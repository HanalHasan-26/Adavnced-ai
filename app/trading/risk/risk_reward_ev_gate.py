"""
Risk/Reward + Expected Value Decision Gate.

This module combines:
1. Risk/Reward validation.
2. Transaction-cost-aware Expected Value validation.

The gate is deterministic and does not depend on an LLM.

The purpose is to provide one final decision for the RR/EV portion
of the trading risk-intelligence layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Optional


class RiskRewardEVGateError(ValueError):
    """Raised when invalid data is supplied to the RR/EV gate."""


class RiskRewardEVDecision(str, Enum):
    """Possible decisions produced by the RR/EV gate."""

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


class RiskRewardEVReasonType(str, Enum):
    """Machine-readable reasons for the final RR/EV decision."""

    RR_VALID = "RR_VALID"
    EV_VALID = "EV_VALID"
    RR_AND_EV_VALID = "RR_AND_EV_VALID"

    INVALID_RR = "INVALID_RR"
    RR_TOO_LOW = "RR_TOO_LOW"

    INVALID_EV = "INVALID_EV"
    EV_TOO_LOW = "EV_TOO_LOW"

    INVALID_INPUT = "INVALID_INPUT"


@dataclass(frozen=True)
class RiskRewardEVGateModel:
    """
    Immutable result produced by the RR/EV decision gate.

    The model intentionally stores the raw RR and net EV values so
    the decision can be audited later.
    """

    timestamp: datetime

    risk_reward: Optional[float]
    minimum_risk_reward: float

    net_expected_value_r: Optional[float]
    minimum_net_expected_value_r: float

    decision: RiskRewardEVDecision
    valid: bool

    reasons: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def allowed(self) -> bool:
        """Return True when the RR/EV gate allows the trade."""

        return self.decision == RiskRewardEVDecision.ALLOW

    @property
    def blocked(self) -> bool:
        """Return True when the RR/EV gate blocks the trade."""

        return self.decision == RiskRewardEVDecision.BLOCK

    @property
    def is_allowed(self) -> bool:
        """Compatibility alias for allowed."""

        return self.allowed

    @property
    def is_blocked(self) -> bool:
        """Compatibility alias for blocked."""

        return self.blocked


class RiskRewardEVGate:
    """
    Deterministic RR + transaction-cost-aware EV gate.

    A trade is allowed only when BOTH conditions are satisfied:

        RR >= minimum RR

    AND

        Net EV >= minimum net EV

    Equality at the configured thresholds is intentionally allowed.
    """

    COMPARISON_TOLERANCE = 1e-12

    def __init__(
        self,
        minimum_risk_reward: float = 2.0,
        minimum_net_expected_value_r: float = 0.0,
    ) -> None:
        """
        Initialize the RR/EV gate.

        Parameters
        ----------
        minimum_risk_reward:
            Minimum acceptable risk/reward ratio.

        minimum_net_expected_value_r:
            Minimum acceptable transaction-cost-adjusted EV in R.
        """

        self._validate_finite(
            minimum_risk_reward,
            "minimum_risk_reward",
        )

        self._validate_finite(
            minimum_net_expected_value_r,
            "minimum_net_expected_value_r",
        )

        if minimum_risk_reward <= 0:
            raise RiskRewardEVGateError(
                "minimum_risk_reward must be greater than zero."
            )

        self.minimum_risk_reward = float(minimum_risk_reward)
        self.minimum_net_expected_value_r = float(
            minimum_net_expected_value_r
        )

    @staticmethod
    def _validate_finite(value: float, field_name: str) -> None:
        """Validate that a numeric value is finite."""

        if isinstance(value, bool):
            raise RiskRewardEVGateError(
                f"{field_name} must be a finite numeric value."
            )

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            raise RiskRewardEVGateError(
                f"{field_name} must be a finite numeric value."
            )

        if not isfinite(numeric_value):
            raise RiskRewardEVGateError(
                f"{field_name} must be a finite numeric value."
            )

    @classmethod
    def _meets_threshold(
        cls,
        value: float,
        threshold: float,
    ) -> bool:
        """
        Compare values using a small floating-point tolerance.

        This prevents values such as:

            1.9999999999999998

        from incorrectly failing an exact threshold comparison.
        """

        return value >= threshold - cls.COMPARISON_TOLERANCE

    @classmethod
    def _normalize_zero(cls, value: float) -> float:
        """
        Normalize values extremely close to zero.

        This prevents floating-point artifacts such as:

            -1.1102230246251565e-16

        from appearing as the actual business result.
        """

        if abs(value) <= cls.COMPARISON_TOLERANCE:
            return 0.0

        return value

    def analyze(
        self,
        *,
        risk_reward: float,
        net_expected_value_r: float,
    ) -> RiskRewardEVGateModel:
        """
        Evaluate RR and net EV together.

        Parameters
        ----------
        risk_reward:
            Calculated trade risk/reward ratio.

        net_expected_value_r:
            Transaction-cost-adjusted expected value in R.

        Returns
        -------
        RiskRewardEVGateModel
            Deterministic ALLOW/BLOCK decision.
        """

        timestamp = datetime.now(timezone.utc)

        try:
            self._validate_finite(
                risk_reward,
                "risk_reward",
            )

            self._validate_finite(
                net_expected_value_r,
                "net_expected_value_r",
            )

            risk_reward = float(risk_reward)
            net_expected_value_r = float(net_expected_value_r)

        except RiskRewardEVGateError as exc:
            return RiskRewardEVGateModel(
                timestamp=timestamp,
                risk_reward=None,
                minimum_risk_reward=self.minimum_risk_reward,
                net_expected_value_r=None,
                minimum_net_expected_value_r=(
                    self.minimum_net_expected_value_r
                ),
                decision=RiskRewardEVDecision.BLOCK,
                valid=False,
                reasons=(
                    RiskRewardEVReasonType.INVALID_INPUT.value,
                    str(exc),
                ),
                warnings=(),
            )

        risk_reward = self._normalize_zero(risk_reward)
        net_expected_value_r = self._normalize_zero(
            net_expected_value_r
        )

        rr_valid = self._meets_threshold(
            risk_reward,
            self.minimum_risk_reward,
        )

        ev_valid = self._meets_threshold(
            net_expected_value_r,
            self.minimum_net_expected_value_r,
        )

        reasons: list[str] = []
        warnings: list[str] = []

        if rr_valid:
            reasons.append(
                RiskRewardEVReasonType.RR_VALID.value
            )
        else:
            reasons.append(
                RiskRewardEVReasonType.RR_TOO_LOW.value
            )

        if ev_valid:
            reasons.append(
                RiskRewardEVReasonType.EV_VALID.value
            )
        else:
            reasons.append(
                RiskRewardEVReasonType.EV_TOO_LOW.value
            )

        if rr_valid and ev_valid:
            decision = RiskRewardEVDecision.ALLOW

            reasons.append(
                RiskRewardEVReasonType.RR_AND_EV_VALID.value
            )
        else:
            decision = RiskRewardEVDecision.BLOCK

        warnings.append(
            "RR/EV gate is deterministic and does not override "
            "the authoritative Risk Engine."
        )

        return RiskRewardEVGateModel(
            timestamp=timestamp,
            risk_reward=risk_reward,
            minimum_risk_reward=self.minimum_risk_reward,
            net_expected_value_r=net_expected_value_r,
            minimum_net_expected_value_r=(
                self.minimum_net_expected_value_r
            ),
            decision=decision,
            valid=True,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
        )

    def evaluate(
        self,
        *,
        risk_reward: float,
        net_expected_value_r: float,
    ) -> RiskRewardEVGateModel:
        """
        Compatibility alias for analyze().
        """

        return self.analyze(
            risk_reward=risk_reward,
            net_expected_value_r=net_expected_value_r,
        )

    def check(
        self,
        *,
        risk_reward: float,
        net_expected_value_r: float,
    ) -> bool:
        """
        Return only the final ALLOW/BLOCK result.

        This is useful for higher-level deterministic engines.
        """

        result = self.analyze(
            risk_reward=risk_reward,
            net_expected_value_r=net_expected_value_r,
        )

        return result.allowed