"""
Expected Value Intelligence Engine.

This module evaluates the mathematical expectancy of a trading strategy
or setup using statistical win/loss assumptions.

Core gross EV formula:

    Gross EV = (Win Rate * Average Win) -
               (Loss Rate * Average Loss)

Transaction-cost-aware EV:

    Total Cost = Spread Cost + Commission Cost + Slippage Cost

    Net EV = Gross EV - Total Cost

All profit/loss and transaction-cost values are expressed in R-multiples.

The engine is intentionally independent from:
- Position sizing
- Account risk management
- Trade execution
- News intelligence
- Machine learning
- Local LLM reasoning
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Optional


class ExpectedValueIntelligenceError(ValueError):
    """Raised when EV engine configuration or input is invalid."""


class ExpectedValueDecision(str, Enum):
    """Final mathematical EV decision."""

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


class ExpectedValueReasonType(str, Enum):
    """Canonical reason types emitted by the EV engine."""

    EXPECTED_VALUE_VALID = "EXPECTED_VALUE_VALID"
    POSITIVE_EXPECTED_VALUE = "POSITIVE_EXPECTED_VALUE"
    MINIMUM_EXPECTED_VALUE_MET = "MINIMUM_EXPECTED_VALUE_MET"

    EXPECTED_VALUE_TOO_LOW = "EXPECTED_VALUE_TOO_LOW"

    INVALID_WIN_RATE = "INVALID_WIN_RATE"
    INVALID_LOSS_RATE = "INVALID_LOSS_RATE"

    INVALID_AVERAGE_WIN = "INVALID_AVERAGE_WIN"
    INVALID_AVERAGE_LOSS = "INVALID_AVERAGE_LOSS"

    INVALID_EXPECTED_VALUE = "INVALID_EXPECTED_VALUE"
    INVALID_MINIMUM_EXPECTED_VALUE = "INVALID_MINIMUM_EXPECTED_VALUE"

    INVALID_SPREAD_COST = "INVALID_SPREAD_COST"
    INVALID_COMMISSION_COST = "INVALID_COMMISSION_COST"
    INVALID_SLIPPAGE_COST = "INVALID_SLIPPAGE_COST"
    INVALID_TOTAL_COST = "INVALID_TOTAL_COST"
    INVALID_NET_EXPECTED_VALUE = "INVALID_NET_EXPECTED_VALUE"

    NET_EXPECTED_VALUE_TOO_LOW = "NET_EXPECTED_VALUE_TOO_LOW"
    POSITIVE_NET_EXPECTED_VALUE = "POSITIVE_NET_EXPECTED_VALUE"
    MINIMUM_NET_EXPECTED_VALUE_MET = "MINIMUM_NET_EXPECTED_VALUE_MET"

    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True, slots=True)
class ExpectedValueReason:
    """A single auditable EV decision reason."""

    reason_type: ExpectedValueReasonType
    message: str


@dataclass(frozen=True, slots=True)
class ExpectedValueModel:
    """Complete auditable result produced by the EV engine."""

    timestamp: datetime

    win_rate: float
    loss_rate: float

    average_win_r: float
    average_loss_r: float

    # Gross expectancy before transaction costs.
    expected_value_r: Optional[float]

    # Transaction costs represented in R.
    spread_cost_r: float
    commission_cost_r: float
    slippage_cost_r: float
    total_cost_r: Optional[float]

    # Final expectancy after transaction costs.
    net_expected_value_r: Optional[float]

    break_even_win_rate: Optional[float]

    # Minimum accepted thresholds.
    minimum_expected_value_r: float
    minimum_net_expected_value_r: float

    decision: ExpectedValueDecision
    valid: bool
    ready: bool

    reasons: tuple[ExpectedValueReason, ...]
    warnings: tuple[str, ...]

    @property
    def allowed(self) -> bool:
        """Return True when the EV engine allows the expectancy."""
        return self.decision == ExpectedValueDecision.ALLOW

    @property
    def blocked(self) -> bool:
        """Return True when the EV engine blocks the expectancy."""
        return self.decision == ExpectedValueDecision.BLOCK

    @property
    def is_allowed(self) -> bool:
        """Compatibility alias for allowed."""
        return self.allowed

    @property
    def is_blocked(self) -> bool:
        """Compatibility alias for blocked."""
        return self.blocked

    @property
    def ev(self) -> Optional[float]:
        """Short alias for gross expected value in R."""
        return self.expected_value_r

    @property
    def expected_value(self) -> Optional[float]:
        """Readable alias for gross expected value in R."""
        return self.expected_value_r

    @property
    def net_ev(self) -> Optional[float]:
        """Short alias for net expected value in R."""
        return self.net_expected_value_r

    @property
    def net_expected_value(self) -> Optional[float]:
        """Readable alias for net expected value in R."""
        return self.net_expected_value_r

    @property
    def breakeven_win_rate(self) -> Optional[float]:
        """Compatibility alias for break-even win rate."""
        return self.break_even_win_rate

    @property
    def total_cost(self) -> Optional[float]:
        """Readable alias for total transaction cost."""
        return self.total_cost_r


class ExpectedValueIntelligenceEngine:
    """
    Deterministic Expected Value Engine.

    Gross mathematical model:

        loss_rate = 1 - win_rate

        Gross EV =
            (win_rate * average_win_r)
            - (loss_rate * average_loss_r)

    Transaction-cost model:

        total_cost_r =
            spread_cost_r
            + commission_cost_r
            + slippage_cost_r

        Net EV =
            Gross EV - total_cost_r

    A result is ALLOW when:

        Net EV >= minimum_net_expected_value_r

    Floating-point tolerance is used for boundary comparisons and
    near-zero normalization.
    """

    # Numerical tolerance used for floating-point comparisons.
    EV_COMPARISON_TOLERANCE = 1e-12

    def __init__(
        self,
        minimum_expected_value_r: float = 0.0,
        minimum_net_expected_value_r: Optional[float] = None,
    ) -> None:
        """
        Initialize the Expected Value Engine.

        Args:
            minimum_expected_value_r:
                Backward-compatible minimum gross EV threshold.

            minimum_net_expected_value_r:
                Minimum acceptable net EV after transaction costs.

                When omitted, the gross threshold is used.
        """

        self._validate_finite(
            minimum_expected_value_r,
            "minimum_expected_value_r",
        )

        if minimum_net_expected_value_r is None:
            minimum_net_expected_value_r = minimum_expected_value_r

        self._validate_finite(
            minimum_net_expected_value_r,
            "minimum_net_expected_value_r",
        )

        self.minimum_expected_value_r = float(
            minimum_expected_value_r
        )

        self.minimum_net_expected_value_r = float(
            minimum_net_expected_value_r
        )

    def analyze(
        self,
        *,
        win_rate: float,
        average_win_r: float,
        average_loss_r: float,
        minimum_expected_value_r: Optional[float] = None,
        minimum_net_expected_value_r: Optional[float] = None,
        spread_cost_r: float = 0.0,
        commission_cost_r: float = 0.0,
        slippage_cost_r: float = 0.0,
    ) -> ExpectedValueModel:
        """
        Calculate gross and transaction-cost-adjusted expected value.
        """

        timestamp = datetime.now(timezone.utc)

        # Resolve gross EV threshold.
        minimum_ev = (
            self.minimum_expected_value_r
            if minimum_expected_value_r is None
            else minimum_expected_value_r
        )

        # Resolve net EV threshold.
        minimum_net_ev = (
            self.minimum_net_expected_value_r
            if minimum_net_expected_value_r is None
            else minimum_net_expected_value_r
        )

        # Validate gross threshold.
        if not self._is_finite(minimum_ev):
            return self._blocked_result(
                timestamp=timestamp,
                win_rate=win_rate,
                average_win_r=average_win_r,
                average_loss_r=average_loss_r,
                minimum_expected_value_r=minimum_ev,
                minimum_net_expected_value_r=minimum_net_ev,
                spread_cost_r=spread_cost_r,
                commission_cost_r=commission_cost_r,
                slippage_cost_r=slippage_cost_r,
                reason_type=(
                    ExpectedValueReasonType
                    .INVALID_MINIMUM_EXPECTED_VALUE
                ),
                message=(
                    "Minimum expected value must be a finite number."
                ),
            )

        minimum_ev = float(minimum_ev)

        # Validate net threshold.
        if not self._is_finite(minimum_net_ev):
            return self._blocked_result(
                timestamp=timestamp,
                win_rate=win_rate,
                average_win_r=average_win_r,
                average_loss_r=average_loss_r,
                minimum_expected_value_r=minimum_ev,
                minimum_net_expected_value_r=minimum_net_ev,
                spread_cost_r=spread_cost_r,
                commission_cost_r=commission_cost_r,
                slippage_cost_r=slippage_cost_r,
                reason_type=(
                    ExpectedValueReasonType
                    .INVALID_MINIMUM_EXPECTED_VALUE
                ),
                message=(
                    "Minimum net expected value must be "
                    "a finite number."
                ),
            )

        minimum_net_ev = float(minimum_net_ev)

        # Validate transaction costs.
        cost_validation = self._validate_transaction_costs(
            spread_cost_r=spread_cost_r,
            commission_cost_r=commission_cost_r,
            slippage_cost_r=slippage_cost_r,
        )

        if cost_validation is not None:
            reason_type, message = cost_validation

            return self._blocked_result(
                timestamp=timestamp,
                win_rate=win_rate,
                average_win_r=average_win_r,
                average_loss_r=average_loss_r,
                minimum_expected_value_r=minimum_ev,
                minimum_net_expected_value_r=minimum_net_ev,
                spread_cost_r=spread_cost_r,
                commission_cost_r=commission_cost_r,
                slippage_cost_r=slippage_cost_r,
                reason_type=reason_type,
                message=message,
            )

        spread_cost_r = float(spread_cost_r)
        commission_cost_r = float(commission_cost_r)
        slippage_cost_r = float(slippage_cost_r)

        # Validate win rate.
        if not self._is_finite(win_rate):
            return self._blocked_result(
                timestamp=timestamp,
                win_rate=win_rate,
                average_win_r=average_win_r,
                average_loss_r=average_loss_r,
                minimum_expected_value_r=minimum_ev,
                minimum_net_expected_value_r=minimum_net_ev,
                spread_cost_r=spread_cost_r,
                commission_cost_r=commission_cost_r,
                slippage_cost_r=slippage_cost_r,
                reason_type=ExpectedValueReasonType.INVALID_WIN_RATE,
                message=(
                    "Win rate must be a finite value between "
                    "0.0 and 1.0."
                ),
            )

        win_rate = float(win_rate)

        if not 0.0 <= win_rate <= 1.0:
            return self._blocked_result(
                timestamp=timestamp,
                win_rate=win_rate,
                average_win_r=average_win_r,
                average_loss_r=average_loss_r,
                minimum_expected_value_r=minimum_ev,
                minimum_net_expected_value_r=minimum_net_ev,
                spread_cost_r=spread_cost_r,
                commission_cost_r=commission_cost_r,
                slippage_cost_r=slippage_cost_r,
                reason_type=ExpectedValueReasonType.INVALID_WIN_RATE,
                message=(
                    "Win rate must be a finite value between "
                    "0.0 and 1.0."
                ),
            )

        # Validate average winning R.
        if not self._is_positive_finite(average_win_r):
            return self._blocked_result(
                timestamp=timestamp,
                win_rate=win_rate,
                average_win_r=average_win_r,
                average_loss_r=average_loss_r,
                minimum_expected_value_r=minimum_ev,
                minimum_net_expected_value_r=minimum_net_ev,
                spread_cost_r=spread_cost_r,
                commission_cost_r=commission_cost_r,
                slippage_cost_r=slippage_cost_r,
                reason_type=(
                    ExpectedValueReasonType.INVALID_AVERAGE_WIN
                ),
                message=(
                    "Average win must be a finite value greater "
                    "than zero."
                ),
            )

        average_win_r = float(average_win_r)

        # Validate average losing R.
        if not self._is_positive_finite(average_loss_r):
            return self._blocked_result(
                timestamp=timestamp,
                win_rate=win_rate,
                average_win_r=average_win_r,
                average_loss_r=average_loss_r,
                minimum_expected_value_r=minimum_ev,
                minimum_net_expected_value_r=minimum_net_ev,
                spread_cost_r=spread_cost_r,
                commission_cost_r=commission_cost_r,
                slippage_cost_r=slippage_cost_r,
                reason_type=(
                    ExpectedValueReasonType.INVALID_AVERAGE_LOSS
                ),
                message=(
                    "Average loss must be a finite value greater "
                    "than zero."
                ),
            )

        average_loss_r = float(average_loss_r)

        # Calculate loss probability.
        loss_rate = 1.0 - win_rate

        # Calculate break-even win rate.
        break_even_win_rate = (
            average_loss_r
            / (average_win_r + average_loss_r)
        )

        # Calculate gross expected value.
        expected_value_r = (
            (win_rate * average_win_r)
            - (loss_rate * average_loss_r)
        )

        # Validate gross EV.
        if not self._is_finite(expected_value_r):
            return self._blocked_result(
                timestamp=timestamp,
                win_rate=win_rate,
                average_win_r=average_win_r,
                average_loss_r=average_loss_r,
                minimum_expected_value_r=minimum_ev,
                minimum_net_expected_value_r=minimum_net_ev,
                spread_cost_r=spread_cost_r,
                commission_cost_r=commission_cost_r,
                slippage_cost_r=slippage_cost_r,
                reason_type=(
                    ExpectedValueReasonType.INVALID_EXPECTED_VALUE
                ),
                message="Calculated expected value is not finite.",
            )

        expected_value_r = float(expected_value_r)

        # ---------------------------------------------------------
        # Normalize values that are mathematically zero but are
        # represented by tiny floating-point residuals.
        #
        # Example:
        #
        #     -1.1102230246251565e-16
        #
        # should be represented as:
        #
        #     0.0
        # ---------------------------------------------------------
        if abs(expected_value_r) <= self.EV_COMPARISON_TOLERANCE:
            expected_value_r = 0.0

        # Calculate total transaction cost.
        total_cost_r = (
            spread_cost_r
            + commission_cost_r
            + slippage_cost_r
        )

        # Validate total cost.
        if not self._is_finite(total_cost_r):
            return self._blocked_result(
                timestamp=timestamp,
                win_rate=win_rate,
                average_win_r=average_win_r,
                average_loss_r=average_loss_r,
                minimum_expected_value_r=minimum_ev,
                minimum_net_expected_value_r=minimum_net_ev,
                spread_cost_r=spread_cost_r,
                commission_cost_r=commission_cost_r,
                slippage_cost_r=slippage_cost_r,
                reason_type=ExpectedValueReasonType.INVALID_TOTAL_COST,
                message="Total transaction cost is not finite.",
            )

        total_cost_r = float(total_cost_r)

        # Calculate net expected value.
        net_expected_value_r = (
            expected_value_r - total_cost_r
        )

        # Validate net EV.
        if not self._is_finite(net_expected_value_r):
            return self._blocked_result(
                timestamp=timestamp,
                win_rate=win_rate,
                average_win_r=average_win_r,
                average_loss_r=average_loss_r,
                minimum_expected_value_r=minimum_ev,
                minimum_net_expected_value_r=minimum_net_ev,
                spread_cost_r=spread_cost_r,
                commission_cost_r=commission_cost_r,
                slippage_cost_r=slippage_cost_r,
                reason_type=(
                    ExpectedValueReasonType
                    .INVALID_NET_EXPECTED_VALUE
                ),
                message=(
                    "Calculated net expected value is not finite."
                ),
            )

        net_expected_value_r = float(net_expected_value_r)

        # Normalize near-zero net EV as well.
        if abs(net_expected_value_r) <= self.EV_COMPARISON_TOLERANCE:
            net_expected_value_r = 0.0

        reasons: list[ExpectedValueReason] = []
        warnings: list[str] = []

        # ---------------------------------------------------------
        # Gross EV audit.
        # ---------------------------------------------------------
        gross_difference = (
            expected_value_r - minimum_ev
        )

        if gross_difference < -self.EV_COMPARISON_TOLERANCE:
            reasons.append(
                ExpectedValueReason(
                    reason_type=(
                        ExpectedValueReasonType.EXPECTED_VALUE_TOO_LOW
                    ),
                    message=(
                        f"Gross expected value "
                        f"{expected_value_r:.6f}R is below "
                        f"the minimum gross requirement "
                        f"{minimum_ev:.6f}R."
                    ),
                )
            )
        else:
            reasons.append(
                ExpectedValueReason(
                    reason_type=(
                        ExpectedValueReasonType.EXPECTED_VALUE_VALID
                    ),
                    message=(
                        "Gross expected value calculation is valid."
                    ),
                )
            )

        # Positive gross EV audit reason.
        if expected_value_r > self.EV_COMPARISON_TOLERANCE:
            reasons.append(
                ExpectedValueReason(
                    reason_type=(
                        ExpectedValueReasonType
                        .POSITIVE_EXPECTED_VALUE
                    ),
                    message=(
                        f"Gross expected value is positive at "
                        f"{expected_value_r:.6f}R."
                    ),
                )
            )

        # ---------------------------------------------------------
        # Final net EV gate.
        # ---------------------------------------------------------
        net_difference = (
            net_expected_value_r - minimum_net_ev
        )

        if net_difference < -self.EV_COMPARISON_TOLERANCE:
            reasons.append(
                ExpectedValueReason(
                    reason_type=(
                        ExpectedValueReasonType
                        .NET_EXPECTED_VALUE_TOO_LOW
                    ),
                    message=(
                        f"Net expected value "
                        f"{net_expected_value_r:.6f}R is below "
                        f"the minimum required net EV "
                        f"{minimum_net_ev:.6f}R."
                    ),
                )
            )

            return ExpectedValueModel(
                timestamp=timestamp,
                win_rate=win_rate,
                loss_rate=loss_rate,
                average_win_r=average_win_r,
                average_loss_r=average_loss_r,
                expected_value_r=expected_value_r,
                spread_cost_r=spread_cost_r,
                commission_cost_r=commission_cost_r,
                slippage_cost_r=slippage_cost_r,
                total_cost_r=total_cost_r,
                net_expected_value_r=net_expected_value_r,
                break_even_win_rate=break_even_win_rate,
                minimum_expected_value_r=minimum_ev,
                minimum_net_expected_value_r=minimum_net_ev,
                decision=ExpectedValueDecision.BLOCK,
                valid=True,
                ready=False,
                reasons=tuple(reasons),
                warnings=tuple(warnings),
            )

        # Normalize net EV when it sits exactly on the threshold.
        if abs(net_difference) <= self.EV_COMPARISON_TOLERANCE:
            net_expected_value_r = minimum_net_ev

        # Positive net EV audit reason.
        if net_expected_value_r > self.EV_COMPARISON_TOLERANCE:
            reasons.append(
                ExpectedValueReason(
                    reason_type=(
                        ExpectedValueReasonType
                        .POSITIVE_NET_EXPECTED_VALUE
                    ),
                    message=(
                        f"Net expected value is positive at "
                        f"{net_expected_value_r:.6f}R after costs."
                    ),
                )
            )

        # Minimum net EV has been met.
        reasons.append(
            ExpectedValueReason(
                reason_type=(
                    ExpectedValueReasonType
                    .MINIMUM_NET_EXPECTED_VALUE_MET
                ),
                message=(
                    f"Net expected value "
                    f"{net_expected_value_r:.6f}R meets "
                    f"the minimum required net EV "
                    f"{minimum_net_ev:.6f}R."
                ),
            )
        )

        # Preserve the gross threshold reason when satisfied.
        if gross_difference >= -self.EV_COMPARISON_TOLERANCE:
            reasons.append(
                ExpectedValueReason(
                    reason_type=(
                        ExpectedValueReasonType
                        .MINIMUM_EXPECTED_VALUE_MET
                    ),
                    message=(
                        f"Gross expected value "
                        f"{expected_value_r:.6f}R meets "
                        f"the minimum gross requirement "
                        f"{minimum_ev:.6f}R."
                    ),
                )
            )

        warnings.append(
            "Transaction costs are represented as R-multiples."
        )

        return ExpectedValueModel(
            timestamp=timestamp,
            win_rate=win_rate,
            loss_rate=loss_rate,
            average_win_r=average_win_r,
            average_loss_r=average_loss_r,
            expected_value_r=expected_value_r,
            spread_cost_r=spread_cost_r,
            commission_cost_r=commission_cost_r,
            slippage_cost_r=slippage_cost_r,
            total_cost_r=total_cost_r,
            net_expected_value_r=net_expected_value_r,
            break_even_win_rate=break_even_win_rate,
            minimum_expected_value_r=minimum_ev,
            minimum_net_expected_value_r=minimum_net_ev,
            decision=ExpectedValueDecision.ALLOW,
            valid=True,
            ready=True,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
        )

    def _validate_transaction_costs(
        self,
        *,
        spread_cost_r: float,
        commission_cost_r: float,
        slippage_cost_r: float,
    ) -> Optional[
        tuple[ExpectedValueReasonType, str]
    ]:
        """Validate all transaction-cost inputs."""

        if not self._is_non_negative_finite(spread_cost_r):
            return (
                ExpectedValueReasonType.INVALID_SPREAD_COST,
                (
                    "Spread cost must be a finite value "
                    "greater than or equal to zero."
                ),
            )

        if not self._is_non_negative_finite(commission_cost_r):
            return (
                ExpectedValueReasonType.INVALID_COMMISSION_COST,
                (
                    "Commission cost must be a finite value "
                    "greater than or equal to zero."
                ),
            )

        if not self._is_non_negative_finite(slippage_cost_r):
            return (
                ExpectedValueReasonType.INVALID_SLIPPAGE_COST,
                (
                    "Slippage cost must be a finite value "
                    "greater than or equal to zero."
                ),
            )

        return None

    def _blocked_result(
        self,
        *,
        timestamp: datetime,
        win_rate: float,
        average_win_r: float,
        average_loss_r: float,
        minimum_expected_value_r: float,
        minimum_net_expected_value_r: float,
        spread_cost_r: float,
        commission_cost_r: float,
        slippage_cost_r: float,
        reason_type: ExpectedValueReasonType,
        message: str,
    ) -> ExpectedValueModel:
        """Build a deterministic blocked result."""

        loss_rate = (
            1.0 - float(win_rate)
            if self._is_finite(win_rate)
            else float("nan")
        )

        return ExpectedValueModel(
            timestamp=timestamp,
            win_rate=win_rate,
            loss_rate=loss_rate,
            average_win_r=average_win_r,
            average_loss_r=average_loss_r,
            expected_value_r=None,
            spread_cost_r=spread_cost_r,
            commission_cost_r=commission_cost_r,
            slippage_cost_r=slippage_cost_r,
            total_cost_r=None,
            net_expected_value_r=None,
            break_even_win_rate=None,
            minimum_expected_value_r=minimum_expected_value_r,
            minimum_net_expected_value_r=minimum_net_expected_value_r,
            decision=ExpectedValueDecision.BLOCK,
            valid=False,
            ready=False,
            reasons=(
                ExpectedValueReason(
                    reason_type=reason_type,
                    message=message,
                ),
            ),
            warnings=(),
        )

    @staticmethod
    def _is_finite(value: float) -> bool:
        """Return True when a value is numeric and finite."""

        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False

    @classmethod
    def _is_positive_finite(cls, value: float) -> bool:
        """Return True when a value is finite and strictly positive."""

        return (
            cls._is_finite(value)
            and float(value) > 0.0
        )

    @classmethod
    def _is_non_negative_finite(cls, value: float) -> bool:
        """Return True when a value is finite and non-negative."""

        return (
            cls._is_finite(value)
            and float(value) >= 0.0
        )

    @classmethod
    def _validate_finite(
        cls,
        value: float,
        field_name: str,
    ) -> None:
        """Validate that a configuration value is finite."""

        if not cls._is_finite(value):
            raise ExpectedValueIntelligenceError(
                f"{field_name} must be a finite number."
            )