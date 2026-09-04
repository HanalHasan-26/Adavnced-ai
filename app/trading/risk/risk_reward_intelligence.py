from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional

from app.trading.entry.entry_model import EntryDirection, EntryModel


class RiskRewardIntelligenceError(ValueError):
    """Raised when risk/reward intelligence receives invalid input."""


class RiskRewardDecision(str, Enum):
    """Final deterministic decision produced by the RR engine."""

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


class RiskRewardReasonType(str, Enum):
    """Canonical reason codes emitted by the RR engine."""

    RISK_REWARD_VALID = "RISK_REWARD_VALID"
    MINIMUM_RISK_REWARD_MET = "MINIMUM_RISK_REWARD_MET"
    RISK_REWARD_TOO_LOW = "RISK_REWARD_TOO_LOW"
    RISK_REWARD_TOO_HIGH = "RISK_REWARD_TOO_HIGH"

    INVALID_ENTRY = "INVALID_ENTRY"
    INVALID_STOP_LOSS = "INVALID_STOP_LOSS"
    INVALID_TAKE_PROFIT = "INVALID_TAKE_PROFIT"
    INVALID_DIRECTION = "INVALID_DIRECTION"
    INVALID_RISK_DISTANCE = "INVALID_RISK_DISTANCE"
    INVALID_REWARD_DISTANCE = "INVALID_REWARD_DISTANCE"
    INVALID_RISK_REWARD = "INVALID_RISK_REWARD"
    INVALID_SYMBOL = "INVALID_SYMBOL"

    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True, slots=True)
class RiskRewardReason:
    """A deterministic explanation for an RR decision."""

    reason_type: RiskRewardReasonType
    message: str


@dataclass(frozen=True, slots=True)
class RiskRewardModel:
    """Immutable result returned by RiskRewardIntelligenceEngine."""

    timestamp: object
    symbol: str
    timeframe: str
    direction: EntryDirection

    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]

    risk_distance: float
    reward_distance: float
    risk_reward_ratio: float

    minimum_risk_reward: float
    maximum_risk_reward: float

    decision: RiskRewardDecision
    valid: bool
    ready: bool

    reasons: tuple[RiskRewardReason, ...]
    warnings: tuple[str, ...]

    @property
    def allowed(self) -> bool:
        """Return True when the RR engine allows the trade."""
        return self.decision is RiskRewardDecision.ALLOW

    @property
    def blocked(self) -> bool:
        """Return True when the RR engine blocks the trade."""
        return self.decision is RiskRewardDecision.BLOCK

    @property
    def is_allowed(self) -> bool:
        """Compatibility alias for allowed."""
        return self.allowed

    @property
    def is_blocked(self) -> bool:
        """Compatibility alias for blocked."""
        return self.blocked

    @property
    def rr(self) -> float:
        """Return the calculated risk/reward ratio."""
        return self.risk_reward_ratio

    @property
    def risk(self) -> float:
        """Return the risk distance."""
        return self.risk_distance

    @property
    def reward(self) -> float:
        """Return the reward distance."""
        return self.reward_distance


class RiskRewardIntelligenceEngine:
    """
    Deterministic risk/reward validation engine.

    Responsibilities:
    - Validate entry, stop-loss, and take-profit.
    - Calculate risk distance.
    - Calculate reward distance.
    - Calculate risk/reward ratio.
    - Enforce minimum RR.
    - Enforce maximum RR sanity limit.
    - Produce an auditable ALLOW/BLOCK decision.

    This engine does NOT:
    - calculate position size;
    - calculate account risk;
    - override the Risk Engine;
    - calculate expected value;
    - execute trades.
    """

    def __init__(
        self,
        minimum_risk_reward: float = 2.0,
        maximum_risk_reward: float = 10.0,
        minimum_distance: float = 0.01,
    ) -> None:
        """Initialize the RR policy."""

        self.minimum_risk_reward = self._validate_positive(
            minimum_risk_reward,
            "minimum_risk_reward",
        )

        self.maximum_risk_reward = self._validate_positive(
            maximum_risk_reward,
            "maximum_risk_reward",
        )

        self.minimum_distance = self._validate_positive(
            minimum_distance,
            "minimum_distance",
        )

        if self.minimum_risk_reward > self.maximum_risk_reward:
            raise RiskRewardIntelligenceError(
                "minimum_risk_reward cannot exceed maximum_risk_reward."
            )

    def analyze(
        self,
        entry: EntryModel,
        *,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        minimum_risk_reward: Optional[float] = None,
        maximum_risk_reward: Optional[float] = None,
    ) -> RiskRewardModel:
        """Analyze the risk/reward relationship of a proposed trade."""

        # Validate the entry model.
        self._validate_entry(entry)

        # Resolve minimum RR.
        minimum_rr = (
            self.minimum_risk_reward
            if minimum_risk_reward is None
            else self._validate_positive(
                minimum_risk_reward,
                "minimum_risk_reward",
            )
        )

        # Resolve maximum RR.
        maximum_rr = (
            self.maximum_risk_reward
            if maximum_risk_reward is None
            else self._validate_positive(
                maximum_risk_reward,
                "maximum_risk_reward",
            )
        )

        # Ensure the RR policy is logically valid.
        if minimum_rr > maximum_rr:
            raise RiskRewardIntelligenceError(
                "minimum_risk_reward cannot exceed maximum_risk_reward."
            )

        # Validate entry price.
        entry_price = self._validate_price(
            entry.entry_price,
            "entry.entry_price",
        )

        # Stop-loss is required.
        if stop_loss is None:
            return self._blocked_result(
                entry=entry,
                entry_price=entry_price,
                stop_loss=None,
                take_profit=take_profit,
                minimum_rr=minimum_rr,
                maximum_rr=maximum_rr,
                reason_type=RiskRewardReasonType.INVALID_STOP_LOSS,
                message=(
                    "A valid stop-loss is required before "
                    "calculating risk/reward."
                ),
            )

        # Validate stop-loss.
        stop_loss_value = self._validate_price(
            stop_loss,
            "stop_loss",
        )

        # Take-profit is required.
        if take_profit is None:
            return self._blocked_result(
                entry=entry,
                entry_price=entry_price,
                stop_loss=stop_loss_value,
                take_profit=None,
                minimum_rr=minimum_rr,
                maximum_rr=maximum_rr,
                reason_type=RiskRewardReasonType.INVALID_TAKE_PROFIT,
                message=(
                    "A valid take-profit is required before "
                    "calculating risk/reward."
                ),
            )

        # Validate take-profit.
        take_profit_value = self._validate_price(
            take_profit,
            "take_profit",
        )

        # Validate direction.
        if entry.direction in (
            EntryDirection.NONE,
            EntryDirection.UNKNOWN,
        ):
            return self._blocked_result(
                entry=entry,
                entry_price=entry_price,
                stop_loss=stop_loss_value,
                take_profit=take_profit_value,
                minimum_rr=minimum_rr,
                maximum_rr=maximum_rr,
                reason_type=RiskRewardReasonType.INVALID_DIRECTION,
                message="Entry direction must be LONG or SHORT.",
            )

        # LONG stop-loss must be below entry.
        if entry.direction is EntryDirection.LONG:
            if stop_loss_value >= entry_price:
                return self._blocked_result(
                    entry=entry,
                    entry_price=entry_price,
                    stop_loss=stop_loss_value,
                    take_profit=take_profit_value,
                    minimum_rr=minimum_rr,
                    maximum_rr=maximum_rr,
                    reason_type=RiskRewardReasonType.INVALID_STOP_LOSS,
                    message=(
                        "For a LONG trade, stop-loss must be "
                        "below the entry price."
                    ),
                )

        # SHORT stop-loss must be above entry.
        elif entry.direction is EntryDirection.SHORT:
            if stop_loss_value <= entry_price:
                return self._blocked_result(
                    entry=entry,
                    entry_price=entry_price,
                    stop_loss=stop_loss_value,
                    take_profit=take_profit_value,
                    minimum_rr=minimum_rr,
                    maximum_rr=maximum_rr,
                    reason_type=RiskRewardReasonType.INVALID_STOP_LOSS,
                    message=(
                        "For a SHORT trade, stop-loss must be "
                        "above the entry price."
                    ),
                )

        # Calculate risk distance.
        risk_distance = abs(
            entry_price - stop_loss_value
        )

        # Reject extremely small risk.
        if risk_distance <= self.minimum_distance:
            return self._blocked_result(
                entry=entry,
                entry_price=entry_price,
                stop_loss=stop_loss_value,
                take_profit=take_profit_value,
                minimum_rr=minimum_rr,
                maximum_rr=maximum_rr,
                reason_type=RiskRewardReasonType.INVALID_RISK_DISTANCE,
                message="Risk distance is too small.",
                risk_distance=risk_distance,
            )

        # LONG take-profit must be above entry.
        if entry.direction is EntryDirection.LONG:
            if take_profit_value <= entry_price:
                return self._blocked_result(
                    entry=entry,
                    entry_price=entry_price,
                    stop_loss=stop_loss_value,
                    take_profit=take_profit_value,
                    minimum_rr=minimum_rr,
                    maximum_rr=maximum_rr,
                    reason_type=RiskRewardReasonType.INVALID_TAKE_PROFIT,
                    message=(
                        "For a LONG trade, take-profit must be "
                        "above the entry price."
                    ),
                    risk_distance=risk_distance,
                )

        # SHORT take-profit must be below entry.
        elif entry.direction is EntryDirection.SHORT:
            if take_profit_value >= entry_price:
                return self._blocked_result(
                    entry=entry,
                    entry_price=entry_price,
                    stop_loss=stop_loss_value,
                    take_profit=take_profit_value,
                    minimum_rr=minimum_rr,
                    maximum_rr=maximum_rr,
                    reason_type=RiskRewardReasonType.INVALID_TAKE_PROFIT,
                    message=(
                        "For a SHORT trade, take-profit must be "
                        "below the entry price."
                    ),
                    risk_distance=risk_distance,
                )

        # Calculate reward distance.
        reward_distance = abs(
            take_profit_value - entry_price
        )

        # Reject extremely small reward.
        if reward_distance <= self.minimum_distance:
            return self._blocked_result(
                entry=entry,
                entry_price=entry_price,
                stop_loss=stop_loss_value,
                take_profit=take_profit_value,
                minimum_rr=minimum_rr,
                maximum_rr=maximum_rr,
                reason_type=RiskRewardReasonType.INVALID_REWARD_DISTANCE,
                message="Reward distance is too small.",
                risk_distance=risk_distance,
                reward_distance=reward_distance,
            )

        # Calculate risk/reward ratio.
        risk_reward_ratio = (
            reward_distance / risk_distance
        )

        # Protect against numerical anomalies.
        if (
            not math.isfinite(risk_reward_ratio)
            or risk_reward_ratio <= 0.0
        ):
            return self._blocked_result(
                entry=entry,
                entry_price=entry_price,
                stop_loss=stop_loss_value,
                take_profit=take_profit_value,
                minimum_rr=minimum_rr,
                maximum_rr=maximum_rr,
                reason_type=RiskRewardReasonType.INVALID_RISK_REWARD,
                message="Calculated risk/reward ratio is invalid.",
                risk_distance=risk_distance,
                reward_distance=reward_distance,
            )

        # Minimum RR is a hard veto.
        if risk_reward_ratio < minimum_rr:
            return self._blocked_result(
                entry=entry,
                entry_price=entry_price,
                stop_loss=stop_loss_value,
                take_profit=take_profit_value,
                minimum_rr=minimum_rr,
                maximum_rr=maximum_rr,
                reason_type=RiskRewardReasonType.RISK_REWARD_TOO_LOW,
                message=(
                    "Risk/reward ratio is below the configured "
                    "minimum requirement."
                ),
                risk_distance=risk_distance,
                reward_distance=reward_distance,
                risk_reward_ratio=risk_reward_ratio,
            )

        # Maximum RR is a sanity boundary.
        if risk_reward_ratio > maximum_rr:
            return self._blocked_result(
                entry=entry,
                entry_price=entry_price,
                stop_loss=stop_loss_value,
                take_profit=take_profit_value,
                minimum_rr=minimum_rr,
                maximum_rr=maximum_rr,
                reason_type=RiskRewardReasonType.RISK_REWARD_TOO_HIGH,
                message=(
                    "Risk/reward ratio exceeds the configured "
                    "maximum sanity limit."
                ),
                risk_distance=risk_distance,
                reward_distance=reward_distance,
                risk_reward_ratio=risk_reward_ratio,
            )

        # Build successful audit reasons.
        reasons = (
            RiskRewardReason(
                RiskRewardReasonType.RISK_REWARD_VALID,
                "Risk/reward calculation is valid.",
            ),
            RiskRewardReason(
                RiskRewardReasonType.MINIMUM_RISK_REWARD_MET,
                (
                    "The configured minimum risk/reward "
                    "requirement has been satisfied."
                ),
            ),
        )

        # Explain what this engine intentionally does not do.
        warnings = (
            "Risk/reward analysis does not calculate position size.",
            "Risk/reward analysis does not authorize live execution.",
            "Expected value is implemented in a later P2.17 layer.",
        )

        # Return successful result.
        return RiskRewardModel(
            timestamp=entry.timestamp,
            symbol=entry.symbol,
            timeframe=entry.timeframe,
            direction=entry.direction,
            entry_price=round(entry_price, 10),
            stop_loss=round(stop_loss_value, 10),
            take_profit=round(take_profit_value, 10),
            risk_distance=round(risk_distance, 10),
            reward_distance=round(reward_distance, 10),
            risk_reward_ratio=round(risk_reward_ratio, 10),
            minimum_risk_reward=minimum_rr,
            maximum_risk_reward=maximum_rr,
            decision=RiskRewardDecision.ALLOW,
            valid=True,
            ready=True,
            reasons=reasons,
            warnings=warnings,
        )

    def analyze_xauusd(
        self,
        entry: EntryModel,
        *,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        minimum_risk_reward: Optional[float] = None,
        maximum_risk_reward: Optional[float] = None,
    ) -> RiskRewardModel:
        """Analyze an XAUUSD trade."""

        if entry.symbol != "XAUUSD":
            raise RiskRewardIntelligenceError(
                "analyze_xauusd requires symbol XAUUSD."
            )

        return self.analyze(
            entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            minimum_risk_reward=minimum_risk_reward,
            maximum_risk_reward=maximum_risk_reward,
        )

    def _blocked_result(
        self,
        *,
        entry: EntryModel,
        entry_price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        minimum_rr: float,
        maximum_rr: float,
        reason_type: RiskRewardReasonType,
        message: str,
        risk_distance: float = 0.0,
        reward_distance: float = 0.0,
        risk_reward_ratio: float = 0.0,
    ) -> RiskRewardModel:
        """Create a deterministic blocked result."""

        return RiskRewardModel(
            timestamp=entry.timestamp,
            symbol=entry.symbol,
            timeframe=entry.timeframe,
            direction=entry.direction,
            entry_price=round(entry_price, 10),
            stop_loss=(
                None
                if stop_loss is None
                else round(stop_loss, 10)
            ),
            take_profit=(
                None
                if take_profit is None
                else round(take_profit, 10)
            ),
            risk_distance=round(
                max(risk_distance, 0.0),
                10,
            ),
            reward_distance=round(
                max(reward_distance, 0.0),
                10,
            ),
            risk_reward_ratio=round(
                max(risk_reward_ratio, 0.0),
                10,
            ),
            minimum_risk_reward=minimum_rr,
            maximum_risk_reward=maximum_rr,
            decision=RiskRewardDecision.BLOCK,
            valid=False,
            ready=False,
            reasons=(
                RiskRewardReason(
                    reason_type,
                    message,
                ),
            ),
            warnings=(
                "Risk/reward vetoed the proposed trade.",
            ),
        )

    @staticmethod
    def _validate_entry(entry: EntryModel) -> None:
        """Validate the EntryModel."""

        if not isinstance(entry, EntryModel):
            raise RiskRewardIntelligenceError(
                "entry must be an EntryModel instance."
            )

        if not entry.symbol:
            raise RiskRewardIntelligenceError(
                "entry.symbol cannot be empty."
            )

        if not entry.timeframe:
            raise RiskRewardIntelligenceError(
                "entry.timeframe cannot be empty."
            )

    @staticmethod
    def _validate_price(
        value: float,
        field_name: str,
    ) -> float:
        """Validate a strictly positive finite price."""

        if isinstance(value, bool):
            raise RiskRewardIntelligenceError(
                f"{field_name} must be a positive number."
            )

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise RiskRewardIntelligenceError(
                f"{field_name} must be a positive number."
            ) from exc

        if (
            not math.isfinite(numeric_value)
            or numeric_value <= 0.0
        ):
            raise RiskRewardIntelligenceError(
                f"{field_name} must be a positive finite number."
            )

        return numeric_value

    @classmethod
    def _validate_positive(
        cls,
        value: float,
        field_name: str,
    ) -> float:
        """Validate a positive finite configuration value."""

        return cls._validate_price(
            value,
            field_name,
        )