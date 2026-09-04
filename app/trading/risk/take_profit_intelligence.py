from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional

from app.trading.entry.entry_model import EntryDirection, EntryModel


class TakeProfitIntelligenceError(ValueError):
    """Raised when take-profit intelligence receives invalid input."""


class TakeProfitMethod(str, Enum):
    """Methods that can be used to determine a take-profit target."""

    NONE = "NONE"
    RISK_REWARD = "RISK_REWARD"
    STRUCTURE = "STRUCTURE"
    SUPPORT_RESISTANCE = "SUPPORT_RESISTANCE"
    TRENDLINE = "TRENDLINE"
    HYBRID = "HYBRID"


class TakeProfitQuality(str, Enum):
    """Quality classification for the selected take-profit."""

    INVALID = "INVALID"
    POOR = "POOR"
    ACCEPTABLE = "ACCEPTABLE"
    GOOD = "GOOD"
    EXCELLENT = "EXCELLENT"


class TakeProfitReasonType(str, Enum):
    """Reason codes explaining the take-profit decision."""

    RISK_REWARD_TARGET = "RISK_REWARD_TARGET"
    STRUCTURAL_TARGET = "STRUCTURAL_TARGET"
    SUPPORT_RESISTANCE_TARGET = "SUPPORT_RESISTANCE_TARGET"
    TRENDLINE_TARGET = "TRENDLINE_TARGET"
    HYBRID_TARGET = "HYBRID_TARGET"

    TARGET_TOO_CLOSE = "TARGET_TOO_CLOSE"
    TARGET_WRONG_SIDE = "TARGET_WRONG_SIDE"

    INVALID_STOP_LOSS = "INVALID_STOP_LOSS"
    INVALID_DIRECTION = "INVALID_DIRECTION"
    INVALID_ENTRY = "INVALID_ENTRY"
    INVALID_SYMBOL = "INVALID_SYMBOL"
    INVALID_RISK_REWARD = "INVALID_RISK_REWARD"

    HIGH_RISK_REWARD = "HIGH_RISK_REWARD"
    GOOD_RISK_REWARD = "GOOD_RISK_REWARD"
    ACCEPTABLE_RISK_REWARD = "ACCEPTABLE_RISK_REWARD"

    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True, slots=True)
class TakeProfitReason:
    """Structured explanation for a take-profit decision."""

    reason_type: TakeProfitReasonType
    message: str


@dataclass(frozen=True, slots=True)
class TakeProfitModel:
    """Immutable result produced by the take-profit engine."""

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

    method: TakeProfitMethod
    quality: TakeProfitQuality
    quality_score: float

    take_profit_ready: bool
    valid: bool

    reasons: tuple[TakeProfitReason, ...]
    warnings: tuple[str, ...]

    @property
    def is_long(self) -> bool:
        """Return True when the trade direction is LONG."""

        return self.direction is EntryDirection.LONG

    @property
    def is_short(self) -> bool:
        """Return True when the trade direction is SHORT."""

        return self.direction is EntryDirection.SHORT

    @property
    def has_take_profit(self) -> bool:
        """Return True when a take-profit exists."""

        return self.take_profit is not None

    @property
    def is_ready(self) -> bool:
        """Return whether the take-profit is ready."""

        return self.take_profit_ready

    @property
    def reward(self) -> float:
        """Return the calculated reward distance."""

        return self.reward_distance

    @property
    def rr(self) -> float:
        """Return the calculated risk/reward ratio."""

        return self.risk_reward_ratio


class TakeProfitIntelligenceEngine:
    """
    Determine a technically valid take-profit target.

    Target sources:

    1. Structural level
    2. Support/resistance
    3. Trendline
    4. Minimum risk/reward fallback

    This engine does NOT calculate:

    - position size
    - lot size
    - account risk
    - daily loss
    - drawdown
    - trade execution

    Those responsibilities belong to the Risk Engine and execution
    layers.
    """

    def __init__(
        self,
        minimum_risk_reward: float = 2.0,
        minimum_distance: float = 0.01,
        maximum_risk_reward: float = 10.0,
        excellent_score: float = 90.0,
        good_score: float = 70.0,
        acceptable_score: float = 50.0,
    ) -> None:
        # Store and validate the minimum RR requirement.
        self.minimum_risk_reward = self._validate_positive(
            minimum_risk_reward,
            "minimum_risk_reward",
        )

        # Store and validate the minimum price distance.
        self.minimum_distance = self._validate_positive(
            minimum_distance,
            "minimum_distance",
        )

        # Store and validate the maximum allowed RR.
        self.maximum_risk_reward = self._validate_positive(
            maximum_risk_reward,
            "maximum_risk_reward",
        )

        # The minimum RR can never exceed the maximum RR.
        if self.minimum_risk_reward > self.maximum_risk_reward:
            raise TakeProfitIntelligenceError(
                "minimum_risk_reward cannot exceed maximum_risk_reward."
            )

        # Validate quality thresholds.
        self.excellent_score = self._validate_score(
            excellent_score,
            "excellent_score",
        )

        self.good_score = self._validate_score(
            good_score,
            "good_score",
        )

        self.acceptable_score = self._validate_score(
            acceptable_score,
            "acceptable_score",
        )

        # Ensure quality thresholds are correctly ordered.
        if not (
            self.excellent_score
            > self.good_score
            > self.acceptable_score
        ):
            raise TakeProfitIntelligenceError(
                "quality thresholds must satisfy "
                "excellent_score > good_score > acceptable_score."
            )

    def analyze(
        self,
        entry: EntryModel,
        *,
        stop_loss: Optional[float] = None,
        structural_level: Optional[float] = None,
        resistance_level: Optional[float] = None,
        support_level: Optional[float] = None,
        trendline_level: Optional[float] = None,
        reference_price: Optional[float] = None,
        minimum_risk_reward: Optional[float] = None,
    ) -> TakeProfitModel:
        """
        Analyze an entry and determine the best take-profit.

        The EntryModel entry price remains the basis for risk-distance
        calculation. An explicit reference_price can override the
        target-selection reference price.
        """

        # Validate the EntryModel before using it.
        self._validate_entry(entry)

        # Resolve the configured minimum RR.
        if minimum_risk_reward is None:
            minimum_rr = self.minimum_risk_reward
        else:
            minimum_rr = self._validate_positive(
                minimum_risk_reward,
                "minimum_risk_reward",
            )

        # The requested minimum RR cannot exceed the engine maximum.
        if minimum_rr > self.maximum_risk_reward:
            raise TakeProfitIntelligenceError(
                "minimum_risk_reward cannot exceed maximum_risk_reward."
            )

        # Validate every optional target level.
        self._validate_optional_price(
            structural_level,
            "structural_level",
        )

        self._validate_optional_price(
            resistance_level,
            "resistance_level",
        )

        self._validate_optional_price(
            support_level,
            "support_level",
        )

        self._validate_optional_price(
            trendline_level,
            "trendline_level",
        )

        # Validate the original entry price.
        original_entry = self._validate_price(
            entry.entry_price,
            "entry.entry_price",
        )

        # By default the effective target reference is the entry price.
        effective_entry = original_entry

        # An explicit reference price overrides the target reference.
        if reference_price is not None:
            effective_entry = self._validate_price(
                reference_price,
                "reference_price",
            )

        # TP calculation cannot continue without an SL.
        if stop_loss is None:
            return self._invalid_result(
                entry=entry,
                entry_price=effective_entry,
                stop_loss=None,
                minimum_rr=minimum_rr,
                reason_type=TakeProfitReasonType.INVALID_STOP_LOSS,
                message=(
                    "A valid stop-loss is required before "
                    "calculating take-profit."
                ),
            )

        # Validate the stop-loss price.
        stop_loss_value = self._validate_price(
            stop_loss,
            "stop_loss",
        )

        # Read the trade direction.
        direction = entry.direction

        # NONE and UNKNOWN directions cannot produce a TP.
        if (
            direction is EntryDirection.NONE
            or direction is EntryDirection.UNKNOWN
        ):
            return self._invalid_result(
                entry=entry,
                entry_price=effective_entry,
                stop_loss=stop_loss_value,
                minimum_rr=minimum_rr,
                reason_type=TakeProfitReasonType.INVALID_DIRECTION,
                message="Entry direction must be LONG or SHORT.",
            )

        # Preserve the existing risk-distance contract.
        calculation_entry = original_entry

        # LONG trades require SL below entry.
        if direction is EntryDirection.LONG:
            if stop_loss_value >= calculation_entry:
                return self._invalid_result(
                    entry=entry,
                    entry_price=effective_entry,
                    stop_loss=stop_loss_value,
                    minimum_rr=minimum_rr,
                    reason_type=TakeProfitReasonType.INVALID_STOP_LOSS,
                    message=(
                        "For a LONG entry, stop-loss must be "
                        "below the entry price."
                    ),
                )

        # SHORT trades require SL above entry.
        elif direction is EntryDirection.SHORT:
            if stop_loss_value <= calculation_entry:
                return self._invalid_result(
                    entry=entry,
                    entry_price=effective_entry,
                    stop_loss=stop_loss_value,
                    minimum_rr=minimum_rr,
                    reason_type=TakeProfitReasonType.INVALID_STOP_LOSS,
                    message=(
                        "For a SHORT entry, stop-loss must be "
                        "above the entry price."
                    ),
                )

        # Calculate the absolute risk distance.
        risk_distance = abs(
            calculation_entry - stop_loss_value
        )

        # Reject an unusably small stop distance.
        if risk_distance <= self.minimum_distance:
            return self._invalid_result(
                entry=entry,
                entry_price=effective_entry,
                stop_loss=stop_loss_value,
                minimum_rr=minimum_rr,
                reason_type=TakeProfitReasonType.INVALID_STOP_LOSS,
                message="Stop-loss is too close to the entry price.",
            )

        # Calculate the deterministic minimum-RR fallback.
        rr_target = self._calculate_rr_target(
            effective_entry,
            risk_distance,
            direction,
            minimum_rr,
        )

        # Build all supplied target candidates.
        candidates = []

        # Add structural target when available.
        if structural_level is not None:
            candidates.append(
                (
                    structural_level,
                    TakeProfitMethod.STRUCTURE,
                    TakeProfitReasonType.STRUCTURAL_TARGET,
                    "A valid structural target was identified.",
                )
            )

        # Add resistance target when available.
        if resistance_level is not None:
            candidates.append(
                (
                    resistance_level,
                    TakeProfitMethod.SUPPORT_RESISTANCE,
                    TakeProfitReasonType.SUPPORT_RESISTANCE_TARGET,
                    "A valid resistance target was identified.",
                )
            )

        # Add support target when available.
        if support_level is not None:
            candidates.append(
                (
                    support_level,
                    TakeProfitMethod.SUPPORT_RESISTANCE,
                    TakeProfitReasonType.SUPPORT_RESISTANCE_TARGET,
                    "A valid support target was identified.",
                )
            )

        # Add trendline target when available.
        if trendline_level is not None:
            candidates.append(
                (
                    trendline_level,
                    TakeProfitMethod.TRENDLINE,
                    TakeProfitReasonType.TRENDLINE_TARGET,
                    "A valid trendline target was identified.",
                )
            )

        # Calculate the exact minimum reward required.
        minimum_reward = (
            risk_distance * minimum_rr
        )

        # Calculate the maximum reward allowed.
        maximum_reward = (
            risk_distance * self.maximum_risk_reward
        )

        # Store only targets that satisfy every TP policy.
        valid_candidates = []

        for (
            level,
            method,
            reason_type,
            message,
        ) in candidates:

            # Reject targets on the wrong side of entry.
            if not self._is_valid_target_side(
                effective_entry,
                level,
                direction,
            ):
                continue

            # Calculate reward distance.
            reward = abs(
                level - effective_entry
            )

            # Reject targets below the exact minimum RR boundary.
            if reward < minimum_reward:
                continue

            # Reject targets above the configured maximum RR boundary.
            if reward > maximum_reward:
                continue

            # Store the valid target candidate.
            valid_candidates.append(
                (
                    level,
                    method,
                    reason_type,
                    message,
                    reward,
                )
            )

        # Default target selection is the RR fallback.
        selected_target = None

        selected_method = TakeProfitMethod.RISK_REWARD

        selected_reason_type = (
            TakeProfitReasonType.RISK_REWARD_TARGET
        )

        selected_message = (
            "Take-profit is calculated from the configured "
            "minimum risk/reward requirement."
        )

        # Prefer the nearest valid market-structure target.
        if valid_candidates:
            valid_candidates.sort(
                key=lambda item: item[4],
            )

            selected_target = valid_candidates[0][0]
            selected_method = valid_candidates[0][1]
            selected_reason_type = valid_candidates[0][2]
            selected_message = valid_candidates[0][3]

            # Find target methods agreeing around the same level.
            same_level_methods = {
                item[1]
                for item in valid_candidates
                if math.isclose(
                    item[0],
                    selected_target,
                    rel_tol=0.0,
                    abs_tol=self.minimum_distance,
                )
            }

            # Structure + S/R agreement becomes HYBRID.
            if (
                TakeProfitMethod.STRUCTURE
                in same_level_methods
                and TakeProfitMethod.SUPPORT_RESISTANCE
                in same_level_methods
            ):
                selected_method = TakeProfitMethod.HYBRID

                selected_reason_type = (
                    TakeProfitReasonType.HYBRID_TARGET
                )

                selected_message = (
                    "Structural and support/resistance targets "
                    "agree on the same take-profit level."
                )

        # If no valid market target exists, use RR fallback.
        else:
            selected_target = rr_target

        # Normalize the final target precision.
        selected_target = round(
            selected_target,
            10,
        )

        # Calculate the final reward distance.
        reward_distance = abs(
            selected_target - effective_entry
        )

        # Defensive invariant.
        if risk_distance <= 0.0:
            raise TakeProfitIntelligenceError(
                "risk distance must be greater than zero."
            )

        # Calculate the final risk/reward ratio.
        risk_reward_ratio = (
            reward_distance / risk_distance
        )

        # Final exact minimum-RR validation.
        if risk_reward_ratio < minimum_rr:
            return self._invalid_result(
                entry=entry,
                entry_price=effective_entry,
                stop_loss=stop_loss_value,
                minimum_rr=minimum_rr,
                reason_type=TakeProfitReasonType.TARGET_TOO_CLOSE,
                message=(
                    "The selected target does not satisfy the "
                    "configured minimum risk/reward requirement."
                ),
            )

        # Final maximum-RR validation.
        if risk_reward_ratio > self.maximum_risk_reward:
            return self._invalid_result(
                entry=entry,
                entry_price=effective_entry,
                stop_loss=stop_loss_value,
                minimum_rr=minimum_rr,
                reason_type=TakeProfitReasonType.INVALID_RISK_REWARD,
                message=(
                    "The selected target exceeds the configured "
                    "maximum risk/reward requirement."
                ),
            )

        # Calculate deterministic quality score.
        quality_score = self._calculate_quality_score(
            risk_reward_ratio,
            selected_method,
        )

        # Convert score into quality enum.
        quality = self._quality_from_score(
            quality_score,
        )

        # Start structured reason collection.
        reasons = [
            TakeProfitReason(
                selected_reason_type,
                selected_message,
            ),
        ]

        # Add high-RR classification.
        if risk_reward_ratio >= 4.0:
            reasons.append(
                TakeProfitReason(
                    TakeProfitReasonType.HIGH_RISK_REWARD,
                    "The target provides a high risk/reward ratio.",
                )
            )

        # Add good-RR classification.
        elif risk_reward_ratio >= 3.0:
            reasons.append(
                TakeProfitReason(
                    TakeProfitReasonType.GOOD_RISK_REWARD,
                    "The target provides a good risk/reward ratio.",
                )
            )

        # Add acceptable-RR classification.
        else:
            reasons.append(
                TakeProfitReason(
                    TakeProfitReasonType.ACCEPTABLE_RISK_REWARD,
                    (
                        "The target satisfies the minimum "
                        "risk/reward requirement."
                    ),
                )
            )

        # P2.15 does not calculate account-level risk.
        warnings = (
            "15R does not calculate position size or account risk.",
        )

        # Return the final immutable TP model.
        return TakeProfitModel(
            timestamp=entry.timestamp,
            symbol=entry.symbol,
            timeframe=entry.timeframe,
            direction=entry.direction,
            entry_price=round(
                effective_entry,
                10,
            ),
            stop_loss=round(
                stop_loss_value,
                10,
            ),
            take_profit=selected_target,
            risk_distance=round(
                risk_distance,
                10,
            ),
            reward_distance=round(
                reward_distance,
                10,
            ),
            risk_reward_ratio=round(
                risk_reward_ratio,
                10,
            ),
            minimum_risk_reward=minimum_rr,
            method=selected_method,
            quality=quality,
            quality_score=round(
                quality_score,
                10,
            ),
            take_profit_ready=True,
            valid=True,
            reasons=tuple(reasons),
            warnings=warnings,
        )

    def analyze_xauusd(
        self,
        entry: EntryModel,
        *,
        stop_loss: Optional[float] = None,
        structural_level: Optional[float] = None,
        resistance_level: Optional[float] = None,
        support_level: Optional[float] = None,
        trendline_level: Optional[float] = None,
        reference_price: Optional[float] = None,
        minimum_risk_reward: Optional[float] = None,
    ) -> TakeProfitModel:
        """Analyze an XAUUSD entry."""

        # Enforce the XAUUSD-specific API contract.
        if entry.symbol != "XAUUSD":
            raise TakeProfitIntelligenceError(
                "analyze_xauusd requires symbol XAUUSD."
            )

        # Delegate to the generic analyzer.
        return self.analyze(
            entry,
            stop_loss=stop_loss,
            structural_level=structural_level,
            resistance_level=resistance_level,
            support_level=support_level,
            trendline_level=trendline_level,
            reference_price=reference_price,
            minimum_risk_reward=minimum_risk_reward,
        )

    def _calculate_rr_target(
        self,
        entry_price: float,
        risk_distance: float,
        direction: EntryDirection,
        minimum_rr: float,
    ) -> float:
        """Calculate the deterministic minimum-RR fallback target."""

        # Convert risk into the required reward.
        reward_distance = (
            risk_distance * minimum_rr
        )

        # LONG TP must be above entry.
        if direction is EntryDirection.LONG:
            return entry_price + reward_distance

        # SHORT TP must be below entry.
        if direction is EntryDirection.SHORT:
            return entry_price - reward_distance

        # Defensive direction validation.
        raise TakeProfitIntelligenceError(
            "direction must be LONG or SHORT."
        )

    def _is_valid_target_side(
        self,
        entry_price: float,
        target_price: float,
        direction: EntryDirection,
    ) -> bool:
        """Check whether the target is on the profitable side."""

        # LONG target must be above entry.
        if direction is EntryDirection.LONG:
            return target_price > entry_price

        # SHORT target must be below entry.
        if direction is EntryDirection.SHORT:
            return target_price < entry_price

        # Unknown directions cannot have valid targets.
        return False

    def _calculate_quality_score(
        self,
        risk_reward_ratio: float,
        method: TakeProfitMethod,
    ) -> float:
        """Calculate deterministic take-profit quality."""

        # Establish a base score from RR.
        if risk_reward_ratio >= 4.0:
            score = 90.0

        elif risk_reward_ratio >= 3.0:
            score = 80.0

        elif risk_reward_ratio >= 2.0:
            score = 70.0

        else:
            score = 40.0

        # Reward market-structure agreement.
        if method is TakeProfitMethod.HYBRID:
            score += 10.0

        # Reward individual structure/SR/trendline targets.
        elif method in (
            TakeProfitMethod.STRUCTURE,
            TakeProfitMethod.SUPPORT_RESISTANCE,
            TakeProfitMethod.TRENDLINE,
        ):
            score += 5.0

        # Never allow scores above 100.
        return min(
            score,
            100.0,
        )

    def _quality_from_score(
        self,
        score: float,
    ) -> TakeProfitQuality:
        """Convert quality score into TakeProfitQuality."""

        # Highest-quality classification.
        if score >= self.excellent_score:
            return TakeProfitQuality.EXCELLENT

        # Good-quality classification.
        if score >= self.good_score:
            return TakeProfitQuality.GOOD

        # Acceptable-quality classification.
        if score >= self.acceptable_score:
            return TakeProfitQuality.ACCEPTABLE

        # Anything below acceptable is poor.
        return TakeProfitQuality.POOR

    def _invalid_result(
        self,
        *,
        entry: EntryModel,
        entry_price: float,
        stop_loss: Optional[float],
        minimum_rr: float,
        reason_type: TakeProfitReasonType,
        message: str,
    ) -> TakeProfitModel:
        """Build a deterministic invalid TP result."""

        # Preserve useful stop-loss/risk information where available.
        risk_distance = 0.0

        if stop_loss is not None:
            risk_distance = abs(
                entry.entry_price - stop_loss
            )

        # Return a blocked immutable model.
        return TakeProfitModel(
            timestamp=entry.timestamp,
            symbol=entry.symbol,
            timeframe=entry.timeframe,
            direction=entry.direction,
            entry_price=round(
                entry_price,
                10,
            ),
            stop_loss=(
                None
                if stop_loss is None
                else round(
                    stop_loss,
                    10,
                )
            ),
            take_profit=None,
            risk_distance=round(
                risk_distance,
                10,
            ),
            reward_distance=0.0,
            risk_reward_ratio=0.0,
            minimum_risk_reward=minimum_rr,
            method=TakeProfitMethod.NONE,
            quality=TakeProfitQuality.INVALID,
            quality_score=0.0,
            take_profit_ready=False,
            valid=False,
            reasons=(
                TakeProfitReason(
                    reason_type,
                    message,
                ),
            ),
            warnings=(
                "Take-profit calculation is blocked.",
            ),
        )

    def _validate_entry(
        self,
        entry: EntryModel,
    ) -> None:
        """Validate the EntryModel contract."""

        # Only EntryModel objects are supported.
        if not isinstance(entry, EntryModel):
            raise TakeProfitIntelligenceError(
                "entry must be an EntryModel."
            )

        # Validate symbol type.
        if not isinstance(entry.symbol, str):
            raise TakeProfitIntelligenceError(
                "entry.symbol must be a string."
            )

        # Symbol cannot be empty.
        if not entry.symbol.strip():
            raise TakeProfitIntelligenceError(
                "entry.symbol cannot be empty."
            )

        # Validate timeframe type.
        if not isinstance(entry.timeframe, str):
            raise TakeProfitIntelligenceError(
                "entry.timeframe must be a string."
            )

        # Timeframe cannot be empty.
        if not entry.timeframe.strip():
            raise TakeProfitIntelligenceError(
                "entry.timeframe cannot be empty."
            )

        # Validate EntryModel entry price.
        self._validate_price(
            entry.entry_price,
            "entry.entry_price",
        )

    def _validate_optional_price(
        self,
        value: Optional[float],
        name: str,
    ) -> None:
        """Validate an optional trading price."""

        # None means that the optional target was not supplied.
        if value is None:
            return

        # Otherwise validate the supplied price.
        self._validate_price(
            value,
            name,
        )

    def _validate_price(
        self,
        value: float,
        name: str,
    ) -> float:
        """
        Validate that a trading price is numeric, finite,
        and strictly greater than zero.
        """

        # bool is a subclass of int in Python, but it is not a price.
        if isinstance(value, bool):
            raise TakeProfitIntelligenceError(
                f"{name} must be a positive finite number."
            )

        # Only numeric values are accepted.
        if not isinstance(value, (int, float)):
            raise TakeProfitIntelligenceError(
                f"{name} must be a positive finite number."
            )

        # Convert to float for consistent financial calculations.
        numeric_value = float(value)

        # Reject NaN and infinity.
        if not math.isfinite(numeric_value):
            raise TakeProfitIntelligenceError(
                f"{name} must be a positive finite number."
            )

        # Zero and negative prices are invalid market prices.
        if numeric_value <= 0.0:
            raise TakeProfitIntelligenceError(
                f"{name} must be greater than zero."
            )

        # Return the validated numeric price.
        return numeric_value

    def _validate_positive(
        self,
        value: float,
        name: str,
    ) -> float:
        """Validate a strictly positive finite number."""

        # Reject booleans explicitly.
        if isinstance(value, bool):
            raise TakeProfitIntelligenceError(
                f"{name} must be greater than zero."
            )

        # Reuse finite-number validation.
        numeric_value = self._validate_price(
            value,
            name,
        )

        # Require strictly positive values.
        if numeric_value <= 0.0:
            raise TakeProfitIntelligenceError(
                f"{name} must be greater than zero."
            )

        return numeric_value

    def _validate_score(
        self,
        value: float,
        name: str,
    ) -> float:
        """Validate a quality score threshold."""

        # Reject booleans.
        if isinstance(value, bool):
            raise TakeProfitIntelligenceError(
                f"{name} must be between 0 and 100."
            )

        # Validate numeric and finite input.
        if not isinstance(value, (int, float)):
            raise TakeProfitIntelligenceError(
                f"{name} must be between 0 and 100."
            )

        numeric_value = float(value)

        # Reject NaN and infinity.
        if not math.isfinite(numeric_value):
            raise TakeProfitIntelligenceError(
                f"{name} must be between 0 and 100."
            )

        # Score thresholds must remain within 0-100.
        if not 0.0 <= numeric_value <= 100.0:
            raise TakeProfitIntelligenceError(
                f"{name} must be between 0 and 100."
            )

        return numeric_value