from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional

from app.trading.entry.entry_model import EntryDirection, EntryModel


class TakeProfitIntelligenceError(ValueError):
    """Raised when take-profit intelligence receives invalid input."""


class TakeProfitMethod(str, Enum):
    NONE = "NONE"
    RISK_REWARD = "RISK_REWARD"
    STRUCTURE = "STRUCTURE"
    SUPPORT_RESISTANCE = "SUPPORT_RESISTANCE"
    TRENDLINE = "TRENDLINE"
    HYBRID = "HYBRID"


class TakeProfitQuality(str, Enum):
    INVALID = "INVALID"
    POOR = "POOR"
    ACCEPTABLE = "ACCEPTABLE"
    GOOD = "GOOD"
    EXCELLENT = "EXCELLENT"


class TakeProfitReasonType(str, Enum):
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
    reason_type: TakeProfitReasonType
    message: str


@dataclass(frozen=True, slots=True)
class TakeProfitModel:
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
        return self.direction is EntryDirection.LONG

    @property
    def is_short(self) -> bool:
        return self.direction is EntryDirection.SHORT

    @property
    def has_take_profit(self) -> bool:
        return self.take_profit is not None

    @property
    def is_ready(self) -> bool:
        return self.take_profit_ready

    @property
    def reward(self) -> float:
        return self.reward_distance

    @property
    def rr(self) -> float:
        return self.risk_reward_ratio


class TakeProfitIntelligenceEngine:
    """
    Determines a take-profit target from:

    1. Structural level
    2. Support/resistance
    3. Trendline
    4. Minimum risk/reward fallback

    This engine does not calculate position size or account risk.
    """

    def __init__(
        self,
        minimum_risk_reward: float = 2.0,
        minimum_distance: float = 0.0001,
        maximum_risk_reward: float = 10.0,
        excellent_score: float = 90.0,
        good_score: float = 70.0,
        acceptable_score: float = 50.0,
    ) -> None:
        self.minimum_risk_reward = self._validate_positive(
            minimum_risk_reward,
            "minimum_risk_reward",
        )
        self.minimum_distance = self._validate_positive(
            minimum_distance,
            "minimum_distance",
        )
        self.maximum_risk_reward = self._validate_positive(
            maximum_risk_reward,
            "maximum_risk_reward",
        )

        if self.minimum_risk_reward > self.maximum_risk_reward:
            raise TakeProfitIntelligenceError(
                "minimum_risk_reward cannot exceed maximum_risk_reward."
            )

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
        self._validate_entry(entry)

        minimum_rr = (
            self.minimum_risk_reward
            if minimum_risk_reward is None
            else self._validate_positive(
                minimum_risk_reward,
                "minimum_risk_reward",
            )
        )

        if minimum_rr > self.maximum_risk_reward:
            raise TakeProfitIntelligenceError(
                "minimum_risk_reward cannot exceed maximum_risk_reward."
            )

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

        original_entry = self._validate_price(
            entry.entry_price,
            "entry.entry_price",
        )

        effective_entry = original_entry

        if reference_price is not None:
            effective_entry = self._validate_price(
                reference_price,
                "reference_price",
            )

        if stop_loss is None:
            return self._invalid_result(
                entry=entry,
                entry_price=effective_entry,
                stop_loss=None,
                minimum_rr=minimum_rr,
                reason_type=TakeProfitReasonType.INVALID_STOP_LOSS,
                message="A valid stop-loss is required before calculating take-profit.",
            )

        stop_loss_value = self._validate_price(
            stop_loss,
            "stop_loss",
        )

        direction = entry.direction

        if direction is EntryDirection.NONE or direction is EntryDirection.UNKNOWN:
            return self._invalid_result(
                entry=entry,
                entry_price=effective_entry,
                stop_loss=stop_loss_value,
                minimum_rr=minimum_rr,
                reason_type=TakeProfitReasonType.INVALID_DIRECTION,
                message="Entry direction must be LONG or SHORT.",
            )

        # Keep the original entry for determining the stop distance.
        #
        # This preserves the semantics of an entry/reference-price override:
        # the override changes the actual target reference price while the
        # supplied stop remains associated with the original entry model.
        calculation_entry = original_entry

        if direction is EntryDirection.LONG:
            if stop_loss_value >= calculation_entry:
                return self._invalid_result(
                    entry=entry,
                    entry_price=effective_entry,
                    stop_loss=stop_loss_value,
                    minimum_rr=minimum_rr,
                    reason_type=TakeProfitReasonType.INVALID_STOP_LOSS,
                    message="For a LONG entry, stop-loss must be below the entry price.",
                )
        elif direction is EntryDirection.SHORT:
            if stop_loss_value <= calculation_entry:
                return self._invalid_result(
                    entry=entry,
                    entry_price=effective_entry,
                    stop_loss=stop_loss_value,
                    minimum_rr=minimum_rr,
                    reason_type=TakeProfitReasonType.INVALID_STOP_LOSS,
                    message="For a SHORT entry, stop-loss must be above the entry price.",
                )

        risk_distance = abs(calculation_entry - stop_loss_value)

        if risk_distance <= self.minimum_distance:
            return self._invalid_result(
                entry=entry,
                entry_price=effective_entry,
                stop_loss=stop_loss_value,
                minimum_rr=minimum_rr,
                reason_type=TakeProfitReasonType.INVALID_STOP_LOSS,
                message="Stop-loss is too close to the entry price.",
            )

        rr_target = self._calculate_rr_target(
            effective_entry,
            risk_distance,
            direction,
            minimum_rr,
        )

        candidates = []

        if structural_level is not None:
            candidates.append(
                (
                    structural_level,
                    TakeProfitMethod.STRUCTURE,
                    TakeProfitReasonType.STRUCTURAL_TARGET,
                    "A valid structural target was identified.",
                )
            )

        if resistance_level is not None:
            candidates.append(
                (
                    resistance_level,
                    TakeProfitMethod.SUPPORT_RESISTANCE,
                    TakeProfitReasonType.SUPPORT_RESISTANCE_TARGET,
                    "A valid resistance target was identified.",
                )
            )

        if support_level is not None:
            candidates.append(
                (
                    support_level,
                    TakeProfitMethod.SUPPORT_RESISTANCE,
                    TakeProfitReasonType.SUPPORT_RESISTANCE_TARGET,
                    "A valid support target was identified.",
                )
            )

        if trendline_level is not None:
            candidates.append(
                (
                    trendline_level,
                    TakeProfitMethod.TRENDLINE,
                    TakeProfitReasonType.TRENDLINE_TARGET,
                    "A valid trendline target was identified.",
                )
            )

        valid_candidates = []

        for (
            level,
            method,
            reason_type,
            message,
        ) in candidates:
            if self._is_valid_target_side(
                effective_entry,
                level,
                direction,
            ):
                reward = abs(level - effective_entry)

                if reward + self.minimum_distance >= (
                    risk_distance * minimum_rr
                ):
                    valid_candidates.append(
                        (
                            level,
                            method,
                            reason_type,
                            message,
                            reward,
                        )
                    )

        selected_target = None
        selected_method = TakeProfitMethod.RISK_REWARD
        selected_reason_type = TakeProfitReasonType.RISK_REWARD_TARGET
        selected_message = (
            "Take-profit is calculated from the configured "
            "minimum risk/reward requirement."
        )

        if valid_candidates:
            valid_candidates.sort(
                key=lambda item: item[4],
            )

            selected_target = valid_candidates[0][0]
            selected_method = valid_candidates[0][1]
            selected_reason_type = valid_candidates[0][2]
            selected_message = valid_candidates[0][3]

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

            if (
                TakeProfitMethod.STRUCTURE in same_level_methods
                and TakeProfitMethod.SUPPORT_RESISTANCE
                in same_level_methods
            ):
                selected_method = TakeProfitMethod.HYBRID
                selected_reason_type = TakeProfitReasonType.HYBRID_TARGET
                selected_message = (
                    "Structural and support/resistance targets agree "
                    "on the same take-profit level."
                )
        else:
            selected_target = rr_target

        selected_target = round(selected_target, 10)

        reward_distance = abs(
            selected_target - effective_entry
        )

        if risk_distance <= 0.0:
            raise TakeProfitIntelligenceError(
                "risk distance must be greater than zero."
            )

        risk_reward_ratio = reward_distance / risk_distance

        if risk_reward_ratio + self.minimum_distance < minimum_rr:
            return self._invalid_result(
                entry=entry,
                entry_price=effective_entry,
                stop_loss=stop_loss_value,
                minimum_rr=minimum_rr,
                reason_type=TakeProfitReasonType.TARGET_TOO_CLOSE,
                message=(
                    "The selected target does not satisfy the configured "
                    "minimum risk/reward requirement."
                ),
            )

        quality_score = self._calculate_quality_score(
            risk_reward_ratio,
            selected_method,
        )

        quality = self._quality_from_score(
            quality_score,
        )

        reasons = [
            TakeProfitReason(
                selected_reason_type,
                selected_message,
            ),
        ]

        if risk_reward_ratio >= 4.0:
            reasons.append(
                TakeProfitReason(
                    TakeProfitReasonType.HIGH_RISK_REWARD,
                    "The target provides a high risk/reward ratio.",
                )
            )
        elif risk_reward_ratio >= 3.0:
            reasons.append(
                TakeProfitReason(
                    TakeProfitReasonType.GOOD_RISK_REWARD,
                    "The target provides a good risk/reward ratio.",
                )
            )
        else:
            reasons.append(
                TakeProfitReason(
                    TakeProfitReasonType.ACCEPTABLE_RISK_REWARD,
                    "The target satisfies the minimum risk/reward requirement.",
                )
            )

        warnings = (
            "15R does not calculate position size or account risk.",
        )

        return TakeProfitModel(
            timestamp=entry.timestamp,
            symbol=entry.symbol,
            timeframe=entry.timeframe,
            direction=entry.direction,
            entry_price=round(effective_entry, 10),
            stop_loss=round(stop_loss_value, 10),
            take_profit=selected_target,
            risk_distance=round(risk_distance, 10),
            reward_distance=round(reward_distance, 10),
            risk_reward_ratio=round(risk_reward_ratio, 10),
            minimum_risk_reward=minimum_rr,
            method=selected_method,
            quality=quality,
            quality_score=round(quality_score, 10),
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
        if entry.symbol != "XAUUSD":
            raise TakeProfitIntelligenceError(
                "analyze_xauusd requires symbol XAUUSD."
            )

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
        reward_distance = risk_distance * minimum_rr

        if direction is EntryDirection.LONG:
            return entry_price + reward_distance

        if direction is EntryDirection.SHORT:
            return entry_price - reward_distance

        raise TakeProfitIntelligenceError(
            "invalid direction."
        )

    def _is_valid_target_side(
        self,
        entry_price: float,
        target: float,
        direction: EntryDirection,
    ) -> bool:
        if direction is EntryDirection.LONG:
            return target > entry_price + self.minimum_distance

        if direction is EntryDirection.SHORT:
            return target < entry_price - self.minimum_distance

        return False

    def _calculate_quality_score(
        self,
        risk_reward_ratio: float,
        method: TakeProfitMethod,
    ) -> float:
        if risk_reward_ratio >= 4.0:
            score = 85.0
        elif risk_reward_ratio >= 3.0:
            score = 75.0
        elif risk_reward_ratio >= 2.0:
            score = 60.0
        else:
            score = 40.0

        if method is TakeProfitMethod.STRUCTURE:
            score += 5.0
        elif method is TakeProfitMethod.SUPPORT_RESISTANCE:
            score += 5.0
        elif method is TakeProfitMethod.TRENDLINE:
            score += 5.0
        elif method is TakeProfitMethod.HYBRID:
            score += 15.0

        return min(100.0, score)

    def _quality_from_score(
        self,
        score: float,
    ) -> TakeProfitQuality:
        if score >= self.excellent_score:
            return TakeProfitQuality.EXCELLENT

        if score >= self.good_score:
            return TakeProfitQuality.GOOD

        if score >= self.acceptable_score:
            return TakeProfitQuality.ACCEPTABLE

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
        return TakeProfitModel(
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
            take_profit=None,
            risk_distance=0.0,
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
                "15R does not calculate position size or account risk.",
            ),
        )

    def _validate_entry(
        self,
        entry: EntryModel,
    ) -> None:
        if not isinstance(entry, EntryModel):
            raise TakeProfitIntelligenceError(
                "entry must be an EntryModel."
            )

        if not isinstance(entry.symbol, str) or not entry.symbol.strip():
            raise TakeProfitIntelligenceError(
                "entry symbol cannot be empty."
            )

        if not isinstance(entry.timeframe, str) or not entry.timeframe.strip():
            raise TakeProfitIntelligenceError(
                "entry timeframe cannot be empty."
            )

        self._validate_price(
            entry.entry_price,
            "entry.entry_price",
        )

    def _validate_optional_price(
        self,
        value: Optional[float],
        name: str,
    ) -> None:
        if value is None:
            return

        self._validate_price(
            value,
            name,
        )

    def _validate_price(
        self,
        value: float,
        name: str,
    ) -> float:
        if isinstance(value, bool):
            raise TakeProfitIntelligenceError(
                f"{name} must be a finite positive number."
            )

        if not isinstance(value, (int, float)):
            raise TakeProfitIntelligenceError(
                f"{name} must be a finite positive number."
            )

        value = float(value)

        if not math.isfinite(value):
            raise TakeProfitIntelligenceError(
                f"{name} must be finite."
            )

        if value <= 0.0:
            raise TakeProfitIntelligenceError(
                f"{name} must be greater than zero."
            )

        return value

    def _validate_positive(
        self,
        value: float,
        name: str,
    ) -> float:
        if isinstance(value, bool):
            raise TakeProfitIntelligenceError(
                f"{name} must be a finite positive number."
            )

        if not isinstance(value, (int, float)):
            raise TakeProfitIntelligenceError(
                f"{name} must be a finite positive number."
            )

        value = float(value)

        if not math.isfinite(value) or value <= 0.0:
            raise TakeProfitIntelligenceError(
                f"{name} must be a finite positive number."
            )

        return value

    def _validate_score(
        self,
        value: float,
        name: str,
    ) -> float:
        value = self._validate_positive(
            value,
            name,
        )

        if value > 100.0:
            raise TakeProfitIntelligenceError(
                f"{name} must be between 0 and 100."
            )

        return value