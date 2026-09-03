from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from app.trading.entry.entry_model import (
    EntryDirection,
    EntryModel,
)


class StopLossIntelligenceError(ValueError):
    """Raised when stop-loss intelligence validation fails."""


class StopLossMethod(str, Enum):
    STRUCTURE = "STRUCTURE"
    SUPPORT_RESISTANCE = "SUPPORT_RESISTANCE"
    TRENDLINE = "TRENDLINE"
    ATR_BUFFER = "ATR_BUFFER"
    HYBRID = "HYBRID"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class StopLossQuality(str, Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    WEAK = "WEAK"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class StopLossReasonType(str, Enum):
    LONG_DIRECTION = "LONG_DIRECTION"
    SHORT_DIRECTION = "SHORT_DIRECTION"

    STRUCTURE_SUPPORT = "STRUCTURE_SUPPORT"
    STRUCTURE_RESISTANCE = "STRUCTURE_RESISTANCE"

    SUPPORT_LEVEL = "SUPPORT_LEVEL"
    RESISTANCE_LEVEL = "RESISTANCE_LEVEL"

    TRENDLINE_SUPPORT = "TRENDLINE_SUPPORT"
    TRENDLINE_RESISTANCE = "TRENDLINE_RESISTANCE"

    ATR_BUFFER = "ATR_BUFFER"

    HYBRID_CONFIRMATION = "HYBRID_CONFIRMATION"

    SUFFICIENT_DATA = "SUFFICIENT_DATA"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

    VALID_LONG_STOP = "VALID_LONG_STOP"
    VALID_SHORT_STOP = "VALID_SHORT_STOP"

    STOP_TOO_CLOSE = "STOP_TOO_CLOSE"
    STOP_TOO_FAR = "STOP_TOO_FAR"
    STOP_WRONG_SIDE = "STOP_WRONG_SIDE"

    INVALID_ENTRY = "INVALID_ENTRY"
    INVALID_DIRECTION = "INVALID_DIRECTION"

    STOP_LOSS_READY = "STOP_LOSS_READY"
    STOP_LOSS_BLOCKED = "STOP_LOSS_BLOCKED"


@dataclass(frozen=True, slots=True)
class StopLossReason:
    reason_type: StopLossReasonType
    message: str


@dataclass(frozen=True, slots=True)
class StopLossModel:
    timestamp: object
    symbol: str
    timeframe: str

    direction: EntryDirection

    entry_price: float
    stop_loss: float | None

    risk_distance: float | None
    risk_percent_of_entry: float | None

    method: StopLossMethod
    quality: StopLossQuality
    quality_score: float

    structural_level: float | None
    support_level: float | None
    resistance_level: float | None
    trendline_level: float | None
    atr_value: float | None
    atr_buffer: float

    valid: bool
    stop_loss_ready: bool

    reasons: tuple[StopLossReason, ...]
    warnings: tuple[str, ...]

    @property
    def is_long(self) -> bool:
        return self.direction is EntryDirection.LONG

    @property
    def is_short(self) -> bool:
        return self.direction is EntryDirection.SHORT

    @property
    def is_valid(self) -> bool:
        return self.valid

    @property
    def has_stop_loss(self) -> bool:
        return self.stop_loss is not None

    @property
    def is_ready(self) -> bool:
        return self.stop_loss_ready

    @property
    def risk(self) -> float | None:
        return self.risk_distance


class StopLossIntelligenceEngine:
    """
    Determines technically valid stop-loss placement.

    Responsibilities:
    - validate the entry model
    - determine the correct side of the market for the stop
    - use structural/support/resistance information
    - optionally apply an ATR buffer
    - calculate risk distance
    - score stop-loss quality

    This engine does NOT:
    - calculate position size
    - calculate take profit
    - calculate reward/risk ratio
    - execute trades
    - fetch market data
    - fetch news
    - call the LLM
    """

    DEFAULT_MINIMUM_DISTANCE = 0.01
    DEFAULT_MAXIMUM_RISK_PERCENT = 5.0

    DEFAULT_ATR_BUFFER_MULTIPLIER = 0.25

    DEFAULT_EXCELLENT_SCORE = 85.0
    DEFAULT_GOOD_SCORE = 70.0
    DEFAULT_ACCEPTABLE_SCORE = 55.0

    def __init__(
        self,
        minimum_distance: float = DEFAULT_MINIMUM_DISTANCE,
        maximum_risk_percent: float = DEFAULT_MAXIMUM_RISK_PERCENT,
        atr_buffer_multiplier: float = DEFAULT_ATR_BUFFER_MULTIPLIER,
        excellent_score: float = DEFAULT_EXCELLENT_SCORE,
        good_score: float = DEFAULT_GOOD_SCORE,
        acceptable_score: float = DEFAULT_ACCEPTABLE_SCORE,
    ) -> None:
        self.minimum_distance = self._validate_positive(
            minimum_distance,
            "minimum_distance",
        )

        self.maximum_risk_percent = self._validate_positive(
            maximum_risk_percent,
            "maximum_risk_percent",
        )

        self.atr_buffer_multiplier = self._validate_non_negative(
            atr_buffer_multiplier,
            "atr_buffer_multiplier",
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
            self.acceptable_score
            <= self.good_score
            <= self.excellent_score
        ):
            raise StopLossIntelligenceError(
                "quality scores must satisfy "
                "acceptable <= good <= excellent."
            )

    def analyze(
        self,
        entry: EntryModel,
        structural_level: float | None = None,
        support_level: float | None = None,
        resistance_level: float | None = None,
        trendline_level: float | None = None,
        atr_value: float | None = None,
        reference_price: float | None = None,
    ) -> StopLossModel:
        self._validate_entry(entry)

        values = {
            "structural_level": structural_level,
            "support_level": support_level,
            "resistance_level": resistance_level,
            "trendline_level": trendline_level,
            "atr_value": atr_value,
            "reference_price": reference_price,
        }

        self._validate_optional_values(values)

        entry_price = (
            float(reference_price)
            if reference_price is not None
            else self._get_entry_price(entry)
        )

        direction = entry.direction

        if direction not in (
            EntryDirection.LONG,
            EntryDirection.SHORT,
        ):
            return self._blocked_result(
                entry=entry,
                entry_price=entry_price,
                structural_level=structural_level,
                support_level=support_level,
                resistance_level=resistance_level,
                trendline_level=trendline_level,
                atr_value=atr_value,
                reason_type=(
                    StopLossReasonType.INVALID_DIRECTION
                ),
                message=(
                    "A long or short direction is required "
                    "for stop-loss analysis."
                ),
            )

        stop_loss, method, structural_reference = (
            self._determine_stop_loss(
                direction=direction,
                entry_price=entry_price,
                structural_level=structural_level,
                support_level=support_level,
                resistance_level=resistance_level,
                trendline_level=trendline_level,
                atr_value=atr_value,
            )
        )

        if stop_loss is None:
            return self._blocked_result(
                entry=entry,
                entry_price=entry_price,
                structural_level=structural_level,
                support_level=support_level,
                resistance_level=resistance_level,
                trendline_level=trendline_level,
                atr_value=atr_value,
                reason_type=(
                    StopLossReasonType.INSUFFICIENT_DATA
                ),
                message=(
                    "No valid structural or technical "
                    "stop-loss reference is available."
                ),
            )

        stop_loss, buffer = self._apply_atr_buffer(
            direction=direction,
            stop_loss=stop_loss,
            atr_value=atr_value,
        )

        valid, validation_reason = self._validate_stop_position(
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
        )

        risk_distance = abs(
            entry_price - stop_loss
        )

        risk_percent = (
            risk_distance / entry_price
        ) * 100.0

        quality_score = self._calculate_quality_score(
            method=method,
            direction=direction,
            structural_reference=structural_reference,
            support_level=support_level,
            resistance_level=resistance_level,
            trendline_level=trendline_level,
            atr_value=atr_value,
            valid=valid,
            risk_percent=risk_percent,
        )

        quality = self._classify_quality(
            quality_score=quality_score,
            valid=valid,
        )

        reasons = self._build_reasons(
            direction=direction,
            method=method,
            valid=valid,
            structural_reference=structural_reference,
            support_level=support_level,
            resistance_level=resistance_level,
            trendline_level=trendline_level,
            atr_value=atr_value,
            buffer=buffer,
            risk_percent=risk_percent,
            validation_reason=validation_reason,
            quality=quality,
        )

        warnings = self._build_warnings(
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            risk_percent=risk_percent,
            atr_value=atr_value,
            valid=valid,
        )

        stop_loss_ready = (
            valid
            and quality
            not in (
                StopLossQuality.INVALID,
                StopLossQuality.UNKNOWN,
                StopLossQuality.WEAK,
            )
        )

        return StopLossModel(
            timestamp=entry.timestamp,
            symbol=entry.symbol,
            timeframe=entry.timeframe,
            direction=direction,
            entry_price=round(entry_price, 10),
            stop_loss=round(stop_loss, 10)
            if valid
            else None,
            risk_distance=round(risk_distance, 10)
            if valid
            else None,
            risk_percent_of_entry=round(risk_percent, 6)
            if valid
            else None,
            method=method,
            quality=quality,
            quality_score=round(
                quality_score,
                6,
            ),
            structural_level=structural_level,
            support_level=support_level,
            resistance_level=resistance_level,
            trendline_level=trendline_level,
            atr_value=atr_value,
            atr_buffer=round(buffer, 10),
            valid=valid,
            stop_loss_ready=stop_loss_ready,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
        )

    def analyze_xauusd(
        self,
        entry: EntryModel,
        structural_level: float | None = None,
        support_level: float | None = None,
        resistance_level: float | None = None,
        trendline_level: float | None = None,
        atr_value: float | None = None,
        reference_price: float | None = None,
    ) -> StopLossModel:
        if entry.symbol != "XAUUSD":
            raise StopLossIntelligenceError(
                "entry must use XAUUSD."
            )

        return self.analyze(
            entry=entry,
            structural_level=structural_level,
            support_level=support_level,
            resistance_level=resistance_level,
            trendline_level=trendline_level,
            atr_value=atr_value,
            reference_price=reference_price,
        )

    def _determine_stop_loss(
        self,
        direction: EntryDirection,
        entry_price: float,
        structural_level: float | None,
        support_level: float | None,
        resistance_level: float | None,
        trendline_level: float | None,
        atr_value: float | None,
    ) -> tuple[float | None, StopLossMethod, float | None]:

        if direction is EntryDirection.LONG:
            candidates: list[
                tuple[float, StopLossMethod]
            ] = []

            if structural_level is not None:
                candidates.append(
                    (
                        structural_level,
                        StopLossMethod.STRUCTURE,
                    )
                )

            if support_level is not None:
                candidates.append(
                    (
                        support_level,
                        StopLossMethod.SUPPORT_RESISTANCE,
                    )
                )

            if trendline_level is not None:
                candidates.append(
                    (
                        trendline_level,
                        StopLossMethod.TRENDLINE,
                    )
                )

            valid_candidates = [
                item
                for item in candidates
                if item[0] < entry_price
            ]

            if not valid_candidates:
                return (
                    None,
                    StopLossMethod.NONE,
                    None,
                )

            # For a long trade the nearest valid structural
            # invalidation level below entry is preferred.
            selected = max(
                valid_candidates,
                key=lambda item: item[0],
            )

            method = selected[1]

            if (
                structural_level is not None
                and support_level is not None
                and structural_level == support_level
            ):
                method = StopLossMethod.HYBRID

            return (
                selected[0],
                method,
                selected[0],
            )

        if direction is EntryDirection.SHORT:
            candidates = []

            if structural_level is not None:
                candidates.append(
                    (
                        structural_level,
                        StopLossMethod.STRUCTURE,
                    )
                )

            if resistance_level is not None:
                candidates.append(
                    (
                        resistance_level,
                        StopLossMethod.SUPPORT_RESISTANCE,
                    )
                )

            if trendline_level is not None:
                candidates.append(
                    (
                        trendline_level,
                        StopLossMethod.TRENDLINE,
                    )
                )

            valid_candidates = [
                item
                for item in candidates
                if item[0] > entry_price
            ]

            if not valid_candidates:
                return (
                    None,
                    StopLossMethod.NONE,
                    None,
                )

            # For a short trade the nearest valid structural
            # invalidation level above entry is preferred.
            selected = min(
                valid_candidates,
                key=lambda item: item[0],
            )

            method = selected[1]

            if (
                structural_level is not None
                and resistance_level is not None
                and structural_level == resistance_level
            ):
                method = StopLossMethod.HYBRID

            return (
                selected[0],
                method,
                selected[0],
            )

        return (
            None,
            StopLossMethod.UNKNOWN,
            None,
        )

    def _apply_atr_buffer(
        self,
        direction: EntryDirection,
        stop_loss: float,
        atr_value: float | None,
    ) -> tuple[float, float]:

        if atr_value is None:
            return stop_loss, 0.0

        buffer = (
            atr_value
            * self.atr_buffer_multiplier
        )

        if direction is EntryDirection.LONG:
            return (
                stop_loss - buffer,
                buffer,
            )

        if direction is EntryDirection.SHORT:
            return (
                stop_loss + buffer,
                buffer,
            )

        return stop_loss, 0.0

    def _validate_stop_position(
        self,
        direction: EntryDirection,
        entry_price: float,
        stop_loss: float,
    ) -> tuple[bool, StopLossReasonType]:

        distance = abs(
            entry_price - stop_loss
        )

        if distance < self.minimum_distance:
            return (
                False,
                StopLossReasonType.STOP_TOO_CLOSE,
            )

        if direction is EntryDirection.LONG:
            if stop_loss >= entry_price:
                return (
                    False,
                    StopLossReasonType.STOP_WRONG_SIDE,
                )

        elif direction is EntryDirection.SHORT:
            if stop_loss <= entry_price:
                return (
                    False,
                    StopLossReasonType.STOP_WRONG_SIDE,
                )

        return (
            True,
            StopLossReasonType.STOP_LOSS_READY,
        )

    def _calculate_quality_score(
        self,
        method: StopLossMethod,
        direction: EntryDirection,
        structural_reference: float | None,
        support_level: float | None,
        resistance_level: float | None,
        trendline_level: float | None,
        atr_value: float | None,
        valid: bool,
        risk_percent: float,
    ) -> float:

        if not valid:
            return 0.0

        score = 40.0

        if method is StopLossMethod.STRUCTURE:
            score += 30.0

        elif method is StopLossMethod.SUPPORT_RESISTANCE:
            score += 25.0

        elif method is StopLossMethod.TRENDLINE:
            score += 20.0

        elif method is StopLossMethod.HYBRID:
            score += 35.0

        elif method is StopLossMethod.ATR_BUFFER:
            score += 15.0

        if structural_reference is not None:
            score += 5.0

        if direction is EntryDirection.LONG:
            if support_level is not None:
                score += 5.0

        elif direction is EntryDirection.SHORT:
            if resistance_level is not None:
                score += 5.0

        if trendline_level is not None:
            score += 5.0

        if atr_value is not None:
            score += 5.0

        if risk_percent > self.maximum_risk_percent:
            score -= 30.0

        elif risk_percent > (
            self.maximum_risk_percent * 0.75
        ):
            score -= 15.0

        return max(
            0.0,
            min(100.0, score),
        )

    def _classify_quality(
        self,
        quality_score: float,
        valid: bool,
    ) -> StopLossQuality:

        if not valid:
            return StopLossQuality.INVALID

        if quality_score >= self.excellent_score:
            return StopLossQuality.EXCELLENT

        if quality_score >= self.good_score:
            return StopLossQuality.GOOD

        if quality_score >= self.acceptable_score:
            return StopLossQuality.ACCEPTABLE

        return StopLossQuality.WEAK

    def _build_reasons(
        self,
        direction: EntryDirection,
        method: StopLossMethod,
        valid: bool,
        structural_reference: float | None,
        support_level: float | None,
        resistance_level: float | None,
        trendline_level: float | None,
        atr_value: float | None,
        buffer: float,
        risk_percent: float,
        validation_reason: StopLossReasonType,
        quality: StopLossQuality,
    ) -> list[StopLossReason]:

        reasons: list[StopLossReason] = []

        if direction is EntryDirection.LONG:
            reasons.append(
                StopLossReason(
                    StopLossReasonType.LONG_DIRECTION,
                    "Long direction requires the stop below entry.",
                )
            )

        elif direction is EntryDirection.SHORT:
            reasons.append(
                StopLossReason(
                    StopLossReasonType.SHORT_DIRECTION,
                    "Short direction requires the stop above entry.",
                )
            )

        if structural_reference is not None:
            reason_type = (
                StopLossReasonType.STRUCTURE_SUPPORT
                if direction is EntryDirection.LONG
                else StopLossReasonType.STRUCTURE_RESISTANCE
            )

            reasons.append(
                StopLossReason(
                    reason_type,
                    "A structural invalidation level is available.",
                )
            )

        if (
            direction is EntryDirection.LONG
            and support_level is not None
        ):
            reasons.append(
                StopLossReason(
                    StopLossReasonType.SUPPORT_LEVEL,
                    "Support provides a long-side invalidation reference.",
                )
            )

        if (
            direction is EntryDirection.SHORT
            and resistance_level is not None
        ):
            reasons.append(
                StopLossReason(
                    StopLossReasonType.RESISTANCE_LEVEL,
                    "Resistance provides a short-side invalidation reference.",
                )
            )

        if trendline_level is not None:
            reasons.append(
                StopLossReason(
                    (
                        StopLossReasonType.TRENDLINE_SUPPORT
                        if direction is EntryDirection.LONG
                        else StopLossReasonType.TRENDLINE_RESISTANCE
                    ),
                    "Trendline structure contributes to stop placement.",
                )
            )

        if atr_value is not None and buffer > 0.0:
            reasons.append(
                StopLossReason(
                    StopLossReasonType.ATR_BUFFER,
                    "ATR buffering was applied to reduce premature stop hits.",
                )
            )

        if method is StopLossMethod.HYBRID:
            reasons.append(
                StopLossReason(
                    StopLossReasonType.HYBRID_CONFIRMATION,
                    "Multiple structural references agree on stop placement.",
                )
            )

        if valid:
            reasons.append(
                StopLossReason(
                    StopLossReasonType.VALID_LONG_STOP
                    if direction is EntryDirection.LONG
                    else StopLossReasonType.VALID_SHORT_STOP,
                    "Stop-loss is on the correct side of the entry.",
                )
            )
        else:
            reasons.append(
                StopLossReason(
                    validation_reason,
                    "Stop-loss placement failed validation.",
                )
            )

        if risk_percent <= self.maximum_risk_percent:
            reasons.append(
                StopLossReason(
                    StopLossReasonType.STOP_LOSS_READY
                    if quality
                    not in (
                        StopLossQuality.INVALID,
                        StopLossQuality.UNKNOWN,
                    )
                    else StopLossReasonType.STOP_LOSS_BLOCKED,
                    (
                        "Stop-loss distance is within the configured "
                        "technical risk boundary."
                    ),
                )
            )
        else:
            reasons.append(
                StopLossReason(
                    StopLossReasonType.STOP_TOO_FAR,
                    "Stop-loss distance exceeds the configured risk boundary.",
                )
            )

        return reasons

    def _build_warnings(
        self,
        direction: EntryDirection,
        entry_price: float,
        stop_loss: float,
        risk_percent: float,
        atr_value: float | None,
        valid: bool,
    ) -> list[str]:

        warnings: list[str] = []

        if not valid:
            warnings.append(
                "Stop-loss is not technically valid."
            )

        if risk_percent > self.maximum_risk_percent:
            warnings.append(
                "Stop-loss distance exceeds the configured maximum risk percentage."
            )

        if atr_value is None:
            warnings.append(
                "ATR was not supplied; no ATR buffer was applied."
            )

        warnings.append(
            "15Q does not calculate position size or take profit."
        )

        return warnings

    def _blocked_result(
        self,
        entry: EntryModel,
        entry_price: float,
        structural_level: float | None,
        support_level: float | None,
        resistance_level: float | None,
        trendline_level: float | None,
        atr_value: float | None,
        reason_type: StopLossReasonType,
        message: str,
    ) -> StopLossModel:

        reason = StopLossReason(
            reason_type,
            message,
        )

        return StopLossModel(
            timestamp=entry.timestamp,
            symbol=entry.symbol,
            timeframe=entry.timeframe,
            direction=entry.direction,
            entry_price=round(
                entry_price,
                10,
            ),
            stop_loss=None,
            risk_distance=None,
            risk_percent_of_entry=None,
            method=StopLossMethod.NONE,
            quality=StopLossQuality.INVALID,
            quality_score=0.0,
            structural_level=structural_level,
            support_level=support_level,
            resistance_level=resistance_level,
            trendline_level=trendline_level,
            atr_value=atr_value,
            atr_buffer=0.0,
            valid=False,
            stop_loss_ready=False,
            reasons=(
                reason,
                StopLossReason(
                    StopLossReasonType.STOP_LOSS_BLOCKED,
                    "Stop-loss generation is blocked.",
                ),
            ),
            warnings=(
                message,
                "15Q does not calculate position size or take profit.",
            ),
        )

    @staticmethod
    def _get_entry_price(
        entry: EntryModel,
    ) -> float:

        if entry.entry_price is None:
            raise StopLossIntelligenceError(
                "entry model does not contain an entry price."
            )

        price = float(entry.entry_price)

        if not isfinite(price) or price <= 0.0:
            raise StopLossIntelligenceError(
                "entry price must be finite and greater than zero."
            )

        return price

    @staticmethod
    def _validate_entry(
        entry: EntryModel,
    ) -> None:

        if not isinstance(entry, EntryModel):
            raise StopLossIntelligenceError(
                "entry must be an EntryModel."
            )

        if not isinstance(entry.symbol, str):
            raise StopLossIntelligenceError(
                "entry symbol must be a string."
            )

        if not entry.symbol.strip():
            raise StopLossIntelligenceError(
                "entry symbol cannot be empty."
            )

        if not isinstance(entry.timeframe, str):
            raise StopLossIntelligenceError(
                "entry timeframe must be a string."
            )

        if not entry.timeframe.strip():
            raise StopLossIntelligenceError(
                "entry timeframe cannot be empty."
            )

        if entry.direction not in (
            EntryDirection.LONG,
            EntryDirection.SHORT,
            EntryDirection.NONE,
            EntryDirection.UNKNOWN,
        ):
            raise StopLossIntelligenceError(
                "entry direction is invalid."
            )

    @staticmethod
    def _validate_optional_values(
        values: dict[str, float | None],
    ) -> None:

        for name, value in values.items():
            if value is None:
                continue

            if isinstance(value, bool):
                raise StopLossIntelligenceError(
                    f"{name} cannot be boolean."
                )

            if not isinstance(
                value,
                (int, float),
            ):
                raise StopLossIntelligenceError(
                    f"{name} must be numeric."
                )

            if not isfinite(float(value)):
                raise StopLossIntelligenceError(
                    f"{name} must be finite."
                )

            if name == "atr_value":
                if value <= 0.0:
                    raise StopLossIntelligenceError(
                        "atr_value must be greater than zero."
                    )

            elif name == "reference_price":
                if value <= 0.0:
                    raise StopLossIntelligenceError(
                        "reference_price must be greater than zero."
                    )

            else:
                if value <= 0.0:
                    raise StopLossIntelligenceError(
                        f"{name} must be greater than zero."
                    )

    @staticmethod
    def _validate_positive(
        value: float,
        name: str,
    ) -> float:

        if isinstance(value, bool):
            raise StopLossIntelligenceError(
                f"{name} cannot be boolean."
            )

        if not isinstance(
            value,
            (int, float),
        ):
            raise StopLossIntelligenceError(
                f"{name} must be numeric."
            )

        value = float(value)

        if not isfinite(value) or value <= 0.0:
            raise StopLossIntelligenceError(
                f"{name} must be greater than zero."
            )

        return value

    @staticmethod
    def _validate_non_negative(
        value: float,
        name: str,
    ) -> float:

        if isinstance(value, bool):
            raise StopLossIntelligenceError(
                f"{name} cannot be boolean."
            )

        if not isinstance(
            value,
            (int, float),
        ):
            raise StopLossIntelligenceError(
                f"{name} must be numeric."
            )

        value = float(value)

        if not isfinite(value) or value < 0.0:
            raise StopLossIntelligenceError(
                f"{name} must be finite and non-negative."
            )

        return value

    @staticmethod
    def _validate_score(
        value: float,
        name: str,
    ) -> float:

        if isinstance(value, bool):
            raise StopLossIntelligenceError(
                f"{name} cannot be boolean."
            )

        if not isinstance(
            value,
            (int, float),
        ):
            raise StopLossIntelligenceError(
                f"{name} must be numeric."
            )

        value = float(value)

        if not isfinite(value):
            raise StopLossIntelligenceError(
                f"{name} must be finite."
            )

        if value < 0.0 or value > 100.0:
            raise StopLossIntelligenceError(
                f"{name} must be between 0 and 100."
            )

        return value