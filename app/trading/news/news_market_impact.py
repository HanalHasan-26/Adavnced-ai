from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.trading.news.event_direction import (
    Direction,
    DirectionTarget,
    EventDirectionResult,
)


class NewsMarketImpactError(ValueError):
    """Raised when news market impact analysis receives invalid input."""


class ImpactLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"
    UNKNOWN = "unknown"


class ImpactReasonType(str, Enum):
    NO_IMPACT = "no_impact"
    LOW_IMPACT = "low_impact"
    MEDIUM_IMPACT = "medium_impact"
    HIGH_IMPACT = "high_impact"
    EXTREME_IMPACT = "extreme_impact"
    BULLISH_DIRECTION = "bullish_direction"
    BEARISH_DIRECTION = "bearish_direction"
    NEUTRAL_DIRECTION = "neutral_direction"
    UNKNOWN_DIRECTION = "unknown_direction"
    USD_XAUUSD_RELATION = "usd_xauusd_relation"
    HIGH_CONFIDENCE = "high_confidence"
    MEDIUM_CONFIDENCE = "medium_confidence"
    LOW_CONFIDENCE = "low_confidence"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True, slots=True)
class NewsMarketImpactReason:
    reason_type: ImpactReasonType
    message: str


@dataclass(frozen=True, slots=True)
class NewsMarketImpactResult:
    timestamp: object
    event_name: str
    symbol: str
    direction: Direction
    impact_level: ImpactLevel
    impact_score: float
    confidence: float
    supports_long: bool
    supports_short: bool
    caution_required: bool
    reasons: tuple[NewsMarketImpactReason, ...]
    sufficient_data: bool

    @property
    def is_bullish(self) -> bool:
        return self.direction == Direction.BULLISH

    @property
    def is_bearish(self) -> bool:
        return self.direction == Direction.BEARISH

    @property
    def is_neutral(self) -> bool:
        return self.direction == Direction.NEUTRAL

    @property
    def is_unknown(self) -> bool:
        return self.direction == Direction.UNKNOWN

    @property
    def is_high_impact(self) -> bool:
        return self.impact_level in (
            ImpactLevel.HIGH,
            ImpactLevel.EXTREME,
        )


class NewsMarketImpactEngine:
    """Converts event-direction evidence into a structured market-impact assessment."""

    def __init__(
        self,
        low_threshold: float = 25.0,
        medium_threshold: float = 50.0,
        high_threshold: float = 75.0,
        extreme_threshold: float = 90.0,
        caution_threshold: float = 60.0,
    ) -> None:
        self.low_threshold = self._validate_threshold(
            low_threshold,
            "low_threshold",
        )
        self.medium_threshold = self._validate_threshold(
            medium_threshold,
            "medium_threshold",
        )
        self.high_threshold = self._validate_threshold(
            high_threshold,
            "high_threshold",
        )
        self.extreme_threshold = self._validate_threshold(
            extreme_threshold,
            "extreme_threshold",
        )
        self.caution_threshold = self._validate_threshold(
            caution_threshold,
            "caution_threshold",
        )

        if not (
            self.low_threshold
            < self.medium_threshold
            < self.high_threshold
            < self.extreme_threshold
        ):
            raise NewsMarketImpactError(
                "impact thresholds must be strictly increasing."
            )

    def analyze(
        self,
        direction_result: EventDirectionResult,
        symbol: str = "XAUUSD",
    ) -> NewsMarketImpactResult:
        self._validate_inputs(direction_result, symbol)

        if not direction_result.sufficient_data:
            return self._unknown_result(direction_result, symbol)

        direction = direction_result.instrument_direction

        if direction == Direction.UNKNOWN:
            return self._unknown_result(direction_result, symbol)

        confidence = self._clamp(direction_result.confidence)

        impact_score = self._calculate_impact_score(
            direction_result,
            confidence,
        )

        impact_level = self._classify_impact(impact_score)

        supports_long = direction == Direction.BULLISH
        supports_short = direction == Direction.BEARISH

        caution_required = (
            impact_level in (
                ImpactLevel.HIGH,
                ImpactLevel.EXTREME,
            )
            or impact_score >= self.caution_threshold
        )

        reasons = self._build_reasons(
            direction_result,
            direction,
            impact_level,
            impact_score,
            confidence,
            symbol,
        )

        return NewsMarketImpactResult(
            timestamp=direction_result.timestamp,
            event_name=direction_result.event_name,
            symbol=symbol,
            direction=direction,
            impact_level=impact_level,
            impact_score=impact_score,
            confidence=confidence,
            supports_long=supports_long,
            supports_short=supports_short,
            caution_required=caution_required,
            reasons=tuple(reasons),
            sufficient_data=True,
        )

    def analyze_xauusd(
        self,
        direction_result: EventDirectionResult,
    ) -> NewsMarketImpactResult:
        return self.analyze(
            direction_result,
            symbol="XAUUSD",
        )

    def _calculate_impact_score(
        self,
        direction_result: EventDirectionResult,
        confidence: float,
    ) -> float:
        score = confidence

        if direction_result.surprise_direction.value == "none":
            score *= 0.5

        return self._clamp(score)

    def _classify_impact(self, score: float) -> ImpactLevel:
        if score >= self.extreme_threshold:
            return ImpactLevel.EXTREME

        if score >= self.high_threshold:
            return ImpactLevel.HIGH

        if score >= self.medium_threshold:
            return ImpactLevel.MEDIUM

        if score >= self.low_threshold:
            return ImpactLevel.LOW

        return ImpactLevel.NONE

    def _build_reasons(
        self,
        direction_result: EventDirectionResult,
        direction: Direction,
        impact_level: ImpactLevel,
        impact_score: float,
        confidence: float,
        symbol: str,
    ) -> list[NewsMarketImpactReason]:
        reasons: list[NewsMarketImpactReason] = []

        reason_map = {
            ImpactLevel.NONE: (
                ImpactReasonType.NO_IMPACT,
                "The event does not provide meaningful directional market impact.",
            ),
            ImpactLevel.LOW: (
                ImpactReasonType.LOW_IMPACT,
                "The event provides low directional market impact.",
            ),
            ImpactLevel.MEDIUM: (
                ImpactReasonType.MEDIUM_IMPACT,
                "The event provides moderate directional market impact.",
            ),
            ImpactLevel.HIGH: (
                ImpactReasonType.HIGH_IMPACT,
                "The event provides high directional market impact.",
            ),
            ImpactLevel.EXTREME: (
                ImpactReasonType.EXTREME_IMPACT,
                "The event provides extreme directional market impact.",
            ),
            ImpactLevel.UNKNOWN: (
                ImpactReasonType.INSUFFICIENT_DATA,
                "The event does not contain enough information for impact analysis.",
            ),
        }

        reason_type, message = reason_map[impact_level]

        reasons.append(
            NewsMarketImpactReason(
                reason_type=reason_type,
                message=message,
            )
        )

        if direction == Direction.BULLISH:
            reasons.append(
                NewsMarketImpactReason(
                    reason_type=ImpactReasonType.BULLISH_DIRECTION,
                    message=f"The event direction supports {symbol} upside.",
                )
            )
        elif direction == Direction.BEARISH:
            reasons.append(
                NewsMarketImpactReason(
                    reason_type=ImpactReasonType.BEARISH_DIRECTION,
                    message=f"The event direction supports {symbol} downside.",
                )
            )
        elif direction == Direction.NEUTRAL:
            reasons.append(
                NewsMarketImpactReason(
                    reason_type=ImpactReasonType.NEUTRAL_DIRECTION,
                    message="The event has no meaningful directional bias.",
                )
            )
        else:
            reasons.append(
                NewsMarketImpactReason(
                    reason_type=ImpactReasonType.UNKNOWN_DIRECTION,
                    message="The event direction is unknown.",
                )
            )

        if (
            direction_result.target == DirectionTarget.XAUUSD
            and direction_result.currency_direction != Direction.UNKNOWN
        ):
            reasons.append(
                NewsMarketImpactReason(
                    reason_type=ImpactReasonType.USD_XAUUSD_RELATION,
                    message=(
                        "The event direction has been translated into "
                        "the XAU/USD relationship."
                    ),
                )
            )

        if confidence >= 75.0:
            reasons.append(
                NewsMarketImpactReason(
                    reason_type=ImpactReasonType.HIGH_CONFIDENCE,
                    message="Directional confidence is high.",
                )
            )
        elif confidence >= 50.0:
            reasons.append(
                NewsMarketImpactReason(
                    reason_type=ImpactReasonType.MEDIUM_CONFIDENCE,
                    message="Directional confidence is moderate.",
                )
            )
        else:
            reasons.append(
                NewsMarketImpactReason(
                    reason_type=ImpactReasonType.LOW_CONFIDENCE,
                    message="Directional confidence is low.",
                )
            )

        if impact_score >= self.caution_threshold:
            reasons.append(
                NewsMarketImpactReason(
                    reason_type=ImpactReasonType.HIGH_IMPACT,
                    message="The impact is strong enough to require additional caution.",
                )
            )

        return reasons

    def _unknown_result(
        self,
        direction_result: EventDirectionResult,
        symbol: str,
    ) -> NewsMarketImpactResult:
        return NewsMarketImpactResult(
            timestamp=direction_result.timestamp,
            event_name=direction_result.event_name,
            symbol=symbol,
            direction=Direction.UNKNOWN,
            impact_level=ImpactLevel.UNKNOWN,
            impact_score=0.0,
            confidence=0.0,
            supports_long=False,
            supports_short=False,
            caution_required=False,
            reasons=(
                NewsMarketImpactReason(
                    reason_type=ImpactReasonType.INSUFFICIENT_DATA,
                    message=(
                        "There is insufficient directional information "
                        "to assess market impact."
                    ),
                ),
            ),
            sufficient_data=False,
        )

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(100.0, float(value)))

    @staticmethod
    def _validate_threshold(value: float, name: str) -> float:
        if not isinstance(value, (int, float)):
            raise NewsMarketImpactError(
                f"{name} must be numeric."
            )

        value = float(value)

        if value != value:
            raise NewsMarketImpactError(
                f"{name} must be finite."
            )

        if value < 0.0 or value > 100.0:
            raise NewsMarketImpactError(
                f"{name} must be between 0 and 100."
            )

        return value

    @staticmethod
    def _validate_inputs(
        direction_result: EventDirectionResult,
        symbol: str,
    ) -> None:
        if not isinstance(direction_result, EventDirectionResult):
            raise NewsMarketImpactError(
                "direction_result must be an EventDirectionResult."
            )

        if not isinstance(symbol, str) or not symbol.strip():
            raise NewsMarketImpactError(
                "symbol must be a non-empty string."
            )