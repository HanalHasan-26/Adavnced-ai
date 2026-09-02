from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from app.trading.news.event_direction import Direction
from app.trading.news.news_market_impact import (
    ImpactLevel,
    NewsMarketImpactResult,
)


class NewsEnvironmentError(ValueError):
    """Raised when news environment analysis receives invalid input."""


class NewsEnvironmentDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class NewsEnvironmentLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"
    UNKNOWN = "unknown"


class NewsEnvironmentReasonType(str, Enum):
    NO_EVENTS = "no_events"
    INSUFFICIENT_DATA = "insufficient_data"
    BULLISH_ENVIRONMENT = "bullish_environment"
    BEARISH_ENVIRONMENT = "bearish_environment"
    NEUTRAL_ENVIRONMENT = "neutral_environment"
    UNKNOWN_ENVIRONMENT = "unknown_environment"
    BULLISH_PRESSURE = "bullish_pressure"
    BEARISH_PRESSURE = "bearish_pressure"
    BALANCED_PRESSURE = "balanced_pressure"
    CONFLICTING_EVENTS = "conflicting_events"
    NO_CONFLICT = "no_conflict"
    LOW_IMPACT_ENVIRONMENT = "low_impact_environment"
    MEDIUM_IMPACT_ENVIRONMENT = "medium_impact_environment"
    HIGH_IMPACT_ENVIRONMENT = "high_impact_environment"
    EXTREME_IMPACT_ENVIRONMENT = "extreme_impact_environment"
    HIGH_CONFIDENCE = "high_confidence"
    MEDIUM_CONFIDENCE = "medium_confidence"
    LOW_CONFIDENCE = "low_confidence"
    CAUTION_REQUIRED = "caution_required"


@dataclass(frozen=True, slots=True)
class NewsEnvironmentReason:
    reason_type: NewsEnvironmentReasonType
    message: str


@dataclass(frozen=True, slots=True)
class NewsEnvironmentResult:
    timestamp: datetime
    symbol: str

    event_count: int
    valid_event_count: int

    bullish_event_count: int
    bearish_event_count: int
    neutral_event_count: int
    unknown_event_count: int

    net_directional_score: float
    average_impact_score: float
    confidence: float

    direction: NewsEnvironmentDirection
    impact_level: NewsEnvironmentLevel

    supports_long: bool
    supports_short: bool
    conflicting_events: bool
    caution_required: bool

    relevant_events: tuple[NewsMarketImpactResult, ...]
    reasons: tuple[NewsEnvironmentReason, ...]

    sufficient_data: bool

    @property
    def is_bullish(self) -> bool:
        return self.direction == NewsEnvironmentDirection.BULLISH

    @property
    def is_bearish(self) -> bool:
        return self.direction == NewsEnvironmentDirection.BEARISH

    @property
    def is_neutral(self) -> bool:
        return self.direction == NewsEnvironmentDirection.NEUTRAL

    @property
    def is_unknown(self) -> bool:
        return self.direction == NewsEnvironmentDirection.UNKNOWN

    @property
    def is_high_impact(self) -> bool:
        return self.impact_level in (
            NewsEnvironmentLevel.HIGH,
            NewsEnvironmentLevel.EXTREME,
        )

    @property
    def has_events(self) -> bool:
        return self.event_count > 0

    @property
    def has_valid_events(self) -> bool:
        return self.valid_event_count > 0


class NewsEnvironmentEngine:
    """
    Aggregates individual news-market impact assessments into
    one XAU/USD news environment.

    This engine summarizes news evidence only.

    It does not make a trading decision.
    """

    IMPACT_WEIGHTS = {
        ImpactLevel.NONE: 0.0,
        ImpactLevel.LOW: 0.50,
        ImpactLevel.MEDIUM: 0.75,
        ImpactLevel.HIGH: 1.00,
        ImpactLevel.EXTREME: 1.25,
    }

    IMPACT_SCORES = {
        ImpactLevel.NONE: 0.0,
        ImpactLevel.LOW: 25.0,
        ImpactLevel.MEDIUM: 50.0,
        ImpactLevel.HIGH: 75.0,
        ImpactLevel.EXTREME: 100.0,
    }

    def __init__(
        self,
        bullish_threshold: float = 15.0,
        bearish_threshold: float = -15.0,
        high_impact_threshold: float = 60.0,
        extreme_impact_threshold: float = 85.0,
        conflict_ratio_threshold: float = 0.35,
        caution_threshold: float = 60.0,
        minimum_valid_events: int = 1,
    ) -> None:
        self.bullish_threshold = self._validate_score_threshold(
            bullish_threshold,
            "bullish_threshold",
        )

        self.bearish_threshold = self._validate_score_threshold(
            bearish_threshold,
            "bearish_threshold",
        )

        self.high_impact_threshold = self._validate_percentage(
            high_impact_threshold,
            "high_impact_threshold",
        )

        self.extreme_impact_threshold = self._validate_percentage(
            extreme_impact_threshold,
            "extreme_impact_threshold",
        )

        self.conflict_ratio_threshold = self._validate_ratio(
            conflict_ratio_threshold,
            "conflict_ratio_threshold",
        )

        self.caution_threshold = self._validate_percentage(
            caution_threshold,
            "caution_threshold",
        )

        if not isinstance(minimum_valid_events, int):
            raise NewsEnvironmentError(
                "minimum_valid_events must be an integer."
            )

        if minimum_valid_events <= 0:
            raise NewsEnvironmentError(
                "minimum_valid_events must be greater than 0."
            )

        self.minimum_valid_events = minimum_valid_events

        if self.bullish_threshold <= 0.0:
            raise NewsEnvironmentError(
                "bullish_threshold must be greater than 0."
            )

        if self.bearish_threshold >= 0.0:
            raise NewsEnvironmentError(
                "bearish_threshold must be less than 0."
            )

        if self.high_impact_threshold >= self.extreme_impact_threshold:
            raise NewsEnvironmentError(
                "high_impact_threshold must be less than "
                "extreme_impact_threshold."
            )

    def analyze(
        self,
        impacts: list[NewsMarketImpactResult] | tuple[
            NewsMarketImpactResult,
            ...,
        ],
        symbol: str = "XAUUSD",
    ) -> NewsEnvironmentResult:
        self._validate_inputs(impacts, symbol)

        impacts_tuple = tuple(impacts)

        if not impacts_tuple:
            return self._empty_result(symbol)

        self._validate_symbol_consistency(
            impacts_tuple,
            symbol,
        )

        relevant_events = tuple(
            impact
            for impact in impacts_tuple
            if impact.sufficient_data
            and impact.direction != Direction.UNKNOWN
            and impact.impact_level != ImpactLevel.UNKNOWN
        )

        event_count = len(impacts_tuple)
        valid_event_count = len(relevant_events)

        if valid_event_count < self.minimum_valid_events:
            return self._insufficient_result(
                impacts_tuple,
                symbol,
            )

        bullish_event_count = sum(
            impact.direction == Direction.BULLISH
            for impact in relevant_events
        )

        bearish_event_count = sum(
            impact.direction == Direction.BEARISH
            for impact in relevant_events
        )

        neutral_event_count = sum(
            impact.direction == Direction.NEUTRAL
            for impact in relevant_events
        )

        unknown_event_count = event_count - valid_event_count

        net_directional_score = self._calculate_net_directional_score(
            relevant_events
        )

        average_impact_score = self._calculate_average_impact(
            relevant_events
        )

        confidence = self._calculate_confidence(
            relevant_events,
            net_directional_score,
        )

        direction = self._classify_direction(
            net_directional_score
        )

        impact_level = self._classify_impact(
            average_impact_score
        )

        conflicting_events = self._detect_conflict(
            bullish_event_count=bullish_event_count,
            bearish_event_count=bearish_event_count,
            valid_event_count=valid_event_count,
        )

        supports_long = direction == NewsEnvironmentDirection.BULLISH
        supports_short = direction == NewsEnvironmentDirection.BEARISH

        caution_required = self._requires_caution(
            average_impact_score=average_impact_score,
            conflicting_events=conflicting_events,
            confidence=confidence,
        )

        ordered_events = tuple(
            sorted(
                relevant_events,
                key=lambda impact: impact.timestamp,
            )
        )

        reasons = self._build_reasons(
            direction=direction,
            impact_level=impact_level,
            net_directional_score=net_directional_score,
            average_impact_score=average_impact_score,
            confidence=confidence,
            conflicting_events=conflicting_events,
            caution_required=caution_required,
            bullish_event_count=bullish_event_count,
            bearish_event_count=bearish_event_count,
        )

        timestamp = max(
            impact.timestamp
            for impact in impacts_tuple
        )

        return NewsEnvironmentResult(
            timestamp=timestamp,
            symbol=symbol,
            event_count=event_count,
            valid_event_count=valid_event_count,
            bullish_event_count=bullish_event_count,
            bearish_event_count=bearish_event_count,
            neutral_event_count=neutral_event_count,
            unknown_event_count=unknown_event_count,
            net_directional_score=net_directional_score,
            average_impact_score=average_impact_score,
            confidence=confidence,
            direction=direction,
            impact_level=impact_level,
            supports_long=supports_long,
            supports_short=supports_short,
            conflicting_events=conflicting_events,
            caution_required=caution_required,
            relevant_events=ordered_events,
            reasons=tuple(reasons),
            sufficient_data=True,
        )

    def analyze_xauusd(
        self,
        impacts: list[NewsMarketImpactResult] | tuple[
            NewsMarketImpactResult,
            ...,
        ],
    ) -> NewsEnvironmentResult:
        return self.analyze(
            impacts,
            symbol="XAUUSD",
        )

    def _calculate_net_directional_score(
        self,
        impacts: tuple[NewsMarketImpactResult, ...],
    ) -> float:
        weighted_direction = 0.0
        total_weight = 0.0

        for impact in impacts:
            if impact.direction == Direction.UNKNOWN:
                continue

            weight = self.IMPACT_WEIGHTS.get(
                impact.impact_level,
                0.0,
            )

            if weight <= 0.0:
                continue

            direction_value = self._direction_value(
                impact.direction
            )

            confidence_factor = (
                self._clamp(impact.confidence) / 100.0
            )

            weighted_direction += (
                direction_value
                * weight
                * confidence_factor
            )

            total_weight += weight * confidence_factor

        if total_weight <= 0.0:
            return 0.0

        score = (
            weighted_direction
            / total_weight
        ) * 100.0

        return self._clamp_signed(score)

    def _calculate_average_impact(
        self,
        impacts: tuple[NewsMarketImpactResult, ...],
    ) -> float:
        if not impacts:
            return 0.0

        weighted_score = 0.0
        total_weight = 0.0

        for impact in impacts:
            impact_weight = self.IMPACT_WEIGHTS.get(
                impact.impact_level,
                0.0,
            )

            if impact_weight <= 0.0:
                continue

            weighted_score += (
                self._clamp(impact.impact_score)
                * impact_weight
            )

            total_weight += impact_weight

        if total_weight <= 0.0:
            return 0.0

        return self._clamp(
            weighted_score / total_weight
        )

    def _calculate_confidence(
        self,
        impacts: tuple[NewsMarketImpactResult, ...],
        net_directional_score: float,
    ) -> float:
        if not impacts:
            return 0.0

        average_confidence = sum(
            self._clamp(impact.confidence)
            for impact in impacts
        ) / len(impacts)

        directional_strength = abs(
            net_directional_score
        )

        event_factor = min(
            1.0,
            len(impacts) / 3.0,
        )

        confidence = (
            average_confidence
            * (0.5 + 0.5 * event_factor)
            * (0.5 + 0.5 * directional_strength / 100.0)
        )

        return self._clamp(confidence)

    def _classify_direction(
        self,
        net_directional_score: float,
    ) -> NewsEnvironmentDirection:
        if net_directional_score >= self.bullish_threshold:
            return NewsEnvironmentDirection.BULLISH

        if net_directional_score <= self.bearish_threshold:
            return NewsEnvironmentDirection.BEARISH

        return NewsEnvironmentDirection.NEUTRAL

    def _classify_impact(
        self,
        average_impact_score: float,
    ) -> NewsEnvironmentLevel:
        if average_impact_score >= self.extreme_impact_threshold:
            return NewsEnvironmentLevel.EXTREME

        if average_impact_score >= self.high_impact_threshold:
            return NewsEnvironmentLevel.HIGH

        if average_impact_score >= 40.0:
            return NewsEnvironmentLevel.MEDIUM

        if average_impact_score >= 20.0:
            return NewsEnvironmentLevel.LOW

        return NewsEnvironmentLevel.NONE

    def _detect_conflict(
        self,
        bullish_event_count: int,
        bearish_event_count: int,
        valid_event_count: int,
    ) -> bool:
        if valid_event_count <= 1:
            return False

        if bullish_event_count == 0:
            return False

        if bearish_event_count == 0:
            return False

        opposing_events = min(
            bullish_event_count,
            bearish_event_count,
        )

        conflict_ratio = (
            opposing_events / valid_event_count
        )

        return conflict_ratio >= self.conflict_ratio_threshold

    def _requires_caution(
        self,
        average_impact_score: float,
        conflicting_events: bool,
        confidence: float,
    ) -> bool:
        if average_impact_score >= self.caution_threshold:
            return True

        if conflicting_events:
            return True

        if confidence < 40.0:
            return True

        return False

    def _build_reasons(
        self,
        direction: NewsEnvironmentDirection,
        impact_level: NewsEnvironmentLevel,
        net_directional_score: float,
        average_impact_score: float,
        confidence: float,
        conflicting_events: bool,
        caution_required: bool,
        bullish_event_count: int,
        bearish_event_count: int,
    ) -> list[NewsEnvironmentReason]:
        reasons: list[NewsEnvironmentReason] = []

        if direction == NewsEnvironmentDirection.BULLISH:
            reasons.append(
                NewsEnvironmentReason(
                    reason_type=(
                        NewsEnvironmentReasonType.BULLISH_ENVIRONMENT
                    ),
                    message=(
                        "The aggregated news environment "
                        "has a bullish XAU/USD directional bias."
                    ),
                )
            )
        elif direction == NewsEnvironmentDirection.BEARISH:
            reasons.append(
                NewsEnvironmentReason(
                    reason_type=(
                        NewsEnvironmentReasonType.BEARISH_ENVIRONMENT
                    ),
                    message=(
                        "The aggregated news environment "
                        "has a bearish XAU/USD directional bias."
                    ),
                )
            )
        elif direction == NewsEnvironmentDirection.NEUTRAL:
            reasons.append(
                NewsEnvironmentReason(
                    reason_type=(
                        NewsEnvironmentReasonType.NEUTRAL_ENVIRONMENT
                    ),
                    message=(
                        "The aggregated news environment "
                        "does not have a strong directional bias."
                    ),
                )
            )
        else:
            reasons.append(
                NewsEnvironmentReason(
                    reason_type=(
                        NewsEnvironmentReasonType.UNKNOWN_ENVIRONMENT
                    ),
                    message=(
                        "The aggregated news environment "
                        "cannot be determined."
                    ),
                )
            )

        if net_directional_score > 0.0:
            reasons.append(
                NewsEnvironmentReason(
                    reason_type=(
                        NewsEnvironmentReasonType.BULLISH_PRESSURE
                    ),
                    message=(
                        "Bullish directional pressure currently "
                        "outweighs bearish pressure."
                    ),
                )
            )
        elif net_directional_score < 0.0:
            reasons.append(
                NewsEnvironmentReason(
                    reason_type=(
                        NewsEnvironmentReasonType.BEARISH_PRESSURE
                    ),
                    message=(
                        "Bearish directional pressure currently "
                        "outweighs bullish pressure."
                    ),
                )
            )
        else:
            reasons.append(
                NewsEnvironmentReason(
                    reason_type=(
                        NewsEnvironmentReasonType.BALANCED_PRESSURE
                    ),
                    message=(
                        "Bullish and bearish news pressure "
                        "is balanced."
                    ),
                )
            )

        impact_reason_map = {
            NewsEnvironmentLevel.NONE: (
                NewsEnvironmentReasonType.LOW_IMPACT_ENVIRONMENT,
                "The aggregated news impact is minimal.",
            ),
            NewsEnvironmentLevel.LOW: (
                NewsEnvironmentReasonType.LOW_IMPACT_ENVIRONMENT,
                "The aggregated news impact is low.",
            ),
            NewsEnvironmentLevel.MEDIUM: (
                NewsEnvironmentReasonType.MEDIUM_IMPACT_ENVIRONMENT,
                "The aggregated news impact is moderate.",
            ),
            NewsEnvironmentLevel.HIGH: (
                NewsEnvironmentReasonType.HIGH_IMPACT_ENVIRONMENT,
                "The aggregated news impact is high.",
            ),
            NewsEnvironmentLevel.EXTREME: (
                NewsEnvironmentReasonType.EXTREME_IMPACT_ENVIRONMENT,
                "The aggregated news impact is extreme.",
            ),
            NewsEnvironmentLevel.UNKNOWN: (
                NewsEnvironmentReasonType.INSUFFICIENT_DATA,
                "The aggregated news impact is unknown.",
            ),
        }

        reason_type, message = impact_reason_map[
            impact_level
        ]

        reasons.append(
            NewsEnvironmentReason(
                reason_type=reason_type,
                message=message,
            )
        )

        if conflicting_events:
            reasons.append(
                NewsEnvironmentReason(
                    reason_type=(
                        NewsEnvironmentReasonType.CONFLICTING_EVENTS
                    ),
                    message=(
                        "Bullish and bearish events are sufficiently "
                        "balanced to create conflicting news pressure."
                    ),
                )
            )
        else:
            reasons.append(
                NewsEnvironmentReason(
                    reason_type=(
                        NewsEnvironmentReasonType.NO_CONFLICT
                    ),
                    message=(
                        "The available events do not create "
                        "significant directional conflict."
                    ),
                )
            )

        if confidence >= 75.0:
            reasons.append(
                NewsEnvironmentReason(
                    reason_type=(
                        NewsEnvironmentReasonType.HIGH_CONFIDENCE
                    ),
                    message="Aggregated news confidence is high.",
                )
            )
        elif confidence >= 50.0:
            reasons.append(
                NewsEnvironmentReason(
                    reason_type=(
                        NewsEnvironmentReasonType.MEDIUM_CONFIDENCE
                    ),
                    message="Aggregated news confidence is moderate.",
                )
            )
        else:
            reasons.append(
                NewsEnvironmentReason(
                    reason_type=(
                        NewsEnvironmentReasonType.LOW_CONFIDENCE
                    ),
                    message="Aggregated news confidence is low.",
                )
            )

        if caution_required:
            reasons.append(
                NewsEnvironmentReason(
                    reason_type=(
                        NewsEnvironmentReasonType.CAUTION_REQUIRED
                    ),
                    message=(
                        "The current news environment requires "
                        "additional caution."
                    ),
                )
            )

        return reasons

    def _empty_result(
        self,
        symbol: str,
    ) -> NewsEnvironmentResult:
        return NewsEnvironmentResult(
            timestamp=datetime.min,
            symbol=symbol,
            event_count=0,
            valid_event_count=0,
            bullish_event_count=0,
            bearish_event_count=0,
            neutral_event_count=0,
            unknown_event_count=0,
            net_directional_score=0.0,
            average_impact_score=0.0,
            confidence=0.0,
            direction=NewsEnvironmentDirection.UNKNOWN,
            impact_level=NewsEnvironmentLevel.UNKNOWN,
            supports_long=False,
            supports_short=False,
            conflicting_events=False,
            caution_required=False,
            relevant_events=(),
            reasons=(
                NewsEnvironmentReason(
                    reason_type=(
                        NewsEnvironmentReasonType.NO_EVENTS
                    ),
                    message="No news impact events were provided.",
                ),
            ),
            sufficient_data=False,
        )

    def _insufficient_result(
        self,
        impacts: tuple[NewsMarketImpactResult, ...],
        symbol: str,
    ) -> NewsEnvironmentResult:
        timestamp = max(
            impact.timestamp
            for impact in impacts
        )

        return NewsEnvironmentResult(
            timestamp=timestamp,
            symbol=symbol,
            event_count=len(impacts),
            valid_event_count=0,
            bullish_event_count=0,
            bearish_event_count=0,
            neutral_event_count=0,
            unknown_event_count=len(impacts),
            net_directional_score=0.0,
            average_impact_score=0.0,
            confidence=0.0,
            direction=NewsEnvironmentDirection.UNKNOWN,
            impact_level=NewsEnvironmentLevel.UNKNOWN,
            supports_long=False,
            supports_short=False,
            conflicting_events=False,
            caution_required=False,
            relevant_events=(),
            reasons=(
                NewsEnvironmentReason(
                    reason_type=(
                        NewsEnvironmentReasonType.INSUFFICIENT_DATA
                    ),
                    message=(
                        "There are not enough valid news impact "
                        "events to build an environment assessment."
                    ),
                ),
            ),
            sufficient_data=False,
        )

    @staticmethod
    def _direction_value(
        direction: Direction,
    ) -> float:
        if direction == Direction.BULLISH:
            return 1.0

        if direction == Direction.BEARISH:
            return -1.0

        return 0.0

    @staticmethod
    def _clamp(value: float) -> float:
        return max(
            0.0,
            min(100.0, float(value)),
        )

    @staticmethod
    def _clamp_signed(value: float) -> float:
        return max(
            -100.0,
            min(100.0, float(value)),
        )

    @staticmethod
    def _validate_score_threshold(
        value: float,
        name: str,
    ) -> float:
        if not isinstance(value, (int, float)):
            raise NewsEnvironmentError(
                f"{name} must be numeric."
            )

        value = float(value)

        if value != value:
            raise NewsEnvironmentError(
                f"{name} must be finite."
            )

        if value < -100.0 or value > 100.0:
            raise NewsEnvironmentError(
                f"{name} must be between -100 and 100."
            )

        return value

    @staticmethod
    def _validate_percentage(
        value: float,
        name: str,
    ) -> float:
        if not isinstance(value, (int, float)):
            raise NewsEnvironmentError(
                f"{name} must be numeric."
            )

        value = float(value)

        if value != value:
            raise NewsEnvironmentError(
                f"{name} must be finite."
            )

        if value < 0.0 or value > 100.0:
            raise NewsEnvironmentError(
                f"{name} must be between 0 and 100."
            )

        return value

    @staticmethod
    def _validate_ratio(
        value: float,
        name: str,
    ) -> float:
        if not isinstance(value, (int, float)):
            raise NewsEnvironmentError(
                f"{name} must be numeric."
            )

        value = float(value)

        if value != value:
            raise NewsEnvironmentError(
                f"{name} must be finite."
            )

        if value < 0.0 or value > 1.0:
            raise NewsEnvironmentError(
                f"{name} must be between 0 and 1."
            )

        return value

    @staticmethod
    def _validate_inputs(
        impacts: list[NewsMarketImpactResult]
        | tuple[NewsMarketImpactResult, ...],
        symbol: str,
    ) -> None:
        if not isinstance(impacts, (list, tuple)):
            raise NewsEnvironmentError(
                "impacts must be a list or tuple."
            )

        if not isinstance(symbol, str) or not symbol.strip():
            raise NewsEnvironmentError(
                "symbol must be a non-empty string."
            )

        for impact in impacts:
            if not isinstance(
                impact,
                NewsMarketImpactResult,
            ):
                raise NewsEnvironmentError(
                    "all impacts must be NewsMarketImpactResult objects."
                )

    @staticmethod
    def _validate_symbol_consistency(
        impacts: tuple[NewsMarketImpactResult, ...],
        symbol: str,
    ) -> None:
        for impact in impacts:
            if impact.symbol != symbol:
                raise NewsEnvironmentError(
                    "all impacts must match the requested symbol."
                )