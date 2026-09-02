from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.trading.context.market_context import ContextBias, MarketContext
from app.trading.news.news_environment import (
    NewsEnvironmentDirection,
    NewsEnvironmentLevel,
    NewsEnvironmentResult,
)
from app.trading.regime.market_regime import MarketRegime, MarketRegimeResult


class MarketEnvironmentError(ValueError):
    """Raised when unified market environment analysis fails validation."""


class EnvironmentDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class EnvironmentQuality(str, Enum):
    CLEAR = "clear"
    FAVORABLE = "favorable"
    MIXED = "mixed"
    CONFLICTED = "conflicted"
    CAUTION = "caution"
    UNKNOWN = "unknown"


class EnvironmentReasonType(str, Enum):
    TECHNICAL_BULLISH = "technical_bullish"
    TECHNICAL_BEARISH = "technical_bearish"
    TECHNICAL_NEUTRAL = "technical_neutral"

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    TRANSITION = "transition"
    UNKNOWN_REGIME = "unknown_regime"

    NEWS_BULLISH = "news_bullish"
    NEWS_BEARISH = "news_bearish"
    NEWS_NEUTRAL = "news_neutral"
    NEWS_UNKNOWN = "news_unknown"
    NEWS_HIGH_IMPACT = "news_high_impact"
    NEWS_CONFLICT = "news_conflict"

    TECHNICAL_NEWS_ALIGNMENT = "technical_news_alignment"
    TECHNICAL_NEWS_CONFLICT = "technical_news_conflict"
    TECHNICAL_NEWS_NEUTRAL = "technical_news_neutral"

    CAUTION_REQUIRED = "caution_required"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True, slots=True)
class EnvironmentReason:
    reason_type: EnvironmentReasonType
    message: str


@dataclass(frozen=True, slots=True)
class MarketEnvironment:
    timestamp: Any
    symbol: str
    timeframe: str

    technical_bias: ContextBias
    technical_strength: float

    market_regime: MarketRegime
    regime_strength: float

    news_direction: NewsEnvironmentDirection
    news_impact_level: NewsEnvironmentLevel
    news_score: float
    news_confidence: float

    overall_direction: EnvironmentDirection
    overall_strength: float
    environment_quality: EnvironmentQuality

    technical_support: bool
    technical_conflict: bool
    news_support: bool
    news_conflict: bool
    environment_conflict: bool
    caution_required: bool

    reasons: tuple[EnvironmentReason, ...]
    warnings: tuple[str, ...]
    sufficient_data: bool

    @property
    def is_bullish(self) -> bool:
        return self.overall_direction is EnvironmentDirection.BULLISH

    @property
    def is_bearish(self) -> bool:
        return self.overall_direction is EnvironmentDirection.BEARISH

    @property
    def is_neutral(self) -> bool:
        return self.overall_direction is EnvironmentDirection.NEUTRAL

    @property
    def is_unknown(self) -> bool:
        return self.overall_direction is EnvironmentDirection.UNKNOWN

    @property
    def is_clear(self) -> bool:
        return self.environment_quality is EnvironmentQuality.CLEAR

    @property
    def is_conflicted(self) -> bool:
        return self.environment_quality is EnvironmentQuality.CONFLICTED

    @property
    def has_caution(self) -> bool:
        return self.caution_required


class MarketEnvironmentEngine:
    """
    Combines technical context, market regime, and news environment.

    This engine describes the market environment only.

    It does not:
    - generate entries
    - calculate stop loss
    - calculate take profit
    - calculate position size
    - execute trades
    - call an LLM
    """

    def __init__(
        self,
        minimum_environment_strength: float = 50.0,
        strong_environment_strength: float = 70.0,
        caution_news_score: float = 60.0,
        caution_news_confidence: float = 40.0,
    ) -> None:
        self.minimum_environment_strength = self._validate_threshold(
            minimum_environment_strength,
            "minimum_environment_strength",
        )
        self.strong_environment_strength = self._validate_threshold(
            strong_environment_strength,
            "strong_environment_strength",
        )
        self.caution_news_score = self._validate_threshold(
            caution_news_score,
            "caution_news_score",
        )
        self.caution_news_confidence = self._validate_threshold(
            caution_news_confidence,
            "caution_news_confidence",
        )

        if self.strong_environment_strength < self.minimum_environment_strength:
            raise MarketEnvironmentError(
                "strong_environment_strength cannot be below "
                "minimum_environment_strength."
            )

    @staticmethod
    def _validate_threshold(value: float, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MarketEnvironmentError(f"{name} must be numeric.")

        value = float(value)

        if value < 0.0 or value > 100.0:
            raise MarketEnvironmentError(
                f"{name} must be between 0 and 100."
            )

        return value

    @staticmethod
    def _validate_inputs(
        context: MarketContext,
        regime: MarketRegimeResult,
        news: NewsEnvironmentResult,
    ) -> None:
        if not isinstance(context, MarketContext):
            raise MarketEnvironmentError(
                "context must be a MarketContext."
            )

        if not isinstance(regime, MarketRegimeResult):
            raise MarketEnvironmentError(
                "regime must be a MarketRegimeResult."
            )

        if not isinstance(news, NewsEnvironmentResult):
            raise MarketEnvironmentError(
                "news must be a NewsEnvironmentResult."
            )

        if context.symbol != regime.symbol:
            raise MarketEnvironmentError(
                "context and regime symbols must match."
            )

        if context.symbol != news.symbol:
            raise MarketEnvironmentError(
                "context and news symbols must match."
            )

        if context.timeframe != regime.timeframe:
            raise MarketEnvironmentError(
                "context and regime timeframes must match."
            )

        if not context.symbol:
            raise MarketEnvironmentError("symbol cannot be empty.")

        if not context.timeframe:
            raise MarketEnvironmentError("timeframe cannot be empty.")

        if context.timestamp != regime.timestamp:
            raise MarketEnvironmentError(
                "context and regime timestamps must match."
            )

        if context.timestamp != news.timestamp:
            raise MarketEnvironmentError(
                "context and news timestamps must match."
            )

    def analyze(
        self,
        context: MarketContext,
        regime: MarketRegimeResult,
        news: NewsEnvironmentResult,
    ) -> MarketEnvironment:
        self._validate_inputs(context, regime, news)

        technical_strength = self._clamp(context.context_strength)
        regime_strength = self._clamp(regime.strength)

        raw_news_score = float(news.net_directional_score)
        news_score = self._clamp(abs(raw_news_score))
        news_confidence = self._clamp(news.confidence)

        technical_bias = context.bias
        news_direction = news.direction

        technical_support = self._has_technical_support(context)
        technical_conflict = self._has_technical_conflict(context)

        news_support = self._has_news_support(news)
        news_conflict = news.conflicting_events

        environment_conflict = self._has_environment_conflict(
            technical_bias,
            news_direction,
            news.sufficient_data,
        )

        overall_direction = self._determine_direction(
            technical_bias=technical_bias,
            technical_strength=technical_strength,
            news_direction=news_direction,
            news_score=news_score,
            news_sufficient=news.sufficient_data,
            environment_conflict=environment_conflict,
        )

        overall_strength = self._calculate_overall_strength(
            technical_strength=technical_strength,
            regime_strength=regime_strength,
            news_score=news_score,
            news_confidence=news_confidence,
            technical_bias=technical_bias,
            news_direction=news_direction,
            news_sufficient=news.sufficient_data,
            environment_conflict=environment_conflict,
        )

        caution_required = self._requires_caution(
            regime=regime.regime,
            news=news,
            environment_conflict=environment_conflict,
            technical_conflict=technical_conflict,
            overall_strength=overall_strength,
        )

        sufficient_data = (
            context.sufficient_history
            and regime.sufficient_history
            and news.sufficient_data
        )

        environment_quality = self._determine_quality(
            overall_direction=overall_direction,
            overall_strength=overall_strength,
            environment_conflict=environment_conflict,
            caution_required=caution_required,
            sufficient_data=sufficient_data,
        )

        reasons = self._build_reasons(
            context=context,
            regime=regime,
            news=news,
            environment_conflict=environment_conflict,
            caution_required=caution_required,
        )

        warnings = self._build_warnings(
            context=context,
            regime=regime,
            news=news,
            environment_conflict=environment_conflict,
            caution_required=caution_required,
        )

        return MarketEnvironment(
            timestamp=context.timestamp,
            symbol=context.symbol,
            timeframe=context.timeframe,
            technical_bias=technical_bias,
            technical_strength=technical_strength,
            market_regime=regime.regime,
            regime_strength=regime_strength,
            news_direction=news_direction,
            news_impact_level=news.impact_level,
            news_score=news_score,
            news_confidence=news_confidence,
            overall_direction=overall_direction,
            overall_strength=overall_strength,
            environment_quality=environment_quality,
            technical_support=technical_support,
            technical_conflict=technical_conflict,
            news_support=news_support,
            news_conflict=news_conflict,
            environment_conflict=environment_conflict,
            caution_required=caution_required,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
            sufficient_data=sufficient_data,
        )

    def analyze_xauusd(
        self,
        context: MarketContext,
        regime: MarketRegimeResult,
        news: NewsEnvironmentResult,
    ) -> MarketEnvironment:
        if context.symbol != "XAUUSD":
            raise MarketEnvironmentError(
                "analyze_xauusd requires symbol XAUUSD."
            )

        return self.analyze(context, regime, news)

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(100.0, float(value)))

    @staticmethod
    def _has_technical_support(context: MarketContext) -> bool:
        return (
            context.bias in {
                ContextBias.BULLISH,
                ContextBias.BEARISH,
            }
            and context.context_strength >= 50.0
        )

    @staticmethod
    def _has_technical_conflict(context: MarketContext) -> bool:
        return len(context.conflicts) > 0

    @staticmethod
    def _has_news_support(news: NewsEnvironmentResult) -> bool:
        if not news.sufficient_data:
            return False

        return news.supports_long or news.supports_short

    @staticmethod
    def _has_environment_conflict(
        technical_bias: ContextBias,
        news_direction: NewsEnvironmentDirection,
        news_sufficient: bool,
    ) -> bool:
        if not news_sufficient:
            return False

        return (
            technical_bias is ContextBias.BULLISH
            and news_direction is NewsEnvironmentDirection.BEARISH
        ) or (
            technical_bias is ContextBias.BEARISH
            and news_direction is NewsEnvironmentDirection.BULLISH
        )

    def _determine_direction(
        self,
        technical_bias: ContextBias,
        technical_strength: float,
        news_direction: NewsEnvironmentDirection,
        news_score: float,
        news_sufficient: bool,
        environment_conflict: bool,
    ) -> EnvironmentDirection:
        if technical_bias is ContextBias.NEUTRAL:
            if news_sufficient:
                if news_direction is NewsEnvironmentDirection.BULLISH:
                    return EnvironmentDirection.BULLISH

                if news_direction is NewsEnvironmentDirection.BEARISH:
                    return EnvironmentDirection.BEARISH

                if news_direction is NewsEnvironmentDirection.NEUTRAL:
                    return EnvironmentDirection.NEUTRAL

            return EnvironmentDirection.NEUTRAL

        if technical_bias is ContextBias.BULLISH:
            if environment_conflict:
                if (
                    technical_strength >= self.strong_environment_strength
                    and news_score < technical_strength
                ):
                    return EnvironmentDirection.BULLISH

                return EnvironmentDirection.NEUTRAL

            return EnvironmentDirection.BULLISH

        if technical_bias is ContextBias.BEARISH:
            if environment_conflict:
                if (
                    technical_strength >= self.strong_environment_strength
                    and news_score < technical_strength
                ):
                    return EnvironmentDirection.BEARISH

                return EnvironmentDirection.NEUTRAL

            return EnvironmentDirection.BEARISH

        return EnvironmentDirection.UNKNOWN

    def _calculate_overall_strength(
        self,
        technical_strength: float,
        regime_strength: float,
        news_score: float,
        news_confidence: float,
        technical_bias: ContextBias,
        news_direction: NewsEnvironmentDirection,
        news_sufficient: bool,
        environment_conflict: bool,
    ) -> float:
        technical_component = technical_strength * 0.45
        regime_component = regime_strength * 0.25

        if not news_sufficient:
            return round(
                self._clamp(
                    technical_component + regime_component
                ),
                6,
            )

        news_component = (
            news_score
            * 0.20
            * (news_confidence / 100.0)
        )

        alignment_bonus = 0.0

        if not environment_conflict:
            if (
                technical_bias is ContextBias.BULLISH
                and news_direction is NewsEnvironmentDirection.BULLISH
            ) or (
                technical_bias is ContextBias.BEARISH
                and news_direction is NewsEnvironmentDirection.BEARISH
            ):
                alignment_bonus = 10.0

            elif (
                technical_bias is ContextBias.NEUTRAL
                and news_direction
                in {
                    NewsEnvironmentDirection.BULLISH,
                    NewsEnvironmentDirection.BEARISH,
                }
            ):
                alignment_bonus = 5.0

        conflict_penalty = 0.0

        if environment_conflict:
            conflict_penalty = 20.0

        strength = (
            technical_component
            + regime_component
            + news_component
            + alignment_bonus
            - conflict_penalty
        )

        return round(self._clamp(strength), 6)

    def _requires_caution(
        self,
        regime: MarketRegime,
        news: NewsEnvironmentResult,
        environment_conflict: bool,
        technical_conflict: bool,
        overall_strength: float,
    ) -> bool:
        # Directional disagreement between technical analysis
        # and news always requires caution.
        if environment_conflict:
            return True

        # Internal technical conflicts require caution.
        if technical_conflict:
            return True

        # Conflicting news events require caution even if the
        # aggregate news direction happens to remain bullish/bearish.
        if news.conflicting_events:
            return True

        # Unstable market regimes require caution.
        if regime in {
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.TRANSITION,
            MarketRegime.UNKNOWN,
        }:
            return True

        # High/extreme impact news requires caution.
        if news.sufficient_data and news.impact_level in {
            NewsEnvironmentLevel.HIGH,
            NewsEnvironmentLevel.EXTREME,
        }:
            return True

        # Very low confidence news should not be treated as
        # strong confirmation.
        if (
            news.sufficient_data
            and news.confidence < self.caution_news_confidence
        ):
            return True

        # A weak combined environment requires caution.
        if overall_strength < self.minimum_environment_strength:
            return True

        return False

    @staticmethod
    def _determine_quality(
        overall_direction: EnvironmentDirection,
        overall_strength: float,
        environment_conflict: bool,
        caution_required: bool,
        sufficient_data: bool,
    ) -> EnvironmentQuality:
        if not sufficient_data:
            return EnvironmentQuality.UNKNOWN

        if environment_conflict:
            return EnvironmentQuality.CONFLICTED

        if caution_required:
            return EnvironmentQuality.CAUTION

        if overall_direction is EnvironmentDirection.NEUTRAL:
            return EnvironmentQuality.MIXED

        if overall_strength >= 70.0:
            return EnvironmentQuality.CLEAR

        return EnvironmentQuality.FAVORABLE

    @staticmethod
    def _build_reasons(
        context: MarketContext,
        regime: MarketRegimeResult,
        news: NewsEnvironmentResult,
        environment_conflict: bool,
        caution_required: bool,
    ) -> list[EnvironmentReason]:
        reasons: list[EnvironmentReason] = []

        if context.bias is ContextBias.BULLISH:
            reasons.append(
                EnvironmentReason(
                    EnvironmentReasonType.TECHNICAL_BULLISH,
                    "Technical market context is bullish.",
                )
            )
        elif context.bias is ContextBias.BEARISH:
            reasons.append(
                EnvironmentReason(
                    EnvironmentReasonType.TECHNICAL_BEARISH,
                    "Technical market context is bearish.",
                )
            )
        else:
            reasons.append(
                EnvironmentReason(
                    EnvironmentReasonType.TECHNICAL_NEUTRAL,
                    "Technical market context is neutral.",
                )
            )

        regime_reason_map = {
            MarketRegime.TRENDING_UP: (
                EnvironmentReasonType.TRENDING_UP,
                "Market regime is trending upward.",
            ),
            MarketRegime.TRENDING_DOWN: (
                EnvironmentReasonType.TRENDING_DOWN,
                "Market regime is trending downward.",
            ),
            MarketRegime.RANGING: (
                EnvironmentReasonType.RANGING,
                "Market regime is ranging.",
            ),
            MarketRegime.HIGH_VOLATILITY: (
                EnvironmentReasonType.HIGH_VOLATILITY,
                "Market regime indicates high volatility.",
            ),
            MarketRegime.TRANSITION: (
                EnvironmentReasonType.TRANSITION,
                "Market regime is in transition.",
            ),
            MarketRegime.UNKNOWN: (
                EnvironmentReasonType.UNKNOWN_REGIME,
                "Market regime is unknown.",
            ),
        }

        regime_reason_type, regime_message = regime_reason_map[
            regime.regime
        ]

        reasons.append(
            EnvironmentReason(
                regime_reason_type,
                regime_message,
            )
        )

        if not news.sufficient_data:
            reasons.append(
                EnvironmentReason(
                    EnvironmentReasonType.NEWS_UNKNOWN,
                    "News environment data is insufficient.",
                )
            )
        elif news.direction is NewsEnvironmentDirection.BULLISH:
            reasons.append(
                EnvironmentReason(
                    EnvironmentReasonType.NEWS_BULLISH,
                    "News environment is directionally bullish.",
                )
            )
        elif news.direction is NewsEnvironmentDirection.BEARISH:
            reasons.append(
                EnvironmentReason(
                    EnvironmentReasonType.NEWS_BEARISH,
                    "News environment is directionally bearish.",
                )
            )
        else:
            reasons.append(
                EnvironmentReason(
                    EnvironmentReasonType.NEWS_NEUTRAL,
                    "News environment is neutral.",
                )
            )

        if news.impact_level in {
            NewsEnvironmentLevel.HIGH,
            NewsEnvironmentLevel.EXTREME,
        }:
            reasons.append(
                EnvironmentReason(
                    EnvironmentReasonType.NEWS_HIGH_IMPACT,
                    "News environment contains significant market impact.",
                )
            )

        if news.conflicting_events:
            reasons.append(
                EnvironmentReason(
                    EnvironmentReasonType.NEWS_CONFLICT,
                    "News environment contains conflicting directional events.",
                )
            )

        if environment_conflict:
            reasons.append(
                EnvironmentReason(
                    EnvironmentReasonType.TECHNICAL_NEWS_CONFLICT,
                    "Technical and news environments point in opposite directions.",
                )
            )
        elif (
            news.sufficient_data
            and (
                (
                    context.bias is ContextBias.BULLISH
                    and news.direction is NewsEnvironmentDirection.BULLISH
                )
                or (
                    context.bias is ContextBias.BEARISH
                    and news.direction is NewsEnvironmentDirection.BEARISH
                )
            )
        ):
            reasons.append(
                EnvironmentReason(
                    EnvironmentReasonType.TECHNICAL_NEWS_ALIGNMENT,
                    "Technical and news environments support the same direction.",
                )
            )
        else:
            reasons.append(
                EnvironmentReason(
                    EnvironmentReasonType.TECHNICAL_NEWS_NEUTRAL,
                    "Technical and news environments do not provide strong alignment.",
                )
            )

        if (
            not context.sufficient_history
            or not regime.sufficient_history
            or not news.sufficient_data
        ):
            reasons.append(
                EnvironmentReason(
                    EnvironmentReasonType.INSUFFICIENT_DATA,
                    "One or more environment inputs do not have sufficient data.",
                )
            )

        if caution_required:
            reasons.append(
                EnvironmentReason(
                    EnvironmentReasonType.CAUTION_REQUIRED,
                    "The combined environment requires caution.",
                )
            )

        return reasons

    @staticmethod
    def _build_warnings(
        context: MarketContext,
        regime: MarketRegimeResult,
        news: NewsEnvironmentResult,
        environment_conflict: bool,
        caution_required: bool,
    ) -> list[str]:
        warnings: list[str] = []

        if not context.sufficient_history:
            warnings.append(
                "Technical context has insufficient history."
            )

        if not regime.sufficient_history:
            warnings.append(
                "Market regime has insufficient history."
            )

        if not news.sufficient_data:
            warnings.append(
                "News environment data is insufficient."
            )

        if environment_conflict:
            warnings.append(
                "Technical and news directional signals conflict."
            )

        if news.conflicting_events:
            warnings.append(
                "News events contain conflicting directional pressure."
            )

        if regime.regime is MarketRegime.HIGH_VOLATILITY:
            warnings.append(
                "High volatility may reduce environment reliability."
            )

        if regime.regime is MarketRegime.TRANSITION:
            warnings.append(
                "Market transition may produce unstable signals."
            )

        if news.impact_level in {
            NewsEnvironmentLevel.HIGH,
            NewsEnvironmentLevel.EXTREME,
        }:
            warnings.append(
                "High-impact news may increase market uncertainty."
            )

        if caution_required:
            warnings.append(
                "Caution is required before using this environment for trade evaluation."
            )

        return warnings