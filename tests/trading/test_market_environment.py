from datetime import datetime, timedelta

import pytest

from app.trading.context.market_context import (
    ContextBias,
    ContextSignal,
    ContextSignalType,
    MarketCondition,
    MarketContext,
)
from app.trading.environment.market_environment import (
    EnvironmentDirection,
    EnvironmentQuality,
    EnvironmentReasonType,
    MarketEnvironment,
    MarketEnvironmentEngine,
    MarketEnvironmentError,
)
from app.trading.news.news_environment import (
    NewsEnvironmentDirection,
    NewsEnvironmentLevel,
    NewsEnvironmentReason,
    NewsEnvironmentReasonType,
    NewsEnvironmentResult,
)
from app.trading.regime.market_regime import (
    MarketRegime,
    MarketRegimeResult,
    RegimeReason,
)


TIMESTAMP = datetime(2026, 1, 1, 12, 0, 0)


def make_context(
    *,
    bias=ContextBias.BULLISH,
    strength=80.0,
    sufficient_history=True,
    symbol="XAUUSD",
    timeframe="H1",
):
    return MarketContext(
        timestamp=TIMESTAMP,
        symbol=symbol,
        timeframe=timeframe,
        close=3000.0,
        trend=None,
        trend_strength=strength,
        rsi=60.0,
        atr=20.0,
        macd=None,
        bollinger_bands=None,
        price_location=50.0,
        volatility_ratio=1.0,
        bias=bias,
        context_strength=strength,
        condition=MarketCondition.TRENDING_UP,
        signals=(
            ContextSignal(
                signal_type=ContextSignalType.RSI,
                bias=bias,
                strength=strength,
                value=60.0,
            ),
        ),
        conflicts=(),
        sufficient_history=sufficient_history,
    )


def make_regime(
    *,
    regime=MarketRegime.TRENDING_UP,
    strength=80.0,
    sufficient_history=True,
    symbol="XAUUSD",
    timeframe="H1",
):
    return MarketRegimeResult(
        timestamp=TIMESTAMP,
        symbol=symbol,
        timeframe=timeframe,
        regime=regime,
        strength=strength,
        trend_strength=strength,
        volatility_ratio=1.0,
        persistence_bars=5,
        sufficient_history=sufficient_history,
        reasons=(
            RegimeReason("test regime"),
        ),
    )


def make_news(
    *,
    direction=NewsEnvironmentDirection.BULLISH,
    impact_level=NewsEnvironmentLevel.MEDIUM,
    score=70.0,
    confidence=80.0,
    sufficient_data=True,
    supports_long=True,
    supports_short=False,
    conflicting_events=False,
    symbol="XAUUSD",
):
    return NewsEnvironmentResult(
        timestamp=TIMESTAMP,
        symbol=symbol,
        event_count=2 if sufficient_data else 0,
        valid_event_count=2 if sufficient_data else 0,
        bullish_event_count=2 if direction is NewsEnvironmentDirection.BULLISH else 0,
        bearish_event_count=2 if direction is NewsEnvironmentDirection.BEARISH else 0,
        neutral_event_count=1
        if direction is NewsEnvironmentDirection.NEUTRAL
        else 0,
        unknown_event_count=0,
        net_directional_score=score,
        average_impact_score=score,
        confidence=confidence,
        direction=direction,
        impact_level=impact_level,
        supports_long=supports_long,
        supports_short=supports_short,
        conflicting_events=conflicting_events,
        caution_required=False,
        relevant_events=(),
        reasons=(
            NewsEnvironmentReason(
                NewsEnvironmentReasonType.BULLISH_ENVIRONMENT,
                "test news",
            ),
        ),
        sufficient_data=sufficient_data,
    )


class TestMarketEnvironmentEngine:
    def test_bullish_technical_and_bullish_news_is_clear(self):
        engine = MarketEnvironmentEngine()

        result = engine.analyze(
            make_context(
                bias=ContextBias.BULLISH,
                strength=85.0,
            ),
            make_regime(
                regime=MarketRegime.TRENDING_UP,
                strength=85.0,
            ),
            make_news(
                direction=NewsEnvironmentDirection.BULLISH,
                score=80.0,
                confidence=90.0,
            ),
        )

        assert isinstance(result, MarketEnvironment)
        assert result.overall_direction is EnvironmentDirection.BULLISH
        assert result.environment_quality is EnvironmentQuality.CLEAR
        assert result.technical_support is True
        assert result.news_support is True
        assert result.environment_conflict is False

    def test_bearish_technical_and_bearish_news_is_bearish(self):
        engine = MarketEnvironmentEngine()

        result = engine.analyze(
            make_context(
                bias=ContextBias.BEARISH,
                strength=85.0,
            ),
            make_regime(
                regime=MarketRegime.TRENDING_DOWN,
                strength=85.0,
            ),
            make_news(
                direction=NewsEnvironmentDirection.BEARISH,
                score=80.0,
                confidence=90.0,
                supports_long=False,
                supports_short=True,
            ),
        )

        assert result.overall_direction is EnvironmentDirection.BEARISH
        assert result.environment_conflict is False
        assert result.news_support is True

    def test_bullish_technical_and_bearish_news_creates_conflict(self):
        engine = MarketEnvironmentEngine()

        result = engine.analyze(
            make_context(
                bias=ContextBias.BULLISH,
                strength=70.0,
            ),
            make_regime(
                regime=MarketRegime.TRENDING_UP,
                strength=70.0,
            ),
            make_news(
                direction=NewsEnvironmentDirection.BEARISH,
                score=90.0,
                confidence=95.0,
                supports_long=False,
                supports_short=True,
            ),
        )

        assert result.environment_conflict is True
        assert result.overall_direction is EnvironmentDirection.NEUTRAL
        assert result.environment_quality is EnvironmentQuality.CONFLICTED
        assert result.caution_required is True

    def test_strong_technical_signal_can_remain_directional_during_conflict(self):
        engine = MarketEnvironmentEngine()

        result = engine.analyze(
            make_context(
                bias=ContextBias.BULLISH,
                strength=90.0,
            ),
            make_regime(
                regime=MarketRegime.TRENDING_UP,
                strength=90.0,
            ),
            make_news(
                direction=NewsEnvironmentDirection.BEARISH,
                score=30.0,
                confidence=70.0,
                supports_long=False,
                supports_short=True,
            ),
        )

        assert result.environment_conflict is True
        assert result.overall_direction is EnvironmentDirection.BULLISH
        assert result.caution_required is True

    def test_neutral_technical_with_bullish_news_becomes_bullish(self):
        engine = MarketEnvironmentEngine()

        result = engine.analyze(
            make_context(
                bias=ContextBias.NEUTRAL,
                strength=45.0,
            ),
            make_regime(
                regime=MarketRegime.RANGING,
                strength=40.0,
            ),
            make_news(
                direction=NewsEnvironmentDirection.BULLISH,
                score=70.0,
                confidence=80.0,
            ),
        )

        assert result.overall_direction is EnvironmentDirection.BULLISH

    def test_neutral_technical_with_bearish_news_becomes_bearish(self):
        engine = MarketEnvironmentEngine()

        result = engine.analyze(
            make_context(
                bias=ContextBias.NEUTRAL,
                strength=45.0,
            ),
            make_regime(
                regime=MarketRegime.RANGING,
                strength=40.0,
            ),
            make_news(
                direction=NewsEnvironmentDirection.BEARISH,
                score=-70.0,
                confidence=80.0,
                supports_long=False,
                supports_short=True,
            ),
        )

        assert result.overall_direction is EnvironmentDirection.BEARISH

    def test_neutral_technical_and_neutral_news_is_neutral(self):
        engine = MarketEnvironmentEngine()

        result = engine.analyze(
            make_context(
                bias=ContextBias.NEUTRAL,
                strength=40.0,
            ),
            make_regime(
                regime=MarketRegime.RANGING,
                strength=35.0,
            ),
            make_news(
                direction=NewsEnvironmentDirection.NEUTRAL,
                score=0.0,
                confidence=70.0,
                supports_long=False,
                supports_short=False,
            ),
        )

        assert result.overall_direction is EnvironmentDirection.NEUTRAL

    def test_insufficient_news_does_not_create_directional_conflict(self):
        engine = MarketEnvironmentEngine()

        result = engine.analyze(
            make_context(
                bias=ContextBias.BULLISH,
                strength=80.0,
            ),
            make_regime(
                regime=MarketRegime.TRENDING_UP,
                strength=80.0,
            ),
            make_news(
                direction=NewsEnvironmentDirection.UNKNOWN,
                score=0.0,
                confidence=0.0,
                sufficient_data=False,
                supports_long=False,
                supports_short=False,
            ),
        )

        assert result.overall_direction is EnvironmentDirection.BULLISH
        assert result.environment_conflict is False
        assert result.sufficient_data is False
        assert result.environment_quality is EnvironmentQuality.UNKNOWN

    def test_high_volatility_requires_caution(self):
        engine = MarketEnvironmentEngine()

        result = engine.analyze(
            make_context(
                bias=ContextBias.BULLISH,
                strength=80.0,
            ),
            make_regime(
                regime=MarketRegime.HIGH_VOLATILITY,
                strength=80.0,
            ),
            make_news(
                direction=NewsEnvironmentDirection.BULLISH,
                score=50.0,
                confidence=80.0,
            ),
        )

        assert result.caution_required is True
        assert any(
            reason.reason_type is EnvironmentReasonType.HIGH_VOLATILITY
            for reason in result.reasons
        )

    def test_transition_requires_caution(self):
        engine = MarketEnvironmentEngine()

        result = engine.analyze(
            make_context(
                bias=ContextBias.BULLISH,
                strength=75.0,
            ),
            make_regime(
                regime=MarketRegime.TRANSITION,
                strength=45.0,
            ),
            make_news(
                direction=NewsEnvironmentDirection.BULLISH,
                score=50.0,
                confidence=80.0,
            ),
        )

        assert result.caution_required is True
        assert any(
            reason.reason_type is EnvironmentReasonType.TRANSITION
            for reason in result.reasons
        )

    def test_news_conflict_requires_caution(self):
        engine = MarketEnvironmentEngine()

        result = engine.analyze(
            make_context(
                bias=ContextBias.BULLISH,
                strength=80.0,
            ),
            make_regime(
                regime=MarketRegime.TRENDING_UP,
                strength=80.0,
            ),
            make_news(
                direction=NewsEnvironmentDirection.BULLISH,
                score=30.0,
                confidence=70.0,
                conflicting_events=True,
            ),
        )

        assert result.news_conflict is True
        assert result.caution_required is True
        assert any(
            reason.reason_type is EnvironmentReasonType.NEWS_CONFLICT
            for reason in result.reasons
        )

    def test_technical_conflict_requires_caution(self):
        context = make_context(
            bias=ContextBias.BULLISH,
            strength=80.0,
        )

        context = MarketContext(
            timestamp=context.timestamp,
            symbol=context.symbol,
            timeframe=context.timeframe,
            close=context.close,
            trend=context.trend,
            trend_strength=context.trend_strength,
            rsi=context.rsi,
            atr=context.atr,
            macd=context.macd,
            bollinger_bands=context.bollinger_bands,
            price_location=context.price_location,
            volatility_ratio=context.volatility_ratio,
            bias=context.bias,
            context_strength=context.context_strength,
            condition=context.condition,
            signals=context.signals,
            conflicts=("technical conflict",),
            sufficient_history=context.sufficient_history,
        )

        engine = MarketEnvironmentEngine()

        result = engine.analyze(
            context,
            make_regime(),
            make_news(),
        )

        assert result.technical_conflict is True
        assert result.caution_required is True

    def test_environment_strength_is_clamped(self):
        engine = MarketEnvironmentEngine()

        result = engine.analyze(
            make_context(
                bias=ContextBias.BULLISH,
                strength=100.0,
            ),
            make_regime(
                regime=MarketRegime.TRENDING_UP,
                strength=100.0,
            ),
            make_news(
                direction=NewsEnvironmentDirection.BULLISH,
                score=100.0,
                confidence=100.0,
            ),
        )

        assert 0.0 <= result.overall_strength <= 100.0
        assert 0.0 <= result.technical_strength <= 100.0
        assert 0.0 <= result.regime_strength <= 100.0
        assert 0.0 <= result.news_score <= 100.0
        assert 0.0 <= result.news_confidence <= 100.0

    def test_properties(self):
        engine = MarketEnvironmentEngine()

        result = engine.analyze(
            make_context(),
            make_regime(),
            make_news(),
        )

        assert result.is_bullish is True
        assert result.is_bearish is False
        assert result.is_neutral is False
        assert result.is_unknown is False
        assert result.has_caution in {True, False}
        assert isinstance(result.is_clear, bool)
        assert isinstance(result.is_conflicted, bool)

    def test_analyze_xauusd_requires_xauusd(self):
        engine = MarketEnvironmentEngine()

        context = make_context(symbol="EURUSD")
        regime = make_regime(symbol="EURUSD")
        news = make_news(symbol="EURUSD")

        with pytest.raises(MarketEnvironmentError):
            engine.analyze_xauusd(context, regime, news)

    def test_symbol_mismatch_is_rejected(self):
        engine = MarketEnvironmentEngine()

        with pytest.raises(MarketEnvironmentError):
            engine.analyze(
                make_context(symbol="XAUUSD"),
                make_regime(symbol="EURUSD"),
                make_news(symbol="XAUUSD"),
            )

    def test_timeframe_mismatch_is_rejected(self):
        engine = MarketEnvironmentEngine()

        with pytest.raises(MarketEnvironmentError):
            engine.analyze(
                make_context(timeframe="H1"),
                make_regime(timeframe="H4"),
                make_news(),
            )

    def test_timestamp_mismatch_is_rejected(self):
        engine = MarketEnvironmentEngine()

        regime = make_regime()

        regime = MarketRegimeResult(
            timestamp=TIMESTAMP + timedelta(hours=1),
            symbol=regime.symbol,
            timeframe=regime.timeframe,
            regime=regime.regime,
            strength=regime.strength,
            trend_strength=regime.trend_strength,
            volatility_ratio=regime.volatility_ratio,
            persistence_bars=regime.persistence_bars,
            sufficient_history=regime.sufficient_history,
            reasons=regime.reasons,
        )

        with pytest.raises(MarketEnvironmentError):
            engine.analyze(
                make_context(),
                regime,
                make_news(),
            )

    def test_invalid_context_type_is_rejected(self):
        engine = MarketEnvironmentEngine()

        with pytest.raises(MarketEnvironmentError):
            engine.analyze(
                object(),
                make_regime(),
                make_news(),
            )

    def test_invalid_regime_type_is_rejected(self):
        engine = MarketEnvironmentEngine()

        with pytest.raises(MarketEnvironmentError):
            engine.analyze(
                make_context(),
                object(),
                make_news(),
            )

    def test_invalid_news_type_is_rejected(self):
        engine = MarketEnvironmentEngine()

        with pytest.raises(MarketEnvironmentError):
            engine.analyze(
                make_context(),
                make_regime(),
                object(),
            )

    def test_invalid_threshold_is_rejected(self):
        with pytest.raises(MarketEnvironmentError):
            MarketEnvironmentEngine(
                minimum_environment_strength=-1.0
            )

        with pytest.raises(MarketEnvironmentError):
            MarketEnvironmentEngine(
                strong_environment_strength=101.0
            )

    def test_strong_threshold_cannot_be_lower_than_minimum(self):
        with pytest.raises(MarketEnvironmentError):
            MarketEnvironmentEngine(
                minimum_environment_strength=80.0,
                strong_environment_strength=70.0,
            )

    def test_reasons_are_present(self):
        engine = MarketEnvironmentEngine()

        result = engine.analyze(
            make_context(),
            make_regime(),
            make_news(),
        )

        assert len(result.reasons) >= 3
        assert all(
            reason.reason_type in EnvironmentReasonType
            for reason in result.reasons
        )

    def test_warnings_are_tuple(self):
        engine = MarketEnvironmentEngine()

        result = engine.analyze(
            make_context(),
            make_regime(),
            make_news(),
        )

        assert isinstance(result.warnings, tuple)

    def test_result_is_immutable(self):
        engine = MarketEnvironmentEngine()

        result = engine.analyze(
            make_context(),
            make_regime(),
            make_news(),
        )

        with pytest.raises(AttributeError):
            result.symbol = "EURUSD"

    def test_analyze_is_deterministic(self):
        engine = MarketEnvironmentEngine()

        context = make_context()
        regime = make_regime()
        news = make_news()

        first = engine.analyze(context, regime, news)
        second = engine.analyze(context, regime, news)

        assert first == second

    def test_bullish_and_bearish_conflict_reasons_are_recorded(self):
        engine = MarketEnvironmentEngine()

        result = engine.analyze(
            make_context(
                bias=ContextBias.BULLISH,
                strength=70.0,
            ),
            make_regime(
                regime=MarketRegime.TRENDING_UP,
                strength=70.0,
            ),
            make_news(
                direction=NewsEnvironmentDirection.BEARISH,
                score=90.0,
                confidence=95.0,
                supports_long=False,
                supports_short=True,
            ),
        )

        reason_types = {
            reason.reason_type
            for reason in result.reasons
        }

        assert EnvironmentReasonType.TECHNICAL_NEWS_CONFLICT in reason_types
        assert EnvironmentReasonType.CAUTION_REQUIRED in reason_types