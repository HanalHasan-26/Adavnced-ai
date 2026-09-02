from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.trading.context.market_context import (
    ContextBias,
    ContextSignal,
    ContextSignalType,
    MarketCondition,
    MarketContext,
)
from app.trading.regime.market_regime import (
    MarketRegime,
    MarketRegimeEngine,
    MarketRegimeError,
)


def make_bar(
    index: int,
    close: float = 3000.0,
    high: float | None = None,
    low: float | None = None,
    symbol: str = "XAUUSD",
    timeframe: str = "M15",
) -> object:
    from app.trading.data.market_bar import MarketBar

    if high is None:
        high = close + 5.0

    if low is None:
        low = close - 5.0

    return MarketBar(
        timestamp=(
            datetime(2026, 1, 1)
            + timedelta(minutes=15 * index)
        ),
        symbol=symbol,
        timeframe=timeframe,
        open=close,
        high=high,
        low=low,
        close=close,
        volume=100.0,
    )


def make_context(
    *,
    timestamp=None,
    symbol="XAUUSD",
    timeframe="M15",
    trend="BULLISH",
    trend_strength=80.0,
    context_strength=80.0,
    condition=MarketCondition.TRENDING_UP,
    sufficient_history=True,
    conflicts=(),
):
    if timestamp is None:
        timestamp = datetime(2026, 1, 1)

    if trend == "BULLISH":
        context_trend = __import__(
            "app.trading.structure.market_structure",
            fromlist=["StructureTrend"],
        ).StructureTrend.BULLISH
        bias = ContextBias.BULLISH

    elif trend == "BEARISH":
        context_trend = __import__(
            "app.trading.structure.market_structure",
            fromlist=["StructureTrend"],
        ).StructureTrend.BEARISH
        bias = ContextBias.BEARISH

    elif trend == "RANGE":
        context_trend = __import__(
            "app.trading.structure.market_structure",
            fromlist=["StructureTrend"],
        ).StructureTrend.RANGE
        bias = ContextBias.NEUTRAL

    else:
        context_trend = __import__(
            "app.trading.structure.market_structure",
            fromlist=["StructureTrend"],
        ).StructureTrend.UNKNOWN
        bias = ContextBias.NEUTRAL

    signals = (
        ContextSignal(
            signal_type=ContextSignalType.STRUCTURE,
            bias=bias,
            strength=trend_strength,
            value=trend_strength,
        ),
    )

    return MarketContext(
        timestamp=timestamp,
        symbol=symbol,
        timeframe=timeframe,
        close=3000.0,
        trend=context_trend,
        trend_strength=trend_strength,
        rsi=50.0,
        atr=10.0,
        macd=None,
        bollinger_bands=None,
        price_location=50.0,
        volatility_ratio=0.5,
        bias=bias,
        context_strength=context_strength,
        condition=condition,
        signals=signals,
        conflicts=tuple(conflicts),
        sufficient_history=sufficient_history,
    )


class TestConfiguration:
    def test_defaults(self):
        engine = MarketRegimeEngine()

        assert engine.volatility_lookback == 20
        assert engine.trend_threshold == 60.0
        assert engine.range_threshold == 40.0
        assert engine.high_volatility_ratio == 1.50
        assert engine.low_volatility_ratio == 0.70
        assert engine.minimum_history == 30
        assert engine.minimum_persistence == 2

    def test_invalid_lookback(self):
        with pytest.raises(MarketRegimeError):
            MarketRegimeEngine(
                volatility_lookback=0
            )

    def test_invalid_minimum_history(self):
        with pytest.raises(MarketRegimeError):
            MarketRegimeEngine(
                minimum_history=0
            )

    def test_invalid_persistence(self):
        with pytest.raises(MarketRegimeError):
            MarketRegimeEngine(
                minimum_persistence=0
            )

    def test_invalid_threshold_order(self):
        with pytest.raises(MarketRegimeError):
            MarketRegimeEngine(
                trend_threshold=30.0,
                range_threshold=50.0,
            )

    def test_invalid_volatility_threshold_order(self):
        with pytest.raises(MarketRegimeError):
            MarketRegimeEngine(
                low_volatility_ratio=2.0,
                high_volatility_ratio=1.0,
            )


class TestValidation:
    def test_bars_must_be_list(self):
        engine = MarketRegimeEngine()

        bars = tuple(
            make_bar(index)
            for index in range(30)
        )

        context = make_context(
            timestamp=bars[-1].timestamp
        )

        with pytest.raises(MarketRegimeError):
            engine.detect(
                bars,
                context,
            )

    def test_empty_bars_rejected(self):
        engine = MarketRegimeEngine()

        with pytest.raises(MarketRegimeError):
            engine.detect(
                [],
                make_context(),
            )

    def test_context_type_required(self):
        engine = MarketRegimeEngine()

        bars = [
            make_bar(index)
            for index in range(30)
        ]

        with pytest.raises(MarketRegimeError):
            engine.detect(
                bars,
                object(),
            )

    def test_context_symbol_must_match(self):
        engine = MarketRegimeEngine()

        bars = [
            make_bar(index)
            for index in range(30)
        ]

        context = make_context(
            timestamp=bars[-1].timestamp,
            symbol="EURUSD",
        )

        with pytest.raises(MarketRegimeError):
            engine.detect(
                bars,
                context,
            )

    def test_context_timeframe_must_match(self):
        engine = MarketRegimeEngine()

        bars = [
            make_bar(index)
            for index in range(30)
        ]

        context = make_context(
            timestamp=bars[-1].timestamp,
            timeframe="H1",
        )

        with pytest.raises(MarketRegimeError):
            engine.detect(
                bars,
                context,
            )

    def test_context_timestamp_must_match(self):
        engine = MarketRegimeEngine()

        bars = [
            make_bar(index)
            for index in range(30)
        ]

        context = make_context()

        with pytest.raises(MarketRegimeError):
            engine.detect(
                bars,
                context,
            )


class TestInsufficientHistory:
    def test_insufficient_history_returns_unknown(self):
        engine = MarketRegimeEngine(
            minimum_history=30
        )

        bars = [
            make_bar(index)
            for index in range(10)
        ]

        context = make_context(
            timestamp=bars[-1].timestamp,
            sufficient_history=False,
        )

        result = engine.detect(
            bars,
            context,
        )

        assert result.regime == MarketRegime.UNKNOWN
        assert result.strength == 0.0
        assert result.sufficient_history is False
        assert result.persistence_bars == 0
        assert result.reasons


class TestTrendClassification:
    def test_strong_bullish_context_is_uptrend(self):
        engine = MarketRegimeEngine(
            minimum_persistence=1
        )

        bars = [
            make_bar(
                index,
                close=3000.0 + index * 5.0,
            )
            for index in range(30)
        ]

        context = make_context(
            timestamp=bars[-1].timestamp,
            trend="BULLISH",
            trend_strength=80.0,
            context_strength=80.0,
            condition=MarketCondition.TRENDING_UP,
        )

        result = engine.detect(
            bars,
            context,
        )

        assert result.regime == MarketRegime.TRENDING_UP
        assert result.is_trending
        assert result.strength == 80.0

    def test_strong_bearish_context_is_downtrend(self):
        engine = MarketRegimeEngine(
            minimum_persistence=1
        )

        bars = [
            make_bar(
                index,
                close=3000.0 - index * 5.0,
            )
            for index in range(30)
        ]

        context = make_context(
            timestamp=bars[-1].timestamp,
            trend="BEARISH",
            trend_strength=80.0,
            context_strength=80.0,
            condition=MarketCondition.TRENDING_DOWN,
        )

        result = engine.detect(
            bars,
            context,
        )

        assert result.regime == MarketRegime.TRENDING_DOWN
        assert result.is_trending
        assert result.strength == 80.0


class TestRangeClassification:
    def test_range_context_is_ranging(self):
        engine = MarketRegimeEngine(
            minimum_persistence=1
        )

        bars = [
            make_bar(
                index,
                close=3000.0
                + (0.5 if index % 2 else -0.5),
            )
            for index in range(30)
        ]

        context = make_context(
            timestamp=bars[-1].timestamp,
            trend="RANGE",
            trend_strength=20.0,
            context_strength=20.0,
            condition=MarketCondition.RANGING,
        )

        result = engine.detect(
            bars,
            context,
        )

        assert result.regime == MarketRegime.RANGING
        assert result.is_ranging


class TestTransition:
    def test_transition_context_is_transition(self):
        engine = MarketRegimeEngine(
            minimum_persistence=1
        )

        bars = [
            make_bar(index)
            for index in range(30)
        ]

        context = make_context(
            timestamp=bars[-1].timestamp,
            trend="BULLISH",
            trend_strength=45.0,
            context_strength=45.0,
            condition=MarketCondition.TRANSITION,
        )

        result = engine.detect(
            bars,
            context,
        )

        assert result.regime == MarketRegime.TRANSITION
        assert result.is_transition


class TestHighVolatility:
    def test_high_current_range_is_high_volatility(self):
        engine = MarketRegimeEngine(
            minimum_persistence=1,
            high_volatility_ratio=1.2,
        )

        bars = []

        for index in range(29):
            bars.append(
                make_bar(
                    index,
                    close=3000.0,
                    high=3001.0,
                    low=2999.0,
                )
            )

        bars.append(
            make_bar(
                29,
                close=3000.0,
                high=3020.0,
                low=2980.0,
            )
        )

        context = make_context(
            timestamp=bars[-1].timestamp,
            trend="RANGE",
            trend_strength=20.0,
            context_strength=20.0,
            condition=MarketCondition.RANGING,
        )

        result = engine.detect(
            bars,
            context,
        )

        assert result.regime == MarketRegime.HIGH_VOLATILITY
        assert result.is_high_volatility
        assert result.volatility_ratio is not None
        assert result.volatility_ratio > 1.2


class TestVolatility:
    def test_volatility_ratio_is_positive(self):
        engine = MarketRegimeEngine(
            minimum_persistence=1
        )

        bars = [
            make_bar(
                index,
                close=3000.0,
                high=3005.0,
                low=2995.0,
            )
            for index in range(30)
        ]

        context = make_context(
            timestamp=bars[-1].timestamp,
            trend="RANGE",
            trend_strength=20.0,
            context_strength=20.0,
            condition=MarketCondition.RANGING,
        )

        result = engine.detect(
            bars,
            context,
        )

        assert result.volatility_ratio is not None
        assert result.volatility_ratio > 0.0

    def test_single_bar_has_no_volatility_ratio(self):
        engine = MarketRegimeEngine(
            minimum_history=1,
            minimum_persistence=1,
        )

        bars = [
            make_bar(0)
        ]

        context = make_context(
            timestamp=bars[-1].timestamp,
            sufficient_history=True,
        )

        result = engine.detect(
            bars,
            context,
        )

        assert result.volatility_ratio is None


class TestPersistence:
    def test_uptrend_persistence_increases(self):
        engine = MarketRegimeEngine(
            minimum_persistence=1
        )

        bars = [
            make_bar(
                index,
                close=3000.0 + index * 5.0,
            )
            for index in range(30)
        ]

        context = make_context(
            timestamp=bars[-1].timestamp,
            trend="BULLISH",
            trend_strength=80.0,
            context_strength=80.0,
            condition=MarketCondition.TRENDING_UP,
        )

        result = engine.detect(
            bars,
            context,
        )

        assert result.persistence_bars >= 2

    def test_first_regime_is_transition_with_persistence_filter(self):
        engine = MarketRegimeEngine(
            minimum_persistence=5
        )

        bars = [
            make_bar(index)
            for index in range(30)
        ]

        context = make_context(
            timestamp=bars[-1].timestamp,
            trend="BULLISH",
            trend_strength=80.0,
            context_strength=80.0,
            condition=MarketCondition.TRENDING_UP,
        )

        result = engine.detect(
            bars,
            context,
        )

        assert result.regime == MarketRegime.TRANSITION


class TestResultProperties:
    def test_trending_property(self):
        engine = MarketRegimeEngine(
            minimum_persistence=1
        )

        bars = [
            make_bar(
                index,
                close=3000.0 + index * 5.0,
            )
            for index in range(30)
        ]

        context = make_context(
            timestamp=bars[-1].timestamp,
            trend="BULLISH",
            trend_strength=80.0,
            context_strength=80.0,
            condition=MarketCondition.TRENDING_UP,
        )

        result = engine.detect(
            bars,
            context,
        )

        assert result.is_trending is True
        assert result.is_ranging is False
        assert result.is_transition is False

    def test_range_property(self):
        engine = MarketRegimeEngine(
            minimum_persistence=1
        )

        bars = [
            make_bar(index)
            for index in range(30)
        ]

        context = make_context(
            timestamp=bars[-1].timestamp,
            trend="RANGE",
            trend_strength=20.0,
            context_strength=20.0,
            condition=MarketCondition.RANGING,
        )

        result = engine.detect(
            bars,
            context,
        )

        assert result.is_ranging is True

    def test_high_volatility_property(self):
        engine = MarketRegimeEngine(
            minimum_persistence=1,
            high_volatility_ratio=1.2,
        )

        bars = [
            make_bar(
                index,
                close=3000.0,
                high=3001.0,
                low=2999.0,
            )
            for index in range(29)
        ]

        bars.append(
            make_bar(
                29,
                close=3000.0,
                high=3020.0,
                low=2980.0,
            )
        )

        context = make_context(
            timestamp=bars[-1].timestamp,
            trend="RANGE",
            trend_strength=20.0,
            context_strength=20.0,
            condition=MarketCondition.RANGING,
        )

        result = engine.detect(
            bars,
            context,
        )

        assert result.is_high_volatility is True


class TestContextSequence:
    def test_detect_from_contexts(self):
        engine = MarketRegimeEngine(
            minimum_persistence=1
        )

        bars = [
            make_bar(index)
            for index in range(30)
        ]

        contexts = [
            make_context(
                timestamp=bars[-1].timestamp,
                trend="BULLISH",
                trend_strength=80.0,
                context_strength=80.0,
                condition=MarketCondition.TRENDING_UP,
            )
        ]

        result = engine.detect_from_contexts(
            bars,
            contexts,
        )

        assert result.regime == MarketRegime.TRENDING_UP

    def test_empty_contexts_rejected(self):
        engine = MarketRegimeEngine()

        bars = [
            make_bar(index)
            for index in range(30)
        ]

        with pytest.raises(MarketRegimeError):
            engine.detect_from_contexts(
                bars,
                [],
            )

    def test_invalid_context_index_rejected(self):
        engine = MarketRegimeEngine()

        bars = [
            make_bar(index)
            for index in range(30)
        ]

        contexts = [
            make_context(
                timestamp=bars[-1].timestamp
            )
        ]

        with pytest.raises(MarketRegimeError):
            engine.detect_from_contexts(
                bars,
                contexts,
                index=10,
            )


class TestReasons:
    def test_reasons_are_present(self):
        engine = MarketRegimeEngine(
            minimum_persistence=1
        )

        bars = [
            make_bar(index)
            for index in range(30)
        ]

        context = make_context(
            timestamp=bars[-1].timestamp,
            trend="BULLISH",
            trend_strength=80.0,
            context_strength=80.0,
            condition=MarketCondition.TRENDING_UP,
        )

        result = engine.detect(
            bars,
            context,
        )

        assert result.reasons
        assert all(
            reason.message
            for reason in result.reasons
        )

    def test_conflict_is_reported(self):
        engine = MarketRegimeEngine(
            minimum_persistence=1
        )

        bars = [
            make_bar(index)
            for index in range(30)
        ]

        context = make_context(
            timestamp=bars[-1].timestamp,
            trend="BULLISH",
            trend_strength=80.0,
            context_strength=80.0,
            condition=MarketCondition.TRENDING_UP,
            conflicts=(
                ContextSignalType.RSI,
            ),
        )

        result = engine.detect(
            bars,
            context,
        )

        messages = [
            reason.message
            for reason in result.reasons
        ]

        assert any(
            "conflicting" in message.lower()
            for message in messages
        )


class TestDeterminism:
    def test_same_input_produces_same_result(self):
        engine = MarketRegimeEngine(
            minimum_persistence=1
        )

        bars = [
            make_bar(
                index,
                close=3000.0 + index,
            )
            for index in range(30)
        ]

        context = make_context(
            timestamp=bars[-1].timestamp,
            trend="BULLISH",
            trend_strength=80.0,
            context_strength=80.0,
            condition=MarketCondition.TRENDING_UP,
        )

        first = engine.detect(
            bars,
            context,
        )

        second = engine.detect(
            bars,
            context,
        )

        assert first == second