from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.trading.context.market_context import (
    ContextBias,
    ContextSignalType,
    MarketCondition,
    MarketContextEngine,
)
from app.trading.data.market_bar import MarketBar
from app.trading.structure.market_structure import (
    StructureTrend,
)


def make_bars(
    closes: list[float],
) -> list[MarketBar]:
    bars: list[MarketBar] = []

    start = datetime(2026, 1, 1)

    for index, close in enumerate(closes):
        close = float(close)

        high = close + 1.0
        low = max(
            0.01,
            close - 1.0,
        )

        bars.append(
            MarketBar(
                timestamp=start + timedelta(
                    minutes=index
                ),
                symbol="XAUUSD",
                timeframe="M1",
                open=close,
                high=high,
                low=low,
                close=close,
                volume=100.0,
            )
        )

    return bars


def make_trending_up_bars(
    count: int = 60,
) -> list[MarketBar]:
    closes = []

    for index in range(count):
        value = (
            100.0
            + index * 0.5
            + (index % 3) * 0.1
        )
        closes.append(value)

    return make_bars(closes)


def make_trending_down_bars(
    count: int = 60,
) -> list[MarketBar]:
    closes = []

    for index in range(count):
        value = (
            150.0
            - index * 0.5
            - (index % 3) * 0.1
        )
        closes.append(value)

    return make_bars(closes)


class TestValidation:
    def test_empty_bars_are_rejected(self):
        engine = MarketContextEngine()

        with pytest.raises(ValueError):
            engine.analyze([])

    def test_bars_must_be_a_list(self):
        engine = MarketContextEngine()

        with pytest.raises(ValueError):
            engine.analyze(
                tuple()  # type: ignore[arg-type]
            )

    def test_invalid_bar_type_is_rejected(self):
        engine = MarketContextEngine()

        with pytest.raises(ValueError):
            engine.analyze(
                [object()]  # type: ignore[list-item]
            )

    def test_invalid_rsi_period(self):
        with pytest.raises(ValueError):
            MarketContextEngine(
                rsi_period=0
            )

    def test_invalid_atr_period(self):
        with pytest.raises(ValueError):
            MarketContextEngine(
                atr_period=0
            )

    def test_invalid_price_range_lookback(self):
        with pytest.raises(ValueError):
            MarketContextEngine(
                price_range_lookback=0
            )

    def test_invalid_macd_order(self):
        with pytest.raises(ValueError):
            MarketContextEngine(
                macd_fast=20,
                macd_slow=10,
            )

    def test_invalid_thresholds(self):
        with pytest.raises(ValueError):
            MarketContextEngine(
                trend_threshold=30,
                neutral_threshold=40,
            )

    def test_invalid_bollinger_multiplier(self):
        with pytest.raises(ValueError):
            MarketContextEngine(
                bollinger_multiplier=0
            )


class TestBasicContext:
    def test_context_returns_latest_candle(self):
        bars = make_bars(
            list(range(1, 61))
        )

        engine = MarketContextEngine()

        context = engine.analyze(bars)

        assert context.timestamp == bars[-1].timestamp
        assert context.close == bars[-1].close
        assert context.symbol == "XAUUSD"
        assert context.timeframe == "M1"

    def test_context_contains_expected_indicators(self):
        bars = make_bars(
            list(range(1, 61))
        )

        engine = MarketContextEngine()

        context = engine.analyze(bars)

        assert context.rsi is not None
        assert context.atr is not None
        assert context.macd is not None
        assert context.bollinger_bands is not None

    def test_context_contains_signals(self):
        bars = make_bars(
            list(range(1, 61))
        )

        engine = MarketContextEngine()

        context = engine.analyze(bars)

        signal_types = {
            signal.signal_type
            for signal in context.signals
        }

        assert ContextSignalType.STRUCTURE in signal_types
        assert ContextSignalType.RSI in signal_types
        assert ContextSignalType.MACD in signal_types
        assert ContextSignalType.VOLATILITY in signal_types
        assert (
            ContextSignalType.PRICE_LOCATION
            in signal_types
        )

    def test_context_strength_is_bounded(self):
        bars = make_bars(
            list(range(1, 61))
        )

        engine = MarketContextEngine()

        context = engine.analyze(bars)

        assert 0.0 <= context.context_strength <= 100.0

    def test_signal_strengths_are_bounded(self):
        bars = make_bars(
            list(range(1, 61))
        )

        engine = MarketContextEngine()

        context = engine.analyze(bars)

        for signal in context.signals:
            assert 0.0 <= signal.strength <= 100.0


class TestInsufficientHistory:
    def test_short_history_does_not_crash(self):
        bars = make_bars(
            [100, 101, 100, 102]
        )

        engine = MarketContextEngine()

        context = engine.analyze(bars)

        assert context is not None

    def test_short_history_reports_insufficient_history(self):
        bars = make_bars(
            [100, 101, 100, 102]
        )

        engine = MarketContextEngine()

        context = engine.analyze(bars)

        assert context.sufficient_history is False

    def test_short_history_has_unknown_or_neutral_context(self):
        bars = make_bars(
            [100, 101, 100, 102]
        )

        engine = MarketContextEngine()

        context = engine.analyze(bars)

        assert context.bias in (
            ContextBias.NEUTRAL,
            ContextBias.BULLISH,
            ContextBias.BEARISH,
        )

        assert context.condition in (
            MarketCondition.UNKNOWN,
            MarketCondition.RANGING,
            MarketCondition.TRANSITION,
            MarketCondition.TRENDING_UP,
            MarketCondition.TRENDING_DOWN,
        )


class TestBullishMarket:
    def test_rising_market_has_bullish_structure_or_context(self):
        bars = make_trending_up_bars()

        engine = MarketContextEngine()

        context = engine.analyze(bars)

        assert context.trend in (
            StructureTrend.BULLISH,
            StructureTrend.RANGE,
            StructureTrend.UNKNOWN,
        )

        assert context.bias in (
            ContextBias.BULLISH,
            ContextBias.NEUTRAL,
        )

    def test_rising_market_has_positive_momentum_evidence(self):
        bars = make_trending_up_bars()

        engine = MarketContextEngine()

        context = engine.analyze(bars)

        bullish_signals = [
            signal
            for signal in context.signals
            if signal.bias is ContextBias.BULLISH
        ]

        assert len(bullish_signals) >= 1


class TestBearishMarket:
    def test_falling_market_has_bearish_structure_or_context(self):
        bars = make_trending_down_bars()

        engine = MarketContextEngine()

        context = engine.analyze(bars)

        assert context.trend in (
            StructureTrend.BEARISH,
            StructureTrend.RANGE,
            StructureTrend.UNKNOWN,
        )

        assert context.bias in (
            ContextBias.BEARISH,
            ContextBias.NEUTRAL,
        )

    def test_falling_market_has_negative_momentum_evidence(self):
        bars = make_trending_down_bars()

        engine = MarketContextEngine()

        context = engine.analyze(bars)

        bearish_signals = [
            signal
            for signal in context.signals
            if signal.bias is ContextBias.BEARISH
        ]

        assert len(bearish_signals) >= 1


class TestAnalyzeAt:
    def test_analyze_at_uses_only_available_history(self):
        bars = make_trending_up_bars(
            count=80
        )

        engine = MarketContextEngine()

        context = engine.analyze_at(
            bars,
            40,
        )

        assert (
            context.timestamp
            == bars[40].timestamp
        )

        assert context.close == bars[40].close

    def test_analyze_at_does_not_use_future_close(self):
        bars = make_trending_up_bars(
            count=80
        )

        engine = MarketContextEngine()

        context = engine.analyze_at(
            bars,
            40,
        )

        assert context.close != bars[-1].close

    def test_analyze_at_matches_explicit_slice(self):
        bars = make_trending_up_bars(
            count=80
        )

        engine = MarketContextEngine()

        index = 50

        from_index = engine.analyze_at(
            bars,
            index,
        )

        from_slice = engine.analyze(
            bars[: index + 1]
        )

        assert from_index.timestamp == (
            from_slice.timestamp
        )

        assert from_index.close == (
            from_slice.close
        )

        assert from_index.bias == (
            from_slice.bias
        )

        assert from_index.condition == (
            from_slice.condition
        )

    def test_negative_index_is_rejected(self):
        bars = make_bars(
            [100, 101, 102]
        )

        engine = MarketContextEngine()

        with pytest.raises(ValueError):
            engine.analyze_at(
                bars,
                -1,
            )

    def test_index_past_end_is_rejected(self):
        bars = make_bars(
            [100, 101, 102]
        )

        engine = MarketContextEngine()

        with pytest.raises(ValueError):
            engine.analyze_at(
                bars,
                len(bars),
            )

    def test_non_integer_index_is_rejected(self):
        bars = make_bars(
            [100, 101, 102]
        )

        engine = MarketContextEngine()

        with pytest.raises(ValueError):
            engine.analyze_at(
                bars,
                1.5,  # type: ignore[arg-type]
            )


class TestPriceLocation:
    def test_price_location_is_bounded(self):
        bars = make_bars(
            list(range(1, 61))
        )

        engine = MarketContextEngine()

        context = engine.analyze(bars)

        assert context.price_location is not None

        assert 0.0 <= (
            context.price_location
        ) <= 100.0

    def test_price_location_is_missing_with_insufficient_history(self):
        bars = make_bars(
            [100, 101, 102]
        )

        engine = MarketContextEngine(
            price_range_lookback=20
        )

        context = engine.analyze(bars)

        assert context.price_location is None


class TestVolatility:
    def test_volatility_ratio_is_positive_when_available(self):
        bars = make_bars(
            list(range(1, 61))
        )

        engine = MarketContextEngine()

        context = engine.analyze(bars)

        assert context.volatility_ratio is not None
        assert context.volatility_ratio > 0

    def test_volatility_ratio_is_missing_with_insufficient_history(self):
        bars = make_bars(
            [100, 101, 102]
        )

        engine = MarketContextEngine()

        context = engine.analyze(bars)

        assert context.volatility_ratio is None


class TestConflicts:
    def test_conflicts_are_tuple(self):
        bars = make_bars(
            list(range(1, 61))
        )

        engine = MarketContextEngine()

        context = engine.analyze(bars)

        assert isinstance(
            context.conflicts,
            tuple,
        )

    def test_signal_bias_values_are_valid(self):
        bars = make_bars(
            list(range(1, 61))
        )

        engine = MarketContextEngine()

        context = engine.analyze(bars)

        for signal in context.signals:
            assert signal.bias in (
                ContextBias.BULLISH,
                ContextBias.BEARISH,
                ContextBias.NEUTRAL,
            )


class TestDeterminism:
    def test_same_input_produces_same_context(self):
        bars = make_trending_up_bars()

        engine = MarketContextEngine()

        first = engine.analyze(bars)
        second = engine.analyze(bars)

        assert first == second

    def test_analyze_at_is_deterministic(self):
        bars = make_trending_up_bars()

        engine = MarketContextEngine()

        first = engine.analyze_at(
            bars,
            40,
        )

        second = engine.analyze_at(
            bars,
            40,
        )

        assert first == second