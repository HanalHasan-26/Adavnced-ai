from __future__ import annotations

from datetime import datetime, timedelta
import math

import pytest

from app.trading.data.market_bar import MarketBar
from app.trading.indicators.technical_indicators import (
    BollingerBands,
    MACDResult,
    TechnicalIndicatorEngine,
)


def make_bars(
    closes: list[float],
    *,
    high_offset: float = 1.0,
    low_offset: float = 1.0,
) -> list[MarketBar]:
    bars: list[MarketBar] = []

    start = datetime(2026, 1, 1)

    for index, close in enumerate(closes):
        close = float(close)

        high = close + high_offset
        low = max(close - low_offset, 0.01)

        bars.append(
            MarketBar(
                timestamp=start + timedelta(minutes=index),
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


def make_custom_bars(
    values: list[tuple[float, float, float, float]],
) -> list[MarketBar]:
    bars: list[MarketBar] = []

    start = datetime(2026, 1, 1)

    for index, (open_, high, low, close) in enumerate(values):
        bars.append(
            MarketBar(
                timestamp=start + timedelta(minutes=index),
                symbol="XAUUSD",
                timeframe="M1",
                open=float(open_),
                high=float(high),
                low=float(low),
                close=float(close),
                volume=100.0,
            )
        )

    return bars


class TestValidation:
    def test_empty_bars_return_empty_result(self):
        result = TechnicalIndicatorEngine.sma(
            [],
            period=3,
        )

        assert result == []

    def test_period_must_be_positive(self):
        bars = make_bars([1, 2, 3])

        with pytest.raises(ValueError):
            TechnicalIndicatorEngine.sma(
                bars,
                period=0,
            )

    def test_period_must_be_integer(self):
        bars = make_bars([1, 2, 3])

        with pytest.raises(ValueError):
            TechnicalIndicatorEngine.sma(
                bars,
                period=2.5,
            )

    def test_insufficient_history_returns_none(self):
        bars = make_bars([1, 2])

        result = TechnicalIndicatorEngine.sma(
            bars,
            period=3,
        )

        assert result == [None, None]


class TestSMA:
    def test_sma(self):
        bars = make_bars(
            [1, 2, 3, 4, 5]
        )

        result = TechnicalIndicatorEngine.sma(
            bars,
            period=3,
        )

        assert result == [
            None,
            None,
            pytest.approx(2.0),
            pytest.approx(3.0),
            pytest.approx(4.0),
        ]

    def test_sma_uses_trailing_window(self):
        bars = make_bars(
            [10, 20, 30, 40, 50]
        )

        result = TechnicalIndicatorEngine.sma(
            bars,
            period=2,
        )

        assert result[0] is None
        assert result[1] == pytest.approx(15.0)
        assert result[2] == pytest.approx(25.0)
        assert result[3] == pytest.approx(35.0)
        assert result[4] == pytest.approx(45.0)

    def test_sma_no_lookahead(self):
        bars_short = make_bars(
            [1, 2, 3, 4, 5]
        )

        bars_long = make_bars(
            [1, 2, 3, 4, 5, 1000]
        )

        short_result = TechnicalIndicatorEngine.sma(
            bars_short,
            period=3,
        )

        long_result = TechnicalIndicatorEngine.sma(
            bars_long,
            period=3,
        )

        assert short_result[4] == long_result[4]


class TestEMA:
    def test_ema(self):
        bars = make_bars(
            [1, 2, 3, 4, 5]
        )

        result = TechnicalIndicatorEngine.ema(
            bars,
            period=3,
        )

        assert result[0] is None
        assert result[1] is None
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_ema_reacts_to_new_price(self):
        bars = make_bars(
            [10, 10, 10, 20]
        )

        result = TechnicalIndicatorEngine.ema(
            bars,
            period=3,
        )

        assert result[2] == pytest.approx(10.0)
        assert result[3] == pytest.approx(15.0)

    def test_ema_no_lookahead(self):
        bars_short = make_bars(
            [1, 2, 3, 4, 5]
        )

        bars_long = make_bars(
            [1, 2, 3, 4, 5, 1000]
        )

        short_result = TechnicalIndicatorEngine.ema(
            bars_short,
            period=3,
        )

        long_result = TechnicalIndicatorEngine.ema(
            bars_long,
            period=3,
        )

        assert short_result[4] == long_result[4]


class TestRSI:
    def test_rsi_requires_period_plus_one_bars(self):
        bars = make_bars(
            [1, 2, 3]
        )

        result = TechnicalIndicatorEngine.rsi(
            bars,
            period=3,
        )

        assert result[0] is None
        assert result[1] is None
        assert result[2] is None

    def test_rsi_all_gains_is_100(self):
        bars = make_bars(
            [1, 2, 3, 4, 5, 6]
        )

        result = TechnicalIndicatorEngine.rsi(
            bars,
            period=3,
        )

        assert result[3] == pytest.approx(100.0)
        assert result[4] == pytest.approx(100.0)
        assert result[5] == pytest.approx(100.0)

    def test_rsi_all_losses_is_0(self):
        bars = make_bars(
            [6, 5, 4, 3, 2, 1]
        )

        result = TechnicalIndicatorEngine.rsi(
            bars,
            period=3,
        )

        assert result[3] == pytest.approx(0.0)
        assert result[4] == pytest.approx(0.0)
        assert result[5] == pytest.approx(0.0)

    def test_rsi_flat_market_is_50(self):
        bars = make_bars(
            [10, 10, 10, 10, 10, 10]
        )

        result = TechnicalIndicatorEngine.rsi(
            bars,
            period=3,
        )

        assert result[3] == pytest.approx(50.0)
        assert result[4] == pytest.approx(50.0)
        assert result[5] == pytest.approx(50.0)

    def test_rsi_mixed_market_is_between_zero_and_hundred(self):
        bars = make_bars(
            [10, 11, 10, 12, 11, 13, 12]
        )

        result = TechnicalIndicatorEngine.rsi(
            bars,
            period=3,
        )

        for value in result:
            if value is not None:
                assert 0.0 <= value <= 100.0

    def test_rsi_no_lookahead(self):
        bars_short = make_bars(
            [10, 11, 10, 12, 11, 13]
        )

        bars_long = make_bars(
            [10, 11, 10, 12, 11, 13, 1000]
        )

        short_result = TechnicalIndicatorEngine.rsi(
            bars_short,
            period=3,
        )

        long_result = TechnicalIndicatorEngine.rsi(
            bars_long,
            period=3,
        )

        assert short_result[5] == long_result[5]


class TestATR:
    def test_atr(self):
        bars = make_custom_bars(
            [
                (10, 12, 9, 11),
                (11, 14, 10, 13),
                (13, 16, 11, 15),
                (15, 17, 13, 14),
            ]
        )

        result = TechnicalIndicatorEngine.atr(
            bars,
            period=3,
        )

        assert result[0] is None
        assert result[1] is None
        assert result[2] is None

        assert result[3] == pytest.approx(
            (4.0 + 5.0 + 4.0) / 3.0
        )

    def test_atr_uses_previous_close_for_true_range(self):
        bars = make_custom_bars(
            [
                (100, 101, 99, 100),
                (110, 111, 109, 110),
                (120, 121, 119, 120),
                (130, 131, 129, 130),
            ]
        )

        result = TechnicalIndicatorEngine.atr(
            bars,
            period=3,
        )

        assert result[0] is None
        assert result[1] is None
        assert result[2] is None

        assert result[3] == pytest.approx(
            11.0
        )

    def test_atr_insufficient_history(self):
        bars = make_bars(
            [100, 101, 102]
        )

        result = TechnicalIndicatorEngine.atr(
            bars,
            period=14,
        )

        assert all(
            value is None
            for value in result
        )


class TestMACD:
    def test_macd_returns_macd_result(self):
        bars = make_bars(
            list(range(1, 40))
        )

        result = TechnicalIndicatorEngine.macd(
            bars,
            fast_period=3,
            slow_period=5,
            signal_period=3,
        )

        assert isinstance(
            result,
            MACDResult,
        )

    def test_macd_requires_slow_history(self):
        bars = make_bars(
            [1, 2, 3, 4]
        )

        result = TechnicalIndicatorEngine.macd(
            bars,
            fast_period=3,
            slow_period=5,
            signal_period=3,
        )

        assert all(
            value is None
            for value in result.macd
        )

    def test_macd_starts_after_slow_period(self):
        bars = make_bars(
            list(range(1, 20))
        )

        result = TechnicalIndicatorEngine.macd(
            bars,
            fast_period=3,
            slow_period=5,
            signal_period=3,
        )

        assert result.macd[0] is None
        assert result.macd[1] is None
        assert result.macd[2] is None
        assert result.macd[3] is None
        assert result.macd[4] is not None

    def test_macd_signal_requires_signal_history(self):
        bars = make_bars(
            list(range(1, 20))
        )

        result = TechnicalIndicatorEngine.macd(
            bars,
            fast_period=3,
            slow_period=5,
            signal_period=3,
        )

        assert result.signal[4] is None
        assert result.signal[5] is None
        assert result.signal[6] == pytest.approx(
            1.0
        )

    def test_macd_histogram_is_macd_minus_signal(self):
        bars = make_bars(
            list(range(1, 40))
        )

        result = TechnicalIndicatorEngine.macd(
            bars,
            fast_period=3,
            slow_period=5,
            signal_period=3,
        )

        for index in range(len(bars)):
            if (
                result.macd[index] is not None
                and result.signal[index] is not None
            ):
                assert result.histogram[index] == pytest.approx(
                    result.macd[index]
                    - result.signal[index]
                )

    def test_macd_no_lookahead(self):
        bars_short = make_bars(
            list(range(1, 20))
        )

        bars_long = make_bars(
            list(range(1, 21))
        )

        short_result = TechnicalIndicatorEngine.macd(
            bars_short,
            fast_period=3,
            slow_period=5,
            signal_period=3,
        )

        long_result = TechnicalIndicatorEngine.macd(
            bars_long,
            fast_period=3,
            slow_period=5,
            signal_period=3,
        )

        assert short_result.macd[10] == long_result.macd[10]
        assert short_result.signal[10] == long_result.signal[10]
        assert short_result.histogram[10] == long_result.histogram[10]


class TestBollingerBands:
    def test_bollinger_returns_result(self):
        bars = make_bars(
            [1, 2, 3, 4, 5]
        )

        result = TechnicalIndicatorEngine.bollinger_bands(
            bars,
            period=3,
            stddev_multiplier=2,
        )

        assert isinstance(
            result,
            BollingerBands,
        )

    def test_bollinger_middle_is_sma(self):
        bars = make_bars(
            [1, 2, 3, 4, 5]
        )

        result = TechnicalIndicatorEngine.bollinger_bands(
            bars,
            period=3,
            stddev_multiplier=2,
        )

        assert result.middle[0] is None
        assert result.middle[1] is None
        assert result.middle[2] == pytest.approx(2.0)
        assert result.middle[3] == pytest.approx(3.0)
        assert result.middle[4] == pytest.approx(4.0)

    def test_bollinger_upper_and_lower(self):
        bars = make_bars(
            [1, 2, 3]
        )

        result = TechnicalIndicatorEngine.bollinger_bands(
            bars,
            period=3,
            stddev_multiplier=2,
        )

        expected_std = (2 / 3) ** 0.5

        assert result.upper[2] == pytest.approx(
            2.0 + 2.0 * expected_std
        )

        assert result.lower[2] == pytest.approx(
            2.0 - 2.0 * expected_std
        )

    def test_bollinger_bands_are_ordered(self):
        bars = make_bars(
            [10, 11, 12, 13, 14]
        )

        result = TechnicalIndicatorEngine.bollinger_bands(
            bars,
            period=3,
            stddev_multiplier=2,
        )

        for index in range(len(bars)):
            if result.middle[index] is not None:
                assert (
                    result.upper[index]
                    >= result.middle[index]
                )

                assert (
                    result.middle[index]
                    >= result.lower[index]
                )

    def test_bollinger_zero_multiplier_is_rejected(self):
        bars = make_bars(
            [10, 11, 12, 13]
        )

        with pytest.raises(ValueError):
            TechnicalIndicatorEngine.bollinger_bands(
                bars,
                period=3,
                stddev_multiplier=0,
            )

    def test_bollinger_no_lookahead(self):
        bars_short = make_bars(
            [1, 2, 3, 4, 5]
        )

        bars_long = make_bars(
            [1, 2, 3, 4, 5, 1000]
        )

        short_result = TechnicalIndicatorEngine.bollinger_bands(
            bars_short,
            period=3,
            stddev_multiplier=2,
        )

        long_result = TechnicalIndicatorEngine.bollinger_bands(
            bars_long,
            period=3,
            stddev_multiplier=2,
        )

        assert short_result.middle[4] == long_result.middle[4]
        assert short_result.upper[4] == long_result.upper[4]
        assert short_result.lower[4] == long_result.lower[4]


class TestHighestHigh:
    def test_highest_high(self):
        bars = make_bars(
            [10, 20, 15, 30, 25]
        )

        result = TechnicalIndicatorEngine.highest_high(
            bars,
            period=3,
        )

        assert result == [
            None,
            None,
            pytest.approx(21.0),
            pytest.approx(31.0),
            pytest.approx(31.0),
        ]

    def test_highest_high_uses_high_values(self):
        bars = make_custom_bars(
            [
                (10, 100, 1, 10),
                (10, 20, 9, 10),
                (10, 30, 9, 10),
            ]
        )

        result = TechnicalIndicatorEngine.highest_high(
            bars,
            period=3,
        )

        assert result[2] == pytest.approx(
            100.0
        )


class TestLowestLow:
    def test_lowest_low(self):
        bars = make_bars(
            [10, 20, 15, 30, 25]
        )

        result = TechnicalIndicatorEngine.lowest_low(
            bars,
            period=3,
        )

        assert result == [
            None,
            None,
            pytest.approx(9.0),
            pytest.approx(14.0),
            pytest.approx(14.0),
        ]

    def test_lowest_low_uses_low_values(self):
        bars = make_custom_bars(
            [
                (10, 20, 1, 10),
                (10, 20, 9, 10),
                (10, 20, 8, 10),
            ]
        )

        result = TechnicalIndicatorEngine.lowest_low(
            bars,
            period=3,
        )

        assert result[2] == pytest.approx(
            1.0
        )


class TestMomentum:
    def test_momentum(self):
        bars = make_bars(
            [10, 20, 30, 40, 50]
        )

        result = TechnicalIndicatorEngine.momentum(
            bars,
            period=2,
        )

        assert result == [
            None,
            None,
            pytest.approx(20.0),
            pytest.approx(20.0),
            pytest.approx(20.0),
        ]

    def test_momentum_can_be_negative(self):
        bars = make_bars(
            [50, 40, 30, 20]
        )

        result = TechnicalIndicatorEngine.momentum(
            bars,
            period=2,
        )

        assert result[2] == pytest.approx(
            -20.0
        )

        assert result[3] == pytest.approx(
            -20.0
        )

    def test_momentum_no_lookahead(self):
        bars_short = make_bars(
            [10, 20, 30, 40, 50]
        )

        bars_long = make_bars(
            [10, 20, 30, 40, 50, 1000]
        )

        short_result = TechnicalIndicatorEngine.momentum(
            bars_short,
            period=2,
        )

        long_result = TechnicalIndicatorEngine.momentum(
            bars_long,
            period=2,
        )

        assert short_result[4] == long_result[4]


class TestOutputSafety:
    def test_sma_outputs_are_finite(self):
        bars = make_bars(
            [100, 101, 102, 103, 104, 105]
        )

        result = TechnicalIndicatorEngine.sma(
            bars,
            period=3,
        )

        for value in result:
            if value is not None:
                assert math.isfinite(value)

    def test_ema_outputs_are_finite(self):
        bars = make_bars(
            [100, 101, 102, 103, 104, 105]
        )

        result = TechnicalIndicatorEngine.ema(
            bars,
            period=3,
        )

        for value in result:
            if value is not None:
                assert math.isfinite(value)

    def test_rsi_outputs_are_finite(self):
        bars = make_bars(
            [100, 101, 99, 102, 98, 103]
        )

        result = TechnicalIndicatorEngine.rsi(
            bars,
            period=3,
        )

        for value in result:
            if value is not None:
                assert math.isfinite(value)

    def test_atr_outputs_are_finite(self):
        bars = make_bars(
            [100, 101, 102, 103, 104, 105]
        )

        result = TechnicalIndicatorEngine.atr(
            bars,
            period=3,
        )

        for value in result:
            if value is not None:
                assert math.isfinite(value)

    def test_macd_outputs_are_finite(self):
        bars = make_bars(
            list(range(100, 150))
        )

        result = TechnicalIndicatorEngine.macd(
            bars,
            fast_period=3,
            slow_period=5,
            signal_period=3,
        )

        for values in (
            result.macd,
            result.signal,
            result.histogram,
        ):
            for value in values:
                if value is not None:
                    assert math.isfinite(value)

    def test_bollinger_outputs_are_finite(self):
        bars = make_bars(
            list(range(100, 150))
        )

        result = TechnicalIndicatorEngine.bollinger_bands(
            bars,
            period=5,
            stddev_multiplier=2,
        )

        for values in (
            result.middle,
            result.upper,
            result.lower,
        ):
            for value in values:
                if value is not None:
                    assert math.isfinite(value)