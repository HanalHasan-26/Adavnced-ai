from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean
from typing import Sequence

from app.trading.data.market_bar import MarketBar


Number = float | int


@dataclass(frozen=True, slots=True)
class MACDResult:
    """MACD line, signal line, and histogram aligned to the input bars."""

    macd: list[float | None]
    signal: list[float | None]
    histogram: list[float | None]


@dataclass(frozen=True, slots=True)
class BollingerBands:
    """Bollinger middle, upper, and lower bands aligned to the input bars."""

    middle: list[float | None]
    upper: list[float | None]
    lower: list[float | None]


class TechnicalIndicatorEngine:
    """Deterministic, no-look-ahead technical indicator calculations.

    Every method returns one value per input bar. Values that cannot be
    calculated yet are represented by None rather than using future data.

    EMA-based indicators are seeded with a simple moving average.
    RSI and ATR use Wilder-style smoothing.
    """

    DEFAULT_SMA_PERIOD = 20
    DEFAULT_EMA_PERIOD = 20
    DEFAULT_RSI_PERIOD = 14
    DEFAULT_ATR_PERIOD = 14

    DEFAULT_MACD_FAST = 12
    DEFAULT_MACD_SLOW = 26
    DEFAULT_MACD_SIGNAL = 9

    DEFAULT_BOLLINGER_PERIOD = 20
    DEFAULT_BOLLINGER_STDDEV = 2.0

    DEFAULT_LOOKBACK = 14

    @staticmethod
    def _validate_bars(
        bars: Sequence[MarketBar],
    ) -> list[MarketBar]:
        if not isinstance(bars, Sequence) or isinstance(bars, (str, bytes)):
            raise ValueError(
                "bars must be a sequence of MarketBar objects."
            )

        values = list(bars)

        for index, bar in enumerate(values):
            if not isinstance(bar, MarketBar):
                raise ValueError(
                    f"bars[{index}] must be a MarketBar."
                )

        return values

    @staticmethod
    def _validate_period(
        period: int,
        name: str = "period",
    ) -> None:
        if isinstance(period, bool) or not isinstance(period, int):
            raise ValueError(
                f"{name} must be a positive integer."
            )

        if period <= 0:
            raise ValueError(
                f"{name} must be a positive integer."
            )

    @staticmethod
    def _validate_multiplier(
        multiplier: Number,
        name: str,
    ) -> float:
        if (
            isinstance(multiplier, bool)
            or not isinstance(multiplier, (int, float))
        ):
            raise ValueError(
                f"{name} must be a finite number."
            )

        value = float(multiplier)

        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"{name} must be a finite number greater than 0."
            )

        return value

    @staticmethod
    def _closes(
        bars: Sequence[MarketBar],
    ) -> list[float]:
        return [float(bar.close) for bar in bars]

    @staticmethod
    def _sma_values(
        values: Sequence[float],
        period: int,
    ) -> list[float | None]:
        TechnicalIndicatorEngine._validate_period(period)

        result: list[float | None] = [None] * len(values)

        if len(values) < period:
            return result

        window_sum = sum(values[:period])

        result[period - 1] = window_sum / period

        for index in range(period, len(values)):
            window_sum += values[index]
            window_sum -= values[index - period]

            result[index] = window_sum / period

        return result

    @staticmethod
    def _ema_values(
        values: Sequence[float],
        period: int,
    ) -> list[float | None]:
        TechnicalIndicatorEngine._validate_period(period)

        result: list[float | None] = [None] * len(values)

        if len(values) < period:
            return result

        previous = sum(values[:period]) / period

        result[period - 1] = previous

        multiplier = 2.0 / (period + 1.0)

        for index in range(period, len(values)):
            previous = (
                (values[index] - previous) * multiplier
            ) + previous

            result[index] = previous

        return result

    @staticmethod
    def sma(
        bars: Sequence[MarketBar],
        period: int = DEFAULT_SMA_PERIOD,
    ) -> list[float | None]:
        """Simple moving average of closing prices."""

        values = TechnicalIndicatorEngine._closes(
            TechnicalIndicatorEngine._validate_bars(bars)
        )

        return TechnicalIndicatorEngine._sma_values(
            values,
            period,
        )

    @staticmethod
    def ema(
        bars: Sequence[MarketBar],
        period: int = DEFAULT_EMA_PERIOD,
    ) -> list[float | None]:
        """Exponential moving average of closing prices."""

        values = TechnicalIndicatorEngine._closes(
            TechnicalIndicatorEngine._validate_bars(bars)
        )

        return TechnicalIndicatorEngine._ema_values(
            values,
            period,
        )

    @staticmethod
    def rsi(
        bars: Sequence[MarketBar],
        period: int = DEFAULT_RSI_PERIOD,
    ) -> list[float | None]:
        """Wilder RSI of closing prices."""

        values = TechnicalIndicatorEngine._closes(
            TechnicalIndicatorEngine._validate_bars(bars)
        )

        TechnicalIndicatorEngine._validate_period(period)

        result: list[float | None] = [None] * len(values)

        if len(values) <= period:
            return result

        gains: list[float] = []
        losses: list[float] = []

        for index in range(1, len(values)):
            change = values[index] - values[index - 1]

            gains.append(max(change, 0.0))
            losses.append(max(-change, 0.0))

        average_gain = sum(gains[:period]) / period
        average_loss = sum(losses[:period]) / period

        result[period] = (
            TechnicalIndicatorEngine._rsi_from_averages(
                average_gain,
                average_loss,
            )
        )

        for gain, loss, index in zip(
            gains[period:],
            losses[period:],
            range(period + 1, len(values)),
        ):
            average_gain = (
                (average_gain * (period - 1)) + gain
            ) / period

            average_loss = (
                (average_loss * (period - 1)) + loss
            ) / period

            result[index] = (
                TechnicalIndicatorEngine._rsi_from_averages(
                    average_gain,
                    average_loss,
                )
            )

        return result

    @staticmethod
    def _rsi_from_averages(
        average_gain: float,
        average_loss: float,
    ) -> float:
        if average_loss == 0.0:
            if average_gain == 0.0:
                return 50.0

            return 100.0

        relative_strength = average_gain / average_loss

        return 100.0 - (
            100.0 / (1.0 + relative_strength)
        )

    @staticmethod
    def atr(
        bars: Sequence[MarketBar],
        period: int = DEFAULT_ATR_PERIOD,
    ) -> list[float | None]:
        """Wilder ATR using true range and prior close."""

        values = TechnicalIndicatorEngine._validate_bars(bars)

        TechnicalIndicatorEngine._validate_period(period)

        result: list[float | None] = [None] * len(values)

        if len(values) <= period:
            return result

        true_ranges: list[float] = [
            float(values[0].high - values[0].low)
        ]

        for index in range(1, len(values)):
            bar = values[index]

            previous_close = float(
                values[index - 1].close
            )

            true_range = max(
                float(bar.high - bar.low),
                abs(float(bar.high) - previous_close),
                abs(float(bar.low) - previous_close),
            )

            true_ranges.append(true_range)

        average_true_range = (
            sum(true_ranges[1 : period + 1]) / period
        )

        result[period] = average_true_range

        for index in range(period + 1, len(values)):
            average_true_range = (
                (
                    average_true_range * (period - 1)
                )
                + true_ranges[index]
            ) / period

            result[index] = average_true_range

        return result

    @staticmethod
    def macd(
        bars: Sequence[MarketBar],
        fast_period: int = DEFAULT_MACD_FAST,
        slow_period: int = DEFAULT_MACD_SLOW,
        signal_period: int = DEFAULT_MACD_SIGNAL,
    ) -> MACDResult:
        """MACD using standard EMA periods."""

        values = TechnicalIndicatorEngine._closes(
            TechnicalIndicatorEngine._validate_bars(bars)
        )

        TechnicalIndicatorEngine._validate_period(
            fast_period,
            "fast_period",
        )

        TechnicalIndicatorEngine._validate_period(
            slow_period,
            "slow_period",
        )

        TechnicalIndicatorEngine._validate_period(
            signal_period,
            "signal_period",
        )

        if fast_period >= slow_period:
            raise ValueError(
                "fast_period must be smaller than slow_period."
            )

        fast_ema = TechnicalIndicatorEngine._ema_values(
            values,
            fast_period,
        )

        slow_ema = TechnicalIndicatorEngine._ema_values(
            values,
            slow_period,
        )

        macd_line: list[float | None] = [None] * len(values)

        defined_macd: list[float] = []
        defined_indices: list[int] = []

        for index, (fast, slow) in enumerate(
            zip(fast_ema, slow_ema)
        ):
            if fast is not None and slow is not None:
                value = fast - slow

                macd_line[index] = value

                defined_macd.append(value)
                defined_indices.append(index)

        signal_values = TechnicalIndicatorEngine._ema_values(
            defined_macd,
            signal_period,
        )

        signal_line: list[float | None] = [None] * len(values)
        histogram: list[float | None] = [None] * len(values)

        for local_index, original_index in enumerate(
            defined_indices
        ):
            signal = signal_values[local_index]

            if signal is not None:
                signal_line[original_index] = signal

                histogram[original_index] = (
                    macd_line[original_index] - signal
                )

        return MACDResult(
            macd=macd_line,
            signal=signal_line,
            histogram=histogram,
        )

    @staticmethod
    def bollinger_bands(
        bars: Sequence[MarketBar],
        period: int = DEFAULT_BOLLINGER_PERIOD,
        stddev_multiplier: Number = DEFAULT_BOLLINGER_STDDEV,
    ) -> BollingerBands:
        """SMA middle band with population standard deviation bands."""

        values = TechnicalIndicatorEngine._closes(
            TechnicalIndicatorEngine._validate_bars(bars)
        )

        TechnicalIndicatorEngine._validate_period(period)

        multiplier = (
            TechnicalIndicatorEngine._validate_multiplier(
                stddev_multiplier,
                "stddev_multiplier",
            )
        )

        middle: list[float | None] = [None] * len(values)
        upper: list[float | None] = [None] * len(values)
        lower: list[float | None] = [None] * len(values)

        if len(values) < period:
            return BollingerBands(
                middle=middle,
                upper=upper,
                lower=lower,
            )

        for index in range(period - 1, len(values)):
            window = values[
                index - period + 1 : index + 1
            ]

            average = mean(window)

            variance = (
                sum(
                    (value - average) ** 2
                    for value in window
                )
                / period
            )

            deviation = math.sqrt(variance)

            middle[index] = average

            upper[index] = (
                average + (multiplier * deviation)
            )

            lower[index] = (
                average - (multiplier * deviation)
            )

        return BollingerBands(
            middle=middle,
            upper=upper,
            lower=lower,
        )

    @staticmethod
    def highest_high(
        bars: Sequence[MarketBar],
        period: int = DEFAULT_LOOKBACK,
    ) -> list[float | None]:
        """Highest high over the lookback window."""

        values = TechnicalIndicatorEngine._validate_bars(bars)

        TechnicalIndicatorEngine._validate_period(period)

        result: list[float | None] = [None] * len(values)

        for index in range(period - 1, len(values)):
            result[index] = max(
                float(bar.high)
                for bar in values[
                    index - period + 1 : index + 1
                ]
            )

        return result

    @staticmethod
    def lowest_low(
        bars: Sequence[MarketBar],
        period: int = DEFAULT_LOOKBACK,
    ) -> list[float | None]:
        """Lowest low over the lookback window."""

        values = TechnicalIndicatorEngine._validate_bars(bars)

        TechnicalIndicatorEngine._validate_period(period)

        result: list[float | None] = [None] * len(values)

        for index in range(period - 1, len(values)):
            result[index] = min(
                float(bar.low)
                for bar in values[
                    index - period + 1 : index + 1
                ]
            )

        return result

    @staticmethod
    def momentum(
        bars: Sequence[MarketBar],
        period: int = DEFAULT_LOOKBACK,
    ) -> list[float | None]:
        """Current close minus close `period` bars ago."""

        values = TechnicalIndicatorEngine._closes(
            TechnicalIndicatorEngine._validate_bars(bars)
        )

        TechnicalIndicatorEngine._validate_period(period)

        result: list[float | None] = [None] * len(values)

        for index in range(period, len(values)):
            result[index] = (
                values[index]
                - values[index - period]
            )

        return result