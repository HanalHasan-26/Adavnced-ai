from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from statistics import mean
from typing import Sequence

from app.trading.data.market_bar import MarketBar
from app.trading.indicators.technical_indicators import (
    BollingerBands,
    MACDResult,
    TechnicalIndicatorEngine,
)
from app.trading.structure.market_structure import (
    MarketStructureEngine,
    StructureTrend,
)


class ContextBias(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class ContextSignalType(str, Enum):
    STRUCTURE = "STRUCTURE"
    RSI = "RSI"
    MACD = "MACD"
    VOLATILITY = "VOLATILITY"
    PRICE_LOCATION = "PRICE_LOCATION"


class MarketCondition(str, Enum):
    UNKNOWN = "UNKNOWN"
    RANGING = "RANGING"
    TRANSITION = "TRANSITION"
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"


@dataclass(frozen=True, slots=True)
class ContextSignal:
    signal_type: ContextSignalType
    bias: ContextBias
    strength: float
    value: float | None = None


@dataclass(frozen=True, slots=True)
class MarketContext:
    timestamp: object
    symbol: str
    timeframe: str
    close: float

    trend: StructureTrend
    trend_strength: float

    rsi: float | None
    atr: float | None
    macd: MACDResult | None
    bollinger_bands: BollingerBands | None

    price_location: float | None
    volatility_ratio: float | None

    bias: ContextBias
    context_strength: float
    condition: MarketCondition

    signals: tuple[ContextSignal, ...]
    conflicts: tuple[ContextSignalType, ...]

    sufficient_history: bool


class MarketContextEngine:
    """
    Combines deterministic technical indicators and market structure
    into a single market-context assessment.

    This engine does not generate trades.

    It provides:
        - structure trend
        - trend strength
        - RSI
        - ATR
        - MACD
        - Bollinger Bands
        - price location
        - volatility ratio
        - directional bias
        - market condition
        - signal conflicts
        - sufficient-history state

    All analysis is deterministic and uses only the supplied bars.

    analyze_at() is specifically designed for historical analysis and
    prevents future candles from being used.
    """

    DEFAULT_RSI_PERIOD = 14
    DEFAULT_ATR_PERIOD = 14

    DEFAULT_MACD_FAST = 12
    DEFAULT_MACD_SLOW = 26
    DEFAULT_MACD_SIGNAL = 9

    DEFAULT_BOLLINGER_PERIOD = 20
    DEFAULT_BOLLINGER_MULTIPLIER = 2.0

    DEFAULT_PRICE_RANGE_LOOKBACK = 20

    DEFAULT_TREND_THRESHOLD = 60.0
    DEFAULT_NEUTRAL_THRESHOLD = 40.0

    DEFAULT_MINIMUM_HISTORY = 30

    def __init__(
        self,
        rsi_period: int = DEFAULT_RSI_PERIOD,
        atr_period: int = DEFAULT_ATR_PERIOD,
        price_range_lookback: int = DEFAULT_PRICE_RANGE_LOOKBACK,
        macd_fast: int = DEFAULT_MACD_FAST,
        macd_slow: int = DEFAULT_MACD_SLOW,
        macd_signal: int = DEFAULT_MACD_SIGNAL,
        bollinger_period: int = DEFAULT_BOLLINGER_PERIOD,
        bollinger_multiplier: float = DEFAULT_BOLLINGER_MULTIPLIER,
        trend_threshold: float = DEFAULT_TREND_THRESHOLD,
        neutral_threshold: float = DEFAULT_NEUTRAL_THRESHOLD,
        minimum_history: int = DEFAULT_MINIMUM_HISTORY,
    ) -> None:
        self._validate_positive_integer(
            rsi_period,
            "rsi_period",
        )

        self._validate_positive_integer(
            atr_period,
            "atr_period",
        )

        self._validate_positive_integer(
            price_range_lookback,
            "price_range_lookback",
        )

        self._validate_positive_integer(
            macd_fast,
            "macd_fast",
        )

        self._validate_positive_integer(
            macd_slow,
            "macd_slow",
        )

        self._validate_positive_integer(
            macd_signal,
            "macd_signal",
        )

        if macd_fast >= macd_slow:
            raise ValueError(
                "macd_fast must be smaller than macd_slow."
            )

        self._validate_positive_integer(
            bollinger_period,
            "bollinger_period",
        )

        self._validate_positive_number(
            bollinger_multiplier,
            "bollinger_multiplier",
        )

        self._validate_percentage(
            trend_threshold,
            "trend_threshold",
        )

        self._validate_percentage(
            neutral_threshold,
            "neutral_threshold",
        )

        if neutral_threshold >= trend_threshold:
            raise ValueError(
                "neutral_threshold must be smaller than "
                "trend_threshold."
            )

        self._validate_positive_integer(
            minimum_history,
            "minimum_history",
        )

        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.price_range_lookback = price_range_lookback

        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

        self.bollinger_period = bollinger_period
        self.bollinger_multiplier = float(
            bollinger_multiplier
        )

        self.trend_threshold = float(
            trend_threshold
        )

        self.neutral_threshold = float(
            neutral_threshold
        )

        self.minimum_history = minimum_history

        self.indicator_engine = (
            TechnicalIndicatorEngine()
        )

        self.structure_engine = (
            MarketStructureEngine()
        )

    # =========================================================
    # VALIDATION
    # =========================================================

    @staticmethod
    def _validate_positive_integer(
        value: int,
        name: str,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                f"{name} must be a positive integer."
            )

    @staticmethod
    def _validate_positive_number(
        value: float,
        name: str,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(
                f"{name} must be a positive finite number."
            )

        numeric_value = float(value)

        if (
            not math.isfinite(numeric_value)
            or numeric_value <= 0
        ):
            raise ValueError(
                f"{name} must be a positive finite number."
            )

    @staticmethod
    def _validate_percentage(
        value: float,
        name: str,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(
                f"{name} must be between 0 and 100."
            )

        numeric_value = float(value)

        if (
            not math.isfinite(numeric_value)
            or numeric_value < 0
            or numeric_value > 100
        ):
            raise ValueError(
                f"{name} must be between 0 and 100."
            )

    @staticmethod
    def _validate_bars(
        bars: list[MarketBar],
    ) -> None:
        if not isinstance(bars, list):
            raise ValueError(
                "bars must be a list of MarketBar objects."
            )

        for index, bar in enumerate(bars):
            if not isinstance(bar, MarketBar):
                raise ValueError(
                    f"bars[{index}] must be a MarketBar."
                )

    # =========================================================
    # PUBLIC ANALYSIS
    # =========================================================

    def analyze(
        self,
        bars: list[MarketBar],
    ) -> MarketContext:
        """
        Analyze the latest available candle.
        """
        self._validate_bars(bars)

        if not bars:
            raise ValueError(
                "bars cannot be empty."
            )

        return self.analyze_at(
            bars,
            len(bars) - 1,
        )

    def analyze_at(
        self,
        bars: list[MarketBar],
        index: int,
    ) -> MarketContext:
        """
        Analyze the market using bars up to and including `index`.

        No candles after `index` are used.
        """
        self._validate_bars(bars)

        if not bars:
            raise ValueError(
                "bars cannot be empty."
            )

        if (
            isinstance(index, bool)
            or not isinstance(index, int)
        ):
            raise ValueError(
                "index must be an integer."
            )

        if index < 0:
            raise ValueError(
                "index cannot be negative."
            )

        if index >= len(bars):
            raise ValueError(
                "index is outside the available bars."
            )

        available = bars[: index + 1]

        latest = available[-1]

        rsi_values = self.indicator_engine.rsi(
            available,
            period=self.rsi_period,
        )

        atr_values = self.indicator_engine.atr(
            available,
            period=self.atr_period,
        )

        macd_values = self.indicator_engine.macd(
            available,
            fast_period=self.macd_fast,
            slow_period=self.macd_slow,
            signal_period=self.macd_signal,
        )

        bollinger_values = (
            self.indicator_engine.bollinger_bands(
                available,
                period=self.bollinger_period,
                stddev_multiplier=self.bollinger_multiplier,
            )
        )

        rsi = self._latest(
            rsi_values
        )

        atr = self._latest(
            atr_values
        )

        macd = macd_values

        bollinger_bands = (
            bollinger_values
            if (
                bollinger_values.middle
                and bollinger_values.middle[-1] is not None
            )
            else None
        )

        structure = self.structure_engine.analyze(
            available
        )

        price_location = (
            self._calculate_price_location(
                available
            )
        )

        volatility_ratio = (
            self._calculate_volatility_ratio(
                latest.close,
                atr,
            )
        )

        trend_strength = (
            self._calculate_trend_strength(
                available,
                structure.trend,
            )
        )

        signals = self._build_signals(
            structure_trend=structure.trend,
            trend_strength=trend_strength,
            rsi=rsi,
            macd=macd,
            price_location=price_location,
            volatility_ratio=volatility_ratio,
        )

        context_strength = (
            self._calculate_context_strength(
                signals
            )
        )

        bias = self._calculate_context_bias(
            signals
        )

        condition = self._calculate_condition(
            structure.trend,
            context_strength,
            trend_strength,
        )

        conflicts = self._calculate_conflicts(
            signals
        )

        sufficient_history = (
            len(available)
            >= self.minimum_history
        )

        return MarketContext(
            timestamp=latest.timestamp,
            symbol=latest.symbol,
            timeframe=latest.timeframe,
            close=float(latest.close),
            trend=structure.trend,
            trend_strength=trend_strength,
            rsi=rsi,
            atr=atr,
            macd=macd,
            bollinger_bands=bollinger_bands,
            price_location=price_location,
            volatility_ratio=volatility_ratio,
            bias=bias,
            context_strength=context_strength,
            condition=condition,
            signals=tuple(signals),
            conflicts=conflicts,
            sufficient_history=sufficient_history,
        )

    # =========================================================
    # INDICATOR HELPERS
    # =========================================================

    @staticmethod
    def _latest(
        values: Sequence[float | None],
    ) -> float | None:
        if not values:
            return None

        value = values[-1]

        if value is None:
            return None

        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            return None

        return numeric_value

    def _calculate_price_location(
        self,
        bars: list[MarketBar],
    ) -> float | None:
        if len(bars) < self.price_range_lookback:
            return None

        window = bars[
            -self.price_range_lookback:
        ]

        highest = max(
            float(bar.high)
            for bar in window
        )

        lowest = min(
            float(bar.low)
            for bar in window
        )

        current_close = float(
            bars[-1].close
        )

        price_range = highest - lowest

        if price_range <= 0:
            return 50.0

        location = (
            (current_close - lowest)
            / price_range
        ) * 100.0

        return max(
            0.0,
            min(100.0, location),
        )

    @staticmethod
    def _calculate_volatility_ratio(
        close: float,
        atr: float | None,
    ) -> float | None:
        if atr is None:
            return None

        if close <= 0:
            return None

        ratio = (
            float(atr)
            / float(close)
        ) * 100.0

        if not math.isfinite(ratio):
            return None

        if ratio <= 0:
            return None

        return ratio

    # =========================================================
    # TREND STRENGTH
    # =========================================================

    def _calculate_trend_strength(
        self,
        bars: list[MarketBar],
        trend: StructureTrend,
    ) -> float:
        if len(bars) < 2:
            return 0.0

        lookback = min(
            self.price_range_lookback,
            len(bars) - 1,
        )

        if lookback <= 0:
            return 0.0

        start_close = float(
            bars[-lookback - 1].close
        )

        end_close = float(
            bars[-1].close
        )

        if start_close <= 0:
            return 0.0

        percentage_change = (
            abs(end_close - start_close)
            / start_close
        ) * 100.0

        movement_strength = min(
            100.0,
            percentage_change * 10.0,
        )

        if trend in (
            StructureTrend.BULLISH,
            StructureTrend.BEARISH,
        ):
            return max(
                50.0,
                movement_strength,
            )

        if trend == StructureTrend.RANGE:
            return min(
                50.0,
                movement_strength,
            )

        return movement_strength

    # =========================================================
    # SIGNAL BUILDING
    # =========================================================

    def _build_signals(
        self,
        structure_trend: StructureTrend,
        trend_strength: float,
        rsi: float | None,
        macd: MACDResult | None,
        price_location: float | None,
        volatility_ratio: float | None,
    ) -> list[ContextSignal]:
        signals: list[ContextSignal] = []

        signals.append(
            self._structure_signal(
                structure_trend,
                trend_strength,
            )
        )

        signals.append(
            self._rsi_signal(
                rsi
            )
        )

        signals.append(
            self._macd_signal(
                macd
            )
        )

        signals.append(
            self._volatility_signal(
                volatility_ratio
            )
        )

        signals.append(
            self._price_location_signal(
                price_location
            )
        )

        return signals

    @staticmethod
    def _structure_signal(
        trend: StructureTrend,
        strength: float,
    ) -> ContextSignal:
        if trend == StructureTrend.BULLISH:
            return ContextSignal(
                signal_type=ContextSignalType.STRUCTURE,
                bias=ContextBias.BULLISH,
                strength=max(
                    0.0,
                    min(100.0, strength),
                ),
                value=strength,
            )

        if trend == StructureTrend.BEARISH:
            return ContextSignal(
                signal_type=ContextSignalType.STRUCTURE,
                bias=ContextBias.BEARISH,
                strength=max(
                    0.0,
                    min(100.0, strength),
                ),
                value=strength,
            )

        return ContextSignal(
            signal_type=ContextSignalType.STRUCTURE,
            bias=ContextBias.NEUTRAL,
            strength=0.0,
            value=strength,
        )

    @staticmethod
    def _rsi_signal(
        rsi: float | None,
    ) -> ContextSignal:
        if rsi is None:
            return ContextSignal(
                signal_type=ContextSignalType.RSI,
                bias=ContextBias.NEUTRAL,
                strength=0.0,
                value=None,
            )

        if rsi > 50.0:
            strength = min(
                100.0,
                abs(rsi - 50.0) * 2.0,
            )

            return ContextSignal(
                signal_type=ContextSignalType.RSI,
                bias=ContextBias.BULLISH,
                strength=strength,
                value=rsi,
            )

        if rsi < 50.0:
            strength = min(
                100.0,
                abs(rsi - 50.0) * 2.0,
            )

            return ContextSignal(
                signal_type=ContextSignalType.RSI,
                bias=ContextBias.BEARISH,
                strength=strength,
                value=rsi,
            )

        return ContextSignal(
            signal_type=ContextSignalType.RSI,
            bias=ContextBias.NEUTRAL,
            strength=0.0,
            value=rsi,
        )

    @staticmethod
    def _macd_signal(
        macd: MACDResult | None,
    ) -> ContextSignal:
        if macd is None:
            return ContextSignal(
                signal_type=ContextSignalType.MACD,
                bias=ContextBias.NEUTRAL,
                strength=0.0,
                value=None,
            )

        if not macd.histogram:
            return ContextSignal(
                signal_type=ContextSignalType.MACD,
                bias=ContextBias.NEUTRAL,
                strength=0.0,
                value=None,
            )

        histogram = macd.histogram[-1]

        if histogram is None:
            return ContextSignal(
                signal_type=ContextSignalType.MACD,
                bias=ContextBias.NEUTRAL,
                strength=0.0,
                value=None,
            )

        histogram = float(histogram)

        if not math.isfinite(histogram):
            return ContextSignal(
                signal_type=ContextSignalType.MACD,
                bias=ContextBias.NEUTRAL,
                strength=0.0,
                value=None,
            )

        strength = min(
            100.0,
            abs(histogram) * 100.0,
        )

        if histogram > 0:
            return ContextSignal(
                signal_type=ContextSignalType.MACD,
                bias=ContextBias.BULLISH,
                strength=strength,
                value=histogram,
            )

        if histogram < 0:
            return ContextSignal(
                signal_type=ContextSignalType.MACD,
                bias=ContextBias.BEARISH,
                strength=strength,
                value=histogram,
            )

        return ContextSignal(
            signal_type=ContextSignalType.MACD,
            bias=ContextBias.NEUTRAL,
            strength=0.0,
            value=histogram,
        )

    @staticmethod
    def _volatility_signal(
        volatility_ratio: float | None,
    ) -> ContextSignal:
        """
        Volatility itself is not directional.

        Therefore this signal never creates a bullish or
        bearish directional bias.
        """
        if volatility_ratio is None:
            return ContextSignal(
                signal_type=ContextSignalType.VOLATILITY,
                bias=ContextBias.NEUTRAL,
                strength=0.0,
                value=None,
            )

        return ContextSignal(
            signal_type=ContextSignalType.VOLATILITY,
            bias=ContextBias.NEUTRAL,
            strength=0.0,
            value=volatility_ratio,
        )

    @staticmethod
    def _price_location_signal(
        price_location: float | None,
    ) -> ContextSignal:
        if price_location is None:
            return ContextSignal(
                signal_type=ContextSignalType.PRICE_LOCATION,
                bias=ContextBias.NEUTRAL,
                strength=0.0,
                value=None,
            )

        if price_location >= 60.0:
            strength = min(
                100.0,
                (price_location - 50.0) * 2.0,
            )

            return ContextSignal(
                signal_type=ContextSignalType.PRICE_LOCATION,
                bias=ContextBias.BULLISH,
                strength=strength,
                value=price_location,
            )

        if price_location <= 40.0:
            strength = min(
                100.0,
                (50.0 - price_location) * 2.0,
            )

            return ContextSignal(
                signal_type=ContextSignalType.PRICE_LOCATION,
                bias=ContextBias.BEARISH,
                strength=strength,
                value=price_location,
            )

        return ContextSignal(
            signal_type=ContextSignalType.PRICE_LOCATION,
            bias=ContextBias.NEUTRAL,
            strength=0.0,
            value=price_location,
        )

    # =========================================================
    # CONTEXT BIAS
    # =========================================================

    @staticmethod
    def _calculate_context_bias(
        signals: Sequence[ContextSignal],
    ) -> ContextBias:
        bullish_strength = sum(
            signal.strength
            for signal in signals
            if signal.bias == ContextBias.BULLISH
        )

        bearish_strength = sum(
            signal.strength
            for signal in signals
            if signal.bias == ContextBias.BEARISH
        )

        if (
            bullish_strength == 0
            and bearish_strength == 0
        ):
            return ContextBias.NEUTRAL

        if bullish_strength > bearish_strength:
            return ContextBias.BULLISH

        if bearish_strength > bullish_strength:
            return ContextBias.BEARISH

        return ContextBias.NEUTRAL

    @staticmethod
    def _calculate_context_strength(
        signals: Sequence[ContextSignal],
    ) -> float:
        bullish_strength = sum(
            signal.strength
            for signal in signals
            if signal.bias == ContextBias.BULLISH
        )

        bearish_strength = sum(
            signal.strength
            for signal in signals
            if signal.bias == ContextBias.BEARISH
        )

        total = (
            bullish_strength
            + bearish_strength
        )

        if total <= 0:
            return 0.0

        directional_difference = abs(
            bullish_strength
            - bearish_strength
        )

        strength = (
            directional_difference
            / total
        ) * 100.0

        return max(
            0.0,
            min(100.0, strength),
        )

    # =========================================================
    # MARKET CONDITION
    # =========================================================

    def _calculate_condition(
        self,
        trend: StructureTrend,
        context_strength: float,
        trend_strength: float,
    ) -> MarketCondition:
        if trend == StructureTrend.BULLISH:
            if (
                trend_strength
                >= self.trend_threshold
            ):
                return MarketCondition.TRENDING_UP

            if (
                context_strength
                <= self.neutral_threshold
            ):
                return MarketCondition.TRANSITION

            return MarketCondition.TRENDING_UP

        if trend == StructureTrend.BEARISH:
            if (
                trend_strength
                >= self.trend_threshold
            ):
                return MarketCondition.TRENDING_DOWN

            if (
                context_strength
                <= self.neutral_threshold
            ):
                return MarketCondition.TRANSITION

            return MarketCondition.TRENDING_DOWN

        if trend == StructureTrend.RANGE:
            return MarketCondition.RANGING

        return MarketCondition.UNKNOWN

    # =========================================================
    # CONFLICT DETECTION
    # =========================================================

    @staticmethod
    def _calculate_conflicts(
        signals: Sequence[ContextSignal],
    ) -> tuple[ContextSignalType, ...]:
        bullish_types = {
            signal.signal_type
            for signal in signals
            if signal.bias == ContextBias.BULLISH
        }

        bearish_types = {
            signal.signal_type
            for signal in signals
            if signal.bias == ContextBias.BEARISH
        }

        conflict_types = (
            bullish_types
            & bearish_types
        )

        return tuple(
            sorted(
                conflict_types,
                key=lambda item: item.value,
            )
        )