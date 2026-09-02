from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from app.trading.context.market_context import (
    MarketCondition,
    MarketContext,
)
from app.trading.data.market_bar import MarketBar


class MarketRegimeError(ValueError):
    """
    Raised when market regime detection receives invalid input.
    """


class MarketRegime(str, Enum):
    """
    High-level market regime classification.
    """

    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    TRANSITION = "TRANSITION"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class RegimeReason:
    """
    Explanation for a market regime classification.
    """

    message: str


@dataclass(frozen=True, slots=True)
class MarketRegimeResult:
    """
    Result of market regime detection.
    """

    timestamp: object
    symbol: str
    timeframe: str

    regime: MarketRegime
    strength: float

    trend_strength: float
    volatility_ratio: float | None

    persistence_bars: int

    sufficient_history: bool

    reasons: tuple[RegimeReason, ...]

    @property
    def is_trending(self) -> bool:
        return self.regime in {
            MarketRegime.TRENDING_UP,
            MarketRegime.TRENDING_DOWN,
        }

    @property
    def is_ranging(self) -> bool:
        return self.regime == MarketRegime.RANGING

    @property
    def is_high_volatility(self) -> bool:
        return self.regime == MarketRegime.HIGH_VOLATILITY

    @property
    def is_transition(self) -> bool:
        return self.regime == MarketRegime.TRANSITION


class MarketRegimeEngine:
    """
    Detects the current market regime using existing Phase 2
    market-context information plus historical volatility.

    The engine does not create trading signals.

    Its purpose is to identify the current market environment:

        - trending up
        - trending down
        - ranging
        - high volatility
        - transition
        - unknown

    No future bars are used.
    """

    DEFAULT_VOLATILITY_LOOKBACK = 20
    DEFAULT_TREND_THRESHOLD = 60.0
    DEFAULT_RANGE_THRESHOLD = 40.0
    DEFAULT_HIGH_VOLATILITY_RATIO = 1.50
    DEFAULT_LOW_VOLATILITY_RATIO = 0.70
    DEFAULT_MINIMUM_HISTORY = 30
    DEFAULT_MINIMUM_PERSISTENCE = 2

    def __init__(
        self,
        volatility_lookback: int = DEFAULT_VOLATILITY_LOOKBACK,
        trend_threshold: float = DEFAULT_TREND_THRESHOLD,
        range_threshold: float = DEFAULT_RANGE_THRESHOLD,
        high_volatility_ratio: float = DEFAULT_HIGH_VOLATILITY_RATIO,
        low_volatility_ratio: float = DEFAULT_LOW_VOLATILITY_RATIO,
        minimum_history: int = DEFAULT_MINIMUM_HISTORY,
        minimum_persistence: int = DEFAULT_MINIMUM_PERSISTENCE,
    ) -> None:
        self.volatility_lookback = volatility_lookback
        self.trend_threshold = trend_threshold
        self.range_threshold = range_threshold
        self.high_volatility_ratio = high_volatility_ratio
        self.low_volatility_ratio = low_volatility_ratio
        self.minimum_history = minimum_history
        self.minimum_persistence = minimum_persistence

        self._validate_configuration()

    def detect(
        self,
        bars: list[MarketBar],
        context: MarketContext,
    ) -> MarketRegimeResult:
        """
        Detect the current market regime.

        The supplied context must describe the final bar in `bars`.
        """

        self._validate_bars(bars)
        self._validate_context(context, bars)

        sufficient_history = (
            len(bars) >= self.minimum_history
            and context.sufficient_history
        )

        volatility_ratio = self._calculate_volatility_ratio(
            bars
        )

        trend_strength = self._normalize_trend_strength(
            context.trend_strength
        )

        if not sufficient_history:
            return MarketRegimeResult(
                timestamp=bars[-1].timestamp,
                symbol=bars[-1].symbol,
                timeframe=bars[-1].timeframe,
                regime=MarketRegime.UNKNOWN,
                strength=0.0,
                trend_strength=trend_strength,
                volatility_ratio=volatility_ratio,
                persistence_bars=0,
                sufficient_history=False,
                reasons=(
                    RegimeReason(
                        "Insufficient historical data for reliable "
                        "regime detection."
                    ),
                ),
            )

        regime = self._classify_regime(
            context=context,
            volatility_ratio=volatility_ratio,
        )

        persistence = self._calculate_persistence(
            bars=bars,
            current_regime=regime,
        )

        if (
            persistence < self.minimum_persistence
            and regime != MarketRegime.HIGH_VOLATILITY
        ):
            regime = self._apply_persistence_filter(
                regime=regime,
                context=context,
                volatility_ratio=volatility_ratio,
            )

        strength = self._calculate_strength(
            regime=regime,
            context=context,
            volatility_ratio=volatility_ratio,
        )

        reasons = self._build_reasons(
            regime=regime,
            context=context,
            volatility_ratio=volatility_ratio,
            persistence=persistence,
        )

        return MarketRegimeResult(
            timestamp=bars[-1].timestamp,
            symbol=bars[-1].symbol,
            timeframe=bars[-1].timeframe,
            regime=regime,
            strength=strength,
            trend_strength=trend_strength,
            volatility_ratio=volatility_ratio,
            persistence_bars=persistence,
            sufficient_history=True,
            reasons=tuple(reasons),
        )

    def detect_from_contexts(
        self,
        bars: list[MarketBar],
        contexts: list[MarketContext],
        index: int = -1,
    ) -> MarketRegimeResult:
        """
        Detect regime using a selected context.
        """

        self._validate_bars(bars)

        if not isinstance(contexts, list):
            raise MarketRegimeError(
                "contexts must be a list."
            )

        if not contexts:
            raise MarketRegimeError(
                "contexts cannot be empty."
            )

        if index < 0:
            index += len(contexts)

        if index < 0 or index >= len(contexts):
            raise MarketRegimeError(
                "context index is out of range."
            )

        context = contexts[index]

        return self.detect(
            bars=bars,
            context=context,
        )

    def _classify_regime(
        self,
        context: MarketContext,
        volatility_ratio: float | None,
    ) -> MarketRegime:
        """
        Classify the current regime.

        Priority:

        1. High volatility
        2. Strong directional trend
        3. Range
        4. Transition
        """

        if volatility_ratio is not None:
            if volatility_ratio >= self.high_volatility_ratio:
                return MarketRegime.HIGH_VOLATILITY

        if (
            context.trend.value == "BULLISH"
            and context.trend_strength >= self.trend_threshold
        ):
            return MarketRegime.TRENDING_UP

        if (
            context.trend.value == "BEARISH"
            and context.trend_strength >= self.trend_threshold
        ):
            return MarketRegime.TRENDING_DOWN

        if (
            context.condition == MarketCondition.RANGING
            and context.context_strength <= self.range_threshold
        ):
            if (
                volatility_ratio is None
                or volatility_ratio <= self.high_volatility_ratio
            ):
                return MarketRegime.RANGING

        if context.condition in {
            MarketCondition.TRANSITION,
            MarketCondition.UNKNOWN,
        }:
            return MarketRegime.TRANSITION

        if context.context_strength < self.trend_threshold:
            return MarketRegime.TRANSITION

        return MarketRegime.TRANSITION

    def _calculate_strength(
        self,
        regime: MarketRegime,
        context: MarketContext,
        volatility_ratio: float | None,
    ) -> float:
        if regime == MarketRegime.UNKNOWN:
            return 0.0

        if regime == MarketRegime.TRENDING_UP:
            return self._clamp(
                context.trend_strength
            )

        if regime == MarketRegime.TRENDING_DOWN:
            return self._clamp(
                context.trend_strength
            )

        if regime == MarketRegime.RANGING:
            if volatility_ratio is None:
                return self._clamp(
                    100.0 - context.context_strength
                )

            if volatility_ratio <= 0:
                return 100.0

            if volatility_ratio <= self.low_volatility_ratio:
                return 100.0

            distance = (
                self.high_volatility_ratio
                - volatility_ratio
            )

            span = (
                self.high_volatility_ratio
                - self.low_volatility_ratio
            )

            if span <= 0:
                return 0.0

            return self._clamp(
                distance / span * 100.0
            )

        if regime == MarketRegime.HIGH_VOLATILITY:
            if volatility_ratio is None:
                return 0.0

            excess = (
                volatility_ratio
                - self.high_volatility_ratio
            )

            denominator = max(
                self.high_volatility_ratio,
                0.000001,
            )

            return self._clamp(
                50.0
                + (excess / denominator) * 50.0
            )

        return self._clamp(
            100.0
            - abs(
                context.context_strength
                - 50.0
            ) * 2.0
        )

    def _calculate_volatility_ratio(
        self,
        bars: list[MarketBar],
    ) -> float | None:
        """
        Compare the latest true range with the recent true-range
        baseline.
        """

        if len(bars) < 2:
            return None

        lookback = min(
            self.volatility_lookback,
            len(bars),
        )

        recent = bars[-lookback:]

        true_ranges: list[float] = []

        previous_close = None

        for bar in recent:
            if previous_close is None:
                previous_close = bar.close
                continue

            true_range = max(
                bar.high - bar.low,
                abs(bar.high - previous_close),
                abs(bar.low - previous_close),
            )

            if (
                not math.isfinite(true_range)
                or true_range < 0
            ):
                return None

            true_ranges.append(true_range)
            previous_close = bar.close

        if not true_ranges:
            return None

        current_range = true_ranges[-1]

        baseline = (
            sum(true_ranges)
            / len(true_ranges)
        )

        if baseline <= 0:
            return None

        return current_range / baseline

    def _calculate_persistence(
        self,
        bars: list[MarketBar],
        current_regime: MarketRegime,
    ) -> int:
        """
        Estimate the number of consecutive bars supporting the
        current regime.

        A directional trend requires STRICT movement in the expected
        direction.

        Equal closes do not count as trend persistence.

        This prevents a flat sequence from falsely appearing to be
        a persistent trend merely because it does not move against
        the trend.
        """

        if len(bars) < 2:
            return 1

        count = 1

        for index in range(
            len(bars) - 2,
            -1,
            -1,
        ):
            current = bars[index]
            following = bars[index + 1]

            movement = (
                following.close
                - current.close
            )

            current_range = (
                current.high
                - current.low
            )

            if current_range <= 0:
                break

            if current_regime == MarketRegime.TRENDING_UP:
                if movement > 0:
                    count += 1
                else:
                    break

            elif current_regime == MarketRegime.TRENDING_DOWN:
                if movement < 0:
                    count += 1
                else:
                    break

            elif current_regime == MarketRegime.RANGING:
                if (
                    abs(movement)
                    <= current_range * 1.5
                ):
                    count += 1
                else:
                    break

            elif current_regime == MarketRegime.HIGH_VOLATILITY:
                if (
                    current_range > 0
                    and abs(movement)
                    >= current_range * 0.25
                ):
                    count += 1
                else:
                    break

            else:
                break

        return count

    def _apply_persistence_filter(
        self,
        regime: MarketRegime,
        context: MarketContext,
        volatility_ratio: float | None,
    ) -> MarketRegime:
        """
        Prevent a newly appearing directional/ranging regime from
        being treated as mature.

        High volatility is retained because a volatility shock is
        itself meaningful information.
        """

        if regime == MarketRegime.HIGH_VOLATILITY:
            return regime

        if regime in {
            MarketRegime.TRENDING_UP,
            MarketRegime.TRENDING_DOWN,
            MarketRegime.RANGING,
        }:
            return MarketRegime.TRANSITION

        return regime

    def _build_reasons(
        self,
        regime: MarketRegime,
        context: MarketContext,
        volatility_ratio: float | None,
        persistence: int,
    ) -> list[RegimeReason]:
        reasons: list[RegimeReason] = []

        if regime == MarketRegime.TRENDING_UP:
            reasons.append(
                RegimeReason(
                    "Market structure and context indicate "
                    "a bullish directional environment."
                )
            )

        elif regime == MarketRegime.TRENDING_DOWN:
            reasons.append(
                RegimeReason(
                    "Market structure and context indicate "
                    "a bearish directional environment."
                )
            )

        elif regime == MarketRegime.RANGING:
            reasons.append(
                RegimeReason(
                    "Context indicates a sideways/ranging market."
                )
            )

        elif regime == MarketRegime.HIGH_VOLATILITY:
            reasons.append(
                RegimeReason(
                    "Current true-range volatility is elevated "
                    "relative to the recent baseline."
                )
            )

        elif regime == MarketRegime.TRANSITION:
            reasons.append(
                RegimeReason(
                    "Trend and range evidence does not provide "
                    "a sufficiently stable directional classification."
                )
            )

        elif regime == MarketRegime.UNKNOWN:
            reasons.append(
                RegimeReason(
                    "The available information is insufficient "
                    "for a reliable regime classification."
                )
            )

        if volatility_ratio is not None:
            reasons.append(
                RegimeReason(
                    f"Volatility ratio is "
                    f"{volatility_ratio:.4f}."
                )
            )

        reasons.append(
            RegimeReason(
                f"Regime persistence is "
                f"{persistence} bar(s)."
            )
        )

        if context.conflicts:
            reasons.append(
                RegimeReason(
                    "Market context contains conflicting signals."
                )
            )

        return reasons

    @staticmethod
    def _validate_context(
        context: MarketContext,
        bars: list[MarketBar],
    ) -> None:
        if not isinstance(
            context,
            MarketContext,
        ):
            raise MarketRegimeError(
                "context must be a MarketContext."
            )

        latest = bars[-1]

        if context.symbol != latest.symbol:
            raise MarketRegimeError(
                "context symbol does not match the latest bar."
            )

        if context.timeframe != latest.timeframe:
            raise MarketRegimeError(
                "context timeframe does not match the latest bar."
            )

        if context.timestamp != latest.timestamp:
            raise MarketRegimeError(
                "context timestamp must match the latest bar."
            )

    @staticmethod
    def _validate_bars(
        bars: list[MarketBar],
    ) -> None:
        if not isinstance(
            bars,
            list,
        ):
            raise MarketRegimeError(
                "bars must be a list."
            )

        if not bars:
            raise MarketRegimeError(
                "bars cannot be empty."
            )

        symbol = bars[0].symbol
        timeframe = bars[0].timeframe

        previous_timestamp = None

        for bar in bars:
            if not isinstance(
                bar,
                MarketBar,
            ):
                raise MarketRegimeError(
                    "every bar must be a MarketBar."
                )

            if bar.symbol != symbol:
                raise MarketRegimeError(
                    "all bars must use the same symbol."
                )

            if bar.timeframe != timeframe:
                raise MarketRegimeError(
                    "all bars must use the same timeframe."
                )

            if (
                previous_timestamp is not None
                and bar.timestamp <= previous_timestamp
            ):
                raise MarketRegimeError(
                    "bars must be strictly chronological."
                )

            previous_timestamp = bar.timestamp

    def _validate_configuration(self) -> None:
        if (
            not isinstance(
                self.volatility_lookback,
                int,
            )
            or self.volatility_lookback <= 0
        ):
            raise MarketRegimeError(
                "volatility_lookback must be greater than 0."
            )

        if (
            not isinstance(
                self.minimum_history,
                int,
            )
            or self.minimum_history <= 0
        ):
            raise MarketRegimeError(
                "minimum_history must be greater than 0."
            )

        if (
            not isinstance(
                self.minimum_persistence,
                int,
            )
            or self.minimum_persistence <= 0
        ):
            raise MarketRegimeError(
                "minimum_persistence must be greater than 0."
            )

        numeric_values = (
            self.trend_threshold,
            self.range_threshold,
            self.high_volatility_ratio,
            self.low_volatility_ratio,
        )

        for value in numeric_values:
            if not isinstance(
                value,
                (int, float),
            ):
                raise MarketRegimeError(
                    "regime thresholds must be numeric."
                )

            if not math.isfinite(float(value)):
                raise MarketRegimeError(
                    "regime thresholds must be finite."
                )

        if not (
            0.0
            <= self.range_threshold
            <= self.trend_threshold
            <= 100.0
        ):
            raise MarketRegimeError(
                "range_threshold and trend_threshold must "
                "satisfy 0 <= range <= trend <= 100."
            )

        if (
            self.low_volatility_ratio <= 0
            or self.high_volatility_ratio <= 0
        ):
            raise MarketRegimeError(
                "volatility ratios must be greater than 0."
            )

        if (
            self.low_volatility_ratio
            >= self.high_volatility_ratio
        ):
            raise MarketRegimeError(
                "low_volatility_ratio must be less than "
                "high_volatility_ratio."
            )

    @staticmethod
    def _normalize_trend_strength(
        value: float,
    ) -> float:
        if not isinstance(
            value,
            (int, float),
        ):
            return 0.0

        if not math.isfinite(float(value)):
            return 0.0

        return max(
            0.0,
            min(100.0, float(value)),
        )

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:
        return max(
            0.0,
            min(100.0, float(value)),
        )