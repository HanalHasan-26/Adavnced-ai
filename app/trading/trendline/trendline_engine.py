from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from app.trading.data.market_bar import MarketBar
from app.trading.structure.market_structure import (
    MarketStructureEngine,
    SwingPoint,
    SwingType,
)


class TrendlineEngineError(ValueError):
    """Raised when trendline analysis validation fails."""


class TrendlineType(str, Enum):
    SUPPORT = "SUPPORT"
    RESISTANCE = "RESISTANCE"


class TrendlineStatus(str, Enum):
    ACTIVE = "ACTIVE"
    TESTED = "TESTED"
    BROKEN = "BROKEN"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class Trendline:
    """A validated trendline derived from market structure."""

    trendline_type: TrendlineType

    first_index: int
    second_index: int

    first_price: float
    second_price: float

    first_timestamp: object
    second_timestamp: object

    slope: float
    intercept: float

    touch_count: int
    rejection_count: int

    strength: float
    status: TrendlineStatus

    distance_from_price: float
    projected_price: float

    source_indices: tuple[int, ...]

    @property
    def is_support(self) -> bool:
        return self.trendline_type is TrendlineType.SUPPORT

    @property
    def is_resistance(self) -> bool:
        return self.trendline_type is TrendlineType.RESISTANCE

    @property
    def is_active(self) -> bool:
        return self.status in (
            TrendlineStatus.ACTIVE,
            TrendlineStatus.TESTED,
        )

    @property
    def is_broken(self) -> bool:
        return self.status is TrendlineStatus.BROKEN

    @property
    def is_strong(self) -> bool:
        return self.strength >= 70.0

    def price_at(self, index: int) -> float:
        """Return the trendline's projected price at a bar index."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise TrendlineEngineError(
                "index must be an integer."
            )

        return self.slope * index + self.intercept


class TrendlineEngine:
    """
    Deterministic trendline and price-structure engine.

    Support trendlines are built from rising swing lows.

    Resistance trendlines are built from falling swing highs.

    The engine identifies candidate connections between confirmed
    structural swing points, validates intervening prices, counts
    touches/rejections, calculates slope and strength, and determines
    whether a trendline remains active or has been broken.

    This engine does not make trade decisions.
    """

    DEFAULT_SWING_LEFT = 2
    DEFAULT_SWING_RIGHT = 2

    DEFAULT_TOUCH_TOLERANCE = 0.001
    DEFAULT_BREAK_TOLERANCE = 0.001

    DEFAULT_MIN_TOUCHES = 2
    DEFAULT_MIN_STRENGTH = 20.0

    DEFAULT_MAX_TRENDLINES = 20

    def __init__(
        self,
        swing_left: int = DEFAULT_SWING_LEFT,
        swing_right: int = DEFAULT_SWING_RIGHT,
        touch_tolerance: float = DEFAULT_TOUCH_TOLERANCE,
        break_tolerance: float = DEFAULT_BREAK_TOLERANCE,
        min_touches: int = DEFAULT_MIN_TOUCHES,
        min_strength: float = DEFAULT_MIN_STRENGTH,
        max_trendlines: int = DEFAULT_MAX_TRENDLINES,
    ) -> None:
        self.swing_left = self._validate_window(
            swing_left,
            "swing_left",
        )

        self.swing_right = self._validate_window(
            swing_right,
            "swing_right",
        )

        self.touch_tolerance = self._validate_non_negative(
            touch_tolerance,
            "touch_tolerance",
        )

        self.break_tolerance = self._validate_non_negative(
            break_tolerance,
            "break_tolerance",
        )

        if isinstance(min_touches, bool) or not isinstance(
            min_touches,
            int,
        ):
            raise TrendlineEngineError(
                "min_touches must be an integer."
            )

        if min_touches < 2:
            raise TrendlineEngineError(
                "min_touches must be at least 2."
            )

        self.min_touches = min_touches

        self.min_strength = self._validate_threshold(
            min_strength,
            "min_strength",
        )

        if isinstance(max_trendlines, bool) or not isinstance(
            max_trendlines,
            int,
        ):
            raise TrendlineEngineError(
                "max_trendlines must be an integer."
            )

        if max_trendlines < 1:
            raise TrendlineEngineError(
                "max_trendlines must be at least 1."
            )

        self.max_trendlines = max_trendlines

        self._structure_engine = MarketStructureEngine(
            left_window=self.swing_left,
            right_window=self.swing_right,
            break_on_close=True,
        )

    def analyze(
        self,
        bars: list[MarketBar],
    ) -> tuple[Trendline, ...]:
        """
        Analyze trendlines from available market structure.

        Returns the strongest/nearest validated trendlines first.
        """

        self._validate_bars(bars)

        if not bars:
            return ()

        structure = self._structure_engine.analyze(bars)

        swing_highs = [
            swing
            for swing in structure.swings
            if swing.swing_type is SwingType.HIGH
        ]

        swing_lows = [
            swing
            for swing in structure.swings
            if swing.swing_type is SwingType.LOW
        ]

        trendlines: list[Trendline] = []

        trendlines.extend(
            self._build_support_trendlines(
                bars,
                swing_lows,
            )
        )

        trendlines.extend(
            self._build_resistance_trendlines(
                bars,
                swing_highs,
            )
        )

        trendlines = [
            trendline
            for trendline in trendlines
            if trendline.status is not TrendlineStatus.INVALID
            and trendline.strength >= self.min_strength
        ]

        trendlines.sort(
            key=lambda trendline: (
                trendline.distance_from_price,
                -trendline.strength,
                trendline.first_index,
                trendline.second_index,
            )
        )

        return tuple(
            trendlines[: self.max_trendlines]
        )

    def analyze_support(
        self,
        bars: list[MarketBar],
    ) -> tuple[Trendline, ...]:
        """Return only support trendlines."""

        return tuple(
            trendline
            for trendline in self.analyze(bars)
            if trendline.is_support
        )

    def analyze_resistance(
        self,
        bars: list[MarketBar],
    ) -> tuple[Trendline, ...]:
        """Return only resistance trendlines."""

        return tuple(
            trendline
            for trendline in self.analyze(bars)
            if trendline.is_resistance
        )

    def nearest_support(
        self,
        bars: list[MarketBar],
    ) -> Trendline | None:
        """Return the nearest active support trendline."""

        self._validate_bars(bars)

        if not bars:
            return None

        current_price = bars[-1].close

        candidates = [
            trendline
            for trendline in self.analyze_support(bars)
            if trendline.is_active
            and trendline.projected_price <= current_price
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda trendline: (
                abs(
                    current_price
                    - trendline.projected_price
                ),
                -trendline.strength,
            ),
        )

    def nearest_resistance(
        self,
        bars: list[MarketBar],
    ) -> Trendline | None:
        """Return the nearest active resistance trendline."""

        self._validate_bars(bars)

        if not bars:
            return None

        current_price = bars[-1].close

        candidates = [
            trendline
            for trendline in self.analyze_resistance(bars)
            if trendline.is_active
            and trendline.projected_price >= current_price
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda trendline: (
                abs(
                    current_price
                    - trendline.projected_price
                ),
                -trendline.strength,
            ),
        )

    def analyze_xauusd(
        self,
        bars: list[MarketBar],
    ) -> tuple[Trendline, ...]:
        """Analyze trendlines specifically for XAUUSD."""

        self._validate_bars(bars)

        if bars and bars[0].symbol != "XAUUSD":
            raise TrendlineEngineError(
                "bars must use XAUUSD."
            )

        return self.analyze(bars)

    def _build_support_trendlines(
        self,
        bars: list[MarketBar],
        swings: list[SwingPoint],
    ) -> list[Trendline]:
        trendlines: list[Trendline] = []

        for first_position in range(
            len(swings)
        ):
            first = swings[first_position]

            for second in swings[
                first_position + 1 :
            ]:
                if second.index <= first.index:
                    continue

                if second.price <= first.price:
                    continue

                trendline = self._create_candidate(
                    bars,
                    first,
                    second,
                    TrendlineType.SUPPORT,
                )

                if trendline is not None:
                    trendlines.append(trendline)

        return trendlines

    def _build_resistance_trendlines(
        self,
        bars: list[MarketBar],
        swings: list[SwingPoint],
    ) -> list[Trendline]:
        trendlines: list[Trendline] = []

        for first_position in range(
            len(swings)
        ):
            first = swings[first_position]

            for second in swings[
                first_position + 1 :
            ]:
                if second.index <= first.index:
                    continue

                if second.price >= first.price:
                    continue

                trendline = self._create_candidate(
                    bars,
                    first,
                    second,
                    TrendlineType.RESISTANCE,
                )

                if trendline is not None:
                    trendlines.append(trendline)

        return trendlines

    def _create_candidate(
        self,
        bars: list[MarketBar],
        first: SwingPoint,
        second: SwingPoint,
        trendline_type: TrendlineType,
    ) -> Trendline | None:
        if second.index <= first.index:
            return None

        denominator = (
            second.index - first.index
        )

        if denominator <= 0:
            return None

        slope = (
            second.price - first.price
        ) / denominator

        intercept = (
            first.price
            - slope * first.index
        )

        if not isfinite(slope):
            return None

        if not isfinite(intercept):
            return None

        if not self._validate_intervening_prices(
            bars,
            first.index,
            second.index,
            slope,
            intercept,
            trendline_type,
        ):
            return None

        touch_count, rejection_count = (
            self._count_interactions(
                bars,
                first.index,
                second.index,
                slope,
                intercept,
                trendline_type,
            )
        )

        if touch_count < self.min_touches:
            return None

        strength = self._calculate_strength(
            bars,
            first,
            second,
            touch_count,
            rejection_count,
        )

        current_index = len(bars) - 1
        projected_price = (
            slope * current_index
            + intercept
        )

        current_price = bars[-1].close

        status = self._determine_status(
            bars,
            first.index,
            second.index,
            slope,
            intercept,
            trendline_type,
        )

        distance = abs(
            current_price
            - projected_price
        )

        return Trendline(
            trendline_type=trendline_type,
            first_index=first.index,
            second_index=second.index,
            first_price=float(first.price),
            second_price=float(second.price),
            first_timestamp=first.timestamp,
            second_timestamp=second.timestamp,
            slope=round(slope, 12),
            intercept=round(intercept, 12),
            touch_count=touch_count,
            rejection_count=rejection_count,
            strength=round(strength, 6),
            status=status,
            distance_from_price=distance,
            projected_price=projected_price,
            source_indices=(
                first.index,
                second.index,
            ),
        )

    def _validate_intervening_prices(
        self,
        bars: list[MarketBar],
        first_index: int,
        second_index: int,
        slope: float,
        intercept: float,
        trendline_type: TrendlineType,
    ) -> bool:
        tolerance = self._absolute_tolerance(
            slope * second_index + intercept,
            bars,
        )

        for index in range(
            first_index + 1,
            second_index,
        ):
            bar = bars[index]

            line_price = (
                slope * index
                + intercept
            )

            if trendline_type is TrendlineType.SUPPORT:
                if bar.low < line_price - tolerance:
                    return False
            else:
                if bar.high > line_price + tolerance:
                    return False

        return True

    def _count_interactions(
        self,
        bars: list[MarketBar],
        first_index: int,
        second_index: int,
        slope: float,
        intercept: float,
        trendline_type: TrendlineType,
    ) -> tuple[int, int]:
        tolerance = self._absolute_tolerance(
            slope * second_index + intercept,
            bars,
        )

        touches = 0
        rejections = 0

        for index, bar in enumerate(bars):
            if index < first_index:
                continue

            line_price = (
                slope * index
                + intercept
            )

            if trendline_type is TrendlineType.SUPPORT:
                distance = abs(
                    bar.low - line_price
                )

                if distance <= tolerance:
                    touches += 1

                    if bar.close > line_price:
                        rejections += 1

            else:
                distance = abs(
                    bar.high - line_price
                )

                if distance <= tolerance:
                    touches += 1

                    if bar.close < line_price:
                        rejections += 1

        return touches, rejections

    def _calculate_strength(
        self,
        bars: list[MarketBar],
        first: SwingPoint,
        second: SwingPoint,
        touch_count: int,
        rejection_count: int,
    ) -> float:
        score = 20.0

        score += min(
            30.0,
            max(
                0.0,
                (touch_count - 2) * 10.0,
            ),
        )

        score += min(
            20.0,
            max(
                0.0,
                rejection_count * 5.0,
            ),
        )

        span = second.index - first.index

        score += min(
            15.0,
            max(
                0.0,
                span / max(
                    1.0,
                    len(bars),
                )
                * 30.0,
            ),
        )

        recency = self._recency_score(
            len(bars),
            second.index,
        )

        score += recency * 15.0

        return max(
            0.0,
            min(
                100.0,
                score,
            ),
        )

    def _determine_status(
        self,
        bars: list[MarketBar],
        first_index: int,
        second_index: int,
        slope: float,
        intercept: float,
        trendline_type: TrendlineType,
    ) -> TrendlineStatus:
        if len(bars) <= second_index:
            return TrendlineStatus.TESTED

        tolerance = self._absolute_tolerance(
            slope * (len(bars) - 1) + intercept,
            bars,
            multiplier=self.break_tolerance,
        )

        touched_after_second = False

        for index in range(
            second_index + 1,
            len(bars),
        ):
            bar = bars[index]

            line_price = (
                slope * index
                + intercept
            )

            if trendline_type is TrendlineType.SUPPORT:
                if bar.close < line_price - tolerance:
                    return TrendlineStatus.BROKEN

                if (
                    bar.low
                    <= line_price + tolerance
                    and bar.high
                    >= line_price - tolerance
                ):
                    touched_after_second = True

            else:
                if bar.close > line_price + tolerance:
                    return TrendlineStatus.BROKEN

                if (
                    bar.high
                    >= line_price - tolerance
                    and bar.low
                    <= line_price + tolerance
                ):
                    touched_after_second = True

        if touched_after_second:
            return TrendlineStatus.TESTED

        return TrendlineStatus.ACTIVE

    def _absolute_tolerance(
        self,
        price: float,
        bars: list[MarketBar],
        multiplier: float | None = None,
    ) -> float:
        if multiplier is None:
            multiplier = self.touch_tolerance

        percentage_amount = (
            abs(price) * multiplier
        )

        ranges = [
            bar.range
            for bar in bars
            if isfinite(bar.range)
            and bar.range > 0.0
        ]

        if not ranges:
            return percentage_amount

        sorted_ranges = sorted(ranges)

        middle = len(sorted_ranges) // 2

        if len(sorted_ranges) % 2:
            median_range = sorted_ranges[middle]
        else:
            median_range = (
                sorted_ranges[middle - 1]
                + sorted_ranges[middle]
            ) / 2.0

        return max(
            percentage_amount,
            median_range * 0.10,
        )

    @staticmethod
    def _recency_score(
        total_bars: int,
        index: int,
    ) -> float:
        if total_bars <= 1:
            return 1.0

        age = max(
            0,
            total_bars - 1 - index,
        )

        normalized_age = min(
            1.0,
            age / max(
                1.0,
                total_bars * 0.50,
            ),
        )

        return max(
            0.0,
            1.0 - normalized_age,
        )

    @staticmethod
    def _validate_bars(
        bars: list[MarketBar],
    ) -> None:
        if not isinstance(bars, list):
            raise TrendlineEngineError(
                "bars must be a list."
            )

        if not bars:
            return

        for bar in bars:
            if not isinstance(bar, MarketBar):
                raise TrendlineEngineError(
                    "all bars must be MarketBar instances."
                )

        symbol = bars[0].symbol
        timeframe = bars[0].timeframe

        for bar in bars:
            if bar.symbol != symbol:
                raise TrendlineEngineError(
                    "all bars must use the same symbol."
                )

            if bar.timeframe != timeframe:
                raise TrendlineEngineError(
                    "all bars must use the same timeframe."
                )

        for previous, current in zip(
            bars,
            bars[1:],
        ):
            if current.timestamp <= previous.timestamp:
                raise TrendlineEngineError(
                    "bars must be strictly chronological."
                )

    @staticmethod
    def _validate_window(
        value: int,
        name: str,
    ) -> int:
        if isinstance(value, bool) or not isinstance(
            value,
            int,
        ):
            raise TrendlineEngineError(
                f"{name} must be an integer."
            )

        if value < 1:
            raise TrendlineEngineError(
                f"{name} must be at least 1."
            )

        return value

    @staticmethod
    def _validate_non_negative(
        value: float,
        name: str,
    ) -> float:
        if isinstance(value, bool):
            raise TrendlineEngineError(
                f"{name} must be numeric."
            )

        if not isinstance(value, (int, float)):
            raise TrendlineEngineError(
                f"{name} must be numeric."
            )

        value = float(value)

        if not isfinite(value):
            raise TrendlineEngineError(
                f"{name} must be finite."
            )

        if value < 0.0:
            raise TrendlineEngineError(
                f"{name} cannot be negative."
            )

        return value

    @staticmethod
    def _validate_threshold(
        value: float,
        name: str,
    ) -> float:
        if isinstance(value, bool):
            raise TrendlineEngineError(
                f"{name} must be numeric."
            )

        if not isinstance(value, (int, float)):
            raise TrendlineEngineError(
                f"{name} must be numeric."
            )

        value = float(value)

        if not isfinite(value):
            raise TrendlineEngineError(
                f"{name} must be finite."
            )

        if value < 0.0 or value > 100.0:
            raise TrendlineEngineError(
                f"{name} must be between 0 and 100."
            )

        return value