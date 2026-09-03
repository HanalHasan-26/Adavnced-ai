from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import isfinite

from app.trading.data.market_bar import MarketBar
from app.trading.structure.market_structure import (
    MarketStructureEngine,
    SwingLabel,
    SwingPoint,
    SwingType,
)


class SupportResistanceError(ValueError):
    """Raised when support/resistance analysis validation fails."""


class SRLevelType(str, Enum):
    """Type of a support/resistance level."""

    SUPPORT = "SUPPORT"
    RESISTANCE = "RESISTANCE"


class SRZoneStatus(str, Enum):
    """Current status of a support/resistance zone."""

    ACTIVE = "ACTIVE"
    TESTED = "TESTED"
    BROKEN = "BROKEN"
    UNTESTED = "UNTESTED"


@dataclass(frozen=True, slots=True)
class SRLevel:
    """A raw support/resistance level derived from a swing."""

    index: int
    confirmation_index: int
    timestamp: datetime
    level_type: SRLevelType
    price: float
    swing_label: SwingLabel | None
    touch_count: int
    strength: float
    distance_from_price: float

    @property
    def is_support(self) -> bool:
        return self.level_type is SRLevelType.SUPPORT

    @property
    def is_resistance(self) -> bool:
        return self.level_type is SRLevelType.RESISTANCE

    @property
    def is_strong(self) -> bool:
        return self.strength >= 70.0


@dataclass(frozen=True, slots=True)
class SRZone:
    """A clustered support/resistance zone."""

    zone_type: SRLevelType
    lower_price: float
    upper_price: float
    center_price: float
    width: float

    touch_count: int
    source_level_count: int
    strength: float

    status: SRZoneStatus
    distance_from_price: float

    first_timestamp: datetime
    last_timestamp: datetime

    source_indices: tuple[int, ...]

    @property
    def is_support(self) -> bool:
        return self.zone_type is SRLevelType.SUPPORT

    @property
    def is_resistance(self) -> bool:
        return self.zone_type is SRLevelType.RESISTANCE

    @property
    def is_active(self) -> bool:
        return self.status in (
            SRZoneStatus.ACTIVE,
            SRZoneStatus.TESTED,
            SRZoneStatus.UNTESTED,
        )

    @property
    def is_broken(self) -> bool:
        return self.status is SRZoneStatus.BROKEN

    @property
    def is_strong(self) -> bool:
        return self.strength >= 70.0

    def contains(self, price: float) -> bool:
        """Return whether price lies inside the zone."""
        if not isinstance(price, (int, float)) or isinstance(price, bool):
            return False

        numeric_price = float(price)

        if not isfinite(numeric_price):
            return False

        return (
            self.lower_price
            <= numeric_price
            <= self.upper_price
        )


class SupportResistanceEngine:
    """
    Deterministic support/resistance intelligence engine.

    Support is derived from swing lows.
    Resistance is derived from swing highs.

    Nearby levels are clustered into zones. Zone strength is based on
    source-level count, touch count, swing significance, and recency.

    The engine does not:
    - generate trade entries
    - generate SL/TP
    - calculate position size
    - execute trades
    - call an LLM
    - fetch external data
    """

    DEFAULT_SWING_LEFT = 2
    DEFAULT_SWING_RIGHT = 2
    DEFAULT_ZONE_TOLERANCE = 0.001
    DEFAULT_MIN_ZONE_STRENGTH = 20.0
    DEFAULT_MAX_ZONES = 20

    def __init__(
        self,
        swing_left: int = DEFAULT_SWING_LEFT,
        swing_right: int = DEFAULT_SWING_RIGHT,
        zone_tolerance: float = DEFAULT_ZONE_TOLERANCE,
        min_zone_strength: float = DEFAULT_MIN_ZONE_STRENGTH,
        max_zones: int = DEFAULT_MAX_ZONES,
    ) -> None:
        self.swing_left = self._validate_window(
            swing_left,
            "swing_left",
        )

        self.swing_right = self._validate_window(
            swing_right,
            "swing_right",
        )

        self.zone_tolerance = self._validate_non_negative(
            zone_tolerance,
            "zone_tolerance",
        )

        self.min_zone_strength = self._validate_threshold(
            min_zone_strength,
            "min_zone_strength",
        )

        if isinstance(max_zones, bool) or not isinstance(
            max_zones,
            int,
        ):
            raise SupportResistanceError(
                "max_zones must be an integer."
            )

        if max_zones < 1:
            raise SupportResistanceError(
                "max_zones must be at least 1."
            )

        self.max_zones = max_zones

        self._structure_engine = MarketStructureEngine(
            left_window=self.swing_left,
            right_window=self.swing_right,
            break_on_close=True,
        )

    def analyze(
        self,
        bars: list[MarketBar],
    ) -> tuple[SRZone, ...]:
        """
        Analyze all available support/resistance zones.

        Returns zones ordered by distance from the latest price,
        then by strength descending.
        """

        self._validate_bars(bars)

        if not bars:
            return ()

        structure = self._structure_engine.analyze(bars)

        levels = self._build_levels(
            bars,
            structure.swings,
        )

        zones = self._build_zones(
            bars,
            levels,
        )

        return tuple(zones[: self.max_zones])

    def analyze_levels(
        self,
        bars: list[MarketBar],
    ) -> tuple[SRLevel, ...]:
        """Return raw support/resistance levels."""

        self._validate_bars(bars)

        if not bars:
            return ()

        structure = self._structure_engine.analyze(bars)

        levels = self._build_levels(
            bars,
            structure.swings,
        )

        return tuple(levels)

    def analyze_xauusd(
        self,
        bars: list[MarketBar],
    ) -> tuple[SRZone, ...]:
        """Analyze support/resistance specifically for XAUUSD."""

        self._validate_bars(bars)

        if bars and bars[0].symbol != "XAUUSD":
            raise SupportResistanceError(
                "bars must use XAUUSD."
            )

        return self.analyze(bars)

    def nearest_support(
        self,
        bars: list[MarketBar],
    ) -> SRZone | None:
        """Return the nearest active support below current price."""

        zones = self.analyze(bars)

        current_price = bars[-1].close

        supports = [
            zone
            for zone in zones
            if zone.is_support
            and zone.is_active
            and zone.center_price <= current_price
        ]

        if not supports:
            return None

        return min(
            supports,
            key=lambda zone: (
                abs(current_price - zone.center_price),
                -zone.strength,
            ),
        )

    def nearest_resistance(
        self,
        bars: list[MarketBar],
    ) -> SRZone | None:
        """Return the nearest active resistance above current price."""

        zones = self.analyze(bars)

        current_price = bars[-1].close

        resistances = [
            zone
            for zone in zones
            if zone.is_resistance
            and zone.is_active
            and zone.center_price >= current_price
        ]

        if not resistances:
            return None

        return min(
            resistances,
            key=lambda zone: (
                abs(current_price - zone.center_price),
                -zone.strength,
            ),
        )

    def _build_levels(
        self,
        bars: list[MarketBar],
        swings: tuple[SwingPoint, ...],
    ) -> list[SRLevel]:
        latest_price = bars[-1].close

        levels: list[SRLevel] = []

        for swing in swings:
            if swing.swing_type is SwingType.LOW:
                level_type = SRLevelType.SUPPORT
            elif swing.swing_type is SwingType.HIGH:
                level_type = SRLevelType.RESISTANCE
            else:
                continue

            touch_count = self._count_touches(
                bars,
                swing.price,
                level_type,
            )

            strength = self._calculate_level_strength(
                bars,
                swing,
                touch_count,
            )

            distance = abs(
                latest_price - swing.price
            )

            levels.append(
                SRLevel(
                    index=swing.index,
                    confirmation_index=swing.confirmation_index,
                    timestamp=swing.timestamp,
                    level_type=level_type,
                    price=float(swing.price),
                    swing_label=swing.label,
                    touch_count=touch_count,
                    strength=strength,
                    distance_from_price=distance,
                )
            )

        levels.sort(
            key=lambda level: (
                level.distance_from_price,
                -level.strength,
                level.index,
            )
        )

        return levels

    def _build_zones(
        self,
        bars: list[MarketBar],
        levels: list[SRLevel],
    ) -> list[SRZone]:
        if not levels:
            return []

        current_price = bars[-1].close

        support_levels = [
            level
            for level in levels
            if level.level_type is SRLevelType.SUPPORT
        ]

        resistance_levels = [
            level
            for level in levels
            if level.level_type is SRLevelType.RESISTANCE
        ]

        zones: list[SRZone] = []

        zones.extend(
            self._cluster_levels(
                bars,
                support_levels,
                current_price,
            )
        )

        zones.extend(
            self._cluster_levels(
                bars,
                resistance_levels,
                current_price,
            )
        )

        zones = [
            zone
            for zone in zones
            if zone.strength >= self.min_zone_strength
        ]

        zones.sort(
            key=lambda zone: (
                zone.distance_from_price,
                -zone.strength,
                zone.center_price,
            )
        )

        return zones

    def _cluster_levels(
        self,
        bars: list[MarketBar],
        levels: list[SRLevel],
        current_price: float,
    ) -> list[SRZone]:
        if not levels:
            return []

        sorted_levels = sorted(
            levels,
            key=lambda level: level.price,
        )

        clusters: list[list[SRLevel]] = []

        for level in sorted_levels:
            if not clusters:
                clusters.append([level])
                continue

            cluster = clusters[-1]

            cluster_center = sum(
                item.price
                for item in cluster
            ) / len(cluster)

            tolerance = self._zone_tolerance_amount(
                cluster_center,
                bars,
            )

            if abs(level.price - cluster_center) <= tolerance:
                cluster.append(level)
            else:
                clusters.append([level])

        zones: list[SRZone] = []

        for cluster in clusters:
            zones.append(
                self._create_zone(
                    bars,
                    cluster,
                    current_price,
                )
            )

        return zones

    def _create_zone(
        self,
        bars: list[MarketBar],
        levels: list[SRLevel],
        current_price: float,
    ) -> SRZone:
        prices = [
            level.price
            for level in levels
        ]

        lower_price = min(prices)
        upper_price = max(prices)

        center_price = sum(prices) / len(prices)

        touch_count = sum(
            level.touch_count
            for level in levels
        )

        strength = self._calculate_zone_strength(
            levels
        )

        status = self._determine_zone_status(
            bars,
            lower_price,
            upper_price,
            levels[0].level_type,
        )

        first_timestamp = min(
            level.timestamp
            for level in levels
        )

        last_timestamp = max(
            level.timestamp
            for level in levels
        )

        source_indices = tuple(
            sorted(
                level.index
                for level in levels
            )
        )

        return SRZone(
            zone_type=levels[0].level_type,
            lower_price=lower_price,
            upper_price=upper_price,
            center_price=center_price,
            width=upper_price - lower_price,
            touch_count=touch_count,
            source_level_count=len(levels),
            strength=strength,
            status=status,
            distance_from_price=abs(
                current_price - center_price
            ),
            first_timestamp=first_timestamp,
            last_timestamp=last_timestamp,
            source_indices=source_indices,
        )

    def _count_touches(
        self,
        bars: list[MarketBar],
        level_price: float,
        level_type: SRLevelType,
    ) -> int:
        if not bars:
            return 0

        tolerance = self._zone_tolerance_amount(
            level_price,
            bars,
        )

        touches = 0

        for bar in bars:
            if level_type is SRLevelType.SUPPORT:
                distance = abs(
                    bar.low - level_price
                )
            else:
                distance = abs(
                    bar.high - level_price
                )

            if distance <= tolerance:
                touches += 1

        return touches

    def _calculate_level_strength(
        self,
        bars: list[MarketBar],
        swing: SwingPoint,
        touch_count: int,
    ) -> float:
        score = 20.0

        if swing.label in (
            SwingLabel.HH,
            SwingLabel.LL,
        ):
            score += 25.0

        elif swing.label in (
            SwingLabel.HL,
            SwingLabel.LH,
        ):
            score += 15.0

        score += min(
            30.0,
            max(
                0.0,
                float(touch_count - 1) * 10.0,
            ),
        )

        recency = self._recency_score(
            len(bars),
            swing.index,
        )

        score += recency * 25.0

        return round(
            max(
                0.0,
                min(
                    100.0,
                    score,
                ),
            ),
            6,
        )

    def _calculate_zone_strength(
        self,
        levels: list[SRLevel],
    ) -> float:
        if not levels:
            return 0.0

        strongest = max(
            level.strength
            for level in levels
        )

        source_bonus = min(
            20.0,
            max(
                0.0,
                float(len(levels) - 1) * 10.0,
            ),
        )

        touch_bonus = min(
            15.0,
            max(
                0.0,
                float(
                    sum(
                        level.touch_count
                        for level in levels
                    )
                    - len(levels)
                )
                * 2.5,
            ),
        )

        return round(
            max(
                0.0,
                min(
                    100.0,
                    strongest
                    + source_bonus
                    + touch_bonus,
                ),
            ),
            6,
        )

    @staticmethod
    def _recency_score(
        total_bars: int,
        swing_index: int,
    ) -> float:
        if total_bars <= 1:
            return 1.0

        age = max(
            0,
            total_bars - 1 - swing_index,
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

    def _determine_zone_status(
        self,
        bars: list[MarketBar],
        lower_price: float,
        upper_price: float,
        level_type: SRLevelType,
    ) -> SRZoneStatus:
        if not bars:
            return SRZoneStatus.UNTESTED

        latest_index = len(bars) - 1

        latest_bar = bars[-1]

        tolerance = self._zone_tolerance_amount(
            (lower_price + upper_price) / 2.0,
            bars,
        )

        tested = False
        broken = False

        for index, bar in enumerate(bars):
            if index == latest_index:
                continue

            if level_type is SRLevelType.SUPPORT:
                if bar.close < lower_price - tolerance:
                    broken = True

                if (
                    bar.low
                    <= upper_price + tolerance
                    and bar.high
                    >= lower_price - tolerance
                ):
                    tested = True

            else:
                if bar.close > upper_price + tolerance:
                    broken = True

                if (
                    bar.high
                    >= lower_price - tolerance
                    and bar.low
                    <= upper_price + tolerance
                ):
                    tested = True

        if level_type is SRLevelType.SUPPORT:
            if latest_bar.close < lower_price - tolerance:
                return SRZoneStatus.BROKEN

        else:
            if latest_bar.close > upper_price + tolerance:
                return SRZoneStatus.BROKEN

        if broken:
            return SRZoneStatus.BROKEN

        if tested:
            return SRZoneStatus.TESTED

        return SRZoneStatus.UNTESTED

    def _zone_tolerance_amount(
        self,
        price: float,
        bars: list[MarketBar],
    ) -> float:
        """
        Convert percentage-style tolerance into an absolute price
        distance.

        A small minimum based on the median candle range prevents
        excessively tiny zones on instruments with natural price
        movement.
        """

        percentage_tolerance = (
            abs(price)
            * self.zone_tolerance
        )

        ranges = [
            bar.range
            for bar in bars
            if isfinite(bar.range)
            and bar.range > 0.0
        ]

        if not ranges:
            return percentage_tolerance

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
            percentage_tolerance,
            median_range * 0.10,
        )

    @staticmethod
    def _validate_bars(
        bars: list[MarketBar],
    ) -> None:
        if not isinstance(bars, list):
            raise SupportResistanceError(
                "bars must be a list."
            )

        if not bars:
            return

        for bar in bars:
            if not isinstance(bar, MarketBar):
                raise SupportResistanceError(
                    "all bars must be MarketBar instances."
                )

        symbol = bars[0].symbol
        timeframe = bars[0].timeframe

        for bar in bars:
            if bar.symbol != symbol:
                raise SupportResistanceError(
                    "all bars must use the same symbol."
                )

            if bar.timeframe != timeframe:
                raise SupportResistanceError(
                    "all bars must use the same timeframe."
                )

        for previous, current in zip(
            bars,
            bars[1:],
        ):
            if current.timestamp <= previous.timestamp:
                raise SupportResistanceError(
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
            raise SupportResistanceError(
                f"{name} must be an integer."
            )

        if value < 1:
            raise SupportResistanceError(
                f"{name} must be at least 1."
            )

        return value

    @staticmethod
    def _validate_non_negative(
        value: float,
        name: str,
    ) -> float:
        if isinstance(value, bool):
            raise SupportResistanceError(
                f"{name} must be numeric."
            )

        if not isinstance(value, (int, float)):
            raise SupportResistanceError(
                f"{name} must be numeric."
            )

        value = float(value)

        if not isfinite(value):
            raise SupportResistanceError(
                f"{name} must be finite."
            )

        if value < 0.0:
            raise SupportResistanceError(
                f"{name} cannot be negative."
            )

        return value

    @staticmethod
    def _validate_threshold(
        value: float,
        name: str,
    ) -> float:
        if isinstance(value, bool):
            raise SupportResistanceError(
                f"{name} must be numeric."
            )

        if not isinstance(value, (int, float)):
            raise SupportResistanceError(
                f"{name} must be numeric."
            )

        value = float(value)

        if not isfinite(value):
            raise SupportResistanceError(
                f"{name} must be finite."
            )

        if value < 0.0 or value > 100.0:
            raise SupportResistanceError(
                f"{name} must be between 0 and 100."
            )

        return value