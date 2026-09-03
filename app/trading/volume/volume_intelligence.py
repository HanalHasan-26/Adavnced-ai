from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from app.trading.data.market_bar import MarketBar


class VolumeIntelligenceError(ValueError):
    """Raised when volume intelligence validation fails."""


class VolumeRegime(str, Enum):
    UNKNOWN = "UNKNOWN"
    NORMAL = "NORMAL"
    EXPANDING = "EXPANDING"
    CONTRACTING = "CONTRACTING"
    SPIKE = "SPIKE"


@dataclass(frozen=True, slots=True)
class VolumeProfile:
    """Volume distribution across price buckets."""

    poc: float | None
    value_area_high: float | None
    value_area_low: float | None

    high_volume_nodes: tuple[float, ...]
    low_volume_nodes: tuple[float, ...]

    total_volume: float
    bucket_size: float
    bucket_count: int

    sufficient_data: bool

    @property
    def has_poc(self) -> bool:
        return self.poc is not None

    @property
    def has_value_area(self) -> bool:
        return (
            self.value_area_high is not None
            and self.value_area_low is not None
        )

    @property
    def is_valid(self) -> bool:
        return self.sufficient_data and self.has_poc

    def price_location(self, price: float) -> str:
        if not isinstance(price, (int, float)) or isinstance(price, bool):
            raise VolumeIntelligenceError(
                "price must be numeric."
            )

        price = float(price)

        if not isfinite(price):
            raise VolumeIntelligenceError(
                "price must be finite."
            )

        if not self.has_value_area:
            return "UNKNOWN"

        if price > self.value_area_high:
            return "ABOVE_VALUE"

        if price < self.value_area_low:
            return "BELOW_VALUE"

        return "INSIDE_VALUE"


@dataclass(frozen=True, slots=True)
class VolumeIntelligence:
    """Complete deterministic volume analysis."""

    timestamp: object
    symbol: str
    timeframe: str

    current_volume: float
    average_volume: float
    relative_volume: float

    volume_regime: VolumeRegime
    volume_strength: float

    expanding: bool
    contracting: bool
    spike: bool

    profile: VolumeProfile

    price: float
    price_location: str
    distance_from_poc: float | None

    sufficient_data: bool

    @property
    def is_unknown(self) -> bool:
        return self.volume_regime is VolumeRegime.UNKNOWN

    @property
    def is_normal(self) -> bool:
        return self.volume_regime is VolumeRegime.NORMAL

    @property
    def is_expanding(self) -> bool:
        return (
            self.volume_regime is VolumeRegime.EXPANDING
            or self.expanding
        )

    @property
    def is_contracting(self) -> bool:
        return (
            self.volume_regime is VolumeRegime.CONTRACTING
            or self.contracting
        )

    @property
    def is_spike(self) -> bool:
        return (
            self.volume_regime is VolumeRegime.SPIKE
            or self.spike
        )

    @property
    def has_profile(self) -> bool:
        return self.profile.is_valid


class VolumeIntelligenceEngine:
    """
    Deterministic volume and volume-profile engine.

    Responsibilities:
    - volume statistics
    - relative volume
    - expansion/contraction detection
    - volume spike detection
    - volume regime classification
    - volume strength
    - fixed-bucket volume profile
    - POC
    - value area
    - high/low volume nodes
    - price location relative to value area

    This engine does not:
    - generate entries
    - calculate SL/TP
    - calculate position size
    - execute trades
    - make final trade decisions
    - call the LLM
    - fetch external data
    """

    DEFAULT_VOLUME_LOOKBACK = 20
    DEFAULT_PROFILE_LOOKBACK = 100
    DEFAULT_BUCKET_COUNT = 24

    DEFAULT_EXPANSION_THRESHOLD = 1.20
    DEFAULT_CONTRACTION_THRESHOLD = 0.80
    DEFAULT_SPIKE_THRESHOLD = 2.00

    DEFAULT_VALUE_AREA_PERCENT = 0.70

    def __init__(
        self,
        volume_lookback: int = DEFAULT_VOLUME_LOOKBACK,
        profile_lookback: int = DEFAULT_PROFILE_LOOKBACK,
        bucket_count: int = DEFAULT_BUCKET_COUNT,
        expansion_threshold: float = DEFAULT_EXPANSION_THRESHOLD,
        contraction_threshold: float = DEFAULT_CONTRACTION_THRESHOLD,
        spike_threshold: float = DEFAULT_SPIKE_THRESHOLD,
        value_area_percent: float = DEFAULT_VALUE_AREA_PERCENT,
    ) -> None:
        self.volume_lookback = self._validate_period(
            volume_lookback,
            "volume_lookback",
        )

        self.profile_lookback = self._validate_period(
            profile_lookback,
            "profile_lookback",
        )

        if isinstance(bucket_count, bool) or not isinstance(
            bucket_count,
            int,
        ):
            raise VolumeIntelligenceError(
                "bucket_count must be an integer."
            )

        if bucket_count < 2:
            raise VolumeIntelligenceError(
                "bucket_count must be at least 2."
            )

        self.bucket_count = bucket_count

        self.expansion_threshold = self._validate_ratio(
            expansion_threshold,
            "expansion_threshold",
        )

        self.contraction_threshold = self._validate_ratio(
            contraction_threshold,
            "contraction_threshold",
        )

        self.spike_threshold = self._validate_ratio(
            spike_threshold,
            "spike_threshold",
        )

        if self.contraction_threshold >= self.expansion_threshold:
            raise VolumeIntelligenceError(
                "contraction_threshold must be below "
                "expansion_threshold."
            )

        if self.spike_threshold <= self.expansion_threshold:
            raise VolumeIntelligenceError(
                "spike_threshold must be above "
                "expansion_threshold."
            )

        self.value_area_percent = self._validate_percentage(
            value_area_percent,
            "value_area_percent",
        )

    def analyze(
        self,
        bars: list[MarketBar],
    ) -> VolumeIntelligence:
        self._validate_bars(bars)

        if not bars:
            return self._unknown_result()

        current = bars[-1]

        volume_start = max(
            0,
            len(bars) - self.volume_lookback,
        )

        volume_window = [
            bar.volume
            for bar in bars[volume_start:]
        ]

        average_volume = (
            sum(volume_window)
            / len(volume_window)
        )

        current_volume = float(current.volume)

        if average_volume > 0.0:
            relative_volume = (
                current_volume
                / average_volume
            )
        else:
            relative_volume = 0.0

        expanding = (
            relative_volume
            >= self.expansion_threshold
        )

        contracting = (
            average_volume > 0.0
            and relative_volume
            <= self.contraction_threshold
        )

        spike = (
            average_volume > 0.0
            and relative_volume
            >= self.spike_threshold
        )

        volume_regime = self._classify_regime(
            relative_volume,
            expanding,
            contracting,
            spike,
        )

        volume_strength = self._calculate_strength(
            relative_volume,
        )

        profile_start = max(
            0,
            len(bars) - self.profile_lookback,
        )

        profile_bars = bars[profile_start:]

        profile = self._build_profile(
            profile_bars,
        )

        price_location = profile.price_location(
            current.close,
        )

        distance_from_poc = None

        if profile.poc is not None:
            distance_from_poc = abs(
                current.close - profile.poc
            )

        sufficient_data = (
            len(volume_window) >= 2
            and average_volume > 0.0
            and profile.sufficient_data
        )

        return VolumeIntelligence(
            timestamp=current.timestamp,
            symbol=current.symbol,
            timeframe=current.timeframe,
            current_volume=current_volume,
            average_volume=average_volume,
            relative_volume=relative_volume,
            volume_regime=volume_regime,
            volume_strength=volume_strength,
            expanding=expanding,
            contracting=contracting,
            spike=spike,
            profile=profile,
            price=float(current.close),
            price_location=price_location,
            distance_from_poc=distance_from_poc,
            sufficient_data=sufficient_data,
        )

    def analyze_xauusd(
        self,
        bars: list[MarketBar],
    ) -> VolumeIntelligence:
        self._validate_bars(bars)

        if bars and bars[0].symbol != "XAUUSD":
            raise VolumeIntelligenceError(
                "bars must use XAUUSD."
            )

        return self.analyze(bars)

    def relative_volume(
        self,
        bars: list[MarketBar],
    ) -> float | None:
        self._validate_bars(bars)

        if not bars:
            return None

        start = max(
            0,
            len(bars) - self.volume_lookback,
        )

        volumes = [
            bar.volume
            for bar in bars[start:]
        ]

        average = sum(volumes) / len(volumes)

        if average <= 0.0:
            return None

        return bars[-1].volume / average

    def volume_profile(
        self,
        bars: list[MarketBar],
    ) -> VolumeProfile:
        self._validate_bars(bars)

        if not bars:
            return self._empty_profile()

        profile_start = max(
            0,
            len(bars) - self.profile_lookback,
        )

        profile_bars = bars[profile_start:]

        return self._build_profile(
            profile_bars,
        )

    def _build_profile(
        self,
        bars: list[MarketBar],
    ) -> VolumeProfile:
        if len(bars) < 2:
            return self._empty_profile()

        total_volume = sum(
            bar.volume
            for bar in bars
        )

        if total_volume <= 0.0:
            return VolumeProfile(
                poc=None,
                value_area_high=None,
                value_area_low=None,
                high_volume_nodes=(),
                low_volume_nodes=(),
                total_volume=0.0,
                bucket_size=0.0,
                bucket_count=self.bucket_count,
                sufficient_data=False,
            )

        lowest = min(
            bar.low
            for bar in bars
        )

        highest = max(
            bar.high
            for bar in bars
        )

        price_range = highest - lowest

        if price_range <= 0.0:
            return VolumeProfile(
                poc=lowest,
                value_area_high=highest,
                value_area_low=lowest,
                high_volume_nodes=(lowest,),
                low_volume_nodes=(),
                total_volume=total_volume,
                bucket_size=0.0,
                bucket_count=1,
                sufficient_data=True,
            )

        bucket_size = (
            price_range
            / self.bucket_count
        )

        volumes = [
            0.0
            for _ in range(self.bucket_count)
        ]

        for bar in bars:
            typical_price = (
                bar.high
                + bar.low
                + bar.close
            ) / 3.0

            bucket_index = int(
                (typical_price - lowest)
                / bucket_size
            )

            bucket_index = max(
                0,
                min(
                    self.bucket_count - 1,
                    bucket_index,
                ),
            )

            volumes[bucket_index] += bar.volume

        poc_index = max(
            range(self.bucket_count),
            key=lambda index: volumes[index],
        )

        poc = self._bucket_center(
            lowest,
            bucket_size,
            poc_index,
        )

        target_volume = (
            total_volume
            * self.value_area_percent
        )

        included = {poc_index}
        accumulated = volumes[poc_index]

        while (
            accumulated < target_volume
            and len(included) < self.bucket_count
        ):
            candidates: list[tuple[float, int]] = []

            left = min(included) - 1
            right = max(included) + 1

            if left >= 0:
                candidates.append(
                    (volumes[left], left)
                )

            if right < self.bucket_count:
                candidates.append(
                    (volumes[right], right)
                )

            if not candidates:
                break

            _, selected = max(
                candidates,
                key=lambda item: item[0],
            )

            included.add(selected)
            accumulated += volumes[selected]

        value_area_low_index = min(included)
        value_area_high_index = max(included)

        value_area_low = self._bucket_lower(
            lowest,
            bucket_size,
            value_area_low_index,
        )

        value_area_high = self._bucket_upper(
            lowest,
            bucket_size,
            value_area_high_index,
        )

        positive_volumes = [
            volume
            for volume in volumes
            if volume > 0.0
        ]

        if positive_volumes:
            average_bucket_volume = (
                sum(positive_volumes)
                / len(positive_volumes)
            )
        else:
            average_bucket_volume = 0.0

        high_threshold = (
            average_bucket_volume
            * 1.50
        )

        low_threshold = (
            average_bucket_volume
            * 0.50
        )

        high_nodes = tuple(
            self._bucket_center(
                lowest,
                bucket_size,
                index,
            )
            for index, volume in enumerate(volumes)
            if volume >= high_threshold
        )

        low_nodes = tuple(
            self._bucket_center(
                lowest,
                bucket_size,
                index,
            )
            for index, volume in enumerate(volumes)
            if (
                volume > 0.0
                and volume <= low_threshold
            )
        )

        return VolumeProfile(
            poc=poc,
            value_area_high=value_area_high,
            value_area_low=value_area_low,
            high_volume_nodes=high_nodes,
            low_volume_nodes=low_nodes,
            total_volume=total_volume,
            bucket_size=bucket_size,
            bucket_count=self.bucket_count,
            sufficient_data=True,
        )

    def _classify_regime(
        self,
        relative_volume: float,
        expanding: bool,
        contracting: bool,
        spike: bool,
    ) -> VolumeRegime:
        if spike:
            return VolumeRegime.SPIKE

        if expanding:
            return VolumeRegime.EXPANDING

        if contracting:
            return VolumeRegime.CONTRACTING

        if relative_volume >= 0.0:
            return VolumeRegime.NORMAL

        return VolumeRegime.UNKNOWN

    @staticmethod
    def _calculate_strength(
        relative_volume: float,
    ) -> float:
        if relative_volume <= 0.0:
            return 0.0

        strength = min(
            100.0,
            relative_volume * 50.0,
        )

        return max(
            0.0,
            strength,
        )

    @staticmethod
    def _bucket_center(
        lowest: float,
        bucket_size: float,
        index: int,
    ) -> float:
        return (
            lowest
            + (index + 0.5)
            * bucket_size
        )

    @staticmethod
    def _bucket_lower(
        lowest: float,
        bucket_size: float,
        index: int,
    ) -> float:
        return (
            lowest
            + index * bucket_size
        )

    @staticmethod
    def _bucket_upper(
        lowest: float,
        bucket_size: float,
        index: int,
    ) -> float:
        return (
            lowest
            + (index + 1)
            * bucket_size
        )

    @staticmethod
    def _validate_bars(
        bars: list[MarketBar],
    ) -> None:
        if not isinstance(bars, list):
            raise VolumeIntelligenceError(
                "bars must be a list."
            )

        if not bars:
            return

        for bar in bars:
            if not isinstance(bar, MarketBar):
                raise VolumeIntelligenceError(
                    "all bars must be MarketBar instances."
                )

            if not isfinite(bar.volume):
                raise VolumeIntelligenceError(
                    "bar volume must be finite."
                )

            if bar.volume < 0.0:
                raise VolumeIntelligenceError(
                    "bar volume cannot be negative."
                )

        symbol = bars[0].symbol
        timeframe = bars[0].timeframe

        for bar in bars:
            if bar.symbol != symbol:
                raise VolumeIntelligenceError(
                    "all bars must use the same symbol."
                )

            if bar.timeframe != timeframe:
                raise VolumeIntelligenceError(
                    "all bars must use the same timeframe."
                )

        for previous, current in zip(
            bars,
            bars[1:],
        ):
            if current.timestamp <= previous.timestamp:
                raise VolumeIntelligenceError(
                    "bars must be strictly chronological."
                )

    @staticmethod
    def _validate_period(
        value: int,
        name: str,
    ) -> int:
        if isinstance(value, bool) or not isinstance(
            value,
            int,
        ):
            raise VolumeIntelligenceError(
                f"{name} must be an integer."
            )

        if value < 1:
            raise VolumeIntelligenceError(
                f"{name} must be at least 1."
            )

        return value

    @staticmethod
    def _validate_ratio(
        value: float,
        name: str,
    ) -> float:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise VolumeIntelligenceError(
                f"{name} must be numeric."
            )

        value = float(value)

        if not isfinite(value):
            raise VolumeIntelligenceError(
                f"{name} must be finite."
            )

        if value <= 0.0:
            raise VolumeIntelligenceError(
                f"{name} must be greater than zero."
            )

        return value

    @staticmethod
    def _validate_percentage(
        value: float,
        name: str,
    ) -> float:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise VolumeIntelligenceError(
                f"{name} must be numeric."
            )

        value = float(value)

        if not isfinite(value):
            raise VolumeIntelligenceError(
                f"{name} must be finite."
            )

        if not 0.0 < value <= 1.0:
            raise VolumeIntelligenceError(
                f"{name} must be greater than 0 and at most 1."
            )

        return value

    @staticmethod
    def _empty_profile() -> VolumeProfile:
        return VolumeProfile(
            poc=None,
            value_area_high=None,
            value_area_low=None,
            high_volume_nodes=(),
            low_volume_nodes=(),
            total_volume=0.0,
            bucket_size=0.0,
            bucket_count=0,
            sufficient_data=False,
        )

    @staticmethod
    def _unknown_result() -> VolumeIntelligence:
        profile = VolumeIntelligenceEngine._empty_profile()

        return VolumeIntelligence(
            timestamp=None,
            symbol="",
            timeframe="",
            current_volume=0.0,
            average_volume=0.0,
            relative_volume=0.0,
            volume_regime=VolumeRegime.UNKNOWN,
            volume_strength=0.0,
            expanding=False,
            contracting=False,
            spike=False,
            profile=profile,
            price=0.0,
            price_location="UNKNOWN",
            distance_from_poc=None,
            sufficient_data=False,
        )