from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from app.trading.mtf.multi_timeframe import (
    MTFAlignment,
    MTFDirection,
    MultiTimeframeResult,
)
from app.trading.support_resistance.support_resistance import (
    SRLevelType,
    SRZone,
)
from app.trading.trendline.trendline_engine import (
    Trendline,
    TrendlineStatus,
    TrendlineType,
)
from app.trading.volume.volume_intelligence import (
    VolumeIntelligence,
    VolumeRegime,
)


class SetupConfluenceError(ValueError):
    """Raised when setup confluence validation fails."""


class ConfluenceDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


class ConfluenceQuality(str, Enum):
    STRONG = "STRONG"
    GOOD = "GOOD"
    MIXED = "MIXED"
    WEAK = "WEAK"
    CONFLICTED = "CONFLICTED"
    UNKNOWN = "UNKNOWN"


class ConfluenceReasonType(str, Enum):
    MTF_ALIGNMENT = "MTF_ALIGNMENT"
    MTF_DIRECTION = "MTF_DIRECTION"
    SUPPORT_CONFIRMATION = "SUPPORT_CONFIRMATION"
    RESISTANCE_CONFIRMATION = "RESISTANCE_CONFIRMATION"
    TRENDLINE_SUPPORT = "TRENDLINE_SUPPORT"
    TRENDLINE_RESISTANCE = "TRENDLINE_RESISTANCE"
    VOLUME_EXPANSION = "VOLUME_EXPANSION"
    VOLUME_CONTRACTION = "VOLUME_CONTRACTION"
    VOLUME_SPIKE = "VOLUME_SPIKE"
    VOLUME_PROFILE_SUPPORT = "VOLUME_PROFILE_SUPPORT"
    VOLUME_PROFILE_RESISTANCE = "VOLUME_PROFILE_RESISTANCE"
    DIRECTIONAL_CONFLICT = "DIRECTIONAL_CONFLICT"
    WEAK_CONFLUENCE = "WEAK_CONFLUENCE"
    STRONG_CONFLUENCE = "STRONG_CONFLUENCE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NO_CLEAR_CONFLUENCE = "NO_CLEAR_CONFLUENCE"


@dataclass(frozen=True, slots=True)
class ConfluenceReason:
    reason_type: ConfluenceReasonType
    message: str


@dataclass(frozen=True, slots=True)
class SetupConfluenceResult:
    timestamp: object
    symbol: str
    timeframe: str

    direction: ConfluenceDirection
    quality: ConfluenceQuality
    score: float

    mtf_score: float
    support_resistance_score: float
    trendline_score: float
    volume_score: float

    bullish_factors: int
    bearish_factors: int
    neutral_factors: int
    conflicting_factors: int

    support_present: bool
    resistance_present: bool
    support_strong: bool
    resistance_strong: bool

    support_trendline_present: bool
    resistance_trendline_present: bool
    volume_confirmed: bool

    sufficient_data: bool
    reasons: tuple[ConfluenceReason, ...]
    warnings: tuple[str, ...]

    @property
    def is_bullish(self) -> bool:
        return self.direction is ConfluenceDirection.BULLISH

    @property
    def is_bearish(self) -> bool:
        return self.direction is ConfluenceDirection.BEARISH

    @property
    def is_neutral(self) -> bool:
        return self.direction is ConfluenceDirection.NEUTRAL

    @property
    def is_unknown(self) -> bool:
        return self.direction is ConfluenceDirection.UNKNOWN

    @property
    def is_strong(self) -> bool:
        return self.quality is ConfluenceQuality.STRONG

    @property
    def is_good(self) -> bool:
        return self.quality is ConfluenceQuality.GOOD

    @property
    def is_conflicted(self) -> bool:
        return self.quality is ConfluenceQuality.CONFLICTED

    @property
    def has_confluence(self) -> bool:
        return self.direction not in (
            ConfluenceDirection.UNKNOWN,
            ConfluenceDirection.NEUTRAL,
        ) and self.score >= 50.0


class SetupConfluenceEngine:
    """
    Deterministic confluence engine.

    Combines:
    - multi-timeframe alignment
    - support/resistance
    - trendlines
    - volume intelligence

    It does not generate entries, SL/TP, position size, execution,
    news, or LLM decisions.
    """

    DEFAULT_MTF_WEIGHT = 0.35
    DEFAULT_SR_WEIGHT = 0.25
    DEFAULT_TRENDLINE_WEIGHT = 0.20
    DEFAULT_VOLUME_WEIGHT = 0.20

    DEFAULT_MIN_SCORE = 50.0
    DEFAULT_GOOD_SCORE = 65.0
    DEFAULT_STRONG_SCORE = 80.0

    def __init__(
        self,
        mtf_weight: float = DEFAULT_MTF_WEIGHT,
        support_resistance_weight: float = DEFAULT_SR_WEIGHT,
        trendline_weight: float = DEFAULT_TRENDLINE_WEIGHT,
        volume_weight: float = DEFAULT_VOLUME_WEIGHT,
        minimum_score: float = DEFAULT_MIN_SCORE,
        good_score: float = DEFAULT_GOOD_SCORE,
        strong_score: float = DEFAULT_STRONG_SCORE,
    ) -> None:
        self.mtf_weight = self._validate_weight(
            mtf_weight, "mtf_weight"
        )
        self.support_resistance_weight = self._validate_weight(
            support_resistance_weight,
            "support_resistance_weight",
        )
        self.trendline_weight = self._validate_weight(
            trendline_weight, "trendline_weight"
        )
        self.volume_weight = self._validate_weight(
            volume_weight, "volume_weight"
        )

        total = (
            self.mtf_weight
            + self.support_resistance_weight
            + self.trendline_weight
            + self.volume_weight
        )

        if abs(total - 1.0) > 1e-9:
            raise SetupConfluenceError(
                "confluence weights must sum to 1.0."
            )

        self.minimum_score = self._validate_score(
            minimum_score, "minimum_score"
        )
        self.good_score = self._validate_score(
            good_score, "good_score"
        )
        self.strong_score = self._validate_score(
            strong_score, "strong_score"
        )

        if not (
            self.minimum_score
            <= self.good_score
            <= self.strong_score
        ):
            raise SetupConfluenceError(
                "scores must satisfy minimum <= good <= strong."
            )

    def analyze(
        self,
        mtf: MultiTimeframeResult,
        support_resistance: tuple[SRZone, ...],
        trendlines: tuple[Trendline, ...],
        volume: VolumeIntelligence,
    ) -> SetupConfluenceResult:
        self._validate_inputs(
            mtf,
            support_resistance,
            trendlines,
            volume,
        )

        timestamp = mtf.timestamp
        symbol = mtf.symbol
        timeframe = mtf.lower.timeframe

        mtf_score, mtf_bull, mtf_bear = self._mtf_component(mtf)

        sr_score, sr_bull, sr_bear, sr_data = (
            self._sr_component(
                support_resistance,
                mtf.direction,
            )
        )

        trend_score, trend_bull, trend_bear, trend_data = (
            self._trendline_component(
                trendlines,
                mtf.direction,
            )
        )

        volume_score, volume_bull, volume_bear, volume_data = (
            self._volume_component(
                volume,
                mtf.direction,
            )
        )

        weighted_score = (
            mtf_score * self.mtf_weight
            + sr_score * self.support_resistance_weight
            + trend_score * self.trendline_weight
            + volume_score * self.volume_weight
        )

        bullish = (
            mtf_bull + sr_bull + trend_bull + volume_bull
        )
        bearish = (
            mtf_bear + sr_bear + trend_bear + volume_bear
        )

        neutral = max(
            0,
            4 - bullish - bearish,
        )

        conflicting = min(
            bullish,
            bearish,
        )

        direction = self._direction(
            bullish,
            bearish,
            mtf.direction,
        )

        quality = self._quality(
            weighted_score,
            conflicting,
            direction,
        )

        sufficient_data = (
            mtf.sufficient_data
            and volume.sufficient_data
            and sr_data
            and trend_data
        )

        reasons = self._build_reasons(
            mtf,
            support_resistance,
            trendlines,
            volume,
            direction,
            quality,
            weighted_score,
        )

        warnings = self._build_warnings(
            mtf,
            volume,
            conflicting,
            sufficient_data,
        )

        return SetupConfluenceResult(
            timestamp=timestamp,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            quality=quality,
            score=round(weighted_score, 6),
            mtf_score=round(mtf_score, 6),
            support_resistance_score=round(sr_score, 6),
            trendline_score=round(trend_score, 6),
            volume_score=round(volume_score, 6),
            bullish_factors=bullish,
            bearish_factors=bearish,
            neutral_factors=neutral,
            conflicting_factors=conflicting,
            support_present=sr_data[0],
            resistance_present=sr_data[1],
            support_strong=sr_data[2],
            resistance_strong=sr_data[3],
            support_trendline_present=trend_data[0],
            resistance_trendline_present=trend_data[1],
            volume_confirmed=volume_data,
            sufficient_data=sufficient_data,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
        )

    def analyze_xauusd(
        self,
        mtf: MultiTimeframeResult,
        support_resistance: tuple[SRZone, ...],
        trendlines: tuple[Trendline, ...],
        volume: VolumeIntelligence,
    ) -> SetupConfluenceResult:
        if mtf.symbol != "XAUUSD":
            raise SetupConfluenceError(
                "mtf must use XAUUSD."
            )

        if volume.symbol != "XAUUSD":
            raise SetupConfluenceError(
                "volume must use XAUUSD."
            )

        return self.analyze(
            mtf,
            support_resistance,
            trendlines,
            volume,
        )

    def _mtf_component(
        self,
        mtf: MultiTimeframeResult,
    ) -> tuple[float, int, int]:
        if not mtf.sufficient_data:
            return 0.0, 0, 0

        score = mtf.alignment_score

        if mtf.direction is MTFDirection.BULLISH:
            return score, 1, 0

        if mtf.direction is MTFDirection.BEARISH:
            return score, 0, 1

        return score, 0, 0

    def _sr_component(
        self,
        zones: tuple[SRZone, ...],
        direction: MTFDirection,
    ) -> tuple[float, int, int, tuple[bool, bool, bool, bool]]:
        active = [
            zone
            for zone in zones
            if zone.is_active
        ]

        supports = [
            zone
            for zone in active
            if zone.zone_type is SRLevelType.SUPPORT
        ]

        resistances = [
            zone
            for zone in active
            if zone.zone_type is SRLevelType.RESISTANCE
        ]

        support_strong = any(
            zone.is_strong for zone in supports
        )
        resistance_strong = any(
            zone.is_strong for zone in resistances
        )

        if direction is MTFDirection.BULLISH:
            score = 0.0
            if supports:
                score += 50.0
            if support_strong:
                score += 35.0
            if resistances and not resistance_strong:
                score += 15.0
            return (
                min(100.0, score),
                1 if supports else 0,
                0,
                (
                    bool(supports),
                    bool(resistances),
                    support_strong,
                    resistance_strong,
                ),
            )

        if direction is MTFDirection.BEARISH:
            score = 0.0
            if resistances:
                score += 50.0
            if resistance_strong:
                score += 35.0
            if supports and not support_strong:
                score += 15.0
            return (
                min(100.0, score),
                0,
                1 if resistances else 0,
                (
                    bool(supports),
                    bool(resistances),
                    support_strong,
                    resistance_strong,
                ),
            )

        return (
            0.0,
            0,
            0,
            (
                bool(supports),
                bool(resistances),
                support_strong,
                resistance_strong,
            ),
        )

    def _trendline_component(
        self,
        trendlines: tuple[Trendline, ...],
        direction: MTFDirection,
    ) -> tuple[float, int, int, tuple[bool, bool]]:
        active = [
            trendline
            for trendline in trendlines
            if trendline.status
            in (
                TrendlineStatus.ACTIVE,
                TrendlineStatus.TESTED,
            )
        ]

        supports = [
            line
            for line in active
            if line.trendline_type is TrendlineType.SUPPORT
        ]

        resistances = [
            line
            for line in active
            if line.trendline_type is TrendlineType.RESISTANCE
        ]

        if direction is MTFDirection.BULLISH:
            score = 0.0
            if supports:
                score += 60.0
            if any(line.is_strong for line in supports):
                score += 40.0
            return (
                min(100.0, score),
                1 if supports else 0,
                0,
                (bool(supports), bool(resistances)),
            )

        if direction is MTFDirection.BEARISH:
            score = 0.0
            if resistances:
                score += 60.0
            if any(line.is_strong for line in resistances):
                score += 40.0
            return (
                min(100.0, score),
                0,
                1 if resistances else 0,
                (bool(supports), bool(resistances)),
            )

        return (
            0.0,
            0,
            0,
            (bool(supports), bool(resistances)),
        )

    def _volume_component(
        self,
        volume: VolumeIntelligence,
        direction: MTFDirection,
    ) -> tuple[float, int, int, bool]:
        if not volume.sufficient_data:
            return 0.0, 0, 0, False

        score = 0.0

        if volume.is_spike:
            score += 35.0
        elif volume.is_expanding:
            score += 30.0
        elif volume.is_normal:
            score += 10.0
        elif volume.is_contracting:
            score += 0.0

        if direction is MTFDirection.BULLISH:
            if volume.price_location == "BELOW_VALUE":
                score += 25.0
            elif volume.price_location == "INSIDE_VALUE":
                score += 15.0

            return (
                min(100.0, score),
                1 if score >= 30.0 else 0,
                0,
                score >= 30.0,
            )

        if direction is MTFDirection.BEARISH:
            if volume.price_location == "ABOVE_VALUE":
                score += 25.0
            elif volume.price_location == "INSIDE_VALUE":
                score += 15.0

            return (
                min(100.0, score),
                0,
                1 if score >= 30.0 else 0,
                score >= 30.0,
            )

        return min(100.0, score), 0, 0, False

    def _direction(
        self,
        bullish: int,
        bearish: int,
        mtf_direction: MTFDirection,
    ) -> ConfluenceDirection:
        if bullish > bearish:
            return ConfluenceDirection.BULLISH

        if bearish > bullish:
            return ConfluenceDirection.BEARISH

        if mtf_direction is MTFDirection.BULLISH:
            return ConfluenceDirection.BULLISH

        if mtf_direction is MTFDirection.BEARISH:
            return ConfluenceDirection.BEARISH

        if mtf_direction is MTFDirection.NEUTRAL:
            return ConfluenceDirection.NEUTRAL

        return ConfluenceDirection.UNKNOWN

    def _quality(
        self,
        score: float,
        conflicting: int,
        direction: ConfluenceDirection,
    ) -> ConfluenceQuality:
        if direction is ConfluenceDirection.UNKNOWN:
            return ConfluenceQuality.UNKNOWN

        if conflicting >= 2:
            return ConfluenceQuality.CONFLICTED

        if score >= self.strong_score:
            return ConfluenceQuality.STRONG

        if score >= self.good_score:
            return ConfluenceQuality.GOOD

        if score >= self.minimum_score:
            return ConfluenceQuality.MIXED

        return ConfluenceQuality.WEAK

    def _build_reasons(
        self,
        mtf: MultiTimeframeResult,
        zones: tuple[SRZone, ...],
        trendlines: tuple[Trendline, ...],
        volume: VolumeIntelligence,
        direction: ConfluenceDirection,
        quality: ConfluenceQuality,
        score: float,
    ) -> list[ConfluenceReason]:
        reasons: list[ConfluenceReason] = []

        if mtf.alignment is MTFAlignment.ALIGNED:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.MTF_ALIGNMENT,
                    "Multi-timeframe structure is aligned.",
                )
            )

        if mtf.direction is MTFDirection.BULLISH:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.MTF_DIRECTION,
                    "Multi-timeframe direction is bullish.",
                )
            )
        elif mtf.direction is MTFDirection.BEARISH:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.MTF_DIRECTION,
                    "Multi-timeframe direction is bearish.",
                )
            )

        if any(
            z.is_active
            and z.zone_type is SRLevelType.SUPPORT
            for z in zones
        ):
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.SUPPORT_CONFIRMATION,
                    "Active support is available.",
                )
            )

        if any(
            z.is_active
            and z.zone_type is SRLevelType.RESISTANCE
            for z in zones
        ):
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.RESISTANCE_CONFIRMATION,
                    "Active resistance is available.",
                )
            )

        if any(
            t.is_active
            and t.trendline_type is TrendlineType.SUPPORT
            for t in trendlines
        ):
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.TRENDLINE_SUPPORT,
                    "Active support trendline is available.",
                )
            )

        if any(
            t.is_active
            and t.trendline_type is TrendlineType.RESISTANCE
            for t in trendlines
        ):
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.TRENDLINE_RESISTANCE,
                    "Active resistance trendline is available.",
                )
            )

        if volume.is_spike:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.VOLUME_SPIKE,
                    "Volume is showing a spike.",
                )
            )
        elif volume.is_expanding:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.VOLUME_EXPANSION,
                    "Volume is expanding.",
                )
            )
        elif volume.is_contracting:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.VOLUME_CONTRACTION,
                    "Volume is contracting.",
                )
            )

        if volume.price_location == "BELOW_VALUE":
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.VOLUME_PROFILE_SUPPORT,
                    "Price is below the volume value area.",
                )
            )
        elif volume.price_location == "ABOVE_VALUE":
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.VOLUME_PROFILE_RESISTANCE,
                    "Price is above the volume value area.",
                )
            )

        if quality in (
            ConfluenceQuality.STRONG,
            ConfluenceQuality.GOOD,
        ):
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.STRONG_CONFLUENCE,
                    f"Confluence score is {score:.2f}.",
                )
            )
        elif quality in (
            ConfluenceQuality.WEAK,
            ConfluenceQuality.UNKNOWN,
        ):
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.WEAK_CONFLUENCE,
                    f"Confluence score is {score:.2f}.",
                )
            )

        if direction in (
            ConfluenceDirection.NEUTRAL,
            ConfluenceDirection.UNKNOWN,
        ):
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.NO_CLEAR_CONFLUENCE,
                    "No clear directional confluence exists.",
                )
            )

        return reasons

    @staticmethod
    def _build_warnings(
        mtf: MultiTimeframeResult,
        volume: VolumeIntelligence,
        conflicting: int,
        sufficient_data: bool,
    ) -> list[str]:
        warnings: list[str] = []

        if not sufficient_data:
            warnings.append(
                "One or more confluence inputs have insufficient data."
            )

        if mtf.is_conflicted:
            warnings.append(
                "Multi-timeframe analysis is conflicted."
            )

        if conflicting:
            warnings.append(
                "Bullish and bearish confluence factors are both present."
            )

        if not volume.sufficient_data:
            warnings.append(
                "Volume intelligence is insufficient."
            )

        return warnings

    @staticmethod
    def _validate_inputs(
        mtf: MultiTimeframeResult,
        zones: tuple[SRZone, ...],
        trendlines: tuple[Trendline, ...],
        volume: VolumeIntelligence,
    ) -> None:
        if not isinstance(mtf, MultiTimeframeResult):
            raise SetupConfluenceError(
                "mtf must be a MultiTimeframeResult."
            )

        if not isinstance(zones, tuple):
            raise SetupConfluenceError(
                "support_resistance must be a tuple."
            )

        if not isinstance(trendlines, tuple):
            raise SetupConfluenceError(
                "trendlines must be a tuple."
            )

        if not isinstance(volume, VolumeIntelligence):
            raise SetupConfluenceError(
                "volume must be a VolumeIntelligence."
            )

        for zone in zones:
            if not isinstance(zone, SRZone):
                raise SetupConfluenceError(
                    "all support/resistance items must be SRZone."
                )

        for trendline in trendlines:
            if not isinstance(trendline, Trendline):
                raise SetupConfluenceError(
                    "all trendline items must be Trendline."
                )

        if volume.symbol != mtf.symbol:
            raise SetupConfluenceError(
                "volume symbol must match mtf symbol."
            )

        if volume.timeframe != mtf.lower.timeframe:
            raise SetupConfluenceError(
                "volume timeframe must match the MTF lower timeframe."
            )

    @staticmethod
    def _validate_weight(
        value: float,
        name: str,
    ) -> float:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise SetupConfluenceError(
                f"{name} must be numeric."
            )

        value = float(value)

        if not isfinite(value):
            raise SetupConfluenceError(
                f"{name} must be finite."
            )

        if value < 0.0:
            raise SetupConfluenceError(
                f"{name} cannot be negative."
            )

        return value

    @staticmethod
    def _validate_score(
        value: float,
        name: str,
    ) -> float:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise SetupConfluenceError(
                f"{name} must be numeric."
            )

        value = float(value)

        if not isfinite(value):
            raise SetupConfluenceError(
                f"{name} must be finite."
            )

        if value < 0.0 or value > 100.0:
            raise SetupConfluenceError(
                f"{name} must be between 0 and 100."
            )

        return value
