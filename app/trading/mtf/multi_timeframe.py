from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import isfinite

from app.trading.context.market_context import (
    ContextBias,
    MarketContext,
)
from app.trading.regime.market_regime import (
    MarketRegime,
    MarketRegimeResult,
)


class MultiTimeframeAnalysisError(ValueError):
    """Raised when multi-timeframe analysis validation fails."""


class TimeframeRole(str, Enum):
    """Role of a timeframe inside multi-timeframe analysis."""

    HIGHER = "HIGHER"
    MIDDLE = "MIDDLE"
    LOWER = "LOWER"


class MTFAlignment(str, Enum):
    """Overall alignment between the supplied timeframes."""

    ALIGNED = "ALIGNED"
    PARTIALLY_ALIGNED = "PARTIALLY_ALIGNED"
    CONFLICTED = "CONFLICTED"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


class MTFDirection(str, Enum):
    """Directional interpretation of multi-timeframe conditions."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


class MTFReasonType(str, Enum):
    """Reason categories produced by the MTF engine."""

    HIGHER_TIMEFRAME_BULLISH = "HIGHER_TIMEFRAME_BULLISH"
    HIGHER_TIMEFRAME_BEARISH = "HIGHER_TIMEFRAME_BEARISH"
    HIGHER_TIMEFRAME_NEUTRAL = "HIGHER_TIMEFRAME_NEUTRAL"
    HIGHER_TIMEFRAME_UNKNOWN = "HIGHER_TIMEFRAME_UNKNOWN"

    MIDDLE_TIMEFRAME_BULLISH = "MIDDLE_TIMEFRAME_BULLISH"
    MIDDLE_TIMEFRAME_BEARISH = "MIDDLE_TIMEFRAME_BEARISH"
    MIDDLE_TIMEFRAME_NEUTRAL = "MIDDLE_TIMEFRAME_NEUTRAL"
    MIDDLE_TIMEFRAME_UNKNOWN = "MIDDLE_TIMEFRAME_UNKNOWN"

    LOWER_TIMEFRAME_BULLISH = "LOWER_TIMEFRAME_BULLISH"
    LOWER_TIMEFRAME_BEARISH = "LOWER_TIMEFRAME_BEARISH"
    LOWER_TIMEFRAME_NEUTRAL = "LOWER_TIMEFRAME_NEUTRAL"
    LOWER_TIMEFRAME_UNKNOWN = "LOWER_TIMEFRAME_UNKNOWN"

    FULL_ALIGNMENT = "FULL_ALIGNMENT"
    PARTIAL_ALIGNMENT = "PARTIAL_ALIGNMENT"
    DIRECTIONAL_CONFLICT = "DIRECTIONAL_CONFLICT"
    NEUTRAL_ALIGNMENT = "NEUTRAL_ALIGNMENT"
    UNKNOWN_ALIGNMENT = "UNKNOWN_ALIGNMENT"

    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    TRANSITION = "TRANSITION"
    UNKNOWN_REGIME = "UNKNOWN_REGIME"

    STRENGTH_SUPPORT = "STRENGTH_SUPPORT"
    STRENGTH_CONFLICT = "STRENGTH_CONFLICT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True, slots=True)
class MTFReason:
    """A single explanation for an MTF result."""

    reason_type: MTFReasonType
    message: str


@dataclass(frozen=True, slots=True)
class TimeframeAnalysis:
    """Analysis of one timeframe."""

    timeframe: str
    role: TimeframeRole
    timestamp: datetime
    bias: ContextBias
    strength: float
    regime: MarketRegime
    regime_strength: float
    sufficient_data: bool

    @property
    def is_bullish(self) -> bool:
        return self.bias is ContextBias.BULLISH

    @property
    def is_bearish(self) -> bool:
        return self.bias is ContextBias.BEARISH

    @property
    def is_neutral(self) -> bool:
        return self.bias is ContextBias.NEUTRAL

    @property
    def is_unknown(self) -> bool:
        return self.bias not in (
            ContextBias.BULLISH,
            ContextBias.BEARISH,
            ContextBias.NEUTRAL,
        )


@dataclass(frozen=True, slots=True)
class MultiTimeframeResult:
    """Complete multi-timeframe market analysis."""

    timestamp: datetime
    symbol: str

    higher: TimeframeAnalysis
    middle: TimeframeAnalysis
    lower: TimeframeAnalysis

    direction: MTFDirection
    alignment: MTFAlignment
    alignment_score: float
    strength: float

    bullish_timeframes: int
    bearish_timeframes: int
    neutral_timeframes: int
    unknown_timeframes: int

    direction_conflict: bool
    sufficient_data: bool

    reasons: tuple[MTFReason, ...]
    warnings: tuple[str, ...]

    @property
    def is_bullish(self) -> bool:
        return self.direction is MTFDirection.BULLISH

    @property
    def is_bearish(self) -> bool:
        return self.direction is MTFDirection.BEARISH

    @property
    def is_aligned(self) -> bool:
        return self.alignment is MTFAlignment.ALIGNED

    @property
    def is_conflicted(self) -> bool:
        return self.alignment is MTFAlignment.CONFLICTED

    @property
    def is_unknown(self) -> bool:
        return self.direction is MTFDirection.UNKNOWN

    @property
    def should_wait(self) -> bool:
        return self.alignment in (
            MTFAlignment.CONFLICTED,
            MTFAlignment.UNKNOWN,
            MTFAlignment.NEUTRAL,
        )


class MultiTimeframeEngine:
    """
    Deterministic multi-timeframe analysis engine.

    The engine combines three already-computed timeframe contexts:

        HIGHER → strategic direction
        MIDDLE → confirmation
        LOWER  → execution context

    It does not:
    - generate entries
    - calculate SL/TP
    - calculate position size
    - execute trades
    - fetch market data
    - fetch news
    - call an LLM
    """

    TIMEFRAME_MINUTES: dict[str, int] = {
        "M1": 1,
        "M3": 3,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H2": 120,
        "H4": 240,
        "H6": 360,
        "H8": 480,
        "H12": 720,
        "D1": 1440,
        "W1": 10080,
        "MN1": 43200,
    }

    def __init__(
        self,
        minimum_alignment_strength: float = 50.0,
        strong_alignment_strength: float = 70.0,
        partial_alignment_threshold: float = 66.67,
    ) -> None:
        self.minimum_alignment_strength = self._validate_threshold(
            minimum_alignment_strength,
            "minimum_alignment_strength",
        )

        self.strong_alignment_strength = self._validate_threshold(
            strong_alignment_strength,
            "strong_alignment_strength",
        )

        if not isfinite(float(partial_alignment_threshold)):
            raise MultiTimeframeAnalysisError(
                "partial_alignment_threshold must be finite."
            )

        if (
            float(partial_alignment_threshold) < 0.0
            or float(partial_alignment_threshold) > 100.0
        ):
            raise MultiTimeframeAnalysisError(
                "partial_alignment_threshold must be between 0 and 100."
            )

        self.partial_alignment_threshold = float(
            partial_alignment_threshold
        )

    def analyze(
        self,
        higher_context: MarketContext,
        middle_context: MarketContext,
        lower_context: MarketContext,
        higher_regime: MarketRegimeResult,
        middle_regime: MarketRegimeResult,
        lower_regime: MarketRegimeResult,
    ) -> MultiTimeframeResult:
        """Analyze higher, middle, and lower timeframe conditions."""

        self._validate_inputs(
            higher_context,
            middle_context,
            lower_context,
            higher_regime,
            middle_regime,
            lower_regime,
        )

        higher = self._build_timeframe_analysis(
            higher_context,
            higher_regime,
            TimeframeRole.HIGHER,
        )

        middle = self._build_timeframe_analysis(
            middle_context,
            middle_regime,
            TimeframeRole.MIDDLE,
        )

        lower = self._build_timeframe_analysis(
            lower_context,
            lower_regime,
            TimeframeRole.LOWER,
        )

        analyses = (higher, middle, lower)

        bullish_count = sum(
            item.bias is ContextBias.BULLISH
            for item in analyses
        )

        bearish_count = sum(
            item.bias is ContextBias.BEARISH
            for item in analyses
        )

        neutral_count = sum(
            item.bias is ContextBias.NEUTRAL
            for item in analyses
        )

        unknown_count = sum(
            item.bias not in (
                ContextBias.BULLISH,
                ContextBias.BEARISH,
                ContextBias.NEUTRAL,
            )
            for item in analyses
        )

        sufficient_data = all(
            item.sufficient_data
            for item in analyses
        )

        direction = self._determine_direction(
            higher,
            middle,
            lower,
        )

        alignment = self._determine_alignment(
            higher,
            middle,
            lower,
        )

        alignment_score = self._calculate_alignment_score(
            higher,
            middle,
            lower,
        )

        strength = self._calculate_strength(
            higher,
            middle,
            lower,
            alignment,
        )

        direction_conflict = (
            bullish_count > 0
            and bearish_count > 0
        )

        reasons = self._build_reasons(
            higher,
            middle,
            lower,
            alignment,
            direction_conflict,
            alignment_score,
        )

        warnings = self._build_warnings(
            higher,
            middle,
            lower,
            alignment,
            sufficient_data,
            direction_conflict,
        )

        return MultiTimeframeResult(
            timestamp=lower.timestamp,
            symbol=lower_context.symbol,
            higher=higher,
            middle=middle,
            lower=lower,
            direction=direction,
            alignment=alignment,
            alignment_score=alignment_score,
            strength=strength,
            bullish_timeframes=bullish_count,
            bearish_timeframes=bearish_count,
            neutral_timeframes=neutral_count,
            unknown_timeframes=unknown_count,
            direction_conflict=direction_conflict,
            sufficient_data=sufficient_data,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
        )

    def analyze_xauusd(
        self,
        higher_context: MarketContext,
        middle_context: MarketContext,
        lower_context: MarketContext,
        higher_regime: MarketRegimeResult,
        middle_regime: MarketRegimeResult,
        lower_regime: MarketRegimeResult,
    ) -> MultiTimeframeResult:
        """Analyze XAUUSD across three timeframes."""

        for context in (
            higher_context,
            middle_context,
            lower_context,
        ):
            if context.symbol != "XAUUSD":
                raise MultiTimeframeAnalysisError(
                    "all contexts must use XAUUSD."
                )

        for regime in (
            higher_regime,
            middle_regime,
            lower_regime,
        ):
            if regime.symbol != "XAUUSD":
                raise MultiTimeframeAnalysisError(
                    "all regimes must use XAUUSD."
                )

        return self.analyze(
            higher_context,
            middle_context,
            lower_context,
            higher_regime,
            middle_regime,
            lower_regime,
        )

    def _build_timeframe_analysis(
        self,
        context: MarketContext,
        regime: MarketRegimeResult,
        role: TimeframeRole,
    ) -> TimeframeAnalysis:
        return TimeframeAnalysis(
            timeframe=context.timeframe,
            role=role,
            timestamp=context.timestamp,
            bias=context.bias,
            strength=self._safe_strength(
                context.context_strength
            ),
            regime=regime.regime,
            regime_strength=self._safe_strength(
                regime.strength
            ),
            sufficient_data=bool(
                context.sufficient_history
                and regime.sufficient_history
            ),
        )

    @staticmethod
    def _determine_direction(
        higher: TimeframeAnalysis,
        middle: TimeframeAnalysis,
        lower: TimeframeAnalysis,
    ) -> MTFDirection:
        analyses = (higher, middle, lower)

        bullish = sum(
            item.bias is ContextBias.BULLISH
            for item in analyses
        )

        bearish = sum(
            item.bias is ContextBias.BEARISH
            for item in analyses
        )

        if bullish == 3:
            return MTFDirection.BULLISH

        if bearish == 3:
            return MTFDirection.BEARISH

        if bullish > bearish and bullish >= 2:
            return MTFDirection.BULLISH

        if bearish > bullish and bearish >= 2:
            return MTFDirection.BEARISH

        if bullish == bearish and bullish > 0:
            return MTFDirection.UNKNOWN

        if bullish == 0 and bearish == 0:
            return MTFDirection.NEUTRAL

        return MTFDirection.UNKNOWN

    def _determine_alignment(
        self,
        higher: TimeframeAnalysis,
        middle: TimeframeAnalysis,
        lower: TimeframeAnalysis,
    ) -> MTFAlignment:
        analyses = (higher, middle, lower)

        known = [
            item
            for item in analyses
            if item.bias in (
                ContextBias.BULLISH,
                ContextBias.BEARISH,
            )
        ]

        if not known:
            if all(
                item.bias is ContextBias.NEUTRAL
                for item in analyses
            ):
                return MTFAlignment.NEUTRAL

            return MTFAlignment.UNKNOWN

        bullish = sum(
            item.bias is ContextBias.BULLISH
            for item in known
        )

        bearish = sum(
            item.bias is ContextBias.BEARISH
            for item in known
        )

        if bullish == len(known) and len(known) == 3:
            return MTFAlignment.ALIGNED

        if bearish == len(known) and len(known) == 3:
            return MTFAlignment.ALIGNED

        if bullish > 0 and bearish > 0:
            return MTFAlignment.CONFLICTED

        directional_count = max(
            bullish,
            bearish,
        )

        if directional_count >= 2:
            return MTFAlignment.PARTIALLY_ALIGNED

        return MTFAlignment.UNKNOWN

    @staticmethod
    def _calculate_alignment_score(
        higher: TimeframeAnalysis,
        middle: TimeframeAnalysis,
        lower: TimeframeAnalysis,
    ) -> float:
        analyses = (higher, middle, lower)

        bullish = sum(
            item.bias is ContextBias.BULLISH
            for item in analyses
        )

        bearish = sum(
            item.bias is ContextBias.BEARISH
            for item in analyses
        )

        directional = bullish + bearish

        if directional == 0:
            return 0.0

        dominant = max(
            bullish,
            bearish,
        )

        return round(
            (dominant / directional) * 100.0,
            6,
        )

    def _calculate_strength(
        self,
        higher: TimeframeAnalysis,
        middle: TimeframeAnalysis,
        lower: TimeframeAnalysis,
        alignment: MTFAlignment,
    ) -> float:
        weighted_strength = (
            higher.strength * 0.50
            + middle.strength * 0.30
            + lower.strength * 0.20
        )

        if alignment is MTFAlignment.ALIGNED:
            alignment_factor = 1.0
        elif alignment is MTFAlignment.PARTIALLY_ALIGNED:
            alignment_factor = 0.80
        elif alignment is MTFAlignment.CONFLICTED:
            alignment_factor = 0.40
        elif alignment is MTFAlignment.NEUTRAL:
            alignment_factor = 0.30
        else:
            alignment_factor = 0.0

        return round(
            max(
                0.0,
                min(
                    100.0,
                    weighted_strength * alignment_factor,
                ),
            ),
            6,
        )

    @staticmethod
    def _build_reasons(
        higher: TimeframeAnalysis,
        middle: TimeframeAnalysis,
        lower: TimeframeAnalysis,
        alignment: MTFAlignment,
        direction_conflict: bool,
        alignment_score: float,
    ) -> list[MTFReason]:
        reasons: list[MTFReason] = []

        timeframe_items = (
            (
                higher,
                MTFReasonType.HIGHER_TIMEFRAME_BULLISH,
                MTFReasonType.HIGHER_TIMEFRAME_BEARISH,
                MTFReasonType.HIGHER_TIMEFRAME_NEUTRAL,
                MTFReasonType.HIGHER_TIMEFRAME_UNKNOWN,
            ),
            (
                middle,
                MTFReasonType.MIDDLE_TIMEFRAME_BULLISH,
                MTFReasonType.MIDDLE_TIMEFRAME_BEARISH,
                MTFReasonType.MIDDLE_TIMEFRAME_NEUTRAL,
                MTFReasonType.MIDDLE_TIMEFRAME_UNKNOWN,
            ),
            (
                lower,
                MTFReasonType.LOWER_TIMEFRAME_BULLISH,
                MTFReasonType.LOWER_TIMEFRAME_BEARISH,
                MTFReasonType.LOWER_TIMEFRAME_NEUTRAL,
                MTFReasonType.LOWER_TIMEFRAME_UNKNOWN,
            ),
        )

        for (
            analysis,
            bullish_reason,
            bearish_reason,
            neutral_reason,
            unknown_reason,
        ) in timeframe_items:
            if analysis.bias is ContextBias.BULLISH:
                reasons.append(
                    MTFReason(
                        bullish_reason,
                        (
                            f"{analysis.timeframe} timeframe "
                            "has bullish context."
                        ),
                    )
                )

            elif analysis.bias is ContextBias.BEARISH:
                reasons.append(
                    MTFReason(
                        bearish_reason,
                        (
                            f"{analysis.timeframe} timeframe "
                            "has bearish context."
                        ),
                    )
                )

            elif analysis.bias is ContextBias.NEUTRAL:
                reasons.append(
                    MTFReason(
                        neutral_reason,
                        (
                            f"{analysis.timeframe} timeframe "
                            "has neutral context."
                        ),
                    )
                )

            else:
                reasons.append(
                    MTFReason(
                        unknown_reason,
                        (
                            f"{analysis.timeframe} timeframe "
                            "direction is unknown."
                        ),
                    )
                )

            if analysis.regime is MarketRegime.TRENDING_UP:
                reasons.append(
                    MTFReason(
                        MTFReasonType.TRENDING_UP,
                        (
                            f"{analysis.timeframe} timeframe "
                            "regime is trending up."
                        ),
                    )
                )

            elif analysis.regime is MarketRegime.TRENDING_DOWN:
                reasons.append(
                    MTFReason(
                        MTFReasonType.TRENDING_DOWN,
                        (
                            f"{analysis.timeframe} timeframe "
                            "regime is trending down."
                        ),
                    )
                )

            elif analysis.regime is MarketRegime.RANGING:
                reasons.append(
                    MTFReason(
                        MTFReasonType.RANGING,
                        (
                            f"{analysis.timeframe} timeframe "
                            "regime is ranging."
                        ),
                    )
                )

            elif analysis.regime is MarketRegime.HIGH_VOLATILITY:
                reasons.append(
                    MTFReason(
                        MTFReasonType.HIGH_VOLATILITY,
                        (
                            f"{analysis.timeframe} timeframe "
                            "has high volatility."
                        ),
                    )
                )

            elif analysis.regime is MarketRegime.TRANSITION:
                reasons.append(
                    MTFReason(
                        MTFReasonType.TRANSITION,
                        (
                            f"{analysis.timeframe} timeframe "
                            "is in transition."
                        ),
                    )
                )

            elif analysis.regime is MarketRegime.UNKNOWN:
                reasons.append(
                    MTFReason(
                        MTFReasonType.UNKNOWN_REGIME,
                        (
                            f"{analysis.timeframe} timeframe "
                            "regime is unknown."
                        ),
                    )
                )

            if not analysis.sufficient_data:
                reasons.append(
                    MTFReason(
                        MTFReasonType.INSUFFICIENT_DATA,
                        (
                            f"{analysis.timeframe} timeframe "
                            "has insufficient data."
                        ),
                    )
                )

        if alignment is MTFAlignment.ALIGNED:
            reasons.append(
                MTFReason(
                    MTFReasonType.FULL_ALIGNMENT,
                    (
                        "All three timeframes agree on the "
                        "directional bias."
                    ),
                )
            )

        elif alignment is MTFAlignment.PARTIALLY_ALIGNED:
            reasons.append(
                MTFReason(
                    MTFReasonType.PARTIAL_ALIGNMENT,
                    (
                        "The majority of directional timeframes "
                        "agree."
                    ),
                )
            )

        elif alignment is MTFAlignment.CONFLICTED:
            reasons.append(
                MTFReason(
                    MTFReasonType.DIRECTIONAL_CONFLICT,
                    (
                        "Bullish and bearish directional "
                        "timeframes are present."
                    ),
                )
            )

        elif alignment is MTFAlignment.NEUTRAL:
            reasons.append(
                MTFReason(
                    MTFReasonType.NEUTRAL_ALIGNMENT,
                    (
                        "The supplied timeframes do not provide "
                        "a directional edge."
                    ),
                )
            )

        else:
            reasons.append(
                MTFReason(
                    MTFReasonType.UNKNOWN_ALIGNMENT,
                    (
                        "Multi-timeframe directional alignment "
                        "cannot be established."
                    ),
                )
            )

        if direction_conflict:
            reasons.append(
                MTFReason(
                    MTFReasonType.STRENGTH_CONFLICT,
                    (
                        "Directional conflict reduces the "
                        "reliability of the combined result."
                    ),
                )
            )

        elif alignment_score >= 100.0:
            reasons.append(
                MTFReason(
                    MTFReasonType.STRENGTH_SUPPORT,
                    (
                        "All directional timeframes support "
                        "the same direction."
                    ),
                )
            )

        return reasons

    @staticmethod
    def _build_warnings(
        higher: TimeframeAnalysis,
        middle: TimeframeAnalysis,
        lower: TimeframeAnalysis,
        alignment: MTFAlignment,
        sufficient_data: bool,
        direction_conflict: bool,
    ) -> list[str]:
        warnings: list[str] = []

        if not sufficient_data:
            warnings.append(
                "At least one timeframe does not contain "
                "sufficient history."
            )

        if direction_conflict:
            warnings.append(
                "Higher, middle, and lower timeframe directions "
                "are not fully aligned."
            )

        if alignment is MTFAlignment.CONFLICTED:
            warnings.append(
                "Multi-timeframe conflict is active."
            )

        if alignment in (
            MTFAlignment.UNKNOWN,
            MTFAlignment.NEUTRAL,
        ):
            warnings.append(
                "Multi-timeframe analysis does not provide "
                "a clear directional bias."
            )

        for analysis in (
            higher,
            middle,
            lower,
        ):
            if analysis.regime in (
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.TRANSITION,
            ):
                warnings.append(
                    f"{analysis.timeframe} timeframe has a "
                    f"{analysis.regime.value.lower()} regime."
                )

        return warnings

    @staticmethod
    def _validate_inputs(
        higher_context: MarketContext,
        middle_context: MarketContext,
        lower_context: MarketContext,
        higher_regime: MarketRegimeResult,
        middle_regime: MarketRegimeResult,
        lower_regime: MarketRegimeResult,
    ) -> None:
        contexts = (
            higher_context,
            middle_context,
            lower_context,
        )

        regimes = (
            higher_regime,
            middle_regime,
            lower_regime,
        )

        for context in contexts:
            if not isinstance(context, MarketContext):
                raise MultiTimeframeAnalysisError(
                    "all contexts must be MarketContext instances."
                )

        for regime in regimes:
            if not isinstance(regime, MarketRegimeResult):
                raise MultiTimeframeAnalysisError(
                    "all regimes must be MarketRegimeResult instances."
                )

        symbols = {
            context.symbol
            for context in contexts
        }

        if len(symbols) != 1:
            raise MultiTimeframeAnalysisError(
                "all contexts must use the same symbol."
            )

        regime_symbols = {
            regime.symbol
            for regime in regimes
        }

        if len(regime_symbols) != 1:
            raise MultiTimeframeAnalysisError(
                "all regimes must use the same symbol."
            )

        if symbols != regime_symbols:
            raise MultiTimeframeAnalysisError(
                "context and regime symbols must match."
            )

        for context, regime in zip(
            contexts,
            regimes,
        ):
            if context.timeframe != regime.timeframe:
                raise MultiTimeframeAnalysisError(
                    "each context timeframe must match its regime timeframe."
                )

            if context.timestamp != regime.timestamp:
                raise MultiTimeframeAnalysisError(
                    "each context timestamp must match its regime timestamp."
                )

        timeframe_values = [
            context.timeframe
            for context in contexts
        ]

        if len(set(timeframe_values)) != 3:
            raise MultiTimeframeAnalysisError(
                "higher, middle, and lower timeframes must be distinct."
            )

        higher_minutes = MultiTimeframeEngine.TIMEFRAME_MINUTES.get(
            higher_context.timeframe
        )

        middle_minutes = MultiTimeframeEngine.TIMEFRAME_MINUTES.get(
            middle_context.timeframe
        )

        lower_minutes = MultiTimeframeEngine.TIMEFRAME_MINUTES.get(
            lower_context.timeframe
        )

        if (
            higher_minutes is None
            or middle_minutes is None
            or lower_minutes is None
        ):
            raise MultiTimeframeAnalysisError(
                "unsupported timeframe."
            )

        if not (
            higher_minutes
            > middle_minutes
            > lower_minutes
        ):
            raise MultiTimeframeAnalysisError(
                "timeframes must satisfy higher > middle > lower."
            )

    @staticmethod
    def _safe_strength(value: float) -> float:
        if isinstance(value, bool):
            return 0.0

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0

        if not isfinite(numeric):
            return 0.0

        return max(
            0.0,
            min(
                100.0,
                numeric,
            ),
        )

    @staticmethod
    def _validate_threshold(
        value: float,
        name: str,
    ) -> float:
        if isinstance(value, bool):
            raise MultiTimeframeAnalysisError(
                f"{name} must be numeric."
            )

        if not isinstance(value, (int, float)):
            raise MultiTimeframeAnalysisError(
                f"{name} must be numeric."
            )

        value = float(value)

        if not isfinite(value):
            raise MultiTimeframeAnalysisError(
                f"{name} must be finite."
            )

        if value < 0.0 or value > 100.0:
            raise MultiTimeframeAnalysisError(
                f"{name} must be between 0 and 100."
            )

        return value