from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

from app.trading.context.market_context import (
    ContextBias,
    ContextSignalType,
    MarketCondition,
    MarketContext,
)
from app.trading.data.market_bar import MarketBar


class SetupDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


class SetupType(str, Enum):
    TREND_CONTINUATION = "TREND_CONTINUATION"
    REVERSAL = "REVERSAL"
    RANGE = "RANGE"
    NONE = "NONE"


class SetupReasonType(str, Enum):
    STRUCTURE_ALIGNMENT = "STRUCTURE_ALIGNMENT"
    MOMENTUM_ALIGNMENT = "MOMENTUM_ALIGNMENT"
    PRICE_ALIGNMENT = "PRICE_ALIGNMENT"
    VOLATILITY_ALIGNMENT = "VOLATILITY_ALIGNMENT"
    TREND_ALIGNMENT = "TREND_ALIGNMENT"
    CONFLICT = "CONFLICT"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    NEUTRAL_CONTEXT = "NEUTRAL_CONTEXT"
    NO_CLEAR_SETUP = "NO_CLEAR_SETUP"


@dataclass(frozen=True, slots=True)
class SetupReason:
    reason_type: SetupReasonType
    message: str


@dataclass(frozen=True, slots=True)
class SetupEvaluation:
    timestamp: object
    symbol: str
    timeframe: str
    close: float

    direction: SetupDirection
    setup_type: SetupType

    valid: bool
    quality_score: float

    context_bias: ContextBias
    context_strength: float
    market_condition: MarketCondition

    supporting_signals: tuple[ContextSignalType, ...]
    conflicting_signals: tuple[ContextSignalType, ...]

    reasons: tuple[SetupReason, ...]
    warnings: tuple[SetupReason, ...]


class SetupEngine:
    """
    Detects and evaluates recognizable trading setups from
    deterministic MarketContext data.

    This engine does not:
        - calculate entry prices
        - calculate stop loss
        - calculate take profit
        - calculate position size
        - calculate monetary risk
        - execute trades

    Its responsibility is only to determine whether the current
    market context represents a sufficiently clear setup.

    The engine is deterministic and does not use an LLM.

    Setup evaluation is deliberately separated from trade outcome.
    A valid setup can lose, and an invalid setup can occasionally win.
    """

    DEFAULT_MINIMUM_SETUP_SCORE = 60.0
    DEFAULT_STRONG_SETUP_SCORE = 80.0

    DEFAULT_MIN_SUPPORTING_SIGNALS = 2
    DEFAULT_MAX_CONFLICTS = 1

    def __init__(
        self,
        minimum_setup_score: float = DEFAULT_MINIMUM_SETUP_SCORE,
        strong_setup_score: float = DEFAULT_STRONG_SETUP_SCORE,
        minimum_supporting_signals: int = (
            DEFAULT_MIN_SUPPORTING_SIGNALS
        ),
        max_conflicts: int = DEFAULT_MAX_CONFLICTS,
    ) -> None:
        self._validate_percentage(
            minimum_setup_score,
            "minimum_setup_score",
        )

        self._validate_percentage(
            strong_setup_score,
            "strong_setup_score",
        )

        if strong_setup_score < minimum_setup_score:
            raise ValueError(
                "strong_setup_score must be greater than or equal "
                "to minimum_setup_score."
            )

        self._validate_positive_integer(
            minimum_supporting_signals,
            "minimum_supporting_signals",
        )

        self._validate_non_negative_integer(
            max_conflicts,
            "max_conflicts",
        )

        self.minimum_setup_score = float(
            minimum_setup_score
        )

        self.strong_setup_score = float(
            strong_setup_score
        )

        self.minimum_supporting_signals = (
            minimum_supporting_signals
        )

        self.max_conflicts = max_conflicts

    # =========================================================
    # VALIDATION
    # =========================================================

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
            or numeric_value < 0.0
            or numeric_value > 100.0
        ):
            raise ValueError(
                f"{name} must be between 0 and 100."
            )

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
    def _validate_non_negative_integer(
        value: int,
        name: str,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(
                f"{name} must be a non-negative integer."
            )

    @staticmethod
    def _validate_context(
        context: MarketContext,
    ) -> None:
        if not isinstance(
            context,
            MarketContext,
        ):
            raise ValueError(
                "context must be a MarketContext."
            )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def evaluate(
        self,
        context: MarketContext,
    ) -> SetupEvaluation:
        """
        Evaluate the supplied market context.

        Returns a deterministic setup assessment.
        """
        self._validate_context(context)

        supporting_signals = (
            self._supporting_signals(context)
        )

        conflicting_signals = (
            context.conflicts
        )

        reasons: list[SetupReason] = []
        warnings: list[SetupReason] = []

        if not context.sufficient_history:
            warnings.append(
                SetupReason(
                    reason_type=(
                        SetupReasonType.INSUFFICIENT_HISTORY
                    ),
                    message=(
                        "Insufficient historical data for a "
                        "high-confidence setup."
                    ),
                )
            )

        if context.bias == ContextBias.NEUTRAL:
            warnings.append(
                SetupReason(
                    reason_type=(
                        SetupReasonType.NEUTRAL_CONTEXT
                    ),
                    message=(
                        "Market context is neutral."
                    ),
                )
            )

        if conflicting_signals:
            warnings.append(
                SetupReason(
                    reason_type=SetupReasonType.CONFLICT,
                    message=(
                        "Directional signals are conflicting."
                    ),
                )
            )

        direction = self._determine_direction(
            context
        )

        setup_type = self._determine_setup_type(
            context,
            direction,
        )

        reasons.extend(
            self._build_reasons(
                context=context,
                direction=direction,
                supporting_signals=supporting_signals,
            )
        )

        quality_score = self._calculate_quality_score(
            context=context,
            direction=direction,
            supporting_signals=supporting_signals,
            conflicting_signals=conflicting_signals,
        )

        valid = self._is_valid_setup(
            context=context,
            direction=direction,
            setup_type=setup_type,
            quality_score=quality_score,
            supporting_signals=supporting_signals,
            conflicting_signals=conflicting_signals,
        )

        if not valid:
            warnings.append(
                SetupReason(
                    reason_type=SetupReasonType.NO_CLEAR_SETUP,
                    message=(
                        "Current conditions do not meet the "
                        "minimum setup requirements."
                    ),
                )
            )

        return SetupEvaluation(
            timestamp=context.timestamp,
            symbol=context.symbol,
            timeframe=context.timeframe,
            close=float(context.close),
            direction=direction,
            setup_type=setup_type,
            valid=valid,
            quality_score=quality_score,
            context_bias=context.bias,
            context_strength=float(
                context.context_strength
            ),
            market_condition=context.condition,
            supporting_signals=(
                tuple(supporting_signals)
            ),
            conflicting_signals=(
                tuple(conflicting_signals)
            ),
            reasons=tuple(reasons),
            warnings=tuple(warnings),
        )

    def evaluate_at(
        self,
        contexts: Sequence[MarketContext],
        index: int,
    ) -> SetupEvaluation:
        """
        Evaluate exactly one historical MarketContext.

        No future context is inspected.

        This helper is useful when the backtesting engine later
        evaluates historical setups one candle at a time.
        """
        if not isinstance(
            contexts,
            Sequence,
        ):
            raise ValueError(
                "contexts must be a sequence of MarketContext objects."
            )

        if isinstance(contexts, (str, bytes)):
            raise ValueError(
                "contexts must be a sequence of MarketContext objects."
            )

        if not contexts:
            raise ValueError(
                "contexts cannot be empty."
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

        if index >= len(contexts):
            raise ValueError(
                "index is outside the available contexts."
            )

        for item_index, context in enumerate(contexts):
            if not isinstance(
                context,
                MarketContext,
            ):
                raise ValueError(
                    "contexts["
                    f"{item_index}"
                    "] must be a MarketContext."
                )

        return self.evaluate(
            contexts[index]
        )

    # =========================================================
    # SIGNAL ANALYSIS
    # =========================================================

    @staticmethod
    def _supporting_signals(
        context: MarketContext,
    ) -> list[ContextSignalType]:
        """
        Return directional signals aligned with the context bias.

        Neutral signals are ignored.
        """
        if context.bias == ContextBias.NEUTRAL:
            return []

        supporting: list[ContextSignalType] = []

        for signal in context.signals:
            if (
                signal.bias == context.bias
                and signal.strength > 0.0
            ):
                supporting.append(
                    signal.signal_type
                )

        return supporting

    @staticmethod
    def _determine_direction(
        context: MarketContext,
    ) -> SetupDirection:
        if context.bias == ContextBias.BULLISH:
            return SetupDirection.LONG

        if context.bias == ContextBias.BEARISH:
            return SetupDirection.SHORT

        return SetupDirection.NONE

    @staticmethod
    def _determine_setup_type(
        context: MarketContext,
        direction: SetupDirection,
    ) -> SetupType:
        if direction == SetupDirection.NONE:
            return SetupType.NONE

        if context.condition == MarketCondition.RANGING:
            return SetupType.RANGE

        if (
            context.condition
            == MarketCondition.TRENDING_UP
            and direction == SetupDirection.LONG
        ):
            return SetupType.TREND_CONTINUATION

        if (
            context.condition
            == MarketCondition.TRENDING_DOWN
            and direction == SetupDirection.SHORT
        ):
            return SetupType.TREND_CONTINUATION

        if context.condition == MarketCondition.TRANSITION:
            return SetupType.REVERSAL

        if (
            context.trend.name == "BULLISH"
            and direction == SetupDirection.SHORT
        ):
            return SetupType.REVERSAL

        if (
            context.trend.name == "BEARISH"
            and direction == SetupDirection.LONG
        ):
            return SetupType.REVERSAL

        return SetupType.REVERSAL

    # =========================================================
    # REASON BUILDING
    # =========================================================

    def _build_reasons(
        self,
        context: MarketContext,
        direction: SetupDirection,
        supporting_signals: Sequence[
            ContextSignalType
        ],
    ) -> list[SetupReason]:
        reasons: list[SetupReason] = []

        if direction == SetupDirection.NONE:
            return reasons

        signal_set = set(
            supporting_signals
        )

        if ContextSignalType.STRUCTURE in signal_set:
            reasons.append(
                SetupReason(
                    reason_type=(
                        SetupReasonType.STRUCTURE_ALIGNMENT
                    ),
                    message=(
                        "Market structure aligns with "
                        f"{direction.value.lower()} direction."
                    ),
                )
            )

        if (
            ContextSignalType.RSI in signal_set
            or ContextSignalType.MACD in signal_set
        ):
            reasons.append(
                SetupReason(
                    reason_type=(
                        SetupReasonType.MOMENTUM_ALIGNMENT
                    ),
                    message=(
                        "Momentum signals support the "
                        f"{direction.value.lower()} direction."
                    ),
                )
            )

        if ContextSignalType.PRICE_LOCATION in signal_set:
            reasons.append(
                SetupReason(
                    reason_type=(
                        SetupReasonType.PRICE_ALIGNMENT
                    ),
                    message=(
                        "Price location supports the "
                        f"{direction.value.lower()} context."
                    ),
                )
            )

        if ContextSignalType.VOLATILITY in signal_set:
            reasons.append(
                SetupReason(
                    reason_type=(
                        SetupReasonType.VOLATILITY_ALIGNMENT
                    ),
                    message=(
                        "Volatility conditions are available "
                        "for setup evaluation."
                    ),
                )
            )

        if (
            context.condition
            in (
                MarketCondition.TRENDING_UP,
                MarketCondition.TRENDING_DOWN,
            )
        ):
            if (
                (
                    context.condition
                    == MarketCondition.TRENDING_UP
                    and direction
                    == SetupDirection.LONG
                )
                or (
                    context.condition
                    == MarketCondition.TRENDING_DOWN
                    and direction
                    == SetupDirection.SHORT
                )
            ):
                reasons.append(
                    SetupReason(
                        reason_type=(
                            SetupReasonType.TREND_ALIGNMENT
                        ),
                        message=(
                            "Market condition aligns with "
                            "the setup direction."
                        ),
                    )
                )

        return reasons

    # =========================================================
    # QUALITY SCORING
    # =========================================================

    def _calculate_quality_score(
        self,
        context: MarketContext,
        direction: SetupDirection,
        supporting_signals: Sequence[
            ContextSignalType
        ],
        conflicting_signals: Sequence[
            ContextSignalType
        ],
    ) -> float:
        """
        Calculate a transparent 0-100 setup quality score.

        Components:

            Context strength       0-30
            Trend alignment        0-25
            Structure alignment    0-20
            Momentum alignment     0-15
            Price alignment        0-10

        Conflict penalties can reduce the score.

        The score is intentionally independent of future trade
        outcome.
        """
        if direction == SetupDirection.NONE:
            return 0.0

        score = 0.0

        # -----------------------------------------------------
        # CONTEXT STRENGTH: 0-30
        # -----------------------------------------------------

        context_strength = max(
            0.0,
            min(
                100.0,
                float(context.context_strength),
            ),
        )

        score += (
            context_strength * 0.30
        )

        # -----------------------------------------------------
        # TREND ALIGNMENT: 0-25
        # -----------------------------------------------------

        trend_alignment = (
            self._trend_alignment_score(
                context,
                direction,
            )
        )

        score += trend_alignment

        # -----------------------------------------------------
        # STRUCTURE: 0-20
        # -----------------------------------------------------

        if ContextSignalType.STRUCTURE in set(
            supporting_signals
        ):
            structure_signal = self._find_signal(
                context.signals,
                ContextSignalType.STRUCTURE,
            )

            if structure_signal is not None:
                score += (
                    max(
                        0.0,
                        min(
                            100.0,
                            structure_signal.strength,
                        ),
                    )
                    * 0.20
                )

        # -----------------------------------------------------
        # MOMENTUM: 0-15
        # -----------------------------------------------------

        momentum_strengths: list[float] = []

        for signal_type in (
            ContextSignalType.RSI,
            ContextSignalType.MACD,
        ):
            signal = self._find_signal(
                context.signals,
                signal_type,
            )

            if (
                signal is not None
                and signal.bias == context.bias
            ):
                momentum_strengths.append(
                    max(
                        0.0,
                        min(
                            100.0,
                            signal.strength,
                        ),
                    )
                )

        if momentum_strengths:
            momentum_score = (
                sum(momentum_strengths)
                / len(momentum_strengths)
            )

            score += (
                momentum_score * 0.15
            )

        # -----------------------------------------------------
        # PRICE LOCATION: 0-10
        # -----------------------------------------------------

        price_signal = self._find_signal(
            context.signals,
            ContextSignalType.PRICE_LOCATION,
        )

        if (
            price_signal is not None
            and price_signal.bias == context.bias
        ):
            score += (
                max(
                    0.0,
                    min(
                        100.0,
                        price_signal.strength,
                    ),
                )
                * 0.10
            )

        # -----------------------------------------------------
        # CONFLICT PENALTY
        # -----------------------------------------------------

        conflict_count = len(
            conflicting_signals
        )

        if conflict_count > 0:
            score -= min(
                30.0,
                conflict_count * 15.0,
            )

        # -----------------------------------------------------
        # INSUFFICIENT HISTORY PENALTY
        # -----------------------------------------------------

        if not context.sufficient_history:
            score -= 20.0

        return max(
            0.0,
            min(100.0, score),
        )

    @staticmethod
    def _trend_alignment_score(
        context: MarketContext,
        direction: SetupDirection,
    ) -> float:
        if direction == SetupDirection.NONE:
            return 0.0

        trend_strength = max(
            0.0,
            min(
                100.0,
                float(context.trend_strength),
            ),
        )

        if (
            context.condition
            == MarketCondition.TRENDING_UP
            and direction == SetupDirection.LONG
        ):
            return trend_strength * 0.25

        if (
            context.condition
            == MarketCondition.TRENDING_DOWN
            and direction == SetupDirection.SHORT
        ):
            return trend_strength * 0.25

        if context.condition == MarketCondition.TRANSITION:
            return trend_strength * 0.10

        if context.condition == MarketCondition.RANGING:
            return 5.0

        return 0.0

    @staticmethod
    def _find_signal(
        signals: Sequence[
            object
        ],
        signal_type: ContextSignalType,
    ):
        for signal in signals:
            if (
                signal.signal_type
                == signal_type
            ):
                return signal

        return None

    # =========================================================
    # SETUP VALIDATION
    # =========================================================

    def _is_valid_setup(
        self,
        context: MarketContext,
        direction: SetupDirection,
        setup_type: SetupType,
        quality_score: float,
        supporting_signals: Sequence[
            ContextSignalType
        ],
        conflicting_signals: Sequence[
            ContextSignalType
        ],
    ) -> bool:
        if not context.sufficient_history:
            return False

        if direction == SetupDirection.NONE:
            return False

        if setup_type == SetupType.NONE:
            return False

        if quality_score < self.minimum_setup_score:
            return False

        if (
            len(supporting_signals)
            < self.minimum_supporting_signals
        ):
            return False

        if (
            len(conflicting_signals)
            > self.max_conflicts
        ):
            return False

        if (
            context.context_strength
            < self.minimum_setup_score
        ):
            return False

        if (
            setup_type
            == SetupType.TREND_CONTINUATION
            and not self._trend_direction_matches(
                context,
                direction,
            )
        ):
            return False

        return True

    @staticmethod
    def _trend_direction_matches(
        context: MarketContext,
        direction: SetupDirection,
    ) -> bool:
        if (
            direction == SetupDirection.LONG
            and context.condition
            == MarketCondition.TRENDING_UP
        ):
            return True

        if (
            direction == SetupDirection.SHORT
            and context.condition
            == MarketCondition.TRENDING_DOWN
        ):
            return True

        return False


__all__ = [
    "SetupDirection",
    "SetupEngine",
    "SetupEvaluation",
    "SetupReason",
    "SetupReasonType",
    "SetupType",
]