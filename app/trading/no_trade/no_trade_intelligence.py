from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import isfinite

from app.trading.candidate.trade_candidate import (
    CandidateDecision,
    SetupDirection,
    TradeCandidate,
)
from app.trading.decision.environment_trade_decision import (
    EnvironmentTradeDecision,
    EnvironmentTradeDecisionResult,
)
from app.trading.environment.market_environment import (
    EnvironmentDirection,
    EnvironmentQuality,
    MarketEnvironment,
)
from app.trading.regime.market_regime import MarketRegime


class NoTradeIntelligenceError(ValueError):
    """Raised when no-trade intelligence validation fails."""


class NoTradeDecision(str, Enum):
    """Final no-trade assessment decision."""

    NO_TRADE = "NO_TRADE"
    CLEAR = "CLEAR"


class NoTradeReasonType(str, Enum):
    """Reason categories produced by the no-trade engine."""

    CANDIDATE_REJECTED = "CANDIDATE_REJECTED"
    CANDIDATE_WAITING = "CANDIDATE_WAITING"
    CANDIDATE_INVALID = "CANDIDATE_INVALID"

    LOW_SETUP_QUALITY = "LOW_SETUP_QUALITY"
    INSUFFICIENT_CONFIRMATION = "INSUFFICIENT_CONFIRMATION"
    NO_CLEAR_DIRECTION = "NO_CLEAR_DIRECTION"

    ENVIRONMENT_UNKNOWN = "ENVIRONMENT_UNKNOWN"
    ENVIRONMENT_NEUTRAL = "ENVIRONMENT_NEUTRAL"
    ENVIRONMENT_CONFLICTED = "ENVIRONMENT_CONFLICTED"
    ENVIRONMENT_CAUTION = "ENVIRONMENT_CAUTION"
    ENVIRONMENT_MIXED = "ENVIRONMENT_MIXED"

    DIRECTION_MISMATCH = "DIRECTION_MISMATCH"
    WEAK_ENVIRONMENT = "WEAK_ENVIRONMENT"

    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    TRANSITION_REGIME = "TRANSITION_REGIME"
    UNKNOWN_REGIME = "UNKNOWN_REGIME"

    TECHNICAL_CONFLICT = "TECHNICAL_CONFLICT"
    NEWS_CONFLICT = "NEWS_CONFLICT"

    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

    TRADE_DECISION_REJECTED = "TRADE_DECISION_REJECTED"
    TRADE_DECISION_WAITING = "TRADE_DECISION_WAITING"

    NO_TRADE_REQUIRED = "NO_TRADE_REQUIRED"
    NO_TRADE_CONDITIONS_CLEARED = "NO_TRADE_CONDITIONS_CLEARED"


@dataclass(frozen=True, slots=True)
class NoTradeReason:
    """A single reason explaining a no-trade assessment."""

    reason_type: NoTradeReasonType
    message: str


@dataclass(frozen=True, slots=True)
class NoTradeAssessment:
    """Complete deterministic no-trade assessment."""

    timestamp: datetime
    symbol: str
    timeframe: str

    candidate_decision: CandidateDecision
    candidate_direction: SetupDirection
    candidate_quality_score: float

    environment_direction: EnvironmentDirection
    environment_strength: float
    environment_quality: EnvironmentQuality

    environment_conflict: bool
    caution_required: bool
    sufficient_environment_data: bool

    trade_decision: EnvironmentTradeDecision | None
    no_trade_decision: NoTradeDecision

    no_trade_score: float
    no_trade_required: bool

    weak_candidate: bool
    weak_environment: bool
    directional_conflict: bool
    environment_conflict_present: bool
    caution_present: bool
    insufficient_data: bool

    reasons: tuple[NoTradeReason, ...]
    warnings: tuple[str, ...]

    @property
    def is_no_trade(self) -> bool:
        return self.no_trade_decision is NoTradeDecision.NO_TRADE

    @property
    def is_clear(self) -> bool:
        return self.no_trade_decision is NoTradeDecision.CLEAR

    @property
    def should_not_trade(self) -> bool:
        return self.no_trade_required


class NoTradeIntelligenceEngine:
    """
    Deterministic no-trade intelligence layer.

    The engine evaluates an already-generated trade candidate against
    the unified market environment and identifies conditions where
    trading should be avoided.

    This engine does not:
    - generate entries
    - calculate stop loss
    - calculate take profit
    - calculate position size
    - execute trades
    - fetch news
    - call an LLM
    """

    DEFAULT_MINIMUM_SETUP_QUALITY = 60.0
    DEFAULT_MINIMUM_ENVIRONMENT_STRENGTH = 50.0
    DEFAULT_NO_TRADE_SCORE = 50.0

    def __init__(
        self,
        minimum_setup_quality: float = DEFAULT_MINIMUM_SETUP_QUALITY,
        minimum_environment_strength: float = (
            DEFAULT_MINIMUM_ENVIRONMENT_STRENGTH
        ),
        no_trade_score_threshold: float = DEFAULT_NO_TRADE_SCORE,
    ) -> None:
        self.minimum_setup_quality = self._validate_threshold(
            minimum_setup_quality,
            "minimum_setup_quality",
        )

        self.minimum_environment_strength = self._validate_threshold(
            minimum_environment_strength,
            "minimum_environment_strength",
        )

        self.no_trade_score_threshold = self._validate_threshold(
            no_trade_score_threshold,
            "no_trade_score_threshold",
        )

    def assess(
        self,
        candidate: TradeCandidate,
        environment: MarketEnvironment,
        trade_decision: EnvironmentTradeDecisionResult | None = None,
    ) -> NoTradeAssessment:
        """Assess whether current conditions require no trade."""

        self._validate_inputs(
            candidate,
            environment,
            trade_decision,
        )

        if trade_decision is not None:
            if (
                trade_decision.symbol != candidate.symbol
                or trade_decision.timeframe != candidate.timeframe
                or trade_decision.timestamp != candidate.timestamp
            ):
                raise NoTradeIntelligenceError(
                    "trade_decision must match candidate timestamp, "
                    "symbol, and timeframe."
                )

        reasons: list[NoTradeReason] = []
        warnings: list[str] = []

        weak_candidate = self._is_weak_candidate(candidate)
        weak_environment = self._is_weak_environment(environment)

        directional_conflict = self._has_directional_conflict(
            candidate,
            environment,
        )

        environment_conflict_present = bool(
            environment.environment_conflict
        )

        caution_present = bool(
            environment.caution_required
        )

        insufficient_data = not bool(
            environment.sufficient_data
        )

        self._collect_candidate_reasons(
            candidate,
            weak_candidate,
            reasons,
        )

        self._collect_environment_reasons(
            environment,
            weak_environment,
            reasons,
        )

        if directional_conflict:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.DIRECTION_MISMATCH,
                    "Candidate direction conflicts with the "
                    "environment direction.",
                )
            )

        if environment.technical_conflict:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.TECHNICAL_CONFLICT,
                    "Technical conditions contain a conflict.",
                )
            )

        if environment.news_conflict:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.NEWS_CONFLICT,
                    "News conditions contain a conflict.",
                )
            )

        if environment_conflict_present:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.ENVIRONMENT_CONFLICTED,
                    "The unified environment is conflicted.",
                )
            )

        if caution_present:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.ENVIRONMENT_CAUTION,
                    "The unified environment requires caution.",
                )
            )

        if insufficient_data:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.INSUFFICIENT_DATA,
                    "The unified environment does not contain "
                    "sufficient data.",
                )
            )

        self._collect_regime_reasons(
            environment,
            reasons,
        )

        if trade_decision is not None:
            self._collect_trade_decision_reasons(
                trade_decision,
                reasons,
            )

        no_trade_score = self._calculate_no_trade_score(
            candidate=candidate,
            environment=environment,
            trade_decision=trade_decision,
            weak_candidate=weak_candidate,
            weak_environment=weak_environment,
            directional_conflict=directional_conflict,
            environment_conflict=environment_conflict_present,
            caution_present=caution_present,
            insufficient_data=insufficient_data,
        )

        no_trade_required = self._requires_no_trade(
            candidate=candidate,
            environment=environment,
            trade_decision=trade_decision,
            weak_candidate=weak_candidate,
            weak_environment=weak_environment,
            directional_conflict=directional_conflict,
            environment_conflict=environment_conflict_present,
            caution_present=caution_present,
            insufficient_data=insufficient_data,
            no_trade_score=no_trade_score,
        )

        if no_trade_required:
            decision = NoTradeDecision.NO_TRADE

            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.NO_TRADE_REQUIRED,
                    "One or more independent no-trade conditions "
                    "are active.",
                )
            )
        else:
            decision = NoTradeDecision.CLEAR

            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.NO_TRADE_CONDITIONS_CLEARED,
                    "No blocking no-trade condition is active.",
                )
            )

        if weak_candidate:
            warnings.append(
                "Setup quality is below the configured minimum."
            )

        if weak_environment:
            warnings.append(
                "Environment strength is below the configured minimum."
            )

        if directional_conflict:
            warnings.append(
                "Candidate and environment directions conflict."
            )

        if environment_conflict_present:
            warnings.append(
                "Environment conflict is active."
            )

        if caution_present:
            warnings.append(
                "Environment caution is active."
            )

        if insufficient_data:
            warnings.append(
                "Environment data is insufficient."
            )

        return NoTradeAssessment(
            timestamp=candidate.timestamp,
            symbol=candidate.symbol,
            timeframe=candidate.timeframe,
            candidate_decision=candidate.decision,
            candidate_direction=candidate.direction,
            candidate_quality_score=float(
                candidate.setup_quality_score
            ),
            environment_direction=environment.overall_direction,
            environment_strength=float(
                environment.overall_strength
            ),
            environment_quality=environment.environment_quality,
            environment_conflict=environment_conflict_present,
            caution_required=caution_present,
            sufficient_environment_data=bool(
                environment.sufficient_data
            ),
            trade_decision=(
                trade_decision.decision
                if trade_decision is not None
                else None
            ),
            no_trade_decision=decision,
            no_trade_score=no_trade_score,
            no_trade_required=no_trade_required,
            weak_candidate=weak_candidate,
            weak_environment=weak_environment,
            directional_conflict=directional_conflict,
            environment_conflict_present=environment_conflict_present,
            caution_present=caution_present,
            insufficient_data=insufficient_data,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
        )

    def assess_xauusd(
        self,
        candidate: TradeCandidate,
        environment: MarketEnvironment,
        trade_decision: EnvironmentTradeDecisionResult | None = None,
    ) -> NoTradeAssessment:
        """Run no-trade assessment specifically for XAUUSD."""

        self._validate_inputs(
            candidate,
            environment,
            trade_decision,
        )

        if candidate.symbol != "XAUUSD":
            raise NoTradeIntelligenceError(
                "candidate symbol must be XAUUSD."
            )

        if environment.symbol != "XAUUSD":
            raise NoTradeIntelligenceError(
                "environment symbol must be XAUUSD."
            )

        return self.assess(
            candidate,
            environment,
            trade_decision,
        )

    def _requires_no_trade(
        self,
        *,
        candidate: TradeCandidate,
        environment: MarketEnvironment,
        trade_decision: EnvironmentTradeDecisionResult | None,
        weak_candidate: bool,
        weak_environment: bool,
        directional_conflict: bool,
        environment_conflict: bool,
        caution_present: bool,
        insufficient_data: bool,
        no_trade_score: float,
    ) -> bool:
        """
        Determine whether the current conditions require NO TRADE.

        Explicit hard-block conditions always take priority over the
        aggregate score.
        """

        # Candidate-level hard blocks.
        if candidate.decision is CandidateDecision.REJECT:
            return True

        if candidate.invalidated:
            return True

        if not candidate.entry_ready:
            return True

        if candidate.direction is SetupDirection.NONE:
            return True

        # Required confirmations.
        if not candidate.structure_confirmed:
            return True

        if not candidate.momentum_confirmed:
            return True

        if not candidate.price_confirmed:
            return True

        if not candidate.trend_confirmed:
            return True

        # Candidate quality.
        if weak_candidate:
            return True

        # Data sufficiency.
        if insufficient_data:
            return True

        # Environment direction.
        if environment.overall_direction is EnvironmentDirection.UNKNOWN:
            return True

        if environment.overall_direction is EnvironmentDirection.NEUTRAL:
            return True

        # Environment strength.
        if weak_environment:
            return True

        # Directional conflict.
        if directional_conflict:
            return True

        # Technical conflict.
        if environment.technical_conflict:
            return True

        # News conflict.
        if environment.news_conflict:
            return True

        # Unified environment conflict.
        if environment_conflict:
            return True

        # Explicit caution.
        if caution_present:
            return True

        # Environment quality.
        if environment.environment_quality in (
            EnvironmentQuality.CONFLICTED,
            EnvironmentQuality.CAUTION,
            EnvironmentQuality.UNKNOWN,
        ):
            return True

        # Dangerous/uncertain market regimes.
        if environment.market_regime is MarketRegime.HIGH_VOLATILITY:
            return True

        if environment.market_regime is MarketRegime.TRANSITION:
            return True

        if environment.market_regime is MarketRegime.UNKNOWN:
            return True

        # Existing environment-aware decision.
        if trade_decision is not None:
            if trade_decision.decision in (
                EnvironmentTradeDecision.REJECT,
                EnvironmentTradeDecision.WAIT,
            ):
                return True

        # Aggregate score as the final additional block.
        return no_trade_score >= self.no_trade_score_threshold

    def _calculate_no_trade_score(
        self,
        *,
        candidate: TradeCandidate,
        environment: MarketEnvironment,
        trade_decision: EnvironmentTradeDecisionResult | None,
        weak_candidate: bool,
        weak_environment: bool,
        directional_conflict: bool,
        environment_conflict: bool,
        caution_present: bool,
        insufficient_data: bool,
    ) -> float:
        """Calculate bounded aggregate no-trade risk score."""

        score = 0.0

        if candidate.decision is CandidateDecision.REJECT:
            score += 100.0
        elif candidate.decision is CandidateDecision.WAIT:
            score += 60.0

        if candidate.invalidated:
            score += 100.0

        if not candidate.entry_ready:
            score += 60.0

        if candidate.direction is SetupDirection.NONE:
            score += 80.0

        if not candidate.structure_confirmed:
            score += 25.0

        if not candidate.momentum_confirmed:
            score += 25.0

        if not candidate.price_confirmed:
            score += 25.0

        if not candidate.trend_confirmed:
            score += 25.0

        if weak_candidate:
            quality_gap = (
                self.minimum_setup_quality
                - float(candidate.setup_quality_score)
            )

            score += max(
                10.0,
                min(40.0, quality_gap),
            )

        if insufficient_data:
            score += 50.0

        if environment.overall_direction in (
            EnvironmentDirection.UNKNOWN,
            EnvironmentDirection.NEUTRAL,
        ):
            score += 45.0

        if weak_environment:
            strength_gap = (
                self.minimum_environment_strength
                - float(environment.overall_strength)
            )

            score += max(
                10.0,
                min(35.0, strength_gap),
            )

        if directional_conflict:
            score += 45.0

        if environment.technical_conflict:
            score += 35.0

        if environment.news_conflict:
            score += 35.0

        if environment_conflict:
            score += 45.0

        if caution_present:
            score += 35.0

        if environment.environment_quality is EnvironmentQuality.MIXED:
            score += 10.0

        if environment.environment_quality in (
            EnvironmentQuality.CONFLICTED,
            EnvironmentQuality.CAUTION,
            EnvironmentQuality.UNKNOWN,
        ):
            score += 35.0

        if environment.market_regime is MarketRegime.HIGH_VOLATILITY:
            score += 30.0

        elif environment.market_regime is MarketRegime.TRANSITION:
            score += 20.0

        elif environment.market_regime is MarketRegime.UNKNOWN:
            score += 30.0

        if trade_decision is not None:
            if trade_decision.decision is EnvironmentTradeDecision.REJECT:
                score += 50.0

            elif trade_decision.decision is EnvironmentTradeDecision.WAIT:
                score += 25.0

        return max(
            0.0,
            min(100.0, score),
        )

    def _is_weak_candidate(
        self,
        candidate: TradeCandidate,
    ) -> bool:
        return (
            float(candidate.setup_quality_score)
            < self.minimum_setup_quality
        )

    def _is_weak_environment(
        self,
        environment: MarketEnvironment,
    ) -> bool:
        return (
            float(environment.overall_strength)
            < self.minimum_environment_strength
        )

    @staticmethod
    def _has_directional_conflict(
        candidate: TradeCandidate,
        environment: MarketEnvironment,
    ) -> bool:
        if candidate.direction is SetupDirection.LONG:
            return (
                environment.overall_direction
                is EnvironmentDirection.BEARISH
            )

        if candidate.direction is SetupDirection.SHORT:
            return (
                environment.overall_direction
                is EnvironmentDirection.BULLISH
            )

        return False

    @staticmethod
    def _collect_candidate_reasons(
        candidate: TradeCandidate,
        weak_candidate: bool,
        reasons: list[NoTradeReason],
    ) -> None:
        if candidate.decision is CandidateDecision.REJECT:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.CANDIDATE_REJECTED,
                    "The candidate was rejected.",
                )
            )

        elif candidate.decision is CandidateDecision.WAIT:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.CANDIDATE_WAITING,
                    "The candidate is still waiting.",
                )
            )

        if candidate.invalidated:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.CANDIDATE_INVALID,
                    "The candidate has been invalidated.",
                )
            )

        if not candidate.entry_ready:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.CANDIDATE_INVALID,
                    "The candidate is not entry-ready.",
                )
            )

        if candidate.direction is SetupDirection.NONE:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.NO_CLEAR_DIRECTION,
                    "The candidate has no clear direction.",
                )
            )

        if weak_candidate:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.LOW_SETUP_QUALITY,
                    "Setup quality is below the configured minimum.",
                )
            )

        if not candidate.structure_confirmed:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.INSUFFICIENT_CONFIRMATION,
                    "Structure confirmation is missing.",
                )
            )

        if not candidate.momentum_confirmed:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.INSUFFICIENT_CONFIRMATION,
                    "Momentum confirmation is missing.",
                )
            )

        if not candidate.price_confirmed:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.INSUFFICIENT_CONFIRMATION,
                    "Price confirmation is missing.",
                )
            )

        if not candidate.trend_confirmed:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.INSUFFICIENT_CONFIRMATION,
                    "Trend confirmation is missing.",
                )
            )

    @staticmethod
    def _collect_environment_reasons(
        environment: MarketEnvironment,
        weak_environment: bool,
        reasons: list[NoTradeReason],
    ) -> None:
        if environment.overall_direction is EnvironmentDirection.UNKNOWN:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.ENVIRONMENT_UNKNOWN,
                    "Environment direction is unknown.",
                )
            )

        elif environment.overall_direction is EnvironmentDirection.NEUTRAL:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.ENVIRONMENT_NEUTRAL,
                    "Environment direction is neutral.",
                )
            )

        if weak_environment:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.WEAK_ENVIRONMENT,
                    "Environment strength is below the "
                    "configured minimum.",
                )
            )

        if environment.environment_quality is EnvironmentQuality.MIXED:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.ENVIRONMENT_MIXED,
                    "The environment contains mixed conditions.",
                )
            )

    @staticmethod
    def _collect_regime_reasons(
        environment: MarketEnvironment,
        reasons: list[NoTradeReason],
    ) -> None:
        regime = environment.market_regime

        if regime is MarketRegime.HIGH_VOLATILITY:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.HIGH_VOLATILITY,
                    "The market is currently classified as "
                    "high volatility.",
                )
            )

        elif regime is MarketRegime.TRANSITION:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.TRANSITION_REGIME,
                    "The market is currently transitioning "
                    "between conditions.",
                )
            )

        elif regime is MarketRegime.UNKNOWN:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.UNKNOWN_REGIME,
                    "The market regime is unknown.",
                )
            )

    @staticmethod
    def _collect_trade_decision_reasons(
        trade_decision: EnvironmentTradeDecisionResult,
        reasons: list[NoTradeReason],
    ) -> None:
        if trade_decision.decision is EnvironmentTradeDecision.REJECT:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.TRADE_DECISION_REJECTED,
                    "The environment-aware trade decision "
                    "rejected the candidate.",
                )
            )

        elif trade_decision.decision is EnvironmentTradeDecision.WAIT:
            reasons.append(
                NoTradeReason(
                    NoTradeReasonType.TRADE_DECISION_WAITING,
                    "The environment-aware trade decision "
                    "requires waiting.",
                )
            )

    @staticmethod
    def _validate_inputs(
        candidate: TradeCandidate,
        environment: MarketEnvironment,
        trade_decision: EnvironmentTradeDecisionResult | None,
    ) -> None:
        if not isinstance(candidate, TradeCandidate):
            raise NoTradeIntelligenceError(
                "candidate must be a TradeCandidate."
            )

        if not isinstance(environment, MarketEnvironment):
            raise NoTradeIntelligenceError(
                "environment must be a MarketEnvironment."
            )

        if trade_decision is not None and not isinstance(
            trade_decision,
            EnvironmentTradeDecisionResult,
        ):
            raise NoTradeIntelligenceError(
                "trade_decision must be an "
                "EnvironmentTradeDecisionResult or None."
            )

        if candidate.symbol != environment.symbol:
            raise NoTradeIntelligenceError(
                "candidate and environment symbols must match."
            )

        if candidate.timeframe != environment.timeframe:
            raise NoTradeIntelligenceError(
                "candidate and environment timeframes must match."
            )

        if candidate.timestamp != environment.timestamp:
            raise NoTradeIntelligenceError(
                "candidate and environment timestamps must match."
            )

    @staticmethod
    def _validate_threshold(
        value: float,
        name: str,
    ) -> float:
        if isinstance(value, bool):
            raise NoTradeIntelligenceError(
                f"{name} must be numeric."
            )

        if not isinstance(value, (int, float)):
            raise NoTradeIntelligenceError(
                f"{name} must be numeric."
            )

        value = float(value)

        if not isfinite(value):
            raise NoTradeIntelligenceError(
                f"{name} must be finite."
            )

        if value < 0.0 or value > 100.0:
            raise NoTradeIntelligenceError(
                f"{name} must be between 0 and 100."
            )

        return value