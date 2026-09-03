from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.trading.candidate.trade_candidate import (
    CandidateDecision,
    SetupDirection,
    TradeCandidate,
)
from app.trading.environment.market_environment import (
    EnvironmentDirection,
    EnvironmentQuality,
    MarketEnvironment,
)


class EnvironmentTradeDecisionError(ValueError):
    """Raised when environment-aware trade decision fails validation."""


class EnvironmentTradeDecision(str, Enum):
    TRADE = "TRADE"
    WAIT = "WAIT"
    REJECT = "REJECT"


class EnvironmentDecisionReasonType(str, Enum):
    CANDIDATE_TRADE_READY = "CANDIDATE_TRADE_READY"
    CANDIDATE_WAITING = "CANDIDATE_WAITING"
    CANDIDATE_REJECTED = "CANDIDATE_REJECTED"
    CANDIDATE_INVALID = "CANDIDATE_INVALID"

    LONG_ENVIRONMENT_ALIGNMENT = "LONG_ENVIRONMENT_ALIGNMENT"
    SHORT_ENVIRONMENT_ALIGNMENT = "SHORT_ENVIRONMENT_ALIGNMENT"

    DIRECTION_CONFLICT = "DIRECTION_CONFLICT"
    ENVIRONMENT_NEUTRAL = "ENVIRONMENT_NEUTRAL"
    ENVIRONMENT_UNKNOWN = "ENVIRONMENT_UNKNOWN"

    ENVIRONMENT_CLEAR = "ENVIRONMENT_CLEAR"
    ENVIRONMENT_FAVORABLE = "ENVIRONMENT_FAVORABLE"
    ENVIRONMENT_MIXED = "ENVIRONMENT_MIXED"
    ENVIRONMENT_CONFLICTED = "ENVIRONMENT_CONFLICTED"
    ENVIRONMENT_CAUTION = "ENVIRONMENT_CAUTION"

    INSUFFICIENT_ENVIRONMENT_STRENGTH = (
        "INSUFFICIENT_ENVIRONMENT_STRENGTH"
    )
    ENVIRONMENT_CAUTION_REQUIRED = "ENVIRONMENT_CAUTION_REQUIRED"

    DECISION_TRADE = "DECISION_TRADE"
    DECISION_WAIT = "DECISION_WAIT"
    DECISION_REJECT = "DECISION_REJECT"


@dataclass(frozen=True, slots=True)
class EnvironmentDecisionReason:
    reason_type: EnvironmentDecisionReasonType
    message: str


@dataclass(frozen=True, slots=True)
class EnvironmentTradeDecisionResult:
    timestamp: object
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

    decision: EnvironmentTradeDecision

    direction_aligned: bool
    environment_supports_trade: bool
    blocked_by_environment: bool

    reasons: tuple[EnvironmentDecisionReason, ...]
    warnings: tuple[str, ...]

    @property
    def is_trade(self) -> bool:
        return self.decision is EnvironmentTradeDecision.TRADE

    @property
    def is_wait(self) -> bool:
        return self.decision is EnvironmentTradeDecision.WAIT

    @property
    def is_rejected(self) -> bool:
        return self.decision is EnvironmentTradeDecision.REJECT

    @property
    def can_proceed(self) -> bool:
        return self.decision is EnvironmentTradeDecision.TRADE


class EnvironmentTradeDecisionEngine:
    """
    Combines a TradeCandidate with the unified MarketEnvironment.

    This engine decides whether a candidate should TRADE, WAIT, or REJECT.

    It does not:
    - generate entries
    - calculate stop loss
    - calculate take profit
    - calculate position size
    - execute trades
    - fetch news
    - call an LLM
    """

    DEFAULT_MINIMUM_ENVIRONMENT_STRENGTH = 50.0

    def __init__(
        self,
        minimum_environment_strength: float = (
            DEFAULT_MINIMUM_ENVIRONMENT_STRENGTH
        ),
    ) -> None:
        if isinstance(minimum_environment_strength, bool):
            raise EnvironmentTradeDecisionError(
                "minimum_environment_strength must be numeric."
            )

        if not isinstance(minimum_environment_strength, (int, float)):
            raise EnvironmentTradeDecisionError(
                "minimum_environment_strength must be numeric."
            )

        minimum_environment_strength = float(
            minimum_environment_strength
        )

        if not 0.0 <= minimum_environment_strength <= 100.0:
            raise EnvironmentTradeDecisionError(
                "minimum_environment_strength must be between 0 and 100."
            )

        self.minimum_environment_strength = (
            minimum_environment_strength
        )

    def decide(
        self,
        candidate: TradeCandidate,
        environment: MarketEnvironment,
    ) -> EnvironmentTradeDecisionResult:
        self._validate_inputs(candidate, environment)

        reasons: list[EnvironmentDecisionReason] = []
        warnings: list[str] = []

        candidate_decision = candidate.decision
        candidate_direction = candidate.direction

        environment_direction = environment.overall_direction
        environment_strength = self._clamp(
            environment.overall_strength
        )
        environment_quality = environment.environment_quality

        environment_conflict = environment.environment_conflict
        caution_required = environment.caution_required
        sufficient_environment_data = environment.sufficient_data

        direction_aligned = self._direction_aligned(
            candidate_direction,
            environment_direction,
        )

        environment_supports_trade = self._environment_supports_trade(
            candidate_direction,
            environment,
            direction_aligned,
        )

        blocked_by_environment = self._blocked_by_environment(
            environment,
            direction_aligned,
            environment_supports_trade,
        )

        self._add_candidate_reasons(
            candidate,
            reasons,
        )

        self._add_environment_reasons(
            candidate_direction,
            environment,
            direction_aligned,
            reasons,
            warnings,
        )

        if candidate_decision is CandidateDecision.REJECT:
            decision = EnvironmentTradeDecision.REJECT

            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.DECISION_REJECT,
                    "The trade candidate was already rejected "
                    "before environment evaluation.",
                )
            )

            return self._build_result(
                candidate,
                environment,
                decision,
                direction_aligned,
                environment_supports_trade,
                blocked_by_environment,
                reasons,
                warnings,
            )

        if candidate_decision is CandidateDecision.WAIT:
            decision = EnvironmentTradeDecision.WAIT

            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.DECISION_WAIT,
                    "The trade candidate is still waiting for "
                    "sufficient confirmation.",
                )
            )

            return self._build_result(
                candidate,
                environment,
                decision,
                direction_aligned,
                environment_supports_trade,
                blocked_by_environment,
                reasons,
                warnings,
            )

        if not candidate.entry_ready or candidate.invalidated:
            decision = EnvironmentTradeDecision.REJECT

            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.CANDIDATE_INVALID,
                    "The candidate is not entry-ready or has "
                    "already been invalidated.",
                )
            )

            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.DECISION_REJECT,
                    "The candidate cannot proceed to trading.",
                )
            )

            return self._build_result(
                candidate,
                environment,
                decision,
                direction_aligned,
                environment_supports_trade,
                blocked_by_environment,
                reasons,
                warnings,
            )

        if not sufficient_environment_data:
            decision = EnvironmentTradeDecision.WAIT

            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.ENVIRONMENT_UNKNOWN,
                    "The unified market environment does not "
                    "contain sufficient data.",
                )
            )

            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.DECISION_WAIT,
                    "Trading is deferred until the environment "
                    "contains sufficient information.",
                )
            )

            return self._build_result(
                candidate,
                environment,
                decision,
                direction_aligned,
                environment_supports_trade,
                blocked_by_environment,
                reasons,
                warnings,
            )

        if environment_direction in (
            EnvironmentDirection.UNKNOWN,
            EnvironmentDirection.NEUTRAL,
        ):
            decision = EnvironmentTradeDecision.WAIT

            reasons.append(
                EnvironmentDecisionReason(
                    (
                        EnvironmentDecisionReasonType.ENVIRONMENT_UNKNOWN
                        if environment_direction
                        is EnvironmentDirection.UNKNOWN
                        else EnvironmentDecisionReasonType.ENVIRONMENT_NEUTRAL
                    ),
                    (
                        "The environment direction is unknown."
                        if environment_direction
                        is EnvironmentDirection.UNKNOWN
                        else
                        "The environment does not provide a "
                        "directional advantage."
                    ),
                )
            )

            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.DECISION_WAIT,
                    "The candidate must wait for clearer "
                    "environmental direction.",
                )
            )

            return self._build_result(
                candidate,
                environment,
                decision,
                direction_aligned,
                environment_supports_trade,
                blocked_by_environment,
                reasons,
                warnings,
            )

        if environment_conflict:
            decision = EnvironmentTradeDecision.WAIT

            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.ENVIRONMENT_CONFLICTED,
                    "The unified environment contains conflicting "
                    "technical or news conditions.",
                )
            )

            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.DECISION_WAIT,
                    "The candidate must wait while the environment "
                    "remains conflicted.",
                )
            )

            return self._build_result(
                candidate,
                environment,
                decision,
                direction_aligned,
                environment_supports_trade,
                blocked_by_environment,
                reasons,
                warnings,
            )

        if caution_required:
            decision = EnvironmentTradeDecision.WAIT

            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.ENVIRONMENT_CAUTION,
                    "The unified environment requires caution.",
                )
            )

            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.ENVIRONMENT_CAUTION_REQUIRED,
                    "Trading is deferred while the environment "
                    "contains a caution condition.",
                )
            )

            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.DECISION_WAIT,
                    "The candidate must wait until environmental "
                    "caution is cleared.",
                )
            )

            return self._build_result(
                candidate,
                environment,
                decision,
                direction_aligned,
                environment_supports_trade,
                blocked_by_environment,
                reasons,
                warnings,
            )

        if not direction_aligned:
            decision = EnvironmentTradeDecision.WAIT

            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.DIRECTION_CONFLICT,
                    "The candidate direction conflicts with the "
                    "current market environment direction.",
                )
            )

            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.DECISION_WAIT,
                    "The candidate should wait for directional "
                    "alignment or a new setup.",
                )
            )

            return self._build_result(
                candidate,
                environment,
                decision,
                direction_aligned,
                environment_supports_trade,
                blocked_by_environment,
                reasons,
                warnings,
            )

        if environment_strength < self.minimum_environment_strength:
            decision = EnvironmentTradeDecision.WAIT

            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.INSUFFICIENT_ENVIRONMENT_STRENGTH,
                    "The environment direction is aligned but "
                    "its strength is below the configured minimum.",
                )
            )

            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.DECISION_WAIT,
                    "The candidate should wait for stronger "
                    "environmental confirmation.",
                )
            )

            return self._build_result(
                candidate,
                environment,
                decision,
                direction_aligned,
                environment_supports_trade,
                blocked_by_environment,
                reasons,
                warnings,
            )

        if not environment_supports_trade:
            decision = EnvironmentTradeDecision.WAIT

            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.DIRECTION_CONFLICT,
                    "The environment does not sufficiently support "
                    "the candidate direction.",
                )
            )

            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.DECISION_WAIT,
                    "The candidate should wait for stronger "
                    "environmental support.",
                )
            )

            return self._build_result(
                candidate,
                environment,
                decision,
                direction_aligned,
                environment_supports_trade,
                blocked_by_environment,
                reasons,
                warnings,
            )

        decision = EnvironmentTradeDecision.TRADE

        reasons.append(
            EnvironmentDecisionReason(
                EnvironmentDecisionReasonType.DECISION_TRADE,
                "The candidate is trade-ready and aligned with "
                "a sufficiently strong market environment.",
            )
        )

        return self._build_result(
            candidate,
            environment,
            decision,
            direction_aligned,
            environment_supports_trade,
            blocked_by_environment,
            reasons,
            warnings,
        )

    def decide_xauusd(
        self,
        candidate: TradeCandidate,
        environment: MarketEnvironment,
    ) -> EnvironmentTradeDecisionResult:
        self._validate_inputs(candidate, environment)

        if candidate.symbol != "XAUUSD":
            raise EnvironmentTradeDecisionError(
                "candidate symbol must be XAUUSD."
            )

        if environment.symbol != "XAUUSD":
            raise EnvironmentTradeDecisionError(
                "environment symbol must be XAUUSD."
            )

        return self.decide(candidate, environment)

    @staticmethod
    def _validate_inputs(
        candidate: TradeCandidate,
        environment: MarketEnvironment,
    ) -> None:
        if not isinstance(candidate, TradeCandidate):
            raise EnvironmentTradeDecisionError(
                "candidate must be a TradeCandidate."
            )

        if not isinstance(environment, MarketEnvironment):
            raise EnvironmentTradeDecisionError(
                "environment must be a MarketEnvironment."
            )

        if candidate.symbol != environment.symbol:
            raise EnvironmentTradeDecisionError(
                "candidate and environment symbols must match."
            )

        if candidate.timeframe != environment.timeframe:
            raise EnvironmentTradeDecisionError(
                "candidate and environment timeframes must match."
            )

        if candidate.timestamp != environment.timestamp:
            raise EnvironmentTradeDecisionError(
                "candidate and environment timestamps must match."
            )

    @staticmethod
    def _direction_aligned(
        candidate_direction: SetupDirection,
        environment_direction: EnvironmentDirection,
    ) -> bool:
        if candidate_direction is SetupDirection.LONG:
            return environment_direction is EnvironmentDirection.BULLISH

        if candidate_direction is SetupDirection.SHORT:
            return environment_direction is EnvironmentDirection.BEARISH

        return False

    @staticmethod
    def _environment_supports_trade(
        candidate_direction: SetupDirection,
        environment: MarketEnvironment,
        direction_aligned: bool,
    ) -> bool:
        if not direction_aligned:
            return False

        if environment.environment_quality in (
            EnvironmentQuality.CONFLICTED,
            EnvironmentQuality.CAUTION,
            EnvironmentQuality.UNKNOWN,
        ):
            return False

        if candidate_direction is SetupDirection.LONG:
            return environment.technical_support or environment.news_support

        if candidate_direction is SetupDirection.SHORT:
            return environment.technical_support or environment.news_support

        return False

    @staticmethod
    def _blocked_by_environment(
        environment: MarketEnvironment,
        direction_aligned: bool,
        environment_supports_trade: bool,
    ) -> bool:
        if environment.environment_conflict:
            return True

        if environment.caution_required:
            return True

        if not direction_aligned:
            return True

        if not environment_supports_trade:
            return True

        return False

    @staticmethod
    def _add_candidate_reasons(
        candidate: TradeCandidate,
        reasons: list[EnvironmentDecisionReason],
    ) -> None:
        if candidate.decision is CandidateDecision.TRADE_READY:
            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.CANDIDATE_TRADE_READY,
                    "The Step 9 candidate is trade-ready.",
                )
            )

        elif candidate.decision is CandidateDecision.WAIT:
            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.CANDIDATE_WAITING,
                    "The Step 9 candidate is waiting.",
                )
            )

        else:
            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.CANDIDATE_REJECTED,
                    "The Step 9 candidate is rejected.",
                )
            )

    @staticmethod
    def _add_environment_reasons(
        candidate_direction: SetupDirection,
        environment: MarketEnvironment,
        direction_aligned: bool,
        reasons: list[EnvironmentDecisionReason],
        warnings: list[str],
    ) -> None:
        if environment.environment_quality is EnvironmentQuality.CLEAR:
            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.ENVIRONMENT_CLEAR,
                    "The market environment is clear.",
                )
            )

        elif environment.environment_quality is EnvironmentQuality.FAVORABLE:
            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.ENVIRONMENT_FAVORABLE,
                    "The market environment is favorable.",
                )
            )

        elif environment.environment_quality is EnvironmentQuality.MIXED:
            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.ENVIRONMENT_MIXED,
                    "The market environment is mixed.",
                )
            )

        elif environment.environment_quality is EnvironmentQuality.CONFLICTED:
            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.ENVIRONMENT_CONFLICTED,
                    "The market environment is conflicted.",
                )
            )

        elif environment.environment_quality is EnvironmentQuality.CAUTION:
            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.ENVIRONMENT_CAUTION,
                    "The market environment requires caution.",
                )
            )

        else:
            reasons.append(
                EnvironmentDecisionReason(
                    EnvironmentDecisionReasonType.ENVIRONMENT_UNKNOWN,
                    "The market environment quality is unknown.",
                )
            )

        if candidate_direction is SetupDirection.LONG:
            if direction_aligned:
                reasons.append(
                    EnvironmentDecisionReason(
                        EnvironmentDecisionReasonType.LONG_ENVIRONMENT_ALIGNMENT,
                        "The LONG candidate aligns with the "
                        "bullish environment direction.",
                    )
                )

        elif candidate_direction is SetupDirection.SHORT:
            if direction_aligned:
                reasons.append(
                    EnvironmentDecisionReason(
                        EnvironmentDecisionReasonType.SHORT_ENVIRONMENT_ALIGNMENT,
                        "The SHORT candidate aligns with the "
                        "bearish environment direction.",
                    )
                )

        if environment.environment_conflict:
            warnings.append(
                "Environment conflict is present."
            )

        if environment.caution_required:
            warnings.append(
                "Environment caution is active."
            )

    @staticmethod
    def _build_result(
        candidate: TradeCandidate,
        environment: MarketEnvironment,
        decision: EnvironmentTradeDecision,
        direction_aligned: bool,
        environment_supports_trade: bool,
        blocked_by_environment: bool,
        reasons: list[EnvironmentDecisionReason],
        warnings: list[str],
    ) -> EnvironmentTradeDecisionResult:
        return EnvironmentTradeDecisionResult(
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
            environment_conflict=environment.environment_conflict,
            caution_required=environment.caution_required,
            sufficient_environment_data=environment.sufficient_data,
            decision=decision,
            direction_aligned=direction_aligned,
            environment_supports_trade=environment_supports_trade,
            blocked_by_environment=blocked_by_environment,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
        )

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(100.0, float(value)))