from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.trading.candidate.trade_candidate import (
    CandidateDecision,
    TradeCandidate,
)
from app.trading.news.news_risk_engine import (
    NewsRiskAssessment,
    NewsRiskLevel,
)


class NewsRiskIntegrationError(ValueError):
    """
    Raised when news-risk integration input is invalid.
    """


class IntegratedDecision(str, Enum):
    """
    Decision produced after combining the trading candidate
    with the economic-event risk assessment.
    """

    TRADE = "TRADE"
    WAIT = "WAIT"
    REJECT = "REJECT"


class IntegrationReasonType(str, Enum):
    """
    Reasons explaining the integrated decision.
    """

    CANDIDATE_READY = "CANDIDATE_READY"
    CANDIDATE_WAITING = "CANDIDATE_WAITING"
    CANDIDATE_REJECTED = "CANDIDATE_REJECTED"
    NEWS_CLEAR = "NEWS_CLEAR"
    NEWS_LOW_RISK = "NEWS_LOW_RISK"
    NEWS_MEDIUM_RISK = "NEWS_MEDIUM_RISK"
    NEWS_HIGH_RISK = "NEWS_HIGH_RISK"
    NEWS_EXTREME_RISK = "NEWS_EXTREME_RISK"
    NEWS_UNKNOWN = "NEWS_UNKNOWN"
    NEWS_DATA_INSUFFICIENT = "NEWS_DATA_INSUFFICIENT"


@dataclass(frozen=True, slots=True)
class IntegrationReason:
    """
    Explanation for the integrated decision.
    """

    reason_type: IntegrationReasonType
    message: str


@dataclass(frozen=True, slots=True)
class NewsRiskIntegrationResult:
    """
    Combined technical-candidate and news-risk assessment.

    News risk does not change the underlying candidate.
    It only adds an additional decision context.
    """

    timestamp: object
    symbol: str
    timeframe: str

    candidate_decision: CandidateDecision
    candidate_direction: object
    candidate_quality_score: float

    news_risk_level: NewsRiskLevel
    news_risk_score: float

    decision: IntegratedDecision

    candidate_ready: bool
    news_risk_present: bool
    news_risk_high: bool

    relevant_event_count: int
    high_impact_event_count: int
    usd_event_count: int

    reasons: tuple[IntegrationReason, ...]
    warnings: tuple[str, ...]

    @property
    def is_trade_allowed(self) -> bool:
        """
        Whether the integration layer permits the candidate
        to continue toward the next decision layer.

        This is deliberately not a hard news lockout policy.
        """

        return self.decision == IntegratedDecision.TRADE

    @property
    def should_wait(self) -> bool:
        return self.decision == IntegratedDecision.WAIT

    @property
    def should_reject(self) -> bool:
        return self.decision == IntegratedDecision.REJECT


class NewsRiskIntegrationEngine:
    """
    Combines an existing TradeCandidate with NewsRiskAssessment.

    Important architectural rule:

    - The candidate engine remains responsible for technical setup.
    - The news engine remains responsible for event risk.
    - This layer combines their information.
    - It does not modify either object.
    - It does not make bullish/bearish predictions.
    - It does not permanently hard-block trading because of news.

    Current policy:

    Candidate REJECTED
        -> REJECT

    Candidate WAIT
        -> WAIT

    Candidate TRADE_READY + EXTREME news risk
        -> WAIT

    Candidate TRADE_READY + HIGH news risk
        -> WAIT

    Candidate TRADE_READY + MEDIUM/LOW/NONE news risk
        -> TRADE

    Unknown/insufficient news data does not automatically reject
    a technically valid candidate. Instead, a warning is attached.
    """

    def integrate(
        self,
        *,
        candidate: TradeCandidate,
        news_risk: NewsRiskAssessment,
    ) -> NewsRiskIntegrationResult:
        self._validate_inputs(
            candidate=candidate,
            news_risk=news_risk,
        )

        candidate_ready = (
            candidate.decision
            == CandidateDecision.TRADE_READY
            and candidate.entry_ready
            and not candidate.invalidated
        )

        if candidate.decision == CandidateDecision.REJECT:
            decision = IntegratedDecision.REJECT

        elif candidate.decision == CandidateDecision.WAIT:
            decision = IntegratedDecision.WAIT

        elif not candidate_ready:
            decision = IntegratedDecision.WAIT

        elif news_risk.risk_level in {
            NewsRiskLevel.HIGH,
            NewsRiskLevel.EXTREME,
        }:
            decision = IntegratedDecision.WAIT

        else:
            decision = IntegratedDecision.TRADE

        reasons = self._build_reasons(
            candidate=candidate,
            news_risk=news_risk,
            decision=decision,
        )

        warnings = self._build_warnings(
            candidate=candidate,
            news_risk=news_risk,
        )

        return NewsRiskIntegrationResult(
            timestamp=candidate.timestamp,
            symbol=candidate.symbol,
            timeframe=candidate.timeframe,
            candidate_decision=candidate.decision,
            candidate_direction=candidate.direction,
            candidate_quality_score=candidate.setup_quality_score,
            news_risk_level=news_risk.risk_level,
            news_risk_score=news_risk.risk_score,
            decision=decision,
            candidate_ready=candidate_ready,
            news_risk_present=news_risk.has_relevant_events,
            news_risk_high=news_risk.is_high_risk,
            relevant_event_count=len(
                news_risk.relevant_events
            ),
            high_impact_event_count=(
                news_risk.high_impact_event_count
            ),
            usd_event_count=news_risk.usd_event_count,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
        )

    def _build_reasons(
        self,
        *,
        candidate: TradeCandidate,
        news_risk: NewsRiskAssessment,
        decision: IntegratedDecision,
    ) -> list[IntegrationReason]:
        reasons: list[IntegrationReason] = []

        if candidate.decision == CandidateDecision.REJECT:
            reasons.append(
                IntegrationReason(
                    IntegrationReasonType.CANDIDATE_REJECTED,
                    "The trade candidate was already rejected.",
                )
            )

        elif candidate.decision == CandidateDecision.WAIT:
            reasons.append(
                IntegrationReason(
                    IntegrationReasonType.CANDIDATE_WAITING,
                    "The trade candidate is not currently ready.",
                )
            )

        elif (
            candidate.decision
            == CandidateDecision.TRADE_READY
            and candidate.entry_ready
            and not candidate.invalidated
        ):
            reasons.append(
                IntegrationReason(
                    IntegrationReasonType.CANDIDATE_READY,
                    "The technical trade candidate is ready.",
                )
            )

        if news_risk.risk_level == NewsRiskLevel.NONE:
            reasons.append(
                IntegrationReason(
                    IntegrationReasonType.NEWS_CLEAR,
                    "No relevant economic-event risk is present.",
                )
            )

        elif news_risk.risk_level == NewsRiskLevel.LOW:
            reasons.append(
                IntegrationReason(
                    IntegrationReasonType.NEWS_LOW_RISK,
                    (
                        "Relevant economic events are present, "
                        "but assessed news risk is low."
                    ),
                )
            )

        elif news_risk.risk_level == NewsRiskLevel.MEDIUM:
            reasons.append(
                IntegrationReason(
                    IntegrationReasonType.NEWS_MEDIUM_RISK,
                    (
                        "Relevant economic-event risk is "
                        "classified as medium."
                    ),
                )
            )

        elif news_risk.risk_level == NewsRiskLevel.HIGH:
            reasons.append(
                IntegrationReason(
                    IntegrationReasonType.NEWS_HIGH_RISK,
                    (
                        "High economic-event risk is present; "
                        "the candidate should wait for the next "
                        "decision layer."
                    ),
                )
            )

        elif news_risk.risk_level == NewsRiskLevel.EXTREME:
            reasons.append(
                IntegrationReason(
                    IntegrationReasonType.NEWS_EXTREME_RISK,
                    (
                        "Extreme economic-event risk is present; "
                        "the candidate should wait."
                    ),
                )
            )

        else:
            reasons.append(
                IntegrationReason(
                    IntegrationReasonType.NEWS_UNKNOWN,
                    "News risk classification is unknown.",
                )
            )

        return reasons

    @staticmethod
    def _build_warnings(
        *,
        candidate: TradeCandidate,
        news_risk: NewsRiskAssessment,
    ) -> list[str]:
        warnings: list[str] = []

        if not news_risk.sufficient_data:
            warnings.append(
                "News-event data is insufficient for a complete risk assessment."
            )

        if news_risk.is_unknown:
            warnings.append(
                "News risk classification is unknown."
            )

        if (
            news_risk.risk_level
            in {
                NewsRiskLevel.HIGH,
                NewsRiskLevel.EXTREME,
            }
        ):
            warnings.append(
                "High news risk is delaying the candidate; "
                "this integration layer does not permanently block trading."
            )

        if candidate.invalidated:
            warnings.append(
                "The candidate is invalidated."
            )

        if not candidate.entry_ready:
            warnings.append(
                "The candidate is not entry-ready."
            )

        return warnings

    @staticmethod
    def _validate_inputs(
        *,
        candidate: TradeCandidate,
        news_risk: NewsRiskAssessment,
    ) -> None:
        if not isinstance(
            candidate,
            TradeCandidate,
        ):
            raise NewsRiskIntegrationError(
                "candidate must be a TradeCandidate."
            )

        if not isinstance(
            news_risk,
            NewsRiskAssessment,
        ):
            raise NewsRiskIntegrationError(
                "news_risk must be a NewsRiskAssessment."
            )

        if candidate.timestamp != news_risk.timestamp:
            raise NewsRiskIntegrationError(
                "candidate and news_risk timestamps must match."
            )

        if candidate.symbol.upper() != news_risk.symbol.upper():
            raise NewsRiskIntegrationError(
                "candidate and news_risk symbols must match."
            )