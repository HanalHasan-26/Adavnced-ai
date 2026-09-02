from datetime import datetime, timezone

import pytest

from app.trading.candidate.trade_candidate import (
    CandidateDecision,
    TradeCandidate,
)
from app.trading.news.economic_event import (
    EconomicEvent,
    EventImpact,
)
from app.trading.news.news_risk_engine import (
    NewsRiskAssessment,
    NewsRiskLevel,
)
from app.trading.news.news_risk_integration import (
    IntegratedDecision,
    IntegrationReasonType,
    NewsRiskIntegrationEngine,
    NewsRiskIntegrationError,
)


BASE_TIME = datetime(
    2026,
    1,
    9,
    13,
    30,
    tzinfo=timezone.utc,
)


def make_candidate(
    *,
    decision=CandidateDecision.TRADE_READY,
    entry_ready=True,
    invalidated=False,
):
    return TradeCandidate(
        timestamp=BASE_TIME,
        symbol="XAUUSD",
        timeframe="M15",
        close=2000.0,
        decision=decision,
        direction="LONG",
        setup_type="TREND_CONTINUATION",
        setup_quality_score=85.0,
        confirmation_score=90.0,
        structure_confirmed=True,
        momentum_confirmed=True,
        price_confirmed=True,
        trend_confirmed=True,
        supporting_signals=(
            "structure",
            "momentum",
            "trend",
        ),
        conflicting_signals=(),
        reasons=(),
        warnings=(),
        entry_ready=entry_ready,
        invalidated=invalidated,
    )


def make_news_risk(
    *,
    level=NewsRiskLevel.NONE,
    score=0.0,
    events=(),
    usd_event_count=0,
    high_impact_event_count=0,
    sufficient_data=True,
):
    return NewsRiskAssessment(
        timestamp=BASE_TIME,
        symbol="XAUUSD",
        risk_level=level,
        risk_score=score,
        relevant_events=tuple(events),
        usd_event_count=usd_event_count,
        high_impact_event_count=high_impact_event_count,
        nearest_event_minutes=(
            10.0 if events else None
        ),
        reasons=(),
        sufficient_data=sufficient_data,
    )


def make_event(
    *,
    name="NFP",
    currency="USD",
    impact=EventImpact.HIGH,
):
    return EconomicEvent(
        timestamp=BASE_TIME,
        name=name,
        currency=currency,
        impact=impact,
        previous=180.0,
        forecast=170.0,
        actual=None,
        source="calendar",
    )


@pytest.fixture
def engine():
    return NewsRiskIntegrationEngine()


class TestValidation:
    def test_candidate_must_be_trade_candidate(
        self,
        engine,
    ):
        with pytest.raises(NewsRiskIntegrationError):
            engine.integrate(
                candidate="invalid",
                news_risk=make_news_risk(),
            )

    def test_news_risk_must_be_assessment(
        self,
        engine,
    ):
        with pytest.raises(NewsRiskIntegrationError):
            engine.integrate(
                candidate=make_candidate(),
                news_risk="invalid",
            )

    def test_timestamp_must_match(
        self,
        engine,
    ):
        candidate = make_candidate()

        news_risk = NewsRiskAssessment(
            timestamp=datetime(
                2026,
                1,
                9,
                14,
                30,
                tzinfo=timezone.utc,
            ),
            symbol="XAUUSD",
            risk_level=NewsRiskLevel.NONE,
            risk_score=0.0,
            relevant_events=(),
            usd_event_count=0,
            high_impact_event_count=0,
            nearest_event_minutes=None,
            reasons=(),
            sufficient_data=True,
        )

        with pytest.raises(NewsRiskIntegrationError):
            engine.integrate(
                candidate=candidate,
                news_risk=news_risk,
            )

    def test_symbol_must_match(
        self,
        engine,
    ):
        news_risk = NewsRiskAssessment(
            timestamp=BASE_TIME,
            symbol="EURUSD",
            risk_level=NewsRiskLevel.NONE,
            risk_score=0.0,
            relevant_events=(),
            usd_event_count=0,
            high_impact_event_count=0,
            nearest_event_minutes=None,
            reasons=(),
            sufficient_data=True,
        )

        with pytest.raises(NewsRiskIntegrationError):
            engine.integrate(
                candidate=make_candidate(),
                news_risk=news_risk,
            )


class TestCandidateDecision:
    def test_ready_candidate_with_no_news_risk_can_trade(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(),
            news_risk=make_news_risk(),
        )

        assert result.decision == IntegratedDecision.TRADE
        assert result.is_trade_allowed is True

    def test_waiting_candidate_remains_waiting(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(
                decision=CandidateDecision.WAIT
            ),
            news_risk=make_news_risk(),
        )

        assert result.decision == IntegratedDecision.WAIT
        assert result.should_wait is True

    def test_rejected_candidate_remains_rejected(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(
                decision=CandidateDecision.REJECT
            ),
            news_risk=make_news_risk(),
        )

        assert result.decision == IntegratedDecision.REJECT
        assert result.should_reject is True

    def test_trade_ready_but_entry_not_ready_waits(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(
                entry_ready=False
            ),
            news_risk=make_news_risk(),
        )

        assert result.decision == IntegratedDecision.WAIT

    def test_trade_ready_but_invalidated_waits(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(
                invalidated=True
            ),
            news_risk=make_news_risk(),
        )

        assert result.decision == IntegratedDecision.WAIT


class TestNewsRisk:
    def test_low_news_risk_does_not_block(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(),
            news_risk=make_news_risk(
                level=NewsRiskLevel.LOW,
                score=10.0,
            ),
        )

        assert result.decision == IntegratedDecision.TRADE

    def test_medium_news_risk_does_not_block(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(),
            news_risk=make_news_risk(
                level=NewsRiskLevel.MEDIUM,
                score=30.0,
            ),
        )

        assert result.decision == IntegratedDecision.TRADE

    def test_high_news_risk_waits(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(),
            news_risk=make_news_risk(
                level=NewsRiskLevel.HIGH,
                score=75.0,
            ),
        )

        assert result.decision == IntegratedDecision.WAIT

    def test_extreme_news_risk_waits(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(),
            news_risk=make_news_risk(
                level=NewsRiskLevel.EXTREME,
                score=90.0,
            ),
        )

        assert result.decision == IntegratedDecision.WAIT

    def test_unknown_news_does_not_automatically_block(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(),
            news_risk=make_news_risk(
                level=NewsRiskLevel.UNKNOWN,
                sufficient_data=True,
            ),
        )

        assert result.decision == IntegratedDecision.TRADE

    def test_insufficient_news_data_does_not_automatically_block(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(),
            news_risk=make_news_risk(
                level=NewsRiskLevel.NONE,
                sufficient_data=False,
            ),
        )

        assert result.decision == IntegratedDecision.TRADE
        assert result.warnings


class TestResultData:
    def test_candidate_data_is_preserved(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(),
            news_risk=make_news_risk(),
        )

        assert result.symbol == "XAUUSD"
        assert result.timeframe == "M15"
        assert result.candidate_decision == (
            CandidateDecision.TRADE_READY
        )
        assert result.candidate_quality_score == 85.0

    def test_news_data_is_preserved(
        self,
        engine,
    ):
        event = make_event()

        result = engine.integrate(
            candidate=make_candidate(),
            news_risk=make_news_risk(
                level=NewsRiskLevel.HIGH,
                score=75.0,
                events=(event,),
                usd_event_count=1,
                high_impact_event_count=1,
            ),
        )

        assert result.news_risk_level == NewsRiskLevel.HIGH
        assert result.news_risk_score == 75.0
        assert result.relevant_event_count == 1
        assert result.usd_event_count == 1
        assert result.high_impact_event_count == 1

    def test_news_risk_present(
        self,
        engine,
    ):
        event = make_event()

        result = engine.integrate(
            candidate=make_candidate(),
            news_risk=make_news_risk(
                level=NewsRiskLevel.HIGH,
                score=75.0,
                events=(event,),
                usd_event_count=1,
                high_impact_event_count=1,
            ),
        )

        assert result.news_risk_present is True
        assert result.news_risk_high is True


class TestReasons:
    def test_candidate_ready_reason(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(),
            news_risk=make_news_risk(),
        )

        assert any(
            reason.reason_type
            == IntegrationReasonType.CANDIDATE_READY
            for reason in result.reasons
        )

    def test_candidate_waiting_reason(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(
                decision=CandidateDecision.WAIT
            ),
            news_risk=make_news_risk(),
        )

        assert any(
            reason.reason_type
            == IntegrationReasonType.CANDIDATE_WAITING
            for reason in result.reasons
        )

    def test_candidate_rejected_reason(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(
                decision=CandidateDecision.REJECT
            ),
            news_risk=make_news_risk(),
        )

        assert any(
            reason.reason_type
            == IntegrationReasonType.CANDIDATE_REJECTED
            for reason in result.reasons
        )

    def test_news_clear_reason(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(),
            news_risk=make_news_risk(
                level=NewsRiskLevel.NONE
            ),
        )

        assert any(
            reason.reason_type
            == IntegrationReasonType.NEWS_CLEAR
            for reason in result.reasons
        )

    def test_low_news_reason(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(),
            news_risk=make_news_risk(
                level=NewsRiskLevel.LOW,
                score=10.0,
            ),
        )

        assert any(
            reason.reason_type
            == IntegrationReasonType.NEWS_LOW_RISK
            for reason in result.reasons
        )

    def test_medium_news_reason(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(),
            news_risk=make_news_risk(
                level=NewsRiskLevel.MEDIUM,
                score=30.0,
            ),
        )

        assert any(
            reason.reason_type
            == IntegrationReasonType.NEWS_MEDIUM_RISK
            for reason in result.reasons
        )

    def test_high_news_reason(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(),
            news_risk=make_news_risk(
                level=NewsRiskLevel.HIGH,
                score=75.0,
            ),
        )

        assert any(
            reason.reason_type
            == IntegrationReasonType.NEWS_HIGH_RISK
            for reason in result.reasons
        )

    def test_extreme_news_reason(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(),
            news_risk=make_news_risk(
                level=NewsRiskLevel.EXTREME,
                score=90.0,
            ),
        )

        assert any(
            reason.reason_type
            == IntegrationReasonType.NEWS_EXTREME_RISK
            for reason in result.reasons
        )

    def test_unknown_news_reason(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(),
            news_risk=make_news_risk(
                level=NewsRiskLevel.UNKNOWN,
            ),
        )

        assert any(
            reason.reason_type
            == IntegrationReasonType.NEWS_UNKNOWN
            for reason in result.reasons
        )


class TestWarnings:
    def test_high_news_risk_generates_warning(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(),
            news_risk=make_news_risk(
                level=NewsRiskLevel.HIGH,
                score=75.0,
            ),
        )

        assert any(
            "High news risk"
            in warning
            for warning in result.warnings
        )

    def test_extreme_news_risk_generates_warning(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(),
            news_risk=make_news_risk(
                level=NewsRiskLevel.EXTREME,
                score=90.0,
            ),
        )

        assert any(
            "High news risk"
            in warning
            for warning in result.warnings
        )

    def test_insufficient_data_generates_warning(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(),
            news_risk=make_news_risk(
                sufficient_data=False
            ),
        )

        assert any(
            "insufficient"
            in warning.lower()
            for warning in result.warnings
        )

    def test_invalidated_candidate_warning(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(
                invalidated=True
            ),
            news_risk=make_news_risk(),
        )

        assert any(
            "invalidated"
            in warning.lower()
            for warning in result.warnings
        )

    def test_not_entry_ready_warning(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(
                entry_ready=False
            ),
            news_risk=make_news_risk(),
        )

        assert any(
            "entry-ready"
            in warning
            for warning in result.warnings
        )


class TestPolicy:
    def test_high_news_risk_does_not_change_candidate(
        self,
        engine,
    ):
        candidate = make_candidate()

        result = engine.integrate(
            candidate=candidate,
            news_risk=make_news_risk(
                level=NewsRiskLevel.HIGH,
                score=75.0,
            ),
        )

        assert (
            result.candidate_decision
            == CandidateDecision.TRADE_READY
        )

    def test_medium_news_risk_keeps_trade_available(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(),
            news_risk=make_news_risk(
                level=NewsRiskLevel.MEDIUM,
                score=30.0,
            ),
        )

        assert result.is_trade_allowed is True

    def test_high_news_risk_requires_wait(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(),
            news_risk=make_news_risk(
                level=NewsRiskLevel.HIGH,
                score=75.0,
            ),
        )

        assert result.should_wait is True

    def test_no_news_does_not_create_block(
        self,
        engine,
    ):
        result = engine.integrate(
            candidate=make_candidate(),
            news_risk=make_news_risk(),
        )

        assert result.decision == IntegratedDecision.TRADE
        assert result.is_trade_allowed is True


class TestDeterminism:
    def test_same_inputs_produce_same_result(
        self,
        engine,
    ):
        candidate = make_candidate()

        news_risk = make_news_risk(
            level=NewsRiskLevel.MEDIUM,
            score=30.0,
        )

        first = engine.integrate(
            candidate=candidate,
            news_risk=news_risk,
        )

        second = engine.integrate(
            candidate=candidate,
            news_risk=news_risk,
        )

        assert first == second