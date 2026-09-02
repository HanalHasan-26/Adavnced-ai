from datetime import datetime, timedelta, timezone

import pytest

from app.trading.news.economic_event import (
    EconomicEvent,
    EventImpact,
)
from app.trading.news.news_risk_engine import (
    NewsRiskEngine,
    NewsRiskEngineError,
    NewsRiskLevel,
    NewsRiskReasonType,
)


BASE_TIME = datetime(
    2026,
    1,
    9,
    13,
    30,
    tzinfo=timezone.utc,
)


def make_event(
    *,
    timestamp=None,
    name="Nonfarm Payrolls",
    currency="USD",
    impact=EventImpact.HIGH,
    previous=180.0,
    forecast=170.0,
    actual=None,
    source="calendar",
):
    if timestamp is None:
        timestamp = BASE_TIME

    return EconomicEvent(
        timestamp=timestamp,
        name=name,
        currency=currency,
        impact=impact,
        previous=previous,
        forecast=forecast,
        actual=actual,
        source=source,
    )


@pytest.fixture
def engine():
    return NewsRiskEngine()


class TestConfiguration:
    def test_default_configuration(self):
        engine = NewsRiskEngine()

        assert engine.relevant_window_minutes == 120.0
        assert engine.imminent_window_minutes == 30.0
        assert engine.high_risk_threshold == 60.0
        assert engine.extreme_risk_threshold == 85.0

    def test_custom_configuration(self):
        engine = NewsRiskEngine(
            relevant_window_minutes=180.0,
            imminent_window_minutes=45.0,
            high_risk_threshold=70.0,
            extreme_risk_threshold=90.0,
        )

        assert engine.relevant_window_minutes == 180.0
        assert engine.imminent_window_minutes == 45.0
        assert engine.high_risk_threshold == 70.0
        assert engine.extreme_risk_threshold == 90.0

    def test_negative_relevant_window_rejected(self):
        with pytest.raises(NewsRiskEngineError):
            NewsRiskEngine(
                relevant_window_minutes=-1
            )

    def test_negative_imminent_window_rejected(self):
        with pytest.raises(NewsRiskEngineError):
            NewsRiskEngine(
                imminent_window_minutes=-1
            )

    def test_negative_high_threshold_rejected(self):
        with pytest.raises(NewsRiskEngineError):
            NewsRiskEngine(
                high_risk_threshold=-1
            )

    def test_negative_extreme_threshold_rejected(self):
        with pytest.raises(NewsRiskEngineError):
            NewsRiskEngine(
                extreme_risk_threshold=-1
            )

    def test_imminent_window_cannot_exceed_relevant_window(self):
        with pytest.raises(NewsRiskEngineError):
            NewsRiskEngine(
                relevant_window_minutes=30,
                imminent_window_minutes=60,
            )

    def test_high_threshold_cannot_exceed_extreme_threshold(self):
        with pytest.raises(NewsRiskEngineError):
            NewsRiskEngine(
                high_risk_threshold=90,
                extreme_risk_threshold=80,
            )

    @pytest.mark.parametrize(
        "field_name",
        [
            "relevant_window_minutes",
            "imminent_window_minutes",
            "high_risk_threshold",
            "extreme_risk_threshold",
        ],
    )
    def test_boolean_configuration_rejected(
        self,
        field_name,
    ):
        with pytest.raises(NewsRiskEngineError):
            NewsRiskEngine(
                **{
                    field_name: True,
                }
            )


class TestInputValidation:
    def test_timestamp_must_be_datetime(
        self,
        engine,
    ):
        with pytest.raises(NewsRiskEngineError):
            engine.assess(
                timestamp="2026-01-09",
                symbol="XAUUSD",
                events=[],
            )

    def test_symbol_must_be_string(
        self,
        engine,
    ):
        with pytest.raises(NewsRiskEngineError):
            engine.assess(
                timestamp=BASE_TIME,
                symbol=123,
                events=[],
            )

    def test_empty_symbol_rejected(
        self,
        engine,
    ):
        with pytest.raises(NewsRiskEngineError):
            engine.assess(
                timestamp=BASE_TIME,
                symbol="",
                events=[],
            )

    def test_events_must_be_list(
        self,
        engine,
    ):
        with pytest.raises(NewsRiskEngineError):
            engine.assess(
                timestamp=BASE_TIME,
                symbol="XAUUSD",
                events=(),
            )

    def test_invalid_event_rejected(
        self,
        engine,
    ):
        with pytest.raises(NewsRiskEngineError):
            engine.assess(
                timestamp=BASE_TIME,
                symbol="XAUUSD",
                events=["invalid"],
            )

    def test_mismatched_timezone_rejected(
        self,
        engine,
    ):
        event = make_event(
            timestamp=datetime(
                2026,
                1,
                9,
                13,
                30,
            )
        )

        with pytest.raises(NewsRiskEngineError):
            engine.assess(
                timestamp=BASE_TIME,
                symbol="XAUUSD",
                events=[event],
            )


class TestNoEvents:
    def test_no_events_returns_none(
        self,
        engine,
    ):
        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[],
        )

        assert result.risk_level == NewsRiskLevel.NONE
        assert result.risk_score == 0.0
        assert result.relevant_events == ()
        assert result.usd_event_count == 0
        assert result.high_impact_event_count == 0
        assert result.nearest_event_minutes is None

    def test_no_events_marks_insufficient_data(
        self,
        engine,
    ):
        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[],
        )

        assert result.sufficient_data is False

    def test_no_events_reason(
        self,
        engine,
    ):
        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[],
        )

        assert any(
            reason.reason_type
            == NewsRiskReasonType.NO_RELEVANT_EVENTS
            for reason in result.reasons
        )


class TestRelevantEvents:
    def test_event_at_decision_time_is_relevant(
        self,
        engine,
    ):
        event = make_event()

        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[event],
        )

        assert len(result.relevant_events) == 1

    def test_event_30_minutes_before_is_relevant(
        self,
        engine,
    ):
        event = make_event(
            timestamp=BASE_TIME
            - timedelta(minutes=30)
        )

        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[event],
        )

        assert len(result.relevant_events) == 1

    def test_event_30_minutes_after_is_relevant(
        self,
        engine,
    ):
        event = make_event(
            timestamp=BASE_TIME
            + timedelta(minutes=30)
        )

        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[event],
        )

        assert len(result.relevant_events) == 1

    def test_event_at_relevant_window_boundary(
        self,
        engine,
    ):
        event = make_event(
            timestamp=BASE_TIME
            + timedelta(minutes=120)
        )

        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[event],
        )

        assert len(result.relevant_events) == 1

    def test_event_outside_relevant_window(
        self,
        engine,
    ):
        event = make_event(
            timestamp=BASE_TIME
            + timedelta(minutes=121)
        )

        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[event],
        )

        assert result.relevant_events == ()

    def test_unrelated_currency_is_ignored(
        self,
        engine,
    ):
        event = make_event(
            currency="JPY"
        )

        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[event],
        )

        assert result.relevant_events == ()

    def test_relevant_currency_for_eurusd(
        self,
        engine,
    ):
        event = make_event(
            currency="EUR",
            name="ECB Rate Decision",
        )

        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="EURUSD",
            events=[event],
        )

        assert len(result.relevant_events) == 1

    def test_usd_is_relevant_for_eurusd(
        self,
        engine,
    ):
        event = make_event(
            currency="USD"
        )

        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="EURUSD",
            events=[event],
        )

        assert len(result.relevant_events) == 1


class TestOrdering:
    def test_nearest_event_first(
        self,
        engine,
    ):
        first = make_event(
            timestamp=BASE_TIME
            + timedelta(minutes=60),
            name="CPI",
        )

        second = make_event(
            timestamp=BASE_TIME
            + timedelta(minutes=10),
            name="NFP",
        )

        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[first, second],
        )

        assert result.relevant_events[0].name == "NFP"
        assert result.relevant_events[1].name == "CPI"

    def test_nearest_event_minutes(
        self,
        engine,
    ):
        event = make_event(
            timestamp=BASE_TIME
            + timedelta(minutes=15)
        )

        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[event],
        )

        assert result.nearest_event_minutes == 15.0


class TestRiskLevels:
    def test_low_impact_event_has_low_risk(
        self,
        engine,
    ):
        event = make_event(
            impact=EventImpact.LOW,
        )

        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="EURUSD",
            events=[event],
        )

        assert result.risk_level == NewsRiskLevel.LOW

    def test_medium_impact_event_has_medium_risk(
        self,
        engine,
    ):
        event = make_event(
            impact=EventImpact.MEDIUM,
        )

        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="EURUSD",
            events=[event],
        )

        assert result.risk_level == NewsRiskLevel.MEDIUM

    def test_high_impact_event_has_high_risk(
        self,
        engine,
    ):
        event = make_event(
            impact=EventImpact.HIGH,
        )

        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="EURUSD",
            events=[event],
        )

        assert result.risk_level == NewsRiskLevel.HIGH

    def test_usd_high_impact_xauusd_can_be_extreme(
        self,
        engine,
    ):
        event = make_event(
            impact=EventImpact.HIGH,
        )

        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[event],
        )

        assert result.risk_score == 75.0
        assert result.risk_level == NewsRiskLevel.HIGH

    def test_imminent_event_has_higher_score(
        self,
        engine,
    ):
        imminent = make_event(
            timestamp=BASE_TIME
            + timedelta(minutes=10),
        )

        later = make_event(
            timestamp=BASE_TIME
            + timedelta(minutes=60),
        )

        imminent_result = engine.assess(
            timestamp=BASE_TIME,
            symbol="EURUSD",
            events=[imminent],
        )

        later_result = engine.assess(
            timestamp=BASE_TIME,
            symbol="EURUSD",
            events=[later],
        )

        assert (
            imminent_result.risk_score
            > later_result.risk_score
        )


class TestCounts:
    def test_usd_event_count(
        self,
        engine,
    ):
        events = [
            make_event(
                name="NFP",
                currency="USD",
            ),
            make_event(
                timestamp=BASE_TIME
                + timedelta(minutes=10),
                name="CPI",
                currency="USD",
            ),
            make_event(
                timestamp=BASE_TIME
                + timedelta(minutes=20),
                name="ECB",
                currency="EUR",
            ),
        ]

        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="EURUSD",
            events=events,
        )

        assert result.usd_event_count == 2

    def test_high_impact_count(
        self,
        engine,
    ):
        events = [
            make_event(
                name="NFP",
                impact=EventImpact.HIGH,
            ),
            make_event(
                timestamp=BASE_TIME
                + timedelta(minutes=10),
                name="PMI",
                impact=EventImpact.MEDIUM,
            ),
        ]

        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=events,
        )

        assert result.high_impact_event_count == 1


class TestMultipleEvents:
    def test_multiple_events_increase_score(
        self,
        engine,
    ):
        first = make_event(
            name="NFP",
        )

        second = make_event(
            timestamp=BASE_TIME
            + timedelta(minutes=10),
            name="CPI",
            impact=EventImpact.MEDIUM,
        )

        single = engine.assess(
            timestamp=BASE_TIME,
            symbol="EURUSD",
            events=[first],
        )

        multiple = engine.assess(
            timestamp=BASE_TIME,
            symbol="EURUSD",
            events=[first, second],
        )

        assert multiple.risk_score > single.risk_score

    def test_multiple_events_reason(
        self,
        engine,
    ):
        events = [
            make_event(name="NFP"),
            make_event(
                timestamp=BASE_TIME
                + timedelta(minutes=10),
                name="CPI",
                impact=EventImpact.MEDIUM,
            ),
        ]

        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="EURUSD",
            events=events,
        )

        assert any(
            reason.reason_type
            == NewsRiskReasonType.MULTIPLE_EVENTS
            for reason in result.reasons
        )


class TestReasons:
    def test_high_impact_reason(
        self,
        engine,
    ):
        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[
                make_event(
                    impact=EventImpact.HIGH
                )
            ],
        )

        assert any(
            reason.reason_type
            == NewsRiskReasonType.HIGH_IMPACT_EVENT
            for reason in result.reasons
        )

    def test_medium_impact_reason(
        self,
        engine,
    ):
        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[
                make_event(
                    impact=EventImpact.MEDIUM
                )
            ],
        )

        assert any(
            reason.reason_type
            == NewsRiskReasonType.MEDIUM_IMPACT_EVENT
            for reason in result.reasons
        )

    def test_low_impact_reason(
        self,
        engine,
    ):
        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[
                make_event(
                    impact=EventImpact.LOW
                )
            ],
        )

        assert any(
            reason.reason_type
            == NewsRiskReasonType.LOW_IMPACT_EVENT
            for reason in result.reasons
        )

    def test_usd_reason(
        self,
        engine,
    ):
        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[
                make_event(
                    currency="USD"
                )
            ],
        )

        assert any(
            reason.reason_type
            == NewsRiskReasonType.USD_EVENT
            for reason in result.reasons
        )

    def test_imminent_reason(
        self,
        engine,
    ):
        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[
                make_event(
                    timestamp=BASE_TIME
                    + timedelta(minutes=10)
                )
            ],
        )

        assert any(
            reason.reason_type
            == NewsRiskReasonType.EVENT_IMMINENT
            for reason in result.reasons
        )

    def test_pending_reason(
        self,
        engine,
    ):
        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[
                make_event(
                    timestamp=BASE_TIME
                    + timedelta(minutes=60)
                )
            ],
        )

        assert any(
            reason.reason_type
            == NewsRiskReasonType.EVENT_PENDING
            for reason in result.reasons
        )

    def test_recent_reason(
        self,
        engine,
    ):
        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[
                make_event(
                    timestamp=BASE_TIME
                    - timedelta(minutes=10)
                )
            ],
        )

        assert any(
            reason.reason_type
            == NewsRiskReasonType.EVENT_RECENT
            for reason in result.reasons
        )

    def test_actual_available_reason(
        self,
        engine,
    ):
        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[
                make_event(
                    actual=175.0
                )
            ],
        )

        assert any(
            reason.reason_type
            == NewsRiskReasonType.ACTUAL_AVAILABLE
            for reason in result.reasons
        )


class TestAssessmentProperties:
    def test_has_relevant_events(
        self,
        engine,
    ):
        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[
                make_event()
            ],
        )

        assert result.has_relevant_events is True

    def test_has_no_relevant_events(
        self,
        engine,
    ):
        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[],
        )

        assert result.has_relevant_events is False

    def test_has_high_impact_event(
        self,
        engine,
    ):
        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[
                make_event(
                    impact=EventImpact.HIGH
                )
            ],
        )

        assert result.has_high_impact_event is True

    def test_has_usd_event(
        self,
        engine,
    ):
        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[
                make_event(
                    currency="USD"
                )
            ],
        )

        assert result.has_usd_event is True

    def test_is_high_risk(
        self,
        engine,
    ):
        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[
                make_event(
                    impact=EventImpact.HIGH
                )
            ],
        )

        assert result.is_high_risk is True

    def test_unknown_property(
        self,
        engine,
    ):
        result = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[],
        )

        assert result.is_unknown is False


class TestAlias:
    def test_assess_symbol_matches_assess(
        self,
        engine,
    ):
        event = make_event()

        first = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[event],
        )

        second = engine.assess_symbol(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=[event],
        )

        assert first == second


class TestDeterminism:
    def test_same_input_produces_same_result(
        self,
        engine,
    ):
        events = [
            make_event(),
            make_event(
                timestamp=BASE_TIME
                + timedelta(minutes=20),
                name="CPI",
                impact=EventImpact.MEDIUM,
            ),
        ]

        first = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=events,
        )

        second = engine.assess(
            timestamp=BASE_TIME,
            symbol="XAUUSD",
            events=events,
        )

        assert first == second