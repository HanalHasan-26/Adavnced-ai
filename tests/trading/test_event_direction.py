from datetime import datetime, timezone

import pytest

from app.trading.news.economic_event import (
    EconomicEvent,
    EventImpact,
)
from app.trading.news.event_direction import (
    Direction,
    DirectionTarget,
    EventDirectionEngine,
    EventDirectionError,
    EventDirectionReasonType,
    SurpriseDirection,
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
    name="Nonfarm Payrolls",
    currency="USD",
    impact=EventImpact.HIGH,
    previous=180.0,
    forecast=170.0,
    actual=190.0,
):
    return EconomicEvent(
        timestamp=BASE_TIME,
        name=name,
        currency=currency,
        impact=impact,
        previous=previous,
        forecast=forecast,
        actual=actual,
        source="calendar",
    )


@pytest.fixture
def engine():
    return EventDirectionEngine()


class TestInputValidation:
    def test_event_must_be_economic_event(
        self,
        engine,
    ):
        with pytest.raises(EventDirectionError):
            engine.analyze(
                "invalid",
            )

    def test_target_must_be_direction_target(
        self,
        engine,
    ):
        with pytest.raises(EventDirectionError):
            engine.analyze(
                make_event(),
                target="XAUUSD",
            )


class TestPositiveSurprise:
    def test_positive_usd_surprise(
        self,
        engine,
    ):
        result = engine.analyze_xauusd(
            make_event(
                forecast=170.0,
                actual=190.0,
            )
        )

        assert result.surprise == 20.0
        assert (
            result.surprise_direction
            == SurpriseDirection.POSITIVE
        )

    def test_positive_usd_surprise_is_usd_bullish(
        self,
        engine,
    ):
        result = engine.analyze_currency(
            make_event(
                forecast=170.0,
                actual=190.0,
            )
        )

        assert (
            result.currency_direction
            == Direction.BULLISH
        )

    def test_positive_usd_surprise_is_xauusd_bearish(
        self,
        engine,
    ):
        result = engine.analyze_xauusd(
            make_event(
                forecast=170.0,
                actual=190.0,
            )
        )

        assert (
            result.instrument_direction
            == Direction.BEARISH
        )

        assert result.is_bearish is True


class TestNegativeSurprise:
    def test_negative_usd_surprise(
        self,
        engine,
    ):
        result = engine.analyze_xauusd(
            make_event(
                forecast=190.0,
                actual=170.0,
            )
        )

        assert result.surprise == -20.0
        assert (
            result.surprise_direction
            == SurpriseDirection.NEGATIVE
        )

    def test_negative_usd_surprise_is_usd_bearish(
        self,
        engine,
    ):
        result = engine.analyze_currency(
            make_event(
                forecast=190.0,
                actual=170.0,
            )
        )

        assert (
            result.currency_direction
            == Direction.BEARISH
        )

    def test_negative_usd_surprise_is_xauusd_bullish(
        self,
        engine,
    ):
        result = engine.analyze_xauusd(
            make_event(
                forecast=190.0,
                actual=170.0,
            )
        )

        assert (
            result.instrument_direction
            == Direction.BULLISH
        )

        assert result.is_bullish is True


class TestNoSurprise:
    def test_actual_equals_forecast(
        self,
        engine,
    ):
        result = engine.analyze_xauusd(
            make_event(
                forecast=180.0,
                actual=180.0,
            )
        )

        assert result.surprise == 0.0
        assert (
            result.surprise_direction
            == SurpriseDirection.NONE
        )

    def test_equal_forecast_is_neutral(
        self,
        engine,
    ):
        result = engine.analyze_xauusd(
            make_event(
                forecast=180.0,
                actual=180.0,
            )
        )

        assert (
            result.instrument_direction
            == Direction.NEUTRAL
        )

        assert result.is_neutral is True

    def test_equal_forecast_has_medium_confidence(
        self,
        engine,
    ):
        result = engine.analyze_xauusd(
            make_event(
                forecast=180.0,
                actual=180.0,
            )
        )

        assert result.confidence == 50.0


class TestMissingData:
    def test_missing_actual(
        self,
        engine,
    ):
        result = engine.analyze_xauusd(
            make_event(
                actual=None,
            )
        )

        assert result.surprise is None
        assert (
            result.surprise_direction
            == SurpriseDirection.UNKNOWN
        )
        assert (
            result.instrument_direction
            == Direction.UNKNOWN
        )
        assert result.confidence == 0.0
        assert result.sufficient_data is False
        assert result.is_unknown is True

    def test_missing_forecast(
        self,
        engine,
    ):
        result = engine.analyze_xauusd(
            make_event(
                forecast=None,
            )
        )

        assert result.surprise is None
        assert (
            result.surprise_direction
            == SurpriseDirection.UNKNOWN
        )
        assert (
            result.instrument_direction
            == Direction.UNKNOWN
        )
        assert result.confidence == 0.0
        assert result.sufficient_data is False

    def test_missing_actual_reason(
        self,
        engine,
    ):
        result = engine.analyze_xauusd(
            make_event(
                actual=None,
            )
        )

        assert any(
            reason.reason_type
            == EventDirectionReasonType.NO_ACTUAL
            for reason in result.reasons
        )

    def test_missing_forecast_reason(
        self,
        engine,
    ):
        result = engine.analyze_xauusd(
            make_event(
                forecast=None,
            )
        )

        assert any(
            reason.reason_type
            == EventDirectionReasonType.NO_FORECAST
            for reason in result.reasons
        )


class TestNonUsdEvents:
    def test_non_usd_currency_is_unknown_for_xauusd(
        self,
        engine,
    ):
        result = engine.analyze_xauusd(
            make_event(
                currency="EUR",
                forecast=100.0,
                actual=120.0,
            )
        )

        assert (
            result.currency_direction
            == Direction.UNKNOWN
        )

        assert (
            result.instrument_direction
            == Direction.UNKNOWN
        )

    def test_non_usd_currency_can_be_analyzed_as_currency(
        self,
        engine,
    ):
        result = engine.analyze_currency(
            make_event(
                currency="EUR",
                forecast=100.0,
                actual=120.0,
            )
        )

        assert (
            result.currency_direction
            == Direction.UNKNOWN
        )

        assert (
            result.instrument_direction
            == Direction.UNKNOWN
        )


class TestTargets:
    def test_currency_target_returns_currency_direction(
        self,
        engine,
    ):
        result = engine.analyze(
            make_event(
                forecast=100.0,
                actual=120.0,
            ),
            target=DirectionTarget.CURRENCY,
        )

        assert (
            result.currency_direction
            == Direction.BULLISH
        )

        assert (
            result.instrument_direction
            == Direction.BULLISH
        )

    def test_xauusd_target_inverts_usd_direction(
        self,
        engine,
    ):
        result = engine.analyze(
            make_event(
                forecast=100.0,
                actual=120.0,
            ),
            target=DirectionTarget.XAUUSD,
        )

        assert (
            result.currency_direction
            == Direction.BULLISH
        )

        assert (
            result.instrument_direction
            == Direction.BEARISH
        )


class TestConfidence:
    def test_usd_currency_direction_confidence(
        self,
        engine,
    ):
        result = engine.analyze_currency(
            make_event(
                forecast=100.0,
                actual=120.0,
            )
        )

        assert result.confidence == 80.0

    def test_usd_xauusd_direction_confidence(
        self,
        engine,
    ):
        result = engine.analyze_xauusd(
            make_event(
                forecast=100.0,
                actual=120.0,
            )
        )

        assert result.confidence == 80.0

    def test_unknown_mapping_has_low_confidence(
        self,
        engine,
    ):
        result = engine.analyze_xauusd(
            make_event(
                currency="EUR",
                forecast=100.0,
                actual=120.0,
            )
        )

        assert result.confidence == 30.0


class TestReasons:
    def test_positive_surprise_reason(
        self,
        engine,
    ):
        result = engine.analyze_xauusd(
            make_event(
                forecast=100.0,
                actual=120.0,
            )
        )

        assert any(
            reason.reason_type
            == EventDirectionReasonType.POSITIVE_SURPRISE
            for reason in result.reasons
        )

    def test_negative_surprise_reason(
        self,
        engine,
    ):
        result = engine.analyze_xauusd(
            make_event(
                forecast=120.0,
                actual=100.0,
            )
        )

        assert any(
            reason.reason_type
            == EventDirectionReasonType.NEGATIVE_SURPRISE
            for reason in result.reasons
        )

    def test_no_surprise_reason(
        self,
        engine,
    ):
        result = engine.analyze_xauusd(
            make_event(
                forecast=100.0,
                actual=100.0,
            )
        )

        assert any(
            reason.reason_type
            == EventDirectionReasonType.NO_SURPRISE
            for reason in result.reasons
        )

    def test_usd_positive_reason(
        self,
        engine,
    ):
        result = engine.analyze_currency(
            make_event(
                forecast=100.0,
                actual=120.0,
            )
        )

        assert any(
            reason.reason_type
            == EventDirectionReasonType.USD_POSITIVE
            for reason in result.reasons
        )

    def test_usd_negative_reason(
        self,
        engine,
    ):
        result = engine.analyze_currency(
            make_event(
                forecast=120.0,
                actual=100.0,
            )
        )

        assert any(
            reason.reason_type
            == EventDirectionReasonType.USD_NEGATIVE
            for reason in result.reasons
        )

    def test_xauusd_bearish_reason(
        self,
        engine,
    ):
        result = engine.analyze_xauusd(
            make_event(
                forecast=100.0,
                actual=120.0,
            )
        )

        assert any(
            reason.reason_type
            == EventDirectionReasonType.XAUUSD_BEARISH
            for reason in result.reasons
        )

    def test_xauusd_bullish_reason(
        self,
        engine,
    ):
        result = engine.analyze_xauusd(
            make_event(
                forecast=120.0,
                actual=100.0,
            )
        )

        assert any(
            reason.reason_type
            == EventDirectionReasonType.XAUUSD_BULLISH
            for reason in result.reasons
        )


class TestProperties:
    def test_bullish_property(
        self,
        engine,
    ):
        result = engine.analyze_xauusd(
            make_event(
                forecast=120.0,
                actual=100.0,
            )
        )

        assert result.is_bullish is True
        assert result.is_bearish is False
        assert result.is_neutral is False
        assert result.is_unknown is False

    def test_bearish_property(
        self,
        engine,
    ):
        result = engine.analyze_xauusd(
            make_event(
                forecast=100.0,
                actual=120.0,
            )
        )

        assert result.is_bearish is True
        assert result.is_bullish is False

    def test_unknown_property(
        self,
        engine,
    ):
        result = engine.analyze_xauusd(
            make_event(
                actual=None,
            )
        )

        assert result.is_unknown is True


class TestAliases:
    def test_analyze_xauusd_matches_target(
        self,
        engine,
    ):
        event = make_event(
            forecast=100.0,
            actual=120.0,
        )

        first = engine.analyze_xauusd(event)

        second = engine.analyze(
            event,
            target=DirectionTarget.XAUUSD,
        )

        assert first == second

    def test_analyze_currency_matches_target(
        self,
        engine,
    ):
        event = make_event(
            forecast=100.0,
            actual=120.0,
        )

        first = engine.analyze_currency(event)

        second = engine.analyze(
            event,
            target=DirectionTarget.CURRENCY,
        )

        assert first == second


class TestDeterminism:
    def test_same_input_same_result(
        self,
        engine,
    ):
        event = make_event(
            forecast=100.0,
            actual=120.0,
        )

        first = engine.analyze_xauusd(event)
        second = engine.analyze_xauusd(event)

        assert first == second