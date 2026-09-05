from datetime import datetime, timedelta, timezone

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
)


# Use one deterministic UTC timestamp for all historical tests.
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
    actual=250.0,
    forecast=180.0,
    currency="USD",
):
    """
    Create a deterministic economic event for timing tests.
    """

    # Default the release time to BASE_TIME.
    if timestamp is None:
        timestamp = BASE_TIME

    # Return a valid high-impact USD event.
    return EconomicEvent(
        timestamp=timestamp,
        name="Nonfarm Payrolls",
        currency=currency,
        impact=EventImpact.HIGH,
        previous=170.0,
        forecast=forecast,
        actual=actual,
        source="test",
    )


class TestHistoricalInformationBoundary:
    """
    Verify that historical analysis cannot use future information.
    """

    def test_before_release_actual_is_not_used(self):
        """
        A decision before the release must not see the actual value.
        """

        # Create an event scheduled for BASE_TIME.
        event = make_event()

        # Make the trading decision ten minutes before release.
        decision_timestamp = (
            BASE_TIME - timedelta(minutes=10)
        )

        # Analyze using the historical-safe API.
        result = EventDirectionEngine().analyze_at(
            event,
            decision_timestamp=decision_timestamp,
            target=DirectionTarget.XAUUSD,
        )

        # Direction must remain unknown before the release.
        assert result.instrument_direction == Direction.UNKNOWN

        # The surprise must not be available before release.
        assert result.surprise is None

        # The result must explicitly indicate insufficient information.
        assert result.sufficient_data is False

        # The reason must explain why the actual cannot be used.
        assert any(
            reason.reason_type
            == EventDirectionReasonType.NO_ACTUAL
            for reason in result.reasons
        )

    def test_at_release_actual_can_be_used(self):
        """
        At the exact release timestamp, actual information may be used.
        """

        # Create an event released exactly at BASE_TIME.
        event = make_event()

        # Make the decision at the exact release timestamp.
        result = EventDirectionEngine().analyze_at(
            event,
            decision_timestamp=BASE_TIME,
            target=DirectionTarget.XAUUSD,
        )

        # Positive USD surprise means USD bullish.
        assert result.currency_direction == Direction.BULLISH

        # USD bullishness is interpreted as bearish pressure on gold.
        assert result.instrument_direction == Direction.BEARISH

        # The actual-versus-forecast surprise is now available.
        assert result.surprise == 70.0

        # The result is sufficiently informed.
        assert result.sufficient_data is True

    def test_after_release_actual_can_be_used(self):
        """
        After release, the actual value may legitimately influence direction.
        """

        # Create an event released at BASE_TIME.
        event = make_event()

        # Make the decision ten minutes after release.
        decision_timestamp = (
            BASE_TIME + timedelta(minutes=10)
        )

        # Analyze using the historical-safe API.
        result = EventDirectionEngine().analyze_at(
            event,
            decision_timestamp=decision_timestamp,
            target=DirectionTarget.XAUUSD,
        )

        # The positive USD surprise should produce bearish XAUUSD direction.
        assert result.instrument_direction == Direction.BEARISH

        # The actual value must be available after release.
        assert result.surprise == 70.0

        # Historical information is now sufficient.
        assert result.sufficient_data is True

    def test_future_actual_cannot_change_pre_release_result(self):
        """
        Changing the future actual value must not change a pre-release result.
        """

        # Create two events with identical release/forecast information
        # but different actual values.
        event_without_large_actual = make_event(
            actual=1.0,
        )

        event_with_large_actual = make_event(
            actual=9999.0,
        )

        # Evaluate both before the release.
        decision_timestamp = (
            BASE_TIME - timedelta(minutes=1)
        )

        engine = EventDirectionEngine()

        # Analyze the first event before release.
        first_result = engine.analyze_at(
            event_without_large_actual,
            decision_timestamp=decision_timestamp,
            target=DirectionTarget.XAUUSD,
        )

        # Analyze the second event before release.
        second_result = engine.analyze_at(
            event_with_large_actual,
            decision_timestamp=decision_timestamp,
            target=DirectionTarget.XAUUSD,
        )

        # Future actual values must have no influence whatsoever.
        assert first_result.instrument_direction == (
            second_result.instrument_direction
        )

        # Both results must have no usable surprise.
        assert first_result.surprise is None
        assert second_result.surprise is None

        # Both results must remain insufficient before release.
        assert first_result.sufficient_data is False
        assert second_result.sufficient_data is False

    def test_previous_does_not_become_forecast(self):
        """
        Previous data alone must never be treated as a forecast.
        """

        # Create an event without a forecast.
        event = make_event(
            forecast=None,
            actual=250.0,
        )

        # Analyze after release when actual is available.
        result = EventDirectionEngine().analyze_at(
            event,
            decision_timestamp=(
                BASE_TIME + timedelta(minutes=1)
            ),
            target=DirectionTarget.XAUUSD,
        )

        # Direction cannot be inferred without a forecast.
        assert result.instrument_direction == Direction.UNKNOWN

        # No surprise should be calculated.
        assert result.surprise is None

        # The engine must report insufficient data.
        assert result.sufficient_data is False

        # The reason must identify the missing forecast.
        assert any(
            reason.reason_type
            == EventDirectionReasonType.NO_FORECAST
            for reason in result.reasons
        )


class TestHistoricalTimezoneSafety:
    """
    Verify deterministic timezone handling for historical decisions.
    """

    def test_mismatched_timezone_is_rejected(self):
        """
        Event and decision timestamps must use identical timezone semantics.
        """

        # Create an event with a timezone-aware UTC timestamp.
        event = make_event()

        # Create a naive historical decision timestamp.
        decision_timestamp = datetime(
            2026,
            1,
            9,
            13,
            20,
        )

        # Mixing naive and aware timestamps must be rejected.
        with pytest.raises(EventDirectionError):
            EventDirectionEngine().analyze_at(
                event,
                decision_timestamp=decision_timestamp,
                target=DirectionTarget.XAUUSD,
            )

    def test_decision_timestamp_must_be_datetime(self):
        """
        Historical decision timestamps must be datetime objects.
        """

        # Create a valid economic event.
        event = make_event()

        # Reject invalid timestamp types.
        with pytest.raises(EventDirectionError):
            EventDirectionEngine().analyze_at(
                event,
                decision_timestamp="2026-01-09T13:20:00Z",
                target=DirectionTarget.XAUUSD,
            )


class TestHistoricalXAUUSDConvenienceMethod:
    """
    Verify the dedicated historical XAUUSD helper.
    """

    def test_xauusd_helper_is_historical_safe(self):
        """
        The XAUUSD convenience method must preserve the release boundary.
        """

        # Create a USD event scheduled at BASE_TIME.
        event = make_event()

        # Analyze ten minutes before release.
        result = EventDirectionEngine().analyze_xauusd_at(
            event,
            decision_timestamp=(
                BASE_TIME - timedelta(minutes=10)
            ),
        )

        # Future information must not be exposed.
        assert result.instrument_direction == Direction.UNKNOWN

        # The surprise must remain unavailable.
        assert result.surprise is None

        # The result must report insufficient information.
        assert result.sufficient_data is False