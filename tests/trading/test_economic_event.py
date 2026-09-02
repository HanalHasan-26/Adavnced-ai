from datetime import datetime, timezone

import pytest

from app.trading.news.economic_event import (
    EconomicEvent,
    EconomicEventError,
    EventActualStatus,
    EventImpact,
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
        timestamp = datetime(
            2026,
            1,
            9,
            13,
            30,
            tzinfo=timezone.utc,
        )

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


class TestEconomicEventCreation:
    def test_valid_event(self):
        event = make_event()

        assert event.name == "Nonfarm Payrolls"
        assert event.currency == "USD"
        assert event.impact == EventImpact.HIGH

    def test_event_is_frozen(self):
        event = make_event()

        with pytest.raises(AttributeError):
            event.name = "CPI"

    def test_event_is_hashable(self):
        event = make_event()

        assert hash(event) is not None


class TestTimestamp:
    def test_timestamp_must_be_datetime(self):
        with pytest.raises(EconomicEventError):
            make_event(timestamp="2026-01-09")

    def test_timezone_aware_timestamp_is_supported(self):
        event = make_event(
            timestamp=datetime(
                2026,
                1,
                9,
                13,
                30,
                tzinfo=timezone.utc,
            )
        )

        assert event.timestamp.tzinfo is not None


class TestName:
    def test_empty_name_rejected(self):
        with pytest.raises(EconomicEventError):
            make_event(name="")

    def test_whitespace_name_rejected(self):
        with pytest.raises(EconomicEventError):
            make_event(name="   ")

    def test_non_string_name_rejected(self):
        with pytest.raises(EconomicEventError):
            make_event(name=123)


class TestCurrency:
    def test_empty_currency_rejected(self):
        with pytest.raises(EconomicEventError):
            make_event(currency="")

    def test_whitespace_currency_rejected(self):
        with pytest.raises(EconomicEventError):
            make_event(currency="   ")

    def test_non_string_currency_rejected(self):
        with pytest.raises(EconomicEventError):
            make_event(currency=123)


class TestImpact:
    def test_invalid_impact_rejected(self):
        with pytest.raises(EconomicEventError):
            make_event(impact="HIGH")

    @pytest.mark.parametrize(
        "impact",
        [
            EventImpact.LOW,
            EventImpact.MEDIUM,
            EventImpact.HIGH,
            EventImpact.UNKNOWN,
        ],
    )
    def test_all_impact_levels_supported(self, impact):
        event = make_event(impact=impact)

        assert event.impact == impact


class TestNumericValues:
    @pytest.mark.parametrize(
        "field_name",
        [
            "previous",
            "forecast",
            "actual",
        ],
    )
    def test_numeric_value_supported(self, field_name):
        event = make_event(
            **{
                field_name: 123.45,
            }
        )

        assert getattr(
            event,
            field_name,
        ) == 123.45

    @pytest.mark.parametrize(
        "field_name",
        [
            "previous",
            "forecast",
            "actual",
        ],
    )
    def test_none_value_supported(self, field_name):
        event = make_event(
            **{
                field_name: None,
            }
        )

        assert getattr(
            event,
            field_name,
        ) is None

    @pytest.mark.parametrize(
        "field_name",
        [
            "previous",
            "forecast",
            "actual",
        ],
    )
    def test_non_numeric_value_rejected(self, field_name):
        with pytest.raises(EconomicEventError):
            make_event(
                **{
                    field_name: "invalid",
                }
            )

    @pytest.mark.parametrize(
        "field_name",
        [
            "previous",
            "forecast",
            "actual",
        ],
    )
    @pytest.mark.parametrize(
        "value",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
        ],
    )
    def test_non_finite_value_rejected(
        self,
        field_name,
        value,
    ):
        with pytest.raises(EconomicEventError):
            make_event(
                **{
                    field_name: value,
                }
            )


class TestSource:
    def test_default_source(self):
        event = EconomicEvent(
            timestamp=datetime(
                2026,
                1,
                1,
            ),
            name="CPI",
            currency="USD",
            impact=EventImpact.HIGH,
        )

        assert event.source == "unknown"

    def test_empty_source_rejected(self):
        with pytest.raises(EconomicEventError):
            make_event(source="")

    def test_whitespace_source_rejected(self):
        with pytest.raises(EconomicEventError):
            make_event(source="   ")


class TestActualStatus:
    def test_pending_when_actual_missing(self):
        event = make_event(
            impact=EventImpact.HIGH,
            actual=None,
        )

        assert event.actual_status == EventActualStatus.PENDING

    def test_available_when_actual_exists(self):
        event = make_event(
            actual=175.0,
        )

        assert event.actual_status == EventActualStatus.AVAILABLE

    def test_unknown_when_impact_unknown_and_actual_missing(self):
        event = make_event(
            impact=EventImpact.UNKNOWN,
            actual=None,
        )

        assert event.actual_status == EventActualStatus.UNKNOWN


class TestValueProperties:
    def test_has_forecast(self):
        assert make_event(
            forecast=170.0
        ).has_forecast is True

    def test_no_forecast(self):
        assert make_event(
            forecast=None
        ).has_forecast is False

    def test_has_actual(self):
        assert make_event(
            actual=175.0
        ).has_actual is True

    def test_no_actual(self):
        assert make_event(
            actual=None
        ).has_actual is False

    def test_surprise(self):
        event = make_event(
            forecast=170.0,
            actual=175.0,
        )

        assert event.surprise == 5.0

    def test_negative_surprise(self):
        event = make_event(
            forecast=170.0,
            actual=160.0,
        )

        assert event.surprise == -10.0

    def test_surprise_missing_actual(self):
        event = make_event(
            forecast=170.0,
            actual=None,
        )

        assert event.surprise is None

    def test_surprise_missing_forecast(self):
        event = make_event(
            forecast=None,
            actual=175.0,
        )

        assert event.surprise is None

    def test_has_surprise(self):
        assert make_event(
            forecast=170.0,
            actual=175.0,
        ).has_surprise is True

    def test_has_no_surprise(self):
        assert make_event(
            forecast=170.0,
            actual=None,
        ).has_surprise is False


class TestImpactProperties:
    def test_high_impact(self):
        assert make_event(
            impact=EventImpact.HIGH
        ).is_high_impact is True

    def test_non_high_impact(self):
        assert make_event(
            impact=EventImpact.MEDIUM
        ).is_high_impact is False


class TestCurrencyProperties:
    def test_usd_event(self):
        assert make_event(
            currency="USD"
        ).is_usd_event is True

    def test_lowercase_usd_event(self):
        assert make_event(
            currency="usd"
        ).is_usd_event is True

    def test_non_usd_event(self):
        assert make_event(
            currency="EUR"
        ).is_usd_event is False


class TestSerialization:
    def test_to_dict(self):
        event = make_event(
            actual=175.0,
        )

        data = event.to_dict()

        assert data["name"] == "Nonfarm Payrolls"
        assert data["currency"] == "USD"
        assert data["impact"] == "HIGH"
        assert data["previous"] == 180.0
        assert data["forecast"] == 170.0
        assert data["actual"] == 175.0
        assert data["actual_status"] == "AVAILABLE"
        assert data["source"] == "calendar"

    def test_to_dict_pending(self):
        event = make_event(
            actual=None,
        )

        data = event.to_dict()

        assert data["actual_status"] == "PENDING"


class TestDeterminism:
    def test_same_input_produces_equal_events(self):
        first = make_event()
        second = make_event()

        assert first == second