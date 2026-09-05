from datetime import datetime, timezone

import pytest

from app.trading.macro.macro_observation import (
    MacroDirection,
    MacroIndicator,
    MacroObservation,
    MacroObservationError,
)


# Use a deterministic UTC timestamp for all tests.
BASE_TIME = datetime(
    2026,
    1,
    9,
    13,
    30,
    tzinfo=timezone.utc,
)


class TestMacroObservation:

    def test_basic_observation(self):
        # Create a valid DXY observation.
        observation = MacroObservation(
            timestamp=BASE_TIME,
            indicator=MacroIndicator.DXY,
            value=100.0,
            source="test",
        )

        # Verify the stored values.
        assert observation.timestamp == BASE_TIME
        assert observation.indicator == MacroIndicator.DXY
        assert observation.value == 100.0

    def test_forecast_surprise(self):
        # Create an observation with a forecast.
        observation = MacroObservation(
            timestamp=BASE_TIME,
            indicator=MacroIndicator.CPI,
            value=3.5,
            forecast=3.2,
            source="test",
        )

        # Verify observed minus forecast.
        assert observation.surprise == pytest.approx(0.3)

        # Forecast availability should be true.
        assert observation.has_forecast is True

        # Surprise availability should be true.
        assert observation.has_surprise is True

    def test_previous_change(self):
        # Create an observation with a previous value.
        observation = MacroObservation(
            timestamp=BASE_TIME,
            indicator=MacroIndicator.US_10Y_YIELD,
            value=4.25,
            previous=4.10,
            source="test",
        )

        # Verify the absolute change.
        assert observation.change_from_previous == pytest.approx(0.15)

        # Previous availability should be true.
        assert observation.has_previous is True

        # Change availability should be true.
        assert observation.has_change is True

    def test_missing_forecast_returns_no_surprise(self):
        # Create an observation without a forecast.
        observation = MacroObservation(
            timestamp=BASE_TIME,
            indicator=MacroIndicator.DXY,
            value=100.0,
            source="test",
        )

        # Surprise cannot be calculated.
        assert observation.surprise is None

        # The convenience flag must also be false.
        assert observation.has_surprise is False

    def test_missing_previous_returns_no_change(self):
        # Create an observation without a previous value.
        observation = MacroObservation(
            timestamp=BASE_TIME,
            indicator=MacroIndicator.DXY,
            value=100.0,
            source="test",
        )

        # Change cannot be calculated.
        assert observation.change_from_previous is None

        # The convenience flag must also be false.
        assert observation.has_change is False

    def test_serialization(self):
        # Create a fully populated observation.
        observation = MacroObservation(
            timestamp=BASE_TIME,
            indicator=MacroIndicator.NFP,
            value=250.0,
            previous=180.0,
            forecast=200.0,
            source="calendar",
            direction=MacroDirection.RISING,
        )

        # Serialize the observation.
        payload = observation.to_dict()

        # Verify primitive serialized values.
        assert payload["timestamp"] == BASE_TIME.isoformat()
        assert payload["indicator"] == "NFP"
        assert payload["value"] == 250.0
        assert payload["previous"] == 180.0
        assert payload["forecast"] == 200.0
        assert payload["source"] == "calendar"
        assert payload["direction"] == "RISING"

    @pytest.mark.parametrize(
        "value",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
        ],
    )
    def test_non_finite_value_rejected(self, value):
        # Invalid numerical values must be rejected.
        with pytest.raises(MacroObservationError):
            MacroObservation(
                timestamp=BASE_TIME,
                indicator=MacroIndicator.DXY,
                value=value,
                source="test",
            )

    def test_boolean_value_rejected(self):
        # Boolean values must not be accepted as numerical observations.
        with pytest.raises(MacroObservationError):
            MacroObservation(
                timestamp=BASE_TIME,
                indicator=MacroIndicator.DXY,
                value=True,
                source="test",
            )

    def test_invalid_timestamp_rejected(self):
        # Timestamp must be a datetime.
        with pytest.raises(MacroObservationError):
            MacroObservation(
                timestamp="2026-01-09",
                indicator=MacroIndicator.DXY,
                value=100.0,
                source="test",
            )

    def test_invalid_indicator_rejected(self):
        # Indicator must use the supported enum.
        with pytest.raises(MacroObservationError):
            MacroObservation(
                timestamp=BASE_TIME,
                indicator="DXY",
                value=100.0,
                source="test",
            )

    def test_invalid_previous_rejected(self):
        # Previous must be numeric when provided.
        with pytest.raises(MacroObservationError):
            MacroObservation(
                timestamp=BASE_TIME,
                indicator=MacroIndicator.DXY,
                value=100.0,
                previous="invalid",
                source="test",
            )

    def test_invalid_forecast_rejected(self):
        # Forecast must be numeric when provided.
        with pytest.raises(MacroObservationError):
            MacroObservation(
                timestamp=BASE_TIME,
                indicator=MacroIndicator.DXY,
                value=100.0,
                forecast="invalid",
                source="test",
            )

    def test_empty_source_rejected(self):
        # Source must identify the observation origin.
        with pytest.raises(MacroObservationError):
            MacroObservation(
                timestamp=BASE_TIME,
                indicator=MacroIndicator.DXY,
                value=100.0,
                source="",
            )

    def test_boolean_previous_rejected(self):
        # Boolean previous values must not be treated as numbers.
        with pytest.raises(MacroObservationError):
            MacroObservation(
                timestamp=BASE_TIME,
                indicator=MacroIndicator.DXY,
                value=100.0,
                previous=True,
                source="test",
            )

    def test_boolean_forecast_rejected(self):
        # Boolean forecast values must not be treated as numbers.
        with pytest.raises(MacroObservationError):
            MacroObservation(
                timestamp=BASE_TIME,
                indicator=MacroIndicator.DXY,
                value=100.0,
                forecast=True,
                source="test",
            )

    def test_immutable_observation(self):
        # Create a valid observation.
        observation = MacroObservation(
            timestamp=BASE_TIME,
            indicator=MacroIndicator.DXY,
            value=100.0,
            source="test",
        )

        # Frozen dataclass must prevent mutation.
        with pytest.raises(AttributeError):
            observation.value = 101.0