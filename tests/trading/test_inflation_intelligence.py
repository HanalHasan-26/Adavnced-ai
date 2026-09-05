# tests/trading/test_inflation_intelligence.py

"""Tests for deterministic inflation intelligence."""

# Import datetime utilities.
from datetime import datetime, timedelta, timezone

# Import pytest for assertions.
import pytest

# Import the existing macro models.
from app.trading.macro import (
    MacroDirection,
    MacroIndicator,
    MacroObservation,
)

# Import inflation intelligence.
from app.trading.macro import (
    InflationIntelligence,
    InflationIntelligenceError,
    InflationLevel,
    InflationSurpriseLevel,
)


# Use a deterministic timezone-aware timestamp.
BASE_TIME = datetime(
    2026,
    1,
    1,
    12,
    0,
    tzinfo=timezone.utc,
)


def make_inflation(
    indicator: MacroIndicator,
    value: float,
    previous: float | None,
    direction: MacroDirection,
    forecast: float | None = None,
    minutes: int = 0,
) -> MacroObservation:
    """Create a deterministic inflation observation."""

    # Calculate the observation timestamp.
    timestamp = BASE_TIME + timedelta(
        minutes=minutes,
    )

    # Return the existing MacroObservation model.
    return MacroObservation(
        timestamp=timestamp,
        indicator=indicator,
        value=value,
        previous=previous,
        forecast=forecast,
        source="test",
        direction=direction,
    )


def test_strong_hot_cpi() -> None:
    """A 0.20-point CPI rise should be strongly hot."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # 3.00 -> 3.20 equals +0.20 percentage points.
    observation = make_inflation(
        MacroIndicator.CPI,
        3.20,
        3.00,
        MacroDirection.RISING,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify strong-hot classification.
    assert result.level is InflationLevel.STRONG_HOT

    # Verify movement.
    assert result.change_from_previous == pytest.approx(0.20)

    # Verify full confidence.
    assert result.confidence == 100.0


def test_hot_cpi() -> None:
    """A meaningful CPI rise should be classified as hot."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # 3.00 -> 3.10 equals +0.10 percentage points.
    observation = make_inflation(
        MacroIndicator.CPI,
        3.10,
        3.00,
        MacroDirection.RISING,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify hot classification.
    assert result.level is InflationLevel.HOT


def test_strong_cooling_cpi() -> None:
    """A 0.20-point CPI fall should be strongly cooling."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # 3.20 -> 3.00 equals -0.20 percentage points.
    observation = make_inflation(
        MacroIndicator.CPI,
        3.00,
        3.20,
        MacroDirection.FALLING,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify strong-cooling classification.
    assert result.level is InflationLevel.STRONG_COOLING

    # Verify movement.
    assert result.change_from_previous == pytest.approx(-0.20)


def test_cooling_cpi() -> None:
    """A meaningful CPI decline should be cooling."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # 3.20 -> 3.10 equals -0.10 percentage points.
    observation = make_inflation(
        MacroIndicator.CPI,
        3.10,
        3.20,
        MacroDirection.FALLING,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify cooling classification.
    assert result.level is InflationLevel.COOLING


def test_small_inflation_change_is_neutral() -> None:
    """Movement below the meaningful threshold is neutral."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # 3.00 -> 3.02 equals +0.02 percentage points.
    observation = make_inflation(
        MacroIndicator.CPI,
        3.02,
        3.00,
        MacroDirection.RISING,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify neutral classification.
    assert result.level is InflationLevel.NEUTRAL


def test_stable_inflation_is_neutral() -> None:
    """Stable inflation should be neutral."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # Create unchanged inflation.
    observation = make_inflation(
        MacroIndicator.CPI,
        3.00,
        3.00,
        MacroDirection.STABLE,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify neutral classification.
    assert result.level is InflationLevel.NEUTRAL

    # Verify zero change.
    assert result.change_from_previous == 0.0


def test_unknown_direction_is_unknown() -> None:
    """Unknown direction must not create an inflation signal."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # Create unknown-direction observation.
    observation = make_inflation(
        MacroIndicator.CPI,
        3.00,
        2.90,
        MacroDirection.UNKNOWN,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify unknown state.
    assert result.level is InflationLevel.UNKNOWN
    assert result.sufficient_data is False
    assert result.confidence == 0.0


def test_direction_without_previous_is_lower_confidence() -> None:
    """Direction alone should have lower confidence."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # Previous value is unavailable.
    observation = make_inflation(
        MacroIndicator.CPI,
        3.00,
        None,
        MacroDirection.RISING,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Direction alone should indicate hot inflation.
    assert result.level is InflationLevel.HOT

    # No change can be calculated.
    assert result.change_from_previous is None

    # Confidence is reduced.
    assert result.confidence == 50.0


def test_upside_forecast_surprise() -> None:
    """Actual inflation above forecast should be upside."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # Actual 3.20 versus forecast 3.10.
    observation = make_inflation(
        MacroIndicator.CPI,
        3.20,
        3.00,
        MacroDirection.RISING,
        forecast=3.10,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify surprise.
    assert result.surprise == pytest.approx(0.10)

    # Verify surprise classification.
    assert result.surprise_level is InflationSurpriseLevel.UPSIDE


def test_strong_upside_forecast_surprise() -> None:
    """A large upside surprise should be strongly upside."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # Actual 3.40 versus forecast 3.10.
    observation = make_inflation(
        MacroIndicator.CPI,
        3.40,
        3.00,
        MacroDirection.RISING,
        forecast=3.10,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify strong upside surprise.
    assert result.surprise_level is InflationSurpriseLevel.STRONG_UPSIDE


def test_downside_forecast_surprise() -> None:
    """Actual inflation below forecast should be downside."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # Actual 3.00 versus forecast 3.10.
    observation = make_inflation(
        MacroIndicator.CPI,
        3.00,
        3.20,
        MacroDirection.FALLING,
        forecast=3.10,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify surprise.
    assert result.surprise == pytest.approx(-0.10)

    # Verify downside classification.
    assert result.surprise_level is InflationSurpriseLevel.DOWNSIDE


def test_strong_downside_forecast_surprise() -> None:
    """A large downside surprise should be strongly downside."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # Actual 2.80 versus forecast 3.10.
    observation = make_inflation(
        MacroIndicator.CPI,
        2.80,
        3.00,
        MacroDirection.FALLING,
        forecast=3.10,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify strong downside.
    assert result.surprise_level is InflationSurpriseLevel.STRONG_DOWNSIDE


def test_forecast_in_line() -> None:
    """Actual inflation matching forecast should be in-line."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # Actual equals forecast.
    observation = make_inflation(
        MacroIndicator.CPI,
        3.10,
        3.00,
        MacroDirection.RISING,
        forecast=3.10,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify zero surprise.
    assert result.surprise == 0.0

    # Verify in-line classification.
    assert result.surprise_level is InflationSurpriseLevel.IN_LINE


def test_missing_forecast_returns_unknown_surprise() -> None:
    """Missing forecast should make surprise classification unknown."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # No forecast is supplied.
    observation = make_inflation(
        MacroIndicator.CPI,
        3.10,
        3.00,
        MacroDirection.RISING,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Surprise comparison is unavailable.
    assert result.surprise is None

    # Surprise level must be unknown.
    assert result.surprise_level is InflationSurpriseLevel.UNKNOWN


def test_future_observation_is_ignored() -> None:
    """Future inflation data must never affect historical analysis."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # Create a future CPI observation.
    future = make_inflation(
        MacroIndicator.CPI,
        5.00,
        4.00,
        MacroDirection.RISING,
        forecast=4.00,
        minutes=10,
    )

    # Analyze before the future observation.
    result = engine.analyze(
        [future],
        BASE_TIME,
    )

    # Future information must be ignored.
    assert result.level is InflationLevel.UNKNOWN
    assert result.sufficient_data is False


def test_exact_timestamp_is_allowed() -> None:
    """Observation exactly at decision time is valid."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # Observation occurs exactly at BASE_TIME.
    observation = make_inflation(
        MacroIndicator.CPI,
        3.20,
        3.00,
        MacroDirection.RISING,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Exact timestamp must be included.
    assert result.level is InflationLevel.STRONG_HOT


def test_latest_historical_observation_is_selected() -> None:
    """Latest historical-safe observation must be selected."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # Older CPI observation is hot.
    older = make_inflation(
        MacroIndicator.CPI,
        3.20,
        3.00,
        MacroDirection.RISING,
        minutes=-10,
    )

    # Newer CPI observation is cooling.
    newer = make_inflation(
        MacroIndicator.CPI,
        2.80,
        3.00,
        MacroDirection.FALLING,
        minutes=-5,
    )

    # Analyze both.
    result = engine.analyze(
        [older, newer],
        BASE_TIME,
    )

    # Newer observation must win.
    assert result.value == pytest.approx(2.80)

    # Newer direction must win.
    assert result.level is InflationLevel.STRONG_COOLING


def test_core_cpi_is_supported() -> None:
    """Core CPI should be independently supported."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # Create Core CPI data.
    observation = make_inflation(
        MacroIndicator.CORE_CPI,
        3.50,
        3.30,
        MacroDirection.RISING,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
        indicator=MacroIndicator.CORE_CPI,
    )

    # Verify indicator.
    assert result.indicator is MacroIndicator.CORE_CPI

    # Verify classification.
    assert result.level is InflationLevel.STRONG_HOT


def test_pce_is_supported() -> None:
    """PCE should be independently supported."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # Create PCE data.
    observation = make_inflation(
        MacroIndicator.PCE,
        2.80,
        2.70,
        MacroDirection.RISING,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
        indicator=MacroIndicator.PCE,
    )

    # Verify indicator.
    assert result.indicator is MacroIndicator.PCE

    # Verify classification.
    assert result.level is InflationLevel.HOT


def test_core_pce_is_supported() -> None:
    """Core PCE should be independently supported."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # Create Core PCE data.
    observation = make_inflation(
        MacroIndicator.CORE_PCE,
        2.50,
        2.60,
        MacroDirection.FALLING,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
        indicator=MacroIndicator.CORE_PCE,
    )

    # Verify indicator.
    assert result.indicator is MacroIndicator.CORE_PCE

    # Verify classification.
    assert result.level is InflationLevel.COOLING


def test_analyze_all_supports_all_indicators() -> None:
    """analyze_all should return all four inflation indicators."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # Create one observation for each indicator.
    observations = [
        make_inflation(
            MacroIndicator.CPI,
            3.20,
            3.00,
            MacroDirection.RISING,
        ),
        make_inflation(
            MacroIndicator.CORE_CPI,
            3.40,
            3.30,
            MacroDirection.RISING,
        ),
        make_inflation(
            MacroIndicator.PCE,
            2.80,
            2.70,
            MacroDirection.RISING,
        ),
        make_inflation(
            MacroIndicator.CORE_PCE,
            2.50,
            2.60,
            MacroDirection.FALLING,
        ),
    ]

    # Analyze all inflation indicators.
    results = engine.analyze_all(
        observations,
        BASE_TIME,
    )

    # Four independent results should exist.
    assert len(results) == 4

    # Every supported indicator should be present.
    for indicator in engine.INFLATION_INDICATORS:
        assert indicator in results
        assert results[indicator].sufficient_data is True


def test_zero_previous_does_not_divide_by_zero() -> None:
    """Zero previous inflation should not cause division by zero."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # Previous value is zero.
    observation = make_inflation(
        MacroIndicator.CPI,
        3.00,
        0.00,
        MacroDirection.RISING,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Absolute movement remains available.
    assert result.change_from_previous == pytest.approx(3.00)

    # Percentage change is unavailable.
    assert result.percentage_change is None


def test_timezone_mismatch_is_rejected() -> None:
    """Naive and timezone-aware timestamps cannot be mixed."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # Create a naive observation.
    observation = MacroObservation(
        timestamp=datetime(
            2026,
            1,
            1,
            12,
            0,
        ),
        indicator=MacroIndicator.CPI,
        value=3.20,
        previous=3.00,
        forecast=None,
        source="test",
        direction=MacroDirection.RISING,
    )

    # Mixed timestamp semantics must raise an error.
    with pytest.raises(InflationIntelligenceError):
        engine.analyze(
            [observation],
            BASE_TIME,
        )


def test_invalid_indicator_is_rejected() -> None:
    """Non-inflation indicators cannot be requested."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # DXY is not an inflation indicator.
    with pytest.raises(InflationIntelligenceError):
        engine.analyze(
            [],
            BASE_TIME,
            indicator=MacroIndicator.DXY,
        )


def test_invalid_thresholds_are_rejected() -> None:
    """Invalid threshold configurations must fail."""

    # Negative significant movement is invalid.
    with pytest.raises(InflationIntelligenceError):
        InflationIntelligence(
            significant_change=-0.01,
        )

    # Strong movement cannot be below significant movement.
    with pytest.raises(InflationIntelligenceError):
        InflationIntelligence(
            significant_change=0.20,
            strong_change=0.10,
        )

    # Negative significant surprise is invalid.
    with pytest.raises(InflationIntelligenceError):
        InflationIntelligence(
            significant_surprise=-0.01,
        )

    # Strong surprise cannot be below significant surprise.
    with pytest.raises(InflationIntelligenceError):
        InflationIntelligence(
            significant_surprise=0.20,
            strong_surprise=0.10,
        )


def test_xauusd_wrapper_returns_inflation_assessment_only() -> None:
    """XAUUSD wrapper must not create a trade decision."""

    # Create the inflation engine.
    engine = InflationIntelligence()

    # Create rising CPI data.
    observation = make_inflation(
        MacroIndicator.CPI,
        3.20,
        3.00,
        MacroDirection.RISING,
    )

    # Analyze through the XAUUSD wrapper.
    result = engine.analyze_xauusd(
        [observation],
        BASE_TIME,
    )

    # Verify the inflation indicator.
    assert result.indicator is MacroIndicator.CPI

    # Verify inflation classification.
    assert result.level is InflationLevel.STRONG_HOT