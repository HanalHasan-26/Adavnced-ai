# tests/trading/test_fed_rate_intelligence.py

"""Tests for deterministic Federal Reserve rate intelligence."""

# Import datetime utilities.
from datetime import datetime, timedelta, timezone

# Import pytest for assertions and exception checks.
import pytest

# Import the existing macro models.
from app.trading.macro import (
    MacroDirection,
    MacroIndicator,
    MacroObservation,
)

# Import Fed-rate intelligence.
from app.trading.macro import (
    FedRateIntelligence,
    FedRateIntelligenceError,
    FedRateLevel,
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


def make_rate(
    value: float,
    previous: float | None,
    direction: MacroDirection,
    forecast: float | None = None,
    minutes: int = 0,
) -> MacroObservation:
    """Create a deterministic Federal Funds Rate observation."""

    # Calculate the observation timestamp.
    timestamp = BASE_TIME + timedelta(
        minutes=minutes,
    )

    # Return the existing MacroObservation model.
    return MacroObservation(
        timestamp=timestamp,
        indicator=MacroIndicator.FED_FUNDS_RATE,
        value=value,
        previous=previous,
        forecast=forecast,
        source="test",
        direction=direction,
    )


def test_strong_hawkish_25_basis_point_rise() -> None:
    """A 25-basis-point increase should be strongly hawkish."""

    # Create the Fed-rate engine.
    engine = FedRateIntelligence()

    # 4.00 -> 4.25 equals +25 basis points.
    observation = make_rate(
        4.25,
        4.00,
        MacroDirection.RISING,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify strong hawkish classification.
    assert result.level is FedRateLevel.STRONG_HAWKISH

    # Verify basis-point movement.
    assert result.change_basis_points == pytest.approx(25.0)

    # Verify full confidence.
    assert result.confidence == 100.0


def test_hawkish_10_basis_point_rise() -> None:
    """A meaningful rate increase should be hawkish."""

    # Create the Fed-rate engine.
    engine = FedRateIntelligence()

    # 4.00 -> 4.10 equals +10 basis points.
    observation = make_rate(
        4.10,
        4.00,
        MacroDirection.RISING,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify hawkish classification.
    assert result.level is FedRateLevel.HAWKISH

    # Verify movement.
    assert result.change_basis_points == pytest.approx(10.0)


def test_strong_dovish_25_basis_point_cut() -> None:
    """A 25-basis-point cut should be strongly dovish."""

    # Create the Fed-rate engine.
    engine = FedRateIntelligence()

    # 4.25 -> 4.00 equals -25 basis points.
    observation = make_rate(
        4.00,
        4.25,
        MacroDirection.FALLING,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify strong dovish classification.
    assert result.level is FedRateLevel.STRONG_DOVISH

    # Verify movement.
    assert result.change_basis_points == pytest.approx(-25.0)


def test_dovish_10_basis_point_cut() -> None:
    """A meaningful rate decrease should be dovish."""

    # Create the Fed-rate engine.
    engine = FedRateIntelligence()

    # 4.25 -> 4.15 equals -10 basis points.
    observation = make_rate(
        4.15,
        4.25,
        MacroDirection.FALLING,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify dovish classification.
    assert result.level is FedRateLevel.DOVISH

    # Verify movement.
    assert result.change_basis_points == pytest.approx(-10.0)


def test_small_rate_change_is_neutral() -> None:
    """Movement below 5 basis points should be neutral."""

    # Create the Fed-rate engine.
    engine = FedRateIntelligence()

    # 4.00 -> 4.02 equals +2 basis points.
    observation = make_rate(
        4.02,
        4.00,
        MacroDirection.RISING,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify neutral classification.
    assert result.level is FedRateLevel.NEUTRAL


def test_stable_rate_is_neutral() -> None:
    """An unchanged Fed rate should be neutral."""

    # Create the Fed-rate engine.
    engine = FedRateIntelligence()

    # Rate remains unchanged.
    observation = make_rate(
        4.00,
        4.00,
        MacroDirection.STABLE,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify neutral classification.
    assert result.level is FedRateLevel.NEUTRAL

    # Verify zero movement.
    assert result.change_basis_points == 0.0


def test_unknown_direction_is_unknown() -> None:
    """Unknown direction must not create a policy signal."""

    # Create the Fed-rate engine.
    engine = FedRateIntelligence()

    # Create unknown-direction observation.
    observation = make_rate(
        4.00,
        3.75,
        MacroDirection.UNKNOWN,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify unknown classification.
    assert result.level is FedRateLevel.UNKNOWN

    # Verify insufficient data.
    assert result.sufficient_data is False

    # Verify zero confidence.
    assert result.confidence == 0.0


def test_direction_without_previous_is_lower_confidence() -> None:
    """Direction alone should produce a lower-confidence assessment."""

    # Create the Fed-rate engine.
    engine = FedRateIntelligence()

    # Previous rate is unavailable.
    observation = make_rate(
        4.00,
        None,
        MacroDirection.RISING,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Direction alone should be hawkish.
    assert result.level is FedRateLevel.HAWKISH

    # No basis-point movement is available.
    assert result.change_basis_points is None

    # Confidence should be reduced.
    assert result.confidence == 50.0


def test_forecast_surprise_is_exposed() -> None:
    """Forecast surprise should be preserved in the assessment."""

    # Create the Fed-rate engine.
    engine = FedRateIntelligence()

    # Actual rate is 25 basis points above forecast.
    observation = make_rate(
        4.25,
        4.00,
        MacroDirection.RISING,
        forecast=4.00,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify forecast.
    assert result.forecast == pytest.approx(4.00)

    # Verify surprise.
    assert result.surprise == pytest.approx(0.25)


def test_negative_forecast_surprise_is_preserved() -> None:
    """A rate below forecast should produce negative surprise."""

    # Create the Fed-rate engine.
    engine = FedRateIntelligence()

    # Actual rate is below forecast.
    observation = make_rate(
        4.00,
        4.25,
        MacroDirection.FALLING,
        forecast=4.25,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify negative surprise.
    assert result.surprise == pytest.approx(-0.25)


def test_future_observation_is_ignored() -> None:
    """Future Fed-rate observations must never affect historical analysis."""

    # Create the Fed-rate engine.
    engine = FedRateIntelligence()

    # Create a future rate observation.
    future = make_rate(
        5.00,
        4.00,
        MacroDirection.RISING,
        minutes=10,
    )

    # Analyze before that observation exists.
    result = engine.analyze(
        [future],
        BASE_TIME,
    )

    # Future information must be ignored.
    assert result.level is FedRateLevel.UNKNOWN
    assert result.sufficient_data is False


def test_exact_timestamp_is_allowed() -> None:
    """An observation exactly at decision time is valid."""

    # Create the Fed-rate engine.
    engine = FedRateIntelligence()

    # Observation timestamp equals decision timestamp.
    observation = make_rate(
        4.25,
        4.00,
        MacroDirection.RISING,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify exact timestamp is included.
    assert result.level is FedRateLevel.STRONG_HAWKISH


def test_latest_historical_observation_is_selected() -> None:
    """The latest available observation must be selected."""

    # Create the Fed-rate engine.
    engine = FedRateIntelligence()

    # Older observation shows a rate hike.
    older = make_rate(
        4.25,
        4.00,
        MacroDirection.RISING,
        minutes=-10,
    )

    # Newer observation shows a rate cut.
    newer = make_rate(
        4.00,
        4.25,
        MacroDirection.FALLING,
        minutes=-5,
    )

    # Analyze both.
    result = engine.analyze(
        [older, newer],
        BASE_TIME,
    )

    # Newer observation must win.
    assert result.value == pytest.approx(4.00)

    # Newer policy direction must win.
    assert result.level is FedRateLevel.STRONG_DOVISH


def test_zero_previous_rate_does_not_divide_by_zero() -> None:
    """Zero previous rate must not cause division by zero."""

    # Create the Fed-rate engine.
    engine = FedRateIntelligence()

    # Previous value is zero.
    observation = make_rate(
        4.00,
        0.00,
        MacroDirection.RISING,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Basis-point movement remains calculable.
    assert result.change_basis_points == pytest.approx(400.0)

    # Percentage change is unavailable.
    assert result.percentage_change is None


def test_timezone_mismatch_is_rejected() -> None:
    """Naive and timezone-aware timestamps cannot be mixed."""

    # Create the Fed-rate engine.
    engine = FedRateIntelligence()

    # Create a naive observation.
    observation = MacroObservation(
        timestamp=datetime(
            2026,
            1,
            1,
            12,
            0,
        ),
        indicator=MacroIndicator.FED_FUNDS_RATE,
        value=4.25,
        previous=4.00,
        forecast=None,
        source="test",
        direction=MacroDirection.RISING,
    )

    # Mixed timestamp semantics must raise an error.
    with pytest.raises(FedRateIntelligenceError):
        engine.analyze(
            [observation],
            BASE_TIME,
        )


def test_no_observations_returns_unknown() -> None:
    """No Fed observations should produce UNKNOWN."""

    # Create the Fed-rate engine.
    engine = FedRateIntelligence()

    # Analyze an empty collection.
    result = engine.analyze(
        [],
        BASE_TIME,
    )

    # Verify unknown result.
    assert result.level is FedRateLevel.UNKNOWN
    assert result.value is None
    assert result.confidence == 0.0


def test_non_macro_observation_is_rejected() -> None:
    """Invalid observation types must be rejected."""

    # Create the Fed-rate engine.
    engine = FedRateIntelligence()

    # Pass an invalid object.
    with pytest.raises(FedRateIntelligenceError):
        engine.analyze(
            [object()],
            BASE_TIME,
        )


def test_invalid_observation_collection_is_rejected() -> None:
    """Observation collection must be list or tuple."""

    # Create the Fed-rate engine.
    engine = FedRateIntelligence()

    # Pass an invalid collection type.
    with pytest.raises(FedRateIntelligenceError):
        engine.analyze(
            "invalid",
            BASE_TIME,
        )


def test_invalid_thresholds_are_rejected() -> None:
    """Invalid threshold configuration must raise errors."""

    # Negative significant threshold is invalid.
    with pytest.raises(FedRateIntelligenceError):
        FedRateIntelligence(
            significant_change_bps=-1.0,
        )

    # Negative strong threshold is invalid.
    with pytest.raises(FedRateIntelligenceError):
        FedRateIntelligence(
            strong_change_bps=-1.0,
        )

    # Strong threshold cannot be below significant threshold.
    with pytest.raises(FedRateIntelligenceError):
        FedRateIntelligence(
            significant_change_bps=30.0,
            strong_change_bps=25.0,
        )


def test_xauusd_wrapper_returns_macro_assessment_only() -> None:
    """XAUUSD wrapper must not produce a trade decision."""

    # Create the Fed-rate engine.
    engine = FedRateIntelligence()

    # Create a rate-hike observation.
    observation = make_rate(
        4.25,
        4.00,
        MacroDirection.RISING,
    )

    # Analyze through the XAUUSD wrapper.
    result = engine.analyze_xauusd(
        [observation],
        BASE_TIME,
    )

    # Verify that this is still Fed-rate intelligence.
    assert result.indicator is MacroIndicator.FED_FUNDS_RATE

    # Verify the policy classification.
    assert result.level is FedRateLevel.STRONG_HAWKISH