"""
Tests for deterministic employment intelligence.

The tests cover:
- NFP
- unemployment rate
- previous-value changes
- forecast surprises
- indicator-specific semantics
- historical/no-lookahead safety
- threshold boundaries
- serialization
- validation
"""

from datetime import datetime, timezone

import pytest

from app.trading.macro.employment_intelligence import (
    EmploymentIntelligence,
    EmploymentIntelligenceError,
    EmploymentLevel,
    EmploymentSurpriseLevel,
)
from app.trading.macro.macro_observation import (
    MacroDirection,
    MacroIndicator,
    MacroObservation,
)


# Create a reusable timezone-aware decision timestamp.
DECISION_TIME = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)


def make_observation(
    *,
    timestamp: datetime,
    indicator: MacroIndicator,
    value: float,
    previous: float | None = None,
    forecast: float | None = None,
    direction: MacroDirection = MacroDirection.UNKNOWN,
) -> MacroObservation:
    """Create a MacroObservation for tests."""

    # Return the existing production observation model.
    return MacroObservation(
        timestamp=timestamp,
        indicator=indicator,
        value=value,
        previous=previous,
        forecast=forecast,
        source="test",
        direction=direction,
    )


def test_nfp_strong_rising_is_strong_hot() -> None:
    """Large positive NFP movement should classify as strongly hot."""

    # Create a large positive NFP change.
    observations = [
        make_observation(
            timestamp=datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc),
            indicator=MacroIndicator.NFP,
            value=250.0,
            previous=150.0,
            forecast=180.0,
        )
    ]

    # Analyze NFP.
    result = EmploymentIntelligence().analyze_nfp(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify the expected classification.
    assert result.level == EmploymentLevel.STRONG_HOT

    # Verify the raw and employment directions.
    assert result.direction == MacroDirection.RISING
    assert result.employment_direction == MacroDirection.RISING

    # Verify the calculated change.
    assert result.change_from_previous == 100.0


def test_nfp_moderately_rising_is_hot() -> None:
    """Moderate positive NFP movement should classify as hot."""

    # Create a moderate positive NFP change.
    observations = [
        make_observation(
            timestamp=datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc),
            indicator=MacroIndicator.NFP,
            value=190.0,
            previous=150.0,
        )
    ]

    # Analyze NFP.
    result = EmploymentIntelligence().analyze_nfp(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify hot classification.
    assert result.level == EmploymentLevel.HOT


def test_nfp_small_change_is_neutral() -> None:
    """Small NFP movement should remain neutral."""

    # Create a small positive NFP change.
    observations = [
        make_observation(
            timestamp=datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc),
            indicator=MacroIndicator.NFP,
            value=160.0,
            previous=150.0,
        )
    ]

    # Analyze NFP.
    result = EmploymentIntelligence().analyze_nfp(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify neutral classification.
    assert result.level == EmploymentLevel.NEUTRAL


def test_nfp_strong_falling_is_strong_cooling() -> None:
    """Large negative NFP movement should classify as strongly cooling."""

    # Create a large negative NFP change.
    observations = [
        make_observation(
            timestamp=datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc),
            indicator=MacroIndicator.NFP,
            value=50.0,
            previous=150.0,
        )
    ]

    # Analyze NFP.
    result = EmploymentIntelligence().analyze_nfp(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify strong cooling.
    assert result.level == EmploymentLevel.STRONG_COOLING
    assert result.employment_direction == MacroDirection.FALLING


def test_nfp_upside_surprise() -> None:
    """NFP above forecast should be an upside surprise."""

    # Create NFP above forecast.
    observations = [
        make_observation(
            timestamp=datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc),
            indicator=MacroIndicator.NFP,
            value=200.0,
            previous=150.0,
            forecast=160.0,
        )
    ]

    # Analyze NFP.
    result = EmploymentIntelligence().analyze_nfp(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify the positive surprise.
    assert result.surprise == 40.0
    assert result.surprise_level == EmploymentSurpriseLevel.UPSIDE


def test_nfp_strong_upside_surprise() -> None:
    """Very large positive NFP surprise should be strongly upside."""

    # Create a large positive forecast surprise.
    observations = [
        make_observation(
            timestamp=datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc),
            indicator=MacroIndicator.NFP,
            value=250.0,
            previous=150.0,
            forecast=160.0,
        )
    ]

    # Analyze NFP.
    result = EmploymentIntelligence().analyze_nfp(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify strong upside.
    assert result.surprise_level == EmploymentSurpriseLevel.STRONG_UPSIDE


def test_nfp_downside_surprise() -> None:
    """NFP below forecast should be a downside surprise."""

    # Create NFP below forecast.
    observations = [
        make_observation(
            timestamp=datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc),
            indicator=MacroIndicator.NFP,
            value=120.0,
            previous=150.0,
            forecast=160.0,
        )
    ]

    # Analyze NFP.
    result = EmploymentIntelligence().analyze_nfp(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify downside classification.
    assert result.surprise == -40.0
    assert result.surprise_level == EmploymentSurpriseLevel.DOWNSIDE


def test_unemployment_falling_is_hot() -> None:
    """Falling unemployment means stronger employment."""

    # Create falling unemployment.
    observations = [
        make_observation(
            timestamp=datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc),
            indicator=MacroIndicator.UNEMPLOYMENT_RATE,
            value=3.8,
            previous=4.1,
        )
    ]

    # Analyze unemployment.
    result = EmploymentIntelligence().analyze_unemployment_rate(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify the inverse employment interpretation.
    assert result.direction == MacroDirection.FALLING
    assert result.employment_direction == MacroDirection.RISING
    assert result.level == EmploymentLevel.STRONG_HOT


def test_unemployment_rising_is_cooling() -> None:
    """Rising unemployment means weaker employment."""

    # Create rising unemployment.
    observations = [
        make_observation(
            timestamp=datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc),
            indicator=MacroIndicator.UNEMPLOYMENT_RATE,
            value=4.4,
            previous=4.1,
        )
    ]

    # Analyze unemployment.
    result = EmploymentIntelligence().analyze_unemployment_rate(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify cooling employment interpretation.
    assert result.direction == MacroDirection.RISING
    assert result.employment_direction == MacroDirection.FALLING
    assert result.level == EmploymentLevel.STRONG_COOLING


def test_unemployment_lower_than_forecast_is_upside() -> None:
    """Lower unemployment than forecast means stronger employment."""

    # Actual unemployment is below forecast.
    observations = [
        make_observation(
            timestamp=datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc),
            indicator=MacroIndicator.UNEMPLOYMENT_RATE,
            value=3.8,
            previous=4.0,
            forecast=4.1,
        )
    ]

    # Analyze unemployment.
    result = EmploymentIntelligence().analyze_unemployment_rate(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Raw surprise is negative because actual is below forecast.
    assert result.surprise == pytest.approx(-0.3)

    # Employment interpretation must invert the surprise.
    assert result.surprise_level == EmploymentSurpriseLevel.STRONG_UPSIDE


def test_unemployment_higher_than_forecast_is_downside() -> None:
    """Higher unemployment than forecast means weaker employment."""

    # Actual unemployment is above forecast.
    observations = [
        make_observation(
            timestamp=datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc),
            indicator=MacroIndicator.UNEMPLOYMENT_RATE,
            value=4.4,
            previous=4.0,
            forecast=4.1,
        )
    ]

    # Analyze unemployment.
    result = EmploymentIntelligence().analyze_unemployment_rate(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify downside classification.
    assert result.surprise_level == EmploymentSurpriseLevel.STRONG_DOWNSIDE


def test_no_previous_uses_direction_with_reduced_confidence() -> None:
    """Direction without previous data should produce 50% confidence."""

    # Provide an observation with explicit direction but no previous value.
    observations = [
        make_observation(
            timestamp=datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc),
            indicator=MacroIndicator.NFP,
            value=200.0,
            direction=MacroDirection.RISING,
        )
    ]

    # Analyze NFP.
    result = EmploymentIntelligence().analyze_nfp(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify reduced confidence.
    assert result.confidence == 50.0

    # Verify directional classification.
    assert result.level == EmploymentLevel.HOT


def test_future_observation_is_ignored() -> None:
    """Future observations must never leak into a historical decision."""

    # Create an older and a future observation.
    observations = [
        make_observation(
            timestamp=datetime(2026, 9, 3, 10, 0, tzinfo=timezone.utc),
            indicator=MacroIndicator.NFP,
            value=150.0,
            previous=140.0,
        ),
        make_observation(
            timestamp=datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc),
            indicator=MacroIndicator.NFP,
            value=500.0,
            previous=100.0,
        ),
    ]

    # Analyze using a decision time before the future observation.
    result = EmploymentIntelligence().analyze_nfp(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # The future value must not be selected.
    assert result.value == 150.0


def test_exact_timestamp_is_allowed() -> None:
    """An observation exactly at decision time is historically available."""

    # Create an observation exactly at decision time.
    observations = [
        make_observation(
            timestamp=DECISION_TIME,
            indicator=MacroIndicator.NFP,
            value=200.0,
            previous=150.0,
        )
    ]

    # Analyze NFP.
    result = EmploymentIntelligence().analyze_nfp(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify exact timestamp inclusion.
    assert result.value == 200.0


def test_latest_historical_observation_is_selected() -> None:
    """The newest eligible observation should be selected."""

    # Create multiple historical observations.
    observations = [
        make_observation(
            timestamp=datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc),
            indicator=MacroIndicator.NFP,
            value=100.0,
        ),
        make_observation(
            timestamp=datetime(2026, 9, 3, 10, 0, tzinfo=timezone.utc),
            indicator=MacroIndicator.NFP,
            value=200.0,
        ),
    ]

    # Analyze NFP.
    result = EmploymentIntelligence().analyze_nfp(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # The latest historical observation must win.
    assert result.value == 200.0


def test_non_employment_indicator_is_rejected() -> None:
    """Unsupported indicators should raise an explicit error."""

    # Create a valid DXY observation.
    observations = [
        make_observation(
            timestamp=datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc),
            indicator=MacroIndicator.DXY,
            value=100.0,
        )
    ]

    # Attempt to analyze an unsupported indicator.
    with pytest.raises(EmploymentIntelligenceError):
        EmploymentIntelligence().analyze(
            observations,
            indicator=MacroIndicator.DXY,
            decision_timestamp=DECISION_TIME,
        )


def test_naive_decision_timestamp_is_rejected() -> None:
    """Naive decision timestamps should be rejected."""

    # Create a valid employment observation.
    observations = [
        make_observation(
            timestamp=datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc),
            indicator=MacroIndicator.NFP,
            value=200.0,
        )
    ]

    # Use a naive decision timestamp.
    with pytest.raises(EmploymentIntelligenceError):
        EmploymentIntelligence().analyze_nfp(
            observations,
            decision_timestamp=datetime(2026, 9, 4, 12, 0),
        )


def test_empty_history_is_rejected() -> None:
    """No historical-safe observation should raise an explicit error."""

    # Attempt analysis without observations.
    with pytest.raises(EmploymentIntelligenceError):
        EmploymentIntelligence().analyze_nfp(
            [],
            decision_timestamp=DECISION_TIME,
        )


def test_analyze_all_returns_both_indicators() -> None:
    """analyze_all should return NFP and unemployment assessments."""

    # Create both employment observations.
    observations = [
        make_observation(
            timestamp=datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc),
            indicator=MacroIndicator.NFP,
            value=200.0,
            previous=150.0,
        ),
        make_observation(
            timestamp=datetime(2026, 9, 4, 10, 1, tzinfo=timezone.utc),
            indicator=MacroIndicator.UNEMPLOYMENT_RATE,
            value=3.9,
            previous=4.1,
        ),
    ]

    # Analyze both indicators.
    result = EmploymentIntelligence().analyze_all(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify both results exist.
    assert MacroIndicator.NFP in result
    assert MacroIndicator.UNEMPLOYMENT_RATE in result


def test_xauusd_wrapper_returns_macro_only() -> None:
    """XAUUSD wrapper should return employment macro data only."""

    # Create both employment observations.
    observations = [
        make_observation(
            timestamp=datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc),
            indicator=MacroIndicator.NFP,
            value=200.0,
            previous=150.0,
        ),
        make_observation(
            timestamp=datetime(2026, 9, 4, 10, 1, tzinfo=timezone.utc),
            indicator=MacroIndicator.UNEMPLOYMENT_RATE,
            value=3.9,
            previous=4.1,
        ),
    ]

    # Run the XAUUSD-specific macro wrapper.
    result = EmploymentIntelligence().analyze_xauusd(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify that only macro assessments are returned.
    assert set(result) == {
        MacroIndicator.NFP,
        MacroIndicator.UNEMPLOYMENT_RATE,
    }


def test_to_dict_serializes_assessment() -> None:
    """Assessment serialization should contain stable fields."""

    # Create a valid NFP observation.
    observations = [
        make_observation(
            timestamp=datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc),
            indicator=MacroIndicator.NFP,
            value=200.0,
            previous=150.0,
            forecast=180.0,
        )
    ]

    # Analyze NFP.
    result = EmploymentIntelligence().analyze_nfp(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Serialize the result.
    data = result.to_dict()

    # Verify important serialized fields.
    assert data["indicator"] == MacroIndicator.NFP.value
    assert data["value"] == 200.0
    assert data["previous"] == 150.0
    assert data["forecast"] == 180.0
    assert data["level"] == EmploymentLevel.HOT.value


def test_equal_thresholds_are_allowed() -> None:
    """Equal significant and strong thresholds are valid configurations."""

    # Create an engine using equal thresholds.
    engine = EmploymentIntelligence(
        significant_nfp_change=50.0,
        strong_nfp_change=50.0,
        significant_unemployment_change=0.2,
        strong_unemployment_change=0.2,
        significant_nfp_surprise=50.0,
        strong_nfp_surprise=50.0,
        significant_unemployment_surprise=0.2,
        strong_unemployment_surprise=0.2,
    )

    # Verify construction succeeded.
    assert engine.significant_nfp_change == 50.0


def test_zero_threshold_is_rejected() -> None:
    """Zero thresholds should not be accepted."""

    # Attempt to create an invalid engine.
    with pytest.raises(EmploymentIntelligenceError):
        EmploymentIntelligence(
            significant_nfp_change=0.0,
        )


def test_boolean_threshold_is_rejected() -> None:
    """Boolean thresholds should not be accepted."""

    # Attempt to create an engine with a boolean threshold.
    with pytest.raises(EmploymentIntelligenceError):
        EmploymentIntelligence(
            significant_nfp_change=True,
        )