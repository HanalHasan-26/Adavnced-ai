# tests/trading/test_treasury_yield_intelligence.py

"""Tests for deterministic Treasury-yield intelligence."""

# Import datetime utilities.
from datetime import datetime, timedelta, timezone

# Import pytest for exception and approximate-value assertions.
import pytest

# Import the existing macro models.
from app.trading.macro import (
    MacroDirection,
    MacroIndicator,
    MacroObservation,
)

# Import Treasury-yield intelligence.
from app.trading.macro import (
    TreasuryYieldIntelligence,
    TreasuryYieldIntelligenceError,
    TreasuryYieldLevel,
)


# Use a deterministic timezone-aware base timestamp.
BASE_TIME = datetime(
    2026,
    1,
    1,
    12,
    0,
    tzinfo=timezone.utc,
)


def make_yield(
    indicator: MacroIndicator,
    value: float,
    previous: float | None,
    direction: MacroDirection,
    minutes: int = 0,
) -> MacroObservation:
    """Create a deterministic Treasury-yield observation."""

    # Calculate the observation timestamp relative to BASE_TIME.
    timestamp = BASE_TIME + timedelta(
        minutes=minutes,
    )

    # Return the existing MacroObservation model.
    return MacroObservation(
        timestamp=timestamp,
        indicator=indicator,
        value=value,
        previous=previous,
        forecast=None,
        source="test",
        direction=direction,
    )


def test_strong_rising_10y() -> None:
    """A 5+ basis-point rise should be strongly rising."""

    # Create the intelligence engine.
    engine = TreasuryYieldIntelligence()

    # 4.20 -> 4.25 represents a 5-basis-point rise.
    observation = make_yield(
        MacroIndicator.US_10Y_YIELD,
        4.25,
        4.20,
        MacroDirection.RISING,
    )

    # Analyze the historical observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify the strong-rising classification.
    assert result.level is TreasuryYieldLevel.STRONG_RISING

    # Verify the basis-point calculation.
    assert result.change_basis_points == pytest.approx(5.0)

    # Verify full confidence.
    assert result.confidence == 100.0


def test_rising_10y() -> None:
    """A meaningful yield rise should be classified as rising."""

    # Create the intelligence engine.
    engine = TreasuryYieldIntelligence()

    # 4.20 -> 4.23 represents a 3-basis-point rise.
    observation = make_yield(
        MacroIndicator.US_10Y_YIELD,
        4.23,
        4.20,
        MacroDirection.RISING,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify the rising classification.
    assert result.level is TreasuryYieldLevel.RISING

    # Verify the basis-point movement.
    assert result.change_basis_points == pytest.approx(3.0)


def test_strong_falling_10y() -> None:
    """A 5+ basis-point fall should be strongly falling."""

    # Create the intelligence engine.
    engine = TreasuryYieldIntelligence()

    # 4.20 -> 4.15 represents a 5-basis-point fall.
    observation = make_yield(
        MacroIndicator.US_10Y_YIELD,
        4.15,
        4.20,
        MacroDirection.FALLING,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify the strong-falling classification.
    assert result.level is TreasuryYieldLevel.STRONG_FALLING

    # Verify the basis-point movement.
    assert result.change_basis_points == pytest.approx(-5.0)


def test_falling_10y() -> None:
    """A meaningful yield fall should be classified as falling."""

    # Create the intelligence engine.
    engine = TreasuryYieldIntelligence()

    # 4.20 -> 4.17 represents a 3-basis-point fall.
    observation = make_yield(
        MacroIndicator.US_10Y_YIELD,
        4.17,
        4.20,
        MacroDirection.FALLING,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify the falling classification.
    assert result.level is TreasuryYieldLevel.FALLING

    # Verify the basis-point movement.
    assert result.change_basis_points == pytest.approx(-3.0)


def test_small_move_is_stable() -> None:
    """Movement below the significant threshold is stable."""

    # Create the intelligence engine.
    engine = TreasuryYieldIntelligence()

    # 4.20 -> 4.21 represents a 1-basis-point rise.
    observation = make_yield(
        MacroIndicator.US_10Y_YIELD,
        4.21,
        4.20,
        MacroDirection.RISING,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify the stable classification.
    assert result.level is TreasuryYieldLevel.STABLE

    # Verify the one-basis-point movement.
    assert result.change_basis_points == pytest.approx(1.0)


def test_stable_direction_is_stable() -> None:
    """Stable direction should produce STABLE."""

    # Create the intelligence engine.
    engine = TreasuryYieldIntelligence()

    # Create an unchanged yield observation.
    observation = make_yield(
        MacroIndicator.US_10Y_YIELD,
        4.20,
        4.20,
        MacroDirection.STABLE,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify stable classification.
    assert result.level is TreasuryYieldLevel.STABLE

    # Verify zero movement.
    assert result.change_basis_points == 0.0


def test_unknown_direction_returns_unknown() -> None:
    """Unknown direction must not produce a false yield signal."""

    # Create the intelligence engine.
    engine = TreasuryYieldIntelligence()

    # Create an unknown-direction observation.
    observation = make_yield(
        MacroIndicator.US_10Y_YIELD,
        4.20,
        4.10,
        MacroDirection.UNKNOWN,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify unknown classification.
    assert result.level is TreasuryYieldLevel.UNKNOWN

    # Verify insufficient data.
    assert result.sufficient_data is False

    # Verify zero confidence.
    assert result.confidence == 0.0


def test_no_previous_uses_direction() -> None:
    """Direction can still be used when previous yield is unavailable."""

    # Create the intelligence engine.
    engine = TreasuryYieldIntelligence()

    # Previous yield is unavailable.
    observation = make_yield(
        MacroIndicator.US_10Y_YIELD,
        4.20,
        None,
        MacroDirection.RISING,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Direction alone should produce RISING.
    assert result.level is TreasuryYieldLevel.RISING

    # No basis-point movement can be calculated.
    assert result.change_basis_points is None

    # Direction-only confidence is reduced.
    assert result.confidence == 50.0


def test_future_observation_is_ignored() -> None:
    """Future Treasury data must never influence historical analysis."""

    # Create the intelligence engine.
    engine = TreasuryYieldIntelligence()

    # Create an observation ten minutes in the future.
    future = make_yield(
        MacroIndicator.US_10Y_YIELD,
        5.00,
        4.00,
        MacroDirection.RISING,
        minutes=10,
    )

    # Analyze at BASE_TIME.
    result = engine.analyze(
        [future],
        BASE_TIME,
    )

    # Future information must be ignored.
    assert result.level is TreasuryYieldLevel.UNKNOWN
    assert result.sufficient_data is False


def test_exact_timestamp_is_allowed() -> None:
    """Observation exactly at decision time is valid."""

    # Create the intelligence engine.
    engine = TreasuryYieldIntelligence()

    # Observation occurs exactly at the decision timestamp.
    observation = make_yield(
        MacroIndicator.US_10Y_YIELD,
        4.25,
        4.20,
        MacroDirection.RISING,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Exact timestamp must be included.
    assert result.level is TreasuryYieldLevel.STRONG_RISING


def test_latest_observation_is_used() -> None:
    """The latest historical-safe observation must be selected."""

    # Create the intelligence engine.
    engine = TreasuryYieldIntelligence()

    # Create an older rising observation.
    older = make_yield(
        MacroIndicator.US_10Y_YIELD,
        4.25,
        4.20,
        MacroDirection.RISING,
        minutes=-10,
    )

    # Create a newer falling observation.
    newer = make_yield(
        MacroIndicator.US_10Y_YIELD,
        4.15,
        4.20,
        MacroDirection.FALLING,
        minutes=-5,
    )

    # Analyze both observations.
    result = engine.analyze(
        [older, newer],
        BASE_TIME,
    )

    # The newer observation must be selected.
    assert result.level is TreasuryYieldLevel.STRONG_FALLING

    # Verify the selected yield value.
    assert result.value == 4.15


def test_non_requested_maturity_is_ignored() -> None:
    """A different maturity must not affect the requested maturity."""

    # Create the intelligence engine.
    engine = TreasuryYieldIntelligence()

    # Supply only 2Y data.
    observation = make_yield(
        MacroIndicator.US_2Y_YIELD,
        4.50,
        4.40,
        MacroDirection.RISING,
    )

    # Request 10Y analysis.
    result = engine.analyze(
        [observation],
        BASE_TIME,
        indicator=MacroIndicator.US_10Y_YIELD,
    )

    # No 10Y data exists.
    assert result.level is TreasuryYieldLevel.UNKNOWN


def test_all_maturities_are_analyzed() -> None:
    """analyze_all should analyze all four Treasury maturities."""

    # Create the intelligence engine.
    engine = TreasuryYieldIntelligence()

    # Create one observation for each supported maturity.
    observations = [
        make_yield(
            MacroIndicator.US_2Y_YIELD,
            4.50,
            4.45,
            MacroDirection.RISING,
        ),
        make_yield(
            MacroIndicator.US_5Y_YIELD,
            4.30,
            4.25,
            MacroDirection.RISING,
        ),
        make_yield(
            MacroIndicator.US_10Y_YIELD,
            4.20,
            4.15,
            MacroDirection.RISING,
        ),
        make_yield(
            MacroIndicator.US_30Y_YIELD,
            4.40,
            4.35,
            MacroDirection.RISING,
        ),
    ]

    # Analyze all maturities.
    results = engine.analyze_all(
        observations,
        BASE_TIME,
    )

    # Four maturity results should exist.
    assert len(results) == 4

    # Every supported maturity must be represented.
    for indicator in engine.TREASURY_INDICATORS:
        assert indicator in results

        # Every supplied observation has usable direction.
        assert results[indicator].sufficient_data is True


def test_zero_previous_value_does_not_divide_by_zero() -> None:
    """Zero previous yield must not cause division by zero."""

    # Create the intelligence engine.
    engine = TreasuryYieldIntelligence()

    # Previous yield is zero.
    observation = make_yield(
        MacroIndicator.US_10Y_YIELD,
        4.20,
        0.0,
        MacroDirection.RISING,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Basis-point movement remains calculable.
    assert result.change_basis_points == pytest.approx(420.0)

    # Percentage movement is unavailable because the denominator is zero.
    assert result.percentage_change is None


def test_timezone_mismatch_is_rejected() -> None:
    """Naive and timezone-aware timestamps cannot be mixed."""

    # Create the intelligence engine.
    engine = TreasuryYieldIntelligence()

    # Create a naive timestamp.
    observation = MacroObservation(
        timestamp=datetime(
            2026,
            1,
            1,
            12,
            0,
        ),
        indicator=MacroIndicator.US_10Y_YIELD,
        value=4.25,
        previous=4.20,
        forecast=None,
        source="test",
        direction=MacroDirection.RISING,
    )

    # Mixed timezone semantics must raise an error.
    with pytest.raises(TreasuryYieldIntelligenceError):
        engine.analyze(
            [observation],
            BASE_TIME,
        )


def test_invalid_indicator_is_rejected() -> None:
    """Non-Treasury indicators cannot be requested."""

    # Create the intelligence engine.
    engine = TreasuryYieldIntelligence()

    # DXY is not a Treasury maturity.
    with pytest.raises(TreasuryYieldIntelligenceError):
        engine.analyze(
            [],
            BASE_TIME,
            indicator=MacroIndicator.DXY,
        )


def test_invalid_thresholds_are_rejected() -> None:
    """Invalid movement thresholds must raise errors."""

    # Significant threshold cannot be negative.
    with pytest.raises(TreasuryYieldIntelligenceError):
        TreasuryYieldIntelligence(
            significant_change_bps=-1.0,
        )

    # Strong threshold cannot be negative.
    with pytest.raises(TreasuryYieldIntelligenceError):
        TreasuryYieldIntelligence(
            strong_change_bps=-1.0,
        )

    # Strong threshold cannot be below significant threshold.
    with pytest.raises(TreasuryYieldIntelligenceError):
        TreasuryYieldIntelligence(
            significant_change_bps=5.0,
            strong_change_bps=2.0,
        )


def test_xauusd_wrapper_returns_yield_assessment_only() -> None:
    """XAUUSD wrapper must return yield intelligence, not a trade signal."""

    # Create the intelligence engine.
    engine = TreasuryYieldIntelligence()

    # Create rising 10Y yield data.
    observation = make_yield(
        MacroIndicator.US_10Y_YIELD,
        4.25,
        4.20,
        MacroDirection.RISING,
    )

    # Analyze through the XAUUSD wrapper.
    result = engine.analyze_xauusd(
        [observation],
        BASE_TIME,
    )

    # Verify that this remains a Treasury-yield assessment.
    assert result.indicator is MacroIndicator.US_10Y_YIELD

    # Verify the corrected strong-rising boundary.
    assert result.level is TreasuryYieldLevel.STRONG_RISING