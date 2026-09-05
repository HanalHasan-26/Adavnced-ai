# tests/trading/test_usd_strength_intelligence.py

"""Tests for deterministic USD-strength intelligence."""

# Import datetime utilities for deterministic timestamps.
from datetime import datetime, timedelta, timezone

# Import pytest for exception assertions.
import pytest

# Import macro observation models.
from app.trading.macro import (
    MacroDirection,
    MacroIndicator,
    MacroObservation,
)

# Import USD-strength intelligence models.
from app.trading.macro import (
    USDStrengthIntelligence,
    USDStrengthIntelligenceError,
    USDStrengthLevel,
)


# Define a fixed UTC timestamp for deterministic tests.
BASE_TIME = datetime(
    2026,
    1,
    1,
    12,
    0,
    tzinfo=timezone.utc,
)


def make_observation(
    indicator: MacroIndicator,
    direction: MacroDirection,
    minutes: int = 0,
) -> MacroObservation:
    """Create a deterministic test observation."""

    # Calculate the timestamp using timedelta so negative and positive
    # offsets work correctly across hour/day boundaries.
    timestamp = BASE_TIME + timedelta(
        minutes=minutes,
    )

    # Return a valid macro observation.
    return MacroObservation(
        timestamp=timestamp,
        indicator=indicator,
        value=100.0,
        previous=99.0,
        forecast=99.5,
        source="test",
        direction=direction,
    )


def test_strong_usd() -> None:
    """Multiple rising USD indicators should produce strong USD."""

    # Create the intelligence engine.
    engine = USDStrengthIntelligence()

    # Provide several rising USD indicators.
    observations = [
        make_observation(
            MacroIndicator.DXY,
            MacroDirection.RISING,
        ),
        make_observation(
            MacroIndicator.US_2Y_YIELD,
            MacroDirection.RISING,
        ),
        make_observation(
            MacroIndicator.US_10Y_YIELD,
            MacroDirection.RISING,
        ),
        make_observation(
            MacroIndicator.FED_FUNDS_RATE,
            MacroDirection.RISING,
        ),
    ]

    # Analyze the observations.
    result = engine.analyze(
        observations,
        BASE_TIME,
    )

    # Verify strong USD classification.
    assert result.level is USDStrengthLevel.STRONG

    # Verify positive score.
    assert result.score > 60.0

    # Verify sufficient data.
    assert result.sufficient_data is True


def test_weak_usd() -> None:
    """Multiple falling USD indicators should produce weak USD."""

    # Create the intelligence engine.
    engine = USDStrengthIntelligence()

    # Provide several falling USD indicators.
    observations = [
        make_observation(
            MacroIndicator.DXY,
            MacroDirection.FALLING,
        ),
        make_observation(
            MacroIndicator.US_2Y_YIELD,
            MacroDirection.FALLING,
        ),
        make_observation(
            MacroIndicator.US_10Y_YIELD,
            MacroDirection.FALLING,
        ),
        make_observation(
            MacroIndicator.FED_FUNDS_RATE,
            MacroDirection.FALLING,
        ),
    ]

    # Analyze the observations.
    result = engine.analyze(
        observations,
        BASE_TIME,
    )

    # Verify weak USD classification.
    assert result.level is USDStrengthLevel.WEAK

    # Verify negative score.
    assert result.score < -60.0

    # Verify sufficient data.
    assert result.sufficient_data is True


def test_mixed_indicators_can_produce_neutral() -> None:
    """Balanced USD inputs should produce a neutral result."""

    # Use equal custom weights for mathematical clarity.
    engine = USDStrengthIntelligence(
        weights={
            MacroIndicator.DXY: 1.0,
            MacroIndicator.US_2Y_YIELD: 1.0,
        },
    )

    # One positive and one negative input should cancel.
    observations = [
        make_observation(
            MacroIndicator.DXY,
            MacroDirection.RISING,
        ),
        make_observation(
            MacroIndicator.US_2Y_YIELD,
            MacroDirection.FALLING,
        ),
    ]

    # Analyze the observations.
    result = engine.analyze(
        observations,
        BASE_TIME,
    )

    # Verify neutral classification.
    assert result.level is USDStrengthLevel.NEUTRAL

    # Verify exact zero after floating-point normalization.
    assert result.score == 0.0


def test_no_data_returns_unknown() -> None:
    """No observations should produce UNKNOWN."""

    # Create the intelligence engine.
    engine = USDStrengthIntelligence()

    # Analyze an empty observation set.
    result = engine.analyze(
        [],
        BASE_TIME,
    )

    # Verify unknown classification.
    assert result.level is USDStrengthLevel.UNKNOWN

    # Verify zero confidence.
    assert result.confidence == 0.0

    # Verify insufficient data.
    assert result.sufficient_data is False

    # Verify no indicators were used.
    assert result.indicators_used == 0


def test_future_observations_are_ignored() -> None:
    """Future observations must never affect historical analysis."""

    # Create the intelligence engine.
    engine = USDStrengthIntelligence()

    # Create an observation ten minutes in the future.
    future = make_observation(
        MacroIndicator.DXY,
        MacroDirection.RISING,
        minutes=10,
    )

    # Analyze at the earlier decision timestamp.
    result = engine.analyze(
        [future],
        BASE_TIME,
    )

    # The future observation must be ignored.
    assert result.level is USDStrengthLevel.UNKNOWN

    # Verify that no indicator was used.
    assert result.indicators_used == 0


def test_latest_observation_is_used() -> None:
    """Only the latest available observation per indicator is used."""

    # Configure only DXY for an unambiguous result.
    engine = USDStrengthIntelligence(
        weights={
            MacroIndicator.DXY: 1.0,
        },
    )

    # Create an older bullish observation.
    older = make_observation(
        MacroIndicator.DXY,
        MacroDirection.RISING,
        minutes=-10,
    )

    # Create a newer bearish observation.
    newer = make_observation(
        MacroIndicator.DXY,
        MacroDirection.FALLING,
        minutes=-5,
    )

    # Analyze both observations.
    result = engine.analyze(
        [older, newer],
        BASE_TIME,
    )

    # The newer observation must determine the result.
    assert result.score == -100.0

    # Verify the newer direction was selected.
    assert (
        result.contributions[0].direction
        is MacroDirection.FALLING
    )


def test_exact_decision_timestamp_is_allowed() -> None:
    """An observation exactly at decision time is usable."""

    # Configure only DXY.
    engine = USDStrengthIntelligence(
        weights={
            MacroIndicator.DXY: 1.0,
        },
    )

    # Create an observation exactly at decision time.
    observation = make_observation(
        MacroIndicator.DXY,
        MacroDirection.RISING,
    )

    # Analyze at the exact same timestamp.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify that equality is allowed.
    assert result.score == 100.0
    assert result.indicators_used == 1


def test_unknown_direction_does_not_add_directional_bias() -> None:
    """UNKNOWN direction must not create a false USD signal."""

    # Configure only DXY.
    engine = USDStrengthIntelligence(
        weights={
            MacroIndicator.DXY: 1.0,
        },
    )

    # Create an unknown-direction observation.
    observation = make_observation(
        MacroIndicator.DXY,
        MacroDirection.UNKNOWN,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify no directional contribution.
    assert result.score == 0.0

    # The observation still counts toward data coverage.
    assert result.indicators_used == 1

    # With sufficient coverage, the result is neutral.
    assert result.level is USDStrengthLevel.NEUTRAL


def test_confidence_represents_weighted_coverage() -> None:
    """Confidence should represent available weighted coverage."""

    # Configure two equally weighted indicators.
    engine = USDStrengthIntelligence(
        weights={
            MacroIndicator.DXY: 1.0,
            MacroIndicator.US_2Y_YIELD: 1.0,
        },
    )

    # Provide only one indicator.
    result = engine.analyze(
        [
            make_observation(
                MacroIndicator.DXY,
                MacroDirection.RISING,
            ),
        ],
        BASE_TIME,
    )

    # Half the configured weight is available.
    assert result.confidence == 50.0

    # The default minimum coverage is 50%.
    assert result.sufficient_data is True


def test_low_coverage_returns_unknown() -> None:
    """Coverage below the configured minimum must return UNKNOWN."""

    # Require complete indicator coverage.
    engine = USDStrengthIntelligence(
        weights={
            MacroIndicator.DXY: 1.0,
            MacroIndicator.US_2Y_YIELD: 1.0,
        },
        min_coverage=1.0,
    )

    # Supply only one indicator.
    result = engine.analyze(
        [
            make_observation(
                MacroIndicator.DXY,
                MacroDirection.RISING,
            ),
        ],
        BASE_TIME,
    )

    # Classification must be unknown.
    assert result.level is USDStrengthLevel.UNKNOWN

    # Confidence still reports actual coverage.
    assert result.confidence == 50.0

    # Data is insufficient.
    assert result.sufficient_data is False


def test_irrelevant_indicators_are_ignored() -> None:
    """Indicators outside the configured model must be ignored."""

    # Configure only DXY.
    engine = USDStrengthIntelligence(
        weights={
            MacroIndicator.DXY: 1.0,
        },
    )

    # Supply an unrelated inflation observation.
    result = engine.analyze(
        [
            make_observation(
                MacroIndicator.CPI,
                MacroDirection.RISING,
            ),
        ],
        BASE_TIME,
    )

    # No configured USD indicator was available.
    assert result.level is USDStrengthLevel.UNKNOWN

    # Verify nothing was used.
    assert result.indicators_used == 0


def test_timezone_awareness_mismatch_is_rejected() -> None:
    """Naive and timezone-aware timestamps must not be mixed."""

    # Create the intelligence engine.
    engine = USDStrengthIntelligence()

    # Create a naive observation.
    observation = MacroObservation(
        timestamp=datetime(
            2026,
            1,
            1,
            12,
            0,
        ),
        indicator=MacroIndicator.DXY,
        value=100.0,
        previous=99.0,
        forecast=99.5,
        source="test",
        direction=MacroDirection.RISING,
    )

    # Mixed timezone semantics must raise an error.
    with pytest.raises(USDStrengthIntelligenceError):
        engine.analyze(
            [observation],
            BASE_TIME,
        )


def test_invalid_empty_weights_are_rejected() -> None:
    """An empty weight configuration is invalid."""

    # An engine without configured indicators cannot operate.
    with pytest.raises(USDStrengthIntelligenceError):
        USDStrengthIntelligence(
            weights={},
        )


def test_invalid_weight_is_rejected() -> None:
    """Zero weights are invalid."""

    # A zero-weight indicator is not meaningful.
    with pytest.raises(USDStrengthIntelligenceError):
        USDStrengthIntelligence(
            weights={
                MacroIndicator.DXY: 0.0,
            },
        )


def test_invalid_thresholds_are_rejected() -> None:
    """Invalid threshold signs must be rejected."""

    # Strong threshold must be positive.
    with pytest.raises(USDStrengthIntelligenceError):
        USDStrengthIntelligence(
            strong_threshold=-1.0,
        )

    # Weak threshold must be negative.
    with pytest.raises(USDStrengthIntelligenceError):
        USDStrengthIntelligence(
            weak_threshold=1.0,
        )


def test_equal_threshold_magnitudes_are_valid() -> None:
    """Equal positive/negative threshold magnitudes are valid."""

    # This was the bug in the previous implementation.
    engine = USDStrengthIntelligence(
        strong_threshold=60.0,
        weak_threshold=-60.0,
    )

    # Verify the engine constructs successfully.
    assert engine is not None


def test_xauusd_wrapper_preserves_usd_assessment() -> None:
    """XAUUSD wrapper must return USD strength, not a trade signal."""

    # Configure only DXY.
    engine = USDStrengthIntelligence(
        weights={
            MacroIndicator.DXY: 1.0,
        },
    )

    # Analyze through the XAUUSD wrapper.
    result = engine.analyze_xauusd(
        [
            make_observation(
                MacroIndicator.DXY,
                MacroDirection.RISING,
            ),
        ],
        BASE_TIME,
    )

    # Verify the result remains a USD-strength assessment.
    assert result.level is USDStrengthLevel.STRONG

    # Verify the score.
    assert result.score == 100.0