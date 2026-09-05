# tests/trading/test_dxy_intelligence.py

"""Tests for deterministic DXY intelligence."""

# Import datetime utilities.
from datetime import datetime, timedelta, timezone

# Import pytest for exception assertions.
import pytest

# Import macro models.
from app.trading.macro import (
    MacroDirection,
    MacroIndicator,
    MacroObservation,
)

# Import DXY intelligence.
from app.trading.macro import (
    DXYIntelligence,
    DXYIntelligenceError,
    DXYLevel,
)


# Define a deterministic UTC base timestamp.
BASE_TIME = datetime(
    2026,
    1,
    1,
    12,
    0,
    tzinfo=timezone.utc,
)


def make_dxy(
    value: float,
    previous: float | None,
    direction: MacroDirection,
    minutes: int = 0,
) -> MacroObservation:
    """Create a deterministic DXY observation."""

    # Calculate the observation timestamp.
    timestamp = BASE_TIME + timedelta(
        minutes=minutes,
    )

    # Return the DXY observation.
    return MacroObservation(
        timestamp=timestamp,
        indicator=MacroIndicator.DXY,
        value=value,
        previous=previous,
        forecast=None,
        source="test",
        direction=direction,
    )


def test_strong_dxy() -> None:
    """Large positive DXY movement should be strongly bullish."""

    # Create the DXY engine.
    engine = DXYIntelligence()

    # Create a 1% DXY increase.
    observation = make_dxy(
        value=101.0,
        previous=100.0,
        direction=MacroDirection.RISING,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify strong classification.
    assert result.level is DXYLevel.STRONG

    # Verify positive movement.
    assert result.percentage_change == 1.0

    # Verify sufficient data.
    assert result.sufficient_data is True


def test_bullish_dxy() -> None:
    """Meaningful positive DXY movement should be bullish."""

    # Create the DXY engine.
    engine = DXYIntelligence()

    # Create a 0.2% DXY increase.
    observation = make_dxy(
        value=100.2,
        previous=100.0,
        direction=MacroDirection.RISING,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify bullish classification.
    assert result.level is DXYLevel.BULLISH

    # Verify percentage movement.
    assert result.percentage_change == pytest.approx(0.2)


def test_weak_dxy() -> None:
    """Large negative DXY movement should be strongly bearish."""

    # Create the DXY engine.
    engine = DXYIntelligence()

    # Create a 1% DXY decrease.
    observation = make_dxy(
        value=99.0,
        previous=100.0,
        direction=MacroDirection.FALLING,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify weak classification.
    assert result.level is DXYLevel.WEAK

    # Verify negative movement.
    assert result.percentage_change == -1.0


def test_bearish_dxy() -> None:
    """Meaningful negative DXY movement should be bearish."""

    # Create the DXY engine.
    engine = DXYIntelligence()

    # Create a 0.2% DXY decrease.
    observation = make_dxy(
        value=99.8,
        previous=100.0,
        direction=MacroDirection.FALLING,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify bearish classification.
    assert result.level is DXYLevel.BEARISH

    # Verify percentage movement.
    assert result.percentage_change == pytest.approx(-0.2)


def test_small_rising_move_is_neutral() -> None:
    """A very small DXY movement should not create a strong bias."""

    # Create the DXY engine.
    engine = DXYIntelligence()

    # Create a 0.05% increase.
    observation = make_dxy(
        value=100.05,
        previous=100.0,
        direction=MacroDirection.RISING,
    )

    # Analyze the observation.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify neutral classification.
    assert result.level is DXYLevel.NEUTRAL


def test_stable_direction_is_neutral() -> None:
    """Stable DXY should produce a neutral result."""

    # Create the engine.
    engine = DXYIntelligence()

    # Create a stable observation.
    observation = make_dxy(
        value=100.0,
        previous=100.0,
        direction=MacroDirection.STABLE,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify neutral classification.
    assert result.level is DXYLevel.NEUTRAL

    # Verify exact zero movement.
    assert result.percentage_change == 0.0


def test_unknown_direction_returns_unknown() -> None:
    """Unknown DXY direction must not create a false signal."""

    # Create the engine.
    engine = DXYIntelligence()

    # Create an unknown-direction observation.
    observation = make_dxy(
        value=100.0,
        previous=99.0,
        direction=MacroDirection.UNKNOWN,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify unknown classification.
    assert result.level is DXYLevel.UNKNOWN

    # Verify insufficient directional data.
    assert result.sufficient_data is False

    # Verify zero confidence.
    assert result.confidence == 0.0


def test_no_previous_value_uses_direction() -> None:
    """Direction can still be used when previous value is unavailable."""

    # Create the engine.
    engine = DXYIntelligence()

    # Create an observation without previous data.
    observation = make_dxy(
        value=100.0,
        previous=None,
        direction=MacroDirection.RISING,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Direction alone should produce bullish classification.
    assert result.level is DXYLevel.BULLISH

    # Percentage change is unavailable.
    assert result.percentage_change is None

    # Direction-only confidence is lower.
    assert result.confidence == 50.0


def test_future_observation_is_ignored() -> None:
    """Future DXY data must never affect historical analysis."""

    # Create the engine.
    engine = DXYIntelligence()

    # Create a future observation.
    future = make_dxy(
        value=110.0,
        previous=100.0,
        direction=MacroDirection.RISING,
        minutes=10,
    )

    # Analyze before that observation exists.
    result = engine.analyze(
        [future],
        BASE_TIME,
    )

    # Verify no-lookahead protection.
    assert result.level is DXYLevel.UNKNOWN
    assert result.sufficient_data is False


def test_exact_timestamp_is_allowed() -> None:
    """An observation exactly at decision time is valid."""

    # Create the engine.
    engine = DXYIntelligence()

    # Create an observation at decision time.
    observation = make_dxy(
        value=101.0,
        previous=100.0,
        direction=MacroDirection.RISING,
    )

    # Analyze at the exact same timestamp.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify inclusion.
    assert result.level is DXYLevel.STRONG


def test_latest_observation_is_used() -> None:
    """Only the latest historical-safe DXY observation is used."""

    # Create the engine.
    engine = DXYIntelligence()

    # Older bullish observation.
    older = make_dxy(
        value=101.0,
        previous=100.0,
        direction=MacroDirection.RISING,
        minutes=-10,
    )

    # Newer bearish observation.
    newer = make_dxy(
        value=99.0,
        previous=100.0,
        direction=MacroDirection.FALLING,
        minutes=-5,
    )

    # Analyze both.
    result = engine.analyze(
        [older, newer],
        BASE_TIME,
    )

    # Newer observation must win.
    assert result.level is DXYLevel.WEAK
    assert result.value == 99.0


def test_non_dxy_indicators_are_ignored() -> None:
    """Non-DXY observations must not affect DXY intelligence."""

    # Create the engine.
    engine = DXYIntelligence()

    # Create a non-DXY observation.
    observation = MacroObservation(
        timestamp=BASE_TIME,
        indicator=MacroIndicator.CPI,
        value=3.0,
        previous=2.9,
        forecast=3.0,
        source="test",
        direction=MacroDirection.RISING,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # No DXY information exists.
    assert result.level is DXYLevel.UNKNOWN


def test_timezone_mismatch_is_rejected() -> None:
    """Naive and timezone-aware timestamps must not be mixed."""

    # Create the engine.
    engine = DXYIntelligence()

    # Create a naive DXY observation.
    observation = MacroObservation(
        timestamp=datetime(
            2026,
            1,
            1,
            12,
            0,
        ),
        indicator=MacroIndicator.DXY,
        value=101.0,
        previous=100.0,
        forecast=None,
        source="test",
        direction=MacroDirection.RISING,
    )

    # Mixed timestamp semantics must raise.
    with pytest.raises(DXYIntelligenceError):
        engine.analyze(
            [observation],
            BASE_TIME,
        )


def test_zero_previous_value_does_not_divide_by_zero() -> None:
    """Zero previous DXY value must not cause division by zero."""

    # Create the engine.
    engine = DXYIntelligence()

    # Previous value is zero.
    observation = make_dxy(
        value=100.0,
        previous=0.0,
        direction=MacroDirection.RISING,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Percentage change is unavailable.
    assert result.percentage_change is None

    # Direction still provides a bullish assessment.
    assert result.level is DXYLevel.BULLISH


def test_custom_thresholds_are_supported() -> None:
    """Custom DXY thresholds should be respected."""

    # Configure lower thresholds.
    engine = DXYIntelligence(
        significant_change_pct=0.05,
        strong_change_pct=0.20,
    )

    # A 0.25% move should now be strong.
    observation = make_dxy(
        value=100.25,
        previous=100.0,
        direction=MacroDirection.RISING,
    )

    # Analyze.
    result = engine.analyze(
        [observation],
        BASE_TIME,
    )

    # Verify custom threshold behavior.
    assert result.level is DXYLevel.STRONG


def test_invalid_thresholds_are_rejected() -> None:
    """Invalid threshold configuration must fail."""

    # Significant threshold cannot be negative.
    with pytest.raises(DXYIntelligenceError):
        DXYIntelligence(
            significant_change_pct=-0.1,
        )

    # Strong threshold cannot be negative.
    with pytest.raises(DXYIntelligenceError):
        DXYIntelligence(
            strong_change_pct=-0.1,
        )

    # Strong threshold cannot be below significant threshold.
    with pytest.raises(DXYIntelligenceError):
        DXYIntelligence(
            significant_change_pct=0.5,
            strong_change_pct=0.1,
        )


def test_xauusd_wrapper_does_not_create_trade_signal() -> None:
    """XAUUSD wrapper should return DXY intelligence only."""

    # Create the engine.
    engine = DXYIntelligence()

    # Create bullish DXY data.
    observation = make_dxy(
        value=101.0,
        previous=100.0,
        direction=MacroDirection.RISING,
    )

    # Analyze through XAUUSD wrapper.
    result = engine.analyze_xauusd(
        [observation],
        BASE_TIME,
    )

    # Verify it remains a DXY assessment.
    assert result.level is DXYLevel.STRONG
    assert result.value == 101.0