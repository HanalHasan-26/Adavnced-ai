"""
Tests for deterministic macro regime intelligence.
"""

from datetime import datetime, timezone

import pytest

from app.trading.macro.macro_observation import (
    MacroDirection,
    MacroIndicator,
    MacroObservation,
)

from app.trading.macro.macro_regime_intelligence import (
    MacroRegime,
    MacroRegimeComponent,
    MacroRegimeIntelligence,
    MacroRegimeIntelligenceError,
)


# Define a shared timezone-aware decision timestamp.
DECISION_TIME = datetime(
    2026,
    9,
    4,
    12,
    0,
    tzinfo=timezone.utc,
)


def make_observation(
    *,
    indicator: MacroIndicator,
    value: float,
    previous: float | None = None,
    timestamp: datetime = DECISION_TIME,
    direction: MacroDirection = MacroDirection.UNKNOWN,
) -> MacroObservation:
    """Create a MacroObservation for tests."""

    # Return the production macro observation model.
    return MacroObservation(
        timestamp=timestamp,
        indicator=indicator,
        value=value,
        previous=previous,
        forecast=None,
        source="test",
        direction=direction,
    )


def test_risk_off_environment_can_produce_risk_off_regime() -> None:
    """Strong defensive macro inputs should produce risk-off."""

    # Create defensive macro observations.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=103.0,
            previous=100.0,
        ),
        make_observation(
            indicator=MacroIndicator.US_10Y_YIELD,
            value=4.5,
            previous=4.0,
        ),
        make_observation(
            indicator=MacroIndicator.FED_FUNDS_RATE,
            value=5.5,
            previous=5.0,
        ),
        make_observation(
            indicator=MacroIndicator.CPI,
            value=4.0,
            previous=3.5,
        ),
        make_observation(
            indicator=MacroIndicator.NFP,
            value=100.0,
            previous=200.0,
        ),
        make_observation(
            indicator=MacroIndicator.UNEMPLOYMENT_RATE,
            value=4.5,
            previous=4.0,
        ),
    ]

    # Use only the relevant components for deterministic testing.
    engine = MacroRegimeIntelligence(
        weights={
            "usd_strength": 1.0,
            "dxy": 1.0,
            "treasury_yields": 1.0,
            "fed_rate": 1.0,
            "inflation": 1.0,
            "employment": 1.0,
            "risk_sentiment": 1.0,
        }
    )

    # Analyze the regime.
    result = engine.analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify sufficient data exists.
    assert result.sufficient_data is True

    # Defensive conditions should not produce a positive regime.
    assert result.regime in {
        MacroRegime.RISK_OFF,
        MacroRegime.TIGHTENING,
        MacroRegime.INFLATIONARY,
        MacroRegime.MIXED,
    }


def test_disinflationary_dovish_environment_can_produce_easing() -> None:
    """Cooling inflation plus falling rates should support easing."""

    # Create disinflationary and dovish observations.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=98.0,
            previous=100.0,
        ),
        make_observation(
            indicator=MacroIndicator.US_10Y_YIELD,
            value=3.5,
            previous=4.0,
        ),
        make_observation(
            indicator=MacroIndicator.FED_FUNDS_RATE,
            value=4.5,
            previous=5.0,
        ),
        make_observation(
            indicator=MacroIndicator.CPI,
            value=3.0,
            previous=3.5,
        ),
        make_observation(
            indicator=MacroIndicator.NFP,
            value=180.0,
            previous=170.0,
        ),
        make_observation(
            indicator=MacroIndicator.UNEMPLOYMENT_RATE,
            value=4.0,
            previous=4.1,
        ),
    ]

    # Analyze the environment.
    result = MacroRegimeIntelligence().analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify sufficient macro data exists.
    assert result.sufficient_data is True

    # The result must not be classified as tightening.
    assert result.regime != MacroRegime.TIGHTENING


def test_insufficient_data_returns_unknown() -> None:
    """Insufficient weighted coverage must return UNKNOWN."""

    # Configure a high coverage requirement.
    engine = MacroRegimeIntelligence(
        min_coverage=0.90,
    )

    # Provide only one macro observation.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=100.0,
            previous=99.0,
        )
    ]

    # Analyze with insufficient data.
    result = engine.analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify unknown regime.
    assert result.sufficient_data is False
    assert result.regime == MacroRegime.UNKNOWN


def test_future_data_is_ignored() -> None:
    """Future observations must never influence regime analysis."""

    # Create a historical and future DXY observation.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=99.0,
            previous=100.0,
            timestamp=datetime(
                2026,
                9,
                3,
                12,
                0,
                tzinfo=timezone.utc,
            ),
        ),
        make_observation(
            indicator=MacroIndicator.DXY,
            value=110.0,
            previous=100.0,
            timestamp=datetime(
                2026,
                9,
                5,
                12,
                0,
                tzinfo=timezone.utc,
            ),
        ),
    ]

    # Use only DXY for deterministic testing.
    engine = MacroRegimeIntelligence(
        weights={
            "usd_strength": 0.0,
            "dxy": 1.0,
            "treasury_yields": 0.0,
            "fed_rate": 0.0,
            "inflation": 0.0,
            "employment": 0.0,
            "risk_sentiment": 0.0,
        }
    )

    # Analyze historical state.
    result = engine.analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Historical falling DXY should produce a non-negative interpretation.
    assert result.dxy.value == 99.0


def test_exact_decision_timestamp_is_allowed() -> None:
    """Data exactly at decision time is valid."""

    # Create an observation exactly at decision time.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=99.0,
            previous=100.0,
            timestamp=DECISION_TIME,
        )
    ]

    # Analyze using only DXY.
    engine = MacroRegimeIntelligence(
        weights={
            "usd_strength": 0.0,
            "dxy": 1.0,
            "treasury_yields": 0.0,
            "fed_rate": 0.0,
            "inflation": 0.0,
            "employment": 0.0,
            "risk_sentiment": 0.0,
        }
    )

    # Run the analysis.
    result = engine.analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify the exact-time observation was used.
    assert result.dxy.value == 99.0


def test_xauusd_wrapper_returns_regime_assessment() -> None:
    """XAUUSD wrapper should return macro regime context only."""

    # Create a simple DXY observation.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=99.0,
            previous=100.0,
        )
    ]

    # Use DXY-only configuration.
    engine = MacroRegimeIntelligence(
        weights={
            "usd_strength": 0.0,
            "dxy": 1.0,
            "treasury_yields": 0.0,
            "fed_rate": 0.0,
            "inflation": 0.0,
            "employment": 0.0,
            "risk_sentiment": 0.0,
        }
    )

    # Analyze XAUUSD macro context.
    result = engine.analyze_xauusd(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify that an assessment object is returned.
    assert result.decision_timestamp == DECISION_TIME


def test_naive_decision_timestamp_is_rejected() -> None:
    """Naive timestamps are unsafe and must be rejected."""

    # Create valid data.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=100.0,
            previous=99.0,
        )
    ]

    # Attempt to analyze using a naive timestamp.
    with pytest.raises(MacroRegimeIntelligenceError):
        MacroRegimeIntelligence().analyze(
            observations,
            decision_timestamp=datetime(
                2026,
                9,
                4,
                12,
                0,
            ),
        )


def test_negative_weight_is_rejected() -> None:
    """Negative subsystem weights must be rejected."""

    # Attempt to create an invalid configuration.
    with pytest.raises(MacroRegimeIntelligenceError):
        MacroRegimeIntelligence(
            weights={
                "dxy": -1.0,
            }
        )


def test_unknown_weight_source_is_rejected() -> None:
    """Unknown subsystem names must be rejected."""

    # Attempt to use an invalid subsystem.
    with pytest.raises(MacroRegimeIntelligenceError):
        MacroRegimeIntelligence(
            weights={
                "unknown_source": 1.0,
            }
        )


def test_invalid_coverage_is_rejected() -> None:
    """Coverage outside the valid range must be rejected."""

    # Attempt invalid coverage.
    with pytest.raises(MacroRegimeIntelligenceError):
        MacroRegimeIntelligence(
            min_coverage=1.5,
        )


def test_strong_threshold_below_normal_threshold_is_rejected() -> None:
    """Strong threshold cannot be below normal threshold."""

    # Attempt invalid threshold ordering.
    with pytest.raises(MacroRegimeIntelligenceError):
        MacroRegimeIntelligence(
            direction_threshold=50.0,
            strong_direction_threshold=20.0,
        )


def test_serialization_contains_regime_and_components() -> None:
    """Serialization should include the regime and contributions."""

    # Create a DXY observation.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=99.0,
            previous=100.0,
        )
    ]

    # Use DXY-only configuration.
    engine = MacroRegimeIntelligence(
        weights={
            "usd_strength": 0.0,
            "dxy": 1.0,
            "treasury_yields": 0.0,
            "fed_rate": 0.0,
            "inflation": 0.0,
            "employment": 0.0,
            "risk_sentiment": 0.0,
        }
    )

    # Analyze.
    result = engine.analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Serialize.
    data = result.to_dict()

    # Verify stable fields.
    assert "regime" in data
    assert "score" in data
    assert "contributions" in data
    assert "dxy" in data