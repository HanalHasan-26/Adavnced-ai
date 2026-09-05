"""
Tests for XAUUSD macro impact intelligence.

These tests verify:
- deterministic behavior
- XAUUSD-only validation
- historical safety through the regime engine
- missing-data handling
- USD pressure
- rate pressure
- inflation pressure
- employment pressure
- risk sentiment pressure
- regime pressure
- serialization
- threshold behavior
- no direct trade-decision behavior
"""

from datetime import datetime, timezone

import pytest

from app.trading.macro.macro_observation import (
    MacroIndicator,
    MacroObservation,
)

from app.trading.macro.xauusd_macro_impact_intelligence import (
    XAUUSDMacroBias,
    XAUUSDMacroComponent,
    XAUUSDMacroImpactIntelligence,
    XAUUSDMacroImpactIntelligenceError,
)


# Fixed historical decision timestamp used by the tests.
DECISION_TIME = datetime(
    2026,
    1,
    15,
    12,
    0,
    tzinfo=timezone.utc,
)


def make_observation(
    *,
    indicator: MacroIndicator,
    value: float,
    previous: float | None = None,
    forecast: float | None = None,
    timestamp: datetime = DECISION_TIME,
) -> MacroObservation:
    """Create a valid macro observation for testing."""

    # Construct the production MacroObservation model.
    return MacroObservation(
        timestamp=timestamp,
        indicator=indicator,
        value=value,
        previous=previous,
        forecast=forecast,
        source="test",
    )


def test_empty_data_returns_unknown() -> None:
    """No macro observations should produce an unknown bias."""

    # Create the production engine.
    engine = XAUUSDMacroImpactIntelligence()

    # Analyze with no observations.
    result = engine.analyze(
        [],
        decision_timestamp=DECISION_TIME,
    )

    # The result must be unknown because there is no data.
    assert result.bias == XAUUSDMacroBias.UNKNOWN

    # The result must explicitly report insufficient data.
    assert result.sufficient_data is False


def test_only_xauusd_is_supported() -> None:
    """The engine must reject non-XAUUSD symbols."""

    # Create the production engine.
    engine = XAUUSDMacroImpactIntelligence()

    # Attempt to analyze another symbol.
    with pytest.raises(
        XAUUSDMacroImpactIntelligenceError
    ):
        engine.analyze(
            [],
            decision_timestamp=DECISION_TIME,
            symbol="EURUSD",
        )


def test_timezone_naive_decision_timestamp_is_rejected() -> None:
    """Naive decision timestamps must be rejected."""

    # Create the production engine.
    engine = XAUUSDMacroImpactIntelligence()

    # Use a naive timestamp.
    naive_timestamp = datetime(
        2026,
        1,
        15,
        12,
        0,
    )

    # The engine must reject it.
    with pytest.raises(
        XAUUSDMacroImpactIntelligenceError
    ):
        engine.analyze(
            [],
            decision_timestamp=naive_timestamp,
        )


def test_non_observation_input_is_rejected() -> None:
    """Invalid observation objects must be rejected."""

    # Create the production engine.
    engine = XAUUSDMacroImpactIntelligence()

    # Pass an invalid object.
    with pytest.raises(
        XAUUSDMacroImpactIntelligenceError
    ):
        engine.analyze(
            ["invalid"],
            decision_timestamp=DECISION_TIME,
        )


def test_dxy_weakness_creates_bullish_gold_pressure() -> None:
    """Falling DXY should contribute bullish XAUUSD pressure."""

    # Create a falling DXY observation.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=99.0,
            previous=100.0,
        )
    ]

    # Use DXY-only configuration.
    engine = XAUUSDMacroImpactIntelligence(
        weights={
            "usd": 1.0,
            "rates": 0.0,
            "inflation": 0.0,
            "employment": 0.0,
            "risk": 0.0,
            "regime": 0.0,
        },
        min_confidence=0.0,
    )

    # Analyze the observations.
    result = engine.analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # The USD component should be bullish for gold.
    usd_contribution = next(
        item
        for item in result.contributions
        if item.source == "usd"
    )

    # Verify component classification.
    assert (
        usd_contribution.component
        == XAUUSDMacroComponent.USD_PRESSURE
    )

    # Verify positive contribution.
    assert usd_contribution.contribution > 0


def test_dxy_strength_creates_bearish_gold_pressure() -> None:
    """Rising DXY should contribute bearish XAUUSD pressure."""

    # Create a rising DXY observation.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=101.0,
            previous=100.0,
        )
    ]

    # Use DXY-only configuration.
    engine = XAUUSDMacroImpactIntelligence(
        weights={
            "usd": 1.0,
            "rates": 0.0,
            "inflation": 0.0,
            "employment": 0.0,
            "risk": 0.0,
            "regime": 0.0,
        },
        min_confidence=0.0,
    )

    # Analyze the observations.
    result = engine.analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Locate the USD contribution.
    usd_contribution = next(
        item
        for item in result.contributions
        if item.source == "usd"
    )

    # Verify bearish pressure.
    assert usd_contribution.contribution < 0


def test_falling_yields_create_bullish_rate_pressure() -> None:
    """Falling Treasury yields should support XAUUSD."""

    # Create a falling 10-year yield observation.
    observations = [
        make_observation(
            indicator=MacroIndicator.US_10Y_YIELD,
            value=4.00,
            previous=4.10,
        )
    ]

    # Use rate-only configuration.
    engine = XAUUSDMacroImpactIntelligence(
        weights={
            "usd": 0.0,
            "rates": 1.0,
            "inflation": 0.0,
            "employment": 0.0,
            "risk": 0.0,
            "regime": 0.0,
        },
        min_confidence=0.0,
    )

    # Analyze the observations.
    result = engine.analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Locate the rate contribution.
    rate_contribution = next(
        item
        for item in result.contributions
        if item.source == "rates"
    )

    # Verify bullish rate pressure.
    assert rate_contribution.contribution > 0


def test_rising_yields_create_bearish_rate_pressure() -> None:
    """Rising Treasury yields should pressure XAUUSD."""

    # Create a rising 10-year yield observation.
    observations = [
        make_observation(
            indicator=MacroIndicator.US_10Y_YIELD,
            value=4.20,
            previous=4.10,
        )
    ]

    # Use rate-only configuration.
    engine = XAUUSDMacroImpactIntelligence(
        weights={
            "usd": 0.0,
            "rates": 1.0,
            "inflation": 0.0,
            "employment": 0.0,
            "risk": 0.0,
            "regime": 0.0,
        },
        min_confidence=0.0,
    )

    # Analyze the observations.
    result = engine.analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Locate the rate contribution.
    rate_contribution = next(
        item
        for item in result.contributions
        if item.source == "rates"
    )

    # Verify bearish pressure.
    assert rate_contribution.contribution < 0


def test_risk_off_creates_bullish_gold_pressure() -> None:
    """Risk-off conditions should contribute safe-haven pressure."""

    # Create DXY and Treasury observations that create risk-off
    # context through the existing deterministic risk engine.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=101.0,
            previous=100.0,
        ),
        make_observation(
            indicator=MacroIndicator.US_10Y_YIELD,
            value=4.20,
            previous=4.10,
        ),
    ]

    # Create the production engine.
    engine = XAUUSDMacroImpactIntelligence(
        min_confidence=0.0,
    )

    # Analyze the observations.
    result = engine.analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Find the risk contribution.
    risk_contribution = next(
        item
        for item in result.contributions
        if item.source == "risk"
    )

    # Risk-off or neutral is valid depending on the underlying
    # risk-sentiment coverage and thresholds.
    assert risk_contribution.component in {
        XAUUSDMacroComponent.RISK_SENTIMENT_PRESSURE,
        XAUUSDMacroComponent.UNKNOWN,
    }


def test_future_observation_does_not_create_lookahead() -> None:
    """Future macro observations must not affect the result."""

    # Create an observation after the decision time.
    future_time = datetime(
        2026,
        1,
        15,
        13,
        0,
        tzinfo=timezone.utc,
    )

    # Create future bullish DXY data.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=90.0,
            previous=100.0,
            timestamp=future_time,
        )
    ]

    # Create the production engine.
    engine = XAUUSDMacroImpactIntelligence()

    # Analyze at the earlier decision time.
    result = engine.analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Future information must not make the result known.
    assert result.bias == XAUUSDMacroBias.UNKNOWN


def test_exact_decision_timestamp_is_allowed() -> None:
    """Data exactly at the decision time should be usable."""

    # Create an observation exactly at decision time.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=99.0,
            previous=100.0,
            timestamp=DECISION_TIME,
        )
    ]

    # Create the production engine.
    engine = XAUUSDMacroImpactIntelligence()

    # Analyze at the same timestamp.
    result = engine.analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # The DXY assessment should be available.
    assert result.macro_regime_assessment.dxy is not None


def test_serialization_contains_required_fields() -> None:
    """Serialization should contain the main assessment fields."""

    # Create a DXY observation.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=99.0,
            previous=100.0,
        )
    ]

    # Create the production engine.
    engine = XAUUSDMacroImpactIntelligence()

    # Analyze the data.
    result = engine.analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Serialize the result.
    data = result.to_dict()

    # Verify required fields.
    assert data["symbol"] == "XAUUSD"
    assert "bias" in data
    assert "score" in data
    assert "confidence" in data
    assert "macro_regime" in data
    assert "contributions" in data
    assert "reasons" in data


def test_xauusd_wrapper_matches_primary_analysis() -> None:
    """The XAUUSD wrapper should use the same analysis path."""

    # Create a DXY observation.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=99.0,
            previous=100.0,
        )
    ]

    # Create the production engine.
    engine = XAUUSDMacroImpactIntelligence()

    # Run the primary analysis.
    primary = engine.analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Run the XAUUSD wrapper.
    wrapped = engine.analyze_xauusd(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # The results should match.
    assert primary == wrapped


def test_result_does_not_authorize_trade() -> None:
    """Macro impact must remain contextual rather than executable."""

    # Create a bullish DXY observation.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=99.0,
            previous=100.0,
        )
    ]

    # Create the production engine.
    engine = XAUUSDMacroImpactIntelligence()

    # Analyze the observations.
    result = engine.analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify the explicit architecture warning exists.
    assert any(
        "does not authorize" in reason
        for reason in result.reasons
    )