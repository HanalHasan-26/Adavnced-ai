"""
Tests for deterministic risk-on / risk-off intelligence.
"""

from datetime import datetime, timezone

import pytest

from app.trading.macro.macro_observation import (
    MacroDirection,
    MacroIndicator,
    MacroObservation,
)
from app.trading.macro.risk_sentiment_intelligence import (
    RiskSentiment,
    RiskSentimentComponent,
    RiskSentimentIntelligence,
    RiskSentimentIntelligenceError,
)


# Define one shared historical decision timestamp.
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
    """Create a test MacroObservation."""

    # Return the production observation model.
    return MacroObservation(
        timestamp=timestamp,
        indicator=indicator,
        value=value,
        previous=previous,
        forecast=None,
        source="test",
        direction=direction,
    )


def test_dxy_rising_is_risk_off() -> None:
    """Rising DXY should contribute risk-off."""

    # Create a rising DXY observation.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=102.0,
            previous=100.0,
        )
    ]

    # Analyze risk sentiment.
    result = RiskSentimentIntelligence().analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify the component.
    assert result.contributions[0].component == RiskSentimentComponent.RISK_OFF


def test_dxy_falling_is_risk_on() -> None:
    """Falling DXY should contribute risk-on."""

    # Create a falling DXY observation.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=98.0,
            previous=100.0,
        )
    ]

    # Analyze risk sentiment.
    result = RiskSentimentIntelligence().analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify risk-on classification.
    assert result.contributions[0].component == RiskSentimentComponent.RISK_ON


def test_rising_treasury_yield_is_risk_off() -> None:
    """Rising Treasury yields should contribute risk-off."""

    # Create a rising 10-year yield observation.
    observations = [
        make_observation(
            indicator=MacroIndicator.US_10Y_YIELD,
            value=4.2,
            previous=4.0,
        )
    ]

    # Analyze risk sentiment.
    result = RiskSentimentIntelligence().analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify risk-off component.
    assert result.contributions[0].component == RiskSentimentComponent.RISK_OFF


def test_falling_treasury_yield_is_risk_on() -> None:
    """Falling Treasury yields should contribute risk-on."""

    # Create a falling 10-year yield observation.
    observations = [
        make_observation(
            indicator=MacroIndicator.US_10Y_YIELD,
            value=3.8,
            previous=4.0,
        )
    ]

    # Analyze risk sentiment.
    result = RiskSentimentIntelligence().analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify risk-on component.
    assert result.contributions[0].component == RiskSentimentComponent.RISK_ON


def test_rising_fed_rate_is_risk_off() -> None:
    """Rising Fed Funds Rate should contribute risk-off."""

    # Create a rising Fed rate observation.
    observations = [
        make_observation(
            indicator=MacroIndicator.FED_FUNDS_RATE,
            value=5.5,
            previous=5.25,
        )
    ]

    # Analyze risk sentiment.
    result = RiskSentimentIntelligence().analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify risk-off component.
    assert result.contributions[0].component == RiskSentimentComponent.RISK_OFF


def test_falling_inflation_is_risk_on() -> None:
    """Cooling inflation should contribute risk-on."""

    # Create falling CPI.
    observations = [
        make_observation(
            indicator=MacroIndicator.CPI,
            value=3.0,
            previous=3.5,
        )
    ]

    # Analyze risk sentiment.
    result = RiskSentimentIntelligence().analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify risk-on component.
    assert result.contributions[0].component == RiskSentimentComponent.RISK_ON


def test_rising_nfp_is_risk_on() -> None:
    """Stronger NFP should contribute risk-on."""

    # Create rising NFP.
    observations = [
        make_observation(
            indicator=MacroIndicator.NFP,
            value=220.0,
            previous=180.0,
        )
    ]

    # Analyze risk sentiment.
    result = RiskSentimentIntelligence().analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify risk-on component.
    assert result.contributions[0].component == RiskSentimentComponent.RISK_ON


def test_falling_nfp_is_risk_off() -> None:
    """Weaker NFP should contribute risk-off."""

    # Create falling NFP.
    observations = [
        make_observation(
            indicator=MacroIndicator.NFP,
            value=120.0,
            previous=180.0,
        )
    ]

    # Analyze risk sentiment.
    result = RiskSentimentIntelligence().analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify risk-off component.
    assert result.contributions[0].component == RiskSentimentComponent.RISK_OFF


def test_falling_unemployment_is_risk_on() -> None:
    """Falling unemployment should contribute risk-on."""

    # Create falling unemployment.
    observations = [
        make_observation(
            indicator=MacroIndicator.UNEMPLOYMENT_RATE,
            value=3.8,
            previous=4.1,
        )
    ]

    # Analyze risk sentiment.
    result = RiskSentimentIntelligence().analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify risk-on component.
    assert result.contributions[0].component == RiskSentimentComponent.RISK_ON


def test_rising_unemployment_is_risk_off() -> None:
    """Rising unemployment should contribute risk-off."""

    # Create rising unemployment.
    observations = [
        make_observation(
            indicator=MacroIndicator.UNEMPLOYMENT_RATE,
            value=4.5,
            previous=4.1,
        )
    ]

    # Analyze risk sentiment.
    result = RiskSentimentIntelligence().analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify risk-off component.
    assert result.contributions[0].component == RiskSentimentComponent.RISK_OFF


def test_stable_indicator_is_neutral() -> None:
    """Stable data should contribute neutral."""

    # Create a stable DXY observation.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=100.0,
            previous=100.0,
        )
    ]

    # Analyze risk sentiment.
    result = RiskSentimentIntelligence().analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify neutral contribution.
    assert result.contributions[0].component == RiskSentimentComponent.NEUTRAL
    assert result.score == 0.0


def test_mixed_data_can_produce_neutral() -> None:
    """Opposing components can cancel each other."""

    # Create equally weighted opposing indicators.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=102.0,
            previous=100.0,
        ),
        make_observation(
            indicator=MacroIndicator.NFP,
            value=220.0,
            previous=180.0,
        ),
    ]

    # Use equal weights to make cancellation deterministic.
    engine = RiskSentimentIntelligence(
        weights={
            MacroIndicator.DXY: 1.0,
            MacroIndicator.NFP: 1.0,
        }
    )

    # Analyze the mixed environment.
    result = engine.analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify exact cancellation.
    assert result.score == 0.0
    assert result.sentiment == RiskSentiment.NEUTRAL


def test_strong_risk_off_is_classified() -> None:
    """Several strong risk-off components should produce strong risk-off."""

    # Create multiple defensive indicators.
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
    ]

    # Use equal weights.
    engine = RiskSentimentIntelligence(
        weights={
            MacroIndicator.DXY: 1.0,
            MacroIndicator.US_10Y_YIELD: 1.0,
            MacroIndicator.FED_FUNDS_RATE: 1.0,
        }
    )

    # Analyze the environment.
    result = engine.analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify strong risk-off classification.
    assert result.sentiment == RiskSentiment.STRONG_RISK_OFF
    assert result.score == -100.0


def test_strong_risk_on_is_classified() -> None:
    """Several risk-on components should produce strong risk-on."""

    # Create several positive-risk indicators.
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
            indicator=MacroIndicator.NFP,
            value=250.0,
            previous=180.0,
        ),
    ]

    # Use equal weights.
    engine = RiskSentimentIntelligence(
        weights={
            MacroIndicator.DXY: 1.0,
            MacroIndicator.US_10Y_YIELD: 1.0,
            MacroIndicator.NFP: 1.0,
        }
    )

    # Analyze the environment.
    result = engine.analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify strong risk-on classification.
    assert result.sentiment == RiskSentiment.STRONG_RISK_ON
    assert result.score == 100.0


def test_future_observation_is_ignored() -> None:
    """Future data must never influence historical sentiment."""

    # Create one historical and one future observation.
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

    # Analyze using the earlier decision time.
    result = RiskSentimentIntelligence(
        weights={MacroIndicator.DXY: 1.0}
    ).analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # The historical falling DXY should be selected.
    assert result.score == 100.0


def test_exact_timestamp_is_allowed() -> None:
    """An observation exactly at decision time is valid."""

    # Create an observation at the decision timestamp.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=99.0,
            previous=100.0,
        )
    ]

    # Analyze sentiment.
    result = RiskSentimentIntelligence(
        weights={MacroIndicator.DXY: 1.0}
    ).analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify the exact-time observation is used.
    assert result.score == 100.0


def test_missing_data_reduces_coverage() -> None:
    """Missing indicators should reduce weighted coverage."""

    # Configure two equal-weight indicators but provide only one.
    engine = RiskSentimentIntelligence(
        weights={
            MacroIndicator.DXY: 1.0,
            MacroIndicator.NFP: 1.0,
        }
    )

    # Provide only DXY.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=99.0,
            previous=100.0,
        )
    ]

    # Analyze with the default 50% coverage requirement.
    result = engine.analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Exactly 50% coverage is sufficient.
    assert result.sufficient_data is True
    assert result.confidence == 50.0


def test_insufficient_coverage_returns_unknown() -> None:
    """Coverage below the configured minimum should return UNKNOWN."""

    # Configure three equal-weight indicators.
    engine = RiskSentimentIntelligence(
        weights={
            MacroIndicator.DXY: 1.0,
            MacroIndicator.NFP: 1.0,
            MacroIndicator.CPI: 1.0,
        },
        min_coverage=0.75,
    )

    # Provide only one indicator.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=99.0,
            previous=100.0,
        )
    ]

    # Analyze sentiment.
    result = engine.analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify insufficient-data behavior.
    assert result.sufficient_data is False
    assert result.sentiment == RiskSentiment.UNKNOWN


def test_xauusd_wrapper_returns_assessment_only() -> None:
    """XAUUSD wrapper should not produce a trade decision."""

    # Create a valid DXY observation.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=99.0,
            previous=100.0,
        )
    ]

    # Analyze through the XAUUSD wrapper.
    result = RiskSentimentIntelligence(
        weights={MacroIndicator.DXY: 1.0}
    ).analyze_xauusd(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Verify the wrapper returns risk sentiment only.
    assert isinstance(result.sentiment, RiskSentiment)
    assert result.score == 100.0


def test_naive_decision_timestamp_is_rejected() -> None:
    """Naive decision timestamps should be rejected."""

    # Create valid data.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=99.0,
            previous=100.0,
        )
    ]

    # Attempt historical analysis with a naive timestamp.
    with pytest.raises(RiskSentimentIntelligenceError):
        RiskSentimentIntelligence().analyze(
            observations,
            decision_timestamp=datetime(2026, 9, 4, 12, 0),
        )


def test_invalid_coverage_is_rejected() -> None:
    """Coverage outside [0, 1] should be rejected."""

    # Attempt an invalid configuration.
    with pytest.raises(RiskSentimentIntelligenceError):
        RiskSentimentIntelligence(
            min_coverage=1.5,
        )


def test_negative_weight_is_rejected() -> None:
    """Negative indicator weights should be rejected."""

    # Attempt to configure a negative weight.
    with pytest.raises(RiskSentimentIntelligenceError):
        RiskSentimentIntelligence(
            weights={
                MacroIndicator.DXY: -1.0,
            }
        )


def test_invalid_threshold_order_is_rejected() -> None:
    """Invalid threshold ordering should be rejected."""

    # Strong risk-on threshold cannot be below the normal threshold.
    with pytest.raises(RiskSentimentIntelligenceError):
        RiskSentimentIntelligence(
            strong_risk_on_threshold=10.0,
            risk_on_threshold=20.0,
        )


def test_equal_positive_thresholds_are_allowed() -> None:
    """Equal positive thresholds are valid."""

    # Construct an engine with equal positive thresholds.
    engine = RiskSentimentIntelligence(
        strong_risk_on_threshold=20.0,
        risk_on_threshold=20.0,
    )

    # Verify successful construction.
    assert engine.strong_risk_on_threshold == 20.0


def test_serialization_contains_contributions() -> None:
    """Assessment serialization should include contribution details."""

    # Create a valid observation.
    observations = [
        make_observation(
            indicator=MacroIndicator.DXY,
            value=99.0,
            previous=100.0,
        )
    ]

    # Analyze the observation.
    result = RiskSentimentIntelligence(
        weights={MacroIndicator.DXY: 1.0}
    ).analyze(
        observations,
        decision_timestamp=DECISION_TIME,
    )

    # Serialize the result.
    data = result.to_dict()

    # Verify stable serialized fields.
    assert data["sentiment"] == RiskSentiment.STRONG_RISK_ON.value
    assert data["score"] == 100.0
    assert len(data["contributions"]) == 1