"""
Tests for P2.20.1 Signal Fusion Intelligence.

The tests verify:

- deterministic weighting
- bullish fusion
- bearish fusion
- neutral fusion
- insufficient evidence
- risk veto
- symbol validation
- timestamp validation
- invalid scores
- serialization
- XAUUSD wrapper
"""

# Import datetime utilities.
from datetime import datetime, timezone

# Import pytest for exception assertions.
import pytest

# Import production Signal Fusion classes.
from app.trading.fusion.signal_fusion_intelligence import (
    FusionDecision,
    FusionDirection,
    SignalFusionEvidence,
    SignalFusionIntelligence,
    SignalFusionIntelligenceError,
)


# Define a deterministic decision timestamp.
DECISION_TIME = datetime(
    2026,
    9,
    5,
    12,
    0,
    tzinfo=timezone.utc,
)


def make_evidence(
    source: str,
    score: float,
    reason: str = "test evidence",
) -> SignalFusionEvidence:
    """Create deterministic test evidence."""

    # Convert the score into the matching direction.
    if score > 0:
        direction = FusionDirection.LONG
    elif score < 0:
        direction = FusionDirection.SHORT
    else:
        direction = FusionDirection.NEUTRAL

    # Return a test evidence object.
    return SignalFusionEvidence(
        source=source,
        direction=direction,
        score=score,
        weight=1.0,
        reason=reason,
    )


def test_empty_evidence_returns_unknown() -> None:
    """No evidence should produce UNKNOWN."""

    # Create the production engine.
    engine = SignalFusionIntelligence()

    # Analyze with no evidence.
    result = engine.analyze(
        evidence=[],
        decision_timestamp=DECISION_TIME,
    )

    # Verify no directional result exists.
    assert result.direction == FusionDirection.UNKNOWN

    # Verify decision state.
    assert result.decision == FusionDecision.UNKNOWN

    # Verify no evidence is counted.
    assert result.sources_used == 0

    # Verify confidence.
    assert result.confidence == 0.0


def test_strong_bullish_fusion() -> None:
    """Strong positive evidence should produce LONG."""

    # Create the production engine.
    engine = SignalFusionIntelligence()

    # Supply strongly bullish evidence.
    evidence = [
        make_evidence("setup", 100.0),
        make_evidence("entry", 100.0),
        make_evidence("confluence", 80.0),
    ]

    # Analyze the evidence.
    result = engine.analyze(
        evidence=evidence,
        decision_timestamp=DECISION_TIME,
    )

    # Verify bullish direction.
    assert result.direction == FusionDirection.LONG

    # Verify decision.
    assert result.decision == FusionDecision.LONG

    # Verify positive score.
    assert result.score > 60.0

    # Verify evidence count.
    assert result.sources_used == 3


def test_strong_bearish_fusion() -> None:
    """Strong negative evidence should produce SHORT."""

    # Create the production engine.
    engine = SignalFusionIntelligence()

    # Supply strongly bearish evidence.
    evidence = [
        make_evidence("setup", -100.0),
        make_evidence("entry", -100.0),
        make_evidence("confluence", -80.0),
    ]

    # Analyze the evidence.
    result = engine.analyze(
        evidence=evidence,
        decision_timestamp=DECISION_TIME,
    )

    # Verify bearish direction.
    assert result.direction == FusionDirection.SHORT

    # Verify decision.
    assert result.decision == FusionDecision.SHORT

    # Verify negative score.
    assert result.score < -60.0


def test_conflicting_evidence_can_be_neutral() -> None:
    """Balanced opposing evidence should produce NEUTRAL."""

    # Create the production engine.
    engine = SignalFusionIntelligence()

    # Supply equal bullish and bearish evidence.
    evidence = [
        make_evidence("setup", 100.0),
        make_evidence("entry", -100.0),
    ]

    # Analyze the evidence.
    result = engine.analyze(
        evidence=evidence,
        decision_timestamp=DECISION_TIME,
    )

    # Verify neutral direction.
    assert result.direction == FusionDirection.NEUTRAL

    # Verify neutral decision.
    assert result.decision == FusionDecision.NEUTRAL

    # Verify exact zero after normalization.
    assert result.score == 0.0


def test_risk_veto_blocks_fusion() -> None:
    """Risk veto must override a bullish fusion result."""

    # Create the production engine.
    engine = SignalFusionIntelligence()

    # Supply strongly bullish evidence.
    evidence = [
        make_evidence("setup", 100.0),
        make_evidence("entry", 100.0),
        make_evidence("confluence", 100.0),
    ]

    # Analyze with an explicit risk veto.
    result = engine.analyze(
        evidence=evidence,
        decision_timestamp=DECISION_TIME,
        risk_blocked=True,
    )

    # Directional context remains available.
    assert result.direction == FusionDirection.LONG

    # Final decision must nevertheless be BLOCKED.
    assert result.decision == FusionDecision.BLOCKED


def test_low_confidence_produces_unknown() -> None:
    """Insufficient evidence should not create a signal."""

    # Require complete evidence coverage.
    engine = SignalFusionIntelligence(
        min_confidence=100.0,
    )

    # Supply only one source.
    evidence = [
        make_evidence("setup", 100.0),
    ]

    # Analyze the evidence.
    result = engine.analyze(
        evidence=evidence,
        decision_timestamp=DECISION_TIME,
    )

    # The direction must remain unknown.
    assert result.direction == FusionDirection.UNKNOWN

    # The decision must remain unknown.
    assert result.decision == FusionDecision.UNKNOWN

    # Confidence is below 100%.
    assert result.confidence < 100.0

    # Evidence is insufficient.
    assert result.sufficient_data is False


def test_symbol_validation_rejects_non_xauusd() -> None:
    """The current fusion engine should reject unsupported symbols."""

    # Create the production engine.
    engine = SignalFusionIntelligence()

    # Verify unsupported symbol rejection.
    with pytest.raises(SignalFusionIntelligenceError):
        engine.analyze(
            evidence=[],
            decision_timestamp=DECISION_TIME,
            symbol="EURUSD",
        )


def test_naive_timestamp_is_rejected() -> None:
    """Decision timestamps must be timezone-aware."""

    # Create the production engine.
    engine = SignalFusionIntelligence()

    # Create a naive timestamp.
    naive_timestamp = datetime(
        2026,
        9,
        5,
        12,
        0,
    )

    # Verify validation.
    with pytest.raises(SignalFusionIntelligenceError):
        engine.analyze(
            evidence=[],
            decision_timestamp=naive_timestamp,
        )


def test_invalid_score_is_rejected() -> None:
    """Scores outside the normalized range must be rejected."""

    # Create the production engine.
    engine = SignalFusionIntelligence()

    # Create invalid evidence.
    evidence = [
        make_evidence("setup", 101.0),
    ]

    # Verify invalid score rejection.
    with pytest.raises(SignalFusionIntelligenceError):
        engine.analyze(
            evidence=evidence,
            decision_timestamp=DECISION_TIME,
        )


def test_unknown_source_is_rejected() -> None:
    """Unknown fusion sources must be rejected."""

    # Create the production engine.
    engine = SignalFusionIntelligence()

    # Create unsupported source evidence.
    evidence = [
        make_evidence("random_source", 100.0),
    ]

    # Verify source validation.
    with pytest.raises(SignalFusionIntelligenceError):
        engine.analyze(
            evidence=evidence,
            decision_timestamp=DECISION_TIME,
        )


def test_custom_weights_are_applied() -> None:
    """Configured source weights should affect the final score."""

    # Give setup twice the weight of entry.
    engine = SignalFusionIntelligence(
        weights={
            "setup": 2.0,
            "entry": 1.0,
        },
    )

    # Setup is bullish while entry is bearish.
    evidence = [
        make_evidence("setup", 100.0),
        make_evidence("entry", -100.0),
    ]

    # Analyze the evidence.
    result = engine.analyze(
        evidence=evidence,
        decision_timestamp=DECISION_TIME,
    )

    # Weighted score should be positive.
    assert result.score > 0.0

    # Direction should therefore be bullish.
    assert result.direction == FusionDirection.LONG


def test_to_dict_is_serializable() -> None:
    """The result should serialize into primitive values."""

    # Create the production engine.
    engine = SignalFusionIntelligence()

    # Supply one evidence item.
    result = engine.analyze(
        evidence=[
            make_evidence("setup", 50.0),
        ],
        decision_timestamp=DECISION_TIME,
    )

    # Serialize the result.
    data = result.to_dict()

    # Verify dictionary output.
    assert isinstance(data, dict)

    # Verify primitive enum values.
    assert isinstance(data["direction"], str)
    assert isinstance(data["decision"], str)

    # Verify evidence serialization.
    assert isinstance(data["evidence"], list)


def test_exact_decision_timestamp_is_allowed() -> None:
    """Evidence is valid when evaluated exactly at its decision timestamp."""

    # Create the production engine.
    engine = SignalFusionIntelligence()

    # Analyze normally at the exact decision timestamp.
    result = engine.analyze(
        evidence=[
            make_evidence("setup", 100.0),
        ],
        decision_timestamp=DECISION_TIME,
    )

    # Evidence should remain usable.
    assert result.sources_used == 1

    # The result should be directional.
    assert result.direction == FusionDirection.LONG


def test_xauusd_wrapper_works() -> None:
    """The XAUUSD convenience wrapper should delegate correctly."""

    # Create the production engine.
    engine = SignalFusionIntelligence()

    # Analyze through the convenience wrapper.
    result = engine.analyze_xauusd(
        evidence=[
            make_evidence("setup", 100.0),
        ],
        decision_timestamp=DECISION_TIME,
    )

    # Verify symbol.
    assert result.symbol == "XAUUSD"

    # Verify directional result.
    assert result.direction == FusionDirection.LONG