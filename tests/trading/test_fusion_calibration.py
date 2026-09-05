"""
Tests for P2.20.3 fusion direction and confidence calibration.
"""

from datetime import datetime, timezone

import pytest

from app.trading.fusion.signal_fusion_intelligence import (
    FusionDirection,
    SignalFusionEvidence,
)

from app.trading.fusion.fusion_calibration import (
    CalibrationStrength,
    FusionCalibrationAssessment,
    FusionCalibrationError,
    FusionCalibrationIntelligence,
)


DECISION_TIME = datetime(
    2026,
    1,
    1,
    12,
    0,
    tzinfo=timezone.utc,
)


def evidence(
    source: str,
    score: float,
    direction: FusionDirection,
    weight: float = 1.0,
) -> SignalFusionEvidence:
    """Create canonical fusion evidence for testing."""

    # Build evidence using the existing P2.20.1 contract.
    return SignalFusionEvidence(
        source=source,
        direction=direction,
        score=score,
        weight=weight,
        reason=f"{source} test evidence",
    )


def test_strong_long_evidence_is_calibrated() -> None:
    """Strong aligned bullish evidence should produce LONG."""

    # Create a calibration engine.
    engine = FusionCalibrationIntelligence()

    # Supply strong aligned evidence from multiple sources.
    result = engine.analyze(
        (
            evidence("setup", 90.0, FusionDirection.LONG),
            evidence("entry", 80.0, FusionDirection.LONG),
            evidence("macro", 70.0, FusionDirection.LONG),
            evidence("confluence", 85.0, FusionDirection.LONG),
        ),
        decision_timestamp=DECISION_TIME,
    )

    # The calibrated direction should be LONG.
    assert result.direction == FusionDirection.LONG

    # Confidence should be high.
    assert result.confidence > 80.0

    # Directional strength should be high.
    assert result.directional_strength > 70.0

    # The evidence should be considered sufficient.
    assert result.sufficient_data is True

    # Strength should be very strong.
    assert result.strength == CalibrationStrength.VERY_STRONG


def test_strong_short_evidence_is_calibrated() -> None:
    """Strong aligned bearish evidence should produce SHORT."""

    # Create the calibration engine.
    engine = FusionCalibrationIntelligence()

    # Supply strongly aligned short evidence.
    result = engine.analyze(
        (
            evidence("setup", -90.0, FusionDirection.SHORT),
            evidence("entry", -80.0, FusionDirection.SHORT),
            evidence("macro", -70.0, FusionDirection.SHORT),
            evidence("confluence", -85.0, FusionDirection.SHORT),
        ),
        decision_timestamp=DECISION_TIME,
    )

    # Verify the short direction.
    assert result.direction == FusionDirection.SHORT

    # Verify high confidence.
    assert result.confidence > 80.0

    # Verify sufficient data.
    assert result.sufficient_data is True


def test_conflicting_evidence_reduces_agreement() -> None:
    """Conflicting evidence should reduce agreement."""

    # Create the calibration engine.
    engine = FusionCalibrationIntelligence()

    # Supply both long and short evidence.
    result = engine.analyze(
        (
            evidence("setup", 90.0, FusionDirection.LONG),
            evidence("entry", -80.0, FusionDirection.SHORT),
            evidence("macro", 70.0, FusionDirection.LONG),
            evidence("news", -60.0, FusionDirection.SHORT),
        ),
        decision_timestamp=DECISION_TIME,
    )

    # Agreement must be below perfect agreement.
    assert result.agreement < 100.0


def test_perfect_agreement_is_detected() -> None:
    """Aligned evidence should produce full directional agreement."""

    # Create the calibration engine.
    engine = FusionCalibrationIntelligence()

    # Supply entirely aligned evidence.
    result = engine.analyze(
        (
            evidence("setup", 80.0, FusionDirection.LONG),
            evidence("entry", 70.0, FusionDirection.LONG),
            evidence("macro", 60.0, FusionDirection.LONG),
        ),
        decision_timestamp=DECISION_TIME,
    )

    # All directional evidence agrees.
    assert result.agreement == 100.0


def test_low_coverage_becomes_unknown() -> None:
    """Insufficient evidence coverage should produce UNKNOWN."""

    # Require complete evidence coverage.
    engine = FusionCalibrationIntelligence(
        min_coverage=100.0,
    )

    # Supply only one source.
    result = engine.analyze(
        (
            evidence("setup", 90.0, FusionDirection.LONG),
        ),
        decision_timestamp=DECISION_TIME,
    )

    # The final direction must not be treated as reliable.
    assert result.direction == FusionDirection.UNKNOWN

    # The result must report insufficient data.
    assert result.sufficient_data is False

    # Strength must also become UNKNOWN.
    assert result.strength == CalibrationStrength.UNKNOWN


def test_low_confidence_becomes_unknown() -> None:
    """Strict confidence requirements should produce UNKNOWN."""

    # Require 100% confidence.
    engine = FusionCalibrationIntelligence(
        min_confidence=100.0,
    )

    # Supply moderate evidence that cannot reach perfect confidence.
    result = engine.analyze(
        (
            evidence("setup", 40.0, FusionDirection.LONG),
        ),
        decision_timestamp=DECISION_TIME,
    )

    # Final direction must be UNKNOWN.
    assert result.direction == FusionDirection.UNKNOWN

    # Sufficient-data flag must be false.
    assert result.sufficient_data is False


def test_empty_evidence_is_unknown() -> None:
    """Empty evidence must never create a directional signal."""

    # Create the calibration engine.
    engine = FusionCalibrationIntelligence()

    # Analyze an empty evidence collection.
    result = engine.analyze(
        (),
        decision_timestamp=DECISION_TIME,
    )

    # No evidence means UNKNOWN.
    assert result.direction == FusionDirection.UNKNOWN

    # No evidence means insufficient data.
    assert result.sufficient_data is False

    # No directional strength exists.
    assert result.directional_strength == 0.0

    # No coverage exists.
    assert result.coverage == 0.0


def test_zero_weight_evidence_does_not_create_coverage() -> None:
    """Zero-weight evidence should not contribute to coverage."""

    # Create the calibration engine.
    engine = FusionCalibrationIntelligence()

    # Supply only zero-weight evidence.
    result = engine.analyze(
        (
            evidence(
                "setup",
                100.0,
                FusionDirection.LONG,
                weight=0.0,
            ),
        ),
        decision_timestamp=DECISION_TIME,
    )

    # No usable weight means no sufficient data.
    assert result.coverage == 0.0
    assert result.sufficient_data is False


def test_neutral_score_is_neutral_when_sufficient() -> None:
    """Balanced evidence should remain neutral."""

    # Disable minimum coverage so the test focuses on direction.
    engine = FusionCalibrationIntelligence(
        min_coverage=0.0,
    )

    # Supply perfectly opposing evidence.
    result = engine.analyze(
        (
            evidence("setup", 80.0, FusionDirection.LONG),
            evidence("entry", -80.0, FusionDirection.SHORT),
        ),
        decision_timestamp=DECISION_TIME,
    )

    # Net score is neutral.
    assert result.calibrated_score == 0.0

    # Direction should be neutral.
    assert result.direction == FusionDirection.NEUTRAL


def test_weighted_evidence_changes_score() -> None:
    """Higher-weight evidence should influence the calibrated score more."""

    # Create the calibration engine.
    engine = FusionCalibrationIntelligence()

    # Give bullish evidence substantially more weight.
    result = engine.analyze(
        (
            evidence(
                "setup",
                100.0,
                FusionDirection.LONG,
                weight=2.0,
            ),
            evidence(
                "entry",
                -50.0,
                FusionDirection.SHORT,
                weight=1.0,
            ),
        ),
        decision_timestamp=DECISION_TIME,
    )

    # Weighted score should be positive.
    assert result.calibrated_score > 0.0

    # Direction should therefore be LONG.
    assert result.direction == FusionDirection.LONG


def test_exact_decision_timestamp_is_allowed() -> None:
    """Evidence at the exact decision timestamp must be accepted."""

    # Create the calibration engine.
    engine = FusionCalibrationIntelligence()

    # Analyze evidence at the exact decision timestamp.
    result = engine.analyze(
        (
            evidence("setup", 80.0, FusionDirection.LONG),
        ),
        decision_timestamp=DECISION_TIME,
    )

    # The timestamp itself must be preserved.
    assert result.decision_timestamp == DECISION_TIME


def test_naive_timestamp_is_rejected() -> None:
    """Naive timestamps must be rejected."""

    # Create the calibration engine.
    engine = FusionCalibrationIntelligence()

    # Attempt to analyze using a naive datetime.
    with pytest.raises(FusionCalibrationError):
        engine.analyze(
            (
                evidence("setup", 80.0, FusionDirection.LONG),
            ),
            decision_timestamp=datetime(
                2026,
                1,
                1,
                12,
                0,
            ),
        )


def test_invalid_evidence_type_is_rejected() -> None:
    """Non-canonical evidence objects must be rejected."""

    # Create the calibration engine.
    engine = FusionCalibrationIntelligence()

    # Attempt to supply an invalid evidence object.
    with pytest.raises(FusionCalibrationError):
        engine.analyze(
            ("invalid",),
            decision_timestamp=DECISION_TIME,
        )


def test_invalid_configuration_is_rejected() -> None:
    """Invalid calibration thresholds must be rejected."""

    # Minimum coverage cannot exceed 100.
    with pytest.raises(FusionCalibrationError):
        FusionCalibrationIntelligence(
            min_coverage=101.0,
        )

    # Strong threshold must exceed moderate threshold.
    with pytest.raises(FusionCalibrationError):
        FusionCalibrationIntelligence(
            strong_threshold=20.0,
            moderate_threshold=20.0,
        )


def test_xauusd_wrapper_matches_analyze() -> None:
    """The XAU/USD wrapper should use the same deterministic calibration."""

    # Create the calibration engine.
    engine = FusionCalibrationIntelligence()

    # Create reusable evidence.
    inputs = (
        evidence("setup", 80.0, FusionDirection.LONG),
        evidence("entry", 70.0, FusionDirection.LONG),
        evidence("macro", 60.0, FusionDirection.LONG),
    )

    # Analyze through the generic method.
    generic = engine.analyze(
        inputs,
        decision_timestamp=DECISION_TIME,
    )

    # Analyze through the XAU/USD wrapper.
    xauusd = engine.analyze_xauusd(
        inputs,
        decision_timestamp=DECISION_TIME,
    )

    # Both results must be equivalent.
    assert xauusd == generic


def test_assessment_serialization_is_deterministic() -> None:
    """Calibration assessments should serialize deterministically."""

    # Create the calibration engine.
    engine = FusionCalibrationIntelligence()

    # Generate an assessment.
    result = engine.analyze(
        (
            evidence("setup", 80.0, FusionDirection.LONG),
            evidence("entry", 70.0, FusionDirection.LONG),
        ),
        decision_timestamp=DECISION_TIME,
    )

    # Serialize the assessment.
    payload = result.to_dict()

    # Verify core serialized fields.
    assert payload["direction"] == "long"
    assert payload["calibrated_score"] > 0.0
    assert payload["confidence"] >= 0.0
    assert payload["decision_timestamp"] == DECISION_TIME.isoformat()


def test_assessment_rejects_invalid_direction() -> None:
    """Assessment validation must enforce the canonical enum."""

    # Attempt to construct an invalid assessment directly.
    with pytest.raises(FusionCalibrationError):
        FusionCalibrationAssessment(
            direction="LONG",
            strength=CalibrationStrength.STRONG,
            calibrated_score=80.0,
            confidence=80.0,
            directional_strength=80.0,
            agreement=100.0,
            coverage=100.0,
            total_weight=1.0,
            used_weight=1.0,
            sources_used=1,
            sufficient_data=True,
            decision_timestamp=DECISION_TIME,
            reasons=("test",),
        )