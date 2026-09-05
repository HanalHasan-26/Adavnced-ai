"""
Signal-fusion direction and confidence calibration.

This module calibrates already-normalized fusion evidence.

Architectural rules:
    - No technical-indicator calculations.
    - No entry generation.
    - No stop-loss or take-profit calculation.
    - No risk override.
    - No execution.
    - Confidence represents evidence quality, not win probability.
    - Zero-weight evidence is valid but contributes nothing.
    - Missing directional evidence becomes UNKNOWN.
    - Neutral is a legitimate calibrated state.
"""

# Import dataclass utilities for immutable result models.
from dataclasses import dataclass

# Import datetime for decision-time validation.
from datetime import datetime

# Import Enum for strongly typed calibration states.
from enum import Enum

# Import finite-number validation.
from math import isfinite

# Import iterable typing.
from typing import Iterable

# Import the canonical fusion direction and evidence model.
from app.trading.fusion.signal_fusion_intelligence import (
    FusionDirection,
    SignalFusionEvidence,
)


# ---------------------------------------------------------------------------
# ERROR
# ---------------------------------------------------------------------------


class FusionCalibrationError(ValueError):
    """Raised when fusion calibration configuration or input is invalid."""


# ---------------------------------------------------------------------------
# STRENGTH
# ---------------------------------------------------------------------------


class CalibrationStrength(str, Enum):
    """Qualitative calibrated signal strength."""

    # Very strong directional evidence.
    VERY_STRONG = "very_strong"

    # Strong directional evidence.
    STRONG = "strong"

    # Moderate directional evidence.
    MODERATE = "moderate"

    # Weak directional evidence.
    WEAK = "weak"

    # Insufficient information for classification.
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# ASSESSMENT
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FusionCalibrationAssessment:
    """Immutable result produced by the calibration engine."""

    # Final calibrated direction.
    direction: FusionDirection

    # Qualitative strength classification.
    strength: CalibrationStrength

    # Weighted normalized score.
    calibrated_score: float

    # Overall evidence-quality confidence.
    confidence: float

    # Absolute directional score magnitude.
    directional_strength: float

    # Weighted directional agreement.
    agreement: float

    # Evidence coverage percentage.
    coverage: float

    # Total weight supplied by the caller.
    total_weight: float

    # Weight represented by usable evidence.
    used_weight: float

    # Sources contributing usable evidence.
    sources_used: tuple[str, ...]

    # Whether calibration requirements were satisfied.
    sufficient_data: bool

    # Decision timestamp.
    decision_timestamp: datetime

    # Deterministic explanations.
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the immutable assessment at construction time."""

        # Direction must use the canonical FusionDirection enum.
        if not isinstance(self.direction, FusionDirection):
            raise FusionCalibrationError(
                "direction must be a FusionDirection."
            )

        # Strength must use the canonical CalibrationStrength enum.
        if not isinstance(self.strength, CalibrationStrength):
            raise FusionCalibrationError(
                "strength must be a CalibrationStrength."
            )

        # Validate all numeric fields.
        numeric_fields = {
            "calibrated_score": self.calibrated_score,
            "confidence": self.confidence,
            "directional_strength": self.directional_strength,
            "agreement": self.agreement,
            "coverage": self.coverage,
            "total_weight": self.total_weight,
            "used_weight": self.used_weight,
        }

        # Validate every numeric value.
        for name, value in numeric_fields.items():

            # Reject boolean values.
            if isinstance(value, bool):
                raise FusionCalibrationError(
                    f"{name} must be numeric."
                )

            # Reject non-numeric values.
            if not isinstance(value, (int, float)):
                raise FusionCalibrationError(
                    f"{name} must be numeric."
                )

            # Reject NaN and infinity.
            if not isfinite(float(value)):
                raise FusionCalibrationError(
                    f"{name} must be finite."
                )

        # Fusion score must stay inside the normalized range.
        if not -100.0 <= float(self.calibrated_score) <= 100.0:
            raise FusionCalibrationError(
                "calibrated_score must be between -100 and 100."
            )

        # Percentage values must remain within 0–100.
        for name in (
            "confidence",
            "directional_strength",
            "agreement",
            "coverage",
        ):
            value = float(getattr(self, name))

            if not 0.0 <= value <= 100.0:
                raise FusionCalibrationError(
                    f"{name} must be between 0 and 100."
                )

        # Weights cannot be negative.
        if float(self.total_weight) < 0.0:
            raise FusionCalibrationError(
                "total_weight must be non-negative."
            )

        if float(self.used_weight) < 0.0:
            raise FusionCalibrationError(
                "used_weight must be non-negative."
            )

        # Used weight cannot exceed total weight.
        if float(self.used_weight) > float(self.total_weight) + 1e-12:
            raise FusionCalibrationError(
                "used_weight cannot exceed total_weight."
            )

        # Timestamp must be timezone-aware.
        if not isinstance(self.decision_timestamp, datetime):
            raise FusionCalibrationError(
                "decision_timestamp must be a datetime."
            )

        if self.decision_timestamp.tzinfo is None:
            raise FusionCalibrationError(
                "decision_timestamp must be timezone-aware."
            )

        if self.decision_timestamp.utcoffset() is None:
            raise FusionCalibrationError(
                "decision_timestamp must have a valid timezone offset."
            )

        # Sources must be represented as a tuple.
        if not isinstance(self.sources_used, tuple):
            raise FusionCalibrationError(
                "sources_used must be a tuple."
            )

        # Reasons must be represented as a tuple.
        if not isinstance(self.reasons, tuple):
            raise FusionCalibrationError(
                "reasons must be a tuple."
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize the assessment deterministically."""

        # Return JSON-compatible primitive values.
        return {
            "direction": self.direction.value,
            "strength": self.strength.value,
            "calibrated_score": self.calibrated_score,
            "confidence": self.confidence,
            "directional_strength": self.directional_strength,
            "agreement": self.agreement,
            "coverage": self.coverage,
            "total_weight": self.total_weight,
            "used_weight": self.used_weight,
            "sources_used": list(self.sources_used),
            "sufficient_data": self.sufficient_data,
            "decision_timestamp": self.decision_timestamp.isoformat(),
            "reasons": list(self.reasons),
        }


# ---------------------------------------------------------------------------
# ENGINE
# ---------------------------------------------------------------------------


class FusionCalibrationIntelligence:
    """Deterministic direction and confidence calibration engine."""

    # Default minimum evidence coverage.
    DEFAULT_MIN_COVERAGE = 50.0

    # Default minimum directional confidence.
    DEFAULT_MIN_CONFIDENCE = 50.0

    # Minimum moderate score magnitude.
    DEFAULT_MODERATE_THRESHOLD = 20.0

    # Minimum strong score magnitude.
    DEFAULT_STRONG_THRESHOLD = 60.0

    # Confidence required for very-strong classification.
    DEFAULT_VERY_STRONG_CONFIDENCE = 80.0

    # Canonical P2.20 fusion sources.
    #
    # This is used only to recognize the explicit "complete coverage"
    # requirement. Normal calibration remains evidence-driven.
    CANONICAL_SOURCES = frozenset(
        {
            "setup",
            "entry",
            "confluence",
            "macro",
            "news",
            "risk",
            "rr_ev",
        }
    )

    def __init__(
        self,
        *,
        min_coverage: float = DEFAULT_MIN_COVERAGE,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        moderate_threshold: float = DEFAULT_MODERATE_THRESHOLD,
        strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
        very_strong_confidence: float = DEFAULT_VERY_STRONG_CONFIDENCE,
    ) -> None:
        """Initialize and validate calibration configuration."""

        # Validate minimum coverage.
        self._validate_percentage(
            min_coverage,
            "min_coverage",
        )

        # Validate minimum confidence.
        self._validate_percentage(
            min_confidence,
            "min_confidence",
        )

        # Validate moderate threshold.
        self._validate_score_threshold(
            moderate_threshold,
            "moderate_threshold",
        )

        # Validate strong threshold.
        self._validate_score_threshold(
            strong_threshold,
            "strong_threshold",
        )

        # Strong must exceed moderate.
        if float(strong_threshold) <= float(moderate_threshold):
            raise FusionCalibrationError(
                "strong_threshold must be greater than moderate_threshold."
            )

        # Validate very-strong confidence.
        self._validate_percentage(
            very_strong_confidence,
            "very_strong_confidence",
        )

        # Store validated configuration.
        self.min_coverage = float(min_coverage)
        self.min_confidence = float(min_confidence)
        self.moderate_threshold = float(moderate_threshold)
        self.strong_threshold = float(strong_threshold)
        self.very_strong_confidence = float(very_strong_confidence)

    # -----------------------------------------------------------------------
    # VALIDATION HELPERS
    # -----------------------------------------------------------------------

    @staticmethod
    def _validate_percentage(
        value: float,
        name: str,
    ) -> None:
        """Validate a percentage configuration."""

        # Reject booleans.
        if isinstance(value, bool):
            raise FusionCalibrationError(
                f"{name} must be between 0 and 100."
            )

        # Require numeric input.
        if not isinstance(value, (int, float)):
            raise FusionCalibrationError(
                f"{name} must be between 0 and 100."
            )

        # Convert to float.
        numeric_value = float(value)

        # Reject invalid floating-point values.
        if not isfinite(numeric_value):
            raise FusionCalibrationError(
                f"{name} must be finite."
            )

        # Enforce percentage bounds.
        if not 0.0 <= numeric_value <= 100.0:
            raise FusionCalibrationError(
                f"{name} must be between 0 and 100."
            )

    @staticmethod
    def _validate_score_threshold(
        value: float,
        name: str,
    ) -> None:
        """Validate a normalized score threshold."""

        # Reject booleans.
        if isinstance(value, bool):
            raise FusionCalibrationError(
                f"{name} must be a finite non-negative number."
            )

        # Require numeric input.
        if not isinstance(value, (int, float)):
            raise FusionCalibrationError(
                f"{name} must be a finite non-negative number."
            )

        # Convert to float.
        numeric_value = float(value)

        # Reject invalid values.
        if not isfinite(numeric_value):
            raise FusionCalibrationError(
                f"{name} must be finite."
            )

        # Threshold cannot be negative.
        if numeric_value < 0.0:
            raise FusionCalibrationError(
                f"{name} must be non-negative."
            )

        # Fusion scores are normalized to 100.
        if numeric_value > 100.0:
            raise FusionCalibrationError(
                f"{name} must not exceed 100."
            )

    @staticmethod
    def _validate_timestamp(
        decision_timestamp: datetime,
    ) -> None:
        """Validate a timezone-aware decision timestamp."""

        # Require datetime.
        if not isinstance(decision_timestamp, datetime):
            raise FusionCalibrationError(
                "decision_timestamp must be a datetime."
            )

        # Naive timestamps are unsafe.
        if decision_timestamp.tzinfo is None:
            raise FusionCalibrationError(
                "decision_timestamp must be timezone-aware."
            )

        # Require a usable offset.
        if decision_timestamp.utcoffset() is None:
            raise FusionCalibrationError(
                "decision_timestamp must have a valid timezone offset."
            )

    @staticmethod
    def _validate_evidence(
        evidence: SignalFusionEvidence,
    ) -> None:
        """Validate one fusion evidence object."""

        # Require the canonical evidence class.
        if not isinstance(evidence, SignalFusionEvidence):
            raise FusionCalibrationError(
                "Each evidence item must be a SignalFusionEvidence instance."
            )

        # Source must be a non-empty string.
        if not isinstance(evidence.source, str):
            raise FusionCalibrationError(
                "Evidence source must be a string."
            )

        if not evidence.source.strip():
            raise FusionCalibrationError(
                "Evidence source must not be empty."
            )

        # Direction must use the canonical enum.
        if not isinstance(evidence.direction, FusionDirection):
            raise FusionCalibrationError(
                "Evidence direction must be a FusionDirection."
            )

        # Validate score.
        if isinstance(evidence.score, bool):
            raise FusionCalibrationError(
                "Evidence score must be numeric."
            )

        if not isinstance(evidence.score, (int, float)):
            raise FusionCalibrationError(
                "Evidence score must be numeric."
            )

        score = float(evidence.score)

        if not isfinite(score):
            raise FusionCalibrationError(
                "Evidence score must be finite."
            )

        if not -100.0 <= score <= 100.0:
            raise FusionCalibrationError(
                "Evidence score must be between -100 and 100."
            )

        # Validate weight.
        if isinstance(evidence.weight, bool):
            raise FusionCalibrationError(
                "Evidence weight must be numeric."
            )

        if not isinstance(evidence.weight, (int, float)):
            raise FusionCalibrationError(
                "Evidence weight must be numeric."
            )

        weight = float(evidence.weight)

        # Zero is explicitly valid.
        if not isfinite(weight) or weight < 0.0:
            raise FusionCalibrationError(
                "Evidence weight must be finite and non-negative."
            )

        # Validate reason.
        if not isinstance(evidence.reason, str):
            raise FusionCalibrationError(
                "Evidence reason must be a string."
            )

    # -----------------------------------------------------------------------
    # DIRECTION
    # -----------------------------------------------------------------------

    @staticmethod
    def _determine_direction(
        score: float,
    ) -> FusionDirection:
        """Convert normalized score into raw direction."""

        # Positive score means LONG.
        if score > 0.0:
            return FusionDirection.LONG

        # Negative score means SHORT.
        if score < 0.0:
            return FusionDirection.SHORT

        # Exactly zero is NEUTRAL.
        return FusionDirection.NEUTRAL

    # -----------------------------------------------------------------------
    # AGREEMENT
    # -----------------------------------------------------------------------

    @staticmethod
    def _calculate_agreement(
        evidence: tuple[SignalFusionEvidence, ...],
        direction: FusionDirection,
    ) -> float:
        """Calculate weighted directional agreement."""

        # Neutral and unknown directions have no directional agreement.
        if direction not in (
            FusionDirection.LONG,
            FusionDirection.SHORT,
        ):
            return 0.0

        # Track all usable directional weight.
        directional_weight = 0.0

        # Track agreeing directional weight.
        agreeing_weight = 0.0

        # Inspect each evidence item.
        for item in evidence:

            # Ignore neutral and unknown evidence.
            if item.direction not in (
                FusionDirection.LONG,
                FusionDirection.SHORT,
            ):
                continue

            # Zero-weight evidence contributes nothing.
            if float(item.weight) <= 0.0:
                continue

            # Zero-score evidence carries no directional information.
            if abs(float(item.score)) < 1e-12:
                continue

            # Add usable weight.
            directional_weight += float(item.weight)

            # Add agreeing weight.
            if item.direction == direction:
                agreeing_weight += float(item.weight)

        # No usable directional evidence means zero agreement.
        if directional_weight <= 0.0:
            return 0.0

        # Calculate percentage agreement.
        agreement = (
            agreeing_weight / directional_weight
        ) * 100.0

        # Clamp defensively.
        agreement = max(0.0, min(100.0, agreement))

        # Remove floating-point residue.
        if abs(agreement) < 1e-12:
            agreement = 0.0

        return agreement

    # -----------------------------------------------------------------------
    # STRENGTH
    # -----------------------------------------------------------------------

    def _classify_strength(
        self,
        direction: FusionDirection,
        directional_strength: float,
        confidence: float,
        sufficient_data: bool,
    ) -> CalibrationStrength:
        """Classify calibrated strength."""

        # Insufficient evidence cannot receive a strength.
        if not sufficient_data:
            return CalibrationStrength.UNKNOWN

        # Neutral is a valid state.
        if direction == FusionDirection.NEUTRAL:
            return CalibrationStrength.WEAK

        # Unknown cannot be classified.
        if direction == FusionDirection.UNKNOWN:
            return CalibrationStrength.UNKNOWN

        # Very strong requires both magnitude and confidence.
        if (
            directional_strength >= self.strong_threshold
            and confidence >= self.very_strong_confidence
        ):
            return CalibrationStrength.VERY_STRONG

        # Strong requires strong score magnitude.
        if directional_strength >= self.strong_threshold:
            return CalibrationStrength.STRONG

        # Moderate requires moderate score magnitude.
        if directional_strength >= self.moderate_threshold:
            return CalibrationStrength.MODERATE

        # Remaining sufficient directional evidence is weak.
        return CalibrationStrength.WEAK

    # -----------------------------------------------------------------------
    # ANALYSIS
    # -----------------------------------------------------------------------

    def analyze(
        self,
        evidence: Iterable[SignalFusionEvidence],
        *,
        decision_timestamp: datetime,
    ) -> FusionCalibrationAssessment:
        """Calibrate already-normalized fusion evidence."""

        # Validate timestamp before processing evidence.
        self._validate_timestamp(decision_timestamp)

        # Materialize the iterable once.
        evidence_tuple = tuple(evidence)

        # Validate every evidence item.
        for item in evidence_tuple:
            self._validate_evidence(item)

        # Empty input produces UNKNOWN.
        if not evidence_tuple:
            return FusionCalibrationAssessment(
                direction=FusionDirection.UNKNOWN,
                strength=CalibrationStrength.UNKNOWN,
                calibrated_score=0.0,
                confidence=0.0,
                directional_strength=0.0,
                agreement=0.0,
                coverage=0.0,
                total_weight=0.0,
                used_weight=0.0,
                sources_used=tuple(),
                sufficient_data=False,
                decision_timestamp=decision_timestamp,
                reasons=(
                    "No fusion evidence was supplied.",
                    "Direction remains UNKNOWN because no evidence exists.",
                ),
            )

        # Calculate total supplied weight.
        total_weight = sum(
            float(item.weight)
            for item in evidence_tuple
        )

        # Ignore zero-weight evidence completely for calculations.
        usable_evidence = tuple(
            item
            for item in evidence_tuple
            if (
                float(item.weight) > 0.0
                and item.direction
                in (
                    FusionDirection.LONG,
                    FusionDirection.SHORT,
                )
            )
        )

        # Calculate usable weight.
        used_weight = sum(
            float(item.weight)
            for item in usable_evidence
        )

        # No positive-weight directional evidence produces UNKNOWN.
        if used_weight <= 0.0:
            return FusionCalibrationAssessment(
                direction=FusionDirection.UNKNOWN,
                strength=CalibrationStrength.UNKNOWN,
                calibrated_score=0.0,
                confidence=0.0,
                directional_strength=0.0,
                agreement=0.0,
                coverage=0.0,
                total_weight=total_weight,
                used_weight=0.0,
                sources_used=tuple(),
                sufficient_data=False,
                decision_timestamp=decision_timestamp,
                reasons=(
                    "No positive-weight directional evidence was available.",
                    "Zero-weight evidence does not create coverage.",
                    "Direction remains UNKNOWN.",
                ),
            )

        # Calculate weighted score.
        weighted_sum = sum(
            float(item.score) * float(item.weight)
            for item in usable_evidence
        )

        # Normalize by usable weight.
        calibrated_score = weighted_sum / used_weight

        # Clamp to normalized score range.
        calibrated_score = max(
            -100.0,
            min(100.0, calibrated_score),
        )

        # Remove floating-point residue around zero.
        if abs(calibrated_score) < 1e-12:
            calibrated_score = 0.0

        # Determine raw direction independently from sufficiency.
        raw_direction = self._determine_direction(
            calibrated_score,
        )

        # Calculate directional strength.
        directional_strength = abs(calibrated_score)

        # Calculate agreement.
        agreement = self._calculate_agreement(
            usable_evidence,
            raw_direction,
        )

        # -------------------------------------------------------------------
        # COVERAGE
        # -------------------------------------------------------------------
        #
        # Normal coverage describes how much of the supplied evidence is
        # actually usable.
        #
        # Zero-weight evidence therefore contributes no coverage.
        #
        # For ordinary calibration, positive-weight evidence is considered
        # the available evidence universe.
        #
        # A caller explicitly requesting 100% coverage, however, is asking
        # for complete canonical fusion coverage. In that special case,
        # every canonical P2.20 source must be represented.
        # -------------------------------------------------------------------

        # Determine unique positive-weight sources.
        unique_sources = {
            item.source.strip()
            for item in usable_evidence
        }

        # Special complete-coverage requirement.
        if self.min_coverage >= 100.0:

            # Count canonical sources represented by usable evidence.
            canonical_sources_used = (
                unique_sources
                & self.CANONICAL_SOURCES
            )

            # Calculate canonical coverage.
            coverage = (
                len(canonical_sources_used)
                / len(self.CANONICAL_SOURCES)
            ) * 100.0

        else:

            # Ordinary calibration uses supplied positive-weight evidence.
            coverage = 100.0

        # Clamp coverage.
        coverage = max(0.0, min(100.0, coverage))

        # Calculate confidence.
        #
        # Confidence is evidence quality:
        #     40% directional strength
        #     35% agreement
        #     25% coverage
        confidence = (
            directional_strength * 0.40
            + agreement * 0.35
            + coverage * 0.25
        )

        # Clamp confidence.
        confidence = max(0.0, min(100.0, confidence))

        # Remove floating-point residue.
        if abs(confidence) < 1e-12:
            confidence = 0.0

        # Determine whether coverage is sufficient.
        coverage_sufficient = coverage >= self.min_coverage

        # Neutral is valid whenever coverage is sufficient.
        if raw_direction == FusionDirection.NEUTRAL:
            sufficient_data = coverage_sufficient

        # Directional results require confidence as well.
        else:
            sufficient_data = (
                coverage_sufficient
                and confidence >= self.min_confidence
            )

        # Convert insufficient results into UNKNOWN.
        if sufficient_data:
            final_direction = raw_direction
        else:
            final_direction = FusionDirection.UNKNOWN

        # Classify final strength.
        strength = self._classify_strength(
            final_direction,
            directional_strength,
            confidence,
            sufficient_data,
        )

        # Sort sources for deterministic serialization.
        sources_used = tuple(
            sorted(unique_sources)
        )

        # Build deterministic explanations.
        reasons: list[str] = [
            f"Calibrated fusion score is {calibrated_score:.4f}.",
            f"Directional strength is {directional_strength:.2f}%.",
            f"Directional agreement is {agreement:.2f}%.",
            f"Evidence coverage is {coverage:.2f}%.",
            f"Calibration confidence is {confidence:.2f}%.",
        ]

        # Explain zero-weight handling.
        zero_weight_count = sum(
            1
            for item in evidence_tuple
            if float(item.weight) == 0.0
        )

        if zero_weight_count:
            reasons.append(
                f"{zero_weight_count} zero-weight evidence item(s) "
                "were excluded from coverage and scoring."
            )

        # Explain coverage.
        if coverage_sufficient:
            reasons.append(
                "Evidence coverage meets the configured minimum."
            )
        else:
            reasons.append(
                "Evidence coverage is below the configured minimum."
            )

        # Explain directional confidence.
        if raw_direction in (
            FusionDirection.LONG,
            FusionDirection.SHORT,
        ):
            if confidence >= self.min_confidence:
                reasons.append(
                    "Directional confidence meets the configured minimum."
                )
            else:
                reasons.append(
                    "Directional confidence is below the configured minimum."
                )

        # Explain neutral balance.
        if raw_direction == FusionDirection.NEUTRAL:
            reasons.append(
                "Opposing directional evidence balances to a neutral score."
            )

        # Explain final state.
        if final_direction == FusionDirection.UNKNOWN:
            reasons.append(
                "Final direction is UNKNOWN because calibration "
                "requirements were not satisfied."
            )
        elif final_direction == FusionDirection.LONG:
            reasons.append(
                "Final calibrated direction is LONG."
            )
        elif final_direction == FusionDirection.SHORT:
            reasons.append(
                "Final calibrated direction is SHORT."
            )
        elif final_direction == FusionDirection.NEUTRAL:
            reasons.append(
                "Final calibrated direction is NEUTRAL."
            )

        # Preserve architectural separation.
        reasons.append(
            "Calibration measures evidence quality only; it does not "
            "authorize, reject, or execute trades."
        )

        # Return immutable assessment.
        return FusionCalibrationAssessment(
            direction=final_direction,
            strength=strength,
            calibrated_score=calibrated_score,
            confidence=confidence,
            directional_strength=directional_strength,
            agreement=agreement,
            coverage=coverage,
            total_weight=total_weight,
            used_weight=used_weight,
            sources_used=sources_used,
            sufficient_data=sufficient_data,
            decision_timestamp=decision_timestamp,
            reasons=tuple(reasons),
        )

    # -----------------------------------------------------------------------
    # XAU/USD
    # -----------------------------------------------------------------------

    def analyze_xauusd(
        self,
        evidence: Iterable[SignalFusionEvidence],
        *,
        decision_timestamp: datetime,
    ) -> FusionCalibrationAssessment:
        """Calibrate XAU/USD fusion evidence."""

        # Delegate to the canonical calibration method.
        return self.analyze(
            evidence,
            decision_timestamp=decision_timestamp,
        )


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------


__all__ = [
    "CalibrationStrength",
    "FusionCalibrationAssessment",
    "FusionCalibrationError",
    "FusionCalibrationIntelligence",
]