"""
Deterministic signal-fusion intelligence.

This module combines normalized evidence from the trading engine into
one deterministic directional assessment.

Architectural rules:
    - This module does not calculate technical indicators.
    - This module does not generate entries.
    - This module does not calculate stop-loss or take-profit.
    - This module does not execute trades.
    - Risk vetoes are authoritative.
    - Directional context can exist with partial evidence.
    - A stricter explicitly configured confidence requirement can suppress
      signal creation when evidence coverage is insufficient.
"""

# Import dataclass helpers for immutable models and serialization.
from dataclasses import dataclass, fields, is_dataclass

# Import datetime for decision-time validation.
from datetime import datetime

# Import Enum for strongly typed fusion states.
from enum import Enum

# Import finite-number validation.
from math import isfinite

# Import iterable and mapping type hints.
from typing import Iterable, Mapping


class SignalFusionIntelligenceError(ValueError):
    """Raised when signal-fusion configuration or input is invalid."""


class FusionDirection(str, Enum):
    """Directional context produced by the fusion engine."""

    # Bullish directional context.
    LONG = "long"

    # Bearish directional context.
    SHORT = "short"

    # Evidence is balanced.
    NEUTRAL = "neutral"

    # No valid directional signal is available.
    UNKNOWN = "unknown"


class FusionDecision(str, Enum):
    """Final fusion decision after applying risk controls."""

    # Bullish decision.
    LONG = "long"

    # Bearish decision.
    SHORT = "short"

    # Neutral decision.
    NEUTRAL = "neutral"

    # No directional decision.
    UNKNOWN = "unknown"

    # Risk veto blocked the result.
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class SignalFusionEvidence:
    """
    One normalized evidence item supplied to the fusion engine.

    Score convention:
        +100 = strongly bullish
        0    = neutral
        -100 = strongly bearish
    """

    # Name of the subsystem producing the evidence.
    source: str

    # Direction represented by the evidence.
    direction: FusionDirection

    # Normalized directional score.
    score: float

    # Per-evidence compatibility weight.
    weight: float

    # Human-readable explanation.
    reason: str

    def __post_init__(self) -> None:
        """Validate the basic evidence structure."""

        # Require a non-empty source name.
        if not isinstance(self.source, str) or not self.source.strip():
            raise SignalFusionIntelligenceError(
                "Evidence source must be a non-empty string."
            )

        # Require the canonical direction enum.
        if not isinstance(self.direction, FusionDirection):
            raise SignalFusionIntelligenceError(
                "Evidence direction must be a FusionDirection."
            )

        # Reject boolean scores.
        if isinstance(self.score, bool):
            raise SignalFusionIntelligenceError(
                "Evidence score must be numeric."
            )

        # Require numeric scores.
        if not isinstance(self.score, (int, float)):
            raise SignalFusionIntelligenceError(
                "Evidence score must be numeric."
            )

        # Reject NaN and infinity.
        if not isfinite(float(self.score)):
            raise SignalFusionIntelligenceError(
                "Evidence score must be finite."
            )

        # IMPORTANT:
        # Do not reject a score outside -100..100 here.
        #
        # The fusion engine is the authoritative validation boundary.
        # This allows tests and callers to construct an evidence object
        # and have analyze() perform the actual acceptance/rejection.

        # Reject boolean evidence weights.
        if isinstance(self.weight, bool):
            raise SignalFusionIntelligenceError(
                "Evidence weight must be numeric."
            )

        # Require numeric evidence weights.
        if not isinstance(self.weight, (int, float)):
            raise SignalFusionIntelligenceError(
                "Evidence weight must be numeric."
            )

        # Reject invalid or negative evidence weights.
        if (
            not isfinite(float(self.weight))
            or float(self.weight) < 0.0
        ):
            raise SignalFusionIntelligenceError(
                "Evidence weight must be finite and non-negative."
            )

        # Reason must be a string.
        if not isinstance(self.reason, str):
            raise SignalFusionIntelligenceError(
                "Evidence reason must be a string."
            )


@dataclass(frozen=True, slots=True)
class SignalFusionAssessment:
    """Complete deterministic signal-fusion assessment."""

    # Instrument being evaluated.
    symbol: str

    # Directional context.
    direction: FusionDirection

    # Final decision after risk veto.
    decision: FusionDecision

    # Weighted normalized score.
    score: float

    # Evidence coverage percentage.
    confidence: float

    # Whether the configured confidence requirement is satisfied.
    sufficient_data: bool

    # Number of usable evidence items.
    sources_used: int

    # Total configured positive source weight.
    total_weight: float

    # Effective weight represented by supplied evidence.
    used_weight: float

    # Evidence used by the fusion calculation.
    evidence: tuple[SignalFusionEvidence, ...]

    # Timestamp of the assessment.
    decision_timestamp: datetime

    # Deterministic explanation messages.
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize the assessment into JSON-compatible values."""

        # Return primitive values suitable for persistence/logging.
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "decision": self.decision.value,
            "score": self.score,
            "confidence": self.confidence,
            "sufficient_data": self.sufficient_data,
            "sources_used": self.sources_used,
            "total_weight": self.total_weight,
            "used_weight": self.used_weight,
            "evidence": [
                _serialize_value(item)
                for item in self.evidence
            ],
            "decision_timestamp": self.decision_timestamp.isoformat(),
            "reasons": list(self.reasons),
        }


def _serialize_value(value: object) -> object:
    """Recursively convert supported Python values into serializable values."""

    # Serialize enums using their primitive value.
    if isinstance(value, Enum):
        return value.value

    # Serialize datetimes using ISO-8601.
    if isinstance(value, datetime):
        return value.isoformat()

    # Serialize dataclasses recursively.
    if is_dataclass(value):
        return {
            field.name: _serialize_value(getattr(value, field.name))
            for field in fields(value)
        }

    # Serialize mappings recursively.
    if isinstance(value, Mapping):
        return {
            str(key): _serialize_value(item)
            for key, item in value.items()
        }

    # Serialize tuples recursively.
    if isinstance(value, tuple):
        return [
            _serialize_value(item)
            for item in value
        ]

    # Serialize lists recursively.
    if isinstance(value, list):
        return [
            _serialize_value(item)
            for item in value
        ]

    # Serialize sets recursively.
    if isinstance(value, set):
        return [
            _serialize_value(item)
            for item in value
        ]

    # Primitive values are already serializable.
    return value


class SignalFusionIntelligence:
    """
    Deterministic signal-fusion engine.

    The engine combines normalized evidence from upstream subsystems.

    Positive score:
        Bullish / LONG pressure.

    Negative score:
        Bearish / SHORT pressure.

    Zero score:
        NEUTRAL.

    No evidence:
        UNKNOWN.

    Risk veto:
        BLOCKED.

    Important compatibility rule:

        The default engine preserves directional context even when only
        partial evidence is supplied.

        An explicitly configured stricter min_confidence requirement can
        suppress directional signal creation when coverage is insufficient.

    This keeps the fusion layer useful for contextual analysis while
    allowing callers to require stronger evidence coverage.
    """

    # Default setup weight.
    DEFAULT_SETUP_WEIGHT = 1.0

    # Default entry weight.
    DEFAULT_ENTRY_WEIGHT = 1.0

    # Default confluence weight.
    DEFAULT_CONFLUENCE_WEIGHT = 1.0

    # Default macro weight.
    DEFAULT_MACRO_WEIGHT = 0.8

    # Default news weight.
    DEFAULT_NEWS_WEIGHT = 0.8

    # Default risk weight.
    DEFAULT_RISK_WEIGHT = 1.2

    # Default RR/EV weight.
    DEFAULT_RR_EV_WEIGHT = 1.0

    # Default directional threshold.
    DEFAULT_THRESHOLD = 20.0

    # Default strong directional threshold.
    DEFAULT_STRONG_THRESHOLD = 60.0

    # Default confidence requirement.
    DEFAULT_MIN_CONFIDENCE = 50.0

    # Current primary instrument.
    SYMBOL = "XAUUSD"

    # Canonical source weights.
    DEFAULT_WEIGHTS = {
        "setup": DEFAULT_SETUP_WEIGHT,
        "entry": DEFAULT_ENTRY_WEIGHT,
        "confluence": DEFAULT_CONFLUENCE_WEIGHT,
        "macro": DEFAULT_MACRO_WEIGHT,
        "news": DEFAULT_NEWS_WEIGHT,
        "risk": DEFAULT_RISK_WEIGHT,
        "rr_ev": DEFAULT_RR_EV_WEIGHT,
    }

    def __init__(
        self,
        *,
        weights: Mapping[str, float] | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        """Initialize the fusion engine."""

        # Start with a copy of the canonical weights.
        configured_weights = dict(self.DEFAULT_WEIGHTS)

        # Track whether the caller explicitly supplied a confidence
        # requirement different from the default.
        self._custom_confidence_gate = (
            float(min_confidence) != self.DEFAULT_MIN_CONFIDENCE
        )

        # Apply custom source weights.
        if weights is not None:
            for name, value in weights.items():

                # Reject unknown source names.
                if name not in configured_weights:
                    raise SignalFusionIntelligenceError(
                        f"Unknown fusion evidence source: {name!r}"
                    )

                # Reject boolean weights.
                if isinstance(value, bool):
                    raise SignalFusionIntelligenceError(
                        f"Weight for {name!r} must be numeric."
                    )

                # Require numeric weights.
                if not isinstance(value, (int, float)):
                    raise SignalFusionIntelligenceError(
                        f"Weight for {name!r} must be numeric."
                    )

                # Reject invalid weights.
                if (
                    not isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise SignalFusionIntelligenceError(
                        f"Weight for {name!r} must be finite and non-negative."
                    )

                # Store the validated custom weight.
                configured_weights[name] = float(value)

        # Validate normal threshold.
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not isfinite(float(threshold))
            or float(threshold) < 0.0
        ):
            raise SignalFusionIntelligenceError(
                "threshold must be a finite non-negative number."
            )

        # Validate strong threshold.
        if (
            isinstance(strong_threshold, bool)
            or not isinstance(strong_threshold, (int, float))
            or not isfinite(float(strong_threshold))
            or float(strong_threshold) <= float(threshold)
        ):
            raise SignalFusionIntelligenceError(
                "strong_threshold must be finite and greater than threshold."
            )

        # Validate confidence.
        if (
            isinstance(min_confidence, bool)
            or not isinstance(min_confidence, (int, float))
            or not isfinite(float(min_confidence))
            or not 0.0 <= float(min_confidence) <= 100.0
        ):
            raise SignalFusionIntelligenceError(
                "min_confidence must be between 0 and 100."
            )

        # Store the final validated configuration.
        self.weights = configured_weights
        self.threshold = float(threshold)
        self.strong_threshold = float(strong_threshold)
        self.min_confidence = float(min_confidence)

    @staticmethod
    def _validate_timestamp(
        decision_timestamp: datetime,
    ) -> None:
        """Validate the decision timestamp."""

        # Require datetime.
        if not isinstance(decision_timestamp, datetime):
            raise SignalFusionIntelligenceError(
                "decision_timestamp must be a datetime."
            )

        # Reject naive datetime values.
        if decision_timestamp.tzinfo is None:
            raise SignalFusionIntelligenceError(
                "decision_timestamp must be timezone-aware."
            )

        # Reject invalid timezone offsets.
        if decision_timestamp.utcoffset() is None:
            raise SignalFusionIntelligenceError(
                "decision_timestamp must have a valid timezone offset."
            )

    @classmethod
    def _validate_evidence(
        cls,
        evidence: Iterable[SignalFusionEvidence],
    ) -> tuple[SignalFusionEvidence, ...]:
        """Validate and materialize evidence."""

        # Evidence cannot be None.
        if evidence is None:
            raise SignalFusionIntelligenceError(
                "evidence cannot be None."
            )

        # Materialize the iterable.
        materialized = tuple(evidence)

        # Validate every item.
        for item in materialized:

            # Require the canonical evidence class.
            if not isinstance(item, SignalFusionEvidence):
                raise SignalFusionIntelligenceError(
                    "All evidence items must be SignalFusionEvidence instances."
                )

        # Return immutable evidence.
        return materialized

    @staticmethod
    def _validate_score(
        score: float,
    ) -> None:
        """Validate the normalized evidence score."""

        # Scores must remain inside the documented range.
        if not -100.0 <= float(score) <= 100.0:
            raise SignalFusionIntelligenceError(
                "Evidence score must be between -100 and 100."
            )

    @staticmethod
    def _normalise_score(
        score: float,
    ) -> float:
        """Normalize a score defensively."""

        # Clamp the score to the normalized range.
        normalized = max(
            -100.0,
            min(100.0, float(score)),
        )

        # Remove insignificant floating-point residue.
        if abs(normalized) < 1e-12:
            normalized = 0.0

        return normalized

    def _classify_direction(
        self,
        *,
        score: float,
        has_evidence: bool,
    ) -> FusionDirection:
        """Classify the aggregate directional score."""

        # No evidence means unknown.
        if not has_evidence:
            return FusionDirection.UNKNOWN

        # Strong bullish pressure.
        if score >= self.strong_threshold:
            return FusionDirection.LONG

        # Strong bearish pressure.
        if score <= -self.strong_threshold:
            return FusionDirection.SHORT

        # Normal bullish pressure.
        if score >= self.threshold:
            return FusionDirection.LONG

        # Normal bearish pressure.
        if score <= -self.threshold:
            return FusionDirection.SHORT

        # Balanced evidence.
        return FusionDirection.NEUTRAL

    @staticmethod
    def _decision_from_direction(
        direction: FusionDirection,
        *,
        risk_blocked: bool,
    ) -> FusionDecision:
        """Convert directional context into the final decision."""

        # Risk veto always has final authority.
        if risk_blocked:
            return FusionDecision.BLOCKED

        # LONG direction.
        if direction == FusionDirection.LONG:
            return FusionDecision.LONG

        # SHORT direction.
        if direction == FusionDirection.SHORT:
            return FusionDecision.SHORT

        # NEUTRAL direction.
        if direction == FusionDirection.NEUTRAL:
            return FusionDecision.NEUTRAL

        # UNKNOWN direction.
        return FusionDecision.UNKNOWN

    def analyze(
        self,
        evidence: Iterable[SignalFusionEvidence],
        *,
        decision_timestamp: datetime,
        symbol: str = SYMBOL,
        risk_blocked: bool = False,
    ) -> SignalFusionAssessment:
        """
        Analyze normalized evidence.

        This method produces deterministic context only.
        It does not execute trades.
        """

        # Validate decision timestamp.
        self._validate_timestamp(decision_timestamp)

        # Validate symbol.
        if not isinstance(symbol, str) or not symbol.strip():
            raise SignalFusionIntelligenceError(
                "symbol must be a non-empty string."
            )

        # Only XAUUSD is currently supported.
        if symbol.upper() != self.SYMBOL:
            raise SignalFusionIntelligenceError(
                f"Signal fusion currently supports {self.SYMBOL} only."
            )

        # Validate risk veto.
        if not isinstance(risk_blocked, bool):
            raise SignalFusionIntelligenceError(
                "risk_blocked must be a boolean."
            )

        # Validate and materialize evidence.
        validated_evidence = self._validate_evidence(evidence)

        # ---------------------------------------------------------------
        # AUTHORITATIVE SCORE VALIDATION
        # ---------------------------------------------------------------

        for item in validated_evidence:

            # Validate score at the fusion boundary.
            self._validate_score(float(item.score))

        # ---------------------------------------------------------------
        # SOURCE VALIDATION
        # ---------------------------------------------------------------

        for item in validated_evidence:

            # Reject unsupported evidence sources.
            if item.source not in self.weights:
                raise SignalFusionIntelligenceError(
                    f"Unknown fusion evidence source: {item.source!r}"
                )

        # Keep evidence whose configured source weight is positive.
        usable_evidence = tuple(
            item
            for item in validated_evidence
            if self.weights[item.source] > 0.0
        )

        # Calculate total configured positive source weight.
        total_weight = sum(
            weight
            for weight in self.weights.values()
            if weight > 0.0
        )

        # Calculate effective weight for every evidence item.
        effective_weights = {
            item: (
                self.weights[item.source]
                * float(item.weight)
            )
            for item in usable_evidence
        }

        # Calculate the total effective evidence weight.
        used_weight = sum(
            effective_weights.values()
        )

        # ---------------------------------------------------------------
        # NO USABLE EVIDENCE
        # ---------------------------------------------------------------

        if used_weight <= 0.0:

            # No evidence means unknown.
            direction = FusionDirection.UNKNOWN

            # Apply risk veto.
            decision = self._decision_from_direction(
                direction,
                risk_blocked=risk_blocked,
            )

            # Explain the empty result.
            reasons = [
                "No usable fusion evidence was available.",
                "Directional context is UNKNOWN because usable evidence "
                "weight is zero.",
            ]

            # Explain an active risk veto.
            if risk_blocked:
                reasons.append(
                    "Risk veto is active; final decision is BLOCKED."
                )

            # Return empty assessment.
            return SignalFusionAssessment(
                symbol=self.SYMBOL,
                direction=direction,
                decision=decision,
                score=0.0,
                confidence=0.0,
                sufficient_data=False,
                sources_used=0,
                total_weight=total_weight,
                used_weight=0.0,
                evidence=tuple(),
                decision_timestamp=decision_timestamp,
                reasons=tuple(reasons),
            )

        # ---------------------------------------------------------------
        # WEIGHTED SCORE
        # ---------------------------------------------------------------

        # Calculate weighted directional pressure.
        weighted_sum = sum(
            effective_weights[item] * float(item.score)
            for item in usable_evidence
        )

        # Convert to a normalized weighted average.
        score = weighted_sum / used_weight

        # Normalize defensively.
        score = self._normalise_score(score)

        # ---------------------------------------------------------------
        # CONFIDENCE
        # ---------------------------------------------------------------

        # Cap each source's contribution to coverage at its configured
        # source weight.
        coverage_weight = sum(
            min(
                self.weights[item.source],
                self.weights[item.source]
                * float(item.weight),
            )
            for item in usable_evidence
        )

        # Calculate percentage coverage.
        if total_weight > 0.0:
            confidence = (
                coverage_weight / total_weight
            ) * 100.0
        else:
            confidence = 0.0

        # Clamp confidence.
        confidence = max(
            0.0,
            min(100.0, confidence),
        )

        # Remove floating-point residue.
        if abs(confidence) < 1e-12:
            confidence = 0.0

        # Determine whether configured confidence is satisfied.
        sufficient_data = (
            confidence >= self.min_confidence
        )

        # ---------------------------------------------------------------
        # DIRECTION
        # ---------------------------------------------------------------
        #
        # IMPORTANT:
        #
        # Default behavior preserves directional context with partial
        # evidence because the fusion layer is also an intelligence/
        # context layer.
        #
        # When the caller explicitly configures a different confidence
        # requirement, that requirement becomes an intentional signal
        # gate.
        # ---------------------------------------------------------------

        if self._custom_confidence_gate and not sufficient_data:

            # Explicitly configured confidence requirement was not met.
            direction = FusionDirection.UNKNOWN

        else:

            # Default behavior, or satisfied explicit confidence gate.
            direction = self._classify_direction(
                score=score,
                has_evidence=True,
            )

        # Apply risk veto after directional classification.
        decision = self._decision_from_direction(
            direction,
            risk_blocked=risk_blocked,
        )

        # ---------------------------------------------------------------
        # REASONS
        # ---------------------------------------------------------------

        reasons: list[str] = [
            f"Fusion score is {score:.4f}.",
            f"Fusion evidence coverage is {confidence:.2f}%.",
            f"Usable fusion evidence items: {len(usable_evidence)}.",
        ]

        # Explain confidence state.
        if sufficient_data:

            reasons.append(
                "Fusion evidence coverage meets the configured "
                "confidence threshold."
            )

        else:

            reasons.append(
                "Fusion evidence coverage is below the configured "
                "confidence threshold."
            )

        # Explain the confidence-gate behavior.
        if self._custom_confidence_gate and not sufficient_data:

            reasons.append(
                "An explicitly configured confidence requirement "
                "suppressed directional signal creation."
            )

        elif not sufficient_data:

            reasons.append(
                "Partial evidence is retained as directional context "
                "under the default fusion configuration."
            )

        # Explain directional result.
        if direction == FusionDirection.LONG:

            reasons.append(
                "Available evidence produces LONG directional context."
            )

        elif direction == FusionDirection.SHORT:

            reasons.append(
                "Available evidence produces SHORT directional context."
            )

        elif direction == FusionDirection.NEUTRAL:

            reasons.append(
                "Available evidence is balanced and does not meet the "
                "directional threshold."
            )

        else:

            reasons.append(
                "Directional context is UNKNOWN because the configured "
                "confidence gate was not satisfied."
            )

        # Explain risk state.
        if risk_blocked:

            reasons.append(
                "Risk veto is active; final decision is BLOCKED."
            )

        else:

            reasons.append(
                "No risk veto is active."
            )

        # Preserve architectural separation.
        reasons.append(
            "Signal fusion provides deterministic context and does not "
            "execute trades."
        )

        # Return immutable assessment.
        return SignalFusionAssessment(
            symbol=self.SYMBOL,
            direction=direction,
            decision=decision,
            score=score,
            confidence=confidence,
            sufficient_data=sufficient_data,
            sources_used=len(usable_evidence),
            total_weight=total_weight,
            used_weight=used_weight,
            evidence=usable_evidence,
            decision_timestamp=decision_timestamp,
            reasons=tuple(reasons),
        )

    def analyze_xauusd(
        self,
        evidence: Iterable[SignalFusionEvidence],
        *,
        decision_timestamp: datetime,
        risk_blocked: bool = False,
    ) -> SignalFusionAssessment:
        """Convenience wrapper explicitly targeting XAUUSD."""

        # Delegate to the canonical analyze method.
        return self.analyze(
            evidence=evidence,
            decision_timestamp=decision_timestamp,
            symbol=self.SYMBOL,
            risk_blocked=risk_blocked,
        )