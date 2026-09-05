"""
Fusion input adapters.

This module converts existing trading-engine assessments into the normalized
SignalFusionEvidence format used by the signal-fusion layer.

Important:
- This module does NOT calculate technical indicators.
- This module does NOT generate entries.
- This module does NOT calculate stop-loss or take-profit.
- This module does NOT override the Risk Engine.
- This module only translates existing subsystem outputs into fusion evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Any, Iterable

from app.trading.fusion.signal_fusion_intelligence import (
    FusionDirection,
    SignalFusionEvidence,
)


class FusionInputAdapterError(ValueError):
    """Raised when fusion input adaptation fails."""


@dataclass(frozen=True, slots=True)
class FusionInputAdapter:
    """
    Convert an existing subsystem result into normalized fusion evidence.

    The adapter is intentionally limited to translation and validation.
    The original subsystem remains authoritative for its calculations.
    """

    source: str
    default_weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate adapter configuration."""

        # Require a non-empty source name.
        if not isinstance(self.source, str) or not self.source.strip():
            raise FusionInputAdapterError(
                "source must be a non-empty string."
            )

        # Reject boolean values because bool is a subclass of int.
        if isinstance(self.default_weight, bool):
            raise FusionInputAdapterError(
                "default_weight must be numeric."
            )

        # Require a numeric weight.
        if not isinstance(self.default_weight, (int, float)):
            raise FusionInputAdapterError(
                "default_weight must be numeric."
            )

        # Require the weight to be finite.
        if not isfinite(float(self.default_weight)):
            raise FusionInputAdapterError(
                "default_weight must be finite."
            )

        # Negative weights are not allowed.
        if self.default_weight < 0.0:
            raise FusionInputAdapterError(
                "default_weight must be non-negative."
            )

    @staticmethod
    def validate_timestamp(timestamp: datetime) -> None:
        """Validate that a timestamp is timezone-aware."""

        # Require an actual datetime object.
        if not isinstance(timestamp, datetime):
            raise FusionInputAdapterError(
                "decision_timestamp must be a datetime."
            )

        # Reject naive datetimes because they are ambiguous.
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise FusionInputAdapterError(
                "decision_timestamp must be timezone-aware."
            )

    @staticmethod
    def _normalize_score(score: Any) -> float:
        """Validate and normalize a fusion score."""

        # Reject booleans explicitly.
        if isinstance(score, bool):
            raise FusionInputAdapterError(
                "score must be numeric."
            )

        # Require numeric input.
        if not isinstance(score, (int, float)):
            raise FusionInputAdapterError(
                "score must be numeric."
            )

        # Convert the score to float.
        normalized = float(score)

        # Reject NaN and infinity.
        if not isfinite(normalized):
            raise FusionInputAdapterError(
                "score must be finite."
            )

        # Fusion scores must remain inside the canonical range.
        if normalized < -100.0 or normalized > 100.0:
            raise FusionInputAdapterError(
                "score must be between -100 and 100."
            )

        return normalized

    @staticmethod
    def _normalize_direction(direction: Any) -> FusionDirection:
        """
        Normalize subsystem direction values into FusionDirection.

        Supports:
        - FusionDirection enum members.
        - Enum values such as "long" and "short".
        - Enum member names such as "LONG" and "SHORT".
        - Common aliases such as BUY, SELL, BULLISH and BEARISH.
        """

        # Missing direction becomes UNKNOWN.
        if direction is None:
            return FusionDirection.UNKNOWN

        # If the value is already the correct enum, return it unchanged.
        if isinstance(direction, FusionDirection):
            return direction

        # Support other Enum-like objects by reading their underlying value.
        value = getattr(direction, "value", direction)

        # Require a string-like representation.
        if not isinstance(value, str):
            raise FusionInputAdapterError(
                "direction must be string-like."
            )

        # Normalize whitespace and casing.
        normalized = value.strip().upper()

        # Translate common terminology used by other trading subsystems.
        aliases = {
            "BUY": "LONG",
            "SELL": "SHORT",
            "BULLISH": "LONG",
            "BEARISH": "SHORT",
            "FLAT": "NEUTRAL",
            "NONE": "NEUTRAL",
        }

        # Apply the aliases.
        normalized = aliases.get(normalized, normalized)

        # First try interpreting the normalized value as the enum value.
        #
        # Example:
        # FusionDirection.LONG.value == "long"
        #
        # This lookup is attempted case-insensitively by trying lowercase.
        try:
            return FusionDirection(normalized.lower())
        except ValueError:
            pass

        # Then try interpreting the normalized value as the enum member name.
        #
        # Example:
        # FusionDirection["LONG"] -> FusionDirection.LONG
        try:
            return FusionDirection[normalized]
        except KeyError as exc:
            raise FusionInputAdapterError(
                f"Unsupported fusion direction: {normalized}."
            ) from exc

    @staticmethod
    def _extract_reason(result: Any) -> str:
        """Extract an explanation from an existing subsystem result."""

        # Prefer a collection of reasons when the subsystem provides one.
        reasons = getattr(result, "reasons", None)

        if reasons:
            # Preserve an existing string reason.
            if isinstance(reasons, str):
                return reasons

            # Convert multiple reasons into a deterministic string.
            try:
                return "; ".join(str(reason) for reason in reasons)
            except TypeError:
                return str(reasons)

        # Fall back to a singular reason field.
        reason = getattr(result, "reason", None)

        if reason is not None:
            return str(reason)

        # Use a deterministic fallback if no explanation exists.
        return "Evidence adapted from an existing trading subsystem."

    @staticmethod
    def _normalize_weight(weight: Any) -> float:
        """Validate and normalize an evidence weight."""

        # Reject booleans explicitly.
        if isinstance(weight, bool):
            raise FusionInputAdapterError(
                "weight must be numeric."
            )

        # Require numeric input.
        if not isinstance(weight, (int, float)):
            raise FusionInputAdapterError(
                "weight must be numeric."
            )

        # Convert to float.
        normalized = float(weight)

        # Reject NaN and infinity.
        if not isfinite(normalized):
            raise FusionInputAdapterError(
                "weight must be finite."
            )

        # Negative weights are invalid.
        if normalized < 0.0:
            raise FusionInputAdapterError(
                "weight must be non-negative."
            )

        return normalized

    def adapt(
        self,
        result: Any,
        *,
        decision_timestamp: datetime,
        score: Any | None = None,
        direction: Any | None = None,
        reason: str | None = None,
        weight: float | None = None,
    ) -> SignalFusionEvidence:
        """
        Convert an existing subsystem result into SignalFusionEvidence.

        Explicit score, direction, reason and weight values override values
        discovered from the subsystem result.
        """

        # Validate the decision timestamp.
        self.validate_timestamp(decision_timestamp)

        # Do not create artificial evidence from missing subsystem output.
        if result is None:
            raise FusionInputAdapterError(
                f"{self.source} result cannot be None."
            )

        # Prefer an explicitly supplied score.
        resolved_score = (
            score
            if score is not None
            else getattr(result, "score", None)
        )

        # Require a score.
        if resolved_score is None:
            raise FusionInputAdapterError(
                f"{self.source} result does not contain a score."
            )

        # Normalize the score.
        normalized_score = self._normalize_score(
            resolved_score
        )

        # Prefer an explicitly supplied direction.
        resolved_direction = (
            direction
            if direction is not None
            else getattr(result, "direction", None)
        )

        # Convert the direction to the canonical enum.
        normalized_direction = self._normalize_direction(
            resolved_direction
        )

        # Use the configured adapter weight unless explicitly overridden.
        resolved_weight = (
            self.default_weight
            if weight is None
            else weight
        )

        # Normalize the weight.
        normalized_weight = self._normalize_weight(
            resolved_weight
        )

        # Prefer an explicitly supplied reason.
        if isinstance(reason, str) and reason.strip():
            resolved_reason = reason.strip()
        else:
            # Otherwise extract the reason from the subsystem result.
            resolved_reason = self._extract_reason(result)

        # Create the exact evidence object required by Signal Fusion.
        return SignalFusionEvidence(
            source=self.source,
            direction=normalized_direction,
            score=normalized_score,
            weight=normalized_weight,
            reason=resolved_reason,
        )


class FusionInputAdapterRegistry:
    """
    Registry of standard fusion input adapters.

    This keeps source names and default weights centralized so different
    parts of the system cannot accidentally create competing conventions.
    """

    # Setup Engine evidence source.
    SETUP = "setup"

    # Entry Model evidence source.
    ENTRY = "entry"

    # Setup Confluence evidence source.
    CONFLUENCE = "confluence"

    # Macro intelligence evidence source.
    MACRO = "macro"

    # News intelligence evidence source.
    NEWS = "news"

    # Risk Engine evidence source.
    RISK = "risk"

    # RR/EV evidence source.
    RR_EV = "rr_ev"

    def __init__(
        self,
        *,
        setup_weight: float = 1.0,
        entry_weight: float = 1.0,
        confluence_weight: float = 1.0,
        macro_weight: float = 0.8,
        news_weight: float = 0.8,
        risk_weight: float = 1.2,
        rr_ev_weight: float = 1.0,
    ) -> None:
        """Create the standard fusion adapter registry."""

        # Create exactly one canonical adapter for each source.
        self._adapters = {
            self.SETUP: FusionInputAdapter(
                self.SETUP,
                setup_weight,
            ),
            self.ENTRY: FusionInputAdapter(
                self.ENTRY,
                entry_weight,
            ),
            self.CONFLUENCE: FusionInputAdapter(
                self.CONFLUENCE,
                confluence_weight,
            ),
            self.MACRO: FusionInputAdapter(
                self.MACRO,
                macro_weight,
            ),
            self.NEWS: FusionInputAdapter(
                self.NEWS,
                news_weight,
            ),
            self.RISK: FusionInputAdapter(
                self.RISK,
                risk_weight,
            ),
            self.RR_EV: FusionInputAdapter(
                self.RR_EV,
                rr_ev_weight,
            ),
        }

    def get(self, source: str) -> FusionInputAdapter:
        """Return the adapter registered for a source."""

        # Reject unknown sources instead of creating architecture implicitly.
        if source not in self._adapters:
            raise FusionInputAdapterError(
                f"Unknown fusion source: {source}."
            )

        return self._adapters[source]

    def adapt(
        self,
        source: str,
        result: Any,
        *,
        decision_timestamp: datetime,
        score: Any | None = None,
        direction: Any | None = None,
        reason: str | None = None,
        weight: float | None = None,
    ) -> SignalFusionEvidence:
        """Adapt one subsystem result using its registered adapter."""

        # Retrieve the canonical adapter and delegate the conversion.
        return self.get(source).adapt(
            result,
            decision_timestamp=decision_timestamp,
            score=score,
            direction=direction,
            reason=reason,
            weight=weight,
        )

    def adapt_many(
        self,
        items: Iterable[tuple[str, Any]],
        *,
        decision_timestamp: datetime,
    ) -> tuple[SignalFusionEvidence, ...]:
        """
        Adapt multiple subsystem results.

        Results remain in their supplied order so the caller can maintain
        deterministic evidence ordering for auditability and testing.
        """

        # Reuse the canonical timestamp validation.
        FusionInputAdapter.validate_timestamp(
            decision_timestamp
        )

        # Collect normalized evidence.
        evidence: list[SignalFusionEvidence] = []

        # Adapt each source/result pair.
        for source, result in items:
            evidence.append(
                self.adapt(
                    source,
                    result,
                    decision_timestamp=decision_timestamp,
                )
            )

        # Return immutable evidence.
        return tuple(evidence)

    def sources(self) -> tuple[str, ...]:
        """Return all canonical fusion source names."""

        # Preserve deterministic insertion order.
        return tuple(self._adapters.keys())


def create_default_fusion_adapters() -> FusionInputAdapterRegistry:
    """
    Create the canonical P2.20.2 fusion adapter registry.

    The default weights remain aligned with the P2.20.1 fusion core.
    """

    # Return the standard registry configuration.
    return FusionInputAdapterRegistry()