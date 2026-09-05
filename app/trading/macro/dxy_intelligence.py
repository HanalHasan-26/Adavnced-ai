# app/trading/macro/dxy_intelligence.py

"""Deterministic DXY intelligence.

This module analyzes DXY observations and produces an auditable
DXY-specific assessment.

Important:
- No external data fetching.
- No LLM dependency.
- No trade decision.
- No XAUUSD buy/sell decision.
- Future observations are never allowed to influence a historical
  decision.
"""

from __future__ import annotations

# Import dataclass for immutable result models.
from dataclasses import dataclass

# Import datetime for historical timestamp validation.
from datetime import datetime

# Import Enum for explicit DXY states.
from enum import Enum

# Import math for finite-number validation.
import math

# Import the existing macro observation models.
from app.trading.macro.macro_observation import (
    MacroDirection,
    MacroIndicator,
    MacroObservation,
)


class DXYIntelligenceError(ValueError):
    """Raised when DXY analysis receives invalid input."""


class DXYLevel(str, Enum):
    """DXY directional classification."""

    # DXY is strongly rising.
    STRONG = "STRONG"

    # DXY is rising but not strongly enough for STRONG.
    BULLISH = "BULLISH"

    # DXY is neither meaningfully bullish nor bearish.
    NEUTRAL = "NEUTRAL"

    # DXY is falling but not strongly enough for WEAK.
    BEARISH = "BEARISH"

    # DXY is strongly falling.
    WEAK = "WEAK"

    # There is not enough valid information.
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class DXYAssessment:
    """Auditable DXY assessment."""

    # DXY observation value used for the assessment.
    value: float | None

    # Previous DXY value, when available.
    previous: float | None

    # Forecast DXY value, when available.
    forecast: float | None

    # Absolute change from the previous value.
    change_from_previous: float | None

    # Percentage change from the previous value.
    percentage_change: float | None

    # Direction supplied by the normalized observation.
    direction: MacroDirection

    # Final DXY classification.
    level: DXYLevel

    # Confidence expressed as a percentage.
    confidence: float

    # Whether enough information exists for a directional assessment.
    sufficient_data: bool

    # Observation timestamp used by the assessment.
    observation_timestamp: datetime | None

    # Decision timestamp used for historical filtering.
    decision_timestamp: datetime

    # Human-readable explanations.
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the DXY assessment."""

        # Validate the direction enum.
        if not isinstance(self.direction, MacroDirection):
            raise DXYIntelligenceError(
                "direction must be a MacroDirection."
            )

        # Validate the DXY level enum.
        if not isinstance(self.level, DXYLevel):
            raise DXYIntelligenceError(
                "level must be a DXYLevel."
            )

        # Validate the decision timestamp.
        if not isinstance(self.decision_timestamp, datetime):
            raise DXYIntelligenceError(
                "decision_timestamp must be a datetime."
            )

        # Validate the optional numeric fields.
        for name, field_value in (
            ("value", self.value),
            ("previous", self.previous),
            ("forecast", self.forecast),
            ("change_from_previous", self.change_from_previous),
            ("percentage_change", self.percentage_change),
        ):
            # Optional values may legitimately be None.
            if field_value is None:
                continue

            # Reject boolean values.
            if isinstance(field_value, bool):
                raise DXYIntelligenceError(
                    f"{name} must be numeric or None."
                )

            # Require numeric values.
            if not isinstance(field_value, (int, float)):
                raise DXYIntelligenceError(
                    f"{name} must be numeric or None."
                )

            # Reject NaN and infinity.
            if not math.isfinite(float(field_value)):
                raise DXYIntelligenceError(
                    f"{name} must be finite."
                )

        # Validate the optional observation timestamp.
        if self.observation_timestamp is not None:
            if not isinstance(
                self.observation_timestamp,
                datetime,
            ):
                raise DXYIntelligenceError(
                    "observation_timestamp must be a datetime or None."
                )

        # Confidence must be finite.
        if not math.isfinite(self.confidence):
            raise DXYIntelligenceError(
                "confidence must be finite."
            )

        # Confidence is a percentage.
        if not 0.0 <= self.confidence <= 100.0:
            raise DXYIntelligenceError(
                "confidence must be between 0 and 100."
            )


class DXYIntelligence:
    """Analyze DXY observations deterministically."""

    # DXY movement of at least this percentage is considered
    # meaningfully directional.
    DEFAULT_SIGNIFICANT_CHANGE_PCT = 0.10

    # DXY movement of at least this percentage is considered strong.
    DEFAULT_STRONG_CHANGE_PCT = 0.50

    def __init__(
        self,
        significant_change_pct: float = DEFAULT_SIGNIFICANT_CHANGE_PCT,
        strong_change_pct: float = DEFAULT_STRONG_CHANGE_PCT,
    ) -> None:
        """Initialize DXY intelligence."""

        # Validate the significant-change threshold.
        self._validate_finite(
            significant_change_pct,
            "significant_change_pct",
        )

        # Validate the strong-change threshold.
        self._validate_finite(
            strong_change_pct,
            "strong_change_pct",
        )

        # Thresholds must be positive.
        if significant_change_pct <= 0:
            raise DXYIntelligenceError(
                "significant_change_pct must be greater than zero."
            )

        if strong_change_pct <= 0:
            raise DXYIntelligenceError(
                "strong_change_pct must be greater than zero."
            )

        # A strong threshold must not be below the significant threshold.
        if strong_change_pct < significant_change_pct:
            raise DXYIntelligenceError(
                "strong_change_pct must be greater than or equal to "
                "significant_change_pct."
            )

        # Store validated configuration.
        self._significant_change_pct = float(
            significant_change_pct
        )

        self._strong_change_pct = float(
            strong_change_pct
        )

    @staticmethod
    def _validate_finite(
        value: float,
        name: str,
    ) -> None:
        """Validate a finite numeric value."""

        # Reject booleans explicitly.
        if isinstance(value, bool):
            raise DXYIntelligenceError(
                f"{name} must be numeric, not bool."
            )

        # Require numeric input.
        if not isinstance(value, (int, float)):
            raise DXYIntelligenceError(
                f"{name} must be numeric."
            )

        # Reject NaN and infinity.
        if not math.isfinite(float(value)):
            raise DXYIntelligenceError(
                f"{name} must be finite."
            )

    @staticmethod
    def _validate_datetime(
        value: datetime,
        name: str,
    ) -> None:
        """Validate datetime input."""

        # Require a datetime object.
        if not isinstance(value, datetime):
            raise DXYIntelligenceError(
                f"{name} must be a datetime."
            )

    @staticmethod
    def _validate_timezone(
        observation: MacroObservation,
        decision_timestamp: datetime,
    ) -> None:
        """Validate compatible timezone-awareness."""

        # Determine decision timestamp timezone-awareness.
        decision_aware = decision_timestamp.tzinfo is not None

        # Determine observation timezone-awareness.
        observation_aware = observation.timestamp.tzinfo is not None

        # Never silently compare incompatible timestamp semantics.
        if decision_aware != observation_aware:
            raise DXYIntelligenceError(
                "decision_timestamp and observation timestamp "
                "must use the same timezone-awareness."
            )

    @staticmethod
    def _percentage_change(
        value: float | None,
        previous: float | None,
    ) -> float | None:
        """Calculate percentage change from the previous value."""

        # Both values are required for percentage change.
        if value is None or previous is None:
            return None

        # A zero previous value cannot be used as a denominator.
        if previous == 0:
            return None

        # Calculate percentage change.
        result = (
            (value - previous)
            / abs(previous)
        ) * 100.0

        # Normalize floating-point noise.
        if abs(result) < 1e-12:
            return 0.0

        # Return the finite result.
        return result

    def _classify(
        self,
        direction: MacroDirection,
        percentage_change: float | None,
    ) -> DXYLevel:
        """Classify DXY using normalized direction and magnitude."""

        # Unknown direction cannot safely produce a directional result.
        if direction is MacroDirection.UNKNOWN:
            return DXYLevel.UNKNOWN

        # Stable direction is neutral.
        if direction is MacroDirection.STABLE:
            return DXYLevel.NEUTRAL

        # Without a percentage change, use the normalized direction.
        if percentage_change is None:
            if direction is MacroDirection.RISING:
                return DXYLevel.BULLISH

            if direction is MacroDirection.FALLING:
                return DXYLevel.BEARISH

            return DXYLevel.UNKNOWN

        # Rising DXY is bullish.
        if direction is MacroDirection.RISING:

            # Strong positive movement.
            if percentage_change >= self._strong_change_pct:
                return DXYLevel.STRONG

            # Meaningful positive movement.
            if percentage_change >= self._significant_change_pct:
                return DXYLevel.BULLISH

            # Direction exists but movement is small.
            return DXYLevel.NEUTRAL

        # Falling DXY is bearish.
        if direction is MacroDirection.FALLING:

            # Strong negative movement.
            if percentage_change <= -self._strong_change_pct:
                return DXYLevel.WEAK

            # Meaningful negative movement.
            if percentage_change <= -self._significant_change_pct:
                return DXYLevel.BEARISH

            # Direction exists but movement is small.
            return DXYLevel.NEUTRAL

        # Defensive fallback.
        return DXYLevel.UNKNOWN

    def analyze(
        self,
        observations: list[MacroObservation]
        | tuple[MacroObservation, ...],
        decision_timestamp: datetime,
    ) -> DXYAssessment:
        """Analyze the latest historical-safe DXY observation."""

        # Validate the decision timestamp.
        self._validate_datetime(
            decision_timestamp,
            "decision_timestamp",
        )

        # Require a list or tuple.
        if not isinstance(observations, (list, tuple)):
            raise DXYIntelligenceError(
                "observations must be a list or tuple."
            )

        # Validate every observation.
        for observation in observations:
            if not isinstance(observation, MacroObservation):
                raise DXYIntelligenceError(
                    "observations must contain MacroObservation values."
                )

        # Keep only DXY observations.
        dxy_observations = [
            observation
            for observation in observations
            if observation.indicator is MacroIndicator.DXY
        ]

        # No DXY observations means unknown.
        if not dxy_observations:
            return DXYAssessment(
                value=None,
                previous=None,
                forecast=None,
                change_from_previous=None,
                percentage_change=None,
                direction=MacroDirection.UNKNOWN,
                level=DXYLevel.UNKNOWN,
                confidence=0.0,
                sufficient_data=False,
                observation_timestamp=None,
                decision_timestamp=decision_timestamp,
                reasons=(
                    "No DXY observation was supplied.",
                ),
            )

        # Validate timestamp semantics.
        for observation in dxy_observations:
            self._validate_timezone(
                observation,
                decision_timestamp,
            )

        # Enforce no-lookahead.
        available = [
            observation
            for observation in dxy_observations
            if observation.timestamp <= decision_timestamp
        ]

        # Future-only observations must produce UNKNOWN.
        if not available:
            return DXYAssessment(
                value=None,
                previous=None,
                forecast=None,
                change_from_previous=None,
                percentage_change=None,
                direction=MacroDirection.UNKNOWN,
                level=DXYLevel.UNKNOWN,
                confidence=0.0,
                sufficient_data=False,
                observation_timestamp=None,
                decision_timestamp=decision_timestamp,
                reasons=(
                    "All DXY observations occur after the "
                    "decision timestamp.",
                ),
            )

        # Select the latest historical-safe observation.
        latest = max(
            available,
            key=lambda observation: observation.timestamp,
        )

        # Calculate absolute change when possible.
        change = latest.change_from_previous

        # Calculate percentage change from previous.
        percentage_change = self._percentage_change(
            latest.value,
            latest.previous,
        )

        # Classify DXY.
        level = self._classify(
            latest.direction,
            percentage_change,
        )

        # Base confidence comes from the quality of available
        # directional information.
        if latest.direction is MacroDirection.UNKNOWN:
            confidence = 0.0

        elif percentage_change is None:
            confidence = 50.0

        else:
            # Having both direction and previous-value movement gives
            # stronger evidence than direction alone.
            confidence = 100.0

        # Build reasons.
        reasons: list[str] = []

        # Explain the direction.
        if latest.direction is MacroDirection.RISING:
            reasons.append(
                "DXY direction is rising, indicating USD strength."
            )

        elif latest.direction is MacroDirection.FALLING:
            reasons.append(
                "DXY direction is falling, indicating USD weakness."
            )

        elif latest.direction is MacroDirection.STABLE:
            reasons.append(
                "DXY direction is stable, indicating no strong "
                "directional pressure."
            )

        else:
            reasons.append(
                "DXY direction is unknown."
            )

        # Explain percentage movement when available.
        if percentage_change is not None:

            # Positive movement.
            if percentage_change > 0:
                reasons.append(
                    f"DXY changed +{percentage_change:.6f}% "
                    "from the previous observation."
                )

            # Negative movement.
            elif percentage_change < 0:
                reasons.append(
                    f"DXY changed {percentage_change:.6f}% "
                    "from the previous observation."
                )

            # Exactly unchanged.
            else:
                reasons.append(
                    "DXY has no percentage change from the "
                    "previous observation."
                )

        # Return the complete assessment.
        return DXYAssessment(
            value=float(latest.value),
            previous=(
                float(latest.previous)
                if latest.previous is not None
                else None
            ),
            forecast=(
                float(latest.forecast)
                if latest.forecast is not None
                else None
            ),
            change_from_previous=(
                float(change)
                if change is not None
                else None
            ),
            percentage_change=percentage_change,
            direction=latest.direction,
            level=level,
            confidence=confidence,
            sufficient_data=(
                latest.direction is not MacroDirection.UNKNOWN
            ),
            observation_timestamp=latest.timestamp,
            decision_timestamp=decision_timestamp,
            reasons=tuple(reasons),
        )

    def analyze_xauusd(
        self,
        observations: list[MacroObservation]
        | tuple[MacroObservation, ...],
        decision_timestamp: datetime,
    ) -> DXYAssessment:
        """Analyze DXY for an XAUUSD macro context."""

        # This method intentionally returns DXY intelligence only.
        #
        # The conversion from DXY/USD strength into an XAUUSD macro
        # bias belongs to P2.19.10.
        return self.analyze(
            observations=observations,
            decision_timestamp=decision_timestamp,
        )