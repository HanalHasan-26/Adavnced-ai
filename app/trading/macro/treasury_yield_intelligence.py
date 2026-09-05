# app/trading/macro/treasury_yield_intelligence.py

"""Deterministic U.S. Treasury yield intelligence.

This module analyzes Treasury-yield observations and produces an
auditable assessment.

Important:
- No external data fetching.
- No LLM dependency.
- No trade decision.
- No direct XAUUSD buy/sell decision.
- Future observations are never used in historical analysis.
"""

from __future__ import annotations

# Import dataclass for immutable result objects.
from dataclasses import dataclass

# Import datetime for timestamp validation and comparison.
from datetime import datetime

# Import Enum for explicit Treasury-yield classifications.
from enum import Enum

# Import math for finite-number validation.
import math

# Import the existing macro observation models.
from app.trading.macro.macro_observation import (
    MacroDirection,
    MacroIndicator,
    MacroObservation,
)


class TreasuryYieldIntelligenceError(ValueError):
    """Raised when Treasury-yield analysis receives invalid input."""


class TreasuryYieldLevel(str, Enum):
    """Treasury-yield directional classification."""

    # Treasury yields are strongly rising.
    STRONG_RISING = "STRONG_RISING"

    # Treasury yields are meaningfully rising.
    RISING = "RISING"

    # Treasury yields are stable.
    STABLE = "STABLE"

    # Treasury yields are meaningfully falling.
    FALLING = "FALLING"

    # Treasury yields are strongly falling.
    STRONG_FALLING = "STRONG_FALLING"

    # There is not enough information to classify the yield.
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class TreasuryYieldAssessment:
    """Immutable and auditable assessment for one Treasury maturity."""

    # Treasury maturity being analyzed.
    indicator: MacroIndicator

    # Latest usable yield value.
    value: float | None

    # Previous yield value, when available.
    previous: float | None

    # Absolute yield change in percentage-point units.
    change_from_previous: float | None

    # Yield movement converted into basis points.
    change_basis_points: float | None

    # Percentage change from the previous yield.
    percentage_change: float | None

    # Existing normalized direction from MacroObservation.
    direction: MacroDirection

    # Treasury-specific classification.
    level: TreasuryYieldLevel

    # Confidence expressed as a percentage from 0 to 100.
    confidence: float

    # Whether enough information exists for a usable assessment.
    sufficient_data: bool

    # Timestamp of the selected observation.
    observation_timestamp: datetime | None

    # Historical decision timestamp.
    decision_timestamp: datetime

    # Human-readable audit reasons.
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the immutable assessment."""

        # Only supported Treasury maturities are valid.
        if self.indicator not in (
            MacroIndicator.US_2Y_YIELD,
            MacroIndicator.US_5Y_YIELD,
            MacroIndicator.US_10Y_YIELD,
            MacroIndicator.US_30Y_YIELD,
        ):
            raise TreasuryYieldIntelligenceError(
                "indicator must be a Treasury-yield MacroIndicator."
            )

        # Direction must use the existing MacroDirection enum.
        if not isinstance(self.direction, MacroDirection):
            raise TreasuryYieldIntelligenceError(
                "direction must be a MacroDirection."
            )

        # Level must use the TreasuryYieldLevel enum.
        if not isinstance(self.level, TreasuryYieldLevel):
            raise TreasuryYieldIntelligenceError(
                "level must be a TreasuryYieldLevel."
            )

        # Decision timestamp must be a datetime.
        if not isinstance(self.decision_timestamp, datetime):
            raise TreasuryYieldIntelligenceError(
                "decision_timestamp must be a datetime."
            )

        # Validate every optional numeric field.
        for name, value in (
            ("value", self.value),
            ("previous", self.previous),
            ("change_from_previous", self.change_from_previous),
            ("change_basis_points", self.change_basis_points),
            ("percentage_change", self.percentage_change),
        ):
            # None represents unavailable information.
            if value is None:
                continue

            # bool is technically an int in Python, so reject it explicitly.
            if isinstance(value, bool):
                raise TreasuryYieldIntelligenceError(
                    f"{name} must be numeric or None."
                )

            # Only int and float are accepted.
            if not isinstance(value, (int, float)):
                raise TreasuryYieldIntelligenceError(
                    f"{name} must be numeric or None."
                )

            # Reject NaN and infinity.
            if not math.isfinite(float(value)):
                raise TreasuryYieldIntelligenceError(
                    f"{name} must be finite."
                )

        # Validate confidence.
        if isinstance(self.confidence, bool):
            raise TreasuryYieldIntelligenceError(
                "confidence must be numeric."
            )

        if not isinstance(self.confidence, (int, float)):
            raise TreasuryYieldIntelligenceError(
                "confidence must be numeric."
            )

        if not math.isfinite(float(self.confidence)):
            raise TreasuryYieldIntelligenceError(
                "confidence must be finite."
            )

        # Confidence is represented as a percentage.
        if not 0.0 <= float(self.confidence) <= 100.0:
            raise TreasuryYieldIntelligenceError(
                "confidence must be between 0 and 100."
            )

        # Observation timestamp is optional.
        if self.observation_timestamp is not None:
            if not isinstance(self.observation_timestamp, datetime):
                raise TreasuryYieldIntelligenceError(
                    "observation_timestamp must be a datetime or None."
                )


class TreasuryYieldIntelligence:
    """Deterministic intelligence for U.S. Treasury yields."""

    # A movement of 2 basis points is considered meaningful.
    DEFAULT_SIGNIFICANT_CHANGE_BPS = 2.0

    # A movement of 5 basis points is considered strong.
    DEFAULT_STRONG_CHANGE_BPS = 5.0

    # Supported Treasury maturities.
    TREASURY_INDICATORS = (
        MacroIndicator.US_2Y_YIELD,
        MacroIndicator.US_5Y_YIELD,
        MacroIndicator.US_10Y_YIELD,
        MacroIndicator.US_30Y_YIELD,
    )

    def __init__(
        self,
        significant_change_bps: float = DEFAULT_SIGNIFICANT_CHANGE_BPS,
        strong_change_bps: float = DEFAULT_STRONG_CHANGE_BPS,
    ) -> None:
        """Initialize Treasury-yield intelligence."""

        # Validate the significant threshold.
        self._validate_finite(
            significant_change_bps,
            "significant_change_bps",
        )

        # Validate the strong threshold.
        self._validate_finite(
            strong_change_bps,
            "strong_change_bps",
        )

        # Both thresholds must be positive.
        if significant_change_bps <= 0:
            raise TreasuryYieldIntelligenceError(
                "significant_change_bps must be greater than zero."
            )

        if strong_change_bps <= 0:
            raise TreasuryYieldIntelligenceError(
                "strong_change_bps must be greater than zero."
            )

        # Strong movement cannot be smaller than significant movement.
        if strong_change_bps < significant_change_bps:
            raise TreasuryYieldIntelligenceError(
                "strong_change_bps must be greater than or equal to "
                "significant_change_bps."
            )

        # Store validated thresholds as floats.
        self._significant_change_bps = float(
            significant_change_bps
        )

        self._strong_change_bps = float(
            strong_change_bps
        )

    @staticmethod
    def _validate_finite(
        value: float,
        name: str,
    ) -> None:
        """Validate that a value is numeric, finite, and not boolean."""

        # Reject booleans explicitly.
        if isinstance(value, bool):
            raise TreasuryYieldIntelligenceError(
                f"{name} must be numeric, not bool."
            )

        # Require numeric input.
        if not isinstance(value, (int, float)):
            raise TreasuryYieldIntelligenceError(
                f"{name} must be numeric."
            )

        # Reject NaN and infinity.
        if not math.isfinite(float(value)):
            raise TreasuryYieldIntelligenceError(
                f"{name} must be finite."
            )

    @staticmethod
    def _validate_datetime(
        value: datetime,
        name: str,
    ) -> None:
        """Validate datetime input."""

        # Require an actual datetime.
        if not isinstance(value, datetime):
            raise TreasuryYieldIntelligenceError(
                f"{name} must be a datetime."
            )

    @staticmethod
    def _validate_timezone(
        observation: MacroObservation,
        decision_timestamp: datetime,
    ) -> None:
        """Ensure timestamps have compatible timezone semantics."""

        # Determine whether the decision timestamp is timezone-aware.
        decision_aware = decision_timestamp.tzinfo is not None

        # Determine whether the observation timestamp is timezone-aware.
        observation_aware = observation.timestamp.tzinfo is not None

        # Reject mixed naive/aware timestamps.
        if decision_aware != observation_aware:
            raise TreasuryYieldIntelligenceError(
                "decision_timestamp and observation timestamp "
                "must use the same timezone-awareness."
            )

    @staticmethod
    def _basis_point_change(
        value: float | None,
        previous: float | None,
    ) -> float | None:
        """Convert a yield change into basis points.

        Treasury yield values are represented as percentage points.

        Example:
            4.20 -> 4.25
            0.05 percentage points
            = 5 basis points
        """

        # Both values are required.
        if value is None or previous is None:
            return None

        # Calculate percentage-point movement and convert to basis points.
        result = (value - previous) * 100.0

        # Remove insignificant floating-point residue.
        if abs(result) < 1e-12:
            return 0.0

        # Return the raw calculated basis-point movement.
        return result

    @staticmethod
    def _percentage_change(
        value: float | None,
        previous: float | None,
    ) -> float | None:
        """Calculate percentage change from the previous yield."""

        # Both values are required.
        if value is None or previous is None:
            return None

        # Avoid division by zero.
        if previous == 0:
            return None

        # Calculate percentage change.
        result = (
            (value - previous)
            / abs(previous)
        ) * 100.0

        # Remove floating-point residue around zero.
        if abs(result) < 1e-12:
            return 0.0

        # Return percentage change.
        return result

    def _classify(
        self,
        direction: MacroDirection,
        change_basis_points: float | None,
    ) -> TreasuryYieldLevel:
        """Classify Treasury-yield direction and magnitude."""

        # Unknown direction cannot safely create a yield classification.
        if direction is MacroDirection.UNKNOWN:
            return TreasuryYieldLevel.UNKNOWN

        # Explicitly stable observations are stable.
        if direction is MacroDirection.STABLE:
            return TreasuryYieldLevel.STABLE

        # Without a previous value, direction alone is lower confidence.
        if change_basis_points is None:

            # Direction indicates rising yields.
            if direction is MacroDirection.RISING:
                return TreasuryYieldLevel.RISING

            # Direction indicates falling yields.
            if direction is MacroDirection.FALLING:
                return TreasuryYieldLevel.FALLING

            # Defensive fallback.
            return TreasuryYieldLevel.UNKNOWN

        # IMPORTANT:
        # Floating-point arithmetic can produce:
        #
        #     4.25 - 4.20 = 0.049999999999...
        #
        # which becomes something like:
        #
        #     4.999999999999996 bps
        #
        # Normalize the value before comparing it with exact thresholds.
        normalized_bps = round(
            change_basis_points,
            10,
        )

        # Handle rising Treasury yields.
        if direction is MacroDirection.RISING:

            # At or above the strong threshold is strongly rising.
            if normalized_bps >= self._strong_change_bps:
                return TreasuryYieldLevel.STRONG_RISING

            # At or above the significant threshold is rising.
            if normalized_bps >= self._significant_change_bps:
                return TreasuryYieldLevel.RISING

            # Smaller movement is treated as stable.
            return TreasuryYieldLevel.STABLE

        # Handle falling Treasury yields.
        if direction is MacroDirection.FALLING:

            # At or below the negative strong threshold is strongly falling.
            if normalized_bps <= -self._strong_change_bps:
                return TreasuryYieldLevel.STRONG_FALLING

            # At or below the negative significant threshold is falling.
            if normalized_bps <= -self._significant_change_bps:
                return TreasuryYieldLevel.FALLING

            # Smaller movement is treated as stable.
            return TreasuryYieldLevel.STABLE

        # Defensive fallback.
        return TreasuryYieldLevel.UNKNOWN

    def analyze(
        self,
        observations: list[MacroObservation]
        | tuple[MacroObservation, ...],
        decision_timestamp: datetime,
        indicator: MacroIndicator = MacroIndicator.US_10Y_YIELD,
    ) -> TreasuryYieldAssessment:
        """Analyze one Treasury maturity using historical-safe data."""

        # Validate the decision timestamp.
        self._validate_datetime(
            decision_timestamp,
            "decision_timestamp",
        )

        # Validate the requested Treasury maturity.
        if indicator not in self.TREASURY_INDICATORS:
            raise TreasuryYieldIntelligenceError(
                "indicator must be one of the supported Treasury yields."
            )

        # Require a list or tuple.
        if not isinstance(observations, (list, tuple)):
            raise TreasuryYieldIntelligenceError(
                "observations must be a list or tuple."
            )

        # Validate every supplied observation.
        for observation in observations:

            # Every item must be a MacroObservation.
            if not isinstance(observation, MacroObservation):
                raise TreasuryYieldIntelligenceError(
                    "observations must contain MacroObservation values."
                )

        # Select only observations for the requested maturity.
        yield_observations = [
            observation
            for observation in observations
            if observation.indicator is indicator
        ]

        # No matching observations means UNKNOWN.
        if not yield_observations:
            return TreasuryYieldAssessment(
                indicator=indicator,
                value=None,
                previous=None,
                change_from_previous=None,
                change_basis_points=None,
                percentage_change=None,
                direction=MacroDirection.UNKNOWN,
                level=TreasuryYieldLevel.UNKNOWN,
                confidence=0.0,
                sufficient_data=False,
                observation_timestamp=None,
                decision_timestamp=decision_timestamp,
                reasons=(
                    f"No observation was supplied for {indicator.value}.",
                ),
            )

        # Validate timestamp semantics for every relevant observation.
        for observation in yield_observations:
            self._validate_timezone(
                observation,
                decision_timestamp,
            )

        # Keep only observations available at the decision timestamp.
        #
        # This is the core no-lookahead protection.
        available = [
            observation
            for observation in yield_observations
            if observation.timestamp <= decision_timestamp
        ]

        # If all observations are in the future, return UNKNOWN.
        if not available:
            return TreasuryYieldAssessment(
                indicator=indicator,
                value=None,
                previous=None,
                change_from_previous=None,
                change_basis_points=None,
                percentage_change=None,
                direction=MacroDirection.UNKNOWN,
                level=TreasuryYieldLevel.UNKNOWN,
                confidence=0.0,
                sufficient_data=False,
                observation_timestamp=None,
                decision_timestamp=decision_timestamp,
                reasons=(
                    f"All {indicator.value} observations occur after "
                    "the decision timestamp.",
                ),
            )

        # Select the latest observation that is historically available.
        latest = max(
            available,
            key=lambda observation: observation.timestamp,
        )

        # Extract the absolute movement.
        change = latest.change_from_previous

        # Convert movement to basis points.
        change_basis_points = self._basis_point_change(
            latest.value,
            latest.previous,
        )

        # Calculate percentage movement.
        percentage_change = self._percentage_change(
            latest.value,
            latest.previous,
        )

        # Determine Treasury-specific classification.
        level = self._classify(
            latest.direction,
            change_basis_points,
        )

        # Determine confidence.
        if latest.direction is MacroDirection.UNKNOWN:
            # Unknown direction gives no usable confidence.
            confidence = 0.0

        elif change_basis_points is None:
            # Direction exists, but previous value is unavailable.
            confidence = 50.0

        else:
            # Direction and measurable movement are both available.
            confidence = 100.0

        # Build audit-friendly reasons.
        reasons: list[str] = []

        # Explain the observed direction.
        if latest.direction is MacroDirection.RISING:
            reasons.append(
                f"{indicator.value} is rising, indicating higher "
                "Treasury yields."
            )

        elif latest.direction is MacroDirection.FALLING:
            reasons.append(
                f"{indicator.value} is falling, indicating lower "
                "Treasury yields."
            )

        elif latest.direction is MacroDirection.STABLE:
            reasons.append(
                f"{indicator.value} is stable."
            )

        else:
            reasons.append(
                f"{indicator.value} direction is unknown."
            )

        # Explain measurable basis-point movement.
        if change_basis_points is not None:

            # Positive yield movement.
            if change_basis_points > 0:
                reasons.append(
                    f"{indicator.value} increased by "
                    f"{change_basis_points:.6f} basis points."
                )

            # Negative yield movement.
            elif change_basis_points < 0:
                reasons.append(
                    f"{indicator.value} decreased by "
                    f"{abs(change_basis_points):.6f} basis points."
                )

            # Zero movement.
            else:
                reasons.append(
                    f"{indicator.value} did not change from the "
                    "previous observation."
                )

        # Return the complete immutable assessment.
        return TreasuryYieldAssessment(
            indicator=indicator,
            value=float(latest.value),
            previous=(
                float(latest.previous)
                if latest.previous is not None
                else None
            ),
            change_from_previous=(
                float(change)
                if change is not None
                else None
            ),
            change_basis_points=change_basis_points,
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

    def analyze_all(
        self,
        observations: list[MacroObservation]
        | tuple[MacroObservation, ...],
        decision_timestamp: datetime,
    ) -> dict[
        MacroIndicator,
        TreasuryYieldAssessment,
    ]:
        """Analyze all supported Treasury maturities."""

        # Prepare the result dictionary.
        assessments: dict[
            MacroIndicator,
            TreasuryYieldAssessment,
        ] = {}

        # Analyze every supported Treasury maturity independently.
        for indicator in self.TREASURY_INDICATORS:
            assessments[indicator] = self.analyze(
                observations=observations,
                decision_timestamp=decision_timestamp,
                indicator=indicator,
            )

        # Return all maturity assessments.
        return assessments

    def analyze_xauusd(
        self,
        observations: list[MacroObservation]
        | tuple[MacroObservation, ...],
        decision_timestamp: datetime,
        indicator: MacroIndicator = MacroIndicator.US_10Y_YIELD,
    ) -> TreasuryYieldAssessment:
        """Analyze Treasury yields in an XAUUSD macro context."""

        # This wrapper intentionally returns Treasury information only.
        #
        # It does NOT convert rising/falling yields into a gold
        # buy/sell decision. That responsibility belongs to later
        # macro-confluence and signal-fusion layers.
        return self.analyze(
            observations=observations,
            decision_timestamp=decision_timestamp,
            indicator=indicator,
        )