# app/trading/macro/inflation_intelligence.py

"""Deterministic U.S. inflation intelligence.

This module analyzes CPI, Core CPI, PCE, and Core PCE observations.

Important:
- No external data fetching.
- No LLM dependency.
- No trade decision.
- No direct XAUUSD buy/sell decision.
- Future observations are excluded from historical analysis.
- Actual/forecast information is only used when it exists in the
  selected historical observation.
"""

from __future__ import annotations

# Import dataclass for immutable assessment objects.
from dataclasses import dataclass

# Import datetime for historical timestamp validation.
from datetime import datetime

# Import Enum for explicit inflation classifications.
from enum import Enum

# Import math for finite-number validation.
import math

# Import the existing macro observation models.
from app.trading.macro.macro_observation import (
    MacroDirection,
    MacroIndicator,
    MacroObservation,
)


class InflationIntelligenceError(ValueError):
    """Raised when inflation analysis receives invalid input."""


class InflationLevel(str, Enum):
    """Classification of inflation movement and surprise."""

    # Inflation is strongly rising.
    STRONG_HOT = "STRONG_HOT"

    # Inflation is meaningfully rising.
    HOT = "HOT"

    # Inflation environment is neutral/stable.
    NEUTRAL = "NEUTRAL"

    # Inflation is meaningfully falling.
    COOLING = "COOLING"

    # Inflation is strongly falling.
    STRONG_COOLING = "STRONG_COOLING"

    # Inflation information is unavailable.
    UNKNOWN = "UNKNOWN"


class InflationSurpriseLevel(str, Enum):
    """Classification of inflation actual-vs-forecast surprise."""

    # Actual inflation is substantially above forecast.
    STRONG_UPSIDE = "STRONG_UPSIDE"

    # Actual inflation is above forecast.
    UPSIDE = "UPSIDE"

    # Actual inflation is approximately equal to forecast.
    IN_LINE = "IN_LINE"

    # Actual inflation is below forecast.
    DOWNSIDE = "DOWNSIDE"

    # Actual inflation is substantially below forecast.
    STRONG_DOWNSIDE = "STRONG_DOWNSIDE"

    # Forecast comparison is unavailable.
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class InflationAssessment:
    """Immutable and auditable inflation assessment."""

    # Inflation indicator being analyzed.
    indicator: MacroIndicator

    # Latest usable inflation value.
    value: float | None

    # Previous inflation value.
    previous: float | None

    # Forecast inflation value.
    forecast: float | None

    # Change from the previous observation.
    change_from_previous: float | None

    # Actual-minus-forecast surprise.
    surprise: float | None

    # Percentage change from previous.
    percentage_change: float | None

    # Existing normalized inflation direction.
    direction: MacroDirection

    # Overall inflation classification.
    level: InflationLevel

    # Actual-versus-forecast classification.
    surprise_level: InflationSurpriseLevel

    # Confidence from 0 to 100.
    confidence: float

    # Whether sufficient information exists.
    sufficient_data: bool

    # Timestamp of the selected observation.
    observation_timestamp: datetime | None

    # Historical decision timestamp.
    decision_timestamp: datetime

    # Human-readable audit explanations.
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the immutable assessment."""

        # Only supported inflation indicators are valid.
        if self.indicator not in (
            MacroIndicator.CPI,
            MacroIndicator.CORE_CPI,
            MacroIndicator.PCE,
            MacroIndicator.CORE_PCE,
        ):
            raise InflationIntelligenceError(
                "indicator must be CPI, CORE_CPI, PCE, or CORE_PCE."
            )

        # Direction must use the existing enum.
        if not isinstance(self.direction, MacroDirection):
            raise InflationIntelligenceError(
                "direction must be a MacroDirection."
            )

        # Level must use the InflationLevel enum.
        if not isinstance(self.level, InflationLevel):
            raise InflationIntelligenceError(
                "level must be an InflationLevel."
            )

        # Surprise classification must use its enum.
        if not isinstance(
            self.surprise_level,
            InflationSurpriseLevel,
        ):
            raise InflationIntelligenceError(
                "surprise_level must be an InflationSurpriseLevel."
            )

        # Decision timestamp must be a datetime.
        if not isinstance(self.decision_timestamp, datetime):
            raise InflationIntelligenceError(
                "decision_timestamp must be a datetime."
            )

        # Validate optional numeric values.
        for name, value in (
            ("value", self.value),
            ("previous", self.previous),
            ("forecast", self.forecast),
            ("change_from_previous", self.change_from_previous),
            ("surprise", self.surprise),
            ("percentage_change", self.percentage_change),
        ):
            # None represents unavailable information.
            if value is None:
                continue

            # Reject booleans.
            if isinstance(value, bool):
                raise InflationIntelligenceError(
                    f"{name} must be numeric or None."
                )

            # Require numeric values.
            if not isinstance(value, (int, float)):
                raise InflationIntelligenceError(
                    f"{name} must be numeric or None."
                )

            # Reject NaN and infinity.
            if not math.isfinite(float(value)):
                raise InflationIntelligenceError(
                    f"{name} must be finite."
                )

        # Validate confidence.
        if isinstance(self.confidence, bool):
            raise InflationIntelligenceError(
                "confidence must be numeric."
            )

        if not isinstance(self.confidence, (int, float)):
            raise InflationIntelligenceError(
                "confidence must be numeric."
            )

        if not math.isfinite(float(self.confidence)):
            raise InflationIntelligenceError(
                "confidence must be finite."
            )

        # Confidence is a percentage.
        if not 0.0 <= float(self.confidence) <= 100.0:
            raise InflationIntelligenceError(
                "confidence must be between 0 and 100."
            )

        # Observation timestamp is optional.
        if self.observation_timestamp is not None:
            if not isinstance(self.observation_timestamp, datetime):
                raise InflationIntelligenceError(
                    "observation_timestamp must be a datetime or None."
                )


class InflationIntelligence:
    """Deterministic U.S. inflation intelligence."""

    # A 0.05 percentage-point movement is meaningful.
    DEFAULT_SIGNIFICANT_CHANGE = 0.05

    # A 0.20 percentage-point movement is considered strong.
    DEFAULT_STRONG_CHANGE = 0.20

    # A 0.05 percentage-point surprise is meaningful.
    DEFAULT_SIGNIFICANT_SURPRISE = 0.05

    # A 0.20 percentage-point surprise is strong.
    DEFAULT_STRONG_SURPRISE = 0.20

    # Supported inflation indicators.
    INFLATION_INDICATORS = (
        MacroIndicator.CPI,
        MacroIndicator.CORE_CPI,
        MacroIndicator.PCE,
        MacroIndicator.CORE_PCE,
    )

    def __init__(
        self,
        significant_change: float = DEFAULT_SIGNIFICANT_CHANGE,
        strong_change: float = DEFAULT_STRONG_CHANGE,
        significant_surprise: float = DEFAULT_SIGNIFICANT_SURPRISE,
        strong_surprise: float = DEFAULT_STRONG_SURPRISE,
    ) -> None:
        """Initialize inflation intelligence."""

        # Validate movement thresholds.
        self._validate_positive(
            significant_change,
            "significant_change",
        )

        self._validate_positive(
            strong_change,
            "strong_change",
        )

        # Validate surprise thresholds.
        self._validate_positive(
            significant_surprise,
            "significant_surprise",
        )

        self._validate_positive(
            strong_surprise,
            "strong_surprise",
        )

        # Strong movement cannot be smaller than significant movement.
        if strong_change < significant_change:
            raise InflationIntelligenceError(
                "strong_change must be greater than or equal to "
                "significant_change."
            )

        # Strong surprise cannot be smaller than significant surprise.
        if strong_surprise < significant_surprise:
            raise InflationIntelligenceError(
                "strong_surprise must be greater than or equal to "
                "significant_surprise."
            )

        # Store validated thresholds.
        self._significant_change = float(
            significant_change
        )

        self._strong_change = float(
            strong_change
        )

        self._significant_surprise = float(
            significant_surprise
        )

        self._strong_surprise = float(
            strong_surprise
        )

    @staticmethod
    def _validate_positive(
        value: float,
        name: str,
    ) -> None:
        """Validate a positive finite numeric value."""

        # Reject booleans.
        if isinstance(value, bool):
            raise InflationIntelligenceError(
                f"{name} must be numeric, not bool."
            )

        # Require numeric values.
        if not isinstance(value, (int, float)):
            raise InflationIntelligenceError(
                f"{name} must be numeric."
            )

        # Reject NaN and infinity.
        if not math.isfinite(float(value)):
            raise InflationIntelligenceError(
                f"{name} must be finite."
            )

        # Thresholds must be positive.
        if float(value) <= 0:
            raise InflationIntelligenceError(
                f"{name} must be greater than zero."
            )

    @staticmethod
    def _validate_datetime(
        value: datetime,
        name: str,
    ) -> None:
        """Validate datetime input."""

        # Require a datetime.
        if not isinstance(value, datetime):
            raise InflationIntelligenceError(
                f"{name} must be a datetime."
            )

    @staticmethod
    def _validate_timezone(
        observation: MacroObservation,
        decision_timestamp: datetime,
    ) -> None:
        """Validate timezone-awareness compatibility."""

        # Determine decision timestamp awareness.
        decision_aware = decision_timestamp.tzinfo is not None

        # Determine observation timestamp awareness.
        observation_aware = observation.timestamp.tzinfo is not None

        # Reject mixed timestamp semantics.
        if decision_aware != observation_aware:
            raise InflationIntelligenceError(
                "decision_timestamp and observation timestamp "
                "must use the same timezone-awareness."
            )

    @staticmethod
    def _percentage_change(
        value: float | None,
        previous: float | None,
    ) -> float | None:
        """Calculate percentage change from previous inflation."""

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

        # Normalize floating-point noise.
        if abs(result) < 1e-12:
            return 0.0

        # Return a deterministic rounded value.
        return round(result, 10)

    @staticmethod
    def _normalize(
        value: float,
    ) -> float:
        """Normalize floating-point comparison values."""

        # Round enough decimal places to eliminate binary residue
        # without changing meaningful economic precision.
        return round(value, 10)

    def _classify_direction(
        self,
        direction: MacroDirection,
        change: float | None,
    ) -> InflationLevel:
        """Classify inflation movement."""

        # Unknown direction cannot safely create an inflation signal.
        if direction is MacroDirection.UNKNOWN:
            return InflationLevel.UNKNOWN

        # Explicitly stable inflation is neutral.
        if direction is MacroDirection.STABLE:
            return InflationLevel.NEUTRAL

        # Without previous inflation, direction alone is lower confidence.
        if change is None:
            if direction is MacroDirection.RISING:
                return InflationLevel.HOT

            if direction is MacroDirection.FALLING:
                return InflationLevel.COOLING

            return InflationLevel.UNKNOWN

        # Normalize floating-point residue.
        normalized_change = self._normalize(change)

        # Rising inflation.
        if direction is MacroDirection.RISING:

            # Strong inflation acceleration.
            if normalized_change >= self._strong_change:
                return InflationLevel.STRONG_HOT

            # Meaningful inflation acceleration.
            if normalized_change >= self._significant_change:
                return InflationLevel.HOT

            # Small movement is neutral.
            return InflationLevel.NEUTRAL

        # Falling inflation.
        if direction is MacroDirection.FALLING:

            # Strong inflation cooling.
            if normalized_change <= -self._strong_change:
                return InflationLevel.STRONG_COOLING

            # Meaningful inflation cooling.
            if normalized_change <= -self._significant_change:
                return InflationLevel.COOLING

            # Small movement is neutral.
            return InflationLevel.NEUTRAL

        # Defensive fallback.
        return InflationLevel.UNKNOWN

    def _classify_surprise(
        self,
        surprise: float | None,
    ) -> InflationSurpriseLevel:
        """Classify actual inflation versus forecast."""

        # No forecast comparison means unknown surprise.
        if surprise is None:
            return InflationSurpriseLevel.UNKNOWN

        # Normalize floating-point comparison residue.
        normalized_surprise = self._normalize(surprise)

        # Strong upside inflation surprise.
        if normalized_surprise >= self._strong_surprise:
            return InflationSurpriseLevel.STRONG_UPSIDE

        # Meaningful upside inflation surprise.
        if normalized_surprise >= self._significant_surprise:
            return InflationSurpriseLevel.UPSIDE

        # Strong downside inflation surprise.
        if normalized_surprise <= -self._strong_surprise:
            return InflationSurpriseLevel.STRONG_DOWNSIDE

        # Meaningful downside inflation surprise.
        if normalized_surprise <= -self._significant_surprise:
            return InflationSurpriseLevel.DOWNSIDE

        # Small difference is considered in-line.
        return InflationSurpriseLevel.IN_LINE

    def analyze(
        self,
        observations: list[MacroObservation]
        | tuple[MacroObservation, ...],
        decision_timestamp: datetime,
        indicator: MacroIndicator = MacroIndicator.CPI,
    ) -> InflationAssessment:
        """Analyze one inflation indicator historically safely."""

        # Validate decision timestamp.
        self._validate_datetime(
            decision_timestamp,
            "decision_timestamp",
        )

        # Validate requested inflation indicator.
        if indicator not in self.INFLATION_INDICATORS:
            raise InflationIntelligenceError(
                "indicator must be CPI, CORE_CPI, PCE, or CORE_PCE."
            )

        # Require list or tuple input.
        if not isinstance(observations, (list, tuple)):
            raise InflationIntelligenceError(
                "observations must be a list or tuple."
            )

        # Validate observation types.
        for observation in observations:

            # Every item must be a MacroObservation.
            if not isinstance(observation, MacroObservation):
                raise InflationIntelligenceError(
                    "observations must contain MacroObservation values."
                )

        # Select observations for the requested inflation indicator.
        inflation_observations = [
            observation
            for observation in observations
            if observation.indicator is indicator
        ]

        # No matching data means UNKNOWN.
        if not inflation_observations:
            return InflationAssessment(
                indicator=indicator,
                value=None,
                previous=None,
                forecast=None,
                change_from_previous=None,
                surprise=None,
                percentage_change=None,
                direction=MacroDirection.UNKNOWN,
                level=InflationLevel.UNKNOWN,
                surprise_level=InflationSurpriseLevel.UNKNOWN,
                confidence=0.0,
                sufficient_data=False,
                observation_timestamp=None,
                decision_timestamp=decision_timestamp,
                reasons=(
                    f"No observation was supplied for {indicator.value}.",
                ),
            )

        # Validate timezone semantics.
        for observation in inflation_observations:
            self._validate_timezone(
                observation,
                decision_timestamp,
            )

        # Keep only observations available at the decision timestamp.
        available = [
            observation
            for observation in inflation_observations
            if observation.timestamp <= decision_timestamp
        ]

        # Never use future inflation data.
        if not available:
            return InflationAssessment(
                indicator=indicator,
                value=None,
                previous=None,
                forecast=None,
                change_from_previous=None,
                surprise=None,
                percentage_change=None,
                direction=MacroDirection.UNKNOWN,
                level=InflationLevel.UNKNOWN,
                surprise_level=InflationSurpriseLevel.UNKNOWN,
                confidence=0.0,
                sufficient_data=False,
                observation_timestamp=None,
                decision_timestamp=decision_timestamp,
                reasons=(
                    f"All {indicator.value} observations occur after "
                    "the decision timestamp.",
                ),
            )

        # Select the latest historical-safe observation.
        latest = max(
            available,
            key=lambda observation: observation.timestamp,
        )

        # Extract previous-value movement.
        change = latest.change_from_previous

        # Extract actual-versus-forecast surprise.
        surprise = latest.surprise

        # Calculate percentage change.
        percentage_change = self._percentage_change(
            latest.value,
            latest.previous,
        )

        # Classify inflation direction and magnitude.
        level = self._classify_direction(
            latest.direction,
            change,
        )

        # Classify forecast surprise.
        surprise_level = self._classify_surprise(
            surprise,
        )

        # Calculate confidence from available information.
        if latest.direction is MacroDirection.UNKNOWN:
            confidence = 0.0

        elif change is None:
            confidence = 50.0

        else:
            confidence = 100.0

        # Build audit reasons.
        reasons: list[str] = []

        # Explain inflation direction.
        if latest.direction is MacroDirection.RISING:
            reasons.append(
                f"{indicator.value} is rising, indicating a hotter "
                "inflation trend."
            )

        elif latest.direction is MacroDirection.FALLING:
            reasons.append(
                f"{indicator.value} is falling, indicating a cooling "
                "inflation trend."
            )

        elif latest.direction is MacroDirection.STABLE:
            reasons.append(
                f"{indicator.value} is stable."
            )

        else:
            reasons.append(
                f"{indicator.value} direction is unknown."
            )

        # Explain previous-value movement.
        if change is not None:

            # Positive inflation change.
            if change > 0:
                reasons.append(
                    f"{indicator.value} increased by "
                    f"{change:.6f} percentage points from the "
                    "previous observation."
                )

            # Negative inflation change.
            elif change < 0:
                reasons.append(
                    f"{indicator.value} decreased by "
                    f"{abs(change):.6f} percentage points from the "
                    "previous observation."
                )

            # No movement.
            else:
                reasons.append(
                    f"{indicator.value} was unchanged from the "
                    "previous observation."
                )

        # Explain forecast availability.
        if latest.forecast is not None:
            reasons.append(
                f"Forecast for {indicator.value} was "
                f"{latest.forecast:.6f}."
            )

        # Explain surprise.
        if surprise is not None:

            # Inflation above forecast.
            if surprise > 0:
                reasons.append(
                    f"Actual {indicator.value} was "
                    f"{surprise:.6f} percentage points above forecast."
                )

            # Inflation below forecast.
            elif surprise < 0:
                reasons.append(
                    f"Actual {indicator.value} was "
                    f"{abs(surprise):.6f} percentage points below forecast."
                )

            # Exact forecast match.
            else:
                reasons.append(
                    f"Actual {indicator.value} matched forecast."
                )

        # Return the complete immutable assessment.
        return InflationAssessment(
            indicator=indicator,
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
            surprise=(
                float(surprise)
                if surprise is not None
                else None
            ),
            percentage_change=percentage_change,
            direction=latest.direction,
            level=level,
            surprise_level=surprise_level,
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
        InflationAssessment,
    ]:
        """Analyze all supported inflation indicators."""

        # Prepare the result dictionary.
        assessments: dict[
            MacroIndicator,
            InflationAssessment,
        ] = {}

        # Analyze CPI, Core CPI, PCE, and Core PCE independently.
        for indicator in self.INFLATION_INDICATORS:
            assessments[indicator] = self.analyze(
                observations=observations,
                decision_timestamp=decision_timestamp,
                indicator=indicator,
            )

        # Return all inflation assessments.
        return assessments

    def analyze_xauusd(
        self,
        observations: list[MacroObservation]
        | tuple[MacroObservation, ...],
        decision_timestamp: datetime,
        indicator: MacroIndicator = MacroIndicator.CPI,
    ) -> InflationAssessment:
        """Analyze inflation in an XAUUSD macro context."""

        # This method returns inflation intelligence only.
        #
        # It intentionally does not create a gold buy/sell decision.
        return self.analyze(
            observations=observations,
            decision_timestamp=decision_timestamp,
            indicator=indicator,
        )