"""
Employment intelligence for the macro analysis layer.

This module provides deterministic analysis for:
- U.S. Non-Farm Payrolls (NFP)
- U.S. Unemployment Rate

Important:
- This module does NOT generate a trade signal.
- This module does NOT execute trades.
- This module does NOT fetch external data.
- Historical/no-lookahead safety is enforced through decision timestamps.
"""

from __future__ import annotations

# Import dataclass utilities for immutable result objects.
from dataclasses import dataclass

# Import Enum for strongly typed classification values.
from enum import Enum

# Import datetime for historical decision-time validation.
from datetime import datetime

# Import math for finite-number validation.
import math

# Import the existing macro observation model.
from app.trading.macro.macro_observation import (
    MacroDirection,
    MacroIndicator,
    MacroObservation,
)


class EmploymentIntelligenceError(ValueError):
    """Raised when employment intelligence receives invalid input."""


class EmploymentLevel(str, Enum):
    """Represents the strength of the U.S. employment environment."""

    STRONG_HOT = "strong_hot"
    HOT = "hot"
    NEUTRAL = "neutral"
    COOLING = "cooling"
    STRONG_COOLING = "strong_cooling"
    UNKNOWN = "unknown"


class EmploymentSurpriseLevel(str, Enum):
    """Represents employment data surprise relative to forecast."""

    STRONG_UPSIDE = "strong_upside"
    UPSIDE = "upside"
    IN_LINE = "in_line"
    DOWNSIDE = "downside"
    STRONG_DOWNSIDE = "strong_downside"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class EmploymentAssessment:
    """
    Immutable employment intelligence result.

    The assessment describes the employment environment only.
    It does not make a trading decision.
    """

    # Indicator that was analyzed.
    indicator: MacroIndicator

    # Latest observed value available at the decision timestamp.
    value: float

    # Previous value when available.
    previous: float | None

    # Forecast value when available.
    forecast: float | None

    # Raw movement direction of the underlying indicator.
    direction: MacroDirection

    # Employment-strength direction after interpreting the indicator.
    employment_direction: MacroDirection

    # Absolute change from the previous observation.
    change_from_previous: float | None

    # Surprise relative to forecast.
    surprise: float | None

    # Percentage change from previous value when mathematically valid.
    percentage_change: float | None

    # Employment environment classification.
    level: EmploymentLevel

    # Forecast-surprise classification.
    surprise_level: EmploymentSurpriseLevel

    # Confidence score from 0 to 100.
    confidence: float

    # Whether enough information was available for a meaningful assessment.
    sufficient_data: bool

    # Timestamp of the observation used.
    observation_timestamp: datetime

    # Timestamp at which the decision was made.
    decision_timestamp: datetime

    # Human-readable deterministic explanations.
    reasons: tuple[str, ...]

    @property
    def has_previous(self) -> bool:
        """Return whether a previous observation was available."""

        # Previous data is available when the field is not None.
        return self.previous is not None

    @property
    def has_forecast(self) -> bool:
        """Return whether a forecast was available."""

        # Forecast data is available when the field is not None.
        return self.forecast is not None

    @property
    def has_change(self) -> bool:
        """Return whether a previous-value change was calculated."""

        # A calculated change exists when the field is not None.
        return self.change_from_previous is not None

    @property
    def has_surprise(self) -> bool:
        """Return whether a forecast surprise was calculated."""

        # A calculated surprise exists when the field is not None.
        return self.surprise is not None

    def to_dict(self) -> dict[str, object]:
        """Serialize the assessment into a JSON-friendly dictionary."""

        # Return every important field explicitly for deterministic serialization.
        return {
            "indicator": self.indicator.value,
            "value": self.value,
            "previous": self.previous,
            "forecast": self.forecast,
            "direction": self.direction.value,
            "employment_direction": self.employment_direction.value,
            "change_from_previous": self.change_from_previous,
            "surprise": self.surprise,
            "percentage_change": self.percentage_change,
            "level": self.level.value,
            "surprise_level": self.surprise_level.value,
            "confidence": self.confidence,
            "sufficient_data": self.sufficient_data,
            "observation_timestamp": self.observation_timestamp.isoformat(),
            "decision_timestamp": self.decision_timestamp.isoformat(),
            "reasons": list(self.reasons),
        }


class EmploymentIntelligence:
    """
    Deterministic U.S. employment intelligence engine.

    Supported indicators:
    - NFP
    - UNEMPLOYMENT_RATE

    No external APIs or LLM decisions are used.
    """

    # Default NFP significant movement in thousands of jobs.
    DEFAULT_SIGNIFICANT_NFP_CHANGE = 25.0

    # Default NFP strong movement in thousands of jobs.
    DEFAULT_STRONG_NFP_CHANGE = 75.0

    # Default unemployment-rate significant movement in percentage points.
    DEFAULT_SIGNIFICANT_UNEMPLOYMENT_CHANGE = 0.10

    # Default unemployment-rate strong movement in percentage points.
    DEFAULT_STRONG_UNEMPLOYMENT_CHANGE = 0.30

    # Default NFP significant forecast surprise.
    DEFAULT_SIGNIFICANT_NFP_SURPRISE = 25.0

    # Default NFP strong forecast surprise.
    DEFAULT_STRONG_NFP_SURPRISE = 75.0

    # Default unemployment significant forecast surprise.
    DEFAULT_SIGNIFICANT_UNEMPLOYMENT_SURPRISE = 0.10

    # Default unemployment strong forecast surprise.
    DEFAULT_STRONG_UNEMPLOYMENT_SURPRISE = 0.30

    def __init__(
        self,
        *,
        significant_nfp_change: float = DEFAULT_SIGNIFICANT_NFP_CHANGE,
        strong_nfp_change: float = DEFAULT_STRONG_NFP_CHANGE,
        significant_unemployment_change: float = DEFAULT_SIGNIFICANT_UNEMPLOYMENT_CHANGE,
        strong_unemployment_change: float = DEFAULT_STRONG_UNEMPLOYMENT_CHANGE,
        significant_nfp_surprise: float = DEFAULT_SIGNIFICANT_NFP_SURPRISE,
        strong_nfp_surprise: float = DEFAULT_STRONG_NFP_SURPRISE,
        significant_unemployment_surprise: float = DEFAULT_SIGNIFICANT_UNEMPLOYMENT_SURPRISE,
        strong_unemployment_surprise: float = DEFAULT_STRONG_UNEMPLOYMENT_SURPRISE,
    ) -> None:
        """Initialize configurable employment thresholds."""

        # Store and validate every configured threshold.
        self._validate_thresholds(
            significant_nfp_change,
            strong_nfp_change,
            significant_unemployment_change,
            strong_unemployment_change,
            significant_nfp_surprise,
            strong_nfp_surprise,
            significant_unemployment_surprise,
            strong_unemployment_surprise,
        )

        # Store NFP movement thresholds.
        self.significant_nfp_change = significant_nfp_change
        self.strong_nfp_change = strong_nfp_change

        # Store unemployment movement thresholds.
        self.significant_unemployment_change = significant_unemployment_change
        self.strong_unemployment_change = strong_unemployment_change

        # Store NFP surprise thresholds.
        self.significant_nfp_surprise = significant_nfp_surprise
        self.strong_nfp_surprise = strong_nfp_surprise

        # Store unemployment surprise thresholds.
        self.significant_unemployment_surprise = significant_unemployment_surprise
        self.strong_unemployment_surprise = strong_unemployment_surprise

    @staticmethod
    def _validate_thresholds(*thresholds: float) -> None:
        """Validate that all thresholds are positive finite numbers."""

        # Iterate over every configured threshold.
        for threshold in thresholds:
            # Reject booleans because bool is a subclass of int in Python.
            if isinstance(threshold, bool):
                raise EmploymentIntelligenceError(
                    "Employment thresholds must be numeric, not boolean."
                )

            # Reject non-numeric values.
            if not isinstance(threshold, (int, float)):
                raise EmploymentIntelligenceError(
                    "Employment thresholds must be numeric."
                )

            # Convert to float for finite-number validation.
            numeric_threshold = float(threshold)

            # Reject NaN and infinity.
            if not math.isfinite(numeric_threshold):
                raise EmploymentIntelligenceError(
                    "Employment thresholds must be finite."
                )

            # Thresholds must be strictly positive.
            if numeric_threshold <= 0:
                raise EmploymentIntelligenceError(
                    "Employment thresholds must be greater than zero."
                )

    @staticmethod
    def _validate_decision_timestamp(decision_timestamp: datetime) -> None:
        """Validate the decision timestamp."""

        # The decision timestamp must be a datetime.
        if not isinstance(decision_timestamp, datetime):
            raise EmploymentIntelligenceError(
                "decision_timestamp must be a datetime."
            )

        # Naive datetimes are rejected because historical ordering becomes ambiguous.
        if decision_timestamp.tzinfo is None:
            raise EmploymentIntelligenceError(
                "decision_timestamp must be timezone-aware."
            )

    @staticmethod
    def _validate_observation_timestamp(observation: MacroObservation) -> None:
        """Validate that the observation timestamp is timezone-aware."""

        # MacroObservation already validates its own timestamp type.
        if observation.timestamp.tzinfo is None:
            raise EmploymentIntelligenceError(
                "Employment observation timestamp must be timezone-aware."
            )

    @staticmethod
    def _validate_indicator(indicator: MacroIndicator) -> None:
        """Validate that the indicator is employment-related."""

        # Only NFP and unemployment rate belong to this engine.
        if indicator not in {
            MacroIndicator.NFP,
            MacroIndicator.UNEMPLOYMENT_RATE,
        }:
            raise EmploymentIntelligenceError(
                "EmploymentIntelligence supports only NFP and UNEMPLOYMENT_RATE."
            )

    @staticmethod
    def _validate_observations(
        observations: list[MacroObservation],
    ) -> None:
        """Validate the observation collection."""

        # Require a list for predictable behavior.
        if not isinstance(observations, list):
            raise EmploymentIntelligenceError(
                "observations must be a list of MacroObservation objects."
            )

        # Validate every observation.
        for observation in observations:
            # Reject unexpected object types.
            if not isinstance(observation, MacroObservation):
                raise EmploymentIntelligenceError(
                    "Every observation must be a MacroObservation."
                )

            # Ensure timestamps are timezone-aware.
            EmploymentIntelligence._validate_observation_timestamp(observation)

    @staticmethod
    def _latest_observation(
        observations: list[MacroObservation],
        indicator: MacroIndicator,
        decision_timestamp: datetime,
    ) -> MacroObservation | None:
        """Return the latest historical-safe observation."""

        # Filter to the requested indicator and observations known by decision time.
        eligible = [
            observation
            for observation in observations
            if observation.indicator == indicator
            and observation.timestamp <= decision_timestamp
        ]

        # Return None when there is no historical-safe observation.
        if not eligible:
            return None

        # Return the most recent eligible observation.
        return max(eligible, key=lambda observation: observation.timestamp)

    @staticmethod
    def _direction_from_change(
        change: float | None,
    ) -> MacroDirection:
        """Convert a numeric change into a raw direction."""

        # Missing change means direction cannot be determined from history.
        if change is None:
            return MacroDirection.UNKNOWN

        # Positive movement means rising.
        if change > 0:
            return MacroDirection.RISING

        # Negative movement means falling.
        if change < 0:
            return MacroDirection.FALLING

        # Zero movement means stable.
        return MacroDirection.STABLE

    @staticmethod
    def _employment_direction(
        indicator: MacroIndicator,
        raw_direction: MacroDirection,
    ) -> MacroDirection:
        """Interpret raw indicator movement as employment strength."""

        # NFP has direct semantics:
        # rising NFP = stronger employment.
        if indicator == MacroIndicator.NFP:
            return raw_direction

        # Unemployment has inverse semantics:
        # falling unemployment = stronger employment.
        if indicator == MacroIndicator.UNEMPLOYMENT_RATE:
            if raw_direction == MacroDirection.RISING:
                return MacroDirection.FALLING

            if raw_direction == MacroDirection.FALLING:
                return MacroDirection.RISING

            return raw_direction

        # Defensive fallback for unsupported indicators.
        return MacroDirection.UNKNOWN

    def _classify_level(
        self,
        indicator: MacroIndicator,
        change: float | None,
        employment_direction: MacroDirection,
    ) -> EmploymentLevel:
        """Classify the employment environment."""

        # If a historical change exists, use magnitude plus employment direction.
        if change is not None:
            # NFP movement is measured in thousands of jobs.
            if indicator == MacroIndicator.NFP:
                magnitude = abs(change)
                significant = self.significant_nfp_change
                strong = self.strong_nfp_change

            # Unemployment movement is measured in percentage points.
            else:
                magnitude = abs(change)
                significant = self.significant_unemployment_change
                strong = self.strong_unemployment_change

            # Normalize floating-point boundary noise.
            magnitude = round(magnitude, 10)

            # Strong employment improvement.
            if (
                employment_direction == MacroDirection.RISING
                and magnitude >= strong
            ):
                return EmploymentLevel.STRONG_HOT

            # Moderate employment improvement.
            if (
                employment_direction == MacroDirection.RISING
                and magnitude >= significant
            ):
                return EmploymentLevel.HOT

            # Strong employment deterioration.
            if (
                employment_direction == MacroDirection.FALLING
                and magnitude >= strong
            ):
                return EmploymentLevel.STRONG_COOLING

            # Moderate employment deterioration.
            if (
                employment_direction == MacroDirection.FALLING
                and magnitude >= significant
            ):
                return EmploymentLevel.COOLING

            # Small movement is treated as neutral.
            return EmploymentLevel.NEUTRAL

        # If no previous value exists, use the latest directional information.
        if employment_direction == MacroDirection.RISING:
            return EmploymentLevel.HOT

        # Falling employment strength is cooling.
        if employment_direction == MacroDirection.FALLING:
            return EmploymentLevel.COOLING

        # Stable data is neutral.
        if employment_direction == MacroDirection.STABLE:
            return EmploymentLevel.NEUTRAL

        # No reliable directional information.
        return EmploymentLevel.UNKNOWN

    def _classify_surprise(
        self,
        indicator: MacroIndicator,
        surprise: float | None,
    ) -> EmploymentSurpriseLevel:
        """Classify forecast surprise using indicator-specific semantics."""

        # No forecast means no surprise classification.
        if surprise is None:
            return EmploymentSurpriseLevel.UNKNOWN

        # Select the appropriate threshold based on the indicator.
        if indicator == MacroIndicator.NFP:
            significant = self.significant_nfp_surprise
            strong = self.strong_nfp_surprise

            # Positive NFP surprise means stronger-than-expected employment.
            employment_surprise = surprise

        else:
            significant = self.significant_unemployment_surprise
            strong = self.strong_unemployment_surprise

            # For unemployment, lower-than-expected is stronger employment.
            employment_surprise = -surprise

        # Normalize tiny floating-point residue.
        employment_surprise = round(employment_surprise, 10)

        # Strong positive employment surprise.
        if employment_surprise >= strong:
            return EmploymentSurpriseLevel.STRONG_UPSIDE

        # Moderate positive employment surprise.
        if employment_surprise >= significant:
            return EmploymentSurpriseLevel.UPSIDE

        # Strong negative employment surprise.
        if employment_surprise <= -strong:
            return EmploymentSurpriseLevel.STRONG_DOWNSIDE

        # Moderate negative employment surprise.
        if employment_surprise <= -significant:
            return EmploymentSurpriseLevel.DOWNSIDE

        # Near-forecast result.
        return EmploymentSurpriseLevel.IN_LINE

    @staticmethod
    def _confidence(
        raw_direction: MacroDirection,
        has_previous: bool,
    ) -> float:
        """Calculate deterministic confidence."""

        # No directional information means zero confidence.
        if raw_direction == MacroDirection.UNKNOWN:
            return 0.0

        # Historical movement gives the strongest confidence.
        if has_previous:
            return 100.0

        # Direction without previous data is weaker.
        return 50.0

    def analyze(
        self,
        observations: list[MacroObservation],
        *,
        indicator: MacroIndicator,
        decision_timestamp: datetime,
    ) -> EmploymentAssessment:
        """
        Analyze one employment indicator using historical-safe data.
        """

        # Validate the requested indicator.
        self._validate_indicator(indicator)

        # Validate the decision timestamp.
        self._validate_decision_timestamp(decision_timestamp)

        # Validate the observation collection.
        self._validate_observations(observations)

        # Find the latest observation known at decision time.
        latest = self._latest_observation(
            observations,
            indicator,
            decision_timestamp,
        )

        # Fail clearly when no historical-safe observation exists.
        if latest is None:
            raise EmploymentIntelligenceError(
                f"No historical-safe observation available for {indicator.value}."
            )

        # Extract the previous value.
        previous = latest.previous

        # Calculate change only when previous data exists.
        change = (
            latest.value - previous
            if previous is not None
            else None
        )

        # Determine raw direction from the underlying indicator.
        raw_direction = (
            latest.direction
            if change is None
            else self._direction_from_change(change)
        )

        # Interpret the direction according to employment semantics.
        employment_direction = self._employment_direction(
            indicator,
            raw_direction,
        )

        # Calculate percentage change when previous is non-zero.
        percentage_change: float | None

        if previous is None:
            percentage_change = None
        elif previous == 0:
            percentage_change = None
        else:
            percentage_change = (change / abs(previous)) * 100.0

        # MacroObservation.surprise already represents value - forecast.
        surprise = latest.surprise

        # Classify the employment environment.
        level = self._classify_level(
            indicator,
            change,
            employment_direction,
        )

        # Classify forecast surprise.
        surprise_level = self._classify_surprise(
            indicator,
            surprise,
        )

        # Calculate confidence from historical information.
        confidence = self._confidence(
            raw_direction,
            previous is not None,
        )

        # A directional employment assessment is sufficient.
        sufficient_data = raw_direction != MacroDirection.UNKNOWN

        # Build deterministic explanations.
        reasons: list[str] = []

        # Explain the selected indicator.
        reasons.append(
            f"Analyzed {indicator.value} using the latest observation available "
            f"at the decision timestamp."
        )

        # Explain historical movement when available.
        if change is not None:
            reasons.append(
                f"Raw change from previous observation: {change:.10g}."
            )
        else:
            reasons.append(
                "No previous observation was available."
            )

        # Explain unemployment inversion explicitly.
        if indicator == MacroIndicator.UNEMPLOYMENT_RATE:
            reasons.append(
                "Unemployment-rate movement is inverted for employment strength: "
                "falling unemployment indicates stronger employment."
            )

        # Explain surprise when available.
        if surprise is not None:
            reasons.append(
                f"Forecast surprise: {surprise:.10g}."
            )
        else:
            reasons.append(
                "No forecast surprise was available."
            )

        # Explain final classification.
        reasons.append(
            f"Employment level classified as {level.value}."
        )

        # Explain surprise classification.
        reasons.append(
            f"Employment surprise classified as {surprise_level.value}."
        )

        # Return the immutable assessment.
        return EmploymentAssessment(
            indicator=indicator,
            value=latest.value,
            previous=previous,
            forecast=latest.forecast,
            direction=raw_direction,
            employment_direction=employment_direction,
            change_from_previous=change,
            surprise=surprise,
            percentage_change=percentage_change,
            level=level,
            surprise_level=surprise_level,
            confidence=confidence,
            sufficient_data=sufficient_data,
            observation_timestamp=latest.timestamp,
            decision_timestamp=decision_timestamp,
            reasons=tuple(reasons),
        )

    def analyze_nfp(
        self,
        observations: list[MacroObservation],
        *,
        decision_timestamp: datetime,
    ) -> EmploymentAssessment:
        """Analyze NFP specifically."""

        # Delegate to the common deterministic analysis method.
        return self.analyze(
            observations,
            indicator=MacroIndicator.NFP,
            decision_timestamp=decision_timestamp,
        )

    def analyze_unemployment_rate(
        self,
        observations: list[MacroObservation],
        *,
        decision_timestamp: datetime,
    ) -> EmploymentAssessment:
        """Analyze the U.S. unemployment rate specifically."""

        # Delegate to the common deterministic analysis method.
        return self.analyze(
            observations,
            indicator=MacroIndicator.UNEMPLOYMENT_RATE,
            decision_timestamp=decision_timestamp,
        )

    def analyze_all(
        self,
        observations: list[MacroObservation],
        *,
        decision_timestamp: datetime,
    ) -> dict[MacroIndicator, EmploymentAssessment]:
        """Analyze both supported employment indicators."""

        # Return both employment assessments.
        return {
            MacroIndicator.NFP: self.analyze_nfp(
                observations,
                decision_timestamp=decision_timestamp,
            ),
            MacroIndicator.UNEMPLOYMENT_RATE: self.analyze_unemployment_rate(
                observations,
                decision_timestamp=decision_timestamp,
            ),
        }

    def analyze_xauusd(
        self,
        observations: list[MacroObservation],
        *,
        decision_timestamp: datetime,
    ) -> dict[MacroIndicator, EmploymentAssessment]:
        """
        Analyze employment conditions for XAUUSD context.

        This returns macro information only.
        It does not generate a gold trade signal.
        """

        # Employment data is relevant to XAUUSD through the USD/macro layer.
        return self.analyze_all(
            observations,
            decision_timestamp=decision_timestamp,
        )