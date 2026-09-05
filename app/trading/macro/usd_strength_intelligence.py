# app/trading/macro/usd_strength_intelligence.py

"""Deterministic USD-strength intelligence.

This module converts normalized macro observations into an auditable
USD-strength assessment.

Important:
- This module does NOT fetch external data.
- This module does NOT generate trade signals.
- This module does NOT directly decide whether XAUUSD should be bought
  or sold.
- Only observations available at the decision timestamp are allowed.
"""

from __future__ import annotations

# Import dataclass utilities for immutable result objects.
from dataclasses import dataclass

# Import datetime for historical/no-lookahead filtering.
from datetime import datetime

# Import Enum for explicit USD-strength states.
from enum import Enum

# Import math for finite-number validation.
import math

# Import the existing macro observation model.
from app.trading.macro.macro_observation import (
    MacroDirection,
    MacroIndicator,
    MacroObservation,
)


class USDStrengthIntelligenceError(ValueError):
    """Raised when USD-strength analysis receives invalid input."""


class USDStrengthLevel(str, Enum):
    """Normalized USD-strength classification."""

    # USD has strong positive directional pressure.
    STRONG = "STRONG"

    # USD has strong negative directional pressure.
    WEAK = "WEAK"

    # USD does not have a sufficiently strong directional bias.
    NEUTRAL = "NEUTRAL"

    # There is not enough usable information.
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class USDStrengthContribution:
    """Explainable contribution from one macro indicator."""

    # Indicator that produced this contribution.
    indicator: MacroIndicator

    # Direction observed for the indicator.
    direction: MacroDirection

    # Configured weight assigned to the indicator.
    weight: float

    # Signed contribution after applying the direction.
    contribution: float

    # Observation timestamp used by the engine.
    timestamp: datetime

    # Human-readable explanation.
    reason: str

    def __post_init__(self) -> None:
        """Validate contribution fields."""

        # Require the correct indicator enum.
        if not isinstance(self.indicator, MacroIndicator):
            raise USDStrengthIntelligenceError(
                "indicator must be a MacroIndicator."
            )

        # Require the correct direction enum.
        if not isinstance(self.direction, MacroDirection):
            raise USDStrengthIntelligenceError(
                "direction must be a MacroDirection."
            )

        # Require a real datetime.
        if not isinstance(self.timestamp, datetime):
            raise USDStrengthIntelligenceError(
                "timestamp must be a datetime."
            )

        # Reject non-finite weights.
        if not math.isfinite(self.weight):
            raise USDStrengthIntelligenceError(
                "weight must be finite."
            )

        # Reject non-finite contributions.
        if not math.isfinite(self.contribution):
            raise USDStrengthIntelligenceError(
                "contribution must be finite."
            )

        # Require positive weights.
        if self.weight <= 0:
            raise USDStrengthIntelligenceError(
                "weight must be greater than zero."
            )

        # Require a non-empty explanation.
        if not self.reason.strip():
            raise USDStrengthIntelligenceError(
                "reason must not be empty."
            )


@dataclass(frozen=True, slots=True)
class USDStrengthAssessment:
    """Final auditable USD-strength assessment."""

    # Final normalized score in the range [-100, +100].
    score: float

    # Final USD-strength classification.
    level: USDStrengthLevel

    # Confidence based on available weighted indicators.
    confidence: float

    # Whether enough configured information was available.
    sufficient_data: bool

    # Number of indicators actually used.
    indicators_used: int

    # Total configured weight.
    total_weight: float

    # Weight represented by usable observations.
    used_weight: float

    # Individual explainable contributions.
    contributions: tuple[USDStrengthContribution, ...]

    # Human-readable reasons.
    reasons: tuple[str, ...]

    # Decision timestamp used for historical filtering.
    decision_timestamp: datetime

    def __post_init__(self) -> None:
        """Validate assessment output."""

        # Score must be finite.
        if not math.isfinite(self.score):
            raise USDStrengthIntelligenceError(
                "score must be finite."
            )

        # Confidence must be finite.
        if not math.isfinite(self.confidence):
            raise USDStrengthIntelligenceError(
                "confidence must be finite."
            )

        # Score must stay inside the documented range.
        if not -100.0 <= self.score <= 100.0:
            raise USDStrengthIntelligenceError(
                "score must be between -100 and 100."
            )

        # Confidence is represented as a percentage.
        if not 0.0 <= self.confidence <= 100.0:
            raise USDStrengthIntelligenceError(
                "confidence must be between 0 and 100."
            )

        # Require a valid classification.
        if not isinstance(self.level, USDStrengthLevel):
            raise USDStrengthIntelligenceError(
                "level must be a USDStrengthLevel."
            )

        # Require a valid decision timestamp.
        if not isinstance(self.decision_timestamp, datetime):
            raise USDStrengthIntelligenceError(
                "decision_timestamp must be a datetime."
            )

        # Indicator count cannot be negative.
        if self.indicators_used < 0:
            raise USDStrengthIntelligenceError(
                "indicators_used cannot be negative."
            )

        # Weights must be finite.
        if not math.isfinite(self.total_weight):
            raise USDStrengthIntelligenceError(
                "total_weight must be finite."
            )

        if not math.isfinite(self.used_weight):
            raise USDStrengthIntelligenceError(
                "used_weight must be finite."
            )

        # Weights cannot be negative.
        if self.total_weight < 0:
            raise USDStrengthIntelligenceError(
                "total_weight cannot be negative."
            )

        if self.used_weight < 0:
            raise USDStrengthIntelligenceError(
                "used_weight cannot be negative."
            )


class USDStrengthIntelligence:
    """Calculate deterministic USD strength from macro observations."""

    # Default direct USD-strength indicators.
    #
    # These indicators are intentionally limited to direct USD/rate
    # proxies. Inflation and employment interpretation will be handled
    # by their dedicated future P2.19 engines.
    DEFAULT_WEIGHTS: dict[MacroIndicator, float] = {
        MacroIndicator.DXY: 1.00,
        MacroIndicator.US_2Y_YIELD: 0.90,
        MacroIndicator.US_5Y_YIELD: 0.80,
        MacroIndicator.US_10Y_YIELD: 0.80,
        MacroIndicator.US_30Y_YIELD: 0.60,
        MacroIndicator.FED_FUNDS_RATE: 1.00,
    }

    # Score at or above this value means strong USD.
    DEFAULT_STRONG_THRESHOLD = 60.0

    # Score at or below this value means weak USD.
    DEFAULT_WEAK_THRESHOLD = -60.0

    # Minimum weighted coverage required for a non-unknown result.
    DEFAULT_MIN_COVERAGE = 0.50

    def __init__(
        self,
        weights: dict[MacroIndicator, float] | None = None,
        strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
        weak_threshold: float = DEFAULT_WEAK_THRESHOLD,
        min_coverage: float = DEFAULT_MIN_COVERAGE,
    ) -> None:
        """Initialize the USD-strength engine."""

        # Copy the supplied configuration so external mutation cannot
        # change the engine after construction.
        configured_weights = (
            dict(self.DEFAULT_WEIGHTS)
            if weights is None
            else dict(weights)
        )

        # Validate the indicator weights.
        self._validate_weights(configured_weights)

        # Validate the strong threshold.
        self._validate_finite(
            strong_threshold,
            "strong_threshold",
        )

        # Validate the weak threshold.
        self._validate_finite(
            weak_threshold,
            "weak_threshold",
        )

        # Strong threshold must be positive.
        if strong_threshold <= 0:
            raise USDStrengthIntelligenceError(
                "strong_threshold must be greater than zero."
            )

        # Weak threshold must be negative.
        if weak_threshold >= 0:
            raise USDStrengthIntelligenceError(
                "weak_threshold must be less than zero."
            )

        # Equal magnitudes are valid.
        #
        # Example:
        # +60 -> STRONG
        # -60 -> WEAK
        #
        # The previous implementation incorrectly rejected this valid
        # configuration because it used <= against abs(weak_threshold).
        #
        # No additional magnitude validation is required because the
        # score itself is bounded to [-100, +100].
        
        # Validate minimum coverage.
        self._validate_finite(
            min_coverage,
            "min_coverage",
        )

        # Coverage must be between zero and one.
        if not 0.0 <= min_coverage <= 1.0:
            raise USDStrengthIntelligenceError(
                "min_coverage must be between 0 and 1."
            )

        # Store a private copy of the configuration.
        self._weights = configured_weights

        # Store the positive classification threshold.
        self._strong_threshold = float(strong_threshold)

        # Store the negative classification threshold.
        self._weak_threshold = float(weak_threshold)

        # Store the minimum required coverage.
        self._min_coverage = float(min_coverage)

    @staticmethod
    def _validate_finite(
        value: float,
        name: str,
    ) -> None:
        """Validate that a value is numeric and finite."""

        # bool is technically an int in Python, so explicitly reject it.
        if isinstance(value, bool):
            raise USDStrengthIntelligenceError(
                f"{name} must be numeric, not bool."
            )

        # Only accept normal numeric values.
        if not isinstance(value, (int, float)):
            raise USDStrengthIntelligenceError(
                f"{name} must be numeric."
            )

        # Reject NaN and infinity.
        if not math.isfinite(float(value)):
            raise USDStrengthIntelligenceError(
                f"{name} must be finite."
            )

    @classmethod
    def _validate_weights(
        cls,
        weights: dict[MacroIndicator, float],
    ) -> None:
        """Validate indicator weights."""

        # Require a dictionary.
        if not isinstance(weights, dict):
            raise USDStrengthIntelligenceError(
                "weights must be a dictionary."
            )

        # At least one indicator must be configured.
        if not weights:
            raise USDStrengthIntelligenceError(
                "weights must not be empty."
            )

        # Validate every configured indicator.
        for indicator, weight in weights.items():

            # Indicator keys must use the MacroIndicator enum.
            if not isinstance(indicator, MacroIndicator):
                raise USDStrengthIntelligenceError(
                    "weight keys must be MacroIndicator values."
                )

            # Validate the numeric weight.
            cls._validate_finite(
                weight,
                f"weight[{indicator.value}]",
            )

            # Zero and negative weights are invalid.
            if weight <= 0:
                raise USDStrengthIntelligenceError(
                    f"weight[{indicator.value}] must be greater than zero."
                )

    @staticmethod
    def _validate_datetime(
        value: datetime,
        name: str,
    ) -> None:
        """Validate datetime input."""

        # Require an actual datetime.
        if not isinstance(value, datetime):
            raise USDStrengthIntelligenceError(
                f"{name} must be a datetime."
            )

    @staticmethod
    def _validate_timezone(
        observations: list[MacroObservation],
        decision_timestamp: datetime,
    ) -> None:
        """Ensure timestamps use compatible timezone semantics."""

        # Determine whether the decision timestamp is timezone-aware.
        decision_aware = decision_timestamp.tzinfo is not None

        # Validate every observation timestamp.
        for observation in observations:

            # Determine whether the observation is timezone-aware.
            observation_aware = observation.timestamp.tzinfo is not None

            # Mixing naive and aware timestamps is unsafe.
            if observation_aware != decision_aware:
                raise USDStrengthIntelligenceError(
                    "decision_timestamp and observation timestamps "
                    "must use the same timezone-awareness."
                )

    @staticmethod
    def _direction_sign(
        direction: MacroDirection,
    ) -> float:
        """Convert a direct macro direction into USD-strength sign."""

        # Rising DXY/rates are treated as positive USD pressure.
        if direction is MacroDirection.RISING:
            return 1.0

        # Falling DXY/rates are treated as negative USD pressure.
        if direction is MacroDirection.FALLING:
            return -1.0

        # Stable and unknown have no directional contribution.
        return 0.0

    @staticmethod
    def _reason(
        indicator: MacroIndicator,
        direction: MacroDirection,
        contribution: float,
    ) -> str:
        """Create a human-readable deterministic explanation."""

        # Explain positive USD contribution.
        if contribution > 0:
            return (
                f"{indicator.value} is rising, contributing to USD strength."
            )

        # Explain negative USD contribution.
        if contribution < 0:
            return (
                f"{indicator.value} is falling, contributing to USD weakness."
            )

        # Explain a neutral/unknown directional contribution.
        return (
            f"{indicator.value} is {direction.value.lower()}, "
            "providing no directional USD contribution."
        )

    def analyze(
        self,
        observations: list[MacroObservation]
        | tuple[MacroObservation, ...],
        decision_timestamp: datetime,
    ) -> USDStrengthAssessment:
        """Analyze USD strength using historical-safe observations."""

        # Validate the decision timestamp.
        self._validate_datetime(
            decision_timestamp,
            "decision_timestamp",
        )

        # Require a list or tuple for deterministic processing.
        if not isinstance(observations, (list, tuple)):
            raise USDStrengthIntelligenceError(
                "observations must be a list or tuple."
            )

        # Validate every observation.
        for observation in observations:
            if not isinstance(observation, MacroObservation):
                raise USDStrengthIntelligenceError(
                    "observations must contain MacroObservation values."
                )

        # Validate timezone-awareness before comparing timestamps.
        self._validate_timezone(
            list(observations),
            decision_timestamp,
        )

        # Calculate the total configured weight.
        total_weight = sum(self._weights.values())

        # Keep only indicators that are part of this USD model.
        relevant = [
            observation
            for observation in observations
            if observation.indicator in self._weights
        ]

        # CRITICAL HISTORICAL SAFETY:
        #
        # An observation can only influence a decision if its timestamp
        # is at or before the decision timestamp.
        available = [
            observation
            for observation in relevant
            if observation.timestamp <= decision_timestamp
        ]

        # Keep only the latest available observation for each indicator.
        latest_by_indicator: dict[
            MacroIndicator,
            MacroObservation,
        ] = {}

        # Process historical-safe observations.
        for observation in available:

            # Retrieve the existing observation for this indicator.
            current = latest_by_indicator.get(
                observation.indicator
            )

            # Replace it only when the new observation is newer.
            if (
                current is None
                or observation.timestamp > current.timestamp
            ):
                latest_by_indicator[observation.indicator] = observation

        # Create the explainable contribution list.
        contributions: list[USDStrengthContribution] = []

        # Evaluate each configured indicator.
        for indicator, weight in self._weights.items():

            # Retrieve the latest historical-safe observation.
            observation = latest_by_indicator.get(indicator)

            # No observation means this indicator is unavailable.
            if observation is None:
                continue

            # Convert direction to signed strength.
            sign = self._direction_sign(
                observation.direction
            )

            # Apply the configured weight.
            contribution = float(weight) * sign

            # Generate the explanation.
            reason = self._reason(
                indicator,
                observation.direction,
                contribution,
            )

            # Store the complete contribution.
            contributions.append(
                USDStrengthContribution(
                    indicator=indicator,
                    direction=observation.direction,
                    weight=float(weight),
                    contribution=contribution,
                    timestamp=observation.timestamp,
                    reason=reason,
                )
            )

        # Calculate the weight represented by available observations.
        used_weight = sum(
            contribution.weight
            for contribution in contributions
        )

        # Calculate weighted data coverage.
        coverage = (
            used_weight / total_weight
            if total_weight > 0
            else 0.0
        )

        # No usable indicators means there is no USD assessment.
        if used_weight <= 0:
            return USDStrengthAssessment(
                score=0.0,
                level=USDStrengthLevel.UNKNOWN,
                confidence=0.0,
                sufficient_data=False,
                indicators_used=0,
                total_weight=total_weight,
                used_weight=0.0,
                contributions=tuple(),
                reasons=(
                    "No configured USD-strength observations were "
                    "available at the decision timestamp.",
                ),
                decision_timestamp=decision_timestamp,
            )

        # Calculate the weighted directional score.
        raw_score = (
            sum(
                contribution.contribution
                for contribution in contributions
            )
            / used_weight
        )

        # Normalize to [-100, +100].
        score = raw_score * 100.0

        # Remove floating-point noise around zero.
        if abs(score) < 1e-12:
            score = 0.0

        # Convert coverage to percentage confidence.
        confidence = min(
            100.0,
            max(0.0, coverage * 100.0),
        )

        # Determine whether enough information is available.
        sufficient_data = coverage >= self._min_coverage

        # Insufficient coverage prevents directional classification.
        if not sufficient_data:
            level = USDStrengthLevel.UNKNOWN

            reasons = (
                "USD-strength data coverage is below the configured "
                "minimum threshold.",
            )

        # Strong USD classification.
        elif score >= self._strong_threshold:
            level = USDStrengthLevel.STRONG

            reasons = (
                "Available weighted USD indicators produce a strong "
                "positive USD-strength score.",
            )

        # Weak USD classification.
        elif score <= self._weak_threshold:
            level = USDStrengthLevel.WEAK

            reasons = (
                "Available weighted USD indicators produce a strong "
                "negative USD-strength score.",
            )

        # Everything between the boundaries is neutral.
        else:
            level = USDStrengthLevel.NEUTRAL

            reasons = (
                "Available weighted USD indicators do not produce a "
                "strong directional USD-strength score.",
            )

        # Return the complete auditable assessment.
        return USDStrengthAssessment(
            score=score,
            level=level,
            confidence=confidence,
            sufficient_data=sufficient_data,
            indicators_used=len(contributions),
            total_weight=total_weight,
            used_weight=used_weight,
            contributions=tuple(contributions),
            reasons=reasons,
            decision_timestamp=decision_timestamp,
        )

    def analyze_xauusd(
        self,
        observations: list[MacroObservation]
        | tuple[MacroObservation, ...],
        decision_timestamp: datetime,
    ) -> USDStrengthAssessment:
        """Analyze USD strength in an XAUUSD macro context."""

        # This method deliberately returns USD strength only.
        #
        # It must NOT convert USD strength into a gold buy/sell signal.
        # That responsibility belongs to the future P2.19.10
        # XAUUSD Macro Bias engine.
        return self.analyze(
            observations=observations,
            decision_timestamp=decision_timestamp,
        )