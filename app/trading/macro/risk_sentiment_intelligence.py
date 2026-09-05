"""
Risk-On / Risk-Off intelligence for the macro analysis layer.

This module provides deterministic assessment of broad market risk sentiment.

The engine:
- Uses existing MacroObservation objects.
- Does not fetch external data.
- Does not use an LLM.
- Does not generate trade signals.
- Does not execute trades.
- Is historical/no-lookahead safe.
- Produces an explainable risk-sentiment assessment.

Primary indicators:
- DXY
- U.S. Treasury yields
- Fed Funds Rate
- CPI / Core CPI
- PCE / Core PCE
- NFP
- Unemployment Rate
"""

from __future__ import annotations

# Import dataclass support for immutable result objects.
from dataclasses import dataclass

# Import Enum for strongly typed classifications.
from enum import Enum

# Import datetime for historical decision-time validation.
from datetime import datetime

# Import math for finite-number validation.
import math

# Import the existing macro observation models.
from app.trading.macro.macro_observation import (
    MacroDirection,
    MacroIndicator,
    MacroObservation,
)


class RiskSentimentIntelligenceError(ValueError):
    """Raised when risk-sentiment intelligence receives invalid input."""


class RiskSentiment(str, Enum):
    """Overall broad-market risk sentiment."""

    STRONG_RISK_ON = "strong_risk_on"
    RISK_ON = "risk_on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"
    STRONG_RISK_OFF = "strong_risk_off"
    UNKNOWN = "unknown"


class RiskSentimentComponent(str, Enum):
    """Direction contributed by an individual macro component."""

    RISK_ON = "risk_on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RiskSentimentContribution:
    """Immutable contribution from one macro indicator."""

    # Indicator responsible for the contribution.
    indicator: MacroIndicator

    # Raw indicator direction.
    direction: MacroDirection

    # Interpreted risk-sentiment direction.
    component: RiskSentimentComponent

    # Configured weight of this indicator.
    weight: float

    # Weighted signed contribution.
    contribution: float

    # Timestamp of the observation used.
    observation_timestamp: datetime

    # Human-readable explanation.
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Serialize the contribution into a JSON-friendly dictionary."""

        # Return the complete contribution explicitly.
        return {
            "indicator": self.indicator.value,
            "direction": self.direction.value,
            "component": self.component.value,
            "weight": self.weight,
            "contribution": self.contribution,
            "observation_timestamp": self.observation_timestamp.isoformat(),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RiskSentimentAssessment:
    """Immutable overall risk-sentiment assessment."""

    # Final normalized sentiment score from -100 to +100.
    score: float

    # Final qualitative sentiment.
    sentiment: RiskSentiment

    # Confidence based on available weighted coverage.
    confidence: float

    # Whether enough weighted information was available.
    sufficient_data: bool

    # Number of indicators successfully used.
    indicators_used: int

    # Total configured weight.
    total_weight: float

    # Weight actually represented by available observations.
    used_weight: float

    # Individual indicator contributions.
    contributions: tuple[RiskSentimentContribution, ...]

    # Decision timestamp used for historical safety.
    decision_timestamp: datetime

    # Deterministic explanations.
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize the assessment into a JSON-friendly dictionary."""

        # Return a stable JSON-friendly representation.
        return {
            "score": self.score,
            "sentiment": self.sentiment.value,
            "confidence": self.confidence,
            "sufficient_data": self.sufficient_data,
            "indicators_used": self.indicators_used,
            "total_weight": self.total_weight,
            "used_weight": self.used_weight,
            "contributions": [
                contribution.to_dict()
                for contribution in self.contributions
            ],
            "decision_timestamp": self.decision_timestamp.isoformat(),
            "reasons": list(self.reasons),
        }


class RiskSentimentIntelligence:
    """
    Deterministic broad-market risk sentiment engine.

    Positive score:
        Risk-On

    Negative score:
        Risk-Off

    Near-zero score:
        Neutral

    Important:
        This engine provides context only.
        It does not approve or reject trades.
    """

    # Default indicator weights.
    DEFAULT_WEIGHTS = {
        MacroIndicator.DXY: 1.00,
        MacroIndicator.US_2Y_YIELD: 0.70,
        MacroIndicator.US_5Y_YIELD: 0.70,
        MacroIndicator.US_10Y_YIELD: 1.00,
        MacroIndicator.US_30Y_YIELD: 0.60,
        MacroIndicator.FED_FUNDS_RATE: 0.80,
        MacroIndicator.CPI: 0.60,
        MacroIndicator.CORE_CPI: 0.60,
        MacroIndicator.PCE: 0.70,
        MacroIndicator.CORE_PCE: 0.70,
        MacroIndicator.NFP: 0.80,
        MacroIndicator.UNEMPLOYMENT_RATE: 0.80,
    }

    # Minimum weighted coverage required for a reliable assessment.
    DEFAULT_MIN_COVERAGE = 0.50

    # Strong risk-on threshold.
    DEFAULT_STRONG_RISK_ON_THRESHOLD = 50.0

    # Risk-on threshold.
    DEFAULT_RISK_ON_THRESHOLD = 15.0

    # Risk-off threshold.
    DEFAULT_RISK_OFF_THRESHOLD = -15.0

    # Strong risk-off threshold.
    DEFAULT_STRONG_RISK_OFF_THRESHOLD = -50.0

    def __init__(
        self,
        *,
        weights: dict[MacroIndicator, float] | None = None,
        min_coverage: float = DEFAULT_MIN_COVERAGE,
        strong_risk_on_threshold: float = DEFAULT_STRONG_RISK_ON_THRESHOLD,
        risk_on_threshold: float = DEFAULT_RISK_ON_THRESHOLD,
        risk_off_threshold: float = DEFAULT_RISK_OFF_THRESHOLD,
        strong_risk_off_threshold: float = DEFAULT_STRONG_RISK_OFF_THRESHOLD,
    ) -> None:
        """Initialize configurable risk-sentiment parameters."""

        # Use a copy of the default weights when no custom weights are supplied.
        selected_weights = (
            dict(self.DEFAULT_WEIGHTS)
            if weights is None
            else dict(weights)
        )

        # Validate the supplied configuration.
        self._validate_configuration(
            selected_weights,
            min_coverage,
            strong_risk_on_threshold,
            risk_on_threshold,
            risk_off_threshold,
            strong_risk_off_threshold,
        )

        # Store the validated weights.
        self.weights = selected_weights

        # Store the minimum coverage requirement.
        self.min_coverage = float(min_coverage)

        # Store positive and negative classification thresholds.
        self.strong_risk_on_threshold = float(strong_risk_on_threshold)
        self.risk_on_threshold = float(risk_on_threshold)
        self.risk_off_threshold = float(risk_off_threshold)
        self.strong_risk_off_threshold = float(strong_risk_off_threshold)

    @staticmethod
    def _validate_configuration(
        weights: dict[MacroIndicator, float],
        min_coverage: float,
        strong_risk_on_threshold: float,
        risk_on_threshold: float,
        risk_off_threshold: float,
        strong_risk_off_threshold: float,
    ) -> None:
        """Validate engine configuration."""

        # Require a dictionary for custom weights.
        if not isinstance(weights, dict):
            raise RiskSentimentIntelligenceError(
                "weights must be a dictionary."
            )

        # Validate every weight.
        for indicator, weight in weights.items():

            # Only MacroIndicator keys are accepted.
            if not isinstance(indicator, MacroIndicator):
                raise RiskSentimentIntelligenceError(
                    "Every risk-sentiment weight key must be a MacroIndicator."
                )

            # Reject boolean weights.
            if isinstance(weight, bool):
                raise RiskSentimentIntelligenceError(
                    "Risk-sentiment weights must be numeric, not boolean."
                )

            # Require numeric weights.
            if not isinstance(weight, (int, float)):
                raise RiskSentimentIntelligenceError(
                    "Risk-sentiment weights must be numeric."
                )

            # Convert to float for validation.
            numeric_weight = float(weight)

            # Reject NaN and infinity.
            if not math.isfinite(numeric_weight):
                raise RiskSentimentIntelligenceError(
                    "Risk-sentiment weights must be finite."
                )

            # Weights must be non-negative.
            if numeric_weight < 0:
                raise RiskSentimentIntelligenceError(
                    "Risk-sentiment weights cannot be negative."
                )

        # Validate coverage.
        if isinstance(min_coverage, bool):
            raise RiskSentimentIntelligenceError(
                "min_coverage must be numeric, not boolean."
            )

        # Convert coverage to float.
        coverage = float(min_coverage)

        # Coverage must be finite and within [0, 1].
        if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            raise RiskSentimentIntelligenceError(
                "min_coverage must be between 0 and 1."
            )

        # Validate every sentiment threshold.
        thresholds = (
            strong_risk_on_threshold,
            risk_on_threshold,
            risk_off_threshold,
            strong_risk_off_threshold,
        )

        # Every threshold must be finite.
        for threshold in thresholds:
            if isinstance(threshold, bool):
                raise RiskSentimentIntelligenceError(
                    "Risk-sentiment thresholds must be numeric, not boolean."
                )

            if not isinstance(threshold, (int, float)):
                raise RiskSentimentIntelligenceError(
                    "Risk-sentiment thresholds must be numeric."
                )

            if not math.isfinite(float(threshold)):
                raise RiskSentimentIntelligenceError(
                    "Risk-sentiment thresholds must be finite."
                )

        # Positive thresholds must be ordered correctly.
        if strong_risk_on_threshold < risk_on_threshold:
            raise RiskSentimentIntelligenceError(
                "strong_risk_on_threshold must be >= risk_on_threshold."
            )

        # Negative thresholds must also be ordered correctly.
        if strong_risk_off_threshold > risk_off_threshold:
            raise RiskSentimentIntelligenceError(
                "strong_risk_off_threshold must be <= risk_off_threshold."
            )

        # The neutral region cannot overlap itself.
        if risk_off_threshold >= risk_on_threshold:
            raise RiskSentimentIntelligenceError(
                "risk_off_threshold must be lower than risk_on_threshold."
            )

    @staticmethod
    def _validate_decision_timestamp(
        decision_timestamp: datetime,
    ) -> None:
        """Validate the decision timestamp."""

        # Require a datetime.
        if not isinstance(decision_timestamp, datetime):
            raise RiskSentimentIntelligenceError(
                "decision_timestamp must be a datetime."
            )

        # Naive timestamps are unsafe for historical analysis.
        if decision_timestamp.tzinfo is None:
            raise RiskSentimentIntelligenceError(
                "decision_timestamp must be timezone-aware."
            )

    @staticmethod
    def _validate_observations(
        observations: list[MacroObservation],
    ) -> None:
        """Validate the observation collection."""

        # Require a list.
        if not isinstance(observations, list):
            raise RiskSentimentIntelligenceError(
                "observations must be a list of MacroObservation objects."
            )

        # Validate each observation.
        for observation in observations:

            # Require the existing production observation type.
            if not isinstance(observation, MacroObservation):
                raise RiskSentimentIntelligenceError(
                    "Every observation must be a MacroObservation."
                )

            # Require timezone-aware observation timestamps.
            if observation.timestamp.tzinfo is None:
                raise RiskSentimentIntelligenceError(
                    "Macro observation timestamps must be timezone-aware."
                )

    @staticmethod
    def _latest_observation(
        observations: list[MacroObservation],
        indicator: MacroIndicator,
        decision_timestamp: datetime,
    ) -> MacroObservation | None:
        """Select the latest observation available at decision time."""

        # Filter to the requested indicator and historical-safe observations.
        eligible = [
            observation
            for observation in observations
            if observation.indicator == indicator
            and observation.timestamp <= decision_timestamp
        ]

        # Return None when no valid observation exists.
        if not eligible:
            return None

        # Return the newest historical-safe observation.
        return max(
            eligible,
            key=lambda observation: observation.timestamp,
        )

    @staticmethod
    def _raw_direction(
        observation: MacroObservation,
    ) -> MacroDirection:
        """Determine raw direction from the observation."""

        # Use previous-value movement whenever previous data exists.
        if observation.previous is not None:

            # Calculate the change.
            change = observation.value - observation.previous

            # Positive change is rising.
            if change > 0:
                return MacroDirection.RISING

            # Negative change is falling.
            if change < 0:
                return MacroDirection.FALLING

            # Zero change is stable.
            return MacroDirection.STABLE

        # Otherwise use the observation's explicit direction.
        return observation.direction

    @staticmethod
    def _component_for_indicator(
        indicator: MacroIndicator,
        direction: MacroDirection,
    ) -> RiskSentimentComponent:
        """
        Interpret an indicator direction as broad risk sentiment.

        The mappings are deliberately conservative and deterministic.
        """

        # Unknown direction cannot produce a reliable component.
        if direction == MacroDirection.UNKNOWN:
            return RiskSentimentComponent.UNKNOWN

        # Stable indicators contribute neutrally.
        if direction == MacroDirection.STABLE:
            return RiskSentimentComponent.NEUTRAL

        # DXY rising generally represents stronger USD / defensive conditions.
        if indicator == MacroIndicator.DXY:
            if direction == MacroDirection.RISING:
                return RiskSentimentComponent.RISK_OFF

            return RiskSentimentComponent.RISK_ON

        # Rising Treasury yields are treated as risk-off in this broad context.
        if indicator in {
            MacroIndicator.US_2Y_YIELD,
            MacroIndicator.US_5Y_YIELD,
            MacroIndicator.US_10Y_YIELD,
            MacroIndicator.US_30Y_YIELD,
        }:
            if direction == MacroDirection.RISING:
                return RiskSentimentComponent.RISK_OFF

            return RiskSentimentComponent.RISK_ON

        # Rising Fed Funds Rate is interpreted as tighter conditions.
        if indicator == MacroIndicator.FED_FUNDS_RATE:
            if direction == MacroDirection.RISING:
                return RiskSentimentComponent.RISK_OFF

            return RiskSentimentComponent.RISK_ON

        # Rising inflation is treated as a macro-risk signal because
        # it can increase pressure for tighter monetary policy.
        if indicator in {
            MacroIndicator.CPI,
            MacroIndicator.CORE_CPI,
            MacroIndicator.PCE,
            MacroIndicator.CORE_PCE,
        }:
            if direction == MacroDirection.RISING:
                return RiskSentimentComponent.RISK_OFF

            return RiskSentimentComponent.RISK_ON

        # Rising NFP indicates stronger employment and is treated as
        # modestly risk-on in isolation.
        if indicator == MacroIndicator.NFP:
            if direction == MacroDirection.RISING:
                return RiskSentimentComponent.RISK_ON

            return RiskSentimentComponent.RISK_OFF

        # Unemployment requires inverse interpretation:
        # falling unemployment = stronger employment.
        if indicator == MacroIndicator.UNEMPLOYMENT_RATE:
            if direction == MacroDirection.FALLING:
                return RiskSentimentComponent.RISK_ON

            return RiskSentimentComponent.RISK_OFF

        # Defensive fallback.
        return RiskSentimentComponent.UNKNOWN

    @staticmethod
    def _signed_value(
        component: RiskSentimentComponent,
    ) -> float:
        """Convert a qualitative component into a signed value."""

        # Risk-on contributes +1.
        if component == RiskSentimentComponent.RISK_ON:
            return 1.0

        # Risk-off contributes -1.
        if component == RiskSentimentComponent.RISK_OFF:
            return -1.0

        # Neutral and unknown contribute zero.
        return 0.0

    def _classify(
        self,
        score: float,
    ) -> RiskSentiment:
        """Classify a normalized risk sentiment score."""

        # Normalize floating-point residue around thresholds.
        normalized_score = round(score, 10)

        # Strong positive environment.
        if normalized_score >= self.strong_risk_on_threshold:
            return RiskSentiment.STRONG_RISK_ON

        # Moderate positive environment.
        if normalized_score >= self.risk_on_threshold:
            return RiskSentiment.RISK_ON

        # Strong negative environment.
        if normalized_score <= self.strong_risk_off_threshold:
            return RiskSentiment.STRONG_RISK_OFF

        # Moderate negative environment.
        if normalized_score <= self.risk_off_threshold:
            return RiskSentiment.RISK_OFF

        # Everything between the boundaries is neutral.
        return RiskSentiment.NEUTRAL

    def analyze(
        self,
        observations: list[MacroObservation],
        *,
        decision_timestamp: datetime,
    ) -> RiskSentimentAssessment:
        """
        Analyze broad market risk sentiment.

        Only the latest historical-safe observation for each configured
        indicator is used.
        """

        # Validate the decision timestamp.
        self._validate_decision_timestamp(decision_timestamp)

        # Validate the observation collection.
        self._validate_observations(observations)

        # Calculate total configured weight.
        total_weight = sum(
            float(weight)
            for weight in self.weights.values()
        )

        # A zero-weight configuration cannot produce a meaningful result.
        if total_weight <= 0:
            raise RiskSentimentIntelligenceError(
                "At least one configured indicator must have positive weight."
            )

        # Prepare contribution storage.
        contributions: list[RiskSentimentContribution] = []

        # Track total available weight.
        used_weight = 0.0

        # Process each configured indicator.
        for indicator, weight in self.weights.items():

            # Ignore zero-weight indicators.
            if weight == 0:
                continue

            # Find the latest historical-safe observation.
            observation = self._latest_observation(
                observations,
                indicator,
                decision_timestamp,
            )

            # Skip indicators with no usable historical data.
            if observation is None:
                continue

            # Determine raw direction.
            direction = self._raw_direction(observation)

            # Interpret the indicator for risk sentiment.
            component = self._component_for_indicator(
                indicator,
                direction,
            )

            # Unknown observations should not count toward coverage.
            if component == RiskSentimentComponent.UNKNOWN:
                continue

            # Mark this indicator's weight as usable.
            used_weight += float(weight)

            # Convert component to signed direction.
            signed_direction = self._signed_value(component)

            # Calculate weighted contribution.
            contribution_value = float(weight) * signed_direction

            # Build a deterministic explanation.
            if component == RiskSentimentComponent.RISK_ON:
                reason = (
                    f"{indicator.value} contributes risk-on because its "
                    f"current interpreted direction is {direction.value}."
                )
            elif component == RiskSentimentComponent.RISK_OFF:
                reason = (
                    f"{indicator.value} contributes risk-off because its "
                    f"current interpreted direction is {direction.value}."
                )
            else:
                reason = (
                    f"{indicator.value} is stable and contributes neutrally."
                )

            # Store the contribution.
            contributions.append(
                RiskSentimentContribution(
                    indicator=indicator,
                    direction=direction,
                    component=component,
                    weight=float(weight),
                    contribution=contribution_value,
                    observation_timestamp=observation.timestamp,
                    reason=reason,
                )
            )

        # Calculate normalized score from -100 to +100.
        raw_score = (
            sum(
                contribution.contribution
                for contribution in contributions
            )
            / total_weight
        ) * 100.0

        # Normalize floating-point noise.
        score = round(raw_score, 10)

        # Calculate weighted coverage.
        coverage = used_weight / total_weight

        # Confidence is the available weighted coverage.
        confidence = round(
            max(0.0, min(100.0, coverage * 100.0)),
            10,
        )

        # Determine whether enough weighted information exists.
        sufficient_data = coverage >= self.min_coverage

        # If coverage is insufficient, do not claim a directional regime.
        if sufficient_data:
            sentiment = self._classify(score)
        else:
            sentiment = RiskSentiment.UNKNOWN

        # Build deterministic explanations.
        reasons: list[str] = []

        # Explain data coverage.
        reasons.append(
            f"Used {len(contributions)} of {len(self.weights)} configured "
            f"indicators."
        )

        # Explain weighted coverage.
        reasons.append(
            f"Weighted data coverage is {coverage:.4f}."
        )

        # Explain insufficient data explicitly.
        if not sufficient_data:
            reasons.append(
                "Weighted coverage is below the configured minimum, so "
                "overall sentiment is UNKNOWN."
            )

        # Explain sufficient data.
        else:
            reasons.append(
                f"Overall risk sentiment score is {score:.4f}."
            )

        # Explain the final classification.
        reasons.append(
            f"Risk sentiment classified as {sentiment.value}."
        )

        # Return the immutable assessment.
        return RiskSentimentAssessment(
            score=score,
            sentiment=sentiment,
            confidence=confidence,
            sufficient_data=sufficient_data,
            indicators_used=len(contributions),
            total_weight=total_weight,
            used_weight=used_weight,
            contributions=tuple(contributions),
            decision_timestamp=decision_timestamp,
            reasons=tuple(reasons),
        )

    def analyze_xauusd(
        self,
        observations: list[MacroObservation],
        *,
        decision_timestamp: datetime,
    ) -> RiskSentimentAssessment:
        """
        Analyze risk sentiment for XAUUSD macro context.

        This method returns market context only.
        It does not generate a gold trade signal.
        """

        # Reuse the common deterministic risk-sentiment analysis.
        return self.analyze(
            observations,
            decision_timestamp=decision_timestamp,
        )