# app/trading/macro/fed_rate_intelligence.py

"""Deterministic Federal Reserve interest-rate intelligence.

This module analyzes FED_FUNDS_RATE observations and produces an
auditable assessment.

Important:
- No external data fetching.
- No LLM dependency.
- No trade decision.
- No direct XAUUSD buy/sell decision.
- Future observations are excluded from historical analysis.
"""

from __future__ import annotations

# Import dataclass for immutable assessment objects.
from dataclasses import dataclass

# Import datetime for historical timestamp validation.
from datetime import datetime

# Import Enum for explicit Fed policy classifications.
from enum import Enum

# Import math for finite-number validation.
import math

# Import the existing macro observation models.
from app.trading.macro.macro_observation import (
    MacroDirection,
    MacroIndicator,
    MacroObservation,
)


class FedRateIntelligenceError(ValueError):
    """Raised when Fed-rate analysis receives invalid input."""


class FedRateLevel(str, Enum):
    """Classification of Federal Reserve policy-rate movement."""

    # The policy rate is strongly rising.
    STRONG_HAWKISH = "STRONG_HAWKISH"

    # The policy rate is rising.
    HAWKISH = "HAWKISH"

    # The policy rate is unchanged or effectively stable.
    NEUTRAL = "NEUTRAL"

    # The policy rate is falling.
    DOVISH = "DOVISH"

    # The policy rate is strongly falling.
    STRONG_DOVISH = "STRONG_DOVISH"

    # There is insufficient information.
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class FedRateAssessment:
    """Immutable and auditable Federal Reserve rate assessment."""

    # Indicator being analyzed.
    indicator: MacroIndicator

    # Latest usable policy-rate value.
    value: float | None

    # Previous policy-rate value.
    previous: float | None

    # Forecast policy-rate value, when available.
    forecast: float | None

    # Actual rate surprise relative to forecast.
    surprise: float | None

    # Absolute change from the previous rate.
    change_from_previous: float | None

    # Rate movement expressed in basis points.
    change_basis_points: float | None

    # Percentage change from the previous rate.
    percentage_change: float | None

    # Existing normalized direction.
    direction: MacroDirection

    # Final Fed policy-rate classification.
    level: FedRateLevel

    # Confidence from 0 to 100.
    confidence: float

    # Whether sufficient data exists.
    sufficient_data: bool

    # Timestamp of the selected observation.
    observation_timestamp: datetime | None

    # Historical decision timestamp.
    decision_timestamp: datetime

    # Human-readable audit explanations.
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the assessment."""

        # Only FED_FUNDS_RATE is valid for this model.
        if self.indicator is not MacroIndicator.FED_FUNDS_RATE:
            raise FedRateIntelligenceError(
                "indicator must be MacroIndicator.FED_FUNDS_RATE."
            )

        # Direction must use the existing enum.
        if not isinstance(self.direction, MacroDirection):
            raise FedRateIntelligenceError(
                "direction must be a MacroDirection."
            )

        # Level must use the FedRateLevel enum.
        if not isinstance(self.level, FedRateLevel):
            raise FedRateIntelligenceError(
                "level must be a FedRateLevel."
            )

        # Decision timestamp must be a datetime.
        if not isinstance(self.decision_timestamp, datetime):
            raise FedRateIntelligenceError(
                "decision_timestamp must be a datetime."
            )

        # Validate all optional numeric fields.
        for name, value in (
            ("value", self.value),
            ("previous", self.previous),
            ("forecast", self.forecast),
            ("surprise", self.surprise),
            ("change_from_previous", self.change_from_previous),
            ("change_basis_points", self.change_basis_points),
            ("percentage_change", self.percentage_change),
        ):
            # None represents unavailable information.
            if value is None:
                continue

            # bool must not be accepted as a numeric value.
            if isinstance(value, bool):
                raise FedRateIntelligenceError(
                    f"{name} must be numeric or None."
                )

            # Require an int or float.
            if not isinstance(value, (int, float)):
                raise FedRateIntelligenceError(
                    f"{name} must be numeric or None."
                )

            # Reject NaN and infinity.
            if not math.isfinite(float(value)):
                raise FedRateIntelligenceError(
                    f"{name} must be finite."
                )

        # Validate confidence.
        if isinstance(self.confidence, bool):
            raise FedRateIntelligenceError(
                "confidence must be numeric."
            )

        if not isinstance(self.confidence, (int, float)):
            raise FedRateIntelligenceError(
                "confidence must be numeric."
            )

        if not math.isfinite(float(self.confidence)):
            raise FedRateIntelligenceError(
                "confidence must be finite."
            )

        # Confidence is represented as a percentage.
        if not 0.0 <= float(self.confidence) <= 100.0:
            raise FedRateIntelligenceError(
                "confidence must be between 0 and 100."
            )

        # Observation timestamp is optional.
        if self.observation_timestamp is not None:
            if not isinstance(self.observation_timestamp, datetime):
                raise FedRateIntelligenceError(
                    "observation_timestamp must be a datetime or None."
                )


class FedRateIntelligence:
    """Deterministic Federal Reserve policy-rate intelligence."""

    # A 5-basis-point movement is meaningful.
    DEFAULT_SIGNIFICANT_CHANGE_BPS = 5.0

    # A 25-basis-point movement is considered a strong policy move.
    DEFAULT_STRONG_CHANGE_BPS = 25.0

    # The only supported macro indicator for this engine.
    INDICATOR = MacroIndicator.FED_FUNDS_RATE

    def __init__(
        self,
        significant_change_bps: float = DEFAULT_SIGNIFICANT_CHANGE_BPS,
        strong_change_bps: float = DEFAULT_STRONG_CHANGE_BPS,
    ) -> None:
        """Initialize Fed-rate intelligence."""

        # Validate significant threshold.
        self._validate_finite(
            significant_change_bps,
            "significant_change_bps",
        )

        # Validate strong threshold.
        self._validate_finite(
            strong_change_bps,
            "strong_change_bps",
        )

        # Both thresholds must be positive.
        if significant_change_bps <= 0:
            raise FedRateIntelligenceError(
                "significant_change_bps must be greater than zero."
            )

        if strong_change_bps <= 0:
            raise FedRateIntelligenceError(
                "strong_change_bps must be greater than zero."
            )

        # Strong threshold cannot be below significant threshold.
        if strong_change_bps < significant_change_bps:
            raise FedRateIntelligenceError(
                "strong_change_bps must be greater than or equal to "
                "significant_change_bps."
            )

        # Store thresholds as floats.
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
        """Validate numeric finite configuration values."""

        # Reject booleans.
        if isinstance(value, bool):
            raise FedRateIntelligenceError(
                f"{name} must be numeric, not bool."
            )

        # Require numeric values.
        if not isinstance(value, (int, float)):
            raise FedRateIntelligenceError(
                f"{name} must be numeric."
            )

        # Reject NaN and infinity.
        if not math.isfinite(float(value)):
            raise FedRateIntelligenceError(
                f"{name} must be finite."
            )

    @staticmethod
    def _validate_datetime(
        value: datetime,
        name: str,
    ) -> None:
        """Validate datetime input."""

        # Require a datetime.
        if not isinstance(value, datetime):
            raise FedRateIntelligenceError(
                f"{name} must be a datetime."
            )

    @staticmethod
    def _validate_timezone(
        observation: MacroObservation,
        decision_timestamp: datetime,
    ) -> None:
        """Validate timezone-awareness compatibility."""

        # Determine whether decision time is timezone-aware.
        decision_aware = decision_timestamp.tzinfo is not None

        # Determine whether observation time is timezone-aware.
        observation_aware = observation.timestamp.tzinfo is not None

        # Mixed timestamp semantics are unsafe.
        if decision_aware != observation_aware:
            raise FedRateIntelligenceError(
                "decision_timestamp and observation timestamp "
                "must use the same timezone-awareness."
            )

    @staticmethod
    def _basis_point_change(
        value: float | None,
        previous: float | None,
    ) -> float | None:
        """Convert a percentage-point rate change into basis points."""

        # Both values are needed.
        if value is None or previous is None:
            return None

        # One percentage point equals 100 basis points.
        result = (value - previous) * 100.0

        # Remove floating-point noise around zero.
        if abs(result) < 1e-12:
            return 0.0

        # Round boundary calculations to avoid binary floating-point
        # classification errors.
        return round(result, 10)

    @staticmethod
    def _percentage_change(
        value: float | None,
        previous: float | None,
    ) -> float | None:
        """Calculate percentage change from the previous rate."""

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

        # Return the percentage movement.
        return round(result, 10)

    def _classify(
        self,
        direction: MacroDirection,
        change_basis_points: float | None,
    ) -> FedRateLevel:
        """Classify Fed policy-rate movement."""

        # Unknown direction must remain unknown.
        if direction is MacroDirection.UNKNOWN:
            return FedRateLevel.UNKNOWN

        # Stable direction is neutral.
        if direction is MacroDirection.STABLE:
            return FedRateLevel.NEUTRAL

        # Without a previous rate, direction alone has lower confidence.
        if change_basis_points is None:

            # Rising policy rate implies hawkish direction.
            if direction is MacroDirection.RISING:
                return FedRateLevel.HAWKISH

            # Falling policy rate implies dovish direction.
            if direction is MacroDirection.FALLING:
                return FedRateLevel.DOVISH

            # Defensive fallback.
            return FedRateLevel.UNKNOWN

        # Normalize floating-point threshold comparisons.
        normalized_bps = round(
            change_basis_points,
            10,
        )

        # Rising Fed rate.
        if direction is MacroDirection.RISING:

            # A 25+ basis-point increase is strongly hawkish.
            if normalized_bps >= self._strong_change_bps:
                return FedRateLevel.STRONG_HAWKISH

            # A meaningful increase is hawkish.
            if normalized_bps >= self._significant_change_bps:
                return FedRateLevel.HAWKISH

            # Smaller movement is effectively neutral.
            return FedRateLevel.NEUTRAL

        # Falling Fed rate.
        if direction is MacroDirection.FALLING:

            # A 25+ basis-point decrease is strongly dovish.
            if normalized_bps <= -self._strong_change_bps:
                return FedRateLevel.STRONG_DOVISH

            # A meaningful decrease is dovish.
            if normalized_bps <= -self._significant_change_bps:
                return FedRateLevel.DOVISH

            # Smaller movement is effectively neutral.
            return FedRateLevel.NEUTRAL

        # Defensive fallback.
        return FedRateLevel.UNKNOWN

    def analyze(
        self,
        observations: list[MacroObservation]
        | tuple[MacroObservation, ...],
        decision_timestamp: datetime,
    ) -> FedRateAssessment:
        """Analyze the latest historical-safe Fed Funds Rate."""

        # Validate decision timestamp.
        self._validate_datetime(
            decision_timestamp,
            "decision_timestamp",
        )

        # Require a list or tuple.
        if not isinstance(observations, (list, tuple)):
            raise FedRateIntelligenceError(
                "observations must be a list or tuple."
            )

        # Validate every supplied observation.
        for observation in observations:

            # Every item must be a MacroObservation.
            if not isinstance(observation, MacroObservation):
                raise FedRateIntelligenceError(
                    "observations must contain MacroObservation values."
                )

        # Keep only Federal Funds Rate observations.
        rate_observations = [
            observation
            for observation in observations
            if observation.indicator is self.INDICATOR
        ]

        # No Fed rate observations means UNKNOWN.
        if not rate_observations:
            return FedRateAssessment(
                indicator=self.INDICATOR,
                value=None,
                previous=None,
                forecast=None,
                surprise=None,
                change_from_previous=None,
                change_basis_points=None,
                percentage_change=None,
                direction=MacroDirection.UNKNOWN,
                level=FedRateLevel.UNKNOWN,
                confidence=0.0,
                sufficient_data=False,
                observation_timestamp=None,
                decision_timestamp=decision_timestamp,
                reasons=(
                    "No FED_FUNDS_RATE observation was supplied.",
                ),
            )

        # Validate timestamp semantics.
        for observation in rate_observations:
            self._validate_timezone(
                observation,
                decision_timestamp,
            )

        # Keep only data available at the historical decision time.
        available = [
            observation
            for observation in rate_observations
            if observation.timestamp <= decision_timestamp
        ]

        # Future-only observations must never influence analysis.
        if not available:
            return FedRateAssessment(
                indicator=self.INDICATOR,
                value=None,
                previous=None,
                forecast=None,
                surprise=None,
                change_from_previous=None,
                change_basis_points=None,
                percentage_change=None,
                direction=MacroDirection.UNKNOWN,
                level=FedRateLevel.UNKNOWN,
                confidence=0.0,
                sufficient_data=False,
                observation_timestamp=None,
                decision_timestamp=decision_timestamp,
                reasons=(
                    "All FED_FUNDS_RATE observations occur after "
                    "the decision timestamp.",
                ),
            )

        # Select the latest historical-safe observation.
        latest = max(
            available,
            key=lambda observation: observation.timestamp,
        )

        # Calculate rate movement.
        change = latest.change_from_previous

        # Convert movement into basis points.
        change_basis_points = self._basis_point_change(
            latest.value,
            latest.previous,
        )

        # Calculate percentage movement.
        percentage_change = self._percentage_change(
            latest.value,
            latest.previous,
        )

        # Calculate forecast surprise using the existing model property.
        surprise = latest.surprise

        # Classify the policy-rate movement.
        level = self._classify(
            latest.direction,
            change_basis_points,
        )

        # Determine confidence.
        if latest.direction is MacroDirection.UNKNOWN:
            confidence = 0.0

        elif change_basis_points is None:
            confidence = 50.0

        else:
            confidence = 100.0

        # Build audit reasons.
        reasons: list[str] = []

        # Explain the policy-rate direction.
        if latest.direction is MacroDirection.RISING:
            reasons.append(
                "The Federal Funds Rate is rising, indicating a "
                "hawkish policy-rate direction."
            )

        elif latest.direction is MacroDirection.FALLING:
            reasons.append(
                "The Federal Funds Rate is falling, indicating a "
                "dovish policy-rate direction."
            )

        elif latest.direction is MacroDirection.STABLE:
            reasons.append(
                "The Federal Funds Rate is stable."
            )

        else:
            reasons.append(
                "Federal Reserve policy-rate direction is unknown."
            )

        # Explain the rate movement.
        if change_basis_points is not None:

            # Positive movement.
            if change_basis_points > 0:
                reasons.append(
                    f"Federal Funds Rate increased by "
                    f"{change_basis_points:.6f} basis points."
                )

            # Negative movement.
            elif change_basis_points < 0:
                reasons.append(
                    f"Federal Funds Rate decreased by "
                    f"{abs(change_basis_points):.6f} basis points."
                )

            # No movement.
            else:
                reasons.append(
                    "Federal Funds Rate did not change from the "
                    "previous observation."
                )

        # Explain forecast availability.
        if latest.forecast is not None:
            reasons.append(
                f"Forecast Federal Funds Rate was "
                f"{latest.forecast:.6f}."
            )

        # Explain the forecast surprise.
        if surprise is not None:

            # Positive surprise means actual rate exceeded forecast.
            if surprise > 0:
                reasons.append(
                    f"Actual rate exceeded forecast by "
                    f"{surprise:.6f} percentage points."
                )

            # Negative surprise means actual rate was below forecast.
            elif surprise < 0:
                reasons.append(
                    f"Actual rate was below forecast by "
                    f"{abs(surprise):.6f} percentage points."
                )

            # Zero surprise means actual matched forecast.
            else:
                reasons.append(
                    "Actual rate matched the forecast."
                )

        # Return the complete immutable assessment.
        return FedRateAssessment(
            indicator=self.INDICATOR,
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
            surprise=(
                float(surprise)
                if surprise is not None
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

    def analyze_xauusd(
        self,
        observations: list[MacroObservation]
        | tuple[MacroObservation, ...],
        decision_timestamp: datetime,
    ) -> FedRateAssessment:
        """Analyze Fed policy rate in an XAUUSD macro context."""

        # This method returns macro intelligence only.
        #
        # It intentionally does not convert Fed policy into an
        # XAUUSD buy/sell decision.
        return self.analyze(
            observations=observations,
            decision_timestamp=decision_timestamp,
        )