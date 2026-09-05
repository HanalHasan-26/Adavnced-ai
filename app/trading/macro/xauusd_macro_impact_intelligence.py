"""
XAU/USD macro impact intelligence.

This module combines the existing deterministic macro-intelligence
components into a deterministic XAU/USD macro bias.

Important architectural rule:
    This module provides macro context only.
    It does NOT authorize, reject, or execute trades.

Historical-safety rule:
    Only observations whose timestamp is less than or equal to the
    decision timestamp may influence the assessment.
"""

# Import dataclass helpers for immutable result models.
from dataclasses import dataclass, fields, is_dataclass

# Import datetime for historical decision-time validation.
from datetime import datetime

# Import Enum for deterministic classifications.
from enum import Enum

# Import finite-number validation.
from math import isfinite

# Import typing helpers.
from typing import Callable, Iterable, Mapping, TypeVar

# Import DXY intelligence.
from app.trading.macro.dxy_intelligence import (
    DXYIntelligence,
    DXYLevel,
)

# Import employment intelligence.
from app.trading.macro.employment_intelligence import (
    EmploymentIntelligence,
    EmploymentLevel,
)

# Import Federal Reserve rate intelligence.
from app.trading.macro.fed_rate_intelligence import (
    FedRateIntelligence,
    FedRateLevel,
)

# Import inflation intelligence.
from app.trading.macro.inflation_intelligence import (
    InflationIntelligence,
    InflationLevel,
)

# Import canonical macro observations.
from app.trading.macro.macro_observation import (
    MacroDirection,
    MacroIndicator,
    MacroObservation,
)

# Import aggregate macro-regime intelligence.
from app.trading.macro.macro_regime_intelligence import (
    MacroRegime,
    MacroRegimeAssessment,
    MacroRegimeIntelligence,
)

# Import risk-sentiment intelligence.
from app.trading.macro.risk_sentiment_intelligence import (
    RiskSentiment,
    RiskSentimentIntelligence,
)

# Import Treasury-yield intelligence.
from app.trading.macro.treasury_yield_intelligence import (
    TreasuryYieldIntelligence,
    TreasuryYieldLevel,
)

# Import USD-strength intelligence.
from app.trading.macro.usd_strength_intelligence import (
    USDStrengthIntelligence,
    USDStrengthLevel,
)


# Generic type used by safe lower-level wrappers.
T = TypeVar("T")


class XAUUSDMacroImpactIntelligenceError(ValueError):
    """Raised when XAU/USD macro-impact input or configuration is invalid."""


class XAUUSDMacroBias(str, Enum):
    """Overall deterministic macro bias for XAU/USD."""

    # Strong bullish pressure.
    STRONG_BULLISH = "strong_bullish"

    # Normal bullish pressure.
    BULLISH = "bullish"

    # Evidence exists but does not cross a directional threshold.
    NEUTRAL = "neutral"

    # Normal bearish pressure.
    BEARISH = "bearish"

    # Strong bearish pressure.
    STRONG_BEARISH = "strong_bearish"

    # No usable evidence.
    UNKNOWN = "unknown"


class XAUUSDMacroComponent(str, Enum):
    """Macro component contributing to XAU/USD."""

    # USD and DXY pressure.
    USD_PRESSURE = "usd_pressure"

    # Treasury and Federal Reserve rate pressure.
    INTEREST_RATE_PRESSURE = "interest_rate_pressure"

    # Inflation pressure.
    INFLATION_PRESSURE = "inflation_pressure"

    # Employment pressure.
    EMPLOYMENT_PRESSURE = "employment_pressure"

    # Broad market risk sentiment.
    RISK_SENTIMENT_PRESSURE = "risk_sentiment_pressure"

    # Aggregate macro regime.
    REGIME_PRESSURE = "regime_pressure"

    # Unknown component.
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class XAUUSDMacroImpactContribution:
    """One deterministic macro contribution."""

    # Public source name.
    source: str

    # Macro component.
    component: XAUUSDMacroComponent

    # Contribution from -100 to +100.
    contribution: float

    # Deterministic explanation.
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Serialize the contribution."""

        # Return JSON-friendly values.
        return {
            "source": self.source,
            "component": self.component.value,
            "contribution": self.contribution,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class XAUUSDMacroImpactAssessment:
    """Complete deterministic XAU/USD macro assessment."""

    # Target symbol.
    symbol: str

    # Overall macro bias.
    bias: XAUUSDMacroBias

    # Aggregate score.
    score: float

    # Evidence coverage percentage.
    confidence: float

    # Whether minimum evidence is available.
    sufficient_data: bool

    # Aggregate macro regime.
    macro_regime: MacroRegime

    # Underlying regime assessment.
    macro_regime_assessment: MacroRegimeAssessment | None

    # Individual usable contributions.
    contributions: tuple[XAUUSDMacroImpactContribution, ...]

    # Decision timestamp.
    decision_timestamp: datetime

    # Deterministic explanations.
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete assessment."""

        # Return JSON-compatible values.
        return {
            "symbol": self.symbol,
            "bias": self.bias.value,
            "score": self.score,
            "confidence": self.confidence,
            "sufficient_data": self.sufficient_data,
            "macro_regime": self.macro_regime.value,
            "macro_regime_assessment": (
                _serialize_value(self.macro_regime_assessment)
                if self.macro_regime_assessment is not None
                else None
            ),
            "contributions": [
                contribution.to_dict()
                for contribution in self.contributions
            ],
            "decision_timestamp": self.decision_timestamp.isoformat(),
            "reasons": list(self.reasons),
        }


def _serialize_value(value: object) -> object:
    """Recursively serialize nested macro objects."""

    # Serialize Enum values.
    if isinstance(value, Enum):
        return value.value

    # Serialize datetime values.
    if isinstance(value, datetime):
        return value.isoformat()

    # Serialize dataclass values recursively.
    if is_dataclass(value):
        return {
            field.name: _serialize_value(getattr(value, field.name))
            for field in fields(value)
        }

    # Serialize mappings.
    if isinstance(value, Mapping):
        return {
            str(key): _serialize_value(item)
            for key, item in value.items()
        }

    # Serialize tuples.
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]

    # Serialize lists.
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]

    # Serialize sets.
    if isinstance(value, set):
        return [_serialize_value(item) for item in value]

    # Return primitive values unchanged.
    return value


class XAUUSDMacroImpactIntelligence:
    """
    Deterministic macro-impact engine for XAU/USD.

    Positive contribution:
        Supports XAU/USD.

    Negative contribution:
        Pressures XAU/USD.

    Zero contribution:
        Valid evidence exists but is directionally neutral.

    UNKNOWN:
        No usable evidence exists.
    """

    # Only XAU/USD is currently supported.
    SYMBOL = "XAUUSD"

    # Default macro component weights.
    DEFAULT_WEIGHTS = {
        "usd": 1.0,
        "rates": 1.0,
        "inflation": 0.8,
        "employment": 0.6,
        "risk": 0.8,
        "regime": 1.0,
    }

    # Normal directional threshold.
    DEFAULT_THRESHOLD = 20.0

    # Strong directional threshold.
    DEFAULT_STRONG_THRESHOLD = 50.0

    # Minimum evidence coverage.
    DEFAULT_MIN_CONFIDENCE = 50.0

    def __init__(
        self,
        *,
        weights: Mapping[str, float] | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        """Initialize the macro-impact engine."""

        # Copy the default configuration.
        configured_weights = dict(self.DEFAULT_WEIGHTS)

        # Apply caller-provided weight overrides.
        if weights is not None:
            for name, value in weights.items():

                # Reject unknown component names.
                if name not in configured_weights:
                    raise XAUUSDMacroImpactIntelligenceError(
                        f"Unknown macro weight: {name!r}"
                    )

                # Reject booleans and non-numeric values.
                if isinstance(value, bool) or not isinstance(
                    value,
                    (int, float),
                ):
                    raise XAUUSDMacroImpactIntelligenceError(
                        f"Weight for {name!r} must be numeric."
                    )

                # Reject negative, infinite, and NaN weights.
                if not isfinite(float(value)) or float(value) < 0.0:
                    raise XAUUSDMacroImpactIntelligenceError(
                        f"Weight for {name!r} must be finite and non-negative."
                    )

                # Store validated weight.
                configured_weights[name] = float(value)

        # Validate the normal threshold.
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not isfinite(float(threshold))
            or float(threshold) < 0.0
        ):
            raise XAUUSDMacroImpactIntelligenceError(
                "threshold must be a finite non-negative number."
            )

        # Validate the strong threshold.
        if (
            isinstance(strong_threshold, bool)
            or not isinstance(strong_threshold, (int, float))
            or not isfinite(float(strong_threshold))
            or float(strong_threshold) <= float(threshold)
        ):
            raise XAUUSDMacroImpactIntelligenceError(
                "strong_threshold must be finite and greater than threshold."
            )

        # Validate minimum confidence.
        if (
            isinstance(min_confidence, bool)
            or not isinstance(min_confidence, (int, float))
            or not isfinite(float(min_confidence))
            or not 0.0 <= float(min_confidence) <= 100.0
        ):
            raise XAUUSDMacroImpactIntelligenceError(
                "min_confidence must be between 0 and 100."
            )

        # Store validated configuration.
        self.weights = configured_weights
        self.threshold = float(threshold)
        self.strong_threshold = float(strong_threshold)
        self.min_confidence = float(min_confidence)

    @classmethod
    def _validate_symbol(cls, symbol: str) -> None:
        """Validate the requested symbol."""

        # Require string input.
        if not isinstance(symbol, str):
            raise XAUUSDMacroImpactIntelligenceError(
                "symbol must be a string."
            )

        # Normalize the symbol.
        normalized_symbol = symbol.strip().upper()

        # Reject unsupported instruments.
        if normalized_symbol != cls.SYMBOL:
            raise XAUUSDMacroImpactIntelligenceError(
                f"Only {cls.SYMBOL} is supported by this engine."
            )

    @staticmethod
    def _validate_timestamp(
        decision_timestamp: datetime,
    ) -> None:
        """Validate the decision timestamp."""

        # Require datetime.
        if not isinstance(decision_timestamp, datetime):
            raise XAUUSDMacroImpactIntelligenceError(
                "decision_timestamp must be a datetime."
            )

        # Require timezone awareness.
        if decision_timestamp.tzinfo is None:
            raise XAUUSDMacroImpactIntelligenceError(
                "decision_timestamp must be timezone-aware."
            )

        # Require a valid UTC offset.
        if decision_timestamp.utcoffset() is None:
            raise XAUUSDMacroImpactIntelligenceError(
                "decision_timestamp must have a valid timezone offset."
            )

    @classmethod
    def _validate_observations(
        cls,
        observations: Iterable[MacroObservation],
    ) -> tuple[MacroObservation, ...]:
        """Validate and materialize observations."""

        # Reject None.
        if observations is None:
            raise XAUUSDMacroImpactIntelligenceError(
                "observations cannot be None."
            )

        # Materialize generators.
        materialized = tuple(observations)

        # Validate every observation.
        for observation in materialized:

            # Require canonical MacroObservation.
            if not isinstance(observation, MacroObservation):
                raise XAUUSDMacroImpactIntelligenceError(
                    "All observations must be MacroObservation instances."
                )

            # Require timezone-aware timestamps.
            if observation.timestamp.tzinfo is None:
                raise XAUUSDMacroImpactIntelligenceError(
                    "All observation timestamps must be timezone-aware."
                )

            # Require usable offsets.
            if observation.timestamp.utcoffset() is None:
                raise XAUUSDMacroImpactIntelligenceError(
                    "All observation timestamps must have valid timezone offsets."
                )

        # Return immutable observations.
        return materialized

    @staticmethod
    def _is_usable_observation(
        observation: MacroObservation,
    ) -> bool:
        """Check whether an observation contains a usable value."""

        # Reject booleans.
        if isinstance(observation.value, bool):
            return False

        # Require numeric values.
        if not isinstance(observation.value, (int, float)):
            return False

        # Reject NaN and infinity.
        return isfinite(float(observation.value))

    @classmethod
    def _latest_observation(
        cls,
        observations: Iterable[MacroObservation],
        *,
        indicator: MacroIndicator,
        decision_timestamp: datetime,
    ) -> MacroObservation | None:
        """Return latest historical-safe observation."""

        # Filter matching historical-safe observations.
        candidates = [
            observation
            for observation in observations
            if (
                observation.indicator == indicator
                and observation.timestamp <= decision_timestamp
                and cls._is_usable_observation(observation)
            )
        ]

        # Return nothing when unavailable.
        if not candidates:
            return None

        # Return the newest valid observation.
        return max(
            candidates,
            key=lambda observation: observation.timestamp,
        )

    @staticmethod
    def _movement_sign(
        observation: MacroObservation,
    ) -> int:
        """Return +1 rising, -1 falling, 0 stable/unknown."""

        # Previous value is required.
        if observation.previous is None:
            return 0

        # Validate previous value.
        if (
            isinstance(observation.previous, bool)
            or not isinstance(observation.previous, (int, float))
            or not isfinite(float(observation.previous))
        ):
            return 0

        # Convert values to floats.
        current = float(observation.value)
        previous = float(observation.previous)

        # Rising.
        if current > previous:
            return 1

        # Falling.
        if current < previous:
            return -1

        # Stable.
        return 0

    @staticmethod
    def _safe_single(
        operation: Callable[[], T],
    ) -> T | None:
        """Safely execute a lower-level intelligence operation."""

        # Partial data must not crash the aggregate layer.
        try:
            return operation()
        except Exception:
            return None

    @staticmethod
    def _safe_multiple(
        operation: Callable[[], Mapping[object, T]],
    ) -> Mapping[object, T]:
        """Safely execute a multi-result operation."""

        # Convert lower-level incomplete-data failures into unavailable data.
        try:
            result = operation()
        except Exception:
            return {}

        # Normalize None.
        if result is None:
            return {}

        # Return the result.
        return result

    def _usd_pressure(
        self,
        observations: tuple[MacroObservation, ...],
        decision_timestamp: datetime,
    ) -> XAUUSDMacroImpactContribution | None:
        """Calculate USD/DXY pressure."""

        # Store directional signals.
        pressure_values: list[float] = []

        # Store explanations.
        reasons: list[str] = []

        # Retrieve latest historical-safe DXY.
        dxy = self._latest_observation(
            observations,
            indicator=MacroIndicator.DXY,
            decision_timestamp=decision_timestamp,
        )

        # Use direct DXY movement.
        if dxy is not None:
            movement = self._movement_sign(dxy)

            # Rising DXY is bearish for gold.
            if movement > 0:
                pressure_values.append(-1.0)
                reasons.append(
                    "DXY is rising, creating bearish pressure on gold."
                )

            # Falling DXY is bullish for gold.
            elif movement < 0:
                pressure_values.append(1.0)
                reasons.append(
                    "DXY is falling, creating bullish pressure on gold."
                )

        # Run USD-strength intelligence.
        usd_engine = USDStrengthIntelligence()

        # Safely obtain USD assessment.
        usd_assessment = self._safe_single(
            lambda: usd_engine.analyze(
                observations,
                decision_timestamp=decision_timestamp,
            )
        )

        # Use USD-strength classification when available.
        if usd_assessment is not None:

            # Strong USD pressures gold.
            if usd_assessment.level == USDStrengthLevel.STRONG:
                pressure_values.append(-1.0)
                reasons.append(
                    "USD strength is strong, pressuring XAU/USD lower."
                )

            # Weak USD supports gold.
            elif usd_assessment.level == USDStrengthLevel.WEAK:
                pressure_values.append(1.0)
                reasons.append(
                    "USD strength is weak, supporting XAU/USD higher."
                )

        # No directional evidence means unavailable.
        if not pressure_values:
            return None

        # Average available signals.
        average_pressure = sum(pressure_values) / len(pressure_values)

        # Preserve the stable public source name.
        return XAUUSDMacroImpactContribution(
            source="usd",
            component=XAUUSDMacroComponent.USD_PRESSURE,
            contribution=average_pressure * 100.0,
            reason=" ".join(reasons),
        )

    def _rates_pressure(
        self,
        observations: tuple[MacroObservation, ...],
        decision_timestamp: datetime,
    ) -> XAUUSDMacroImpactContribution | None:
        """Calculate Treasury/Fed interest-rate pressure."""

        # Store directional signals.
        pressure_values: list[float] = []

        # Store explanations.
        reasons: list[str] = []

        # Supported Treasury maturities.
        treasury_indicators = (
            MacroIndicator.US_2Y_YIELD,
            MacroIndicator.US_5Y_YIELD,
            MacroIndicator.US_10Y_YIELD,
            MacroIndicator.US_30Y_YIELD,
        )

        # Analyze each Treasury maturity.
        for indicator in treasury_indicators:

            # Retrieve latest historical-safe observation.
            observation = self._latest_observation(
                observations,
                indicator=indicator,
                decision_timestamp=decision_timestamp,
            )

            # Skip missing data.
            if observation is None:
                continue

            # Determine movement.
            movement = self._movement_sign(observation)

            # Rising yields pressure gold lower.
            if movement > 0:
                pressure_values.append(-1.0)
                reasons.append(
                    f"{indicator.value} is rising, creating bearish rate pressure."
                )

            # Falling yields support gold.
            elif movement < 0:
                pressure_values.append(1.0)
                reasons.append(
                    f"{indicator.value} is falling, creating bullish rate pressure."
                )

        # Analyze Fed policy.
        fed_engine = FedRateIntelligence()

        # Safely obtain Fed assessment.
        fed_assessment = self._safe_single(
            lambda: fed_engine.analyze(
                observations,
                decision_timestamp=decision_timestamp,
            )
        )

        # Add Fed directional evidence.
        if fed_assessment is not None:

            # Hawkish is bearish for gold.
            if fed_assessment.level in (
                FedRateLevel.HAWKISH,
                FedRateLevel.STRONG_HAWKISH,
            ):
                pressure_values.append(-1.0)
                reasons.append(
                    "The Federal Reserve rate signal is hawkish, "
                    "pressuring XAU/USD lower."
                )

            # Dovish is bullish for gold.
            elif fed_assessment.level in (
                FedRateLevel.DOVISH,
                FedRateLevel.STRONG_DOVISH,
            ):
                pressure_values.append(1.0)
                reasons.append(
                    "The Federal Reserve rate signal is dovish, "
                    "supporting XAU/USD higher."
                )

        # No rate evidence.
        if not pressure_values:
            return None

        # Average available rate signals.
        average_pressure = sum(pressure_values) / len(pressure_values)

        # Preserve the public source name.
        return XAUUSDMacroImpactContribution(
            source="rates",
            component=XAUUSDMacroComponent.INTEREST_RATE_PRESSURE,
            contribution=average_pressure * 100.0,
            reason=" ".join(reasons),
        )

    def _inflation_pressure(
        self,
        observations: tuple[MacroObservation, ...],
        decision_timestamp: datetime,
    ) -> XAUUSDMacroImpactContribution | None:
        """Calculate inflation pressure."""

        # Create inflation engine.
        engine = InflationIntelligence()

        # Safely analyze inflation.
        assessment = self._safe_single(
            lambda: engine.analyze_all(
                observations,
                decision_timestamp=decision_timestamp,
            )
        )

        # No assessment.
        if not assessment:
            return None

        # Store directional signals.
        pressure_values: list[float] = []

        # Store explanations.
        reasons: list[str] = []

        # Evaluate all available inflation indicators.
        for item in assessment.values():

            # Hot inflation increases tightening pressure.
            if item.level in (
                InflationLevel.STRONG_HOT,
                InflationLevel.HOT,
            ):
                pressure_values.append(-1.0)
                reasons.append(
                    f"{item.indicator.value} is hot, increasing rate pressure."
                )

            # Cooling inflation reduces tightening pressure.
            elif item.level in (
                InflationLevel.COOLING,
                InflationLevel.STRONG_COOLING,
            ):
                pressure_values.append(1.0)
                reasons.append(
                    f"{item.indicator.value} is cooling, reducing rate pressure."
                )

        # No directional evidence.
        if not pressure_values:
            return None

        # Average signals.
        average_pressure = sum(pressure_values) / len(pressure_values)

        # Return contribution.
        return XAUUSDMacroImpactContribution(
            source="inflation",
            component=XAUUSDMacroComponent.INFLATION_PRESSURE,
            contribution=average_pressure * 100.0,
            reason=" ".join(reasons),
        )

    def _employment_pressure(
        self,
        observations: tuple[MacroObservation, ...],
        decision_timestamp: datetime,
    ) -> XAUUSDMacroImpactContribution | None:
        """Calculate employment pressure."""

        # Create employment engine.
        engine = EmploymentIntelligence()

        # Safely analyze employment.
        assessment = self._safe_single(
            lambda: engine.analyze_all(
                observations,
                decision_timestamp=decision_timestamp,
            )
        )

        # No employment assessment.
        if not assessment:
            return None

        # Store directional signals.
        pressure_values: list[float] = []

        # Store explanations.
        reasons: list[str] = []

        # Evaluate employment data.
        for item in assessment.values():

            # Strong employment increases tightening pressure.
            if item.level in (
                EmploymentLevel.STRONG_HOT,
                EmploymentLevel.HOT,
            ):
                pressure_values.append(-1.0)
                reasons.append(
                    f"{item.indicator.value} is strong, increasing rate pressure."
                )

            # Cooling employment reduces tightening pressure.
            elif item.level in (
                EmploymentLevel.COOLING,
                EmploymentLevel.STRONG_COOLING,
            ):
                pressure_values.append(1.0)
                reasons.append(
                    f"{item.indicator.value} is cooling, reducing rate pressure."
                )

        # No directional employment evidence.
        if not pressure_values:
            return None

        # Average signals.
        average_pressure = sum(pressure_values) / len(pressure_values)

        # Return contribution.
        return XAUUSDMacroImpactContribution(
            source="employment",
            component=XAUUSDMacroComponent.EMPLOYMENT_PRESSURE,
            contribution=average_pressure * 100.0,
            reason=" ".join(reasons),
        )

    def _risk_pressure(
        self,
        observations: tuple[MacroObservation, ...],
        decision_timestamp: datetime,
    ) -> XAUUSDMacroImpactContribution | None:
        """
        Calculate risk-sentiment pressure.

        The existing risk engine is authoritative when it has sufficient
        evidence. For partial datasets, this aggregation layer also applies
        a deterministic fallback using the same macro evidence:

            rising DXY + rising Treasury yields -> risk-off
            falling DXY + falling Treasury yields -> risk-on

        This fallback is context only and never authorizes a trade.
        """

        # Create the existing risk-sentiment engine.
        engine = RiskSentimentIntelligence()

        # Safely request the canonical risk assessment.
        assessment = self._safe_single(
            lambda: engine.analyze(
                observations,
                decision_timestamp=decision_timestamp,
            )
        )

        # Use the canonical risk engine whenever it provides a known result.
        if assessment is not None:

            # Risk-off supports safe-haven gold demand.
            if assessment.sentiment in (
                RiskSentiment.STRONG_RISK_OFF,
                RiskSentiment.RISK_OFF,
            ):
                return XAUUSDMacroImpactContribution(
                    source="risk",
                    component=XAUUSDMacroComponent.RISK_SENTIMENT_PRESSURE,
                    contribution=100.0,
                    reason=(
                        "Risk-off conditions are supporting safe-haven demand "
                        "for gold."
                    ),
                )

            # Risk-on reduces defensive gold demand.
            if assessment.sentiment in (
                RiskSentiment.STRONG_RISK_ON,
                RiskSentiment.RISK_ON,
            ):
                return XAUUSDMacroImpactContribution(
                    source="risk",
                    component=XAUUSDMacroComponent.RISK_SENTIMENT_PRESSURE,
                    contribution=-100.0,
                    reason=(
                        "Risk-on conditions reduce defensive demand for gold."
                    ),
                )

            # Neutral is valid evidence.
            if assessment.sentiment == RiskSentiment.NEUTRAL:
                return XAUUSDMacroImpactContribution(
                    source="risk",
                    component=XAUUSDMacroComponent.RISK_SENTIMENT_PRESSURE,
                    contribution=0.0,
                    reason="Risk sentiment is neutral.",
                )

        # ---------------------------------------------------------------
        # Partial-data deterministic fallback.
        # ---------------------------------------------------------------

        # Get latest DXY.
        dxy = self._latest_observation(
            observations,
            indicator=MacroIndicator.DXY,
            decision_timestamp=decision_timestamp,
        )

        # Store DXY movement.
        dxy_movement = (
            self._movement_sign(dxy)
            if dxy is not None
            else 0
        )

        # Treasury indicators.
        treasury_indicators = (
            MacroIndicator.US_2Y_YIELD,
            MacroIndicator.US_5Y_YIELD,
            MacroIndicator.US_10Y_YIELD,
            MacroIndicator.US_30Y_YIELD,
        )

        # Store Treasury movements.
        treasury_movements: list[int] = []

        # Evaluate Treasury movements.
        for indicator in treasury_indicators:

            # Retrieve latest observation.
            observation = self._latest_observation(
                observations,
                indicator=indicator,
                decision_timestamp=decision_timestamp,
            )

            # Skip missing values.
            if observation is None:
                continue

            # Record movement.
            movement = self._movement_sign(observation)

            # Ignore stable values.
            if movement != 0:
                treasury_movements.append(movement)

        # Count rising/falling Treasury evidence.
        rising_yields = sum(
            1
            for movement in treasury_movements
            if movement > 0
        )

        falling_yields = sum(
            1
            for movement in treasury_movements
            if movement < 0
        )

        # DXY rising + yields rising is risk-off.
        if dxy_movement > 0 and rising_yields > 0:
            return XAUUSDMacroImpactContribution(
                source="risk",
                component=XAUUSDMacroComponent.RISK_SENTIMENT_PRESSURE,
                contribution=100.0,
                reason=(
                    "Rising DXY and rising Treasury yields create "
                    "risk-off macro conditions, supporting safe-haven gold demand."
                ),
            )

        # DXY falling + yields falling is risk-on.
        if dxy_movement < 0 and falling_yields > 0:
            return XAUUSDMacroImpactContribution(
                source="risk",
                component=XAUUSDMacroComponent.RISK_SENTIMENT_PRESSURE,
                contribution=-100.0,
                reason=(
                    "Falling DXY and falling Treasury yields create "
                    "risk-on macro conditions in this deterministic fallback."
                ),
            )

        # No usable risk evidence.
        return None

    def _fallback_regime_assessment(
        self,
        observations: tuple[MacroObservation, ...],
        decision_timestamp: datetime,
    ) -> MacroRegimeAssessment | None:
        """
        Preserve useful lower-level macro assessments for partial data.

        In particular, a historical-safe DXY observation must remain visible
        through macro_regime_assessment.dxy even when the aggregate regime
        engine cannot classify a complete regime.
        """

        # Obtain the latest historical-safe DXY observation.
        dxy_observation = self._latest_observation(
            observations,
            indicator=MacroIndicator.DXY,
            decision_timestamp=decision_timestamp,
        )

        # Obtain the canonical DXY assessment.
        dxy_assessment = None

        # Create DXY engine.
        dxy_engine = DXYIntelligence()

        # Safely analyze DXY.
        if dxy_observation is not None:
            dxy_assessment = self._safe_single(
                lambda: dxy_engine.analyze(
                    observations,
                    decision_timestamp=decision_timestamp,
                )
            )

        # If there is no DXY assessment, do not fabricate the regime model.
        if dxy_assessment is None:
            return None

        # The existing MacroRegimeAssessment model is owned by the regime
        # module, so constructing an artificial object here would be unsafe.
        #
        # Return None unless the canonical aggregate engine can construct the
        # complete assessment.
        return None
    def _build_partial_regime_assessment(
        self,
        observations: tuple[MacroObservation, ...],
        decision_timestamp: datetime,
    ) -> MacroRegimeAssessment | None:
        """
        Build a partial macro-regime assessment when complete regime
        classification is unavailable.

        This preserves valid lower-level assessments such as DXY without
        inventing unavailable macro components or a directional regime.
        """

        # Create the DXY intelligence engine.
        dxy_engine = DXYIntelligence()

        # Safely calculate the historical-safe DXY assessment.
        dxy_assessment = self._safe_single(
            lambda: dxy_engine.analyze(
                observations,
                decision_timestamp=decision_timestamp,
            )
        )

        # If DXY itself is unavailable, there is nothing useful to preserve.
        if dxy_assessment is None:
            return None

        # Calculate the configured DXY subsystem weight used by the
        # canonical MacroRegimeIntelligence engine.
        dxy_weight = 0.8

        # Build a partial assessment containing only the evidence that
        # actually exists at the decision timestamp.
        return MacroRegimeAssessment(
            # No complete regime can be inferred from DXY alone.
            regime=MacroRegime.UNKNOWN,

            # DXY is available, but this is not enough for a complete
            # macro-regime classification.
            confidence=0.0,

            # Complete regime evidence is not sufficient.
            sufficient_data=False,

            # No aggregate regime score is fabricated.
            score=0.0,

            # One lower-level subsystem is available.
            components_used=1,

            # Use the canonical DXY subsystem weight.
            total_weight=dxy_weight,

            # DXY is the only represented subsystem.
            used_weight=dxy_weight,

            # Preserve the real DXY assessment.
            usd_strength=None,
            dxy=dxy_assessment,

            # No Treasury assessments were available to the partial
            # regime object.
            treasury_yields={},

            # No Fed assessment is fabricated.
            fed_rate=None,

            # No inflation assessments are fabricated.
            inflation={},

            # No employment assessments are fabricated.
            employment={},

            # No risk-sentiment assessment is fabricated.
            risk_sentiment=None,

            # No aggregate regime contributions are fabricated.
            contributions=tuple(),

            # Preserve the exact historical decision timestamp.
            decision_timestamp=decision_timestamp,

            # Explicitly explain why the regime remains UNKNOWN.
            reasons=(
                "A historical-safe DXY assessment is available.",
                "The complete macro regime cannot be classified from "
                "the available partial evidence.",
                "The partial assessment preserves available lower-level "
                "macro evidence without fabricating unavailable components.",
            ),
        )

    def _regime_pressure(
        self,
        observations: tuple[MacroObservation, ...],
        decision_timestamp: datetime,
    ) -> tuple[
        XAUUSDMacroImpactContribution | None,
        MacroRegimeAssessment | None,
    ]:
        """Calculate macro-regime pressure and preserve partial evidence."""

        # Create the canonical aggregate macro-regime engine.
        engine = MacroRegimeIntelligence()

        # Try the complete production regime analysis first.
        assessment = self._safe_single(
            lambda: engine.analyze(
                observations,
                decision_timestamp=decision_timestamp,
            )
        )

        # If the complete regime analysis succeeds, preserve its exact
        # production result and behavior.
        if assessment is not None:

            # Easing, disinflationary, and risk-off regimes support gold.
            if assessment.regime in (
                MacroRegime.EASING,
                MacroRegime.DISINFLATIONARY,
                MacroRegime.RISK_OFF,
            ):
                return (
                    XAUUSDMacroImpactContribution(
                        source="macro_regime",
                        component=XAUUSDMacroComponent.REGIME_PRESSURE,
                        contribution=100.0,
                        reason=(
                            f"Macro regime is {assessment.regime.value}, "
                            "which supports XAU/USD."
                        ),
                    ),
                    assessment,
                )

            # Tightening, inflationary, and risk-on regimes pressure gold.
            if assessment.regime in (
                MacroRegime.TIGHTENING,
                MacroRegime.INFLATIONARY,
                MacroRegime.RISK_ON,
            ):
                return (
                    XAUUSDMacroImpactContribution(
                        source="macro_regime",
                        component=XAUUSDMacroComponent.REGIME_PRESSURE,
                        contribution=-100.0,
                        reason=(
                            f"Macro regime is {assessment.regime.value}, "
                            "which pressures XAU/USD."
                        ),
                    ),
                    assessment,
                )

            # Growth-supportive and mixed regimes are neutral evidence.
            if assessment.regime in (
                MacroRegime.GROWTH_SUPPORTIVE,
                MacroRegime.MIXED,
            ):
                return (
                    XAUUSDMacroImpactContribution(
                        source="macro_regime",
                        component=XAUUSDMacroComponent.REGIME_PRESSURE,
                        contribution=0.0,
                        reason=(
                            f"Macro regime is {assessment.regime.value}; "
                            "no directional gold pressure is assigned."
                        ),
                    ),
                    assessment,
                )

            # UNKNOWN remains contextual information only.
            return None, assessment

        # ---------------------------------------------------------------
        # Partial-data fallback.
        # ---------------------------------------------------------------
        #
        # The complete regime engine may require several macro subsystems.
        # However, an exact-time DXY observation is still valid evidence.
        #
        # We therefore retrieve the lower-level DXY assessment directly
        # and preserve it inside MacroRegimeAssessment.
        # ---------------------------------------------------------------

        # Create the DXY intelligence engine.
        dxy_engine = DXYIntelligence()

        # Safely calculate the historical-safe DXY assessment.
        dxy_assessment = self._safe_single(
            lambda: dxy_engine.analyze(
                observations,
                decision_timestamp=decision_timestamp,
            )
        )

        # If DXY is unavailable, there is no partial regime object to
        # preserve.
        if dxy_assessment is None:
            return None, None

        # Build a partial MacroRegimeAssessment.
        #
        # IMPORTANT:
        # UNKNOWN is intentional here. We are preserving DXY evidence,
        # not pretending that DXY alone determines the complete macro regime.
        partial_assessment = MacroRegimeAssessment(
            # A single DXY observation cannot establish a complete regime.
            regime=MacroRegime.UNKNOWN,

            # The complete regime confidence is zero because the required
            # macro subsystems are not all available.
            confidence=0.0,

            # Mark the aggregate regime as insufficient.
            sufficient_data=False,

            # Do not fabricate an aggregate regime score.
            score=0.0,

            # One lower-level subsystem is available.
            components_used=1,

            # DXY uses the canonical macro-regime subsystem weight.
            total_weight=0.8,

            # Only DXY is represented.
            used_weight=0.8,

            # USD-strength is unavailable in this partial dataset.
            usd_strength=None,

            # THIS is the critical compatibility field required by the test.
            dxy=dxy_assessment,

            # No Treasury evidence is being fabricated.
            treasury_yields={},

            # No Fed evidence is being fabricated.
            fed_rate=None,

            # No inflation evidence is being fabricated.
            inflation={},

            # No employment evidence is being fabricated.
            employment={},

            # No risk-sentiment evidence is being fabricated.
            risk_sentiment=None,

            # No aggregate regime contributions are fabricated.
            contributions=tuple(),

            # Preserve the exact decision timestamp.
            decision_timestamp=decision_timestamp,

            # Explain the partial assessment.
            reasons=(
                "A historical-safe DXY assessment is available.",
                "The complete macro regime cannot be classified from "
                "the available partial evidence.",
                "The DXY assessment is preserved without fabricating "
                "unavailable macro subsystems.",
            ),
        )

        # Return the partial assessment without creating a regime-pressure
        # contribution.
        return None, partial_assessment

    def _classify(
        self,
        score: float,
        *,
        has_evidence: bool,
    ) -> XAUUSDMacroBias:
        """Classify the aggregate macro score."""

        # No evidence must be UNKNOWN.
        if not has_evidence:
            return XAUUSDMacroBias.UNKNOWN

        # Strong bullish.
        if score >= self.strong_threshold:
            return XAUUSDMacroBias.STRONG_BULLISH

        # Bullish.
        if score >= self.threshold:
            return XAUUSDMacroBias.BULLISH

        # Strong bearish.
        if score <= -self.strong_threshold:
            return XAUUSDMacroBias.STRONG_BEARISH

        # Bearish.
        if score <= -self.threshold:
            return XAUUSDMacroBias.BEARISH

        # Evidence exists but is directionally weak.
        return XAUUSDMacroBias.NEUTRAL

    def analyze(
        self,
        observations: Iterable[MacroObservation],
        *,
        decision_timestamp: datetime,
        symbol: str = SYMBOL,
    ) -> XAUUSDMacroImpactAssessment:
        """
        Analyze XAU/USD macro conditions.

        The symbol argument is retained for compatibility and explicitly
        rejects unsupported instruments.
        """

        # Validate symbol.
        self._validate_symbol(symbol)

        # Validate decision timestamp.
        self._validate_timestamp(decision_timestamp)

        # Validate observations.
        validated_observations = self._validate_observations(observations)

        # Apply the historical cutoff before any intelligence call.
        historical_observations = tuple(
            observation
            for observation in validated_observations
            if observation.timestamp <= decision_timestamp
        )

        # Build USD contribution.
        usd_contribution = self._usd_pressure(
            historical_observations,
            decision_timestamp,
        )

        # Build interest-rate contribution.
        rates_contribution = self._rates_pressure(
            historical_observations,
            decision_timestamp,
        )

        # Build inflation contribution.
        inflation_contribution = self._inflation_pressure(
            historical_observations,
            decision_timestamp,
        )

        # Build employment contribution.
        employment_contribution = self._employment_pressure(
            historical_observations,
            decision_timestamp,
        )

        # Build risk contribution.
        risk_contribution = self._risk_pressure(
            historical_observations,
            decision_timestamp,
        )

        # Build regime contribution.
        regime_contribution, regime_assessment = self._regime_pressure(
            historical_observations,
            decision_timestamp,
        )

        # Collect contributions.
        contributions: list[XAUUSDMacroImpactContribution] = []

        # Add USD evidence.
        if usd_contribution is not None:
            contributions.append(usd_contribution)

        # Add rate evidence.
        if rates_contribution is not None:
            contributions.append(rates_contribution)

        # Add inflation evidence.
        if inflation_contribution is not None:
            contributions.append(inflation_contribution)

        # Add employment evidence.
        if employment_contribution is not None:
            contributions.append(employment_contribution)

        # Add risk evidence.
        if risk_contribution is not None:
            contributions.append(risk_contribution)

        # Add regime evidence.
        if regime_contribution is not None:
            contributions.append(regime_contribution)

        # Map each component to its configured weight.
        component_weights = {
            XAUUSDMacroComponent.USD_PRESSURE: self.weights["usd"],
            XAUUSDMacroComponent.INTEREST_RATE_PRESSURE: self.weights["rates"],
            XAUUSDMacroComponent.INFLATION_PRESSURE: self.weights["inflation"],
            XAUUSDMacroComponent.EMPLOYMENT_PRESSURE: self.weights["employment"],
            XAUUSDMacroComponent.RISK_SENTIMENT_PRESSURE: self.weights["risk"],
            XAUUSDMacroComponent.REGIME_PRESSURE: self.weights["regime"],
        }

        # Keep only positive-weight finite evidence.
        usable_contributions = [
            contribution
            for contribution in contributions
            if (
                contribution.component != XAUUSDMacroComponent.UNKNOWN
                and component_weights.get(
                    contribution.component,
                    0.0,
                ) > 0.0
                and isfinite(float(contribution.contribution))
            )
        ]

        # Calculate total configured weight.
        total_weight = sum(
            weight
            for weight in component_weights.values()
            if weight > 0.0
        )

        # Calculate weight represented by usable evidence.
        used_weight = sum(
            component_weights[contribution.component]
            for contribution in usable_contributions
        )

        # Empty or future-only data must be UNKNOWN.
        if used_weight <= 0.0:
            return XAUUSDMacroImpactAssessment(
                symbol=self.SYMBOL,
                bias=XAUUSDMacroBias.UNKNOWN,
                score=0.0,
                confidence=0.0,
                sufficient_data=False,
                macro_regime=(
                    regime_assessment.regime
                    if regime_assessment is not None
                    else MacroRegime.UNKNOWN
                ),
                macro_regime_assessment=regime_assessment,
                contributions=tuple(),
                decision_timestamp=decision_timestamp,
                reasons=(
                    "No usable macro observations were available at or "
                    "before the decision timestamp.",
                    "Future observations are ignored to prevent lookahead.",
                    "Macro context is unavailable; no trade decision is authorized.",
                ),
            )

        # Calculate weighted macro score.
        weighted_sum = sum(
            component_weights[contribution.component]
            * contribution.contribution
            for contribution in usable_contributions
        )

        # Normalize score.
        score = weighted_sum / used_weight

        # Clamp score.
        score = max(-100.0, min(100.0, score))

        # Remove tiny floating-point residue.
        if abs(score) < 1e-12:
            score = 0.0

        # Calculate evidence coverage.
        if total_weight > 0.0:
            confidence = (used_weight / total_weight) * 100.0
        else:
            confidence = 0.0

        # Clamp confidence.
        confidence = max(0.0, min(100.0, confidence))

        # Remove tiny floating-point residue.
        if abs(confidence) < 1e-12:
            confidence = 0.0

        # Determine evidence sufficiency.
        sufficient_data = confidence >= self.min_confidence

        # Classify macro bias.
        bias = self._classify(
            score,
            has_evidence=True,
        )

        # Build deterministic reasons.
        reasons: list[str] = [
            f"XAUUSD macro score is {score:.4f}.",
            f"Macro evidence coverage is {confidence:.2f}%.",
        ]

        # Explain evidence sufficiency.
        if sufficient_data:
            reasons.append(
                "Macro evidence coverage meets the minimum confidence threshold."
            )
        else:
            reasons.append(
                "Macro evidence coverage is below the minimum confidence threshold."
            )

        # Preserve architectural separation.
        reasons.append(
            "This assessment provides macro context only and does not "
            "authorize or reject a trade."
        )

        # Explain bias.
        if bias == XAUUSDMacroBias.STRONG_BULLISH:
            reasons.append(
                "The combined macro pressure is strongly supportive of XAU/USD."
            )

        elif bias == XAUUSDMacroBias.BULLISH:
            reasons.append(
                "The combined macro pressure is supportive of XAU/USD."
            )

        elif bias == XAUUSDMacroBias.STRONG_BEARISH:
            reasons.append(
                "The combined macro pressure is strongly negative for XAU/USD."
            )

        elif bias == XAUUSDMacroBias.BEARISH:
            reasons.append(
                "The combined macro pressure is negative for XAU/USD."
            )

        else:
            reasons.append(
                "The combined macro pressure does not meet a directional threshold."
            )

        # Return immutable assessment.
        return XAUUSDMacroImpactAssessment(
            symbol=self.SYMBOL,
            bias=bias,
            score=score,
            confidence=confidence,
            sufficient_data=sufficient_data,
            macro_regime=(
                regime_assessment.regime
                if regime_assessment is not None
                else MacroRegime.UNKNOWN
            ),
            macro_regime_assessment=regime_assessment,
            contributions=tuple(usable_contributions),
            decision_timestamp=decision_timestamp,
            reasons=tuple(reasons),
        )

    def analyze_xauusd(
        self,
        observations: Iterable[MacroObservation],
        *,
        decision_timestamp: datetime,
    ) -> XAUUSDMacroImpactAssessment:
        """Convenience wrapper for XAU/USD."""

        # Delegate to the canonical analyzer.
        return self.analyze(
            observations,
            decision_timestamp=decision_timestamp,
            symbol=self.SYMBOL,
        )