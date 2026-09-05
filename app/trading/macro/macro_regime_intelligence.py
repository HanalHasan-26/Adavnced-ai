"""
Macro regime intelligence for the Advanced AI trading engine.

This module combines the existing deterministic macro intelligence
components into a higher-level macro regime assessment.

Supported regime classifications:
- GROWTH_SUPPORTIVE
- INFLATIONARY
- DISINFLATIONARY
- TIGHTENING
- EASING
- RISK_ON
- RISK_OFF
- MIXED
- UNKNOWN

Important:
- No external data fetching.
- No LLM dependency.
- No direct trade execution.
- Historical/no-lookahead safety is preserved.
- Missing macro datasets are tolerated at the aggregate layer.
"""

from __future__ import annotations

# Import dataclass support for immutable result objects.
from dataclasses import dataclass, fields, is_dataclass

# Import Enum for strongly typed classifications.
from enum import Enum

# Import datetime for historical decision-time validation.
from datetime import datetime

# Import math for finite-number validation.
import math

# Import the existing macro observation model.
from app.trading.macro.macro_observation import (
    MacroIndicator,
    MacroObservation,
)

# Import USD strength intelligence.
from app.trading.macro.usd_strength_intelligence import (
    USDStrengthAssessment,
    USDStrengthIntelligence,
    USDStrengthIntelligenceError,
)

# Import DXY intelligence.
from app.trading.macro.dxy_intelligence import (
    DXYAssessment,
    DXYIntelligence,
    DXYIntelligenceError,
)

# Import Treasury yield intelligence.
from app.trading.macro.treasury_yield_intelligence import (
    TreasuryYieldAssessment,
    TreasuryYieldIntelligence,
    TreasuryYieldIntelligenceError,
)

# Import Federal Reserve rate intelligence.
from app.trading.macro.fed_rate_intelligence import (
    FedRateAssessment,
    FedRateIntelligence,
    FedRateIntelligenceError,
)

# Import inflation intelligence.
from app.trading.macro.inflation_intelligence import (
    InflationAssessment,
    InflationIntelligence,
    InflationIntelligenceError,
)

# Import employment intelligence.
from app.trading.macro.employment_intelligence import (
    EmploymentAssessment,
    EmploymentIntelligence,
    EmploymentIntelligenceError,
)

# Import risk sentiment intelligence.
from app.trading.macro.risk_sentiment_intelligence import (
    RiskSentiment,
    RiskSentimentAssessment,
    RiskSentimentIntelligence,
    RiskSentimentIntelligenceError,
)


class MacroRegimeIntelligenceError(ValueError):
    """Raised when macro regime intelligence receives invalid input."""


class MacroRegime(str, Enum):
    """High-level macroeconomic regime."""

    GROWTH_SUPPORTIVE = "growth_supportive"
    INFLATIONARY = "inflationary"
    DISINFLATIONARY = "disinflationary"
    TIGHTENING = "tightening"
    EASING = "easing"
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class MacroRegimeComponent(str, Enum):
    """Interpretation of an individual macro subsystem."""

    SUPPORTIVE = "supportive"
    HAWKISH = "hawkish"
    DOVISH = "dovish"
    INFLATIONARY = "inflationary"
    DISINFLATIONARY = "disinflationary"
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class MacroRegimeContribution:
    """Immutable contribution from one macro subsystem."""

    # Name of the subsystem.
    source: str

    # Interpreted macro component.
    component: MacroRegimeComponent

    # Configured subsystem weight.
    weight: float

    # Signed contribution to the aggregate score.
    contribution: float

    # Human-readable deterministic explanation.
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Serialize the contribution."""

        # Return a JSON-friendly representation.
        return {
            "source": self.source,
            "component": self.component.value,
            "weight": self.weight,
            "contribution": self.contribution,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class MacroRegimeAssessment:
    """Immutable macro regime assessment."""

    # Final macro regime.
    regime: MacroRegime

    # Confidence from 0 to 100.
    confidence: float

    # Whether sufficient weighted information exists.
    sufficient_data: bool

    # Aggregate normalized score.
    score: float

    # Number of usable macro subsystems.
    components_used: int

    # Total configured subsystem weight.
    total_weight: float

    # Weight represented by usable subsystems.
    used_weight: float

    # USD strength assessment.
    usd_strength: USDStrengthAssessment | None

    # DXY assessment.
    dxy: DXYAssessment | None

    # Treasury yield assessments.
    treasury_yields: dict[
        MacroIndicator,
        TreasuryYieldAssessment,
    ]

    # Federal Reserve rate assessment.
    fed_rate: FedRateAssessment | None

    # Inflation assessments.
    inflation: dict[
        MacroIndicator,
        InflationAssessment,
    ]

    # Employment assessments.
    employment: dict[
        MacroIndicator,
        EmploymentAssessment,
    ]

    # Risk sentiment assessment.
    risk_sentiment: RiskSentimentAssessment | None

    # Individual subsystem contributions.
    contributions: tuple[MacroRegimeContribution, ...]

    # Decision timestamp.
    decision_timestamp: datetime

    # Deterministic explanations.
    reasons: tuple[str, ...]

    @staticmethod
    def _serialize_value(value: object) -> object:
        """
        Convert an assessment value into a JSON-friendly structure.

        This method intentionally supports lower-level assessment objects
        that may not expose their own to_dict() method.
        """

        # Use the object's native serializer when available.
        serializer = getattr(value, "to_dict", None)

        # Call the serializer when it is callable.
        if callable(serializer):
            return serializer()

        # Serialize Enum values using their string value.
        if isinstance(value, Enum):
            return value.value

        # Serialize datetime values as ISO strings.
        if isinstance(value, datetime):
            return value.isoformat()

        # Recursively serialize dictionaries.
        if isinstance(value, dict):
            return {
                (
                    key.value
                    if isinstance(key, Enum)
                    else str(key)
                ): MacroRegimeAssessment._serialize_value(item)
                for key, item in value.items()
            }

        # Recursively serialize lists and tuples.
        if isinstance(value, (list, tuple)):
            return [
                MacroRegimeAssessment._serialize_value(item)
                for item in value
            ]

        # Recursively serialize dataclass instances.
        if is_dataclass(value) and not isinstance(value, type):
            return {
                field.name: MacroRegimeAssessment._serialize_value(
                    getattr(value, field.name)
                )
                for field in fields(value)
            }

        # Primitive values can be returned directly.
        return value

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete assessment."""

        # Serialize Treasury assessments.
        treasury_data = {
            indicator.value: self._serialize_value(assessment)
            for indicator, assessment in self.treasury_yields.items()
        }

        # Serialize inflation assessments.
        inflation_data = {
            indicator.value: self._serialize_value(assessment)
            for indicator, assessment in self.inflation.items()
        }

        # Serialize employment assessments.
        employment_data = {
            indicator.value: self._serialize_value(assessment)
            for indicator, assessment in self.employment.items()
        }

        # Return the complete JSON-friendly structure.
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "sufficient_data": self.sufficient_data,
            "score": self.score,
            "components_used": self.components_used,
            "total_weight": self.total_weight,
            "used_weight": self.used_weight,

            # Serialize USD strength when available.
            "usd_strength": self._serialize_value(
                self.usd_strength
            ),

            # Serialize DXY when available.
            "dxy": self._serialize_value(
                self.dxy
            ),

            # Include Treasury assessments.
            "treasury_yields": treasury_data,

            # Serialize Fed rate when available.
            "fed_rate": self._serialize_value(
                self.fed_rate
            ),

            # Include inflation assessments.
            "inflation": inflation_data,

            # Include employment assessments.
            "employment": employment_data,

            # Serialize risk sentiment when available.
            "risk_sentiment": self._serialize_value(
                self.risk_sentiment
            ),

            # Serialize regime contributions.
            "contributions": [
                contribution.to_dict()
                for contribution in self.contributions
            ],

            # Serialize the decision timestamp.
            "decision_timestamp": (
                self.decision_timestamp.isoformat()
            ),

            # Serialize deterministic explanations.
            "reasons": list(self.reasons),
        }


class MacroRegimeIntelligence:
    """
    Deterministic macro regime aggregation engine.

    This engine combines the existing macro intelligence layers.

    It does not replace those engines and does not make trade decisions.
    """

    # Default weights for each macro subsystem.
    DEFAULT_WEIGHTS = {
        "usd_strength": 1.00,
        "dxy": 0.80,
        "treasury_yields": 1.00,
        "fed_rate": 1.00,
        "inflation": 1.00,
        "employment": 0.90,
        "risk_sentiment": 1.00,
    }

    # Minimum weighted coverage required for a known regime.
    DEFAULT_MIN_COVERAGE = 0.50

    # Normal directional threshold.
    DEFAULT_DIRECTION_THRESHOLD = 20.0

    # Strong directional threshold.
    DEFAULT_STRONG_DIRECTION_THRESHOLD = 50.0

    def __init__(
        self,
        *,
        weights: dict[str, float] | None = None,
        min_coverage: float = DEFAULT_MIN_COVERAGE,
        direction_threshold: float = DEFAULT_DIRECTION_THRESHOLD,
        strong_direction_threshold: float = (
            DEFAULT_STRONG_DIRECTION_THRESHOLD
        ),
        usd_strength_engine: USDStrengthIntelligence | None = None,
        dxy_engine: DXYIntelligence | None = None,
        treasury_engine: TreasuryYieldIntelligence | None = None,
        fed_rate_engine: FedRateIntelligence | None = None,
        inflation_engine: InflationIntelligence | None = None,
        employment_engine: EmploymentIntelligence | None = None,
        risk_sentiment_engine: RiskSentimentIntelligence | None = None,
    ) -> None:
        """Initialize the macro regime engine."""

        # Copy the defaults so the class-level dictionary is never mutated.
        selected_weights = (
            dict(self.DEFAULT_WEIGHTS)
            if weights is None
            else dict(weights)
        )

        # Validate configuration.
        self._validate_configuration(
            selected_weights,
            min_coverage,
            direction_threshold,
            strong_direction_threshold,
        )

        # Store weights.
        self.weights = selected_weights

        # Store minimum coverage.
        self.min_coverage = float(min_coverage)

        # Store normal threshold.
        self.direction_threshold = float(
            direction_threshold
        )

        # Store strong threshold.
        self.strong_direction_threshold = float(
            strong_direction_threshold
        )

        # Initialize or reuse USD-strength engine.
        self.usd_strength_engine = (
            usd_strength_engine
            if usd_strength_engine is not None
            else USDStrengthIntelligence()
        )

        # Initialize or reuse DXY engine.
        self.dxy_engine = (
            dxy_engine
            if dxy_engine is not None
            else DXYIntelligence()
        )

        # Initialize or reuse Treasury engine.
        self.treasury_engine = (
            treasury_engine
            if treasury_engine is not None
            else TreasuryYieldIntelligence()
        )

        # Initialize or reuse Fed-rate engine.
        self.fed_rate_engine = (
            fed_rate_engine
            if fed_rate_engine is not None
            else FedRateIntelligence()
        )

        # Initialize or reuse inflation engine.
        self.inflation_engine = (
            inflation_engine
            if inflation_engine is not None
            else InflationIntelligence()
        )

        # Initialize or reuse employment engine.
        self.employment_engine = (
            employment_engine
            if employment_engine is not None
            else EmploymentIntelligence()
        )

        # Initialize or reuse risk-sentiment engine.
        self.risk_sentiment_engine = (
            risk_sentiment_engine
            if risk_sentiment_engine is not None
            else RiskSentimentIntelligence()
        )

    @classmethod
    def _validate_configuration(
        cls,
        weights: dict[str, float],
        min_coverage: float,
        direction_threshold: float,
        strong_direction_threshold: float,
    ) -> None:
        """Validate engine configuration."""

        # Define supported subsystem names.
        valid_sources = set(cls.DEFAULT_WEIGHTS)

        # Require a dictionary.
        if not isinstance(weights, dict):
            raise MacroRegimeIntelligenceError(
                "weights must be a dictionary."
            )

        # Validate configured weights.
        for source, weight in weights.items():

            # Reject unknown subsystem names.
            if source not in valid_sources:
                raise MacroRegimeIntelligenceError(
                    f"Unknown macro regime source: {source}."
                )

            # Reject boolean values.
            if isinstance(weight, bool):
                raise MacroRegimeIntelligenceError(
                    "Macro regime weights must be numeric."
                )

            # Require numeric values.
            if not isinstance(weight, (int, float)):
                raise MacroRegimeIntelligenceError(
                    "Macro regime weights must be numeric."
                )

            # Convert to float.
            numeric_weight = float(weight)

            # Reject NaN and infinity.
            if not math.isfinite(numeric_weight):
                raise MacroRegimeIntelligenceError(
                    "Macro regime weights must be finite."
                )

            # Negative weights are not supported.
            if numeric_weight < 0:
                raise MacroRegimeIntelligenceError(
                    "Macro regime weights cannot be negative."
                )

        # Validate minimum coverage.
        if isinstance(min_coverage, bool):
            raise MacroRegimeIntelligenceError(
                "min_coverage must be numeric."
            )

        # Convert coverage to float.
        coverage = float(min_coverage)

        # Coverage must be within 0..1.
        if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            raise MacroRegimeIntelligenceError(
                "min_coverage must be between 0 and 1."
            )

        # Validate directional thresholds.
        for threshold in (
            direction_threshold,
            strong_direction_threshold,
        ):
            # Reject boolean values.
            if isinstance(threshold, bool):
                raise MacroRegimeIntelligenceError(
                    "Macro regime thresholds must be numeric."
                )

            # Require numeric values.
            if not isinstance(threshold, (int, float)):
                raise MacroRegimeIntelligenceError(
                    "Macro regime thresholds must be numeric."
                )

            # Convert to float.
            numeric_threshold = float(threshold)

            # Reject invalid numeric values.
            if not math.isfinite(numeric_threshold):
                raise MacroRegimeIntelligenceError(
                    "Macro regime thresholds must be finite."
                )

            # Negative thresholds are invalid.
            if numeric_threshold < 0:
                raise MacroRegimeIntelligenceError(
                    "Macro regime thresholds cannot be negative."
                )

        # Strong threshold cannot be lower than normal threshold.
        if strong_direction_threshold < direction_threshold:
            raise MacroRegimeIntelligenceError(
                "strong_direction_threshold must be >= "
                "direction_threshold."
            )

    @staticmethod
    def _validate_inputs(
        observations: list[MacroObservation],
        decision_timestamp: datetime,
    ) -> None:
        """Validate analysis inputs."""

        # Require a list.
        if not isinstance(observations, list):
            raise MacroRegimeIntelligenceError(
                "observations must be a list."
            )

        # Require datetime.
        if not isinstance(decision_timestamp, datetime):
            raise MacroRegimeIntelligenceError(
                "decision_timestamp must be a datetime."
            )

        # Require timezone-aware decision time.
        if decision_timestamp.tzinfo is None:
            raise MacroRegimeIntelligenceError(
                "decision_timestamp must be timezone-aware."
            )

        # Validate observations.
        for observation in observations:

            # Require the expected model.
            if not isinstance(observation, MacroObservation):
                raise MacroRegimeIntelligenceError(
                    "Every observation must be a MacroObservation."
                )

            # Require timezone-aware observations.
            if observation.timestamp.tzinfo is None:
                raise MacroRegimeIntelligenceError(
                    "Observation timestamps must be timezone-aware."
                )

    @staticmethod
    def _safe_single(
        analyzer: object,
        observations: list[MacroObservation],
        *,
        decision_timestamp: datetime,
        expected_error: type[Exception],
    ) -> object | None:
        """
        Execute a single-indicator analyzer safely.

        Missing historical data is represented by None.
        """

        # Run the strict lower-level analyzer.
        try:
            return analyzer.analyze(
                observations,
                decision_timestamp=decision_timestamp,
            )

        # Only convert the known lower-level data error.
        except expected_error:
            return None

    @staticmethod
    def _safe_multiple(
        analyzer: object,
        observations: list[MacroObservation],
        *,
        decision_timestamp: datetime,
        expected_error: type[Exception],
    ) -> dict:
        """
        Execute a multi-indicator analyzer safely.

        Missing historical data is represented by an empty dictionary.
        """

        # Run the strict lower-level analyzer.
        try:
            result = analyzer.analyze_all(
                observations,
                decision_timestamp=decision_timestamp,
            )

        # Convert known no-data errors to an empty result.
        except expected_error:
            return {}

        # Protect the aggregate layer from invalid return shapes.
        if not isinstance(result, dict):
            return {}

        # Return the available assessments.
        return result

    @staticmethod
    def _average_direction(
        assessments: list[object],
    ) -> MacroRegimeComponent:
        """Convert directional assessments into a macro component."""

        # Store usable directions.
        directions: list[str] = []

        # Inspect every assessment.
        for assessment in assessments:

            # Read the direction attribute.
            direction = getattr(
                assessment,
                "direction",
                None,
            )

            # Ignore missing direction.
            if direction is None:
                continue

            # Read enum value.
            direction_value = getattr(
                direction,
                "value",
                None,
            )

            # Ignore unknown direction.
            if direction_value in {
                None,
                "unknown",
            }:
                continue

            # Store valid direction.
            directions.append(direction_value)

        # No directional information.
        if not directions:
            return MacroRegimeComponent.UNKNOWN

        # Count rising values.
        rising = directions.count("rising")

        # Count falling values.
        falling = directions.count("falling")

        # Rising means hawkish.
        if rising > falling:
            return MacroRegimeComponent.HAWKISH

        # Falling means dovish.
        if falling > rising:
            return MacroRegimeComponent.DOVISH

        # Equal evidence is neutral.
        return MacroRegimeComponent.NEUTRAL

    @staticmethod
    def _inflation_component(
        inflation: dict[
            MacroIndicator,
            InflationAssessment,
        ],
    ) -> MacroRegimeComponent:
        """Interpret inflation assessments."""

        # No data means unknown.
        if not inflation:
            return MacroRegimeComponent.UNKNOWN

        # Count hot observations.
        hot = sum(
            assessment.level.value in {
                "hot",
                "strong_hot",
            }
            for assessment in inflation.values()
        )

        # Count cooling observations.
        cooling = sum(
            assessment.level.value in {
                "cooling",
                "strong_cooling",
            }
            for assessment in inflation.values()
        )

        # Hot inflation dominates.
        if hot > cooling and hot > 0:
            return MacroRegimeComponent.INFLATIONARY

        # Cooling inflation dominates.
        if cooling > hot and cooling > 0:
            return MacroRegimeComponent.DISINFLATIONARY

        # Conflicting or neutral inflation.
        return MacroRegimeComponent.NEUTRAL

    @staticmethod
    def _employment_component(
        employment: dict[
            MacroIndicator,
            EmploymentAssessment,
        ],
    ) -> MacroRegimeComponent:
        """Interpret employment assessments."""

        # No employment data.
        if not employment:
            return MacroRegimeComponent.UNKNOWN

        # Count strong employment.
        hot = sum(
            assessment.level.value in {
                "hot",
                "strong_hot",
            }
            for assessment in employment.values()
        )

        # Count cooling employment.
        cooling = sum(
            assessment.level.value in {
                "cooling",
                "strong_cooling",
            }
            for assessment in employment.values()
        )

        # Strong employment supports growth.
        if hot > cooling and hot > 0:
            return MacroRegimeComponent.SUPPORTIVE

        # Weakening employment is dovish.
        if cooling > hot and cooling > 0:
            return MacroRegimeComponent.DOVISH

        # Conflicting or neutral employment.
        return MacroRegimeComponent.NEUTRAL

    @staticmethod
    def _risk_component(
        risk_sentiment: RiskSentimentAssessment | None,
    ) -> MacroRegimeComponent:
        """Convert risk sentiment into a macro component."""

        # No assessment means unknown.
        if risk_sentiment is None:
            return MacroRegimeComponent.UNKNOWN

        # Risk-on classifications.
        if risk_sentiment.sentiment in {
            RiskSentiment.STRONG_RISK_ON,
            RiskSentiment.RISK_ON,
        }:
            return MacroRegimeComponent.RISK_ON

        # Risk-off classifications.
        if risk_sentiment.sentiment in {
            RiskSentiment.STRONG_RISK_OFF,
            RiskSentiment.RISK_OFF,
        }:
            return MacroRegimeComponent.RISK_OFF

        # Neutral classification.
        if risk_sentiment.sentiment == RiskSentiment.NEUTRAL:
            return MacroRegimeComponent.NEUTRAL

        # Unknown classification.
        return MacroRegimeComponent.UNKNOWN

    @staticmethod
    def _component_score(
        component: MacroRegimeComponent,
    ) -> float:
        """Convert a macro component into a signed score."""

        # Positive/supportive conditions.
        if component in {
            MacroRegimeComponent.SUPPORTIVE,
            MacroRegimeComponent.RISK_ON,
            MacroRegimeComponent.DISINFLATIONARY,
            MacroRegimeComponent.DOVISH,
        }:
            return 1.0

        # Negative/defensive conditions.
        if component in {
            MacroRegimeComponent.HAWKISH,
            MacroRegimeComponent.INFLATIONARY,
            MacroRegimeComponent.RISK_OFF,
        }:
            return -1.0

        # Neutral and unknown contribute zero.
        return 0.0

    def _classify(
        self,
        score: float,
        *,
        inflation_component: MacroRegimeComponent,
        monetary_component: MacroRegimeComponent,
        risk_component: MacroRegimeComponent,
        employment_component: MacroRegimeComponent,
    ) -> MacroRegime:
        """Classify the aggregate macro regime."""

        # Normalize floating-point residue.
        normalized_score = round(score, 10)

        # Inflation plus hawkish policy indicates tightening.
        if (
            inflation_component
            == MacroRegimeComponent.INFLATIONARY
            and monetary_component
            == MacroRegimeComponent.HAWKISH
        ):
            return MacroRegime.TIGHTENING

        # Disinflation plus dovish policy indicates easing.
        if (
            inflation_component
            == MacroRegimeComponent.DISINFLATIONARY
            and monetary_component
            == MacroRegimeComponent.DOVISH
        ):
            return MacroRegime.EASING

        # Strong inflationary environment.
        if (
            inflation_component
            == MacroRegimeComponent.INFLATIONARY
            and normalized_score
            <= -self.direction_threshold
        ):
            return MacroRegime.INFLATIONARY

        # Strong disinflationary environment.
        if (
            inflation_component
            == MacroRegimeComponent.DISINFLATIONARY
            and normalized_score
            >= self.direction_threshold
        ):
            return MacroRegime.DISINFLATIONARY

        # Risk-off environment.
        if (
            risk_component
            == MacroRegimeComponent.RISK_OFF
            and normalized_score
            <= -self.direction_threshold
        ):
            return MacroRegime.RISK_OFF

        # Risk-on environment.
        if (
            risk_component
            == MacroRegimeComponent.RISK_ON
            and normalized_score
            >= self.direction_threshold
        ):
            return MacroRegime.RISK_ON

        # Growth-supportive environment.
        if (
            employment_component
            == MacroRegimeComponent.SUPPORTIVE
            and normalized_score
            >= self.direction_threshold
        ):
            return MacroRegime.GROWTH_SUPPORTIVE

        # Hawkish monetary environment.
        if (
            monetary_component
            == MacroRegimeComponent.HAWKISH
            and normalized_score
            <= -self.direction_threshold
        ):
            return MacroRegime.TIGHTENING

        # Dovish monetary environment.
        if (
            monetary_component
            == MacroRegimeComponent.DOVISH
            and normalized_score
            >= self.direction_threshold
        ):
            return MacroRegime.EASING

        # No dominant regime.
        return MacroRegime.MIXED

    def analyze(
        self,
        observations: list[MacroObservation],
        *,
        decision_timestamp: datetime,
    ) -> MacroRegimeAssessment:
        """
        Build a historical-safe macro regime assessment.

        Partial macro data is supported.
        """

        # Validate all inputs.
        self._validate_inputs(
            observations,
            decision_timestamp,
        )

        # Analyze USD strength without allowing missing data to crash
        # the aggregate regime engine.
        usd_strength = self._safe_single(
            self.usd_strength_engine,
            observations,
            decision_timestamp=decision_timestamp,
            expected_error=USDStrengthIntelligenceError,
        )

        # Analyze DXY safely.
        dxy = self._safe_single(
            self.dxy_engine,
            observations,
            decision_timestamp=decision_timestamp,
            expected_error=DXYIntelligenceError,
        )

        # Analyze Treasury yields safely.
        treasury_yields = self._safe_multiple(
            self.treasury_engine,
            observations,
            decision_timestamp=decision_timestamp,
            expected_error=TreasuryYieldIntelligenceError,
        )

        # Analyze Fed rate safely.
        fed_rate = self._safe_single(
            self.fed_rate_engine,
            observations,
            decision_timestamp=decision_timestamp,
            expected_error=FedRateIntelligenceError,
        )

        # Analyze inflation safely.
        inflation = self._safe_multiple(
            self.inflation_engine,
            observations,
            decision_timestamp=decision_timestamp,
            expected_error=InflationIntelligenceError,
        )

        # Analyze employment safely.
        employment = self._safe_multiple(
            self.employment_engine,
            observations,
            decision_timestamp=decision_timestamp,
            expected_error=EmploymentIntelligenceError,
        )

        # Analyze risk sentiment safely.
        risk_sentiment = self._safe_single(
            self.risk_sentiment_engine,
            observations,
            decision_timestamp=decision_timestamp,
            expected_error=RiskSentimentIntelligenceError,
        )

        # Interpret inflation.
        inflation_component = self._inflation_component(
            inflation
        )

        # Interpret employment.
        employment_component = self._employment_component(
            employment
        )

        # Interpret monetary conditions.
        if fed_rate is None:
            monetary_component = MacroRegimeComponent.UNKNOWN
        else:
            monetary_component = self._average_direction(
                [fed_rate]
            )

        # Interpret risk sentiment.
        risk_component = self._risk_component(
            risk_sentiment
        )

        # Prepare contributions.
        contributions: list[
            MacroRegimeContribution
        ] = []

        # --------------------------------------------------------------
        # USD STRENGTH
        # --------------------------------------------------------------

        # Determine USD component.
        if usd_strength is None:
            usd_component = MacroRegimeComponent.UNKNOWN
            usd_reason = (
                "USD strength data is unavailable."
            )
        else:
            # Strong USD is treated as hawkish/defensive.
            if usd_strength.level.value == "strong":
                usd_component = MacroRegimeComponent.HAWKISH

            # Weak USD is treated as dovish.
            elif usd_strength.level.value == "weak":
                usd_component = MacroRegimeComponent.DOVISH

            # Neutral USD.
            else:
                usd_component = MacroRegimeComponent.NEUTRAL

            # Explain classification.
            usd_reason = (
                "USD strength is classified as "
                f"{usd_strength.level.value}."
            )

        # Add USD contribution.
        contributions.append(
            MacroRegimeContribution(
                source="usd_strength",
                component=usd_component,
                weight=self.weights["usd_strength"],
                contribution=(
                    self.weights["usd_strength"]
                    * self._component_score(
                        usd_component
                    )
                ),
                reason=usd_reason,
            )
        )

        # --------------------------------------------------------------
        # DXY
        # --------------------------------------------------------------

        # Determine DXY component.
        if dxy is None:
            dxy_component = MacroRegimeComponent.UNKNOWN
            dxy_reason = "DXY data is unavailable."
        else:
            # Strong/bullish DXY is hawkish.
            if dxy.level.value in {
                "strong",
                "bullish",
            }:
                dxy_component = MacroRegimeComponent.HAWKISH

            # Weak/bearish DXY is dovish.
            elif dxy.level.value in {
                "weak",
                "bearish",
            }:
                dxy_component = MacroRegimeComponent.DOVISH

            # Neutral DXY.
            else:
                dxy_component = MacroRegimeComponent.NEUTRAL

            # Explain classification.
            dxy_reason = (
                f"DXY is classified as {dxy.level.value}."
            )

        # Add DXY contribution.
        contributions.append(
            MacroRegimeContribution(
                source="dxy",
                component=dxy_component,
                weight=self.weights["dxy"],
                contribution=(
                    self.weights["dxy"]
                    * self._component_score(
                        dxy_component
                    )
                ),
                reason=dxy_reason,
            )
        )

        # --------------------------------------------------------------
        # TREASURY YIELDS
        # --------------------------------------------------------------

        # Collect Treasury directions.
        treasury_components: list[
            MacroRegimeComponent
        ] = []

        # Inspect every Treasury maturity.
        for assessment in treasury_yields.values():

            # Rising yields are hawkish.
            if assessment.direction.value == "rising":
                treasury_components.append(
                    MacroRegimeComponent.HAWKISH
                )

            # Falling yields are dovish.
            elif assessment.direction.value == "falling":
                treasury_components.append(
                    MacroRegimeComponent.DOVISH
                )

            # Stable yields are neutral.
            elif assessment.direction.value == "stable":
                treasury_components.append(
                    MacroRegimeComponent.NEUTRAL
                )

        # No Treasury information.
        if not treasury_components:
            treasury_component = MacroRegimeComponent.UNKNOWN

        else:
            # Count hawkish assessments.
            hawkish_count = treasury_components.count(
                MacroRegimeComponent.HAWKISH
            )

            # Count dovish assessments.
            dovish_count = treasury_components.count(
                MacroRegimeComponent.DOVISH
            )

            # Hawkish majority.
            if hawkish_count > dovish_count:
                treasury_component = (
                    MacroRegimeComponent.HAWKISH
                )

            # Dovish majority.
            elif dovish_count > hawkish_count:
                treasury_component = (
                    MacroRegimeComponent.DOVISH
                )

            # Equal evidence.
            else:
                treasury_component = (
                    MacroRegimeComponent.NEUTRAL
                )

        # Add Treasury contribution.
        contributions.append(
            MacroRegimeContribution(
                source="treasury_yields",
                component=treasury_component,
                weight=self.weights["treasury_yields"],
                contribution=(
                    self.weights["treasury_yields"]
                    * self._component_score(
                        treasury_component
                    )
                ),
                reason=(
                    "Treasury yields are interpreted as "
                    f"{treasury_component.value}."
                ),
            )
        )

        # --------------------------------------------------------------
        # FED RATE
        # --------------------------------------------------------------

        # Add Fed contribution.
        contributions.append(
            MacroRegimeContribution(
                source="fed_rate",
                component=monetary_component,
                weight=self.weights["fed_rate"],
                contribution=(
                    self.weights["fed_rate"]
                    * self._component_score(
                        monetary_component
                    )
                ),
                reason=(
                    "Fed-rate environment is interpreted as "
                    f"{monetary_component.value}."
                ),
            )
        )

        # --------------------------------------------------------------
        # INFLATION
        # --------------------------------------------------------------

        # Add inflation contribution.
        contributions.append(
            MacroRegimeContribution(
                source="inflation",
                component=inflation_component,
                weight=self.weights["inflation"],
                contribution=(
                    self.weights["inflation"]
                    * self._component_score(
                        inflation_component
                    )
                ),
                reason=(
                    "Inflation environment is classified as "
                    f"{inflation_component.value}."
                ),
            )
        )

        # --------------------------------------------------------------
        # EMPLOYMENT
        # --------------------------------------------------------------

        # Add employment contribution.
        contributions.append(
            MacroRegimeContribution(
                source="employment",
                component=employment_component,
                weight=self.weights["employment"],
                contribution=(
                    self.weights["employment"]
                    * self._component_score(
                        employment_component
                    )
                ),
                reason=(
                    "Employment environment is classified as "
                    f"{employment_component.value}."
                ),
            )
        )

        # --------------------------------------------------------------
        # RISK SENTIMENT
        # --------------------------------------------------------------

        # Add risk-sentiment contribution.
        contributions.append(
            MacroRegimeContribution(
                source="risk_sentiment",
                component=risk_component,
                weight=self.weights["risk_sentiment"],
                contribution=(
                    self.weights["risk_sentiment"]
                    * self._component_score(
                        risk_component
                    )
                ),
                reason=(
                    (
                        "Risk sentiment is classified as "
                        f"{risk_sentiment.sentiment.value}."
                    )
                    if risk_sentiment is not None
                    else "Risk sentiment data is unavailable."
                ),
            )
        )

        # Calculate total configured weight.
        total_weight = sum(
            float(weight)
            for weight in self.weights.values()
        )

        # Keep only usable weighted components.
        usable_contributions = [
            contribution
            for contribution in contributions
            if (
                contribution.component
                != MacroRegimeComponent.UNKNOWN
                and contribution.weight > 0
            )
        ]

        # Calculate represented weight.
        used_weight = sum(
            contribution.weight
            for contribution in usable_contributions
        )

        # Calculate weighted coverage.
        coverage = (
            used_weight / total_weight
            if total_weight > 0
            else 0.0
        )

        # Calculate aggregate normalized score.
        raw_score = (
            (
                sum(
                    contribution.contribution
                    for contribution in usable_contributions
                )
                / total_weight
            )
            * 100.0
            if total_weight > 0
            else 0.0
        )

        # Remove floating-point residue.
        score = round(raw_score, 10)

        # Convert coverage to confidence.
        confidence = round(
            max(
                0.0,
                min(
                    100.0,
                    coverage * 100.0,
                ),
            ),
            10,
        )

        # Determine whether enough data exists.
        sufficient_data = (
            coverage >= self.min_coverage
        )

        # Classify only with sufficient coverage.
        if sufficient_data:
            regime = self._classify(
                score,
                inflation_component=inflation_component,
                monetary_component=monetary_component,
                risk_component=risk_component,
                employment_component=employment_component,
            )
        else:
            regime = MacroRegime.UNKNOWN

        # Build deterministic explanations.
        reasons: list[str] = []

        # Explain component coverage.
        reasons.append(
            f"Used {len(usable_contributions)} of "
            f"{len(contributions)} macro regime components."
        )

        # Explain weighted coverage.
        reasons.append(
            f"Weighted macro coverage is {coverage:.4f}."
        )

        # Explain aggregate score.
        reasons.append(
            f"Macro regime score is {score:.4f}."
        )

        # Explain insufficient data.
        if not sufficient_data:
            reasons.append(
                "Macro coverage is below the configured minimum; "
                "regime is UNKNOWN."
            )

        # Explain final regime.
        reasons.append(
            f"Macro regime classified as {regime.value}."
        )

        # Explicitly separate regime from trade decision.
        reasons.append(
            "Macro regime is contextual information and does not "
            "authorize or reject a trade."
        )

        # Return immutable assessment.
        return MacroRegimeAssessment(
            regime=regime,
            confidence=confidence,
            sufficient_data=sufficient_data,
            score=score,
            components_used=len(
                usable_contributions
            ),
            total_weight=total_weight,
            used_weight=used_weight,
            usd_strength=usd_strength,
            dxy=dxy,
            treasury_yields=treasury_yields,
            fed_rate=fed_rate,
            inflation=inflation,
            employment=employment,
            risk_sentiment=risk_sentiment,
            contributions=tuple(contributions),
            decision_timestamp=decision_timestamp,
            reasons=tuple(reasons),
        )

    def analyze_xauusd(
        self,
        observations: list[MacroObservation],
        *,
        decision_timestamp: datetime,
    ) -> MacroRegimeAssessment:
        """
        Analyze macro regime for XAUUSD context.

        This returns macro context only.
        It does not generate a trade signal.
        """

        # Reuse the common historical-safe analysis path.
        return self.analyze(
            observations,
            decision_timestamp=decision_timestamp,
        )