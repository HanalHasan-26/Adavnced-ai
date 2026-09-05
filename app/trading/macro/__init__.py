"""
Macro intelligence package.

This package contains deterministic macroeconomic analysis components
used by the Advanced AI trading engine.

Architecture:
- Macro observations provide normalized economic data.
- Individual intelligence modules analyze specific macro factors.
- Macro regime intelligence combines those factors.
- XAUUSD macro impact intelligence translates macro conditions
  into gold-specific directional pressure.

Important:
- The macro package does not execute trades.
- The macro package does not directly authorize trades.
- The macro package does not depend on an LLM.
- Historical/no-lookahead safety is handled by the intelligence layers.
"""


# ---------------------------------------------------------------------------
# CORE MACRO OBSERVATION
# ---------------------------------------------------------------------------

# Export the normalized macro observation model and related enums/errors.
from app.trading.macro.macro_observation import (
    MacroDirection,
    MacroIndicator,
    MacroObservation,
    MacroObservationError,
)


# ---------------------------------------------------------------------------
# USD STRENGTH INTELLIGENCE
# ---------------------------------------------------------------------------

# Export deterministic USD-strength analysis.
from app.trading.macro.usd_strength_intelligence import (
    USDStrengthAssessment,
    USDStrengthContribution,
    USDStrengthIntelligence,
    USDStrengthIntelligenceError,
    USDStrengthLevel,
)


# ---------------------------------------------------------------------------
# DXY INTELLIGENCE
# ---------------------------------------------------------------------------

# Export deterministic DXY analysis.
from app.trading.macro.dxy_intelligence import (
    DXYAssessment,
    DXYIntelligence,
    DXYIntelligenceError,
    DXYLevel,
)


# ---------------------------------------------------------------------------
# TREASURY YIELD INTELLIGENCE
# ---------------------------------------------------------------------------

# Export Treasury-yield analysis.
from app.trading.macro.treasury_yield_intelligence import (
    TreasuryYieldAssessment,
    TreasuryYieldIntelligence,
    TreasuryYieldIntelligenceError,
    TreasuryYieldLevel,
)


# ---------------------------------------------------------------------------
# FEDERAL RESERVE RATE INTELLIGENCE
# ---------------------------------------------------------------------------

# Export Federal Reserve policy-rate analysis.
from app.trading.macro.fed_rate_intelligence import (
    FedRateAssessment,
    FedRateIntelligence,
    FedRateIntelligenceError,
    FedRateLevel,
)


# ---------------------------------------------------------------------------
# INFLATION INTELLIGENCE
# ---------------------------------------------------------------------------

# Export U.S. inflation analysis.
from app.trading.macro.inflation_intelligence import (
    InflationAssessment,
    InflationIntelligence,
    InflationIntelligenceError,
    InflationLevel,
    InflationSurpriseLevel,
)


# ---------------------------------------------------------------------------
# EMPLOYMENT INTELLIGENCE
# ---------------------------------------------------------------------------

# Export U.S. employment analysis.
from app.trading.macro.employment_intelligence import (
    EmploymentAssessment,
    EmploymentIntelligence,
    EmploymentIntelligenceError,
    EmploymentLevel,
    EmploymentSurpriseLevel,
)


# ---------------------------------------------------------------------------
# RISK SENTIMENT INTELLIGENCE
# ---------------------------------------------------------------------------

# Export broad risk-on/risk-off analysis.
from app.trading.macro.risk_sentiment_intelligence import (
    RiskSentiment,
    RiskSentimentAssessment,
    RiskSentimentComponent,
    RiskSentimentContribution,
    RiskSentimentIntelligence,
    RiskSentimentIntelligenceError,
)


# ---------------------------------------------------------------------------
# MACRO REGIME INTELLIGENCE
# ---------------------------------------------------------------------------

# Export the higher-level macro regime aggregation layer.
from app.trading.macro.macro_regime_intelligence import (
    MacroRegime,
    MacroRegimeAssessment,
    MacroRegimeComponent,
    MacroRegimeContribution,
    MacroRegimeIntelligence,
    MacroRegimeIntelligenceError,
)


# ---------------------------------------------------------------------------
# XAUUSD MACRO IMPACT INTELLIGENCE
# ---------------------------------------------------------------------------

# Export XAUUSD-specific macro impact analysis.
from app.trading.macro.xauusd_macro_impact_intelligence import (
    XAUUSDMacroBias,
    XAUUSDMacroComponent,
    XAUUSDMacroImpactAssessment,
    XAUUSDMacroImpactContribution,
    XAUUSDMacroImpactIntelligence,
    XAUUSDMacroImpactIntelligenceError,
)


# ---------------------------------------------------------------------------
# PUBLIC PACKAGE API
# ---------------------------------------------------------------------------

# Define everything intentionally exposed by app.trading.macro.
__all__ = [

    # Core macro observation models.
    "MacroDirection",
    "MacroIndicator",
    "MacroObservation",
    "MacroObservationError",

    # USD strength intelligence.
    "USDStrengthAssessment",
    "USDStrengthContribution",
    "USDStrengthIntelligence",
    "USDStrengthIntelligenceError",
    "USDStrengthLevel",

    # DXY intelligence.
    "DXYAssessment",
    "DXYIntelligence",
    "DXYIntelligenceError",
    "DXYLevel",

    # Treasury yield intelligence.
    "TreasuryYieldAssessment",
    "TreasuryYieldIntelligence",
    "TreasuryYieldIntelligenceError",
    "TreasuryYieldLevel",

    # Federal Reserve rate intelligence.
    "FedRateAssessment",
    "FedRateIntelligence",
    "FedRateIntelligenceError",
    "FedRateLevel",

    # Inflation intelligence.
    "InflationAssessment",
    "InflationIntelligence",
    "InflationIntelligenceError",
    "InflationLevel",
    "InflationSurpriseLevel",

    # Employment intelligence.
    "EmploymentAssessment",
    "EmploymentIntelligence",
    "EmploymentIntelligenceError",
    "EmploymentLevel",
    "EmploymentSurpriseLevel",

    # Risk sentiment intelligence.
    "RiskSentiment",
    "RiskSentimentAssessment",
    "RiskSentimentComponent",
    "RiskSentimentContribution",
    "RiskSentimentIntelligence",
    "RiskSentimentIntelligenceError",

    # Macro regime intelligence.
    "MacroRegime",
    "MacroRegimeAssessment",
    "MacroRegimeComponent",
    "MacroRegimeContribution",
    "MacroRegimeIntelligence",
    "MacroRegimeIntelligenceError",

    # XAUUSD macro impact intelligence.
    "XAUUSDMacroBias",
    "XAUUSDMacroComponent",
    "XAUUSDMacroImpactAssessment",
    "XAUUSDMacroImpactContribution",
    "XAUUSDMacroImpactIntelligence",
    "XAUUSDMacroImpactIntelligenceError",
]