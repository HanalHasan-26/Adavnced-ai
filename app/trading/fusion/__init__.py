"""
Trading signal-fusion package.

This package combines normalized evidence from existing trading subsystems,
then calibrates the resulting directional signal without duplicating the
underlying trading intelligence.
"""

from app.trading.fusion.signal_fusion_intelligence import (
    FusionDecision,
    FusionDirection,
    SignalFusionAssessment,
    SignalFusionEvidence,
    SignalFusionIntelligence,
    SignalFusionIntelligenceError,
)

from app.trading.fusion.fusion_input_adapters import (
    FusionInputAdapter,
    FusionInputAdapterError,
    FusionInputAdapterRegistry,
    create_default_fusion_adapters,
)

from app.trading.fusion.fusion_calibration import (
    CalibrationStrength,
    FusionCalibrationAssessment,
    FusionCalibrationError,
    FusionCalibrationIntelligence,
)

__all__ = [
    "FusionDecision",
    "FusionDirection",
    "SignalFusionAssessment",
    "SignalFusionEvidence",
    "SignalFusionIntelligence",
    "SignalFusionIntelligenceError",
    "FusionInputAdapter",
    "FusionInputAdapterError",
    "FusionInputAdapterRegistry",
    "create_default_fusion_adapters",
    "CalibrationStrength",
    "FusionCalibrationAssessment",
    "FusionCalibrationError",
    "FusionCalibrationIntelligence",
]