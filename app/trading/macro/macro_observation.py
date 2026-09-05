from __future__ import annotations

# Import dataclass support for immutable macro observations.
from dataclasses import dataclass

# Import datetime so every observation has a precise timestamp.
from datetime import datetime

# Import Enum for strongly typed macro indicators and directions.
from enum import Enum

# Import math for finite-number validation.
import math


class MacroObservationError(ValueError):
    """
    Raised when a macroeconomic observation is invalid.
    """


class MacroIndicator(str, Enum):
    """
    Supported macroeconomic indicators.

    These are deliberately data categories rather than trade signals.
    """

    # US Dollar Index.
    DXY = "DXY"

    # US 2-year Treasury yield.
    US_2Y_YIELD = "US_2Y_YIELD"

    # US 5-year Treasury yield.
    US_5Y_YIELD = "US_5Y_YIELD"

    # US 10-year Treasury yield.
    US_10Y_YIELD = "US_10Y_YIELD"

    # US 30-year Treasury yield.
    US_30Y_YIELD = "US_30Y_YIELD"

    # Federal Funds target rate.
    FED_FUNDS_RATE = "FED_FUNDS_RATE"

    # CPI inflation.
    CPI = "CPI"

    # Core CPI inflation.
    CORE_CPI = "CORE_CPI"

    # PCE inflation.
    PCE = "PCE"

    # Core PCE inflation.
    CORE_PCE = "CORE_PCE"

    # Nonfarm Payrolls.
    NFP = "NFP"

    # Unemployment rate.
    UNEMPLOYMENT_RATE = "UNEMPLOYMENT_RATE"

    # GDP growth.
    GDP = "GDP"

    # Generic macroeconomic observation.
    OTHER = "OTHER"


class MacroDirection(str, Enum):
    """
    Direction of a macro observation relative to its reference.

    This is intentionally generic. It does not directly mean
    bullish or bearish for XAUUSD.
    """

    RISING = "RISING"
    FALLING = "FALLING"
    STABLE = "STABLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class MacroObservation:
    """
    Immutable representation of one macroeconomic observation.

    The observation stores the raw value and optional reference
    information. Interpretation belongs to later macro-intelligence
    engines.
    """

    # Time at which this observation became available to the system.
    timestamp: datetime

    # Macro indicator represented by this observation.
    indicator: MacroIndicator

    # Observed numerical value.
    value: float

    # Optional previous observation.
    previous: float | None = None

    # Optional market/economic expectation.
    forecast: float | None = None

    # Data source identifier.
    source: str = "unknown"

    # Optional explicit direction relative to the previous observation.
    direction: MacroDirection = MacroDirection.UNKNOWN

    @property
    def has_previous(self) -> bool:
        """
        Return whether a previous value is available.
        """

        # A previous value is available when it is not None.
        return self.previous is not None

    @property
    def has_forecast(self) -> bool:
        """
        Return whether a forecast is available.
        """

        # A forecast is available when it is not None.
        return self.forecast is not None

    @property
    def surprise(self) -> float | None:
        """
        Calculate actual/observed value minus forecast.

        Returns None when no forecast exists.
        """

        # A surprise cannot be calculated without a forecast.
        if self.forecast is None:
            return None

        # Return the deterministic observed-minus-forecast difference.
        return self.value - self.forecast

    @property
    def change_from_previous(self) -> float | None:
        """
        Calculate the absolute change from the previous observation.

        Returns None when no previous value exists.
        """

        # A change cannot be calculated without a previous value.
        if self.previous is None:
            return None

        # Return current value minus previous value.
        return self.value - self.previous

    @property
    def has_surprise(self) -> bool:
        """
        Return whether a forecast-based surprise can be calculated.
        """

        # A surprise exists when the calculated value is available.
        return self.surprise is not None

    @property
    def has_change(self) -> bool:
        """
        Return whether a previous-value change can be calculated.
        """

        # A change exists when a previous value is available.
        return self.change_from_previous is not None

    def to_dict(self) -> dict[str, object]:
        """
        Serialize the observation into a JSON-compatible dictionary.
        """

        # Return only primitive/serializable values.
        return {
            "timestamp": self.timestamp.isoformat(),
            "indicator": self.indicator.value,
            "value": self.value,
            "previous": self.previous,
            "forecast": self.forecast,
            "source": self.source,
            "direction": self.direction.value,
        }

    def __post_init__(self) -> None:
        """
        Validate the observation after construction.
        """

        # Timestamp must be a datetime.
        if not isinstance(self.timestamp, datetime):
            raise MacroObservationError(
                "timestamp must be a datetime."
            )

        # Indicator must be a supported MacroIndicator.
        if not isinstance(self.indicator, MacroIndicator):
            raise MacroObservationError(
                "indicator must be a MacroIndicator."
            )

        # Value must be numeric and finite.
        if not isinstance(self.value, (int, float)):
            raise MacroObservationError(
                "value must be numeric."
            )

        # Boolean values are technically integers in Python, so reject them.
        if isinstance(self.value, bool):
            raise MacroObservationError(
                "value must not be boolean."
            )

        # Reject NaN and infinity.
        if not math.isfinite(float(self.value)):
            raise MacroObservationError(
                "value must be finite."
            )

        # Validate optional previous value.
        if self.previous is not None:
            if not isinstance(self.previous, (int, float)):
                raise MacroObservationError(
                    "previous must be numeric or None."
                )

            # Reject boolean previous values.
            if isinstance(self.previous, bool):
                raise MacroObservationError(
                    "previous must not be boolean."
                )

            # Reject non-finite previous values.
            if not math.isfinite(float(self.previous)):
                raise MacroObservationError(
                    "previous must be finite."
                )

        # Validate optional forecast value.
        if self.forecast is not None:
            if not isinstance(self.forecast, (int, float)):
                raise MacroObservationError(
                    "forecast must be numeric or None."
                )

            # Reject boolean forecast values.
            if isinstance(self.forecast, bool):
                raise MacroObservationError(
                    "forecast must not be boolean."
                )

            # Reject non-finite forecast values.
            if not math.isfinite(float(self.forecast)):
                raise MacroObservationError(
                    "forecast must be finite."
                )

        # Source must identify where the observation came from.
        if (
            not isinstance(self.source, str)
            or not self.source.strip()
        ):
            raise MacroObservationError(
                "source must be a non-empty string."
            )

        # Direction must use the supported enum.
        if not isinstance(self.direction, MacroDirection):
            raise MacroObservationError(
                "direction must be a MacroDirection."
            )