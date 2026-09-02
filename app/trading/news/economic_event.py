from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import math


class EconomicEventError(ValueError):
    """
    Raised when an economic event is invalid.
    """


class EventImpact(str, Enum):
    """
    Expected market impact of an economic event.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class EventActualStatus(str, Enum):
    """
    Availability state of an event's actual value.
    """

    PENDING = "PENDING"
    AVAILABLE = "AVAILABLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class EconomicEvent:
    """
    Structured representation of an economic-calendar event.

    The model intentionally stores the raw numerical values rather
    than attempting to determine bullish/bearish direction.

    Directional interpretation belongs to a later event-analysis
    layer.
    """

    timestamp: datetime
    name: str
    currency: str
    impact: EventImpact

    previous: float | None = None
    forecast: float | None = None
    actual: float | None = None

    source: str = "unknown"

    @property
    def actual_status(self) -> EventActualStatus:
        if self.actual is not None:
            return EventActualStatus.AVAILABLE

        if self.impact == EventImpact.UNKNOWN:
            return EventActualStatus.UNKNOWN

        return EventActualStatus.PENDING

    @property
    def has_forecast(self) -> bool:
        return self.forecast is not None

    @property
    def has_actual(self) -> bool:
        return self.actual is not None

    @property
    def surprise(self) -> float | None:
        """
        Actual minus forecast.

        Returns None when either value is unavailable.
        """

        if (
            self.actual is None
            or self.forecast is None
        ):
            return None

        return self.actual - self.forecast

    @property
    def has_surprise(self) -> bool:
        return self.surprise is not None

    @property
    def is_high_impact(self) -> bool:
        return self.impact == EventImpact.HIGH

    @property
    def is_usd_event(self) -> bool:
        return self.currency.upper() == "USD"

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "name": self.name,
            "currency": self.currency,
            "impact": self.impact.value,
            "previous": self.previous,
            "forecast": self.forecast,
            "actual": self.actual,
            "actual_status": self.actual_status.value,
            "source": self.source,
        }

    def __post_init__(self) -> None:
        if not isinstance(
            self.timestamp,
            datetime,
        ):
            raise EconomicEventError(
                "timestamp must be a datetime."
            )

        if not isinstance(
            self.name,
            str,
        ) or not self.name.strip():
            raise EconomicEventError(
                "name must be a non-empty string."
            )

        if not isinstance(
            self.currency,
            str,
        ) or not self.currency.strip():
            raise EconomicEventError(
                "currency must be a non-empty string."
            )

        if not isinstance(
            self.impact,
            EventImpact,
        ):
            raise EconomicEventError(
                "impact must be an EventImpact."
            )

        if not isinstance(
            self.source,
            str,
        ) or not self.source.strip():
            raise EconomicEventError(
                "source must be a non-empty string."
            )

        for field_name, value in (
            ("previous", self.previous),
            ("forecast", self.forecast),
            ("actual", self.actual),
        ):
            if value is None:
                continue

            if not isinstance(
                value,
                (int, float),
            ):
                raise EconomicEventError(
                    f"{field_name} must be numeric or None."
                )

            if not math.isfinite(
                float(value)
            ):
                raise EconomicEventError(
                    f"{field_name} must be finite."
                )