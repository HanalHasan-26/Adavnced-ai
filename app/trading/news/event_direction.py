from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.trading.news.economic_event import EconomicEvent


class EventDirectionError(ValueError):
    """Raised when event-direction analysis input is invalid."""


class Direction(str, Enum):
    """Directional interpretation of an economic event."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


class DirectionTarget(str, Enum):
    """Asset/currency whose directional bias is being evaluated."""

    CURRENCY = "CURRENCY"
    XAUUSD = "XAUUSD"
    UNKNOWN = "UNKNOWN"


class SurpriseDirection(str, Enum):
    """Direction of the actual-versus-forecast surprise."""

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class EventDirectionReasonType(str, Enum):
    """Reason categories for directional event analysis."""

    NO_ACTUAL = "NO_ACTUAL"
    NO_FORECAST = "NO_FORECAST"
    POSITIVE_SURPRISE = "POSITIVE_SURPRISE"
    NEGATIVE_SURPRISE = "NEGATIVE_SURPRISE"
    NO_SURPRISE = "NO_SURPRISE"
    USD_POSITIVE = "USD_POSITIVE"
    USD_NEGATIVE = "USD_NEGATIVE"
    XAUUSD_BULLISH = "XAUUSD_BULLISH"
    XAUUSD_BEARISH = "XAUUSD_BEARISH"
    NEUTRAL_EVENT = "NEUTRAL_EVENT"
    UNKNOWN_EVENT = "UNKNOWN_EVENT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True, slots=True)
class EventDirectionReason:
    """Explanation for a directional event assessment."""

    reason_type: EventDirectionReasonType
    message: str


@dataclass(frozen=True, slots=True)
class EventDirectionResult:
    """
    Deterministic directional interpretation of an economic event.

    The result separates:
    - surprise direction,
    - currency direction,
    - instrument direction,
    - confidence.

    It does not make a trade decision.
    """

    timestamp: object
    event_name: str
    currency: str
    target: DirectionTarget

    surprise: float | None
    surprise_direction: SurpriseDirection

    currency_direction: Direction
    instrument_direction: Direction

    confidence: float

    reasons: tuple[EventDirectionReason, ...]

    sufficient_data: bool

    @property
    def is_bullish(self) -> bool:
        return self.instrument_direction == Direction.BULLISH

    @property
    def is_bearish(self) -> bool:
        return self.instrument_direction == Direction.BEARISH

    @property
    def is_neutral(self) -> bool:
        return self.instrument_direction == Direction.NEUTRAL

    @property
    def is_unknown(self) -> bool:
        return self.instrument_direction == Direction.UNKNOWN


class EventDirectionEngine:
    """
    Deterministic economic-event directional analysis.

    Current supported directional model:

    For USD events:
        positive surprise -> USD bullish
        negative surprise -> USD bearish

    For XAUUSD:
        USD bullish -> XAUUSD bearish
        USD bearish -> XAUUSD bullish

    For non-USD currencies:
        the engine can determine currency direction,
        but it does not automatically convert that into
        XAUUSD direction.

    This prevents unsupported assumptions about how
    unrelated economic releases affect gold.

    Important:
        Actual and forecast are required for directional
        surprise analysis.

        Previous alone is not treated as the market expectation.
    """

    def analyze(
        self,
        event: EconomicEvent,
        *,
        target: DirectionTarget = DirectionTarget.XAUUSD,
    ) -> EventDirectionResult:
        self._validate_inputs(
            event=event,
            target=target,
        )

        if event.actual is None:
            return self._unknown_result(
                event=event,
                target=target,
                reason_type=EventDirectionReasonType.NO_ACTUAL,
                message=(
                    "Actual value is not available; "
                    "directional surprise cannot be determined."
                ),
            )

        if event.forecast is None:
            return self._unknown_result(
                event=event,
                target=target,
                reason_type=EventDirectionReasonType.NO_FORECAST,
                message=(
                    "Forecast value is not available; "
                    "directional surprise cannot be determined."
                ),
            )

        surprise = event.actual - event.forecast

        if surprise > 0:
            surprise_direction = SurpriseDirection.POSITIVE
        elif surprise < 0:
            surprise_direction = SurpriseDirection.NEGATIVE
        else:
            surprise_direction = SurpriseDirection.NONE

        currency_direction = self._currency_direction(
            event=event,
            surprise_direction=surprise_direction,
        )

        instrument_direction = self._instrument_direction(
            event=event,
            target=target,
            currency_direction=currency_direction,
        )

        confidence = self._calculate_confidence(
            event=event,
            target=target,
            surprise_direction=surprise_direction,
            instrument_direction=instrument_direction,
        )

        reasons = self._build_reasons(
            event=event,
            target=target,
            surprise_direction=surprise_direction,
            currency_direction=currency_direction,
            instrument_direction=instrument_direction,
        )

        return EventDirectionResult(
            timestamp=event.timestamp,
            event_name=event.name,
            currency=event.currency.upper(),
            target=target,
            surprise=surprise,
            surprise_direction=surprise_direction,
            currency_direction=currency_direction,
            instrument_direction=instrument_direction,
            confidence=confidence,
            reasons=tuple(reasons),
            sufficient_data=True,
        )
    def analyze_at(
        self,
        event: EconomicEvent,
        *,
        decision_timestamp,
        target: DirectionTarget = DirectionTarget.XAUUSD,
    ) -> EventDirectionResult:
        """
        Analyze an economic event using only information available
        at the supplied historical decision timestamp.

        The event timestamp represents the scheduled/release time.

        Before the release time:
        - actual must not be used.
        - directional surprise must remain unknown.
        - the engine must not leak future information.

        At or after the release time:
        - the stored actual value may be used if available.

        This method is specifically intended for historical
        backtesting and walk-forward evaluation.
        """

        # Validate that the event itself and requested target are valid.
        self._validate_inputs(
            event=event,
            target=target,
        )

        # Validate the historical decision timestamp explicitly.
        from datetime import datetime

        if not isinstance(
            decision_timestamp,
            datetime,
        ):
            raise EventDirectionError(
                "decision_timestamp must be a datetime."
            )

        # Require matching timezone semantics so that the comparison
        # cannot silently mix naive and timezone-aware timestamps.
        if (
            event.timestamp.tzinfo
            != decision_timestamp.tzinfo
        ):
            raise EventDirectionError(
                "event and decision timestamps must use the same timezone."
            )

        # If the decision occurs before the economic release,
        # the actual value is future information and must be hidden.
        if decision_timestamp < event.timestamp:
            return self._unknown_result(
                event=event,
                target=target,
                reason_type=EventDirectionReasonType.NO_ACTUAL,
                message=(
                    "The decision occurs before the event release; "
                    "actual information is not yet available."
                ),
            )

        # Once the release time has arrived, use the normal
        # directional analysis because the actual may legitimately
        # be available to the historical decision.
        return self.analyze(
            event,
            target=target,
        )
    def analyze_xauusd_at(
        self,
        event: EconomicEvent,
        *,
        decision_timestamp,
    ) -> EventDirectionResult:
        """
        Analyze XAUUSD event direction using only information
        available at the historical decision timestamp.
        """

        # Delegate to the historical-safe analysis method while
        # explicitly selecting XAUUSD as the target instrument.
        return self.analyze_at(
            event,
            decision_timestamp=decision_timestamp,
            target=DirectionTarget.XAUUSD,
        )
    def analyze_xauusd(
        self,
        event: EconomicEvent,
    ) -> EventDirectionResult:
        """Analyze the event specifically for XAUUSD."""

        return self.analyze(
            event,
            target=DirectionTarget.XAUUSD,
        )

    def analyze_currency(
        self,
        event: EconomicEvent,
    ) -> EventDirectionResult:
        """Analyze the event from the currency perspective."""

        return self.analyze(
            event,
            target=DirectionTarget.CURRENCY,
        )

    @staticmethod
    def _currency_direction(
        *,
        event: EconomicEvent,
        surprise_direction: SurpriseDirection,
    ) -> Direction:
        currency = event.currency.upper()

        if currency != "USD":
            return Direction.UNKNOWN

        if surprise_direction == SurpriseDirection.POSITIVE:
            return Direction.BULLISH

        if surprise_direction == SurpriseDirection.NEGATIVE:
            return Direction.BEARISH

        if surprise_direction == SurpriseDirection.NONE:
            return Direction.NEUTRAL

        return Direction.UNKNOWN

    @staticmethod
    def _instrument_direction(
        *,
        event: EconomicEvent,
        target: DirectionTarget,
        currency_direction: Direction,
    ) -> Direction:
        if target == DirectionTarget.CURRENCY:
            return currency_direction

        if target == DirectionTarget.XAUUSD:
            if event.currency.upper() != "USD":
                return Direction.UNKNOWN

            if currency_direction == Direction.BULLISH:
                return Direction.BEARISH

            if currency_direction == Direction.BEARISH:
                return Direction.BULLISH

            if currency_direction == Direction.NEUTRAL:
                return Direction.NEUTRAL

            return Direction.UNKNOWN

        return Direction.UNKNOWN

    @staticmethod
    def _calculate_confidence(
        *,
        event: EconomicEvent,
        target: DirectionTarget,
        surprise_direction: SurpriseDirection,
        instrument_direction: Direction,
    ) -> float:
        if (
            event.actual is None
            or event.forecast is None
        ):
            return 0.0

        if surprise_direction == SurpriseDirection.NONE:
            return 50.0

        if (
            target == DirectionTarget.XAUUSD
            and event.currency.upper() == "USD"
            and instrument_direction
            in {
                Direction.BULLISH,
                Direction.BEARISH,
            }
        ):
            return 80.0

        if target == DirectionTarget.CURRENCY:
            return 80.0

        return 30.0

    @staticmethod
    def _build_reasons(
        *,
        event: EconomicEvent,
        target: DirectionTarget,
        surprise_direction: SurpriseDirection,
        currency_direction: Direction,
        instrument_direction: Direction,
    ) -> list[EventDirectionReason]:
        reasons: list[EventDirectionReason] = []

        if surprise_direction == SurpriseDirection.POSITIVE:
            reasons.append(
                EventDirectionReason(
                    EventDirectionReasonType.POSITIVE_SURPRISE,
                    (
                        f"Actual value for {event.name} "
                        "is above the forecast."
                    ),
                )
            )

        elif surprise_direction == SurpriseDirection.NEGATIVE:
            reasons.append(
                EventDirectionReason(
                    EventDirectionReasonType.NEGATIVE_SURPRISE,
                    (
                        f"Actual value for {event.name} "
                        "is below the forecast."
                    ),
                )
            )

        elif surprise_direction == SurpriseDirection.NONE:
            reasons.append(
                EventDirectionReason(
                    EventDirectionReasonType.NO_SURPRISE,
                    (
                        f"Actual value for {event.name} "
                        "matches the forecast."
                    ),
                )
            )

        else:
            reasons.append(
                EventDirectionReason(
                    EventDirectionReasonType.UNKNOWN_EVENT,
                    (
                        f"Directional surprise for "
                        f"{event.name} is unknown."
                    ),
                )
            )

        if currency_direction == Direction.BULLISH:
            reasons.append(
                EventDirectionReason(
                    EventDirectionReasonType.USD_POSITIVE,
                    (
                        f"{event.name} is interpreted as "
                        "USD-positive based on the surprise."
                    ),
                )
            )

        elif currency_direction == Direction.BEARISH:
            reasons.append(
                EventDirectionReason(
                    EventDirectionReasonType.USD_NEGATIVE,
                    (
                        f"{event.name} is interpreted as "
                        "USD-negative based on the surprise."
                    ),
                )
            )

        if instrument_direction == Direction.BULLISH:
            reasons.append(
                EventDirectionReason(
                    EventDirectionReasonType.XAUUSD_BULLISH,
                    (
                        "The USD interpretation implies "
                        "bullish pressure for XAUUSD."
                    ),
                )
            )

        elif instrument_direction == Direction.BEARISH:
            reasons.append(
                EventDirectionReason(
                    EventDirectionReasonType.XAUUSD_BEARISH,
                    (
                        "The USD interpretation implies "
                        "bearish pressure for XAUUSD."
                    ),
                )
            )

        elif instrument_direction == Direction.NEUTRAL:
            reasons.append(
                EventDirectionReason(
                    EventDirectionReasonType.NEUTRAL_EVENT,
                    (
                        "The event produces no directional "
                        "surprise."
                    ),
                )
            )

        elif instrument_direction == Direction.UNKNOWN:
            reasons.append(
                EventDirectionReason(
                    EventDirectionReasonType.UNKNOWN_EVENT,
                    (
                        "The event cannot be reliably mapped "
                        "to the requested instrument."
                    ),
                )
            )

        return reasons

    @staticmethod
    def _unknown_result(
        *,
        event: EconomicEvent,
        target: DirectionTarget,
        reason_type: EventDirectionReasonType,
        message: str,
    ) -> EventDirectionResult:
        return EventDirectionResult(
            timestamp=event.timestamp,
            event_name=event.name,
            currency=event.currency.upper(),
            target=target,
            surprise=None,
            surprise_direction=SurpriseDirection.UNKNOWN,
            currency_direction=Direction.UNKNOWN,
            instrument_direction=Direction.UNKNOWN,
            confidence=0.0,
            reasons=(
                EventDirectionReason(
                    reason_type,
                    message,
                ),
                EventDirectionReason(
                    EventDirectionReasonType.INSUFFICIENT_DATA,
                    "Insufficient data for directional analysis.",
                ),
            ),
            sufficient_data=False,
        )

    @staticmethod
    def _validate_inputs(
        *,
        event: EconomicEvent,
        target: DirectionTarget,
    ) -> None:
        if not isinstance(
            event,
            EconomicEvent,
        ):
            raise EventDirectionError(
                "event must be an EconomicEvent."
            )

        if not isinstance(
            target,
            DirectionTarget,
        ):
            raise EventDirectionError(
                "target must be a DirectionTarget."
            )