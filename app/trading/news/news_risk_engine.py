from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from app.trading.news.economic_event import (
    EconomicEvent,
    EventImpact,
)


class NewsRiskEngineError(ValueError):
    """
    Raised when news-risk analysis input or configuration is invalid.
    """


class NewsRiskLevel(str, Enum):
    """
    Overall economic-event risk around a trading decision.
    """

    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"
    UNKNOWN = "UNKNOWN"


class NewsRiskReasonType(str, Enum):
    """
    Reason categories contributing to a news-risk assessment.
    """

    NO_RELEVANT_EVENTS = "NO_RELEVANT_EVENTS"
    LOW_IMPACT_EVENT = "LOW_IMPACT_EVENT"
    MEDIUM_IMPACT_EVENT = "MEDIUM_IMPACT_EVENT"
    HIGH_IMPACT_EVENT = "HIGH_IMPACT_EVENT"
    USD_EVENT = "USD_EVENT"
    EVENT_IMMINENT = "EVENT_IMMINENT"
    EVENT_RECENT = "EVENT_RECENT"
    EVENT_PENDING = "EVENT_PENDING"
    ACTUAL_AVAILABLE = "ACTUAL_AVAILABLE"
    UNKNOWN_EVENT_DATA = "UNKNOWN_EVENT_DATA"
    MULTIPLE_EVENTS = "MULTIPLE_EVENTS"
    OUTSIDE_WINDOW = "OUTSIDE_WINDOW"


@dataclass(frozen=True, slots=True)
class NewsRiskReason:
    """
    Explanation for a news-risk assessment.
    """

    reason_type: NewsRiskReasonType
    message: str


@dataclass(frozen=True, slots=True)
class NewsRiskAssessment:
    """
    Result of evaluating economic-event risk around a decision time.
    """

    timestamp: datetime
    symbol: str
    risk_level: NewsRiskLevel
    risk_score: float
    relevant_events: tuple[EconomicEvent, ...]
    usd_event_count: int
    high_impact_event_count: int
    nearest_event_minutes: float | None
    reasons: tuple[NewsRiskReason, ...]
    sufficient_data: bool

    @property
    def has_relevant_events(self) -> bool:
        return bool(self.relevant_events)

    @property
    def has_high_impact_event(self) -> bool:
        return self.high_impact_event_count > 0

    @property
    def has_usd_event(self) -> bool:
        return self.usd_event_count > 0

    @property
    def is_high_risk(self) -> bool:
        return self.risk_level in {
            NewsRiskLevel.HIGH,
            NewsRiskLevel.EXTREME,
        }

    @property
    def is_unknown(self) -> bool:
        return self.risk_level == NewsRiskLevel.UNKNOWN


class NewsRiskEngine:
    """
    Deterministic economic-event risk evaluator.

    This engine measures event risk around a decision timestamp.
    It deliberately does not determine bullish/bearish direction.

    The default windows are:

    - imminent window: 30 minutes before/after an event
    - relevant window: 120 minutes before/after an event

    Risk is increased by:
    - high-impact events
    - USD events
    - event proximity
    - multiple relevant events

    Risk is not directly determined by whether an event is
    expected to be bullish or bearish.
    """

    SYMBOL_CURRENCIES = {
        "XAUUSD": {"USD"},
        "EURUSD": {"EUR", "USD"},
        "GBPUSD": {"GBP", "USD"},
        "USDJPY": {"USD", "JPY"},
        "USDCHF": {"USD", "CHF"},
        "AUDUSD": {"AUD", "USD"},
        "NZDUSD": {"NZD", "USD"},
        "USDCAD": {"USD", "CAD"},
    }

    IMPACT_SCORE = {
        EventImpact.LOW: 10.0,
        EventImpact.MEDIUM: 30.0,
        EventImpact.HIGH: 60.0,
        EventImpact.UNKNOWN: 0.0,
    }

    def __init__(
        self,
        *,
        relevant_window_minutes: float = 120.0,
        imminent_window_minutes: float = 30.0,
        high_risk_threshold: float = 60.0,
        extreme_risk_threshold: float = 85.0,
    ) -> None:
        self.relevant_window_minutes = relevant_window_minutes
        self.imminent_window_minutes = imminent_window_minutes
        self.high_risk_threshold = high_risk_threshold
        self.extreme_risk_threshold = extreme_risk_threshold

        self._validate_configuration()

    def assess(
        self,
        *,
        timestamp: datetime,
        symbol: str,
        events: list[EconomicEvent],
    ) -> NewsRiskAssessment:
        self._validate_inputs(
            timestamp=timestamp,
            symbol=symbol,
            events=events,
        )

        if not events:
            return NewsRiskAssessment(
                timestamp=timestamp,
                symbol=symbol,
                risk_level=NewsRiskLevel.NONE,
                risk_score=0.0,
                relevant_events=(),
                usd_event_count=0,
                high_impact_event_count=0,
                nearest_event_minutes=None,
                reasons=(
                    NewsRiskReason(
                        NewsRiskReasonType.NO_RELEVANT_EVENTS,
                        "No economic events were provided.",
                    ),
                ),
                sufficient_data=False,
            )

        relevant_currencies = self._get_relevant_currencies(
            symbol
        )

        relevant_events = self._find_relevant_events(
            timestamp=timestamp,
            events=events,
            currencies=relevant_currencies,
        )

        if not relevant_events:
            return NewsRiskAssessment(
                timestamp=timestamp,
                symbol=symbol,
                risk_level=NewsRiskLevel.NONE,
                risk_score=0.0,
                relevant_events=(),
                usd_event_count=0,
                high_impact_event_count=0,
                nearest_event_minutes=None,
                reasons=(
                    NewsRiskReason(
                        NewsRiskReasonType.NO_RELEVANT_EVENTS,
                        "No relevant economic events are within the configured window.",
                    ),
                    NewsRiskReason(
                        NewsRiskReasonType.OUTSIDE_WINDOW,
                        "Available events are outside the configured time window or unrelated to the symbol.",
                    ),
                ),
                sufficient_data=True,
            )

        ordered_events = tuple(
            sorted(
                relevant_events,
                key=lambda event: (
                    abs(
                        (
                            event.timestamp
                            - timestamp
                        ).total_seconds()
                    ),
                    event.timestamp,
                    event.name,
                ),
            )
        )

        usd_event_count = sum(
            1
            for event in ordered_events
            if event.is_usd_event
        )

        high_impact_event_count = sum(
            1
            for event in ordered_events
            if event.impact == EventImpact.HIGH
        )

        nearest_event_minutes = (
            abs(
                (
                    ordered_events[0].timestamp
                    - timestamp
                ).total_seconds()
            )
            / 60.0
        )

        risk_score = self._calculate_risk_score(
            timestamp=timestamp,
            symbol=symbol,
            events=ordered_events,
        )

        risk_level = self._risk_level(
            risk_score
        )

        reasons = self._build_reasons(
            timestamp=timestamp,
            events=ordered_events,
            risk_score=risk_score,
        )

        return NewsRiskAssessment(
            timestamp=timestamp,
            symbol=symbol,
            risk_level=risk_level,
            risk_score=risk_score,
            relevant_events=ordered_events,
            usd_event_count=usd_event_count,
            high_impact_event_count=high_impact_event_count,
            nearest_event_minutes=nearest_event_minutes,
            reasons=tuple(reasons),
            sufficient_data=True,
        )

    def assess_symbol(
        self,
        *,
        timestamp: datetime,
        symbol: str,
        events: list[EconomicEvent],
    ) -> NewsRiskAssessment:
        """
        Alias for assess() for callers that prefer explicit naming.
        """

        return self.assess(
            timestamp=timestamp,
            symbol=symbol,
            events=events,
        )

    def _find_relevant_events(
        self,
        *,
        timestamp: datetime,
        events: list[EconomicEvent],
        currencies: set[str],
    ) -> list[EconomicEvent]:
        relevant: list[EconomicEvent] = []

        for event in events:
            if event.currency.upper() not in currencies:
                continue

            distance_minutes = abs(
                (
                    event.timestamp
                    - timestamp
                ).total_seconds()
            ) / 60.0

            if (
                distance_minutes
                <= self.relevant_window_minutes
            ):
                relevant.append(event)

        return relevant

    def _calculate_risk_score(
        self,
        *,
        timestamp: datetime,
        symbol: str,
        events: tuple[EconomicEvent, ...],
    ) -> float:
        score = 0.0

        for event in events:
            base_score = self.IMPACT_SCORE[
                event.impact
            ]

            if base_score <= 0.0:
                continue

            distance_minutes = abs(
                (
                    event.timestamp
                    - timestamp
                ).total_seconds()
            ) / 60.0

            if (
                distance_minutes
                <= self.imminent_window_minutes
            ):
                proximity_multiplier = 1.0
            else:
                proximity_multiplier = 0.5

            currency_multiplier = 1.0

            if (
                event.is_usd_event
                and symbol.upper() == "XAUUSD"
            ):
                currency_multiplier = 1.25

            score += (
                base_score
                * proximity_multiplier
                * currency_multiplier
            )

        if len(events) >= 2:
            score += min(
                15.0,
                (len(events) - 1) * 5.0,
            )

        return min(
            100.0,
            score,
        )

    def _risk_level(
        self,
        score: float,
    ) -> NewsRiskLevel:
        if score <= 0.0:
            return NewsRiskLevel.NONE

        if score >= self.extreme_risk_threshold:
            return NewsRiskLevel.EXTREME

        if score >= self.high_risk_threshold:
            return NewsRiskLevel.HIGH

        if score >= 30.0:
            return NewsRiskLevel.MEDIUM

        return NewsRiskLevel.LOW

    def _build_reasons(
        self,
        *,
        timestamp: datetime,
        events: tuple[EconomicEvent, ...],
        risk_score: float,
    ) -> list[NewsRiskReason]:
        reasons: list[NewsRiskReason] = []

        if len(events) > 1:
            reasons.append(
                NewsRiskReason(
                    NewsRiskReasonType.MULTIPLE_EVENTS,
                    (
                        f"{len(events)} relevant economic events "
                        "are within the configured window."
                    ),
                )
            )

        for event in events:
            time_difference_minutes = (
                (
                    event.timestamp
                    - timestamp
                ).total_seconds()
                / 60.0
            )

            distance_minutes = abs(
                time_difference_minutes
            )

            if event.impact == EventImpact.HIGH:
                reasons.append(
                    NewsRiskReason(
                        NewsRiskReasonType.HIGH_IMPACT_EVENT,
                        (
                            f"High-impact event: "
                            f"{event.name}."
                        ),
                    )
                )

            elif event.impact == EventImpact.MEDIUM:
                reasons.append(
                    NewsRiskReason(
                        NewsRiskReasonType.MEDIUM_IMPACT_EVENT,
                        (
                            f"Medium-impact event: "
                            f"{event.name}."
                        ),
                    )
                )

            elif event.impact == EventImpact.LOW:
                reasons.append(
                    NewsRiskReason(
                        NewsRiskReasonType.LOW_IMPACT_EVENT,
                        (
                            f"Low-impact event: "
                            f"{event.name}."
                        ),
                    )
                )

            else:
                reasons.append(
                    NewsRiskReason(
                        NewsRiskReasonType.UNKNOWN_EVENT_DATA,
                        (
                            f"Unknown impact classification: "
                            f"{event.name}."
                        ),
                    )
                )

            if event.is_usd_event:
                reasons.append(
                    NewsRiskReason(
                        NewsRiskReasonType.USD_EVENT,
                        (
                            f"USD event: "
                            f"{event.name}."
                        ),
                    )
                )

            # IMPORTANT:
            # Determine whether the event is in the past or future
            # BEFORE applying the proximity classification.
            if time_difference_minutes < 0:
                reasons.append(
                    NewsRiskReason(
                        NewsRiskReasonType.EVENT_RECENT,
                        (
                            f"{event.name} occurred "
                            f"{distance_minutes:.1f} minutes ago."
                        ),
                    )
                )
            elif (
                time_difference_minutes
                <= self.imminent_window_minutes
            ):
                reasons.append(
                    NewsRiskReason(
                        NewsRiskReasonType.EVENT_IMMINENT,
                        (
                            f"{event.name} is "
                            f"{distance_minutes:.1f} minutes "
                            "away."
                        ),
                    )
                )
            else:
                reasons.append(
                    NewsRiskReason(
                        NewsRiskReasonType.EVENT_PENDING,
                        (
                            f"{event.name} is "
                            f"{distance_minutes:.1f} minutes "
                            "away."
                        ),
                    )
                )

            if event.has_actual:
                reasons.append(
                    NewsRiskReason(
                        NewsRiskReasonType.ACTUAL_AVAILABLE,
                        (
                            f"Actual value is available "
                            f"for {event.name}."
                        ),
                    )
                )

        return reasons

    def _get_relevant_currencies(
        self,
        symbol: str,
    ) -> set[str]:
        normalized_symbol = symbol.upper()

        if normalized_symbol in self.SYMBOL_CURRENCIES:
            return set(
                self.SYMBOL_CURRENCIES[
                    normalized_symbol
                ]
            )

        if (
            len(normalized_symbol) == 6
            and normalized_symbol.isalpha()
        ):
            return {
                normalized_symbol[:3],
                normalized_symbol[3:],
            }

        if normalized_symbol == "XAUUSD":
            return {"USD"}

        return set()

    @staticmethod
    def _validate_inputs(
        *,
        timestamp: datetime,
        symbol: str,
        events: list[EconomicEvent],
    ) -> None:
        if not isinstance(
            timestamp,
            datetime,
        ):
            raise NewsRiskEngineError(
                "timestamp must be a datetime."
            )

        if (
            not isinstance(symbol, str)
            or not symbol.strip()
        ):
            raise NewsRiskEngineError(
                "symbol must be a non-empty string."
            )

        if not isinstance(events, list):
            raise NewsRiskEngineError(
                "events must be a list."
            )

        for event in events:
            if not isinstance(
                event,
                EconomicEvent,
            ):
                raise NewsRiskEngineError(
                    "events must contain EconomicEvent objects."
                )

            if (
                event.timestamp.tzinfo
                != timestamp.tzinfo
            ):
                raise NewsRiskEngineError(
                    "event timestamps must use the same timezone as the decision timestamp."
                )

    def _validate_configuration(self) -> None:
        numeric_values = (
            (
                "relevant_window_minutes",
                self.relevant_window_minutes,
            ),
            (
                "imminent_window_minutes",
                self.imminent_window_minutes,
            ),
            (
                "high_risk_threshold",
                self.high_risk_threshold,
            ),
            (
                "extreme_risk_threshold",
                self.extreme_risk_threshold,
            ),
        )

        for name, value in numeric_values:
            if (
                not isinstance(
                    value,
                    (int, float),
                )
                or isinstance(value, bool)
            ):
                raise NewsRiskEngineError(
                    f"{name} must be numeric."
                )

            if value < 0:
                raise NewsRiskEngineError(
                    f"{name} must be non-negative."
                )

        if (
            self.imminent_window_minutes
            > self.relevant_window_minutes
        ):
            raise NewsRiskEngineError(
                "imminent_window_minutes cannot exceed relevant_window_minutes."
            )

        if (
            self.high_risk_threshold
            > self.extreme_risk_threshold
        ):
            raise NewsRiskEngineError(
                "high_risk_threshold cannot exceed extreme_risk_threshold."
            )