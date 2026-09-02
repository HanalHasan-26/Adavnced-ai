from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.trading.data.market_bar import MarketBar


class SwingType(str, Enum):
    HIGH = "HIGH"
    LOW = "LOW"


class SwingLabel(str, Enum):
    HH = "HH"
    HL = "HL"
    LH = "LH"
    LL = "LL"


class StructureTrend(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGE = "RANGE"
    UNKNOWN = "UNKNOWN"


class StructureEventType(str, Enum):
    BOS_UP = "BOS_UP"
    BOS_DOWN = "BOS_DOWN"
    CHOCH_UP = "CHOCH_UP"
    CHOCH_DOWN = "CHOCH_DOWN"


@dataclass(frozen=True, slots=True)
class SwingPoint:
    """
    A confirmed market swing.

    index:
        Index of the candle that actually formed the swing.

    confirmation_index:
        First candle index at which the swing became knowable.

    This distinction is important for avoiding look-ahead bias.
    """

    index: int
    confirmation_index: int
    timestamp: object
    swing_type: SwingType
    price: float
    label: SwingLabel | None = None


@dataclass(frozen=True, slots=True)
class StructureEvent:
    """
    A confirmed market-structure event.

    index:
        Candle index where the structure break was confirmed.

    level:
        Swing level that was broken.
    """

    index: int
    timestamp: object
    event_type: StructureEventType
    level: float
    broken_swing_index: int


@dataclass(frozen=True, slots=True)
class MarketStructureResult:
    """
    Complete market-structure analysis result.
    """

    swings: tuple[SwingPoint, ...]
    events: tuple[StructureEvent, ...]
    trend: StructureTrend


class MarketStructureEngine:
    """
    Deterministic market-structure engine.

    The engine identifies confirmed pivot highs/lows and classifies
    subsequent swings as HH, HL, LH, or LL.

    A pivot at index i with right_window N is only known at:

        i + N

    This prevents future candles from being used before they actually
    become available.

    BOS:
        Break of Structure in the current directional structure.

    CHoCH:
        Change of Character against the current directional structure.
    """

    def __init__(
        self,
        left_window: int = 2,
        right_window: int = 2,
        break_on_close: bool = True,
    ) -> None:
        self._validate_window(
            left_window,
            "left_window",
        )

        self._validate_window(
            right_window,
            "right_window",
        )

        if not isinstance(break_on_close, bool):
            raise ValueError(
                "break_on_close must be a boolean."
            )

        self.left_window = left_window
        self.right_window = right_window
        self.break_on_close = break_on_close

    @staticmethod
    def _validate_bars(
        bars: list[MarketBar],
    ) -> None:
        if not isinstance(bars, list):
            raise ValueError(
                "bars must be a list of MarketBar objects."
            )

        for index, bar in enumerate(bars):
            if not isinstance(bar, MarketBar):
                raise ValueError(
                    f"bars[{index}] must be a MarketBar."
                )

    @staticmethod
    def _validate_window(
        value: int,
        name: str,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                f"{name} must be a positive integer."
            )

    @staticmethod
    def _is_swing_high(
        bars: list[MarketBar],
        index: int,
        left_window: int,
        right_window: int,
    ) -> bool:
        current = bars[index].high

        left_start = index - left_window
        left_end = index

        right_start = index + 1
        right_end = index + right_window + 1

        for candidate_index in range(
            left_start,
            left_end,
        ):
            if bars[candidate_index].high >= current:
                return False

        for candidate_index in range(
            right_start,
            right_end,
        ):
            if bars[candidate_index].high >= current:
                return False

        return True

    @staticmethod
    def _is_swing_low(
        bars: list[MarketBar],
        index: int,
        left_window: int,
        right_window: int,
    ) -> bool:
        current = bars[index].low

        left_start = index - left_window
        left_end = index

        right_start = index + 1
        right_end = index + right_window + 1

        for candidate_index in range(
            left_start,
            left_end,
        ):
            if bars[candidate_index].low <= current:
                return False

        for candidate_index in range(
            right_start,
            right_end,
        ):
            if bars[candidate_index].low <= current:
                return False

        return True

    def find_swings(
        self,
        bars: list[MarketBar],
    ) -> list[SwingPoint]:
        """
        Find confirmed swing highs and lows.

        Only candles with enough candles on both sides can become
        confirmed swings.

        Returned swings are ordered chronologically.
        """

        self._validate_bars(bars)

        if not bars:
            return []

        minimum_bars = (
            self.left_window
            + self.right_window
            + 1
        )

        if len(bars) < minimum_bars:
            return []

        swings: list[SwingPoint] = []

        start_index = self.left_window
        end_index = (
            len(bars)
            - self.right_window
        )

        for index in range(
            start_index,
            end_index,
        ):
            is_high = self._is_swing_high(
                bars,
                index,
                self.left_window,
                self.right_window,
            )

            is_low = self._is_swing_low(
                bars,
                index,
                self.left_window,
                self.right_window,
            )

            confirmation_index = (
                index + self.right_window
            )

            if is_high:
                swings.append(
                    SwingPoint(
                        index=index,
                        confirmation_index=confirmation_index,
                        timestamp=bars[index].timestamp,
                        swing_type=SwingType.HIGH,
                        price=bars[index].high,
                    )
                )

            if is_low:
                swings.append(
                    SwingPoint(
                        index=index,
                        confirmation_index=confirmation_index,
                        timestamp=bars[index].timestamp,
                        swing_type=SwingType.LOW,
                        price=bars[index].low,
                    )
                )

        swings.sort(
            key=lambda swing: (
                swing.confirmation_index,
                swing.index,
                swing.swing_type.value,
            )
        )

        return swings

    @staticmethod
    def _classify_swings(
        swings: list[SwingPoint],
    ) -> list[SwingPoint]:
        previous_high: SwingPoint | None = None
        previous_low: SwingPoint | None = None

        classified: list[SwingPoint] = []

        for swing in swings:
            label: SwingLabel | None = None

            if swing.swing_type is SwingType.HIGH:
                if previous_high is not None:
                    if swing.price > previous_high.price:
                        label = SwingLabel.HH
                    elif swing.price < previous_high.price:
                        label = SwingLabel.LH

                previous_high = swing

            elif swing.swing_type is SwingType.LOW:
                if previous_low is not None:
                    if swing.price > previous_low.price:
                        label = SwingLabel.HL
                    elif swing.price < previous_low.price:
                        label = SwingLabel.LL

                previous_low = swing

            classified.append(
                SwingPoint(
                    index=swing.index,
                    confirmation_index=swing.confirmation_index,
                    timestamp=swing.timestamp,
                    swing_type=swing.swing_type,
                    price=swing.price,
                    label=label,
                )
            )

        return classified

    @staticmethod
    def _determine_trend(
        swings: list[SwingPoint],
    ) -> StructureTrend:
        highs = [
            swing
            for swing in swings
            if swing.swing_type is SwingType.HIGH
            and swing.label is not None
        ]

        lows = [
            swing
            for swing in swings
            if swing.swing_type is SwingType.LOW
            and swing.label is not None
        ]

        if not highs or not lows:
            return StructureTrend.UNKNOWN

        latest_high = highs[-1]
        latest_low = lows[-1]

        if (
            latest_high.label is SwingLabel.HH
            and latest_low.label is SwingLabel.HL
        ):
            return StructureTrend.BULLISH

        if (
            latest_high.label is SwingLabel.LH
            and latest_low.label is SwingLabel.LL
        ):
            return StructureTrend.BEARISH

        return StructureTrend.RANGE

    def detect_structure_events(
        self,
        bars: list[MarketBar],
        swings: list[SwingPoint] | None = None,
    ) -> list[StructureEvent]:
        """
        Detect BOS and CHoCH events.

        Structure levels become usable only after their swing has been
        confirmed.

        A level is considered broken when price crosses it.

        If break_on_close=True:
            close must cross the level.

        If break_on_close=False:
            high/low crossing is sufficient.

        A structure level is consumed after being broken, preventing
        repeated BOS/CHoCH events from the same level.
        """

        self._validate_bars(bars)

        if not bars:
            return []

        if swings is None:
            swings = self.find_swings(bars)

        swings = self._classify_swings(swings)

        events: list[StructureEvent] = []

        active_high: SwingPoint | None = None
        active_low: SwingPoint | None = None

        broken_high_index: int | None = None
        broken_low_index: int | None = None

        trend = StructureTrend.UNKNOWN

        swings_by_confirmation: dict[int, list[SwingPoint]] = {}

        for swing in swings:
            swings_by_confirmation.setdefault(
                swing.confirmation_index,
                [],
            ).append(swing)

        for index, bar in enumerate(bars):
            newly_confirmed = swings_by_confirmation.get(
                index,
                [],
            )

            for swing in newly_confirmed:
                if swing.swing_type is SwingType.HIGH:
                    active_high = swing
                    broken_high_index = None

                elif swing.swing_type is SwingType.LOW:
                    active_low = swing
                    broken_low_index = None

            if active_high is not None:
                if (
                    broken_high_index is None
                    and index > active_high.confirmation_index
                ):
                    if self.break_on_close:
                        broken = (
                            bar.close
                            > active_high.price
                        )
                    else:
                        broken = (
                            bar.high
                            > active_high.price
                        )

                    if broken:
                        if trend is StructureTrend.BEARISH:
                            event_type = (
                                StructureEventType.CHOCH_UP
                            )
                        else:
                            event_type = (
                                StructureEventType.BOS_UP
                            )

                        events.append(
                            StructureEvent(
                                index=index,
                                timestamp=bar.timestamp,
                                event_type=event_type,
                                level=active_high.price,
                                broken_swing_index=active_high.index,
                            )
                        )

                        broken_high_index = (
                            active_high.index
                        )

                        trend = StructureTrend.BULLISH

            if active_low is not None:
                if (
                    broken_low_index is None
                    and index > active_low.confirmation_index
                ):
                    if self.break_on_close:
                        broken = (
                            bar.close
                            < active_low.price
                        )
                    else:
                        broken = (
                            bar.low
                            < active_low.price
                        )

                    if broken:
                        if trend is StructureTrend.BULLISH:
                            event_type = (
                                StructureEventType.CHOCH_DOWN
                            )
                        else:
                            event_type = (
                                StructureEventType.BOS_DOWN
                            )

                        events.append(
                            StructureEvent(
                                index=index,
                                timestamp=bar.timestamp,
                                event_type=event_type,
                                level=active_low.price,
                                broken_swing_index=active_low.index,
                            )
                        )

                        broken_low_index = (
                            active_low.index
                        )

                        trend = StructureTrend.BEARISH

        return events

    def analyze(
        self,
        bars: list[MarketBar],
    ) -> MarketStructureResult:
        """
        Run the complete market-structure analysis.
        """

        self._validate_bars(bars)

        if not bars:
            return MarketStructureResult(
                swings=(),
                events=(),
                trend=StructureTrend.UNKNOWN,
            )

        raw_swings = self.find_swings(bars)

        classified_swings = self._classify_swings(
            raw_swings
        )

        events = self.detect_structure_events(
            bars,
            classified_swings,
        )

        trend = self._determine_trend(
            classified_swings
        )

        if events:
            last_event = events[-1]

            if last_event.event_type in (
                StructureEventType.BOS_UP,
                StructureEventType.CHOCH_UP,
            ):
                trend = StructureTrend.BULLISH

            elif last_event.event_type in (
                StructureEventType.BOS_DOWN,
                StructureEventType.CHOCH_DOWN,
            ):
                trend = StructureTrend.BEARISH

        return MarketStructureResult(
            swings=tuple(classified_swings),
            events=tuple(events),
            trend=trend,
        )