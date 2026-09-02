from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.trading.data.market_bar import MarketBar
from app.trading.structure.market_structure import (
    MarketStructureEngine,
    StructureEventType,
    StructureTrend,
    SwingLabel,
    SwingType,
)


def make_bars(
    highs: list[float],
    lows: list[float] | None = None,
) -> list[MarketBar]:
    if lows is None:
        lows = [
            high - 2.0
            for high in highs
        ]

    if len(highs) != len(lows):
        raise ValueError(
            "highs and lows must have the same length."
        )

    bars: list[MarketBar] = []

    start = datetime(2026, 1, 1)

    for index, (high, low) in enumerate(
        zip(highs, lows)
    ):
        high = float(high)
        low = float(low)

        close = (high + low) / 2.0

        bars.append(
            MarketBar(
                timestamp=start + timedelta(
                    minutes=index
                ),
                symbol="XAUUSD",
                timeframe="M1",
                open=close,
                high=high,
                low=low,
                close=close,
                volume=100.0,
            )
        )

    return bars


def make_bars_with_closes(
    highs: list[float],
    lows: list[float],
    closes: list[float],
) -> list[MarketBar]:
    if not (
        len(highs)
        == len(lows)
        == len(closes)
    ):
        raise ValueError(
            "highs, lows and closes must have the same length."
        )

    bars: list[MarketBar] = []

    start = datetime(2026, 1, 1)

    for index, (
        high,
        low,
        close,
    ) in enumerate(
        zip(highs, lows, closes)
    ):
        close = float(close)
        high = float(high)
        low = float(low)

        bars.append(
            MarketBar(
                timestamp=start + timedelta(
                    minutes=index
                ),
                symbol="XAUUSD",
                timeframe="M1",
                open=close,
                high=high,
                low=low,
                close=close,
                volume=100.0,
            )
        )

    return bars


class TestValidation:
    def test_empty_bars_return_empty_result(self):
        engine = MarketStructureEngine()

        result = engine.analyze([])

        assert result.swings == ()
        assert result.events == ()
        assert result.trend is StructureTrend.UNKNOWN

    def test_bars_must_be_a_list(self):
        engine = MarketStructureEngine()

        with pytest.raises(ValueError):
            engine.analyze(
                tuple()
            )  # type: ignore[arg-type]

    def test_invalid_bar_type_is_rejected(self):
        engine = MarketStructureEngine()

        with pytest.raises(ValueError):
            engine.analyze(
                [object()]  # type: ignore[list-item]
            )

    def test_left_window_must_be_positive(self):
        with pytest.raises(ValueError):
            MarketStructureEngine(
                left_window=0
            )

    def test_right_window_must_be_positive(self):
        with pytest.raises(ValueError):
            MarketStructureEngine(
                right_window=0
            )

    def test_windows_must_be_integers(self):
        with pytest.raises(ValueError):
            MarketStructureEngine(
                left_window=2.5  # type: ignore[arg-type]
            )


class TestSwingDetection:
    def test_detects_swing_high(self):
        bars = make_bars(
            [10, 11, 15, 11, 10]
        )

        engine = MarketStructureEngine(
            left_window=2,
            right_window=2,
        )

        swings = engine.find_swings(bars)

        highs = [
            swing
            for swing in swings
            if swing.swing_type is SwingType.HIGH
        ]

        assert len(highs) == 1

        assert highs[0].index == 2
        assert highs[0].price == pytest.approx(15.0)

    def test_detects_swing_low(self):
        bars = make_bars(
            [15, 11, 10, 11, 15]
        )

        engine = MarketStructureEngine(
            left_window=2,
            right_window=2,
        )

        swings = engine.find_swings(bars)

        lows = [
            swing
            for swing in swings
            if swing.swing_type is SwingType.LOW
        ]

        assert len(lows) == 1

        assert lows[0].index == 2
        assert lows[0].price == pytest.approx(8.0)

    def test_swing_requires_both_sides(self):
        bars = make_bars(
            [10, 20, 10]
        )

        engine = MarketStructureEngine(
            left_window=2,
            right_window=2,
        )

        swings = engine.find_swings(bars)

        assert swings == []

    def test_equal_high_does_not_create_unique_swing_high(self):
        bars = make_bars(
            [10, 15, 15, 10, 9]
        )

        engine = MarketStructureEngine(
            left_window=1,
            right_window=1,
        )

        swings = engine.find_swings(bars)

        highs = [
            swing
            for swing in swings
            if swing.swing_type is SwingType.HIGH
        ]

        assert highs == []

    def test_equal_low_does_not_create_unique_swing_low(self):
        bars = make_bars(
            [15, 10, 10, 15, 16]
        )

        engine = MarketStructureEngine(
            left_window=1,
            right_window=1,
        )

        swings = engine.find_swings(bars)

        lows = [
            swing
            for swing in swings
            if swing.swing_type is SwingType.LOW
        ]

        assert lows == []

    def test_confirmation_index_is_delayed_by_right_window(self):
        bars = make_bars(
            [10, 11, 15, 11, 10, 9]
        )

        engine = MarketStructureEngine(
            left_window=2,
            right_window=2,
        )

        swings = engine.find_swings(bars)

        high = next(
            swing
            for swing in swings
            if swing.swing_type is SwingType.HIGH
        )

        assert high.index == 2
        assert high.confirmation_index == 4

    def test_swings_are_chronological(self):
        bars = make_bars(
            [
                10,
                15,
                10,
                14,
                9,
                13,
                8,
            ]
        )

        engine = MarketStructureEngine(
            left_window=1,
            right_window=1,
        )

        swings = engine.find_swings(bars)

        confirmation_indexes = [
            swing.confirmation_index
            for swing in swings
        ]

        assert confirmation_indexes == sorted(
            confirmation_indexes
        )


class TestSwingClassification:
    def test_higher_high_is_classified(self):
        bars = make_bars(
            [
                10,
                15,
                10,
                17,
                11,
                20,
                12,
            ]
        )

        engine = MarketStructureEngine(
            left_window=1,
            right_window=1,
        )

        result = engine.analyze(bars)

        highs = [
            swing
            for swing in result.swings
            if swing.swing_type is SwingType.HIGH
        ]

        assert len(highs) >= 2

        assert highs[1].label is SwingLabel.HH

    def test_lower_high_is_classified(self):
        bars = make_bars(
            [
                10,
                20,
                10,
                17,
                11,
            ]
        )

        engine = MarketStructureEngine(
            left_window=1,
            right_window=1,
        )

        result = engine.analyze(bars)

        highs = [
            swing
            for swing in result.swings
            if swing.swing_type is SwingType.HIGH
        ]

        assert len(highs) >= 2

        assert highs[1].label is SwingLabel.LH

    def test_higher_low_is_classified(self):
        bars = make_bars(
            [15, 10, 16, 12, 17, 14, 18]
        )

        engine = MarketStructureEngine(
            left_window=1,
            right_window=1,
        )

        result = engine.analyze(bars)

        lows = [
            swing
            for swing in result.swings
            if swing.swing_type is SwingType.LOW
        ]

        assert len(lows) >= 2

        assert lows[1].label is SwingLabel.HL

    def test_lower_low_is_classified(self):
        bars = make_bars(
            [15, 10, 14, 8, 13, 6, 12]
        )

        engine = MarketStructureEngine(
            left_window=1,
            right_window=1,
        )

        result = engine.analyze(bars)

        lows = [
            swing
            for swing in result.swings
            if swing.swing_type is SwingType.LOW
        ]

        assert len(lows) >= 2

        assert lows[1].label is SwingLabel.LL


class TestTrend:
    def test_bullish_structure(self):
        bars = make_bars(
            [
                10,
                15,
                10,
                17,
                12,
                20,
                14,
            ]
        )

        engine = MarketStructureEngine(
            left_window=1,
            right_window=1,
        )

        result = engine.analyze(bars)

        assert result.trend is StructureTrend.BULLISH

    def test_bearish_structure(self):
        bars = make_bars(
            [
                20,
                15,
                18,
                12,
                16,
                9,
                14,
            ]
        )

        engine = MarketStructureEngine(
            left_window=1,
            right_window=1,
        )

        result = engine.analyze(bars)

        assert result.trend is StructureTrend.BEARISH

    def test_unknown_trend_without_enough_structure(self):
        bars = make_bars(
            [10, 15, 10]
        )

        engine = MarketStructureEngine(
            left_window=1,
            right_window=1,
        )

        result = engine.analyze(bars)

        assert result.trend is StructureTrend.UNKNOWN


class TestBOS:
    def test_bos_up(self):
        highs = [
            10,
            15,
            10,
            14,
            11,
            17,
            12,
            18,
        ]

        lows = [
            8,
            13,
            8,
            12,
            9,
            15,
            10,
            16,
        ]

        closes = [
            9,
            14,
            9,
            13,
            10,
            16,
            11,
            17,
        ]

        bars = make_bars_with_closes(
            highs,
            lows,
            closes,
        )

        engine = MarketStructureEngine(
            left_window=1,
            right_window=1,
        )

        events = engine.detect_structure_events(
            bars
        )

        assert any(
            event.event_type
            is StructureEventType.BOS_UP
            for event in events
        )

    def test_bos_down(self):
        highs = [
            20,
            17,
            19,
            14,
            18,
            12,
            16,
            10,
        ]

        lows = [
            18,
            15,
            17,
            12,
            16,
            10,
            14,
            8,
        ]

        closes = [
            19,
            16,
            18,
            13,
            17,
            11,
            15,
            9,
        ]

        bars = make_bars_with_closes(
            highs,
            lows,
            closes,
        )

        engine = MarketStructureEngine(
            left_window=1,
            right_window=1,
        )

        events = engine.detect_structure_events(
            bars
        )

        assert any(
            event.event_type
            is StructureEventType.BOS_DOWN
            for event in events
        )

    def test_broken_level_is_not_repeated(self):
        highs = [
            10,
            15,
            10,
            14,
            11,
            17,
            12,
            18,
            19,
        ]

        lows = [
            8,
            13,
            8,
            12,
            9,
            15,
            10,
            16,
            17,
        ]

        closes = [
            9,
            14,
            9,
            13,
            10,
            16,
            11,
            17,
            18,
        ]

        bars = make_bars_with_closes(
            highs,
            lows,
            closes,
        )

        engine = MarketStructureEngine(
            left_window=1,
            right_window=1,
        )

        events = engine.detect_structure_events(
            bars
        )

        first_level_events = [
            event
            for event in events
            if event.level == pytest.approx(15.0)
        ]

        assert len(first_level_events) <= 1


class TestCHoCH:
    def test_choch_down_after_bullish_structure_break(self):
        highs = [
            10,
            15,
            10,
            17,
            12,
            19,
            14,
            16,
            11,
        ]

        lows = [
            8,
            13,
            8,
            15,
            10,
            17,
            12,
            14,
            9,
        ]

        closes = [
            9,
            14,
            9,
            16,
            11,
            18,
            13,
            15,
            10,
        ]

        bars = make_bars_with_closes(
            highs,
            lows,
            closes,
        )

        engine = MarketStructureEngine(
            left_window=1,
            right_window=1,
        )

        events = engine.detect_structure_events(
            bars
        )

        assert any(
            event.event_type
            is StructureEventType.CHOCH_DOWN
            for event in events
        )

    def test_choch_up_after_bearish_structure_break(self):
        highs = [
            20,
            16,
            18,
            13,
            17,
            11,
            15,
            18,
            20,
        ]

        lows = [
            18,
            14,
            16,
            11,
            15,
            9,
            13,
            16,
            18,
        ]

        closes = [
            19,
            15,
            17,
            12,
            16,
            10,
            14,
            17,
            19,
        ]

        bars = make_bars_with_closes(
            highs,
            lows,
            closes,
        )

        engine = MarketStructureEngine(
            left_window=1,
            right_window=1,
        )

        events = engine.detect_structure_events(
            bars
        )

        assert any(
            event.event_type
            is StructureEventType.CHOCH_UP
            for event in events
        )


class TestNoLookahead:
    def test_future_candle_cannot_change_already_confirmed_swing(self):
        bars_short = make_bars(
            [
                10,
                15,
                10,
                14,
                11,
            ]
        )

        bars_long = make_bars(
            [
                10,
                15,
                10,
                14,
                11,
                100,
                5,
            ]
        )

        engine = MarketStructureEngine(
            left_window=1,
            right_window=1,
        )

        short_swings = engine.find_swings(
            bars_short
        )

        long_swings = engine.find_swings(
            bars_long
        )

        short_confirmed = [
            swing
            for swing in short_swings
            if swing.confirmation_index <= 4
        ]

        long_confirmed = [
            swing
            for swing in long_swings
            if swing.confirmation_index <= 4
        ]

        assert short_confirmed == long_confirmed

    def test_swing_is_not_confirmed_before_right_window(self):
        bars = make_bars(
            [
                10,
                15,
                10,
                14,
                11,
            ]
        )

        engine = MarketStructureEngine(
            left_window=1,
            right_window=2,
        )

        swings = engine.find_swings(bars)

        high = next(
            swing
            for swing in swings
            if swing.swing_type is SwingType.HIGH
        )

        assert high.index == 1
        assert high.confirmation_index == 3

        assert high.confirmation_index > high.index