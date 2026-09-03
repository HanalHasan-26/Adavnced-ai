from datetime import datetime, timedelta

import pytest

from app.trading.data.market_bar import MarketBar
from app.trading.trendline.trendline_engine import (
    TrendlineEngine,
    TrendlineEngineError,
    TrendlineStatus,
    TrendlineType,
)


BASE_TIME = datetime(2026, 1, 1)


def make_bar(
    index: int,
    open_price: float,
    high: float,
    low: float,
    close: float,
    *,
    symbol: str = "XAUUSD",
    timeframe: str = "M15",
) -> MarketBar:
    return MarketBar(
        timestamp=BASE_TIME + timedelta(
            minutes=index * 15
        ),
        symbol=symbol,
        timeframe=timeframe,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=100.0,
    )


def make_bars() -> list[MarketBar]:
    return [
        make_bar(0, 100, 103, 99, 102),
        make_bar(1, 102, 105, 100, 104),
        make_bar(2, 104, 107, 102, 106),
        make_bar(3, 106, 108, 103, 105),
        make_bar(4, 105, 109, 104, 108),
        make_bar(5, 108, 110, 105, 109),
        make_bar(6, 109, 111, 106, 110),
        make_bar(7, 110, 112, 107, 108),
        make_bar(8, 108, 109, 104, 105),
        make_bar(9, 105, 107, 101, 103),
        make_bar(10, 103, 106, 99, 101),
        make_bar(11, 101, 105, 100, 104),
        make_bar(12, 104, 109, 103, 107),
        make_bar(13, 107, 111, 105, 109),
        make_bar(14, 109, 112, 107, 110),
        make_bar(15, 110, 113, 108, 111),
        make_bar(16, 111, 114, 109, 112),
        make_bar(17, 112, 115, 110, 113),
        make_bar(18, 113, 116, 111, 114),
        make_bar(19, 114, 117, 112, 115),
    ]


def test_engine_defaults():
    engine = TrendlineEngine()

    assert engine.swing_left == 2
    assert engine.swing_right == 2
    assert engine.touch_tolerance == 0.001
    assert engine.break_tolerance == 0.001
    assert engine.min_touches == 2
    assert engine.min_strength == 20.0
    assert engine.max_trendlines == 20


@pytest.mark.parametrize(
    "kwargs",
    [
        {"swing_left": 0},
        {"swing_right": 0},
        {"swing_left": -1},
        {"swing_right": -1},
    ],
)
def test_invalid_swing_windows(kwargs):
    with pytest.raises(TrendlineEngineError):
        TrendlineEngine(**kwargs)


@pytest.mark.parametrize(
    "value",
    [
        -1.0,
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_invalid_touch_tolerance(value):
    with pytest.raises(TrendlineEngineError):
        TrendlineEngine(
            touch_tolerance=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        -1.0,
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_invalid_break_tolerance(value):
    with pytest.raises(TrendlineEngineError):
        TrendlineEngine(
            break_tolerance=value,
        )


@pytest.mark.parametrize(
    "value",
    [0, 1, True, 1.5],
)
def test_invalid_min_touches(value):
    with pytest.raises(TrendlineEngineError):
        TrendlineEngine(
            min_touches=value,
        )


@pytest.mark.parametrize(
    "value",
    [-1.0, 101.0, float("inf"), float("nan")],
)
def test_invalid_min_strength(value):
    with pytest.raises(TrendlineEngineError):
        TrendlineEngine(
            min_strength=value,
        )


@pytest.mark.parametrize(
    "value",
    [0, -1, True, 1.5],
)
def test_invalid_max_trendlines(value):
    with pytest.raises(TrendlineEngineError):
        TrendlineEngine(
            max_trendlines=value,
        )


def test_empty_bars_returns_empty():
    assert TrendlineEngine().analyze([]) == ()


def test_invalid_bars_type():
    with pytest.raises(TrendlineEngineError):
        TrendlineEngine().analyze("invalid")


def test_invalid_bar_type():
    with pytest.raises(TrendlineEngineError):
        TrendlineEngine().analyze([object()])


def test_mixed_symbols_rejected():
    bars = make_bars()

    bars[5] = make_bar(
        5,
        108,
        110,
        105,
        109,
        symbol="EURUSD",
    )

    with pytest.raises(TrendlineEngineError):
        TrendlineEngine().analyze(bars)


def test_mixed_timeframes_rejected():
    bars = make_bars()

    bars[5] = make_bar(
        5,
        108,
        110,
        105,
        109,
        timeframe="H1",
    )

    with pytest.raises(TrendlineEngineError):
        TrendlineEngine().analyze(bars)


def test_out_of_order_rejected():
    bars = make_bars()

    bars[5], bars[6] = bars[6], bars[5]

    with pytest.raises(TrendlineEngineError):
        TrendlineEngine().analyze(bars)


def test_duplicate_timestamp_rejected():
    bars = make_bars()

    bars[5] = make_bar(
        4,
        108,
        110,
        105,
        109,
    )

    with pytest.raises(TrendlineEngineError):
        TrendlineEngine().analyze(bars)


def test_result_is_tuple():
    result = TrendlineEngine().analyze(
        make_bars()
    )

    assert isinstance(result, tuple)


def test_result_contains_only_trendlines():
    result = TrendlineEngine().analyze(
        make_bars()
    )

    assert all(
        hasattr(item, "trendline_type")
        for item in result
    )


def test_support_filter():
    engine = TrendlineEngine()

    result = engine.analyze_support(
        make_bars()
    )

    assert all(
        item.is_support
        for item in result
    )


def test_resistance_filter():
    engine = TrendlineEngine()

    result = engine.analyze_resistance(
        make_bars()
    )

    assert all(
        item.is_resistance
        for item in result
    )


def test_trendline_type_is_valid():
    result = TrendlineEngine().analyze(
        make_bars()
    )

    assert all(
        item.trendline_type in (
            TrendlineType.SUPPORT,
            TrendlineType.RESISTANCE,
        )
        for item in result
    )


def test_support_property():
    result = TrendlineEngine().analyze(
        make_bars()
    )

    for item in result:
        if item.trendline_type is TrendlineType.SUPPORT:
            assert item.is_support
            assert not item.is_resistance


def test_resistance_property():
    result = TrendlineEngine().analyze(
        make_bars()
    )

    for item in result:
        if item.trendline_type is TrendlineType.RESISTANCE:
            assert item.is_resistance
            assert not item.is_support


def test_slope_is_finite():
    result = TrendlineEngine().analyze(
        make_bars()
    )

    assert all(
        isinstance(item.slope, float)
        for item in result
    )


def test_price_at_first_index():
    result = TrendlineEngine().analyze(
        make_bars()
    )

    for item in result:
        assert (
            abs(
                item.price_at(item.first_index)
                - item.first_price
            )
            < 1e-8
        )


def test_price_at_second_index():
    result = TrendlineEngine().analyze(
        make_bars()
    )

    for item in result:
        assert (
            abs(
                item.price_at(item.second_index)
                - item.second_price
            )
            < 1e-8
        )


def test_source_indices_match_endpoints():
    result = TrendlineEngine().analyze(
        make_bars()
    )

    for item in result:
        assert item.source_indices == (
            item.first_index,
            item.second_index,
        )


def test_indices_are_chronological():
    result = TrendlineEngine().analyze(
        make_bars()
    )

    assert all(
        item.first_index < item.second_index
        for item in result
    )


def test_prices_are_positive():
    result = TrendlineEngine().analyze(
        make_bars()
    )

    assert all(
        item.first_price > 0
        and item.second_price > 0
        for item in result
    )


def test_touch_count_minimum():
    result = TrendlineEngine().analyze(
        make_bars()
    )

    assert all(
        item.touch_count >= 2
        for item in result
    )


def test_rejection_count_non_negative():
    result = TrendlineEngine().analyze(
        make_bars()
    )

    assert all(
        item.rejection_count >= 0
        for item in result
    )


def test_rejection_count_not_above_touches():
    result = TrendlineEngine().analyze(
        make_bars()
    )

    assert all(
        item.rejection_count
        <= item.touch_count
        for item in result
    )


def test_strength_is_bounded():
    result = TrendlineEngine().analyze(
        make_bars()
    )

    assert all(
        0.0 <= item.strength <= 100.0
        for item in result
    )


def test_distance_is_non_negative():
    result = TrendlineEngine().analyze(
        make_bars()
    )

    assert all(
        item.distance_from_price >= 0.0
        for item in result
    )


def test_projected_price_is_finite():
    result = TrendlineEngine().analyze(
        make_bars()
    )

    assert all(
        isinstance(item.projected_price, float)
        for item in result
    )


def test_status_is_valid():
    result = TrendlineEngine().analyze(
        make_bars()
    )

    assert all(
        item.status in (
            TrendlineStatus.ACTIVE,
            TrendlineStatus.TESTED,
            TrendlineStatus.BROKEN,
            TrendlineStatus.INVALID,
        )
        for item in result
    )


def test_active_property():
    result = TrendlineEngine().analyze(
        make_bars()
    )

    for item in result:
        assert item.is_active == (
            item.status
            in (
                TrendlineStatus.ACTIVE,
                TrendlineStatus.TESTED,
            )
        )


def test_broken_property():
    result = TrendlineEngine().analyze(
        make_bars()
    )

    for item in result:
        assert item.is_broken == (
            item.status
            is TrendlineStatus.BROKEN
        )


def test_strong_property():
    result = TrendlineEngine().analyze(
        make_bars()
    )

    for item in result:
        assert item.is_strong == (
            item.strength >= 70.0
        )


def test_analyze_is_deterministic():
    bars = make_bars()

    engine = TrendlineEngine()

    first = engine.analyze(bars)
    second = engine.analyze(bars)

    assert first == second


def test_support_and_resistance_are_subsets():
    bars = make_bars()

    engine = TrendlineEngine()

    all_lines = engine.analyze(bars)
    supports = engine.analyze_support(bars)
    resistances = engine.analyze_resistance(bars)

    assert all(
        item in all_lines
        for item in supports
    )

    assert all(
        item in all_lines
        for item in resistances
    )


def test_max_trendlines_respected():
    engine = TrendlineEngine(
        max_trendlines=2,
    )

    result = engine.analyze(
        make_bars()
    )

    assert len(result) <= 2


def test_min_strength_filter_respected():
    engine = TrendlineEngine(
        min_strength=90.0,
    )

    result = engine.analyze(
        make_bars()
    )

    assert all(
        item.strength >= 90.0
        for item in result
    )


def test_results_ordered_by_distance():
    result = TrendlineEngine().analyze(
        make_bars()
    )

    distances = [
        item.distance_from_price
        for item in result
    ]

    assert distances == sorted(distances)


def test_xauusd_wrapper_accepts_xauusd():
    result = TrendlineEngine().analyze_xauusd(
        make_bars()
    )

    assert isinstance(result, tuple)


def test_xauusd_wrapper_rejects_other_symbol():
    bars = [
        make_bar(
            index,
            100 + index,
            103 + index,
            99 + index,
            102 + index,
            symbol="EURUSD",
        )
        for index in range(20)
    ]

    with pytest.raises(TrendlineEngineError):
        TrendlineEngine().analyze_xauusd(bars)


def test_nearest_support_is_none_or_support():
    result = TrendlineEngine().nearest_support(
        make_bars()
    )

    assert (
        result is None
        or result.is_support
    )


def test_nearest_resistance_is_none_or_resistance():
    result = TrendlineEngine().nearest_resistance(
        make_bars()
    )

    assert (
        result is None
        or result.is_resistance
    )


def test_nearest_support_is_active():
    result = TrendlineEngine().nearest_support(
        make_bars()
    )

    if result is not None:
        assert result.is_active


def test_nearest_resistance_is_active():
    result = TrendlineEngine().nearest_resistance(
        make_bars()
    )

    if result is not None:
        assert result.is_active