from datetime import datetime, timedelta

import pytest

from app.trading.data.market_bar import MarketBar
from app.trading.support_resistance.support_resistance import (
    SRLevelType,
    SRZoneStatus,
    SupportResistanceEngine,
    SupportResistanceError,
)


BASE_TIME = datetime(2026, 1, 1, 0, 0, 0)


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
        timestamp=BASE_TIME + timedelta(minutes=index * 15),
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
        make_bar(1, 102, 106, 101, 105),
        make_bar(2, 105, 110, 104, 108),
        make_bar(3, 108, 109, 103, 104),
        make_bar(4, 104, 105, 98, 100),
        make_bar(5, 100, 102, 97, 101),
        make_bar(6, 101, 107, 100, 106),
        make_bar(7, 106, 112, 105, 110),
        make_bar(8, 110, 111, 104, 105),
        make_bar(9, 105, 106, 99, 101),
        make_bar(10, 101, 103, 96, 98),
        make_bar(11, 98, 102, 97, 101),
        make_bar(12, 101, 108, 100, 106),
        make_bar(13, 106, 113, 105, 111),
        make_bar(14, 111, 112, 106, 108),
    ]


def test_engine_defaults():
    engine = SupportResistanceEngine()

    assert engine.swing_left == 2
    assert engine.swing_right == 2
    assert engine.zone_tolerance == 0.001
    assert engine.min_zone_strength == 20.0
    assert engine.max_zones == 20


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
    with pytest.raises(SupportResistanceError):
        SupportResistanceEngine(**kwargs)


@pytest.mark.parametrize(
    "value",
    [-1.0, float("inf"), float("-inf"), float("nan")],
)
def test_invalid_zone_tolerance(value):
    with pytest.raises(SupportResistanceError):
        SupportResistanceEngine(
            zone_tolerance=value,
        )


@pytest.mark.parametrize(
    "value",
    [-1.0, 101.0, float("inf"), float("-inf"), float("nan")],
)
def test_invalid_min_zone_strength(value):
    with pytest.raises(SupportResistanceError):
        SupportResistanceEngine(
            min_zone_strength=value,
        )


@pytest.mark.parametrize(
    "value",
    [0, -1, True, 1.5],
)
def test_invalid_max_zones(value):
    with pytest.raises(SupportResistanceError):
        SupportResistanceEngine(
            max_zones=value,
        )


def test_empty_bars_returns_empty_zones():
    engine = SupportResistanceEngine()

    assert engine.analyze([]) == ()


def test_empty_bars_returns_empty_levels():
    engine = SupportResistanceEngine()

    assert engine.analyze_levels([]) == ()


def test_invalid_bars_type():
    engine = SupportResistanceEngine()

    with pytest.raises(SupportResistanceError):
        engine.analyze("not-a-list")


def test_invalid_bar_type():
    engine = SupportResistanceEngine()

    with pytest.raises(SupportResistanceError):
        engine.analyze([object()])


def test_mixed_symbols_are_rejected():
    bars = make_bars()

    bars[5] = make_bar(
        5,
        100,
        102,
        97,
        101,
        symbol="EURUSD",
    )

    with pytest.raises(SupportResistanceError):
        SupportResistanceEngine().analyze(bars)


def test_mixed_timeframes_are_rejected():
    bars = make_bars()

    bars[5] = make_bar(
        5,
        100,
        102,
        97,
        101,
        timeframe="H1",
    )

    with pytest.raises(SupportResistanceError):
        SupportResistanceEngine().analyze(bars)


def test_out_of_order_bars_are_rejected():
    bars = make_bars()

    bars[5], bars[6] = bars[6], bars[5]

    with pytest.raises(SupportResistanceError):
        SupportResistanceEngine().analyze(bars)


def test_duplicate_timestamps_are_rejected():
    bars = make_bars()

    bars[5] = make_bar(
        4,
        100,
        102,
        97,
        101,
    )

    with pytest.raises(SupportResistanceError):
        SupportResistanceEngine().analyze(bars)


def test_levels_are_generated_from_structure():
    engine = SupportResistanceEngine()

    levels = engine.analyze_levels(make_bars())

    assert levels

    assert all(
        level.level_type in (
            SRLevelType.SUPPORT,
            SRLevelType.RESISTANCE,
        )
        for level in levels
    )


def test_support_levels_come_from_swing_lows():
    engine = SupportResistanceEngine()

    levels = engine.analyze_levels(make_bars())

    supports = [
        level
        for level in levels
        if level.is_support
    ]

    assert supports


def test_resistance_levels_come_from_swing_highs():
    engine = SupportResistanceEngine()

    levels = engine.analyze_levels(make_bars())

    resistances = [
        level
        for level in levels
        if level.is_resistance
    ]

    assert resistances


def test_level_prices_are_positive():
    engine = SupportResistanceEngine()

    levels = engine.analyze_levels(make_bars())

    assert all(
        level.price > 0
        for level in levels
    )


def test_level_strength_is_bounded():
    engine = SupportResistanceEngine()

    levels = engine.analyze_levels(make_bars())

    assert all(
        0.0 <= level.strength <= 100.0
        for level in levels
    )


def test_level_distance_is_non_negative():
    engine = SupportResistanceEngine()

    levels = engine.analyze_levels(make_bars())

    assert all(
        level.distance_from_price >= 0.0
        for level in levels
    )


def test_touch_count_is_non_negative():
    engine = SupportResistanceEngine()

    levels = engine.analyze_levels(make_bars())

    assert all(
        level.touch_count >= 0
        for level in levels
    )


def test_strong_property_matches_strength():
    engine = SupportResistanceEngine()

    levels = engine.analyze_levels(make_bars())

    assert all(
        level.is_strong == (level.strength >= 70.0)
        for level in levels
    )


def test_zones_are_generated():
    engine = SupportResistanceEngine()

    zones = engine.analyze(make_bars())

    assert zones


def test_zones_have_valid_types():
    engine = SupportResistanceEngine()

    zones = engine.analyze(make_bars())

    assert all(
        zone.zone_type in (
            SRLevelType.SUPPORT,
            SRLevelType.RESISTANCE,
        )
        for zone in zones
    )


def test_zone_price_order_is_valid():
    engine = SupportResistanceEngine()

    zones = engine.analyze(make_bars())

    assert all(
        zone.lower_price <= zone.center_price <= zone.upper_price
        for zone in zones
    )


def test_zone_width_is_non_negative():
    engine = SupportResistanceEngine()

    zones = engine.analyze(make_bars())

    assert all(
        zone.width >= 0.0
        for zone in zones
    )


def test_zone_strength_is_bounded():
    engine = SupportResistanceEngine()

    zones = engine.analyze(make_bars())

    assert all(
        0.0 <= zone.strength <= 100.0
        for zone in zones
    )


def test_zone_distance_is_non_negative():
    engine = SupportResistanceEngine()

    zones = engine.analyze(make_bars())

    assert all(
        zone.distance_from_price >= 0.0
        for zone in zones
    )


def test_zone_source_indices_are_sorted():
    engine = SupportResistanceEngine()

    zones = engine.analyze(make_bars())

    assert all(
        zone.source_indices
        == tuple(sorted(zone.source_indices))
        for zone in zones
    )


def test_zone_source_level_count_matches_indices():
    engine = SupportResistanceEngine()

    zones = engine.analyze(make_bars())

    assert all(
        zone.source_level_count == len(zone.source_indices)
        for zone in zones
    )


def test_zone_support_property():
    engine = SupportResistanceEngine()

    zones = engine.analyze(make_bars())

    for zone in zones:
        if zone.zone_type is SRLevelType.SUPPORT:
            assert zone.is_support
            assert not zone.is_resistance


def test_zone_resistance_property():
    engine = SupportResistanceEngine()

    zones = engine.analyze(make_bars())

    for zone in zones:
        if zone.zone_type is SRLevelType.RESISTANCE:
            assert zone.is_resistance
            assert not zone.is_support


def test_zone_contains_center_price():
    engine = SupportResistanceEngine()

    zones = engine.analyze(make_bars())

    assert all(
        zone.contains(zone.center_price)
        for zone in zones
    )


def test_zone_contains_lower_boundary():
    engine = SupportResistanceEngine()

    zones = engine.analyze(make_bars())

    assert all(
        zone.contains(zone.lower_price)
        for zone in zones
    )


def test_zone_contains_upper_boundary():
    engine = SupportResistanceEngine()

    zones = engine.analyze(make_bars())

    assert all(
        zone.contains(zone.upper_price)
        for zone in zones
    )


def test_zone_contains_rejects_invalid_price():
    engine = SupportResistanceEngine()

    zones = engine.analyze(make_bars())

    for zone in zones:
        assert zone.contains(float("nan")) is False
        assert zone.contains(float("inf")) is False
        assert zone.contains("invalid") is False
        assert zone.contains(True) is False


def test_active_property_for_non_broken_zone():
    engine = SupportResistanceEngine()

    zones = engine.analyze(make_bars())

    for zone in zones:
        if zone.status is not SRZoneStatus.BROKEN:
            assert zone.is_active


def test_broken_property():
    engine = SupportResistanceEngine()

    zones = engine.analyze(make_bars())

    for zone in zones:
        assert zone.is_broken == (
            zone.status is SRZoneStatus.BROKEN
        )


def test_nearest_support_returns_none_or_support():
    engine = SupportResistanceEngine()

    result = engine.nearest_support(make_bars())

    assert result is None or result.is_support


def test_nearest_resistance_returns_none_or_resistance():
    engine = SupportResistanceEngine()

    result = engine.nearest_resistance(make_bars())

    assert result is None or result.is_resistance


def test_nearest_support_is_below_or_at_price():
    bars = make_bars()

    result = SupportResistanceEngine().nearest_support(bars)

    if result is not None:
        assert result.center_price <= bars[-1].close


def test_nearest_resistance_is_above_or_at_price():
    bars = make_bars()

    result = SupportResistanceEngine().nearest_resistance(bars)

    if result is not None:
        assert result.center_price >= bars[-1].close


def test_xauusd_wrapper_accepts_xauusd():
    bars = make_bars()

    zones = SupportResistanceEngine().analyze_xauusd(bars)

    assert isinstance(zones, tuple)


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
        for index in range(15)
    ]

    with pytest.raises(SupportResistanceError):
        SupportResistanceEngine().analyze_xauusd(bars)


def test_max_zones_is_respected():
    engine = SupportResistanceEngine(
        max_zones=2,
    )

    zones = engine.analyze(make_bars())

    assert len(zones) <= 2


def test_min_strength_filter_is_respected():
    engine = SupportResistanceEngine(
        min_zone_strength=90.0,
    )

    zones = engine.analyze(make_bars())

    assert all(
        zone.strength >= 90.0
        for zone in zones
    )


def test_zones_are_ordered_by_distance():
    engine = SupportResistanceEngine()

    zones = engine.analyze(make_bars())

    distances = [
        zone.distance_from_price
        for zone in zones
    ]

    assert distances == sorted(distances)


def test_levels_are_ordered_by_distance():
    engine = SupportResistanceEngine()

    levels = engine.analyze_levels(make_bars())

    distances = [
        level.distance_from_price
        for level in levels
    ]

    assert distances == sorted(distances)


def test_result_is_deterministic():
    bars = make_bars()

    engine = SupportResistanceEngine()

    first = engine.analyze(bars)
    second = engine.analyze(bars)

    assert first == second


def test_level_result_is_deterministic():
    bars = make_bars()

    engine = SupportResistanceEngine()

    first = engine.analyze_levels(bars)
    second = engine.analyze_levels(bars)

    assert first == second