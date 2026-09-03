from datetime import datetime, timedelta

import pytest

from app.trading.data.market_bar import MarketBar
from app.trading.volume.volume_intelligence import (
    VolumeIntelligenceEngine,
    VolumeIntelligenceError,
    VolumeRegime,
)


BASE_TIME = datetime(2026, 1, 1)


def make_bar(
    index: int,
    close: float,
    volume: float,
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
        open=close - 0.5,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=volume,
    )


def make_bars(
    count: int = 30,
    volume: float = 100.0,
) -> list[MarketBar]:
    return [
        make_bar(
            index,
            2000.0 + index,
            volume,
        )
        for index in range(count)
    ]


def test_engine_defaults():
    engine = VolumeIntelligenceEngine()

    assert engine.volume_lookback == 20
    assert engine.profile_lookback == 100
    assert engine.bucket_count == 24
    assert engine.expansion_threshold == 1.20
    assert engine.contraction_threshold == 0.80
    assert engine.spike_threshold == 2.00
    assert engine.value_area_percent == 0.70


@pytest.mark.parametrize(
    "kwargs",
    [
        {"volume_lookback": 0},
        {"volume_lookback": -1},
        {"profile_lookback": 0},
        {"profile_lookback": -1},
        {"bucket_count": 0},
        {"bucket_count": 1},
    ],
)
def test_invalid_integer_configuration(kwargs):
    with pytest.raises(VolumeIntelligenceError):
        VolumeIntelligenceEngine(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expansion_threshold": 0},
        {"expansion_threshold": -1},
        {"contraction_threshold": 0},
        {"contraction_threshold": -1},
        {"spike_threshold": 0},
        {"spike_threshold": -1},
    ],
)
def test_invalid_ratio_configuration(kwargs):
    with pytest.raises(VolumeIntelligenceError):
        VolumeIntelligenceEngine(**kwargs)


def test_threshold_order_is_validated():
    with pytest.raises(VolumeIntelligenceError):
        VolumeIntelligenceEngine(
            contraction_threshold=1.0,
            expansion_threshold=0.8,
        )


def test_spike_threshold_must_exceed_expansion():
    with pytest.raises(VolumeIntelligenceError):
        VolumeIntelligenceEngine(
            expansion_threshold=1.5,
            spike_threshold=1.2,
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -0.1,
        1.1,
        float("inf"),
        float("nan"),
    ],
)
def test_invalid_value_area_percentage(value):
    with pytest.raises(VolumeIntelligenceError):
        VolumeIntelligenceEngine(
            value_area_percent=value,
        )


def test_empty_analysis():
    result = VolumeIntelligenceEngine().analyze([])

    assert result.volume_regime is VolumeRegime.UNKNOWN
    assert result.sufficient_data is False
    assert result.profile.poc is None


def test_invalid_bars_type():
    with pytest.raises(VolumeIntelligenceError):
        VolumeIntelligenceEngine().analyze("invalid")


def test_invalid_bar_type():
    with pytest.raises(VolumeIntelligenceError):
        VolumeIntelligenceEngine().analyze([object()])


def test_negative_volume_rejected():
    with pytest.raises(ValueError):
        make_bar(
            29,
            2030.0,
            -1.0,
        )


def test_mixed_symbols_rejected():
    bars = make_bars()

    bars[-1] = make_bar(
        len(bars) - 1,
        2030.0,
        100.0,
        symbol="EURUSD",
    )

    with pytest.raises(VolumeIntelligenceError):
        VolumeIntelligenceEngine().analyze(bars)


def test_mixed_timeframes_rejected():
    bars = make_bars()

    bars[-1] = make_bar(
        len(bars) - 1,
        2030.0,
        100.0,
        timeframe="H1",
    )

    with pytest.raises(VolumeIntelligenceError):
        VolumeIntelligenceEngine().analyze(bars)


def test_out_of_order_rejected():
    bars = make_bars()

    bars[-1], bars[-2] = (
        bars[-2],
        bars[-1],
    )

    with pytest.raises(VolumeIntelligenceError):
        VolumeIntelligenceEngine().analyze(bars)


def test_duplicate_timestamp_rejected():
    bars = make_bars()

    bars[-1] = MarketBar(
        timestamp=bars[-2].timestamp,
        symbol="XAUUSD",
        timeframe="M15",
        open=2029.5,
        high=2031.0,
        low=2029.0,
        close=2030.0,
        volume=100.0,
    )

    with pytest.raises(VolumeIntelligenceError):
        VolumeIntelligenceEngine().analyze(bars)


def test_relative_volume_normal():
    bars = make_bars()

    result = VolumeIntelligenceEngine().analyze(
        bars
    )

    assert result.relative_volume == pytest.approx(
        1.0
    )


def test_relative_volume_expanding():
    bars = make_bars()

    bars[-1] = make_bar(
        len(bars) - 1,
        2030.0,
        140.0,
    )

    result = VolumeIntelligenceEngine().analyze(
        bars
    )

    assert result.relative_volume > 1.2
    assert result.expanding is True


def test_relative_volume_contracting():
    bars = make_bars()

    bars[-1] = make_bar(
        len(bars) - 1,
        2030.0,
        50.0,
    )

    result = VolumeIntelligenceEngine().analyze(
        bars
    )

    assert result.relative_volume < 0.8
    assert result.contracting is True


def test_volume_spike():
    bars = make_bars()

    bars[-1] = make_bar(
        len(bars) - 1,
        2030.0,
        250.0,
    )

    result = VolumeIntelligenceEngine().analyze(
        bars
    )

    assert result.spike is True
    assert result.volume_regime is VolumeRegime.SPIKE


def test_normal_volume_regime():
    result = VolumeIntelligenceEngine().analyze(
        make_bars()
    )

    assert result.volume_regime is VolumeRegime.NORMAL


def test_expanding_volume_regime():
    bars = make_bars()

    bars[-1] = make_bar(
        len(bars) - 1,
        2030.0,
        140.0,
    )

    result = VolumeIntelligenceEngine().analyze(
        bars
    )

    assert result.volume_regime is VolumeRegime.EXPANDING


def test_contracting_volume_regime():
    bars = make_bars()

    bars[-1] = make_bar(
        len(bars) - 1,
        2030.0,
        50.0,
    )

    result = VolumeIntelligenceEngine().analyze(
        bars
    )

    assert result.volume_regime is VolumeRegime.CONTRACTING


def test_volume_strength_bounded():
    result = VolumeIntelligenceEngine().analyze(
        make_bars()
    )

    assert 0.0 <= result.volume_strength <= 100.0


def test_profile_exists_with_volume():
    result = VolumeIntelligenceEngine().analyze(
        make_bars()
    )

    assert result.profile.sufficient_data
    assert result.profile.total_volume > 0
    assert result.profile.poc is not None


def test_profile_poc_is_inside_price_range():
    bars = make_bars()

    result = VolumeIntelligenceEngine().analyze(
        bars
    )

    poc = result.profile.poc

    assert poc is not None
    assert bars[0].low <= poc <= bars[-1].high


def test_value_area_is_ordered():
    result = VolumeIntelligenceEngine().analyze(
        make_bars()
    )

    profile = result.profile

    assert (
        profile.value_area_low
        <= profile.value_area_high
    )


def test_poc_is_inside_value_area():
    result = VolumeIntelligenceEngine().analyze(
        make_bars()
    )

    profile = result.profile

    assert (
        profile.value_area_low
        <= profile.poc
        <= profile.value_area_high
    )


def test_profile_total_volume():
    bars = make_bars()

    result = VolumeIntelligenceEngine().volume_profile(
        bars
    )

    assert result.total_volume == pytest.approx(
        sum(bar.volume for bar in bars)
    )


def test_profile_bucket_count():
    result = VolumeIntelligenceEngine().volume_profile(
        make_bars()
    )

    assert result.bucket_count == 24


def test_profile_bucket_size_positive():
    result = VolumeIntelligenceEngine().volume_profile(
        make_bars()
    )

    assert result.bucket_size > 0


def test_high_volume_nodes_are_tuple():
    result = VolumeIntelligenceEngine().volume_profile(
        make_bars()
    )

    assert isinstance(
        result.high_volume_nodes,
        tuple,
    )


def test_low_volume_nodes_are_tuple():
    result = VolumeIntelligenceEngine().volume_profile(
        make_bars()
    )

    assert isinstance(
        result.low_volume_nodes,
        tuple,
    )


def test_price_location_inside_value():
    bars = make_bars()

    result = VolumeIntelligenceEngine().analyze(
        bars
    )

    assert result.price_location in {
        "INSIDE_VALUE",
        "ABOVE_VALUE",
        "BELOW_VALUE",
        "UNKNOWN",
    }


def test_price_location_is_deterministic():
    bars = make_bars()

    engine = VolumeIntelligenceEngine()

    first = engine.analyze(bars)
    second = engine.analyze(bars)

    assert first == second


def test_distance_from_poc_is_non_negative():
    result = VolumeIntelligenceEngine().analyze(
        make_bars()
    )

    if result.distance_from_poc is not None:
        assert result.distance_from_poc >= 0.0


def test_profile_price_location_validation():
    profile = VolumeIntelligenceEngine().volume_profile(
        make_bars()
    )

    assert profile.price_location(
        2030.0
    ) in {
        "INSIDE_VALUE",
        "ABOVE_VALUE",
        "BELOW_VALUE",
    }


def test_profile_price_location_rejects_nan():
    profile = VolumeIntelligenceEngine().volume_profile(
        make_bars()
    )

    with pytest.raises(VolumeIntelligenceError):
        profile.price_location(float("nan"))


def test_profile_price_location_rejects_inf():
    profile = VolumeIntelligenceEngine().volume_profile(
        make_bars()
    )

    with pytest.raises(VolumeIntelligenceError):
        profile.price_location(float("inf"))


def test_analyze_xauusd_accepts_xauusd():
    result = VolumeIntelligenceEngine().analyze_xauusd(
        make_bars()
    )

    assert result.symbol == "XAUUSD"


def test_analyze_xauusd_rejects_other_symbol():
    bars = [
        make_bar(
            index,
            2000.0 + index,
            100.0,
            symbol="EURUSD",
        )
        for index in range(30)
    ]

    with pytest.raises(VolumeIntelligenceError):
        VolumeIntelligenceEngine().analyze_xauusd(
            bars
        )


def test_relative_volume_method():
    bars = make_bars()

    result = VolumeIntelligenceEngine().relative_volume(
        bars
    )

    assert result == pytest.approx(1.0)


def test_relative_volume_empty():
    assert (
        VolumeIntelligenceEngine().relative_volume([])
        is None
    )


def test_profile_empty():
    profile = VolumeIntelligenceEngine().volume_profile(
        []
    )

    assert profile.poc is None
    assert profile.sufficient_data is False


def test_zero_volume_profile_is_insufficient():
    bars = make_bars(
        volume=0.0,
    )

    profile = VolumeIntelligenceEngine().volume_profile(
        bars
    )

    assert profile.sufficient_data is False
    assert profile.poc is None


def test_zero_volume_analysis_is_not_sufficient():
    bars = make_bars(
        volume=0.0,
    )

    result = VolumeIntelligenceEngine().analyze(
        bars
    )

    assert result.sufficient_data is False
    assert result.volume_regime is VolumeRegime.NORMAL


def test_profile_uses_recent_bars():
    engine = VolumeIntelligenceEngine(
        profile_lookback=5,
    )

    bars = make_bars(
        count=30,
    )

    profile = engine.volume_profile(
        bars
    )

    expected = sum(
        bar.volume
        for bar in bars[-5:]
    )

    assert profile.total_volume == pytest.approx(
        expected
    )


def test_volume_lookback_uses_recent_bars():
    engine = VolumeIntelligenceEngine(
        volume_lookback=5,
    )

    bars = make_bars(
        count=30,
    )

    result = engine.analyze(
        bars
    )

    assert result.average_volume == pytest.approx(
        100.0
    )


def test_current_volume_is_latest_bar():
    bars = make_bars()

    bars[-1] = make_bar(
        len(bars) - 1,
        2030.0,
        175.0,
    )

    result = VolumeIntelligenceEngine().analyze(
        bars
    )

    assert result.current_volume == pytest.approx(
        175.0
    )


def test_result_symbol_and_timeframe():
    result = VolumeIntelligenceEngine().analyze(
        make_bars()
    )

    assert result.symbol == "XAUUSD"
    assert result.timeframe == "M15"


def test_result_timestamp_matches_latest_bar():
    bars = make_bars()

    result = VolumeIntelligenceEngine().analyze(
        bars
    )

    assert result.timestamp == bars[-1].timestamp


def test_result_price_matches_latest_close():
    bars = make_bars()

    result = VolumeIntelligenceEngine().analyze(
        bars
    )

    assert result.price == pytest.approx(
        bars[-1].close
    )


def test_profile_is_valid_with_sufficient_volume():
    profile = VolumeIntelligenceEngine().volume_profile(
        make_bars()
    )

    assert profile.is_valid


def test_profile_has_poc():
    profile = VolumeIntelligenceEngine().volume_profile(
        make_bars()
    )

    assert profile.has_poc


def test_profile_has_value_area():
    profile = VolumeIntelligenceEngine().volume_profile(
        make_bars()
    )

    assert profile.has_value_area


def test_result_has_profile():
    result = VolumeIntelligenceEngine().analyze(
        make_bars()
    )

    assert result.has_profile


def test_unknown_result_properties():
    result = VolumeIntelligenceEngine().analyze([])

    assert result.is_unknown
    assert not result.is_expanding
    assert not result.is_contracting
    assert not result.is_spike


def test_normal_result_property():
    result = VolumeIntelligenceEngine().analyze(
        make_bars()
    )

    assert result.is_normal


def test_expanding_result_property():
    bars = make_bars()

    bars[-1] = make_bar(
        len(bars) - 1,
        2030.0,
        140.0,
    )

    result = VolumeIntelligenceEngine().analyze(
        bars
    )

    assert result.is_expanding


def test_contracting_result_property():
    bars = make_bars()

    bars[-1] = make_bar(
        len(bars) - 1,
        2030.0,
        50.0,
    )

    result = VolumeIntelligenceEngine().analyze(
        bars
    )

    assert result.is_contracting


def test_spike_result_property():
    bars = make_bars()

    bars[-1] = make_bar(
        len(bars) - 1,
        2030.0,
        250.0,
    )

    result = VolumeIntelligenceEngine().analyze(
        bars
    )

    assert result.is_spike