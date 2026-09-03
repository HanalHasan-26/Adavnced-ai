from datetime import datetime

import pytest

from app.trading.confluence.setup_confluence import (
    ConfluenceDirection,
    ConfluenceQuality,
    SetupConfluenceEngine,
    SetupConfluenceError,
    SetupConfluenceResult,
)
from app.trading.mtf.multi_timeframe import (
    MTFAlignment,
    MTFDirection,
    MultiTimeframeResult,
    TimeframeAnalysis,
    TimeframeRole,
)
from app.trading.context.market_context import ContextBias
from app.trading.regime.market_regime import MarketRegime
from app.trading.support_resistance.support_resistance import (
    SRLevelType,
    SRZone,
    SRZoneStatus,
)
from app.trading.trendline.trendline_engine import (
    Trendline,
    TrendlineStatus,
    TrendlineType,
)
from app.trading.volume.volume_intelligence import (
    VolumeIntelligence,
    VolumeProfile,
    VolumeRegime,
)


def make_timeframe(timeframe="H4", role=TimeframeRole.HIGHER,
                   bias=ContextBias.BULLISH, strength=80.0):
    return TimeframeAnalysis(
        timeframe=timeframe,
        role=role,
        timestamp=datetime(2026, 1, 1),
        bias=bias,
        strength=strength,
        regime=MarketRegime.TRENDING_UP,
        regime_strength=80.0,
        sufficient_data=True,
    )


def make_mtf(direction=MTFDirection.BULLISH,
             alignment=MTFAlignment.ALIGNED,
             score=90.0):
    higher = make_timeframe("H4", TimeframeRole.HIGHER)
    middle = make_timeframe(
        "H1", TimeframeRole.MIDDLE
    )
    lower = make_timeframe(
        "M15", TimeframeRole.LOWER
    )

    return MultiTimeframeResult(
        timestamp=datetime(2026, 1, 1),
        symbol="XAUUSD",
        higher=higher,
        middle=middle,
        lower=lower,
        direction=direction,
        alignment=alignment,
        alignment_score=score,
        strength=80.0,
        bullish_timeframes=3 if direction is MTFDirection.BULLISH else 0,
        bearish_timeframes=3 if direction is MTFDirection.BEARISH else 0,
        neutral_timeframes=0,
        unknown_timeframes=0,
        direction_conflict=alignment is MTFAlignment.CONFLICTED,
        sufficient_data=True,
        reasons=(),
        warnings=(),
    )


def make_zone(
    zone_type=SRLevelType.SUPPORT,
    strength=80.0,
):
    return SRZone(
        zone_type=zone_type,
        lower_price=2000.0,
        upper_price=2002.0,
        center_price=2001.0,
        width=2.0,
        touch_count=4,
        source_level_count=2,
        strength=strength,
        status=SRZoneStatus.ACTIVE,
        distance_from_price=5.0,
        first_timestamp=datetime(2026, 1, 1),
        last_timestamp=datetime(2026, 1, 1),
        source_indices=(1, 2),
    )


def make_trendline(
    trendline_type=TrendlineType.SUPPORT,
    strength=80.0,
):
    return Trendline(
        trendline_type=trendline_type,
        first_index=1,
        second_index=5,
        first_price=1990.0,
        second_price=1995.0,
        first_timestamp=datetime(2026, 1, 1),
        second_timestamp=datetime(2026, 1, 1),
        slope=1.25,
        intercept=1988.75,
        touch_count=3,
        rejection_count=2,
        strength=strength,
        status=TrendlineStatus.ACTIVE,
        distance_from_price=5.0,
        projected_price=2000.0,
        source_indices=(1, 5),
    )


def make_profile():
    return VolumeProfile(
        poc=2000.0,
        value_area_high=2002.0,
        value_area_low=1998.0,
        high_volume_nodes=(2000.0,),
        low_volume_nodes=(),
        total_volume=3000.0,
        bucket_size=1.0,
        bucket_count=24,
        sufficient_data=True,
    )


def make_volume(
    regime=VolumeRegime.EXPANDING,
    price_location="BELOW_VALUE",
):
    return VolumeIntelligence(
        timestamp=datetime(2026, 1, 1),
        symbol="XAUUSD",
        timeframe="M15",
        current_volume=150.0,
        average_volume=100.0,
        relative_volume=1.5,
        volume_regime=regime,
        volume_strength=75.0,
        expanding=regime is VolumeRegime.EXPANDING,
        contracting=regime is VolumeRegime.CONTRACTING,
        spike=regime is VolumeRegime.SPIKE,
        profile=make_profile(),
        price=1995.0,
        price_location=price_location,
        distance_from_poc=5.0,
        sufficient_data=True,
    )


def test_defaults():
    engine = SetupConfluenceEngine()
    assert engine.mtf_weight == 0.35
    assert engine.support_resistance_weight == 0.25
    assert engine.trendline_weight == 0.20
    assert engine.volume_weight == 0.20
    assert engine.minimum_score == 50.0
    assert engine.good_score == 65.0
    assert engine.strong_score == 80.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mtf_weight": -1},
        {"support_resistance_weight": -1},
        {"trendline_weight": -1},
        {"volume_weight": -1},
        {"mtf_weight": float("nan")},
        {"minimum_score": -1},
        {"minimum_score": 101},
        {"good_score": 101},
        {"strong_score": 101},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(SetupConfluenceError):
        SetupConfluenceEngine(**kwargs)


def test_weights_must_sum_to_one():
    with pytest.raises(SetupConfluenceError):
        SetupConfluenceEngine(
            mtf_weight=0.5,
            support_resistance_weight=0.5,
            trendline_weight=0.5,
            volume_weight=0.5,
        )


def test_score_order_is_validated():
    with pytest.raises(SetupConfluenceError):
        SetupConfluenceEngine(
            minimum_score=70,
            good_score=60,
            strong_score=80,
        )


def test_invalid_mtf():
    with pytest.raises(SetupConfluenceError):
        SetupConfluenceEngine().analyze(
            object(), (), (), make_volume()
        )


def test_invalid_sr_container():
    with pytest.raises(SetupConfluenceError):
        SetupConfluenceEngine().analyze(
            make_mtf(), [], (), make_volume()
        )


def test_invalid_trendline_container():
    with pytest.raises(SetupConfluenceError):
        SetupConfluenceEngine().analyze(
            make_mtf(), (), [], make_volume()
        )


def test_invalid_volume():
    with pytest.raises(SetupConfluenceError):
        SetupConfluenceEngine().analyze(
            make_mtf(), (), (), object()
        )


def test_invalid_sr_item():
    with pytest.raises(SetupConfluenceError):
        SetupConfluenceEngine().analyze(
            make_mtf(), (object(),), (), make_volume()
        )


def test_invalid_trendline_item():
    with pytest.raises(SetupConfluenceError):
        SetupConfluenceEngine().analyze(
            make_mtf(), (), (object(),), make_volume()
        )


def test_symbol_mismatch():
    volume = make_volume()
    volume = VolumeIntelligence(
        timestamp=volume.timestamp,
        symbol="EURUSD",
        timeframe=volume.timeframe,
        current_volume=volume.current_volume,
        average_volume=volume.average_volume,
        relative_volume=volume.relative_volume,
        volume_regime=volume.volume_regime,
        volume_strength=volume.volume_strength,
        expanding=volume.expanding,
        contracting=volume.contracting,
        spike=volume.spike,
        profile=volume.profile,
        price=volume.price,
        price_location=volume.price_location,
        distance_from_poc=volume.distance_from_poc,
        sufficient_data=volume.sufficient_data,
    )
    with pytest.raises(SetupConfluenceError):
        SetupConfluenceEngine().analyze(
            make_mtf(), (), (), volume
        )


def test_timeframe_mismatch():
    volume = make_volume()
    volume = VolumeIntelligence(
        timestamp=volume.timestamp,
        symbol=volume.symbol,
        timeframe="H1",
        current_volume=volume.current_volume,
        average_volume=volume.average_volume,
        relative_volume=volume.relative_volume,
        volume_regime=volume.volume_regime,
        volume_strength=volume.volume_strength,
        expanding=volume.expanding,
        contracting=volume.contracting,
        spike=volume.spike,
        profile=volume.profile,
        price=volume.price,
        price_location=volume.price_location,
        distance_from_poc=volume.distance_from_poc,
        sufficient_data=volume.sufficient_data,
    )
    with pytest.raises(SetupConfluenceError):
        SetupConfluenceEngine().analyze(
            make_mtf(), (), (), volume
        )


def test_bullish_confluence():
    result = SetupConfluenceEngine().analyze(
        make_mtf(),
        (make_zone(SRLevelType.SUPPORT),),
        (make_trendline(TrendlineType.SUPPORT),),
        make_volume(),
    )

    assert isinstance(result, SetupConfluenceResult)
    assert result.direction is ConfluenceDirection.BULLISH
    assert result.score > 0
    assert result.bullish_factors > 0
    assert result.support_present
    assert result.support_trendline_present
    assert result.volume_confirmed


def test_bearish_confluence():
    result = SetupConfluenceEngine().analyze(
        make_mtf(
            MTFDirection.BEARISH
        ),
        (make_zone(SRLevelType.RESISTANCE),),
        (make_trendline(TrendlineType.RESISTANCE),),
        make_volume(
            price_location="ABOVE_VALUE"
        ),
    )

    assert result.direction is ConfluenceDirection.BEARISH
    assert result.bearish_factors > 0
    assert result.resistance_present
    assert result.resistance_trendline_present


def test_neutral_mtf():
    result = SetupConfluenceEngine().analyze(
        make_mtf(
            MTFDirection.NEUTRAL,
            MTFAlignment.NEUTRAL,
            20.0,
        ),
        (),
        (),
        make_volume(
            regime=VolumeRegime.NORMAL,
            price_location="INSIDE_VALUE",
        ),
    )

    assert result.direction is ConfluenceDirection.NEUTRAL


def test_unknown_mtf():
    result = SetupConfluenceEngine().analyze(
        make_mtf(
            MTFDirection.UNKNOWN,
            MTFAlignment.UNKNOWN,
            0.0,
        ),
        (),
        (),
        make_volume(),
    )

    assert result.direction is ConfluenceDirection.UNKNOWN
    assert result.quality is ConfluenceQuality.UNKNOWN


def test_conflicted_mtf_warning():
    mtf = make_mtf(
        MTFDirection.BULLISH,
        MTFAlignment.CONFLICTED,
        45.0,
    )

    result = SetupConfluenceEngine().analyze(
        mtf,
        (
            make_zone(SRLevelType.RESISTANCE),
            make_zone(SRLevelType.SUPPORT),
        ),
        (
            make_trendline(TrendlineType.RESISTANCE),
            make_trendline(TrendlineType.SUPPORT),
        ),
        make_volume(),
    )

    assert any(
        "conflicted" in warning.lower()
        for warning in result.warnings
    )


def test_score_is_bounded():
    result = SetupConfluenceEngine().analyze(
        make_mtf(),
        (make_zone(),),
        (make_trendline(),),
        make_volume(),
    )

    assert 0.0 <= result.score <= 100.0


def test_component_scores_are_bounded():
    result = SetupConfluenceEngine().analyze(
        make_mtf(),
        (make_zone(),),
        (make_trendline(),),
        make_volume(),
    )

    assert 0.0 <= result.mtf_score <= 100.0
    assert 0.0 <= result.support_resistance_score <= 100.0
    assert 0.0 <= result.trendline_score <= 100.0
    assert 0.0 <= result.volume_score <= 100.0


def test_result_flags():
    result = SetupConfluenceEngine().analyze(
        make_mtf(),
        (make_zone(),),
        (make_trendline(),),
        make_volume(),
    )

    assert result.is_bullish
    assert not result.is_bearish
    assert not result.is_unknown


def test_result_reasons_exist():
    result = SetupConfluenceEngine().analyze(
        make_mtf(),
        (make_zone(),),
        (make_trendline(),),
        make_volume(),
    )

    assert result.reasons


def test_result_is_immutable():
    result = SetupConfluenceEngine().analyze(
        make_mtf(),
        (),
        (),
        make_volume(),
    )

    with pytest.raises(AttributeError):
        result.score = 10.0


def test_empty_confluence_is_weak_or_unknown():
    result = SetupConfluenceEngine().analyze(
        make_mtf(
            MTFDirection.UNKNOWN,
            MTFAlignment.UNKNOWN,
            0.0,
        ),
        (),
        (),
        make_volume(),
    )

    assert result.quality is ConfluenceQuality.UNKNOWN


def test_insufficient_volume_generates_warning():
    volume = make_volume()
    volume = VolumeIntelligence(
        timestamp=volume.timestamp,
        symbol=volume.symbol,
        timeframe=volume.timeframe,
        current_volume=volume.current_volume,
        average_volume=volume.average_volume,
        relative_volume=volume.relative_volume,
        volume_regime=volume.volume_regime,
        volume_strength=volume.volume_strength,
        expanding=volume.expanding,
        contracting=volume.contracting,
        spike=volume.spike,
        profile=volume.profile,
        price=volume.price,
        price_location=volume.price_location,
        distance_from_poc=volume.distance_from_poc,
        sufficient_data=False,
    )

    result = SetupConfluenceEngine().analyze(
        make_mtf(),
        (),
        (),
        volume,
    )

    assert result.sufficient_data is False
    assert any(
        "volume" in warning.lower()
        for warning in result.warnings
    )


def test_xauusd_wrapper():
    result = SetupConfluenceEngine().analyze_xauusd(
        make_mtf(),
        (),
        (),
        make_volume(),
    )

    assert result.symbol == "XAUUSD"


def test_xauusd_wrapper_rejects_other_symbol():
    mtf = make_mtf()
    bad_mtf = MultiTimeframeResult(
        timestamp=mtf.timestamp,
        symbol="EURUSD",
        higher=mtf.higher,
        middle=mtf.middle,
        lower=mtf.lower,
        direction=mtf.direction,
        alignment=mtf.alignment,
        alignment_score=mtf.alignment_score,
        strength=mtf.strength,
        bullish_timeframes=mtf.bullish_timeframes,
        bearish_timeframes=mtf.bearish_timeframes,
        neutral_timeframes=mtf.neutral_timeframes,
        unknown_timeframes=mtf.unknown_timeframes,
        direction_conflict=mtf.direction_conflict,
        sufficient_data=mtf.sufficient_data,
        reasons=mtf.reasons,
        warnings=mtf.warnings,
    )

    with pytest.raises(SetupConfluenceError):
        SetupConfluenceEngine().analyze_xauusd(
            bad_mtf,
            (),
            (),
            make_volume(),
        )
