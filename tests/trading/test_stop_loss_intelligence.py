from datetime import datetime

import pytest

from app.trading.entry.entry_model import (
    EntryDirection,
    EntryModel,
    EntryQuality,
    EntryTrigger,
)
from app.trading.risk.stop_loss_intelligence import (
    StopLossIntelligenceEngine,
    StopLossIntelligenceError,
    StopLossMethod,
    StopLossQuality,
    StopLossReasonType,
)


def make_entry(
    direction=EntryDirection.LONG,
    entry_price=2000.0,
    symbol="XAUUSD",
    timeframe="M15",
):
    return EntryModel(
        timestamp=datetime(2026, 1, 1),
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        trigger=EntryTrigger.MARKET,
        quality=EntryQuality.STRONG,
        reference_price=entry_price,
        entry_price=entry_price,
        confluence_score=90.0,
        entry_confidence=90.0,
        support_present=direction is EntryDirection.LONG,
        resistance_present=direction is EntryDirection.SHORT,
        trendline_present=True,
        volume_confirmed=True,
        mtf_confirmed=True,
        valid=True,
        entry_allowed=True,
        reasons=(),
        warnings=(),
    )


def test_defaults():
    engine = StopLossIntelligenceEngine()

    assert engine.minimum_distance == 0.01
    assert engine.maximum_risk_percent == 5.0
    assert engine.atr_buffer_multiplier == 0.25


def test_long_structure_stop():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(),
        structural_level=1980.0,
    )

    assert result.direction is EntryDirection.LONG
    assert result.stop_loss is not None
    assert result.stop_loss < result.entry_price
    assert result.valid
    assert result.stop_loss_ready


def test_short_structure_stop():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(
            direction=EntryDirection.SHORT,
        ),
        structural_level=2020.0,
    )

    assert result.direction is EntryDirection.SHORT
    assert result.stop_loss is not None
    assert result.stop_loss > result.entry_price
    assert result.valid
    assert result.stop_loss_ready


def test_long_support_stop():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(),
        support_level=1985.0,
    )

    assert result.stop_loss == pytest.approx(
        1985.0 - 1985.0 * 0.0,
    )
    assert result.valid


def test_short_resistance_stop():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(
            direction=EntryDirection.SHORT,
        ),
        resistance_level=2015.0,
    )

    assert result.stop_loss == pytest.approx(
        2015.0,
    )
    assert result.valid


def test_long_trendline_stop():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(),
        trendline_level=1990.0,
    )

    assert result.stop_loss is not None
    assert result.stop_loss < result.entry_price
    assert result.valid
    assert result.method is StopLossMethod.TRENDLINE


def test_short_trendline_stop():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(
            direction=EntryDirection.SHORT,
        ),
        trendline_level=2010.0,
    )

    assert result.stop_loss is not None
    assert result.stop_loss > result.entry_price
    assert result.valid


def test_atr_buffer_long():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(),
        support_level=1980.0,
        atr_value=20.0,
    )

    assert result.stop_loss == pytest.approx(
        1975.0,
    )
    assert result.atr_buffer == pytest.approx(
        5.0,
    )


def test_atr_buffer_short():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(
            direction=EntryDirection.SHORT,
        ),
        resistance_level=2020.0,
        atr_value=20.0,
    )

    assert result.stop_loss == pytest.approx(
        2025.0,
    )
    assert result.atr_buffer == pytest.approx(
        5.0,
    )


def test_structural_level_preferred_over_support_when_valid():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(),
        structural_level=1985.0,
        support_level=1970.0,
    )

    assert result.stop_loss == pytest.approx(
        1985.0,
    )


def test_nearest_valid_long_level_is_selected():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(),
        structural_level=1970.0,
        support_level=1985.0,
    )

    assert result.stop_loss == pytest.approx(
        1985.0,
    )


def test_nearest_valid_short_level_is_selected():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(
            direction=EntryDirection.SHORT,
        ),
        structural_level=2030.0,
        resistance_level=2015.0,
    )

    assert result.stop_loss == pytest.approx(
        2015.0,
    )


def test_long_wrong_side_level_is_rejected():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(),
        structural_level=2010.0,
    )

    assert result.valid is False
    assert result.stop_loss is None


def test_short_wrong_side_level_is_rejected():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(
            direction=EntryDirection.SHORT,
        ),
        structural_level=1990.0,
    )

    assert result.valid is False
    assert result.stop_loss is None


def test_no_level_blocks_stop_loss():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(),
    )

    assert result.stop_loss is None
    assert result.valid is False
    assert result.stop_loss_ready is False
    assert result.quality is StopLossQuality.INVALID


def test_none_direction_blocks():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(
            direction=EntryDirection.NONE,
        ),
        structural_level=1980.0,
    )

    assert result.valid is False
    assert result.stop_loss is None


def test_unknown_direction_blocks():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(
            direction=EntryDirection.UNKNOWN,
        ),
        structural_level=1980.0,
    )

    assert result.valid is False


def test_risk_distance_long():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(
            entry_price=2000.0,
        ),
        structural_level=1980.0,
    )

    assert result.risk_distance == pytest.approx(
        20.0,
    )


def test_risk_distance_short():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(
            direction=EntryDirection.SHORT,
            entry_price=2000.0,
        ),
        structural_level=2020.0,
    )

    assert result.risk_distance == pytest.approx(
        20.0,
    )


def test_risk_percent():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(
            entry_price=2000.0,
        ),
        structural_level=1980.0,
    )

    assert result.risk_percent_of_entry == pytest.approx(
        1.0,
    )


def test_xauusd_wrapper():
    result = StopLossIntelligenceEngine().analyze_xauusd(
        make_entry(),
        structural_level=1980.0,
    )

    assert result.symbol == "XAUUSD"


def test_xauusd_wrapper_rejects_other_symbol():
    with pytest.raises(StopLossIntelligenceError):
        StopLossIntelligenceEngine().analyze_xauusd(
            make_entry(symbol="EURUSD"),
            structural_level=1.09,
        )


def test_reference_price_override():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(entry_price=2000.0),
        structural_level=1980.0,
        reference_price=2050.0,
    )

    assert result.entry_price == pytest.approx(
        2050.0,
    )


def test_invalid_reference_price():
    with pytest.raises(StopLossIntelligenceError):
        StopLossIntelligenceEngine().analyze(
            make_entry(),
            structural_level=1980.0,
            reference_price=-1.0,
        )


def test_invalid_atr():
    with pytest.raises(StopLossIntelligenceError):
        StopLossIntelligenceEngine().analyze(
            make_entry(),
            structural_level=1980.0,
            atr_value=-1.0,
        )


def test_invalid_support():
    with pytest.raises(StopLossIntelligenceError):
        StopLossIntelligenceEngine().analyze(
            make_entry(),
            support_level=0.0,
        )


def test_invalid_engine_minimum_distance():
    with pytest.raises(StopLossIntelligenceError):
        StopLossIntelligenceEngine(
            minimum_distance=0.0,
        )


def test_invalid_engine_maximum_risk():
    with pytest.raises(StopLossIntelligenceError):
        StopLossIntelligenceEngine(
            maximum_risk_percent=0.0,
        )


def test_invalid_atr_multiplier():
    with pytest.raises(StopLossIntelligenceError):
        StopLossIntelligenceEngine(
            atr_buffer_multiplier=-1.0,
        )


def test_quality_is_bounded():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(),
        structural_level=1980.0,
    )

    assert 0.0 <= result.quality_score <= 100.0


def test_structure_quality():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(),
        structural_level=1980.0,
    )

    assert result.quality in (
        StopLossQuality.EXCELLENT,
        StopLossQuality.GOOD,
        StopLossQuality.ACCEPTABLE,
        StopLossQuality.WEAK,
    )


def test_reasons_are_present():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(),
        structural_level=1980.0,
    )

    assert result.reasons


def test_warnings_are_present():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(),
        structural_level=1980.0,
    )

    assert result.warnings


def test_long_reason_present():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(),
        structural_level=1980.0,
    )

    assert any(
        reason.reason_type
        is StopLossReasonType.LONG_DIRECTION
        for reason in result.reasons
    )


def test_short_reason_present():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(
            direction=EntryDirection.SHORT,
        ),
        structural_level=2020.0,
    )

    assert any(
        reason.reason_type
        is StopLossReasonType.SHORT_DIRECTION
        for reason in result.reasons
    )


def test_atr_reason_present():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(),
        support_level=1980.0,
        atr_value=20.0,
    )

    assert any(
        reason.reason_type
        is StopLossReasonType.ATR_BUFFER
        for reason in result.reasons
    )


def test_result_is_immutable():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(),
        structural_level=1980.0,
    )

    with pytest.raises(AttributeError):
        result.stop_loss = 1900.0


def test_result_properties():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(),
        structural_level=1980.0,
    )

    assert result.is_long
    assert not result.is_short
    assert result.is_valid
    assert result.has_stop_loss
    assert result.is_ready
    assert result.risk is not None


def test_short_result_properties():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(
            direction=EntryDirection.SHORT,
        ),
        structural_level=2020.0,
    )

    assert result.is_short
    assert not result.is_long
    assert result.is_valid


def test_no_stop_result_properties():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(),
    )

    assert not result.has_stop_loss
    assert not result.is_ready
    assert result.risk is None


def test_engine_is_deterministic():
    engine = StopLossIntelligenceEngine()

    entry = make_entry()

    first = engine.analyze(
        entry,
        structural_level=1980.0,
        atr_value=20.0,
    )

    second = engine.analyze(
        entry,
        structural_level=1980.0,
        atr_value=20.0,
    )

    assert first == second


def test_timestamp_preserved():
    entry = make_entry()

    result = StopLossIntelligenceEngine().analyze(
        entry,
        structural_level=1980.0,
    )

    assert result.timestamp == entry.timestamp


def test_symbol_preserved():
    entry = make_entry()

    result = StopLossIntelligenceEngine().analyze(
        entry,
        structural_level=1980.0,
    )

    assert result.symbol == entry.symbol


def test_timeframe_preserved():
    entry = make_entry()

    result = StopLossIntelligenceEngine().analyze(
        entry,
        structural_level=1980.0,
    )

    assert result.timeframe == entry.timeframe


def test_stop_is_below_long_entry():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(),
        structural_level=1980.0,
    )

    assert result.stop_loss < result.entry_price


def test_stop_is_above_short_entry():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(
            direction=EntryDirection.SHORT,
        ),
        structural_level=2020.0,
    )

    assert result.stop_loss > result.entry_price


def test_does_not_calculate_position_size():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(),
        structural_level=1980.0,
    )

    assert not hasattr(result, "position_size")


def test_does_not_calculate_take_profit():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(),
        structural_level=1980.0,
    )

    assert not hasattr(result, "take_profit")


def test_hybrid_when_structure_and_support_agree():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(),
        structural_level=1980.0,
        support_level=1980.0,
    )

    assert result.method is StopLossMethod.HYBRID


def test_hybrid_when_structure_and_resistance_agree():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(
            direction=EntryDirection.SHORT,
        ),
        structural_level=2020.0,
        resistance_level=2020.0,
    )

    assert result.method is StopLossMethod.HYBRID


def test_atr_without_structural_reference_does_not_create_stop():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(),
        atr_value=20.0,
    )

    assert result.stop_loss is None
    assert result.valid is False


def test_support_long_level_below_entry_is_valid():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(
            entry_price=2000.0,
        ),
        support_level=1990.0,
    )

    assert result.valid
    assert result.stop_loss == pytest.approx(
        1990.0,
    )


def test_resistance_short_level_above_entry_is_valid():
    result = StopLossIntelligenceEngine().analyze(
        make_entry(
            direction=EntryDirection.SHORT,
            entry_price=2000.0,
        ),
        resistance_level=2010.0,
    )

    assert result.valid
    assert result.stop_loss == pytest.approx(
        2010.0,
    )