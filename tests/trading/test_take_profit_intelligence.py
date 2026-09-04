from datetime import datetime, timedelta

import pytest

from app.trading.entry.entry_model import (
    EntryDirection,
    EntryModel,
    EntryQuality,
    EntryTrigger,
)
from app.trading.risk.take_profit_intelligence import (
    TakeProfitIntelligenceEngine,
    TakeProfitIntelligenceError,
    TakeProfitMethod,
    TakeProfitQuality,
    TakeProfitReasonType,
)


def make_entry(
    direction=EntryDirection.LONG,
    entry_price=2030.0,
    symbol="XAUUSD",
    timeframe="M15",
):
    return EntryModel(
        timestamp=datetime(2026, 1, 1, 7, 15),
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        trigger=EntryTrigger.MARKET,
        quality=EntryQuality.GOOD,
        reference_price=entry_price,
        entry_price=entry_price,
        confluence_score=80.0,
        entry_confidence=80.0,
        support_present=False,
        resistance_present=False,
        trendline_present=False,
        volume_confirmed=False,
        mtf_confirmed=False,
        valid=True,
        entry_allowed=True,
        reasons=(),
        warnings=(),
    )


def test_long_rr_target():
    engine = TakeProfitIntelligenceEngine()

    entry = make_entry(
        EntryDirection.LONG,
        2030.0,
    )

    result = engine.analyze(
        entry,
        stop_loss=2020.0,
    )

    assert result.valid is True
    assert result.take_profit == 2050.0
    assert result.risk_distance == 10.0
    assert result.reward_distance == 20.0
    assert result.risk_reward_ratio == 2.0
    assert result.method is TakeProfitMethod.RISK_REWARD
    assert result.take_profit_ready is True


def test_short_rr_target():
    engine = TakeProfitIntelligenceEngine()

    entry = make_entry(
        EntryDirection.SHORT,
        2030.0,
    )

    result = engine.analyze(
        entry,
        stop_loss=2040.0,
    )

    assert result.valid is True
    assert result.take_profit == 2010.0
    assert result.risk_distance == 10.0
    assert result.reward_distance == 20.0
    assert result.risk_reward_ratio == 2.0
    assert result.take_profit_ready is True


def test_long_structural_target_is_used_when_beyond_minimum_rr():
    engine = TakeProfitIntelligenceEngine()

    entry = make_entry(
        EntryDirection.LONG,
        2030.0,
    )

    result = engine.analyze(
        entry,
        stop_loss=2020.0,
        structural_level=2060.0,
    )

    assert result.valid is True
    assert result.take_profit == 2060.0
    assert result.method is TakeProfitMethod.STRUCTURE
    assert result.risk_reward_ratio == 3.0


def test_short_structural_target_is_used_when_beyond_minimum_rr():
    engine = TakeProfitIntelligenceEngine()

    entry = make_entry(
        EntryDirection.SHORT,
        2030.0,
    )

    result = engine.analyze(
        entry,
        stop_loss=2040.0,
        structural_level=2000.0,
    )

    assert result.valid is True
    assert result.take_profit == 2000.0
    assert result.method is TakeProfitMethod.STRUCTURE
    assert result.risk_reward_ratio == 3.0


def test_long_resistance_target():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(),
        stop_loss=2020.0,
        resistance_level=2055.0,
    )

    assert result.valid is True
    assert result.take_profit == 2055.0
    assert result.method is TakeProfitMethod.SUPPORT_RESISTANCE
    assert result.risk_reward_ratio == 2.5


def test_short_support_target():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(EntryDirection.SHORT),
        stop_loss=2040.0,
        support_level=2005.0,
    )

    assert result.valid is True
    assert result.take_profit == 2005.0
    assert result.method is TakeProfitMethod.SUPPORT_RESISTANCE
    assert result.risk_reward_ratio == 2.5


def test_long_trendline_target():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(),
        stop_loss=2020.0,
        trendline_level=2060.0,
    )

    assert result.valid is True
    assert result.take_profit == 2060.0
    assert result.method is TakeProfitMethod.TRENDLINE


def test_short_trendline_target():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(EntryDirection.SHORT),
        stop_loss=2040.0,
        trendline_level=2000.0,
    )

    assert result.valid is True
    assert result.take_profit == 2000.0
    assert result.method is TakeProfitMethod.TRENDLINE


def test_long_uses_rr_when_structural_target_is_too_close():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(),
        stop_loss=2020.0,
        structural_level=2040.0,
    )

    assert result.valid is True
    assert result.take_profit == 2050.0
    assert result.method is TakeProfitMethod.RISK_REWARD


def test_short_uses_rr_when_structural_target_is_too_close():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(EntryDirection.SHORT),
        stop_loss=2040.0,
        structural_level=2020.0,
    )

    assert result.valid is True
    assert result.take_profit == 2010.0
    assert result.method is TakeProfitMethod.RISK_REWARD


def test_long_ignores_resistance_below_entry():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(),
        stop_loss=2020.0,
        resistance_level=2025.0,
    )

    assert result.valid is True
    assert result.take_profit == 2050.0
    assert result.method is TakeProfitMethod.RISK_REWARD


def test_short_ignores_support_above_entry():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(EntryDirection.SHORT),
        stop_loss=2040.0,
        support_level=2035.0,
    )

    assert result.valid is True
    assert result.take_profit == 2010.0
    assert result.method is TakeProfitMethod.RISK_REWARD


def test_long_wrong_side_structural_target_falls_back_to_rr():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(),
        stop_loss=2020.0,
        structural_level=2010.0,
    )

    assert result.valid is True
    assert result.take_profit == 2050.0
    assert result.method is TakeProfitMethod.RISK_REWARD


def test_short_wrong_side_structural_target_falls_back_to_rr():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(EntryDirection.SHORT),
        stop_loss=2040.0,
        structural_level=2050.0,
    )

    assert result.valid is True
    assert result.take_profit == 2010.0
    assert result.method is TakeProfitMethod.RISK_REWARD


def test_custom_minimum_rr():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(),
        stop_loss=2020.0,
        minimum_risk_reward=3.0,
    )

    assert result.valid is True
    assert result.take_profit == 2060.0
    assert result.risk_reward_ratio == 3.0
    assert result.minimum_risk_reward == 3.0


def test_custom_minimum_rr_short():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(EntryDirection.SHORT),
        stop_loss=2040.0,
        minimum_risk_reward=3.0,
    )

    assert result.valid is True
    assert result.take_profit == 2000.0
    assert result.risk_reward_ratio == 3.0


def test_hybrid_long_target():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(),
        stop_loss=2020.0,
        structural_level=2060.0,
        resistance_level=2060.0,
    )

    assert result.valid is True
    assert result.take_profit == 2060.0
    assert result.method is TakeProfitMethod.HYBRID


def test_hybrid_short_target():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(EntryDirection.SHORT),
        stop_loss=2040.0,
        structural_level=2000.0,
        support_level=2000.0,
    )

    assert result.valid is True
    assert result.take_profit == 2000.0
    assert result.method is TakeProfitMethod.HYBRID


def test_long_multiple_targets_selects_nearest_valid_target():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(),
        stop_loss=2020.0,
        structural_level=2070.0,
        resistance_level=2055.0,
        trendline_level=2060.0,
    )

    assert result.valid is True
    assert result.take_profit == 2055.0
    assert result.method is TakeProfitMethod.SUPPORT_RESISTANCE


def test_short_multiple_targets_selects_nearest_valid_target():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(EntryDirection.SHORT),
        stop_loss=2040.0,
        structural_level=1990.0,
        support_level=2005.0,
        trendline_level=2000.0,
    )

    assert result.valid is True
    assert result.take_profit == 2005.0
    assert result.method is TakeProfitMethod.SUPPORT_RESISTANCE


def test_long_entry_price_override():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(
            EntryDirection.LONG,
            entry_price=2030.0,
        ),
        stop_loss=2020.0,
        reference_price=2040.0,
    )

    assert result.entry_price == 2040.0
    assert result.take_profit == 2060.0


def test_short_entry_price_override():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(
            EntryDirection.SHORT,
            entry_price=2030.0,
        ),
        stop_loss=2040.0,
        reference_price=2040.0,
    )

    assert result.entry_price == 2040.0
    assert result.take_profit == 2020.0


def test_missing_stop_loss_is_blocked():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(),
    )

    assert result.valid is False
    assert result.take_profit is None
    assert result.take_profit_ready is False
    assert result.method is TakeProfitMethod.NONE
    assert result.quality is TakeProfitQuality.INVALID

    assert any(
        reason.reason_type
        is TakeProfitReasonType.INVALID_STOP_LOSS
        for reason in result.reasons
    )


def test_invalid_stop_loss_for_long_is_blocked():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(EntryDirection.LONG),
        stop_loss=2040.0,
    )

    assert result.valid is False
    assert result.take_profit is None
    assert result.take_profit_ready is False


def test_invalid_stop_loss_for_short_is_blocked():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(EntryDirection.SHORT),
        stop_loss=2020.0,
    )

    assert result.valid is False
    assert result.take_profit is None
    assert result.take_profit_ready is False


def test_stop_loss_too_close_is_blocked():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(),
        stop_loss=2029.999,
    )

    assert result.valid is False
    assert result.take_profit is None


def test_invalid_direction_is_blocked():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(EntryDirection.NONE),
        stop_loss=2020.0,
    )

    assert result.valid is False
    assert result.take_profit is None
    assert result.method is TakeProfitMethod.NONE


def test_unknown_direction_is_blocked():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(EntryDirection.UNKNOWN),
        stop_loss=2020.0,
    )

    assert result.valid is False
    assert result.take_profit is None


def test_non_entry_model_is_rejected():
    engine = TakeProfitIntelligenceEngine()

    with pytest.raises(TakeProfitIntelligenceError):
        engine.analyze(
            object(),
            stop_loss=2020.0,
        )


def test_empty_symbol_is_rejected():
    engine = TakeProfitIntelligenceEngine()

    entry = make_entry()
    entry = EntryModel(
        timestamp=entry.timestamp,
        symbol="",
        timeframe=entry.timeframe,
        direction=entry.direction,
        entry_price=entry.entry_price,
        reference_price=entry.reference_price,
        model=entry.model,
        confidence=entry.confidence,
        valid=entry.valid,
        entry_ready=entry.entry_ready,
        reasons=entry.reasons,
        warnings=entry.warnings,
    )

    with pytest.raises(TakeProfitIntelligenceError):
        engine.analyze(
            entry,
            stop_loss=2020.0,
        )


def test_invalid_minimum_rr():
    with pytest.raises(TakeProfitIntelligenceError):
        TakeProfitIntelligenceEngine(
            minimum_risk_reward=0.0,
        )


def test_invalid_minimum_distance():
    with pytest.raises(TakeProfitIntelligenceError):
        TakeProfitIntelligenceEngine(
            minimum_distance=0.0,
        )


def test_invalid_maximum_rr():
    with pytest.raises(TakeProfitIntelligenceError):
        TakeProfitIntelligenceEngine(
            maximum_risk_reward=0.0,
        )


def test_minimum_rr_cannot_exceed_maximum_rr():
    with pytest.raises(TakeProfitIntelligenceError):
        TakeProfitIntelligenceEngine(
            minimum_risk_reward=5.0,
            maximum_risk_reward=4.0,
        )


def test_invalid_quality_threshold_order():
    with pytest.raises(TakeProfitIntelligenceError):
        TakeProfitIntelligenceEngine(
            excellent_score=50.0,
            good_score=70.0,
            acceptable_score=60.0,
        )


def test_boolean_minimum_rr_rejected():
    with pytest.raises(TakeProfitIntelligenceError):
        TakeProfitIntelligenceEngine(
            minimum_risk_reward=True,
        )


def test_boolean_minimum_distance_rejected():
    with pytest.raises(TakeProfitIntelligenceError):
        TakeProfitIntelligenceEngine(
            minimum_distance=True,
        )


def test_nan_minimum_rr_rejected():
    with pytest.raises(TakeProfitIntelligenceError):
        TakeProfitIntelligenceEngine(
            minimum_risk_reward=float("nan"),
        )


def test_infinite_minimum_rr_rejected():
    with pytest.raises(TakeProfitIntelligenceError):
        TakeProfitIntelligenceEngine(
            minimum_risk_reward=float("inf"),
        )


def test_nan_stop_loss_rejected():
    with pytest.raises(TakeProfitIntelligenceError):
        TakeProfitIntelligenceEngine().analyze(
            make_entry(),
            stop_loss=float("nan"),
        )


def test_infinite_stop_loss_rejected():
    with pytest.raises(TakeProfitIntelligenceError):
        TakeProfitIntelligenceEngine().analyze(
            make_entry(),
            stop_loss=float("inf"),
        )


def test_zero_stop_loss_rejected():
    with pytest.raises(TakeProfitIntelligenceError):
        TakeProfitIntelligenceEngine().analyze(
            make_entry(),
            stop_loss=0.0,
        )


def test_negative_stop_loss_rejected():
    with pytest.raises(TakeProfitIntelligenceError):
        TakeProfitIntelligenceEngine().analyze(
            make_entry(),
            stop_loss=-1.0,
        )


def test_negative_structural_level_rejected():
    with pytest.raises(TakeProfitIntelligenceError):
        TakeProfitIntelligenceEngine().analyze(
            make_entry(),
            stop_loss=2020.0,
            structural_level=-1.0,
        )


def test_zero_resistance_rejected():
    with pytest.raises(TakeProfitIntelligenceError):
        TakeProfitIntelligenceEngine().analyze(
            make_entry(),
            stop_loss=2020.0,
            resistance_level=0.0,
        )


def test_negative_support_rejected():
    with pytest.raises(TakeProfitIntelligenceError):
        TakeProfitIntelligenceEngine().analyze(
            make_entry(),
            stop_loss=2020.0,
            support_level=-5.0,
        )


def test_xauusd_wrapper():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze_xauusd(
        make_entry(),
        stop_loss=2020.0,
    )

    assert result.valid is True
    assert result.symbol == "XAUUSD"
    assert result.take_profit == 2050.0


def test_xauusd_wrapper_rejects_other_symbol():
    engine = TakeProfitIntelligenceEngine()

    entry = make_entry(
        symbol="EURUSD",
    )

    with pytest.raises(TakeProfitIntelligenceError):
        engine.analyze_xauusd(
            entry,
            stop_loss=2020.0,
        )


def test_is_long_property():
    result = TakeProfitIntelligenceEngine().analyze(
        make_entry(EntryDirection.LONG),
        stop_loss=2020.0,
    )

    assert result.is_long is True
    assert result.is_short is False


def test_is_short_property():
    result = TakeProfitIntelligenceEngine().analyze(
        make_entry(EntryDirection.SHORT),
        stop_loss=2040.0,
    )

    assert result.is_long is False
    assert result.is_short is True


def test_has_take_profit_property():
    result = TakeProfitIntelligenceEngine().analyze(
        make_entry(),
        stop_loss=2020.0,
    )

    assert result.has_take_profit is True


def test_is_ready_property():
    result = TakeProfitIntelligenceEngine().analyze(
        make_entry(),
        stop_loss=2020.0,
    )

    assert result.is_ready is True


def test_reward_alias():
    result = TakeProfitIntelligenceEngine().analyze(
        make_entry(),
        stop_loss=2020.0,
    )

    assert result.reward == result.reward_distance


def test_rr_alias():
    result = TakeProfitIntelligenceEngine().analyze(
        make_entry(),
        stop_loss=2020.0,
    )

    assert result.rr == result.risk_reward_ratio


def test_take_profit_is_rounded():
    engine = TakeProfitIntelligenceEngine()

    result = engine.analyze(
        make_entry(
            EntryDirection.LONG,
            2030.123456789,
        ),
        stop_loss=2020.123456789,
    )

    assert result.entry_price == round(
        2030.123456789,
        10,
    )


def test_quality_is_at_least_acceptable_for_standard_rr():
    result = TakeProfitIntelligenceEngine().analyze(
        make_entry(),
        stop_loss=2020.0,
    )

    assert result.quality in (
        TakeProfitQuality.ACCEPTABLE,
        TakeProfitQuality.GOOD,
        TakeProfitQuality.EXCELLENT,
    )


def test_structural_target_can_produce_excellent_quality():
    result = TakeProfitIntelligenceEngine().analyze(
        make_entry(),
        stop_loss=2020.0,
        structural_level=2060.0,
        resistance_level=2060.0,
    )

    assert result.quality in (
        TakeProfitQuality.GOOD,
        TakeProfitQuality.EXCELLENT,
    )
    assert result.quality_score >= 70.0


def test_high_rr_increases_quality():
    result = TakeProfitIntelligenceEngine().analyze(
        make_entry(),
        stop_loss=2020.0,
        minimum_risk_reward=4.0,
    )

    assert result.risk_reward_ratio == 4.0
    assert result.quality_score >= 50.0


def test_reasons_are_present_for_valid_target():
    result = TakeProfitIntelligenceEngine().analyze(
        make_entry(),
        stop_loss=2020.0,
    )

    assert result.reasons
    assert any(
        reason.reason_type
        is TakeProfitReasonType.RISK_REWARD_TARGET
        for reason in result.reasons
    )


def test_warnings_are_present():
    result = TakeProfitIntelligenceEngine().analyze(
        make_entry(),
        stop_loss=2020.0,
    )

    assert result.warnings


def test_result_metadata_matches_entry():
    entry = make_entry(
        direction=EntryDirection.LONG,
        entry_price=2035.0,
        symbol="XAUUSD",
        timeframe="H1",
    )

    result = TakeProfitIntelligenceEngine().analyze(
        entry,
        stop_loss=2025.0,
    )

    assert result.timestamp == entry.timestamp
    assert result.symbol == entry.symbol
    assert result.timeframe == entry.timeframe
    assert result.direction == entry.direction


def test_target_respects_custom_rr():
    result = TakeProfitIntelligenceEngine().analyze(
        make_entry(),
        stop_loss=2020.0,
        structural_level=2055.0,
        minimum_risk_reward=3.0,
    )

    assert result.take_profit == 2060.0
    assert result.risk_reward_ratio == 3.0


def test_short_target_respects_custom_rr():
    result = TakeProfitIntelligenceEngine().analyze(
        make_entry(EntryDirection.SHORT),
        stop_loss=2040.0,
        support_level=2005.0,
        minimum_risk_reward=3.0,
    )

    assert result.take_profit == 2000.0
    assert result.risk_reward_ratio == 3.0

# ============================================================
# P2.15 TAKE-PROFIT HARDENING TESTS
# ============================================================
#
# These tests strengthen the existing P2.15 implementation
# without changing its current public API.
#
# The purpose is to verify:
# - exact RR boundaries
# - below-minimum RR protection
# - long/short symmetry
# - wrong-side target protection
# - extreme RR handling
# - invalid numeric inputs
# - XAUUSD enforcement
# - reference-price behavior
# - auditability
# - readiness consistency
# ============================================================


def test_long_target_exactly_at_minimum_rr_is_allowed():
    # Create the TP engine using the default minimum RR of 2.0.
    engine = TakeProfitIntelligenceEngine()

    # Create a standard LONG entry at 2030.
    entry = make_entry(
        EntryDirection.LONG,
        2030.0,
    )

    # A stop at 2020 creates exactly 10 points of risk.
    # A 2R target must therefore be 2050.
    result = engine.analyze(
        entry,
        stop_loss=2020.0,
        structural_level=2050.0,
    )

    # The exact minimum RR boundary must be accepted.
    assert result.valid is True
    assert result.take_profit == 2050.0
    assert result.risk_reward_ratio == 2.0
    assert result.take_profit_ready is True


def test_short_target_exactly_at_minimum_rr_is_allowed():
    # Create the TP engine using the default minimum RR.
    engine = TakeProfitIntelligenceEngine()

    # Create a standard SHORT entry at 2030.
    entry = make_entry(
        EntryDirection.SHORT,
        2030.0,
    )

    # A stop at 2040 creates 10 points of risk.
    # A 2R target must therefore be 2010.
    result = engine.analyze(
        entry,
        stop_loss=2040.0,
        structural_level=2010.0,
    )

    # The exact minimum RR boundary must be accepted.
    assert result.valid is True
    assert result.take_profit == 2010.0
    assert result.risk_reward_ratio == 2.0
    assert result.take_profit_ready is True


def test_long_target_below_minimum_rr_falls_back_to_rr_target():
    # Create the TP engine.
    engine = TakeProfitIntelligenceEngine()

    # Create a LONG entry.
    entry = make_entry(
        EntryDirection.LONG,
        2030.0,
    )

    # Risk is 10 points.
    # The supplied structural target only provides 1.5R.
    result = engine.analyze(
        entry,
        stop_loss=2020.0,
        structural_level=2045.0,
    )

    # The engine must not accept the weaker structural target.
    # It should fall back to the configured 2R target.
    assert result.valid is True
    assert result.take_profit == 2050.0
    assert result.method is TakeProfitMethod.RISK_REWARD
    assert result.risk_reward_ratio == 2.0


def test_short_target_below_minimum_rr_falls_back_to_rr_target():
    # Create the TP engine.
    engine = TakeProfitIntelligenceEngine()

    # Create a SHORT entry.
    entry = make_entry(
        EntryDirection.SHORT,
        2030.0,
    )

    # Risk is 10 points.
    # The supplied structural target only provides 1.5R.
    result = engine.analyze(
        entry,
        stop_loss=2040.0,
        structural_level=2015.0,
    )

    # The weaker structural target must be rejected.
    assert result.valid is True
    assert result.take_profit == 2010.0
    assert result.method is TakeProfitMethod.RISK_REWARD
    assert result.risk_reward_ratio == 2.0


def test_long_target_exactly_below_minimum_rr_is_not_selected():
    # Create the engine with a 2R minimum.
    engine = TakeProfitIntelligenceEngine(
        minimum_risk_reward=2.0,
    )

    # Create the LONG entry.
    entry = make_entry(
        EntryDirection.LONG,
        2030.0,
    )

    # 10 points of risk means 2050 is the exact 2R boundary.
    # 2049.99 is below the required reward.
    result = engine.analyze(
        entry,
        stop_loss=2020.0,
        resistance_level=2049.99,
    )

    # The invalid candidate must not be selected.
    assert result.valid is True
    assert result.take_profit == 2050.0
    assert result.method is TakeProfitMethod.RISK_REWARD
    assert result.risk_reward_ratio == 2.0


def test_short_target_exactly_below_minimum_rr_is_not_selected():
    # Create the engine with a 2R minimum.
    engine = TakeProfitIntelligenceEngine(
        minimum_risk_reward=2.0,
    )

    # Create the SHORT entry.
    entry = make_entry(
        EntryDirection.SHORT,
        2030.0,
    )

    # 10 points of risk means 2010 is the exact 2R boundary.
    # 2010.01 provides less than the required reward.
    result = engine.analyze(
        entry,
        stop_loss=2040.0,
        support_level=2010.01,
    )

    # The invalid candidate must not be selected.
    assert result.valid is True
    assert result.take_profit == 2010.0
    assert result.method is TakeProfitMethod.RISK_REWARD
    assert result.risk_reward_ratio == 2.0


def test_long_target_wrong_side_is_ignored_even_if_rr_is_large():
    # Create the engine.
    engine = TakeProfitIntelligenceEngine()

    # Create a LONG entry.
    entry = make_entry(
        EntryDirection.LONG,
        2030.0,
    )

    # The structural level is below the entry.
    # It cannot be a LONG take-profit target.
    result = engine.analyze(
        entry,
        stop_loss=2020.0,
        structural_level=1900.0,
    )

    # The wrong-side target must be ignored.
    assert result.valid is True
    assert result.take_profit == 2050.0
    assert result.method is TakeProfitMethod.RISK_REWARD


def test_short_target_wrong_side_is_ignored_even_if_rr_is_large():
    # Create the engine.
    engine = TakeProfitIntelligenceEngine()

    # Create a SHORT entry.
    entry = make_entry(
        EntryDirection.SHORT,
        2030.0,
    )

    # The structural level is above the entry.
    # It cannot be a SHORT take-profit target.
    result = engine.analyze(
        entry,
        stop_loss=2040.0,
        structural_level=2160.0,
    )

    # The wrong-side target must be ignored.
    assert result.valid is True
    assert result.take_profit == 2010.0
    assert result.method is TakeProfitMethod.RISK_REWARD


def test_long_target_with_zero_reward_is_not_selected():
    # Create the engine.
    engine = TakeProfitIntelligenceEngine()

    # Create the LONG entry.
    entry = make_entry(
        EntryDirection.LONG,
        2030.0,
    )

    # A target equal to the entry provides zero reward.
    result = engine.analyze(
        entry,
        stop_loss=2020.0,
        structural_level=2030.0,
    )

    # The zero-reward target must be ignored.
    assert result.valid is True
    assert result.take_profit == 2050.0
    assert result.method is TakeProfitMethod.RISK_REWARD


def test_short_target_with_zero_reward_is_not_selected():
    # Create the engine.
    engine = TakeProfitIntelligenceEngine()

    # Create the SHORT entry.
    entry = make_entry(
        EntryDirection.SHORT,
        2030.0,
    )

    # A target equal to the entry provides zero reward.
    result = engine.analyze(
        entry,
        stop_loss=2040.0,
        structural_level=2030.0,
    )

    # The zero-reward target must be ignored.
    assert result.valid is True
    assert result.take_profit == 2010.0
    assert result.method is TakeProfitMethod.RISK_REWARD


def test_long_target_far_above_entry_can_be_valid():
    # Create the engine.
    engine = TakeProfitIntelligenceEngine()

    # Create a LONG entry.
    entry = make_entry(
        EntryDirection.LONG,
        2030.0,
    )

    # The engine allows targets above the normal 2R minimum.
    result = engine.analyze(
        entry,
        stop_loss=2020.0,
        structural_level=2100.0,
    )

    # 70 points of reward over 10 points of risk equals 7R.
    assert result.valid is True
    assert result.take_profit == 2100.0
    assert result.risk_reward_ratio == 7.0
    assert result.method is TakeProfitMethod.STRUCTURE


def test_short_target_far_below_entry_can_be_valid():
    # Create the engine.
    engine = TakeProfitIntelligenceEngine()

    # Create a SHORT entry.
    entry = make_entry(
        EntryDirection.SHORT,
        2030.0,
    )

    # The engine allows targets below the normal 2R minimum.
    result = engine.analyze(
        entry,
        stop_loss=2040.0,
        structural_level=1960.0,
    )

    # 70 points of reward over 10 points of risk equals 7R.
    assert result.valid is True
    assert result.take_profit == 1960.0
    assert result.risk_reward_ratio == 7.0
    assert result.method is TakeProfitMethod.STRUCTURE


def test_long_target_above_maximum_rr_is_rejected_or_falls_back():
    # Configure a maximum RR of 5.
    engine = TakeProfitIntelligenceEngine(
        minimum_risk_reward=2.0,
        maximum_risk_reward=5.0,
    )

    # Create a LONG entry.
    entry = make_entry(
        EntryDirection.LONG,
        2030.0,
    )

    # 100 points of reward over 10 points of risk equals 10R.
    # This is above the configured maximum RR.
    result = engine.analyze(
        entry,
        stop_loss=2020.0,
        structural_level=2130.0,
    )

    # The selected target must never violate the configured
    # maximum RR policy.
    assert result.valid is True
    assert result.take_profit == 2050.0
    assert result.risk_reward_ratio == 2.0


def test_short_target_below_maximum_rr_is_rejected_or_falls_back():
    # Configure a maximum RR of 5.
    engine = TakeProfitIntelligenceEngine(
        minimum_risk_reward=2.0,
        maximum_risk_reward=5.0,
    )

    # Create a SHORT entry.
    entry = make_entry(
        EntryDirection.SHORT,
        2030.0,
    )

    # 100 points of reward over 10 points of risk equals 10R.
    result = engine.analyze(
        entry,
        stop_loss=2040.0,
        structural_level=1930.0,
    )

    # The selected target must respect the maximum RR policy.
    assert result.valid is True
    assert result.take_profit == 2010.0
    assert result.risk_reward_ratio == 2.0


def test_long_result_contains_auditable_risk_and_reward_values():
    # Create the engine.
    engine = TakeProfitIntelligenceEngine()

    # Create a LONG entry.
    entry = make_entry(
        EntryDirection.LONG,
        2030.0,
    )

    # Use a 3R structural target.
    result = engine.analyze(
        entry,
        stop_loss=2020.0,
        structural_level=2060.0,
    )

    # Verify the result preserves the core calculation values.
    assert result.entry_price == 2030.0
    assert result.stop_loss == 2020.0
    assert result.take_profit == 2060.0
    assert result.risk_distance == 10.0
    assert result.reward_distance == 30.0
    assert result.risk_reward_ratio == 3.0
    assert result.minimum_risk_reward == 2.0


def test_short_result_contains_auditable_risk_and_reward_values():
    # Create the engine.
    engine = TakeProfitIntelligenceEngine()

    # Create a SHORT entry.
    entry = make_entry(
        EntryDirection.SHORT,
        2030.0,
    )

    # Use a 3R structural target.
    result = engine.analyze(
        entry,
        stop_loss=2040.0,
        structural_level=2000.0,
    )

    # Verify the result preserves the core calculation values.
    assert result.entry_price == 2030.0
    assert result.stop_loss == 2040.0
    assert result.take_profit == 2000.0
    assert result.risk_distance == 10.0
    assert result.reward_distance == 30.0
    assert result.risk_reward_ratio == 3.0
    assert result.minimum_risk_reward == 2.0


def test_long_ready_result_has_valid_consistent_state():
    # Create the engine.
    engine = TakeProfitIntelligenceEngine()

    # Create a normal LONG entry.
    result = engine.analyze(
        make_entry(EntryDirection.LONG),
        stop_loss=2020.0,
    )

    # A ready TP must also be valid and contain a target.
    assert result.take_profit_ready is True
    assert result.valid is True
    assert result.has_take_profit is True
    assert result.is_ready is True
    assert result.take_profit is not None
    assert result.risk_reward_ratio >= result.minimum_risk_reward


def test_short_ready_result_has_valid_consistent_state():
    # Create the engine.
    engine = TakeProfitIntelligenceEngine()

    # Create a normal SHORT entry.
    result = engine.analyze(
        make_entry(EntryDirection.SHORT),
        stop_loss=2040.0,
    )

    # A ready TP must also be valid and contain a target.
    assert result.take_profit_ready is True
    assert result.valid is True
    assert result.has_take_profit is True
    assert result.is_ready is True
    assert result.take_profit is not None
    assert result.risk_reward_ratio >= result.minimum_risk_reward


def test_non_xauusd_analyze_xauusd_is_blocked():
    # Create an entry for a different instrument.
    entry = make_entry(
        EntryDirection.LONG,
        2030.0,
        symbol="EURUSD",
    )

    # The XAUUSD-specific API must reject the wrong symbol.
    with pytest.raises(TakeProfitIntelligenceError):
        TakeProfitIntelligenceEngine().analyze_xauusd(
            entry,
            stop_loss=2020.0,
        )


def test_negative_structural_target_is_rejected():
    # Create the engine.
    engine = TakeProfitIntelligenceEngine()

    # A negative price is invalid.
    with pytest.raises(TakeProfitIntelligenceError):
        engine.analyze(
            make_entry(),
            stop_loss=2020.0,
            structural_level=-1.0,
        )


def test_nan_structural_target_is_rejected():
    # Create the engine.
    engine = TakeProfitIntelligenceEngine()

    # NaN cannot represent a valid market price.
    with pytest.raises(TakeProfitIntelligenceError):
        engine.analyze(
            make_entry(),
            stop_loss=2020.0,
            structural_level=float("nan"),
        )


def test_infinite_structural_target_is_rejected():
    # Create the engine.
    engine = TakeProfitIntelligenceEngine()

    # Infinity cannot represent a valid market price.
    with pytest.raises(TakeProfitIntelligenceError):
        engine.analyze(
            make_entry(),
            stop_loss=2020.0,
            structural_level=float("inf"),
        )


def test_negative_support_target_is_rejected():
    # Create the engine.
    engine = TakeProfitIntelligenceEngine()

    # A negative support price is invalid.
    with pytest.raises(TakeProfitIntelligenceError):
        engine.analyze(
            make_entry(),
            stop_loss=2020.0,
            support_level=-1.0,
        )


def test_negative_resistance_target_is_rejected():
    # Create the engine.
    engine = TakeProfitIntelligenceEngine()

    # A negative resistance price is invalid.
    with pytest.raises(TakeProfitIntelligenceError):
        engine.analyze(
            make_entry(),
            stop_loss=2020.0,
            resistance_level=-1.0,
        )


def test_negative_trendline_target_is_rejected():
    # Create the engine.
    engine = TakeProfitIntelligenceEngine()

    # A negative trendline price is invalid.
    with pytest.raises(TakeProfitIntelligenceError):
        engine.analyze(
            make_entry(),
            stop_loss=2020.0,
            trendline_level=-1.0,
        )


def test_boolean_structural_target_is_rejected():
    # Create the engine.
    engine = TakeProfitIntelligenceEngine()

    # Boolean values must never be accepted as prices.
    with pytest.raises(TakeProfitIntelligenceError):
        engine.analyze(
            make_entry(),
            stop_loss=2020.0,
            structural_level=True,
        )


def test_boolean_stop_loss_is_rejected():
    # Create the engine.
    engine = TakeProfitIntelligenceEngine()

    # Boolean values must never be accepted as stop prices.
    with pytest.raises(TakeProfitIntelligenceError):
        engine.analyze(
            make_entry(),
            stop_loss=True,
        )


def test_boolean_reference_price_is_rejected():
    # Create the engine.
    engine = TakeProfitIntelligenceEngine()

    # Boolean values must never be accepted as market prices.
    with pytest.raises(TakeProfitIntelligenceError):
        engine.analyze(
            make_entry(),
            stop_loss=2020.0,
            reference_price=True,
        )


def test_reference_price_preserves_long_target_geometry():
    # Create the engine.
    engine = TakeProfitIntelligenceEngine()

    # The EntryModel contains 2030, but the supplied market reference
    # price is 2040.
    entry = make_entry(
        EntryDirection.LONG,
        2030.0,
    )

    result = engine.analyze(
        entry,
        stop_loss=2020.0,
        reference_price=2040.0,
    )

    # The RR target is calculated from the effective reference price.
    assert result.entry_price == 2040.0
    assert result.take_profit == 2060.0
    assert result.risk_reward_ratio == 2.0


def test_reference_price_preserves_short_target_geometry():
    # Create the engine.
    engine = TakeProfitIntelligenceEngine()

    # The EntryModel contains 2030, but the effective reference price
    # is 2040.
    entry = make_entry(
        EntryDirection.SHORT,
        2030.0,
    )

    result = engine.analyze(
        entry,
        stop_loss=2040.0,
        reference_price=2040.0,
    )

    # The RR target is calculated from the effective reference price.
    assert result.entry_price == 2040.0
    assert result.take_profit == 2020.0
    assert result.risk_reward_ratio == 2.0


def test_missing_all_target_levels_uses_rr_fallback():
    # Create the engine.
    engine = TakeProfitIntelligenceEngine()

    # No structural, S/R, or trendline target is supplied.
    result = engine.analyze(
        make_entry(EntryDirection.LONG),
        stop_loss=2020.0,
    )

    # The deterministic RR fallback must still produce a valid TP.
    assert result.valid is True
    assert result.take_profit == 2050.0
    assert result.method is TakeProfitMethod.RISK_REWARD
    assert result.take_profit_ready is True


def test_short_missing_all_target_levels_uses_rr_fallback():
    # Create the engine.
    engine = TakeProfitIntelligenceEngine()

    # No structural, S/R, or trendline target is supplied.
    result = engine.analyze(
        make_entry(EntryDirection.SHORT),
        stop_loss=2040.0,
    )

    # The deterministic RR fallback must still produce a valid TP.
    assert result.valid is True
    assert result.take_profit == 2010.0
    assert result.method is TakeProfitMethod.RISK_REWARD
    assert result.take_profit_ready is True