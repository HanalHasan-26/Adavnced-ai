from datetime import datetime

import pytest

from app.trading.confluence.setup_confluence import (
    ConfluenceDirection,
    ConfluenceQuality,
    SetupConfluenceResult,
)
from app.trading.entry.entry_model import (
    EntryDirection,
    EntryModel,
    EntryModelEngine,
    EntryModelError,
    EntryQuality,
    EntryTrigger,
)


def make_confluence(
    direction=ConfluenceDirection.BULLISH,
    quality=ConfluenceQuality.STRONG,
    score=85.0,
    sufficient_data=True,
    conflicting_factors=0,
):
    return SetupConfluenceResult(
        timestamp=datetime(2026, 1, 1),
        symbol="XAUUSD",
        timeframe="M15",
        direction=direction,
        quality=quality,
        score=score,
        mtf_score=90.0,
        support_resistance_score=80.0,
        trendline_score=80.0,
        volume_score=75.0,
        bullish_factors=3,
        bearish_factors=0,
        neutral_factors=1,
        conflicting_factors=conflicting_factors,
        support_present=True,
        resistance_present=False,
        support_strong=True,
        resistance_strong=False,
        support_trendline_present=True,
        resistance_trendline_present=False,
        volume_confirmed=True,
        sufficient_data=sufficient_data,
        reasons=(),
        warnings=(),
    )


def test_defaults():
    engine = EntryModelEngine()

    assert engine.minimum_confluence_score == 50.0
    assert engine.good_confluence_score == 65.0
    assert engine.strong_confluence_score == 80.0
    assert engine.minimum_entry_confidence == 50.0
    assert engine.good_entry_confidence == 65.0
    assert engine.strong_entry_confidence == 80.0


def test_bullish_confluence_creates_long_entry():
    result = EntryModelEngine().analyze(
        make_confluence()
    )

    assert isinstance(result, EntryModel)
    assert result.direction is EntryDirection.LONG
    assert result.entry_allowed is True
    assert result.valid is True
    assert result.entry_price is not None


def test_bearish_confluence_creates_short_entry():
    confluence = make_confluence(
        direction=ConfluenceDirection.BEARISH,
    )

    confluence = SetupConfluenceResult(
        timestamp=confluence.timestamp,
        symbol=confluence.symbol,
        timeframe=confluence.timeframe,
        direction=ConfluenceDirection.BEARISH,
        quality=confluence.quality,
        score=confluence.score,
        mtf_score=confluence.mtf_score,
        support_resistance_score=confluence.support_resistance_score,
        trendline_score=confluence.trendline_score,
        volume_score=confluence.volume_score,
        bullish_factors=0,
        bearish_factors=3,
        neutral_factors=1,
        conflicting_factors=0,
        support_present=False,
        resistance_present=True,
        support_strong=False,
        resistance_strong=True,
        support_trendline_present=False,
        resistance_trendline_present=True,
        volume_confirmed=True,
        sufficient_data=True,
        reasons=(),
        warnings=(),
    )

    result = EntryModelEngine().analyze(
        confluence
    )

    assert result.direction is EntryDirection.SHORT
    assert result.entry_allowed is True
    assert result.valid is True


def test_neutral_confluence_is_blocked():
    result = EntryModelEngine().analyze(
        make_confluence(
            direction=ConfluenceDirection.NEUTRAL,
            quality=ConfluenceQuality.MIXED,
            score=55.0,
        )
    )

    assert result.direction is EntryDirection.NONE
    assert result.entry_allowed is False
    assert result.valid is False
    assert result.entry_price is None


def test_unknown_confluence_is_blocked():
    result = EntryModelEngine().analyze(
        make_confluence(
            direction=ConfluenceDirection.UNKNOWN,
            quality=ConfluenceQuality.UNKNOWN,
            score=0.0,
        )
    )

    assert result.direction is EntryDirection.UNKNOWN
    assert result.entry_allowed is False
    assert result.valid is False


def test_weak_confluence_is_blocked():
    result = EntryModelEngine().analyze(
        make_confluence(
            quality=ConfluenceQuality.WEAK,
            score=30.0,
        )
    )

    assert result.quality is EntryQuality.WEAK
    assert result.entry_allowed is False
    assert result.valid is False


def test_conflicted_confluence_is_blocked():
    result = EntryModelEngine().analyze(
        make_confluence(
            quality=ConfluenceQuality.CONFLICTED,
            score=75.0,
            conflicting_factors=2,
        )
    )

    assert result.quality is EntryQuality.REJECTED
    assert result.entry_allowed is False
    assert result.valid is False


def test_insufficient_data_is_blocked():
    result = EntryModelEngine().analyze(
        make_confluence(
            sufficient_data=False,
        )
    )

    assert result.quality is EntryQuality.REJECTED
    assert result.entry_allowed is False
    assert result.valid is False


def test_support_changes_bullish_trigger():
    result = EntryModelEngine().analyze(
        make_confluence()
    )

    assert result.trigger is EntryTrigger.PULLBACK


def test_bearish_resistance_trigger():
    confluence = make_confluence(
        direction=ConfluenceDirection.BEARISH,
    )

    confluence = SetupConfluenceResult(
        timestamp=confluence.timestamp,
        symbol=confluence.symbol,
        timeframe=confluence.timeframe,
        direction=ConfluenceDirection.BEARISH,
        quality=confluence.quality,
        score=confluence.score,
        mtf_score=confluence.mtf_score,
        support_resistance_score=confluence.support_resistance_score,
        trendline_score=confluence.trendline_score,
        volume_score=confluence.volume_score,
        bullish_factors=0,
        bearish_factors=3,
        neutral_factors=1,
        conflicting_factors=0,
        support_present=False,
        resistance_present=True,
        support_strong=False,
        resistance_strong=True,
        support_trendline_present=False,
        resistance_trendline_present=True,
        volume_confirmed=True,
        sufficient_data=True,
        reasons=(),
        warnings=(),
    )

    result = EntryModelEngine().analyze(
        confluence
    )

    assert result.trigger is EntryTrigger.PULLBACK


def test_confidence_is_bounded():
    result = EntryModelEngine().analyze(
        make_confluence()
    )

    assert 0.0 <= result.entry_confidence <= 100.0


def test_score_is_preserved():
    confluence = make_confluence(
        score=72.5,
    )

    result = EntryModelEngine().analyze(
        confluence
    )

    assert result.confluence_score == pytest.approx(
        72.5
    )


def test_mtf_confirmation_flag():
    result = EntryModelEngine().analyze(
        make_confluence()
    )

    assert result.mtf_confirmed is True


def test_volume_confirmation_flag():
    result = EntryModelEngine().analyze(
        make_confluence()
    )

    assert result.volume_confirmed is True


def test_support_flag():
    result = EntryModelEngine().analyze(
        make_confluence()
    )

    assert result.support_present is True


def test_trendline_flag():
    result = EntryModelEngine().analyze(
        make_confluence()
    )

    assert result.trendline_present is True


def test_xauusd_wrapper():
    result = EntryModelEngine().analyze_xauusd(
        make_confluence()
    )

    assert result.symbol == "XAUUSD"


def test_xauusd_wrapper_rejects_other_symbol():
    confluence = make_confluence()

    confluence = SetupConfluenceResult(
        timestamp=confluence.timestamp,
        symbol="EURUSD",
        timeframe=confluence.timeframe,
        direction=confluence.direction,
        quality=confluence.quality,
        score=confluence.score,
        mtf_score=confluence.mtf_score,
        support_resistance_score=confluence.support_resistance_score,
        trendline_score=confluence.trendline_score,
        volume_score=confluence.volume_score,
        bullish_factors=confluence.bullish_factors,
        bearish_factors=confluence.bearish_factors,
        neutral_factors=confluence.neutral_factors,
        conflicting_factors=confluence.conflicting_factors,
        support_present=confluence.support_present,
        resistance_present=confluence.resistance_present,
        support_strong=confluence.support_strong,
        resistance_strong=confluence.resistance_strong,
        support_trendline_present=confluence.support_trendline_present,
        resistance_trendline_present=confluence.resistance_trendline_present,
        volume_confirmed=confluence.volume_confirmed,
        sufficient_data=confluence.sufficient_data,
        reasons=(),
        warnings=(),
    )

    with pytest.raises(EntryModelError):
        EntryModelEngine().analyze_xauusd(
            confluence
        )


def test_invalid_confluence_type():
    with pytest.raises(EntryModelError):
        EntryModelEngine().analyze(
            object()
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_confluence_score": -1},
        {"minimum_confluence_score": 101},
        {"good_confluence_score": -1},
        {"good_confluence_score": 101},
        {"strong_confluence_score": -1},
        {"strong_confluence_score": 101},
        {"minimum_entry_confidence": -1},
        {"minimum_entry_confidence": 101},
        {"good_entry_confidence": -1},
        {"good_entry_confidence": 101},
        {"strong_entry_confidence": -1},
        {"strong_entry_confidence": 101},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(EntryModelError):
        EntryModelEngine(**kwargs)


def test_confluence_threshold_order():
    with pytest.raises(EntryModelError):
        EntryModelEngine(
            minimum_confluence_score=70.0,
            good_confluence_score=60.0,
            strong_confluence_score=80.0,
        )


def test_confidence_threshold_order():
    with pytest.raises(EntryModelError):
        EntryModelEngine(
            minimum_entry_confidence=70.0,
            good_entry_confidence=60.0,
            strong_entry_confidence=80.0,
        )


def test_result_is_immutable():
    result = EntryModelEngine().analyze(
        make_confluence()
    )

    with pytest.raises(AttributeError):
        result.entry_allowed = False


def test_result_properties():
    result = EntryModelEngine().analyze(
        make_confluence()
    )

    assert result.is_long
    assert not result.is_short
    assert not result.is_none
    assert not result.is_unknown
    assert result.is_strong
    assert not result.is_rejected
    assert result.has_entry
    assert result.is_valid


def test_reasons_are_present():
    result = EntryModelEngine().analyze(
        make_confluence()
    )

    assert result.reasons


def test_warnings_are_present():
    result = EntryModelEngine().analyze(
        make_confluence()
    )

    assert result.warnings


def test_entry_price_is_not_used_for_risk_calculation():
    result = EntryModelEngine().analyze(
        make_confluence()
    )

    assert result.entry_price == pytest.approx(1.0)


def test_low_score_is_blocked():
    result = EntryModelEngine().analyze(
        make_confluence(
            quality=ConfluenceQuality.MIXED,
            score=45.0,
        )
    )

    assert result.entry_allowed is False
    assert result.valid is False


def test_conflicting_factor_blocks_entry():
    result = EntryModelEngine().analyze(
        make_confluence(
            score=90.0,
            conflicting_factors=1,
        )
    )

    assert result.entry_allowed is False
    assert result.valid is False


def test_result_timestamp_matches_confluence():
    confluence = make_confluence()

    result = EntryModelEngine().analyze(
        confluence
    )

    assert result.timestamp == confluence.timestamp


def test_result_symbol_matches_confluence():
    confluence = make_confluence()

    result = EntryModelEngine().analyze(
        confluence
    )

    assert result.symbol == confluence.symbol


def test_result_timeframe_matches_confluence():
    confluence = make_confluence()

    result = EntryModelEngine().analyze(
        confluence
    )

    assert result.timeframe == confluence.timeframe


def test_good_confluence_quality():
    result = EntryModelEngine().analyze(
        make_confluence(
            quality=ConfluenceQuality.GOOD,
            score=70.0,
        )
    )

    assert result.quality is EntryQuality.GOOD


def test_acceptable_confluence_quality():
    result = EntryModelEngine().analyze(
        make_confluence(
            quality=ConfluenceQuality.MIXED,
            score=55.0,
        )
    )

    assert result.quality is EntryQuality.ACCEPTABLE


def test_strong_confluence_quality():
    result = EntryModelEngine().analyze(
        make_confluence(
            quality=ConfluenceQuality.STRONG,
            score=90.0,
        )
    )

    assert result.quality is EntryQuality.STRONG


def test_engine_is_deterministic():
    confluence = make_confluence()

    engine = EntryModelEngine()

    first = engine.analyze(confluence)
    second = engine.analyze(confluence)

    assert first == second


def test_no_llm_or_execution_fields():
    result = EntryModelEngine().analyze(
        make_confluence()
    )

    assert not hasattr(result, "stop_loss")
    assert not hasattr(result, "take_profit")
    assert not hasattr(result, "position_size")
    assert not hasattr(result, "pnl")