from datetime import datetime, timedelta

import pytest

from app.trading.context.market_context import (
    ContextBias,
    ContextSignal,
    ContextSignalType,
    MarketCondition,
    MarketContext,
)
from app.trading.setup.setup_engine import (
    SetupDirection,
    SetupEngine,
    SetupReasonType,
    SetupType,
)
from app.trading.structure.market_structure import (
    StructureTrend,
)


def make_context(
    *,
    bias=ContextBias.BULLISH,
    context_strength=85.0,
    trend=StructureTrend.BULLISH,
    trend_strength=85.0,
    condition=MarketCondition.TRENDING_UP,
    sufficient_history=True,
    conflicts=(),
    rsi_bias=ContextBias.BULLISH,
    rsi_strength=70.0,
    macd_bias=ContextBias.BULLISH,
    macd_strength=65.0,
    structure_strength=90.0,
    price_bias=ContextBias.BULLISH,
    price_strength=60.0,
):
    signals = (
        ContextSignal(
            signal_type=ContextSignalType.STRUCTURE,
            bias=trend
            if trend in (
                StructureTrend.BULLISH,
                StructureTrend.BEARISH,
            )
            else ContextBias.NEUTRAL,
            strength=structure_strength,
            value=structure_strength,
        ),
        ContextSignal(
            signal_type=ContextSignalType.RSI,
            bias=rsi_bias,
            strength=rsi_strength,
            value=65.0,
        ),
        ContextSignal(
            signal_type=ContextSignalType.MACD,
            bias=macd_bias,
            strength=macd_strength,
            value=0.1,
        ),
        ContextSignal(
            signal_type=ContextSignalType.VOLATILITY,
            bias=ContextBias.NEUTRAL,
            strength=0.0,
            value=1.2,
        ),
        ContextSignal(
            signal_type=ContextSignalType.PRICE_LOCATION,
            bias=price_bias,
            strength=price_strength,
            value=75.0,
        ),
    )

    return MarketContext(
        timestamp=datetime(2026, 1, 1, 12, 0),
        symbol="XAUUSD",
        timeframe="M15",
        close=3400.0,
        trend=trend,
        trend_strength=trend_strength,
        rsi=65.0,
        atr=10.0,
        macd=None,
        bollinger_bands=None,
        price_location=75.0,
        volatility_ratio=0.3,
        bias=bias,
        context_strength=context_strength,
        condition=condition,
        signals=signals,
        conflicts=tuple(conflicts),
        sufficient_history=sufficient_history,
    )


def test_default_engine_configuration():
    engine = SetupEngine()

    assert engine.minimum_setup_score == 60.0
    assert engine.strong_setup_score == 80.0
    assert engine.minimum_supporting_signals == 2
    assert engine.max_conflicts == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_setup_score": -1},
        {"minimum_setup_score": 101},
        {"strong_setup_score": -1},
        {"strong_setup_score": 101},
        {"minimum_supporting_signals": 0},
        {"minimum_supporting_signals": -1},
        {"max_conflicts": -1},
    ],
)
def test_invalid_engine_configuration(kwargs):
    with pytest.raises(ValueError):
        SetupEngine(**kwargs)


def test_strong_score_cannot_be_below_minimum_score():
    with pytest.raises(ValueError):
        SetupEngine(
            minimum_setup_score=80,
            strong_setup_score=60,
        )


def test_context_must_be_market_context():
    engine = SetupEngine()

    with pytest.raises(ValueError):
        engine.evaluate(None)


def test_bullish_trending_context_produces_long_setup():
    engine = SetupEngine()

    context = make_context()

    result = engine.evaluate(context)

    assert result.direction == SetupDirection.LONG
    assert result.setup_type == SetupType.TREND_CONTINUATION
    assert result.context_bias == ContextBias.BULLISH
    assert result.market_condition == (
        MarketCondition.TRENDING_UP
    )
    assert result.quality_score > 0
    assert result.valid is True


def test_bearish_trending_context_produces_short_setup():
    engine = SetupEngine()

    context = make_context(
        bias=ContextBias.BEARISH,
        trend=StructureTrend.BEARISH,
        trend_strength=85.0,
        condition=MarketCondition.TRENDING_DOWN,
        rsi_bias=ContextBias.BEARISH,
        macd_bias=ContextBias.BEARISH,
        price_bias=ContextBias.BEARISH,
    )

    result = engine.evaluate(context)

    assert result.direction == SetupDirection.SHORT
    assert result.setup_type == SetupType.TREND_CONTINUATION
    assert result.valid is True


def test_neutral_context_produces_no_trade():
    engine = SetupEngine()

    context = make_context(
        bias=ContextBias.NEUTRAL,
        context_strength=0.0,
        trend=StructureTrend.RANGE,
        trend_strength=20.0,
        condition=MarketCondition.RANGING,
        rsi_bias=ContextBias.NEUTRAL,
        rsi_strength=0.0,
        macd_bias=ContextBias.NEUTRAL,
        macd_strength=0.0,
        price_bias=ContextBias.NEUTRAL,
        price_strength=0.0,
    )

    result = engine.evaluate(context)

    assert result.direction == SetupDirection.NONE
    assert result.setup_type == SetupType.NONE
    assert result.valid is False
    assert result.quality_score == 0.0
    assert (
        SetupReasonType.NEUTRAL_CONTEXT
        in {
            warning.reason_type
            for warning in result.warnings
        }
    )


def test_insufficient_history_cannot_produce_valid_setup():
    engine = SetupEngine()

    context = make_context(
        sufficient_history=False,
    )

    result = engine.evaluate(context)

    assert result.valid is False
    assert (
        SetupReasonType.INSUFFICIENT_HISTORY
        in {
            warning.reason_type
            for warning in result.warnings
        }
    )


def test_too_many_conflicts_invalidates_setup():
    engine = SetupEngine(
        max_conflicts=1,
    )

    context = make_context(
        conflicts=(
            ContextSignalType.RSI,
            ContextSignalType.MACD,
        ),
    )

    result = engine.evaluate(context)

    assert result.valid is False
    assert result.quality_score < 60.0
    assert (
        SetupReasonType.CONFLICT
        in {
            warning.reason_type
            for warning in result.warnings
        }
    )


def test_one_supporting_signal_is_not_enough():
    engine = SetupEngine(
        minimum_supporting_signals=3,
    )

    context = make_context(
        rsi_bias=ContextBias.NEUTRAL,
        rsi_strength=0.0,
        macd_bias=ContextBias.NEUTRAL,
        macd_strength=0.0,
        price_bias=ContextBias.NEUTRAL,
        price_strength=0.0,
    )

    result = engine.evaluate(context)

    assert result.valid is False


def test_context_strength_is_required():
    engine = SetupEngine()

    context = make_context(
        context_strength=40.0,
    )

    result = engine.evaluate(context)

    assert result.valid is False


def test_quality_score_is_bounded():
    engine = SetupEngine()

    context = make_context(
        context_strength=100.0,
        trend_strength=100.0,
        structure_strength=100.0,
        rsi_strength=100.0,
        macd_strength=100.0,
        price_strength=100.0,
    )

    result = engine.evaluate(context)

    assert 0.0 <= result.quality_score <= 100.0


def test_quality_score_is_deterministic():
    engine = SetupEngine()

    context = make_context()

    first = engine.evaluate(context)
    second = engine.evaluate(context)

    assert first == second


def test_supporting_signals_match_context_bias():
    engine = SetupEngine()

    context = make_context(
        bias=ContextBias.BULLISH,
        rsi_bias=ContextBias.BEARISH,
        macd_bias=ContextBias.BULLISH,
    )

    result = engine.evaluate(context)

    assert (
        ContextSignalType.RSI
        not in result.supporting_signals
    )

    assert (
        ContextSignalType.MACD
        in result.supporting_signals
    )

    assert (
        ContextSignalType.STRUCTURE
        in result.supporting_signals
    )


def test_transition_can_be_classified_as_reversal():
    engine = SetupEngine()

    context = make_context(
        condition=MarketCondition.TRANSITION,
    )

    result = engine.evaluate(context)

    assert result.direction == SetupDirection.LONG
    assert result.setup_type == SetupType.REVERSAL


def test_ranging_context_is_classified_as_range():
    engine = SetupEngine()

    context = make_context(
        condition=MarketCondition.RANGING,
    )

    result = engine.evaluate(context)

    assert result.setup_type == SetupType.RANGE


def test_evaluate_at_returns_selected_context_only():
    engine = SetupEngine()

    first = make_context()
    second = make_context(
        bias=ContextBias.BEARISH,
        trend=StructureTrend.BEARISH,
        condition=MarketCondition.TRENDING_DOWN,
        rsi_bias=ContextBias.BEARISH,
        macd_bias=ContextBias.BEARISH,
        price_bias=ContextBias.BEARISH,
    )

    contexts = [first, second]

    result = engine.evaluate_at(
        contexts,
        1,
    )

    assert result.timestamp == second.timestamp
    assert result.direction == SetupDirection.SHORT


def test_evaluate_at_validates_index():
    engine = SetupEngine()

    context = make_context()

    with pytest.raises(ValueError):
        engine.evaluate_at(
            [context],
            -1,
        )

    with pytest.raises(ValueError):
        engine.evaluate_at(
            [context],
            1,
        )


def test_evaluate_at_rejects_invalid_context():
    engine = SetupEngine()

    with pytest.raises(ValueError):
        engine.evaluate_at(
            [make_context(), None],
            0,
        )


def test_setup_result_preserves_market_identity():
    engine = SetupEngine()

    context = make_context()

    result = engine.evaluate(context)

    assert result.symbol == "XAUUSD"
    assert result.timeframe == "M15"
    assert result.close == 3400.0
    assert result.timestamp == context.timestamp


def test_reversal_does_not_require_trending_condition():
    engine = SetupEngine()

    context = make_context(
        bias=ContextBias.BULLISH,
        trend=StructureTrend.BEARISH,
        trend_strength=55.0,
        condition=MarketCondition.TRANSITION,
        rsi_bias=ContextBias.BULLISH,
        macd_bias=ContextBias.BULLISH,
        price_bias=ContextBias.BULLISH,
    )

    result = engine.evaluate(context)

    assert result.direction == SetupDirection.LONG
    assert result.setup_type == SetupType.REVERSAL


def test_zero_conflict_context_can_be_valid():
    engine = SetupEngine(
        max_conflicts=0,
    )

    context = make_context(
        conflicts=(),
    )

    result = engine.evaluate(context)

    assert result.valid is True