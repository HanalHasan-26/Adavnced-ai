from datetime import datetime, timedelta

import pytest

from app.trading.context.market_context import (
    ContextBias,
    MarketCondition,
    MarketContext,
    ContextSignal,
    ContextSignalType,
)
from app.trading.mtf.multi_timeframe import (
    MTFAlignment,
    MTFDirection,
    MTFReasonType,
    MultiTimeframeAnalysisError,
    MultiTimeframeEngine,
    TimeframeRole,
)
from app.trading.regime.market_regime import (
    MarketRegime,
    MarketRegimeResult,
)


BASE_TIME = datetime(2026, 1, 1, 12, 0, 0)


def make_context(
    timeframe: str,
    *,
    bias: ContextBias = ContextBias.BULLISH,
    strength: float = 80.0,
    timestamp: datetime = BASE_TIME,
    symbol: str = "XAUUSD",
    sufficient_history: bool = True,
) -> MarketContext:
    return MarketContext(
        timestamp=timestamp,
        symbol=symbol,
        timeframe=timeframe,
        close=3000.0,
        trend=(
            MarketRegime.TRENDING_UP
            if bias is ContextBias.BULLISH
            else MarketRegime.TRENDING_DOWN
            if bias is ContextBias.BEARISH
            else MarketRegime.RANGING
        ),
        trend_strength=strength,
        rsi=60.0,
        atr=10.0,
        macd=None,
        bollinger_bands=None,
        price_location=50.0,
        volatility_ratio=1.0,
        bias=bias,
        context_strength=strength,
        condition=(
            MarketCondition.TRENDING_UP
            if bias is ContextBias.BULLISH
            else MarketCondition.TRENDING_DOWN
            if bias is ContextBias.BEARISH
            else MarketCondition.RANGING
        ),
        signals=(
            ContextSignal(
                signal_type=ContextSignalType.STRUCTURE,
                bias=bias,
                strength=strength,
                value=1.0,
            ),
        ),
        conflicts=(),
        sufficient_history=sufficient_history,
    )


def make_regime(
    timeframe: str,
    *,
    regime: MarketRegime = MarketRegime.TRENDING_UP,
    strength: float = 80.0,
    timestamp: datetime = BASE_TIME,
    symbol: str = "XAUUSD",
    sufficient_history: bool = True,
) -> MarketRegimeResult:
    return MarketRegimeResult(
        timestamp=timestamp,
        symbol=symbol,
        timeframe=timeframe,
        regime=regime,
        strength=strength,
        trend_strength=strength,
        volatility_ratio=1.0,
        persistence_bars=3,
        sufficient_history=sufficient_history,
        reasons=(),
    )


def make_inputs(
    *,
    higher_bias=ContextBias.BULLISH,
    middle_bias=ContextBias.BULLISH,
    lower_bias=ContextBias.BULLISH,
    higher_strength=80.0,
    middle_strength=80.0,
    lower_strength=80.0,
    higher_regime=MarketRegime.TRENDING_UP,
    middle_regime=MarketRegime.TRENDING_UP,
    lower_regime=MarketRegime.TRENDING_UP,
):
    higher_context = make_context(
        "H4",
        bias=higher_bias,
        strength=higher_strength,
    )

    middle_context = make_context(
        "H1",
        bias=middle_bias,
        strength=middle_strength,
    )

    lower_context = make_context(
        "M15",
        bias=lower_bias,
        strength=lower_strength,
    )

    higher_regime_result = make_regime(
        "H4",
        regime=higher_regime,
        strength=higher_strength,
    )

    middle_regime_result = make_regime(
        "H1",
        regime=middle_regime,
        strength=middle_strength,
    )

    lower_regime_result = make_regime(
        "M15",
        regime=lower_regime,
        strength=lower_strength,
    )

    return (
        higher_context,
        middle_context,
        lower_context,
        higher_regime_result,
        middle_regime_result,
        lower_regime_result,
    )


def test_engine_defaults():
    engine = MultiTimeframeEngine()

    assert engine.minimum_alignment_strength == 50.0
    assert engine.strong_alignment_strength == 70.0
    assert engine.partial_alignment_threshold == 66.67


@pytest.mark.parametrize(
    "value",
    [-1.0, 101.0, float("inf"), float("-inf"), float("nan")],
)
def test_invalid_minimum_alignment_strength(value):
    with pytest.raises(MultiTimeframeAnalysisError):
        MultiTimeframeEngine(
            minimum_alignment_strength=value,
        )


@pytest.mark.parametrize(
    "value",
    [-1.0, 101.0, float("inf"), float("-inf"), float("nan")],
)
def test_invalid_strong_alignment_strength(value):
    with pytest.raises(MultiTimeframeAnalysisError):
        MultiTimeframeEngine(
            strong_alignment_strength=value,
        )


@pytest.mark.parametrize(
    "value",
    [-1.0, 101.0, float("inf"), float("-inf"), float("nan")],
)
def test_invalid_partial_alignment_threshold(value):
    with pytest.raises(MultiTimeframeAnalysisError):
        MultiTimeframeEngine(
            partial_alignment_threshold=value,
        )


def test_full_bullish_alignment():
    engine = MultiTimeframeEngine()

    inputs = make_inputs()

    result = engine.analyze(*inputs)

    assert result.direction is MTFDirection.BULLISH
    assert result.alignment is MTFAlignment.ALIGNED
    assert result.alignment_score == 100.0
    assert result.bullish_timeframes == 3
    assert result.bearish_timeframes == 0
    assert result.neutral_timeframes == 0
    assert result.unknown_timeframes == 0
    assert result.direction_conflict is False
    assert result.sufficient_data is True
    assert result.is_bullish
    assert result.is_aligned
    assert result.should_wait is False


def test_full_bearish_alignment():
    engine = MultiTimeframeEngine()

    inputs = make_inputs(
        higher_bias=ContextBias.BEARISH,
        middle_bias=ContextBias.BEARISH,
        lower_bias=ContextBias.BEARISH,
        higher_regime=MarketRegime.TRENDING_DOWN,
        middle_regime=MarketRegime.TRENDING_DOWN,
        lower_regime=MarketRegime.TRENDING_DOWN,
    )

    result = engine.analyze(*inputs)

    assert result.direction is MTFDirection.BEARISH
    assert result.alignment is MTFAlignment.ALIGNED
    assert result.alignment_score == 100.0
    assert result.bearish_timeframes == 3
    assert result.direction_conflict is False
    assert result.is_bearish
    assert result.is_aligned
    assert result.should_wait is False


def test_two_bullish_one_neutral_is_partial_alignment():
    engine = MultiTimeframeEngine()

    inputs = make_inputs(
        higher_bias=ContextBias.BULLISH,
        middle_bias=ContextBias.BULLISH,
        lower_bias=ContextBias.NEUTRAL,
    )

    result = engine.analyze(*inputs)

    assert result.direction is MTFDirection.BULLISH
    assert result.alignment is MTFAlignment.PARTIALLY_ALIGNED
    assert result.alignment_score == 100.0
    assert result.bullish_timeframes == 2
    assert result.neutral_timeframes == 1
    assert result.direction_conflict is False


def test_two_bearish_one_neutral_is_partial_alignment():
    engine = MultiTimeframeEngine()

    inputs = make_inputs(
        higher_bias=ContextBias.BEARISH,
        middle_bias=ContextBias.BEARISH,
        lower_bias=ContextBias.NEUTRAL,
        higher_regime=MarketRegime.TRENDING_DOWN,
        middle_regime=MarketRegime.TRENDING_DOWN,
        lower_regime=MarketRegime.RANGING,
    )

    result = engine.analyze(*inputs)

    assert result.direction is MTFDirection.BEARISH
    assert result.alignment is MTFAlignment.PARTIALLY_ALIGNED
    assert result.bearish_timeframes == 2
    assert result.neutral_timeframes == 1


def test_bullish_bearish_neutral_is_conflicted():
    engine = MultiTimeframeEngine()

    inputs = make_inputs(
        higher_bias=ContextBias.BULLISH,
        middle_bias=ContextBias.BEARISH,
        lower_bias=ContextBias.NEUTRAL,
        higher_regime=MarketRegime.TRENDING_UP,
        middle_regime=MarketRegime.TRENDING_DOWN,
        lower_regime=MarketRegime.RANGING,
    )

    result = engine.analyze(*inputs)

    assert result.alignment is MTFAlignment.CONFLICTED
    assert result.direction is MTFDirection.UNKNOWN
    assert result.direction_conflict is True
    assert result.is_conflicted
    assert result.should_wait is True


def test_all_neutral_is_neutral():
    engine = MultiTimeframeEngine()

    inputs = make_inputs(
        higher_bias=ContextBias.NEUTRAL,
        middle_bias=ContextBias.NEUTRAL,
        lower_bias=ContextBias.NEUTRAL,
        higher_regime=MarketRegime.RANGING,
        middle_regime=MarketRegime.RANGING,
        lower_regime=MarketRegime.RANGING,
    )

    result = engine.analyze(*inputs)

    assert result.direction is MTFDirection.NEUTRAL
    assert result.alignment is MTFAlignment.NEUTRAL
    assert result.alignment_score == 0.0
    assert result.neutral_timeframes == 3
    assert result.should_wait is True


def test_bullish_and_bearish_without_neutral_is_conflicted():
    engine = MultiTimeframeEngine()

    inputs = make_inputs(
        higher_bias=ContextBias.BULLISH,
        middle_bias=ContextBias.BEARISH,
        lower_bias=ContextBias.BEARISH,
        higher_regime=MarketRegime.TRENDING_UP,
        middle_regime=MarketRegime.TRENDING_DOWN,
        lower_regime=MarketRegime.TRENDING_DOWN,
    )

    result = engine.analyze(*inputs)

    assert result.alignment is MTFAlignment.CONFLICTED
    assert result.direction is MTFDirection.BEARISH
    assert result.direction_conflict is True
    assert result.alignment_score == 66.666667


def test_higher_timeframe_has_50_percent_weight():
    engine = MultiTimeframeEngine()

    inputs = make_inputs(
        higher_strength=100.0,
        middle_strength=0.0,
        lower_strength=0.0,
    )

    result = engine.analyze(*inputs)

    assert result.strength == 50.0


def test_middle_timeframe_has_30_percent_weight():
    engine = MultiTimeframeEngine()

    inputs = make_inputs(
        higher_strength=0.0,
        middle_strength=100.0,
        lower_strength=0.0,
    )

    result = engine.analyze(*inputs)

    assert result.strength == 30.0


def test_lower_timeframe_has_20_percent_weight():
    engine = MultiTimeframeEngine()

    inputs = make_inputs(
        higher_strength=0.0,
        middle_strength=0.0,
        lower_strength=100.0,
    )

    result = engine.analyze(*inputs)

    assert result.strength == 20.0


def test_aligned_strength_is_preserved():
    engine = MultiTimeframeEngine()

    inputs = make_inputs(
        higher_strength=70.0,
        middle_strength=80.0,
        lower_strength=90.0,
    )

    result = engine.analyze(*inputs)

    expected = (
        70.0 * 0.50
        + 80.0 * 0.30
        + 90.0 * 0.20
    )

    assert result.strength == expected


def test_partial_alignment_reduces_strength():
    engine = MultiTimeframeEngine()

    inputs = make_inputs(
        higher_strength=80.0,
        middle_strength=80.0,
        lower_strength=80.0,
        lower_bias=ContextBias.NEUTRAL,
    )

    result = engine.analyze(*inputs)

    assert result.alignment is MTFAlignment.PARTIALLY_ALIGNED
    assert result.strength == 64.0


def test_conflicted_alignment_reduces_strength():
    engine = MultiTimeframeEngine()

    inputs = make_inputs(
        higher_bias=ContextBias.BULLISH,
        middle_bias=ContextBias.BEARISH,
        lower_bias=ContextBias.BEARISH,
        higher_regime=MarketRegime.TRENDING_UP,
        middle_regime=MarketRegime.TRENDING_DOWN,
        lower_regime=MarketRegime.TRENDING_DOWN,
    )

    result = engine.analyze(*inputs)

    raw_strength = (
        80.0 * 0.50
        + 80.0 * 0.30
        + 80.0 * 0.20
    )

    assert result.strength == raw_strength * 0.40


def test_neutral_alignment_reduces_strength():
    engine = MultiTimeframeEngine()

    inputs = make_inputs(
        higher_bias=ContextBias.NEUTRAL,
        middle_bias=ContextBias.NEUTRAL,
        lower_bias=ContextBias.NEUTRAL,
        higher_regime=MarketRegime.RANGING,
        middle_regime=MarketRegime.RANGING,
        lower_regime=MarketRegime.RANGING,
    )

    result = engine.analyze(*inputs)

    assert result.strength == 24.0


def test_unknown_alignment_has_zero_strength():
    engine = MultiTimeframeEngine()

    higher = make_context(
        "H4",
        bias=ContextBias.NEUTRAL,
        strength=80.0,
        sufficient_history=False,
    )

    middle = make_context(
        "H1",
        bias=ContextBias.NEUTRAL,
        strength=80.0,
        sufficient_history=False,
    )

    lower = make_context(
        "M15",
        bias=ContextBias.NEUTRAL,
        strength=80.0,
        sufficient_history=False,
    )

    higher_regime = make_regime(
        "H4",
        regime=MarketRegime.UNKNOWN,
        strength=80.0,
        sufficient_history=False,
    )

    middle_regime = make_regime(
        "H1",
        regime=MarketRegime.UNKNOWN,
        strength=80.0,
        sufficient_history=False,
    )

    lower_regime = make_regime(
        "M15",
        regime=MarketRegime.UNKNOWN,
        strength=80.0,
        sufficient_history=False,
    )

    result = engine.analyze(
        higher,
        middle,
        lower,
        higher_regime,
        middle_regime,
        lower_regime,
    )

    assert result.alignment is MTFAlignment.NEUTRAL
    assert result.strength == 24.0


def test_higher_role_is_correct():
    engine = MultiTimeframeEngine()

    result = engine.analyze(*make_inputs())

    assert result.higher.role is TimeframeRole.HIGHER
    assert result.higher.timeframe == "H4"


def test_middle_role_is_correct():
    engine = MultiTimeframeEngine()

    result = engine.analyze(*make_inputs())

    assert result.middle.role is TimeframeRole.MIDDLE
    assert result.middle.timeframe == "H1"


def test_lower_role_is_correct():
    engine = MultiTimeframeEngine()

    result = engine.analyze(*make_inputs())

    assert result.lower.role is TimeframeRole.LOWER
    assert result.lower.timeframe == "M15"


def test_higher_context_is_copied_correctly():
    engine = MultiTimeframeEngine()

    result = engine.analyze(
        *make_inputs(
            higher_bias=ContextBias.BEARISH,
            higher_strength=65.0,
            higher_regime=MarketRegime.TRENDING_DOWN,
        )
    )

    assert result.higher.bias is ContextBias.BEARISH
    assert result.higher.strength == 65.0
    assert result.higher.regime is MarketRegime.TRENDING_DOWN
    assert result.higher.regime_strength == 65.0


def test_middle_context_is_copied_correctly():
    engine = MultiTimeframeEngine()

    result = engine.analyze(
        *make_inputs(
            middle_bias=ContextBias.BEARISH,
            middle_strength=55.0,
            middle_regime=MarketRegime.TRENDING_DOWN,
        )
    )

    assert result.middle.bias is ContextBias.BEARISH
    assert result.middle.strength == 55.0
    assert result.middle.regime is MarketRegime.TRENDING_DOWN
    assert result.middle.regime_strength == 55.0


def test_lower_context_is_copied_correctly():
    engine = MultiTimeframeEngine()

    result = engine.analyze(
        *make_inputs(
            lower_bias=ContextBias.BEARISH,
            lower_strength=45.0,
            lower_regime=MarketRegime.TRENDING_DOWN,
        )
    )

    assert result.lower.bias is ContextBias.BEARISH
    assert result.lower.strength == 45.0
    assert result.lower.regime is MarketRegime.TRENDING_DOWN
    assert result.lower.regime_strength == 45.0


def test_is_bullish_property():
    result = MultiTimeframeEngine().analyze(*make_inputs())

    assert result.is_bullish is True
    assert result.is_bearish is False
    assert result.is_unknown is False


def test_is_bearish_property():
    result = MultiTimeframeEngine().analyze(
        *make_inputs(
            higher_bias=ContextBias.BEARISH,
            middle_bias=ContextBias.BEARISH,
            lower_bias=ContextBias.BEARISH,
            higher_regime=MarketRegime.TRENDING_DOWN,
            middle_regime=MarketRegime.TRENDING_DOWN,
            lower_regime=MarketRegime.TRENDING_DOWN,
        )
    )

    assert result.is_bearish is True
    assert result.is_bullish is False
    assert result.is_unknown is False


def test_is_conflicted_property():
    result = MultiTimeframeEngine().analyze(
        *make_inputs(
            higher_bias=ContextBias.BULLISH,
            middle_bias=ContextBias.BEARISH,
            lower_bias=ContextBias.BEARISH,
            higher_regime=MarketRegime.TRENDING_UP,
            middle_regime=MarketRegime.TRENDING_DOWN,
            lower_regime=MarketRegime.TRENDING_DOWN,
        )
    )

    assert result.is_conflicted is True
    assert result.should_wait is True


def test_reason_for_full_alignment():
    result = MultiTimeframeEngine().analyze(*make_inputs())

    reason_types = {
        reason.reason_type
        for reason in result.reasons
    }

    assert MTFReasonType.FULL_ALIGNMENT in reason_types
    assert MTFReasonType.STRENGTH_SUPPORT in reason_types


def test_reason_for_partial_alignment():
    result = MultiTimeframeEngine().analyze(
        *make_inputs(
            lower_bias=ContextBias.NEUTRAL,
        )
    )

    reason_types = {
        reason.reason_type
        for reason in result.reasons
    }

    assert MTFReasonType.PARTIAL_ALIGNMENT in reason_types


def test_reason_for_conflict():
    result = MultiTimeframeEngine().analyze(
        *make_inputs(
            higher_bias=ContextBias.BULLISH,
            middle_bias=ContextBias.BEARISH,
            lower_bias=ContextBias.BEARISH,
            higher_regime=MarketRegime.TRENDING_UP,
            middle_regime=MarketRegime.TRENDING_DOWN,
            lower_regime=MarketRegime.TRENDING_DOWN,
        )
    )

    reason_types = {
        reason.reason_type
        for reason in result.reasons
    }

    assert MTFReasonType.DIRECTIONAL_CONFLICT in reason_types
    assert MTFReasonType.STRENGTH_CONFLICT in reason_types


def test_reason_for_neutral_alignment():
    result = MultiTimeframeEngine().analyze(
        *make_inputs(
            higher_bias=ContextBias.NEUTRAL,
            middle_bias=ContextBias.NEUTRAL,
            lower_bias=ContextBias.NEUTRAL,
            higher_regime=MarketRegime.RANGING,
            middle_regime=MarketRegime.RANGING,
            lower_regime=MarketRegime.RANGING,
        )
    )

    reason_types = {
        reason.reason_type
        for reason in result.reasons
    }

    assert MTFReasonType.NEUTRAL_ALIGNMENT in reason_types


def test_regime_reason_trending_up():
    result = MultiTimeframeEngine().analyze(*make_inputs())

    reason_types = {
        reason.reason_type
        for reason in result.reasons
    }

    assert MTFReasonType.TRENDING_UP in reason_types


def test_regime_reason_trending_down():
    result = MultiTimeframeEngine().analyze(
        *make_inputs(
            higher_bias=ContextBias.BEARISH,
            middle_bias=ContextBias.BEARISH,
            lower_bias=ContextBias.BEARISH,
            higher_regime=MarketRegime.TRENDING_DOWN,
            middle_regime=MarketRegime.TRENDING_DOWN,
            lower_regime=MarketRegime.TRENDING_DOWN,
        )
    )

    reason_types = {
        reason.reason_type
        for reason in result.reasons
    }

    assert MTFReasonType.TRENDING_DOWN in reason_types


def test_regime_reason_ranging():
    result = MultiTimeframeEngine().analyze(
        *make_inputs(
            higher_bias=ContextBias.NEUTRAL,
            middle_bias=ContextBias.NEUTRAL,
            lower_bias=ContextBias.NEUTRAL,
            higher_regime=MarketRegime.RANGING,
            middle_regime=MarketRegime.RANGING,
            lower_regime=MarketRegime.RANGING,
        )
    )

    reason_types = {
        reason.reason_type
        for reason in result.reasons
    }

    assert MTFReasonType.RANGING in reason_types


def test_regime_reason_high_volatility():
    result = MultiTimeframeEngine().analyze(
        *make_inputs(
            higher_regime=MarketRegime.HIGH_VOLATILITY,
        )
    )

    reason_types = {
        reason.reason_type
        for reason in result.reasons
    }

    assert MTFReasonType.HIGH_VOLATILITY in reason_types


def test_regime_reason_transition():
    result = MultiTimeframeEngine().analyze(
        *make_inputs(
            higher_regime=MarketRegime.TRANSITION,
        )
    )

    reason_types = {
        reason.reason_type
        for reason in result.reasons
    }

    assert MTFReasonType.TRANSITION in reason_types


def test_regime_reason_unknown():
    result = MultiTimeframeEngine().analyze(
        *make_inputs(
            higher_regime=MarketRegime.UNKNOWN,
        )
    )

    reason_types = {
        reason.reason_type
        for reason in result.reasons
    }

    assert MTFReasonType.UNKNOWN_REGIME in reason_types


def test_high_volatility_warning():
    result = MultiTimeframeEngine().analyze(
        *make_inputs(
            middle_regime=MarketRegime.HIGH_VOLATILITY,
        )
    )

    assert any(
        "high_volatility" in warning
        for warning in result.warnings
    )


def test_transition_warning():
    result = MultiTimeframeEngine().analyze(
        *make_inputs(
            lower_regime=MarketRegime.TRANSITION,
        )
    )

    assert any(
        "transition" in warning
        for warning in result.warnings
    )


def test_insufficient_data_warning():
    higher_context = make_context(
        "H4",
        sufficient_history=False,
    )

    middle_context = make_context(
        "H1",
    )

    lower_context = make_context(
        "M15",
    )

    higher_regime = make_regime(
        "H4",
        sufficient_history=False,
    )

    middle_regime = make_regime("H1")
    lower_regime = make_regime("M15")

    result = MultiTimeframeEngine().analyze(
        higher_context,
        middle_context,
        lower_context,
        higher_regime,
        middle_regime,
        lower_regime,
    )

    assert result.sufficient_data is False
    assert any(
        "sufficient history" in warning
        for warning in result.warnings
    )


def test_context_timestamp_is_preserved():
    timestamp = BASE_TIME + timedelta(minutes=15)

    contexts = make_inputs()

    higher = make_context(
        "H4",
        timestamp=timestamp,
    )

    middle = make_context(
        "H1",
        timestamp=timestamp,
    )

    lower = make_context(
        "M15",
        timestamp=timestamp,
    )

    higher_regime = make_regime(
        "H4",
        timestamp=timestamp,
    )

    middle_regime = make_regime(
        "H1",
        timestamp=timestamp,
    )

    lower_regime = make_regime(
        "M15",
        timestamp=timestamp,
    )

    result = MultiTimeframeEngine().analyze(
        higher,
        middle,
        lower,
        higher_regime,
        middle_regime,
        lower_regime,
    )

    assert result.timestamp == timestamp


def test_xauusd_wrapper_accepts_xauusd():
    result = MultiTimeframeEngine().analyze_xauusd(
        *make_inputs()
    )

    assert result.symbol == "XAUUSD"
    assert result.is_bullish


@pytest.mark.parametrize(
    "bad_symbol",
    ["EURUSD", "GBPUSD", "BTCUSD"],
)
def test_xauusd_wrapper_rejects_non_xauusd_context(bad_symbol):
    inputs = make_inputs()

    higher = make_context(
        "H4",
        symbol=bad_symbol,
    )

    with pytest.raises(MultiTimeframeAnalysisError):
        MultiTimeframeEngine().analyze_xauusd(
            higher,
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
        )


def test_xauusd_wrapper_rejects_non_xauusd_regime():
    inputs = make_inputs()

    bad_regime = make_regime(
        "H4",
        symbol="EURUSD",
    )

    with pytest.raises(MultiTimeframeAnalysisError):
        MultiTimeframeEngine().analyze_xauusd(
            inputs[0],
            inputs[1],
            inputs[2],
            bad_regime,
            inputs[4],
            inputs[5],
        )


def test_mismatched_symbols_are_rejected():
    inputs = make_inputs()

    middle = make_context(
        "H1",
        symbol="EURUSD",
    )

    with pytest.raises(MultiTimeframeAnalysisError):
        MultiTimeframeEngine().analyze(
            inputs[0],
            middle,
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
        )


def test_mismatched_regime_symbols_are_rejected():
    inputs = make_inputs()

    middle_regime = make_regime(
        "H1",
        symbol="EURUSD",
    )

    with pytest.raises(MultiTimeframeAnalysisError):
        MultiTimeframeEngine().analyze(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            middle_regime,
            inputs[5],
        )


def test_context_regime_timeframe_mismatch_is_rejected():
    inputs = make_inputs()

    bad_regime = make_regime(
        "H1",
    )

    with pytest.raises(MultiTimeframeAnalysisError):
        MultiTimeframeEngine().analyze(
            inputs[0],
            inputs[1],
            inputs[2],
            bad_regime,
            inputs[4],
            inputs[5],
        )


def test_context_regime_timestamp_mismatch_is_rejected():
    inputs = make_inputs()

    bad_regime = make_regime(
        "H4",
        timestamp=BASE_TIME + timedelta(minutes=1),
    )

    with pytest.raises(MultiTimeframeAnalysisError):
        MultiTimeframeEngine().analyze(
            inputs[0],
            inputs[1],
            inputs[2],
            bad_regime,
            inputs[4],
            inputs[5],
        )


@pytest.mark.parametrize(
    "higher,middle,lower",
    [
        ("H1", "H4", "M15"),
        ("H4", "M15", "H1"),
        ("M15", "H1", "H4"),
        ("H4", "H4", "M15"),
        ("H4", "H1", "H1"),
    ],
)
def test_invalid_timeframe_order_is_rejected(
    higher,
    middle,
    lower,
):
    higher_context = make_context(higher)
    middle_context = make_context(middle)
    lower_context = make_context(lower)

    higher_regime = make_regime(higher)
    middle_regime = make_regime(middle)
    lower_regime = make_regime(lower)

    with pytest.raises(MultiTimeframeAnalysisError):
        MultiTimeframeEngine().analyze(
            higher_context,
            middle_context,
            lower_context,
            higher_regime,
            middle_regime,
            lower_regime,
        )


def test_unsupported_timeframe_is_rejected():
    higher_context = make_context("H4")
    middle_context = make_context("H1")
    lower_context = make_context("XYZ")

    higher_regime = make_regime("H4")
    middle_regime = make_regime("H1")
    lower_regime = make_regime("XYZ")

    with pytest.raises(MultiTimeframeAnalysisError):
        MultiTimeframeEngine().analyze(
            higher_context,
            middle_context,
            lower_context,
            higher_regime,
            middle_regime,
            lower_regime,
        )


def test_duplicate_timeframes_are_rejected():
    higher_context = make_context("H4")
    middle_context = make_context("H1")
    lower_context = make_context("H1")

    higher_regime = make_regime("H4")
    middle_regime = make_regime("H1")
    lower_regime = make_regime("H1")

    with pytest.raises(MultiTimeframeAnalysisError):
        MultiTimeframeEngine().analyze(
            higher_context,
            middle_context,
            lower_context,
            higher_regime,
            middle_regime,
            lower_regime,
        )


def test_invalid_context_type_is_rejected():
    inputs = make_inputs()

    with pytest.raises(MultiTimeframeAnalysisError):
        MultiTimeframeEngine().analyze(
            "not-a-context",
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
        )


def test_invalid_regime_type_is_rejected():
    inputs = make_inputs()

    with pytest.raises(MultiTimeframeAnalysisError):
        MultiTimeframeEngine().analyze(
            inputs[0],
            inputs[1],
            inputs[2],
            "not-a-regime",
            inputs[4],
            inputs[5],
        )


def test_timeframe_analysis_unknown_bias_is_supported():
    context = make_context(
        "H4",
        bias=ContextBias.NEUTRAL,
    )

    regime = make_regime(
        "H4",
        regime=MarketRegime.UNKNOWN,
    )

    analysis = MultiTimeframeEngine()._build_timeframe_analysis(
        context,
        regime,
        TimeframeRole.HIGHER,
    )

    assert analysis.is_neutral
    assert analysis.is_unknown is False


def test_result_contains_all_three_timeframes():
    result = MultiTimeframeEngine().analyze(*make_inputs())

    assert result.higher.timeframe == "H4"
    assert result.middle.timeframe == "H1"
    assert result.lower.timeframe == "M15"


def test_result_symbol_is_preserved():
    result = MultiTimeframeEngine().analyze(*make_inputs())

    assert result.symbol == "XAUUSD"


def test_result_reason_messages_are_nonempty():
    result = MultiTimeframeEngine().analyze(*make_inputs())

    assert result.reasons
    assert all(
        reason.message.strip()
        for reason in result.reasons
    )


def test_result_warning_messages_are_strings():
    result = MultiTimeframeEngine().analyze(*make_inputs())

    assert all(
        isinstance(warning, str)
        for warning in result.warnings
    )


def test_full_alignment_does_not_require_wait():
    result = MultiTimeframeEngine().analyze(*make_inputs())

    assert result.alignment is MTFAlignment.ALIGNED
    assert result.should_wait is False


def test_conflict_requires_wait():
    result = MultiTimeframeEngine().analyze(
        *make_inputs(
            higher_bias=ContextBias.BULLISH,
            middle_bias=ContextBias.BEARISH,
            lower_bias=ContextBias.BEARISH,
            higher_regime=MarketRegime.TRENDING_UP,
            middle_regime=MarketRegime.TRENDING_DOWN,
            lower_regime=MarketRegime.TRENDING_DOWN,
        )
    )

    assert result.should_wait is True


def test_neutral_requires_wait():
    result = MultiTimeframeEngine().analyze(
        *make_inputs(
            higher_bias=ContextBias.NEUTRAL,
            middle_bias=ContextBias.NEUTRAL,
            lower_bias=ContextBias.NEUTRAL,
            higher_regime=MarketRegime.RANGING,
            middle_regime=MarketRegime.RANGING,
            lower_regime=MarketRegime.RANGING,
        )
    )

    assert result.should_wait is True


def test_alignment_score_is_bounded():
    result = MultiTimeframeEngine().analyze(*make_inputs())

    assert 0.0 <= result.alignment_score <= 100.0


def test_strength_is_bounded():
    result = MultiTimeframeEngine().analyze(*make_inputs())

    assert 0.0 <= result.strength <= 100.0


def test_context_strength_is_clamped():
    context = make_context(
        "H4",
        strength=150.0,
    )

    regime = make_regime(
        "H4",
        strength=150.0,
    )

    analysis = MultiTimeframeEngine()._build_timeframe_analysis(
        context,
        regime,
        TimeframeRole.HIGHER,
    )

    assert analysis.strength == 100.0
    assert analysis.regime_strength == 100.0


def test_negative_context_strength_is_clamped():
    context = make_context(
        "H4",
        strength=-50.0,
    )

    regime = make_regime(
        "H4",
        strength=-50.0,
    )

    analysis = MultiTimeframeEngine()._build_timeframe_analysis(
        context,
        regime,
        TimeframeRole.HIGHER,
    )

    assert analysis.strength == 0.0
    assert analysis.regime_strength == 0.0