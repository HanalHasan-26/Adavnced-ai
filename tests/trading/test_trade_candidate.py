from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.trading.candidate.trade_candidate import (
    CandidateDecision,
    CandidateReasonType,
    TradeCandidate,
    TradeCandidateEngine,
)
from app.trading.context.market_context import (
    ContextBias,
    ContextSignalType,
    MarketCondition,
)
from app.trading.setup.setup_engine import (
    SetupDirection,
    SetupEvaluation,
    SetupReason,
    SetupReasonType,
    SetupType,
)


def make_setup(
    *,
    direction: SetupDirection = SetupDirection.LONG,
    setup_type: SetupType = SetupType.TREND_CONTINUATION,
    valid: bool = True,
    quality_score: float = 80.0,
    context_bias: ContextBias = ContextBias.BULLISH,
    context_strength: float = 80.0,
    supporting_signals=None,
    conflicting_signals=None,
    reasons=None,
    warnings=None,
):
    if supporting_signals is None:
        supporting_signals = (
            ContextSignalType.STRUCTURE,
            ContextSignalType.MACD,
            ContextSignalType.PRICE_LOCATION,
        )

    if conflicting_signals is None:
        conflicting_signals = ()

    if reasons is None:
        reasons = (
            SetupReason(
                SetupReasonType.STRUCTURE_ALIGNMENT,
                "Structure supports direction.",
            ),
            SetupReason(
                SetupReasonType.MOMENTUM_ALIGNMENT,
                "Momentum supports direction.",
            ),
            SetupReason(
                SetupReasonType.PRICE_ALIGNMENT,
                "Price supports direction.",
            ),
            SetupReason(
                SetupReasonType.TREND_ALIGNMENT,
                "Trend supports direction.",
            ),
        )

    if warnings is None:
        warnings = ()

    return SetupEvaluation(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        symbol="XAUUSD",
        timeframe="M15",
        close=2650.0,
        direction=direction,
        setup_type=setup_type,
        valid=valid,
        quality_score=quality_score,
        context_bias=context_bias,
        context_strength=context_strength,
        market_condition=MarketCondition.TRENDING_UP,
        supporting_signals=tuple(supporting_signals),
        conflicting_signals=tuple(conflicting_signals),
        reasons=tuple(reasons),
        warnings=tuple(warnings),
    )


def test_default_engine_configuration():
    engine = TradeCandidateEngine()

    assert engine.minimum_confirmation_score == 65.0
    assert engine.trade_ready_score == 75.0
    assert engine.max_conflicts == 1


def test_engine_rejects_invalid_confirmation_threshold():
    with pytest.raises(ValueError):
        TradeCandidateEngine(minimum_confirmation_score=-1)

    with pytest.raises(ValueError):
        TradeCandidateEngine(minimum_confirmation_score=101)


def test_engine_rejects_invalid_trade_ready_score():
    with pytest.raises(ValueError):
        TradeCandidateEngine(trade_ready_score=-1)

    with pytest.raises(ValueError):
        TradeCandidateEngine(trade_ready_score=101)


def test_engine_requires_trade_ready_score_above_confirmation_threshold():
    with pytest.raises(ValueError):
        TradeCandidateEngine(
            minimum_confirmation_score=80,
            trade_ready_score=70,
        )


def test_engine_rejects_invalid_max_conflicts():
    with pytest.raises(ValueError):
        TradeCandidateEngine(max_conflicts=-1)

    with pytest.raises(TypeError):
        TradeCandidateEngine(max_conflicts=1.5)


def test_invalid_setup_type_is_rejected():
    engine = TradeCandidateEngine()

    with pytest.raises(TypeError):
        engine.evaluate(object())


def test_strong_long_setup_becomes_trade_ready():
    engine = TradeCandidateEngine()

    candidate = engine.evaluate(make_setup())

    assert isinstance(candidate, TradeCandidate)
    assert candidate.decision == CandidateDecision.TRADE_READY
    assert candidate.direction == SetupDirection.LONG
    assert candidate.entry_ready is True
    assert candidate.invalidated is False


def test_strong_short_setup_becomes_trade_ready():
    engine = TradeCandidateEngine()

    candidate = engine.evaluate(
        make_setup(
            direction=SetupDirection.SHORT,
            setup_type=SetupType.TREND_CONTINUATION,
            context_bias=ContextBias.BEARISH,
            reasons=(
                SetupReason(
                    SetupReasonType.STRUCTURE_ALIGNMENT,
                    "Structure supports direction.",
                ),
                SetupReason(
                    SetupReasonType.MOMENTUM_ALIGNMENT,
                    "Momentum supports direction.",
                ),
                SetupReason(
                    SetupReasonType.PRICE_ALIGNMENT,
                    "Price supports direction.",
                ),
                SetupReason(
                    SetupReasonType.TREND_ALIGNMENT,
                    "Trend supports direction.",
                ),
            ),
        )
    )

    assert candidate.decision == CandidateDecision.TRADE_READY
    assert candidate.direction == SetupDirection.SHORT


def test_setup_with_one_conflict_can_still_be_trade_ready():
    engine = TradeCandidateEngine(max_conflicts=1)

    candidate = engine.evaluate(
        make_setup(
            conflicting_signals=(ContextSignalType.RSI,),
        )
    )

    assert candidate.decision == CandidateDecision.TRADE_READY
    assert candidate.invalidated is False


def test_setup_with_too_many_conflicts_is_rejected():
    engine = TradeCandidateEngine(max_conflicts=1)

    candidate = engine.evaluate(
        make_setup(
            conflicting_signals=(
                ContextSignalType.RSI,
                ContextSignalType.MACD,
            )
        )
    )

    assert candidate.decision == CandidateDecision.REJECT
    assert candidate.invalidated is True
    assert CandidateReasonType.CONFLICT_PRESENT in {
        reason.reason_type for reason in candidate.reasons
    }


def test_invalid_setup_is_rejected():
    engine = TradeCandidateEngine()

    candidate = engine.evaluate(
        make_setup(valid=False)
    )

    assert candidate.decision == CandidateDecision.REJECT
    assert candidate.invalidated is True


def test_none_direction_is_rejected():
    engine = TradeCandidateEngine()

    candidate = engine.evaluate(
        make_setup(direction=SetupDirection.NONE)
    )

    assert candidate.decision == CandidateDecision.REJECT
    assert candidate.invalidated is True


def test_none_setup_type_is_rejected():
    engine = TradeCandidateEngine()

    candidate = engine.evaluate(
        make_setup(setup_type=SetupType.NONE)
    )

    assert candidate.decision == CandidateDecision.REJECT
    assert candidate.invalidated is True


def test_low_quality_setup_waits_when_not_otherwise_invalidated():
    engine = TradeCandidateEngine(
        minimum_confirmation_score=65,
        trade_ready_score=75,
    )

    candidate = engine.evaluate(
        make_setup(
            quality_score=50,
            reasons=(
                SetupReason(
                    SetupReasonType.STRUCTURE_ALIGNMENT,
                    "Structure supports direction.",
                ),
                SetupReason(
                    SetupReasonType.MOMENTUM_ALIGNMENT,
                    "Momentum supports direction.",
                ),
            ),
        )
    )

    assert candidate.decision in {
        CandidateDecision.WAIT,
        CandidateDecision.REJECT,
    }


def test_insufficient_confirmation_waits():
    engine = TradeCandidateEngine()

    candidate = engine.evaluate(
        make_setup(
            quality_score=65,
            reasons=(
                SetupReason(
                    SetupReasonType.STRUCTURE_ALIGNMENT,
                    "Structure supports direction.",
                ),
            ),
        )
    )

    assert candidate.decision == CandidateDecision.WAIT
    assert candidate.entry_ready is False


def test_missing_trend_confirmation_for_continuation_waits():
    engine = TradeCandidateEngine()

    candidate = engine.evaluate(
        make_setup(
            reasons=(
                SetupReason(
                    SetupReasonType.STRUCTURE_ALIGNMENT,
                    "Structure supports direction.",
                ),
                SetupReason(
                    SetupReasonType.MOMENTUM_ALIGNMENT,
                    "Momentum supports direction.",
                ),
                SetupReason(
                    SetupReasonType.PRICE_ALIGNMENT,
                    "Price supports direction.",
                ),
            )
        )
    )

    assert candidate.trend_confirmed is False
    assert candidate.decision == CandidateDecision.WAIT


def test_reversal_requires_structure_confirmation():
    engine = TradeCandidateEngine()

    candidate = engine.evaluate(
        make_setup(
            setup_type=SetupType.REVERSAL,
            reasons=(
                SetupReason(
                    SetupReasonType.MOMENTUM_ALIGNMENT,
                    "Momentum supports direction.",
                ),
                SetupReason(
                    SetupReasonType.PRICE_ALIGNMENT,
                    "Price supports direction.",
                ),
                SetupReason(
                    SetupReasonType.TREND_ALIGNMENT,
                    "Trend supports direction.",
                ),
            )
        )
    )

    assert candidate.decision == CandidateDecision.REJECT
    assert candidate.invalidated is True
    assert CandidateReasonType.REVERSAL_REQUIRES_CONFIRMATION in {
        reason.reason_type for reason in candidate.reasons
    }


def test_reversal_with_structure_confirmation_can_be_ready():
    engine = TradeCandidateEngine()

    candidate = engine.evaluate(
        make_setup(
            setup_type=SetupType.REVERSAL,
        )
    )

    assert candidate.structure_confirmed is True
    assert candidate.decision == CandidateDecision.TRADE_READY


def test_confirmation_score_is_capped_at_100():
    engine = TradeCandidateEngine()

    candidate = engine.evaluate(
        make_setup(quality_score=100)
    )

    assert candidate.confirmation_score <= 100


def test_confirmation_score_cannot_be_negative():
    engine = TradeCandidateEngine()

    candidate = engine.evaluate(
        make_setup(
            quality_score=0,
            direction=SetupDirection.NONE,
            setup_type=SetupType.NONE,
            conflicting_signals=(
                ContextSignalType.RSI,
                ContextSignalType.MACD,
            ),
            reasons=(),
        )
    )

    assert candidate.confirmation_score >= 0


def test_supporting_and_conflicting_signals_are_preserved():
    engine = TradeCandidateEngine()

    candidate = engine.evaluate(
        make_setup(
            supporting_signals=(
                ContextSignalType.STRUCTURE,
                ContextSignalType.MACD,
            ),
            conflicting_signals=(ContextSignalType.RSI,),
        )
    )

    assert candidate.supporting_signals == (
        ContextSignalType.STRUCTURE,
        ContextSignalType.MACD,
    )
    assert candidate.conflicting_signals == (
        ContextSignalType.RSI,
    )


def test_setup_metadata_is_preserved():
    engine = TradeCandidateEngine()

    candidate = engine.evaluate(make_setup())

    assert candidate.timestamp == datetime(
        2026, 1, 1, tzinfo=timezone.utc
    )
    assert candidate.symbol == "XAUUSD"
    assert candidate.timeframe == "M15"
    assert candidate.close == 2650.0
    assert candidate.setup_quality_score == 80.0


def test_evaluate_at_uses_only_selected_setup():
    engine = TradeCandidateEngine()

    first = make_setup(quality_score=40)
    second = make_setup(quality_score=90)

    candidate = engine.evaluate_at([first, second], 1)

    assert candidate.setup_quality_score == 90.0
    assert candidate.decision == CandidateDecision.TRADE_READY


def test_evaluate_at_rejects_invalid_sequence():
    engine = TradeCandidateEngine()

    with pytest.raises(TypeError):
        engine.evaluate_at(object(), 0)


def test_evaluate_at_rejects_empty_sequence():
    engine = TradeCandidateEngine()

    with pytest.raises(ValueError):
        engine.evaluate_at([], 0)


def test_evaluate_at_rejects_invalid_index():
    engine = TradeCandidateEngine()
    setup = make_setup()

    with pytest.raises(IndexError):
        engine.evaluate_at([setup], 1)

    with pytest.raises(IndexError):
        engine.evaluate_at([setup], -1)


def test_candidate_is_deterministic():
    engine = TradeCandidateEngine()
    setup = make_setup()

    first = engine.evaluate(setup)
    second = engine.evaluate(setup)

    assert first == second


def test_no_trade_ready_when_quality_is_below_trade_ready_threshold():
    engine = TradeCandidateEngine(
        minimum_confirmation_score=50,
        trade_ready_score=95,
    )

    candidate = engine.evaluate(
        make_setup(quality_score=60)
    )

    assert candidate.decision == CandidateDecision.WAIT
    assert candidate.entry_ready is False