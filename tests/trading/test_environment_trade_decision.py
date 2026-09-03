from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.trading.candidate.trade_candidate import (
    CandidateDecision,
    SetupDirection,
    SetupType,
    TradeCandidate,
)
from app.trading.context.market_context import ContextBias, MarketCondition
from app.trading.environment.market_environment import (
    EnvironmentDirection,
    EnvironmentQuality,
    MarketEnvironment,
)
from app.trading.decision.environment_trade_decision import (
    EnvironmentDecisionReasonType,
    EnvironmentTradeDecision,
    EnvironmentTradeDecisionEngine,
    EnvironmentTradeDecisionError,
)
from app.trading.news.news_environment import (
    NewsEnvironmentDirection,
    NewsEnvironmentLevel,
)
from app.trading.regime.market_regime import MarketRegime


TIMESTAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_candidate(
    *,
    decision: CandidateDecision = CandidateDecision.TRADE_READY,
    direction: SetupDirection = SetupDirection.LONG,
    entry_ready: bool = True,
    invalidated: bool = False,
    quality: float = 80.0,
) -> TradeCandidate:
    return TradeCandidate(
        timestamp=TIMESTAMP,
        symbol="XAUUSD",
        timeframe="H1",
        close=3000.0,
        decision=decision,
        direction=direction,
        setup_type=SetupType.TREND_CONTINUATION,
        setup_quality_score=quality,
        confirmation_score=80.0,
        structure_confirmed=True,
        momentum_confirmed=True,
        price_confirmed=True,
        trend_confirmed=True,
        supporting_signals=(),
        conflicting_signals=(),
        reasons=(),
        warnings=(),
        entry_ready=entry_ready,
        invalidated=invalidated,
    )


def make_environment(
    *,
    direction: EnvironmentDirection = EnvironmentDirection.BULLISH,
    strength: float = 80.0,
    quality: EnvironmentQuality = EnvironmentQuality.CLEAR,
    technical_support: bool = True,
    news_support: bool = False,
    conflict: bool = False,
    caution: bool = False,
    sufficient: bool = True,
) -> MarketEnvironment:
    return MarketEnvironment(
        timestamp=TIMESTAMP,
        symbol="XAUUSD",
        timeframe="H1",
        technical_bias=ContextBias.BULLISH,
        technical_strength=80.0,
        market_regime=MarketRegime.TRENDING_UP,
        regime_strength=80.0,
        news_direction=NewsEnvironmentDirection.UNKNOWN,
        news_impact_level=NewsEnvironmentLevel.UNKNOWN,
        news_score=0.0,
        news_confidence=0.0,
        overall_direction=direction,
        overall_strength=strength,
        environment_quality=quality,
        technical_support=technical_support,
        technical_conflict=False,
        news_support=news_support,
        news_conflict=False,
        environment_conflict=conflict,
        caution_required=caution,
        reasons=(),
        warnings=(),
        sufficient_data=sufficient,
    )


class TestEnvironmentTradeDecisionEngine:
    def test_bullish_long_candidate_trades(self):
        engine = EnvironmentTradeDecisionEngine()

        result = engine.decide(
            make_candidate(direction=SetupDirection.LONG),
            make_environment(
                direction=EnvironmentDirection.BULLISH,
            ),
        )

        assert result.decision is EnvironmentTradeDecision.TRADE
        assert result.direction_aligned is True
        assert result.environment_supports_trade is True
        assert result.can_proceed is True

    def test_bearish_short_candidate_trades(self):
        engine = EnvironmentTradeDecisionEngine()

        result = engine.decide(
            make_candidate(direction=SetupDirection.SHORT),
            make_environment(
                direction=EnvironmentDirection.BEARISH,
                technical_support=True,
            ),
        )

        assert result.decision is EnvironmentTradeDecision.TRADE
        assert result.direction_aligned is True
        assert result.environment_supports_trade is True

    def test_long_against_bearish_environment_waits(self):
        engine = EnvironmentTradeDecisionEngine()

        result = engine.decide(
            make_candidate(direction=SetupDirection.LONG),
            make_environment(
                direction=EnvironmentDirection.BEARISH,
            ),
        )

        assert result.decision is EnvironmentTradeDecision.WAIT
        assert result.direction_aligned is False
        assert result.blocked_by_environment is True

    def test_short_against_bullish_environment_waits(self):
        engine = EnvironmentTradeDecisionEngine()

        result = engine.decide(
            make_candidate(direction=SetupDirection.SHORT),
            make_environment(
                direction=EnvironmentDirection.BULLISH,
            ),
        )

        assert result.decision is EnvironmentTradeDecision.WAIT
        assert result.direction_aligned is False

    def test_neutral_environment_waits(self):
        engine = EnvironmentTradeDecisionEngine()

        result = engine.decide(
            make_candidate(),
            make_environment(
                direction=EnvironmentDirection.NEUTRAL,
            ),
        )

        assert result.decision is EnvironmentTradeDecision.WAIT
        assert result.environment_supports_trade is False

    def test_unknown_environment_waits(self):
        engine = EnvironmentTradeDecisionEngine()

        result = engine.decide(
            make_candidate(),
            make_environment(
                direction=EnvironmentDirection.UNKNOWN,
            ),
        )

        assert result.decision is EnvironmentTradeDecision.WAIT

    def test_conflicted_environment_waits(self):
        engine = EnvironmentTradeDecisionEngine()

        result = engine.decide(
            make_candidate(),
            make_environment(
                quality=EnvironmentQuality.CONFLICTED,
                conflict=True,
            ),
        )

        assert result.decision is EnvironmentTradeDecision.WAIT
        assert result.environment_conflict is True
        assert result.blocked_by_environment is True

    def test_caution_environment_waits(self):
        engine = EnvironmentTradeDecisionEngine()

        result = engine.decide(
            make_candidate(),
            make_environment(
                quality=EnvironmentQuality.CAUTION,
                caution=True,
            ),
        )

        assert result.decision is EnvironmentTradeDecision.WAIT
        assert result.caution_required is True

    def test_insufficient_environment_data_waits(self):
        engine = EnvironmentTradeDecisionEngine()

        result = engine.decide(
            make_candidate(),
            make_environment(
                sufficient=False,
                direction=EnvironmentDirection.BULLISH,
            ),
        )

        assert result.decision is EnvironmentTradeDecision.WAIT
        assert result.sufficient_environment_data is False

    def test_low_environment_strength_waits(self):
        engine = EnvironmentTradeDecisionEngine(
            minimum_environment_strength=50.0,
        )

        result = engine.decide(
            make_candidate(),
            make_environment(
                direction=EnvironmentDirection.BULLISH,
                strength=40.0,
            ),
        )

        assert result.decision is EnvironmentTradeDecision.WAIT

    def test_candidate_wait_is_preserved(self):
        engine = EnvironmentTradeDecisionEngine()

        result = engine.decide(
            make_candidate(
                decision=CandidateDecision.WAIT,
            ),
            make_environment(),
        )

        assert result.decision is EnvironmentTradeDecision.WAIT
        assert result.candidate_decision is CandidateDecision.WAIT

    def test_candidate_reject_is_preserved(self):
        engine = EnvironmentTradeDecisionEngine()

        result = engine.decide(
            make_candidate(
                decision=CandidateDecision.REJECT,
            ),
            make_environment(),
        )

        assert result.decision is EnvironmentTradeDecision.REJECT
        assert result.candidate_decision is CandidateDecision.REJECT

    def test_invalid_candidate_is_rejected(self):
        engine = EnvironmentTradeDecisionEngine()

        result = engine.decide(
            make_candidate(
                entry_ready=False,
            ),
            make_environment(),
        )

        assert result.decision is EnvironmentTradeDecision.REJECT
        assert result.is_rejected is True

    def test_invalidated_candidate_is_rejected(self):
        engine = EnvironmentTradeDecisionEngine()

        result = engine.decide(
            make_candidate(
                invalidated=True,
            ),
            make_environment(),
        )

        assert result.decision is EnvironmentTradeDecision.REJECT

    def test_news_support_can_support_trade(self):
        engine = EnvironmentTradeDecisionEngine()

        result = engine.decide(
            make_candidate(),
            make_environment(
                technical_support=False,
                news_support=True,
            ),
        )

        assert result.decision is EnvironmentTradeDecision.TRADE
        assert result.environment_supports_trade is True

    def test_no_technical_or_news_support_waits(self):
        engine = EnvironmentTradeDecisionEngine()

        result = engine.decide(
            make_candidate(),
            make_environment(
                technical_support=False,
                news_support=False,
            ),
        )

        assert result.decision is EnvironmentTradeDecision.WAIT
        assert result.environment_supports_trade is False

    def test_mixed_environment_can_trade_when_direction_and_support_are_valid(self):
        engine = EnvironmentTradeDecisionEngine()

        result = engine.decide(
            make_candidate(),
            make_environment(
                quality=EnvironmentQuality.MIXED,
                direction=EnvironmentDirection.BULLISH,
                technical_support=True,
            ),
        )

        assert result.decision is EnvironmentTradeDecision.TRADE

    def test_favorable_environment_can_trade(self):
        engine = EnvironmentTradeDecisionEngine()

        result = engine.decide(
            make_candidate(),
            make_environment(
                quality=EnvironmentQuality.FAVORABLE,
                direction=EnvironmentDirection.BULLISH,
                strength=70.0,
            ),
        )

        assert result.decision is EnvironmentTradeDecision.TRADE

    def test_xauusd_wrapper_accepts_xauusd(self):
        engine = EnvironmentTradeDecisionEngine()

        result = engine.decide_xauusd(
            make_candidate(),
            make_environment(),
        )

        assert result.decision is EnvironmentTradeDecision.TRADE

    def test_xauusd_wrapper_rejects_other_candidate_symbol(self):
        engine = EnvironmentTradeDecisionEngine()

        candidate = make_candidate()

        object.__setattr__(candidate, "symbol", "EURUSD")

        with pytest.raises(EnvironmentTradeDecisionError):
            engine.decide_xauusd(
                candidate,
                make_environment(),
            )

    def test_mismatched_symbols_are_rejected(self):
        engine = EnvironmentTradeDecisionEngine()

        environment = make_environment()
        object.__setattr__(environment, "symbol", "EURUSD")

        with pytest.raises(EnvironmentTradeDecisionError):
            engine.decide(
                make_candidate(),
                environment,
            )

    def test_mismatched_timeframes_are_rejected(self):
        engine = EnvironmentTradeDecisionEngine()

        environment = make_environment()
        object.__setattr__(environment, "timeframe", "M15")

        with pytest.raises(EnvironmentTradeDecisionError):
            engine.decide(
                make_candidate(),
                environment,
            )

    def test_mismatched_timestamps_are_rejected(self):
        engine = EnvironmentTradeDecisionEngine()

        environment = make_environment()
        object.__setattr__(
            environment,
            "timestamp",
            datetime(2026, 1, 2, tzinfo=timezone.utc),
        )

        with pytest.raises(EnvironmentTradeDecisionError):
            engine.decide(
                make_candidate(),
                environment,
            )

    def test_invalid_constructor_strength_rejected(self):
        with pytest.raises(EnvironmentTradeDecisionError):
            EnvironmentTradeDecisionEngine(
                minimum_environment_strength=101.0,
            )

    def test_negative_constructor_strength_rejected(self):
        with pytest.raises(EnvironmentTradeDecisionError):
            EnvironmentTradeDecisionEngine(
                minimum_environment_strength=-1.0,
            )

    def test_bool_constructor_strength_rejected(self):
        with pytest.raises(EnvironmentTradeDecisionError):
            EnvironmentTradeDecisionEngine(
                minimum_environment_strength=True,
            )

    def test_result_properties(self):
        engine = EnvironmentTradeDecisionEngine()

        trade = engine.decide(
            make_candidate(),
            make_environment(),
        )

        assert trade.is_trade is True
        assert trade.is_wait is False
        assert trade.is_rejected is False
        assert trade.can_proceed is True

        wait = engine.decide(
            make_candidate(),
            make_environment(
                direction=EnvironmentDirection.NEUTRAL,
            ),
        )

        assert wait.is_trade is False
        assert wait.is_wait is True
        assert wait.is_rejected is False
        assert wait.can_proceed is False

        reject = engine.decide(
            make_candidate(
                decision=CandidateDecision.REJECT,
            ),
            make_environment(),
        )

        assert reject.is_trade is False
        assert reject.is_wait is False
        assert reject.is_rejected is True
        assert reject.can_proceed is False

    def test_trade_result_contains_trade_reason(self):
        engine = EnvironmentTradeDecisionEngine()

        result = engine.decide(
            make_candidate(),
            make_environment(),
        )

        reason_types = {
            reason.reason_type
            for reason in result.reasons
        }

        assert (
            EnvironmentDecisionReasonType.DECISION_TRADE
            in reason_types
        )

    def test_wait_result_contains_wait_reason(self):
        engine = EnvironmentTradeDecisionEngine()

        result = engine.decide(
            make_candidate(),
            make_environment(
                direction=EnvironmentDirection.NEUTRAL,
            ),
        )

        reason_types = {
            reason.reason_type
            for reason in result.reasons
        }

        assert (
            EnvironmentDecisionReasonType.DECISION_WAIT
            in reason_types
        )

    def test_reject_result_contains_reject_reason(self):
        engine = EnvironmentTradeDecisionEngine()

        result = engine.decide(
            make_candidate(
                decision=CandidateDecision.REJECT,
            ),
            make_environment(),
        )

        reason_types = {
            reason.reason_type
            for reason in result.reasons
        }

        assert (
            EnvironmentDecisionReasonType.DECISION_REJECT
            in reason_types
        )

    def test_candidate_quality_is_preserved(self):
        engine = EnvironmentTradeDecisionEngine()

        result = engine.decide(
            make_candidate(quality=91.5),
            make_environment(),
        )

        assert result.candidate_quality_score == 91.5