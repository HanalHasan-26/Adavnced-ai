from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.trading.candidate.trade_candidate import (
    CandidateDecision,
    SetupDirection,
    SetupType,
    TradeCandidate,
)
from app.trading.context.market_context import ContextBias
from app.trading.decision.environment_trade_decision import (
    EnvironmentTradeDecision,
    EnvironmentTradeDecisionResult,
)
from app.trading.environment.market_environment import (
    EnvironmentDirection,
    EnvironmentQuality,
    MarketEnvironment,
)
from app.trading.no_trade.no_trade_intelligence import (
    NoTradeAssessment,
    NoTradeDecision,
    NoTradeIntelligenceEngine,
    NoTradeIntelligenceError,
    NoTradeReasonType,
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
    structure_confirmed: bool = True,
    momentum_confirmed: bool = True,
    price_confirmed: bool = True,
    trend_confirmed: bool = True,
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
        structure_confirmed=structure_confirmed,
        momentum_confirmed=momentum_confirmed,
        price_confirmed=price_confirmed,
        trend_confirmed=trend_confirmed,
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
    technical_conflict: bool = False,
    news_support: bool = False,
    news_conflict: bool = False,
    conflict: bool = False,
    caution: bool = False,
    sufficient: bool = True,
    regime: MarketRegime = MarketRegime.TRENDING_UP,
) -> MarketEnvironment:
    return MarketEnvironment(
        timestamp=TIMESTAMP,
        symbol="XAUUSD",
        timeframe="H1",
        technical_bias=ContextBias.BULLISH,
        technical_strength=80.0,
        market_regime=regime,
        regime_strength=80.0,
        news_direction=NewsEnvironmentDirection.UNKNOWN,
        news_impact_level=NewsEnvironmentLevel.UNKNOWN,
        news_score=0.0,
        news_confidence=0.0,
        overall_direction=direction,
        overall_strength=strength,
        environment_quality=quality,
        technical_support=technical_support,
        technical_conflict=technical_conflict,
        news_support=news_support,
        news_conflict=news_conflict,
        environment_conflict=conflict,
        caution_required=caution,
        reasons=(),
        warnings=(),
        sufficient_data=sufficient,
    )


def make_trade_decision(
    *,
    decision: EnvironmentTradeDecision = EnvironmentTradeDecision.TRADE,
) -> EnvironmentTradeDecisionResult:
    candidate = make_candidate()

    return EnvironmentTradeDecisionResult(
        timestamp=TIMESTAMP,
        symbol="XAUUSD",
        timeframe="H1",
        candidate_decision=candidate.decision,
        candidate_direction=candidate.direction,
        candidate_quality_score=candidate.setup_quality_score,
        environment_direction=EnvironmentDirection.BULLISH,
        environment_strength=80.0,
        environment_quality=EnvironmentQuality.CLEAR,
        environment_conflict=False,
        caution_required=False,
        sufficient_environment_data=True,
        decision=decision,
        direction_aligned=True,
        environment_supports_trade=True,
        blocked_by_environment=False,
        reasons=(),
        warnings=(),
    )


class TestNoTradeIntelligenceEngine:
    def test_clear_trade_ready_candidate_returns_clear(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(),
            make_environment(),
        )

        assert isinstance(result, NoTradeAssessment)
        assert result.no_trade_decision is NoTradeDecision.CLEAR
        assert result.no_trade_required is False
        assert result.is_clear is True
        assert result.is_no_trade is False

    def test_rejected_candidate_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(
                decision=CandidateDecision.REJECT,
            ),
            make_environment(),
        )

        assert result.no_trade_decision is NoTradeDecision.NO_TRADE
        assert result.no_trade_required is True
        assert result.is_no_trade is True

    def test_waiting_candidate_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(
                decision=CandidateDecision.WAIT,
            ),
            make_environment(),
        )

        assert result.no_trade_required is True
        assert result.no_trade_decision is NoTradeDecision.NO_TRADE

    def test_invalidated_candidate_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(
                invalidated=True,
            ),
            make_environment(),
        )

        assert result.no_trade_required is True
        assert result.no_trade_decision is NoTradeDecision.NO_TRADE

    def test_not_entry_ready_candidate_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(
                entry_ready=False,
            ),
            make_environment(),
        )

        assert result.no_trade_required is True
        assert result.no_trade_decision is NoTradeDecision.NO_TRADE

    def test_none_direction_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(
                direction=SetupDirection.NONE,
            ),
            make_environment(),
        )

        assert result.no_trade_required is True
        assert result.no_trade_decision is NoTradeDecision.NO_TRADE

    def test_low_setup_quality_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine(
            minimum_setup_quality=60.0,
        )

        result = engine.assess(
            make_candidate(
                quality=40.0,
            ),
            make_environment(),
        )

        assert result.weak_candidate is True
        assert result.no_trade_required is True
        assert result.no_trade_decision is NoTradeDecision.NO_TRADE

    def test_setup_quality_at_threshold_is_not_weak(self):
        engine = NoTradeIntelligenceEngine(
            minimum_setup_quality=60.0,
        )

        result = engine.assess(
            make_candidate(
                quality=60.0,
            ),
            make_environment(),
        )

        assert result.weak_candidate is False

    def test_missing_structure_confirmation_is_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(
                structure_confirmed=False,
            ),
            make_environment(),
        )

        assert result.no_trade_required is True

        reason_types = {
            reason.reason_type
            for reason in result.reasons
        }

        assert (
            NoTradeReasonType.INSUFFICIENT_CONFIRMATION
            in reason_types
        )

    def test_missing_momentum_confirmation_is_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(
                momentum_confirmed=False,
            ),
            make_environment(),
        )

        assert result.no_trade_required is True

    def test_missing_price_confirmation_is_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(
                price_confirmed=False,
            ),
            make_environment(),
        )

        assert result.no_trade_required is True

    def test_missing_trend_confirmation_is_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(
                trend_confirmed=False,
            ),
            make_environment(),
        )

        assert result.no_trade_required is True

    def test_unknown_environment_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(),
            make_environment(
                direction=EnvironmentDirection.UNKNOWN,
            ),
        )

        assert result.no_trade_required is True
        assert result.no_trade_decision is NoTradeDecision.NO_TRADE
        assert result.insufficient_data is False

    def test_neutral_environment_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(),
            make_environment(
                direction=EnvironmentDirection.NEUTRAL,
            ),
        )

        assert result.no_trade_required is True

        reason_types = {
            reason.reason_type
            for reason in result.reasons
        }

        assert (
            NoTradeReasonType.ENVIRONMENT_NEUTRAL
            in reason_types
        )

    def test_insufficient_environment_data_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(),
            make_environment(
                sufficient=False,
            ),
        )

        assert result.insufficient_data is True
        assert result.no_trade_required is True

    def test_weak_environment_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine(
            minimum_environment_strength=50.0,
        )

        result = engine.assess(
            make_candidate(),
            make_environment(
                strength=40.0,
            ),
        )

        assert result.weak_environment is True
        assert result.no_trade_required is True

    def test_environment_strength_at_threshold_is_not_weak(self):
        engine = NoTradeIntelligenceEngine(
            minimum_environment_strength=50.0,
        )

        result = engine.assess(
            make_candidate(),
            make_environment(
                strength=50.0,
            ),
        )

        assert result.weak_environment is False

    def test_long_against_bearish_environment_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(
                direction=SetupDirection.LONG,
            ),
            make_environment(
                direction=EnvironmentDirection.BEARISH,
            ),
        )

        assert result.directional_conflict is True
        assert result.no_trade_required is True

    def test_short_against_bullish_environment_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(
                direction=SetupDirection.SHORT,
            ),
            make_environment(
                direction=EnvironmentDirection.BULLISH,
            ),
        )

        assert result.directional_conflict is True
        assert result.no_trade_required is True

    def test_environment_conflict_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(),
            make_environment(
                conflict=True,
                quality=EnvironmentQuality.CONFLICTED,
            ),
        )

        assert result.environment_conflict_present is True
        assert result.no_trade_required is True

    def test_technical_conflict_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(),
            make_environment(
                technical_conflict=True,
            ),
        )

        assert result.no_trade_required is True

        reason_types = {
            reason.reason_type
            for reason in result.reasons
        }

        assert (
            NoTradeReasonType.TECHNICAL_CONFLICT
            in reason_types
        )

    def test_news_conflict_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(),
            make_environment(
                news_conflict=True,
            ),
        )

        assert result.no_trade_required is True

        reason_types = {
            reason.reason_type
            for reason in result.reasons
        }

        assert (
            NoTradeReasonType.NEWS_CONFLICT
            in reason_types
        )

    def test_caution_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(),
            make_environment(
                caution=True,
                quality=EnvironmentQuality.CAUTION,
            ),
        )

        assert result.caution_present is True
        assert result.no_trade_required is True

    def test_high_volatility_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(),
            make_environment(
                regime=MarketRegime.HIGH_VOLATILITY,
            ),
        )

        assert result.no_trade_required is True

        reason_types = {
            reason.reason_type
            for reason in result.reasons
        }

        assert (
            NoTradeReasonType.HIGH_VOLATILITY
            in reason_types
        )

    def test_transition_regime_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(),
            make_environment(
                regime=MarketRegime.TRANSITION,
            ),
        )

        assert result.no_trade_required is True

    def test_unknown_regime_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(),
            make_environment(
                regime=MarketRegime.UNKNOWN,
            ),
        )

        assert result.no_trade_required is True

    def test_conflicted_quality_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(),
            make_environment(
                quality=EnvironmentQuality.CONFLICTED,
                conflict=True,
            ),
        )

        assert result.no_trade_required is True

    def test_caution_quality_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(),
            make_environment(
                quality=EnvironmentQuality.CAUTION,
                caution=True,
            ),
        )

        assert result.no_trade_required is True

    def test_unknown_quality_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(),
            make_environment(
                quality=EnvironmentQuality.UNKNOWN,
                direction=EnvironmentDirection.UNKNOWN,
            ),
        )

        assert result.no_trade_required is True

    def test_mixed_environment_adds_reason_but_is_not_by_itself_blocking(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(),
            make_environment(
                quality=EnvironmentQuality.MIXED,
            ),
        )

        reason_types = {
            reason.reason_type
            for reason in result.reasons
        }

        assert (
            NoTradeReasonType.ENVIRONMENT_MIXED
            in reason_types
        )

    def test_trade_decision_reject_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(),
            make_environment(),
            make_trade_decision(
                decision=EnvironmentTradeDecision.REJECT,
            ),
        )

        assert result.no_trade_required is True
        assert result.trade_decision is EnvironmentTradeDecision.REJECT

    def test_trade_decision_wait_requires_no_trade(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(),
            make_environment(),
            make_trade_decision(
                decision=EnvironmentTradeDecision.WAIT,
            ),
        )

        assert result.no_trade_required is True
        assert result.trade_decision is EnvironmentTradeDecision.WAIT

    def test_trade_decision_trade_can_be_clear(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(),
            make_environment(),
            make_trade_decision(
                decision=EnvironmentTradeDecision.TRADE,
            ),
        )

        assert result.no_trade_required is False
        assert result.no_trade_decision is NoTradeDecision.CLEAR
        assert result.trade_decision is EnvironmentTradeDecision.TRADE

    def test_no_trade_score_is_bounded(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(
                decision=CandidateDecision.REJECT,
                entry_ready=False,
                invalidated=True,
                direction=SetupDirection.NONE,
                quality=0.0,
                structure_confirmed=False,
                momentum_confirmed=False,
                price_confirmed=False,
                trend_confirmed=False,
            ),
            make_environment(
                direction=EnvironmentDirection.UNKNOWN,
                strength=0.0,
                quality=EnvironmentQuality.CONFLICTED,
                conflict=True,
                caution=True,
                sufficient=False,
                technical_conflict=True,
                news_conflict=True,
                regime=MarketRegime.HIGH_VOLATILITY,
            ),
        )

        assert 0.0 <= result.no_trade_score <= 100.0
        assert result.no_trade_required is True

    def test_no_trade_score_increases_for_more_risk_conditions(self):
        engine = NoTradeIntelligenceEngine()

        clear = engine.assess(
            make_candidate(),
            make_environment(),
        )

        risky = engine.assess(
            make_candidate(
                quality=40.0,
            ),
            make_environment(
                direction=EnvironmentDirection.BEARISH,
                strength=30.0,
                quality=EnvironmentQuality.CONFLICTED,
                conflict=True,
                caution=True,
            ),
        )

        assert risky.no_trade_score > clear.no_trade_score

    def test_reasons_are_recorded(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(
                quality=40.0,
            ),
            make_environment(
                direction=EnvironmentDirection.BEARISH,
            ),
        )

        assert result.reasons

    def test_warnings_are_recorded_for_blocking_conditions(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(
                quality=40.0,
            ),
            make_environment(
                direction=EnvironmentDirection.BEARISH,
                strength=30.0,
                conflict=True,
                caution=True,
                sufficient=False,
            ),
        )

        assert result.warnings

    def test_assessment_preserves_candidate_information(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(
                direction=SetupDirection.SHORT,
                quality=91.5,
            ),
            make_environment(
                direction=EnvironmentDirection.BEARISH,
                strength=88.0,
            ),
        )

        assert result.symbol == "XAUUSD"
        assert result.timeframe == "H1"
        assert result.timestamp == TIMESTAMP
        assert result.candidate_direction is SetupDirection.SHORT
        assert result.candidate_quality_score == 91.5
        assert result.environment_direction is EnvironmentDirection.BEARISH
        assert result.environment_strength == 88.0

    def test_assessment_preserves_environment_information(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(),
            make_environment(
                strength=72.0,
                quality=EnvironmentQuality.FAVORABLE,
            ),
        )

        assert result.environment_strength == 72.0
        assert result.environment_quality is EnvironmentQuality.FAVORABLE
        assert result.environment_conflict is False
        assert result.caution_required is False
        assert result.sufficient_environment_data is True

    def test_xauusd_wrapper_accepts_xauusd(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess_xauusd(
            make_candidate(),
            make_environment(),
        )

        assert result.symbol == "XAUUSD"

    def test_xauusd_wrapper_rejects_other_candidate(self):
        engine = NoTradeIntelligenceEngine()

        candidate = make_candidate()
        object.__setattr__(candidate, "symbol", "EURUSD")

        with pytest.raises(NoTradeIntelligenceError):
            engine.assess_xauusd(
                candidate,
                make_environment(),
            )

    def test_xauusd_wrapper_rejects_other_environment(self):
        engine = NoTradeIntelligenceEngine()

        environment = make_environment()
        object.__setattr__(environment, "symbol", "EURUSD")

        with pytest.raises(NoTradeIntelligenceError):
            engine.assess_xauusd(
                make_candidate(),
                environment,
            )

    def test_mismatched_candidate_environment_symbol_rejected(self):
        engine = NoTradeIntelligenceEngine()

        environment = make_environment()
        object.__setattr__(environment, "symbol", "EURUSD")

        with pytest.raises(NoTradeIntelligenceError):
            engine.assess(
                make_candidate(),
                environment,
            )

    def test_mismatched_timeframe_rejected(self):
        engine = NoTradeIntelligenceEngine()

        environment = make_environment()
        object.__setattr__(environment, "timeframe", "M15")

        with pytest.raises(NoTradeIntelligenceError):
            engine.assess(
                make_candidate(),
                environment,
            )

    def test_mismatched_timestamp_rejected(self):
        engine = NoTradeIntelligenceEngine()

        environment = make_environment()
        object.__setattr__(
            environment,
            "timestamp",
            datetime(2026, 1, 2, tzinfo=timezone.utc),
        )

        with pytest.raises(NoTradeIntelligenceError):
            engine.assess(
                make_candidate(),
                environment,
            )

    def test_invalid_candidate_type_rejected(self):
        engine = NoTradeIntelligenceEngine()

        with pytest.raises(NoTradeIntelligenceError):
            engine.assess(
                object(),  # type: ignore[arg-type]
                make_environment(),
            )

    def test_invalid_environment_type_rejected(self):
        engine = NoTradeIntelligenceEngine()

        with pytest.raises(NoTradeIntelligenceError):
            engine.assess(
                make_candidate(),
                object(),  # type: ignore[arg-type]
            )

    def test_invalid_trade_decision_type_rejected(self):
        engine = NoTradeIntelligenceEngine()

        with pytest.raises(NoTradeIntelligenceError):
            engine.assess(
                make_candidate(),
                make_environment(),
                object(),  # type: ignore[arg-type]
            )

    def test_trade_decision_metadata_mismatch_rejected(self):
        engine = NoTradeIntelligenceEngine()

        trade_decision = make_trade_decision()

        object.__setattr__(
            trade_decision,
            "timestamp",
            datetime(2026, 1, 2, tzinfo=timezone.utc),
        )

        with pytest.raises(NoTradeIntelligenceError):
            engine.assess(
                make_candidate(),
                make_environment(),
                trade_decision,
            )

    def test_invalid_minimum_setup_quality_rejected(self):
        with pytest.raises(NoTradeIntelligenceError):
            NoTradeIntelligenceEngine(
                minimum_setup_quality=101.0,
            )

    def test_negative_minimum_setup_quality_rejected(self):
        with pytest.raises(NoTradeIntelligenceError):
            NoTradeIntelligenceEngine(
                minimum_setup_quality=-1.0,
            )

    def test_invalid_environment_strength_threshold_rejected(self):
        with pytest.raises(NoTradeIntelligenceError):
            NoTradeIntelligenceEngine(
                minimum_environment_strength=101.0,
            )

    def test_invalid_no_trade_score_threshold_rejected(self):
        with pytest.raises(NoTradeIntelligenceError):
            NoTradeIntelligenceEngine(
                no_trade_score_threshold=101.0,
            )

    def test_boolean_threshold_rejected(self):
        with pytest.raises(NoTradeIntelligenceError):
            NoTradeIntelligenceEngine(
                minimum_setup_quality=True,
            )

    def test_string_threshold_rejected(self):
        with pytest.raises(NoTradeIntelligenceError):
            NoTradeIntelligenceEngine(
                minimum_setup_quality="60",  # type: ignore[arg-type]
            )

    def test_no_trade_reason_types_are_present_for_candidate_rejection(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
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
            NoTradeReasonType.CANDIDATE_REJECTED
            in reason_types
        )

    def test_no_trade_reason_types_are_present_for_environment_conflict(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(),
            make_environment(
                conflict=True,
                quality=EnvironmentQuality.CONFLICTED,
            ),
        )

        reason_types = {
            reason.reason_type
            for reason in result.reasons
        }

        assert (
            NoTradeReasonType.ENVIRONMENT_CONFLICTED
            in reason_types
        )

    def test_no_trade_reason_types_are_present_for_caution(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(),
            make_environment(
                caution=True,
                quality=EnvironmentQuality.CAUTION,
            ),
        )

        reason_types = {
            reason.reason_type
            for reason in result.reasons
        }

        assert (
            NoTradeReasonType.ENVIRONMENT_CAUTION
            in reason_types
        )

    def test_clear_result_contains_clear_reason(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(),
            make_environment(),
        )

        reason_types = {
            reason.reason_type
            for reason in result.reasons
        }

        assert (
            NoTradeReasonType.NO_TRADE_CONDITIONS_CLEARED
            in reason_types
        )

    def test_no_trade_result_contains_required_reason(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(
                quality=40.0,
            ),
            make_environment(),
        )

        reason_types = {
            reason.reason_type
            for reason in result.reasons
        }

        assert (
            NoTradeReasonType.NO_TRADE_REQUIRED
            in reason_types
        )

    def test_no_trade_decision_enum_values(self):
        assert NoTradeDecision.NO_TRADE.value == "NO_TRADE"
        assert NoTradeDecision.CLEAR.value == "CLEAR"

    def test_assessment_should_not_trade_property(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(
                quality=40.0,
            ),
            make_environment(),
        )

        assert result.should_not_trade is True

    def test_clear_assessment_should_not_trade_property(self):
        engine = NoTradeIntelligenceEngine()

        result = engine.assess(
            make_candidate(),
            make_environment(),
        )

        assert result.should_not_trade is False