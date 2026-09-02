from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.trading.candidate.trade_candidate import (
    CandidateDecision,
    TradeCandidate,
)
from app.trading.risk.trade_planner import (
    RiskPlanningError,
    TradePlan,
    TradePlanningEngine,
)
from app.trading.setup.setup_engine import SetupDirection, SetupType


def make_candidate(
    *,
    direction: SetupDirection = SetupDirection.LONG,
    decision: CandidateDecision = CandidateDecision.TRADE_READY,
    entry_ready: bool = True,
    invalidated: bool = False,
    close: float = 3000.0,
) -> TradeCandidate:
    return TradeCandidate(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        symbol="XAUUSD",
        timeframe="M15",
        close=close,
        decision=decision,
        direction=direction,
        setup_type=SetupType.TREND_CONTINUATION,
        setup_quality_score=90.0,
        confirmation_score=90.0,
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


def make_engine(
    *,
    minimum_risk_reward: float = 2.0,
    maximum_risk_percent: float = 2.0,
) -> TradePlanningEngine:
    return TradePlanningEngine(
        minimum_risk_reward=minimum_risk_reward,
        maximum_risk_percent=maximum_risk_percent,
    )


class TestTradePlanningEngineConfiguration:
    def test_default_configuration(self) -> None:
        engine = TradePlanningEngine()

        assert engine.minimum_risk_reward == 2.0
        assert engine.maximum_risk_percent == 2.0

    def test_custom_configuration(self) -> None:
        engine = TradePlanningEngine(
            minimum_risk_reward=1.5,
            maximum_risk_percent=3.0,
        )

        assert engine.minimum_risk_reward == 1.5
        assert engine.maximum_risk_percent == 3.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"minimum_risk_reward": 0},
            {"minimum_risk_reward": -1},
            {"maximum_risk_percent": 0},
            {"maximum_risk_percent": -1},
            {"price_tolerance": 0},
            {"price_tolerance": -1},
        ],
    )
    def test_invalid_configuration_is_rejected(
        self,
        kwargs: dict,
    ) -> None:
        with pytest.raises(RiskPlanningError):
            TradePlanningEngine(**kwargs)


class TestLongTradePlanning:
    def test_valid_long_trade(self) -> None:
        engine = make_engine()
        candidate = make_candidate(
            direction=SetupDirection.LONG,
            close=3000.0,
        )

        plan = engine.plan(
            candidate,
            account_balance=5000.0,
            risk_percent=1.0,
            stop_loss=2990.0,
            take_profit=3020.0,
            value_per_price_unit=1.0,
        )

        assert isinstance(plan, TradePlan)
        assert plan.valid is True
        assert plan.direction == SetupDirection.LONG
        assert plan.entry_price == 3000.0
        assert plan.stop_loss == 2990.0
        assert plan.take_profit == 3020.0
        assert plan.risk_distance == 10.0
        assert plan.reward_distance == 20.0
        assert plan.risk_reward_ratio == 2.0
        assert plan.risk_amount == 50.0
        assert plan.position_size == 5.0

    def test_long_plan_uses_candidate_close_as_entry(self) -> None:
        engine = make_engine()
        candidate = make_candidate(
            direction=SetupDirection.LONG,
            close=3050.0,
        )

        plan = engine.plan(
            candidate,
            account_balance=10000.0,
            risk_percent=1.0,
            stop_loss=3040.0,
            take_profit=3070.0,
            value_per_price_unit=1.0,
        )

        assert plan.entry_price == 3050.0

    def test_long_stop_must_be_below_entry(self) -> None:
        engine = make_engine()
        candidate = make_candidate(direction=SetupDirection.LONG)

        with pytest.raises(RiskPlanningError):
            engine.plan(
                candidate,
                account_balance=5000.0,
                risk_percent=1.0,
                stop_loss=3000.0,
                take_profit=3020.0,
                value_per_price_unit=1.0,
            )

    def test_long_take_profit_must_be_above_entry(self) -> None:
        engine = make_engine()
        candidate = make_candidate(direction=SetupDirection.LONG)

        with pytest.raises(RiskPlanningError):
            engine.plan(
                candidate,
                account_balance=5000.0,
                risk_percent=1.0,
                stop_loss=2990.0,
                take_profit=3000.0,
                value_per_price_unit=1.0,
            )


class TestShortTradePlanning:
    def test_valid_short_trade(self) -> None:
        engine = make_engine()
        candidate = make_candidate(
            direction=SetupDirection.SHORT,
            close=3000.0,
        )

        plan = engine.plan(
            candidate,
            account_balance=5000.0,
            risk_percent=1.0,
            stop_loss=3010.0,
            take_profit=2980.0,
            value_per_price_unit=1.0,
        )

        assert isinstance(plan, TradePlan)
        assert plan.valid is True
        assert plan.direction == SetupDirection.SHORT
        assert plan.entry_price == 3000.0
        assert plan.stop_loss == 3010.0
        assert plan.take_profit == 2980.0
        assert plan.risk_distance == 10.0
        assert plan.reward_distance == 20.0
        assert plan.risk_reward_ratio == 2.0
        assert plan.risk_amount == 50.0
        assert plan.position_size == 5.0

    def test_short_stop_must_be_above_entry(self) -> None:
        engine = make_engine()
        candidate = make_candidate(direction=SetupDirection.SHORT)

        with pytest.raises(RiskPlanningError):
            engine.plan(
                candidate,
                account_balance=5000.0,
                risk_percent=1.0,
                stop_loss=3000.0,
                take_profit=2980.0,
                value_per_price_unit=1.0,
            )

    def test_short_take_profit_must_be_below_entry(self) -> None:
        engine = make_engine()
        candidate = make_candidate(direction=SetupDirection.SHORT)

        with pytest.raises(RiskPlanningError):
            engine.plan(
                candidate,
                account_balance=5000.0,
                risk_percent=1.0,
                stop_loss=3010.0,
                take_profit=3000.0,
                value_per_price_unit=1.0,
            )


class TestRiskReward:
    def test_exact_minimum_risk_reward_is_allowed(self) -> None:
        engine = make_engine(minimum_risk_reward=2.0)
        candidate = make_candidate()

        plan = engine.plan(
            candidate,
            account_balance=5000.0,
            risk_percent=1.0,
            stop_loss=2990.0,
            take_profit=3020.0,
            value_per_price_unit=1.0,
        )

        assert plan.risk_reward_ratio == 2.0

    def test_below_minimum_risk_reward_is_rejected(self) -> None:
        engine = make_engine(minimum_risk_reward=2.0)
        candidate = make_candidate()

        with pytest.raises(RiskPlanningError):
            engine.plan(
                candidate,
                account_balance=5000.0,
                risk_percent=1.0,
                stop_loss=2990.0,
                take_profit=3010.0,
                value_per_price_unit=1.0,
            )

    def test_higher_risk_reward_is_allowed(self) -> None:
        engine = make_engine(minimum_risk_reward=2.0)
        candidate = make_candidate()

        plan = engine.plan(
            candidate,
            account_balance=5000.0,
            risk_percent=1.0,
            stop_loss=2990.0,
            take_profit=3030.0,
            value_per_price_unit=1.0,
        )

        assert plan.risk_reward_ratio == 3.0

    def test_close_to_minimum_rr_generates_warning(self) -> None:
        engine = make_engine(minimum_risk_reward=2.0)
        candidate = make_candidate()

        plan = engine.plan(
            candidate,
            account_balance=5000.0,
            risk_percent=1.0,
            stop_loss=2990.0,
            take_profit=3020.0,
            value_per_price_unit=1.0,
        )

        assert any(
            "close to the configured minimum" in warning
            for warning in plan.warnings
        )


class TestRiskCalculation:
    def test_risk_amount_is_account_balance_times_risk_percent(
        self,
    ) -> None:
        engine = make_engine()
        candidate = make_candidate()

        plan = engine.plan(
            candidate,
            account_balance=5000.0,
            risk_percent=0.5,
            stop_loss=2990.0,
            take_profit=3020.0,
            value_per_price_unit=1.0,
        )

        assert plan.risk_amount == 25.0

    def test_position_size_uses_risk_amount_and_stop_distance(
        self,
    ) -> None:
        engine = make_engine()
        candidate = make_candidate()

        plan = engine.plan(
            candidate,
            account_balance=5000.0,
            risk_percent=1.0,
            stop_loss=2990.0,
            take_profit=3020.0,
            value_per_price_unit=1.0,
        )

        assert plan.position_size == 5.0

    def test_position_size_respects_value_per_price_unit(
        self,
    ) -> None:
        engine = make_engine()
        candidate = make_candidate()

        plan = engine.plan(
            candidate,
            account_balance=5000.0,
            risk_percent=1.0,
            stop_loss=2990.0,
            take_profit=3020.0,
            value_per_price_unit=2.0,
        )

        assert plan.position_size == 2.5

    def test_maximum_risk_amount_is_calculated(self) -> None:
        engine = make_engine(maximum_risk_percent=2.0)
        candidate = make_candidate()

        plan = engine.plan(
            candidate,
            account_balance=5000.0,
            risk_percent=1.0,
            stop_loss=2990.0,
            take_profit=3020.0,
            value_per_price_unit=1.0,
        )

        assert plan.maximum_risk_amount == 100.0

    def test_risk_at_maximum_generates_warning(self) -> None:
        engine = make_engine(maximum_risk_percent=2.0)
        candidate = make_candidate()

        plan = engine.plan(
            candidate,
            account_balance=5000.0,
            risk_percent=2.0,
            stop_loss=2990.0,
            take_profit=3020.0,
            value_per_price_unit=1.0,
        )

        assert any(
            "at the configured maximum" in warning
            for warning in plan.warnings
        )

    def test_risk_above_maximum_is_rejected(self) -> None:
        engine = make_engine(maximum_risk_percent=2.0)
        candidate = make_candidate()

        with pytest.raises(RiskPlanningError):
            engine.plan(
                candidate,
                account_balance=5000.0,
                risk_percent=2.1,
                stop_loss=2990.0,
                take_profit=3020.0,
                value_per_price_unit=1.0,
            )


class TestDistancePlanning:
    def test_long_distance_plan(self) -> None:
        engine = make_engine()
        candidate = make_candidate(
            direction=SetupDirection.LONG,
            close=3000.0,
        )

        plan = engine.plan_with_distances(
            candidate,
            account_balance=5000.0,
            risk_percent=1.0,
            stop_distance=10.0,
            reward_distance=20.0,
            value_per_price_unit=1.0,
        )

        assert plan.entry_price == 3000.0
        assert plan.stop_loss == 2990.0
        assert plan.take_profit == 3020.0
        assert plan.risk_reward_ratio == 2.0

    def test_short_distance_plan(self) -> None:
        engine = make_engine()
        candidate = make_candidate(
            direction=SetupDirection.SHORT,
            close=3000.0,
        )

        plan = engine.plan_with_distances(
            candidate,
            account_balance=5000.0,
            risk_percent=1.0,
            stop_distance=10.0,
            reward_distance=20.0,
            value_per_price_unit=1.0,
        )

        assert plan.entry_price == 3000.0
        assert plan.stop_loss == 3010.0
        assert plan.take_profit == 2980.0
        assert plan.risk_reward_ratio == 2.0

    @pytest.mark.parametrize(
        "stop_distance,reward_distance",
        [
            (0, 20),
            (-1, 20),
            (10, 0),
            (10, -1),
        ],
    )
    def test_invalid_distances_are_rejected(
        self,
        stop_distance: float,
        reward_distance: float,
    ) -> None:
        engine = make_engine()
        candidate = make_candidate()

        with pytest.raises(RiskPlanningError):
            engine.plan_with_distances(
                candidate,
                account_balance=5000.0,
                risk_percent=1.0,
                stop_distance=stop_distance,
                reward_distance=reward_distance,
                value_per_price_unit=1.0,
            )


class TestCandidateValidation:
    @pytest.mark.parametrize(
        "decision",
        [
            CandidateDecision.WAIT,
            CandidateDecision.REJECT,
        ],
    )
    def test_non_trade_ready_candidate_is_rejected(
        self,
        decision: CandidateDecision,
    ) -> None:
        engine = make_engine()
        candidate = make_candidate(decision=decision)

        with pytest.raises(RiskPlanningError):
            engine.plan(
                candidate,
                account_balance=5000.0,
                risk_percent=1.0,
                stop_loss=2990.0,
                take_profit=3020.0,
                value_per_price_unit=1.0,
            )

    def test_invalidated_candidate_is_rejected(self) -> None:
        engine = make_engine()
        candidate = make_candidate(invalidated=True)

        with pytest.raises(RiskPlanningError):
            engine.plan(
                candidate,
                account_balance=5000.0,
                risk_percent=1.0,
                stop_loss=2990.0,
                take_profit=3020.0,
                value_per_price_unit=1.0,
            )

    def test_not_entry_ready_candidate_is_rejected(self) -> None:
        engine = make_engine()
        candidate = make_candidate(entry_ready=False)

        with pytest.raises(RiskPlanningError):
            engine.plan(
                candidate,
                account_balance=5000.0,
                risk_percent=1.0,
                stop_loss=2990.0,
                take_profit=3020.0,
                value_per_price_unit=1.0,
            )

    def test_wrong_candidate_type_is_rejected(self) -> None:
        engine = make_engine()

        with pytest.raises(RiskPlanningError):
            engine.plan(
                object(),  # type: ignore[arg-type]
                account_balance=5000.0,
                risk_percent=1.0,
                stop_loss=2990.0,
                take_profit=3020.0,
                value_per_price_unit=1.0,
            )


class TestInputValidation:
    @pytest.mark.parametrize(
        "account_balance",
        [0, -1, float("inf"), float("nan")],
    )
    def test_invalid_account_balance_is_rejected(
        self,
        account_balance: float,
    ) -> None:
        engine = make_engine()
        candidate = make_candidate()

        with pytest.raises(RiskPlanningError):
            engine.plan(
                candidate,
                account_balance=account_balance,
                risk_percent=1.0,
                stop_loss=2990.0,
                take_profit=3020.0,
                value_per_price_unit=1.0,
            )

    @pytest.mark.parametrize(
        "risk_percent",
        [0, -1, float("inf"), float("nan")],
    )
    def test_invalid_risk_percent_is_rejected(
        self,
        risk_percent: float,
    ) -> None:
        engine = make_engine()
        candidate = make_candidate()

        with pytest.raises(RiskPlanningError):
            engine.plan(
                candidate,
                account_balance=5000.0,
                risk_percent=risk_percent,
                stop_loss=2990.0,
                take_profit=3020.0,
                value_per_price_unit=1.0,
            )

    @pytest.mark.parametrize(
        "value_per_price_unit",
        [0, -1, float("inf"), float("nan")],
    )
    def test_invalid_value_per_price_unit_is_rejected(
        self,
        value_per_price_unit: float,
    ) -> None:
        engine = make_engine()
        candidate = make_candidate()

        with pytest.raises(RiskPlanningError):
            engine.plan(
                candidate,
                account_balance=5000.0,
                risk_percent=1.0,
                stop_loss=2990.0,
                take_profit=3020.0,
                value_per_price_unit=value_per_price_unit,
            )

    @pytest.mark.parametrize(
        "stop_loss,take_profit",
        [
            (0, 3020),
            (-1, 3020),
            (float("inf"), 3020),
            (2990, float("nan")),
        ],
    )
    def test_invalid_prices_are_rejected(
        self,
        stop_loss: float,
        take_profit: float,
    ) -> None:
        engine = make_engine()
        candidate = make_candidate()

        with pytest.raises(RiskPlanningError):
            engine.plan(
                candidate,
                account_balance=5000.0,
                risk_percent=1.0,
                stop_loss=stop_loss,
                take_profit=take_profit,
                value_per_price_unit=1.0,
            )


class TestDeterminism:
    def test_same_input_produces_same_plan(self) -> None:
        engine = make_engine()
        candidate = make_candidate()

        plan_a = engine.plan(
            candidate,
            account_balance=5000.0,
            risk_percent=1.0,
            stop_loss=2990.0,
            take_profit=3020.0,
            value_per_price_unit=1.0,
        )

        plan_b = engine.plan(
            candidate,
            account_balance=5000.0,
            risk_percent=1.0,
            stop_loss=2990.0,
            take_profit=3020.0,
            value_per_price_unit=1.0,
        )

        assert plan_a == plan_b

    def test_planner_does_not_mutate_candidate(self) -> None:
        engine = make_engine()
        candidate = make_candidate()

        original = candidate

        engine.plan(
            candidate,
            account_balance=5000.0,
            risk_percent=1.0,
            stop_loss=2990.0,
            take_profit=3020.0,
            value_per_price_unit=1.0,
        )

        assert candidate == original