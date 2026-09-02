from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.trading.data.market_bar import MarketBar
from app.trading.execution.paper_execution import (
    ExecutionStatus,
    ExitReason,
    PaperExecutionEngine,
    PaperExecutionError,
)
from app.trading.risk.trade_planner import TradePlan
from app.trading.setup.setup_engine import SetupDirection


def make_plan(
    *,
    direction: SetupDirection = SetupDirection.LONG,
    entry: float = 3000.0,
    stop: float = 2990.0,
    target: float = 3020.0,
    position_size: float = 5.0,
    risk_amount: float = 50.0,
) -> TradePlan:
    return TradePlan(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        symbol="XAUUSD",
        timeframe="M15",
        direction=direction,
        entry_price=entry,
        stop_loss=stop,
        take_profit=target,
        risk_distance=abs(entry - stop),
        reward_distance=abs(target - entry),
        risk_reward_ratio=2.0,
        account_balance=5000.0,
        risk_percent=1.0,
        risk_amount=risk_amount,
        value_per_price_unit=1.0,
        position_size=position_size,
        maximum_risk_amount=100.0,
        valid=True,
        warnings=(),
    )


def make_bar(
    *,
    index: int = 1,
    open_price: float | None = None,
    high: float = 3005.0,
    low: float = 2995.0,
    close: float = 3002.0,
    symbol: str = "XAUUSD",
    timeframe: str = "M15",
) -> MarketBar:
    timestamp = (
        datetime(2026, 1, 1, tzinfo=timezone.utc)
        + timedelta(minutes=15 * index)
    )

    if open_price is None:
        open_price = close

    return MarketBar(
        timestamp=timestamp,
        symbol=symbol,
        timeframe=timeframe,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=100.0,
    )


class TestLongExecution:
    def test_long_take_profit(self) -> None:
        engine = PaperExecutionEngine()
        plan = make_plan()

        bars = [
            make_bar(
                high=3025.0,
                low=2995.0,
                close=3015.0,
            )
        ]

        result = engine.execute(plan, bars)

        assert result.status == ExecutionStatus.CLOSED
        assert result.exit_reason == ExitReason.TAKE_PROFIT
        assert result.exit_price == 3020.0
        assert result.pnl == 100.0
        assert result.r_multiple == 2.0
        assert result.duration_bars == 1

    def test_long_stop_loss(self) -> None:
        engine = PaperExecutionEngine()
        plan = make_plan()

        bars = [
            make_bar(
                high=3005.0,
                low=2985.0,
                close=2990.0,
            )
        ]

        result = engine.execute(plan, bars)

        assert result.status == ExecutionStatus.CLOSED
        assert result.exit_reason == ExitReason.STOP_LOSS
        assert result.exit_price == 2990.0
        assert result.pnl == -50.0
        assert result.r_multiple == -1.0

    def test_long_remains_open_when_neither_level_is_hit(self) -> None:
        engine = PaperExecutionEngine()
        plan = make_plan()

        bars = [
            make_bar(
                close=3005.0,
            ),
            make_bar(
                index=2,
                high=3010.0,
                low=3002.0,
                close=3008.0,
            ),
        ]

        result = engine.execute(plan, bars)

        assert result.status == ExecutionStatus.OPEN
        assert result.exit_reason == ExitReason.END_OF_DATA
        assert result.exit_timestamp is None
        assert result.exit_price == 3008.0
        assert result.pnl == 40.0
        assert result.r_multiple == 0.8
        assert result.duration_bars == 2


class TestShortExecution:
    def test_short_take_profit(self) -> None:
        engine = PaperExecutionEngine()

        plan = make_plan(
            direction=SetupDirection.SHORT,
            entry=3000.0,
            stop=3010.0,
            target=2980.0,
        )

        bars = [
            make_bar(
                low=2975.0,
                high=3005.0,
                close=2985.0,
            )
        ]

        result = engine.execute(plan, bars)

        assert result.status == ExecutionStatus.CLOSED
        assert result.exit_reason == ExitReason.TAKE_PROFIT
        assert result.exit_price == 2980.0
        assert result.pnl == 100.0
        assert result.r_multiple == 2.0

    def test_short_stop_loss(self) -> None:
        engine = PaperExecutionEngine()

        plan = make_plan(
            direction=SetupDirection.SHORT,
            entry=3000.0,
            stop=3010.0,
            target=2980.0,
        )

        bars = [
            make_bar(
                low=2995.0,
                high=3015.0,
                close=3010.0,
            )
        ]

        result = engine.execute(plan, bars)

        assert result.status == ExecutionStatus.CLOSED
        assert result.exit_reason == ExitReason.STOP_LOSS
        assert result.exit_price == 3010.0
        assert result.pnl == -50.0
        assert result.r_multiple == -1.0


class TestSameCandleAmbiguity:
    def test_long_same_candle_sl_and_tp_uses_stop_loss(self) -> None:
        engine = PaperExecutionEngine()
        plan = make_plan()

        bars = [
            make_bar(
                high=3025.0,
                low=2985.0,
                close=3005.0,
            )
        ]

        result = engine.execute(plan, bars)

        assert result.exit_reason == ExitReason.STOP_LOSS
        assert result.exit_price == 2990.0
        assert result.pnl == -50.0
        assert result.r_multiple == -1.0

    def test_short_same_candle_sl_and_tp_uses_stop_loss(self) -> None:
        engine = PaperExecutionEngine()

        plan = make_plan(
            direction=SetupDirection.SHORT,
            entry=3000.0,
            stop=3010.0,
            target=2980.0,
        )

        bars = [
            make_bar(
                high=3015.0,
                low=2975.0,
                close=2995.0,
            )
        ]

        result = engine.execute(plan, bars)

        assert result.exit_reason == ExitReason.STOP_LOSS
        assert result.exit_price == 3010.0
        assert result.pnl == -50.0


class TestGapExecution:
    def test_long_gap_below_stop_exits_at_open(self) -> None:
        engine = PaperExecutionEngine()
        plan = make_plan()

        bars = [
            make_bar(
                open_price=2980.0,
                high=2990.0,
                low=2975.0,
                close=2985.0,
            )
        ]

        result = engine.execute(plan, bars)

        assert result.exit_reason == ExitReason.STOP_LOSS
        assert result.exit_price == 2980.0
        assert result.gap_exit is True
        assert result.pnl == -100.0
        assert result.r_multiple == -2.0

    def test_long_gap_above_target_exits_at_open(self) -> None:
        engine = PaperExecutionEngine()
        plan = make_plan()

        bars = [
            make_bar(
                open_price=3030.0,
                high=3040.0,
                low=3025.0,
                close=3035.0,
            )
        ]

        result = engine.execute(plan, bars)

        assert result.exit_reason == ExitReason.TAKE_PROFIT
        assert result.exit_price == 3030.0
        assert result.gap_exit is True
        assert result.pnl == 150.0
        assert result.r_multiple == 3.0

    def test_short_gap_above_stop_exits_at_open(self) -> None:
        engine = PaperExecutionEngine()

        plan = make_plan(
            direction=SetupDirection.SHORT,
            entry=3000.0,
            stop=3010.0,
            target=2980.0,
        )

        bars = [
            make_bar(
                open_price=3020.0,
                high=3030.0,
                low=3015.0,
                close=3025.0,
            )
        ]

        result = engine.execute(plan, bars)

        assert result.exit_reason == ExitReason.STOP_LOSS
        assert result.exit_price == 3020.0
        assert result.gap_exit is True
        assert result.pnl == -100.0

    def test_short_gap_below_target_exits_at_open(self) -> None:
        engine = PaperExecutionEngine()

        plan = make_plan(
            direction=SetupDirection.SHORT,
            entry=3000.0,
            stop=3010.0,
            target=2980.0,
        )

        bars = [
            make_bar(
                open_price=2970.0,
                high=2980.0,
                low=2960.0,
                close=2975.0,
            )
        ]

        result = engine.execute(plan, bars)

        assert result.exit_reason == ExitReason.TAKE_PROFIT
        assert result.exit_price == 2970.0
        assert result.gap_exit is True
        assert result.pnl == 150.0


class TestExcursion:
    def test_long_mfe_and_mae(self) -> None:
        engine = PaperExecutionEngine()
        plan = make_plan()

        bars = [
            make_bar(
                high=3010.0,
                low=2995.0,
                close=3005.0,
            ),
            make_bar(
                index=2,
                high=3018.0,
                low=2988.0,
                close=3002.0,
            ),
        ]

        result = engine.execute(plan, bars)

        assert result.maximum_favorable_excursion == 18.0
        assert result.maximum_adverse_excursion == 12.0

    def test_short_mfe_and_mae(self) -> None:
        engine = PaperExecutionEngine()

        plan = make_plan(
            direction=SetupDirection.SHORT,
            entry=3000.0,
            stop=3010.0,
            target=2980.0,
        )

        bars = [
            make_bar(
                high=3005.0,
                low=2990.0,
                close=2995.0,
            ),
            make_bar(
                index=2,
                high=3012.0,
                low=2982.0,
                close=2990.0,
            ),
        ]

        result = engine.execute(plan, bars)

        assert result.maximum_favorable_excursion == 18.0
        assert result.maximum_adverse_excursion == 12.0


class TestMultipleBars:
    def test_trade_closes_on_first_trigger(self) -> None:
        engine = PaperExecutionEngine()
        plan = make_plan()

        bars = [
            make_bar(
                high=3008.0,
                low=2995.0,
                close=3005.0,
            ),
            make_bar(
                index=2,
                high=3025.0,
                low=2995.0,
                close=3015.0,
            ),
            make_bar(
                index=3,
                high=3040.0,
                low=3030.0,
                close=3035.0,
            ),
        ]

        result = engine.execute(plan, bars)

        assert result.exit_reason == ExitReason.TAKE_PROFIT
        assert result.exit_timestamp == bars[1].timestamp
        assert result.duration_bars == 2

    def test_bars_after_exit_are_ignored(self) -> None:
        engine = PaperExecutionEngine()
        plan = make_plan()

        bars = [
            make_bar(
                high=3025.0,
                low=2995.0,
                close=3015.0,
            ),
            make_bar(
                index=2,
                high=3050.0,
                low=3040.0,
                close=3045.0,
            ),
        ]

        result = engine.execute(plan, bars)

        assert result.exit_timestamp == bars[0].timestamp
        assert result.exit_price == 3020.0


class TestValidation:
    def test_empty_bars_rejected(self) -> None:
        engine = PaperExecutionEngine()
        plan = make_plan()

        with pytest.raises(PaperExecutionError):
            engine.execute(plan, [])

    def test_non_list_bars_rejected(self) -> None:
        engine = PaperExecutionEngine()
        plan = make_plan()

        with pytest.raises(PaperExecutionError):
            engine.execute(plan, ())  # type: ignore[arg-type]

    def test_invalid_bar_type_rejected(self) -> None:
        engine = PaperExecutionEngine()
        plan = make_plan()

        with pytest.raises(PaperExecutionError):
            engine.execute(plan, [object()])  # type: ignore[list-item]

    def test_symbol_mismatch_rejected(self) -> None:
        engine = PaperExecutionEngine()
        plan = make_plan()

        bars = [
            make_bar(symbol="EURUSD"),
        ]

        with pytest.raises(PaperExecutionError):
            engine.execute(plan, bars)

    def test_timeframe_mismatch_rejected(self) -> None:
        engine = PaperExecutionEngine()
        plan = make_plan()

        bars = [
            make_bar(timeframe="H1"),
        ]

        with pytest.raises(PaperExecutionError):
            engine.execute(plan, bars)

    def test_bar_before_entry_rejected(self) -> None:
        engine = PaperExecutionEngine()
        plan = make_plan()

        bar = MarketBar(
            timestamp=plan.timestamp,
            symbol="XAUUSD",
            timeframe="M15",
            open=3000.0,
            high=3005.0,
            low=2995.0,
            close=3002.0,
            volume=100.0,
        )

        with pytest.raises(PaperExecutionError):
            engine.execute(plan, [bar])

    def test_invalid_plan_type_rejected(self) -> None:
        engine = PaperExecutionEngine()

        with pytest.raises(PaperExecutionError):
            engine.execute(
                object(),  # type: ignore[arg-type]
                [make_bar()],
            )

    def test_invalid_plan_rejected(self) -> None:
        engine = PaperExecutionEngine()

        plan = make_plan()

        invalid_plan = TradePlan(
            timestamp=plan.timestamp,
            symbol=plan.symbol,
            timeframe=plan.timeframe,
            direction=plan.direction,
            entry_price=plan.entry_price,
            stop_loss=plan.stop_loss,
            take_profit=plan.take_profit,
            risk_distance=plan.risk_distance,
            reward_distance=plan.reward_distance,
            risk_reward_ratio=plan.risk_reward_ratio,
            account_balance=plan.account_balance,
            risk_percent=plan.risk_percent,
            risk_amount=plan.risk_amount,
            value_per_price_unit=plan.value_per_price_unit,
            position_size=plan.position_size,
            maximum_risk_amount=plan.maximum_risk_amount,
            valid=False,
            warnings=(),
        )

        with pytest.raises(PaperExecutionError):
            engine.execute(
                invalid_plan,
                [make_bar()],
            )


class TestDeterminism:
    def test_same_input_produces_same_result(self) -> None:
        engine = PaperExecutionEngine()
        plan = make_plan()

        bars = [
            make_bar(
                high=3025.0,
                low=2995.0,
                close=3015.0,
            )
        ]

        result_a = engine.execute(plan, bars)
        result_b = engine.execute(plan, bars)

        assert result_a == result_b