from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import isfinite

from app.trading.data.market_bar import MarketBar
from app.trading.risk.trade_planner import TradePlan
from app.trading.setup.setup_engine import SetupDirection


class PaperExecutionError(ValueError):
    """Raised when paper execution cannot be performed safely."""


class ExecutionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class ExitReason(str, Enum):
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    END_OF_DATA = "END_OF_DATA"


@dataclass(frozen=True, slots=True)
class PaperTrade:
    timestamp: datetime
    symbol: str
    timeframe: str
    direction: SetupDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_amount: float


@dataclass(frozen=True, slots=True)
class TradeResult:
    timestamp: datetime
    symbol: str
    timeframe: str
    direction: SetupDirection

    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    position_size: float

    risk_amount: float
    pnl: float
    r_multiple: float

    status: ExecutionStatus
    exit_reason: ExitReason | None
    exit_timestamp: datetime | None

    duration_bars: int
    maximum_favorable_excursion: float
    maximum_adverse_excursion: float

    gap_exit: bool

    @property
    def is_closed(self) -> bool:
        return self.status == ExecutionStatus.CLOSED

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    @property
    def is_loser(self) -> bool:
        return self.pnl < 0

    @property
    def is_breakeven(self) -> bool:
        return self.pnl == 0


class PaperExecutionEngine:
    """
    Deterministic paper-trading execution engine.

    Execution assumptions:

    1. The trade enters at TradePlan.entry_price.
    2. Future candles are used only after the entry.
    3. LONG:
         TP is reached when candle.high >= TP.
         SL is reached when candle.low <= SL.
    4. SHORT:
         TP is reached when candle.low <= TP.
         SL is reached when candle.high >= SL.
    5. If both SL and TP are touched in the same candle,
       SL is assumed to have happened first.
    6. If a candle opens beyond the SL/TP level, the trade exits
       at the candle open to model a gap.
    7. No future candles are used before their timestamp.
    """

    def execute(
        self,
        plan: TradePlan,
        future_bars: list[MarketBar],
    ) -> TradeResult:
        self._validate_plan(plan)
        self._validate_bars(future_bars)

        if not future_bars:
            raise PaperExecutionError(
                "future_bars must contain at least one bar"
            )

        for bar in future_bars:
            if bar.symbol != plan.symbol:
                raise PaperExecutionError(
                    "future bar symbol does not match trade plan"
                )

            if bar.timeframe != plan.timeframe:
                raise PaperExecutionError(
                    "future bar timeframe does not match trade plan"
                )

            if bar.timestamp <= plan.timestamp:
                raise PaperExecutionError(
                    "future bars must occur after trade entry"
                )

        entry_price = plan.entry_price

        max_favorable = 0.0
        max_adverse = 0.0

        for index, bar in enumerate(future_bars, start=1):
            favorable, adverse = self._calculate_excursion(
                plan.direction,
                entry_price,
                bar,
            )

            max_favorable = max(max_favorable, favorable)
            max_adverse = max(max_adverse, adverse)

            exit_price, exit_reason, gap_exit = self._check_exit(
                plan,
                bar,
            )

            if exit_price is not None:
                pnl = self._calculate_pnl(
                    plan.direction,
                    entry_price,
                    exit_price,
                    plan.position_size,
                )

                r_multiple = self._calculate_r_multiple(
                    pnl,
                    plan.risk_amount,
                )

                return TradeResult(
                    timestamp=plan.timestamp,
                    symbol=plan.symbol,
                    timeframe=plan.timeframe,
                    direction=plan.direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    stop_loss=plan.stop_loss,
                    take_profit=plan.take_profit,
                    position_size=plan.position_size,
                    risk_amount=plan.risk_amount,
                    pnl=pnl,
                    r_multiple=r_multiple,
                    status=ExecutionStatus.CLOSED,
                    exit_reason=exit_reason,
                    exit_timestamp=bar.timestamp,
                    duration_bars=index,
                    maximum_favorable_excursion=max_favorable,
                    maximum_adverse_excursion=max_adverse,
                    gap_exit=gap_exit,
                )

        return TradeResult(
            timestamp=plan.timestamp,
            symbol=plan.symbol,
            timeframe=plan.timeframe,
            direction=plan.direction,
            entry_price=entry_price,
            exit_price=future_bars[-1].close,
            stop_loss=plan.stop_loss,
            take_profit=plan.take_profit,
            position_size=plan.position_size,
            risk_amount=plan.risk_amount,
            pnl=self._calculate_pnl(
                plan.direction,
                entry_price,
                future_bars[-1].close,
                plan.position_size,
            ),
            r_multiple=self._calculate_r_multiple(
                self._calculate_pnl(
                    plan.direction,
                    entry_price,
                    future_bars[-1].close,
                    plan.position_size,
                ),
                plan.risk_amount,
            ),
            status=ExecutionStatus.OPEN,
            exit_reason=ExitReason.END_OF_DATA,
            exit_timestamp=None,
            duration_bars=len(future_bars),
            maximum_favorable_excursion=max_favorable,
            maximum_adverse_excursion=max_adverse,
            gap_exit=False,
        )

    def _check_exit(
        self,
        plan: TradePlan,
        bar: MarketBar,
    ) -> tuple[float | None, ExitReason | None, bool]:
        if plan.direction == SetupDirection.LONG:
            if bar.open <= plan.stop_loss:
                return bar.open, ExitReason.STOP_LOSS, True

            if bar.open >= plan.take_profit:
                return bar.open, ExitReason.TAKE_PROFIT, True

            stop_hit = bar.low <= plan.stop_loss
            target_hit = bar.high >= plan.take_profit

            if stop_hit:
                return plan.stop_loss, ExitReason.STOP_LOSS, False

            if target_hit:
                return plan.take_profit, ExitReason.TAKE_PROFIT, False

            return None, None, False

        if plan.direction == SetupDirection.SHORT:
            if bar.open >= plan.stop_loss:
                return bar.open, ExitReason.STOP_LOSS, True

            if bar.open <= plan.take_profit:
                return bar.open, ExitReason.TAKE_PROFIT, True

            stop_hit = bar.high >= plan.stop_loss
            target_hit = bar.low <= plan.take_profit

            if stop_hit:
                return plan.stop_loss, ExitReason.STOP_LOSS, False

            if target_hit:
                return plan.take_profit, ExitReason.TAKE_PROFIT, False

            return None, None, False

        raise PaperExecutionError(
            "unsupported trade direction"
        )

    @staticmethod
    def _calculate_pnl(
        direction: SetupDirection,
        entry_price: float,
        exit_price: float,
        position_size: float,
    ) -> float:
        if direction == SetupDirection.LONG:
            return (exit_price - entry_price) * position_size

        if direction == SetupDirection.SHORT:
            return (entry_price - exit_price) * position_size

        raise PaperExecutionError(
            "unsupported trade direction"
        )

    @staticmethod
    def _calculate_r_multiple(
        pnl: float,
        risk_amount: float,
    ) -> float:
        if risk_amount <= 0:
            raise PaperExecutionError(
                "risk amount must be greater than zero"
            )

        return pnl / risk_amount

    @staticmethod
    def _calculate_excursion(
        direction: SetupDirection,
        entry_price: float,
        bar: MarketBar,
    ) -> tuple[float, float]:
        if direction == SetupDirection.LONG:
            favorable = max(
                0.0,
                bar.high - entry_price,
            )
            adverse = max(
                0.0,
                entry_price - bar.low,
            )
            return favorable, adverse

        if direction == SetupDirection.SHORT:
            favorable = max(
                0.0,
                entry_price - bar.low,
            )
            adverse = max(
                0.0,
                bar.high - entry_price,
            )
            return favorable, adverse

        raise PaperExecutionError(
            "unsupported trade direction"
        )

    @staticmethod
    def _validate_plan(plan: TradePlan) -> None:
        if not isinstance(plan, TradePlan):
            raise PaperExecutionError(
                "plan must be a TradePlan"
            )

        if not plan.valid:
            raise PaperExecutionError(
                "trade plan must be valid"
            )

        if plan.direction not in (
            SetupDirection.LONG,
            SetupDirection.SHORT,
        ):
            raise PaperExecutionError(
                "trade plan must have LONG or SHORT direction"
            )

        numeric_values = (
            plan.entry_price,
            plan.stop_loss,
            plan.take_profit,
            plan.position_size,
            plan.risk_amount,
        )

        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(float(value))
            for value in numeric_values
        ):
            raise PaperExecutionError(
                "trade plan contains invalid numeric values"
            )

    @staticmethod
    def _validate_bars(
        bars: list[MarketBar],
    ) -> None:
        if not isinstance(bars, list):
            raise PaperExecutionError(
                "future_bars must be a list"
            )

        for bar in bars:
            if not isinstance(bar, MarketBar):
                raise PaperExecutionError(
                    "future_bars must contain only MarketBar objects"
                )