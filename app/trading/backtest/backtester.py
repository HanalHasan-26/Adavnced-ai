from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.trading.data.market_bar import MarketBar
from app.trading.context.market_context import (
    MarketContext,
    MarketContextEngine,
)
from app.trading.setup.setup_engine import (
    SetupEvaluation,
    SetupEngine,
)
from app.trading.candidate.trade_candidate import (
    CandidateDecision,
    TradeCandidate,
    TradeCandidateEngine,
)
from app.trading.risk.trade_planner import (
    TradePlan,
    TradePlanningEngine,
)
from app.trading.execution.paper_execution import (
    PaperExecutionEngine,
    TradeResult,
)


class BacktestError(ValueError):
    """
    Raised when the backtesting engine receives invalid input
    or encounters an invalid backtesting configuration.
    """


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """
    Configuration for one historical backtest.

    The default planning policy uses ATR to calculate the
    initial stop-loss and take-profit distances.

    stop_loss_atr_multiplier:
        ATR multiple used for stop-loss distance.

    take_profit_atr_multiplier:
        ATR multiple used for take-profit distance.

    account_balance:
        Starting account balance used by the risk engine.

    risk_percent:
        Percentage of account balance risked per trade.

    value_per_price_unit:
        Monetary value represented by one unit of price
        movement for one unit of position size.

    minimum_history:
        Minimum number of candles required before the
        backtester begins evaluating trading decisions.

    allow_overlapping_trades:
        Whether another trade may be opened while a previous
        trade is still open.

    The first implementation intentionally defaults to
    non-overlapping trades.
    """

    account_balance: float = 5000.0
    risk_percent: float = 1.0
    value_per_price_unit: float = 1.0

    stop_loss_atr_multiplier: float = 1.0
    take_profit_atr_multiplier: float = 2.0

    minimum_history: int = 30

    allow_overlapping_trades: bool = False


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """
    Result of one historical backtest.
    """

    symbol: str
    timeframe: str

    bars_processed: int
    decisions_evaluated: int

    candidates_trade_ready: int
    trades_executed: int

    skipped_no_trade: int
    skipped_insufficient_history: int
    skipped_invalid_plan: int

    trade_results: tuple[TradeResult, ...]

    @property
    def closed_trades(self) -> tuple[TradeResult, ...]:
        """
        Return only trades that reached a normal exit or
        an end-of-data state.
        """
        return tuple(
            result
            for result in self.trade_results
            if result.is_closed
        )

    @property
    def winning_trades(self) -> tuple[TradeResult, ...]:
        """
        Return winning trade results.
        """
        return tuple(
            result
            for result in self.trade_results
            if result.is_winner
        )

    @property
    def losing_trades(self) -> tuple[TradeResult, ...]:
        """
        Return losing trade results.
        """
        return tuple(
            result
            for result in self.trade_results
            if result.is_loser
        )

    @property
    def open_trades(self) -> tuple[TradeResult, ...]:
        """
        Return trades that remained open at the end of the
        available historical data.
        """
        return tuple(
            result
            for result in self.trade_results
            if not result.is_closed
        )


class Backtester:
    """
    Historical backtesting orchestrator.

    The backtester does not duplicate trading logic.

    It reuses:

        MarketContextEngine
        SetupEngine
        TradeCandidateEngine
        TradePlanningEngine
        PaperExecutionEngine

    For every decision candle, only historical information
    available up to that candle is supplied to the analysis
    engines.

    Future candles are supplied only to PaperExecutionEngine
    after a trade has been planned.

    This prevents the decision logic from seeing future data.
    """

    def __init__(
        self,
        config: BacktestConfig | None = None,
        context_engine: MarketContextEngine | None = None,
        setup_engine: SetupEngine | None = None,
        candidate_engine: TradeCandidateEngine | None = None,
        planning_engine: TradePlanningEngine | None = None,
        execution_engine: PaperExecutionEngine | None = None,
    ) -> None:
        self.config = config or BacktestConfig()

        self._validate_config(self.config)

        self.context_engine = (
            context_engine
            or MarketContextEngine(
                minimum_history=self.config.minimum_history,
            )
        )

        self.setup_engine = (
            setup_engine
            or SetupEngine()
        )

        self.candidate_engine = (
            candidate_engine
            or TradeCandidateEngine()
        )

        self.planning_engine = (
            planning_engine
            or TradePlanningEngine()
        )

        self.execution_engine = (
            execution_engine
            or PaperExecutionEngine()
        )

    def run(
        self,
        bars: list[MarketBar],
    ) -> BacktestResult:
        """
        Run a historical backtest.

        The bars must:

        - be a list
        - contain MarketBar objects
        - use one symbol
        - use one timeframe
        - be strictly chronological

        Returns a deterministic BacktestResult.
        """

        self._validate_bars(bars)

        symbol = bars[0].symbol
        timeframe = bars[0].timeframe

        decisions_evaluated = 0
        candidates_trade_ready = 0
        trades_executed = 0
        skipped_no_trade = 0
        skipped_insufficient_history = 0
        skipped_invalid_plan = 0

        trade_results: list[TradeResult] = []

        index = self.config.minimum_history - 1

        while index < len(bars):
            decisions_evaluated += 1

            historical_bars = bars[: index + 1]

            context = self.context_engine.analyze_at(
                historical_bars,
                len(historical_bars) - 1,
            )

            if not context.sufficient_history:
                skipped_insufficient_history += 1
                index += 1
                continue

            setup = self.setup_engine.evaluate(
                context
            )

            candidate = self.candidate_engine.evaluate(
                setup
            )

            if (
                candidate.decision
                != CandidateDecision.TRADE_READY
            ):
                skipped_no_trade += 1
                index += 1
                continue

            candidates_trade_ready += 1

            plan = self._build_trade_plan(
                candidate=candidate,
                context=context,
            )

            if plan is None:
                skipped_invalid_plan += 1
                index += 1
                continue

            future_bars = bars[index + 1 :]

            if not future_bars:
                break

            result = self.execution_engine.execute(
                plan,
                future_bars,
            )

            trade_results.append(result)
            trades_executed += 1

            if self.config.allow_overlapping_trades:
                index += 1
                continue

            exit_index = self._find_exit_index(
                bars=bars,
                entry_index=index,
                result=result,
            )

            if exit_index is None:
                break

            index = exit_index + 1

        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            bars_processed=len(bars),
            decisions_evaluated=decisions_evaluated,
            candidates_trade_ready=candidates_trade_ready,
            trades_executed=trades_executed,
            skipped_no_trade=skipped_no_trade,
            skipped_insufficient_history=skipped_insufficient_history,
            skipped_invalid_plan=skipped_invalid_plan,
            trade_results=tuple(trade_results),
        )

    def _build_trade_plan(
        self,
        candidate: TradeCandidate,
        context: MarketContext,
    ) -> TradePlan | None:
        """
        Build a TradePlan from a trade candidate.

        The first backtesting strategy uses ATR multiples:

            stop loss = entry +/- ATR * stop multiplier
            take profit = entry +/- ATR * target multiplier

        Direction determines whether the distances are placed
        above or below the entry.

        If ATR is unavailable, no plan is created.
        """

        if context.atr is None:
            return None

        if context.atr <= 0:
            return None

        entry = candidate.close

        stop_distance = (
            context.atr
            * self.config.stop_loss_atr_multiplier
        )

        reward_distance = (
            context.atr
            * self.config.take_profit_atr_multiplier
        )

        if stop_distance <= 0:
            return None

        if reward_distance <= 0:
            return None

        try:
            return self.planning_engine.plan_with_distances(
                candidate=candidate,
                account_balance=self.config.account_balance,
                risk_percent=self.config.risk_percent,
                stop_distance=stop_distance,
                reward_distance=reward_distance,
                value_per_price_unit=(
                    self.config.value_per_price_unit
                ),
            )
        except (
            ValueError,
            RuntimeError,
        ):
            return None

    @staticmethod
    def _find_exit_index(
        bars: list[MarketBar],
        entry_index: int,
        result: TradeResult,
    ) -> int | None:
        """
        Find the historical candle index on which the trade
        exited.

        For END_OF_DATA, there is no actual exit candle beyond
        the final available candle, so return the final index.
        """

        if result.exit_reason.name == "END_OF_DATA":
            return len(bars) - 1

        exit_timestamp = result.exit_timestamp

        if exit_timestamp is None:
            return None

        for index in range(
            entry_index + 1,
            len(bars),
        ):
            if bars[index].timestamp == exit_timestamp:
                return index

        return None

    @staticmethod
    def _validate_config(
        config: BacktestConfig,
    ) -> None:
        if not isinstance(config, BacktestConfig):
            raise BacktestError(
                "config must be a BacktestConfig."
            )

        if (
            not isinstance(
                config.account_balance,
                (int, float),
            )
            or config.account_balance <= 0
        ):
            raise BacktestError(
                "account_balance must be greater than 0."
            )

        if (
            not isinstance(
                config.risk_percent,
                (int, float),
            )
            or config.risk_percent <= 0
        ):
            raise BacktestError(
                "risk_percent must be greater than 0."
            )

        if (
            not isinstance(
                config.value_per_price_unit,
                (int, float),
            )
            or config.value_per_price_unit <= 0
        ):
            raise BacktestError(
                "value_per_price_unit must be greater than 0."
            )

        if (
            not isinstance(
                config.stop_loss_atr_multiplier,
                (int, float),
            )
            or config.stop_loss_atr_multiplier <= 0
        ):
            raise BacktestError(
                "stop_loss_atr_multiplier must be greater than 0."
            )

        if (
            not isinstance(
                config.take_profit_atr_multiplier,
                (int, float),
            )
            or config.take_profit_atr_multiplier <= 0
        ):
            raise BacktestError(
                "take_profit_atr_multiplier must be greater than 0."
            )

        if (
            not isinstance(
                config.minimum_history,
                int,
            )
            or config.minimum_history <= 0
        ):
            raise BacktestError(
                "minimum_history must be greater than 0."
            )

        if not isinstance(
            config.allow_overlapping_trades,
            bool,
        ):
            raise BacktestError(
                "allow_overlapping_trades must be a bool."
            )

    @staticmethod
    def _validate_bars(
        bars: list[MarketBar],
    ) -> None:
        if not isinstance(bars, list):
            raise BacktestError(
                "bars must be a list."
            )

        if not bars:
            raise BacktestError(
                "bars cannot be empty."
            )

        for bar in bars:
            if not isinstance(
                bar,
                MarketBar,
            ):
                raise BacktestError(
                    "every bar must be a MarketBar."
                )

        symbol = bars[0].symbol
        timeframe = bars[0].timeframe

        previous_timestamp = None

        for bar in bars:
            if bar.symbol != symbol:
                raise BacktestError(
                    "all bars must use the same symbol."
                )

            if bar.timeframe != timeframe:
                raise BacktestError(
                    "all bars must use the same timeframe."
                )

            if (
                previous_timestamp is not None
                and bar.timestamp
                <= previous_timestamp
            ):
                raise BacktestError(
                    "bars must be strictly chronological."
                )

            previous_timestamp = bar.timestamp