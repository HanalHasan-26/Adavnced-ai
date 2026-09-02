from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from app.trading.execution.paper_execution import TradeResult


class PerformanceAnalyticsError(ValueError):
    """
    Raised when performance analytics receives invalid input.
    """


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    """
    Statistical summary of a collection of trade results.

    P&L and win/loss statistics are based on CLOSED trades only.

    Open trades are reported separately and are not counted as
    wins or losses.

    Drawdown is calculated from the cumulative P&L of closed
    trades in chronological order.
    """

    symbol: str
    timeframe: str

    total_trades: int
    closed_trades: int
    open_trades: int

    winning_trades: int
    losing_trades: int
    breakeven_trades: int

    long_trades: int
    short_trades: int

    total_pnl: float
    average_pnl: float

    gross_profit: float
    gross_loss: float
    profit_factor: float | None

    win_rate: float
    loss_rate: float

    average_win: float | None
    average_loss: float | None

    total_r: float
    average_r: float
    expectancy_r: float

    maximum_drawdown: float
    maximum_drawdown_r: float

    maximum_consecutive_wins: int
    maximum_consecutive_losses: int

    average_duration_bars: float | None

    total_mfe: float
    average_mfe: float

    total_mae: float
    average_mae: float

    @property
    def net_profit(self) -> float:
        """
        Alias for total P&L.
        """
        return self.total_pnl

    @property
    def has_enough_trades(self) -> bool:
        """
        Whether at least one closed trade exists.
        """
        return self.closed_trades > 0


class PerformanceAnalyzer:
    """
    Calculates deterministic performance statistics from TradeResult
    objects.

    This class does not execute trades and does not modify trade
    results.

    It is deliberately independent from the backtesting engine so
    that the same analytics can later be used for:

    - backtests
    - paper trading
    - live trading history
    - strategy comparisons
    """

    def analyze(
        self,
        results: Sequence[TradeResult],
    ) -> PerformanceReport:
        """
        Analyze a sequence of trade results.

        Results are sorted chronologically internally before
        drawdown and consecutive-win/loss calculations.

        All results must belong to the same symbol and timeframe.
        """

        self._validate_results(results)

        ordered_results = tuple(
            sorted(
                results,
                key=lambda result: (
                    result.timestamp,
                    result.exit_timestamp
                    if result.exit_timestamp is not None
                    else result.timestamp,
                ),
            )
        )

        symbol = ordered_results[0].symbol
        timeframe = ordered_results[0].timeframe

        for result in ordered_results:
            if result.symbol != symbol:
                raise PerformanceAnalyticsError(
                    "all trade results must use the same symbol."
                )

            if result.timeframe != timeframe:
                raise PerformanceAnalyticsError(
                    "all trade results must use the same timeframe."
                )

        closed = tuple(
            result
            for result in ordered_results
            if result.is_closed
        )

        open_results = tuple(
            result
            for result in ordered_results
            if not result.is_closed
        )

        winning = tuple(
            result
            for result in closed
            if result.is_winner
        )

        losing = tuple(
            result
            for result in closed
            if result.is_loser
        )

        breakeven = tuple(
            result
            for result in closed
            if result.is_breakeven
        )

        long_trades = tuple(
            result
            for result in ordered_results
            if self._direction_value(result) == "LONG"
        )

        short_trades = tuple(
            result
            for result in ordered_results
            if self._direction_value(result) == "SHORT"
        )

        total_pnl = sum(
            result.pnl
            for result in closed
        )

        average_pnl = (
            total_pnl / len(closed)
            if closed
            else 0.0
        )

        gross_profit = sum(
            max(result.pnl, 0.0)
            for result in closed
        )

        gross_loss = sum(
            abs(min(result.pnl, 0.0))
            for result in closed
        )

        if gross_loss > 0:
            profit_factor = (
                gross_profit / gross_loss
            )
        else:
            profit_factor = (
                math.inf
                if gross_profit > 0
                else None
            )

        closed_count = len(closed)

        win_rate = (
            len(winning) / closed_count * 100.0
            if closed_count
            else 0.0
        )

        loss_rate = (
            len(losing) / closed_count * 100.0
            if closed_count
            else 0.0
        )

        average_win = (
            sum(
                result.pnl
                for result in winning
            )
            / len(winning)
            if winning
            else None
        )

        average_loss = (
            sum(
                result.pnl
                for result in losing
            )
            / len(losing)
            if losing
            else None
        )

        total_r = sum(
            result.r_multiple
            for result in closed
        )

        average_r = (
            total_r / closed_count
            if closed_count
            else 0.0
        )

        expectancy_r = self._calculate_expectancy_r(
            closed
        )

        maximum_drawdown = (
            self._calculate_maximum_drawdown(
                closed
            )
        )

        maximum_drawdown_r = (
            self._calculate_maximum_drawdown_r(
                closed
            )
        )

        max_consecutive_wins = (
            self._calculate_max_consecutive(
                closed,
                outcome="win",
            )
        )

        max_consecutive_losses = (
            self._calculate_max_consecutive(
                closed,
                outcome="loss",
            )
        )

        durations = [
            result.duration_bars
            for result in closed
            if result.duration_bars >= 0
        ]

        average_duration_bars = (
            sum(durations) / len(durations)
            if durations
            else None
        )

        total_mfe = sum(
            result.maximum_favorable_excursion
            for result in closed
        )

        average_mfe = (
            total_mfe / closed_count
            if closed_count
            else 0.0
        )

        total_mae = sum(
            result.maximum_adverse_excursion
            for result in closed
        )

        average_mae = (
            total_mae / closed_count
            if closed_count
            else 0.0
        )

        return PerformanceReport(
            symbol=symbol,
            timeframe=timeframe,
            total_trades=len(ordered_results),
            closed_trades=closed_count,
            open_trades=len(open_results),
            winning_trades=len(winning),
            losing_trades=len(losing),
            breakeven_trades=len(breakeven),
            long_trades=len(long_trades),
            short_trades=len(short_trades),
            total_pnl=total_pnl,
            average_pnl=average_pnl,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=profit_factor,
            win_rate=win_rate,
            loss_rate=loss_rate,
            average_win=average_win,
            average_loss=average_loss,
            total_r=total_r,
            average_r=average_r,
            expectancy_r=expectancy_r,
            maximum_drawdown=maximum_drawdown,
            maximum_drawdown_r=maximum_drawdown_r,
            maximum_consecutive_wins=max_consecutive_wins,
            maximum_consecutive_losses=max_consecutive_losses,
            average_duration_bars=average_duration_bars,
            total_mfe=total_mfe,
            average_mfe=average_mfe,
            total_mae=total_mae,
            average_mae=average_mae,
        )

    @staticmethod
    def _calculate_expectancy_r(
        results: Sequence[TradeResult],
    ) -> float:
        """
        Calculate expectancy in R.

        E[R] = average R-multiple per closed trade.
        """

        if not results:
            return 0.0

        return (
            sum(
                result.r_multiple
                for result in results
            )
            / len(results)
        )

    @staticmethod
    def _calculate_maximum_drawdown(
        results: Sequence[TradeResult],
    ) -> float:
        """
        Calculate maximum peak-to-trough drawdown in account
        currency based on cumulative closed-trade P&L.

        Returns a positive number representing the magnitude
        of the drawdown.
        """

        if not results:
            return 0.0

        equity = 0.0
        peak = 0.0
        maximum_drawdown = 0.0

        for result in results:
            equity += result.pnl

            if equity > peak:
                peak = equity

            drawdown = peak - equity

            if drawdown > maximum_drawdown:
                maximum_drawdown = drawdown

        return maximum_drawdown

    @staticmethod
    def _calculate_maximum_drawdown_r(
        results: Sequence[TradeResult],
    ) -> float:
        """
        Calculate maximum drawdown in R.

        This uses cumulative R-multiple rather than account
        currency, making the statistic useful across different
        account sizes and risk amounts.
        """

        if not results:
            return 0.0

        cumulative_r = 0.0
        peak_r = 0.0
        maximum_drawdown_r = 0.0

        for result in results:
            cumulative_r += result.r_multiple

            if cumulative_r > peak_r:
                peak_r = cumulative_r

            drawdown_r = (
                peak_r - cumulative_r
            )

            if drawdown_r > maximum_drawdown_r:
                maximum_drawdown_r = drawdown_r

        return maximum_drawdown_r

    @staticmethod
    def _calculate_max_consecutive(
        results: Sequence[TradeResult],
        outcome: str,
    ) -> int:
        """
        Calculate the maximum consecutive wins or losses.

        Breakeven trades break both streaks.
        """

        if outcome not in {
            "win",
            "loss",
        }:
            raise PerformanceAnalyticsError(
                "outcome must be 'win' or 'loss'."
            )

        current = 0
        maximum = 0

        for result in results:
            matches = (
                result.is_winner
                if outcome == "win"
                else result.is_loser
            )

            if matches:
                current += 1
                maximum = max(
                    maximum,
                    current,
                )
            else:
                current = 0

        return maximum

    @staticmethod
    def _direction_value(
        result: TradeResult,
    ) -> str:
        """
        Safely normalize the direction value.

        TradeResult.direction is normally an enum, but using its
        value when available keeps analytics compatible with the
        existing trading models.
        """

        direction = result.direction

        value = getattr(
            direction,
            "value",
            direction,
        )

        return str(value).upper()

    @staticmethod
    def _validate_results(
        results: Sequence[TradeResult],
    ) -> None:
        if not isinstance(
            results,
            Sequence,
        ):
            raise PerformanceAnalyticsError(
                "results must be a sequence."
            )

        if len(results) == 0:
            raise PerformanceAnalyticsError(
                "results cannot be empty."
            )

        for result in results:
            if not isinstance(
                result,
                TradeResult,
            ):
                raise PerformanceAnalyticsError(
                    "every result must be a TradeResult."
                )

            numeric_values = (
                result.pnl,
                result.r_multiple,
                result.maximum_favorable_excursion,
                result.maximum_adverse_excursion,
            )

            for value in numeric_values:
                if not isinstance(
                    value,
                    (int, float),
                ):
                    raise PerformanceAnalyticsError(
                        "trade result numeric values must be numbers."
                    )

                if not math.isfinite(
                    float(value)
                ):
                    raise PerformanceAnalyticsError(
                        "trade result numeric values must be finite."
                    )