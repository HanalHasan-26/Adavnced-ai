from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.trading.analytics.performance import (
    PerformanceAnalyzer,
    PerformanceAnalyticsError,
)
from app.trading.execution.paper_execution import (
    ExecutionStatus,
    ExitReason,
    TradeResult,
)
from app.trading.setup.setup_engine import SetupDirection


BASE_TIME = datetime(2026, 1, 1)


def make_result(
    *,
    index: int = 0,
    direction=SetupDirection.LONG,
    pnl: float = 100.0,
    r_multiple: float = 1.0,
    status=ExecutionStatus.CLOSED,
    exit_reason=ExitReason.TAKE_PROFIT,
    duration_bars: int = 3,
    mfe: float = 10.0,
    mae: float = 5.0,
) -> TradeResult:
    timestamp = (
        BASE_TIME
        + timedelta(minutes=15 * index)
    )

    exit_timestamp = (
        timestamp
        + timedelta(minutes=15 * duration_bars)
        if status == ExecutionStatus.CLOSED
        else None
    )

    return TradeResult(
        timestamp=timestamp,
        symbol="XAUUSD",
        timeframe="M15",
        direction=direction,
        entry_price=3000.0,
        exit_price=3000.0 + pnl,
        stop_loss=2990.0,
        take_profit=3020.0,
        position_size=1.0,
        risk_amount=100.0,
        pnl=pnl,
        r_multiple=r_multiple,
        status=status,
        exit_reason=exit_reason,
        exit_timestamp=exit_timestamp,
        duration_bars=duration_bars,
        maximum_favorable_excursion=mfe,
        maximum_adverse_excursion=mae,
        gap_exit=False,
    )


class TestValidation:
    def test_results_must_be_a_sequence(self):
        analyzer = PerformanceAnalyzer()

        with pytest.raises(PerformanceAnalyticsError):
            analyzer.analyze(123)

    def test_every_item_must_be_trade_result(self):
        analyzer = PerformanceAnalyzer()

        with pytest.raises(PerformanceAnalyticsError):
            analyzer.analyze(
                [make_result(), "invalid"]
            )

    def test_mixed_symbols_are_rejected(self):
        first = make_result()
        second = TradeResult(
            timestamp=BASE_TIME + timedelta(minutes=15),
            symbol="EURUSD",
            timeframe="M15",
            direction=SetupDirection.LONG,
            entry_price=1.0,
            exit_price=1.1,
            stop_loss=0.9,
            take_profit=1.2,
            position_size=1.0,
            risk_amount=100.0,
            pnl=100.0,
            r_multiple=1.0,
            status=ExecutionStatus.CLOSED,
            exit_reason=ExitReason.TAKE_PROFIT,
            exit_timestamp=BASE_TIME + timedelta(minutes=30),
            duration_bars=1,
            maximum_favorable_excursion=0.1,
            maximum_adverse_excursion=0.05,
            gap_exit=False,
        )

        analyzer = PerformanceAnalyzer()

        with pytest.raises(PerformanceAnalyticsError):
            analyzer.analyze([first, second])

    def test_mixed_timeframes_are_rejected(self):
        first = make_result()

        second = TradeResult(
            timestamp=BASE_TIME + timedelta(minutes=15),
            symbol="XAUUSD",
            timeframe="H1",
            direction=SetupDirection.LONG,
            entry_price=3000.0,
            exit_price=3010.0,
            stop_loss=2990.0,
            take_profit=3020.0,
            position_size=1.0,
            risk_amount=100.0,
            pnl=100.0,
            r_multiple=1.0,
            status=ExecutionStatus.CLOSED,
            exit_reason=ExitReason.TAKE_PROFIT,
            exit_timestamp=BASE_TIME + timedelta(minutes=75),
            duration_bars=1,
            maximum_favorable_excursion=10.0,
            maximum_adverse_excursion=5.0,
            gap_exit=False,
        )

        analyzer = PerformanceAnalyzer()

        with pytest.raises(PerformanceAnalyticsError):
            analyzer.analyze([first, second])


class TestEmptyResults:
    def test_empty_results_are_rejected(self):
        analyzer = PerformanceAnalyzer()

        with pytest.raises(PerformanceAnalyticsError):
            analyzer.analyze([])


class TestBasicMetrics:
    def test_single_winning_trade(self):
        analyzer = PerformanceAnalyzer()

        report = analyzer.analyze(
            [
                make_result(
                    pnl=200.0,
                    r_multiple=2.0,
                )
            ]
        )

        assert report.total_trades == 1
        assert report.closed_trades == 1
        assert report.open_trades == 0

        assert report.winning_trades == 1
        assert report.losing_trades == 0
        assert report.breakeven_trades == 0

        assert report.total_pnl == 200.0
        assert report.average_pnl == 200.0

        assert report.gross_profit == 200.0
        assert report.gross_loss == 0.0

        assert report.win_rate == 100.0
        assert report.loss_rate == 0.0

        assert report.average_win == 200.0
        assert report.average_loss is None

        assert report.total_r == 2.0
        assert report.average_r == 2.0
        assert report.expectancy_r == 2.0

    def test_single_losing_trade(self):
        analyzer = PerformanceAnalyzer()

        report = analyzer.analyze(
            [
                make_result(
                    pnl=-100.0,
                    r_multiple=-1.0,
                    exit_reason=ExitReason.STOP_LOSS,
                )
            ]
        )

        assert report.total_pnl == -100.0
        assert report.gross_profit == 0.0
        assert report.gross_loss == 100.0

        assert report.win_rate == 0.0
        assert report.loss_rate == 100.0

        assert report.average_win is None
        assert report.average_loss == -100.0

        assert report.total_r == -1.0
        assert report.average_r == -1.0
        assert report.expectancy_r == -1.0

    def test_win_rate_uses_closed_trades_only(self):
        analyzer = PerformanceAnalyzer()

        report = analyzer.analyze(
            [
                make_result(
                    index=0,
                    pnl=100.0,
                    r_multiple=1.0,
                ),
                make_result(
                    index=1,
                    pnl=0.0,
                    r_multiple=0.0,
                    status=ExecutionStatus.OPEN,
                    exit_reason=ExitReason.END_OF_DATA,
                ),
            ]
        )

        assert report.total_trades == 2
        assert report.closed_trades == 1
        assert report.open_trades == 1

        assert report.winning_trades == 1
        assert report.losing_trades == 0
        assert report.win_rate == 100.0

    def test_breakeven_trade_is_not_win_or_loss(self):
        analyzer = PerformanceAnalyzer()

        report = analyzer.analyze(
            [
                make_result(
                    pnl=0.0,
                    r_multiple=0.0,
                )
            ]
        )

        assert report.winning_trades == 0
        assert report.losing_trades == 0
        assert report.breakeven_trades == 1

        assert report.win_rate == 0.0
        assert report.loss_rate == 0.0


class TestProfitFactor:
    def test_profit_factor(self):
        analyzer = PerformanceAnalyzer()

        report = analyzer.analyze(
            [
                make_result(
                    index=0,
                    pnl=200.0,
                    r_multiple=2.0,
                ),
                make_result(
                    index=1,
                    pnl=100.0,
                    r_multiple=1.0,
                ),
                make_result(
                    index=2,
                    pnl=-100.0,
                    r_multiple=-1.0,
                    exit_reason=ExitReason.STOP_LOSS,
                ),
            ]
        )

        assert report.gross_profit == 300.0
        assert report.gross_loss == 100.0
        assert report.profit_factor == 3.0

    def test_profit_factor_is_infinite_when_only_winners_exist(self):
        analyzer = PerformanceAnalyzer()

        report = analyzer.analyze(
            [
                make_result(pnl=100.0)
            ]
        )

        assert math_is_infinite(report.profit_factor)

    def test_profit_factor_is_none_when_no_profit_or_loss(self):
        analyzer = PerformanceAnalyzer()

        report = analyzer.analyze(
            [
                make_result(
                    pnl=0.0,
                    r_multiple=0.0,
                )
            ]
        )

        assert report.profit_factor is None


class TestDrawdown:
    def test_maximum_drawdown(self):
        analyzer = PerformanceAnalyzer()

        report = analyzer.analyze(
            [
                make_result(
                    index=0,
                    pnl=200.0,
                    r_multiple=2.0,
                ),
                make_result(
                    index=1,
                    pnl=-100.0,
                    r_multiple=-1.0,
                    exit_reason=ExitReason.STOP_LOSS,
                ),
                make_result(
                    index=2,
                    pnl=-150.0,
                    r_multiple=-1.5,
                    exit_reason=ExitReason.STOP_LOSS,
                ),
            ]
        )

        assert report.maximum_drawdown == 250.0
        assert report.maximum_drawdown_r == 2.5

    def test_drawdown_does_not_become_negative(self):
        analyzer = PerformanceAnalyzer()

        report = analyzer.analyze(
            [
                make_result(
                    pnl=100.0,
                    r_multiple=1.0,
                )
            ]
        )

        assert report.maximum_drawdown == 0.0
        assert report.maximum_drawdown_r == 0.0


class TestConsecutiveResults:
    def test_consecutive_wins(self):
        analyzer = PerformanceAnalyzer()

        report = analyzer.analyze(
            [
                make_result(index=0, pnl=100.0),
                make_result(index=1, pnl=200.0),
                make_result(
                    index=2,
                    pnl=-100.0,
                    r_multiple=-1.0,
                    exit_reason=ExitReason.STOP_LOSS,
                ),
            ]
        )

        assert report.maximum_consecutive_wins == 2
        assert report.maximum_consecutive_losses == 1

    def test_consecutive_losses(self):
        analyzer = PerformanceAnalyzer()

        report = analyzer.analyze(
            [
                make_result(
                    index=0,
                    pnl=-100.0,
                    r_multiple=-1.0,
                    exit_reason=ExitReason.STOP_LOSS,
                ),
                make_result(
                    index=1,
                    pnl=-100.0,
                    r_multiple=-1.0,
                    exit_reason=ExitReason.STOP_LOSS,
                ),
                make_result(index=2, pnl=100.0),
            ]
        )

        assert report.maximum_consecutive_losses == 2
        assert report.maximum_consecutive_wins == 1


class TestDirection:
    def test_long_and_short_counts(self):
        analyzer = PerformanceAnalyzer()

        report = analyzer.analyze(
            [
                make_result(
                    index=0,
                    direction=SetupDirection.LONG,
                ),
                make_result(
                    index=1,
                    direction=SetupDirection.LONG,
                ),
                make_result(
                    index=2,
                    direction=SetupDirection.SHORT,
                ),
            ]
        )

        assert report.long_trades == 2
        assert report.short_trades == 1


class TestDuration:
    def test_average_duration(self):
        analyzer = PerformanceAnalyzer()

        report = analyzer.analyze(
            [
                make_result(
                    index=0,
                    duration_bars=2,
                ),
                make_result(
                    index=1,
                    duration_bars=4,
                ),
            ]
        )

        assert report.average_duration_bars == 3.0


class TestExcursion:
    def test_mfe_and_mae(self):
        analyzer = PerformanceAnalyzer()

        report = analyzer.analyze(
            [
                make_result(
                    index=0,
                    mfe=10.0,
                    mae=5.0,
                ),
                make_result(
                    index=1,
                    mfe=20.0,
                    mae=10.0,
                ),
            ]
        )

        assert report.total_mfe == 30.0
        assert report.average_mfe == 15.0

        assert report.total_mae == 15.0
        assert report.average_mae == 7.5


class TestOrdering:
    def test_results_are_analyzed_chronologically(self):
        analyzer = PerformanceAnalyzer()

        report = analyzer.analyze(
            [
                make_result(
                    index=2,
                    pnl=-100.0,
                    r_multiple=-1.0,
                    exit_reason=ExitReason.STOP_LOSS,
                ),
                make_result(
                    index=0,
                    pnl=200.0,
                    r_multiple=2.0,
                ),
                make_result(
                    index=1,
                    pnl=-50.0,
                    r_multiple=-0.5,
                    exit_reason=ExitReason.STOP_LOSS,
                ),
            ]
        )

        assert report.maximum_drawdown == 150.0
        assert report.maximum_drawdown_r == 1.5


class TestProperties:
    def test_net_profit_alias(self):
        analyzer = PerformanceAnalyzer()

        report = analyzer.analyze(
            [
                make_result(
                    pnl=150.0,
                    r_multiple=1.5,
                )
            ]
        )

        assert report.net_profit == report.total_pnl

    def test_has_enough_trades(self):
        analyzer = PerformanceAnalyzer()

        report = analyzer.analyze(
            [
                make_result()
            ]
        )

        assert report.has_enough_trades is True


def math_is_infinite(value):
    return value is not None and value == float("inf")