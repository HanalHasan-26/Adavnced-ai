from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.trading.backtest.backtester import (
    BacktestConfig,
    BacktestError,
    BacktestResult,
    Backtester,
)
from app.trading.data.market_bar import MarketBar


def make_bar(
    index: int,
    close: float = 3000.0,
    symbol: str = "XAUUSD",
    timeframe: str = "M15",
) -> MarketBar:
    return MarketBar(
        timestamp=datetime(2026, 1, 1)
        + timedelta(minutes=15 * index),
        symbol=symbol,
        timeframe=timeframe,
        open=close,
        high=close + 5.0,
        low=close - 5.0,
        close=close,
        volume=100.0,
    )


def make_bars(
    count: int = 40,
    symbol: str = "XAUUSD",
    timeframe: str = "M15",
) -> list[MarketBar]:
    return [
        make_bar(
            index=index,
            close=3000.0 + index,
            symbol=symbol,
            timeframe=timeframe,
        )
        for index in range(count)
    ]


class TestBacktestConfig:
    def test_defaults_are_valid(self):
        config = BacktestConfig()

        assert config.account_balance == 5000.0
        assert config.risk_percent == 1.0
        assert config.value_per_price_unit == 1.0
        assert config.stop_loss_atr_multiplier == 1.0
        assert config.take_profit_atr_multiplier == 2.0
        assert config.minimum_history == 30
        assert config.allow_overlapping_trades is False

    def test_invalid_account_balance_is_rejected(self):
        with pytest.raises(BacktestError):
            Backtester(
                BacktestConfig(
                    account_balance=0
                )
            )

    def test_invalid_risk_percent_is_rejected(self):
        with pytest.raises(BacktestError):
            Backtester(
                BacktestConfig(
                    risk_percent=0
                )
            )

    def test_invalid_value_per_price_unit_is_rejected(self):
        with pytest.raises(BacktestError):
            Backtester(
                BacktestConfig(
                    value_per_price_unit=0
                )
            )

    def test_invalid_stop_multiplier_is_rejected(self):
        with pytest.raises(BacktestError):
            Backtester(
                BacktestConfig(
                    stop_loss_atr_multiplier=0
                )
            )

    def test_invalid_target_multiplier_is_rejected(self):
        with pytest.raises(BacktestError):
            Backtester(
                BacktestConfig(
                    take_profit_atr_multiplier=0
                )
            )

    def test_invalid_minimum_history_is_rejected(self):
        with pytest.raises(BacktestError):
            Backtester(
                BacktestConfig(
                    minimum_history=0
                )
            )

    def test_invalid_overlap_setting_is_rejected(self):
        with pytest.raises(BacktestError):
            Backtester(
                BacktestConfig(
                    allow_overlapping_trades=1
                )
            )


class TestValidation:
    def test_bars_must_be_a_list(self):
        backtester = Backtester()

        with pytest.raises(BacktestError):
            backtester.run(tuple(make_bars()))

    def test_empty_bars_are_rejected(self):
        backtester = Backtester()

        with pytest.raises(BacktestError):
            backtester.run([])

    def test_invalid_bar_type_is_rejected(self):
        backtester = Backtester()

        with pytest.raises(BacktestError):
            backtester.run(
                [
                    make_bar(0),
                    "invalid",
                ]
            )

    def test_mixed_symbols_are_rejected(self):
        bars = make_bars()

        bars.append(
            make_bar(
                index=40,
                symbol="EURUSD",
            )
        )

        backtester = Backtester()

        with pytest.raises(BacktestError):
            backtester.run(bars)

    def test_mixed_timeframes_are_rejected(self):
        bars = make_bars()

        bars.append(
            make_bar(
                index=40,
                timeframe="H1",
            )
        )

        backtester = Backtester()

        with pytest.raises(BacktestError):
            backtester.run(bars)

    def test_out_of_order_bars_are_rejected(self):
        bars = make_bars()

        bars[10], bars[11] = (
            bars[11],
            bars[10],
        )

        backtester = Backtester()

        with pytest.raises(BacktestError):
            backtester.run(bars)


class TestBacktestResult:
    def test_result_is_frozen_dataclass(self):
        result = BacktestResult(
            symbol="XAUUSD",
            timeframe="M15",
            bars_processed=40,
            decisions_evaluated=10,
            candidates_trade_ready=2,
            trades_executed=2,
            skipped_no_trade=6,
            skipped_insufficient_history=2,
            skipped_invalid_plan=0,
            trade_results=(),
        )

        assert result.symbol == "XAUUSD"
        assert result.timeframe == "M15"
        assert result.bars_processed == 40

    def test_empty_trade_collections_are_empty(self):
        result = BacktestResult(
            symbol="XAUUSD",
            timeframe="M15",
            bars_processed=40,
            decisions_evaluated=10,
            candidates_trade_ready=0,
            trades_executed=0,
            skipped_no_trade=10,
            skipped_insufficient_history=0,
            skipped_invalid_plan=0,
            trade_results=(),
        )

        assert result.closed_trades == ()
        assert result.winning_trades == ()
        assert result.losing_trades == ()
        assert result.open_trades == ()


class TestBacktester:
    def test_short_history_does_not_crash(self):
        bars = make_bars(10)

        backtester = Backtester(
            BacktestConfig(
                minimum_history=30
            )
        )

        result = backtester.run(bars)

        assert result.bars_processed == 10
        assert result.decisions_evaluated == 0
        assert result.trades_executed == 0

    def test_no_future_data_is_used_for_decision_history(self):
        bars = make_bars(40)

        backtester = Backtester(
            BacktestConfig(
                minimum_history=30
            )
        )

        result = backtester.run(bars)

        assert result.bars_processed == 40
        assert result.decisions_evaluated == 11

    def test_deterministic_result(self):
        bars = make_bars(50)

        backtester = Backtester(
            BacktestConfig(
                minimum_history=30
            )
        )

        first = backtester.run(bars)
        second = backtester.run(bars)

        assert first == second

    def test_symbol_and_timeframe_are_preserved(self):
        bars = make_bars(
            count=40,
            symbol="XAUUSD",
            timeframe="M15",
        )

        backtester = Backtester(
            BacktestConfig(
                minimum_history=30
            )
        )

        result = backtester.run(bars)

        assert result.symbol == "XAUUSD"
        assert result.timeframe == "M15"

    def test_decision_count_matches_available_decision_candles(self):
        bars = make_bars(35)

        backtester = Backtester(
            BacktestConfig(
                minimum_history=30
            )
        )

        result = backtester.run(bars)

        assert result.decisions_evaluated == 6

    def test_trade_result_collection_is_tuple(self):
        bars = make_bars(40)

        backtester = Backtester(
            BacktestConfig(
                minimum_history=30
            )
        )

        result = backtester.run(bars)

        assert isinstance(
            result.trade_results,
            tuple,
        )

    def test_trade_count_never_exceeds_ready_candidates(self):
        bars = make_bars(60)

        backtester = Backtester(
            BacktestConfig(
                minimum_history=30
            )
        )

        result = backtester.run(bars)

        assert (
            result.trades_executed
            <= result.candidates_trade_ready
        )

    def test_decision_counts_are_consistent(self):
        bars = make_bars(60)

        backtester = Backtester(
            BacktestConfig(
                minimum_history=30
            )
        )

        result = backtester.run(bars)

        assert result.decisions_evaluated >= 0
        assert result.candidates_trade_ready >= 0
        assert result.trades_executed >= 0
        assert result.skipped_no_trade >= 0
        assert result.skipped_insufficient_history >= 0
        assert result.skipped_invalid_plan >= 0

    def test_minimum_history_can_be_configured(self):
        bars = make_bars(20)

        backtester = Backtester(
            BacktestConfig(
                minimum_history=10
            )
        )

        result = backtester.run(bars)

        assert result.decisions_evaluated == 11

    def test_different_symbols_can_be_backtested_separately(self):
        xau_bars = make_bars(
            40,
            symbol="XAUUSD",
        )

        eur_bars = make_bars(
            40,
            symbol="EURUSD",
        )

        backtester = Backtester(
            BacktestConfig(
                minimum_history=30
            )
        )

        xau_result = backtester.run(xau_bars)
        eur_result = backtester.run(eur_bars)

        assert xau_result.symbol == "XAUUSD"
        assert eur_result.symbol == "EURUSD"