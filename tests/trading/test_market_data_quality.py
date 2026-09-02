from datetime import datetime

import pytest

from app.trading.data.market_bar import MarketBar
from app.trading.data.market_data_quality import (
    MarketDataQualityChecker,
    QualityIssueType,
    QualitySeverity,
)


def make_bar(
    minute: int,
    *,
    symbol: str = "XAUUSD",
    timeframe: str = "M5",
    high: float = 2005.0,
    low: float = 1998.0,
) -> MarketBar:
    return MarketBar(
        timestamp=datetime(
            2026,
            1,
            1,
            10,
            minute,
        ),
        symbol=symbol,
        timeframe=timeframe,
        open=2000.0,
        high=high,
        low=low,
        close=2003.0,
        volume=100.0,
    )


def test_valid_data_passes():
    bars = [
        make_bar(0),
        make_bar(5),
        make_bar(10),
    ]

    report = MarketDataQualityChecker().check(bars)

    assert report.valid is True
    assert report.bars_checked == 3
    assert report.errors == ()
    assert report.warnings == ()


def test_empty_data_is_error():
    report = MarketDataQualityChecker().check([])

    assert report.valid is False
    assert report.bars_checked == 0

    assert report.issues[0].issue_type == (
        QualityIssueType.EMPTY_DATA
    )

    assert report.issues[0].severity == (
        QualitySeverity.ERROR
    )


def test_insufficient_history_is_error():
    checker = MarketDataQualityChecker(
        minimum_history=10
    )

    report = checker.check(
        [make_bar(0)]
    )

    assert report.valid is False

    assert any(
        issue.issue_type
        == QualityIssueType.INSUFFICIENT_HISTORY
        for issue in report.errors
    )


def test_invalid_item_type_is_error():
    report = MarketDataQualityChecker().check(
        [make_bar(0), "invalid"]
    )

    assert report.valid is False

    assert any(
        issue.issue_type
        == QualityIssueType.INVALID_BAR
        for issue in report.errors
    )


def test_duplicate_timestamp_is_error():
    bars = [
        make_bar(0),
        make_bar(5),
        make_bar(5),
    ]

    report = MarketDataQualityChecker().check(bars)

    assert report.valid is False

    assert any(
        issue.issue_type
        == QualityIssueType.DUPLICATE_TIMESTAMP
        for issue in report.errors
    )


def test_out_of_order_is_error():
    bars = [
        make_bar(10),
        make_bar(5),
    ]

    report = MarketDataQualityChecker().check(bars)

    assert report.valid is False

    assert any(
        issue.issue_type
        == QualityIssueType.OUT_OF_ORDER
        for issue in report.errors
    )


def test_mixed_symbols_are_error():
    bars = [
        make_bar(
            0,
            symbol="XAUUSD",
        ),
        make_bar(
            5,
            symbol="EURUSD",
        ),
    ]

    report = MarketDataQualityChecker().check(bars)

    assert report.valid is False

    assert any(
        issue.issue_type
        == QualityIssueType.MIXED_SYMBOL
        for issue in report.errors
    )


def test_mixed_timeframes_are_error():
    bars = [
        make_bar(
            0,
            timeframe="M5",
        ),
        make_bar(
            5,
            timeframe="H1",
        ),
    ]

    report = MarketDataQualityChecker().check(bars)

    assert report.valid is False

    assert any(
        issue.issue_type
        == QualityIssueType.MIXED_TIMEFRAME
        for issue in report.errors
    )


def test_expected_symbol_is_enforced():
    report = MarketDataQualityChecker().check(
        [make_bar(0)],
        expected_symbol="EURUSD",
    )

    assert report.valid is False

    assert any(
        issue.issue_type
        == QualityIssueType.MIXED_SYMBOL
        for issue in report.errors
    )


def test_expected_timeframe_is_enforced():
    report = MarketDataQualityChecker().check(
        [make_bar(0)],
        expected_timeframe="H1",
    )

    assert report.valid is False

    assert any(
        issue.issue_type
        == QualityIssueType.MIXED_TIMEFRAME
        for issue in report.errors
    )


def test_gap_is_warning_not_error():
    bars = [
        make_bar(0),
        make_bar(30),
    ]

    report = MarketDataQualityChecker().check(bars)

    assert report.valid is True

    assert any(
        issue.issue_type
        == QualityIssueType.GAP
        for issue in report.warnings
    )

    assert not any(
        issue.issue_type
        == QualityIssueType.GAP
        for issue in report.errors
    )


def test_normal_timeframe_spacing_has_no_gap():
    bars = [
        make_bar(0),
        make_bar(5),
        make_bar(10),
    ]

    report = MarketDataQualityChecker().check(bars)

    assert not any(
        issue.issue_type
        == QualityIssueType.GAP
        for issue in report.issues
    )


def test_invalid_ohlc_is_rejected_by_market_bar():
    with pytest.raises(ValueError):
        MarketBar(
            timestamp=datetime(
                2026,
                1,
                1,
                10,
                0,
            ),
            symbol="XAUUSD",
            timeframe="M5",
            open=2000.0,
            high=1990.0,
            low=1980.0,
            close=2003.0,
            volume=100.0,
        )


def test_negative_volume_is_rejected_by_market_bar():
    with pytest.raises(ValueError):
        MarketBar(
            timestamp=datetime(
                2026,
                1,
                1,
                10,
                0,
            ),
            symbol="XAUUSD",
            timeframe="M5",
            open=2000.0,
            high=2005.0,
            low=1998.0,
            close=2003.0,
            volume=-1.0,
        )


def test_non_finite_value_is_detected():
    valid_bar = make_bar(0)

    invalid_bar = MarketBar.__new__(
        MarketBar
    )

    object.__setattr__(
        invalid_bar,
        "timestamp",
        valid_bar.timestamp,
    )

    object.__setattr__(
        invalid_bar,
        "symbol",
        valid_bar.symbol,
    )

    object.__setattr__(
        invalid_bar,
        "timeframe",
        valid_bar.timeframe,
    )

    object.__setattr__(
        invalid_bar,
        "open",
        float("nan"),
    )

    object.__setattr__(
        invalid_bar,
        "high",
        valid_bar.high,
    )

    object.__setattr__(
        invalid_bar,
        "low",
        valid_bar.low,
    )

    object.__setattr__(
        invalid_bar,
        "close",
        valid_bar.close,
    )

    object.__setattr__(
        invalid_bar,
        "volume",
        valid_bar.volume,
    )

    report = MarketDataQualityChecker().check(
        [invalid_bar]
    )

    assert report.valid is False

    assert any(
        issue.issue_type
        == QualityIssueType.NON_FINITE_VALUE
        for issue in report.errors
    )


def test_abnormal_range_is_warning():
    bars = [
        make_bar(
            0,
            high=2005.0,
            low=1998.0,
        ),
        make_bar(
            5,
            high=2005.0,
            low=1998.0,
        ),
        make_bar(
            10,
            high=2100.0,
            low=1900.0,
        ),
    ]

    checker = MarketDataQualityChecker(
        abnormal_range_multiplier=5.0
    )

    report = checker.check(bars)

    assert report.valid is True

    assert any(
        issue.issue_type
        == QualityIssueType.ABNORMAL_RANGE
        for issue in report.warnings
    )


def test_report_separates_errors_and_warnings():
    bars = [
        make_bar(0),
        make_bar(30),
    ]

    checker = MarketDataQualityChecker(
        minimum_history=5
    )

    report = checker.check(bars)

    assert report.valid is False
    assert len(report.errors) >= 1
    assert len(report.warnings) >= 1


def test_custom_gap_tolerance():
    bars = [
        make_bar(0),
        make_bar(10),
    ]

    checker = MarketDataQualityChecker(
        gap_tolerance=3.0
    )

    report = checker.check(bars)

    assert not any(
        issue.issue_type
        == QualityIssueType.GAP
        for issue in report.issues
    )


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        MarketDataQualityChecker(
            minimum_history=0
        )

    with pytest.raises(ValueError):
        MarketDataQualityChecker(
            gap_tolerance=1.0
        )

    with pytest.raises(ValueError):
        MarketDataQualityChecker(
            abnormal_range_multiplier=1.0
        )


def test_checker_does_not_modify_bars():
    bars = [
        make_bar(0),
        make_bar(5),
    ]

    original = list(bars)

    MarketDataQualityChecker().check(bars)

    assert bars == original


def test_multiple_series_can_be_checked_independently():
    bars = [
        make_bar(
            0,
            symbol="XAUUSD",
        ),
        make_bar(
            30,
            symbol="XAUUSD",
        ),
        make_bar(
            0,
            symbol="EURUSD",
        ),
        make_bar(
            5,
            symbol="EURUSD",
        ),
    ]

    report = MarketDataQualityChecker().check(bars)

    gaps = [
        issue
        for issue in report.warnings
        if issue.issue_type
        == QualityIssueType.GAP
    ]

    assert len(gaps) == 1