from datetime import datetime, timezone

import pytest

from app.trading.data.market_data_loader import (
    MarketDataLoadError,
    MarketDataLoader,
)


def write_csv(tmp_path, content: str):
    path = tmp_path / "market.csv"
    path.write_text(
        content,
        encoding="utf-8",
    )
    return path


def test_load_valid_csv(tmp_path):
    path = write_csv(
        tmp_path,
        "timestamp,symbol,timeframe,open,high,low,close,volume\n"
        "2026-01-01T10:00:00+00:00,"
        "XAUUSD,M5,2000,2005,1998,2003,120\n"
        "2026-01-01T10:05:00+00:00,"
        "XAUUSD,M5,2003,2008,2001,2007,150\n",
    )

    bars = MarketDataLoader().load_csv(path)

    assert len(bars) == 2
    assert bars[0].symbol == "XAUUSD"
    assert bars[0].close == 2003

    assert bars[1].timestamp == datetime(
        2026,
        1,
        1,
        10,
        5,
        tzinfo=timezone.utc,
    )


def test_volume_defaults_to_zero_when_missing(tmp_path):
    path = write_csv(
        tmp_path,
        "timestamp,symbol,timeframe,open,high,low,close\n"
        "2026-01-01T10:00:00,"
        "XAUUSD,M5,2000,2005,1998,2003\n",
    )

    bars = MarketDataLoader().load_csv(path)

    assert bars[0].volume == 0.0


def test_missing_file_raises(tmp_path):
    with pytest.raises(
        MarketDataLoadError,
        match="does not exist",
    ):
        MarketDataLoader().load_csv(
            tmp_path / "missing.csv"
        )


def test_directory_path_raises(tmp_path):
    with pytest.raises(
        MarketDataLoadError,
        match="not a file",
    ):
        MarketDataLoader().load_csv(tmp_path)


def test_empty_csv_raises(tmp_path):
    path = write_csv(
        tmp_path,
        "",
    )

    with pytest.raises(
        MarketDataLoadError,
        match="no header",
    ):
        MarketDataLoader().load_csv(path)


def test_missing_required_column_raises(tmp_path):
    path = write_csv(
        tmp_path,
        "timestamp,symbol,timeframe,open,high,low\n"
        "2026-01-01T10:00:00,"
        "XAUUSD,M5,2000,2005,1998\n",
    )

    with pytest.raises(
        MarketDataLoadError,
        match="close",
    ):
        MarketDataLoader().load_csv(path)


def test_missing_required_value_raises(tmp_path):
    path = write_csv(
        tmp_path,
        "timestamp,symbol,timeframe,open,high,low,close\n"
        "2026-01-01T10:00:00,"
        "XAUUSD,M5,,2005,1998,2003\n",
    )

    with pytest.raises(
        MarketDataLoadError,
        match="open.*required",
    ):
        MarketDataLoader().load_csv(path)


def test_invalid_timestamp_raises(tmp_path):
    path = write_csv(
        tmp_path,
        "timestamp,symbol,timeframe,open,high,low,close\n"
        "not-a-date,"
        "XAUUSD,M5,2000,2005,1998,2003\n",
    )

    with pytest.raises(
        MarketDataLoadError,
        match="invalid timestamp",
    ):
        MarketDataLoader().load_csv(path)


def test_invalid_numeric_value_raises(tmp_path):
    path = write_csv(
        tmp_path,
        "timestamp,symbol,timeframe,open,high,low,close\n"
        "2026-01-01T10:00:00,"
        "XAUUSD,M5,nope,2005,1998,2003\n",
    )

    with pytest.raises(
        MarketDataLoadError,
        match="open.*numeric",
    ):
        MarketDataLoader().load_csv(path)


def test_market_bar_validation_is_preserved(tmp_path):
    path = write_csv(
        tmp_path,
        "timestamp,symbol,timeframe,open,high,low,close\n"
        "2026-01-01T10:00:00,"
        "XAUUSD,M5,2000,1990,1998,2003\n",
    )

    with pytest.raises(
        MarketDataLoadError,
        match="invalid market bar",
    ):
        MarketDataLoader().load_csv(path)


def test_duplicate_candle_raises(tmp_path):
    path = write_csv(
        tmp_path,
        "timestamp,symbol,timeframe,open,high,low,close\n"
        "2026-01-01T10:00:00,"
        "XAUUSD,M5,2000,2005,1998,2003\n"
        "2026-01-01T10:00:00,"
        "XAUUSD,M5,2003,2008,2001,2007\n",
    )

    with pytest.raises(
        MarketDataLoadError,
        match="Duplicate",
    ):
        MarketDataLoader().load_csv(path)


def test_out_of_order_candle_raises(tmp_path):
    path = write_csv(
        tmp_path,
        "timestamp,symbol,timeframe,open,high,low,close\n"
        "2026-01-01T10:05:00,"
        "XAUUSD,M5,2003,2008,2001,2007\n"
        "2026-01-01T10:00:00,"
        "XAUUSD,M5,2000,2005,1998,2003\n",
    )

    with pytest.raises(
        MarketDataLoadError,
        match="Out-of-order",
    ):
        MarketDataLoader().load_csv(path)


def test_different_symbols_can_have_same_timestamp(tmp_path):
    path = write_csv(
        tmp_path,
        "timestamp,symbol,timeframe,open,high,low,close\n"
        "2026-01-01T10:00:00,"
        "XAUUSD,M5,2000,2005,1998,2003\n"
        "2026-01-01T10:00:00,"
        "EURUSD,M5,1.1,1.2,1.0,1.15\n",
    )

    bars = MarketDataLoader().load_csv(path)

    assert len(bars) == 2


def test_different_timeframes_can_have_same_timestamp(tmp_path):
    path = write_csv(
        tmp_path,
        "timestamp,symbol,timeframe,open,high,low,close\n"
        "2026-01-01T10:00:00,"
        "XAUUSD,M5,2000,2005,1998,2003\n"
        "2026-01-01T10:00:00,"
        "XAUUSD,H1,2000,2010,1990,2008\n",
    )

    bars = MarketDataLoader().load_csv(path)

    assert len(bars) == 2


def test_z_timestamp_is_supported(tmp_path):
    path = write_csv(
        tmp_path,
        "timestamp,symbol,timeframe,open,high,low,close\n"
        "2026-01-01T10:00:00Z,"
        "XAUUSD,M5,2000,2005,1998,2003\n",
    )

    bars = MarketDataLoader().load_csv(path)

    assert bars[0].timestamp.tzinfo is not None


def test_utf8_bom_is_supported(tmp_path):
    path = tmp_path / "market.csv"

    path.write_text(
        "timestamp,symbol,timeframe,open,high,low,close\n"
        "2026-01-01T10:00:00,"
        "XAUUSD,M5,2000,2005,1998,2003\n",
        encoding="utf-8-sig",
    )

    bars = MarketDataLoader().load_csv(path)

    assert len(bars) == 1


def test_whitespace_is_trimmed(tmp_path):
    path = write_csv(
        tmp_path,
        "timestamp,symbol,timeframe,open,high,low,close,volume\n"
        " 2026-01-01T10:00:00 ,"
        " XAUUSD , M5 , 2000 , 2005 , 1998 , 2003 , 10 \n",
    )

    bars = MarketDataLoader().load_csv(path)

    assert bars[0].symbol == "XAUUSD"
    assert bars[0].timeframe == "M5"
    assert bars[0].volume == 10


def test_multiple_series_are_validated_independently(tmp_path):
    path = write_csv(
        tmp_path,
        "timestamp,symbol,timeframe,open,high,low,close\n"
        "2026-01-01T10:05:00,"
        "XAUUSD,M5,2003,2008,2001,2007\n"
        "2026-01-01T10:00:00,"
        "EURUSD,M5,1.1,1.2,1.0,1.15\n"
        "2026-01-01T10:10:00,"
        "XAUUSD,M5,2000,2005,1998,2003\n",
    )

    bars = MarketDataLoader().load_csv(path)

    assert len(bars) == 3