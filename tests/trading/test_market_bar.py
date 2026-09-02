from datetime import datetime, timezone

import pytest

from app.trading.data.market_bar import MarketBar


def make_bar(**overrides):
    values = {
        "timestamp": datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        "symbol": "XAUUSD",
        "timeframe": "M15",
        "open": 4500.0,
        "high": 4510.0,
        "low": 4495.0,
        "close": 4505.0,
        "volume": 123.0,
    }
    values.update(overrides)
    return MarketBar(**values)


def test_valid_market_bar():
    bar = make_bar()

    assert bar.symbol == "XAUUSD"
    assert bar.timeframe == "M15"
    assert bar.open == 4500.0
    assert bar.high == 4510.0
    assert bar.low == 4495.0
    assert bar.close == 4505.0
    assert bar.volume == 123.0


def test_bullish_bar():
    assert make_bar().is_bullish is True
    assert make_bar().is_bearish is False
    assert make_bar().is_doji is False


def test_bearish_bar():
    bar = make_bar(open=4505.0, close=4500.0)

    assert bar.is_bullish is False
    assert bar.is_bearish is True
    assert bar.is_doji is False


def test_doji_bar():
    bar = make_bar(open=4500.0, close=4500.0)

    assert bar.is_bullish is False
    assert bar.is_bearish is False
    assert bar.is_doji is True


def test_range_and_body():
    bar = make_bar(open=4500.0, high=4512.0, low=4494.0, close=4506.0)

    assert bar.range == 18.0
    assert bar.body == 6.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("open", 0),
        ("high", 0),
        ("low", 0),
        ("close", 0),
        ("open", float("nan")),
        ("high", float("inf")),
    ],
)
def test_invalid_price_values(field, value):
    with pytest.raises(ValueError):
        make_bar(**{field: value})


def test_high_cannot_be_below_open_or_close():
    with pytest.raises(ValueError):
        make_bar(high=4499.0)


def test_low_cannot_be_above_open_or_close():
    with pytest.raises(ValueError):
        make_bar(low=4501.0)


def test_negative_volume_is_rejected():
    with pytest.raises(ValueError):
        make_bar(volume=-1.0)


def test_empty_symbol_is_rejected():
    with pytest.raises(ValueError):
        make_bar(symbol="   ")


def test_empty_timeframe_is_rejected():
    with pytest.raises(ValueError):
        make_bar(timeframe="")


def test_non_datetime_timestamp_is_rejected():
    with pytest.raises(ValueError):
        make_bar(timestamp="2026-01-01T12:00:00Z")


def test_market_bar_is_immutable():
    bar = make_bar()

    with pytest.raises(AttributeError):
        bar.close = 9999.0


def test_to_dict_contains_ohlcv_fields():
    bar = make_bar()
    data = bar.to_dict()

    assert data["symbol"] == "XAUUSD"
    assert data["timeframe"] == "M15"
    assert data["open"] == 4500.0
    assert data["high"] == 4510.0
    assert data["low"] == 4495.0
    assert data["close"] == 4505.0
    assert data["volume"] == 123.0
