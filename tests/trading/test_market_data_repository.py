from datetime import datetime, timezone

import pytest

from app.trading.data.market_data_repository import (
    MarketDataRepository,
    MarketDataRepositoryError,
)
from app.trading.data.market_bar import MarketBar


def make_bar(
    timestamp: datetime,
    symbol: str = "XAUUSD",
    timeframe: str = "M5",
    close: float = 2003.0,
) -> MarketBar:
    return MarketBar(
        timestamp=timestamp,
        symbol=symbol,
        timeframe=timeframe,
        open=2000.0,
        high=max(2005.0, close),
        low=1998.0,
        close=close,
        volume=100.0,
    )


def test_database_is_created(tmp_path):
    database = tmp_path / "market.db"

    repository = MarketDataRepository(database)

    assert database.exists()
    assert repository.count() == 0


def test_save_one_bar(tmp_path):
    repository = MarketDataRepository(
        tmp_path / "market.db"
    )

    bar = make_bar(
        datetime(
            2026,
            1,
            1,
            10,
            0,
        )
    )

    assert repository.save(bar) is True
    assert repository.count() == 1


def test_save_duplicate_returns_false(tmp_path):
    repository = MarketDataRepository(
        tmp_path / "market.db"
    )

    bar = make_bar(
        datetime(
            2026,
            1,
            1,
            10,
            0,
        )
    )

    assert repository.save(bar) is True
    assert repository.save(bar) is False
    assert repository.count() == 1


def test_same_timestamp_different_symbol_is_allowed(tmp_path):
    repository = MarketDataRepository(
        tmp_path / "market.db"
    )

    timestamp = datetime(
        2026,
        1,
        1,
        10,
        0,
    )

    assert repository.save(
        make_bar(timestamp, symbol="XAUUSD")
    )

    assert repository.save(
        make_bar(timestamp, symbol="EURUSD")
    )

    assert repository.count() == 2


def test_same_timestamp_different_timeframe_is_allowed(tmp_path):
    repository = MarketDataRepository(
        tmp_path / "market.db"
    )

    timestamp = datetime(
        2026,
        1,
        1,
        10,
        0,
    )

    assert repository.save(
        make_bar(timestamp, timeframe="M5")
    )

    assert repository.save(
        make_bar(timestamp, timeframe="H1")
    )

    assert repository.count() == 2


def test_save_many_inserts_multiple_bars(tmp_path):
    repository = MarketDataRepository(
        tmp_path / "market.db"
    )

    bars = [
        make_bar(
            datetime(
                2026,
                1,
                1,
                10,
                0,
            )
        ),
        make_bar(
            datetime(
                2026,
                1,
                1,
                10,
                5,
            )
        ),
        make_bar(
            datetime(
                2026,
                1,
                1,
                10,
                10,
            )
        ),
    ]

    assert repository.save_many(bars) == 3
    assert repository.count() == 3


def test_save_many_ignores_duplicates(tmp_path):
    repository = MarketDataRepository(
        tmp_path / "market.db"
    )

    timestamp = datetime(
        2026,
        1,
        1,
        10,
        0,
    )

    bars = [
        make_bar(timestamp),
        make_bar(timestamp),
        make_bar(
            datetime(
                2026,
                1,
                1,
                10,
                5,
            )
        ),
    ]

    assert repository.save_many(bars) == 2
    assert repository.count() == 2


def test_save_many_empty_list(tmp_path):
    repository = MarketDataRepository(
        tmp_path / "market.db"
    )

    assert repository.save_many([]) == 0
    assert repository.count() == 0


def test_get_returns_chronological_order(tmp_path):
    repository = MarketDataRepository(
        tmp_path / "market.db"
    )

    bars = [
        make_bar(
            datetime(
                2026,
                1,
                1,
                10,
                10,
            ),
            close=2010,
        ),
        make_bar(
            datetime(
                2026,
                1,
                1,
                10,
                0,
            ),
            close=2000,
        ),
        make_bar(
            datetime(
                2026,
                1,
                1,
                10,
                5,
            ),
            close=2005,
        ),
    ]

    repository.save_many(bars)

    result = repository.get(
        "XAUUSD",
        "M5",
    )

    assert [
        bar.timestamp
        for bar in result
    ] == [
        datetime(2026, 1, 1, 10, 0),
        datetime(2026, 1, 1, 10, 5),
        datetime(2026, 1, 1, 10, 10),
    ]


def test_get_filters_by_symbol(tmp_path):
    repository = MarketDataRepository(
        tmp_path / "market.db"
    )

    timestamp = datetime(
        2026,
        1,
        1,
        10,
        0,
    )

    repository.save(
        make_bar(
            timestamp,
            symbol="XAUUSD",
        )
    )

    repository.save(
        make_bar(
            timestamp,
            symbol="EURUSD",
        )
    )

    result = repository.get(
        "XAUUSD",
        "M5",
    )

    assert len(result) == 1
    assert result[0].symbol == "XAUUSD"


def test_get_filters_by_timeframe(tmp_path):
    repository = MarketDataRepository(
        tmp_path / "market.db"
    )

    timestamp = datetime(
        2026,
        1,
        1,
        10,
        0,
    )

    repository.save(
        make_bar(
            timestamp,
            timeframe="M5",
        )
    )

    repository.save(
        make_bar(
            timestamp,
            timeframe="H1",
        )
    )

    result = repository.get(
        "XAUUSD",
        "H1",
    )

    assert len(result) == 1
    assert result[0].timeframe == "H1"


def test_get_filters_by_time_range(tmp_path):
    repository = MarketDataRepository(
        tmp_path / "market.db"
    )

    repository.save_many(
        [
            make_bar(
                datetime(
                    2026,
                    1,
                    1,
                    10,
                    0,
                )
            ),
            make_bar(
                datetime(
                    2026,
                    1,
                    1,
                    10,
                    5,
                )
            ),
            make_bar(
                datetime(
                    2026,
                    1,
                    1,
                    10,
                    10,
                )
            ),
        ]
    )

    result = repository.get(
        "XAUUSD",
        "M5",
        start=datetime(
            2026,
            1,
            1,
            10,
            5,
        ),
        end=datetime(
            2026,
            1,
            1,
            10,
            10,
        ),
    )

    assert len(result) == 2


def test_timezone_aware_timestamp_survives_round_trip(
    tmp_path,
):
    repository = MarketDataRepository(
        tmp_path / "market.db"
    )

    timestamp = datetime(
        2026,
        1,
        1,
        10,
        0,
        tzinfo=timezone.utc,
    )

    repository.save(
        make_bar(timestamp)
    )

    result = repository.get(
        "XAUUSD",
        "M5",
    )

    assert result[0].timestamp == timestamp


def test_count_can_filter_by_symbol(tmp_path):
    repository = MarketDataRepository(
        tmp_path / "market.db"
    )

    timestamp = datetime(
        2026,
        1,
        1,
        10,
        0,
    )

    repository.save(
        make_bar(timestamp, symbol="XAUUSD")
    )

    repository.save(
        make_bar(
            timestamp,
            symbol="EURUSD",
        )
    )

    assert repository.count(
        symbol="XAUUSD"
    ) == 1


def test_count_can_filter_by_timeframe(tmp_path):
    repository = MarketDataRepository(
        tmp_path / "market.db"
    )

    timestamp = datetime(
        2026,
        1,
        1,
        10,
        0,
    )

    repository.save(
        make_bar(
            timestamp,
            timeframe="M5",
        )
    )

    repository.save(
        make_bar(
            timestamp,
            timeframe="H1",
        )
    )

    assert repository.count(
        timeframe="M5"
    ) == 1


def test_invalid_bar_type_raises(tmp_path):
    repository = MarketDataRepository(
        tmp_path / "market.db"
    )

    with pytest.raises(
        MarketDataRepositoryError,
        match="MarketBar",
    ):
        repository.save("not a bar")


def test_invalid_symbol_filter_raises(tmp_path):
    repository = MarketDataRepository(
        tmp_path / "market.db"
    )

    with pytest.raises(
        MarketDataRepositoryError,
        match="symbol",
    ):
        repository.get(
            "",
            "M5",
        )


def test_invalid_timeframe_filter_raises(tmp_path):
    repository = MarketDataRepository(
        tmp_path / "market.db"
    )

    with pytest.raises(
        MarketDataRepositoryError,
        match="timeframe",
    ):
        repository.get(
            "XAUUSD",
            "",
        )


def test_invalid_date_range_raises(tmp_path):
    repository = MarketDataRepository(
        tmp_path / "market.db"
    )

    with pytest.raises(
        MarketDataRepositoryError,
        match="before or equal",
    ):
        repository.get(
            "XAUUSD",
            "M5",
            start=datetime(
                2026,
                1,
                1,
                11,
            ),
            end=datetime(
                2026,
                1,
                1,
                10,
            ),
        )


def test_data_survives_repository_recreation(tmp_path):
    database = tmp_path / "market.db"

    repository = MarketDataRepository(database)

    repository.save(
        make_bar(
            datetime(
                2026,
                1,
                1,
                10,
                0,
            )
        )
    )

    del repository

    new_repository = MarketDataRepository(database)

    assert new_repository.count() == 1

    result = new_repository.get(
        "XAUUSD",
        "M5",
    )

    assert len(result) == 1


def test_delete_all_removes_data(tmp_path):
    repository = MarketDataRepository(
        tmp_path / "market.db"
    )

    repository.save_many(
        [
            make_bar(
                datetime(
                    2026,
                    1,
                    1,
                    10,
                    0,
                )
            ),
            make_bar(
                datetime(
                    2026,
                    1,
                    1,
                    10,
                    5,
                )
            ),
        ]
    )

    assert repository.delete_all() == 2
    assert repository.count() == 0