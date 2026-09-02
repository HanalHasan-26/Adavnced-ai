from datetime import datetime, timedelta, timezone

import pytest

from app.trading.news.economic_event import (
    EconomicEvent,
    EventImpact,
)
from app.trading.news.economic_event_repository import (
    EconomicEventRepository,
    EconomicEventRepositoryError,
)


def make_event(
    *,
    timestamp=None,
    name="Nonfarm Payrolls",
    currency="USD",
    impact=EventImpact.HIGH,
    previous=180.0,
    forecast=170.0,
    actual=None,
    source="calendar",
):
    if timestamp is None:
        timestamp = datetime(
            2026,
            1,
            9,
            13,
            30,
            tzinfo=timezone.utc,
        )

    return EconomicEvent(
        timestamp=timestamp,
        name=name,
        currency=currency,
        impact=impact,
        previous=previous,
        forecast=forecast,
        actual=actual,
        source=source,
    )


@pytest.fixture
def repository(tmp_path):
    return EconomicEventRepository(
        str(tmp_path / "events.db")
    )


class TestInitialization:
    def test_creates_database(self, tmp_path):
        database = tmp_path / "events.db"

        repository = EconomicEventRepository(
            str(database)
        )

        assert database.exists()
        assert repository.count() == 0

    def test_reuses_existing_database(self, tmp_path):
        database = tmp_path / "events.db"

        first = EconomicEventRepository(
            str(database)
        )

        first.save(make_event())

        second = EconomicEventRepository(
            str(database)
        )

        assert second.count() == 1


class TestValidation:
    def test_empty_database_path_rejected(self):
        with pytest.raises(
            EconomicEventRepositoryError
        ):
            EconomicEventRepository("")

    def test_whitespace_database_path_rejected(self):
        with pytest.raises(
            EconomicEventRepositoryError
        ):
            EconomicEventRepository("   ")

    def test_non_string_database_path_rejected(self):
        with pytest.raises(
            EconomicEventRepositoryError
        ):
            EconomicEventRepository(None)

    def test_invalid_event_rejected(self, repository):
        with pytest.raises(
            EconomicEventRepositoryError
        ):
            repository.save("invalid")

    def test_save_many_requires_list(self, repository):
        with pytest.raises(
            EconomicEventRepositoryError
        ):
            repository.save_many(
                (make_event(),)
            )

    def test_invalid_event_in_save_many_rejected(
        self,
        repository,
    ):
        with pytest.raises(
            EconomicEventRepositoryError
        ):
            repository.save_many(
                [
                    make_event(),
                    "invalid",
                ]
            )


class TestSave:
    def test_save_event(self, repository):
        event = make_event()

        assert repository.save(event) is True
        assert repository.count() == 1

    def test_duplicate_save_is_ignored(
        self,
        repository,
    ):
        event = make_event()

        assert repository.save(event) is True
        assert repository.save(event) is False
        assert repository.count() == 1

    def test_different_source_is_distinct(
        self,
        repository,
    ):
        event = make_event(source="calendar")
        other = make_event(source="other")

        assert repository.save(event) is True
        assert repository.save(other) is True
        assert repository.count() == 2

    def test_different_timestamp_is_distinct(
        self,
        repository,
    ):
        event = make_event()
        other = make_event(
            timestamp=event.timestamp
            + timedelta(minutes=30)
        )

        assert repository.save(event) is True
        assert repository.save(other) is True
        assert repository.count() == 2


class TestSaveMany:
    def test_save_many(self, repository):
        events = [
            make_event(
                timestamp=datetime(
                    2026,
                    1,
                    9,
                    13,
                    30,
                    tzinfo=timezone.utc,
                )
            ),
            make_event(
                timestamp=datetime(
                    2026,
                    2,
                    6,
                    13,
                    30,
                    tzinfo=timezone.utc,
                ),
                name="CPI",
            ),
        ]

        assert repository.save_many(events) == 2
        assert repository.count() == 2

    def test_save_many_empty_list(
        self,
        repository,
    ):
        assert repository.save_many([]) == 0

    def test_save_many_ignores_duplicates(
        self,
        repository,
    ):
        event = make_event()

        assert repository.save_many(
            [event, event]
        ) == 1

        assert repository.count() == 1


class TestUpdate:
    def test_update_existing_event(
        self,
        repository,
    ):
        original = make_event(
            actual=None,
        )

        repository.save(original)

        updated = make_event(
            actual=175.0,
        )

        assert repository.update(updated) is True

        events = repository.get()

        assert len(events) == 1
        assert events[0].actual == 175.0

    def test_update_forecast(
        self,
        repository,
    ):
        original = make_event(
            forecast=170.0,
        )

        repository.save(original)

        updated = make_event(
            forecast=175.0,
        )

        assert repository.update(updated) is True

        event = repository.get()[0]

        assert event.forecast == 175.0

    def test_update_nonexistent_event(
        self,
        repository,
    ):
        assert repository.update(
            make_event()
        ) is False

    def test_update_does_not_change_identity(
        self,
        repository,
    ):
        event = make_event()

        repository.save(event)

        updated = make_event(
            actual=175.0,
            impact=EventImpact.MEDIUM,
        )

        repository.update(updated)

        result = repository.get()[0]

        assert result.timestamp == event.timestamp
        assert result.name == event.name
        assert result.currency == event.currency
        assert result.source == event.source


class TestGet:
    def test_get_all(self, repository):
        first = make_event()

        second = make_event(
            timestamp=first.timestamp
            + timedelta(hours=1),
            name="CPI",
        )

        repository.save_many(
            [second, first]
        )

        events = repository.get()

        assert len(events) == 2
        assert events[0].timestamp < events[1].timestamp

    def test_get_by_currency(self, repository):
        repository.save(
            make_event(currency="USD")
        )

        repository.save(
            make_event(
                timestamp=datetime(
                    2026,
                    1,
                    10,
                    13,
                    30,
                    tzinfo=timezone.utc,
                ),
                currency="EUR",
                name="ECB Rate Decision",
            )
        )

        events = repository.get(
            currency="USD"
        )

        assert len(events) == 1
        assert events[0].currency == "USD"

    def test_get_by_impact(self, repository):
        repository.save(
            make_event(
                impact=EventImpact.HIGH
            )
        )

        repository.save(
            make_event(
                timestamp=datetime(
                    2026,
                    1,
                    10,
                    13,
                    30,
                    tzinfo=timezone.utc,
                ),
                name="PMI",
                impact=EventImpact.MEDIUM,
            )
        )

        events = repository.get(
            impact=EventImpact.HIGH
        )

        assert len(events) == 1
        assert events[0].impact == EventImpact.HIGH

    def test_get_by_currency_and_impact(
        self,
        repository,
    ):
        repository.save(
            make_event(
                currency="USD",
                impact=EventImpact.HIGH,
            )
        )

        repository.save(
            make_event(
                timestamp=datetime(
                    2026,
                    1,
                    10,
                    13,
                    30,
                    tzinfo=timezone.utc,
                ),
                currency="USD",
                impact=EventImpact.MEDIUM,
                name="PMI",
            )
        )

        repository.save(
            make_event(
                timestamp=datetime(
                    2026,
                    1,
                    11,
                    13,
                    30,
                    tzinfo=timezone.utc,
                ),
                currency="EUR",
                impact=EventImpact.HIGH,
                name="ECB Rate Decision",
            )
        )

        events = repository.get(
            currency="USD",
            impact=EventImpact.HIGH,
        )

        assert len(events) == 1
        assert events[0].currency == "USD"
        assert events[0].impact == EventImpact.HIGH

    def test_get_start_inclusive(
        self,
        repository,
    ):
        first = make_event()

        repository.save(first)

        events = repository.get(
            start=first.timestamp
        )

        assert len(events) == 1

    def test_get_end_inclusive(
        self,
        repository,
    ):
        first = make_event()

        repository.save(first)

        events = repository.get(
            end=first.timestamp
        )

        assert len(events) == 1

    def test_get_time_range(
        self,
        repository,
    ):
        first = make_event()

        second = make_event(
            timestamp=first.timestamp
            + timedelta(hours=1),
            name="CPI",
        )

        third = make_event(
            timestamp=first.timestamp
            + timedelta(hours=2),
            name="PCE",
        )

        repository.save_many(
            [first, second, third]
        )

        events = repository.get(
            start=second.timestamp,
            end=third.timestamp,
        )

        assert len(events) == 2
        assert events[0].name == "CPI"
        assert events[1].name == "PCE"

    def test_get_empty_result(
        self,
        repository,
    ):
        assert repository.get(
            currency="USD"
        ) == []


class TestFilterValidation:
    def test_invalid_currency_rejected(
        self,
        repository,
    ):
        with pytest.raises(
            EconomicEventRepositoryError
        ):
            repository.get(
                currency=""
            )

    def test_whitespace_currency_rejected(
        self,
        repository,
    ):
        with pytest.raises(
            EconomicEventRepositoryError
        ):
            repository.get(
                currency="   "
            )

    def test_non_string_currency_rejected(
        self,
        repository,
    ):
        with pytest.raises(
            EconomicEventRepositoryError
        ):
            repository.get(
                currency=123
            )

    def test_invalid_impact_rejected(
        self,
        repository,
    ):
        with pytest.raises(
            EconomicEventRepositoryError
        ):
            repository.get(
                impact="HIGH"
            )

    def test_invalid_start_rejected(
        self,
        repository,
    ):
        with pytest.raises(
            EconomicEventRepositoryError
        ):
            repository.get(
                start="2026-01-01"
            )

    def test_invalid_end_rejected(
        self,
        repository,
    ):
        with pytest.raises(
            EconomicEventRepositoryError
        ):
            repository.get(
                end="2026-01-01"
            )

    def test_start_after_end_rejected(
        self,
        repository,
    ):
        start = datetime(
            2026,
            1,
            2,
            tzinfo=timezone.utc,
        )

        end = datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        )

        with pytest.raises(
            EconomicEventRepositoryError
        ):
            repository.get(
                start=start,
                end=end,
            )


class TestCount:
    def test_count_all(self, repository):
        repository.save(
            make_event()
        )

        repository.save(
            make_event(
                timestamp=datetime(
                    2026,
                    1,
                    10,
                    13,
                    30,
                    tzinfo=timezone.utc,
                ),
                name="CPI",
            )
        )

        assert repository.count() == 2

    def test_count_by_currency(
        self,
        repository,
    ):
        repository.save(
            make_event(currency="USD")
        )

        repository.save(
            make_event(
                timestamp=datetime(
                    2026,
                    1,
                    10,
                    13,
                    30,
                    tzinfo=timezone.utc,
                ),
                currency="EUR",
                name="ECB Rate Decision",
            )
        )

        assert repository.count(
            currency="USD"
        ) == 1

    def test_count_by_impact(
        self,
        repository,
    ):
        repository.save(
            make_event(
                impact=EventImpact.HIGH
            )
        )

        repository.save(
            make_event(
                timestamp=datetime(
                    2026,
                    1,
                    10,
                    13,
                    30,
                    tzinfo=timezone.utc,
                ),
                name="PMI",
                impact=EventImpact.MEDIUM,
            )
        )

        assert repository.count(
            impact=EventImpact.HIGH
        ) == 1


class TestDelete:
    def test_delete_all(
        self,
        repository,
    ):
        repository.save_many(
            [
                make_event(),
                make_event(
                    timestamp=datetime(
                        2026,
                        1,
                        10,
                        13,
                        30,
                        tzinfo=timezone.utc,
                    ),
                    name="CPI",
                ),
            ]
        )

        assert repository.delete_all() == 2
        assert repository.count() == 0

    def test_delete_empty_database(
        self,
        repository,
    ):
        assert repository.delete_all() == 0


class TestPersistence:
    def test_data_survives_new_repository(
        self,
        tmp_path,
    ):
        database = tmp_path / "persistent.db"

        first = EconomicEventRepository(
            str(database)
        )

        event = make_event(
            actual=175.0
        )

        first.save(event)

        second = EconomicEventRepository(
            str(database)
        )

        events = second.get()

        assert len(events) == 1
        assert events[0] == event