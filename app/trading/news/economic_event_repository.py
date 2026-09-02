from __future__ import annotations

from datetime import datetime
import sqlite3

from app.trading.news.economic_event import (
    EconomicEvent,
    EconomicEventError,
    EventImpact,
)


class EconomicEventRepositoryError(RuntimeError):
    """
    Raised when an economic-event repository operation fails.
    """


class EconomicEventRepository:
    """
    SQLite-backed persistence for economic-calendar events.
    """

    def __init__(self, database_path: str) -> None:
        if not isinstance(database_path, str) or not database_path.strip():
            raise EconomicEventRepositoryError(
                "database_path must be a non-empty string."
            )

        self.database_path = database_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS economic_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        name TEXT NOT NULL,
                        currency TEXT NOT NULL,
                        impact TEXT NOT NULL,
                        previous REAL,
                        forecast REAL,
                        actual REAL,
                        source TEXT NOT NULL,
                        UNIQUE(timestamp, name, currency, source)
                    )
                    """
                )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_economic_events_timestamp
                    ON economic_events(timestamp)
                    """
                )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_economic_events_currency_timestamp
                    ON economic_events(currency, timestamp)
                    """
                )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_economic_events_impact_timestamp
                    ON economic_events(impact, timestamp)
                    """
                )

        except sqlite3.Error as exc:
            raise EconomicEventRepositoryError(
                "failed to initialize economic-event database."
            ) from exc

    def save(self, event: EconomicEvent) -> bool:
        self._validate_event(event)

        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO economic_events (
                        timestamp,
                        name,
                        currency,
                        impact,
                        previous,
                        forecast,
                        actual,
                        source
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.timestamp.isoformat(),
                        event.name,
                        event.currency,
                        event.impact.value,
                        event.previous,
                        event.forecast,
                        event.actual,
                        event.source,
                    ),
                )

                return cursor.rowcount > 0

        except sqlite3.Error as exc:
            raise EconomicEventRepositoryError(
                "failed to save economic event."
            ) from exc

    def save_many(self, events: list[EconomicEvent]) -> int:
        if not isinstance(events, list):
            raise EconomicEventRepositoryError(
                "events must be a list."
            )

        for event in events:
            self._validate_event(event)

        if not events:
            return 0

        try:
            with self._connect() as connection:
                before = connection.total_changes

                connection.executemany(
                    """
                    INSERT OR IGNORE INTO economic_events (
                        timestamp,
                        name,
                        currency,
                        impact,
                        previous,
                        forecast,
                        actual,
                        source
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            event.timestamp.isoformat(),
                            event.name,
                            event.currency,
                            event.impact.value,
                            event.previous,
                            event.forecast,
                            event.actual,
                            event.source,
                        )
                        for event in events
                    ],
                )

                return connection.total_changes - before

        except sqlite3.Error as exc:
            raise EconomicEventRepositoryError(
                "failed to save economic events."
            ) from exc

    def update(
        self,
        event: EconomicEvent,
    ) -> bool:
        self._validate_event(event)

        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE economic_events
                    SET
                        impact = ?,
                        previous = ?,
                        forecast = ?,
                        actual = ?
                    WHERE
                        timestamp = ?
                        AND name = ?
                        AND currency = ?
                        AND source = ?
                    """,
                    (
                        event.impact.value,
                        event.previous,
                        event.forecast,
                        event.actual,
                        event.timestamp.isoformat(),
                        event.name,
                        event.currency,
                        event.source,
                    ),
                )

                return cursor.rowcount > 0

        except sqlite3.Error as exc:
            raise EconomicEventRepositoryError(
                "failed to update economic event."
            ) from exc

    def get(
        self,
        *,
        currency: str | None = None,
        impact: EventImpact | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[EconomicEvent]:
        self._validate_filters(
            currency=currency,
            impact=impact,
            start=start,
            end=end,
        )

        query = """
            SELECT
                timestamp,
                name,
                currency,
                impact,
                previous,
                forecast,
                actual,
                source
            FROM economic_events
            WHERE 1 = 1
        """

        parameters: list[object] = []

        if currency is not None:
            query += " AND currency = ?"
            parameters.append(currency)

        if impact is not None:
            query += " AND impact = ?"
            parameters.append(impact.value)

        if start is not None:
            query += " AND timestamp >= ?"
            parameters.append(start.isoformat())

        if end is not None:
            query += " AND timestamp <= ?"
            parameters.append(end.isoformat())

        query += """
            ORDER BY timestamp ASC, id ASC
        """

        try:
            with self._connect() as connection:
                rows = connection.execute(
                    query,
                    parameters,
                ).fetchall()

        except sqlite3.Error as exc:
            raise EconomicEventRepositoryError(
                "failed to retrieve economic events."
            ) from exc

        return [
            self._row_to_event(row)
            for row in rows
        ]

    def count(
        self,
        *,
        currency: str | None = None,
        impact: EventImpact | None = None,
    ) -> int:
        self._validate_filters(
            currency=currency,
            impact=impact,
        )

        query = """
            SELECT COUNT(*)
            FROM economic_events
            WHERE 1 = 1
        """

        parameters: list[object] = []

        if currency is not None:
            query += " AND currency = ?"
            parameters.append(currency)

        if impact is not None:
            query += " AND impact = ?"
            parameters.append(impact.value)

        try:
            with self._connect() as connection:
                row = connection.execute(
                    query,
                    parameters,
                ).fetchone()

        except sqlite3.Error as exc:
            raise EconomicEventRepositoryError(
                "failed to count economic events."
            ) from exc

        return int(row[0])

    def delete_all(self) -> int:
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "DELETE FROM economic_events"
                )

                return cursor.rowcount

        except sqlite3.Error as exc:
            raise EconomicEventRepositoryError(
                "failed to delete economic events."
            ) from exc

    @staticmethod
    def _validate_event(
        event: EconomicEvent,
    ) -> None:
        if not isinstance(event, EconomicEvent):
            raise EconomicEventRepositoryError(
                "event must be an EconomicEvent."
            )

    @staticmethod
    def _validate_filters(
        *,
        currency: str | None = None,
        impact: EventImpact | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> None:
        if currency is not None:
            if (
                not isinstance(currency, str)
                or not currency.strip()
            ):
                raise EconomicEventRepositoryError(
                    "currency must be a non-empty string or None."
                )

        if impact is not None:
            if not isinstance(
                impact,
                EventImpact,
            ):
                raise EconomicEventRepositoryError(
                    "impact must be an EventImpact or None."
                )

        if start is not None and not isinstance(
            start,
            datetime,
        ):
            raise EconomicEventRepositoryError(
                "start must be a datetime or None."
            )

        if end is not None and not isinstance(
            end,
            datetime,
        ):
            raise EconomicEventRepositoryError(
                "end must be a datetime or None."
            )

        if (
            start is not None
            and end is not None
            and start > end
        ):
            raise EconomicEventRepositoryError(
                "start must be earlier than or equal to end."
            )

    @staticmethod
    def _row_to_event(
        row: sqlite3.Row,
    ) -> EconomicEvent:
        try:
            timestamp = datetime.fromisoformat(
                row["timestamp"]
            )

            return EconomicEvent(
                timestamp=timestamp,
                name=row["name"],
                currency=row["currency"],
                impact=EventImpact(row["impact"]),
                previous=row["previous"],
                forecast=row["forecast"],
                actual=row["actual"],
                source=row["source"],
            )

        except (
            ValueError,
            TypeError,
            EconomicEventError,
        ) as exc:
            raise EconomicEventRepositoryError(
                "stored economic event is invalid."
            ) from exc