from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from .market_bar import MarketBar


class MarketDataRepositoryError(RuntimeError):
    """Raised when historical market data storage fails."""


class MarketDataRepository:
    """Persistent SQLite storage for historical market candles."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

        if self.database_path.exists() and self.database_path.is_dir():
            raise MarketDataRepositoryError(
                f"Database path is a directory: {self.database_path}"
            )

        try:
            self.database_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._initialize_database()

        except OSError as exc:
            raise MarketDataRepositoryError(
                f"Unable to prepare database directory: {exc}"
            ) from exc

        except sqlite3.Error as exc:
            raise MarketDataRepositoryError(
                f"Unable to initialize database: {exc}"
            ) from exc

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)

        connection.row_factory = sqlite3.Row

        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS market_bars (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    UNIQUE(symbol, timeframe, timestamp)
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_market_bars_lookup
                ON market_bars(symbol, timeframe, timestamp)
                """
            )

            connection.commit()

    def save(self, bar: MarketBar) -> bool:
        """Save one candle.

        Returns True when inserted and False when the candle
        already exists.
        """

        if not isinstance(bar, MarketBar):
            raise MarketDataRepositoryError(
                "bar must be a MarketBar instance."
            )

        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO market_bars (
                        timestamp,
                        symbol,
                        timeframe,
                        open,
                        high,
                        low,
                        close,
                        volume
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    self._values(bar),
                )

                connection.commit()

                return cursor.rowcount == 1

        except sqlite3.Error as exc:
            raise MarketDataRepositoryError(
                f"Unable to save market bar: {exc}"
            ) from exc

    def save_many(self, bars: list[MarketBar]) -> int:
        """Save multiple candles.

        Returns the number of newly inserted candles.
        """

        if not isinstance(bars, list):
            raise MarketDataRepositoryError(
                "bars must be a list."
            )

        for bar in bars:
            if not isinstance(bar, MarketBar):
                raise MarketDataRepositoryError(
                    "Every item in bars must be a MarketBar instance."
                )

        if not bars:
            return 0

        try:
            with self._connect() as connection:
                cursor = connection.executemany(
                    """
                    INSERT OR IGNORE INTO market_bars (
                        timestamp,
                        symbol,
                        timeframe,
                        open,
                        high,
                        low,
                        close,
                        volume
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [self._values(bar) for bar in bars],
                )

                connection.commit()

                return cursor.rowcount

        except sqlite3.Error as exc:
            raise MarketDataRepositoryError(
                f"Unable to save market bars: {exc}"
            ) from exc

    def get(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[MarketBar]:
        """Retrieve candles in chronological order."""

        self._validate_filter_text(
            symbol,
            "symbol",
        )

        self._validate_filter_text(
            timeframe,
            "timeframe",
        )

        if start is not None and not isinstance(start, datetime):
            raise MarketDataRepositoryError(
                "start must be a datetime or None."
            )

        if end is not None and not isinstance(end, datetime):
            raise MarketDataRepositoryError(
                "end must be a datetime or None."
            )

        if (
            start is not None
            and end is not None
            and start > end
        ):
            raise MarketDataRepositoryError(
                "start must be before or equal to end."
            )

        query = """
            SELECT
                timestamp,
                symbol,
                timeframe,
                open,
                high,
                low,
                close,
                volume
            FROM market_bars
            WHERE symbol = ?
              AND timeframe = ?
        """

        parameters: list[object] = [
            symbol.strip(),
            timeframe.strip(),
        ]

        if start is not None:
            query += " AND timestamp >= ?"
            parameters.append(self._timestamp(start))

        if end is not None:
            query += " AND timestamp <= ?"
            parameters.append(self._timestamp(end))

        query += " ORDER BY timestamp ASC"

        try:
            with self._connect() as connection:
                rows = connection.execute(
                    query,
                    parameters,
                ).fetchall()

        except sqlite3.Error as exc:
            raise MarketDataRepositoryError(
                f"Unable to retrieve market bars: {exc}"
            ) from exc

        return [
            self._row_to_bar(row)
            for row in rows
        ]

    def count(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> int:
        """Return the number of stored candles."""

        if symbol is not None:
            self._validate_filter_text(
                symbol,
                "symbol",
            )

        if timeframe is not None:
            self._validate_filter_text(
                timeframe,
                "timeframe",
            )

        query = "SELECT COUNT(*) FROM market_bars"

        conditions: list[str] = []
        parameters: list[str] = []

        if symbol is not None:
            conditions.append("symbol = ?")
            parameters.append(symbol.strip())

        if timeframe is not None:
            conditions.append("timeframe = ?")
            parameters.append(timeframe.strip())

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        try:
            with self._connect() as connection:
                row = connection.execute(
                    query,
                    parameters,
                ).fetchone()

        except sqlite3.Error as exc:
            raise MarketDataRepositoryError(
                f"Unable to count market bars: {exc}"
            ) from exc

        return int(row[0])

    def delete_all(self) -> int:
        """Delete all stored candles.

        Primarily useful for tests or deliberate data resets.
        """

        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "DELETE FROM market_bars"
                )

                connection.commit()

                return cursor.rowcount

        except sqlite3.Error as exc:
            raise MarketDataRepositoryError(
                f"Unable to delete market bars: {exc}"
            ) from exc

    @staticmethod
    def _values(
        bar: MarketBar,
    ) -> tuple[object, ...]:
        return (
            MarketDataRepository._timestamp(
                bar.timestamp
            ),
            bar.symbol.strip(),
            bar.timeframe.strip(),
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.volume,
        )

    @staticmethod
    def _timestamp(timestamp: datetime) -> str:
        return timestamp.isoformat()

    @staticmethod
    def _row_to_bar(
        row: sqlite3.Row,
    ) -> MarketBar:
        try:
            timestamp = datetime.fromisoformat(
                row["timestamp"]
            )

            return MarketBar(
                timestamp=timestamp,
                symbol=row["symbol"],
                timeframe=row["timeframe"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )

        except (TypeError, ValueError) as exc:
            raise MarketDataRepositoryError(
                f"Stored market data is invalid: {exc}"
            ) from exc

    @staticmethod
    def _validate_filter_text(
        value: str,
        name: str,
    ) -> None:
        if not isinstance(value, str):
            raise MarketDataRepositoryError(
                f"{name} must be a string."
            )

        if not value.strip():
            raise MarketDataRepositoryError(
                f"{name} must not be empty."
            )