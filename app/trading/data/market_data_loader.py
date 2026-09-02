from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from .market_bar import MarketBar


class MarketDataLoadError(ValueError):
    """Raised when market data cannot be loaded or validated."""


class MarketDataLoader:
    """Load and validate OHLCV candles from CSV files."""

    REQUIRED_COLUMNS = (
        "timestamp",
        "symbol",
        "timeframe",
        "open",
        "high",
        "low",
        "close",
    )

    def load_csv(self, path: str | Path) -> list[MarketBar]:
        file_path = Path(path)

        if not file_path.exists():
            raise MarketDataLoadError(
                f"CSV file does not exist: {file_path}"
            )

        if not file_path.is_file():
            raise MarketDataLoadError(
                f"CSV path is not a file: {file_path}"
            )

        try:
            with file_path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as handle:
                reader = csv.DictReader(handle)

                if reader.fieldnames is None:
                    raise MarketDataLoadError(
                        "CSV file has no header row."
                    )

                columns = {
                    column.strip()
                    for column in reader.fieldnames
                    if column
                }

                missing = [
                    column
                    for column in self.REQUIRED_COLUMNS
                    if column not in columns
                ]

                if missing:
                    raise MarketDataLoadError(
                        "CSV is missing required columns: "
                        + ", ".join(missing)
                    )

                bars: list[MarketBar] = []

                for row_number, row in enumerate(reader, start=2):
                    bars.append(
                        self._parse_row(
                            row,
                            row_number,
                        )
                    )

        except UnicodeDecodeError as exc:
            raise MarketDataLoadError(
                "CSV file is not valid UTF-8 text."
            ) from exc

        except csv.Error as exc:
            raise MarketDataLoadError(
                f"Malformed CSV: {exc}"
            ) from exc

        except OSError as exc:
            raise MarketDataLoadError(
                f"Unable to read CSV file: {exc}"
            ) from exc

        self._validate_sequence(bars)

        return bars

    def _parse_row(
        self,
        row: dict[str, str | None],
        row_number: int,
    ) -> MarketBar:

        def required_text(name: str) -> str:
            value = row.get(name)

            if value is None or not value.strip():
                raise MarketDataLoadError(
                    f"Row {row_number}: '{name}' is required."
                )

            return value.strip()

        def number(
            name: str,
            default: float | None = None,
        ) -> float:
            value = row.get(name)

            if value is None or not value.strip():
                if default is not None:
                    return default

                raise MarketDataLoadError(
                    f"Row {row_number}: '{name}' is required."
                )

            try:
                return float(value.strip())

            except ValueError as exc:
                raise MarketDataLoadError(
                    f"Row {row_number}: '{name}' must be numeric."
                ) from exc

        timestamp_text = required_text("timestamp")

        try:
            timestamp = datetime.fromisoformat(
                timestamp_text.replace(
                    "Z",
                    "+00:00",
                )
            )

        except ValueError as exc:
            raise MarketDataLoadError(
                f"Row {row_number}: invalid timestamp "
                f"'{timestamp_text}'."
            ) from exc

        try:
            return MarketBar(
                timestamp=timestamp,
                symbol=required_text("symbol"),
                timeframe=required_text("timeframe"),
                open=number("open"),
                high=number("high"),
                low=number("low"),
                close=number("close"),
                volume=number(
                    "volume",
                    default=0.0,
                ),
            )

        except ValueError as exc:
            raise MarketDataLoadError(
                f"Row {row_number}: invalid market bar: {exc}"
            ) from exc

    @staticmethod
    def _validate_sequence(
        bars: list[MarketBar],
    ) -> None:
        seen: set[
            tuple[str, str, datetime]
        ] = set()

        previous: dict[
            tuple[str, str],
            datetime,
        ] = {}

        for index, bar in enumerate(
            bars,
            start=1,
        ):
            key = (
                bar.symbol,
                bar.timeframe,
                bar.timestamp,
            )

            if key in seen:
                raise MarketDataLoadError(
                    f"Duplicate candle at row {index + 1}: "
                    f"{bar.symbol} "
                    f"{bar.timeframe} "
                    f"{bar.timestamp.isoformat()}."
                )

            seen.add(key)

            series_key = (
                bar.symbol,
                bar.timeframe,
            )

            prior_timestamp = previous.get(
                series_key
            )

            if (
                prior_timestamp is not None
                and bar.timestamp <= prior_timestamp
            ):
                raise MarketDataLoadError(
                    f"Out-of-order candle at row {index + 1}: "
                    f"{bar.timestamp.isoformat()} must be after "
                    f"{prior_timestamp.isoformat()}."
                )

            previous[series_key] = bar.timestamp