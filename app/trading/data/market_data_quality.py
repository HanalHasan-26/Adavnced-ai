from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import math

from .market_bar import MarketBar


class QualitySeverity(str, Enum):
    """Severity assigned to a market-data quality issue."""

    WARNING = "warning"
    ERROR = "error"


class QualityIssueType(str, Enum):
    """Types of market-data quality problems."""

    EMPTY_DATA = "empty_data"
    INVALID_BAR = "invalid_bar"
    DUPLICATE_TIMESTAMP = "duplicate_timestamp"
    OUT_OF_ORDER = "out_of_order"
    INVALID_TIMESTAMP = "invalid_timestamp"
    NON_FINITE_VALUE = "non_finite_value"
    INVALID_OHLC = "invalid_ohlc"
    NEGATIVE_VOLUME = "negative_volume"
    GAP = "gap"
    INSUFFICIENT_HISTORY = "insufficient_history"
    ABNORMAL_RANGE = "abnormal_range"
    MIXED_SYMBOL = "mixed_symbol"
    MIXED_TIMEFRAME = "mixed_timeframe"


@dataclass(frozen=True, slots=True)
class QualityIssue:
    """One detected market-data quality issue."""

    issue_type: QualityIssueType
    severity: QualitySeverity
    message: str
    index: int | None = None
    timestamp: datetime | None = None


@dataclass(frozen=True, slots=True)
class MarketDataQualityReport:
    """Result of validating a collection of market bars."""

    valid: bool
    bars_checked: int
    issues: tuple[QualityIssue, ...]

    @property
    def errors(self) -> tuple[QualityIssue, ...]:
        """Return only error-level issues."""

        return tuple(
            issue
            for issue in self.issues
            if issue.severity == QualitySeverity.ERROR
        )

    @property
    def warnings(self) -> tuple[QualityIssue, ...]:
        """Return only warning-level issues."""

        return tuple(
            issue
            for issue in self.issues
            if issue.severity == QualitySeverity.WARNING
        )


class MarketDataQualityChecker:
    """
    Validate market data before it reaches trading analysis.

    The checker never modifies the supplied market bars.
    """

    TIMEFRAME_MINUTES = {
        "M1": 1,
        "M2": 2,
        "M3": 3,
        "M4": 4,
        "M5": 5,
        "M6": 6,
        "M10": 10,
        "M12": 12,
        "M15": 15,
        "M20": 20,
        "M30": 30,
        "H1": 60,
        "H2": 120,
        "H3": 180,
        "H4": 240,
        "H6": 360,
        "H8": 480,
        "H12": 720,
        "D1": 1440,
        "W1": 10080,
        "MN1": 43200,
    }

    def __init__(
        self,
        *,
        minimum_history: int = 1,
        gap_tolerance: float = 1.5,
        abnormal_range_multiplier: float = 10.0,
    ) -> None:
        if minimum_history < 1:
            raise ValueError(
                "minimum_history must be greater than or equal to 1."
            )

        if not math.isfinite(gap_tolerance) or gap_tolerance <= 1.0:
            raise ValueError(
                "gap_tolerance must be greater than 1."
            )

        if (
            not math.isfinite(abnormal_range_multiplier)
            or abnormal_range_multiplier <= 1.0
        ):
            raise ValueError(
                "abnormal_range_multiplier must be greater than 1."
            )

        self.minimum_history = minimum_history
        self.gap_tolerance = gap_tolerance
        self.abnormal_range_multiplier = abnormal_range_multiplier

    def check(
        self,
        bars: list[MarketBar],
        *,
        expected_symbol: str | None = None,
        expected_timeframe: str | None = None,
    ) -> MarketDataQualityReport:
        """Validate a collection of market bars."""

        issues: list[QualityIssue] = []

        if not isinstance(bars, list):
            issues.append(
                QualityIssue(
                    issue_type=QualityIssueType.INVALID_BAR,
                    severity=QualitySeverity.ERROR,
                    message="Market data must be provided as a list.",
                )
            )

            return MarketDataQualityReport(
                valid=False,
                bars_checked=0,
                issues=tuple(issues),
            )

        if not bars:
            issues.append(
                QualityIssue(
                    issue_type=QualityIssueType.EMPTY_DATA,
                    severity=QualitySeverity.ERROR,
                    message="Market data contains no candles.",
                )
            )

            return MarketDataQualityReport(
                valid=False,
                bars_checked=0,
                issues=tuple(issues),
            )

        if len(bars) < self.minimum_history:
            issues.append(
                QualityIssue(
                    issue_type=QualityIssueType.INSUFFICIENT_HISTORY,
                    severity=QualitySeverity.ERROR,
                    message=(
                        f"Only {len(bars)} candle(s) available; "
                        f"at least {self.minimum_history} required."
                    ),
                )
            )

        symbols: set[str] = set()
        timeframes: set[str] = set()

        for index, bar in enumerate(bars):
            if not isinstance(bar, MarketBar):
                issues.append(
                    QualityIssue(
                        issue_type=QualityIssueType.INVALID_BAR,
                        severity=QualitySeverity.ERROR,
                        message=(
                            f"Item at index {index} is not a MarketBar."
                        ),
                        index=index,
                    )
                )
                continue

            symbols.add(bar.symbol)
            timeframes.add(bar.timeframe)

            self._check_bar_values(
                bar,
                index,
                issues,
            )

            if (
                expected_symbol is not None
                and bar.symbol != expected_symbol
            ):
                issues.append(
                    QualityIssue(
                        issue_type=QualityIssueType.MIXED_SYMBOL,
                        severity=QualitySeverity.ERROR,
                        message=(
                            f"Expected symbol '{expected_symbol}', "
                            f"but received '{bar.symbol}'."
                        ),
                        index=index,
                        timestamp=bar.timestamp,
                    )
                )

            if (
                expected_timeframe is not None
                and bar.timeframe != expected_timeframe
            ):
                issues.append(
                    QualityIssue(
                        issue_type=QualityIssueType.MIXED_TIMEFRAME,
                        severity=QualitySeverity.ERROR,
                        message=(
                            f"Expected timeframe "
                            f"'{expected_timeframe}', "
                            f"but received '{bar.timeframe}'."
                        ),
                        index=index,
                        timestamp=bar.timestamp,
                    )
                )

        if len(symbols) > 1:
            issues.append(
                QualityIssue(
                    issue_type=QualityIssueType.MIXED_SYMBOL,
                    severity=QualitySeverity.ERROR,
                    message=(
                        "Market data contains multiple symbols: "
                        + ", ".join(sorted(symbols))
                    ),
                )
            )

        if len(timeframes) > 1:
            issues.append(
                QualityIssue(
                    issue_type=QualityIssueType.MIXED_TIMEFRAME,
                    severity=QualitySeverity.ERROR,
                    message=(
                        "Market data contains multiple timeframes: "
                        + ", ".join(sorted(timeframes))
                    ),
                )
            )

        self._check_sequence(
            bars,
            issues,
        )

        self._check_gaps(
            bars,
            issues,
        )

        self._check_abnormal_ranges(
            bars,
            issues,
        )

        has_errors = any(
            issue.severity == QualitySeverity.ERROR
            for issue in issues
        )

        return MarketDataQualityReport(
            valid=not has_errors,
            bars_checked=len(bars),
            issues=tuple(issues),
        )

    @staticmethod
    def _check_bar_values(
        bar: MarketBar,
        index: int,
        issues: list[QualityIssue],
    ) -> None:
        values = {
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }

        for name, value in values.items():
            if not math.isfinite(float(value)):
                issues.append(
                    QualityIssue(
                        issue_type=QualityIssueType.NON_FINITE_VALUE,
                        severity=QualitySeverity.ERROR,
                        message=(
                            f"{name} contains a non-finite value."
                        ),
                        index=index,
                        timestamp=bar.timestamp,
                    )
                )

        if bar.open <= 0:
            issues.append(
                QualityIssue(
                    issue_type=QualityIssueType.INVALID_OHLC,
                    severity=QualitySeverity.ERROR,
                    message="Open price must be greater than zero.",
                    index=index,
                    timestamp=bar.timestamp,
                )
            )

        if bar.high <= 0:
            issues.append(
                QualityIssue(
                    issue_type=QualityIssueType.INVALID_OHLC,
                    severity=QualitySeverity.ERROR,
                    message="High price must be greater than zero.",
                    index=index,
                    timestamp=bar.timestamp,
                )
            )

        if bar.low <= 0:
            issues.append(
                QualityIssue(
                    issue_type=QualityIssueType.INVALID_OHLC,
                    severity=QualitySeverity.ERROR,
                    message="Low price must be greater than zero.",
                    index=index,
                    timestamp=bar.timestamp,
                )
            )

        if bar.close <= 0:
            issues.append(
                QualityIssue(
                    issue_type=QualityIssueType.INVALID_OHLC,
                    severity=QualitySeverity.ERROR,
                    message="Close price must be greater than zero.",
                    index=index,
                    timestamp=bar.timestamp,
                )
            )

        if bar.high < max(bar.open, bar.close):
            issues.append(
                QualityIssue(
                    issue_type=QualityIssueType.INVALID_OHLC,
                    severity=QualitySeverity.ERROR,
                    message="High price is below open or close.",
                    index=index,
                    timestamp=bar.timestamp,
                )
            )

        if bar.low > min(bar.open, bar.close):
            issues.append(
                QualityIssue(
                    issue_type=QualityIssueType.INVALID_OHLC,
                    severity=QualitySeverity.ERROR,
                    message="Low price is above open or close.",
                    index=index,
                    timestamp=bar.timestamp,
                )
            )

        if bar.high < bar.low:
            issues.append(
                QualityIssue(
                    issue_type=QualityIssueType.INVALID_OHLC,
                    severity=QualitySeverity.ERROR,
                    message="High price is below low price.",
                    index=index,
                    timestamp=bar.timestamp,
                )
            )

        if bar.volume < 0:
            issues.append(
                QualityIssue(
                    issue_type=QualityIssueType.NEGATIVE_VOLUME,
                    severity=QualitySeverity.ERROR,
                    message="Volume cannot be negative.",
                    index=index,
                    timestamp=bar.timestamp,
                )
            )

    @staticmethod
    def _check_sequence(
        bars: list[MarketBar],
        issues: list[QualityIssue],
    ) -> None:
        seen: set[tuple[str, str, datetime]] = set()

        previous: dict[
            tuple[str, str],
            datetime,
        ] = {}

        for index, bar in enumerate(bars):
            if not isinstance(bar, MarketBar):
                continue

            key = (
                bar.symbol,
                bar.timeframe,
                bar.timestamp,
            )

            if key in seen:
                issues.append(
                    QualityIssue(
                        issue_type=QualityIssueType.DUPLICATE_TIMESTAMP,
                        severity=QualitySeverity.ERROR,
                        message=(
                            "Duplicate candle timestamp for "
                            f"{bar.symbol} {bar.timeframe}."
                        ),
                        index=index,
                        timestamp=bar.timestamp,
                    )
                )
            else:
                seen.add(key)

            series_key = (
                bar.symbol,
                bar.timeframe,
            )

            prior = previous.get(series_key)

            if prior is not None and bar.timestamp <= prior:
                issues.append(
                    QualityIssue(
                        issue_type=QualityIssueType.OUT_OF_ORDER,
                        severity=QualitySeverity.ERROR,
                        message=(
                            "Candle timestamps are not strictly "
                            "in chronological order."
                        ),
                        index=index,
                        timestamp=bar.timestamp,
                    )
                )

            previous[series_key] = bar.timestamp

    def _check_gaps(
        self,
        bars: list[MarketBar],
        issues: list[QualityIssue],
    ) -> None:
        previous: dict[
            tuple[str, str],
            datetime,
        ] = {}

        for index, bar in enumerate(bars):
            if not isinstance(bar, MarketBar):
                continue

            series_key = (
                bar.symbol,
                bar.timeframe,
            )

            prior = previous.get(series_key)

            if prior is not None:
                expected_minutes = self.TIMEFRAME_MINUTES.get(
                    bar.timeframe.upper()
                )

                if expected_minutes is not None:
                    expected_delta = timedelta(
                        minutes=expected_minutes
                    )

                    actual_delta = (
                        bar.timestamp - prior
                    )

                    if actual_delta > (
                        expected_delta * self.gap_tolerance
                    ):
                        issues.append(
                            QualityIssue(
                                issue_type=QualityIssueType.GAP,
                                severity=QualitySeverity.WARNING,
                                message=(
                                    "Gap detected between candles: "
                                    f"{actual_delta} elapsed; "
                                    "expected approximately "
                                    f"{expected_delta}."
                                ),
                                index=index,
                                timestamp=bar.timestamp,
                            )
                        )

            previous[series_key] = bar.timestamp

    def _check_abnormal_ranges(
        self,
        bars: list[MarketBar],
        issues: list[QualityIssue],
    ) -> None:
        valid_ranges = [
            bar.range
            for bar in bars
            if isinstance(bar, MarketBar)
            and math.isfinite(bar.range)
            and bar.range > 0
        ]

        if len(valid_ranges) < 3:
            return

        sorted_ranges = sorted(valid_ranges)

        middle = len(sorted_ranges) // 2

        if len(sorted_ranges) % 2 == 0:
            baseline = (
                sorted_ranges[middle - 1]
                + sorted_ranges[middle]
            ) / 2
        else:
            baseline = sorted_ranges[middle]

        if baseline <= 0:
            return

        threshold = (
            baseline * self.abnormal_range_multiplier
        )

        for index, bar in enumerate(bars):
            if not isinstance(bar, MarketBar):
                continue

            if bar.range > threshold:
                issues.append(
                    QualityIssue(
                        issue_type=QualityIssueType.ABNORMAL_RANGE,
                        severity=QualitySeverity.WARNING,
                        message=(
                            f"Candle range {bar.range:g} is unusually "
                            f"large compared with the median range "
                            f"{baseline:g}."
                        ),
                        index=index,
                        timestamp=bar.timestamp,
                    )
                )