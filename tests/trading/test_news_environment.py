from datetime import datetime, timedelta, timezone

import pytest

from app.trading.news.event_direction import Direction
from app.trading.news.news_environment import (
    NewsEnvironmentDirection,
    NewsEnvironmentEngine,
    NewsEnvironmentError,
    NewsEnvironmentLevel,
    NewsEnvironmentReasonType,
)
from app.trading.news.news_market_impact import (
    ImpactLevel,
    NewsMarketImpactResult,
)


BASE_TIME = datetime(
    2026,
    1,
    1,
    12,
    0,
    tzinfo=timezone.utc,
)


def make_impact(
    *,
    timestamp: datetime = BASE_TIME,
    symbol: str = "XAUUSD",
    direction: Direction = Direction.BULLISH,
    impact_level: ImpactLevel = ImpactLevel.HIGH,
    impact_score: float = 80.0,
    confidence: float = 80.0,
    supports_long: bool = True,
    supports_short: bool = False,
    sufficient_data: bool = True,
) -> NewsMarketImpactResult:
    return NewsMarketImpactResult(
        timestamp=timestamp,
        event_name="CPI",
        symbol=symbol,
        direction=direction,
        impact_level=impact_level,
        impact_score=impact_score,
        confidence=confidence,
        supports_long=supports_long,
        supports_short=supports_short,
        caution_required=impact_score >= 60.0,
        reasons=(),
        sufficient_data=sufficient_data,
    )


@pytest.fixture
def engine():
    return NewsEnvironmentEngine()


class TestEmptyInput:
    def test_empty_input_returns_unknown(self, engine):
        result = engine.analyze([])

        assert result.direction == NewsEnvironmentDirection.UNKNOWN
        assert result.impact_level == NewsEnvironmentLevel.UNKNOWN
        assert result.event_count == 0
        assert result.valid_event_count == 0
        assert result.sufficient_data is False
        assert result.has_events is False

    def test_empty_input_has_no_events_reason(self, engine):
        result = engine.analyze([])

        assert any(
            reason.reason_type
            == NewsEnvironmentReasonType.NO_EVENTS
            for reason in result.reasons
        )


class TestBullishEnvironment:
    def test_bullish_events_create_bullish_environment(
        self,
        engine,
    ):
        impacts = [
            make_impact(
                direction=Direction.BULLISH,
                impact_level=ImpactLevel.HIGH,
            ),
            make_impact(
                timestamp=BASE_TIME + timedelta(minutes=5),
                direction=Direction.BULLISH,
                impact_level=ImpactLevel.HIGH,
            ),
        ]

        result = engine.analyze(impacts)

        assert result.direction == NewsEnvironmentDirection.BULLISH
        assert result.supports_long is True
        assert result.supports_short is False
        assert result.bullish_event_count == 2
        assert result.bearish_event_count == 0
        assert result.sufficient_data is True

    def test_bullish_environment_property(
        self,
        engine,
    ):
        result = engine.analyze(
            [
                make_impact(
                    direction=Direction.BULLISH,
                )
            ]
        )

        assert result.is_bullish is True
        assert result.is_bearish is False
        assert result.is_neutral is False
        assert result.is_unknown is False


class TestBearishEnvironment:
    def test_bearish_events_create_bearish_environment(
        self,
        engine,
    ):
        impacts = [
            make_impact(
                direction=Direction.BEARISH,
                supports_long=False,
                supports_short=True,
            ),
            make_impact(
                timestamp=BASE_TIME + timedelta(minutes=5),
                direction=Direction.BEARISH,
                supports_long=False,
                supports_short=True,
            ),
        ]

        result = engine.analyze(impacts)

        assert result.direction == NewsEnvironmentDirection.BEARISH
        assert result.supports_long is False
        assert result.supports_short is True
        assert result.bearish_event_count == 2

    def test_bearish_environment_property(
        self,
        engine,
    ):
        result = engine.analyze(
            [
                make_impact(
                    direction=Direction.BEARISH,
                    supports_long=False,
                    supports_short=True,
                )
            ]
        )

        assert result.is_bearish is True
        assert result.is_bullish is False
        assert result.is_neutral is False
        assert result.is_unknown is False


class TestNeutralEnvironment:
    def test_balanced_events_create_neutral_environment(
        self,
        engine,
    ):
        impacts = [
            make_impact(
                direction=Direction.BULLISH,
                impact_level=ImpactLevel.MEDIUM,
            ),
            make_impact(
                timestamp=BASE_TIME + timedelta(minutes=5),
                direction=Direction.BEARISH,
                impact_level=ImpactLevel.MEDIUM,
                supports_long=False,
                supports_short=True,
            ),
        ]

        result = engine.analyze(impacts)

        assert result.direction == NewsEnvironmentDirection.NEUTRAL

    def test_neutral_event_is_counted(
        self,
        engine,
    ):
        result = engine.analyze(
            [
                make_impact(
                    direction=Direction.NEUTRAL,
                    impact_level=ImpactLevel.MEDIUM,
                    supports_long=False,
                    supports_short=False,
                )
            ]
        )

        assert result.neutral_event_count == 1
        assert result.direction == NewsEnvironmentDirection.NEUTRAL


class TestDirectionalScore:
    def test_bullish_pressure_is_positive(
        self,
        engine,
    ):
        result = engine.analyze(
            [
                make_impact(
                    direction=Direction.BULLISH,
                )
            ]
        )

        assert result.net_directional_score > 0.0

    def test_bearish_pressure_is_negative(
        self,
        engine,
    ):
        result = engine.analyze(
            [
                make_impact(
                    direction=Direction.BEARISH,
                    supports_long=False,
                    supports_short=True,
                )
            ]
        )

        assert result.net_directional_score < 0.0

    def test_balanced_pressure_is_near_zero(
        self,
        engine,
    ):
        result = engine.analyze(
            [
                make_impact(
                    direction=Direction.BULLISH,
                    impact_level=ImpactLevel.MEDIUM,
                ),
                make_impact(
                    direction=Direction.BEARISH,
                    impact_level=ImpactLevel.MEDIUM,
                    supports_long=False,
                    supports_short=True,
                ),
            ]
        )

        assert abs(result.net_directional_score) < 1e-9


class TestImpactAggregation:
    def test_high_impact_events_produce_high_environment(
        self,
        engine,
    ):
        impacts = [
            make_impact(
                impact_level=ImpactLevel.HIGH,
                impact_score=80.0,
            ),
            make_impact(
                timestamp=BASE_TIME + timedelta(minutes=5),
                impact_level=ImpactLevel.HIGH,
                impact_score=80.0,
            ),
        ]

        result = engine.analyze(impacts)

        assert result.average_impact_score >= 60.0
        assert result.impact_level == NewsEnvironmentLevel.HIGH
        assert result.is_high_impact is True

    def test_extreme_impact_events_produce_extreme_environment(
        self,
        engine,
    ):
        impacts = [
            make_impact(
                impact_level=ImpactLevel.EXTREME,
                impact_score=100.0,
            ),
            make_impact(
                timestamp=BASE_TIME + timedelta(minutes=5),
                impact_level=ImpactLevel.EXTREME,
                impact_score=100.0,
            ),
        ]

        result = engine.analyze(impacts)

        assert result.impact_level == NewsEnvironmentLevel.EXTREME

    def test_low_impact_events_produce_low_environment(
        self,
        engine,
    ):
        impacts = [
            make_impact(
                impact_level=ImpactLevel.LOW,
                impact_score=25.0,
                confidence=80.0,
            )
        ]

        result = engine.analyze(impacts)

        assert result.impact_level == NewsEnvironmentLevel.LOW


class TestConflicts:
    def test_conflicting_events_are_detected(
        self,
        engine,
    ):
        impacts = [
            make_impact(
                direction=Direction.BULLISH,
            ),
            make_impact(
                timestamp=BASE_TIME + timedelta(minutes=5),
                direction=Direction.BEARISH,
                supports_long=False,
                supports_short=True,
            ),
        ]

        result = engine.analyze(impacts)

        assert result.conflicting_events is True

    def test_same_direction_events_are_not_conflicting(
        self,
        engine,
    ):
        impacts = [
            make_impact(
                direction=Direction.BULLISH,
            ),
            make_impact(
                timestamp=BASE_TIME + timedelta(minutes=5),
                direction=Direction.BULLISH,
            ),
        ]

        result = engine.analyze(impacts)

        assert result.conflicting_events is False

    def test_conflict_requires_caution(
        self,
        engine,
    ):
        impacts = [
            make_impact(
                direction=Direction.BULLISH,
            ),
            make_impact(
                timestamp=BASE_TIME + timedelta(minutes=5),
                direction=Direction.BEARISH,
                supports_long=False,
                supports_short=True,
            ),
        ]

        result = engine.analyze(impacts)

        assert result.caution_required is True


class TestUnknownEvents:
    def test_unknown_events_are_excluded_from_valid_count(
        self,
        engine,
    ):
        impacts = [
            make_impact(
                direction=Direction.UNKNOWN,
                impact_level=ImpactLevel.UNKNOWN,
                impact_score=0.0,
                confidence=0.0,
                supports_long=False,
                supports_short=False,
                sufficient_data=False,
            )
        ]

        result = engine.analyze(impacts)

        assert result.event_count == 1
        assert result.valid_event_count == 0
        assert result.unknown_event_count == 1
        assert result.sufficient_data is False
        assert result.direction == NewsEnvironmentDirection.UNKNOWN

    def test_valid_and_unknown_events_are_separated(
        self,
        engine,
    ):
        impacts = [
            make_impact(
                direction=Direction.BULLISH,
            ),
            make_impact(
                timestamp=BASE_TIME + timedelta(minutes=5),
                direction=Direction.UNKNOWN,
                impact_level=ImpactLevel.UNKNOWN,
                impact_score=0.0,
                confidence=0.0,
                supports_long=False,
                supports_short=False,
                sufficient_data=False,
            ),
        ]

        result = engine.analyze(impacts)

        assert result.event_count == 2
        assert result.valid_event_count == 1
        assert result.unknown_event_count == 1
        assert result.sufficient_data is True


class TestConfidence:
    def test_confidence_is_bounded(
        self,
        engine,
    ):
        impacts = [
            make_impact(
                confidence=100.0,
            )
            for _ in range(5)
        ]

        result = engine.analyze(impacts)

        assert 0.0 <= result.confidence <= 100.0

    def test_more_consistent_events_can_increase_confidence(
        self,
        engine,
    ):
        one = engine.analyze(
            [
                make_impact(
                    direction=Direction.BULLISH,
                )
            ]
        )

        multiple = engine.analyze(
            [
                make_impact(
                    direction=Direction.BULLISH,
                ),
                make_impact(
                    timestamp=BASE_TIME + timedelta(minutes=5),
                    direction=Direction.BULLISH,
                ),
                make_impact(
                    timestamp=BASE_TIME + timedelta(minutes=10),
                    direction=Direction.BULLISH,
                ),
            ]
        )

        assert multiple.confidence >= one.confidence


class TestCaution:
    def test_high_average_impact_requires_caution(
        self,
        engine,
    ):
        result = engine.analyze(
            [
                make_impact(
                    impact_score=90.0,
                    impact_level=ImpactLevel.EXTREME,
                )
            ]
        )

        assert result.caution_required is True

    def test_low_confidence_requires_caution(
        self,
        engine,
    ):
        result = engine.analyze(
            [
                make_impact(
                    confidence=20.0,
                    impact_score=25.0,
                    impact_level=ImpactLevel.LOW,
                )
            ]
        )

        assert result.caution_required is True


class TestEvents:
    def test_relevant_events_are_preserved(
        self,
        engine,
    ):
        first = make_impact(
            timestamp=BASE_TIME,
        )

        second = make_impact(
            timestamp=BASE_TIME + timedelta(minutes=10),
        )

        result = engine.analyze(
            [second, first]
        )

        assert len(result.relevant_events) == 2
        assert (
            result.relevant_events[0].timestamp
            < result.relevant_events[1].timestamp
        )

    def test_latest_timestamp_is_used(
        self,
        engine,
    ):
        latest = BASE_TIME + timedelta(minutes=30)

        result = engine.analyze(
            [
                make_impact(timestamp=BASE_TIME),
                make_impact(timestamp=latest),
            ]
        )

        assert result.timestamp == latest


class TestXAUUSD:
    def test_analyze_xauusd_sets_symbol(
        self,
        engine,
    ):
        result = engine.analyze_xauusd(
            [
                make_impact(),
            ]
        )

        assert result.symbol == "XAUUSD"


class TestReasons:
    def test_bullish_reason_exists(
        self,
        engine,
    ):
        result = engine.analyze(
            [
                make_impact(
                    direction=Direction.BULLISH,
                )
            ]
        )

        assert any(
            reason.reason_type
            == NewsEnvironmentReasonType.BULLISH_ENVIRONMENT
            for reason in result.reasons
        )

    def test_bearish_reason_exists(
        self,
        engine,
    ):
        result = engine.analyze(
            [
                make_impact(
                    direction=Direction.BEARISH,
                    supports_long=False,
                    supports_short=True,
                )
            ]
        )

        assert any(
            reason.reason_type
            == NewsEnvironmentReasonType.BEARISH_ENVIRONMENT
            for reason in result.reasons
        )

    def test_conflict_reason_exists(
        self,
        engine,
    ):
        result = engine.analyze(
            [
                make_impact(
                    direction=Direction.BULLISH,
                ),
                make_impact(
                    timestamp=BASE_TIME + timedelta(minutes=5),
                    direction=Direction.BEARISH,
                    supports_long=False,
                    supports_short=True,
                ),
            ]
        )

        assert any(
            reason.reason_type
            == NewsEnvironmentReasonType.CONFLICTING_EVENTS
            for reason in result.reasons
        )

    def test_caution_reason_exists(
        self,
        engine,
    ):
        result = engine.analyze(
            [
                make_impact(
                    impact_level=ImpactLevel.EXTREME,
                    impact_score=100.0,
                )
            ]
        )

        assert any(
            reason.reason_type
            == NewsEnvironmentReasonType.CAUTION_REQUIRED
            for reason in result.reasons
        )


class TestValidation:
    def test_invalid_impacts_type_raises(
        self,
        engine,
    ):
        with pytest.raises(NewsEnvironmentError):
            engine.analyze("invalid")

    def test_invalid_impact_object_raises(
        self,
        engine,
    ):
        with pytest.raises(NewsEnvironmentError):
            engine.analyze(
                ["invalid"]
            )

    def test_empty_symbol_raises(
        self,
        engine,
    ):
        with pytest.raises(NewsEnvironmentError):
            engine.analyze(
                [
                    make_impact(),
                ],
                symbol="",
            )

    def test_mixed_symbols_raise(
        self,
        engine,
    ):
        impacts = [
            make_impact(symbol="XAUUSD"),
            make_impact(
                symbol="EURUSD",
                timestamp=BASE_TIME + timedelta(minutes=5),
            ),
        ]

        with pytest.raises(NewsEnvironmentError):
            engine.analyze(impacts)

    def test_invalid_minimum_events_raises(self):
        with pytest.raises(NewsEnvironmentError):
            NewsEnvironmentEngine(
                minimum_valid_events=0,
            )

    def test_invalid_conflict_ratio_raises(self):
        with pytest.raises(NewsEnvironmentError):
            NewsEnvironmentEngine(
                conflict_ratio_threshold=1.5,
            )

    def test_invalid_impact_thresholds_raise(self):
        with pytest.raises(NewsEnvironmentError):
            NewsEnvironmentEngine(
                high_impact_threshold=90.0,
                extreme_impact_threshold=80.0,
            )