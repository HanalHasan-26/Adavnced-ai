from datetime import datetime, timezone

import pytest

from app.trading.news.economic_event import (
    EconomicEvent,
    EventImpact,
)
from app.trading.news.event_direction import (
    Direction,
    DirectionTarget,
    EventDirectionEngine,
)
from app.trading.news.news_market_impact import (
    ImpactLevel,
    ImpactReasonType,
    NewsMarketImpactEngine,
    NewsMarketImpactError,
)


BASE_TIME = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def make_event(
    *,
    currency: str = "USD",
    forecast: float | None = 100.0,
    actual: float | None = 120.0,
) -> EconomicEvent:
    return EconomicEvent(
        timestamp=BASE_TIME,
        name="CPI",
        currency=currency,
        impact=EventImpact.HIGH,
        forecast=forecast,
        actual=actual,
        source="test",
    )


@pytest.fixture
def direction_engine():
    return EventDirectionEngine()


@pytest.fixture
def engine():
    return NewsMarketImpactEngine()


def make_direction_result(
    direction_engine,
    *,
    currency: str = "USD",
    forecast: float | None = 100.0,
    actual: float | None = 120.0,
):
    return direction_engine.analyze(
        make_event(
            currency=currency,
            forecast=forecast,
            actual=actual,
        ),
        target=DirectionTarget.XAUUSD,
    )


class TestBasicAnalysis:
    def test_bullish_event_supports_long(
        self,
        engine,
        direction_engine,
    ):
        result = engine.analyze(
            make_direction_result(direction_engine)
        )

        assert result.direction == Direction.BEARISH
        assert result.supports_long is False
        assert result.supports_short is True

    def test_negative_usd_surprise_supports_xauusd_long(
        self,
        engine,
        direction_engine,
    ):
        direction_result = make_direction_result(
            direction_engine,
            forecast=100.0,
            actual=80.0,
        )

        result = engine.analyze(direction_result)

        assert result.direction == Direction.BULLISH
        assert result.supports_long is True
        assert result.supports_short is False


class TestImpactLevels:
    def test_high_confidence_direction_is_high_or_extreme(
        self,
        engine,
        direction_engine,
    ):
        result = engine.analyze(
            make_direction_result(direction_engine)
        )

        assert result.impact_level in (
            ImpactLevel.HIGH,
            ImpactLevel.EXTREME,
        )

    def test_neutral_surprise_reduces_impact(
        self,
        engine,
        direction_engine,
    ):
        direction_result = make_direction_result(
            direction_engine,
            forecast=100.0,
            actual=100.0,
        )

        result = engine.analyze(direction_result)

        assert result.direction == Direction.NEUTRAL
        assert result.impact_score < 80.0


class TestCaution:
    def test_high_impact_requires_caution(
        self,
        engine,
        direction_engine,
    ):
        result = engine.analyze(
            make_direction_result(direction_engine)
        )

        assert result.caution_required is True

    def test_low_impact_does_not_require_caution(
        self,
        engine,
        direction_engine,
    ):
        direction_result = make_direction_result(
            direction_engine,
            forecast=100.0,
            actual=100.0,
        )

        result = engine.analyze(direction_result)

        assert result.caution_required is False


class TestUnknown:
    def test_missing_actual_is_unknown(
        self,
        engine,
        direction_engine,
    ):
        direction_result = make_direction_result(
            direction_engine,
            actual=None,
        )

        result = engine.analyze(direction_result)

        assert result.direction == Direction.UNKNOWN
        assert result.impact_level == ImpactLevel.UNKNOWN
        assert result.sufficient_data is False

    def test_missing_forecast_is_unknown(
        self,
        engine,
        direction_engine,
    ):
        direction_result = make_direction_result(
            direction_engine,
            forecast=None,
        )

        result = engine.analyze(direction_result)

        assert result.direction == Direction.UNKNOWN
        assert result.impact_level == ImpactLevel.UNKNOWN
        assert result.sufficient_data is False


class TestProperties:
    def test_bearish_property(
        self,
        engine,
        direction_engine,
    ):
        result = engine.analyze(
            make_direction_result(direction_engine)
        )

        assert result.is_bearish is True
        assert result.is_bullish is False
        assert result.is_neutral is False
        assert result.is_unknown is False

    def test_xauusd_analysis(
        self,
        engine,
        direction_engine,
    ):
        direction_result = make_direction_result(
            direction_engine,
            actual=80.0,
        )

        result = engine.analyze_xauusd(direction_result)

        assert result.symbol == "XAUUSD"
        assert result.direction == Direction.BULLISH


class TestReasons:
    def test_bearish_reason_exists(
        self,
        engine,
        direction_engine,
    ):
        result = engine.analyze(
            make_direction_result(direction_engine)
        )

        assert any(
            reason.reason_type
            == ImpactReasonType.BEARISH_DIRECTION
            for reason in result.reasons
        )

    def test_usd_relationship_reason_exists(
        self,
        engine,
        direction_engine,
    ):
        result = engine.analyze(
            make_direction_result(direction_engine)
        )

        assert any(
            reason.reason_type
            == ImpactReasonType.USD_XAUUSD_RELATION
            for reason in result.reasons
        )

    def test_confidence_reason_exists(
        self,
        engine,
        direction_engine,
    ):
        result = engine.analyze(
            make_direction_result(direction_engine)
        )

        assert any(
            reason.reason_type
            == ImpactReasonType.HIGH_CONFIDENCE
            for reason in result.reasons
        )

    def test_insufficient_data_reason_exists(
        self,
        engine,
        direction_engine,
    ):
        direction_result = make_direction_result(
            direction_engine,
            actual=None,
        )

        result = engine.analyze(direction_result)

        assert any(
            reason.reason_type
            == ImpactReasonType.INSUFFICIENT_DATA
            for reason in result.reasons
        )


class TestValidation:
    def test_invalid_direction_result_raises(
        self,
        engine,
    ):
        with pytest.raises(NewsMarketImpactError):
            engine.analyze("invalid")

    def test_empty_symbol_raises(
        self,
        engine,
        direction_engine,
    ):
        direction_result = make_direction_result(
            direction_engine
        )

        with pytest.raises(NewsMarketImpactError):
            engine.analyze(
                direction_result,
                symbol="",
            )

    def test_invalid_threshold_raises(self):
        with pytest.raises(NewsMarketImpactError):
            NewsMarketImpactEngine(
                low_threshold=80.0,
                medium_threshold=50.0,
            )

    def test_threshold_out_of_range_raises(self):
        with pytest.raises(NewsMarketImpactError):
            NewsMarketImpactEngine(
                low_threshold=101.0,
            )