from datetime import datetime, timezone

import pytest

from app.trading.entry.entry_model import EntryDirection, EntryModel
from app.trading.risk.risk_reward_intelligence import (
    RiskRewardDecision,
    RiskRewardIntelligenceEngine,
    RiskRewardIntelligenceError,
    RiskRewardReasonType,
)


def make_entry(
    direction: EntryDirection,
    *,
    symbol: str = "XAUUSD",
    timeframe: str = "M15",
    entry_price: float = 2000.0,
) -> EntryModel:
    """Create a minimal valid EntryModel."""

    return EntryModel(
        timestamp=datetime.now(timezone.utc),
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        entry_price=entry_price,
    )


def test_long_1_to_2_rr_is_allowed() -> None:
    """Exactly 1:2 RR should be allowed."""

    engine = RiskRewardIntelligenceEngine()

    entry = make_entry(
        EntryDirection.LONG,
        entry_price=2000.0,
    )

    result = engine.analyze(
        entry,
        stop_loss=1990.0,
        take_profit=2020.0,
    )

    assert result.decision is RiskRewardDecision.ALLOW
    assert result.allowed is True
    assert result.valid is True
    assert result.ready is True
    assert result.risk_distance == 10.0
    assert result.reward_distance == 20.0
    assert result.risk_reward_ratio == 2.0


def test_short_1_to_2_rr_is_allowed() -> None:
    """Exactly 1:2 RR should work for SHORT trades."""

    engine = RiskRewardIntelligenceEngine()

    entry = make_entry(
        EntryDirection.SHORT,
        entry_price=2000.0,
    )

    result = engine.analyze(
        entry,
        stop_loss=2010.0,
        take_profit=1980.0,
    )

    assert result.allowed is True
    assert result.risk_distance == 10.0
    assert result.reward_distance == 20.0
    assert result.risk_reward_ratio == 2.0


def test_rr_above_minimum_is_allowed() -> None:
    """RR above minimum but below maximum is allowed."""

    engine = RiskRewardIntelligenceEngine()

    entry = make_entry(EntryDirection.LONG)

    result = engine.analyze(
        entry,
        stop_loss=1990.0,
        take_profit=2040.0,
    )

    assert result.allowed is True
    assert result.rr == 4.0


def test_rr_below_minimum_is_blocked() -> None:
    """RR below 1:2 must be blocked."""

    engine = RiskRewardIntelligenceEngine()

    entry = make_entry(EntryDirection.LONG)

    result = engine.analyze(
        entry,
        stop_loss=1990.0,
        take_profit=2015.0,
    )

    assert result.blocked is True
    assert result.valid is False
    assert result.ready is False
    assert (
        result.reasons[0].reason_type
        is RiskRewardReasonType.RISK_REWARD_TOO_LOW
    )


def test_rr_above_maximum_is_blocked() -> None:
    """RR above maximum must be blocked."""

    engine = RiskRewardIntelligenceEngine(
        maximum_risk_reward=5.0,
    )

    entry = make_entry(EntryDirection.LONG)

    result = engine.analyze(
        entry,
        stop_loss=1990.0,
        take_profit=2060.0,
    )

    assert result.blocked is True
    assert (
        result.reasons[0].reason_type
        is RiskRewardReasonType.RISK_REWARD_TOO_HIGH
    )


def test_exact_maximum_rr_is_allowed() -> None:
    """RR exactly equal to maximum should be allowed."""

    engine = RiskRewardIntelligenceEngine(
        maximum_risk_reward=5.0,
    )

    entry = make_entry(EntryDirection.LONG)

    result = engine.analyze(
        entry,
        stop_loss=1990.0,
        take_profit=2050.0,
    )

    assert result.allowed is True
    assert result.rr == 5.0


def test_missing_stop_loss_is_blocked() -> None:
    """Missing SL must block the trade."""

    engine = RiskRewardIntelligenceEngine()

    entry = make_entry(EntryDirection.LONG)

    result = engine.analyze(
        entry,
        take_profit=2020.0,
    )

    assert result.blocked is True
    assert (
        result.reasons[0].reason_type
        is RiskRewardReasonType.INVALID_STOP_LOSS
    )


def test_missing_take_profit_is_blocked() -> None:
    """Missing TP must block the trade."""

    engine = RiskRewardIntelligenceEngine()

    entry = make_entry(EntryDirection.LONG)

    result = engine.analyze(
        entry,
        stop_loss=1990.0,
    )

    assert result.blocked is True
    assert (
        result.reasons[0].reason_type
        is RiskRewardReasonType.INVALID_TAKE_PROFIT
    )


def test_long_wrong_side_stop_is_blocked() -> None:
    """LONG SL must be below entry."""

    engine = RiskRewardIntelligenceEngine()

    entry = make_entry(EntryDirection.LONG)

    result = engine.analyze(
        entry,
        stop_loss=2005.0,
        take_profit=2020.0,
    )

    assert result.blocked is True


def test_short_wrong_side_stop_is_blocked() -> None:
    """SHORT SL must be above entry."""

    engine = RiskRewardIntelligenceEngine()

    entry = make_entry(EntryDirection.SHORT)

    result = engine.analyze(
        entry,
        stop_loss=1995.0,
        take_profit=1980.0,
    )

    assert result.blocked is True


def test_long_wrong_side_take_profit_is_blocked() -> None:
    """LONG TP must be above entry."""

    engine = RiskRewardIntelligenceEngine()

    entry = make_entry(EntryDirection.LONG)

    result = engine.analyze(
        entry,
        stop_loss=1990.0,
        take_profit=1995.0,
    )

    assert result.blocked is True


def test_short_wrong_side_take_profit_is_blocked() -> None:
    """SHORT TP must be below entry."""

    engine = RiskRewardIntelligenceEngine()

    entry = make_entry(EntryDirection.SHORT)

    result = engine.analyze(
        entry,
        stop_loss=2010.0,
        take_profit=2005.0,
    )

    assert result.blocked is True


def test_zero_stop_loss_is_rejected() -> None:
    """Zero SL is invalid."""

    engine = RiskRewardIntelligenceEngine()

    entry = make_entry(EntryDirection.LONG)

    with pytest.raises(RiskRewardIntelligenceError):
        engine.analyze(
            entry,
            stop_loss=0.0,
            take_profit=2020.0,
        )


def test_negative_stop_loss_is_rejected() -> None:
    """Negative SL is invalid."""

    engine = RiskRewardIntelligenceEngine()

    entry = make_entry(EntryDirection.LONG)

    with pytest.raises(RiskRewardIntelligenceError):
        engine.analyze(
            entry,
            stop_loss=-1.0,
            take_profit=2020.0,
        )


def test_zero_take_profit_is_rejected() -> None:
    """Zero TP is invalid."""

    engine = RiskRewardIntelligenceEngine()

    entry = make_entry(EntryDirection.LONG)

    with pytest.raises(RiskRewardIntelligenceError):
        engine.analyze(
            entry,
            stop_loss=1990.0,
            take_profit=0.0,
        )


def test_negative_take_profit_is_rejected() -> None:
    """Negative TP is invalid."""

    engine = RiskRewardIntelligenceEngine()

    entry = make_entry(EntryDirection.LONG)

    with pytest.raises(RiskRewardIntelligenceError):
        engine.analyze(
            entry,
            stop_loss=1990.0,
            take_profit=-1.0,
        )


def test_invalid_direction_is_blocked() -> None:
    """NONE direction cannot pass RR validation."""

    engine = RiskRewardIntelligenceEngine()

    entry = make_entry(EntryDirection.NONE)

    result = engine.analyze(
        entry,
        stop_loss=1990.0,
        take_profit=2020.0,
    )

    assert result.blocked is True
    assert (
        result.reasons[0].reason_type
        is RiskRewardReasonType.INVALID_DIRECTION
    )


def test_custom_minimum_rr() -> None:
    """A custom minimum RR can be supplied."""

    engine = RiskRewardIntelligenceEngine()

    entry = make_entry(EntryDirection.LONG)

    result = engine.analyze(
        entry,
        stop_loss=1990.0,
        take_profit=2030.0,
        minimum_risk_reward=3.0,
    )

    assert result.allowed is True
    assert result.minimum_risk_reward == 3.0
    assert result.rr == 3.0


def test_custom_minimum_rr_can_block() -> None:
    """A stricter minimum RR can block a trade."""

    engine = RiskRewardIntelligenceEngine()

    entry = make_entry(EntryDirection.LONG)

    result = engine.analyze(
        entry,
        stop_loss=1990.0,
        take_profit=2020.0,
        minimum_risk_reward=3.0,
    )

    assert result.blocked is True


def test_invalid_policy_is_rejected() -> None:
    """Minimum RR cannot exceed maximum RR."""

    with pytest.raises(RiskRewardIntelligenceError):
        RiskRewardIntelligenceEngine(
            minimum_risk_reward=5.0,
            maximum_risk_reward=4.0,
        )


def test_custom_maximum_rr() -> None:
    """A custom maximum RR can be supplied."""

    engine = RiskRewardIntelligenceEngine()

    entry = make_entry(EntryDirection.LONG)

    result = engine.analyze(
        entry,
        stop_loss=1990.0,
        take_profit=2050.0,
        maximum_risk_reward=5.0,
    )

    assert result.allowed is True
    assert result.maximum_risk_reward == 5.0
    assert result.rr == 5.0


def test_xauusd_wrapper_accepts_xauusd() -> None:
    """XAUUSD wrapper accepts XAUUSD."""

    engine = RiskRewardIntelligenceEngine()

    entry = make_entry(
        EntryDirection.LONG,
        symbol="XAUUSD",
    )

    result = engine.analyze_xauusd(
        entry,
        stop_loss=1990.0,
        take_profit=2020.0,
    )

    assert result.allowed is True


def test_xauusd_wrapper_rejects_other_symbol() -> None:
    """XAUUSD wrapper rejects other symbols."""

    engine = RiskRewardIntelligenceEngine()

    entry = make_entry(
        EntryDirection.LONG,
        symbol="EURUSD",
        entry_price=1.10,
    )

    with pytest.raises(RiskRewardIntelligenceError):
        engine.analyze_xauusd(
            entry,
            stop_loss=1.09,
            take_profit=1.12,
        )


def test_result_is_auditable() -> None:
    """Successful result contains important audit fields."""

    engine = RiskRewardIntelligenceEngine()

    entry = make_entry(EntryDirection.LONG)

    result = engine.analyze(
        entry,
        stop_loss=1990.0,
        take_profit=2020.0,
    )

    assert result.timestamp == entry.timestamp
    assert result.symbol == "XAUUSD"
    assert result.timeframe == "M15"
    assert result.direction is EntryDirection.LONG
    assert result.entry_price == 2000.0
    assert result.stop_loss == 1990.0
    assert result.take_profit == 2020.0
    assert result.risk_distance == 10.0
    assert result.reward_distance == 20.0
    assert result.rr == 2.0
    assert len(result.reasons) >= 2
    assert len(result.warnings) >= 1