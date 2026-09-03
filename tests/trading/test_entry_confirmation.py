# Enable modern type-hint behavior.
from __future__ import annotations

# Import datetime utilities.
from datetime import datetime, timedelta

# Import pytest for exception assertions.
import pytest

# Import the existing EntryModel and enums.
from app.trading.entry.entry_model import (
    EntryDirection,
    EntryModel,
    EntryQuality,
    EntryTrigger,
)

# Import the P2.13 confirmation engine.
from app.trading.confirmation.entry_confirmation import (
    ConfirmationReasonType,
    ConfirmationStatus,
    EntryConfirmationEngine,
    EntryConfirmationError,
)

# Import the canonical MarketBar model.
from app.trading.data.market_bar import MarketBar


# Use a deterministic timestamp for all tests.
TIMESTAMP = datetime(2026, 1, 1, 12, 0, 0)


def make_bar(
    *,
    timestamp: datetime,
    symbol: str = "XAUUSD",
    timeframe: str = "M5",
    open_price: float = 100.0,
    high: float = 102.0,
    low: float = 99.0,
    close: float = 101.0,
    volume: float = 100.0,
) -> MarketBar:
    """Create a valid MarketBar."""

    # Return the canonical market-bar object.
    return MarketBar(
        timestamp=timestamp,
        symbol=symbol,
        timeframe=timeframe,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def make_entry(
    *,
    direction: EntryDirection = EntryDirection.LONG,
    trigger: EntryTrigger = EntryTrigger.MARKET,
    volume_confirmed: bool = False,
    mtf_confirmed: bool = False,
) -> EntryModel:
    """Create a valid entry candidate."""

    # Return an entry already approved by the EntryModel stage.
    return EntryModel(
        timestamp=TIMESTAMP,
        symbol="XAUUSD",
        timeframe="M5",
        direction=direction,
        trigger=trigger,
        quality=EntryQuality.GOOD,
        reference_price=100.0,
        entry_price=100.5,
        confluence_score=80.0,
        entry_confidence=80.0,
        volume_confirmed=volume_confirmed,
        mtf_confirmed=mtf_confirmed,
        valid=True,
        entry_allowed=True,
    )


def test_market_entry_can_be_confirmed() -> None:
    """A clean market entry should pass confirmation."""

    # Create the confirmation engine.
    engine = EntryConfirmationEngine(
        minimum_score=60.0,
    )

    # Create two bullish candles.
    bars = (
        make_bar(
            timestamp=TIMESTAMP - timedelta(minutes=5),
            open_price=99.0,
            high=100.5,
            low=98.5,
            close=100.0,
        ),
        make_bar(
            timestamp=TIMESTAMP,
            open_price=100.0,
            high=102.0,
            low=99.5,
            close=101.5,
        ),
    )

    # Run confirmation.
    result = engine.confirm(
        entry=make_entry(),
        bars=bars,
        current_timestamp=TIMESTAMP,
        volume_confirmed=True,
        mtf_confirmed=True,
    )

    # Confirmation should pass.
    assert result.status is ConfirmationStatus.CONFIRMED

    # The final permission must be true.
    assert result.confirmation_allowed is True

    # Candle confirmation must pass.
    assert result.candle_close_confirmed is True


def test_invalid_entry_is_rejected() -> None:
    """An EntryModel that is not allowed cannot be confirmed."""

    # Create the engine.
    engine = EntryConfirmationEngine()

    # Create an invalid entry.
    entry = EntryModel(
        timestamp=TIMESTAMP,
        symbol="XAUUSD",
        timeframe="M5",
        direction=EntryDirection.LONG,
        trigger=EntryTrigger.MARKET,
        quality=EntryQuality.WEAK,
        reference_price=100.0,
        valid=False,
        entry_allowed=False,
    )

    # Provide valid market data.
    bars = (
        make_bar(
            timestamp=TIMESTAMP,
            close=101.0,
        ),
    )

    # Run confirmation.
    result = engine.confirm(
        entry=entry,
        bars=bars,
        current_timestamp=TIMESTAMP,
    )

    # Confirmation must be rejected.
    assert result.status is ConfirmationStatus.REJECTED

    # Permission must remain disabled.
    assert result.confirmation_allowed is False

    # The explicit invalid-entry reason must exist.
    assert any(
        reason.reason_type is ConfirmationReasonType.INVALID_ENTRY
        for reason in result.reasons
    )


def test_high_spread_blocks_confirmation() -> None:
    """Excessive spread must block confirmation."""

    # Create the engine.
    engine = EntryConfirmationEngine(
        max_spread=5.0,
    )

    # Create valid bullish candles.
    bars = (
        make_bar(
            timestamp=TIMESTAMP - timedelta(minutes=5),
            close=100.0,
        ),
        make_bar(
            timestamp=TIMESTAMP,
            open_price=100.0,
            high=103.0,
            low=99.5,
            close=102.0,
        ),
    )

    # Supply a spread above the configured limit.
    result = engine.confirm(
        entry=make_entry(),
        bars=bars,
        current_timestamp=TIMESTAMP,
        spread=6.0,
        volume_confirmed=True,
        mtf_confirmed=True,
    )

    # Confirmation must be blocked.
    assert result.confirmation_allowed is False

    # Spread must be marked unacceptable.
    assert result.spread_acceptable is False


def test_high_volatility_blocks_confirmation() -> None:
    """Excessive volatility must block confirmation."""

    # Create the engine.
    engine = EntryConfirmationEngine(
        max_volatility_ratio=3.0,
    )

    # Create valid candles.
    bars = (
        make_bar(
            timestamp=TIMESTAMP - timedelta(minutes=5),
            close=100.0,
        ),
        make_bar(
            timestamp=TIMESTAMP,
            open_price=100.0,
            high=103.0,
            low=99.5,
            close=102.0,
        ),
    )

    # Supply excessive volatility.
    result = engine.confirm(
        entry=make_entry(),
        bars=bars,
        current_timestamp=TIMESTAMP,
        volatility_ratio=4.0,
        volume_confirmed=True,
        mtf_confirmed=True,
    )

    # Confirmation must be rejected.
    assert result.confirmation_allowed is False

    # Volatility must be marked unacceptable.
    assert result.volatility_acceptable is False


def test_stale_confirmation_is_rejected() -> None:
    """Old confirmation evidence must not be tradable."""

    # Create the engine.
    engine = EntryConfirmationEngine(
        max_confirmation_age_minutes=15,
    )

    # Create an hour-old candle.
    old_timestamp = TIMESTAMP - timedelta(hours=1)

    # Supply the stale candle.
    bars = (
        make_bar(
            timestamp=old_timestamp,
            close=101.0,
        ),
    )

    # Confirm using the current timestamp.
    result = engine.confirm(
        entry=make_entry(),
        bars=bars,
        current_timestamp=TIMESTAMP,
        volume_confirmed=True,
        mtf_confirmed=True,
    )

    # Stale confirmation must be blocked.
    assert result.confirmation_allowed is False

    # Stale state must be exposed.
    assert result.stale is True


def test_breakout_requires_follow_through() -> None:
    """Breakout entries require breakout confirmation and protection."""

    # Create the confirmation engine.
    engine = EntryConfirmationEngine()

    # Create three valid candles.
    #
    # Candle 1 establishes the resistance level at 100.
    # Candle 2 breaks above 100.
    # Candle 3 holds above 100.
    bars = (
        make_bar(
            timestamp=TIMESTAMP - timedelta(minutes=10),
            open_price=98.0,
            high=100.0,
            low=97.0,
            close=99.0,
        ),
        make_bar(
            timestamp=TIMESTAMP - timedelta(minutes=5),
            open_price=100.0,
            high=103.0,
            low=99.5,
            close=102.0,
        ),
        make_bar(
            timestamp=TIMESTAMP,
            open_price=102.0,
            high=104.0,
            low=100.5,
            close=103.0,
        ),
    )

    # Create the breakout entry.
    entry = make_entry(
        trigger=EntryTrigger.BREAKOUT,
    )

    # Run confirmation.
    result = engine.confirm(
        entry=entry,
        bars=bars,
        current_timestamp=TIMESTAMP,
        volume_confirmed=True,
        mtf_confirmed=True,
    )

    # The breakout should be confirmed.
    assert result.breakout_confirmed is True

    # The breakout should also have follow-through protection.
    assert result.false_breakout_protected is True


def test_wrong_direction_candle_blocks_long() -> None:
    """A bearish candle cannot confirm a long entry."""

    # Create the engine.
    engine = EntryConfirmationEngine()

    # Create a bearish candle.
    bars = (
        make_bar(
            timestamp=TIMESTAMP,
            open_price=101.0,
            high=102.0,
            low=98.0,
            close=99.0,
        ),
    )

    # Run confirmation.
    result = engine.confirm(
        entry=make_entry(),
        bars=bars,
        current_timestamp=TIMESTAMP,
        volume_confirmed=True,
        mtf_confirmed=True,
    )

    # Candle confirmation must fail.
    assert result.candle_close_confirmed is False

    # Final confirmation must fail.
    assert result.confirmation_allowed is False


def test_wrong_symbol_is_rejected() -> None:
    """Entry and market data must use the same symbol."""

    # Create the engine.
    engine = EntryConfirmationEngine()

    # Create EURUSD data.
    bars = (
        make_bar(
            timestamp=TIMESTAMP,
            symbol="EURUSD",
            close=101.0,
        ),
    )

    # Symbol mismatch must raise the domain error.
    with pytest.raises(EntryConfirmationError):
        engine.confirm(
            entry=make_entry(),
            bars=bars,
            current_timestamp=TIMESTAMP,
        )


def test_bars_must_be_chronological() -> None:
    """Out-of-order candles must be rejected."""

    # Create the engine.
    engine = EntryConfirmationEngine()

    # Intentionally reverse the timestamps.
    bars = (
        make_bar(
            timestamp=TIMESTAMP,
            close=101.0,
        ),
        make_bar(
            timestamp=TIMESTAMP - timedelta(minutes=5),
            close=100.0,
        ),
    )

    # Invalid chronological ordering must raise.
    with pytest.raises(EntryConfirmationError):
        engine.confirm(
            entry=make_entry(),
            bars=bars,
            current_timestamp=TIMESTAMP,
        )


def test_xauusd_helper_rejects_other_symbol() -> None:
    """The XAUUSD helper must reject non-XAUUSD entries."""

    # Create the engine.
    engine = EntryConfirmationEngine()

    # Create an EURUSD entry.
    entry = EntryModel(
        timestamp=TIMESTAMP,
        symbol="EURUSD",
        timeframe="M5",
        direction=EntryDirection.LONG,
        trigger=EntryTrigger.MARKET,
        quality=EntryQuality.GOOD,
        reference_price=100.0,
        valid=True,
        entry_allowed=True,
    )

    # Create matching EURUSD data.
    bars = (
        make_bar(
            timestamp=TIMESTAMP,
            symbol="EURUSD",
            close=101.0,
        ),
    )

    # XAUUSD-specific API must reject it.
    with pytest.raises(EntryConfirmationError):
        engine.confirm_xauusd(
            entry=entry,
            bars=bars,
            current_timestamp=TIMESTAMP,
        )