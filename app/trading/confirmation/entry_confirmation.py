"""
Entry confirmation engine for the Phase 2 trading pipeline.

This module determines whether an EntryModel has sufficient market
confirmation to proceed toward the later risk-management stages.

Important architectural rules:
- This module does NOT execute trades.
- This module does NOT calculate position size.
- This module does NOT bypass the RiskEngine.
- Invalid market data must not be silently accepted.
- Confirmation must remain deterministic and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import math

from app.trading.data.market_bar import MarketBar
from app.trading.entry.entry_model import (
    EntryDirection,
    EntryModel,
    EntryTrigger,
)


class EntryConfirmationError(ValueError):
    """Raised when entry-confirmation input or configuration is invalid."""


class ConfirmationStatus(Enum):
    """Overall confirmation state."""

    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


class ConfirmationReasonType(Enum):
    """Machine-readable reason codes for confirmation decisions."""

    CANDLE_CLOSE_CONFIRMED = "CANDLE_CLOSE_CONFIRMED"
    CANDLE_CLOSE_REQUIRED = "CANDLE_CLOSE_REQUIRED"

    BREAKOUT_CONFIRMED = "BREAKOUT_CONFIRMED"
    BREAKOUT_NOT_CONFIRMED = "BREAKOUT_NOT_CONFIRMED"

    RETEST_CONFIRMED = "RETEST_CONFIRMED"
    RETEST_REQUIRED = "RETEST_REQUIRED"

    REJECTION_CONFIRMED = "REJECTION_CONFIRMED"
    REJECTION_NOT_CONFIRMED = "REJECTION_NOT_CONFIRMED"

    MOMENTUM_CONFIRMED = "MOMENTUM_CONFIRMED"
    MOMENTUM_NOT_CONFIRMED = "MOMENTUM_NOT_CONFIRMED"

    VOLUME_CONFIRMED = "VOLUME_CONFIRMED"
    VOLUME_NOT_CONFIRMED = "VOLUME_NOT_CONFIRMED"

    MTF_CONFIRMED = "MTF_CONFIRMED"
    MTF_NOT_CONFIRMED = "MTF_NOT_CONFIRMED"

    SPREAD_ACCEPTABLE = "SPREAD_ACCEPTABLE"
    SPREAD_TOO_HIGH = "SPREAD_TOO_HIGH"

    VOLATILITY_ACCEPTABLE = "VOLATILITY_ACCEPTABLE"
    VOLATILITY_TOO_HIGH = "VOLATILITY_TOO_HIGH"

    FALSE_BREAKOUT_PROTECTED = "FALSE_BREAKOUT_PROTECTED"
    FALSE_BREAKOUT_DETECTED = "FALSE_BREAKOUT_DETECTED"

    STALE_CONFIRMATION = "STALE_CONFIRMATION"

    INVALID_ENTRY = "INVALID_ENTRY"

    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

    CONFIRMATION_PASSED = "CONFIRMATION_PASSED"
    CONFIRMATION_BLOCKED = "CONFIRMATION_BLOCKED"


@dataclass(frozen=True, slots=True)
class ConfirmationReason:
    """Structured explanation for a confirmation decision."""

    type: ConfirmationReasonType
    message: str

    @property
    def reason_type(self) -> ConfirmationReasonType:
        """Compatibility alias for the existing public API."""

        return self.type


@dataclass(frozen=True, slots=True)
class EntryConfirmationResult:
    """Immutable result produced by the entry confirmation engine."""

    timestamp: datetime
    symbol: str
    timeframe: str

    status: ConfirmationStatus

    direction: EntryDirection
    trigger: EntryTrigger

    candle_close_confirmed: bool
    breakout_confirmed: bool
    retest_confirmed: bool
    rejection_confirmed: bool
    momentum_confirmed: bool
    volume_confirmed: bool
    mtf_confirmed: bool

    spread_acceptable: bool
    volatility_acceptable: bool

    false_breakout_protected: bool
    stale: bool

    score: float

    valid: bool
    confirmation_allowed: bool

    reasons: tuple[ConfirmationReason, ...]
    warnings: tuple[str, ...]

    @property
    def is_confirmed(self) -> bool:
        """Return True when confirmation succeeded."""

        return self.status is ConfirmationStatus.CONFIRMED

    @property
    def is_rejected(self) -> bool:
        """Return True when confirmation was rejected."""

        return self.status is ConfirmationStatus.REJECTED

    @property
    def is_unknown(self) -> bool:
        """Return True when confirmation could not be determined."""

        return self.status is ConfirmationStatus.UNKNOWN


class EntryConfirmationEngine:
    """Deterministic entry confirmation engine."""

    def __init__(
        self,
        *,
        minimum_score: float = 60.0,
        max_spread: float = 5.0,
        max_volatility_ratio: float = 3.0,
        max_confirmation_age_minutes: float = 15.0,
        maximum_spread: float | None = None,
        maximum_volatility_ratio: float | None = None,
        maximum_confirmation_age_minutes: float | None = None,
    ) -> None:
        """Initialize the confirmation engine."""

        if maximum_spread is not None:
            max_spread = maximum_spread

        if maximum_volatility_ratio is not None:
            max_volatility_ratio = maximum_volatility_ratio

        if maximum_confirmation_age_minutes is not None:
            max_confirmation_age_minutes = maximum_confirmation_age_minutes

        self._validate_configuration(
            minimum_score=minimum_score,
            max_spread=max_spread,
            max_volatility_ratio=max_volatility_ratio,
            max_confirmation_age_minutes=max_confirmation_age_minutes,
        )

        self.minimum_score = float(minimum_score)
        self.max_spread = float(max_spread)
        self.max_volatility_ratio = float(max_volatility_ratio)
        self.max_confirmation_age_minutes = float(
            max_confirmation_age_minutes
        )

        self.maximum_spread = self.max_spread
        self.maximum_volatility_ratio = self.max_volatility_ratio
        self.maximum_confirmation_age_minutes = (
            self.max_confirmation_age_minutes
        )

    def confirm(
        self,
        *,
        entry: EntryModel,
        bars: tuple[MarketBar, ...],
        current_timestamp: datetime,
        spread: float | None = None,
        volatility_ratio: float | None = None,
        volume_confirmed: bool | None = None,
        mtf_confirmed: bool | None = None,
    ) -> EntryConfirmationResult:
        """Confirm an entry using deterministic market evidence."""

        self._validate_confirm_inputs(
            entry=entry,
            bars=bars,
            current_timestamp=current_timestamp,
            spread=spread,
            volatility_ratio=volatility_ratio,
            volume_confirmed=volume_confirmed,
            mtf_confirmed=mtf_confirmed,
        )

        if not entry.valid or not entry.entry_allowed:
            return self._blocked_result(
                entry=entry,
                current_timestamp=current_timestamp,
                reason=ConfirmationReason(
                    ConfirmationReasonType.INVALID_ENTRY,
                    "Entry model is invalid or entry is not allowed.",
                ),
            )

        if entry.direction not in (
            EntryDirection.LONG,
            EntryDirection.SHORT,
        ):
            return self._blocked_result(
                entry=entry,
                current_timestamp=current_timestamp,
                reason=ConfirmationReason(
                    ConfirmationReasonType.INVALID_ENTRY,
                    "Entry direction is not actionable.",
                ),
            )

        if not bars:
            return EntryConfirmationResult(
                timestamp=current_timestamp,
                symbol=entry.symbol,
                timeframe=entry.timeframe,
                status=ConfirmationStatus.UNKNOWN,
                direction=entry.direction,
                trigger=entry.trigger,
                candle_close_confirmed=False,
                breakout_confirmed=False,
                retest_confirmed=False,
                rejection_confirmed=False,
                momentum_confirmed=False,
                volume_confirmed=False,
                mtf_confirmed=False,
                spread_acceptable=spread is None,
                volatility_acceptable=volatility_ratio is None,
                false_breakout_protected=False,
                stale=False,
                score=0.0,
                valid=True,
                confirmation_allowed=False,
                reasons=(
                    ConfirmationReason(
                        ConfirmationReasonType.INSUFFICIENT_DATA,
                        "No market candles are available for confirmation.",
                    ),
                ),
                warnings=(
                    "Confirmation does not execute trades or calculate risk.",
                ),
            )

        for bar in bars:
            if bar.symbol != entry.symbol:
                raise EntryConfirmationError(
                    "Market-bar symbol does not match the entry symbol."
                )

            if bar.timeframe != entry.timeframe:
                raise EntryConfirmationError(
                    "Market-bar timeframe does not match the entry timeframe."
                )

        candle_close_confirmed = self._candle_close_confirmation(
            entry=entry,
            bars=bars,
        )

        breakout_confirmed = self._breakout_confirmation(
            entry=entry,
            bars=bars,
        )

        retest_confirmed = self._retest_confirmation(
            entry=entry,
            bars=bars,
        )

        rejection_confirmed = self._rejection_confirmation(
            entry=entry,
            bars=bars,
        )

        momentum_confirmed = self._momentum_confirmation(
            entry=entry,
            bars=bars,
        )

        if volume_confirmed is None:
            volume_confirmed_result = bool(entry.volume_confirmed)
        else:
            volume_confirmed_result = volume_confirmed

        if mtf_confirmed is None:
            mtf_confirmed_result = bool(entry.mtf_confirmed)
        else:
            mtf_confirmed_result = mtf_confirmed

        if spread is None:
            spread_acceptable = True
        else:
            spread_acceptable = spread <= self.max_spread

        if volatility_ratio is None:
            volatility_acceptable = True
        else:
            volatility_acceptable = (
                volatility_ratio <= self.max_volatility_ratio
            )

        false_breakout_protected = self._false_breakout_protection(
            entry=entry,
            bars=bars,
        )

        stale = self._is_stale(
            latest_bar_timestamp=bars[-1].timestamp,
            current_timestamp=current_timestamp,
        )

        score = self._calculate_score(
            entry=entry,
            candle_close_confirmed=candle_close_confirmed,
            breakout_confirmed=breakout_confirmed,
            retest_confirmed=retest_confirmed,
            rejection_confirmed=rejection_confirmed,
            momentum_confirmed=momentum_confirmed,
            volume_confirmed=volume_confirmed_result,
            mtf_confirmed=mtf_confirmed_result,
            spread_acceptable=spread_acceptable,
            volatility_acceptable=volatility_acceptable,
            false_breakout_protected=false_breakout_protected,
        )

        reasons = self._build_reasons(
            entry=entry,
            candle_close_confirmed=candle_close_confirmed,
            breakout_confirmed=breakout_confirmed,
            retest_confirmed=retest_confirmed,
            rejection_confirmed=rejection_confirmed,
            momentum_confirmed=momentum_confirmed,
            volume_confirmed=volume_confirmed_result,
            mtf_confirmed=mtf_confirmed_result,
            spread_acceptable=spread_acceptable,
            volatility_acceptable=volatility_acceptable,
            false_breakout_protected=false_breakout_protected,
            stale=stale,
            score=score,
        )

        warnings = self._build_warnings(
            entry=entry,
            spread=spread,
            volatility_ratio=volatility_ratio,
        )

        confirmation_allowed = self._confirmation_allowed(
            entry=entry,
            candle_close_confirmed=candle_close_confirmed,
            breakout_confirmed=breakout_confirmed,
            retest_confirmed=retest_confirmed,
            rejection_confirmed=rejection_confirmed,
            momentum_confirmed=momentum_confirmed,
            volume_confirmed=volume_confirmed_result,
            mtf_confirmed=mtf_confirmed_result,
            spread_acceptable=spread_acceptable,
            volatility_acceptable=volatility_acceptable,
            false_breakout_protected=false_breakout_protected,
            stale=stale,
            score=score,
        )

        if confirmation_allowed:
            status = ConfirmationStatus.CONFIRMED

            reasons = reasons + (
                ConfirmationReason(
                    ConfirmationReasonType.CONFIRMATION_PASSED,
                    "All required entry confirmation gates passed.",
                ),
            )
        else:
            status = ConfirmationStatus.REJECTED

            reasons = reasons + (
                ConfirmationReason(
                    ConfirmationReasonType.CONFIRMATION_BLOCKED,
                    "One or more required confirmation gates failed.",
                ),
            )

        return EntryConfirmationResult(
            timestamp=current_timestamp,
            symbol=entry.symbol,
            timeframe=entry.timeframe,
            status=status,
            direction=entry.direction,
            trigger=entry.trigger,
            candle_close_confirmed=candle_close_confirmed,
            breakout_confirmed=breakout_confirmed,
            retest_confirmed=retest_confirmed,
            rejection_confirmed=rejection_confirmed,
            momentum_confirmed=momentum_confirmed,
            volume_confirmed=volume_confirmed_result,
            mtf_confirmed=mtf_confirmed_result,
            spread_acceptable=spread_acceptable,
            volatility_acceptable=volatility_acceptable,
            false_breakout_protected=false_breakout_protected,
            stale=stale,
            score=score,
            valid=True,
            confirmation_allowed=confirmation_allowed,
            reasons=reasons,
            warnings=warnings,
        )

    def confirm_xauusd(
        self,
        *,
        entry: EntryModel,
        bars: tuple[MarketBar, ...],
        current_timestamp: datetime,
        spread: float | None = None,
        volatility_ratio: float | None = None,
        volume_confirmed: bool | None = None,
        mtf_confirmed: bool | None = None,
    ) -> EntryConfirmationResult:
        """Confirm an XAU/USD entry."""

        if entry.symbol.upper() != "XAUUSD":
            raise EntryConfirmationError(
                "XAU/USD confirmation requires symbol XAUUSD."
            )

        return self.confirm(
            entry=entry,
            bars=bars,
            current_timestamp=current_timestamp,
            spread=spread,
            volatility_ratio=volatility_ratio,
            volume_confirmed=volume_confirmed,
            mtf_confirmed=mtf_confirmed,
        )

    def _candle_close_confirmation(
        self,
        *,
        entry: EntryModel,
        bars: tuple[MarketBar, ...],
    ) -> bool:
        """Confirm that the latest candle closes in the expected direction."""

        if not bars:
            return False

        latest = bars[-1]

        if entry.direction is EntryDirection.LONG:
            return (
                latest.close >= entry.reference_price
                and latest.close >= latest.open
            )

        if entry.direction is EntryDirection.SHORT:
            return (
                latest.close <= entry.reference_price
                and latest.close <= latest.open
            )

        return False

    def _breakout_confirmation(
        self,
        *,
        entry: EntryModel,
        bars: tuple[MarketBar, ...],
    ) -> bool:
        """Confirm a breakout using structural follow-through."""

        if entry.trigger is not EntryTrigger.BREAKOUT:
            return True

        if len(bars) < 3:
            return False

        structure_bar = bars[-3]
        breakout_bar = bars[-2]
        confirmation_bar = bars[-1]

        if entry.direction is EntryDirection.LONG:
            return (
                breakout_bar.close > structure_bar.high
                and confirmation_bar.close > structure_bar.high
            )

        if entry.direction is EntryDirection.SHORT:
            return (
                breakout_bar.close < structure_bar.low
                and confirmation_bar.close < structure_bar.low
            )

        return False

    def _retest_confirmation(
        self,
        *,
        entry: EntryModel,
        bars: tuple[MarketBar, ...],
    ) -> bool:
        """Confirm a pullback/retest around the reference price."""

        if entry.trigger is not EntryTrigger.PULLBACK:
            return True

        if len(bars) < 2:
            return False

        latest = bars[-1]

        tolerance = max(
            abs(entry.reference_price) * 0.002,
            0.01,
        )

        touched_reference = (
            abs(latest.close - entry.reference_price) <= tolerance
            or abs(latest.low - entry.reference_price) <= tolerance
            or abs(latest.high - entry.reference_price) <= tolerance
        )

        if entry.direction is EntryDirection.LONG:
            return touched_reference and latest.close >= latest.open

        if entry.direction is EntryDirection.SHORT:
            return touched_reference and latest.close <= latest.open

        return False

    def _rejection_confirmation(
        self,
        *,
        entry: EntryModel,
        bars: tuple[MarketBar, ...],
    ) -> bool:
        """Confirm a directional rejection candle."""

        if entry.trigger is not EntryTrigger.REJECTION:
            return True

        if not bars:
            return False

        latest = bars[-1]

        candle_range = latest.high - latest.low

        if candle_range <= 0:
            return False

        body = abs(latest.close - latest.open)

        upper_wick = latest.high - max(
            latest.open,
            latest.close,
        )

        lower_wick = min(
            latest.open,
            latest.close,
        ) - latest.low

        if entry.direction is EntryDirection.LONG:
            return (
                lower_wick >= candle_range * 0.40
                and lower_wick >= body
                and latest.close >= latest.open
            )

        if entry.direction is EntryDirection.SHORT:
            return (
                upper_wick >= candle_range * 0.40
                and upper_wick >= body
                and latest.close <= latest.open
            )

        return False

    def _momentum_confirmation(
        self,
        *,
        entry: EntryModel,
        bars: tuple[MarketBar, ...],
    ) -> bool:
        """Confirm directional momentum using the latest two candles."""

        if len(bars) < 2:
            return False

        previous = bars[-2]
        latest = bars[-1]

        if entry.direction is EntryDirection.LONG:
            return (
                latest.close > previous.close
                and latest.close >= latest.open
            )

        if entry.direction is EntryDirection.SHORT:
            return (
                latest.close < previous.close
                and latest.close <= latest.open
            )

        return False

    def _false_breakout_protection(
        self,
        *,
        entry: EntryModel,
        bars: tuple[MarketBar, ...],
    ) -> bool:
        """Protect against a breakout immediately failing."""

        if entry.trigger is not EntryTrigger.BREAKOUT:
            return True

        if len(bars) < 3:
            return False

        structure_bar = bars[-3]
        breakout_bar = bars[-2]
        confirmation_bar = bars[-1]

        if entry.direction is EntryDirection.LONG:
            return (
                breakout_bar.close > structure_bar.high
                and confirmation_bar.low >= structure_bar.high
            )

        if entry.direction is EntryDirection.SHORT:
            return (
                breakout_bar.close < structure_bar.low
                and confirmation_bar.high <= structure_bar.low
            )

        return False

    def _calculate_score(
        self,
        *,
        entry: EntryModel,
        candle_close_confirmed: bool,
        breakout_confirmed: bool,
        retest_confirmed: bool,
        rejection_confirmed: bool,
        momentum_confirmed: bool,
        volume_confirmed: bool,
        mtf_confirmed: bool,
        spread_acceptable: bool,
        volatility_acceptable: bool,
        false_breakout_protected: bool,
    ) -> float:
        """Calculate deterministic confirmation score from 0 to 100."""

        score = 0.0

        if candle_close_confirmed:
            score += 20.0

        if entry.trigger is EntryTrigger.BREAKOUT:
            if breakout_confirmed:
                score += 20.0

            if false_breakout_protected:
                score += 15.0

        elif entry.trigger is EntryTrigger.PULLBACK:
            if retest_confirmed:
                score += 20.0

        elif entry.trigger is EntryTrigger.REJECTION:
            if rejection_confirmed:
                score += 20.0

        else:
            if momentum_confirmed:
                score += 15.0

        if momentum_confirmed:
            score += 15.0

        if volume_confirmed:
            score += 10.0

        if mtf_confirmed:
            score += 10.0

        if spread_acceptable:
            score += 5.0

        if volatility_acceptable:
            score += 5.0

        return min(score, 100.0)

    def _confirmation_allowed(
        self,
        *,
        entry: EntryModel,
        candle_close_confirmed: bool,
        breakout_confirmed: bool,
        retest_confirmed: bool,
        rejection_confirmed: bool,
        momentum_confirmed: bool,
        volume_confirmed: bool,
        mtf_confirmed: bool,
        spread_acceptable: bool,
        volatility_acceptable: bool,
        false_breakout_protected: bool,
        stale: bool,
        score: float,
    ) -> bool:
        """Apply hard confirmation gates."""

        if not entry.valid or not entry.entry_allowed:
            return False

        if entry.direction not in (
            EntryDirection.LONG,
            EntryDirection.SHORT,
        ):
            return False

        if not candle_close_confirmed:
            return False

        if not spread_acceptable:
            return False

        if not volatility_acceptable:
            return False

        if stale:
            return False

        if entry.mtf_confirmed and not mtf_confirmed:
            return False

        if entry.volume_confirmed and not volume_confirmed:
            return False

        if entry.trigger is EntryTrigger.BREAKOUT:
            if not breakout_confirmed:
                return False

            if not false_breakout_protected:
                return False

        if entry.trigger is EntryTrigger.PULLBACK:
            if not retest_confirmed:
                return False

        if entry.trigger is EntryTrigger.REJECTION:
            if not rejection_confirmed:
                return False

        if score < self.minimum_score:
            return False

        return True

    def _build_reasons(
        self,
        *,
        entry: EntryModel,
        candle_close_confirmed: bool,
        breakout_confirmed: bool,
        retest_confirmed: bool,
        rejection_confirmed: bool,
        momentum_confirmed: bool,
        volume_confirmed: bool,
        mtf_confirmed: bool,
        spread_acceptable: bool,
        volatility_acceptable: bool,
        false_breakout_protected: bool,
        stale: bool,
        score: float,
    ) -> tuple[ConfirmationReason, ...]:
        """Build structured confirmation reasons."""

        reasons: list[ConfirmationReason] = []

        if candle_close_confirmed:
            reasons.append(
                ConfirmationReason(
                    ConfirmationReasonType.CANDLE_CLOSE_CONFIRMED,
                    "Latest candle closed in the expected entry direction.",
                )
            )
        else:
            reasons.append(
                ConfirmationReason(
                    ConfirmationReasonType.CANDLE_CLOSE_REQUIRED,
                    "Latest candle did not provide the required directional close.",
                )
            )

        if entry.trigger is EntryTrigger.BREAKOUT:
            if breakout_confirmed:
                reasons.append(
                    ConfirmationReason(
                        ConfirmationReasonType.BREAKOUT_CONFIRMED,
                        "Breakout and follow-through were confirmed.",
                    )
                )
            else:
                reasons.append(
                    ConfirmationReason(
                        ConfirmationReasonType.BREAKOUT_NOT_CONFIRMED,
                        "Breakout follow-through was not confirmed.",
                    )
                )

            if false_breakout_protected:
                reasons.append(
                    ConfirmationReason(
                        ConfirmationReasonType.FALSE_BREAKOUT_PROTECTED,
                        "Confirmation candle held beyond the breakout level.",
                    )
                )
            else:
                reasons.append(
                    ConfirmationReason(
                        ConfirmationReasonType.FALSE_BREAKOUT_DETECTED,
                        "False-breakout protection failed.",
                    )
                )

        if entry.trigger is EntryTrigger.PULLBACK:
            if retest_confirmed:
                reasons.append(
                    ConfirmationReason(
                        ConfirmationReasonType.RETEST_CONFIRMED,
                        "Pullback/retest confirmation passed.",
                    )
                )
            else:
                reasons.append(
                    ConfirmationReason(
                        ConfirmationReasonType.RETEST_REQUIRED,
                        "Required pullback/retest confirmation was not present.",
                    )
                )

        if entry.trigger is EntryTrigger.REJECTION:
            if rejection_confirmed:
                reasons.append(
                    ConfirmationReason(
                        ConfirmationReasonType.REJECTION_CONFIRMED,
                        "Directional rejection candle was confirmed.",
                    )
                )
            else:
                reasons.append(
                    ConfirmationReason(
                        ConfirmationReasonType.REJECTION_NOT_CONFIRMED,
                        "Required rejection candle confirmation was not present.",
                    )
                )

        if momentum_confirmed:
            reasons.append(
                ConfirmationReason(
                    ConfirmationReasonType.MOMENTUM_CONFIRMED,
                    "Directional momentum was confirmed.",
                )
            )
        else:
            reasons.append(
                ConfirmationReason(
                    ConfirmationReasonType.MOMENTUM_NOT_CONFIRMED,
                    "Directional momentum was not confirmed.",
                )
            )

        if volume_confirmed:
            reasons.append(
                ConfirmationReason(
                    ConfirmationReasonType.VOLUME_CONFIRMED,
                    "Volume confirmation is present.",
                )
            )
        else:
            reasons.append(
                ConfirmationReason(
                    ConfirmationReasonType.VOLUME_NOT_CONFIRMED,
                    "Volume confirmation is not present.",
                )
            )

        if mtf_confirmed:
            reasons.append(
                ConfirmationReason(
                    ConfirmationReasonType.MTF_CONFIRMED,
                    "Multi-timeframe confirmation is present.",
                )
            )
        else:
            reasons.append(
                ConfirmationReason(
                    ConfirmationReasonType.MTF_NOT_CONFIRMED,
                    "Multi-timeframe confirmation is not present.",
                )
            )

        if spread_acceptable:
            reasons.append(
                ConfirmationReason(
                    ConfirmationReasonType.SPREAD_ACCEPTABLE,
                    "Spread is within the configured limit.",
                )
            )
        else:
            reasons.append(
                ConfirmationReason(
                    ConfirmationReasonType.SPREAD_TOO_HIGH,
                    "Spread exceeds the configured limit.",
                )
            )

        if volatility_acceptable:
            reasons.append(
                ConfirmationReason(
                    ConfirmationReasonType.VOLATILITY_ACCEPTABLE,
                    "Volatility is within the configured limit.",
                )
            )
        else:
            reasons.append(
                ConfirmationReason(
                    ConfirmationReasonType.VOLATILITY_TOO_HIGH,
                    "Volatility exceeds the configured limit.",
                )
            )

        if stale:
            reasons.append(
                ConfirmationReason(
                    ConfirmationReasonType.STALE_CONFIRMATION,
                    "The confirmation is older than the configured maximum age.",
                )
            )

        reasons.append(
            ConfirmationReason(
                ConfirmationReasonType.CONFIRMATION_PASSED
                if score >= self.minimum_score
                else ConfirmationReasonType.CONFIRMATION_BLOCKED,
                (
                    f"Confirmation score is {score:.2f}; "
                    f"minimum required score is {self.minimum_score:.2f}."
                ),
            )
        )

        return tuple(reasons)

    def _build_warnings(
        self,
        *,
        entry: EntryModel,
        spread: float | None,
        volatility_ratio: float | None,
    ) -> tuple[str, ...]:
        """Build human-readable confirmation warnings."""

        warnings: list[str] = [
            "Confirmation does not execute trades or calculate risk."
        ]

        if spread is None:
            warnings.append(
                "Spread was not supplied; spread gate was treated as acceptable."
            )

        if volatility_ratio is None:
            warnings.append(
                "Volatility ratio was not supplied; volatility gate was treated as acceptable."
            )

        warnings.extend(entry.warnings)

        return tuple(warnings)

    def _is_stale(
        self,
        *,
        latest_bar_timestamp: datetime,
        current_timestamp: datetime,
    ) -> bool:
        """Determine whether confirmation evidence is stale."""

        age_seconds = (
            current_timestamp - latest_bar_timestamp
        ).total_seconds()

        if age_seconds < 0:
            return True

        maximum_age_seconds = (
            self.max_confirmation_age_minutes * 60.0
        )

        return age_seconds > maximum_age_seconds

    def _blocked_result(
        self,
        *,
        entry: EntryModel,
        current_timestamp: datetime,
        reason: ConfirmationReason,
    ) -> EntryConfirmationResult:
        """Create a standardized rejected confirmation result."""

        return EntryConfirmationResult(
            timestamp=current_timestamp,
            symbol=entry.symbol,
            timeframe=entry.timeframe,
            status=ConfirmationStatus.REJECTED,
            direction=entry.direction,
            trigger=entry.trigger,
            candle_close_confirmed=False,
            breakout_confirmed=False,
            retest_confirmed=False,
            rejection_confirmed=False,
            momentum_confirmed=False,
            volume_confirmed=False,
            mtf_confirmed=False,
            spread_acceptable=False,
            volatility_acceptable=False,
            false_breakout_protected=False,
            stale=False,
            score=0.0,
            valid=False,
            confirmation_allowed=False,
            reasons=(reason,),
            warnings=(
                "Confirmation does not execute trades or calculate risk.",
            ),
        )

    def _validate_confirm_inputs(
        self,
        *,
        entry: EntryModel,
        bars: tuple[MarketBar, ...],
        current_timestamp: datetime,
        spread: float | None,
        volatility_ratio: float | None,
        volume_confirmed: bool | None,
        mtf_confirmed: bool | None,
    ) -> None:
        """Validate all confirmation inputs."""

        if not isinstance(entry, EntryModel):
            raise EntryConfirmationError(
                "entry must be an EntryModel instance."
            )

        if not isinstance(bars, tuple):
            raise EntryConfirmationError(
                "bars must be a tuple of MarketBar objects."
            )

        for bar in bars:
            if not isinstance(bar, MarketBar):
                raise EntryConfirmationError(
                    "Every item in bars must be a MarketBar instance."
                )

        if not isinstance(current_timestamp, datetime):
            raise EntryConfirmationError(
                "current_timestamp must be a datetime."
            )

        for previous, current in zip(bars, bars[1:]):
            if current.timestamp <= previous.timestamp:
                raise EntryConfirmationError(
                    "Market bars must be strictly chronological."
                )

        if spread is not None:
            self._validate_finite_nonnegative(
                value=spread,
                field_name="spread",
            )

        if volatility_ratio is not None:
            self._validate_finite_nonnegative(
                value=volatility_ratio,
                field_name="volatility_ratio",
            )

        if volume_confirmed is not None and not isinstance(
            volume_confirmed,
            bool,
        ):
            raise EntryConfirmationError(
                "volume_confirmed must be bool or None."
            )

        if mtf_confirmed is not None and not isinstance(
            mtf_confirmed,
            bool,
        ):
            raise EntryConfirmationError(
                "mtf_confirmed must be bool or None."
            )

    @staticmethod
    def _validate_configuration(
        *,
        minimum_score: float,
        max_spread: float,
        max_volatility_ratio: float,
        max_confirmation_age_minutes: float,
    ) -> None:
        """Validate confirmation-engine configuration."""

        if not isinstance(minimum_score, (int, float)):
            raise EntryConfirmationError(
                "minimum_score must be numeric."
            )

        if not math.isfinite(float(minimum_score)):
            raise EntryConfirmationError(
                "minimum_score must be finite."
            )

        if not 0.0 <= float(minimum_score) <= 100.0:
            raise EntryConfirmationError(
                "minimum_score must be between 0 and 100."
            )

        if not isinstance(max_spread, (int, float)):
            raise EntryConfirmationError(
                "max_spread must be numeric."
            )

        if not math.isfinite(float(max_spread)):
            raise EntryConfirmationError(
                "max_spread must be finite."
            )

        if float(max_spread) < 0.0:
            raise EntryConfirmationError(
                "max_spread cannot be negative."
            )

        if not isinstance(max_volatility_ratio, (int, float)):
            raise EntryConfirmationError(
                "max_volatility_ratio must be numeric."
            )

        if not math.isfinite(float(max_volatility_ratio)):
            raise EntryConfirmationError(
                "max_volatility_ratio must be finite."
            )

        if float(max_volatility_ratio) < 0.0:
            raise EntryConfirmationError(
                "max_volatility_ratio cannot be negative."
            )

        if not isinstance(
            max_confirmation_age_minutes,
            (int, float),
        ):
            raise EntryConfirmationError(
                "max_confirmation_age_minutes must be numeric."
            )

        if not math.isfinite(
            float(max_confirmation_age_minutes)
        ):
            raise EntryConfirmationError(
                "max_confirmation_age_minutes must be finite."
            )

        if float(max_confirmation_age_minutes) < 0.0:
            raise EntryConfirmationError(
                "max_confirmation_age_minutes cannot be negative."
            )

    @staticmethod
    def _validate_finite_nonnegative(
        *,
        value: float,
        field_name: str,
    ) -> None:
        """Validate that a numeric value is finite and non-negative."""

        if not isinstance(value, (int, float)):
            raise EntryConfirmationError(
                f"{field_name} must be numeric."
            )

        if not math.isfinite(float(value)):
            raise EntryConfirmationError(
                f"{field_name} must be finite."
            )

        if float(value) < 0.0:
            raise EntryConfirmationError(
                f"{field_name} cannot be negative."
            )