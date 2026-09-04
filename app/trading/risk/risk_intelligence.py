"""
Deterministic trading risk intelligence engine.

The risk engine is the final risk-control layer before a trade can
be considered executable.

Architecture rule:

    Entry/Setup Engine -> identifies a possible trade.
    Risk Engine       -> decides whether that trade is allowed.

The risk engine must never increase position size to satisfy a minimum
volume requirement because doing so could increase monetary risk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RiskIntelligenceError(ValueError):
    """Raised when risk-engine input or configuration is invalid."""


class RiskDecision(str, Enum):
    """Final risk decision."""

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


class RiskReasonType(str, Enum):
    """Deterministic reason codes returned by the risk engine."""

    RISK_ALLOWED = "RISK_ALLOWED"

    INVALID_BALANCE = "INVALID_BALANCE"
    INVALID_EQUITY = "INVALID_EQUITY"
    INVALID_RISK_PERCENT = "INVALID_RISK_PERCENT"
    INVALID_ENTRY_PRICE = "INVALID_ENTRY_PRICE"
    INVALID_STOP_LOSS = "INVALID_STOP_LOSS"
    INVALID_STOP_DISTANCE = "INVALID_STOP_DISTANCE"

    INVALID_TICK_SIZE = "INVALID_TICK_SIZE"
    INVALID_TICK_VALUE = "INVALID_TICK_VALUE"
    INVALID_CONTRACT_SIZE = "INVALID_CONTRACT_SIZE"
    INVALID_VOLUME_STEP = "INVALID_VOLUME_STEP"
    INVALID_BROKER_VOLUME = "INVALID_BROKER_VOLUME"

    RISK_AMOUNT_TOO_HIGH = "RISK_AMOUNT_TOO_HIGH"

    POSITION_SIZE_TOO_SMALL = "POSITION_SIZE_TOO_SMALL"
    POSITION_SIZE_TOO_LARGE = "POSITION_SIZE_TOO_LARGE"
    POSITION_SIZE_ROUNDED_TO_ZERO = "POSITION_SIZE_ROUNDED_TO_ZERO"
    POSITION_SIZE_EXCEEDS_RISK = "POSITION_SIZE_EXCEEDS_RISK"

    SPREAD_TOO_HIGH = "SPREAD_TOO_HIGH"
    COMMISSION_TOO_HIGH = "COMMISSION_TOO_HIGH"
    SLIPPAGE_TOO_HIGH = "SLIPPAGE_TOO_HIGH"

    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"

    # Canonical public reason code used by the existing test/API contract.
    MAX_DRAWDOWN_LIMIT = "MAX_DRAWDOWN_LIMIT"

    # Backward-compatible alias.
    MAX_DRAWDOWN = "MAX_DRAWDOWN_LIMIT"

    MAX_OPEN_TRADES = "MAX_OPEN_TRADES"

    # Canonical public reason code used by the existing test/API contract.
    CONSECUTIVE_LOSS_LIMIT = "CONSECUTIVE_LOSS_LIMIT"

    # Backward-compatible alias.
    MAX_CONSECUTIVE_LOSSES = "CONSECUTIVE_LOSS_LIMIT"

    COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"
    SESSION_BLOCKED = "SESSION_BLOCKED"
    NEWS_BLOCKED = "NEWS_BLOCKED"
    CORRELATION_BLOCKED = "CORRELATION_BLOCKED"

    RISK_ENGINE_ERROR = "RISK_ENGINE_ERROR"


@dataclass(frozen=True, slots=True)
class RiskReason:
    """One deterministic explanation for a risk decision."""

    reason_type: RiskReasonType
    message: str


@dataclass(frozen=True, slots=True)
class RiskModel:
    """Complete result produced by the risk engine."""

    decision: RiskDecision

    balance: float
    equity: float

    risk_percent: float
    risk_amount: float

    entry_price: float
    stop_loss: float
    stop_distance: float

    position_size: float
    raw_position_size: float

    tick_size: float
    tick_value: float
    contract_size: float
    volume_step: float

    broker_min_position_size: float
    broker_max_position_size: float

    spread: float
    commission: float
    slippage: float

    daily_loss: float
    drawdown_percent: float

    open_trades: int
    consecutive_losses: int

    session_allowed: bool
    news_allowed: bool
    correlation_allowed: bool
    cooldown_active: bool

    reasons: tuple[RiskReason, ...]
    warnings: tuple[str, ...]

    @property
    def allowed(self) -> bool:
        """Return True when the trade is allowed."""

        return self.decision == RiskDecision.ALLOW

    @property
    def blocked(self) -> bool:
        """Return True when the trade is blocked."""

        return self.decision == RiskDecision.BLOCK

    @property
    def is_allowed(self) -> bool:
        """Compatibility alias."""

        return self.allowed

    @property
    def is_blocked(self) -> bool:
        """Compatibility alias."""

        return self.blocked


class RiskIntelligenceEngine:
    """Deterministic trading risk-control engine."""

    def __init__(
        self,
        *,
        default_risk_percent: float = 1.0,
        minimum_risk_percent: float = 0.01,
        maximum_risk_percent: float = 2.0,
        minimum_stop_distance: float = 0.01,
        minimum_position_size: float = 0.01,
        maximum_position_size: float = 100.0,
        maximum_spread: float = 5.0,
        maximum_commission: float = 100.0,
        maximum_slippage: float = 5.0,
        maximum_daily_loss: float = 5.0,
        maximum_drawdown_percent: float = 10.0,
        maximum_open_trades: int = 3,
        maximum_consecutive_losses: int = 3,
    ) -> None:
        """Initialize and validate risk-policy configuration."""

        self.default_risk_percent = self._validate_positive(
            default_risk_percent,
            "default_risk_percent",
        )

        self.minimum_risk_percent = self._validate_positive(
            minimum_risk_percent,
            "minimum_risk_percent",
        )

        self.maximum_risk_percent = self._validate_positive(
            maximum_risk_percent,
            "maximum_risk_percent",
        )

        self.minimum_stop_distance = self._validate_positive(
            minimum_stop_distance,
            "minimum_stop_distance",
        )

        self.minimum_position_size = self._validate_positive(
            minimum_position_size,
            "minimum_position_size",
        )

        self.maximum_position_size = self._validate_positive(
            maximum_position_size,
            "maximum_position_size",
        )

        self.maximum_spread = self._validate_non_negative(
            maximum_spread,
            "maximum_spread",
        )

        self.maximum_commission = self._validate_non_negative(
            maximum_commission,
            "maximum_commission",
        )

        self.maximum_slippage = self._validate_non_negative(
            maximum_slippage,
            "maximum_slippage",
        )

        self.maximum_daily_loss = self._validate_non_negative(
            maximum_daily_loss,
            "maximum_daily_loss",
        )

        self.maximum_drawdown_percent = self._validate_non_negative(
            maximum_drawdown_percent,
            "maximum_drawdown_percent",
        )

        self.maximum_open_trades = self._validate_non_negative_int(
            maximum_open_trades,
            "maximum_open_trades",
        )

        self.maximum_consecutive_losses = self._validate_non_negative_int(
            maximum_consecutive_losses,
            "maximum_consecutive_losses",
        )

        # The default risk percentage must itself be inside the
        # configured risk-policy range.
        if self.default_risk_percent < self.minimum_risk_percent:
            raise RiskIntelligenceError(
                "default_risk_percent cannot be below "
                "minimum_risk_percent."
            )

        if self.default_risk_percent > self.maximum_risk_percent:
            raise RiskIntelligenceError(
                "default_risk_percent cannot exceed "
                "maximum_risk_percent."
            )

        if self.minimum_risk_percent > self.maximum_risk_percent:
            raise RiskIntelligenceError(
                "minimum_risk_percent cannot exceed "
                "maximum_risk_percent."
            )

        if self.minimum_position_size > self.maximum_position_size:
            raise RiskIntelligenceError(
                "minimum_position_size cannot exceed "
                "maximum_position_size."
            )

    def analyze(
        self,
        *,
        balance: float,
        equity: float,
        entry_price: float,
        stop_loss: float,
        risk_percent: Optional[float] = None,
        tick_size: float = 1.0,
        tick_value: float = 1.0,
        contract_size: float = 1.0,
        volume_step: float = 0.01,
        broker_min_position_size: float = 0.01,
        broker_max_position_size: float = 100.0,
        spread: float = 0.0,
        commission: float = 0.0,
        slippage: float = 0.0,
        daily_loss: float = 0.0,
        drawdown_percent: float = 0.0,
        open_trades: int = 0,
        consecutive_losses: int = 0,
        session_allowed: bool = True,
        news_allowed: bool = True,
        correlation_allowed: bool = True,
        cooldown_active: bool = False,
    ) -> RiskModel:
        """Analyze a proposed trade."""

        # Validate account balance.
        balance_value = self._validate_positive(
            balance,
            "balance",
        )

        # Validate account equity.
        equity_value = self._validate_positive(
            equity,
            "equity",
        )

        # Use the configured default risk when no explicit value is
        # supplied.
        if risk_percent is None:
            risk_value = self.default_risk_percent
        else:
            risk_value = self._validate_positive(
                risk_percent,
                "risk_percent",
            )

        # Risk percentage below policy minimum is blocked.
        if risk_value < self.minimum_risk_percent:
            return self._blocked_result(
                balance=balance_value,
                equity=equity_value,
                risk_percent=risk_value,
                entry_price=entry_price,
                stop_loss=stop_loss,
                tick_size=tick_size,
                tick_value=tick_value,
                contract_size=contract_size,
                volume_step=volume_step,
                broker_min_position_size=broker_min_position_size,
                broker_max_position_size=broker_max_position_size,
                spread=spread,
                commission=commission,
                slippage=slippage,
                daily_loss=daily_loss,
                drawdown_percent=drawdown_percent,
                open_trades=open_trades,
                consecutive_losses=consecutive_losses,
                session_allowed=session_allowed,
                news_allowed=news_allowed,
                correlation_allowed=correlation_allowed,
                cooldown_active=cooldown_active,
                reason_type=RiskReasonType.INVALID_RISK_PERCENT,
                message="Risk percentage is below the configured minimum.",
            )

        # Risk percentage above policy maximum is blocked.
        if risk_value > self.maximum_risk_percent:
            return self._blocked_result(
                balance=balance_value,
                equity=equity_value,
                risk_percent=risk_value,
                entry_price=entry_price,
                stop_loss=stop_loss,
                tick_size=tick_size,
                tick_value=tick_value,
                contract_size=contract_size,
                volume_step=volume_step,
                broker_min_position_size=broker_min_position_size,
                broker_max_position_size=broker_max_position_size,
                spread=spread,
                commission=commission,
                slippage=slippage,
                daily_loss=daily_loss,
                drawdown_percent=drawdown_percent,
                open_trades=open_trades,
                consecutive_losses=consecutive_losses,
                session_allowed=session_allowed,
                news_allowed=news_allowed,
                correlation_allowed=correlation_allowed,
                cooldown_active=cooldown_active,
                reason_type=RiskReasonType.INVALID_RISK_PERCENT,
                message="Risk percentage exceeds the configured maximum.",
            )

        # Validate entry price.
        entry_value = self._validate_positive(
            entry_price,
            "entry_price",
        )

        # Validate stop loss.
        stop_value = self._validate_positive(
            stop_loss,
            "stop_loss",
        )

        # Entry and stop cannot be equal.
        if stop_value == entry_value:
            return self._blocked_result(
                balance=balance_value,
                equity=equity_value,
                risk_percent=risk_value,
                entry_price=entry_value,
                stop_loss=stop_value,
                tick_size=tick_size,
                tick_value=tick_value,
                contract_size=contract_size,
                volume_step=volume_step,
                broker_min_position_size=broker_min_position_size,
                broker_max_position_size=broker_max_position_size,
                spread=spread,
                commission=commission,
                slippage=slippage,
                daily_loss=daily_loss,
                drawdown_percent=drawdown_percent,
                open_trades=open_trades,
                consecutive_losses=consecutive_losses,
                session_allowed=session_allowed,
                news_allowed=news_allowed,
                correlation_allowed=correlation_allowed,
                cooldown_active=cooldown_active,
                reason_type=RiskReasonType.INVALID_STOP_DISTANCE,
                message="Entry price and stop loss cannot be equal.",
            )

        # Calculate absolute stop distance.
        stop_distance = abs(
            entry_value - stop_value,
        )

        # Stop distance must satisfy the policy minimum.
        if stop_distance < self.minimum_stop_distance:
            return self._blocked_result(
                balance=balance_value,
                equity=equity_value,
                risk_percent=risk_value,
                entry_price=entry_value,
                stop_loss=stop_value,
                tick_size=tick_size,
                tick_value=tick_value,
                contract_size=contract_size,
                volume_step=volume_step,
                broker_min_position_size=broker_min_position_size,
                broker_max_position_size=broker_max_position_size,
                spread=spread,
                commission=commission,
                slippage=slippage,
                daily_loss=daily_loss,
                drawdown_percent=drawdown_percent,
                open_trades=open_trades,
                consecutive_losses=consecutive_losses,
                session_allowed=session_allowed,
                news_allowed=news_allowed,
                correlation_allowed=correlation_allowed,
                cooldown_active=cooldown_active,
                reason_type=RiskReasonType.INVALID_STOP_DISTANCE,
                message="Stop distance is below the configured minimum.",
            )

        # Validate tick size.
        tick_size_value = self._validate_positive(
            tick_size,
            "tick_size",
        )

        # Validate tick value.
        tick_value_value = self._validate_positive(
            tick_value,
            "tick_value",
        )

        # Validate contract size.
        contract_size_value = self._validate_positive(
            contract_size,
            "contract_size",
        )

        # Validate broker volume step.
        volume_step_value = self._validate_positive(
            volume_step,
            "volume_step",
        )

        # Validate broker minimum volume.
        broker_min_value = self._validate_positive(
            broker_min_position_size,
            "broker_min_position_size",
        )

        # Validate broker maximum volume.
        broker_max_value = self._validate_positive(
            broker_max_position_size,
            "broker_max_position_size",
        )

        # Broker metadata with minimum > maximum is invalid configuration.
        if broker_min_value > broker_max_value:
            raise RiskIntelligenceError(
                "broker_min_position_size cannot exceed "
                "broker_max_position_size."
            )

        # Validate spread.
        spread_value = self._validate_non_negative(
            spread,
            "spread",
        )

        # Validate commission.
        commission_value = self._validate_non_negative(
            commission,
            "commission",
        )

        # Validate slippage.
        slippage_value = self._validate_non_negative(
            slippage,
            "slippage",
        )

        # Validate daily loss.
        daily_loss_value = self._validate_non_negative(
            daily_loss,
            "daily_loss",
        )

        # Validate drawdown.
        drawdown_value = self._validate_non_negative(
            drawdown_percent,
            "drawdown_percent",
        )

        # Validate open trade count.
        open_trades_value = self._validate_non_negative_int(
            open_trades,
            "open_trades",
        )

        # Validate consecutive losses.
        consecutive_losses_value = self._validate_non_negative_int(
            consecutive_losses,
            "consecutive_losses",
        )

        # Validate boolean risk controls.
        session_allowed = self._validate_bool(
            session_allowed,
            "session_allowed",
        )

        news_allowed = self._validate_bool(
            news_allowed,
            "news_allowed",
        )

        correlation_allowed = self._validate_bool(
            correlation_allowed,
            "correlation_allowed",
        )

        cooldown_active = self._validate_bool(
            cooldown_active,
            "cooldown_active",
        )

        # Calculate maximum monetary risk.
        risk_amount = balance_value * (
            risk_value / 100.0
        )

        if risk_amount <= 0.0 or not math.isfinite(risk_amount):
            return self._blocked_result(
                balance=balance_value,
                equity=equity_value,
                risk_percent=risk_value,
                entry_price=entry_value,
                stop_loss=stop_value,
                tick_size=tick_size_value,
                tick_value=tick_value_value,
                contract_size=contract_size_value,
                volume_step=volume_step_value,
                broker_min_position_size=broker_min_value,
                broker_max_position_size=broker_max_value,
                spread=spread_value,
                commission=commission_value,
                slippage=slippage_value,
                daily_loss=daily_loss_value,
                drawdown_percent=drawdown_value,
                open_trades=open_trades_value,
                consecutive_losses=consecutive_losses_value,
                session_allowed=session_allowed,
                news_allowed=news_allowed,
                correlation_allowed=correlation_allowed,
                cooldown_active=cooldown_active,
                reason_type=RiskReasonType.RISK_AMOUNT_TOO_HIGH,
                message="Calculated risk amount is invalid.",
            )

        # Calculate monetary risk per one position unit.
        risk_per_unit = (
            (stop_distance / tick_size_value)
            * tick_value_value
            * contract_size_value
        )

        if risk_per_unit <= 0.0 or not math.isfinite(
            risk_per_unit
        ):
            return self._blocked_result(
                balance=balance_value,
                equity=equity_value,
                risk_percent=risk_value,
                entry_price=entry_value,
                stop_loss=stop_value,
                tick_size=tick_size_value,
                tick_value=tick_value_value,
                contract_size=contract_size_value,
                volume_step=volume_step_value,
                broker_min_position_size=broker_min_value,
                broker_max_position_size=broker_max_value,
                spread=spread_value,
                commission=commission_value,
                slippage=slippage_value,
                daily_loss=daily_loss_value,
                drawdown_percent=drawdown_value,
                open_trades=open_trades_value,
                consecutive_losses=consecutive_losses_value,
                session_allowed=session_allowed,
                news_allowed=news_allowed,
                correlation_allowed=correlation_allowed,
                cooldown_active=cooldown_active,
                reason_type=RiskReasonType.INVALID_STOP_DISTANCE,
                message="Risk per position unit is invalid.",
            )

        # Calculate theoretical position size before broker rounding.
        raw_position_size = (
            risk_amount / risk_per_unit
        )

        if raw_position_size <= 0.0 or not math.isfinite(
            raw_position_size
        ):
            return self._blocked_result(
                balance=balance_value,
                equity=equity_value,
                risk_percent=risk_value,
                entry_price=entry_value,
                stop_loss=stop_value,
                tick_size=tick_size_value,
                tick_value=tick_value_value,
                contract_size=contract_size_value,
                volume_step=volume_step_value,
                broker_min_position_size=broker_min_value,
                broker_max_position_size=broker_max_value,
                spread=spread_value,
                commission=commission_value,
                slippage=slippage_value,
                daily_loss=daily_loss_value,
                drawdown_percent=drawdown_value,
                open_trades=open_trades_value,
                consecutive_losses=consecutive_losses_value,
                session_allowed=session_allowed,
                news_allowed=news_allowed,
                correlation_allowed=correlation_allowed,
                cooldown_active=cooldown_active,
                reason_type=RiskReasonType.POSITION_SIZE_ROUNDED_TO_ZERO,
                message="Calculated position size is invalid.",
                stop_distance=stop_distance,
                risk_amount=risk_amount,
                raw_position_size=raw_position_size,
            )

        # Round DOWN to the broker's volume step.
        final_position_size = self._floor_to_step(
            raw_position_size,
            volume_step_value,
        )

        # If rounding produces zero, the trade is not executable.
        if final_position_size <= 0.0:
            return self._blocked_result(
                balance=balance_value,
                equity=equity_value,
                risk_percent=risk_value,
                entry_price=entry_value,
                stop_loss=stop_value,
                tick_size=tick_size_value,
                tick_value=tick_value_value,
                contract_size=contract_size_value,
                volume_step=volume_step_value,
                broker_min_position_size=broker_min_value,
                broker_max_position_size=broker_max_value,
                spread=spread_value,
                commission=commission_value,
                slippage=slippage_value,
                daily_loss=daily_loss_value,
                drawdown_percent=drawdown_value,
                open_trades=open_trades_value,
                consecutive_losses=consecutive_losses_value,
                session_allowed=session_allowed,
                news_allowed=news_allowed,
                correlation_allowed=correlation_allowed,
                cooldown_active=cooldown_active,
                reason_type=RiskReasonType.POSITION_SIZE_ROUNDED_TO_ZERO,
                message=(
                    "Position size became zero after volume-step "
                    "rounding."
                ),
                stop_distance=stop_distance,
                risk_amount=risk_amount,
                raw_position_size=raw_position_size,
            )

        # Internal minimum position size is a hard veto.
        #
        # Never increase the calculated position size to satisfy this
        # minimum because that could increase monetary risk.
        if final_position_size < self.minimum_position_size:
            return self._blocked_result(
                balance=balance_value,
                equity=equity_value,
                risk_percent=risk_value,
                entry_price=entry_value,
                stop_loss=stop_value,
                tick_size=tick_size_value,
                tick_value=tick_value_value,
                contract_size=contract_size_value,
                volume_step=volume_step_value,
                broker_min_position_size=broker_min_value,
                broker_max_position_size=broker_max_value,
                spread=spread_value,
                commission=commission_value,
                slippage=slippage_value,
                daily_loss=daily_loss_value,
                drawdown_percent=drawdown_value,
                open_trades=open_trades_value,
                consecutive_losses=consecutive_losses_value,
                session_allowed=session_allowed,
                news_allowed=news_allowed,
                correlation_allowed=correlation_allowed,
                cooldown_active=cooldown_active,
                reason_type=RiskReasonType.POSITION_SIZE_TOO_SMALL,
                message=(
                    "Calculated position size is below the configured "
                    "minimum position size."
                ),
                stop_distance=stop_distance,
                risk_amount=risk_amount,
                raw_position_size=raw_position_size,
            )

        # Broker minimum is also a hard veto.
        #
        # Never increase volume to broker minimum because that can
        # exceed the requested monetary risk.
        if final_position_size < broker_min_value:
            return self._blocked_result(
                balance=balance_value,
                equity=equity_value,
                risk_percent=risk_value,
                entry_price=entry_value,
                stop_loss=stop_value,
                tick_size=tick_size_value,
                tick_value=tick_value_value,
                contract_size=contract_size_value,
                volume_step=volume_step_value,
                broker_min_position_size=broker_min_value,
                broker_max_position_size=broker_max_value,
                spread=spread_value,
                commission=commission_value,
                slippage=slippage_value,
                daily_loss=daily_loss_value,
                drawdown_percent=drawdown_value,
                open_trades=open_trades_value,
                consecutive_losses=consecutive_losses_value,
                session_allowed=session_allowed,
                news_allowed=news_allowed,
                correlation_allowed=correlation_allowed,
                cooldown_active=cooldown_active,
                reason_type=RiskReasonType.POSITION_SIZE_TOO_SMALL,
                message=(
                    "Calculated position size is below the broker's "
                    "minimum position size."
                ),
                stop_distance=stop_distance,
                risk_amount=risk_amount,
                raw_position_size=raw_position_size,
            )

        # The effective maximum is the stricter of the internal policy
        # maximum and broker maximum.
        effective_maximum_position = min(
            self.maximum_position_size,
            broker_max_value,
        )

        if final_position_size > effective_maximum_position:
            return self._blocked_result(
                balance=balance_value,
                equity=equity_value,
                risk_percent=risk_value,
                entry_price=entry_value,
                stop_loss=stop_value,
                tick_size=tick_size_value,
                tick_value=tick_value_value,
                contract_size=contract_size_value,
                volume_step=volume_step_value,
                broker_min_position_size=broker_min_value,
                broker_max_position_size=broker_max_value,
                spread=spread_value,
                commission=commission_value,
                slippage=slippage_value,
                daily_loss=daily_loss_value,
                drawdown_percent=drawdown_value,
                open_trades=open_trades_value,
                consecutive_losses=consecutive_losses_value,
                session_allowed=session_allowed,
                news_allowed=news_allowed,
                correlation_allowed=correlation_allowed,
                cooldown_active=cooldown_active,
                reason_type=RiskReasonType.POSITION_SIZE_TOO_LARGE,
                message=(
                    "Calculated position size exceeds the configured "
                    "position-size maximum."
                ),
                stop_distance=stop_distance,
                risk_amount=risk_amount,
                raw_position_size=raw_position_size,
            )

        # Recalculate actual monetary risk from the final executable
        # position size.
        actual_risk = (
            final_position_size * risk_per_unit
        )

        # The final executable size must never exceed the requested risk.
        risk_tolerance = max(
            1e-9,
            risk_amount * 1e-12,
        )

        if actual_risk > risk_amount + risk_tolerance:
            return self._blocked_result(
                balance=balance_value,
                equity=equity_value,
                risk_percent=risk_value,
                entry_price=entry_value,
                stop_loss=stop_value,
                tick_size=tick_size_value,
                tick_value=tick_value_value,
                contract_size=contract_size_value,
                volume_step=volume_step_value,
                broker_min_position_size=broker_min_value,
                broker_max_position_size=broker_max_value,
                spread=spread_value,
                commission=commission_value,
                slippage=slippage_value,
                daily_loss=daily_loss_value,
                drawdown_percent=drawdown_value,
                open_trades=open_trades_value,
                consecutive_losses=consecutive_losses_value,
                session_allowed=session_allowed,
                news_allowed=news_allowed,
                correlation_allowed=correlation_allowed,
                cooldown_active=cooldown_active,
                reason_type=RiskReasonType.POSITION_SIZE_EXCEEDS_RISK,
                message=(
                    "Final position size would exceed the requested "
                    "monetary risk."
                ),
                stop_distance=stop_distance,
                risk_amount=risk_amount,
                raw_position_size=raw_position_size,
            )

        # Spread veto.
        if spread_value > self.maximum_spread:
            return self._blocked_result(
                balance=balance_value,
                equity=equity_value,
                risk_percent=risk_value,
                entry_price=entry_value,
                stop_loss=stop_value,
                tick_size=tick_size_value,
                tick_value=tick_value_value,
                contract_size=contract_size_value,
                volume_step=volume_step_value,
                broker_min_position_size=broker_min_value,
                broker_max_position_size=broker_max_value,
                spread=spread_value,
                commission=commission_value,
                slippage=slippage_value,
                daily_loss=daily_loss_value,
                drawdown_percent=drawdown_value,
                open_trades=open_trades_value,
                consecutive_losses=consecutive_losses_value,
                session_allowed=session_allowed,
                news_allowed=news_allowed,
                correlation_allowed=correlation_allowed,
                cooldown_active=cooldown_active,
                reason_type=RiskReasonType.SPREAD_TOO_HIGH,
                message="Spread exceeds the configured maximum.",
                stop_distance=stop_distance,
                risk_amount=risk_amount,
                raw_position_size=raw_position_size,
            )

        # Commission veto.
        if commission_value > self.maximum_commission:
            return self._blocked_result(
                balance=balance_value,
                equity=equity_value,
                risk_percent=risk_value,
                entry_price=entry_value,
                stop_loss=stop_value,
                tick_size=tick_size_value,
                tick_value=tick_value_value,
                contract_size=contract_size_value,
                volume_step=volume_step_value,
                broker_min_position_size=broker_min_value,
                broker_max_position_size=broker_max_value,
                spread=spread_value,
                commission=commission_value,
                slippage=slippage_value,
                daily_loss=daily_loss_value,
                drawdown_percent=drawdown_value,
                open_trades=open_trades_value,
                consecutive_losses=consecutive_losses_value,
                session_allowed=session_allowed,
                news_allowed=news_allowed,
                correlation_allowed=correlation_allowed,
                cooldown_active=cooldown_active,
                reason_type=RiskReasonType.COMMISSION_TOO_HIGH,
                message="Commission exceeds the configured maximum.",
                stop_distance=stop_distance,
                risk_amount=risk_amount,
                raw_position_size=raw_position_size,
            )

        # Slippage veto.
        if slippage_value > self.maximum_slippage:
            return self._blocked_result(
                balance=balance_value,
                equity=equity_value,
                risk_percent=risk_value,
                entry_price=entry_value,
                stop_loss=stop_value,
                tick_size=tick_size_value,
                tick_value=tick_value_value,
                contract_size=contract_size_value,
                volume_step=volume_step_value,
                broker_min_position_size=broker_min_value,
                broker_max_position_size=broker_max_value,
                spread=spread_value,
                commission=commission_value,
                slippage=slippage_value,
                daily_loss=daily_loss_value,
                drawdown_percent=drawdown_value,
                open_trades=open_trades_value,
                consecutive_losses=consecutive_losses_value,
                session_allowed=session_allowed,
                news_allowed=news_allowed,
                correlation_allowed=correlation_allowed,
                cooldown_active=cooldown_active,
                reason_type=RiskReasonType.SLIPPAGE_TOO_HIGH,
                message="Slippage exceeds the configured maximum.",
                stop_distance=stop_distance,
                risk_amount=risk_amount,
                raw_position_size=raw_position_size,
            )

        # Daily loss reaches the configured limit -> BLOCK.
        if daily_loss_value >= self.maximum_daily_loss:
            return self._blocked_result(
                balance=balance_value,
                equity=equity_value,
                risk_percent=risk_value,
                entry_price=entry_value,
                stop_loss=stop_value,
                tick_size=tick_size_value,
                tick_value=tick_value_value,
                contract_size=contract_size_value,
                volume_step=volume_step_value,
                broker_min_position_size=broker_min_value,
                broker_max_position_size=broker_max_value,
                spread=spread_value,
                commission=commission_value,
                slippage=slippage_value,
                daily_loss=daily_loss_value,
                drawdown_percent=drawdown_value,
                open_trades=open_trades_value,
                consecutive_losses=consecutive_losses_value,
                session_allowed=session_allowed,
                news_allowed=news_allowed,
                correlation_allowed=correlation_allowed,
                cooldown_active=cooldown_active,
                reason_type=RiskReasonType.DAILY_LOSS_LIMIT,
                message="Daily loss has reached the configured maximum.",
                stop_distance=stop_distance,
                risk_amount=risk_amount,
                raw_position_size=raw_position_size,
            )

        # Drawdown reaching the configured limit -> BLOCK.
        #
        # Equality is intentionally blocked.
        if drawdown_value >= self.maximum_drawdown_percent:
            return self._blocked_result(
                balance=balance_value,
                equity=equity_value,
                risk_percent=risk_value,
                entry_price=entry_value,
                stop_loss=stop_value,
                tick_size=tick_size_value,
                tick_value=tick_value_value,
                contract_size=contract_size_value,
                volume_step=volume_step_value,
                broker_min_position_size=broker_min_value,
                broker_max_position_size=broker_max_value,
                spread=spread_value,
                commission=commission_value,
                slippage=slippage_value,
                daily_loss=daily_loss_value,
                drawdown_percent=drawdown_value,
                open_trades=open_trades_value,
                consecutive_losses=consecutive_losses_value,
                session_allowed=session_allowed,
                news_allowed=news_allowed,
                correlation_allowed=correlation_allowed,
                cooldown_active=cooldown_active,
                reason_type=RiskReasonType.MAX_DRAWDOWN_LIMIT,
                message="Drawdown has reached the configured maximum.",
                stop_distance=stop_distance,
                risk_amount=risk_amount,
                raw_position_size=raw_position_size,
            )

        # Maximum open trades veto.
        if open_trades_value >= self.maximum_open_trades:
            return self._blocked_result(
                balance=balance_value,
                equity=equity_value,
                risk_percent=risk_value,
                entry_price=entry_value,
                stop_loss=stop_value,
                tick_size=tick_size_value,
                tick_value=tick_value_value,
                contract_size=contract_size_value,
                volume_step=volume_step_value,
                broker_min_position_size=broker_min_value,
                broker_max_position_size=broker_max_value,
                spread=spread_value,
                commission=commission_value,
                slippage=slippage_value,
                daily_loss=daily_loss_value,
                drawdown_percent=drawdown_value,
                open_trades=open_trades_value,
                consecutive_losses=consecutive_losses_value,
                session_allowed=session_allowed,
                news_allowed=news_allowed,
                correlation_allowed=correlation_allowed,
                cooldown_active=cooldown_active,
                reason_type=RiskReasonType.MAX_OPEN_TRADES,
                message="Maximum number of open trades has been reached.",
                stop_distance=stop_distance,
                risk_amount=risk_amount,
                raw_position_size=raw_position_size,
            )

        # Consecutive-loss veto.
        if consecutive_losses_value >= self.maximum_consecutive_losses:
            return self._blocked_result(
                balance=balance_value,
                equity=equity_value,
                risk_percent=risk_value,
                entry_price=entry_value,
                stop_loss=stop_value,
                tick_size=tick_size_value,
                tick_value=tick_value_value,
                contract_size=contract_size_value,
                volume_step=volume_step_value,
                broker_min_position_size=broker_min_value,
                broker_max_position_size=broker_max_value,
                spread=spread_value,
                commission=commission_value,
                slippage=slippage_value,
                daily_loss=daily_loss_value,
                drawdown_percent=drawdown_value,
                open_trades=open_trades_value,
                consecutive_losses=consecutive_losses_value,
                session_allowed=session_allowed,
                news_allowed=news_allowed,
                correlation_allowed=correlation_allowed,
                cooldown_active=cooldown_active,
                reason_type=RiskReasonType.CONSECUTIVE_LOSS_LIMIT,
                message=(
                    "Maximum consecutive-loss limit has been reached."
                ),
                stop_distance=stop_distance,
                risk_amount=risk_amount,
                raw_position_size=raw_position_size,
            )

        # Cooldown veto.
        if cooldown_active:
            return self._blocked_result(
                balance=balance_value,
                equity=equity_value,
                risk_percent=risk_value,
                entry_price=entry_value,
                stop_loss=stop_value,
                tick_size=tick_size_value,
                tick_value=tick_value_value,
                contract_size=contract_size_value,
                volume_step=volume_step_value,
                broker_min_position_size=broker_min_value,
                broker_max_position_size=broker_max_value,
                spread=spread_value,
                commission=commission_value,
                slippage=slippage_value,
                daily_loss=daily_loss_value,
                drawdown_percent=drawdown_value,
                open_trades=open_trades_value,
                consecutive_losses=consecutive_losses_value,
                session_allowed=session_allowed,
                news_allowed=news_allowed,
                correlation_allowed=correlation_allowed,
                cooldown_active=cooldown_active,
                reason_type=RiskReasonType.COOLDOWN_ACTIVE,
                message="Trading cooldown is currently active.",
                stop_distance=stop_distance,
                risk_amount=risk_amount,
                raw_position_size=raw_position_size,
            )

        # Session veto.
        if not session_allowed:
            return self._blocked_result(
                balance=balance_value,
                equity=equity_value,
                risk_percent=risk_value,
                entry_price=entry_value,
                stop_loss=stop_value,
                tick_size=tick_size_value,
                tick_value=tick_value_value,
                contract_size=contract_size_value,
                volume_step=volume_step_value,
                broker_min_position_size=broker_min_value,
                broker_max_position_size=broker_max_value,
                spread=spread_value,
                commission=commission_value,
                slippage=slippage_value,
                daily_loss=daily_loss_value,
                drawdown_percent=drawdown_value,
                open_trades=open_trades_value,
                consecutive_losses=consecutive_losses_value,
                session_allowed=session_allowed,
                news_allowed=news_allowed,
                correlation_allowed=correlation_allowed,
                cooldown_active=cooldown_active,
                reason_type=RiskReasonType.SESSION_BLOCKED,
                message="Trading session is currently blocked.",
                stop_distance=stop_distance,
                risk_amount=risk_amount,
                raw_position_size=raw_position_size,
            )

        # News veto.
        if not news_allowed:
            return self._blocked_result(
                balance=balance_value,
                equity=equity_value,
                risk_percent=risk_value,
                entry_price=entry_value,
                stop_loss=stop_value,
                tick_size=tick_size_value,
                tick_value=tick_value_value,
                contract_size=contract_size_value,
                volume_step=volume_step_value,
                broker_min_position_size=broker_min_value,
                broker_max_position_size=broker_max_value,
                spread=spread_value,
                commission=commission_value,
                slippage=slippage_value,
                daily_loss=daily_loss_value,
                drawdown_percent=drawdown_value,
                open_trades=open_trades_value,
                consecutive_losses=consecutive_losses_value,
                session_allowed=session_allowed,
                news_allowed=news_allowed,
                correlation_allowed=correlation_allowed,
                cooldown_active=cooldown_active,
                reason_type=RiskReasonType.NEWS_BLOCKED,
                message="News filter currently blocks trading.",
                stop_distance=stop_distance,
                risk_amount=risk_amount,
                raw_position_size=raw_position_size,
            )

        # Correlation veto.
        if not correlation_allowed:
            return self._blocked_result(
                balance=balance_value,
                equity=equity_value,
                risk_percent=risk_value,
                entry_price=entry_value,
                stop_loss=stop_value,
                tick_size=tick_size_value,
                tick_value=tick_value_value,
                contract_size=contract_size_value,
                volume_step=volume_step_value,
                broker_min_position_size=broker_min_value,
                broker_max_position_size=broker_max_value,
                spread=spread_value,
                commission=commission_value,
                slippage=slippage_value,
                daily_loss=daily_loss_value,
                drawdown_percent=drawdown_value,
                open_trades=open_trades_value,
                consecutive_losses=consecutive_losses_value,
                session_allowed=session_allowed,
                news_allowed=news_allowed,
                correlation_allowed=correlation_allowed,
                cooldown_active=cooldown_active,
                reason_type=RiskReasonType.CORRELATION_BLOCKED,
                message="Correlation filter currently blocks trading.",
                stop_distance=stop_distance,
                risk_amount=risk_amount,
                raw_position_size=raw_position_size,
            )

        # All risk controls passed.
        return RiskModel(
            decision=RiskDecision.ALLOW,
            balance=balance_value,
            equity=equity_value,
            risk_percent=risk_value,
            risk_amount=risk_amount,
            entry_price=entry_value,
            stop_loss=stop_value,
            stop_distance=stop_distance,
            position_size=final_position_size,
            raw_position_size=raw_position_size,
            tick_size=tick_size_value,
            tick_value=tick_value_value,
            contract_size=contract_size_value,
            volume_step=volume_step_value,
            broker_min_position_size=broker_min_value,
            broker_max_position_size=broker_max_value,
            spread=spread_value,
            commission=commission_value,
            slippage=slippage_value,
            daily_loss=daily_loss_value,
            drawdown_percent=drawdown_value,
            open_trades=open_trades_value,
            consecutive_losses=consecutive_losses_value,
            session_allowed=session_allowed,
            news_allowed=news_allowed,
            correlation_allowed=correlation_allowed,
            cooldown_active=cooldown_active,
            reasons=(
                RiskReason(
                    reason_type=RiskReasonType.RISK_ALLOWED,
                    message="All configured risk controls passed.",
                ),
            ),
            warnings=(),
        )

    def analyze_xauusd(
        self,
        **kwargs: object,
    ) -> RiskModel:
        """Convenience wrapper for the XAU/USD risk pipeline."""

        return self.analyze(**kwargs)

    def _blocked_result(
        self,
        *,
        balance: float,
        equity: float,
        risk_percent: float,
        entry_price: float,
        stop_loss: float,
        tick_size: float,
        tick_value: float,
        contract_size: float,
        volume_step: float,
        broker_min_position_size: float,
        broker_max_position_size: float,
        spread: float,
        commission: float,
        slippage: float,
        daily_loss: float,
        drawdown_percent: float,
        open_trades: int,
        consecutive_losses: int,
        session_allowed: bool,
        news_allowed: bool,
        correlation_allowed: bool,
        cooldown_active: bool,
        reason_type: RiskReasonType,
        message: str,
        raw_position_size: float = 0.0,
        position_size: float = 0.0,
        stop_distance: Optional[float] = None,
        risk_amount: Optional[float] = None,
    ) -> RiskModel:
        """
        Build a blocked result.

        IMPORTANT:

        position_size is ALWAYS zero for a blocked trade.

        raw_position_size is preserved for audit/debugging, but it is
        never exposed as an executable size.
        """

        calculated_stop_distance = (
            abs(entry_price - stop_loss)
            if stop_distance is None
            else stop_distance
        )

        calculated_risk_amount = (
            balance * (risk_percent / 100.0)
            if risk_amount is None
            else risk_amount
        )

        return RiskModel(
            decision=RiskDecision.BLOCK,
            balance=balance,
            equity=equity,
            risk_percent=risk_percent,
            risk_amount=calculated_risk_amount,
            entry_price=entry_price,
            stop_loss=stop_loss,
            stop_distance=calculated_stop_distance,

            # A blocked trade must never expose an executable volume.
            position_size=0.0,

            # Keep the theoretical calculation for auditability.
            raw_position_size=raw_position_size,

            tick_size=tick_size,
            tick_value=tick_value,
            contract_size=contract_size,
            volume_step=volume_step,
            broker_min_position_size=broker_min_position_size,
            broker_max_position_size=broker_max_position_size,
            spread=spread,
            commission=commission,
            slippage=slippage,
            daily_loss=daily_loss,
            drawdown_percent=drawdown_percent,
            open_trades=open_trades,
            consecutive_losses=consecutive_losses,
            session_allowed=session_allowed,
            news_allowed=news_allowed,
            correlation_allowed=correlation_allowed,
            cooldown_active=cooldown_active,
            reasons=(
                RiskReason(
                    reason_type=reason_type,
                    message=message,
                ),
            ),
            warnings=(),
        )

    @staticmethod
    def _floor_to_step(
        value: float,
        step: float,
    ) -> float:
        """Round a position size DOWN to the broker volume step."""

        if value <= 0.0:
            return 0.0

        if step <= 0.0:
            raise RiskIntelligenceError(
                "step must be greater than zero."
            )

        units = math.floor(
            (value / step) + 1e-12,
        )

        rounded_value = units * step

        return round(
            rounded_value,
            10,
        )

    @staticmethod
    def _validate_positive(
        value: float,
        name: str,
    ) -> float:
        """Validate a strictly positive finite number."""

        # bool is a subclass of int in Python, so explicitly reject it.
        if isinstance(value, bool):
            raise RiskIntelligenceError(
                f"{name} must be a positive finite number."
            )

        if not isinstance(value, (int, float)):
            raise RiskIntelligenceError(
                f"{name} must be a positive finite number."
            )

        numeric_value = float(value)

        # Reject NaN and infinity.
        if not math.isfinite(numeric_value):
            raise RiskIntelligenceError(
                f"{name} must be a positive finite number."
            )

        # Financial quantities requiring positivity cannot be zero
        # or negative.
        if numeric_value <= 0.0:
            raise RiskIntelligenceError(
                f"{name} must be greater than zero."
            )

        return numeric_value

    @staticmethod
    def _validate_non_negative(
        value: float,
        name: str,
    ) -> float:
        """Validate a finite number that may equal zero."""

        if isinstance(value, bool):
            raise RiskIntelligenceError(
                f"{name} must be a non-negative finite number."
            )

        if not isinstance(value, (int, float)):
            raise RiskIntelligenceError(
                f"{name} must be a non-negative finite number."
            )

        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            raise RiskIntelligenceError(
                f"{name} must be a non-negative finite number."
            )

        if numeric_value < 0.0:
            raise RiskIntelligenceError(
                f"{name} cannot be negative."
            )

        return numeric_value

    @staticmethod
    def _validate_non_negative_int(
        value: int,
        name: str,
    ) -> int:
        """Validate a non-negative integer."""

        if isinstance(value, bool):
            raise RiskIntelligenceError(
                f"{name} must be a non-negative integer."
            )

        if not isinstance(value, int):
            raise RiskIntelligenceError(
                f"{name} must be a non-negative integer."
            )

        if value < 0:
            raise RiskIntelligenceError(
                f"{name} cannot be negative."
            )

        return value

    @staticmethod
    def _validate_bool(
        value: bool,
        name: str,
    ) -> bool:
        """Validate a boolean value."""

        if not isinstance(value, bool):
            raise RiskIntelligenceError(
                f"{name} must be a boolean."
            )

        return value