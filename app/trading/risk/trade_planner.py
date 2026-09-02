from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from app.trading.candidate.trade_candidate import (
    CandidateDecision,
    TradeCandidate,
)
from app.trading.setup.setup_engine import SetupDirection


class RiskPlanningError(ValueError):
    """Raised when a trade plan cannot be created safely."""


@dataclass(frozen=True, slots=True)
class TradePlan:
    """
    Deterministic trade plan generated from a confirmed trade candidate.

    All prices are expressed in the instrument's native price units.

    Position sizing uses:

        position_size =
            risk_amount / (stop_distance * value_per_price_unit)

    This keeps the engine independent from broker-specific contract
    specifications.
    """

    timestamp: object
    symbol: str
    timeframe: str

    direction: SetupDirection

    entry_price: float
    stop_loss: float
    take_profit: float

    risk_distance: float
    reward_distance: float
    risk_reward_ratio: float

    account_balance: float
    risk_percent: float
    risk_amount: float

    value_per_price_unit: float
    position_size: float

    maximum_risk_amount: float

    valid: bool
    warnings: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return self.valid


class TradePlanningEngine:
    """
    Converts a trade-ready candidate into a deterministic trade plan.

    The engine does not decide whether a market setup exists.
    It only handles trade-level risk and price planning.

    The engine is intentionally broker/instrument agnostic.
    `value_per_price_unit` must be supplied by the caller according
    to the instrument and broker contract specification.
    """

    DEFAULT_RISK_PERCENT = 1.0
    DEFAULT_MINIMUM_RR = 2.0
    DEFAULT_MAXIMUM_RISK_PERCENT = 2.0
    DEFAULT_PRICE_TOLERANCE = 1e-12

    def __init__(
        self,
        minimum_risk_reward: float = DEFAULT_MINIMUM_RR,
        maximum_risk_percent: float = DEFAULT_MAXIMUM_RISK_PERCENT,
        price_tolerance: float = DEFAULT_PRICE_TOLERANCE,
    ) -> None:
        self._validate_positive(
            minimum_risk_reward,
            "minimum_risk_reward",
        )

        self._validate_positive(
            maximum_risk_percent,
            "maximum_risk_percent",
        )

        self._validate_positive(
            price_tolerance,
            "price_tolerance",
        )

        self.minimum_risk_reward = float(minimum_risk_reward)
        self.maximum_risk_percent = float(maximum_risk_percent)
        self.price_tolerance = float(price_tolerance)

    def plan(
        self,
        candidate: TradeCandidate,
        *,
        account_balance: float,
        risk_percent: float = DEFAULT_RISK_PERCENT,
        stop_loss: float,
        take_profit: float,
        value_per_price_unit: float,
    ) -> TradePlan:
        """
        Create a deterministic trade plan.

        The candidate must be TRADE_READY.

        Parameters
        ----------
        candidate:
            Confirmed trade candidate.

        account_balance:
            Current account/equity amount.

        risk_percent:
            Percentage of account balance to risk.

        stop_loss:
            Explicit stop-loss price.

        take_profit:
            Explicit take-profit price.

        value_per_price_unit:
            Monetary value of one price unit for one unit of position size.
        """

        self._validate_candidate(candidate)

        self._validate_positive(
            account_balance,
            "account_balance",
        )

        self._validate_positive(
            risk_percent,
            "risk_percent",
        )

        self._validate_price(
            stop_loss,
            "stop_loss",
        )

        self._validate_price(
            take_profit,
            "take_profit",
        )

        self._validate_positive(
            value_per_price_unit,
            "value_per_price_unit",
        )

        if risk_percent > self.maximum_risk_percent:
            raise RiskPlanningError(
                "risk_percent exceeds maximum allowed risk_percent"
            )

        entry_price = float(candidate.close)

        direction = candidate.direction

        if direction not in (
            SetupDirection.LONG,
            SetupDirection.SHORT,
        ):
            raise RiskPlanningError(
                "trade candidate must have LONG or SHORT direction"
            )

        self._validate_price_relationship(
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        risk_distance = abs(entry_price - stop_loss)
        reward_distance = abs(take_profit - entry_price)

        if risk_distance <= self.price_tolerance:
            raise RiskPlanningError(
                "stop-loss distance must be greater than zero"
            )

        if reward_distance <= self.price_tolerance:
            raise RiskPlanningError(
                "take-profit distance must be greater than zero"
            )

        risk_reward_ratio = reward_distance / risk_distance

        if risk_reward_ratio + self.price_tolerance < self.minimum_risk_reward:
            raise RiskPlanningError(
                "risk/reward ratio is below minimum allowed value"
            )

        risk_amount = account_balance * (risk_percent / 100.0)

        maximum_risk_amount = (
            account_balance * (self.maximum_risk_percent / 100.0)
        )

        if risk_amount > maximum_risk_amount + self.price_tolerance:
            raise RiskPlanningError(
                "calculated risk amount exceeds maximum allowed risk"
            )

        position_size = risk_amount / (
            risk_distance * value_per_price_unit
        )

        if not isfinite(position_size) or position_size <= 0:
            raise RiskPlanningError(
                "calculated position size must be finite and greater than zero"
            )

        warnings: list[str] = []

        if risk_percent >= self.maximum_risk_percent:
            warnings.append(
                "risk_percent is at the configured maximum"
            )

        if (
            risk_reward_ratio
            < self.minimum_risk_reward + 0.5
        ):
            warnings.append(
                "risk/reward ratio is close to the configured minimum"
            )

        return TradePlan(
            timestamp=candidate.timestamp,
            symbol=candidate.symbol,
            timeframe=candidate.timeframe,
            direction=direction,
            entry_price=entry_price,
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            risk_distance=risk_distance,
            reward_distance=reward_distance,
            risk_reward_ratio=risk_reward_ratio,
            account_balance=float(account_balance),
            risk_percent=float(risk_percent),
            risk_amount=risk_amount,
            value_per_price_unit=float(value_per_price_unit),
            position_size=position_size,
            maximum_risk_amount=maximum_risk_amount,
            valid=True,
            warnings=tuple(warnings),
        )

    def plan_with_distances(
        self,
        candidate: TradeCandidate,
        *,
        account_balance: float,
        risk_percent: float = DEFAULT_RISK_PERCENT,
        stop_distance: float,
        reward_distance: float,
        value_per_price_unit: float,
    ) -> TradePlan:
        """
        Create a plan using distances rather than absolute SL/TP prices.

        LONG:
            SL = entry - stop_distance
            TP = entry + reward_distance

        SHORT:
            SL = entry + stop_distance
            TP = entry - reward_distance
        """

        self._validate_candidate(candidate)

        self._validate_positive(
            stop_distance,
            "stop_distance",
        )

        self._validate_positive(
            reward_distance,
            "reward_distance",
        )

        entry_price = float(candidate.close)

        if candidate.direction == SetupDirection.LONG:
            stop_loss = entry_price - float(stop_distance)
            take_profit = entry_price + float(reward_distance)

        elif candidate.direction == SetupDirection.SHORT:
            stop_loss = entry_price + float(stop_distance)
            take_profit = entry_price - float(reward_distance)

        else:
            raise RiskPlanningError(
                "trade candidate must have LONG or SHORT direction"
            )

        return self.plan(
            candidate,
            account_balance=account_balance,
            risk_percent=risk_percent,
            stop_loss=stop_loss,
            take_profit=take_profit,
            value_per_price_unit=value_per_price_unit,
        )

    def _validate_candidate(
        self,
        candidate: TradeCandidate,
    ) -> None:
        if not isinstance(candidate, TradeCandidate):
            raise RiskPlanningError(
                "candidate must be a TradeCandidate"
            )

        if candidate.decision != CandidateDecision.TRADE_READY:
            raise RiskPlanningError(
                "trade candidate must be TRADE_READY"
            )

        if candidate.invalidated:
            raise RiskPlanningError(
                "trade candidate is invalidated"
            )

        if not candidate.entry_ready:
            raise RiskPlanningError(
                "trade candidate is not entry-ready"
            )

        if candidate.direction not in (
            SetupDirection.LONG,
            SetupDirection.SHORT,
        ):
            raise RiskPlanningError(
                "trade candidate must have LONG or SHORT direction"
            )

    def _validate_price_relationship(
        self,
        *,
        direction: SetupDirection,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> None:
        if direction == SetupDirection.LONG:
            if stop_loss >= entry_price - self.price_tolerance:
                raise RiskPlanningError(
                    "LONG stop-loss must be below entry price"
                )

            if take_profit <= entry_price + self.price_tolerance:
                raise RiskPlanningError(
                    "LONG take-profit must be above entry price"
                )

        elif direction == SetupDirection.SHORT:
            if stop_loss <= entry_price + self.price_tolerance:
                raise RiskPlanningError(
                    "SHORT stop-loss must be above entry price"
                )

            if take_profit >= entry_price - self.price_tolerance:
                raise RiskPlanningError(
                    "SHORT take-profit must be below entry price"
                )

        else:
            raise RiskPlanningError(
                "unsupported trade direction"
            )

    @staticmethod
    def _validate_positive(
        value: float,
        name: str,
    ) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RiskPlanningError(
                f"{name} must be numeric"
            )

        if not isfinite(float(value)):
            raise RiskPlanningError(
                f"{name} must be finite"
            )

        if float(value) <= 0:
            raise RiskPlanningError(
                f"{name} must be greater than zero"
            )

    @staticmethod
    def _validate_price(
        value: float,
        name: str,
    ) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RiskPlanningError(
                f"{name} must be numeric"
            )

        if not isfinite(float(value)):
            raise RiskPlanningError(
                f"{name} must be finite"
            )

        if float(value) <= 0:
            raise RiskPlanningError(
                f"{name} must be greater than zero"
            )