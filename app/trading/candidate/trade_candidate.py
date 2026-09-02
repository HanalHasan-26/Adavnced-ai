from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from app.trading.setup.setup_engine import (
    SetupDirection,
    SetupEvaluation,
    SetupReasonType,
    SetupType,
)


class CandidateDecision(str, Enum):
    TRADE_READY = "TRADE_READY"
    WAIT = "WAIT"
    REJECT = "REJECT"


class CandidateReasonType(str, Enum):
    SETUP_VALID = "SETUP_VALID"
    SETUP_INVALID = "SETUP_INVALID"
    LONG_CONFIRMED = "LONG_CONFIRMED"
    SHORT_CONFIRMED = "SHORT_CONFIRMED"
    STRUCTURE_CONFIRMED = "STRUCTURE_CONFIRMED"
    MOMENTUM_CONFIRMED = "MOMENTUM_CONFIRMED"
    PRICE_CONFIRMED = "PRICE_CONFIRMED"
    TREND_CONFIRMED = "TREND_CONFIRMED"
    RANGE_CONFIRMED = "RANGE_CONFIRMED"
    CONFLICT_PRESENT = "CONFLICT_PRESENT"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    LOW_QUALITY = "LOW_QUALITY"
    INSUFFICIENT_CONFIRMATION = "INSUFFICIENT_CONFIRMATION"
    NO_CLEAR_DIRECTION = "NO_CLEAR_DIRECTION"
    REVERSAL_REQUIRES_CONFIRMATION = "REVERSAL_REQUIRES_CONFIRMATION"
    INVALID_SETUP_TYPE = "INVALID_SETUP_TYPE"


@dataclass(frozen=True, slots=True)
class CandidateReason:
    reason_type: CandidateReasonType
    message: str


@dataclass(frozen=True, slots=True)
class TradeCandidate:
    timestamp: object
    symbol: str
    timeframe: str
    close: float

    decision: CandidateDecision
    direction: SetupDirection
    setup_type: SetupType

    setup_quality_score: float
    confirmation_score: float

    structure_confirmed: bool
    momentum_confirmed: bool
    price_confirmed: bool
    trend_confirmed: bool

    supporting_signals: tuple
    conflicting_signals: tuple

    reasons: tuple[CandidateReason, ...]
    warnings: tuple[str, ...]

    entry_ready: bool
    invalidated: bool

    @property
    def is_trade_ready(self) -> bool:
        return self.decision == CandidateDecision.TRADE_READY


class TradeCandidateEngine:
    DEFAULT_MINIMUM_CONFIRMATION_SCORE = 65.0
    DEFAULT_TRADE_READY_SCORE = 75.0
    DEFAULT_MAX_CONFLICTS = 1

    def __init__(
        self,
        minimum_confirmation_score: float = DEFAULT_MINIMUM_CONFIRMATION_SCORE,
        trade_ready_score: float = DEFAULT_TRADE_READY_SCORE,
        max_conflicts: int = DEFAULT_MAX_CONFLICTS,
    ) -> None:
        if not isinstance(minimum_confirmation_score, (int, float)):
            raise TypeError("minimum_confirmation_score must be numeric")

        if minimum_confirmation_score < 0 or minimum_confirmation_score > 100:
            raise ValueError(
                "minimum_confirmation_score must be between 0 and 100"
            )

        if not isinstance(trade_ready_score, (int, float)):
            raise TypeError("trade_ready_score must be numeric")

        if trade_ready_score < 0 or trade_ready_score > 100:
            raise ValueError("trade_ready_score must be between 0 and 100")

        if trade_ready_score < minimum_confirmation_score:
            raise ValueError(
                "trade_ready_score must be greater than or equal to "
                "minimum_confirmation_score"
            )

        if not isinstance(max_conflicts, int):
            raise TypeError("max_conflicts must be an integer")

        if max_conflicts < 0:
            raise ValueError("max_conflicts must be non-negative")

        self.minimum_confirmation_score = float(minimum_confirmation_score)
        self.trade_ready_score = float(trade_ready_score)
        self.max_conflicts = max_conflicts

    def evaluate(self, setup: SetupEvaluation) -> TradeCandidate:
        self._validate_setup(setup)

        reasons: list[CandidateReason] = []
        warnings: list[str] = []

        supporting = tuple(setup.supporting_signals)
        conflicting = tuple(setup.conflicting_signals)

        structure_confirmed = self._structure_confirmed(setup)
        momentum_confirmed = self._momentum_confirmed(setup)
        price_confirmed = self._price_confirmed(setup)
        trend_confirmed = self._trend_confirmed(setup)

        if setup.valid:
            reasons.append(
                CandidateReason(
                    CandidateReasonType.SETUP_VALID,
                    "The setup passed the Step 8 setup evaluation.",
                )
            )
        else:
            reasons.append(
                CandidateReason(
                    CandidateReasonType.SETUP_INVALID,
                    "The setup was not valid during Step 8 evaluation.",
                )
            )

        if setup.direction == SetupDirection.LONG:
            reasons.append(
                CandidateReason(
                    CandidateReasonType.LONG_CONFIRMED,
                    "The setup direction is LONG.",
                )
            )
        elif setup.direction == SetupDirection.SHORT:
            reasons.append(
                CandidateReason(
                    CandidateReasonType.SHORT_CONFIRMED,
                    "The setup direction is SHORT.",
                )
            )
        else:
            reasons.append(
                CandidateReason(
                    CandidateReasonType.NO_CLEAR_DIRECTION,
                    "The setup does not have a clear trading direction.",
                )
            )

        if structure_confirmed:
            reasons.append(
                CandidateReason(
                    CandidateReasonType.STRUCTURE_CONFIRMED,
                    "Market structure supports the setup direction.",
                )
            )

        if momentum_confirmed:
            reasons.append(
                CandidateReason(
                    CandidateReasonType.MOMENTUM_CONFIRMED,
                    "Momentum signals support the setup direction.",
                )
            )

        if price_confirmed:
            reasons.append(
                CandidateReason(
                    CandidateReasonType.PRICE_CONFIRMED,
                    "Price-location signals support the setup direction.",
                )
            )

        if trend_confirmed:
            reasons.append(
                CandidateReason(
                    CandidateReasonType.TREND_CONFIRMED,
                    "The market condition is compatible with the setup.",
                )
            )

        if setup.setup_type == SetupType.RANGE:
            reasons.append(
                CandidateReason(
                    CandidateReasonType.RANGE_CONFIRMED,
                    "The setup is classified as a range setup.",
                )
            )

        if len(conflicting) > self.max_conflicts:
            reasons.append(
                CandidateReason(
                    CandidateReasonType.CONFLICT_PRESENT,
                    "The setup contains more conflicts than allowed.",
                )
            )
            warnings.append("Too many conflicting signals.")

        if not self._has_sufficient_history(setup):
            reasons.append(
                CandidateReason(
                    CandidateReasonType.INSUFFICIENT_HISTORY,
                    "There is not enough historical data for confirmation.",
                )
            )
            warnings.append("Insufficient historical context.")

        if setup.quality_score < self.minimum_confirmation_score:
            reasons.append(
                CandidateReason(
                    CandidateReasonType.LOW_QUALITY,
                    "The setup quality is below the confirmation threshold.",
                )
            )

        confirmation_score = self._calculate_confirmation_score(
            setup=setup,
            structure_confirmed=structure_confirmed,
            momentum_confirmed=momentum_confirmed,
            price_confirmed=price_confirmed,
            trend_confirmed=trend_confirmed,
        )

        if confirmation_score < self.minimum_confirmation_score:
            reasons.append(
                CandidateReason(
                    CandidateReasonType.INSUFFICIENT_CONFIRMATION,
                    "Confirmation strength is below the required threshold.",
                )
            )

        if (
            setup.setup_type == SetupType.REVERSAL
            and not structure_confirmed
        ):
            reasons.append(
                CandidateReason(
                    CandidateReasonType.REVERSAL_REQUIRES_CONFIRMATION,
                    "A reversal setup requires structural confirmation.",
                )
            )

        invalidated = self._is_invalidated(
            setup=setup,
            structure_confirmed=structure_confirmed,
            conflicting=conflicting,
        )

        if invalidated:
            warnings.append("The setup is invalidated for candidate generation.")

        decision = self._determine_decision(
            setup=setup,
            confirmation_score=confirmation_score,
            structure_confirmed=structure_confirmed,
            momentum_confirmed=momentum_confirmed,
            price_confirmed=price_confirmed,
            trend_confirmed=trend_confirmed,
            conflicting=conflicting,
            invalidated=invalidated,
        )

        entry_ready = decision == CandidateDecision.TRADE_READY

        return TradeCandidate(
            timestamp=setup.timestamp,
            symbol=setup.symbol,
            timeframe=setup.timeframe,
            close=setup.close,
            decision=decision,
            direction=setup.direction,
            setup_type=setup.setup_type,
            setup_quality_score=float(setup.quality_score),
            confirmation_score=confirmation_score,
            structure_confirmed=structure_confirmed,
            momentum_confirmed=momentum_confirmed,
            price_confirmed=price_confirmed,
            trend_confirmed=trend_confirmed,
            supporting_signals=supporting,
            conflicting_signals=conflicting,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
            entry_ready=entry_ready,
            invalidated=invalidated,
        )

    def evaluate_at(
        self,
        setups: Sequence[SetupEvaluation],
        index: int,
    ) -> TradeCandidate:
        if not isinstance(setups, Sequence):
            raise TypeError("setups must be a sequence")

        if not setups:
            raise ValueError("setups must not be empty")

        if not isinstance(index, int):
            raise TypeError("index must be an integer")

        if index < 0 or index >= len(setups):
            raise IndexError("index out of range")

        return self.evaluate(setups[index])

    @staticmethod
    def _validate_setup(setup: SetupEvaluation) -> None:
        if not isinstance(setup, SetupEvaluation):
            raise TypeError("setup must be a SetupEvaluation")

    @staticmethod
    def _has_sufficient_history(setup: SetupEvaluation) -> bool:
        return (
            SetupReasonType.INSUFFICIENT_HISTORY
            not in {
                reason.reason_type
                for reason in setup.reasons
            }
        )

    @staticmethod
    def _structure_confirmed(setup: SetupEvaluation) -> bool:
        return (
            setup.direction != SetupDirection.NONE
            and SetupReasonType.STRUCTURE_ALIGNMENT
            in {
                reason.reason_type
                for reason in setup.reasons
            }
        )

    @staticmethod
    def _momentum_confirmed(setup: SetupEvaluation) -> bool:
        return (
            SetupReasonType.MOMENTUM_ALIGNMENT
            in {
                reason.reason_type
                for reason in setup.reasons
            }
        )

    @staticmethod
    def _price_confirmed(setup: SetupEvaluation) -> bool:
        return (
            SetupReasonType.PRICE_ALIGNMENT
            in {
                reason.reason_type
                for reason in setup.reasons
            }
        )

    @staticmethod
    def _trend_confirmed(setup: SetupEvaluation) -> bool:
        return (
            SetupReasonType.TREND_ALIGNMENT
            in {
                reason.reason_type
                for reason in setup.reasons
            }
        )

    def _calculate_confirmation_score(
        self,
        setup: SetupEvaluation,
        structure_confirmed: bool,
        momentum_confirmed: bool,
        price_confirmed: bool,
        trend_confirmed: bool,
    ) -> float:
        score = float(setup.quality_score)

        confirmations = sum(
            (
                structure_confirmed,
                momentum_confirmed,
                price_confirmed,
                trend_confirmed,
            )
        )

        score += confirmations * 5.0

        score -= len(setup.conflicting_signals) * 10.0

        if not self._has_sufficient_history(setup):
            score -= 20.0

        if setup.direction == SetupDirection.NONE:
            score -= 30.0

        if setup.setup_type == SetupType.NONE:
            score -= 20.0

        return round(max(0.0, min(100.0, score)), 4)

    def _is_invalidated(
        self,
        setup: SetupEvaluation,
        structure_confirmed: bool,
        conflicting: tuple,
    ) -> bool:
        if not setup.valid:
            return True

        if setup.direction == SetupDirection.NONE:
            return True

        if setup.setup_type == SetupType.NONE:
            return True

        if len(conflicting) > self.max_conflicts:
            return True

        if not self._has_sufficient_history(setup):
            return True

        if (
            setup.setup_type == SetupType.REVERSAL
            and not structure_confirmed
        ):
            return True

        return False

    def _determine_decision(
        self,
        setup: SetupEvaluation,
        confirmation_score: float,
        structure_confirmed: bool,
        momentum_confirmed: bool,
        price_confirmed: bool,
        trend_confirmed: bool,
        conflicting: tuple,
        invalidated: bool,
    ) -> CandidateDecision:
        if invalidated:
            return CandidateDecision.REJECT

        if confirmation_score < self.minimum_confirmation_score:
            return CandidateDecision.WAIT

        confirmation_count = sum(
            (
                structure_confirmed,
                momentum_confirmed,
                price_confirmed,
                trend_confirmed,
            )
        )

        if confirmation_count < 2:
            return CandidateDecision.WAIT

        if (
            setup.setup_type == SetupType.TREND_CONTINUATION
            and not trend_confirmed
        ):
            return CandidateDecision.WAIT

        if len(conflicting) > self.max_conflicts:
            return CandidateDecision.REJECT

        if confirmation_score < self.trade_ready_score:
            return CandidateDecision.WAIT

        if setup.direction == SetupDirection.NONE:
            return CandidateDecision.REJECT

        return CandidateDecision.TRADE_READY