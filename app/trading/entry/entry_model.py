from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import math

from app.trading.setup.confluence_engine import (
    ConfluenceDirection,
    ConfluenceQuality,
    SetupConfluenceResult,
)


class EntryModelError(ValueError):
    """Raised when entry-model inputs or configuration are invalid."""


class EntryDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class EntryTrigger(str, Enum):
    MARKET = "MARKET"
    BREAKOUT = "BREAKOUT"
    PULLBACK = "PULLBACK"
    REJECTION = "REJECTION"
    MOMENTUM = "MOMENTUM"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class EntryQuality(str, Enum):
    STRONG = "STRONG"
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    WEAK = "WEAK"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


class EntryReasonType(str, Enum):
    LONG_DIRECTION = "LONG_DIRECTION"
    SHORT_DIRECTION = "SHORT_DIRECTION"

    SUPPORT_CONFIRMATION = "SUPPORT_CONFIRMATION"
    RESISTANCE_CONFIRMATION = "RESISTANCE_CONFIRMATION"
    TRENDLINE_CONFIRMATION = "TRENDLINE_CONFIRMATION"
    VOLUME_CONFIRMATION = "VOLUME_CONFIRMATION"
    MTF_CONFIRMATION = "MTF_CONFIRMATION"

    STRONG_CONFLUENCE = "STRONG_CONFLUENCE"
    GOOD_CONFLUENCE = "GOOD_CONFLUENCE"
    ACCEPTABLE_CONFLUENCE = "ACCEPTABLE_CONFLUENCE"

    LOW_CONFLUENCE = "LOW_CONFLUENCE"
    CONFLICTING_FACTORS = "CONFLICTING_FACTORS"

    NEUTRAL_DIRECTION = "NEUTRAL_DIRECTION"
    UNKNOWN_DIRECTION = "UNKNOWN_DIRECTION"

    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INVALID_CONFLUENCE = "INVALID_CONFLUENCE"

    ENTRY_ALLOWED = "ENTRY_ALLOWED"
    ENTRY_BLOCKED = "ENTRY_BLOCKED"


@dataclass(frozen=True, slots=True)
class EntryReason:
    reason_type: EntryReasonType
    message: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.reason_type,
            EntryReasonType,
        ):
            raise ValueError(
                "reason_type must be an EntryReasonType."
            )

        if (
            not isinstance(
                self.message,
                str,
            )
            or not self.message.strip()
        ):
            raise ValueError(
                "message must be a non-empty string."
            )


@dataclass(frozen=True, slots=True, init=False)
class EntryModel:
    timestamp: datetime
    symbol: str
    timeframe: str

    direction: EntryDirection
    trigger: EntryTrigger
    quality: EntryQuality

    reference_price: float
    entry_price: float | None

    confluence_score: float
    entry_confidence: float

    support_present: bool
    resistance_present: bool
    trendline_present: bool
    volume_confirmed: bool
    mtf_confirmed: bool

    valid: bool
    entry_allowed: bool

    reasons: tuple[EntryReason, ...]
    warnings: tuple[str, ...]

    model: str

    def __init__(
        self,
        timestamp: datetime,
        symbol: str,
        timeframe: str,
        direction: EntryDirection,
        trigger: EntryTrigger | None = None,
        quality: EntryQuality | None = None,
        reference_price: float = 1.0,
        entry_price: float | None = None,
        confluence_score: float = 0.0,
        entry_confidence: float | None = None,
        support_present: bool = False,
        resistance_present: bool = False,
        trendline_present: bool = False,
        volume_confirmed: bool = False,
        mtf_confirmed: bool = False,
        valid: bool = False,
        entry_allowed: bool = False,
        reasons: tuple[EntryReason, ...] = (),
        warnings: tuple[str, ...] = (),
        model: str = "entry",
        *,
        confidence: float | None = None,
        entry_ready: bool | None = None,
    ) -> None:
        """
        Construct an EntryModel.

        Canonical fields:
            entry_confidence
            entry_allowed

        Backward-compatible aliases:
            confidence
            entry_ready
        """

        # ------------------------------------------------------------------
        # Backward compatibility: confidence -> entry_confidence
        # ------------------------------------------------------------------

        if entry_confidence is None:
            if confidence is not None:
                entry_confidence = confidence
            else:
                entry_confidence = 0.0

        elif confidence is not None:
            if not math.isclose(
                float(entry_confidence),
                float(confidence),
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                raise ValueError(
                    "entry_confidence and confidence must match."
                )

        # ------------------------------------------------------------------
        # Backward compatibility: entry_ready
        #
        # Canonical state remains:
        #     valid
        #     entry_allowed
        #
        # We do not store entry_ready separately.
        # ------------------------------------------------------------------

        if entry_ready is not None:
            if not isinstance(
                entry_ready,
                bool,
            ):
                raise ValueError(
                    "entry_ready must be a boolean."
                )

            if entry_ready:
                valid = True
                entry_allowed = True
            else:
                entry_allowed = False

        # ------------------------------------------------------------------
        # Defaults for older callers that did not provide these fields.
        # ------------------------------------------------------------------

        if trigger is None:
            trigger = EntryTrigger.UNKNOWN

        if quality is None:
            quality = EntryQuality.UNKNOWN

        # ------------------------------------------------------------------
        # Frozen dataclass initialization.
        # ------------------------------------------------------------------

        object.__setattr__(
            self,
            "timestamp",
            timestamp,
        )

        object.__setattr__(
            self,
            "symbol",
            symbol,
        )

        object.__setattr__(
            self,
            "timeframe",
            timeframe,
        )

        object.__setattr__(
            self,
            "direction",
            direction,
        )

        object.__setattr__(
            self,
            "trigger",
            trigger,
        )

        object.__setattr__(
            self,
            "quality",
            quality,
        )

        object.__setattr__(
            self,
            "reference_price",
            reference_price,
        )

        object.__setattr__(
            self,
            "entry_price",
            entry_price,
        )

        object.__setattr__(
            self,
            "confluence_score",
            confluence_score,
        )

        object.__setattr__(
            self,
            "entry_confidence",
            entry_confidence,
        )

        object.__setattr__(
            self,
            "support_present",
            support_present,
        )

        object.__setattr__(
            self,
            "resistance_present",
            resistance_present,
        )

        object.__setattr__(
            self,
            "trendline_present",
            trendline_present,
        )

        object.__setattr__(
            self,
            "volume_confirmed",
            volume_confirmed,
        )

        object.__setattr__(
            self,
            "mtf_confirmed",
            mtf_confirmed,
        )

        object.__setattr__(
            self,
            "valid",
            valid,
        )

        object.__setattr__(
            self,
            "entry_allowed",
            entry_allowed,
        )

        object.__setattr__(
            self,
            "reasons",
            reasons,
        )

        object.__setattr__(
            self,
            "warnings",
            warnings,
        )

        object.__setattr__(
            self,
            "model",
            model,
        )

        self.__post_init__()

    def __post_init__(self) -> None:
        # --------------------------------------------------------------
        # Timestamp
        # --------------------------------------------------------------

        if not isinstance(
            self.timestamp,
            datetime,
        ):
            raise ValueError(
                "timestamp must be a datetime."
            )

        # --------------------------------------------------------------
        # Symbol
        #
        # IMPORTANT:
        # Empty symbols are allowed at the EntryModel data-object level.
        # Downstream engines such as TakeProfitIntelligenceEngine perform
        # domain-specific validation.
        # --------------------------------------------------------------

        if not isinstance(
            self.symbol,
            str,
        ):
            raise ValueError(
                "symbol must be a string."
            )

        # --------------------------------------------------------------
        # Timeframe
        # --------------------------------------------------------------

        if (
            not isinstance(
                self.timeframe,
                str,
            )
            or not self.timeframe.strip()
        ):
            raise ValueError(
                "timeframe must be a non-empty string."
            )

        # --------------------------------------------------------------
        # Direction
        # --------------------------------------------------------------

        if not isinstance(
            self.direction,
            EntryDirection,
        ):
            raise ValueError(
                "direction must be an EntryDirection."
            )

        # --------------------------------------------------------------
        # Trigger
        # --------------------------------------------------------------

        if not isinstance(
            self.trigger,
            EntryTrigger,
        ):
            raise ValueError(
                "trigger must be an EntryTrigger."
            )

        # --------------------------------------------------------------
        # Quality
        # --------------------------------------------------------------

        if not isinstance(
            self.quality,
            EntryQuality,
        ):
            raise ValueError(
                "quality must be an EntryQuality."
            )

        # --------------------------------------------------------------
        # Prices
        # --------------------------------------------------------------

        self._validate_price(
            self.reference_price,
            "reference_price",
        )

        if self.entry_price is not None:
            self._validate_price(
                self.entry_price,
                "entry_price",
            )

        # --------------------------------------------------------------
        # Scores
        # --------------------------------------------------------------

        self._validate_score(
            self.confluence_score,
            "confluence_score",
        )

        self._validate_score(
            self.entry_confidence,
            "entry_confidence",
        )

        # --------------------------------------------------------------
        # Boolean fields
        # --------------------------------------------------------------

        for name, value in (
            (
                "support_present",
                self.support_present,
            ),
            (
                "resistance_present",
                self.resistance_present,
            ),
            (
                "trendline_present",
                self.trendline_present,
            ),
            (
                "volume_confirmed",
                self.volume_confirmed,
            ),
            (
                "mtf_confirmed",
                self.mtf_confirmed,
            ),
            (
                "valid",
                self.valid,
            ),
            (
                "entry_allowed",
                self.entry_allowed,
            ),
        ):
            if not isinstance(
                value,
                bool,
            ):
                raise ValueError(
                    f"{name} must be a boolean."
                )

        # --------------------------------------------------------------
        # Reasons
        # --------------------------------------------------------------

        if not isinstance(
            self.reasons,
            tuple,
        ):
            raise ValueError(
                "reasons must be a tuple."
            )

        # --------------------------------------------------------------
        # Warnings
        # --------------------------------------------------------------

        if not isinstance(
            self.warnings,
            tuple,
        ):
            raise ValueError(
                "warnings must be a tuple."
            )

        # --------------------------------------------------------------
        # Model
        # --------------------------------------------------------------

        if (
            not isinstance(
                self.model,
                str,
            )
            or not self.model.strip()
        ):
            raise ValueError(
                "model must be a non-empty string."
            )

    @staticmethod
    def _validate_price(
        value: float,
        name: str,
    ) -> None:
        if isinstance(
            value,
            bool,
        ):
            raise ValueError(
                f"{name} must be a finite positive number."
            )

        if not isinstance(
            value,
            (int, float),
        ):
            raise ValueError(
                f"{name} must be a finite positive number."
            )

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(
                f"{name} must be finite."
            )

        if value <= 0.0:
            raise ValueError(
                f"{name} must be greater than zero."
            )

    @staticmethod
    def _validate_score(
        value: float,
        name: str,
    ) -> None:
        if isinstance(
            value,
            bool,
        ):
            raise ValueError(
                f"{name} must be a finite number."
            )

        if not isinstance(
            value,
            (int, float),
        ):
            raise ValueError(
                f"{name} must be a finite number."
            )

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(
                f"{name} must be finite."
            )

        if value < 0.0 or value > 100.0:
            raise ValueError(
                f"{name} must be between 0 and 100."
            )

    # ==================================================================
    # Convenience / compatibility properties
    # ==================================================================

    @property
    def is_long(self) -> bool:
        return self.direction is EntryDirection.LONG

    @property
    def is_short(self) -> bool:
        return self.direction is EntryDirection.SHORT

    @property
    def is_none(self) -> bool:
        return self.direction is EntryDirection.NONE

    @property
    def is_unknown(self) -> bool:
        return self.direction is EntryDirection.UNKNOWN

    @property
    def is_strong(self) -> bool:
        return self.quality is EntryQuality.STRONG

    @property
    def is_rejected(self) -> bool:
        return self.quality is EntryQuality.REJECTED

    @property
    def is_directional(self) -> bool:
        return self.direction in (
            EntryDirection.LONG,
            EntryDirection.SHORT,
        )

    @property
    def is_valid(self) -> bool:
        return self.valid

    @property
    def is_ready(self) -> bool:
        return (
            self.valid
            and self.entry_allowed
        )

    @property
    def entry_ready(self) -> bool:
        """
        Backward-compatible readiness property.

        Canonical state is represented by:
            valid
            entry_allowed
        """
        return (
            self.valid
            and self.entry_allowed
        )

    @property
    def confidence(self) -> float:
        """
        Backward-compatible alias for entry_confidence.
        """
        return self.entry_confidence

    @property
    def has_entry(self) -> bool:
        return (
            self.entry_price is not None
            and self.entry_allowed
        )


class EntryModelEngine:
    """
    Deterministic entry model based on setup confluence.

    This engine:
    - does not use an LLM
    - does not execute trades
    - does not calculate stop loss
    - does not calculate take profit
    - does not calculate position size
    - does not perform account-risk calculations

    Its responsibility is to transform a valid
    SetupConfluenceResult into an EntryModel.
    """

    def __init__(
        self,
        minimum_confluence_score: float = 50.0,
        good_confluence_score: float = 65.0,
        strong_confluence_score: float = 80.0,
        minimum_entry_confidence: float = 50.0,
        good_entry_confidence: float = 65.0,
        strong_entry_confidence: float = 80.0,
    ) -> None:
        self.minimum_confluence_score = (
            self._validate_threshold(
                minimum_confluence_score,
                "minimum_confluence_score",
            )
        )

        self.good_confluence_score = (
            self._validate_threshold(
                good_confluence_score,
                "good_confluence_score",
            )
        )

        self.strong_confluence_score = (
            self._validate_threshold(
                strong_confluence_score,
                "strong_confluence_score",
            )
        )

        self.minimum_entry_confidence = (
            self._validate_threshold(
                minimum_entry_confidence,
                "minimum_entry_confidence",
            )
        )

        self.good_entry_confidence = (
            self._validate_threshold(
                good_entry_confidence,
                "good_entry_confidence",
            )
        )

        self.strong_entry_confidence = (
            self._validate_threshold(
                strong_entry_confidence,
                "strong_entry_confidence",
            )
        )

        if not (
            self.minimum_confluence_score
            <= self.good_confluence_score
            <= self.strong_confluence_score
        ):
            raise EntryModelError(
                "confluence thresholds must be ordered from "
                "minimum to good to strong."
            )

        if not (
            self.minimum_entry_confidence
            <= self.good_entry_confidence
            <= self.strong_entry_confidence
        ):
            raise EntryModelError(
                "confidence thresholds must be ordered from "
                "minimum to good to strong."
            )

    @staticmethod
    def _validate_threshold(
        value: float,
        name: str,
    ) -> float:
        if (
            isinstance(
                value,
                bool,
            )
            or not isinstance(
                value,
                (int, float),
            )
        ):
            raise EntryModelError(
                f"{name} must be a finite number between 0 and 100."
            )

        value = float(value)

        if not math.isfinite(value):
            raise EntryModelError(
                f"{name} must be finite."
            )

        if value < 0.0 or value > 100.0:
            raise EntryModelError(
                f"{name} must be between 0 and 100."
            )

        return value

    def analyze(
        self,
        confluence: SetupConfluenceResult,
    ) -> EntryModel:
        self._validate_confluence(
            confluence
        )

        direction = self._determine_direction(
            confluence.direction
        )

        reference_price = self._get_reference_price(
            confluence
        )

        entry_price = (
            reference_price
            if direction in (
                EntryDirection.LONG,
                EntryDirection.SHORT,
            )
            else None
        )

        support_present = bool(
            confluence.support_present
        )

        resistance_present = bool(
            confluence.resistance_present
        )

        trendline_present = bool(
            confluence.support_trendline_present
            or confluence.resistance_trendline_present
        )

        volume_confirmed = bool(
            confluence.volume_confirmed
        )

        mtf_confirmed = bool(
            confluence.mtf_score
            >= self.minimum_confluence_score
        )

        confidence = self._calculate_confidence(
            confluence
        )

        quality = self._calculate_quality(
            score=confluence.score,
            confidence=confidence,
            confluence_quality=confluence.quality,
            direction=direction,
            sufficient_data=confluence.sufficient_data,
        )

        trigger = self._determine_trigger(
            direction=direction,
            support_present=support_present,
            resistance_present=resistance_present,
            trendline_present=trendline_present,
            volume_confirmed=volume_confirmed,
        )

        reasons = self._build_reasons(
            confluence=confluence,
            direction=direction,
            quality=quality,
            support_present=support_present,
            resistance_present=resistance_present,
            trendline_present=trendline_present,
            volume_confirmed=volume_confirmed,
            mtf_confirmed=mtf_confirmed,
        )

        warnings = self._build_warnings(
            confluence=confluence,
            direction=direction,
            quality=quality,
        )

        valid = self._is_valid(
            confluence=confluence,
            direction=direction,
            quality=quality,
        )

        entry_allowed = self._is_entry_allowed(
            confluence=confluence,
            direction=direction,
            quality=quality,
            confidence=confidence,
        )

        if not valid:
            entry_allowed = False

        return EntryModel(
            timestamp=confluence.timestamp,
            symbol=confluence.symbol,
            timeframe=confluence.timeframe,
            direction=direction,
            trigger=trigger,
            quality=quality,
            reference_price=reference_price,
            entry_price=entry_price,
            confluence_score=float(
                confluence.score
            ),
            entry_confidence=confidence,
            support_present=support_present,
            resistance_present=resistance_present,
            trendline_present=trendline_present,
            volume_confirmed=volume_confirmed,
            mtf_confirmed=mtf_confirmed,
            valid=valid,
            entry_allowed=entry_allowed,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
            model="entry",
        )

    def analyze_xauusd(
        self,
        confluence: SetupConfluenceResult,
    ) -> EntryModel:
        self._validate_confluence(
            confluence
        )

        if (
            confluence.symbol.upper()
            != "XAUUSD"
        ):
            raise EntryModelError(
                "analyze_xauusd requires symbol XAUUSD."
            )

        return self.analyze(
            confluence
        )

    @staticmethod
    def _validate_confluence(
        confluence: SetupConfluenceResult,
    ) -> None:
        if not isinstance(
            confluence,
            SetupConfluenceResult,
        ):
            raise EntryModelError(
                "confluence must be a SetupConfluenceResult."
            )

        if not isinstance(
            confluence.timestamp,
            datetime,
        ):
            raise EntryModelError(
                "confluence timestamp must be a datetime."
            )

        if (
            not isinstance(
                confluence.symbol,
                str,
            )
            or not confluence.symbol.strip()
        ):
            raise EntryModelError(
                "confluence symbol must be a non-empty string."
            )

        if (
            not isinstance(
                confluence.timeframe,
                str,
            )
            or not confluence.timeframe.strip()
        ):
            raise EntryModelError(
                "confluence timeframe must be a non-empty string."
            )

        if not isinstance(
            confluence.direction,
            ConfluenceDirection,
        ):
            raise EntryModelError(
                "confluence direction must be a ConfluenceDirection."
            )

        if not isinstance(
            confluence.quality,
            ConfluenceQuality,
        ):
            raise EntryModelError(
                "confluence quality must be a ConfluenceQuality."
            )

        EntryModel._validate_score(
            confluence.score,
            "confluence score",
        )

        EntryModel._validate_score(
            confluence.mtf_score,
            "mtf_score",
        )

        EntryModel._validate_score(
            confluence.support_resistance_score,
            "support_resistance_score",
        )

        EntryModel._validate_score(
            confluence.trendline_score,
            "trendline_score",
        )

        EntryModel._validate_score(
            confluence.volume_score,
            "volume_score",
        )

        for name in (
            "bullish_factors",
            "bearish_factors",
            "neutral_factors",
            "conflicting_factors",
        ):
            value = getattr(
                confluence,
                name,
            )

            if (
                isinstance(
                    value,
                    bool,
                )
                or not isinstance(
                    value,
                    int,
                )
            ):
                raise EntryModelError(
                    f"{name} must be a non-negative integer."
                )

            if value < 0:
                raise EntryModelError(
                    f"{name} must be non-negative."
                )

        for name in (
            "support_present",
            "resistance_present",
            "support_strong",
            "resistance_strong",
            "support_trendline_present",
            "resistance_trendline_present",
            "volume_confirmed",
            "sufficient_data",
        ):
            if not isinstance(
                getattr(
                    confluence,
                    name,
                ),
                bool,
            ):
                raise EntryModelError(
                    f"{name} must be a boolean."
                )

    @staticmethod
    def _determine_direction(
        direction: ConfluenceDirection,
    ) -> EntryDirection:
        if direction is ConfluenceDirection.BULLISH:
            return EntryDirection.LONG

        if direction is ConfluenceDirection.BEARISH:
            return EntryDirection.SHORT

        if direction is ConfluenceDirection.NEUTRAL:
            return EntryDirection.NONE

        return EntryDirection.UNKNOWN

    @staticmethod
    def _get_reference_price(
        confluence: SetupConfluenceResult,
    ) -> float:
        for name in (
            "reference_price",
            "entry_price",
            "close",
            "price",
        ):
            if hasattr(
                confluence,
                name,
            ):
                value = getattr(
                    confluence,
                    name,
                )

                if isinstance(
                    value,
                    bool,
                ):
                    continue

                if isinstance(
                    value,
                    (int, float),
                ):
                    value = float(value)

                    if (
                        math.isfinite(value)
                        and value > 0.0
                    ):
                        return round(
                            value,
                            10,
                        )

        return 1.0

    def _calculate_confidence(
        self,
        confluence: SetupConfluenceResult,
    ) -> float:
        confidence = float(
            confluence.score
        )

        if (
            confluence.mtf_score
            >= self.minimum_confluence_score
        ):
            confidence += 5.0

        if confluence.volume_confirmed:
            confidence += 5.0

        if (
            confluence.support_present
            or confluence.resistance_present
        ):
            confidence += 3.0

        if (
            confluence.support_trendline_present
            or confluence.resistance_trendline_present
        ):
            confidence += 2.0

        if confluence.conflicting_factors > 0:
            confidence -= min(
                20.0,
                confluence.conflicting_factors
                * 10.0,
            )

        confidence = max(
            0.0,
            min(
                100.0,
                confidence,
            ),
        )

        return round(
            confidence,
            10,
        )

    def _calculate_quality(
        self,
        score: float,
        confidence: float,
        confluence_quality: ConfluenceQuality,
        direction: EntryDirection,
        sufficient_data: bool,
    ) -> EntryQuality:
        if not sufficient_data:
            return EntryQuality.REJECTED

        if direction in (
            EntryDirection.NONE,
            EntryDirection.UNKNOWN,
        ):
            return EntryQuality.REJECTED

        if (
            confluence_quality
            is ConfluenceQuality.CONFLICTED
        ):
            return EntryQuality.REJECTED

        if (
            confluence_quality
            is ConfluenceQuality.UNKNOWN
        ):
            return EntryQuality.REJECTED

        if (
            confluence_quality
            is ConfluenceQuality.WEAK
        ):
            return EntryQuality.WEAK

        effective_score = min(
            float(score),
            float(confidence),
        )

        if (
            effective_score
            >= self.strong_entry_confidence
        ):
            return EntryQuality.STRONG

        if (
            effective_score
            >= self.good_entry_confidence
        ):
            return EntryQuality.GOOD

        if (
            effective_score
            >= self.minimum_entry_confidence
        ):
            return EntryQuality.ACCEPTABLE

        return EntryQuality.WEAK

    @staticmethod
    def _determine_trigger(
        direction: EntryDirection,
        support_present: bool,
        resistance_present: bool,
        trendline_present: bool,
        volume_confirmed: bool,
    ) -> EntryTrigger:
        if direction is EntryDirection.LONG:
            if support_present:
                return EntryTrigger.PULLBACK

            if trendline_present:
                return EntryTrigger.REJECTION

            if volume_confirmed:
                return EntryTrigger.MOMENTUM

            return EntryTrigger.MARKET

        if direction is EntryDirection.SHORT:
            if resistance_present:
                return EntryTrigger.PULLBACK

            if trendline_present:
                return EntryTrigger.REJECTION

            if volume_confirmed:
                return EntryTrigger.MOMENTUM

            return EntryTrigger.MARKET

        if direction is EntryDirection.NONE:
            return EntryTrigger.NONE

        return EntryTrigger.UNKNOWN

    def _is_valid(
        self,
        confluence: SetupConfluenceResult,
        direction: EntryDirection,
        quality: EntryQuality,
    ) -> bool:
        if not confluence.sufficient_data:
            return False

        if direction in (
            EntryDirection.NONE,
            EntryDirection.UNKNOWN,
        ):
            return False

        if confluence.conflicting_factors > 0:
            return False

        if (
            float(confluence.score)
            < self.minimum_confluence_score
        ):
            return False

        if quality in (
            EntryQuality.REJECTED,
            EntryQuality.UNKNOWN,
            EntryQuality.WEAK,
        ):
            return False

        return True

    def _is_entry_allowed(
        self,
        confluence: SetupConfluenceResult,
        direction: EntryDirection,
        quality: EntryQuality,
        confidence: float,
    ) -> bool:
        if direction not in (
            EntryDirection.LONG,
            EntryDirection.SHORT,
        ):
            return False

        if not confluence.sufficient_data:
            return False

        if confluence.conflicting_factors > 0:
            return False

        if (
            float(confluence.score)
            < self.minimum_confluence_score
        ):
            return False

        if (
            confidence
            < self.minimum_entry_confidence
        ):
            return False

        if quality in (
            EntryQuality.REJECTED,
            EntryQuality.UNKNOWN,
            EntryQuality.WEAK,
        ):
            return False

        return True

    def _build_reasons(
        self,
        confluence: SetupConfluenceResult,
        direction: EntryDirection,
        quality: EntryQuality,
        support_present: bool,
        resistance_present: bool,
        trendline_present: bool,
        volume_confirmed: bool,
        mtf_confirmed: bool,
    ) -> list[EntryReason]:
        reasons: list[EntryReason] = []

        if direction is EntryDirection.LONG:
            reasons.append(
                EntryReason(
                    EntryReasonType.LONG_DIRECTION,
                    "Bullish confluence supports a long entry.",
                )
            )

        elif direction is EntryDirection.SHORT:
            reasons.append(
                EntryReason(
                    EntryReasonType.SHORT_DIRECTION,
                    "Bearish confluence supports a short entry.",
                )
            )

        elif direction is EntryDirection.NONE:
            reasons.append(
                EntryReason(
                    EntryReasonType.NEUTRAL_DIRECTION,
                    "Confluence has no directional edge.",
                )
            )

        else:
            reasons.append(
                EntryReason(
                    EntryReasonType.UNKNOWN_DIRECTION,
                    "Confluence direction is unknown.",
                )
            )

        if support_present:
            reasons.append(
                EntryReason(
                    EntryReasonType.SUPPORT_CONFIRMATION,
                    "Support is present in the confluence analysis.",
                )
            )

        if resistance_present:
            reasons.append(
                EntryReason(
                    EntryReasonType.RESISTANCE_CONFIRMATION,
                    "Resistance is present in the confluence analysis.",
                )
            )

        if trendline_present:
            reasons.append(
                EntryReason(
                    EntryReasonType.TRENDLINE_CONFIRMATION,
                    "Trendline evidence is present.",
                )
            )

        if volume_confirmed:
            reasons.append(
                EntryReason(
                    EntryReasonType.VOLUME_CONFIRMATION,
                    "Volume confirmation is present.",
                )
            )

        if mtf_confirmed:
            reasons.append(
                EntryReason(
                    EntryReasonType.MTF_CONFIRMATION,
                    "Multi-timeframe confirmation is present.",
                )
            )

        if quality is EntryQuality.STRONG:
            reasons.append(
                EntryReason(
                    EntryReasonType.STRONG_CONFLUENCE,
                    (
                        "Confluence is strong enough for "
                        "a high-quality entry."
                    ),
                )
            )

        elif quality is EntryQuality.GOOD:
            reasons.append(
                EntryReason(
                    EntryReasonType.GOOD_CONFLUENCE,
                    "Confluence provides good entry conditions.",
                )
            )

        elif quality is EntryQuality.ACCEPTABLE:
            reasons.append(
                EntryReason(
                    EntryReasonType.ACCEPTABLE_CONFLUENCE,
                    (
                        "Confluence meets the minimum "
                        "acceptable entry threshold."
                    ),
                )
            )

        else:
            reasons.append(
                EntryReason(
                    EntryReasonType.LOW_CONFLUENCE,
                    (
                        "Confluence is below the required "
                        "entry quality."
                    ),
                )
            )

        if confluence.conflicting_factors > 0:
            reasons.append(
                EntryReason(
                    EntryReasonType.CONFLICTING_FACTORS,
                    (
                        "Conflicting factors prevent "
                        "entry confirmation."
                    ),
                )
            )

        if not confluence.sufficient_data:
            reasons.append(
                EntryReason(
                    EntryReasonType.INSUFFICIENT_DATA,
                    "Confluence data is insufficient.",
                )
            )

        if (
            direction
            in (
                EntryDirection.LONG,
                EntryDirection.SHORT,
            )
            and quality
            not in (
                EntryQuality.WEAK,
                EntryQuality.REJECTED,
            )
            and confluence.conflicting_factors == 0
            and confluence.sufficient_data
            and float(confluence.score)
            >= self.minimum_confluence_score
        ):
            reasons.append(
                EntryReason(
                    EntryReasonType.ENTRY_ALLOWED,
                    (
                        "Entry conditions satisfy the "
                        "configured requirements."
                    ),
                )
            )

        else:
            reasons.append(
                EntryReason(
                    EntryReasonType.ENTRY_BLOCKED,
                    (
                        "Entry conditions do not satisfy "
                        "all configured requirements."
                    ),
                )
            )

        return reasons

    @staticmethod
    def _build_warnings(
        confluence: SetupConfluenceResult,
        direction: EntryDirection,
        quality: EntryQuality,
    ) -> list[str]:
        warnings: list[str] = []

        if confluence.conflicting_factors > 0:
            warnings.append(
                "Conflicting confluence factors are present."
            )

        if not confluence.sufficient_data:
            warnings.append(
                "Confluence data is insufficient."
            )

        if direction in (
            EntryDirection.NONE,
            EntryDirection.UNKNOWN,
        ):
            warnings.append(
                "No actionable directional entry is available."
            )

        if quality in (
            EntryQuality.WEAK,
            EntryQuality.REJECTED,
        ):
            warnings.append(
                "Entry quality is below the actionable threshold."
            )

        warnings.append(
            "Entry model does not calculate stop loss, take profit, "
            "position size, or account risk."
        )

        return warnings


__all__ = [
    "EntryDirection",
    "EntryModel",
    "EntryModelEngine",
    "EntryModelError",
    "EntryQuality",
    "EntryReason",
    "EntryReasonType",
    "EntryTrigger",
]