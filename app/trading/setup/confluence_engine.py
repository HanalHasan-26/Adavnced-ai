from __future__ import annotations

import math
from datetime import datetime

from app.trading.confluence.setup_confluence import (
    ConfluenceDirection,
    ConfluenceQuality,
    ConfluenceReason,
    ConfluenceReasonType,
    SetupConfluenceResult,
)


class ConfluenceEngineError(Exception):
    """Base exception for confluence engine errors."""


class SetupConfluenceEngine:
    """
    Deterministic setup-confluence engine.

    This engine evaluates agreement between:

    - multi-timeframe analysis
    - support/resistance
    - trendlines
    - volume

    It does NOT:

    - calculate entry price
    - calculate stop loss
    - calculate take profit
    - calculate position size
    - calculate account risk
    - execute trades
    - fetch news
    - call an LLM
    """

    DEFAULT_MINIMUM_SCORE = 50.0
    DEFAULT_GOOD_SCORE = 65.0
    DEFAULT_STRONG_SCORE = 80.0

    def __init__(
        self,
        minimum_score: float = DEFAULT_MINIMUM_SCORE,
        good_score: float = DEFAULT_GOOD_SCORE,
        strong_score: float = DEFAULT_STRONG_SCORE,
    ) -> None:
        self.minimum_score = self._validate_threshold(
            minimum_score,
            "minimum_score",
        )
        self.good_score = self._validate_threshold(
            good_score,
            "good_score",
        )
        self.strong_score = self._validate_threshold(
            strong_score,
            "strong_score",
        )

        if not (
            self.minimum_score
            <= self.good_score
            <= self.strong_score
        ):
            raise ValueError(
                "score thresholds must be ordered from "
                "minimum to good to strong."
            )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def analyze(
        self,
        *,
        timestamp: datetime,
        symbol: str,
        timeframe: str,
        mtf_score: float = 0.0,
        support_resistance_score: float = 0.0,
        trendline_score: float = 0.0,
        volume_score: float = 0.0,
        support_present: bool = False,
        resistance_present: bool = False,
        support_strong: bool = False,
        resistance_strong: bool = False,
        support_trendline_present: bool = False,
        resistance_trendline_present: bool = False,
        volume_confirmed: bool = False,
        bullish_factors: int = 0,
        bearish_factors: int = 0,
        neutral_factors: int = 0,
        conflicting_factors: int = 0,
        sufficient_data: bool = True,
    ) -> SetupConfluenceResult:

        self._validate_inputs(
            timestamp=timestamp,
            symbol=symbol,
            timeframe=timeframe,
            mtf_score=mtf_score,
            support_resistance_score=support_resistance_score,
            trendline_score=trendline_score,
            volume_score=volume_score,
            support_present=support_present,
            resistance_present=resistance_present,
            support_strong=support_strong,
            resistance_strong=resistance_strong,
            support_trendline_present=support_trendline_present,
            resistance_trendline_present=resistance_trendline_present,
            volume_confirmed=volume_confirmed,
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors,
            neutral_factors=neutral_factors,
            conflicting_factors=conflicting_factors,
            sufficient_data=sufficient_data,
        )

        score = self._calculate_score(
            mtf_score=mtf_score,
            support_resistance_score=support_resistance_score,
            trendline_score=trendline_score,
            volume_score=volume_score,
            conflicting_factors=conflicting_factors,
            sufficient_data=sufficient_data,
        )

        direction = self._determine_direction(
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors,
            neutral_factors=neutral_factors,
            conflicting_factors=conflicting_factors,
            support_present=support_present,
            resistance_present=resistance_present,
        )

        quality = self._determine_quality(
            score=score,
            direction=direction,
            conflicting_factors=conflicting_factors,
            sufficient_data=sufficient_data,
        )

        reasons = self._build_reasons(
            direction=direction,
            quality=quality,
            mtf_score=mtf_score,
            support_present=support_present,
            resistance_present=resistance_present,
            support_strong=support_strong,
            resistance_strong=resistance_strong,
            support_trendline_present=support_trendline_present,
            resistance_trendline_present=resistance_trendline_present,
            volume_confirmed=volume_confirmed,
            conflicting_factors=conflicting_factors,
            sufficient_data=sufficient_data,
        )

        warnings = self._build_warnings(
            direction=direction,
            quality=quality,
            conflicting_factors=conflicting_factors,
            sufficient_data=sufficient_data,
        )

        return SetupConfluenceResult(
            timestamp=timestamp,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            quality=quality,
            score=round(score, 10),
            mtf_score=float(mtf_score),
            support_resistance_score=float(
                support_resistance_score
            ),
            trendline_score=float(trendline_score),
            volume_score=float(volume_score),
            support_present=support_present,
            resistance_present=resistance_present,
            support_strong=support_strong,
            resistance_strong=resistance_strong,
            support_trendline_present=(
                support_trendline_present
            ),
            resistance_trendline_present=(
                resistance_trendline_present
            ),
            volume_confirmed=volume_confirmed,
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors,
            neutral_factors=neutral_factors,
            conflicting_factors=conflicting_factors,
            sufficient_data=sufficient_data,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
        )

    def analyze_xauusd(
        self,
        *,
        timestamp: datetime,
        timeframe: str,
        mtf_score: float = 0.0,
        support_resistance_score: float = 0.0,
        trendline_score: float = 0.0,
        volume_score: float = 0.0,
        support_present: bool = False,
        resistance_present: bool = False,
        support_strong: bool = False,
        resistance_strong: bool = False,
        support_trendline_present: bool = False,
        resistance_trendline_present: bool = False,
        volume_confirmed: bool = False,
        bullish_factors: int = 0,
        bearish_factors: int = 0,
        neutral_factors: int = 0,
        conflicting_factors: int = 0,
        sufficient_data: bool = True,
    ) -> SetupConfluenceResult:

        return self.analyze(
            timestamp=timestamp,
            symbol="XAUUSD",
            timeframe=timeframe,
            mtf_score=mtf_score,
            support_resistance_score=support_resistance_score,
            trendline_score=trendline_score,
            volume_score=volume_score,
            support_present=support_present,
            resistance_present=resistance_present,
            support_strong=support_strong,
            resistance_strong=resistance_strong,
            support_trendline_present=(
                support_trendline_present
            ),
            resistance_trendline_present=(
                resistance_trendline_present
            ),
            volume_confirmed=volume_confirmed,
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors,
            neutral_factors=neutral_factors,
            conflicting_factors=conflicting_factors,
            sufficient_data=sufficient_data,
        )

    # =========================================================
    # SCORE
    # =========================================================

    @staticmethod
    def _calculate_score(
        *,
        mtf_score: float,
        support_resistance_score: float,
        trendline_score: float,
        volume_score: float,
        conflicting_factors: int,
        sufficient_data: bool,
    ) -> float:

        score = (
            float(mtf_score) * 0.30
            + float(support_resistance_score) * 0.30
            + float(trendline_score) * 0.20
            + float(volume_score) * 0.20
        )

        if conflicting_factors > 0:
            score -= min(
                30.0,
                conflicting_factors * 10.0,
            )

        if not sufficient_data:
            score -= 25.0

        return max(
            0.0,
            min(100.0, score),
        )

    # =========================================================
    # DIRECTION
    # =========================================================

    @staticmethod
    def _determine_direction(
        *,
        bullish_factors: int,
        bearish_factors: int,
        neutral_factors: int,
        conflicting_factors: int,
        support_present: bool,
        resistance_present: bool,
    ) -> ConfluenceDirection:

        if bullish_factors == 0 and bearish_factors == 0:

            if support_present and not resistance_present:
                return ConfluenceDirection.BULLISH

            if resistance_present and not support_present:
                return ConfluenceDirection.BEARISH

            if neutral_factors > 0:
                return ConfluenceDirection.NEUTRAL

            return ConfluenceDirection.UNKNOWN

        if bullish_factors > bearish_factors:
            return ConfluenceDirection.BULLISH

        if bearish_factors > bullish_factors:
            return ConfluenceDirection.BEARISH

        if conflicting_factors > 0:
            return ConfluenceDirection.NEUTRAL

        return ConfluenceDirection.NEUTRAL

    # =========================================================
    # QUALITY
    # =========================================================

    def _determine_quality(
        self,
        *,
        score: float,
        direction: ConfluenceDirection,
        conflicting_factors: int,
        sufficient_data: bool,
    ) -> ConfluenceQuality:

        if not sufficient_data:
            return ConfluenceQuality.UNKNOWN

        if direction in (
            ConfluenceDirection.UNKNOWN,
            ConfluenceDirection.NEUTRAL,
        ):
            if conflicting_factors > 0:
                return ConfluenceQuality.CONFLICTED

            return ConfluenceQuality.WEAK

        if conflicting_factors >= 2:
            return ConfluenceQuality.CONFLICTED

        if score >= self.strong_score:
            return ConfluenceQuality.STRONG

        if score >= self.good_score:
            return ConfluenceQuality.GOOD

        if score >= self.minimum_score:
            return ConfluenceQuality.ACCEPTABLE

        return ConfluenceQuality.WEAK

    # =========================================================
    # REASONS
    # =========================================================

    @staticmethod
    def _build_reasons(
        *,
        direction: ConfluenceDirection,
        quality: ConfluenceQuality,
        mtf_score: float,
        support_present: bool,
        resistance_present: bool,
        support_strong: bool,
        resistance_strong: bool,
        support_trendline_present: bool,
        resistance_trendline_present: bool,
        volume_confirmed: bool,
        conflicting_factors: int,
        sufficient_data: bool,
    ) -> list[ConfluenceReason]:

        reasons: list[ConfluenceReason] = []

        if direction is ConfluenceDirection.BULLISH:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.BULLISH_ALIGNMENT,
                    "Confluence factors favor the bullish direction.",
                )
            )

        elif direction is ConfluenceDirection.BEARISH:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.BEARISH_ALIGNMENT,
                    "Confluence factors favor the bearish direction.",
                )
            )

        elif direction is ConfluenceDirection.NEUTRAL:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.NEUTRAL_DIRECTION,
                    "Confluence factors do not provide a clear directional edge.",
                )
            )

        else:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.UNKNOWN_DIRECTION,
                    "Confluence direction cannot be determined.",
                )
            )

        if support_present:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.SUPPORT_PRESENT,
                    "Support is present in the setup.",
                )
            )

        if resistance_present:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.RESISTANCE_PRESENT,
                    "Resistance is present in the setup.",
                )
            )

        if support_strong:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.SUPPORT_STRONG,
                    "Support is classified as strong.",
                )
            )

        if resistance_strong:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.RESISTANCE_STRONG,
                    "Resistance is classified as strong.",
                )
            )

        if support_trendline_present:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.TRENDLINE_SUPPORT,
                    "A supporting trendline is present.",
                )
            )

        if resistance_trendline_present:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.TRENDLINE_RESISTANCE,
                    "A resisting trendline is present.",
                )
            )

        if volume_confirmed:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.VOLUME_CONFIRMATION,
                    "Volume confirms the setup.",
                )
            )

        if mtf_score >= 50.0:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.MTF_CONFIRMATION,
                    "Multi-timeframe evidence contributes positively.",
                )
            )

        if conflicting_factors > 0:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.CONFLICTING_FACTORS,
                    "Conflicting factors reduce confluence quality.",
                )
            )

        if not sufficient_data:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.INSUFFICIENT_DATA,
                    "Available data is insufficient for reliable confluence evaluation.",
                )
            )

        if quality is ConfluenceQuality.STRONG:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.STRONG_CONFLUENCE,
                    "The combined confluence score is strong.",
                )
            )

        elif quality is ConfluenceQuality.GOOD:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.GOOD_CONFLUENCE,
                    "The combined confluence score is good.",
                )
            )

        elif quality is ConfluenceQuality.ACCEPTABLE:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.ACCEPTABLE_CONFLUENCE,
                    "The combined confluence score meets the acceptable threshold.",
                )
            )

        else:
            reasons.append(
                ConfluenceReason(
                    ConfluenceReasonType.LOW_CONFLUENCE,
                    "The combined confluence score is below the preferred level.",
                )
            )

        return reasons

    # =========================================================
    # WARNINGS
    # =========================================================

    @staticmethod
    def _build_warnings(
        *,
        direction: ConfluenceDirection,
        quality: ConfluenceQuality,
        conflicting_factors: int,
        sufficient_data: bool,
    ) -> list[str]:

        warnings: list[str] = []

        if not sufficient_data:
            warnings.append(
                "Confluence data is insufficient."
            )

        if conflicting_factors > 0:
            warnings.append(
                "Conflicting confluence factors are present."
            )

        if direction in (
            ConfluenceDirection.NEUTRAL,
            ConfluenceDirection.UNKNOWN,
        ):
            warnings.append(
                "There is no actionable directional confluence."
            )

        if quality in (
            ConfluenceQuality.WEAK,
            ConfluenceQuality.CONFLICTED,
            ConfluenceQuality.UNKNOWN,
        ):
            warnings.append(
                "Confluence quality is below the preferred entry threshold."
            )

        warnings.append(
            "Confluence analysis does not calculate entry, stop loss, take profit, or position size."
        )

        return warnings

    # =========================================================
    # VALIDATION
    # =========================================================

    @staticmethod
    def _validate_threshold(
        value: float,
        name: str,
    ) -> float:

        if isinstance(value, bool):
            raise ValueError(
                f"{name} must be a finite number between 0 and 100."
            )

        if not isinstance(value, (int, float)):
            raise ValueError(
                f"{name} must be a finite number between 0 and 100."
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

        return value

    @staticmethod
    def _validate_inputs(
        *,
        timestamp: datetime,
        symbol: str,
        timeframe: str,
        mtf_score: float,
        support_resistance_score: float,
        trendline_score: float,
        volume_score: float,
        support_present: bool,
        resistance_present: bool,
        support_strong: bool,
        resistance_strong: bool,
        support_trendline_present: bool,
        resistance_trendline_present: bool,
        volume_confirmed: bool,
        bullish_factors: int,
        bearish_factors: int,
        neutral_factors: int,
        conflicting_factors: int,
        sufficient_data: bool,
    ) -> None:

        if not isinstance(timestamp, datetime):
            raise ValueError(
                "timestamp must be a datetime."
            )

        if (
            not isinstance(symbol, str)
            or not symbol.strip()
        ):
            raise ValueError(
                "symbol must be a non-empty string."
            )

        if (
            not isinstance(timeframe, str)
            or not timeframe.strip()
        ):
            raise ValueError(
                "timeframe must be a non-empty string."
            )

        for name, value in (
            ("mtf_score", mtf_score),
            (
                "support_resistance_score",
                support_resistance_score,
            ),
            ("trendline_score", trendline_score),
            ("volume_score", volume_score),
        ):
            SetupConfluenceResult._validate_score(
                value,
                name,
            )

        for name, value in (
            ("support_present", support_present),
            ("resistance_present", resistance_present),
            ("support_strong", support_strong),
            ("resistance_strong", resistance_strong),
            (
                "support_trendline_present",
                support_trendline_present,
            ),
            (
                "resistance_trendline_present",
                resistance_trendline_present,
            ),
            ("volume_confirmed", volume_confirmed),
            ("sufficient_data", sufficient_data),
        ):
            if not isinstance(value, bool):
                raise ValueError(
                    f"{name} must be a boolean."
                )

        for name, value in (
            ("bullish_factors", bullish_factors),
            ("bearish_factors", bearish_factors),
            ("neutral_factors", neutral_factors),
            ("conflicting_factors", conflicting_factors),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
            ):
                raise ValueError(
                    f"{name} must be a non-negative integer."
                )

            if value < 0:
                raise ValueError(
                    f"{name} must be non-negative."
                )


__all__ = [
    "ConfluenceDirection",
    "ConfluenceQuality",
    "ConfluenceReason",
    "ConfluenceReasonType",
    "ConfluenceEngineError",
    "SetupConfluenceEngine",
    "SetupConfluenceResult",
]