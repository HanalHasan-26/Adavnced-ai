from __future__ import annotations

import re

from app.web.mode import WebMode


class WebAutoDecider:

    # ---------------------------------------------------------
    # EXPLICIT WEB COMMAND
    # ---------------------------------------------------------

    WEB_PREFIXES = (
        "web:",
        "web :",
    )

    OFFLINE_PREFIXES = (
        "offline:",
        "offline :",
    )

    # ---------------------------------------------------------
    # CURRENT / EXTERNAL INFORMATION
    # ---------------------------------------------------------

    WEB_INDICATORS = (
        "latest",
        "current",
        "today",
        "tonight",
        "this week",
        "this month",
        "right now",
        "recent",
        "news",
        "breaking",
        "live",
        "update",
        "updates",
        "price now",
        "current price",
        "current population",
        "current weather",
        "weather today",
        "search the web",
        "look online",
        "look it up online",
        "online",
        "internet",
    )

    # ---------------------------------------------------------
    # CONSTRUCTOR
    # ---------------------------------------------------------

    def __init__(self) -> None:
        pass

    # =========================================================
    # COMMAND DETECTION
    # =========================================================

    def is_direct_web_query(
        self,
        query: str,
    ) -> bool:

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        query = query.strip().lower()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        return query.startswith(
            self.WEB_PREFIXES
        )

    def is_explicit_offline_query(
        self,
        query: str,
    ) -> bool:

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        query = query.strip().lower()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        return query.startswith(
            self.OFFLINE_PREFIXES
        )

    # =========================================================
    # REMOVE COMMAND PREFIX
    # =========================================================

    def clean_query(
        self,
        query: str,
    ) -> str:

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        lowered = query.lower()

        for prefix in (
            *self.WEB_PREFIXES,
            *self.OFFLINE_PREFIXES,
        ):

            if lowered.startswith(prefix):

                cleaned = query[
                    len(prefix):
                ].strip()

                if not cleaned:
                    raise ValueError(
                        "query cannot be empty after "
                        "the web/offline command."
                    )

                return cleaned

        return query

    # =========================================================
    # CURRENT INFORMATION DETECTION
    # =========================================================

    def should_use_web(
        self,
        query: str,
    ) -> bool:

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        query = query.strip().lower()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        # Explicit web command always wins.
        if self.is_direct_web_query(query):
            return True

        # Explicit offline command never uses web.
        if self.is_explicit_offline_query(query):
            return False

        return any(
            indicator in query
            for indicator in self.WEB_INDICATORS
        )

    # =========================================================
    # MODE DECISION
    # =========================================================

    def decide(
        self,
        query: str,
    ) -> WebMode:

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        if self.is_direct_web_query(query):
            return WebMode.WEB

        if self.is_explicit_offline_query(query):
            return WebMode.OFFLINE

        if self.should_use_web(query):
            return WebMode.WEB

        return WebMode.OFFLINE

    # =========================================================
    # AUTO WEB FALLBACK
    # =========================================================

    def should_fallback_to_web(
        self,
        query: str,
    ) -> bool:

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        if self.is_direct_web_query(query):
            return True

        if self.is_explicit_offline_query(query):
            return False

        # Explicitly current/external questions should
        # go directly to web.
        if self.should_use_web(query):
            return True

        # Normal unknown questions are allowed to fall
        # back to web after local sources fail.
        return True