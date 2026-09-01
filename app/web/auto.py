from __future__ import annotations

import re

from app.web.mode import WebMode


class WebAutoDecider:

    # =========================================================
    # EXPLICIT WEB INDICATORS
    # =========================================================

    WEB_INDICATORS = (
        "latest",
        "current",
        "today",
        "tonight",
        "this week",
        "this month",
        "this year",
        "right now",
        "recent",
        "news",
        "breaking",
        "live",
        "update",
        "updates",
        "price now",
        "current price",
        "weather",
        "forecast",
        "search the web",
        "search online",
        "look online",
        "look it up",
        "look this up",
        "online",
        "internet",
        "on the internet",
    )

    # =========================================================
    # TIME-SENSITIVE SUBJECTS
    # =========================================================

    TIME_SENSITIVE_TERMS = (
        "gold price",
        "silver price",
        "oil price",
        "bitcoin price",
        "ethereum price",
        "stock price",
        "share price",
        "exchange rate",
        "forex price",
        "xauusd",
        "xau/usd",
        "usd/inr",
        "eur/usd",
        "economic calendar",
        "nfp",
        "fomc",
        "cpi",
        "ppi",
        "pce",
        "interest rate",
    )

    # =========================================================
    # DIRECT WEB COMMANDS
    # =========================================================

    DIRECT_WEB_PATTERNS = (
        re.compile(
            r"^\s*web\s*:\s*(.+?)\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*\(\s*web\s*:\s*(.+?)\s*\)\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*\[\s*web\s*:\s*(.+?)\s*\]\s*$",
            re.IGNORECASE,
        ),
    )

    # =========================================================
    # UNKNOWN / EXTERNAL QUESTION INDICATORS
    # =========================================================

    EXTERNAL_QUESTION_PATTERNS = (
        r"\bwho\s+(?:was|is|were|are)\b",
        r"\bwhen\s+(?:was|is|did|does|were)\b",
        r"\bwhere\s+(?:was|is|did|does|were)\b",
        r"\bwhich\s+(?:person|company|country|organization|team)\b",
        r"\bhow\s+many\b",
        r"\bhow\s+much\b",
        r"\bwhat\s+is\s+the\b",
        r"\bwhat\s+was\s+the\b",
        r"\bwhat\s+are\s+the\b",
        r"\bwhat\s+were\s+the\b",
    )

    # =========================================================
    # DIRECT WEB COMMAND
    # =========================================================

    def extract_direct_web_query(
        self,
        query: str,
    ) -> str | None:

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        for pattern in self.DIRECT_WEB_PATTERNS:

            match = pattern.match(query)

            if match is not None:

                web_query = match.group(1).strip()

                if web_query:
                    return web_query

        return None

    # =========================================================
    # DIRECT WEB CHECK
    # =========================================================

    def is_direct_web_query(
        self,
        query: str,
    ) -> bool:

        return (
            self.extract_direct_web_query(
                query
            )
            is not None
        )

    # =========================================================
    # NORMALIZE WEB QUERY
    # =========================================================

    def normalize_web_query(
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

        direct_query = (
            self.extract_direct_web_query(
                query
            )
        )

        if direct_query is not None:
            return direct_query

        return query

    # =========================================================
    # CURRENT / WEB INDICATOR CHECK
    # =========================================================

    def contains_web_indicator(
        self,
        query: str,
    ) -> bool:

        query = query.lower()

        return any(
            indicator in query
            for indicator in self.WEB_INDICATORS
        )

    # =========================================================
    # TIME-SENSITIVE CHECK
    # =========================================================

    def is_time_sensitive(
        self,
        query: str,
    ) -> bool:

        query = query.strip().lower()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        return any(
            term in query
            for term in self.TIME_SENSITIVE_TERMS
        )

    # =========================================================
    # EXTERNAL QUESTION CHECK
    # =========================================================

    def looks_like_external_question(
        self,
        query: str,
    ) -> bool:

        query = query.strip().lower()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        for pattern in self.EXTERNAL_QUESTION_PATTERNS:

            if re.search(
                pattern,
                query,
                re.IGNORECASE,
            ):
                return True

        return False

    # =========================================================
    # SHOULD USE WEB
    # =========================================================

    def should_use_web(
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

        # -----------------------------------------------------
        # DIRECT WEB COMMAND
        # -----------------------------------------------------

        if self.is_direct_web_query(query):
            return True

        normalized = query.lower()

        # -----------------------------------------------------
        # EXPLICIT WEB INDICATORS
        # -----------------------------------------------------

        if self.contains_web_indicator(
            normalized
        ):
            return True

        # -----------------------------------------------------
        # TIME-SENSITIVE TOPICS
        # -----------------------------------------------------

        if self.is_time_sensitive(
            normalized
        ):
            return True

        return False

    # =========================================================
    # DECIDE
    # =========================================================

    def decide(
        self,
        query: str,
    ) -> WebMode:

        if self.should_use_web(query):
            return WebMode.WEB

        return WebMode.OFFLINE