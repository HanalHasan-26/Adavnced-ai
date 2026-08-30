from __future__ import annotations

from app.web.mode import WebMode


class WebAutoDecider:

    # Phrases that strongly indicate the user wants
    # current or external information.
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
        "weather",
        "search the web",
        "look online",
        "online",
        "internet",
    )

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

        return any(
            indicator in query
            for indicator in self.WEB_INDICATORS
        )

    def decide(
        self,
        query: str,
    ) -> WebMode:

        if self.should_use_web(query):
            return WebMode.WEB

        return WebMode.OFFLINE