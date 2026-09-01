from __future__ import annotations

from urllib.parse import urlparse
from app.web.search import SearchResult


class WebResultValidator:
    NON_NEWS_TERMS = (
        "calculator",
        "currency converter",
        "exchange rate",
        "gold calculator",
        "price calculator",
        "live price of gold",
        "gold prices worldwide",
    )

    def validate(
        self,
        result: SearchResult,
        query: str | None = None,
    ) -> SearchResult | None:
        if not isinstance(result, SearchResult):
            return None

        title = result.title.strip()
        url = result.url.strip()
        snippet = result.snippet.strip()

        if not title or not url:
            return None

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return None

        if query and self._is_news_query(query):
            combined = f"{title} {url} {snippet}".lower()
            if any(term in combined for term in self.NON_NEWS_TERMS):
                return None

        return SearchResult(title=title, url=url, snippet=snippet)

    @staticmethod
    def _is_news_query(query: str) -> bool:
        q = query.lower()
        return any(
            x in q
            for x in (
                "news", "latest", "today", "breaking",
                "recent", "headline", "headlines", "updates",
            )
        )

    def validate_many(
        self,
        results: list[SearchResult],
        query: str | None = None,
    ) -> list[SearchResult]:
        validated = []
        for result in results:
            cleaned = self.validate(result, query=query)
            if cleaned is not None:
                validated.append(cleaned)
        return validated
