from __future__ import annotations

from app.web.fetcher import WebPageFetcher
from app.web.extractor import HTMLTextExtractor
from app.web.security import WebContentSecurity
from app.web.validation import WebResultValidator
from app.web.search import SearchResult


class WebResearcher:
    def __init__(
        self,
        fetcher: WebPageFetcher | None = None,
        extractor: HTMLTextExtractor | None = None,
        security: WebContentSecurity | None = None,
        validator: WebResultValidator | None = None,
    ):
        self.fetcher = fetcher or WebPageFetcher()
        self.extractor = extractor or HTMLTextExtractor()
        self.security = security or WebContentSecurity()
        self.validator = validator or WebResultValidator()

    def read_result(self, result: SearchResult) -> str:
        result = self.validator.validate(result)
        if result is None:
            return ""

        try:
            html = self.fetcher.fetch(result.url)
            text = self.extractor.extract(html)
            if not text:
                return ""
            return self.security.build_safe_context(text)
        except Exception:
            return ""

    def research(
        self,
        results: list[SearchResult],
        max_sources: int = 5,
    ) -> str:
        if max_sources <= 0:
            raise ValueError("max_sources must be greater than 0.")

        sections = []
        validated = self.validator.validate_many(results)

        for index, result in enumerate(validated[:max_sources], start=1):
            content = self.read_result(result)
            if not content:
                continue

            sections.append(
                f"[Web Source {index}]\n"
                f"Title: {result.title}\n"
                f"URL: {result.url}\n\n"
                f"{content}"
            )

        return "\n\n".join(sections)
