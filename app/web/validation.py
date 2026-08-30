from __future__ import annotations

# Import URL parsing utilities.
from urllib.parse import urlparse

# Import the existing search result model.
from app.web.search import SearchResult


class WebResultValidator:

    # Validate and clean one web-search result.
    def validate(
        self,
        result: SearchResult,
    ) -> SearchResult | None:

        # Make sure the object is a SearchResult.
        if not isinstance(
            result,
            SearchResult,
        ):
            return None

        # Clean the fields.
        title = result.title.strip()
        url = result.url.strip()
        snippet = result.snippet.strip()

        # A title is required.
        if not title:
            return None

        # A URL is required.
        if not url:
            return None

        # Parse the URL.
        parsed = urlparse(url)

        # Only accept HTTP and HTTPS URLs.
        if parsed.scheme not in {
            "http",
            "https",
        }:
            return None

        # A network location is required.
        if not parsed.netloc:
            return None

        # Return a cleaned result.
        return SearchResult(
            title=title,
            url=url,
            snippet=snippet,
        )

    # Validate multiple search results.
    def validate_many(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:

        # Store valid results.
        validated = []

        # Validate every result.
        for result in results:

            cleaned = self.validate(
                result
            )

            if cleaned is not None:
                validated.append(
                    cleaned
                )

        return validated