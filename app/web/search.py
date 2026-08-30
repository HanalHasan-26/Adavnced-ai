from __future__ import annotations

# Import dataclass support.
from dataclasses import dataclass

# Import HTML parsing support.
from html.parser import HTMLParser

# Import URL handling.
from urllib.parse import urlencode, urljoin

# Import HTTP request support.
from urllib.request import Request, urlopen

# Import URL-related errors.
from urllib.error import HTTPError, URLError


# Store one web-search result.
@dataclass
class SearchResult:

    title: str
    url: str
    snippet: str

    def to_dict(
        self,
    ) -> dict[str, str]:

        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
        }


# Parse DuckDuckGo HTML search results.
class _DuckDuckGoParser(HTMLParser):

    def __init__(self):

        super().__init__()

        # Store completed search results.
        self.results: list[SearchResult] = []

        # Store the currently parsed result.
        self.current_title: str | None = None
        self.current_url: str | None = None
        self.current_snippet: str | None = None

        # Track whether we are inside a title.
        self.inside_title = False

        # Track whether we are inside a snippet.
        self.inside_snippet = False

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ):

        # Convert attributes into a dictionary.
        attributes = dict(attrs)

        # Get the CSS class.
        classes = attributes.get(
            "class",
            "",
        )

        # Detect a search-result title.
        if (
            tag == "a"
            and "result__a" in classes
        ):

            # If another result is still waiting to be
            # completed, save it before starting a new one.
            self._finish_current_result()

            # Start a new result.
            self.current_title = ""
            self.current_url = (
                attributes.get("href")
                or ""
            )
            self.current_snippet = ""

            # Start collecting title text.
            self.inside_title = True

        # Detect a search-result snippet.
        elif (
            tag in {"a", "div"}
            and "result__snippet" in classes
        ):

            # Make sure a result exists.
            if self.current_title is not None:

                # Start collecting snippet text.
                self.current_snippet = ""
                self.inside_snippet = True

    def handle_endtag(
        self,
        tag: str,
    ):

        # Finish title collection.
        if (
            self.inside_title
            and tag == "a"
        ):

            self.inside_title = False

        # Finish snippet collection.
        if (
            self.inside_snippet
            and tag in {"a", "div"}
        ):

            self.inside_snippet = False

    def handle_data(
        self,
        data: str,
    ):

        # Collect title text.
        if self.inside_title:

            self.current_title = (
                self.current_title or ""
            ) + data

        # Collect snippet text.
        elif self.inside_snippet:

            self.current_snippet = (
                self.current_snippet or ""
            ) + data

    def _finish_current_result(
        self,
    ) -> None:

        # Nothing to finish.
        if self.current_title is None:
            return

        # A result must have a URL.
        if not self.current_url:
            self.current_title = None
            self.current_url = None
            self.current_snippet = None
            self.inside_title = False
            self.inside_snippet = False
            return

        # Create the result.
        result = SearchResult(
            title=self.current_title.strip(),
            url=self.current_url.strip(),
            snippet=(
                self.current_snippet or ""
            ).strip(),
        )

        # Store the completed result.
        self.results.append(result)

        # Reset the current result.
        self.current_title = None
        self.current_url = None
        self.current_snippet = None

        self.inside_title = False
        self.inside_snippet = False

    def close(self):

        # Finish the final result because there may be
        # no following result to trigger the normal flush.
        self._finish_current_result()

        # Finish normal HTML parsing.
        return super().close()


# Create the web-search client.
class WebSearch:

    # DuckDuckGo's HTML search endpoint.
    SEARCH_URL = (
        "https://html.duckduckgo.com/html/"
    )

    # Initialize the search client.
    def __init__(
        self,
        timeout: float = 10.0,
    ):

        # Make sure the timeout is valid.
        if timeout <= 0:
            raise ValueError(
                "timeout must be greater than 0."
            )

        self.timeout = timeout

    # Search the web.
    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[SearchResult]:

        # Validate the query.
        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        # Remove unnecessary whitespace.
        query = query.strip()

        # Reject empty queries.
        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        # Validate the result limit.
        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        # Build the search URL.
        params = urlencode(
            {
                "q": query,
            }
        )

        url = (
            f"{self.SEARCH_URL}?{params}"
        )

        # Create a browser-like HTTP request.
        request = Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/126.0 Safari/537.36"
                )
            },
        )

        try:

            # Request the search page.
            with urlopen(
                request,
                timeout=self.timeout,
            ) as response:

                # Read and decode the response.
                html = response.read().decode(
                    "utf-8",
                    errors="replace",
                )

        except HTTPError as error:

            raise RuntimeError(
                "Web search failed with HTTP "
                f"status {error.code}."
            ) from error

        except URLError as error:

            raise RuntimeError(
                "Unable to connect to the web "
                "search service."
            ) from error

        except TimeoutError as error:

            raise RuntimeError(
                "Web search timed out."
            ) from error

        # Parse the returned HTML.
        parser = _DuckDuckGoParser()

        parser.feed(html)

        # Explicitly close the parser so the final
        # result is flushed.
        parser.close()

        # Limit the number of returned results.
        results = parser.results[:limit]

        # Convert relative URLs into absolute URLs.
        for result in results:

            result.url = urljoin(
                self.SEARCH_URL,
                result.url,
            )

        return results