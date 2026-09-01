from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
        }


class _DuckDuckGoParser(HTMLParser):
    def __init__(self):
        super().__init__()

        self.results: list[SearchResult] = []

        self.current_title: str | None = None
        self.current_url: str | None = None
        self.current_snippet: str | None = None

        self.inside_title = False
        self.inside_snippet = False

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ):
        attributes = dict(attrs)
        classes = attributes.get("class", "") or ""

        if (
            tag == "a"
            and "result__a" in classes
        ):
            self._finish_current_result()

            self.current_title = ""
            self.current_url = (
                attributes.get("href") or ""
            )
            self.current_snippet = ""

            self.inside_title = True

        elif (
            tag in {"a", "div"}
            and "result__snippet" in classes
        ):
            if self.current_title is not None:
                self.current_snippet = ""
                self.inside_snippet = True

    def handle_endtag(self, tag: str):
        if (
            self.inside_title
            and tag == "a"
        ):
            self.inside_title = False

        if (
            self.inside_snippet
            and tag in {"a", "div"}
        ):
            self.inside_snippet = False

    def handle_data(self, data: str):
        if self.inside_title:
            self.current_title = (
                self.current_title or ""
            ) + data

        elif self.inside_snippet:
            self.current_snippet = (
                self.current_snippet or ""
            ) + data

    def _finish_current_result(self):
        if self.current_title is None:
            return

        if not self.current_url:
            self.current_title = None
            self.current_url = None
            self.current_snippet = None
            self.inside_title = False
            self.inside_snippet = False
            return

        self.results.append(
            SearchResult(
                title=self.current_title.strip(),
                url=self.current_url.strip(),
                snippet=(
                    self.current_snippet or ""
                ).strip(),
            )
        )

        self.current_title = None
        self.current_url = None
        self.current_snippet = None
        self.inside_title = False
        self.inside_snippet = False

    def close(self):
        self._finish_current_result()
        return super().close()


class WebSearch:
    SEARCH_URL = (
        "https://html.duckduckgo.com/html/"
    )

    USER_AGENT = (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    )

    def __init__(
        self,
        timeout: float = 10.0,
    ):
        if timeout <= 0:
            raise ValueError(
                "timeout must be greater than 0."
            )

        self.timeout = timeout

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[SearchResult]:
        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        params = urlencode(
            {"q": query}
        )

        url = (
            f"{self.SEARCH_URL}?{params}"
        )

        request = Request(
            url,
            headers={
                "User-Agent": self.USER_AGENT,
                "Accept-Language": "en-US,en;q=0.9",
            },
        )

        try:
            with urlopen(
                request,
                timeout=self.timeout,
            ) as response:
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

        parser = _DuckDuckGoParser()

        parser.feed(html)
        parser.close()

        results = parser.results[:limit]

        for result in results:
            result.url = urljoin(
                self.SEARCH_URL,
                result.url,
            )

        return results
