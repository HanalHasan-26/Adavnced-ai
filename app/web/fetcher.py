from __future__ import annotations

# Import URL parsing.
from urllib.parse import urlparse

# Import HTTP request support.
from urllib.request import Request, urlopen

# Import HTTP/network errors.
from urllib.error import HTTPError, URLError


class WebPageFetcher:

    # Default maximum number of bytes to download.
    DEFAULT_MAX_BYTES = 2_000_000

    # Browser-like user agent.
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
        max_bytes: int = DEFAULT_MAX_BYTES,
    ):

        if timeout <= 0:
            raise ValueError(
                "timeout must be greater than 0."
            )

        if max_bytes <= 0:
            raise ValueError(
                "max_bytes must be greater than 0."
            )

        self.timeout = timeout
        self.max_bytes = max_bytes

    def fetch(
        self,
        url: str,
    ) -> str:

        if not isinstance(url, str):
            raise ValueError(
                "url must be a string."
            )

        url = url.strip()

        if not url:
            raise ValueError(
                "url cannot be empty."
            )

        # Parse and validate the URL.
        parsed = urlparse(url)

        if parsed.scheme not in {
            "http",
            "https",
        }:
            raise ValueError(
                "url must use HTTP or HTTPS."
            )

        if not parsed.netloc:
            raise ValueError(
                "url must contain a hostname."
            )

        # Create the HTTP request.
        request = Request(
            url,
            headers={
                "User-Agent": self.USER_AGENT,
            },
        )

        try:

            # Open the webpage.
            with urlopen(
                request,
                timeout=self.timeout,
            ) as response:

                # Read only the configured maximum amount.
                data = response.read(
                    self.max_bytes + 1
                )

        except HTTPError as error:

            raise RuntimeError(
                "Webpage fetch failed with HTTP "
                f"status {error.code}."
            ) from error

        except URLError as error:

            raise RuntimeError(
                "Unable to connect to webpage."
            ) from error

        except TimeoutError as error:

            raise RuntimeError(
                "Webpage fetch timed out."
            ) from error

        # Protect against oversized responses.
        if len(data) > self.max_bytes:

            raise RuntimeError(
                "Webpage exceeds the maximum "
                "allowed size."
            )

        # Decode the webpage safely.
        return data.decode(
            "utf-8",
            errors="replace",
        )