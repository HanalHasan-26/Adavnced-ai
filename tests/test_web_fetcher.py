from unittest.mock import Mock, patch

import pytest

from app.web.fetcher import WebPageFetcher


def make_response(
    content: bytes,
):
    response = Mock()

    response.__enter__ = Mock(
        return_value=response
    )

    response.__exit__ = Mock(
        return_value=False
    )

    response.read.return_value = content

    return response


def test_fetch_returns_html():

    response = make_response(
        b"<html>Hello</html>"
    )

    with patch(
        "app.web.fetcher.urlopen",
        return_value=response,
    ):

        fetcher = WebPageFetcher()

        result = fetcher.fetch(
            "https://example.com"
        )

    assert result == (
        "<html>Hello</html>"
    )


def test_fetch_sends_user_agent():

    response = make_response(
        b"test"
    )

    with patch(
        "app.web.fetcher.urlopen",
        return_value=response,
    ) as mock_urlopen:

        fetcher = WebPageFetcher()

        fetcher.fetch(
            "https://example.com"
        )

    request = mock_urlopen.call_args.args[0]

    assert request.get_header(
        "User-agent"
    ) == fetcher.USER_AGENT


def test_fetch_uses_timeout():

    response = make_response(
        b"test"
    )

    with patch(
        "app.web.fetcher.urlopen",
        return_value=response,
    ) as mock_urlopen:

        fetcher = WebPageFetcher(
            timeout=25.0
        )

        fetcher.fetch(
            "https://example.com"
        )

    assert (
        mock_urlopen.call_args.kwargs[
            "timeout"
        ]
        == 25.0
    )


def test_fetch_rejects_empty_url():

    fetcher = WebPageFetcher()

    with pytest.raises(
        ValueError,
        match="url cannot be empty",
    ):
        fetcher.fetch("")


def test_fetch_rejects_non_string_url():

    fetcher = WebPageFetcher()

    with pytest.raises(
        ValueError,
        match="url must be a string",
    ):
        fetcher.fetch(None)


def test_fetch_rejects_invalid_scheme():

    fetcher = WebPageFetcher()

    with pytest.raises(
        ValueError,
        match="HTTP or HTTPS",
    ):
        fetcher.fetch(
            "ftp://example.com"
        )


def test_fetch_rejects_missing_hostname():

    fetcher = WebPageFetcher()

    with pytest.raises(
        ValueError,
        match="hostname",
    ):
        fetcher.fetch(
            "https://"
        )


def test_fetch_handles_http_error():

    from urllib.error import HTTPError

    error = HTTPError(
        "https://example.com",
        404,
        "Not Found",
        {},
        None,
    )

    with patch(
        "app.web.fetcher.urlopen",
        side_effect=error,
    ):

        fetcher = WebPageFetcher()

        with pytest.raises(
            RuntimeError,
            match="HTTP status 404",
        ):
            fetcher.fetch(
                "https://example.com"
            )


def test_fetch_handles_connection_error():

    from urllib.error import URLError

    with patch(
        "app.web.fetcher.urlopen",
        side_effect=URLError(
            "connection failed"
        ),
    ):

        fetcher = WebPageFetcher()

        with pytest.raises(
            RuntimeError,
            match="Unable to connect",
        ):
            fetcher.fetch(
                "https://example.com"
            )


def test_fetch_handles_timeout():

    with patch(
        "app.web.fetcher.urlopen",
        side_effect=TimeoutError(),
    ):

        fetcher = WebPageFetcher()

        with pytest.raises(
            RuntimeError,
            match="timed out",
        ):
            fetcher.fetch(
                "https://example.com"
            )


def test_fetch_rejects_oversized_page():

    response = make_response(
        b"123456789"
    )

    with patch(
        "app.web.fetcher.urlopen",
        return_value=response,
    ):

        fetcher = WebPageFetcher(
            max_bytes=5
        )

        with pytest.raises(
            RuntimeError,
            match="maximum allowed size",
        ):
            fetcher.fetch(
                "https://example.com"
            )


def test_fetch_decodes_invalid_utf8_safely():

    response = make_response(
        b"hello \xff world"
    )

    with patch(
        "app.web.fetcher.urlopen",
        return_value=response,
    ):

        fetcher = WebPageFetcher()

        result = fetcher.fetch(
            "https://example.com"
        )

    assert "hello" in result
    assert "world" in result