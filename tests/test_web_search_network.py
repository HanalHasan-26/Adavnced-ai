from urllib.error import HTTPError, URLError
from unittest.mock import MagicMock, patch

import pytest

from app.web.search import WebSearch


def make_response(html: str):

    response = MagicMock()

    response.read.return_value = (
        html.encode("utf-8")
    )

    # Important:
    # urlopen() is used as a context manager.
    # Make __enter__ return the actual response.
    response.__enter__.return_value = response

    response.__exit__.return_value = False

    return response


def test_search_success():

    html = """
    <a
        class="result__a"
        href="https://example.com/forex"
    >
        Forex Trading
    </a>

    <div class="result__snippet">
        Forex trading basics.
    </div>
    """

    response = make_response(html)

    with patch(
        "app.web.search.urlopen",
        return_value=response,
    ):

        search = WebSearch()

        results = search.search(
            "forex trading"
        )

    assert len(results) == 1

    assert results[0].title == (
        "Forex Trading"
    )

    assert results[0].url == (
        "https://example.com/forex"
    )

    assert results[0].snippet == (
        "Forex trading basics."
    )


def test_search_sends_query():

    html = """
    <a
        class="result__a"
        href="https://example.com"
    >
        Test
    </a>
    """

    response = make_response(html)

    with patch(
        "app.web.search.urlopen",
        return_value=response,
    ) as mock_urlopen:

        search = WebSearch()

        search.search(
            "order block"
        )

    request = mock_urlopen.call_args.args[0]

    assert "order+block" in request.full_url


def test_search_uses_timeout():

    response = make_response("")

    with patch(
        "app.web.search.urlopen",
        return_value=response,
    ) as mock_urlopen:

        search = WebSearch(
            timeout=25.0
        )

        search.search(
            "forex"
        )

    assert mock_urlopen.call_args.kwargs[
        "timeout"
    ] == 25.0


def test_search_handles_http_error():

    error = HTTPError(
        url="https://example.com",
        code=500,
        msg="Server Error",
        hdrs=None,
        fp=None,
    )

    with patch(
        "app.web.search.urlopen",
        side_effect=error,
    ):

        search = WebSearch()

        with pytest.raises(
            RuntimeError,
            match="HTTP status 500",
        ):

            search.search(
                "forex"
            )


def test_search_handles_connection_error():

    error = URLError(
        "Connection failed"
    )

    with patch(
        "app.web.search.urlopen",
        side_effect=error,
    ):

        search = WebSearch()

        with pytest.raises(
            RuntimeError,
            match="Unable to connect",
        ):

            search.search(
                "forex"
            )


def test_search_handles_timeout():

    with patch(
        "app.web.search.urlopen",
        side_effect=TimeoutError(),
    ):

        search = WebSearch()

        with pytest.raises(
            RuntimeError,
            match="timed out",
        ):

            search.search(
                "forex"
            )


def test_search_handles_malformed_html():

    html = """
    <html>
        <body>

            <a
                class="result__a"
                href="https://example.com"
            >
                Broken Result
    """

    response = make_response(html)

    with patch(
        "app.web.search.urlopen",
        return_value=response,
    ):

        search = WebSearch()

        results = search.search(
            "forex"
        )

    assert len(results) == 1

    assert results[0].title == (
        "Broken Result"
    )


def test_search_returns_empty_for_no_results():

    html = """
    <html>
        <body>

            <p>
                No search results.
            </p>

        </body>
    </html>
    """

    response = make_response(html)

    with patch(
        "app.web.search.urlopen",
        return_value=response,
    ):

        search = WebSearch()

        results = search.search(
            "something"
        )

    assert results == []


def test_search_respects_limit():

    html = """
    <a
        class="result__a"
        href="https://example.com/1"
    >
        Result One
    </a>

    <a
        class="result__a"
        href="https://example.com/2"
    >
        Result Two
    </a>

    <a
        class="result__a"
        href="https://example.com/3"
    >
        Result Three
    </a>

    <a
        class="result__a"
        href="https://example.com/4"
    >
        Result Four
    </a>
    """

    response = make_response(html)

    with patch(
        "app.web.search.urlopen",
        return_value=response,
    ):

        search = WebSearch()

        results = search.search(
            "forex",
            limit=2,
        )

    assert len(results) == 2

    assert results[0].title == (
        "Result One"
    )

    assert results[1].title == (
        "Result Two"
    )


def test_search_uses_user_agent():

    response = make_response("")

    with patch(
        "app.web.search.urlopen",
        return_value=response,
    ) as mock_urlopen:

        search = WebSearch()

        search.search(
            "forex"
        )

    request = mock_urlopen.call_args.args[0]

    user_agent = request.get_header(
        "User-agent"
    )

    assert user_agent is not None

    assert "Mozilla" in user_agent