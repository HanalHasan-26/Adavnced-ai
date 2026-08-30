import pytest

from app.web.search import (
    SearchResult,
    WebSearch,
    _DuckDuckGoParser,
)


def test_search_result_to_dict():

    result = SearchResult(
        title="Forex Trading",
        url="https://example.com/forex",
        snippet="Forex trading basics.",
    )

    assert result.to_dict() == {
        "title": "Forex Trading",
        "url": "https://example.com/forex",
        "snippet": "Forex trading basics.",
    }


def test_search_result_attributes():

    result = SearchResult(
        title="Test",
        url="https://example.com",
        snippet="Example",
    )

    assert result.title == "Test"
    assert result.url == "https://example.com"
    assert result.snippet == "Example"


def test_default_timeout():

    search = WebSearch()

    assert search.timeout == 10.0


def test_custom_timeout():

    search = WebSearch(
        timeout=20.0
    )

    assert search.timeout == 20.0


def test_invalid_timeout():

    with pytest.raises(
        ValueError,
        match="timeout must be greater than 0",
    ):

        WebSearch(
            timeout=0
        )


def test_negative_timeout():

    with pytest.raises(
        ValueError,
        match="timeout must be greater than 0",
    ):

        WebSearch(
            timeout=-1
        )


def test_non_string_query():

    search = WebSearch()

    with pytest.raises(
        ValueError,
        match="query must be a string",
    ):

        search.search(
            query=123
        )


def test_empty_query():

    search = WebSearch()

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):

        search.search(
            query=""
        )


def test_whitespace_query():

    search = WebSearch()

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):

        search.search(
            query="   "
        )


def test_zero_limit():

    search = WebSearch()

    with pytest.raises(
        ValueError,
        match="limit must be greater than 0",
    ):

        search.search(
            query="forex",
            limit=0,
        )


def test_negative_limit():

    search = WebSearch()

    with pytest.raises(
        ValueError,
        match="limit must be greater than 0",
    ):

        search.search(
            query="forex",
            limit=-1,
        )


def test_parser_extracts_result():

    html = """
    <a
        class="result__a"
        href="https://example.com/forex"
    >
        Forex Trading Basics
    </a>

    <div class="result__snippet">
        Learn the basics of forex trading.
    </div>
    """

    parser = _DuckDuckGoParser()

    parser.feed(html)
    parser.close()

    assert len(parser.results) == 1

    result = parser.results[0]

    assert result.title == (
        "Forex Trading Basics"
    )

    assert result.url == (
        "https://example.com/forex"
    )

    assert result.snippet == (
        "Learn the basics of forex trading."
    )


def test_parser_handles_result_without_snippet():

    html = """
    <a
        class="result__a"
        href="https://example.com/test"
    >
        Test Result
    </a>
    """

    parser = _DuckDuckGoParser()

    parser.feed(html)
    parser.close()

    assert len(parser.results) == 1

    result = parser.results[0]

    assert result.title == "Test Result"

    assert result.url == (
        "https://example.com/test"
    )

    assert result.snippet == ""


def test_parser_extracts_multiple_results():

    html = """
    <a
        class="result__a"
        href="https://example.com/one"
    >
        First Result
    </a>

    <div class="result__snippet">
        First snippet.
    </div>

    <a
        class="result__a"
        href="https://example.com/two"
    >
        Second Result
    </a>

    <div class="result__snippet">
        Second snippet.
    </div>
    """

    parser = _DuckDuckGoParser()

    parser.feed(html)
    parser.close()

    assert len(parser.results) == 2

    assert parser.results[0].title == (
        "First Result"
    )

    assert parser.results[1].title == (
        "Second Result"
    )


def test_parser_handles_empty_html():

    parser = _DuckDuckGoParser()

    parser.feed("")
    parser.close()

    assert parser.results == []


def test_parser_handles_result_without_url():

    html = """
    <a class="result__a">
        Result Without URL
    </a>
    """

    parser = _DuckDuckGoParser()

    parser.feed(html)
    parser.close()

    assert parser.results == []