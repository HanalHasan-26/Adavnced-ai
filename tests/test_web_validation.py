from app.web.search import SearchResult
from app.web.validation import WebResultValidator


def test_valid_result_is_accepted():

    validator = WebResultValidator()

    result = SearchResult(
        title="Forex Trading",
        url="https://example.com/forex",
        snippet="Forex trading basics.",
    )

    validated = validator.validate(
        result
    )

    assert validated is not None
    assert validated.title == "Forex Trading"
    assert validated.url == "https://example.com/forex"
    assert validated.snippet == "Forex trading basics."


def test_whitespace_is_removed():

    validator = WebResultValidator()

    result = SearchResult(
        title="  Forex Trading  ",
        url="  https://example.com/forex  ",
        snippet="  Forex basics.  ",
    )

    validated = validator.validate(
        result
    )

    assert validated is not None
    assert validated.title == "Forex Trading"
    assert validated.url == "https://example.com/forex"
    assert validated.snippet == "Forex basics."


def test_empty_title_is_rejected():

    validator = WebResultValidator()

    result = SearchResult(
        title="",
        url="https://example.com",
        snippet="Test",
    )

    assert validator.validate(
        result
    ) is None


def test_empty_url_is_rejected():

    validator = WebResultValidator()

    result = SearchResult(
        title="Test",
        url="",
        snippet="Test",
    )

    assert validator.validate(
        result
    ) is None


def test_http_url_is_accepted():

    validator = WebResultValidator()

    result = SearchResult(
        title="Test",
        url="http://example.com",
        snippet="Test",
    )

    assert validator.validate(
        result
    ) is not None


def test_https_url_is_accepted():

    validator = WebResultValidator()

    result = SearchResult(
        title="Test",
        url="https://example.com",
        snippet="Test",
    )

    assert validator.validate(
        result
    ) is not None


def test_non_http_url_is_rejected():

    validator = WebResultValidator()

    result = SearchResult(
        title="Test",
        url="javascript:alert(1)",
        snippet="Test",
    )

    assert validator.validate(
        result
    ) is None


def test_missing_hostname_is_rejected():

    validator = WebResultValidator()

    result = SearchResult(
        title="Test",
        url="https:///missing-host",
        snippet="Test",
    )

    assert validator.validate(
        result
    ) is None


def test_empty_snippet_is_allowed():

    validator = WebResultValidator()

    result = SearchResult(
        title="Test",
        url="https://example.com",
        snippet="",
    )

    validated = validator.validate(
        result
    )

    assert validated is not None
    assert validated.snippet == ""


def test_invalid_object_is_rejected():

    validator = WebResultValidator()

    assert validator.validate(
        None
    ) is None


def test_validate_many_keeps_only_valid_results():

    validator = WebResultValidator()

    results = [
        SearchResult(
            title="Valid One",
            url="https://example.com/1",
            snippet="One",
        ),
        SearchResult(
            title="",
            url="https://example.com/2",
            snippet="Invalid",
        ),
        SearchResult(
            title="Valid Two",
            url="http://example.com/3",
            snippet="Two",
        ),
    ]

    validated = validator.validate_many(
        results
    )

    assert len(validated) == 2
    assert validated[0].title == "Valid One"
    assert validated[1].title == "Valid Two"