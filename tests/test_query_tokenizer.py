# Import pytest for validation tests.
import pytest

# Import the query tokenizer.
from app.knowledge.retrieval.query_tokenizer import QueryTokenizer


# Test that a query is split into individual terms.
def test_tokenize_query():

    # Create the tokenizer.
    tokenizer = QueryTokenizer()

    # Tokenize a multi-word query.
    result = tokenizer.tokenize(
        "support resistance"
    )

    # Verify the individual terms.
    assert result == [
        "support",
        "resistance",
    ]


# Test that extra whitespace is handled.
def test_tokenize_extra_whitespace():

    # Create the tokenizer.
    tokenizer = QueryTokenizer()

    # Tokenize a query with repeated whitespace.
    result = tokenizer.tokenize(
        "  support    resistance  "
    )

    # Verify clean terms.
    assert result == [
        "support",
        "resistance",
    ]


# Test that duplicate terms are removed.
def test_tokenize_removes_duplicates():

    # Create the tokenizer.
    tokenizer = QueryTokenizer()

    # Tokenize a query containing duplicates.
    result = tokenizer.tokenize(
        "support resistance support support"
    )

    # Verify duplicates were removed.
    assert result == [
        "support",
        "resistance",
    ]


# Test that term order is preserved.
def test_tokenize_preserves_order():

    # Create the tokenizer.
    tokenizer = QueryTokenizer()

    # Tokenize the query.
    result = tokenizer.tokenize(
        "gold support resistance"
    )

    # Verify the original order.
    assert result == [
        "gold",
        "support",
        "resistance",
    ]


# Test that an empty query returns no terms.
def test_tokenize_empty_query():

    # Create the tokenizer.
    tokenizer = QueryTokenizer()

    # Tokenize an empty query.
    result = tokenizer.tokenize("")

    # Verify that no terms are returned.
    assert result == []


# Test that a whitespace-only query returns no terms.
def test_tokenize_whitespace_query():

    # Create the tokenizer.
    tokenizer = QueryTokenizer()

    # Tokenize whitespace-only input.
    result = tokenizer.tokenize("     ")

    # Verify that no terms are returned.
    assert result == []


# Test that a single term is preserved.
def test_tokenize_single_term():

    # Create the tokenizer.
    tokenizer = QueryTokenizer()

    # Tokenize one term.
    result = tokenizer.tokenize("gold")

    # Verify the result.
    assert result == ["gold"]


# Test that numbers are preserved.
def test_tokenize_numbers():

    # Create the tokenizer.
    tokenizer = QueryTokenizer()

    # Tokenize a query containing numbers.
    result = tokenizer.tokenize(
        "xauusd 2500"
    )

    # Verify that numbers remain.
    assert result == [
        "xauusd",
        "2500",
    ]


# Test that non-string input is rejected.
def test_tokenize_rejects_non_string():

    # Create the tokenizer.
    tokenizer = QueryTokenizer()

    # Verify that invalid input raises TypeError.
    with pytest.raises(AttributeError):
        tokenizer.tokenize(123)