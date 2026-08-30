# Import pytest for validation tests.
import pytest

# Import the query normalizer.
from app.knowledge.retrieval.query_normalizer import QueryNormalizer


# Test that text is converted to lowercase.
def test_normalize_lowercase():

    # Create the normalizer.
    normalizer = QueryNormalizer()

    # Normalize uppercase text.
    result = normalizer.normalize("SUPPORT")

    # Verify lowercase output.
    assert result == "support"


# Test that leading and trailing whitespace is removed.
def test_normalize_strips_whitespace():

    # Create the normalizer.
    normalizer = QueryNormalizer()

    # Normalize text containing surrounding whitespace.
    result = normalizer.normalize("   support   ")

    # Verify that surrounding whitespace was removed.
    assert result == "support"


# Test that repeated whitespace is collapsed.
def test_normalize_collapses_whitespace():

    # Create the normalizer.
    normalizer = QueryNormalizer()

    # Normalize text containing repeated spaces.
    result = normalizer.normalize(
        "support    and     resistance"
    )

    # Verify that only single spaces remain.
    assert result == "support and resistance"


# Test that punctuation is removed.
def test_normalize_removes_punctuation():

    # Create the normalizer.
    normalizer = QueryNormalizer()

    # Normalize punctuation-heavy text.
    result = normalizer.normalize(
        "Support, resistance! What?"
    )

    # Verify the normalized result.
    assert result == "support resistance what"


# Test that symbols are separated rather than joined.
def test_normalize_separates_symbols():

    # Create the normalizer.
    normalizer = QueryNormalizer()

    # Normalize text containing a slash.
    result = normalizer.normalize(
        "Gold/XAUUSD"
    )

    # Verify that the slash becomes a separator.
    assert result == "gold xauusd"


# Test that an empty string remains empty.
def test_normalize_empty_string():

    # Create the normalizer.
    normalizer = QueryNormalizer()

    # Normalize an empty query.
    result = normalizer.normalize("")

    # Verify the result.
    assert result == ""


# Test that whitespace-only text becomes empty.
def test_normalize_whitespace_only():

    # Create the normalizer.
    normalizer = QueryNormalizer()

    # Normalize whitespace-only input.
    result = normalizer.normalize("     ")

    # Verify the result.
    assert result == ""


# Test that newline and tab characters are normalized.
def test_normalize_newlines_and_tabs():

    # Create the normalizer.
    normalizer = QueryNormalizer()

    # Normalize text containing different whitespace characters.
    result = normalizer.normalize(
        "support\n\tand\r\nresistance"
    )

    # Verify normalized spacing.
    assert result == "support and resistance"


# Test that numbers are preserved.
def test_normalize_preserves_numbers():

    # Create the normalizer.
    normalizer = QueryNormalizer()

    # Normalize text containing numbers.
    result = normalizer.normalize(
        "XAUUSD 2500"
    )

    # Verify that numbers remain.
    assert result == "xauusd 2500"


# Test that non-string input is rejected.
def test_normalize_rejects_non_string():

    # Create the normalizer.
    normalizer = QueryNormalizer()

    # Verify that a non-string query raises TypeError.
    with pytest.raises(TypeError):
        normalizer.normalize(123)