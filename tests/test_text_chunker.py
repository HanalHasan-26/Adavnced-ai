# Import pytest so we can test invalid configurations.
import pytest

# Import the text chunker we want to test.
from app.knowledge.chunking.text_chunker import TextChunker


# Test that normal text is split into multiple chunks.
def test_chunk_text():

    # Create a chunker with a small size so the test is easy to verify.
    chunker = TextChunker(
        chunk_size=10,
        overlap=2,
    )

    # Create text longer than one chunk.
    text = "abcdefghijklmnopqrstuvwxyz"

    # Split the text into chunks.
    chunks = chunker.chunk(text)

    # Make sure multiple chunks were created.
    assert len(chunks) > 1

    # Make sure every chunk is no longer than the configured size.
    assert all(len(chunk) <= 10 for chunk in chunks)


# Test that chunks contain overlapping text.
def test_chunk_overlap():

    # Create a chunker with a 10-character size and 2-character overlap.
    chunker = TextChunker(
        chunk_size=10,
        overlap=2,
    )

    # Create text long enough to produce multiple chunks.
    text = "abcdefghijklmnopqrstuvwxyz"

    # Split the text.
    chunks = chunker.chunk(text)

    # Make sure there are at least two chunks.
    assert len(chunks) >= 2

    # The last two characters of the first chunk should
    # appear at the beginning of the next chunk.
    assert chunks[0][-2:] == chunks[1][:2]


# Test that empty text produces no chunks.
def test_empty_text_returns_no_chunks():

    # Create the chunker.
    chunker = TextChunker()

    # Chunk an empty string.
    chunks = chunker.chunk("")

    # Make sure the result is an empty list.
    assert chunks == []


# Test that whitespace-only text produces no chunks.
def test_whitespace_text_returns_no_chunks():

    # Create the chunker.
    chunker = TextChunker()

    # Chunk text containing only whitespace.
    chunks = chunker.chunk("   \n   \t   ")

    # Make sure no meaningless chunks are created.
    assert chunks == []


# Test that short text remains one chunk.
def test_short_text_returns_one_chunk():

    # Create a chunker with a large chunk size.
    chunker = TextChunker(
        chunk_size=100,
        overlap=20,
    )

    # Create text shorter than the chunk size.
    text = "This is short text."

    # Split the text.
    chunks = chunker.chunk(text)

    # Make sure exactly one chunk was created.
    assert chunks == [text]


# Test that chunk_size must be positive.
def test_invalid_chunk_size():

    # Make sure a zero chunk size is rejected.
    with pytest.raises(ValueError):

        # Try to create an invalid chunker.
        TextChunker(
            chunk_size=0,
            overlap=0,
        )


# Test that overlap cannot be negative.
def test_negative_overlap():

    # Make sure negative overlap is rejected.
    with pytest.raises(ValueError):

        # Try to create an invalid chunker.
        TextChunker(
            chunk_size=100,
            overlap=-1,
        )


# Test that overlap must be smaller than chunk size.
def test_overlap_too_large():

    # Make sure overlap equal to chunk size is rejected.
    with pytest.raises(ValueError):

        # Try to create an invalid chunker.
        TextChunker(
            chunk_size=100,
            overlap=100,
        )


# Test that overlap larger than chunk size is rejected.
def test_overlap_larger_than_chunk_size():

    # Make sure excessive overlap is rejected.
    with pytest.raises(ValueError):

        # Try to create an invalid chunker.
        TextChunker(
            chunk_size=100,
            overlap=150,
        )