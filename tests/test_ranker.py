# Import pytest so we can test validation errors.
import pytest

# Import the knowledge chunk model.
from app.knowledge.chunking.chunk import KnowledgeChunk

# Import the chunk ranker.
from app.knowledge.retrieval.ranker import ChunkRanker


# Create a helper for making test chunks.
def create_chunk(
    chunk_id: str,
    chunk_index: int,
    content: str,
) -> KnowledgeChunk:

    # Return a knowledge chunk with predictable test values.
    return KnowledgeChunk(
        id=chunk_id,
        document_id="document-1",
        chunk_index=chunk_index,
        content=content,
    )


# Test that relevant chunks are ranked first.
def test_rank_relevant_chunks():

    # Create the ranker.
    ranker = ChunkRanker()

    # Create test chunks.
    chunks = [
        create_chunk(
            "chunk-1",
            0,
            "Support is a price level.",
        ),
        create_chunk(
            "chunk-2",
            1,
            "Support support support is important.",
        ),
        create_chunk(
            "chunk-3",
            2,
            "Resistance is another price level.",
        ),
    ]

    # Rank the chunks using the query.
    results = ranker.rank(
        "support",
        chunks,
    )

    # The chunk containing support three times should rank first.
    assert results[0].id == "chunk-2"

    # The chunk containing support once should rank second.
    assert results[1].id == "chunk-1"

    # The unrelated resistance chunk should not be returned.
    assert len(results) == 2


# Test that ranking is case-insensitive.
def test_rank_is_case_insensitive():

    # Create the ranker.
    ranker = ChunkRanker()

    # Create a chunk containing lowercase text.
    chunks = [
        create_chunk(
            "chunk-1",
            0,
            "Support is important.",
        )
    ]

    # Search using uppercase text.
    results = ranker.rank(
        "SUPPORT",
        chunks,
    )

    # Make sure the chunk was found.
    assert len(results) == 1

    # Verify the correct chunk.
    assert results[0].id == "chunk-1"


# Test that multiple query words are scored.
def test_rank_multiple_query_words():

    # Create the ranker.
    ranker = ChunkRanker()

    # Create test chunks.
    chunks = [
        create_chunk(
            "chunk-1",
            0,
            "Support is important.",
        ),
        create_chunk(
            "chunk-2",
            1,
            "Support and resistance are important concepts.",
        ),
        create_chunk(
            "chunk-3",
            2,
            "Resistance is important.",
        ),
    ]

    # Search for two concepts.
    results = ranker.rank(
        "support resistance",
        chunks,
    )

    # The chunk containing both words should rank first.
    assert results[0].id == "chunk-2"


# Test that the limit is respected.
def test_rank_respects_limit():

    # Create the ranker.
    ranker = ChunkRanker()

    # Create several matching chunks.
    chunks = [
        create_chunk(
            f"chunk-{index}",
            index,
            "Support is important.",
        )
        for index in range(5)
    ]

    # Request only two results.
    results = ranker.rank(
        "support",
        chunks,
        limit=2,
    )

    # Make sure only two results are returned.
    assert len(results) == 2


# Test that chunks are returned in a stable order when scores tie.
def test_rank_uses_chunk_index_as_tie_breaker():

    # Create the ranker.
    ranker = ChunkRanker()

    # Create chunks with identical scores but different indexes.
    chunks = [
        create_chunk(
            "chunk-2",
            2,
            "Support is important.",
        ),
        create_chunk(
            "chunk-0",
            0,
            "Support is important.",
        ),
        create_chunk(
            "chunk-1",
            1,
            "Support is important.",
        ),
    ]

    # Rank the chunks.
    results = ranker.rank(
        "support",
        chunks,
    )

    # Verify that lower chunk indexes come first.
    assert [chunk.chunk_index for chunk in results] == [
        0,
        1,
        2,
    ]


# Test that an empty query returns no results.
def test_rank_empty_query():

    # Create the ranker.
    ranker = ChunkRanker()

    # Create a test chunk.
    chunks = [
        create_chunk(
            "chunk-1",
            0,
            "Support is important.",
        )
    ]

    # Rank using an empty query.
    results = ranker.rank(
        "",
        chunks,
    )

    # No results should be returned.
    assert results == []


# Test that a whitespace-only query returns no results.
def test_rank_whitespace_query():

    # Create the ranker.
    ranker = ChunkRanker()

    # Create a test chunk.
    chunks = [
        create_chunk(
            "chunk-1",
            0,
            "Support is important.",
        )
    ]

    # Rank using whitespace.
    results = ranker.rank(
        "   ",
        chunks,
    )

    # No results should be returned.
    assert results == []


# Test that invalid limits are rejected.
def test_rank_invalid_limit():

    # Create the ranker.
    ranker = ChunkRanker()

    # Create a test chunk.
    chunks = [
        create_chunk(
            "chunk-1",
            0,
            "Support is important.",
        )
    ]

    # Zero should be rejected.
    with pytest.raises(ValueError):
        ranker.rank(
            "support",
            chunks,
            limit=0,
        )

    # Negative values should be rejected.
    with pytest.raises(ValueError):
        ranker.rank(
            "support",
            chunks,
            limit=-1,
        )


# Test that unrelated chunks are excluded.
def test_rank_ignores_unrelated_chunks():

    # Create the ranker.
    ranker = ChunkRanker()

    # Create unrelated chunks.
    chunks = [
        create_chunk(
            "chunk-1",
            0,
            "Moving averages are technical indicators.",
        ),
        create_chunk(
            "chunk-2",
            1,
            "Candlestick patterns show price movement.",
        ),
    ]

    # Search for support.
    results = ranker.rank(
        "support",
        chunks,
    )

    # Nothing should match.
    assert results == []


# Test that repeated query words increase the score correctly.
def test_rank_repeated_words():

    # Create the ranker.
    ranker = ChunkRanker()

    # Create chunks with different frequencies.
    chunks = [
        create_chunk(
            "chunk-1",
            0,
            "support",
        ),
        create_chunk(
            "chunk-2",
            1,
            "support support",
        ),
        create_chunk(
            "chunk-3",
            2,
            "support support support",
        ),
    ]

    # Rank the chunks.
    results = ranker.rank(
        "support",
        chunks,
    )

    # Higher frequency should rank first.
    assert [chunk.id for chunk in results] == [
        "chunk-3",
        "chunk-2",
        "chunk-1",
    ]

# Test that a chunk matching more distinct query terms
# ranks above a chunk matching only one term.
def test_rank_prefers_more_distinct_query_terms():

    # Create the ranker.
    ranker = ChunkRanker()

    # Create chunks with different numbers of matching terms.
    chunks = [
        create_chunk(
            "chunk-one",
            0,
            "Support is important.",
        ),
        create_chunk(
            "chunk-both",
            1,
            "Support and resistance are important.",
        ),
        create_chunk(
            "chunk-two",
            2,
            "Resistance is important.",
        ),
    ]

    # Search for both concepts.
    results = ranker.rank(
        "support resistance",
        chunks,
    )

    # The chunk containing both concepts should rank first.
    assert results[0].id == "chunk-both"


# Test that a chunk containing both query terms
# ranks above a chunk containing one term many times.
def test_rank_distinct_terms_are_more_important_than_frequency():

    # Create the ranker.
    ranker = ChunkRanker()

    # Create chunks with different relevance.
    chunks = [
        create_chunk(
            "chunk-frequency",
            0,
            "support support support support support",
        ),
        create_chunk(
            "chunk-both",
            1,
            "support and resistance",
        ),
    ]

    # Search for both concepts.
    results = ranker.rank(
        "support resistance",
        chunks,
    )

    # Matching both query concepts should win.
    assert results[0].id == "chunk-both"


# Test that irrelevant chunks are never returned,
# even when the result limit is larger than the
# number of relevant chunks.
def test_rank_returns_only_relevant_chunks():

    # Create the ranker.
    ranker = ChunkRanker()

    # Create relevant and irrelevant chunks.
    chunks = [
        create_chunk(
            "chunk-support",
            0,
            "Support is a price level.",
        ),
        create_chunk(
            "chunk-resistance",
            1,
            "Resistance is a price level.",
        ),
        create_chunk(
            "chunk-unrelated",
            2,
            "Moving averages are technical indicators.",
        ),
    ]

    # Search for support and resistance.
    results = ranker.rank(
        "support resistance",
        chunks,
        limit=10,
    )

    # Only the two relevant chunks should be returned.
    assert [chunk.id for chunk in results] == [
        "chunk-support",
        "chunk-resistance",
    ]

# Test that score returns zero for an empty query.
def test_score_empty_query():

    # Create the ranker.
    ranker = ChunkRanker()

    # Create a test chunk.
    chunk = create_chunk(
        "chunk-1",
        0,
        "Support is important.",
    )

    # Calculate the score.
    score = ranker.score(
        "",
        chunk,
    )

    # An empty query should have no relevance.
    assert score == 0


# Test that score rewards matching terms.
def test_score_matching_term():

    # Create the ranker.
    ranker = ChunkRanker()

    # Create a test chunk.
    chunk = create_chunk(
        "chunk-1",
        0,
        "Support is important.",
    )

    # Calculate the score.
    score = ranker.score(
        "support",
        chunk,
    )

    # One distinct match gives 100 points,
    # plus one occurrence.
    assert score == 101


# Test that repeated occurrences increase the score.
def test_score_repeated_occurrences():

    # Create the ranker.
    ranker = ChunkRanker()

    # Create a chunk containing support three times.
    chunk = create_chunk(
        "chunk-1",
        0,
        "Support support support.",
    )

    # Calculate the score.
    score = ranker.score(
        "support",
        chunk,
    )

    # One matched term = 100.
    # Three occurrences = 3.
    assert score == 103


# Test that multiple distinct query terms receive
# a higher relevance score.
def test_score_multiple_distinct_terms():

    # Create the ranker.
    ranker = ChunkRanker()

    # Create a chunk containing both terms.
    chunk = create_chunk(
        "chunk-1",
        0,
        "Support and resistance are important.",
    )

    # Calculate the score.
    score = ranker.score(
        "support resistance",
        chunk,
    )

    # Two matched terms = 200.
    # Two total occurrences = 2.
    assert score == 202


# Test that score matching is case-insensitive.
def test_score_is_case_insensitive():

    # Create the ranker.
    ranker = ChunkRanker()

    # Create a chunk containing lowercase text.
    chunk = create_chunk(
        "chunk-1",
        0,
        "support is important.",
    )

    # Search using uppercase text.
    score = ranker.score(
        "SUPPORT",
        chunk,
    )

    # The term should still match.
    assert score == 101


# Test that unrelated chunks receive a zero score.
def test_score_unrelated_chunk():

    # Create the ranker.
    ranker = ChunkRanker()

    # Create an unrelated chunk.
    chunk = create_chunk(
        "chunk-1",
        0,
        "Moving averages are technical indicators.",
    )

    # Calculate the score.
    score = ranker.score(
        "support",
        chunk,
    )

    # There should be no relevance.
    assert score == 0