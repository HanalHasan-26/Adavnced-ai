# Import the knowledge chunk model.
from app.knowledge.chunking.chunk import KnowledgeChunk

# Import the retrieval result.
from app.knowledge.retrieval.result import KnowledgeRetrievalResult


# Test that a retrieval result stores all information correctly.
def test_retrieval_result():

    # Create a test chunk.
    chunk = KnowledgeChunk(
        id="chunk-1",
        document_id="document-1",
        chunk_index=0,
        content="Support is a price level.",
    )

    # Create a retrieval result.
    result = KnowledgeRetrievalResult(
        query="support",
        chunks=[chunk],
        context="[Knowledge 1]\nSupport is a price level.",
    )

    # Verify the original query.
    assert result.query == "support"

    # Verify the retrieved chunks.
    assert result.chunks == [chunk]

    # Verify the assembled context.
    assert result.context == (
        "[Knowledge 1]\n"
        "Support is a price level."
    )

    # Verify the chunk count.
    assert result.count == 1


# Test that an empty result reports zero chunks.
def test_empty_retrieval_result():

    # Create an empty result.
    result = KnowledgeRetrievalResult(
        query="unknown",
        chunks=[],
        context="",
    )

    # Verify the count.
    assert result.count == 0

    # Verify the context.
    assert result.context == ""