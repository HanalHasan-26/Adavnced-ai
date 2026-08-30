# Import pytest so we can test validation errors.
import pytest

# Import the knowledge chunk model.
from app.knowledge.chunking.chunk import KnowledgeChunk

# Import the knowledge context assembler.
from app.knowledge.retrieval.context import KnowledgeContext


# Create a helper for making test chunks.
def create_chunk(
    chunk_id: str,
    chunk_index: int,
    content: str,
) -> KnowledgeChunk:

    # Return a predictable knowledge chunk.
    return KnowledgeChunk(
        id=chunk_id,
        document_id="document-1",
        chunk_index=chunk_index,
        content=content,
    )


# Test that chunks are assembled correctly.
def test_assemble_chunks():

    # Create the context assembler.
    context = KnowledgeContext()

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
            "Resistance is a price barrier.",
        ),
    ]

    # Assemble the chunks.
    result = context.assemble(chunks)

    # Verify the final context.
    assert result == (
        "[Knowledge 1]\n"
        "Support is a price level.\n\n"
        "[Knowledge 2]\n"
        "Resistance is a price barrier."
    )


# Test that an empty chunk list produces empty context.
def test_assemble_empty_chunks():

    # Create the context assembler.
    context = KnowledgeContext()

    # Assemble no chunks.
    result = context.assemble([])

    # Verify that the result is empty.
    assert result == ""


# Test that the maximum context size is respected.
def test_context_respects_max_characters():

    # Create a context with a small limit.
    context = KnowledgeContext(
        max_characters=60,
    )

    # Create several chunks.
    chunks = [
        create_chunk(
            "chunk-1",
            0,
            "Support is important.",
        ),
        create_chunk(
            "chunk-2",
            1,
            "Resistance is important.",
        ),
        create_chunk(
            "chunk-3",
            2,
            "Moving averages are indicators.",
        ),
    ]

    # Assemble the chunks.
    result = context.assemble(chunks)

    # Verify that the result does not exceed the limit.
    assert len(result) <= 60


# Test that the assembler keeps earlier chunks when
# a later chunk would exceed the limit.
def test_context_keeps_chunks_that_fit():

    # Create a context with a controlled limit.
    context = KnowledgeContext(
        max_characters=70,
    )

    # Create chunks.
    chunks = [
        create_chunk(
            "chunk-1",
            0,
            "First knowledge chunk.",
        ),
        create_chunk(
            "chunk-2",
            1,
            "Second knowledge chunk.",
        ),
        create_chunk(
            "chunk-3",
            2,
            "Third knowledge chunk.",
        ),
    ]

    # Assemble the chunks.
    result = context.assemble(chunks)

    # The first chunk should remain.
    assert "First knowledge chunk." in result


# Test that an individual oversized chunk is not returned.
def test_context_rejects_oversized_first_chunk():

    # Create a very small context limit.
    context = KnowledgeContext(
        max_characters=10,
    )

    # Create a chunk larger than the limit.
    chunks = [
        create_chunk(
            "chunk-1",
            0,
            "This chunk is too large.",
        )
    ]

    # Assemble the chunks.
    result = context.assemble(chunks)

    # Nothing should be returned because the first
    # chunk cannot fit.
    assert result == ""


# Test that invalid maximum sizes are rejected.
def test_context_invalid_max_characters():

    # Zero should be rejected.
    with pytest.raises(ValueError):
        KnowledgeContext(
            max_characters=0,
        )

    # Negative values should be rejected.
    with pytest.raises(ValueError):
        KnowledgeContext(
            max_characters=-1,
        )


# Test that chunk order is preserved.
def test_context_preserves_order():

    # Create the context assembler.
    context = KnowledgeContext()

    # Create chunks in a deliberate order.
    chunks = [
        create_chunk(
            "chunk-3",
            2,
            "Third.",
        ),
        create_chunk(
            "chunk-1",
            0,
            "First.",
        ),
        create_chunk(
            "chunk-2",
            1,
            "Second.",
        ),
    ]

    # Assemble the chunks.
    result = context.assemble(chunks)

    # Verify the supplied order is preserved.
    assert result.index("Third.") < result.index("First.")
    assert result.index("First.") < result.index("Second.")