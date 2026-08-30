# Import the chunk model we want to test.
from app.knowledge.chunking.chunk import KnowledgeChunk


# Test that a KnowledgeChunk stores its information correctly.
def test_knowledge_chunk():

    # Create a sample knowledge chunk.
    chunk = KnowledgeChunk(
        id="chunk-1",
        document_id="document-1",
        chunk_index=0,
        content="Support and resistance are important trading concepts.",
    )

    # Verify the chunk ID.
    assert chunk.id == "chunk-1"

    # Verify the source document ID.
    assert chunk.document_id == "document-1"

    # Verify the chunk position.
    assert chunk.chunk_index == 0

    # Verify the chunk content.
    assert chunk.content == (
        "Support and resistance are important trading concepts."
    )


# Test that chunks cannot be modified after creation.
def test_knowledge_chunk_is_immutable():

    # Create a sample chunk.
    chunk = KnowledgeChunk(
        id="chunk-1",
        document_id="document-1",
        chunk_index=0,
        content="Test content.",
    )

    # Attempting to change a frozen field should raise an error.
    try:

        # Try to modify the chunk content.
        chunk.content = "Changed content."

    except AttributeError:

        # AttributeError confirms that the model is immutable.
        pass

    else:

        # Fail the test if the modification was unexpectedly allowed.
        raise AssertionError("KnowledgeChunk should be immutable.")