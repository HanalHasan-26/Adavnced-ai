# Import pytest so we can test validation errors.
import pytest

# Import the knowledge document model.
from app.knowledge.document import KnowledgeDocument

# Import the knowledge chunk model.
from app.knowledge.chunking.chunk import KnowledgeChunk

# Import the knowledge storage.
from app.knowledge.storage import KnowledgeStorage

# Import the chunk retriever.
from app.knowledge.retrieval.chunk_retriever import ChunkRetriever


# Create a storage system and retriever for tests.
def create_retriever(database_path):

    # Create the knowledge storage.
    storage = KnowledgeStorage(database_path)

    # Create the chunk retriever.
    retriever = ChunkRetriever(storage)

    # Return both objects.
    return storage, retriever


# Test that a matching chunk can be retrieved.
def test_retrieve_matching_chunk(tmp_path):

    # Create a temporary database.
    database_path = tmp_path / "knowledge.db"

    # Create the storage and retriever.
    storage, retriever = create_retriever(database_path)

    # Create a knowledge document.
    document = KnowledgeDocument.create(
        content="Support is a price level where buying interest may appear.",
        source="trading.txt",
        source_type="text",
    )

    # Store the document.
    storage.add(document)

    # Store a chunk containing the requested concept.
    storage.add_chunk(
        KnowledgeChunk(
            id="chunk-1",
            document_id=str(document.id),
            chunk_index=0,
            content="Support is a price level where buying interest may appear.",
        )
    )

    # Search for the word support.
    results = retriever.retrieve("support")

    # Make sure one result was found.
    assert len(results) == 1

    # Verify the returned chunk.
    assert results[0].id == "chunk-1"


# Test that retrieval is case-insensitive.
def test_retrieve_is_case_insensitive(tmp_path):

    # Create a temporary database.
    database_path = tmp_path / "knowledge.db"

    # Create the storage and retriever.
    storage, retriever = create_retriever(database_path)

    # Create a knowledge document.
    document = KnowledgeDocument.create(
        content="Resistance can act as a barrier to rising prices.",
        source="trading.txt",
        source_type="text",
    )

    # Store the document.
    storage.add(document)

    # Store a chunk.
    storage.add_chunk(
        KnowledgeChunk(
            id="chunk-1",
            document_id=str(document.id),
            chunk_index=0,
            content="Resistance can act as a barrier to rising prices.",
        )
    )

    # Search using uppercase text.
    results = retriever.retrieve("RESISTANCE")

    # Make sure the chunk was found.
    assert len(results) == 1

    # Verify the correct chunk.
    assert results[0].id == "chunk-1"


# Test that an empty query returns no results.
def test_retrieve_empty_query(tmp_path):

    # Create a temporary database.
    database_path = tmp_path / "knowledge.db"

    # Create the storage and retriever.
    _, retriever = create_retriever(database_path)

    # Search using an empty query.
    results = retriever.retrieve("")

    # No chunks should be returned.
    assert results == []


# Test that whitespace-only queries return no results.
def test_retrieve_whitespace_query(tmp_path):

    # Create a temporary database.
    database_path = tmp_path / "knowledge.db"

    # Create the storage and retriever.
    _, retriever = create_retriever(database_path)

    # Search using only whitespace.
    results = retriever.retrieve("   ")

    # No chunks should be returned.
    assert results == []


# Test that invalid limits are rejected.
def test_retrieve_invalid_limit(tmp_path):

    # Create a temporary database.
    database_path = tmp_path / "knowledge.db"

    # Create the storage and retriever.
    _, retriever = create_retriever(database_path)

    # Verify that zero is rejected.
    with pytest.raises(ValueError):
        retriever.retrieve("support", limit=0)

    # Verify that negative values are rejected.
    with pytest.raises(ValueError):
        retriever.retrieve("support", limit=-1)


# Test that the limit restricts the number of results.
def test_retrieve_respects_limit(tmp_path):

    # Create a temporary database.
    database_path = tmp_path / "knowledge.db"

    # Create the storage and retriever.
    storage, retriever = create_retriever(database_path)

    # Create a document containing several matching concepts.
    document = KnowledgeDocument.create(
        content="support support support support",
        source="trading.txt",
        source_type="text",
    )

    # Store the document.
    storage.add(document)

    # Store multiple matching chunks.
    for index in range(5):

        storage.add_chunk(
            KnowledgeChunk(
                id=f"chunk-{index}",
                document_id=str(document.id),
                chunk_index=index,
                content=f"Support level number {index}.",
            )
        )

    # Retrieve only three results.
    results = retriever.retrieve(
        "support",
        limit=3,
    )

    # Make sure exactly three results were returned.
    assert len(results) == 3


# Test that non-matching chunks are ignored.
def test_non_matching_chunks_are_ignored(tmp_path):

    # Create a temporary database.
    database_path = tmp_path / "knowledge.db"

    # Create the storage and retriever.
    storage, retriever = create_retriever(database_path)

    # Create a knowledge document.
    document = KnowledgeDocument.create(
        content="Trading contains many different concepts.",
        source="trading.txt",
        source_type="text",
    )

    # Store the document.
    storage.add(document)

    # Store a chunk that does not match.
    storage.add_chunk(
        KnowledgeChunk(
            id="chunk-1",
            document_id=str(document.id),
            chunk_index=0,
            content="Moving averages are technical indicators.",
        )
    )

    # Search for a completely different concept.
    results = retriever.retrieve("gold")

    # Make sure nothing was returned.
    assert results == []


# Test that chunks from multiple documents can be retrieved.
def test_retrieve_from_multiple_documents(tmp_path):

    # Create a temporary database.
    database_path = tmp_path / "knowledge.db"

    # Create the storage and retriever.
    storage, retriever = create_retriever(database_path)

    # Create the first document.
    document_one = KnowledgeDocument.create(
        content="Gold support can be important.",
        source="gold.txt",
        source_type="text",
    )

    # Create the second document.
    document_two = KnowledgeDocument.create(
        content="Gold resistance can also be important.",
        source="gold2.txt",
        source_type="text",
    )

    # Store both documents.
    storage.add(document_one)
    storage.add(document_two)

    # Add a chunk to the first document.
    storage.add_chunk(
        KnowledgeChunk(
            id="chunk-one",
            document_id=str(document_one.id),
            chunk_index=0,
            content="Gold support can be important.",
        )
    )

    # Add a chunk to the second document.
    storage.add_chunk(
        KnowledgeChunk(
            id="chunk-two",
            document_id=str(document_two.id),
            chunk_index=0,
            content="Gold resistance can also be important.",
        )
    )

    # Search for gold.
    results = retriever.retrieve("gold")

    # Both chunks should be returned.
    assert len(results) == 2

    # Verify both chunk IDs are present.
    result_ids = {chunk.id for chunk in results}

    assert result_ids == {
        "chunk-one",
        "chunk-two",
    }