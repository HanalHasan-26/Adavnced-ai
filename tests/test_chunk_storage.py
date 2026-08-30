# Import the KnowledgeChunk model.
from app.knowledge.chunking.chunk import KnowledgeChunk

# Import the KnowledgeDocument model.
from app.knowledge.document import KnowledgeDocument

# Import the storage system.
from app.knowledge.storage import KnowledgeStorage


# Test that a chunk can be saved and retrieved.
def test_add_and_get_chunk(tmp_path):

    # Create a temporary database.
    database_path = tmp_path / "knowledge.db"

    # Create the storage system.
    storage = KnowledgeStorage(database_path)

    # Create a knowledge document.
    document = KnowledgeDocument.create(
        content="Support and resistance are important trading concepts.",
        source="trading.txt",
        source_type="text",
    )

    # Save the parent document first.
    storage.add(document)

    # Create a chunk belonging to that document.
    chunk = KnowledgeChunk(
        id="chunk-1",
        document_id=str(document.id),
        chunk_index=0,
        content="Support and resistance are important.",
    )

    # Save the chunk.
    storage.add_chunk(chunk)

    # Retrieve all chunks for the document.
    chunks = storage.get_chunks(str(document.id))

    # Make sure exactly one chunk was returned.
    assert len(chunks) == 1

    # Retrieve the first chunk.
    stored_chunk = chunks[0]

    # Verify the chunk ID.
    assert stored_chunk.id == "chunk-1"

    # Verify the parent document ID.
    assert stored_chunk.document_id == str(document.id)

    # Verify the chunk position.
    assert stored_chunk.chunk_index == 0

    # Verify the chunk content.
    assert stored_chunk.content == (
        "Support and resistance are important."
    )


# Test that multiple chunks are returned in document order.
def test_get_chunks_returns_correct_order(tmp_path):

    # Create a temporary database.
    database_path = tmp_path / "knowledge.db"

    # Create the storage system.
    storage = KnowledgeStorage(database_path)

    # Create the parent document.
    document = KnowledgeDocument.create(
        content="Trading knowledge.",
        source="trading.txt",
        source_type="text",
    )

    # Save the parent document.
    storage.add(document)

    # Create the chunks in a deliberately mixed insertion order.
    chunks_to_add = [
        KnowledgeChunk(
            id="chunk-2",
            document_id=str(document.id),
            chunk_index=2,
            content="Third chunk.",
        ),
        KnowledgeChunk(
            id="chunk-0",
            document_id=str(document.id),
            chunk_index=0,
            content="First chunk.",
        ),
        KnowledgeChunk(
            id="chunk-1",
            document_id=str(document.id),
            chunk_index=1,
            content="Second chunk.",
        ),
    ]

    # Save every chunk.
    for chunk in chunks_to_add:
        storage.add_chunk(chunk)

    # Retrieve the chunks.
    stored_chunks = storage.get_chunks(str(document.id))

    # Verify that the database returned them in the correct order.
    assert [chunk.chunk_index for chunk in stored_chunks] == [0, 1, 2]

    # Verify their contents are also in the correct order.
    assert [chunk.content for chunk in stored_chunks] == [
        "First chunk.",
        "Second chunk.",
        "Third chunk.",
    ]


# Test that different documents have separate chunks.
def test_get_chunks_only_returns_requested_document(tmp_path):

    # Create a temporary database.
    database_path = tmp_path / "knowledge.db"

    # Create the storage system.
    storage = KnowledgeStorage(database_path)

    # Create the first document.
    document_one = KnowledgeDocument.create(
        content="Document one.",
        source="one.txt",
        source_type="text",
    )

    # Create the second document.
    document_two = KnowledgeDocument.create(
        content="Document two.",
        source="two.txt",
        source_type="text",
    )

    # Save both documents.
    storage.add(document_one)
    storage.add(document_two)

    # Add a chunk to the first document.
    storage.add_chunk(
        KnowledgeChunk(
            id="chunk-one",
            document_id=str(document_one.id),
            chunk_index=0,
            content="Chunk from document one.",
        )
    )

    # Add a chunk to the second document.
    storage.add_chunk(
        KnowledgeChunk(
            id="chunk-two",
            document_id=str(document_two.id),
            chunk_index=0,
            content="Chunk from document two.",
        )
    )

    # Retrieve chunks belonging only to document one.
    chunks = storage.get_chunks(str(document_one.id))

    # Make sure only one chunk was returned.
    assert len(chunks) == 1

    # Make sure it belongs to document one.
    assert chunks[0].document_id == str(document_one.id)


# Test that chunks can be deleted.
def test_delete_chunks(tmp_path):

    # Create a temporary database.
    database_path = tmp_path / "knowledge.db"

    # Create the storage system.
    storage = KnowledgeStorage(database_path)

    # Create a document.
    document = KnowledgeDocument.create(
        content="Knowledge document.",
        source="knowledge.txt",
        source_type="text",
    )

    # Save the document.
    storage.add(document)

    # Add multiple chunks.
    storage.add_chunk(
        KnowledgeChunk(
            id="chunk-1",
            document_id=str(document.id),
            chunk_index=0,
            content="First chunk.",
        )
    )

    storage.add_chunk(
        KnowledgeChunk(
            id="chunk-2",
            document_id=str(document.id),
            chunk_index=1,
            content="Second chunk.",
        )
    )

    # Delete all chunks belonging to the document.
    deleted_count = storage.delete_chunks(str(document.id))

    # Verify that two chunks were deleted.
    assert deleted_count == 2

    # Verify that no chunks remain.
    assert storage.get_chunks(str(document.id)) == []