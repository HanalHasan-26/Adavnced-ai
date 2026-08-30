# Import the UUID type so we can verify the document ID.
from uuid import UUID

# Import the KnowledgeDocument model we are testing.
from app.knowledge.document import KnowledgeDocument

# Import the KnowledgeStorage class we want to test.
from app.knowledge.storage import KnowledgeStorage

# Test that a saved knowledge document can be retrieved.
def test_get_knowledge_document(tmp_path):

    # Create a temporary database for this test.
        database_path = tmp_path / "test_knowledge.db"

    # Create the knowledge storage using the temporary database.
        storage = KnowledgeStorage(database_path)

    # Create a document that we want to save.
        document = KnowledgeDocument.create(
        content="Python is a programming language.",
        source="test",
        source_type="text",
        )

    # Save the document into the database.
        storage.add(document)

    # Retrieve the document using its unique ID.
        retrieved = storage.get(str(document.id))

    # Make sure a document was actually returned.
        assert retrieved is not None

    # Make sure the retrieved content matches the original.
        assert retrieved.content == document.content

    # Make sure the retrieved source matches the original.
        assert retrieved.source == document.source

    # Make sure the retrieved source type matches the original.
        assert retrieved.source_type == document.source_type

    # Make sure the retrieved ID matches the original ID.
        assert retrieved.id == document.id

# Define a test for saving a knowledge document.
def test_add_knowledge_document(tmp_path):

    # Create a temporary database path for this test.
    # pytest automatically provides a separate temporary folder.
    database_path = tmp_path / "test_knowledge.db"

    # Create a storage object using the temporary database.
    storage = KnowledgeStorage(database_path)

    # Create a sample knowledge document.
    document = KnowledgeDocument.create(
        content="Python is a programming language.",
        source="example.txt",
        source_type="txt",
    )

    # Save the document to the SQLite database.
    storage.add(document)

    # Verify that the database file was actually created.
    assert database_path.exists()

    # Verify that the document has a valid UUID.
    assert isinstance(document.id, UUID)

# Test that the storage system can find knowledge using a search query.
def test_search_knowledge(tmp_path):

    # Create a temporary database for this test.
    database_path = tmp_path / "test_knowledge.db"

    # Create the knowledge storage using the temporary database.
    storage = KnowledgeStorage(database_path)

    # Create the first sample knowledge document.
    python_document = KnowledgeDocument.create(
        content="Python is a programming language.",
        source="python.txt",
        source_type="text",
    )

    # Create a second sample knowledge document.
    trading_document = KnowledgeDocument.create(
        content="Gold is traded in the financial markets.",
        source="trading.txt",
        source_type="text",
    )

    # Save the first document.
    storage.add(python_document)

    # Save the second document.
    storage.add(trading_document)

    # Search for knowledge containing the word Python.
    results = storage.search("Python")

    # Make sure exactly one matching document was found.
    assert len(results) == 1

    # Make sure the correct document was returned.
    assert results[0].id == python_document.id

    # Make sure unrelated knowledge was not returned.
    assert results[0].id != trading_document.id

# Test that one specific knowledge document can be deleted.
def test_delete_knowledge_document(tmp_path):

    # Create a temporary database for this test.
    database_path = tmp_path / "test_knowledge.db"

    # Create the knowledge storage using the temporary database.
    storage = KnowledgeStorage(database_path)

    # Create the knowledge document we want to delete.
    document = KnowledgeDocument.create(
        content="This knowledge should be deleted.",
        source="delete_test.txt",
        source_type="text",
    )

    # Create another document that must NOT be deleted.
    other_document = KnowledgeDocument.create(
        content="This knowledge should remain.",
        source="keep_test.txt",
        source_type="text",
    )

    # Save the first document.
    storage.add(document)

    # Save the second document.
    storage.add(other_document)

    # Delete only the first document using its ID.
    deleted = storage.delete(str(document.id))

    # Make sure the deletion actually happened.
    assert deleted is True

    # Make sure the deleted document can no longer be found.
    assert storage.get(str(document.id)) is None

    # Make sure the unrelated document still exists.
    assert storage.get(str(other_document.id)) is not None

    # Make sure trying to delete the same document again returns False.
    assert storage.delete(str(document.id)) is False

# Test that an existing knowledge document can be updated.
def test_update_knowledge_document(tmp_path):

    # Create a temporary database for this test.
    database_path = tmp_path / "test_knowledge.db"

    # Create the knowledge storage using the temporary database.
    storage = KnowledgeStorage(database_path)

    # Create the original knowledge document.
    document = KnowledgeDocument.create(
        content="The old information.",
        source="old.txt",
        source_type="text",
    )

    # Save the original document.
    storage.add(document)

    # Create an updated version using the SAME ID.
    updated_document = KnowledgeDocument(
        id=document.id,
        content="The updated information.",
        source="updated.txt",
        source_type="text",
        created_at=document.created_at,
    )

    # Update the existing document in the database.
    updated = storage.update(updated_document)

    # Make sure the update was successful.
    assert updated is True

    # Retrieve the document after the update.
    retrieved = storage.get(str(document.id))

    # Make sure the document still exists.
    assert retrieved is not None

    # Make sure the content was updated.
    assert retrieved.content == "The updated information."

    # Make sure the source was updated.
    assert retrieved.source == "updated.txt"

    # Make sure the ID did not change.
    assert retrieved.id == document.id

    # Test that all stored knowledge documents can be retrieved.
def test_list_knowledge_documents(tmp_path):

    # Create a temporary database for this test.
    database_path = tmp_path / "test_knowledge.db"

    # Create the knowledge storage using the temporary database.
    storage = KnowledgeStorage(database_path)

    # Create the first knowledge document.
    first_document = KnowledgeDocument.create(
        content="Python is a programming language.",
        source="python.txt",
        source_type="text",
    )

    # Create the second knowledge document.
    second_document = KnowledgeDocument.create(
        content="Gold is traded in financial markets.",
        source="trading.txt",
        source_type="text",
    )

    # Save the first document.
    storage.add(first_document)

    # Save the second document.
    storage.add(second_document)

    # Retrieve all stored documents.
    documents = storage.list()

    # Make sure both documents were returned.
    assert len(documents) == 2

    # Make sure both document IDs are present.
    document_ids = {document.id for document in documents}

    # Verify that the first document is in the results.
    assert first_document.id in document_ids

    # Verify that the second document is in the results.
    assert second_document.id in document_ids