# Import the UUID type so we can verify the document ID.
from uuid import UUID

# Import the KnowledgeDocument model we are testing.
from app.knowledge.document import KnowledgeDocument

# Import the KnowledgeStorage class we want to test.
from app.knowledge.storage import KnowledgeStorage


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