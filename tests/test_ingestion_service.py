# Import the classes needed to test the ingestion pipeline.
from app.knowledge.ingestion.service import KnowledgeIngestionService
from app.knowledge.ingestion.text_loader import TextLoader
from app.knowledge.storage import KnowledgeStorage


# Test that a text file can be ingested and stored.
def test_ingest_text_file(tmp_path):

    # Create a temporary text file.
    file_path = tmp_path / "learning.txt"

    # Write knowledge into the temporary file.
    file_path.write_text(
        "Python is commonly used for artificial intelligence.",
        encoding="utf-8",
    )

    # Create a temporary database for this test.
    database_path = tmp_path / "knowledge.db"

    # Create the storage system using the temporary database.
    storage = KnowledgeStorage(database_path)

    # Create the text-file loader.
    text_loader = TextLoader()

    # Create the ingestion service.
    service = KnowledgeIngestionService(
        storage=storage,
        text_loader=text_loader,
    )

    # Ingest the text file into the knowledge system.
    document = service.ingest_text_file(file_path)

    # Make sure the extracted content is correct.
    assert document.content == (
        "Python is commonly used for artificial intelligence."
    )

    # Make sure the source points to the original file.
    assert document.source == str(file_path)

    # Make sure the source type was recorded correctly.
    assert document.source_type == "text"

    # Retrieve the document from the database.
    stored_document = storage.get(str(document.id))

    # Make sure the document was actually persisted.
    assert stored_document is not None

    # Make sure the stored content matches the original.
    assert stored_document.content == document.content