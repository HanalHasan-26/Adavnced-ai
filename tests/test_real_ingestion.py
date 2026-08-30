# Import Path so we can work with file paths.
from pathlib import Path

# Import the knowledge ingestion service.
from app.knowledge.ingestion.service import KnowledgeIngestionService

# Import persistent knowledge storage.
from app.knowledge.storage import KnowledgeStorage

# Import the text loader.
from app.knowledge.ingestion.text_loader import TextLoader

# Import the PDF loader.
from app.knowledge.ingestion.pdf_loader import PDFLoader


# Test real text-file ingestion.
def test_real_text_ingestion():

    # Locate the real trading knowledge document.
    file_path = Path("data/trading_basics.txt")

    # Make sure the document exists.
    assert file_path.is_file()

    # Create the test database path.
    database_path = Path("data/test_ingestion.db")

    # Remove an old test database if it exists.
    if database_path.exists():
        database_path.unlink()

    # Create persistent knowledge storage.
    storage = KnowledgeStorage(
        database_path=database_path
    )

    # Create the text loader.
    text_loader = TextLoader()

    # Create the PDF loader.
    pdf_loader = PDFLoader()

    # Create the ingestion service.
    service = KnowledgeIngestionService(
        storage=storage,
        text_loader=text_loader,
        pdf_loader=pdf_loader,
    )

    # Ingest the real trading document.
    document = service.ingest_text_file(
        file_path
    )

    # Verify that a document was created.
    assert document is not None

    # Retrieve the document from SQLite.
    stored_document = storage.get(
        str(document.id)
    )

    # Verify that the document was stored.
    assert stored_document is not None

    # Verify important trading knowledge exists.
    assert "Support" in stored_document.content
    assert "Resistance" in stored_document.content
    assert "Order Block" in stored_document.content

    # Retrieve the document's chunks.
    chunks = storage.get_chunks(
        str(document.id)
    )

    # Make sure chunks were created.
    assert len(chunks) > 0

    # Combine all chunks for verification.
    combined_content = " ".join(
        chunk.content
        for chunk in chunks
    )

    # Verify important knowledge exists in chunks.
    assert "Support" in combined_content
    assert "Resistance" in combined_content
    assert "Order Block" in combined_content