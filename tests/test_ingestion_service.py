# Import pytest so we can test expected errors.
import pytest

# Import PyMuPDF so we can create test PDFs.
import fitz

# Import the ingestion service.
from app.knowledge.ingestion.service import KnowledgeIngestionService

# Import the PDF loader.
from app.knowledge.ingestion.pdf_loader import PDFLoader

# Import the text loader.
from app.knowledge.ingestion.text_loader import TextLoader

# Import the knowledge storage.
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

    # Create the knowledge storage.
    storage = KnowledgeStorage(database_path)

    # Create the text loader.
    text_loader = TextLoader()

    # Create the PDF loader.
    pdf_loader = PDFLoader()

    # Create the ingestion service with both loaders.
    service = KnowledgeIngestionService(
        storage=storage,
        text_loader=text_loader,
        pdf_loader=pdf_loader,
    )

    # Ingest the text file into the knowledge system.
    document = service.ingest_text_file(file_path)

    # Make sure the extracted content is correct.
    assert document.content == (
        "Python is commonly used for artificial intelligence."
    )

    # Make sure the source points to the original file.
    assert document.source == str(file_path)

    # Make sure the source type is recorded correctly.
    assert document.source_type == "text"

    # Retrieve the document from the database.
    stored_document = storage.get(str(document.id))

    # Make sure the document was actually persisted.
    assert stored_document is not None

    # Make sure the stored content matches the original.
    assert stored_document.content == document.content


# Test that an empty text file is rejected.
def test_ingest_empty_text_file(tmp_path):

    # Create a temporary empty text file.
    file_path = tmp_path / "empty.txt"

    # Write only whitespace into the file.
    file_path.write_text(
        "   \n   \n   ",
        encoding="utf-8",
    )

    # Create a temporary database.
    database_path = tmp_path / "knowledge.db"

    # Create the knowledge storage.
    storage = KnowledgeStorage(database_path)

    # Create the text loader.
    text_loader = TextLoader()

    # Create the PDF loader.
    pdf_loader = PDFLoader()

    # Create the ingestion service with both loaders.
    service = KnowledgeIngestionService(
        storage=storage,
        text_loader=text_loader,
        pdf_loader=pdf_loader,
    )

    # Verify that empty knowledge is rejected.
    with pytest.raises(ValueError):

        # Try to ingest the empty file.
        service.ingest_text_file(file_path)


# Test that a PDF can be ingested and stored.
def test_ingest_pdf_file(tmp_path):

    # Create the temporary PDF path.
    file_path = tmp_path / "learning.pdf"

    # Create a new PDF document.
    pdf = fitz.open()

    # Add a page to the PDF.
    page = pdf.new_page()

    # Write knowledge onto the page.
    page.insert_text(
        (72, 72),
        "Gold is traded on financial markets.",
    )

    # Save the PDF to the temporary path.
    pdf.save(file_path)

    # Close the PDF.
    pdf.close()

    # Create a temporary database.
    database_path = tmp_path / "knowledge.db"

    # Create the knowledge storage.
    storage = KnowledgeStorage(database_path)

    # Create the text loader.
    text_loader = TextLoader()

    # Create the PDF loader.
    pdf_loader = PDFLoader()

    # Create the ingestion service with both loaders.
    service = KnowledgeIngestionService(
        storage=storage,
        text_loader=text_loader,
        pdf_loader=pdf_loader,
    )

    # Ingest the PDF into the knowledge system.
    document = service.ingest_pdf_file(file_path)

    # Make sure the PDF text was extracted.
    assert "Gold is traded on financial markets." in document.content

    # Make sure the source is the PDF file.
    assert document.source == str(file_path)

    # Make sure the source type is recorded as PDF.
    assert document.source_type == "pdf"

    # Retrieve the document from persistent storage.
    stored_document = storage.get(str(document.id))

    # Make sure the document was actually saved.
    assert stored_document is not None

    # Make sure the stored content matches the extracted content.
    assert stored_document.content == document.content