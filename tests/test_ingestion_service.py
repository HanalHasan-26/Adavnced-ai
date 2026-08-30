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

# Import the text chunker.
from app.knowledge.chunking.text_chunker import TextChunker


# Create an ingestion service for tests.
def create_service(database_path, chunker=None):

    # Create the knowledge storage.
    storage = KnowledgeStorage(database_path)

    # Create the text loader.
    text_loader = TextLoader()

    # Create the PDF loader.
    pdf_loader = PDFLoader()

    # Create the ingestion service.
    service = KnowledgeIngestionService(
        storage=storage,
        text_loader=text_loader,
        pdf_loader=pdf_loader,
        chunker=chunker,
    )

    # Return both objects so tests can inspect the database.
    return service, storage


# Test that a text file can be ingested and stored.
def test_ingest_text_file(tmp_path):

    # Create a temporary text file.
    file_path = tmp_path / "learning.txt"

    # Write knowledge into the temporary file.
    file_path.write_text(
        "Python is commonly used for artificial intelligence.",
        encoding="utf-8",
    )

    # Create a temporary database.
    database_path = tmp_path / "knowledge.db"

    # Create the ingestion service.
    service, storage = create_service(database_path)

    # Ingest the text file.
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

    # Create the ingestion service.
    service, _ = create_service(database_path)

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

    # Create the ingestion service.
    service, storage = create_service(database_path)

    # Ingest the PDF.
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


# Test that ingest() automatically detects and loads a TXT file.
def test_ingest_txt_file(tmp_path):

    # Create a temporary TXT file.
    file_path = tmp_path / "automatic.txt"

    # Write knowledge into the TXT file.
    file_path.write_text(
        "Artificial intelligence can learn from data.",
        encoding="utf-8",
    )

    # Create a temporary database.
    database_path = tmp_path / "knowledge.db"

    # Create the ingestion service.
    service, storage = create_service(database_path)

    # Let the service automatically detect the TXT file.
    document = service.ingest(file_path)

    # Make sure the correct content was loaded.
    assert document.content == (
        "Artificial intelligence can learn from data."
    )

    # Make sure the source type was detected as text.
    assert document.source_type == "text"

    # Make sure chunks were created.
    chunks = storage.get_chunks(str(document.id))

    # Make sure at least one chunk exists.
    assert len(chunks) >= 1


# Test that ingest() automatically detects a PDF file.
def test_ingest_pdf_automatically(tmp_path):

    # Create the temporary PDF path.
    file_path = tmp_path / "automatic.pdf"

    # Create a new PDF document.
    pdf = fitz.open()

    # Add a page to the PDF.
    page = pdf.new_page()

    # Write knowledge onto the PDF page.
    page.insert_text(
        (72, 72),
        "Machine learning uses data to find patterns.",
    )

    # Save the PDF.
    pdf.save(file_path)

    # Close the PDF.
    pdf.close()

    # Create a temporary database.
    database_path = tmp_path / "knowledge.db"

    # Create the ingestion service.
    service, storage = create_service(database_path)

    # Let the service automatically detect the PDF.
    document = service.ingest(file_path)

    # Make sure the PDF text was extracted.
    assert "Machine learning uses data to find patterns." in document.content

    # Make sure the source type was detected as PDF.
    assert document.source_type == "pdf"

    # Make sure chunks were created.
    chunks = storage.get_chunks(str(document.id))

    # Make sure at least one chunk exists.
    assert len(chunks) >= 1


# Test that uppercase file extensions are supported.
def test_ingest_uppercase_extension(tmp_path):

    # Create a TXT file with an uppercase extension.
    file_path = tmp_path / "uppercase.TXT"

    # Write text into the file.
    file_path.write_text(
        "Uppercase extensions should work.",
        encoding="utf-8",
    )

    # Create a temporary database.
    database_path = tmp_path / "knowledge.db"

    # Create the ingestion service.
    service, _ = create_service(database_path)

    # Ingest the file automatically.
    document = service.ingest(file_path)

    # Verify that the uppercase extension was handled correctly.
    assert document.source_type == "text"


# Test that unsupported file types are rejected.
def test_ingest_unsupported_file(tmp_path):

    # Create a file with an unsupported extension.
    file_path = tmp_path / "unsupported.jpg"

    # Create a small placeholder file.
    file_path.write_bytes(
        b"not a supported knowledge file"
    )

    # Create a temporary database.
    database_path = tmp_path / "knowledge.db"

    # Create the ingestion service.
    service, _ = create_service(database_path)

    # Verify that unsupported file types are rejected.
    with pytest.raises(ValueError):

        # Try to ingest the unsupported file.
        service.ingest(file_path)


# Test that ingestion creates the expected chunks.
def test_ingestion_creates_chunks(tmp_path):

    # Create a temporary text file.
    file_path = tmp_path / "chunked.txt"

    # Create enough text to make chunking observable.
    content = (
        "Python is a programming language. "
        "Python is useful for artificial intelligence. "
        "Machine learning uses data to discover patterns. "
        "Neural networks are used in many AI systems."
    )

    # Write the content to the file.
    file_path.write_text(
        content,
        encoding="utf-8",
    )

    # Create a temporary database.
    database_path = tmp_path / "knowledge.db"

    # Use a small chunk size so the test definitely creates
    # multiple chunks.
    chunker = TextChunker(
        chunk_size=50,
        overlap=10,
    )

    # Create the ingestion service.
    service, storage = create_service(
        database_path,
        chunker=chunker,
    )

    # Ingest the document.
    document = service.ingest_text_file(file_path)

    # Retrieve all chunks belonging to the document.
    chunks = storage.get_chunks(str(document.id))

    # Make sure multiple chunks were created.
    assert len(chunks) > 1

    # Make sure every chunk belongs to the same document.
    assert all(
        chunk.document_id == str(document.id)
        for chunk in chunks
    )

    # Make sure chunk indexes start at zero.
    assert chunks[0].chunk_index == 0

    # Make sure indexes are sequential.
    assert [chunk.chunk_index for chunk in chunks] == list(
        range(len(chunks))
    )

    # Make sure every chunk contains actual content.
    assert all(
        chunk.content.strip()
        for chunk in chunks
    )


# Test that a short document produces one chunk.
def test_short_document_creates_one_chunk(tmp_path):

    # Create a temporary text file.
    file_path = tmp_path / "short.txt"

    # Write a short piece of knowledge.
    file_path.write_text(
        "Artificial intelligence learns from data.",
        encoding="utf-8",
    )

    # Create a temporary database.
    database_path = tmp_path / "knowledge.db"

    # Create the ingestion service.
    service, storage = create_service(database_path)

    # Ingest the document.
    document = service.ingest_text_file(file_path)

    # Retrieve the chunks.
    chunks = storage.get_chunks(str(document.id))

    # A short document should fit inside one chunk.
    assert len(chunks) == 1

    # The first chunk should have index zero.
    assert chunks[0].chunk_index == 0

    # The chunk should contain the document content.
    assert chunks[0].content == document.content