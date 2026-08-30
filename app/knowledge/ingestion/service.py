# Import Path so we can work with file paths.
from pathlib import Path

# Import UUID so we can create unique IDs for knowledge chunks.
from uuid import uuid4

# Import our knowledge document model.
from app.knowledge.document import KnowledgeDocument

# Import our knowledge chunk model.
from app.knowledge.chunking.chunk import KnowledgeChunk

# Import our text chunker.
from app.knowledge.chunking.text_chunker import TextChunker

# Import our persistent knowledge storage.
from app.knowledge.storage import KnowledgeStorage

# Import the text-file loader.
from app.knowledge.ingestion.text_loader import TextLoader

# Import the PDF loader.
from app.knowledge.ingestion.pdf_loader import PDFLoader


# Create the service responsible for turning files into stored knowledge.
class KnowledgeIngestionService:

    # Initialize the ingestion service.
    def __init__(
        self,
        storage: KnowledgeStorage,
        text_loader: TextLoader,
        pdf_loader: PDFLoader,
        chunker: TextChunker | None = None,
    ):

        # Store the knowledge storage dependency.
        self.storage = storage

        # Store the text loader dependency.
        self.text_loader = text_loader

        # Store the PDF loader dependency.
        self.pdf_loader = pdf_loader

        # Use the supplied chunker when one is provided.
        # Otherwise create one with the default configuration.
        self.chunker = chunker or TextChunker()

    # Ingest one text file into the knowledge database.
    def ingest_text_file(self, file_path: Path) -> KnowledgeDocument:

        # Read the contents of the text file.
        content = self.text_loader.load(file_path)

        # Convert the extracted text into stored knowledge.
        return self._create_and_store_document(
            content=content,
            source=file_path,
            source_type="text",
        )

    # Ingest one PDF file into the knowledge database.
    def ingest_pdf_file(self, file_path: Path) -> KnowledgeDocument:

        # Extract text from the PDF.
        content = self.pdf_loader.load(file_path)

        # Convert the extracted text into stored knowledge.
        return self._create_and_store_document(
            content=content,
            source=file_path,
            source_type="pdf",
        )

    # Ingest a supported file automatically based on its extension.
    def ingest(self, file_path: Path) -> KnowledgeDocument:

        # Convert the path to lowercase so extension checks are case-insensitive.
        suffix = file_path.suffix.lower()

        # Use the text loader for TXT files.
        if suffix == ".txt":
            return self.ingest_text_file(file_path)

        # Use the PDF loader for PDF files.
        if suffix == ".pdf":
            return self.ingest_pdf_file(file_path)

        # Reject file types that we don't support yet.
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}"
        )

    # Create a document, split it into chunks, and store everything.
    def _create_and_store_document(
        self,
        content: str,
        source: Path,
        source_type: str,
    ) -> KnowledgeDocument:

        # Reject files that contain no meaningful text.
        if not content.strip():

            # Stop instead of storing empty knowledge.
            raise ValueError(
                "Cannot ingest a file with empty content."
            )

        # Create a new knowledge document.
        document = KnowledgeDocument.create(
            content=content,
            source=str(source),
            source_type=source_type,
        )

        # Save the document permanently.
        self.storage.add(document)

        # Split the document content into smaller chunks.
        chunks = self.chunker.chunk(content)

        # Create and store every chunk.
        for chunk_index, chunk_content in enumerate(chunks):

            # Create a KnowledgeChunk belonging to this document.
            chunk = KnowledgeChunk(
                id=str(uuid4()),
                document_id=str(document.id),
                chunk_index=chunk_index,
                content=chunk_content,
            )

            # Save the chunk permanently.
            self.storage.add_chunk(chunk)

        # Return the newly created document.
        return document