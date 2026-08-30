# Import Path so we can work with file paths.
from pathlib import Path

# Import our knowledge document model.
from app.knowledge.document import KnowledgeDocument

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
    ):

        # Store the knowledge storage dependency.
        self.storage = storage

        # Store the text loader dependency.
        self.text_loader = text_loader

        # Store the PDF loader dependency.
        self.pdf_loader = pdf_loader

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

    # Create and store a KnowledgeDocument.
    def _create_and_store_document(
        self,
        content: str,
        source: Path,
        source_type: str,
    ) -> KnowledgeDocument:

        # Reject files that contain no meaningful text.
        if not content.strip():

            # Stop instead of storing empty knowledge.
            raise ValueError("Cannot ingest a file with empty content.")

        # Create a new knowledge document.
        document = KnowledgeDocument.create(
            content=content,
            source=str(source),
            source_type=source_type,
        )

        # Save the document permanently.
        self.storage.add(document)

        # Return the newly created document.
        return document