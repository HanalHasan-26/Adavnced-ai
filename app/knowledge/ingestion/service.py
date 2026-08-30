# Import Path so we can work with file paths.
from pathlib import Path

# Import our knowledge document model.
from app.knowledge.document import KnowledgeDocument

# Import our persistent knowledge storage.
from app.knowledge.storage import KnowledgeStorage

# Import the text-file loader.
from app.knowledge.ingestion.text_loader import TextLoader


# Create the service responsible for turning files into stored knowledge.
class KnowledgeIngestionService:

    # Initialize the ingestion service.
    def __init__(
        self,
        storage: KnowledgeStorage,
        text_loader: TextLoader,
    ):

        # Store the knowledge storage dependency.
        self.storage = storage

        # Store the text loader dependency.
        self.text_loader = text_loader

    # Ingest one text file into the knowledge database.
    def ingest_text_file(self, file_path: Path) -> KnowledgeDocument:

        # Read the contents of the text file.
        content = self.text_loader.load(file_path)

        # Create a KnowledgeDocument from the extracted text.
        document = KnowledgeDocument.create(
            content=content,
            source=str(file_path),
            source_type="text",
        )

        # Save the document into persistent storage.
        self.storage.add(document)

        # Return the newly created knowledge document.
        return document