# Import dataclass so we can create a simple data model.
from dataclasses import dataclass


# Represent one chunk of knowledge extracted from a document.
@dataclass(frozen=True)
class KnowledgeChunk:

    # Store the unique ID of the chunk.
    id: str

    # Store the ID of the original document.
    document_id: str

    # Store the position of this chunk inside the document.
    chunk_index: int

    # Store the actual text contained in this chunk.
    content: str