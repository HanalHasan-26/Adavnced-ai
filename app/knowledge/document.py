# Import dataclass so we can create a clean data structure.
from dataclasses import dataclass

# Import datetime so we can record when knowledge was created.
from datetime import datetime

# Import UUID tools so every knowledge item gets a unique ID.
from uuid import UUID, uuid4


# Define the structure of a knowledge document.
@dataclass
class KnowledgeDocument:

    # Give every knowledge document a unique identifier.
    id: UUID

    # Store the actual text/content of the knowledge.
    content: str

    # Store where the knowledge came from.
    source: str

    # Store the type of knowledge source, such as PDF, TXT, or web.
    source_type: str

    # Store when this knowledge was added.
    created_at: datetime

    # Create a new knowledge document with an automatically generated ID.
    @classmethod
    def create(
        cls,
        content: str,
        source: str,
        source_type: str,
    ) -> "KnowledgeDocument":

        # Return a new document with the supplied information.
        return cls(
            id=uuid4(),
            content=content,
            source=source,
            source_type=source_type,
            created_at=datetime.now(),
        )