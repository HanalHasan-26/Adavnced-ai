# Import the dataclass decorator.
from dataclasses import dataclass

# Import the knowledge chunk model.
from app.knowledge.chunking.chunk import KnowledgeChunk


# Represent the complete result of a knowledge retrieval operation.
@dataclass(frozen=True)
class KnowledgeRetrievalResult:

    # The original user query.
    query: str

    # The chunks selected as relevant knowledge.
    chunks: list[KnowledgeChunk]

    # The assembled context sent to the AI.
    context: str

    # Return the number of retrieved chunks.
    @property
    def count(self) -> int:
        return len(self.chunks)