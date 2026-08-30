# Import the knowledge chunk model.
from app.knowledge.chunking.chunk import KnowledgeChunk

# Import persistent knowledge storage.
from app.knowledge.storage import KnowledgeStorage

# Import the chunk ranking component.
from app.knowledge.retrieval.ranker import ChunkRanker


# Create a component responsible for retrieving knowledge chunks.
class ChunkRetriever:

    # Initialize the retriever.
    def __init__(
        self,
        storage: KnowledgeStorage,
        ranker: ChunkRanker | None = None,
    ):

        # Store the storage dependency.
        self.storage = storage

        # Use the supplied ranker when one is provided.
        # Otherwise create a default ranker.
        self.ranker = ranker or ChunkRanker()

    # Retrieve the most relevant chunks for a query.
    def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> list[KnowledgeChunk]:

        # Remove unnecessary whitespace.
        query = query.strip()

        # Return no results for an empty query.
        if not query:
            return []

        # Make sure the requested limit is valid.
        if limit <= 0:
            raise ValueError("limit must be greater than 0.")

        # Search through stored documents to find candidates.
        documents = self.storage.search(query)

        # Store candidate chunks.
        candidate_chunks: list[KnowledgeChunk] = []

        # Examine every matching document.
        for document in documents:

            # Retrieve all chunks belonging to this document.
            chunks = self.storage.get_chunks(
                str(document.id)
            )

            # Add the document's chunks to the candidates.
            candidate_chunks.extend(chunks)

        # Let the ranker determine which chunks are most relevant.
        return self.ranker.rank(
            query=query,
            chunks=candidate_chunks,
            limit=limit,
        )