# Import the knowledge chunk model.
from app.knowledge.chunking.chunk import KnowledgeChunk

# Import persistent knowledge storage.
from app.knowledge.storage import KnowledgeStorage

# Import the chunk ranking component.
from app.knowledge.retrieval.ranker import ChunkRanker

# Import the query normalization component.
from app.knowledge.retrieval.query_normalizer import QueryNormalizer

# Import the query tokenizer.
from app.knowledge.retrieval.query_tokenizer import QueryTokenizer


# Create a component responsible for retrieving knowledge chunks.
class ChunkRetriever:

    # Initialize the retriever.
    def __init__(
        self,
        storage: KnowledgeStorage,
        ranker: ChunkRanker | None = None,
        normalizer: QueryNormalizer | None = None,
        tokenizer: QueryTokenizer | None = None,
    ):

        # Store the storage dependency.
        self.storage = storage

        # Use the supplied ranker when provided.
        # Otherwise create a default ranker.
        self.ranker = ranker or ChunkRanker()

        # Use the supplied normalizer when provided.
        # Otherwise create a default normalizer.
        self.normalizer = normalizer or QueryNormalizer()

        # Use the supplied tokenizer when provided.
        # Otherwise create a default tokenizer.
        self.tokenizer = tokenizer or QueryTokenizer()

    # Retrieve the most relevant chunks for a query.
    def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> list[KnowledgeChunk]:

        # Normalize the user's query.
        query = self.normalizer.normalize(query)

        # Return no results for an empty normalized query.
        if not query:
            return []

        # Make sure the requested limit is valid.
        if limit <= 0:
            raise ValueError("limit must be greater than 0.")

        # Convert the normalized query into individual terms.
        terms = self.tokenizer.tokenize(query)

        # Store candidate chunks.
        candidate_chunks: list[KnowledgeChunk] = []

        if not terms:
            return []

        # Keep track of chunk IDs so the same chunk
        # is not added more than once.
        seen_chunk_ids: set[str] = set()

        # Search for each individual query term.
        for term in terms:

            # Retrieve candidate chunks containing this term.
            chunks = self.storage.search_chunks(
                query=term,
                limit=max(limit * 10, 50),
            )

            # Add each unique chunk to the candidate set.
            for chunk in chunks:

                # Skip chunks already collected from another term.
                if chunk.id in seen_chunk_ids:
                    continue

                # Remember this chunk ID.
                seen_chunk_ids.add(chunk.id)

                # Add the chunk to the candidates.
                candidate_chunks.append(chunk)

        # Rank all collected candidates.
        return self.ranker.rank(
            query=query,
            chunks=candidate_chunks,
            limit=limit,
        )