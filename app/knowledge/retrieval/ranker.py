# Import the knowledge chunk model.
from app.knowledge.chunking.chunk import KnowledgeChunk


# Create a component responsible for ranking knowledge chunks.
class ChunkRanker:

    # Rank chunks according to how relevant they are to a query.
    def rank(
        self,
        query: str,
        chunks: list[KnowledgeChunk],
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

        # Split the query into individual words.
        query_words = query.lower().split()

        # Store each chunk together with its relevance score.
        scored_chunks = []

        # Score every supplied chunk.
        for chunk in chunks:

            # Convert the chunk content to lowercase.
            content = chunk.content.lower()

            # Calculate the score.
            score = sum(
                content.count(word)
                for word in query_words
            )

            # Only keep chunks that contain at least one
            # query word.
            if score > 0:

                # Store the score and chunk together.
                scored_chunks.append(
                    (score, chunk)
                )

        # Sort by relevance score from highest to lowest.
        # Use chunk_index as a stable tie-breaker.
        scored_chunks.sort(
            key=lambda item: (
                -item[0],
                item[1].chunk_index,
            )
        )

        # Return only the requested number of chunks.
        return [
            chunk
            for _, chunk in scored_chunks[:limit]
        ]
    