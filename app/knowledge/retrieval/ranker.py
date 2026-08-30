# Import the knowledge chunk model.
from app.knowledge.chunking.chunk import KnowledgeChunk


# Create a component responsible for ranking knowledge chunks.
class ChunkRanker:

    # Calculate the relevance score for one chunk.
    def score(
        self,
        query: str,
        chunk: KnowledgeChunk,
    ) -> int:

        # Remove unnecessary whitespace.
        query = query.strip()

        # Return zero for an empty query.
        if not query:
            return 0

        # Split the query into individual words.
        query_words = query.lower().split()

        # Convert the chunk content to lowercase.
        content = chunk.content.lower()

        # Count how many different query terms
        # appear in the chunk.
        matched_terms = sum(
            1
            for word in query_words
            if word in content
        )

        # Count the total number of occurrences
        # of all query terms.
        occurrence_score = sum(
            content.count(word)
            for word in query_words
        )

        # Give priority to chunks containing
        # more distinct query terms.
        return (
            matched_terms * 100
        ) + occurrence_score

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

        # Store each chunk together with its relevance scores.
        scored_chunks: list[
            tuple[int, int, KnowledgeChunk]
        ] = []

        # Score every supplied chunk.
        for chunk in chunks:

            # Convert the chunk content to lowercase.
            content = chunk.content.lower()

            # Calculate how many different query terms
            # appear in the chunk.
            matched_terms = sum(
                1
                for word in query_words
                if word in content
            )

            # Calculate the total number of occurrences
            # of all query terms.
            occurrence_score = sum(
                content.count(word)
                for word in query_words
            )

            # Calculate the final relevance score.
            relevance_score = (
                matched_terms * 100
            ) + occurrence_score

            # Only keep chunks containing at least one
            # query term.
            if relevance_score > 0:

                # Store:
                # 1. relevance score
                # 2. occurrence score
                # 3. chunk
                scored_chunks.append(
                    (
                        relevance_score,
                        occurrence_score,
                        chunk,
                    )
                )

        # Sort from most relevant to least relevant.
        #
        # First:
        #   higher relevance score
        #
        # Then:
        #   higher occurrence score
        #
        # Finally:
        #   lower chunk index for stable ordering.
        scored_chunks.sort(
            key=lambda item: (
                -item[0],
                -item[1],
                item[2].chunk_index,
            )
        )

        # Return only the requested number of chunks.
        return [
            chunk
            for _, _, chunk in scored_chunks[:limit]
        ]