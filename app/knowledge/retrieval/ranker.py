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

        # Convert the query to lowercase.
        query_lower = query.lower()

        # Convert the chunk content to lowercase.
        content_lower = chunk.content.lower()

        # Split the query into individual words.
        query_words = list(
            dict.fromkeys(
                query_lower.split()
            )
        )

        # Return zero when the query contains no words.
        if not query_words:
            return 0

        # -------------------------------------------------
        # 1. EXACT PHRASE MATCH
        # -------------------------------------------------

        # Give a strong bonus when the complete query
        # appears inside the chunk.
        exact_phrase_score = 0

        if query_lower in content_lower:
            exact_phrase_score = 1000

        # -------------------------------------------------
        # 2. INDIVIDUAL TERM MATCHING
        # -------------------------------------------------

        matched_terms = 0
        occurrence_score = 0

        for word in query_words:

            # Ignore extremely short terms because they
            # create too many accidental matches.
            if len(word) <= 2:
                continue

            # Check whether the exact word appears.
            if word in content_lower:

                matched_terms += 1

                # Count occurrences as a secondary signal.
                occurrence_score += content_lower.count(
                    word
                )

        # -------------------------------------------------
        # 3. QUERY COVERAGE
        # -------------------------------------------------

        meaningful_words = [
            word
            for word in query_words
            if len(word) > 2
        ]

        coverage_score = 0

        if meaningful_words:

            coverage = (
                matched_terms
                / len(meaningful_words)
            )

            # Convert coverage into a score.
            coverage_score = int(
                coverage * 500
            )

        # -------------------------------------------------
        # 4. MULTI-WORD PHRASES
        # -------------------------------------------------

        # Look for adjacent pairs of query words.
        phrase_score = 0

        for index in range(
            len(query_words) - 1
        ):

            first = query_words[index]
            second = query_words[index + 1]

            if len(first) <= 2 or len(second) <= 2:
                continue

            phrase = (
                f"{first} {second}"
            )

            if phrase in content_lower:

                # Reward matching phrases because they
                # preserve more meaning than isolated words.
                phrase_score += 250

        # -------------------------------------------------
        # 5. FINAL SCORE
        # -------------------------------------------------

        return (
            exact_phrase_score
            + phrase_score
            + coverage_score
            + (matched_terms * 100)
            + occurrence_score
        )

    # Rank chunks according to relevance.
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
            raise ValueError(
                "limit must be greater than 0."
            )

        # Store each chunk together with its scores.
        scored_chunks: list[
            tuple[
                int,
                int,
                int,
                KnowledgeChunk,
            ]
        ] = []

        # Score every candidate chunk.
        for chunk in chunks:

            # Calculate the complete relevance score.
            relevance_score = self.score(
                query=query,
                chunk=chunk,
            )

            # Calculate occurrence score separately
            # for deterministic secondary sorting.
            content_lower = chunk.content.lower()

            query_words = list(
                dict.fromkeys(
                    query.lower().split()
                )
            )

            occurrence_score = sum(
                content_lower.count(word)
                for word in query_words
                if len(word) > 2
            )

            # Count how many meaningful query terms match.
            matched_terms = sum(
                1
                for word in query_words
                if len(word) > 2
                and word in content_lower
            )

            # Only keep relevant chunks.
            if relevance_score > 0:

                scored_chunks.append(
                    (
                        relevance_score,
                        matched_terms,
                        occurrence_score,
                        chunk,
                    )
                )

        # Sort from most relevant to least relevant.
        #
        # Priority:
        #   1. Overall relevance
        #   2. Number of matched terms
        #   3. Number of occurrences
        #   4. Earlier chunk index for stable ordering
        scored_chunks.sort(
            key=lambda item: (
                -item[0],
                -item[1],
                -item[2],
                item[3].chunk_index,
            )
        )

        # Return only the requested number of chunks.
        return [
            chunk
            for _, _, _, chunk
            in scored_chunks[:limit]
        ]