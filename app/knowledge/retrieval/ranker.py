# Import the knowledge chunk model.
from app.knowledge.chunking.chunk import KnowledgeChunk


# Create a component responsible for ranking knowledge chunks.
class ChunkRanker:

    # Words that usually do not carry useful search meaning.
    STOP_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }

    # Convert a query into meaningful searchable words.
    def _query_words(self, query: str) -> list[str]:

        # Normalize whitespace.
        query = query.strip().lower()

        # Return no words for an empty query.
        if not query:
            return []

        # Split the query into individual words.
        words = query.split()

        # Remove stop words and duplicate terms.
        return list(
            dict.fromkeys(
                word
                for word in words
                if word not in self.STOP_WORDS
            )
        )

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

        # Get meaningful query words.
        query_words = self._query_words(query)

        # Return zero when no meaningful words remain.
        if not query_words:
            return 0

        # Convert chunk content to lowercase.
        content = chunk.content.lower()

        # Split chunk content into complete words.
        content_words = set(
            content.split()
        )

        # Count meaningful query terms that appear.
        matched_terms = sum(
            1
            for word in query_words
            if word in content_words
        )

        # Count exact occurrences of each term.
        occurrence_score = sum(
            content_words and content.count(
                f" {word} "
            )
            for word in query_words
        )

        # Give a strong bonus when the complete query
        # appears inside the chunk.
        phrase_bonus = (
            1000
            if query.lower() in content
            else 0
        )

        # Give priority to chunks containing
        # more distinct meaningful terms.
        return (
            phrase_bonus
            + (matched_terms * 100)
            + occurrence_score
        )

    # Rank chunks according to their relevance.
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

        # Get meaningful query words.
        query_words = self._query_words(query)

        # Return no results when there are no meaningful terms.
        if not query_words:
            return []

        # Store chunks together with their scores.
        scored_chunks: list[
            tuple[int, int, KnowledgeChunk]
        ] = []

        # Score every candidate chunk.
        for chunk in chunks:

            # Convert chunk content to lowercase.
            content = chunk.content.lower()

            # Split content into complete words.
            content_words = set(
                content.split()
            )

            # Count matching query terms.
            matched_terms = sum(
                1
                for word in query_words
                if word in content_words
            )

            # Count occurrences.
            occurrence_score = sum(
                content.count(word)
                for word in query_words
            )

            # Give a large bonus for an exact phrase match.
            phrase_bonus = (
                1000
                if query.lower() in content
                else 0
            )

            # Calculate final relevance score.
            relevance_score = (
                phrase_bonus
                + (matched_terms * 100)
                + occurrence_score
            )

            # Ignore chunks with no meaningful match.
            if matched_terms > 0:

                scored_chunks.append(
                    (
                        relevance_score,
                        occurrence_score,
                        chunk,
                    )
                )

        # Sort from most relevant to least relevant.
        #
        # 1. Higher relevance score
        # 2. Higher occurrence score
        # 3. Lower chunk index
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