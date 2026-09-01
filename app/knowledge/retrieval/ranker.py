from __future__ import annotations

# =========================================================
# STANDARD LIBRARY
# =========================================================

import re


# =========================================================
# KNOWLEDGE
# =========================================================

from app.knowledge.chunking.chunk import KnowledgeChunk


# =========================================================
# CHUNK RANKER
# =========================================================

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
        "when",
        "where",
        "which",
        "who",
        "with",
    }

    # Word tokenizer.
    #
    # This handles punctuation safely.
    #
    # Examples:
    #
    # "Support"          -> ["support"]
    # "support."         -> ["support"]
    # "support,"         -> ["support"]
    # "XAU/USD"          -> ["xau", "usd"]
    #
    WORD_PATTERN = re.compile(
        r"\b[\w]+\b",
        re.UNICODE,
    )

    # =====================================================
    # QUERY TOKENIZATION
    # =====================================================

    def _tokenize(
        self,
        text: str,
    ) -> list[str]:

        """
        Convert text into lowercase word tokens.

        Punctuation is ignored so that words at the
        beginning/end of sentences are counted correctly.
        """

        if not isinstance(text, str):
            raise ValueError(
                "text must be a string."
            )

        return [
            match.group(0).lower()
            for match in self.WORD_PATTERN.finditer(
                text
            )
        ]

    # =====================================================
    # QUERY WORDS
    # =====================================================

    def _query_words(
        self,
        query: str,
    ) -> list[str]:

        """
        Convert a query into meaningful searchable words.

        Stop words are removed and duplicate terms are
        removed while preserving their original order.
        """

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        # Normalize whitespace.
        query = query.strip().lower()

        # Return no words for an empty query.
        if not query:
            return []

        # Tokenize the query safely.
        words = self._tokenize(
            query
        )

        # Remove stop words and duplicate terms.
        return list(
            dict.fromkeys(
                word
                for word in words
                if word not in self.STOP_WORDS
            )
        )

    # =====================================================
    # PHRASE MATCHING
    # =====================================================

    def _contains_phrase(
        self,
        query_words: list[str],
        content_words: list[str],
    ) -> bool:

        """
        Check whether the complete meaningful query appears
        as a contiguous sequence of words.

        A phrase bonus is only applied when there are at
        least two meaningful query terms.

        This prevents:

            query = "support"

        from incorrectly receiving the phrase bonus.
        """

        # A single term is not treated as a phrase.
        if len(query_words) < 2:
            return False

        phrase_length = len(
            query_words
        )

        if len(content_words) < phrase_length:
            return False

        for index in range(
            len(content_words) - phrase_length + 1
        ):

            candidate = content_words[
                index:index + phrase_length
            ]

            if candidate == query_words:
                return True

        return False

    # =====================================================
    # SCORE
    # =====================================================

    def score(
        self,
        query: str,
        chunk: KnowledgeChunk,
    ) -> int:

        """
        Calculate the relevance score for one chunk.

        Scoring:

            matched distinct term = 100 points

            each occurrence of a matched term = 1 point

            exact multi-word phrase = 1000 points
        """

        # Remove unnecessary whitespace.
        query = query.strip()

        # Return zero for an empty query.
        if not query:
            return 0

        # Get meaningful query words.
        query_words = self._query_words(
            query
        )

        # Return zero when no meaningful words remain.
        if not query_words:
            return 0

        # Validate chunk.
        if not isinstance(
            chunk,
            KnowledgeChunk,
        ):
            raise ValueError(
                "chunk must be a KnowledgeChunk."
            )

        # Tokenize chunk content.
        content_words = self._tokenize(
            chunk.content
        )

        # Create a set for fast membership testing.
        content_word_set = set(
            content_words
        )

        # -------------------------------------------------
        # DISTINCT TERM SCORE
        # -------------------------------------------------

        matched_terms = sum(
            1
            for word in query_words
            if word in content_word_set
        )

        # -------------------------------------------------
        # OCCURRENCE SCORE
        # -------------------------------------------------

        occurrence_score = sum(
            content_words.count(word)
            for word in query_words
        )

        # -------------------------------------------------
        # PHRASE BONUS
        # -------------------------------------------------

        phrase_bonus = (
            1000
            if self._contains_phrase(
                query_words,
                content_words,
            )
            else 0
        )

        # -------------------------------------------------
        # FINAL SCORE
        # -------------------------------------------------

        return (
            phrase_bonus
            + (matched_terms * 100)
            + occurrence_score
        )

    # =====================================================
    # RANK
    # =====================================================

    def rank(
        self,
        query: str,
        chunks: list[KnowledgeChunk],
        limit: int = 5,
    ) -> list[KnowledgeChunk]:

        """
        Rank knowledge chunks according to relevance.

        Ranking priority:

            1. Higher relevance score
            2. Higher occurrence score
            3. Lower chunk index
        """

        # Remove unnecessary whitespace.
        query = query.strip()

        # Return no results for an empty query.
        if not query:
            return []

        # Validate limit.
        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        # Get meaningful query words.
        query_words = self._query_words(
            query
        )

        # Return no results when no meaningful terms exist.
        if not query_words:
            return []

        # Store chunks together with their scores.
        scored_chunks: list[
            tuple[
                int,
                int,
                KnowledgeChunk,
            ]
        ] = []

        # =================================================
        # SCORE EVERY CHUNK
        # =================================================

        for chunk in chunks:

            # Ignore invalid chunk objects.
            if not isinstance(
                chunk,
                KnowledgeChunk,
            ):
                continue

            # Tokenize chunk content.
            content_words = self._tokenize(
                chunk.content
            )

            # Create a set for fast membership testing.
            content_word_set = set(
                content_words
            )

            # -------------------------------------------------
            # DISTINCT MATCHES
            # -------------------------------------------------

            matched_terms = sum(
                1
                for word in query_words
                if word in content_word_set
            )

            # -------------------------------------------------
            # OCCURRENCES
            # -------------------------------------------------

            occurrence_score = sum(
                content_words.count(word)
                for word in query_words
            )

            # -------------------------------------------------
            # PHRASE BONUS
            # -------------------------------------------------

            phrase_bonus = (
                1000
                if self._contains_phrase(
                    query_words,
                    content_words,
                )
                else 0
            )

            # -------------------------------------------------
            # FINAL SCORE
            # -------------------------------------------------

            relevance_score = (
                phrase_bonus
                + (matched_terms * 100)
                + occurrence_score
            )

            # Ignore chunks with no meaningful match.
            if matched_terms <= 0:
                continue

            # Save the scored chunk.
            scored_chunks.append(
                (
                    relevance_score,
                    occurrence_score,
                    chunk,
                )
            )

        # =================================================
        # SORT RESULTS
        # =================================================

        scored_chunks.sort(
            key=lambda item: (
                # Higher relevance first.
                -item[0],

                # Higher occurrence count first.
                -item[1],

                # Lower chunk index first.
                item[2].chunk_index,
            )
        )

        # =================================================
        # RETURN LIMITED RESULTS
        # =================================================

        return [
            chunk
            for _, _, chunk in scored_chunks[:limit]
        ]