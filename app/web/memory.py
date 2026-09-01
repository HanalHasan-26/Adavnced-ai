from __future__ import annotations

import re
import sqlite3

from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from config.settings import DATA_DIR


DATABASE_PATH = (
    DATA_DIR / "web_memory.db"
)


class WebMemory:

    # ---------------------------------------------------------
    # DEFAULT FRESHNESS
    # ---------------------------------------------------------

    DEFAULT_TTL_SECONDS = 30 * 24 * 60 * 60

    CURRENT_TTL_SECONDS = 10 * 60

    NEWS_TTL_SECONDS = 15 * 60

    PRICE_TTL_SECONDS = 5 * 60

    WEATHER_TTL_SECONDS = 15 * 60

    POPULATION_TTL_SECONDS = 24 * 60 * 60

    # ---------------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------------

    def __init__(
        self,
        database_path: Path = DATABASE_PATH,
    ) -> None:

        self.database_path = database_path

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_database()

    # =========================================================
    # DATABASE
    # =========================================================

    def _initialize_database(
        self,
    ) -> None:

        with sqlite3.connect(
            self.database_path
        ) as connection:

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS web_memories (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    normalized_query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    research TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_web_memories_normalized_query
                ON web_memories(normalized_query)
                """
            )

            connection.commit()

    # =========================================================
    # NORMALIZATION
    # =========================================================

    def normalize_query(
        self,
        query: str,
    ) -> str:

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        query = query.strip().lower()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        # Remove explicit command prefixes.
        if query.startswith("web:"):
            query = query[4:].strip()

        if query.startswith("web :"):
            query = query[5:].strip()

        # Keep words and numbers.
        words = re.findall(
            r"\b[\w]+\b",
            query,
            flags=re.UNICODE,
        )

        return " ".join(words)

    # =========================================================
    # TTL
    # =========================================================

    def get_ttl_seconds(
        self,
        query: str,
    ) -> int:

        normalized = self.normalize_query(
            query
        )

        current_indicators = (
            "current",
            "right now",
            "today",
            "live",
            "now",
        )

        news_indicators = (
            "news",
            "latest news",
            "breaking",
            "recent news",
        )

        price_indicators = (
            "price",
            "gold price",
            "silver price",
            "bitcoin price",
            "forex price",
            "stock price",
            "market price",
        )

        weather_indicators = (
            "weather",
            "temperature",
            "forecast",
        )

        population_indicators = (
            "population",
        )

        if any(
            indicator in normalized
            for indicator in price_indicators
        ):
            return self.PRICE_TTL_SECONDS

        if any(
            indicator in normalized
            for indicator in news_indicators
        ):
            return self.NEWS_TTL_SECONDS

        if any(
            indicator in normalized
            for indicator in weather_indicators
        ):
            return self.WEATHER_TTL_SECONDS

        if any(
            indicator in normalized
            for indicator in population_indicators
        ):
            return self.POPULATION_TTL_SECONDS

        if any(
            indicator in normalized
            for indicator in current_indicators
        ):
            return self.CURRENT_TTL_SECONDS

        return self.DEFAULT_TTL_SECONDS

    # =========================================================
    # TOKENIZATION
    # =========================================================

    def _tokens(
        self,
        query: str,
    ) -> set[str]:

        normalized = self.normalize_query(
            query
        )

        return set(
            normalized.split()
        )

    # =========================================================
    # SIMILARITY
    # =========================================================

    def _similarity(
        self,
        query_a: str,
        query_b: str,
    ) -> float:

        tokens_a = self._tokens(
            query_a
        )

        tokens_b = self._tokens(
            query_b
        )

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = (
            tokens_a & tokens_b
        )

        union = (
            tokens_a | tokens_b
        )

        if not union:
            return 0.0

        return (
            len(intersection)
            / len(union)
        )

    # =========================================================
    # SAVE
    # =========================================================

    def save(
        self,
        query: str,
        answer: str,
        research: str = "",
        ttl_seconds: int | None = None,
    ) -> str:

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        if not isinstance(answer, str):
            raise ValueError(
                "answer must be a string."
            )

        query = query.strip()
        answer = answer.strip()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        if not answer:
            raise ValueError(
                "answer cannot be empty."
            )

        if not isinstance(
            research,
            str,
        ):
            research = ""

        if ttl_seconds is None:
            ttl_seconds = (
                self.get_ttl_seconds(
                    query
                )
            )

        if ttl_seconds <= 0:
            raise ValueError(
                "ttl_seconds must be greater than 0."
            )

        now = datetime.now()

        expires = (
            now
            + timedelta(
                seconds=ttl_seconds
            )
        )

        memory_id = str(
            uuid4()
        )

        normalized_query = (
            self.normalize_query(
                query
            )
        )

        with sqlite3.connect(
            self.database_path
        ) as connection:

            connection.execute(
                """
                INSERT INTO web_memories (
                    id,
                    query,
                    normalized_query,
                    answer,
                    research,
                    created_at,
                    expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    query,
                    normalized_query,
                    answer,
                    research,
                    now.isoformat(),
                    expires.isoformat(),
                ),
            )

            connection.commit()

        return memory_id

    # =========================================================
    # FIND
    # =========================================================

    def find(
        self,
        query: str,
        min_similarity: float = 0.60,
    ) -> dict | None:

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        query = query.strip()

        if not query:
            return None

        if not (
            0.0
            <= min_similarity
            <= 1.0
        ):
            raise ValueError(
                "min_similarity must be between 0 and 1."
            )

        now = datetime.now()

        with sqlite3.connect(
            self.database_path
        ) as connection:

            connection.row_factory = (
                sqlite3.Row
            )

            rows = connection.execute(
                """
                SELECT
                    id,
                    query,
                    normalized_query,
                    answer,
                    research,
                    created_at,
                    expires_at
                FROM web_memories
                ORDER BY created_at DESC
                """
            ).fetchall()

        best = None
        best_score = 0.0

        for row in rows:

            try:
                expires_at = (
                    datetime.fromisoformat(
                        row["expires_at"]
                    )
                )

            except ValueError:
                continue

            if expires_at <= now:
                continue

            score = self._similarity(
                query,
                row["normalized_query"],
            )

            if score < min_similarity:
                continue

            if score > best_score:

                best_score = score

                best = {
                    "id": row["id"],
                    "query": row["query"],
                    "normalized_query": (
                        row["normalized_query"]
                    ),
                    "answer": row["answer"],
                    "research": row["research"],
                    "created_at": (
                        row["created_at"]
                    ),
                    "expires_at": (
                        row["expires_at"]
                    ),
                    "similarity": score,
                }

        return best

    # =========================================================
    # DELETE EXPIRED
    # =========================================================

    def delete_expired(
        self,
    ) -> int:

        now = datetime.now().isoformat()

        with sqlite3.connect(
            self.database_path
        ) as connection:

            cursor = connection.execute(
                """
                DELETE FROM web_memories
                WHERE expires_at <= ?
                """,
                (now,),
            )

            connection.commit()

            return cursor.rowcount

    # =========================================================
    # LIST
    # =========================================================

    def list(
        self,
    ) -> list[dict]:

        with sqlite3.connect(
            self.database_path
        ) as connection:

            connection.row_factory = (
                sqlite3.Row
            )

            rows = connection.execute(
                """
                SELECT
                    id,
                    query,
                    answer,
                    research,
                    created_at,
                    expires_at
                FROM web_memories
                ORDER BY created_at DESC
                """
            ).fetchall()

        return [
            {
                "id": row["id"],
                "query": row["query"],
                "answer": row["answer"],
                "research": row["research"],
                "created_at": row[
                    "created_at"
                ],
                "expires_at": row[
                    "expires_at"
                ],
            }
            for row in rows
        ]

    # =========================================================
    # DELETE
    # =========================================================

    def delete(
        self,
        memory_id: str,
    ) -> bool:

        if not isinstance(
            memory_id,
            str,
        ):
            raise ValueError(
                "memory_id must be a string."
            )

        memory_id = memory_id.strip()

        if not memory_id:
            raise ValueError(
                "memory_id cannot be empty."
            )

        with sqlite3.connect(
            self.database_path
        ) as connection:

            cursor = connection.execute(
                """
                DELETE FROM web_memories
                WHERE id = ?
                """,
                (memory_id,),
            )

            connection.commit()

            return cursor.rowcount > 0