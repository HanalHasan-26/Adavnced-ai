from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from config.settings import DATA_DIR


DATABASE_PATH = DATA_DIR / "memory.db"


class UserFacts:
    """
    Persistent structured facts explicitly stated by the user.

    User facts are trusted separately from normal conversation memory.

    Important rules:

    1. Only explicit user statements create facts.
    2. Assistant responses never create user facts.
    3. A newer explicit user statement replaces an older fact
       with the same fact key.
    4. User facts are authoritative over old conversation/assistant
       responses.
    5. Name is treated as a high-priority fact.
    """

    def __init__(
        self,
        database_path: Path = DATABASE_PATH,
    ):
        self.database_path = Path(database_path)

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_database()

    # =========================================================
    # DATABASE
    # =========================================================

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path
        )

        connection.row_factory = sqlite3.Row

        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_facts (
                    id TEXT PRIMARY KEY,
                    fact_key TEXT NOT NULL UNIQUE,
                    fact_value TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            connection.commit()

    # =========================================================
    # SET FACT
    # =========================================================

    def set_fact(
        self,
        fact_key: str,
        fact_value: str,
    ) -> str:

        if (
            not isinstance(fact_key, str)
            or not fact_key.strip()
        ):
            raise ValueError(
                "fact_key must be a non-empty string."
            )

        if (
            not isinstance(fact_value, str)
            or not fact_value.strip()
        ):
            raise ValueError(
                "fact_value must be a non-empty string."
            )

        fact_key = fact_key.strip().lower()
        fact_value = fact_value.strip()

        now = datetime.now().isoformat()

        with self._connect() as connection:

            existing = connection.execute(
                """
                SELECT id
                FROM user_facts
                WHERE fact_key = ?
                """,
                (fact_key,),
            ).fetchone()

            if existing is None:

                fact_id = str(uuid4())

                connection.execute(
                    """
                    INSERT INTO user_facts (
                        id,
                        fact_key,
                        fact_value,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        fact_id,
                        fact_key,
                        fact_value,
                        now,
                        now,
                    ),
                )

            else:

                fact_id = existing["id"]

                connection.execute(
                    """
                    UPDATE user_facts
                    SET
                        fact_value = ?,
                        updated_at = ?
                    WHERE fact_key = ?
                    """,
                    (
                        fact_value,
                        now,
                        fact_key,
                    ),
                )

            connection.commit()

        return fact_id

    # =========================================================
    # GET FACT
    # =========================================================

    def get_fact(
        self,
        fact_key: str,
    ) -> dict | None:

        if (
            not isinstance(fact_key, str)
            or not fact_key.strip()
        ):
            raise ValueError(
                "fact_key must be a non-empty string."
            )

        normalized_key = (
            fact_key.strip().lower()
        )

        with self._connect() as connection:

            row = connection.execute(
                """
                SELECT
                    id,
                    fact_key,
                    fact_value,
                    created_at,
                    updated_at
                FROM user_facts
                WHERE fact_key = ?
                """,
                (normalized_key,),
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    # =========================================================
    # LIST FACTS
    # =========================================================

    def list_facts(self) -> list[dict]:

        with self._connect() as connection:

            rows = connection.execute(
                """
                SELECT
                    id,
                    fact_key,
                    fact_value,
                    created_at,
                    updated_at
                FROM user_facts
                ORDER BY updated_at DESC
                """
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    # =========================================================
    # DELETE FACT
    # =========================================================

    def delete_fact(
        self,
        fact_key: str,
    ) -> bool:

        if (
            not isinstance(fact_key, str)
            or not fact_key.strip()
        ):
            raise ValueError(
                "fact_key must be a non-empty string."
            )

        normalized_key = (
            fact_key.strip().lower()
        )

        with self._connect() as connection:

            cursor = connection.execute(
                """
                DELETE FROM user_facts
                WHERE fact_key = ?
                """,
                (normalized_key,),
            )

            connection.commit()

            return cursor.rowcount > 0

    # =========================================================
    # VALUE CLEANING
    # =========================================================

    @staticmethod
    def _clean_value(
        value: str,
    ) -> str:

        value = value.strip()

        # Normalize repeated whitespace.
        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        # Remove common conversational endings.
        value = re.sub(
            r"""
            \s*,?\s*
            (?:
                (?:u|you)\s+got(?:\s+it)?
                |
                remember(?:\s+that)?
                |
                right
                |
                okay
                |
                ok
            )
            \s*[.!?]*$
            """,
            "",
            value,
            flags=re.IGNORECASE | re.VERBOSE,
        )

        # Remove trailing punctuation.
        value = value.strip(
            " \t\r\n.,!?;:"
        )

        return value

    # =========================================================
    # NAME EXTRACTION
    # =========================================================

    def _extract_name(
        self,
        message: str,
    ) -> str | None:

        patterns = [
            # My name is Hanal
            r"""
            ^\s*
            (?:actually\s*[,;:]?\s*)?
            my\s+
            (?:actual\s+)?
            name\s+is\s+
            (?P<value>
                [A-Za-z][A-Za-z'\-]*
                (?:\s+[A-Za-z][A-Za-z'\-]*){0,3}
            )
            """,

            # I am Hanal
            # I'm Hanal
            r"""
            ^\s*
            (?:actually\s*[,;:]?\s*)?
            i\s*(?:am|'m)\s+
            (?P<value>
                [A-Za-z][A-Za-z'\-]*
                (?:\s+[A-Za-z][A-Za-z'\-]*){0,3}
            )
            """,

            # Call me Hanal
            r"""
            ^\s*
            (?:actually\s*[,;:]?\s*)?
            call\s+me\s+
            (?P<value>
                [A-Za-z][A-Za-z'\-]*
                (?:\s+[A-Za-z][A-Za-z'\-]*){0,3}
            )
            """,

            # I'm Hanal, remember that
            # My name is Hanal, you got it?
            r"""
            ^\s*
            (?:actually\s*[,;:]?\s*)?
            (?:
                my\s+(?:actual\s+)?name\s+is
                |
                i\s*(?:am|'m)
                |
                call\s+me
            )
            \s+
            (?P<value>
                [A-Za-z][A-Za-z'\-]*
                (?:\s+[A-Za-z][A-Za-z'\-]*){0,3}
            )
            """,
        ]

        for pattern in patterns:

            match = re.match(
                pattern,
                message,
                flags=re.IGNORECASE | re.VERBOSE,
            )

            if not match:
                continue

            value = self._clean_value(
                match.group("value")
            )

            if not value:
                continue

            if value.lower() in {
                "not",
                "unknown",
                "nothing",
                "nobody",
            }:
                continue

            return value

        return None

    # =========================================================
    # FACT EXTRACTION
    # =========================================================

    def extract_facts(
        self,
        message: str,
    ) -> dict[str, str]:
        """
        Extract only facts explicitly asserted about the user.

        Assistant messages must never be passed here.

        Example:

            My name is Hanal.
                ->
            {"name": "Hanal"}

            My favorite color is blue.
                ->
            {"favorite_color": "blue"}

            I like football.
                ->
            {"likes": "football"}
        """

        if not isinstance(message, str):
            raise ValueError(
                "message must be a string."
            )

        message = message.strip()

        if not message:
            return {}

        facts: dict[str, str] = {}

        # -----------------------------------------------------
        # NAME
        # -----------------------------------------------------

        name = self._extract_name(
            message
        )

        if name:
            facts["name"] = name

        # -----------------------------------------------------
        # COMMON PERSONAL FACTS
        # -----------------------------------------------------

        patterns = [
            (
                "favorite_color",
                r"""
                ^\s*
                (?:actually\s*[,;:]?\s*)?
                my\s+favorite\s+color\s+is\s+
                (.+?)
                \s*$
                """,
            ),
            (
                "favorite_food",
                r"""
                ^\s*
                (?:actually\s*[,;:]?\s*)?
                my\s+favorite\s+food\s+is\s+
                (.+?)
                \s*$
                """,
            ),
            (
                "favorite_sport",
                r"""
                ^\s*
                (?:actually\s*[,;:]?\s*)?
                my\s+favorite\s+sport\s+is\s+
                (.+?)
                \s*$
                """,
            ),
            (
                "favorite_team",
                r"""
                ^\s*
                (?:actually\s*[,;:]?\s*)?
                my\s+favorite\s+
                (?:sports?\s+)?
                team\s+is\s+
                (.+?)
                \s*$
                """,
            ),
            (
                "favorite_movie",
                r"""
                ^\s*
                (?:actually\s*[,;:]?\s*)?
                my\s+favorite\s+movie\s+is\s+
                (.+?)
                \s*$
                """,
            ),
            (
                "favorite_game",
                r"""
                ^\s*
                (?:actually\s*[,;:]?\s*)?
                my\s+favorite\s+game\s+is\s+
                (.+?)
                \s*$
                """,
            ),
            (
                "favorite_song",
                r"""
                ^\s*
                (?:actually\s*[,;:]?\s*)?
                my\s+favorite\s+song\s+is\s+
                (.+?)
                \s*$
                """,
            ),
            (
                "favorite_animal",
                r"""
                ^\s*
                (?:actually\s*[,;:]?\s*)?
                my\s+favorite\s+animal\s+is\s+
                (.+?)
                \s*$
                """,
            ),
            (
                "location",
                r"""
                ^\s*
                (?:actually\s*[,;:]?\s*)?
                i\s+
                (?:live\s+in|am\s+from)
                \s+
                (.+?)
                \s*$
                """,
            ),
            (
                "job",
                r"""
                ^\s*
                (?:actually\s*[,;:]?\s*)?
                i\s+work\s+as\s+
                (.+?)
                \s*$
                """,
            ),
            (
                "job",
                r"""
                ^\s*
                (?:actually\s*[,;:]?\s*)?
                my\s+job\s+is\s+
                (.+?)
                \s*$
                """,
            ),
            (
                "likes",
                r"""
                ^\s*
                (?:actually\s*[,;:]?\s*)?
                i\s+like\s+
                (.+?)
                \s*$
                """,
            ),
            (
                "dislikes",
                r"""
                ^\s*
                (?:actually\s*[,;:]?\s*)?
                i\s+
                (?:do\s+not|don't|dont)
                \s+like\s+
                (.+?)
                \s*$
                """,
            ),
        ]

        for fact_key, pattern in patterns:

            match = re.match(
                pattern,
                message,
                flags=re.IGNORECASE | re.VERBOSE,
            )

            if not match:
                continue

            value = self._clean_value(
                match.group(1)
            )

            if value:
                facts[fact_key] = value

            break

        # -----------------------------------------------------
        # GENERIC "MY X IS Y"
        # -----------------------------------------------------

        generic = re.match(
            r"""
            ^\s*
            (?:actually\s*[,;:]?\s*)?
            my\s+
            (?P<key>
                [A-Za-z][A-Za-z0-9_ ]{1,40}?
            )
            \s+is\s+
            (?P<value>.+?)
            \s*$
            """,
            message,
            flags=re.IGNORECASE | re.VERBOSE,
        )

        if generic:

            raw_key = self._clean_value(
                generic.group("key")
            ).lower()

            value = self._clean_value(
                generic.group("value")
            )

            raw_key = re.sub(
                r"\s+",
                "_",
                raw_key,
            )

            # Name has its own authoritative extractor.
            if raw_key not in {
                "name",
                "actual_name",
            } and value:

                facts.setdefault(
                    raw_key,
                    value,
                )

        return facts

    # =========================================================
    # LEARN
    # =========================================================

    def learn_from_user_message(
        self,
        message: str,
    ) -> dict[str, str]:

        facts = self.extract_facts(
            message
        )

        for fact_key, fact_value in facts.items():

            self.set_fact(
                fact_key,
                fact_value,
            )

        return facts

    # =========================================================
    # BUILD CONTEXT
    # =========================================================

    def build_context(
        self,
        query: str = "",
        limit: int = 20,
    ) -> str:

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        facts = self.list_facts()

        if not facts:
            return ""

        # -----------------------------------------------------
        # PRIORITY
        # -----------------------------------------------------
        #
        # Name should always be included first.
        #

        name_facts = [
            fact
            for fact in facts
            if fact["fact_key"] == "name"
        ]

        other_facts = [
            fact
            for fact in facts
            if fact["fact_key"] != "name"
        ]

        ordered_facts = (
            name_facts + other_facts
        )

        ordered_facts = ordered_facts[
            :limit
        ]

        # -----------------------------------------------------
        # CONTEXT
        # -----------------------------------------------------

        lines = [
            "AUTHORITATIVE USER FACTS:",
            (
                "These facts were explicitly stated "
                "by the user."
            ),
            (
                "They are more authoritative than "
                "older assistant responses or conflicting "
                "conversation text."
            ),
        ]

        for fact in ordered_facts:

            lines.append(
                f"- {fact['fact_key']}: "
                f"{fact['fact_value']}"
            )

        return "\n".join(lines)