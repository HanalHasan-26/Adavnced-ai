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

    Important behavior:
    - Stores facts separately from conversation memory.
    - Corrects common spelling mistakes in fact keys.
    - Prevents duplicate keys such as:
        favorite_color
        fevorite_colore
    - New explicit statements replace older values.
    - Only explicit user statements are stored.
    """

    # ---------------------------------------------------------
    # COMMON FACT-KEY CORRECTIONS
    # ---------------------------------------------------------

    KEY_ALIASES = {
        "fevorite_color": "favorite_color",
        "fevorite_colore": "favorite_color",
        "favourite_color": "favorite_color",
        "favourite_colour": "favorite_color",
        "favorite_colour": "favorite_color",

        "fevorite_food": "favorite_food",
        "favourite_food": "favorite_food",

        "fevorite_sport": "favorite_sport",
        "favourite_sport": "favorite_sport",

        "fevorite_team": "favorite_team",
        "favourite_team": "favorite_team",

        "fevorite_movie": "favorite_movie",
        "favourite_movie": "favorite_movie",

        "fevorite_game": "favorite_game",
        "favourite_game": "favorite_game",

        "fevorite_song": "favorite_song",
        "favourite_song": "favorite_song",

        "fevorite_animal": "favorite_animal",
        "favourite_animal": "favorite_animal",
    }

    # ---------------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # DATABASE CONNECTION
    # ---------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path
        )

        connection.row_factory = sqlite3.Row

        return connection

    # ---------------------------------------------------------
    # DATABASE INITIALIZATION
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # NORMALIZE FACT KEY
    # ---------------------------------------------------------

    @classmethod
    def _normalize_fact_key(
        cls,
        fact_key: str,
    ) -> str:

        if not isinstance(fact_key, str):
            raise ValueError(
                "fact_key must be a string."
            )

        fact_key = fact_key.strip().lower()

        if not fact_key:
            raise ValueError(
                "fact_key must be a non-empty string."
            )

        # Convert spaces and hyphens into underscores.
        fact_key = re.sub(
            r"[\s\-]+",
            "_",
            fact_key,
        )

        # Remove unusual characters.
        fact_key = re.sub(
            r"[^a-z0-9_]",
            "",
            fact_key,
        )

        # Remove duplicate underscores.
        fact_key = re.sub(
            r"_+",
            "_",
            fact_key,
        )

        fact_key = fact_key.strip("_")

        if not fact_key:
            raise ValueError(
                "fact_key must contain valid characters."
            )

        # Correct known misspellings / aliases.
        fact_key = cls.KEY_ALIASES.get(
            fact_key,
            fact_key,
        )

        return fact_key

    # ---------------------------------------------------------
    # CLEAN FACT VALUE
    # ---------------------------------------------------------

    @staticmethod
    def _clean_value(
        value: str,
    ) -> str:

        if not isinstance(value, str):
            raise ValueError(
                "fact_value must be a string."
            )

        value = value.strip()

        # Normalize repeated whitespace.
        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        # Remove common trailing conversational phrases.
        value = re.sub(
            r"\s*,?\s*"
            r"(?:(?:u|you)\s+got(?:\s+it)?|"
            r"(?:remember(?:\s+that)?|right|okay|ok))"
            r"\s*[.!?]*$",
            "",
            value,
            flags=re.IGNORECASE,
        )

        # Remove surrounding punctuation.
        value = value.strip(
            " \t\r\n.,!?;:"
        )

        return value

    # ---------------------------------------------------------
    # SET FACT
    # ---------------------------------------------------------

    def set_fact(
        self,
        fact_key: str,
        fact_value: str,
    ) -> str:

        fact_key = self._normalize_fact_key(
            fact_key
        )

        fact_value = self._clean_value(
            fact_value
        )

        if not fact_value:
            raise ValueError(
                "fact_value must be a non-empty string."
            )

        now = datetime.now().isoformat()

        with self._connect() as connection:

            row = connection.execute(
                """
                SELECT id
                FROM user_facts
                WHERE fact_key = ?
                """,
                (fact_key,),
            ).fetchone()

            # -------------------------------------------------
            # CREATE NEW FACT
            # -------------------------------------------------

            if row is None:

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

            # -------------------------------------------------
            # UPDATE EXISTING FACT
            # -------------------------------------------------

            else:

                fact_id = row["id"]

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

    # ---------------------------------------------------------
    # GET FACT
    # ---------------------------------------------------------

    def get_fact(
        self,
        fact_key: str,
    ) -> dict | None:

        fact_key = self._normalize_fact_key(
            fact_key
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
                (fact_key,),
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    # ---------------------------------------------------------
    # LIST FACTS
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # DELETE FACT
    # ---------------------------------------------------------

    def delete_fact(
        self,
        fact_key: str,
    ) -> bool:

        fact_key = self._normalize_fact_key(
            fact_key
        )

        with self._connect() as connection:

            cursor = connection.execute(
                """
                DELETE FROM user_facts
                WHERE fact_key = ?
                """,
                (fact_key,),
            )

            connection.commit()

            return cursor.rowcount > 0

    # ---------------------------------------------------------
    # EXTRACT FACTS
    # ---------------------------------------------------------

    def extract_facts(
        self,
        message: str,
    ) -> dict[str, str]:
        """
        Extract facts explicitly asserted by the user.

        Examples:

        My name is Hanal
        -> {"name": "Hanal"}

        My favorite color is blue
        -> {"favorite_color": "blue"}

        My fevorite colore is black
        -> {"favorite_color": "black"}
        """

        if not isinstance(message, str):
            raise ValueError(
                "message must be a string."
            )

        message = message.strip()

        if not message:
            return {}

        facts: dict[str, str] = {}

        # =====================================================
        # NAME
        # =====================================================

        name_patterns = [

            r"^\s*"
            r"(?:actually\s*[,;:]?\s*)?"
            r"my\s+"
            r"(?:actual\s+)?"
            r"name\s+is\s+"
            r"([A-Za-z][A-Za-z'\-]*"
            r"(?:\s+[A-Za-z][A-Za-z'\-]*){0,3})",

            r"^\s*"
            r"(?:actually\s*[,;:]?\s*)?"
            r"i\s*(?:am|'m)\s+"
            r"([A-Za-z][A-Za-z'\-]*"
            r"(?:\s+[A-Za-z][A-Za-z'\-]*){0,3})",

            r"^\s*"
            r"(?:actually\s*[,;:]?\s*)?"
            r"call\s+me\s+"
            r"([A-Za-z][A-Za-z'\-]*"
            r"(?:\s+[A-Za-z][A-Za-z'\-]*){0,3})",
        ]

        for pattern in name_patterns:

            match = re.match(
                pattern
                + r"(?=\s*(?:[,!?.]|$|"
                r"(?:\s+(?:u|you)\s+got\b)|"
                r"(?:\s+(?:remember(?:\s+that)?|"
                r"right|okay|ok)\b)|"
                r"(?:\s+not\b)))",
                message,
                re.IGNORECASE,
            )

            if match:

                value = self._clean_value(
                    match.group(1)
                )

                if (
                    value
                    and value.lower()
                    not in {
                        "not",
                        "unknown",
                    }
                ):
                    facts["name"] = value

                break

        # =====================================================
        # STANDARD FACTS
        # =====================================================

        patterns = {

            "favorite_color":
                r"^\s*"
                r"(?:actually\s*[,;:]?\s*)?"
                r"my\s+"
                r"favou?rite\s+"
                r"colou?r\s+is\s+(.+)$",

            "favorite_food":
                r"^\s*"
                r"(?:actually\s*[,;:]?\s*)?"
                r"my\s+"
                r"favou?rite\s+"
                r"food\s+is\s+(.+)$",

            "favorite_sport":
                r"^\s*"
                r"(?:actually\s*[,;:]?\s*)?"
                r"my\s+"
                r"favou?rite\s+"
                r"sport\s+is\s+(.+)$",

            "favorite_team":
                r"^\s*"
                r"(?:actually\s*[,;:]?\s*)?"
                r"my\s+"
                r"favou?rite\s+"
                r"(?:sports?\s+)?"
                r"team\s+is\s+(.+)$",

            "favorite_movie":
                r"^\s*"
                r"(?:actually\s*[,;:]?\s*)?"
                r"my\s+"
                r"favou?rite\s+"
                r"movie\s+is\s+(.+)$",

            "favorite_game":
                r"^\s*"
                r"(?:actually\s*[,;:]?\s*)?"
                r"my\s+"
                r"favou?rite\s+"
                r"game\s+is\s+(.+)$",

            "favorite_song":
                r"^\s*"
                r"(?:actually\s*[,;:]?\s*)?"
                r"my\s+"
                r"favou?rite\s+"
                r"song\s+is\s+(.+)$",

            "favorite_animal":
                r"^\s*"
                r"(?:actually\s*[,;:]?\s*)?"
                r"my\s+"
                r"favou?rite\s+"
                r"animal\s+is\s+(.+)$",

            "location":
                r"^\s*"
                r"(?:actually\s*[,;:]?\s*)?"
                r"i\s+"
                r"(?:live\s+in|am\s+from)\s+(.+)$",

            "job":
                r"^\s*"
                r"(?:actually\s*[,;:]?\s*)?"
                r"i\s+"
                r"work\s+as\s+(.+)$",

            "likes":
                r"^\s*"
                r"(?:actually\s*[,;:]?\s*)?"
                r"i\s+like\s+(.+)$",

            "dislikes":
                r"^\s*"
                r"(?:actually\s*[,;:]?\s*)?"
                r"i\s+"
                r"(?:do\s+not|don't|dont)\s+"
                r"like\s+(.+)$",
        }

        for fact_key, pattern in patterns.items():

            match = re.match(
                pattern,
                message,
                re.IGNORECASE,
            )

            if not match:
                continue

            value = self._clean_value(
                match.group(1)
            )

            if value:

                normalized_key = (
                    self._normalize_fact_key(
                        fact_key
                    )
                )

                facts[
                    normalized_key
                ] = value

            break

        # =====================================================
        # GENERIC "MY X IS Y"
        # =====================================================

        generic = re.match(
            r"^\s*"
            r"(?:actually\s*[,;:]?\s*)?"
            r"my\s+"
            r"([A-Za-z][A-Za-z0-9_ ]{1,40}?)"
            r"\s+is\s+"
            r"(.+)$",
            message,
            re.IGNORECASE,
        )

        if generic:

            raw_key = self._clean_value(
                generic.group(1)
            )

            value = self._clean_value(
                generic.group(2)
            )

            raw_key = re.sub(
                r"\s+",
                "_",
                raw_key.lower(),
            )

            if (
                raw_key
                not in {
                    "name",
                    "actual_name",
                }
                and value
            ):

                normalized_key = (
                    self._normalize_fact_key(
                        raw_key
                    )
                )

                facts.setdefault(
                    normalized_key,
                    value,
                )

        return facts

    # ---------------------------------------------------------
    # LEARN FROM USER MESSAGE
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # CLEAN OLD DUPLICATE / MISSPELLED FACTS
    # ---------------------------------------------------------

    def repair_fact_keys(self) -> int:
        """
        Repairs old incorrectly stored keys.

        Example:

        fevorite_colore: black
        favorite_color: blue

        The canonical key becomes:

        favorite_color: black

        The newer/incorrect duplicate is removed.
        """

        facts = self.list_facts()

        repaired = 0

        for fact in facts:

            old_key = fact["fact_key"]

            new_key = self._normalize_fact_key(
                old_key
            )

            if old_key == new_key:
                continue

            # -------------------------------------------------
            # Check whether canonical key already exists.
            # -------------------------------------------------

            existing = self.get_fact(
                new_key
            )

            if existing is None:

                with self._connect() as connection:

                    connection.execute(
                        """
                        UPDATE user_facts
                        SET fact_key = ?
                        WHERE id = ?
                        """,
                        (
                            new_key,
                            fact["id"],
                        ),
                    )

                    connection.commit()

            else:

                # The incorrectly named fact is newer,
                # so preserve its value.
                old_updated = datetime.fromisoformat(
                    fact["updated_at"]
                )

                existing_updated = datetime.fromisoformat(
                    existing["updated_at"]
                )

                if old_updated > existing_updated:

                    self.set_fact(
                        new_key,
                        fact["fact_value"],
                    )

                # Delete duplicate old key.
                with self._connect() as connection:

                    connection.execute(
                        """
                        DELETE FROM user_facts
                        WHERE id = ?
                        """,
                        (fact["id"],),
                    )

                    connection.commit()

            repaired += 1

        return repaired

    # ---------------------------------------------------------
    # BUILD CONTEXT
    # ---------------------------------------------------------

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

        # Repair old misspelled keys before building context.
        self.repair_fact_keys()

        facts = self.list_facts()

        if not facts:
            return ""

        # Prevent unlimited prompt growth.
        facts = facts[:limit]

        lines = [
            "AUTHORITATIVE USER FACTS:",
            (
                "These facts were explicitly stated by the user."
            ),
            (
                "They are more authoritative than older "
                "assistant responses or conflicting "
                "conversation text."
            ),
        ]

        for fact in facts:

            lines.append(
                f"- {fact['fact_key']}: "
                f"{fact['fact_value']}"
            )

        return "\n".join(lines)