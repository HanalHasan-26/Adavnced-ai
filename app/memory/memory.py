from __future__ import annotations

# Import SQLite, which is included with Python.
import sqlite3

# Import Path for safe filesystem handling.
from pathlib import Path

# Import UUID generation for unique memory IDs.
from uuid import uuid4

# Import the project's data directory.
from config.settings import DATA_DIR


# Define the default location of the memory database.
DATABASE_PATH = DATA_DIR / "memory.db"


# Create the persistent memory system.
class Memory:

    # Initialize the memory system.
    def __init__(
        self,
        database_path: Path = DATABASE_PATH,
    ):

        # Store the database path.
        self.database_path = database_path

        # Make sure the parent directory exists.
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Create the database structure.
        self._initialize_database()

    # Initialize the SQLite database.
    def _initialize_database(self) -> None:

        # Open a database connection.
        with sqlite3.connect(
            self.database_path
        ) as connection:

            # Create the memories table.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

            # Save the database structure.
            connection.commit()

    # Add a new memory.
    def add(
        self,
        content: str,
    ) -> str:

        # Make sure the content is a string.
        if not isinstance(content, str):
            raise ValueError(
                "content must be a string."
            )

        # Remove unnecessary whitespace.
        content = content.strip()

        # Reject empty memories.
        if not content:
            raise ValueError(
                "content cannot be empty."
            )

        # Generate a unique memory ID.
        memory_id = str(uuid4())

        # Generate the creation timestamp.
        from datetime import datetime

        created_at = datetime.now().isoformat()

        # Open the database connection.
        with sqlite3.connect(
            self.database_path
        ) as connection:

            # Insert the memory.
            connection.execute(
                """
                INSERT INTO memories (
                    id,
                    content,
                    created_at
                )
                VALUES (?, ?, ?)
                """,
                (
                    memory_id,
                    content,
                    created_at,
                ),
            )

            # Save the memory permanently.
            connection.commit()

        # Return the generated ID.
        return memory_id

    # Retrieve one memory by ID.
    def get(
        self,
        memory_id: str,
    ) -> dict | None:

        # Reject empty IDs.
        if not isinstance(memory_id, str):
            raise ValueError(
                "memory_id must be a string."
            )

        memory_id = memory_id.strip()

        if not memory_id:
            raise ValueError(
                "memory_id cannot be empty."
            )

        # Open the database connection.
        with sqlite3.connect(
            self.database_path
        ) as connection:

            # Return rows as dictionary-like objects.
            connection.row_factory = sqlite3.Row

            # Find the requested memory.
            row = connection.execute(
                """
                SELECT id, content, created_at
                FROM memories
                WHERE id = ?
                """,
                (memory_id,),
            ).fetchone()

        # Return nothing when the memory doesn't exist.
        if row is None:
            return None

        # Convert the row into a normal dictionary.
        return {
            "id": row["id"],
            "content": row["content"],
            "created_at": row["created_at"],
        }

    # Search memories using text.
    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict]:

        # Make sure the query is a string.
        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        # Remove unnecessary whitespace.
        query = query.strip()

        # Empty searches return no results.
        if not query:
            return []

        # Make sure the limit is valid.
        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        # Create the SQLite search pattern.
        search_pattern = f"%{query}%"

        # Open the database connection.
        with sqlite3.connect(
            self.database_path
        ) as connection:

            # Return rows using column names.
            connection.row_factory = sqlite3.Row

            # Search memory content.
            rows = connection.execute(
                """
                SELECT id, content, created_at
                FROM memories
                WHERE content LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (
                    search_pattern,
                    limit,
                ),
            ).fetchall()

        # Convert database rows into dictionaries.
        return [
            {
                "id": row["id"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    # Retrieve all stored memories.
    def list(
        self,
    ) -> list[dict]:

        # Open the database connection.
        with sqlite3.connect(
            self.database_path
        ) as connection:

            # Return rows using column names.
            connection.row_factory = sqlite3.Row

            # Retrieve all memories.
            rows = connection.execute(
                """
                SELECT id, content, created_at
                FROM memories
                ORDER BY created_at DESC
                """
            ).fetchall()

        # Convert rows into dictionaries.
        return [
            {
                "id": row["id"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    # Delete one memory.
    def delete(
        self,
        memory_id: str,
    ) -> bool:

        # Make sure the ID is a string.
        if not isinstance(memory_id, str):
            raise ValueError(
                "memory_id must be a string."
            )

        # Remove unnecessary whitespace.
        memory_id = memory_id.strip()

        # Reject an empty ID.
        if not memory_id:
            raise ValueError(
                "memory_id cannot be empty."
            )

        # Open the database connection.
        with sqlite3.connect(
            self.database_path
        ) as connection:

            # Delete the requested memory.
            cursor = connection.execute(
                """
                DELETE FROM memories
                WHERE id = ?
                """,
                (memory_id,),
            )

            # Save the deletion.
            connection.commit()

            # Return whether a memory was deleted.
            return cursor.rowcount > 0