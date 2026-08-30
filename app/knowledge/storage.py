# Import SQLite, which comes built into Python.
import sqlite3

# Import Path so we can work with file paths.
from pathlib import Path

# Import datetime so we can convert timestamps.
from datetime import datetime

# Import UUID so we can convert document IDs.
from uuid import UUID

# Import our knowledge document model.
from app.knowledge.document import KnowledgeDocument

# Import the project's data directory.
from config.settings import DATA_DIR


# Define the location of our knowledge database.
DATABASE_PATH = DATA_DIR / "knowledge.db"


# Create the class responsible for storing knowledge.
class KnowledgeStorage:

    # Initialize the storage system.
    def __init__(self, database_path: Path = DATABASE_PATH):

        # Store the database path.
        self.database_path = database_path

        # Create the parent directory if it doesn't exist.
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        # Create the database table if it doesn't exist.
        self._initialize_database()

    # Create the database structure.
    def _initialize_database(self) -> None:

        # Open a connection to the SQLite database.
        with sqlite3.connect(self.database_path) as connection:

            # Create a cursor for executing SQL commands.
            cursor = connection.cursor()

            # Create the knowledge table if it doesn't exist.
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

            # Save the database changes.
            connection.commit()

    # Add a knowledge document to the database.
    def add(self, document: KnowledgeDocument) -> None:

        # Open a connection to the database.
        with sqlite3.connect(self.database_path) as connection:

            # Insert the document into the knowledge table.
            connection.execute(
                """
                INSERT INTO knowledge (
                    id,
                    content,
                    source,
                    source_type,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(document.id),
                    document.content,
                    document.source,
                    document.source_type,
                    document.created_at.isoformat(),
                ),
            )

            # Save the changes permanently.
            connection.commit()

    # Retrieve one document using its ID.
    def get(self, document_id: str) -> KnowledgeDocument | None:

        # Open a connection to the database.
        with sqlite3.connect(self.database_path) as connection:

            # Make database rows accessible using column names.
            connection.row_factory = sqlite3.Row

            # Find the document with the requested ID.
            row = connection.execute(
                """
                SELECT id, content, source, source_type, created_at
                FROM knowledge
                WHERE id = ?
                """,
                (document_id,),
            ).fetchone()

        # Return nothing if the document doesn't exist.
        if row is None:
            return None

        # Convert the database row back into a KnowledgeDocument.
        return KnowledgeDocument(
            id=UUID(row["id"]),
            content=row["content"],
            source=row["source"],
            source_type=row["source_type"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

        # Search stored knowledge using a text query.
    def search(self, query: str) -> list[KnowledgeDocument]:

        # Remove unnecessary spaces from the user's search query.
        query = query.strip()

        # Return an empty list when the query is empty.
        if not query:
            return []

        # Add wildcard characters so SQLite can find the query inside text.
        search_pattern = f"%{query}%"

        # Open a connection to our SQLite database.
        with sqlite3.connect(self.database_path) as connection:

            # Make database rows accessible using column names.
            connection.row_factory = sqlite3.Row

            # Search the content, source, and source type.
            rows = connection.execute(
                """
                SELECT id, content, source, source_type, created_at
                FROM knowledge
                WHERE content LIKE ?
                   OR source LIKE ?
                   OR source_type LIKE ?
                ORDER BY created_at DESC
                """,
                (search_pattern, search_pattern, search_pattern),
            ).fetchall()

        # Convert every database row into a KnowledgeDocument.
        return [
            KnowledgeDocument(
                id=UUID(row["id"]),
                content=row["content"],
                source=row["source"],
                source_type=row["source_type"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

        # Delete one knowledge document using its unique ID.
    def delete(self, document_id: str) -> bool:

        # Open a connection to our SQLite database.
        with sqlite3.connect(self.database_path) as connection:

            # Delete the document with the requested ID.
            cursor = connection.execute(
                """
                DELETE FROM knowledge
                WHERE id = ?
                """,
                (document_id,),
            )

            # Save the deletion permanently.
            connection.commit()

            # Check whether a document was actually deleted.
            return cursor.rowcount > 0

        # Update an existing knowledge document.
    def update(self, document: KnowledgeDocument) -> bool:

        # Open a connection to the SQLite database.
        with sqlite3.connect(self.database_path) as connection:

            # Update the document that has the matching ID.
            cursor = connection.execute(
                """
                UPDATE knowledge
                SET content = ?,
                    source = ?,
                    source_type = ?,
                    created_at = ?
                WHERE id = ?
                """,
                (
                    document.content,
                    document.source,
                    document.source_type,
                    document.created_at.isoformat(),
                    str(document.id),
                ),
            )

            # Save the changes permanently.
            connection.commit()

            # Return True if a document was actually updated.
            return cursor.rowcount > 0

        # Retrieve all knowledge documents from the database.
    def list(self) -> list[KnowledgeDocument]:

        # Open a connection to the SQLite database.
        with sqlite3.connect(self.database_path) as connection:

            # Make database rows accessible using column names.
            connection.row_factory = sqlite3.Row

            # Retrieve every knowledge document.
            rows = connection.execute(
                """
                SELECT id, content, source, source_type, created_at
                FROM knowledge
                ORDER BY created_at DESC
                """
            ).fetchall()

        # Convert every database row into a KnowledgeDocument object.
        return [
            KnowledgeDocument(
                id=UUID(row["id"]),
                content=row["content"],
                source=row["source"],
                source_type=row["source_type"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]