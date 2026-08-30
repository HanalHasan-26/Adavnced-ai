# Import SQLite, which is built into Python.
import sqlite3

# Import Path so we can safely work with file paths.
from pathlib import Path

# Import our knowledge document model.
from app.knowledge.document import KnowledgeDocument

# Import the data directory from our project configuration.
from config.settings import DATA_DIR


# Define where our local knowledge database will be stored.
DATABASE_PATH = DATA_DIR / "knowledge.db"


# Create a class responsible for storing knowledge.
class KnowledgeStorage:

    # Initialize the storage system.
    def __init__(self, database_path: Path = DATABASE_PATH):

        # Store the database path inside this object.
        self.database_path = database_path

        # Make sure the database's parent directory exists.
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        # Create the database table if it does not exist.
        self._initialize_database()

    # Create the database structure.
    def _initialize_database(self) -> None:

        # Open a connection to the SQLite database.
        with sqlite3.connect(self.database_path) as connection:

            # Create a cursor for executing SQL commands.
            cursor = connection.cursor()

            # Create the knowledge table if it does not already exist.
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

        # Open a connection to the SQLite database.
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

            # Save the document permanently.
            connection.commit()