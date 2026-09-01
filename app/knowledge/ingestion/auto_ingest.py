from __future__ import annotations

from pathlib import Path

from app.knowledge.ingestion.service import KnowledgeIngestionService
from app.knowledge.storage import KnowledgeStorage


class KnowledgeAutoIngestion:
    """
    Automatically discover and ingest knowledge files from the data directory.

    Supported:
        .txt
        .pdf

    Existing unchanged files are skipped.
    Modified files are re-ingested.
    """

    SUPPORTED_EXTENSIONS = {
        ".txt",
        ".pdf",
    }

    def __init__(
        self,
        storage: KnowledgeStorage,
        ingestion_service: KnowledgeIngestionService,
        data_directory: Path,
    ) -> None:

        self.storage = storage
        self.ingestion_service = ingestion_service
        self.data_directory = Path(data_directory)

    def scan_and_ingest(self) -> dict[str, int]:
        """
        Scan the data directory and automatically ingest knowledge files.

        Returns:
            A dictionary containing:
                discovered
                ingested
                updated
                skipped
                failed
        """

        self.data_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        discovered = 0
        ingested = 0
        updated = 0
        skipped = 0
        failed = 0

        existing_documents = self.storage.list()

        # Normalize existing database sources.
        existing_by_source = {
            self._normalize_path(document.source): document
            for document in existing_documents
        }

        for file_path in sorted(self.data_directory.iterdir()):

            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            # Never treat the SQLite database as a knowledge document.
            if file_path.name == "knowledge.db":
                continue

            discovered += 1

            normalized_source = self._normalize_path(
                file_path
            )

            existing_document = existing_by_source.get(
                normalized_source
            )

            try:

                # -------------------------------------------------
                # NEW FILE
                # -------------------------------------------------

                if existing_document is None:

                    self.ingestion_service.ingest(
                        file_path
                    )

                    ingested += 1

                    continue

                # -------------------------------------------------
                # EXISTING FILE
                # -------------------------------------------------

                if self._is_modified(
                    file_path,
                    existing_document,
                ):

                    # Remove old chunks first.
                    self.storage.delete_chunks(
                        str(existing_document.id)
                    )

                    # Remove old document.
                    self.storage.delete(
                        str(existing_document.id)
                    )

                    # Ingest the updated document.
                    self.ingestion_service.ingest(
                        file_path
                    )

                    updated += 1

                    continue

                # -------------------------------------------------
                # UNCHANGED FILE
                # -------------------------------------------------

                skipped += 1

            except Exception as error:

                failed += 1

                print(
                    f"⚠ Failed to ingest "
                    f"{file_path.name}: {error}"
                )

        return {
            "discovered": discovered,
            "ingested": ingested,
            "updated": updated,
            "skipped": skipped,
            "failed": failed,
        }

    def _is_modified(
        self,
        file_path: Path,
        existing_document,
    ) -> bool:
        """
        Determine whether the physical file differs from the
        stored document.

        The comparison is based on the actual file content.
        """

        try:

            if file_path.suffix.lower() == ".txt":

                current_content = file_path.read_text(
                    encoding="utf-8"
                )

            elif file_path.suffix.lower() == ".pdf":

                current_content = (
                    self.ingestion_service.pdf_loader.load(
                        file_path
                    )
                )

            else:

                return False

            return (
                current_content
                != existing_document.content
            )

        except Exception:

            # If we cannot compare safely, treat the file
            # as modified so it can be re-ingested.
            return True

    @staticmethod
    def _normalize_path(
        file_path: str | Path,
    ) -> str:
        """
        Normalize paths so relative and absolute representations
        can be compared consistently.
        """

        return str(
            Path(file_path)
            .resolve()
        ).lower()