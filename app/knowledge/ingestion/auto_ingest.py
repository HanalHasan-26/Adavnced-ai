from __future__ import annotations

from pathlib import Path

from app.knowledge.ingestion.service import KnowledgeIngestionService
from app.knowledge.storage import KnowledgeStorage


class KnowledgeAutoIngestion:
    """
    Automatically synchronize TXT/PDF files in the data directory
    with the persistent knowledge database.

    Supported:
        .txt
        .pdf

    Behavior:
        New file       -> ingest
        Unchanged file -> skip
        Modified file  -> update
        Deleted file   -> remove from knowledge database
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
        Synchronize files in the data directory with the
        knowledge database.

        Returns:
            discovered
            ingested
            updated
            deleted
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
        deleted = 0
        skipped = 0
        failed = 0

        existing_documents = self.storage.list()

        existing_by_source = {
            self._normalize_path(document.source): document
            for document in existing_documents
        }

        current_files: set[str] = set()

        # =========================================================
        # SCAN CURRENT FILES
        # =========================================================

        for file_path in sorted(
            self.data_directory.iterdir()
        ):

            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            # Never treat the database itself as knowledge.
            if file_path.name == "knowledge.db":
                continue

            discovered += 1

            normalized_source = self._normalize_path(
                file_path
            )

            current_files.add(
                normalized_source
            )

            existing_document = existing_by_source.get(
                normalized_source
            )

            try:

                # =================================================
                # NEW FILE
                # =================================================

                if existing_document is None:

                    self.ingestion_service.ingest(
                        file_path
                    )

                    ingested += 1

                    continue

                # =================================================
                # MODIFIED FILE
                # =================================================

                if self._is_modified(
                    file_path,
                    existing_document,
                ):

                    # Delete old chunks.
                    self.storage.delete_chunks(
                        str(existing_document.id)
                    )

                    # Delete old document.
                    self.storage.delete(
                        str(existing_document.id)
                    )

                    # Ingest the new version.
                    self.ingestion_service.ingest(
                        file_path
                    )

                    updated += 1

                    continue

                # =================================================
                # UNCHANGED FILE
                # =================================================

                skipped += 1

            except Exception as error:

                failed += 1

                print(
                    f"⚠ Failed to synchronize "
                    f"{file_path.name}: {error}"
                )

        # =========================================================
        # DETECT DELETED FILES
        # =========================================================

        for source, document in existing_by_source.items():

            # If the source is no longer present on disk,
            # remove its stored knowledge.
            if source not in current_files:

                try:

                    self.storage.delete_chunks(
                        str(document.id)
                    )

                    self.storage.delete(
                        str(document.id)
                    )

                    deleted += 1

                except Exception as error:

                    failed += 1

                    print(
                        f"⚠ Failed to delete knowledge "
                        f"for {document.source}: {error}"
                    )

        return {
            "discovered": discovered,
            "ingested": ingested,
            "updated": updated,
            "deleted": deleted,
            "skipped": skipped,
            "failed": failed,
        }

    def _is_modified(
        self,
        file_path: Path,
        existing_document,
    ) -> bool:
        """
        Determine whether the physical file content differs
        from the content stored in the knowledge database.
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

            # If the file cannot be safely compared,
            # treat it as modified.
            return True

    @staticmethod
    def _normalize_path(
        file_path: str | Path,
    ) -> str:
        """
        Normalize paths so relative and absolute paths
        compare consistently.
        """

        return str(
            Path(file_path).resolve()
        ).lower()