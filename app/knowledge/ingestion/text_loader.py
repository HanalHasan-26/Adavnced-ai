from __future__ import annotations

from pathlib import Path


class TextLoader:
    """Load plain-text knowledge files."""

    def load(self, file_path: Path) -> str:
        """Read and return the complete contents of a text file."""

        file_path = Path(file_path)

        if not file_path.is_file():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        return file_path.read_text(
            encoding="utf-8"
        )