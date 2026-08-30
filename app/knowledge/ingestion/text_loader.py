# Import Path so we can safely work with file paths.
from pathlib import Path


# Create a class responsible for loading text files.
class TextLoader:

    # Load text from a file.
    def load(self, file_path: Path) -> str:

        # Make sure the supplied path points to a file.
        if not file_path.is_file():

            # Stop with a clear error when the file doesn't exist.
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read the complete file using UTF-8 encoding.
        return file_path.read_text(encoding="utf-8")