# Import Path so we can safely work with PDF file paths.
from pathlib import Path

# Import PyMuPDF.
# The package is installed as PyMuPDF but imported as "fitz".
import fitz


# Create a class responsible for extracting text from PDF files.
class PDFLoader:

    # Load all text from a PDF file.
    def load(self, file_path: Path) -> str:

        # Make sure the supplied path points to a real file.
        if not file_path.is_file():

            # Stop with a clear error when the file doesn't exist.
            raise FileNotFoundError(f"File not found: {file_path}")

        # Open the PDF document.
        pdf = fitz.open(file_path)

        # Create a list to hold text from every page.
        pages = []

        # Go through every page in the PDF.
        for page in pdf:

            # Extract the text from the current page.
            pages.append(page.get_text())

        # Close the PDF after extracting the text.
        pdf.close()

        # Join all page text into one string.
        return "\n".join(pages)