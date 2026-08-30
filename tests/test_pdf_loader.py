# Import Path so we can work with file paths.
from pathlib import Path

# Import pytest so we can test expected errors.
import pytest

# Import PyMuPDF so we can create a small PDF for our test.
import fitz

# Import the PDF loader we want to test.
from app.knowledge.ingestion.pdf_loader import PDFLoader


# Test that PDFLoader can extract text from a PDF.
def test_load_pdf(tmp_path):

    # Create the path for our temporary PDF.
    file_path = tmp_path / "example.pdf"

    # Create a new PDF document in memory.
    pdf = fitz.open()

    # Add one page to the PDF.
    page = pdf.new_page()

    # Write text onto the PDF page.
    page.insert_text(
        (72, 72),
        "Python is useful for artificial intelligence.",
    )

    # Save the PDF to our temporary file.
    pdf.save(file_path)

    # Close the PDF we created.
    pdf.close()

    # Create our PDF loader.
    loader = PDFLoader()

    # Extract text from the PDF.
    content = loader.load(file_path)

    # Make sure the expected text was extracted.
    assert "Python is useful for artificial intelligence." in content


# Test that PDFLoader can handle multiple pages.
def test_load_multi_page_pdf(tmp_path):

    # Create the path for our temporary PDF.
    file_path = tmp_path / "multi_page.pdf"

    # Create a new PDF document.
    pdf = fitz.open()

    # Create the first page.
    first_page = pdf.new_page()

    # Add text to the first page.
    first_page.insert_text(
        (72, 72),
        "This is page one.",
    )

    # Create the second page.
    second_page = pdf.new_page()

    # Add text to the second page.
    second_page.insert_text(
        (72, 72),
        "This is page two.",
    )

    # Save the multi-page PDF.
    pdf.save(file_path)

    # Close the PDF.
    pdf.close()

    # Create our PDF loader.
    loader = PDFLoader()

    # Extract text from both pages.
    content = loader.load(file_path)

    # Verify that page one was extracted.
    assert "This is page one." in content

    # Verify that page two was extracted.
    assert "This is page two." in content


# Test that a missing PDF raises the correct error.
def test_load_missing_pdf():

    # Create our PDF loader.
    loader = PDFLoader()

    # Create a path that does not exist.
    missing_file = Path("this_pdf_does_not_exist.pdf")

    # Verify that the correct error is raised.
    with pytest.raises(FileNotFoundError):

        # Try to load the nonexistent PDF.
        loader.load(missing_file)