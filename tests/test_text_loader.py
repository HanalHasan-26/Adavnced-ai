# Import Path so we can work with the temporary test file.
from pathlib import Path

# Import pytest so we can test expected errors.
import pytest

# Import the TextLoader we want to test.
from app.knowledge.ingestion.text_loader import TextLoader


# Test that TextLoader can read a normal text file.
def test_load_text_file(tmp_path):

    # Create a temporary text file path.
    file_path = tmp_path / "example.txt"

    # Write some text into the temporary file.
    file_path.write_text(
        "Python is useful for building AI systems.",
        encoding="utf-8",
    )

    # Create our text loader.
    loader = TextLoader()

    # Load the text from the file.
    content = loader.load(file_path)

    # Make sure the returned content is correct.
    assert content == "Python is useful for building AI systems."


# Test that TextLoader correctly handles UTF-8 text.
def test_load_utf8_text(tmp_path):

    # Create a temporary text file path.
    file_path = tmp_path / "unicode.txt"

    # Create text containing non-English characters.
    expected_text = "Hello മലയാളം — مرحبا — AI 🤖"

    # Write the UTF-8 text to the temporary file.
    file_path.write_text(
        expected_text,
        encoding="utf-8",
    )

    # Create our text loader.
    loader = TextLoader()

    # Load the text from the file.
    content = loader.load(file_path)

    # Make sure the Unicode text was preserved correctly.
    assert content == expected_text


# Test that a missing file produces the correct error.
def test_load_missing_file():

    # Create our text loader.
    loader = TextLoader()

    # Create a path that does not exist.
    missing_file = Path("this_file_does_not_exist.txt")

    # Verify that loading the missing file raises FileNotFoundError.
    with pytest.raises(FileNotFoundError):

        # Try to load the nonexistent file.
        loader.load(missing_file)