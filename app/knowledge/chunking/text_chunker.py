# Create a component responsible for splitting text into smaller chunks.
class TextChunker:

    # Initialize the chunker with configurable chunk size and overlap.
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
    ):

        # Make sure the chunk size is a positive number.
        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0."
            )

        # Make sure the overlap is not negative.
        if overlap < 0:
            raise ValueError(
                "overlap cannot be negative."
            )

        # Make sure the overlap is smaller than the chunk size.
        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size."
            )

        # Store the configured chunk size.
        self.chunk_size = chunk_size

        # Store the configured overlap.
        self.overlap = overlap

    # Split text into overlapping chunks.
    def chunk(self, text: str) -> list[str]:

        # Make sure the input is a string.
        if not isinstance(text, str):
            raise TypeError(
                "text must be a string."
            )

        # Remove unnecessary whitespace from the beginning
        # and end of the document.
        text = text.strip()

        # Return no chunks when there is no meaningful text.
        if not text:
            return []

        # Calculate how far we move forward after each chunk.
        step = self.chunk_size - self.overlap

        # Store the generated chunks.
        chunks: list[str] = []

        # Start reading from the beginning of the text.
        start = 0

        # Continue until the complete document is processed.
        while start < len(text):

            # Calculate the end position of this chunk.
            end = min(
                start + self.chunk_size,
                len(text),
            )

            # Extract the current chunk.
            current_chunk = text[start:end]

            # Make sure the extracted chunk contains content.
            if current_chunk:

                # Store the chunk.
                chunks.append(current_chunk)

            # Stop when the end of the document has been reached.
            if end >= len(text):
                break

            # Move forward while preserving the configured overlap.
            start += step

        # Return all generated chunks.
        return chunks