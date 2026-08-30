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
            raise ValueError("chunk_size must be greater than 0.")

        # Make sure the overlap is not negative.
        if overlap < 0:
            raise ValueError("overlap cannot be negative.")

        # Make sure the overlap is smaller than the chunk size.
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

        # Store the configured chunk size.
        self.chunk_size = chunk_size

        # Store the configured overlap.
        self.overlap = overlap

    # Split text into overlapping chunks.
    def chunk(self, text: str) -> list[str]:

        # Remove unnecessary whitespace from the beginning and end.
        text = text.strip()

        # Return no chunks when there is no meaningful text.
        if not text:
            return []

        # Calculate how far we move forward after each chunk.
        step = self.chunk_size - self.overlap

        # Store the generated chunks.
        chunks = []

        # Start reading from the beginning of the text.
        start = 0

        # Continue until we reach the end of the text.
        while start < len(text):

            # Calculate where the current chunk ends.
            end = start + self.chunk_size

            # Extract the current chunk.
            chunk = text[start:end]

            # Store the chunk.
            chunks.append(chunk)

            # Move forward while keeping the configured overlap.
            start += step

        # Return all generated chunks.
        return chunks