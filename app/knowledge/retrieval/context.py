# Import the knowledge chunk model.
from app.knowledge.chunking.chunk import KnowledgeChunk


# Create a component responsible for assembling
# retrieved chunks into AI-ready context.
class KnowledgeContext:

    # Initialize the context assembler.
    def __init__(
        self,
        max_characters: int = 12000,
    ):

        # Make sure the maximum size is valid.
        if max_characters <= 0:
            raise ValueError(
                "max_characters must be greater than 0."
            )

        # Store the maximum allowed context size.
        self.max_characters = max_characters

    # Assemble retrieved chunks into one context string.
    def assemble(
        self,
        chunks: list[KnowledgeChunk],
    ) -> str:

        # Return an empty context when there are no chunks.
        if not chunks:
            return ""

        # Store formatted chunk sections.
        sections: list[str] = []

        # Track the current context size.
        current_length = 0

        # Format every retrieved chunk.
        for index, chunk in enumerate(chunks, start=1):

            # Create the formatted section.
            section = (
                f"[Knowledge {index}]\n"
                f"{chunk.content}"
            )

            # Calculate how much space this section needs.
            separator_length = 2 if sections else 0

            # Calculate the total size if this section
            # were added.
            required_length = (
                current_length
                + separator_length
                + len(section)
            )

            # Stop before exceeding the configured limit.
            if required_length > self.max_characters:

                # If no chunk fits at all, return an empty
                # context rather than returning oversized text.
                if not sections:
                    return ""

                break

            # Add the section.
            sections.append(section)

            # Update the current context size.
            current_length = required_length

        # Join all accepted knowledge sections together.
        return "\n\n".join(sections)