# Import the persistent memory system.
from app.memory.memory import Memory


# Create a component responsible for storing
# and retrieving conversation messages.
class ConversationMemory:

    # Initialize conversation memory.
    def __init__(
        self,
        memory: Memory | None = None,
    ):

        # Use the supplied memory system when provided.
        # Otherwise create the default persistent memory system.
        self.memory = memory or Memory()

    # Save a user message.
    def save_user_message(
        self,
        message: str,
    ) -> str:

        # Remove unnecessary whitespace.
        message = message.strip()

        # Reject an empty message.
        if not message:
            raise ValueError(
                "message cannot be empty."
            )

        # Store the message with a clear role prefix.
        return self.memory.add(
            f"User: {message}"
        )

    # Save an assistant response.
    def save_assistant_message(
        self,
        message: str,
    ) -> str:

        # Remove unnecessary whitespace.
        message = message.strip()

        # Reject an empty message.
        if not message:
            raise ValueError(
                "message cannot be empty."
            )

        # Store the response with a clear role prefix.
        return self.memory.add(
            f"Assistant: {message}"
        )

    # Retrieve conversation memories relevant
    # to the current query.
    def recall(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:

        # Search the persistent memory database.
        return self.memory.search(
            query=query,
            limit=limit,
        )

    # Build conversation context for the LLM.
    def build_context(
        self,
        query: str,
        limit: int = 5,
    ) -> str:

        # Retrieve relevant conversation memories.
        memories = self.recall(
            query=query,
            limit=limit,
        )

        # Return empty context when nothing was found.
        if not memories:
            return ""

        # Store formatted conversation messages.
        sections = []

        # Format each recalled memory.
        for memory in memories:

            sections.append(
                memory["content"]
            )

        # Join messages into one context block.
        return "\n".join(sections)