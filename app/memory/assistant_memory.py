# Import the persistent memory system.
from app.memory.memory import Memory


# Create a component responsible for managing
# memories used by the AI assistant.
class AssistantMemory:

    # Initialize the assistant memory component.
    def __init__(
        self,
        memory: Memory | None = None,
    ):

        # Use the supplied memory system when provided.
        # Otherwise create the default persistent memory system.
        self.memory = memory or Memory()

    # Save a piece of information as a memory.
    def remember(
        self,
        content: str,
    ) -> str:

        # Store the memory permanently.
        return self.memory.add(content)

    # Search for memories relevant to a query.
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

    # Convert recalled memories into prompt context.
    def build_context(
        self,
        query: str,
        limit: int = 5,
    ) -> str:

        # Retrieve relevant memories.
        memories = self.recall(
            query=query,
            limit=limit,
        )

        # Return an empty context when nothing was found.
        if not memories:
            return ""

        # Build readable memory context.
        sections = []

        # Add each memory to the context.
        for index, memory in enumerate(
            memories,
            start=1,
        ):

            sections.append(
                f"[Memory {index}]\n"
                f"{memory['content']}"
            )

        # Separate individual memories clearly.
        return "\n\n".join(sections)