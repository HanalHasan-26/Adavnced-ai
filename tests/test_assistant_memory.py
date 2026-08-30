# Import pytest for validation tests.
import pytest

# Import the assistant memory component.
from app.memory.assistant_memory import AssistantMemory


# Create a fake memory system for testing.
class FakeMemory:

    # Initialize the fake memory system.
    def __init__(self):

        # Store memories in memory for the test.
        self.memories = []

        # Store the last search query.
        self.received_query = None

        # Store the last search limit.
        self.received_limit = None

    # Add a memory.
    def add(
        self,
        content: str,
    ) -> str:

        # Create a predictable memory ID.
        memory_id = f"memory-{len(self.memories) + 1}"

        # Store the memory.
        self.memories.append(
            {
                "id": memory_id,
                "content": content,
            }
        )

        # Return the memory ID.
        return memory_id

    # Search memories.
    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:

        # Remember the received query.
        self.received_query = query

        # Remember the received limit.
        self.received_limit = limit

        # Return matching memories.
        return self.memories[:limit]


# Test that a memory can be saved.
def test_remember():

    # Create the fake memory system.
    memory = FakeMemory()

    # Create the assistant memory component.
    assistant_memory = AssistantMemory(
        memory=memory
    )

    # Save a memory.
    result = assistant_memory.remember(
        "My name is Pirlo."
    )

    # Verify the returned memory ID.
    assert result == "memory-1"

    # Verify that the memory was stored.
    assert len(memory.memories) == 1

    # Verify the stored content.
    assert (
        memory.memories[0]["content"]
        == "My name is Pirlo."
    )


# Test that memories can be recalled.
def test_recall():

    # Create the fake memory system.
    memory = FakeMemory()

    # Add test memories.
    memory.add("My name is Pirlo.")
    memory.add("I am learning AI.")
    memory.add("I am learning forex trading.")

    # Create the assistant memory component.
    assistant_memory = AssistantMemory(
        memory=memory
    )

    # Recall memories.
    result = assistant_memory.recall(
        query="AI",
        limit=2,
    )

    # Verify the returned memories.
    assert len(result) == 2

    # Verify the query reached the memory system.
    assert memory.received_query == "AI"

    # Verify the limit reached the memory system.
    assert memory.received_limit == 2


# Test that memory context is created correctly.
def test_build_context():

    # Create the fake memory system.
    memory = FakeMemory()

    # Add test memories.
    memory.add("My name is Pirlo.")
    memory.add("I am learning AI.")

    # Create the assistant memory component.
    assistant_memory = AssistantMemory(
        memory=memory
    )

    # Build memory context.
    result = assistant_memory.build_context(
        query="AI",
        limit=5,
    )

    # Verify the first memory heading.
    assert "[Memory 1]" in result

    # Verify the second memory heading.
    assert "[Memory 2]" in result

    # Verify the memory contents.
    assert "My name is Pirlo." in result
    assert "I am learning AI." in result


# Test that empty memory produces empty context.
def test_build_context_without_memories():

    # Create an empty fake memory system.
    memory = FakeMemory()

    # Create the assistant memory component.
    assistant_memory = AssistantMemory(
        memory=memory
    )

    # Build context without memories.
    result = assistant_memory.build_context(
        query="unknown",
    )

    # The result should be empty.
    assert result == ""


# Test that the default limit is used.
def test_recall_uses_default_limit():

    # Create the fake memory system.
    memory = FakeMemory()

    # Create the assistant memory component.
    assistant_memory = AssistantMemory(
        memory=memory
    )

    # Recall memories without specifying a limit.
    assistant_memory.recall(
        query="AI"
    )

    # Verify the default limit.
    assert memory.received_limit == 5


# Test that the requested limit is respected.
def test_build_context_uses_limit():

    # Create the fake memory system.
    memory = FakeMemory()

    # Add several memories.
    memory.add("Memory one.")
    memory.add("Memory two.")
    memory.add("Memory three.")

    # Create the assistant memory component.
    assistant_memory = AssistantMemory(
        memory=memory
    )

    # Request only two memories.
    result = assistant_memory.build_context(
        query="memory",
        limit=2,
    )

    # Verify only two memories were included.
    assert "[Memory 1]" in result
    assert "[Memory 2]" in result
    assert "[Memory 3]" not in result


# Test that the memory dependency can be replaced.
def test_custom_memory_dependency():

    # Create a fake memory system.
    memory = FakeMemory()

    # Create the assistant memory component.
    assistant_memory = AssistantMemory(
        memory=memory
    )

    # Verify the supplied dependency is used.
    assert assistant_memory.memory is memory


# Test that an empty memory result produces empty context.
def test_empty_memory_result():

    # Create a fake memory system.
    memory = FakeMemory()

    # Create the assistant memory component.
    assistant_memory = AssistantMemory(
        memory=memory
    )

    # Build context.
    result = assistant_memory.build_context(
        query="nothing",
    )

    # Verify that no context was created.
    assert result == ""