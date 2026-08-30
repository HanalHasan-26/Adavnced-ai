# Import pytest for validation tests.
import pytest

# Import the conversation memory component.
from app.memory.conversation import ConversationMemory


# Create a fake memory system for testing.
class FakeMemory:

    # Initialize the fake memory system.
    def __init__(self):

        # Store memories in a simple list.
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

        # Generate a predictable ID.
        memory_id = (
            f"memory-{len(self.memories) + 1}"
        )

        # Store the memory.
        self.memories.append(
            {
                "id": memory_id,
                "content": content,
            }
        )

        # Return the generated ID.
        return memory_id

    # Search memories.
    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:

        # Remember the query.
        self.received_query = query

        # Remember the limit.
        self.received_limit = limit

        # Return the requested number of memories.
        return self.memories[:limit]


# Test saving a user message.
def test_save_user_message():

    # Create the fake memory system.
    memory = FakeMemory()

    # Create conversation memory.
    conversation = ConversationMemory(
        memory=memory
    )

    # Save a user message.
    result = conversation.save_user_message(
        "My name is Pirlo."
    )

    # Verify the generated ID.
    assert result == "memory-1"

    # Verify the stored message.
    assert memory.memories[0]["content"] == (
        "User: My name is Pirlo."
    )


# Test saving an assistant message.
def test_save_assistant_message():

    # Create the fake memory system.
    memory = FakeMemory()

    # Create conversation memory.
    conversation = ConversationMemory(
        memory=memory
    )

    # Save an assistant response.
    result = conversation.save_assistant_message(
        "Nice to meet you, Pirlo."
    )

    # Verify the generated ID.
    assert result == "memory-1"

    # Verify the stored message.
    assert memory.memories[0]["content"] == (
        "Assistant: Nice to meet you, Pirlo."
    )


# Test recalling conversation messages.
def test_recall():

    # Create the fake memory system.
    memory = FakeMemory()

    # Create conversation memory.
    conversation = ConversationMemory(
        memory=memory
    )

    # Save conversation messages.
    conversation.save_user_message(
        "My name is Pirlo."
    )

    conversation.save_assistant_message(
        "Nice to meet you."
    )

    # Recall memories.
    result = conversation.recall(
        query="Pirlo",
        limit=2,
    )

    # Verify the results.
    assert len(result) == 2

    # Verify the query was passed correctly.
    assert memory.received_query == "Pirlo"

    # Verify the limit was passed correctly.
    assert memory.received_limit == 2


# Test building conversation context.
def test_build_context():

    # Create the fake memory system.
    memory = FakeMemory()

    # Create conversation memory.
    conversation = ConversationMemory(
        memory=memory
    )

    # Save messages.
    conversation.save_user_message(
        "My name is Pirlo."
    )

    conversation.save_assistant_message(
        "Nice to meet you."
    )

    # Build context.
    result = conversation.build_context(
        query="Pirlo"
    )

    # Verify user message is included.
    assert "User: My name is Pirlo." in result

    # Verify assistant response is included.
    assert "Assistant: Nice to meet you." in result


# Test that empty conversation context
# produces an empty string.
def test_empty_context():

    # Create an empty fake memory system.
    memory = FakeMemory()

    # Create conversation memory.
    conversation = ConversationMemory(
        memory=memory
    )

    # Build context without memories.
    result = conversation.build_context(
        query="unknown"
    )

    # Verify empty context.
    assert result == ""


# Test that an empty user message is rejected.
def test_empty_user_message():

    # Create conversation memory.
    conversation = ConversationMemory(
        memory=FakeMemory()
    )

    # Verify empty input raises an error.
    with pytest.raises(ValueError):

        conversation.save_user_message("")


# Test that whitespace-only user messages
# are rejected.
def test_whitespace_user_message():

    # Create conversation memory.
    conversation = ConversationMemory(
        memory=FakeMemory()
    )

    # Verify whitespace-only input is rejected.
    with pytest.raises(ValueError):

        conversation.save_user_message(
            "   "
        )


# Test that an empty assistant message is rejected.
def test_empty_assistant_message():

    # Create conversation memory.
    conversation = ConversationMemory(
        memory=FakeMemory()
    )

    # Verify empty input raises an error.
    with pytest.raises(ValueError):

        conversation.save_assistant_message("")


# Test that whitespace-only assistant messages
# are rejected.
def test_whitespace_assistant_message():

    # Create conversation memory.
    conversation = ConversationMemory(
        memory=FakeMemory()
    )

    # Verify whitespace-only input is rejected.
    with pytest.raises(ValueError):

        conversation.save_assistant_message(
            "   "
        )


# Test that the default recall limit is used.
def test_default_recall_limit():

    # Create the fake memory system.
    memory = FakeMemory()

    # Create conversation memory.
    conversation = ConversationMemory(
        memory=memory
    )

    # Recall without specifying a limit.
    conversation.recall(
        query="test"
    )

    # Verify the default limit.
    assert memory.received_limit == 5


# Test that a custom memory dependency
# is actually used.
def test_custom_memory_dependency():

    # Create a fake memory system.
    memory = FakeMemory()

    # Create conversation memory.
    conversation = ConversationMemory(
        memory=memory
    )

    # Verify the dependency is stored.
    assert conversation.memory is memory


# Test that context respects the requested limit.
def test_context_limit():

    # Create the fake memory system.
    memory = FakeMemory()

    # Create conversation memory.
    conversation = ConversationMemory(
        memory=memory
    )

    # Add several messages.
    conversation.save_user_message(
        "Message one."
    )

    conversation.save_user_message(
        "Message two."
    )

    conversation.save_user_message(
        "Message three."
    )

    # Request only two memories.
    result = conversation.build_context(
        query="message",
        limit=2,
    )

    # Verify the first two are present.
    assert "Message one." in result
    assert "Message two." in result

    # Verify the third isn't present.
    assert "Message three." not in result