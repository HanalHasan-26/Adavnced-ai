# Import pytest for validation tests.
import pytest

# Import the knowledge-aware assistant.
from app.knowledge.assistant import KnowledgeAssistant

# Import the conversation memory component.
from app.memory.conversation import ConversationMemory

# Import the knowledge retrieval result.
from app.knowledge.retrieval.result import KnowledgeRetrievalResult


# Create a fake retrieval pipeline.
class FakePipeline:

    # Return a predictable retrieval result.
    def run(
        self,
        query: str,
        limit: int = 5,
    ) -> KnowledgeRetrievalResult:

        # Return a result containing no external knowledge.
        return KnowledgeRetrievalResult(
            query=query,
            chunks=[],
            context="",
        )


# Create a fake memory system.
class FakeMemory:

    # Initialize the fake memory.
    def __init__(self):

        # Store memories here.
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

        # Create a predictable ID.
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

    # Search stored memories.
    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:

        # Remember the query.
        self.received_query = query

        # Remember the limit.
        self.received_limit = limit

        # Return the most recent memories.
        #
        # This fake memory intentionally does not
        # implement semantic search.
        #
        # The purpose of these tests is to verify
        # that ConversationMemory is correctly
        # integrated into KnowledgeAssistant.
        return self.memories[-limit:]


# Create a fake LLM.
class FakeLLM:

    # Initialize the fake LLM.
    def __init__(self):

        # Store every prompt received.
        self.prompts = []

    # Generate a predictable answer.
    def generate(
        self,
        prompt: str,
    ) -> str:

        # Store the prompt.
        self.prompts.append(prompt)

        # Return an answer when the name appears
        # anywhere in the prompt.
        if "My name is Pirlo." in prompt:

            return "Your name is Pirlo."

        # Return a fallback response.
        return "I don't know."


# Test that the assistant saves the user message.
def test_assistant_saves_user_message():

    # Create the fake memory system.
    memory = FakeMemory()

    # Create conversation memory.
    conversation = ConversationMemory(
        memory=memory
    )

    # Create the fake LLM.
    llm = FakeLLM()

    # Create the assistant.
    assistant = KnowledgeAssistant(
        retrieval_pipeline=FakePipeline(),
        llm=llm,
        conversation_memory=conversation,
    )

    # Ask the assistant a question.
    assistant.ask(
        "My name is Pirlo."
    )

    # Verify that the user message was saved.
    assert any(
        item["content"]
        == "User: My name is Pirlo."
        for item in memory.memories
    )


# Test that the assistant saves its response.
def test_assistant_saves_response():

    # Create the fake memory system.
    memory = FakeMemory()

    # Create conversation memory.
    conversation = ConversationMemory(
        memory=memory
    )

    # Create the fake LLM.
    llm = FakeLLM()

    # Create the assistant.
    assistant = KnowledgeAssistant(
        retrieval_pipeline=FakePipeline(),
        llm=llm,
        conversation_memory=conversation,
    )

    # Ask the assistant a question.
    result = assistant.ask(
        "My name is Pirlo."
    )

    # Verify the answer.
    assert result == "Your name is Pirlo."

    # Verify that the response was saved.
    assert any(
        item["content"]
        == "Assistant: Your name is Pirlo."
        for item in memory.memories
    )


# Test that previous conversation is included
# in a later prompt.
def test_assistant_uses_previous_conversation():

    # Create the fake memory system.
    memory = FakeMemory()

    # Create conversation memory.
    conversation = ConversationMemory(
        memory=memory
    )

    # Create the fake LLM.
    llm = FakeLLM()

    # Create the assistant.
    assistant = KnowledgeAssistant(
        retrieval_pipeline=FakePipeline(),
        llm=llm,
        conversation_memory=conversation,
    )

    # First conversation turn.
    assistant.ask(
        "My name is Pirlo."
    )

    # Second conversation turn.
    assistant.ask(
        "What is my name?"
    )

    # Make sure two prompts were generated.
    assert len(llm.prompts) == 2

    # Get the second prompt.
    second_prompt = llm.prompts[1]

    # The previous user message should be present.
    assert "User: My name is Pirlo." in (
        second_prompt
    )

    # The previous assistant response should
    # also be present.
    assert "Assistant: Your name is Pirlo." in (
        second_prompt
    )


# Test the complete two-turn memory flow.
def test_two_turn_memory_flow():

    # Create the fake memory system.
    memory = FakeMemory()

    # Create conversation memory.
    conversation = ConversationMemory(
        memory=memory
    )

    # Create the fake LLM.
    llm = FakeLLM()

    # Create the assistant.
    assistant = KnowledgeAssistant(
        retrieval_pipeline=FakePipeline(),
        llm=llm,
        conversation_memory=conversation,
    )

    # First message.
    first_answer = assistant.ask(
        "My name is Pirlo."
    )

    # Verify the first answer.
    assert first_answer == (
        "Your name is Pirlo."
    )

    # Second message.
    second_answer = assistant.ask(
        "What is my name?"
    )

    # Verify that the assistant remembered
    # the information from the previous turn.
    assert second_answer == (
        "Your name is Pirlo."
    )

    # Four memory entries should exist:
    #
    # 1. User message
    # 2. Assistant response
    # 3. User message
    # 4. Assistant response
    assert len(memory.memories) == 4


# Test that default conversation memory
# is created automatically.
def test_default_conversation_memory():

    # Create the assistant with no explicit
    # conversation memory.
    assistant = KnowledgeAssistant(
        retrieval_pipeline=FakePipeline(),
        llm=FakeLLM(),
    )

    # Verify that conversation memory exists.
    assert (
        assistant.conversation_memory
        is not None
    )


# Test that a custom conversation memory
# dependency is used.
def test_custom_conversation_memory_is_used():

    # Create a fake memory system.
    memory = FakeMemory()

    # Create conversation memory.
    conversation = ConversationMemory(
        memory=memory
    )

    # Create the assistant.
    assistant = KnowledgeAssistant(
        retrieval_pipeline=FakePipeline(),
        llm=FakeLLM(),
        conversation_memory=conversation,
    )

    # Verify that the exact dependency is used.
    assert (
        assistant.conversation_memory
        is conversation
    )


# Test that LLM errors are propagated.
def test_llm_error_does_not_save_assistant_response():

    # Create a fake memory system.
    memory = FakeMemory()

    # Create conversation memory.
    conversation = ConversationMemory(
        memory=memory
    )

    # Create a failing LLM.
    class FailingLLM:

        # Always raise an error.
        def generate(
            self,
            prompt: str,
        ) -> str:

            raise RuntimeError(
                "llm failed"
            )

    # Create the assistant.
    assistant = KnowledgeAssistant(
        retrieval_pipeline=FakePipeline(),
        llm=FailingLLM(),
        conversation_memory=conversation,
    )

    # Verify that the error is propagated.
    with pytest.raises(
        RuntimeError,
        match="llm failed",
    ):

        assistant.ask(
            "My name is Pirlo."
        )

    # The assistant should not save anything
    # when LLM generation fails.
    assert memory.memories == []