# Import pytest for validation tests.
import pytest

# Import the assistant.
from app.llm.assistant import KnowledgeAssistant

# Import the knowledge retrieval result.
from app.knowledge.retrieval.result import KnowledgeRetrievalResult

# Import the knowledge chunk model.
from app.knowledge.chunking.chunk import KnowledgeChunk


# Create a fake retrieval pipeline.
class FakePipeline:

    # Initialize the fake pipeline.
    def __init__(self, result):

        # Store the result that should be returned.
        self.result = result

        # Store the received query.
        self.received_query = None

        # Store the received limit.
        self.received_limit = None

    # Return the configured retrieval result.
    def run(
        self,
        query: str,
        limit: int = 5,
    ):

        # Remember the query.
        self.received_query = query

        # Remember the limit.
        self.received_limit = limit

        # Return the fake result.
        return self.result


# Create a fake prompt builder.
class FakePromptBuilder:

    # Initialize the fake prompt builder.
    def __init__(self):

        # Store the received query.
        self.received_query = None

        # Store the received context.
        self.received_context = None

    # Build a predictable prompt.
    def build(
        self,
        query: str,
        context: str,
    ) -> str:

        # Remember the query.
        self.received_query = query

        # Remember the context.
        self.received_context = context

        # Return a predictable prompt.
        return f"PROMPT: {query}\nCONTEXT: {context}"


# Create a fake LLM client.
class FakeLLM:

    # Initialize the fake LLM.
    def __init__(self, response="AI answer."):

        # Store the response.
        self.response = response

        # Store the received prompt.
        self.received_prompt = None

    # Generate a predictable answer.
    def generate(self, prompt: str) -> str:

        # Remember the prompt.
        self.received_prompt = prompt

        # Return the configured response.
        return self.response


# Create a reusable retrieval result.
def create_result():

    # Create test chunks.
    chunks = [
        KnowledgeChunk(
            id="chunk-1",
            document_id="document-1",
            chunk_index=0,
            content="Support is a price level.",
        ),
        KnowledgeChunk(
            id="chunk-2",
            document_id="document-1",
            chunk_index=1,
            content="Resistance is a price barrier.",
        ),
    ]

    # Create a retrieval result.
    return KnowledgeRetrievalResult(
        query="support resistance",
        chunks=chunks,
        context=(
            "[Knowledge 1]\n"
            "Support is a price level.\n\n"
            "[Knowledge 2]\n"
            "Resistance is a price barrier."
        ),
    )


# Test the complete assistant flow.
def test_assistant_generates_answer():

    # Create the retrieval result.
    retrieval_result = create_result()

    # Create the fake pipeline.
    pipeline = FakePipeline(
        retrieval_result
    )

    # Create the fake prompt builder.
    prompt_builder = FakePromptBuilder()

    # Create the fake LLM.
    llm = FakeLLM(
        response="Support is a price level where buyers may appear."
    )

    # Create the assistant.
    assistant = KnowledgeAssistant(
        retrieval_pipeline=pipeline,
        prompt_builder=prompt_builder,
        llm=llm,
    )

    # Ask a question.
    result = assistant.ask(
        query="support resistance",
        limit=2,
    )

    # Verify the final answer.
    assert result == (
        "Support is a price level where buyers may appear."
    )

    # Verify retrieval received the query.
    assert pipeline.received_query == (
        "support resistance"
    )

    # Verify retrieval received the limit.
    assert pipeline.received_limit == 2

    # Verify the prompt builder received the query.
    assert prompt_builder.received_query == (
        "support resistance"
    )

    # Verify the prompt builder received the context.
    assert "Support is a price level." in (
        prompt_builder.received_context
    )

    # Verify the LLM received the generated prompt.
    assert llm.received_prompt == (
        "PROMPT: support resistance\n"
        "CONTEXT: "
        "[Knowledge 1]\n"
        "Support is a price level.\n\n"
        "[Knowledge 2]\n"
        "Resistance is a price barrier."
    )


# Test that the assistant can work without retrieved knowledge.
def test_assistant_without_knowledge():

    # Create an empty retrieval result.
    retrieval_result = KnowledgeRetrievalResult(
        query="unknown",
        chunks=[],
        context="",
    )

    # Create the fake pipeline.
    pipeline = FakePipeline(
        retrieval_result
    )

    # Create the fake prompt builder.
    prompt_builder = FakePromptBuilder()

    # Create the fake LLM.
    llm = FakeLLM(
        response="I don't know."
    )

    # Create the assistant.
    assistant = KnowledgeAssistant(
        retrieval_pipeline=pipeline,
        prompt_builder=prompt_builder,
        llm=llm,
    )

    # Ask an unknown question.
    result = assistant.ask("unknown")

    # Verify that the LLM response is returned.
    assert result == "I don't know."

    # Verify that empty context reached the builder.
    assert prompt_builder.received_context == ""

    # Verify that the LLM still received a prompt.
    assert llm.received_prompt is not None


# Test that the default retrieval limit is used.
def test_assistant_uses_default_limit():

    # Create the retrieval result.
    retrieval_result = create_result()

    # Create dependencies.
    pipeline = FakePipeline(
        retrieval_result
    )

    prompt_builder = FakePromptBuilder()
    llm = FakeLLM()

    # Create the assistant.
    assistant = KnowledgeAssistant(
        retrieval_pipeline=pipeline,
        prompt_builder=prompt_builder,
        llm=llm,
    )

    # Ask without specifying a limit.
    assistant.ask("support")

    # Verify the default limit.
    assert pipeline.received_limit == 5


# Test that an empty query is rejected.
def test_assistant_rejects_empty_query():

    # Create dependencies.
    pipeline = FakePipeline(
        create_result()
    )

    prompt_builder = FakePromptBuilder()
    llm = FakeLLM()

    # Create the assistant.
    assistant = KnowledgeAssistant(
        retrieval_pipeline=pipeline,
        prompt_builder=prompt_builder,
        llm=llm,
    )

    # Verify that empty input is rejected.
    with pytest.raises(ValueError):

        assistant.ask("")


# Test that whitespace-only queries are rejected.
def test_assistant_rejects_whitespace_query():

    # Create dependencies.
    pipeline = FakePipeline(
        create_result()
    )

    prompt_builder = FakePromptBuilder()
    llm = FakeLLM()

    # Create the assistant.
    assistant = KnowledgeAssistant(
        retrieval_pipeline=pipeline,
        prompt_builder=prompt_builder,
        llm=llm,
    )

    # Verify whitespace-only input is rejected.
    with pytest.raises(ValueError):

        assistant.ask("   ")


# Test that invalid limits are rejected.
def test_assistant_rejects_invalid_limit():

    # Create dependencies.
    pipeline = FakePipeline(
        create_result()
    )

    prompt_builder = FakePromptBuilder()
    llm = FakeLLM()

    # Create the assistant.
    assistant = KnowledgeAssistant(
        retrieval_pipeline=pipeline,
        prompt_builder=prompt_builder,
        llm=llm,
    )

    # Verify zero is rejected.
    with pytest.raises(ValueError):

        assistant.ask(
            "support",
            limit=0,
        )

    # Verify negative values are rejected.
    with pytest.raises(ValueError):

        assistant.ask(
            "support",
            limit=-1,
        )


# Test that retrieval errors are propagated.
def test_assistant_propagates_retrieval_errors():

    # Create a pipeline that raises an error.
    class FailingPipeline:

        def run(
            self,
            query: str,
            limit: int = 5,
        ):

            raise RuntimeError(
                "retrieval failed"
            )

    # Create the assistant.
    assistant = KnowledgeAssistant(
        retrieval_pipeline=FailingPipeline(),
        prompt_builder=FakePromptBuilder(),
        llm=FakeLLM(),
    )

    # Verify the error is not silently swallowed.
    with pytest.raises(RuntimeError, match="retrieval failed"):

        assistant.ask("support")


# Test that LLM errors are propagated.
def test_assistant_propagates_llm_errors():

    # Create a failing LLM.
    class FailingLLM:

        def generate(self, prompt: str) -> str:

            raise RuntimeError(
                "llm failed"
            )

    # Create the assistant.
    assistant = KnowledgeAssistant(
        retrieval_pipeline=FakePipeline(
            create_result()
        ),
        prompt_builder=FakePromptBuilder(),
        llm=FailingLLM(),
    )

    # Verify the error is not silently swallowed.
    with pytest.raises(RuntimeError, match="llm failed"):

        assistant.ask("support")