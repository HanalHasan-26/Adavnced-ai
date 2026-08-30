# Import pytest for validation tests.
import pytest

# Import the knowledge chunk model.
from app.knowledge.chunking.chunk import KnowledgeChunk

# Import the assistant.
from app.knowledge.assistant import KnowledgeAssistant

# Import the retrieval pipeline.
from app.knowledge.retrieval.pipeline import KnowledgeRetrievalPipeline


# Create a fake retriever for testing.
class FakeRetriever:

    # Initialize the fake retriever.
    def __init__(self, chunks):

        # Store the chunks.
        self.chunks = chunks

        # Store the received query.
        self.received_query = None

        # Store the received limit.
        self.received_limit = None

    # Return the predefined chunks.
    def retrieve(
        self,
        query: str,
        limit: int = 5,
    ):

        # Remember the arguments.
        self.received_query = query
        self.received_limit = limit

        # Return the test chunks.
        return self.chunks


# Test that the assistant prepares a knowledge-aware prompt.
def test_assistant_prepare():

    # Create a knowledge chunk.
    chunk = KnowledgeChunk(
        id="chunk-1",
        document_id="document-1",
        chunk_index=0,
        content="Support is a price level.",
    )

    # Create the fake retriever.
    retriever = FakeRetriever([chunk])

    # Create the retrieval pipeline.
    pipeline = KnowledgeRetrievalPipeline(
        retriever
    )

    # Create the assistant.
    assistant = KnowledgeAssistant(
        pipeline
    )

    # Prepare a prompt.
    prompt = assistant.prepare(
        query="What is support?",
        limit=3,
    )

    # Verify the query reached the retriever.
    assert retriever.received_query == (
        "What is support?"
    )

    # Verify the limit reached the retriever.
    assert retriever.received_limit == 3

    # Verify the knowledge is present.
    assert "Support is a price level." in prompt

    # Verify the question is present.
    assert "What is support?" in prompt


# Test that retrieve() returns the complete result.
def test_assistant_retrieve():

    # Create a knowledge chunk.
    chunk = KnowledgeChunk(
        id="chunk-1",
        document_id="document-1",
        chunk_index=0,
        content="Resistance can act as a barrier.",
    )

    # Create the fake retriever.
    retriever = FakeRetriever([chunk])

    # Create the pipeline.
    pipeline = KnowledgeRetrievalPipeline(
        retriever
    )

    # Create the assistant.
    assistant = KnowledgeAssistant(
        pipeline
    )

    # Retrieve knowledge.
    result = assistant.retrieve(
        query="What is resistance?"
    )

    # Verify the retrieved chunk.
    assert len(result.chunks) == 1

    # Verify the chunk content.
    assert result.chunks[0].content == (
        "Resistance can act as a barrier."
    )

    # Verify the query.
    assert result.query == (
        "What is resistance?"
    )


# Test that empty queries are rejected by prepare().
def test_assistant_prepare_empty_query():

    # Create an empty fake retriever.
    retriever = FakeRetriever([])

    # Create the pipeline.
    pipeline = KnowledgeRetrievalPipeline(
        retriever
    )

    # Create the assistant.
    assistant = KnowledgeAssistant(
        pipeline
    )

    # Verify empty input is rejected.
    with pytest.raises(ValueError):

        assistant.prepare("")


# Test that whitespace-only queries are rejected.
def test_assistant_prepare_whitespace_query():

    # Create an empty fake retriever.
    retriever = FakeRetriever([])

    # Create the pipeline.
    pipeline = KnowledgeRetrievalPipeline(
        retriever
    )

    # Create the assistant.
    assistant = KnowledgeAssistant(
        pipeline
    )

    # Verify whitespace input is rejected.
    with pytest.raises(ValueError):

        assistant.prepare("   ")


# Test that empty queries are rejected by retrieve().
def test_assistant_retrieve_empty_query():

    # Create an empty fake retriever.
    retriever = FakeRetriever([])

    # Create the pipeline.
    pipeline = KnowledgeRetrievalPipeline(
        retriever
    )

    # Create the assistant.
    assistant = KnowledgeAssistant(
        pipeline
    )

    # Verify empty input is rejected.
    with pytest.raises(ValueError):

        assistant.retrieve("")


# Test that the assistant works when no knowledge exists.
def test_assistant_prepare_without_knowledge():

    # Create a retriever with no results.
    retriever = FakeRetriever([])

    # Create the pipeline.
    pipeline = KnowledgeRetrievalPipeline(
        retriever
    )

    # Create the assistant.
    assistant = KnowledgeAssistant(
        pipeline
    )

    # Prepare a prompt.
    prompt = assistant.prepare(
        "What is something unknown?"
    )

    # Verify the question is still present.
    assert "What is something unknown?" in prompt

    # Verify that the prompt acknowledges
    # that no knowledge was retrieved.
    assert (
        "No relevant knowledge was retrieved"
        in prompt
    )