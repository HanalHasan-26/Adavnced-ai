# Import the knowledge chunk model.
from app.knowledge.chunking.chunk import KnowledgeChunk

# Import the pipeline.
from app.knowledge.retrieval.pipeline import KnowledgeRetrievalPipeline


# Create a fake retriever for testing.
class FakeRetriever:

    # Initialize the fake retriever.
    def __init__(self, chunks):

        # Store the chunks that should be returned.
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

        # Remember the arguments received.
        self.received_query = query
        self.received_limit = limit

        # Return the test chunks.
        return self.chunks


# Test that the pipeline retrieves and assembles knowledge.
def test_pipeline_retrieves_and_assembles():

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

    # Create the fake retriever.
    retriever = FakeRetriever(chunks)

    # Create the pipeline.
    pipeline = KnowledgeRetrievalPipeline(retriever)

    # Run the pipeline.
    result = pipeline.run(
        query="support resistance",
        limit=2,
    )

    # Verify the query reached the retriever.
    assert retriever.received_query == "support resistance"

    # Verify the limit reached the retriever.
    assert retriever.received_limit == 2

    # Verify the assembled context.
    assert result.context == (
        "[Knowledge 1]\n"
        "Support is a price level.\n\n"
        "[Knowledge 2]\n"
        "Resistance is a price barrier."
    )

    # Verify both chunks were returned.
    assert len(result.chunks) == 2

    # Verify the result count.
    assert result.count == 2


# Test that an empty retrieval result produces empty context.
def test_pipeline_with_no_results():

    # Create a fake retriever with no chunks.
    retriever = FakeRetriever([])

    # Create the pipeline.
    pipeline = KnowledgeRetrievalPipeline(retriever)

    # Run the pipeline.
    result = pipeline.run("unknown")

    # No knowledge should produce empty context.
    assert result.context == ""

    # No chunks should be returned.
    assert result.chunks == []

    # Count should be zero.
    assert result.count == 0


# Test that the pipeline respects the requested limit.
def test_pipeline_respects_limit():

    # Create one test chunk.
    chunks = [
        KnowledgeChunk(
            id="chunk-1",
            document_id="document-1",
            chunk_index=0,
            content="Trading knowledge.",
        )
    ]

    # Create the fake retriever.
    retriever = FakeRetriever(chunks)

    # Create the pipeline.
    pipeline = KnowledgeRetrievalPipeline(retriever)

    # Run with a custom limit.
    pipeline.run(
        query="trading",
        limit=3,
    )

    # Verify the limit was forwarded.
    assert retriever.received_limit == 3


# Test that the pipeline preserves retrieved chunk order.
def test_pipeline_preserves_chunk_order():

    # Create chunks in a deliberate order.
    chunks = [
        KnowledgeChunk(
            id="chunk-3",
            document_id="document-1",
            chunk_index=2,
            content="Third.",
        ),
        KnowledgeChunk(
            id="chunk-1",
            document_id="document-1",
            chunk_index=0,
            content="First.",
        ),
        KnowledgeChunk(
            id="chunk-2",
            document_id="document-1",
            chunk_index=1,
            content="Second.",
        ),
    ]

    # Create the fake retriever.
    retriever = FakeRetriever(chunks)

    # Create the pipeline.
    pipeline = KnowledgeRetrievalPipeline(retriever)

    # Run the pipeline.
    result = pipeline.run("test")

    # Read the assembled context.
    context = result.context

    # Verify the original retrieval order is preserved.
    assert context.index("Third.") < context.index("First.")
    assert context.index("First.") < context.index("Second.")

    # Verify all three chunks were returned.
    assert result.count == 3