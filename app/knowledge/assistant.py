# Import the retrieval pipeline.
from app.knowledge.retrieval.pipeline import KnowledgeRetrievalPipeline

# Import the prompt builder.
from app.knowledge.retrieval.prompt_builder import KnowledgePromptBuilder

# Import the retrieval result model.
from app.knowledge.retrieval.result import KnowledgeRetrievalResult


# Create the service responsible for knowledge retrieval
# and prompt preparation.
class KnowledgeAssistant:

    # Initialize the knowledge assistant.
    def __init__(
        self,
        pipeline: KnowledgeRetrievalPipeline,
        prompt_builder: KnowledgePromptBuilder | None = None,
    ):

        # Make sure a retrieval pipeline was supplied.
        if pipeline is None:
            raise ValueError(
                "pipeline cannot be None."
            )

        # Store the retrieval pipeline.
        self.pipeline = pipeline

        # Use the supplied prompt builder when provided.
        # Otherwise create a default one.
        self.prompt_builder = (
            prompt_builder
            or KnowledgePromptBuilder()
        )

    # Prepare a knowledge-aware prompt.
    def prepare(
        self,
        query: str,
        limit: int = 5,
    ) -> str:

        # Remove unnecessary whitespace.
        query = query.strip()

        # Reject an empty query.
        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        # Retrieve relevant knowledge.
        result: KnowledgeRetrievalResult = (
            self.pipeline.run(
                query=query,
                limit=limit,
            )
        )

        # Build the final prompt.
        return self.prompt_builder.build(
            query=result.query,
            context=result.context,
        )

    # Retrieve knowledge and return the complete result.
    def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> KnowledgeRetrievalResult:

        # Remove unnecessary whitespace.
        query = query.strip()

        # Reject an empty query.
        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        # Make sure the limit is valid.
        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        # Run the retrieval pipeline.
        return self.pipeline.run(
            query=query,
            limit=limit,
        )