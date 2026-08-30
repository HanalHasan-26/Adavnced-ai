# Import the retrieval pipeline.
from app.knowledge.retrieval.pipeline import KnowledgeRetrievalPipeline

# Import the prompt builder.
from app.knowledge.retrieval.prompt_builder import KnowledgePromptBuilder

# Import the retrieval result model.
from app.knowledge.retrieval.result import KnowledgeRetrievalResult


# Create the service that connects knowledge retrieval
# with prompt construction.
class KnowledgeAssistant:

    # Initialize the knowledge assistant.
    def __init__(
        self,
        pipeline: KnowledgeRetrievalPipeline,
        prompt_builder: KnowledgePromptBuilder | None = None,
    ):

        # Store the retrieval pipeline.
        self.pipeline = pipeline

        # Use the supplied prompt builder when provided.
        # Otherwise create a default one.
        self.prompt_builder = (
            prompt_builder
            or KnowledgePromptBuilder()
        )

    # Prepare a knowledge-aware prompt for a user question.
    def prepare(
        self,
        query: str,
        limit: int = 5,
    ) -> str:

        # Reject empty queries before starting retrieval.
        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        # Run the knowledge retrieval pipeline.
        result: KnowledgeRetrievalResult = (
            self.pipeline.run(
                query=query,
                limit=limit,
            )
        )

        # Build the final prompt using the retrieved context.
        return self.prompt_builder.build(
            query=result.query,
            context=result.context,
        )

    # Retrieve knowledge and return the complete retrieval result.
    def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> KnowledgeRetrievalResult:

        # Reject empty queries.
        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        # Run the retrieval pipeline.
        return self.pipeline.run(
            query=query,
            limit=limit,
        )