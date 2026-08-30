# Import the LLM client interface.
from app.llm.client import LLMClient

# Import the prompt builder.
from app.llm.prompt_builder import PromptBuilder

# Import the knowledge retrieval pipeline.
from app.knowledge.retrieval.pipeline import KnowledgeRetrievalPipeline


# Create the main assistant responsible for answering questions.
class KnowledgeAssistant:

    # Initialize the assistant.
    def __init__(
        self,
        retrieval_pipeline: KnowledgeRetrievalPipeline,
        prompt_builder: PromptBuilder,
        llm: LLMClient,
    ):

        # Make sure a retrieval pipeline was supplied.
        if retrieval_pipeline is None:
            raise ValueError(
                "retrieval_pipeline cannot be None."
            )

        # Make sure a prompt builder was supplied.
        if prompt_builder is None:
            raise ValueError(
                "prompt_builder cannot be None."
            )

        # Make sure an LLM client was supplied.
        if llm is None:
            raise ValueError(
                "llm cannot be None."
            )

        # Store the retrieval pipeline.
        self.retrieval_pipeline = retrieval_pipeline

        # Store the prompt builder.
        self.prompt_builder = prompt_builder

        # Store the LLM client.
        self.llm = llm

    # Ask the assistant a question.
    def ask(
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

        # Make sure the retrieval limit is valid.
        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        # Retrieve relevant knowledge.
        retrieval_result = self.retrieval_pipeline.run(
            query=query,
            limit=limit,
        )

        # Build a prompt using the retrieved context.
        prompt = self.prompt_builder.build(
            query=query,
            context=retrieval_result.context,
        )

        # Generate the final answer using the LLM.
        answer = self.llm.generate(prompt)

        # Make sure the LLM returned text.
        if not isinstance(answer, str):
            raise TypeError(
                "LLM must return a string."
            )

        # Remove unnecessary whitespace.
        answer = answer.strip()

        # Reject an empty answer.
        if not answer:
            raise ValueError(
                "LLM returned an empty response."
            )

        # Return the final answer.
        return answer
