# Import the retrieval pipeline.
from app.knowledge.retrieval.pipeline import KnowledgeRetrievalPipeline

# Import the prompt builder.
from app.llm.prompt_builder import PromptBuilder

# Import the LLM client interface.
from app.llm.client import LLMClient

# Import the retrieval result model.
from app.knowledge.retrieval.result import KnowledgeRetrievalResult

# Import conversation memory.
from app.memory.conversation import ConversationMemory


# Create the main knowledge-aware AI assistant.
class KnowledgeAssistant:

    # Initialize the assistant.
    def __init__(
        self,
        retrieval_pipeline: KnowledgeRetrievalPipeline,
        prompt_builder: PromptBuilder | None = None,
        llm: LLMClient | None = None,
        conversation_memory: ConversationMemory | None = None,
    ):

        # Make sure a retrieval pipeline was supplied.
        if retrieval_pipeline is None:
            raise ValueError(
                "retrieval_pipeline cannot be None."
            )

        # Store the retrieval pipeline.
        self.retrieval_pipeline = retrieval_pipeline

        # Use the supplied prompt builder when provided.
        # Otherwise create a default one.
        self.prompt_builder = (
            prompt_builder
            or PromptBuilder()
        )

        # Store the LLM client.
        #
        # The LLM is optional because prepare()
        # and retrieve() can work without a model.
        self.llm = llm

        # Use the supplied conversation memory when provided.
        # Otherwise create the default persistent memory system.
        self.conversation_memory = (
            conversation_memory
            or ConversationMemory()
        )

    # Retrieve knowledge and prepare the final prompt.
    def prepare(
        self,
        query: str,
        limit: int = 5,
    ) -> str:

        # Reject empty queries.
        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        # Run the knowledge retrieval pipeline.
        result: KnowledgeRetrievalResult = (
            self.retrieval_pipeline.run(
                query=query,
                limit=limit,
            )
        )

        # Build the final prompt using the
        # retrieved knowledge context.
        return self.prompt_builder.build(
            query=result.query,
            context=result.context,
        )

    # Retrieve knowledge only.
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
        return self.retrieval_pipeline.run(
            query=query,
            limit=limit,
        )

    # Ask the local LLM and return its answer.
    def ask(
        self,
        query: str,
        limit: int = 5,
    ) -> str:

        # An LLM is required when generating
        # an actual answer.
        if self.llm is None:
            raise ValueError(
                "llm cannot be None."
            )

        # Reject empty queries.
        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        # Retrieve relevant knowledge.
        retrieval_result = self.retrieve(
            query=query,
            limit=limit,
        )

        # Retrieve relevant previous conversation.
        memory_context = (
            self.conversation_memory.build_context(
                query=query,
                limit=5,
            )
        )

        # Start with the retrieved knowledge context.
        context_parts = []

        if retrieval_result.context:
            context_parts.append(
                retrieval_result.context
            )

        # Add conversation memory when available.
        if memory_context:
            context_parts.append(
                "Previous conversation:\n"
                f"{memory_context}"
            )

        # Combine all available context.
        combined_context = "\n\n".join(
            context_parts
        )

        # Build the final prompt.
        prompt = self.prompt_builder.build(
            query=retrieval_result.query,
            context=combined_context,
        )

        # Generate the answer.
        answer = self.llm.generate(prompt)

        # Make sure the LLM returned a valid string.
        if not isinstance(answer, str):
            raise ValueError(
                "llm returned a non-string response."
            )

        # Remove unnecessary whitespace.
        answer = answer.strip()

        # Save the user message.
        self.conversation_memory.save_user_message(
            query
        )

        # Save the assistant response.
        self.conversation_memory.save_assistant_message(
            answer
        )

        # Return the final answer.
        return answer