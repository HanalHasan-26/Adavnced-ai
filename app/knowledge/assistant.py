from __future__ import annotations

# Import the retrieval pipeline.
from app.knowledge.retrieval.pipeline import (
    KnowledgeRetrievalPipeline,
)

# Import the prompt builder.
from app.llm.prompt_builder import PromptBuilder

# Import the LLM client interface.
from app.llm.client import LLMClient

# Import the retrieval result model.
from app.knowledge.retrieval.result import (
    KnowledgeRetrievalResult,
)

# Import conversation memory.
from app.memory.conversation import ConversationMemory

# Import web mode support.
from app.web.mode import (
    WebMode,
    WebModeController,
)

# Import web search.
from app.web.search import (
    SearchResult,
    WebSearch,
)

# Import web-result validation.
from app.web.validation import (
    WebResultValidator,
)

# Import webpage fetching.
from app.web.fetcher import (
    WebPageFetcher,
)

# Import HTML text extraction.
from app.web.extractor import (
    HTMLTextExtractor,
)

# Import web-content security.
from app.web.security import (
    WebContentSecurity,
)


class KnowledgeAssistant:

    def __init__(
        self,
        retrieval_pipeline: KnowledgeRetrievalPipeline,
        prompt_builder: PromptBuilder | None = None,
        llm: LLMClient | None = None,
        conversation_memory: ConversationMemory | None = None,
        web_mode_controller: WebModeController | None = None,
        web_search: WebSearch | None = None,
        web_result_validator: WebResultValidator | None = None,
        web_page_fetcher: WebPageFetcher | None = None,
        web_text_extractor: HTMLTextExtractor | None = None,
        web_content_security: WebContentSecurity | None = None,
    ):

        # -----------------------------------------------------
        # LOCAL COMPONENTS
        # -----------------------------------------------------

        if retrieval_pipeline is None:
            raise ValueError(
                "retrieval_pipeline cannot be None."
            )

        self.retrieval_pipeline = (
            retrieval_pipeline
        )

        self.prompt_builder = (
            prompt_builder
            or PromptBuilder()
        )

        self.llm = llm

        self.conversation_memory = (
            conversation_memory
            or ConversationMemory()
        )

        # -----------------------------------------------------
        # WEB COMPONENTS
        # -----------------------------------------------------

        # Create the web mode controller.
        #
        # The default mode is OFFLINE, so the assistant
        # will not access the internet unless WEB mode
        # is explicitly enabled.
        self.web_mode_controller = (
            web_mode_controller
            or WebModeController()
        )

        # Store the web-search client.
        #
        # Do NOT validate this here.
        #
        # WEB mode validates it when a web operation is
        # actually requested. This allows the assistant
        # object itself to be created without a web client.
        self.web_search = web_search

        # Validate search results before using them.
        self.web_result_validator = (
            web_result_validator
            or WebResultValidator()
        )

        # Fetch webpage HTML.
        self.web_page_fetcher = (
            web_page_fetcher
            or WebPageFetcher()
        )

        # Extract readable text from HTML.
        self.web_text_extractor = (
            web_text_extractor
            or HTMLTextExtractor()
        )

        # Protect the LLM from untrusted webpage content.
        self.web_content_security = (
            web_content_security
            or WebContentSecurity()
        )

    # =========================================================
    # LOCAL KNOWLEDGE
    # =========================================================

    def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> KnowledgeRetrievalResult:

        # Validate query type.
        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        # Reject empty queries.
        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        # Validate result limit.
        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        # Run the local retrieval pipeline.
        return self.retrieval_pipeline.run(
            query=query,
            limit=limit,
        )

    # =========================================================
    # WEB SEARCH
    # =========================================================

    def web_retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> list[SearchResult]:

        # Validate query type.
        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        # Reject empty queries.
        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        # Validate limit.
        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        # OFFLINE mode must never access the internet.
        if (
            self.web_mode_controller.mode
            == WebMode.OFFLINE
        ):
            return []

        # WEB mode requires a search client.
        #
        # This check intentionally happens here instead
        # of __init__, because web_search is only required
        # when a web operation is actually performed.
        if self.web_search is None:
            raise ValueError(
                "web_search cannot be None"
            )

        # Perform the web search.
        results = self.web_search.search(
            query=query,
            limit=limit,
        )

        # Validate all returned search results.
        return (
            self.web_result_validator.validate_many(
                results
            )
        )

    # =========================================================
    # WEB RESEARCH
    # =========================================================

    def web_research(
        self,
        query: str,
        limit: int = 5,
    ) -> str:

        # Validate query type.
        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        # Reject empty queries.
        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        # Validate limit.
        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        # OFFLINE mode means no internet access.
        if (
            self.web_mode_controller.mode
            == WebMode.OFFLINE
        ):
            return ""

        # WEB mode requires a search client.
        if self.web_search is None:
            raise ValueError(
                "web_search cannot be None"
            )

        # Search the web.
        results = self.web_retrieve(
            query=query,
            limit=limit,
        )

        # No search results.
        if not results:
            return ""

        research_sections: list[str] = []

        # Process every search result.
        for result in results:

            # -------------------------------------------------
            # ALWAYS KEEP THE SEARCH RESULT
            # -------------------------------------------------
            #
            # Even if the webpage cannot be downloaded,
            # the title, URL and snippet are still useful.
            #

            section_parts = [
                f"Title: {result.title}",
                f"URL: {result.url}",
                f"Snippet: {result.snippet}",
            ]

            # -------------------------------------------------
            # TRY TO FETCH THE WEBPAGE
            # -------------------------------------------------

            try:

                # Fetch webpage HTML.
                html = (
                    self.web_page_fetcher.fetch(
                        result.url
                    )
                )

                # Convert HTML to readable text.
                text = (
                    self.web_text_extractor.extract(
                        html
                    )
                )

                # Only add webpage text when extraction
                # produced useful content.
                if text:

                    # Treat webpage content as untrusted.
                    safe_text = (
                        self.web_content_security
                        .build_safe_context(
                            text
                        )
                    )

                    section_parts.append(
                        safe_text
                    )

            except (
                RuntimeError,
                ValueError,
            ):
                # Do not discard the search result.
                #
                # Title + URL + snippet remain available
                # even when webpage fetching fails.
                pass

            # Add this result to the research.
            research_sections.append(
                "\n".join(
                    section_parts
                )
            )

        # Combine all search results.
        return "\n\n".join(
            research_sections
        )

    # =========================================================
    # PREPARE PROMPT
    # =========================================================

    def prepare(
        self,
        query: str,
        limit: int = 5,
    ) -> str:

        # Validate query type.
        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        # Reject empty queries.
        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        # -----------------------------------------------------
        # LOCAL KNOWLEDGE
        # -----------------------------------------------------

        retrieval_result = self.retrieve(
            query=query,
            limit=limit,
        )

        context_parts: list[str] = []

        # Add local knowledge.
        if retrieval_result.context:

            context_parts.append(
                retrieval_result.context
            )

        # -----------------------------------------------------
        # CONVERSATION MEMORY
        # -----------------------------------------------------

        memory_context = (
            self.conversation_memory.build_context(
                query=query,
                limit=5,
            )
        )

        if memory_context:

            context_parts.append(
                "Previous conversation:\n"
                f"{memory_context}"
            )

        # -----------------------------------------------------
        # WEB RESEARCH
        # -----------------------------------------------------

        # web_research() itself checks the current mode.
        #
        # Therefore OFFLINE mode returns an empty string
        # without accessing the internet.
        web_context = self.web_research(
            query=query,
            limit=limit,
        )

        if web_context:

            context_parts.append(
                "Web research results:\n"
                f"{web_context}"
            )

        # -----------------------------------------------------
        # COMBINE CONTEXT
        # -----------------------------------------------------

        combined_context = (
            "\n\n".join(
                context_parts
            )
        )

        # Build the final prompt.
        return self.prompt_builder.build(
            query=retrieval_result.query,
            context=combined_context,
        )

    # =========================================================
    # ASK LOCAL LLM
    # =========================================================

    def ask(
        self,
        query: str,
        limit: int = 5,
    ) -> str:

        # An LLM is required to generate an answer.
        if self.llm is None:
            raise ValueError(
                "llm cannot be None."
            )

        # Validate query type.
        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        # Reject empty queries.
        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        # Build the complete prompt.
        prompt = self.prepare(
            query=query,
            limit=limit,
        )

        # Generate the answer.
        answer = self.llm.generate(
            prompt
        )

        # Validate returned answer.
        if not isinstance(answer, str):
            raise ValueError(
                "llm returned a non-string response."
            )

        # Remove unnecessary whitespace.
        answer = answer.strip()

        # Reject empty answers.
        if not answer:
            raise ValueError(
                "llm returned an empty response."
            )

        # -----------------------------------------------------
        # SAVE CONVERSATION
        # -----------------------------------------------------
        #
        # Only save the conversation after successful
        # answer generation.

        self.conversation_memory.save_user_message(
            query
        )

        self.conversation_memory.save_assistant_message(
            answer
        )

        return answer