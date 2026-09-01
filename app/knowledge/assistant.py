from __future__ import annotations

# =========================================================
# STANDARD LIBRARY
# =========================================================

import re


# =========================================================
# KNOWLEDGE
# =========================================================

from app.knowledge.retrieval.pipeline import (
    KnowledgeRetrievalPipeline,
)

from app.knowledge.retrieval.result import (
    KnowledgeRetrievalResult,
)


# =========================================================
# LLM
# =========================================================

from app.llm.prompt_builder import PromptBuilder
from app.llm.client import LLMClient


# =========================================================
# MEMORY
# =========================================================

from app.memory.conversation import ConversationMemory
from app.memory.assistant_memory import AssistantMemory
from app.memory.user_facts import UserFacts


# =========================================================
# WEB
# =========================================================

from app.web.mode import (
    WebMode,
    WebModeController,
)

from app.web.search import (
    SearchResult,
    WebSearch,
)

from app.web.validation import (
    WebResultValidator,
)

from app.web.fetcher import (
    WebPageFetcher,
)

from app.web.extractor import (
    HTMLTextExtractor,
)

from app.web.security import (
    WebContentSecurity,
)


# =========================================================
# KNOWLEDGE ASSISTANT
# =========================================================

class KnowledgeAssistant:

    def __init__(
        self,
        retrieval_pipeline: KnowledgeRetrievalPipeline,
        prompt_builder: PromptBuilder | None = None,
        llm: LLMClient | None = None,
        conversation_memory: ConversationMemory | None = None,
        assistant_memory: AssistantMemory | None = None,
        user_facts: UserFacts | None = None,
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

        self.retrieval_pipeline = retrieval_pipeline

        self.prompt_builder = (
            prompt_builder
            or PromptBuilder()
        )

        self.llm = llm

        # Short-term conversation memory.
        self.conversation_memory = (
            conversation_memory
            or ConversationMemory()
        )

        # Long-term assistant memory.
        self.assistant_memory = (
            assistant_memory
            or AssistantMemory()
        )

        # Structured persistent user facts.
        self.user_facts = (
            user_facts
            or UserFacts()
        )

        # -----------------------------------------------------
        # WEB COMPONENTS
        # -----------------------------------------------------

        self.web_mode_controller = (
            web_mode_controller
            or WebModeController()
        )

        self.web_search = web_search

        self.web_result_validator = (
            web_result_validator
            or WebResultValidator()
        )

        self.web_page_fetcher = (
            web_page_fetcher
            or WebPageFetcher()
        )

        self.web_text_extractor = (
            web_text_extractor
            or HTMLTextExtractor()
        )

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

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        return self.retrieval_pipeline.run(
            query=query,
            limit=limit,
        )

    # =========================================================
    # LONG-TERM MEMORY
    # =========================================================

    def remember(
        self,
        content: str,
    ) -> str:

        return self.assistant_memory.remember(
            content
        )

    def recall_memory(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:

        return self.assistant_memory.recall(
            query=query,
            limit=limit,
        )

    def build_memory_context(
        self,
        query: str,
        limit: int = 5,
    ) -> str:

        return self.assistant_memory.build_context(
            query=query,
            limit=limit,
        )

    # =========================================================
    # AUTHORITATIVE USER FACTS
    # =========================================================

    def _build_authoritative_user_context(
        self,
    ) -> str:

        try:
            return self.user_facts.build_context(
                limit=50
            )

        except (
            RuntimeError,
            ValueError,
        ):
            return ""

    # =========================================================
    # WEB SEARCH
    # =========================================================

    def web_retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> list[SearchResult]:

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        # Offline mode must NEVER access the internet.
        if (
            self.web_mode_controller.mode
            == WebMode.OFFLINE
        ):
            return []

        if self.web_search is None:
            raise ValueError(
                "web_search cannot be None."
            )

        results = self.web_search.search(
            query=query,
            limit=limit,
        )

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

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        if (
            self.web_mode_controller.mode
            == WebMode.OFFLINE
        ):
            return ""

        if self.web_search is None:
            raise ValueError(
                "web_search cannot be None."
            )

        results = self.web_retrieve(
            query=query,
            limit=limit,
        )

        if not results:
            return ""

        research_sections: list[str] = []

        for result in results:

            section_parts = [
                f"Title: {result.title}",
                f"URL: {result.url}",
                f"Snippet: {result.snippet}",
            ]

            try:

                html = (
                    self.web_page_fetcher.fetch(
                        result.url
                    )
                )

                text = (
                    self.web_text_extractor.extract(
                        html
                    )
                )

                if text:

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

                # Keep search metadata even if
                # webpage fetching fails.
                pass

            research_sections.append(
                "\n".join(
                    section_parts
                )
            )

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

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        # -----------------------------------------------------
        # LOCAL KNOWLEDGE
        # -----------------------------------------------------

        retrieval_result = self.retrieve(
            query=query,
            limit=limit,
        )

        knowledge_context = (
            retrieval_result.context
        )

        # -----------------------------------------------------
        # PREVIOUS CONVERSATION
        # -----------------------------------------------------

        conversation_context = (
            self.conversation_memory.build_context(
                query=query,
                limit=5,
            )
        )

        # -----------------------------------------------------
        # AUTHORITATIVE USER FACTS
        # -----------------------------------------------------

        user_fact_context = (
            self._build_authoritative_user_context()
        )

        # -----------------------------------------------------
        # LONG-TERM ASSISTANT MEMORY
        # -----------------------------------------------------

        memory_context = (
            self.build_memory_context(
                query=query,
                limit=5,
            )
        )

        # -----------------------------------------------------
        # WEB RESEARCH
        # -----------------------------------------------------

        web_context = self.web_research(
            query=query,
            limit=limit,
        )

        # -----------------------------------------------------
        # BUILD KNOWLEDGE SECTION
        # -----------------------------------------------------

        if knowledge_context:

            knowledge = knowledge_context

        else:

            knowledge = (
                "No relevant knowledge was retrieved."
            )

        # -----------------------------------------------------
        # BUILD FINAL CONTEXT
        # -----------------------------------------------------

        context_parts: list[str] = []

        if conversation_context:

            context_parts.append(
                "Previous conversation:\n"
                + conversation_context
            )

        if user_fact_context:

            context_parts.append(
                user_fact_context
            )

        if memory_context:

            context_parts.append(
                "Long-term memory:\n"
                + memory_context
            )

        if web_context:

            context_parts.append(
                "Web research results:\n"
                + web_context
            )

        combined_context = (
            "\n\n".join(
                context_parts
            )
        )

        # -----------------------------------------------------
        # BUILD PROMPT
        # -----------------------------------------------------

        return self.prompt_builder.build(
            query=retrieval_result.query,
            context=combined_context,
            knowledge=knowledge,
        )

    # =========================================================
    # ASK LOCAL LLM
    # =========================================================

    def ask(
        self,
        query: str,
        limit: int = 5,
    ) -> str:

        if self.llm is None:
            raise ValueError(
                "llm cannot be None."
            )

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        # -----------------------------------------------------
        # BUILD PROMPT
        # -----------------------------------------------------

        prompt = self.prepare(
            query=query,
            limit=limit,
        )

        # -----------------------------------------------------
        # GENERATE ANSWER
        # -----------------------------------------------------

        answer = self.llm.generate(
            prompt
        )

        if not isinstance(answer, str):
            raise ValueError(
                "llm returned a non-string response."
            )

        answer = answer.strip()

        if not answer:
            raise ValueError(
                "llm returned an empty response."
            )

        # -----------------------------------------------------
        # LEARN EXPLICIT USER FACTS
        # -----------------------------------------------------

        # IMPORTANT:
        #
        # Only the user's message is passed to UserFacts.
        #
        # The AI's answer is NEVER passed to UserFacts.
        #
        # Therefore the AI cannot accidentally invent
        # or overwrite a personal fact.

        self.user_facts.learn_from_user_message(
            query
        )

        # -----------------------------------------------------
        # SAVE CONVERSATION
        # -----------------------------------------------------

        self.conversation_memory.save_user_message(
            query
        )

        self.conversation_memory.save_assistant_message(
            answer
        )

        return answer