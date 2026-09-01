from __future__ import annotations

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

from app.web.auto import (
    WebAutoDecider,
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
        web_auto_decider: WebAutoDecider | None = None,
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

        self.conversation_memory = (
            conversation_memory
            or ConversationMemory()
        )

        self.assistant_memory = (
            assistant_memory
            or AssistantMemory()
        )

        self.user_facts = (
            user_facts
            or UserFacts()
        )

        # -----------------------------------------------------
        # WEB COMPONENTS
        # -----------------------------------------------------

        self.web_mode_controller = (
            web_mode_controller
            or WebModeController(
                mode=WebMode.AUTO
            )
        )

        self.web_auto_decider = (
            web_auto_decider
            or WebAutoDecider()
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
    # QUERY HELPERS
    # =========================================================

    @staticmethod
    def _is_direct_web_query(
        query: str,
    ) -> bool:

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        return query.strip().lower().startswith(
            "web:"
        )

    @staticmethod
    def _remove_direct_web_prefix(
        query: str,
    ) -> str:

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        query = query.strip()

        if not query.lower().startswith("web:"):
            return query

        cleaned_query = query[4:].strip()

        if not cleaned_query:
            raise ValueError(
                "web query cannot be empty."
            )

        return cleaned_query

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
    # STORE WEB ANSWER
    # =========================================================

    def _remember_web_answer(
        self,
        query: str,
        answer: str,
    ) -> None:
        """
        Store a successful web-derived answer in long-term
        assistant memory.

        Memory failures must never break the answer.
        """

        if not isinstance(query, str):
            return

        if not isinstance(answer, str):
            return

        query = query.strip()
        answer = answer.strip()

        if not query or not answer:
            return

        memory_content = (
            "[WEB ANSWER]\n"
            f"Question: {query}\n"
            f"Answer: {answer}"
        )

        try:

            self.assistant_memory.remember(
                memory_content
            )

        except (
            RuntimeError,
            ValueError,
        ):

            pass

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
    # WEB MODE DECISION
    # =========================================================

    def _should_use_web(
        self,
        query: str,
    ) -> bool:

        mode = (
            self.web_mode_controller.mode
        )

        # -----------------------------------------------------
        # OFFLINE
        # -----------------------------------------------------

        if mode == WebMode.OFFLINE:
            return False

        # -----------------------------------------------------
        # WEB
        # -----------------------------------------------------

        if mode == WebMode.WEB:
            return True

        # -----------------------------------------------------
        # AUTO
        # -----------------------------------------------------

        if mode == WebMode.AUTO:

            return self.web_auto_decider.should_use_web(
                query
            )

        return False

    # =========================================================
    # WEB SEARCH
    # =========================================================

    def web_retrieve(
        self,
        query: str,
        limit: int = 5,
        force: bool = False,
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

        # -----------------------------------------------------
        # DECIDE WHETHER WEB SHOULD BE USED
        # -----------------------------------------------------

        if not force:

            if not self._should_use_web(
                query
            ):
                return []

        # -----------------------------------------------------
        # WEB CLIENT
        # -----------------------------------------------------

        if self.web_search is None:

            mode = (
                self.web_mode_controller.mode
            )

            # WEB mode explicitly requires web.
            if mode == WebMode.WEB:

                raise ValueError(
                    "web_search cannot be None."
                )

            # Explicit web: request requires web.
            if force:

                raise ValueError(
                    "web_search cannot be None."
                )

            # AUTO mode gracefully falls back
            # to local knowledge/memory.
            return []

        # -----------------------------------------------------
        # SEARCH
        # -----------------------------------------------------

        results = self.web_search.search(
            query=query,
            limit=limit,
        )

        if results is None:
            return []

        # -----------------------------------------------------
        # VALIDATION
        # -----------------------------------------------------

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
        force: bool = False,
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
        # SEARCH
        # -----------------------------------------------------

        results = self.web_retrieve(
            query=query,
            limit=limit,
            force=force,
        )

        if not results:
            return ""

        # -----------------------------------------------------
        # BUILD RESEARCH CONTEXT
        # -----------------------------------------------------

        research_sections: list[str] = []

        for result in results:

            section_parts = [
                f"Title: {result.title}",
                f"URL: {result.url}",
                f"Snippet: {result.snippet}",
            ]

            # -------------------------------------------------
            # FETCH PAGE
            # -------------------------------------------------

            try:

                html = (
                    self.web_page_fetcher.fetch(
                        result.url
                    )
                )

                # ---------------------------------------------
                # EXTRACT TEXT
                # ---------------------------------------------

                text = (
                    self.web_text_extractor.extract(
                        html
                    )
                )

                # ---------------------------------------------
                # SECURITY PROCESSING
                # ---------------------------------------------

                if text:

                    safe_text = (
                        self.web_content_security
                        .build_safe_context(
                            text
                        )
                    )

                    if safe_text:

                        section_parts.append(
                            safe_text
                        )

            except (
                RuntimeError,
                ValueError,
            ):

                # Search metadata is still useful when
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
    # DIRECT WEB PROMPT
    # =========================================================

    def prepare_direct_web(
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
        # CLEAN QUERY
        # -----------------------------------------------------

        clean_query = (
            self._remove_direct_web_prefix(
                query
            )
        )

        # -----------------------------------------------------
        # WEB RESEARCH
        # -----------------------------------------------------

        web_context = self.web_research(
            query=clean_query,
            limit=limit,
            force=True,
        )

        # -----------------------------------------------------
        # FALLBACK
        # -----------------------------------------------------

        if not web_context:

            web_context = (
                "No web search results were available."
            )

        # -----------------------------------------------------
        # WEB-ONLY KNOWLEDGE
        # -----------------------------------------------------

        knowledge = (
            "No local knowledge was used. "
            "This request was explicitly sent "
            "to the web."
        )

        # -----------------------------------------------------
        # CONTEXT
        # -----------------------------------------------------

        context = (
            "DIRECT WEB RESEARCH:\n"
            + web_context
        )

        # -----------------------------------------------------
        # BUILD PROMPT
        # -----------------------------------------------------

        return self.prompt_builder.build(
            query=clean_query,
            context=context,
            knowledge=knowledge,
        )

    # =========================================================
    # NORMAL PROMPT
    # =========================================================

    def prepare(
        self,
        query: str,
        limit: int = 5,
    ) -> str:
        """
        Normal assistant flow.

        OFFLINE:
            Local knowledge + memory only.

        AUTO:
            Local knowledge + memory.
            If there is no useful long-term memory,
            WebAutoDecider may trigger web research.

        WEB:
            Web is mandatory and is always searched,
            regardless of existing memory.

        Explicit:
            web: question
            always bypasses this method entirely.
        """

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
        # LONG-TERM MEMORY
        # -----------------------------------------------------

        memory_context = (
            self.build_memory_context(
                query=query,
                limit=5,
            )
        )

        # -----------------------------------------------------
        # KNOWLEDGE
        # -----------------------------------------------------

        if knowledge_context:

            knowledge = knowledge_context

        else:

            knowledge = (
                "No relevant knowledge was retrieved."
            )

        # -----------------------------------------------------
        # WEB RESEARCH
        # -----------------------------------------------------

        web_context = ""

        mode = (
            self.web_mode_controller.mode
        )

        # -----------------------------------------------------
        # WEB MODE
        # -----------------------------------------------------
        #
        # IMPORTANT:
        #
        # WEB mode must ALWAYS perform a web search.
        #
        # Existing memory does NOT suppress the search.
        #
        if mode == WebMode.WEB:

            web_context = self.web_research(
                query=query,
                limit=limit,
                force=False,
            )

        # -----------------------------------------------------
        # AUTO MODE
        # -----------------------------------------------------
        #
        # In AUTO mode, useful long-term memory can prevent
        # another web search.
        #
        # If there is no memory, ask WebAutoDecider.
        #
        elif mode == WebMode.AUTO:

            if not memory_context:

                should_search_web = (
                    self._should_use_web(
                        query
                    )
                )

                if should_search_web:

                    web_context = self.web_research(
                        query=query,
                        limit=limit,
                        force=False,
                    )

        # -----------------------------------------------------
        # OFFLINE MODE
        # -----------------------------------------------------
        #
        # Nothing to do. Web remains disabled.
        #
        elif mode == WebMode.OFFLINE:

            web_context = ""

        # -----------------------------------------------------
        # BUILD CONTEXT
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
        # DETERMINE ROUTE
        # -----------------------------------------------------

        direct_web = (
            self._is_direct_web_query(
                query
            )
        )

        # -----------------------------------------------------
        # NORMALIZE QUERY
        # -----------------------------------------------------

        if direct_web:

            clean_query = (
                self._remove_direct_web_prefix(
                    query
                )
            )

        else:

            clean_query = query.strip()

        # -----------------------------------------------------
        # BUILD PROMPT
        # -----------------------------------------------------

        if direct_web:

            prompt = self.prepare_direct_web(
                query=query,
                limit=limit,
            )

        else:

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
        # DETERMINE WHETHER WEB WAS ACTUALLY USED
        # -----------------------------------------------------

        web_answer = direct_web

        if not web_answer:

            if "Web research results:" in prompt:

                web_answer = True

        # -----------------------------------------------------
        # STORE WEB ANSWER
        # -----------------------------------------------------

        if web_answer:

            self._remember_web_answer(
                query=clean_query,
                answer=answer,
            )

        # -----------------------------------------------------
        # LEARN USER FACTS
        # -----------------------------------------------------
        #
        # Only the user's original message is passed.
        #
        # The assistant's answer is NEVER passed to
        # UserFacts.
        #
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