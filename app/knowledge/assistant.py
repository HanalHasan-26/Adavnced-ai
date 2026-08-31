from __future__ import annotations

import re

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

# Import long-term assistant memory.
from app.memory.assistant_memory import AssistantMemory

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
        assistant_memory: AssistantMemory | None = None,
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

        # Short/episodic conversation memory.
        self.conversation_memory = (
            conversation_memory
            or ConversationMemory()
        )

        # Long-term assistant memory.
        self.assistant_memory = (
            assistant_memory
            or AssistantMemory()
        )

        # -----------------------------------------------------
        # WEB COMPONENTS
        # -----------------------------------------------------

        # The default mode is OFFLINE.
        self.web_mode_controller = (
            web_mode_controller
            or WebModeController()
        )

        # Store the web-search client.
        self.web_search = web_search

        # Validate search results.
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

        # Store persistent assistant memory.
        return self.assistant_memory.remember(
            content
        )

    def recall_memory(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:

        # Search persistent long-term memories.
        return self.assistant_memory.recall(
            query=query,
            limit=limit,
        )

    def build_memory_context(
        self,
        query: str,
        limit: int = 5,
    ) -> str:

        # Build prompt-ready memory context.
        return self.assistant_memory.build_context(
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

        # OFFLINE mode must never access the internet.
        if (
            self.web_mode_controller.mode
            == WebMode.OFFLINE
        ):
            return []

        if self.web_search is None:
            raise ValueError(
                "web_search cannot be None"
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
                "web_search cannot be None"
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
                # Keep title, URL and snippet
                # even when fetching fails.
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
    # AUTHORITATIVE USER FACTS
    # =========================================================

    def _build_authoritative_user_context(self) -> str:
        """
        Extract explicit facts stated by the user.

        User messages are authoritative for personal information.
        Assistant-generated answers are deliberately ignored.

        The newest explicit user statement wins when the same
        fact has previously been stated differently.
        """

        memory = getattr(
            self.conversation_memory,
            "memory",
            None,
        )

        if memory is None or not hasattr(memory, "list"):
            return ""

        try:
            records = memory.list()
        except Exception:
            return ""

        if not isinstance(records, list):
            return ""

        # Memory.list() normally returns newest first.
        # Sort by timestamp when available so this also works
        # with memory implementations that return another order.
        def record_time(record: dict) -> str:
            if not isinstance(record, dict):
                return ""
            value = record.get("created_at", "")
            return value if isinstance(value, str) else ""

        records = sorted(
            records,
            key=record_time,
            reverse=True,
        )

        # ---------------------------------------------------------
        # NAME
        # ---------------------------------------------------------

        # Match the name itself without swallowing the rest of a
        # natural-language sentence.
        #
        # Examples:
        #
        #   My name is Hanal.
        #       -> Hanal
        #
        #   My name is Hanal, remember that.
        #       -> Hanal
        #
        #   My name is Hanal u got it?
        #       -> Hanal
        #
        #   My actual name is Hanal Hasan.
        #       -> Hanal Hasan
        #
        # The look-ahead stops the name at punctuation or common
        # follow-up phrases instead of storing the entire sentence.
        name_patterns = [
            re.compile(
                r"^\s*(?:my\s+name\s+is|my\s+actual\s+name\s+is)"
                r"\s+([A-Za-z][A-Za-z'-]*"
                r"(?:\s+[A-Za-z][A-Za-z'-]*){0,3})"
                r"(?=\s*(?:[,!?\.]|$)"
                r"|\s+(?:u|you)\s+got\b"
                r"|\s+(?:remember|right|okay|ok)\b)",
                re.IGNORECASE,
            ),
            re.compile(
                r"^\s*(?:i\s+am|i'm|im)"
                r"\s+([A-Za-z][A-Za-z'-]*"
                r"(?:\s+[A-Za-z][A-Za-z'-]*){0,3})"
                r"(?=\s*(?:[,!?\.]|$)"
                r"|\s+(?:u|you)\s+got\b"
                r"|\s+(?:remember|right|okay|ok)\b)",
                re.IGNORECASE,
            ),
            re.compile(
                r"^\s*(?:call\s+me)"
                r"\s+([A-Za-z][A-Za-z'-]*"
                r"(?:\s+[A-Za-z][A-Za-z'-]*){0,3})"
                r"(?=\s*(?:[,!?\.]|$)"
                r"|\s+(?:u|you)\s+got\b"
                r"|\s+(?:remember|right|okay|ok)\b)",
                re.IGNORECASE,
            ),
        ]

        latest_name: str | None = None

        for record in records:
            if not isinstance(record, dict):
                continue

            content = record.get("content", "")

            if not isinstance(content, str):
                continue

            content = content.strip()

            # Only explicit USER records can establish user facts.
            if not content.lower().startswith("user:"):
                continue

            message = content[5:].strip()

            # Ignore nested console prefixes such as:
            # "User: You: My name is Hanal."
            while message.lower().startswith("you:"):
                message = message[4:].strip()

            for pattern in name_patterns:
                match = pattern.match(message)

                if match:
                    candidate = match.group(1).strip()

                    # Do not accept obviously negative statements.
                    if candidate.lower() not in {
                        "not",
                        "unknown",
                    }:
                        latest_name = candidate

                    break

            # Newest matching user statement wins.
            if latest_name is not None:
                break

        if latest_name is None:
            return ""

        return (
            "AUTHORITATIVE USER FACTS:\n"
            f"- The user's current name is {latest_name}.\n"
            "- This fact comes from the user's own explicit statement.\n"
            "- If older conversation or assistant responses contain a "
            "different name, ignore the older conflicting information.\n"
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

        context_parts: list[str] = []

        if retrieval_result.context:

            context_parts.append(
                retrieval_result.context
            )
        else:

            context_parts.append(
            "No relevant knowledge was retrieved."
            )

        # -----------------------------------------------------
        # CONVERSATION MEMORY
        # -----------------------------------------------------

        conversation_context = (
            self.conversation_memory.build_context(
                query=query,
                limit=5,
            )
        )

        if conversation_context:

            context_parts.append(
                "Previous conversation:\n"
                f"{conversation_context}"
            )

        # -----------------------------------------------------
        # AUTHORITATIVE USER FACTS
        # -----------------------------------------------------

        user_fact_context = (
            self._build_authoritative_user_context()
        )

        if user_fact_context:

            context_parts.append(
                user_fact_context
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

        if memory_context:

            context_parts.append(
                "Long-term memory:\n"
                f"{memory_context}"
            )

        # -----------------------------------------------------
        # WEB RESEARCH
        # -----------------------------------------------------

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

        # Build complete prompt.
        prompt = self.prepare(
            query=query,
            limit=limit,
        )

        # Generate answer.
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
        # SAVE CONVERSATION
        # -----------------------------------------------------

        self.conversation_memory.save_user_message(
            query
        )

        self.conversation_memory.save_assistant_message(
            answer
        )

        return answer