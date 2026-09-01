from __future__ import annotations

import re
from urllib.parse import parse_qs, unquote, urlparse

# =========================================================
# LOCAL KNOWLEDGE
# =========================================================

from app.knowledge.retrieval.pipeline import (
    KnowledgeRetrievalPipeline,
)

from app.knowledge.retrieval.result import (
    KnowledgeRetrievalResult,
)

from app.llm.prompt_builder import PromptBuilder
from app.llm.client import LLMClient

# =========================================================
# MEMORY
# =========================================================

from app.memory.conversation import ConversationMemory
from app.memory.assistant_memory import AssistantMemory

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


class KnowledgeAssistant:

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        retrieval_pipeline: KnowledgeRetrievalPipeline,
        prompt_builder: PromptBuilder | None = None,
        llm: LLMClient | None = None,

        # -----------------------------------------------------
        # MEMORY
        # -----------------------------------------------------

        conversation_memory: ConversationMemory | None = None,
        assistant_memory: AssistantMemory | None = None,

        # -----------------------------------------------------
        # WEB
        # -----------------------------------------------------

        web_mode_controller: WebModeController | None = None,
        web_auto_decider: WebAutoDecider | None = None,
        web_search: WebSearch | None = None,
        web_result_validator: WebResultValidator | None = None,
        web_page_fetcher: WebPageFetcher | None = None,
        web_text_extractor: HTMLTextExtractor | None = None,
        web_content_security: WebContentSecurity | None = None,

        # -----------------------------------------------------
        # DEBUG
        # -----------------------------------------------------

        debug_web: bool = False,
    ) -> None:

        # =====================================================
        # LOCAL COMPONENTS
        # =====================================================

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

        # =====================================================
        # MEMORY
        # =====================================================

        self.conversation_memory = (
            conversation_memory
            or ConversationMemory()
        )

        self.assistant_memory = (
            assistant_memory
            or AssistantMemory()
        )

        # =====================================================
        # WEB COMPONENTS
        # =====================================================

        self.web_mode_controller = (
            web_mode_controller
            or WebModeController(
                mode=WebMode.OFFLINE
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

        # =====================================================
        # DEBUG
        # =====================================================

        self.debug_web = bool(
            debug_web
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

        query = query.strip()

        if not query:
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

        if not isinstance(content, str):
            raise ValueError(
                "content must be a string."
            )

        content = content.strip()

        if not content:
            raise ValueError(
                "content cannot be empty."
            )

        return self.assistant_memory.remember(
            content
        )

    # =========================================================

    def recall_memory(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        return self.assistant_memory.recall(
            query=query,
            limit=limit,
        )

    # =========================================================

    def build_memory_context(
        self,
        query: str,
        limit: int = 5,
    ) -> str:

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        return self.assistant_memory.build_context(
            query=query,
            limit=limit,
        )

    # =========================================================
    # WEB MODE
    # =========================================================

    def _extract_web_query(
        self,
        query: str,
    ) -> tuple[str, bool]:

        query = query.strip()

        # -----------------------------------------------------
        # web: query
        # -----------------------------------------------------

        direct_pattern = re.compile(
            r"^\s*web\s*:\s*(.+?)\s*$",
            re.IGNORECASE,
        )

        match = direct_pattern.match(
            query
        )

        if match:

            clean_query = (
                match.group(1)
                .strip()
            )

            return clean_query, True

        # -----------------------------------------------------
        # [web: query]
        # -----------------------------------------------------

        bracket_pattern = re.compile(
            r"^\s*\[\s*web\s*:\s*(.+?)\s*\]\s*$",
            re.IGNORECASE,
        )

        match = bracket_pattern.match(
            query
        )

        if match:

            clean_query = (
                match.group(1)
                .strip()
            )

            return clean_query, True

        # -----------------------------------------------------
        # (web: query)
        # -----------------------------------------------------

        parenthesis_pattern = re.compile(
            r"^\s*\(\s*web\s*:\s*(.+?)\s*\)\s*$",
            re.IGNORECASE,
        )

        match = parenthesis_pattern.match(
            query
        )

        if match:

            clean_query = (
                match.group(1)
                .strip()
            )

            return clean_query, True

        return query, False

    # =========================================================
    # SHOULD USE WEB
    # =========================================================

    def should_use_web(
        self,
        query: str,
    ) -> bool:

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        query = query.strip()

        if not query:
            return False

        # -----------------------------------------------------
        # OFFLINE MODE
        # -----------------------------------------------------

        if (
            self.web_mode_controller.mode
            == WebMode.OFFLINE
        ):
            return False

        # -----------------------------------------------------
        # EXPLICIT WEB COMMAND
        # -----------------------------------------------------

        clean_query, explicit_web = (
            self._extract_web_query(
                query
            )
        )

        if explicit_web:
            return True

        # -----------------------------------------------------
        # WEB MODE
        # -----------------------------------------------------

        if (
            self.web_mode_controller.mode
            == WebMode.WEB
        ):
            return True

        # -----------------------------------------------------
        # AUTO MODE
        # -----------------------------------------------------

        if (
            self.web_mode_controller.mode
            == WebMode.AUTO
        ):

            decider = (
                self.web_auto_decider
            )

            if decider is not None:

                # =================================================
                # CURRENT IMPLEMENTATION
                # =================================================

                if hasattr(
                    decider,
                    "should_use_web",
                ):

                    try:

                        return bool(
                            decider.should_use_web(
                                clean_query
                            )
                        )

                    except (
                        RuntimeError,
                        ValueError,
                        TypeError,
                    ) as error:

                        if self.debug_web:

                            print(
                                "\n"
                                "[WEB AUTO DECISION FAILED]"
                            )

                            print(
                                f"Reason: {error}"
                            )

                # =================================================
                # BACKWARD COMPATIBILITY
                # =================================================

                if hasattr(
                    decider,
                    "should_search",
                ):

                    try:

                        return bool(
                            decider.should_search(
                                clean_query
                            )
                        )

                    except (
                        RuntimeError,
                        ValueError,
                        TypeError,
                    ) as error:

                        if self.debug_web:

                            print(
                                "\n"
                                "[WEB AUTO DECISION FAILED]"
                            )

                            print(
                                f"Reason: {error}"
                            )

                # =================================================
                # decide()
                # =================================================

                if hasattr(
                    decider,
                    "decide",
                ):

                    try:

                        decision = (
                            decider.decide(
                                clean_query
                            )
                        )

                        if isinstance(
                            decision,
                            WebMode,
                        ):

                            return (
                                decision
                                == WebMode.WEB
                            )

                    except (
                        RuntimeError,
                        ValueError,
                        TypeError,
                    ) as error:

                        if self.debug_web:

                            print(
                                "\n"
                                "[WEB AUTO DECISION FAILED]"
                            )

                            print(
                                f"Reason: {error}"
                            )

            # -----------------------------------------------------
            # SAFE FALLBACK
            # -----------------------------------------------------

            lower = (
                clean_query.lower()
            )

            indicators = (
                "today",
                "latest",
                "current",
                "currently",
                "now",
                "right now",
                "recent",
                "recently",
                "news",
                "breaking",
                "live",
                "update",
                "updates",
                "price",
                "market",
                "weather",
                "forecast",
                "gold",
                "silver",
                "oil",
                "bitcoin",
                "ethereum",
                "crypto",
                "stock",
                "forex",
                "xauusd",
                "xau/usd",
                "nfp",
                "fomc",
                "cpi",
                "ppi",
                "pce",
                "economic calendar",
                "exchange rate",
            )

            return any(
                indicator in lower
                for indicator in indicators
            )

        return False

    # =========================================================
    # UNWRAP SEARCH URL
    # =========================================================

    @staticmethod
    def unwrap_search_url(
        url: str,
    ) -> str:

        if not isinstance(url, str):
            return ""

        url = url.strip()

        if not url:
            return ""

        try:

            parsed = urlparse(
                url
            )

            hostname = (
                parsed.hostname
                or ""
            ).lower()

            # -------------------------------------------------
            # DuckDuckGo redirect
            # -------------------------------------------------

            if (
                "duckduckgo.com"
                in hostname
                and parsed.path.startswith(
                    "/l/"
                )
            ):

                query_values = (
                    parse_qs(
                        parsed.query
                    )
                )

                target_values = (
                    query_values.get(
                        "uddg"
                    )
                )

                if target_values:

                    target = (
                        target_values[0]
                        .strip()
                    )

                    target = unquote(
                        target
                    )

                    target_parsed = (
                        urlparse(
                            target
                        )
                    )

                    if (
                        target_parsed.scheme
                        in {
                            "http",
                            "https",
                        }
                        and target_parsed.netloc
                    ):
                        return target

        except Exception:

            pass

        return url

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

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        # -----------------------------------------------------
        # OFFLINE
        # -----------------------------------------------------

        if (
            self.web_mode_controller.mode
            == WebMode.OFFLINE
        ):
            return []

        # -----------------------------------------------------
        # WEB SEARCH CLIENT
        # -----------------------------------------------------

        if self.web_search is None:

            if self.debug_web:

                print(
                    "\n[WEB SEARCH UNAVAILABLE]"
                )

            return []

        # -----------------------------------------------------
        # SEARCH
        # -----------------------------------------------------

        try:

            results = (
                self.web_search.search(
                    query=query,
                    limit=limit,
                )
            )

        except (
            RuntimeError,
            ValueError,
        ) as error:

            if self.debug_web:

                print(
                    "\n[WEB SEARCH FAILED]"
                )

                print(
                    f"Reason: {error}"
                )

            return []

        if not results:
            return []

        # -----------------------------------------------------
        # VALIDATE
        # -----------------------------------------------------

        try:

            results = (
                self.web_result_validator
                .validate_many(
                    results
                )
            )

        except (
            RuntimeError,
            ValueError,
            TypeError,
        ) as error:

            if self.debug_web:

                print(
                    "\n[WEB RESULT VALIDATION FAILED]"
                )

                print(
                    f"Reason: {error}"
                )

            return []

        # -----------------------------------------------------
        # UNWRAP DUCKDUCKGO URLS
        # -----------------------------------------------------

        cleaned_results: list[
            SearchResult
        ] = []

        for result in results:

            if not isinstance(
                result,
                SearchResult,
            ):
                continue

            real_url = (
                self.unwrap_search_url(
                    result.url
                )
            )

            cleaned_result = (
                SearchResult(
                    title=(
                        result.title
                        or ""
                    ).strip(),
                    url=real_url,
                    snippet=(
                        result.snippet
                        or ""
                    ).strip(),
                )
            )

            if not cleaned_result.title:
                continue

            if not cleaned_result.url:
                continue

            cleaned_results.append(
                cleaned_result
            )

        return cleaned_results[:limit]

    # =========================================================
    # FETCH ONE WEB RESULT
    # =========================================================

    def _fetch_result_text(
        self,
        result: SearchResult,
    ) -> str:

        if self.web_page_fetcher is None:
            return ""

        if self.web_text_extractor is None:
            return ""

        if not result.url:
            return ""

        try:

            html = (
                self.web_page_fetcher.fetch(
                    result.url
                )
            )

            if not html:
                return ""

            text = (
                self.web_text_extractor.extract(
                    html
                )
            )

            if not text:
                return ""

            # -------------------------------------------------
            # SECURITY
            # -------------------------------------------------

            if (
                self.web_content_security
                is not None
            ):

                try:

                    safe_text = (
                        self.web_content_security
                        .build_safe_context(
                            text
                        )
                    )

                    if safe_text:
                        return safe_text

                except (
                    RuntimeError,
                    ValueError,
                    TypeError,
                ) as error:

                    if self.debug_web:

                        print(
                            "\n"
                            "[WEB SECURITY FILTER FAILED]"
                        )

                        print(
                            f"Reason: {error}"
                        )

                    return ""

            return text

        except (
            RuntimeError,
            ValueError,
            TypeError,
        ) as error:

            if self.debug_web:

                print(
                    "\n[WEB FETCH FAILED]"
                )

                print(
                    f"URL: {result.url}"
                )

                print(
                    f"Reason: {error}"
                )

            return ""

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

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        # -----------------------------------------------------
        # SHOULD USE WEB
        # -----------------------------------------------------

        if not self.should_use_web(
            query
        ):
            return ""

        # -----------------------------------------------------
        # REMOVE web: PREFIX
        # -----------------------------------------------------

        clean_query, _ = (
            self._extract_web_query(
                query
            )
        )

        # -----------------------------------------------------
        # SEARCH
        # -----------------------------------------------------

        results = self.web_retrieve(
            query=clean_query,
            limit=limit,
        )

        if not results:
            return ""

        # -----------------------------------------------------
        # BUILD RESEARCH
        # -----------------------------------------------------

        research_sections: list[str] = []

        for index, result in enumerate(
            results,
            start=1,
        ):

            # =================================================
            # START WITH SEARCH RESULT
            # =================================================

            parts: list[str] = []

            parts.append(
                f"WEB SOURCE {index}"
            )

            parts.append(
                f"Title: {result.title}"
            )

            parts.append(
                f"URL: {result.url}"
            )

            if result.snippet:

                parts.append(
                    "Search snippet:"
                )

                parts.append(
                    result.snippet
                )

            # =================================================
            # TRY ARTICLE FETCH
            # =================================================

            article_text = (
                self._fetch_result_text(
                    result
                )
            )

            # =================================================
            # ONLY ADD ARTICLE CONTENT IF
            # ACTUALLY EXTRACTED
            # =================================================

            if article_text:

                # Keep web context bounded.
                #
                # This prevents a huge webpage from
                # overwhelming the 1.7B local model.

                max_chars = 5000

                article_text = (
                    article_text[
                        :max_chars
                    ]
                )

                parts.append(
                    "Article content:"
                )

                parts.append(
                    article_text
                )

            # =================================================
            # SAVE SOURCE
            # =================================================

            research_sections.append(
                "\n".join(
                    parts
                )
            )

        # -----------------------------------------------------
        # FINAL RESULT
        # -----------------------------------------------------

        if not research_sections:
            return ""

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

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        # =====================================================
        # RETRIEVE LOCAL KNOWLEDGE
        # =====================================================

        retrieval_result = (
            self.retrieve(
                query=query,
                limit=limit,
            )
        )

        context_parts: list[str] = []

        if retrieval_result.context:

            context_parts.append(
                "LOCAL KNOWLEDGE\n"
                "---------------\n"
                f"{retrieval_result.context}"
            )

        # =====================================================
        # CONVERSATION MEMORY
        # =====================================================

        try:

            conversation_context = (
                self.conversation_memory
                .build_context(
                    query=query,
                    limit=5,
                )
            )

            if conversation_context:

                context_parts.append(
                    "CONVERSATION MEMORY\n"
                    "-------------------\n"
                    f"{conversation_context}"
                )

        except (
            RuntimeError,
            ValueError,
            TypeError,
        ):

            pass

        # =====================================================
        # LONG-TERM MEMORY
        # =====================================================

        try:

            memory_context = (
                self.build_memory_context(
                    query=query,
                    limit=5,
                )
            )

            if memory_context:

                context_parts.append(
                    "LONG-TERM MEMORY\n"
                    "----------------\n"
                    f"{memory_context}"
                )

        except (
            RuntimeError,
            ValueError,
            TypeError,
        ):

            pass

        # =====================================================
        # WEB RESEARCH
        # =====================================================

        web_context = (
            self.web_research(
                query=query,
                limit=limit,
            )
        )

        if web_context:

            context_parts.append(
                web_context
            )

        # =====================================================
        # COMBINE
        # =====================================================

        combined_context = (
            "\n\n".join(
                context_parts
            )
        )

        # =====================================================
        # PROMPT
        # =====================================================

        return self.prompt_builder.build(
            query=retrieval_result.query,
            context=combined_context,
        )

    # =========================================================
    # ASK
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

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        # =====================================================
        # BUILD PROMPT
        # =====================================================

        prompt = self.prepare(
            query=query,
            limit=limit,
        )

        # =====================================================
        # GENERATE
        # =====================================================

        answer = self.llm.generate(
            prompt
        )

        if not isinstance(
            answer,
            str,
        ):
            raise ValueError(
                "llm returned a non-string response."
            )

        answer = answer.strip()

        if not answer:
            raise ValueError(
                "llm returned an empty response."
            )

        # =====================================================
        # SAVE CONVERSATION
        # =====================================================

        try:

            self.conversation_memory.save_user_message(
                query
            )

            self.conversation_memory.save_assistant_message(
                answer
            )

        except (
            RuntimeError,
            ValueError,
            TypeError,
        ):

            # Memory failure must not destroy
            # an otherwise valid AI response.

            pass

        return answer