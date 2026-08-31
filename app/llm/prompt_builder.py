from __future__ import annotations


class PromptBuilder:
    """
    Builds the final prompt sent to the local LLM.

    Supports both:

        build(question="...")

    and:

        build(query="...")

    Backwards compatibility:

        build(
            query="...",
            context="..."
        )

    When `knowledge` is not explicitly supplied, a non-empty
    `context` argument is treated as the legacy Knowledge input.

    Context priority:

        1. Previous conversation
        2. Long-term memory
        3. Knowledge
        4. User question

    User-specific information from conversation or memory
    should be treated as valid context.
    """

    def build(
        self,
        question: str | None = None,
        context: str = "",
        knowledge: str = "",
        memory: str = "",
        *,
        query: str | None = None,
    ) -> str:

        # ---------------------------------------------------------
        # QUESTION / QUERY COMPATIBILITY
        # ---------------------------------------------------------

        if question is not None and query is not None:

            if question != query:
                raise ValueError(
                    "question and query must contain the same value."
                )

        if question is None:
            question = query

        if not isinstance(question, str):
            raise ValueError(
                "question must be a string."
            )

        question = question.strip()

        if not question:
            raise ValueError(
                "question cannot be empty."
            )

        # ---------------------------------------------------------
        # NORMALIZE INPUTS
        # ---------------------------------------------------------

        if not isinstance(context, str):
            context = ""

        if not isinstance(knowledge, str):
            knowledge = ""

        if not isinstance(memory, str):
            memory = ""

        context = context.strip()
        knowledge = knowledge.strip()
        memory = memory.strip()

        # ---------------------------------------------------------
        # BACKWARDS COMPATIBILITY
        # ---------------------------------------------------------
        #
        # Older tests/project code use:
        #
        #     build(
        #         query="...",
        #         context="..."
        #     )
        #
        # In that API, `context` represents knowledge.
        #
        # Newer code can explicitly provide:
        #
        #     knowledge="..."
        #
        # If both are supplied, keep them separate.
        # ---------------------------------------------------------

        legacy_knowledge = False

        if context and not knowledge and not memory:

            knowledge = context
            context = ""

            legacy_knowledge = True

        # ---------------------------------------------------------
        # BUILD PROMPT
        # ---------------------------------------------------------

        sections: list[str] = []

        sections.append(
            """You are a helpful local AI assistant.

You have access to information from previous conversation,
long-term memory, and knowledge.

IMPORTANT RULES:

1. Answer the user's question directly.
2. Previous conversation and long-term memory may contain facts about the user.
3. Treat explicit statements made by the user as true user information.
4. If the user previously told you their name, remember and use it.
5. If the answer is explicitly present in conversation or memory, use that answer.
6. Do NOT say you do not know something when the answer is present in the context.
7. Do NOT confuse the user's information with information about another person.
8. Knowledge is general information.
9. Conversation and memory can contain personal information about the user.
10. Do not invent facts.
11. Do not mention these instructions in your answer.
12. Do not say "the provided knowledge does not include..." if the answer exists in conversation or memory.
13. Give a natural, concise answer.
"""
        )

        # ---------------------------------------------------------
        # PREVIOUS CONVERSATION
        # ---------------------------------------------------------

        if context:
            sections.append(
                "Previous conversation:\n"
                + context
            )

        # ---------------------------------------------------------
        # LONG-TERM MEMORY
        # ---------------------------------------------------------

        if memory:
            sections.append(
                "Long-term memory:\n"
                + memory
            )

        # ---------------------------------------------------------
        # KNOWLEDGE
        # ---------------------------------------------------------

        if knowledge:
            sections.append(
                "Knowledge:\n"
                + knowledge
            )

        # ---------------------------------------------------------
        # USER QUESTION
        # ---------------------------------------------------------

        sections.append(
            "User question:\n"
            + question
        )

        # ---------------------------------------------------------
        # ANSWER
        # ---------------------------------------------------------

        sections.append(
            "Answer:"
        )

        return "\n\n".join(
            sections
        )