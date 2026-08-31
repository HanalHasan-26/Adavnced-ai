from __future__ import annotations


class ConversationMemory:
    """
    Manages conversation messages on top of the project's
    persistent Memory storage.

    Supports:

        ConversationMemory()

    and:

        ConversationMemory(memory=FakeMemory())
    """

    def __init__(
        self,
        memory=None,
    ):
        # Use the real persistent Memory implementation
        # when no memory object is supplied.
        if memory is None:
            from app.memory.memory import Memory

            memory = Memory()

        self.memory = memory

    # ---------------------------------------------------------
    # SAVE USER MESSAGE
    # ---------------------------------------------------------

    def save_user_message(
        self,
        message: str,
    ) -> str:

        if not isinstance(message, str):
            raise ValueError(
                "message must be a string."
            )

        message = message.strip()

        if not message:
            raise ValueError(
                "message cannot be empty."
            )

        return self.memory.add(
            f"User: {message}"
        )

    # ---------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # ---------------------------------------------------------

    def save_assistant_message(
        self,
        message: str,
    ) -> str:

        if not isinstance(message, str):
            raise ValueError(
                "message must be a string."
            )

        message = message.strip()

        if not message:
            raise ValueError(
                "message cannot be empty."
            )

        return self.memory.add(
            f"Assistant: {message}"
        )

    # ---------------------------------------------------------
    # RECALL
    # ---------------------------------------------------------

    def recall(
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
            return []

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        return self.memory.search(
            query=query,
            limit=limit,
        )

    # ---------------------------------------------------------
    # RECALL RECENT
    # ---------------------------------------------------------

    def recall_recent(
        self,
        limit: int = 5,
    ) -> list[dict]:

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        # Preferred path:
        # The real Memory class provides list().
        if hasattr(self.memory, "list"):

            memories = self.memory.list()

            if isinstance(memories, list):

                # Memory.list() returns newest first.
                return memories[:limit]

        # Test doubles may expose their records directly.
        if hasattr(self.memory, "memories"):

            memories = self.memory.memories

            if isinstance(memories, list):

                # If records are stored oldest first,
                # take the newest records.
                return memories[-limit:]

        return []

    # ---------------------------------------------------------
    # BUILD CONTEXT
    # ---------------------------------------------------------

    def build_context(
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
            return ""

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        # IMPORTANT:
        #
        # Conversation context should use recent
        # conversation history instead of searching
        # for words matching the current question.
        #
        # Example:
        #
        # User:
        #     My name is Pirlo.
        #
        # Later:
        #     What's my name?
        #
        # A normal text search for "What's my name?"
        # will NOT reliably find "My name is Pirlo."
        #
        # Therefore we use recent conversation history.

        memories = self.recall_recent(
            limit=limit,
        )

        if not memories:
            return ""

        sections: list[str] = []

        for memory in memories:

            if not isinstance(
                memory,
                dict,
            ):
                continue

            content = memory.get(
                "content",
                "",
            )

            if not isinstance(
                content,
                str,
            ):
                continue

            content = content.strip()

            if not content:
                continue

            sections.append(
                content
            )

        if not sections:
            return ""

        # recall_recent() returns newest first.
        #
        # Reverse the records so the LLM sees the
        # conversation in normal chronological order.
        sections.reverse()

        return "\n".join(
            sections
        )