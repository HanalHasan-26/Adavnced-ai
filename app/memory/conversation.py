from __future__ import annotations


class ConversationMemory:
    """
    Manages conversation messages on top of the project's
    persistent Memory storage.

    Supports:

        ConversationMemory()

    and:

        ConversationMemory(memory=FakeMemory())

    Conversation context is chronological and is NOT selected
    by searching for words from the current question.

    This is important because a previous message such as:

        User: My name is Pirlo.

    does not contain the words:

        What is my name?

    Therefore conversation history must be retrieved as
    conversation history rather than keyword search.
    """

    def __init__(
        self,
        memory=None,
    ):
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

        # -----------------------------------------------------
        # REAL MEMORY
        # -----------------------------------------------------

        if hasattr(self.memory, "list"):

            memories = self.memory.list()

            if isinstance(memories, list):

                # Memory.list() returns newest first.
                #
                # Keep that behavior for recall_recent().
                return memories[:limit]

        # -----------------------------------------------------
        # TEST / FAKE MEMORY
        # -----------------------------------------------------

        if hasattr(self.memory, "memories"):

            memories = self.memory.memories

            if isinstance(memories, list):

                # FakeMemory normally stores records in
                # chronological insertion order.
                #
                # Return newest first to match the public
                # recall_recent() contract.
                return list(
                    reversed(memories)
                )[:limit]

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

        # -----------------------------------------------------
        # IMPORTANT DESIGN DECISION
        # -----------------------------------------------------
        #
        # Conversation context must NOT use:
        #
        #     memory.search(query)
        #
        # because the current question usually has different
        # words from the previous statement.
        #
        # Example:
        #
        # Previous:
        #     User: My name is Pirlo.
        #
        # Current:
        #     What is my name?
        #
        # Searching for "What is my name?" cannot find
        # "My name is Pirlo."
        #
        # Instead, conversation context is retrieved from the
        # stored conversation history.
        # -----------------------------------------------------

        memories: list[dict] = []

        # -----------------------------------------------------
        # FAKE MEMORY / TEST MEMORY
        # -----------------------------------------------------

        if hasattr(self.memory, "memories"):

            stored = self.memory.memories

            if isinstance(stored, list):

                for memory in stored:

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

                    if not content.strip():
                        continue

                    memories.append(
                        memory
                    )

        # -----------------------------------------------------
        # REAL MEMORY
        # -----------------------------------------------------

        else:

            if hasattr(self.memory, "list"):

                stored = self.memory.list()

                if isinstance(stored, list):

                    for memory in stored:

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

                        if not content.strip():
                            continue

                        memories.append(
                            memory
                        )

        # Nothing stored.
        if not memories:
            return ""

        # -----------------------------------------------------
        # NORMALIZE ORDER
        # -----------------------------------------------------
        #
        # FakeMemory:
        #     oldest -> newest
        #
        # Real Memory.list():
        #     newest -> oldest
        #
        # We use created_at when available so both cases
        # become chronological.
        # -----------------------------------------------------

        def created_at_value(
            memory: dict,
        ) -> str:

            value = memory.get(
                "created_at",
                "",
            )

            if isinstance(
                value,
                str,
            ):
                return value

            return ""

        memories.sort(
            key=created_at_value
        )

        # -----------------------------------------------------
        # LIMIT
        # -----------------------------------------------------
        #
        # The project tests expect the context limit to keep
        # the earliest records when the stored conversation is
        # viewed chronologically.
        #
        # Example:
        #
        # Message one
        # Message two
        # Message three
        #
        # limit=2
        #
        # Result:
        #
        # Message one
        # Message two
        # -----------------------------------------------------

        memories = memories[:limit]

        # -----------------------------------------------------
        # FORMAT
        # -----------------------------------------------------

        sections: list[str] = []

        for memory in memories:

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

        return "\n".join(
            sections
        )