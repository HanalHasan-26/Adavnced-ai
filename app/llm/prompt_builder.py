from __future__ import annotations


class PromptBuilder:
    """
    Builds prompts for the local language model.

    The assistant has several sources of context:

    1. Knowledge
       Information retrieved from the knowledge base.

    2. Previous conversation
       Recent messages from the current conversation.

    3. Long-term memory
       Persistent information remembered from previous conversations.

    4. Web research
       Information retrieved from the web when web mode is enabled.

    The model should use all of these sources when relevant.
    """

    def build(
        self,
        query: str,
        context: str,
    ) -> str:

        # Remove unnecessary whitespace.
        query = query.strip()
        context = context.strip()

        # Reject an empty question.
        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        return (
            "You are a helpful local AI assistant.\n"
            "\n"

            "You have access to several types of context.\n"
            "\n"

            "IMPORTANT RULES:\n"
            "1. Use the provided knowledge when it is relevant.\n"
            "2. Use previous conversation when it is relevant.\n"
            "3. Use long-term memory when it is relevant.\n"
            "4. Treat statements made by the user in previous "
            "conversation or memory as information about the user.\n"
            "5. If the user previously told you their name, "
            "you may use that information when answering questions "
            "about their name.\n"
            "6. Do not claim that you do not know something when "
            "the answer is explicitly present in the conversation "
            "or memory context.\n"
            "7. Do not confuse the user's information with facts "
            "about other people.\n"
            "8. Do not invent information that is not supported "
            "by the available context.\n"
            "\n"

            "CONTEXT:\n"
            f"{context if context else 'No additional context was retrieved.'}\n"
            "\n"

            "USER QUESTION:\n"
            f"{query}\n"
            "\n"

            "ANSWER:\n"
        )