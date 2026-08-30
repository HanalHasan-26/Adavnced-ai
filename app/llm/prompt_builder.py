# Create a component responsible for building
# prompts for the language model.
class PromptBuilder:

    # Build an answer-generation prompt.
    def build(
        self,
        query: str,
        context: str,
    ) -> str:

        # Remove unnecessary whitespace from the query.
        query = query.strip()

        # Remove unnecessary whitespace from the context.
        context = context.strip()

        # Reject an empty question.
        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        # Build a prompt when relevant knowledge exists.
        if context:

            return (
                "You are a helpful local AI assistant.\n"
                "\n"
                "Answer the user's question using the "
                "provided knowledge when relevant.\n"
                "Do not invent facts that are not supported "
                "by the provided knowledge.\n"
                "\n"
                "Knowledge:\n"
                f"{context}\n"
                "\n"
                "User question:\n"
                f"{query}\n"
                "\n"
                "Answer:"
            )

        # Build a prompt when no knowledge was retrieved.
        return (
    "You are a helpful local AI assistant.\n"
    "\n"
    "No relevant knowledge was retrieved for this question.\n"
    "Answer the user's question as accurately as "
    "possible.\n"
    "If you do not know the answer, clearly say so "
    "instead of inventing information.\n"
    "\n"
    "User question:\n"
    f"{query}\n"
    "\n"
    "Answer:"
        )