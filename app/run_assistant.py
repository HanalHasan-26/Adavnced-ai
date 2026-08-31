from __future__ import annotations

from config.settings import (
    APP_NAME,
    MODEL_NAME,
    DEFAULT_RETRIEVAL_LIMIT,
)

from app.core.startup import startup
from app.core.lifecycle import shutdown
from app.core.errors import handle_error
from app.core.logger import logger

from app.knowledge.storage import KnowledgeStorage
from app.knowledge.retrieval.chunk_retriever import ChunkRetriever
from app.knowledge.retrieval.pipeline import KnowledgeRetrievalPipeline
from app.knowledge.assistant import KnowledgeAssistant

from app.llm.local import LocalLLMClient
from app.llm.ollama import OllamaBackend


# =========================================================
# CREATE ASSISTANT
# =========================================================

def create_assistant() -> KnowledgeAssistant:

    # Create persistent knowledge storage.
    storage = KnowledgeStorage()

    # Create the knowledge retriever.
    retriever = ChunkRetriever(
        storage=storage,
    )

    # Create the retrieval pipeline.
    pipeline = KnowledgeRetrievalPipeline(
        retriever=retriever,
    )

    # Create the Ollama backend.
    backend = OllamaBackend(
        model=MODEL_NAME,
    )

    # Create the local LLM client.
    llm = LocalLLMClient(
        backend=backend,
    )

    # Create the complete assistant.
    assistant = KnowledgeAssistant(
        retrieval_pipeline=pipeline,
        llm=llm,
    )

    return assistant


# =========================================================
# MAIN APPLICATION
# =========================================================

def main() -> None:

    assistant: KnowledgeAssistant | None = None

    try:

        # -------------------------------------------------
        # STARTUP
        # -------------------------------------------------

        startup()

        print()
        print("==============================")
        print(f"        {APP_NAME}")
        print("==============================")
        print()

        logger.info(
            "Creating AI assistant."
        )

        print(
            "Initializing AI assistant..."
        )

        assistant = create_assistant()

        print(
            "✓ Knowledge storage ready."
        )

        print(
            "✓ Retrieval system ready."
        )

        print(
            "✓ Local LLM ready."
        )

        logger.info(
            "AI assistant initialized successfully."
        )

        print()
        print(
            "================================"
        )
        print(
            "       Local AI Assistant"
        )
        print(
            "================================"
        )

        print(
            f"Model: {MODEL_NAME}"
        )

        print(
            "Type 'exit' or 'quit' to stop."
        )

        print()

        # -------------------------------------------------
        # CONVERSATION LOOP
        # -------------------------------------------------

        while True:

            try:

                query = input(
                    "You: "
                ).strip()

            except EOFError:

                print()

                logger.info(
                    "Input stream closed."
                )

                break

            # -------------------------------------------------
            # EXIT
            # -------------------------------------------------

            if query.lower() in {
                "exit",
                "quit",
            }:

                print()

                print(
                    "Shutting down..."
                )

                logger.info(
                    "Shutdown requested by user."
                )

                break

            # Ignore empty input.
            if not query:
                continue

            # -------------------------------------------------
            # ASK AI
            # -------------------------------------------------

            try:

                answer = assistant.ask(
                    query=query,
                    limit=DEFAULT_RETRIEVAL_LIMIT,
                )

                print()
                print("AI:")
                print(answer)
                print()

            except Exception as error:

                handle_error(error)

                print()

    # ---------------------------------------------------------
    # CTRL+C
    # ---------------------------------------------------------

    except KeyboardInterrupt:

        print()
        print()

        print(
            "Shutdown requested."
        )

        logger.info(
            "Application interrupted by user."
        )

    # ---------------------------------------------------------
    # UNEXPECTED ERROR
    # ---------------------------------------------------------

    except Exception as error:

        print()

        print(
            "Application failed."
        )

        handle_error(error)

    # ---------------------------------------------------------
    # SHUTDOWN
    # ---------------------------------------------------------

    finally:

        try:

            shutdown()

            print()

            print(
                "✓ Application stopped."
            )

            print(
                "Goodbye."
            )

        except Exception as error:

            handle_error(error)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()