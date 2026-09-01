from __future__ import annotations

import sys
import time
import threading

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
# LOADING UI
# =========================================================

def show_loading(
    step: int,
    total_steps: int,
    message: str,
    elapsed: float,
) -> None:
    """
    Display startup progress.
    """

    bar_width = 20

    progress = step / total_steps

    filled = int(
        bar_width * progress
    )

    empty = bar_width - filled

    bar = (
        "/" * filled
        + "." * empty
    )

    percentage = int(
        progress * 100
    )

    print(
        f"[{bar}] "
        f"{percentage:3d}% "
        f"{message:<32} "
        f"{elapsed:.2f}s"
    )


# =========================================================
# AI THINKING / GENERATION LOADING UI
# =========================================================

class ThinkingAnimation:
    """
    Displays a live animation while the AI is generating
    a response.
    """

    def __init__(self) -> None:

        self.running = False

        self.thread: threading.Thread | None = None

        self.frames = [
            "/",
            "-",
            "\\",
            "|",
        ]

    def _animate(self) -> None:

        frame_index = 0

        start_time = time.perf_counter()

        while self.running:

            elapsed = (
                time.perf_counter()
                - start_time
            )

            frame = self.frames[
                frame_index % len(self.frames)
            ]

            message = (
                f"\rAI is thinking... "
                f"{frame} "
                f"{elapsed:.1f}s"
            )

            sys.stdout.write(
                message
            )

            sys.stdout.flush()

            frame_index += 1

            time.sleep(0.15)

    def start(self) -> None:

        if self.running:
            return

        self.running = True

        self.thread = threading.Thread(
            target=self._animate,
            daemon=True,
        )

        self.thread.start()

    def stop(self) -> None:

        self.running = False

        if self.thread is not None:

            self.thread.join(
                timeout=1.0
            )

        # Clear the current thinking line.
        sys.stdout.write(
            "\r"
            + (" " * 60)
            + "\r"
        )

        sys.stdout.flush()

        self.thread = None


# =========================================================
# CREATE ASSISTANT
# =========================================================

def create_assistant() -> KnowledgeAssistant:

    # -----------------------------------------------------
    # STEP 1 - KNOWLEDGE STORAGE
    # -----------------------------------------------------

    step_start = time.perf_counter()

    storage = KnowledgeStorage()

    elapsed = (
        time.perf_counter()
        - step_start
    )

    show_loading(
        step=1,
        total_steps=5,
        message="Loading knowledge storage...",
        elapsed=elapsed,
    )

    # -----------------------------------------------------
    # STEP 2 - RETRIEVAL SYSTEM
    # -----------------------------------------------------

    step_start = time.perf_counter()

    retriever = ChunkRetriever(
        storage=storage,
    )

    elapsed = (
        time.perf_counter()
        - step_start
    )

    show_loading(
        step=2,
        total_steps=5,
        message="Loading retrieval system...",
        elapsed=elapsed,
    )

    # -----------------------------------------------------
    # STEP 3 - KNOWLEDGE PIPELINE
    # -----------------------------------------------------

    step_start = time.perf_counter()

    pipeline = KnowledgeRetrievalPipeline(
        retriever=retriever,
    )

    elapsed = (
        time.perf_counter()
        - step_start
    )

    show_loading(
        step=3,
        total_steps=5,
        message="Loading knowledge pipeline...",
        elapsed=elapsed,
    )

    # -----------------------------------------------------
    # STEP 4 - LOCAL LLM
    # -----------------------------------------------------

    step_start = time.perf_counter()

    backend = OllamaBackend(
        model=MODEL_NAME,
    )

    llm = LocalLLMClient(
        backend=backend,
    )

    elapsed = (
        time.perf_counter()
        - step_start
    )

    show_loading(
        step=4,
        total_steps=5,
        message="Loading local LLM...",
        elapsed=elapsed,
    )

    # -----------------------------------------------------
    # STEP 5 - ASSISTANT
    # -----------------------------------------------------

    step_start = time.perf_counter()

    assistant = KnowledgeAssistant(
        retrieval_pipeline=pipeline,
        llm=llm,
    )

    elapsed = (
        time.perf_counter()
        - step_start
    )

    show_loading(
        step=5,
        total_steps=5,
        message="Initializing AI assistant...",
        elapsed=elapsed,
    )

    return assistant


# =========================================================
# ASK AI WITH THINKING ANIMATION
# =========================================================

def ask_ai(
    assistant: KnowledgeAssistant,
    query: str,
) -> str:
    """
    Generate an AI response while displaying a live
    thinking animation.
    """

    thinking = ThinkingAnimation()

    try:

        thinking.start()

        answer = assistant.ask(
            query=query,
            limit=DEFAULT_RETRIEVAL_LIMIT,
        )

        return answer

    finally:

        thinking.stop()


# =========================================================
# MAIN APPLICATION
# =========================================================

def main() -> None:

    assistant: KnowledgeAssistant | None = None

    total_start = time.perf_counter()

    try:

        # -------------------------------------------------
        # STARTUP
        # -------------------------------------------------

        startup()

        print()

        print(
            "=============================="
        )

        print(
            f"        {APP_NAME}"
        )

        print(
            "=============================="
        )

        print()

        logger.info(
            "Creating AI assistant."
        )

        print(
            "Initializing AI assistant..."
        )

        print()

        # -------------------------------------------------
        # CREATE ASSISTANT
        # -------------------------------------------------

        assistant = create_assistant()

        # -------------------------------------------------
        # STARTUP COMPLETE
        # -------------------------------------------------

        total_elapsed = (
            time.perf_counter()
            - total_start
        )

        print()

        print(
            "================================"
        )

        print(
            "       Initialization Complete"
        )

        print(
            "================================"
        )

        print(
            f"Total startup time: "
            f"{total_elapsed:.2f}s"
        )

        print()

        logger.info(
            "AI assistant initialized successfully "
            f"in {total_elapsed:.2f}s."
        )

        # -------------------------------------------------
        # ASSISTANT UI
        # -------------------------------------------------

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

            # -------------------------------------------------
            # IGNORE EMPTY INPUT
            # -------------------------------------------------

            if not query:
                continue

            # -------------------------------------------------
            # ASK AI
            # -------------------------------------------------

            try:

                answer = ask_ai(
                    assistant=assistant,
                    query=query,
                )

                print(
                    "AI:"
                )

                print(
                    answer
                )

                print()

            except Exception as error:

                handle_error(
                    error
                )

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

        handle_error(
            error
        )

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

            handle_error(
                error
            )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()