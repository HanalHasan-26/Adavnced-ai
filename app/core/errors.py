from __future__ import annotations

from app.core.logger import logger


def handle_error(error: Exception) -> None:
    """
    Handle an application error.

    The error is logged and a readable message
    is displayed in the terminal.
    """

    logger.exception(
        "Application error: %s",
        error,
    )

    print()
    print(f"An error occurred: {error}")
    print()