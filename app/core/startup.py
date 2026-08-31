from __future__ import annotations

from app.core.logger import logger


def startup() -> None:
    """
    Start the application lifecycle.
    """

    logger.info("Application startup.")

    print()
    print("==============================")
    print("       Starting Advanced AI")
    print("==============================")
    print("Application state: STARTING")