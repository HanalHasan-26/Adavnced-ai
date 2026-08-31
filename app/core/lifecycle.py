from __future__ import annotations

from app.core.logger import logger


def shutdown() -> None:
    """
    Shut down the application cleanly.
    """

    logger.info("Application shutdown.")

    print()
    print("==============================")
    print("       Shutting Down")
    print("==============================")
    print("Application state: SHUTTING DOWN")
    print("Application state: STOPPED")