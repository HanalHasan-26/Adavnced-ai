from app.core.logger import logger

def handle_error(error: Exception) -> None:

    logger.error("Application error: %s", error)
    print(f"An error occurred: {error}")

