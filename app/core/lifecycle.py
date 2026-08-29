# Import our application logger so lifecycle events are recorded.
from app.core.logger import logger


# Define the function that handles application startup.
def startup() -> None:

    # Record that the application is entering the startup stage.
    logger.info("Application startup.")

    # Display the current lifecycle state in the terminal.
    print("Application state: STARTING")


# Define the function that handles application shutdown.
def shutdown() -> None:

    # Record that the application is shutting down.
    logger.info("Application shutdown.")

    # Display the current lifecycle state in the terminal.
    print("Application state: SHUTTING DOWN")