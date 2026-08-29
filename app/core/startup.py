from app.core.logger import logger
from app.core.errors import handle_error




def start():

    try:

        logger.info("Core system initialized.")
        print("Core system initialized.")

    except Exception as error:

        handle_error(error)