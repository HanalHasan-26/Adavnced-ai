from app.core.startup import start
from config.settings import APP_NAME, APP_VERSION
from app.core.lifecycle import startup, shutdown



def main():

    print(f"{APP_NAME} is starting...")

    print(f"Version: {APP_VERSION}")

    startup()

    start()

    shutdown()


if __name__ == "__main__":
    main()