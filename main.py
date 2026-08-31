from app.core.startup import startup
from app.core.lifecycle import shutdown
from config.settings import APP_NAME, APP_VERSION


def main() -> None:

    print(f"{APP_NAME} is starting...")
    print(f"Version: {APP_VERSION}")

    startup()

    shutdown()


if __name__ == "__main__":
    main()