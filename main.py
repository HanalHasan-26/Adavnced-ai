from app.core.startup import start
from config.settings import APP_NAME, APP_VERSION


def main():

    print(f"{APP_NAME} is starting...")

    print(f"Version: {APP_VERSION}")

    start()


if __name__ == "__main__":
    main()