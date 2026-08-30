# Create a small interface for the local model backend.
class LocalModelBackend:

    # Generate text using the local model.
    def generate(self, prompt: str) -> str:

        # This method must be implemented by the
        # actual local model integration.
        raise NotImplementedError