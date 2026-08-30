# Import the LLM client interface.
from app.llm.client import LLMClient

# Import the local model backend interface.
from app.llm.local_backend import LocalModelBackend


# Create an LLM client that communicates with a
# local model backend.
class LocalLLMClient(LLMClient):

    # Initialize the local LLM client.
    def __init__(self, backend: LocalModelBackend):

        # Make sure a backend was supplied.
        if backend is None:
            raise ValueError(
                "backend cannot be None."
            )

        # Store the local model backend.
        self.backend = backend

    # Generate a response using the local model.
    def generate(self, prompt: str) -> str:

        # Remove unnecessary whitespace from the prompt.
        prompt = prompt.strip()

        # Reject an empty prompt.
        if not prompt:
            raise ValueError(
                "prompt cannot be empty."
            )

        # Send the prompt to the local model.
        response = self.backend.generate(prompt)

        # Make sure the backend returned text.
        if not isinstance(response, str):
            raise TypeError(
                "LLM backend must return a string."
            )

        # Remove unnecessary whitespace from the response.
        response = response.strip()

        # Reject an empty model response.
        if not response:
            raise ValueError(
                "LLM backend returned an empty response."
            )

        # Return the cleaned response.
        return response