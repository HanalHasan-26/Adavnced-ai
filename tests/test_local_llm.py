# Import pytest for validation tests.
import pytest

# Import the local LLM client.
from app.llm.local import LocalLLMClient

# Import the local backend interface.
from app.llm.local import LocalModelBackend


# Create a fake backend for testing.
class FakeBackend(LocalModelBackend):

    # Initialize the fake backend.
    def __init__(self, response="Test response"):

        # Store the response that should be returned.
        self.response = response

        # Store the received prompt.
        self.received_prompt = None

    # Generate a predictable response.
    def generate(self, prompt: str) -> str:

        # Remember the prompt.
        self.received_prompt = prompt

        # Return the configured response.
        return self.response


# Test that the local client sends the prompt
# to the backend and returns the response.
def test_local_llm_generate():

    # Create the fake backend.
    backend = FakeBackend(
        response="Hello from local AI."
    )

    # Create the local LLM client.
    client = LocalLLMClient(backend)

    # Generate a response.
    result = client.generate(
        "What is artificial intelligence?"
    )

    # Verify the response.
    assert result == "Hello from local AI."

    # Verify that the prompt reached the backend.
    assert backend.received_prompt == (
        "What is artificial intelligence?"
    )


# Test that surrounding prompt whitespace is removed.
def test_local_llm_strips_prompt():

    # Create the fake backend.
    backend = FakeBackend()

    # Create the client.
    client = LocalLLMClient(backend)

    # Generate using a padded prompt.
    client.generate(
        "   Hello local model   "
    )

    # Verify the cleaned prompt reached the backend.
    assert backend.received_prompt == "Hello local model"


# Test that the model response is cleaned.
def test_local_llm_strips_response():

    # Create a backend with padded output.
    backend = FakeBackend(
        response="   Local response   "
    )

    # Create the client.
    client = LocalLLMClient(backend)

    # Generate a response.
    result = client.generate("Hello")

    # Verify that whitespace was removed.
    assert result == "Local response"


# Test that an empty prompt is rejected.
def test_local_llm_empty_prompt():

    # Create the backend.
    backend = FakeBackend()

    # Create the client.
    client = LocalLLMClient(backend)

    # Verify empty input is rejected.
    with pytest.raises(ValueError):
        client.generate("")


# Test that a whitespace-only prompt is rejected.
def test_local_llm_whitespace_prompt():

    # Create the backend.
    backend = FakeBackend()

    # Create the client.
    client = LocalLLMClient(backend)

    # Verify whitespace-only input is rejected.
    with pytest.raises(ValueError):
        client.generate("   ")


# Test that a None backend is rejected.
def test_local_llm_none_backend():

    # Verify that None is not accepted.
    with pytest.raises(ValueError):
        LocalLLMClient(None)


# Test that a non-string backend response is rejected.
def test_local_llm_invalid_response_type():

    # Create a backend returning an invalid type.
    backend = FakeBackend()

    # Replace its response with an invalid value.
    backend.response = 123

    # Create the client.
    client = LocalLLMClient(backend)

    # Verify the invalid response is rejected.
    with pytest.raises(TypeError):
        client.generate("Hello")


# Test that an empty backend response is rejected.
def test_local_llm_empty_response():

    # Create a backend returning empty text.
    backend = FakeBackend(
        response="   "
    )

    # Create the client.
    client = LocalLLMClient(backend)

    # Verify the empty response is rejected.
    with pytest.raises(ValueError):
        client.generate("Hello")


# Test that the backend interface requires
# generate() to be implemented.
def test_local_backend_must_be_implemented():

    # Create an incomplete backend.
    class IncompleteBackend(LocalModelBackend):
        pass

    # The base backend itself isn't abstract yet,
    # so calling generate() should raise the intended error.
    backend = IncompleteBackend()

    with pytest.raises(NotImplementedError):
        backend.generate("Hello")