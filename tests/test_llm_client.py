# Import pytest for validation and abstract-class tests.
import pytest

# Import the LLM client interface.
from app.llm.client import LLMClient


# Create a fake implementation for testing.
class FakeLLMClient(LLMClient):

    # Generate a predictable response.
    def generate(self, prompt: str) -> str:

        # Return a predictable test response.
        return f"Response to: {prompt}"


# Test that a concrete implementation can be created.
def test_llm_client_implementation():

    # Create the fake client.
    client = FakeLLMClient()

    # Generate a response.
    result = client.generate("Hello")

    # Verify the response.
    assert result == "Response to: Hello"


# Test that the abstract interface cannot be
# instantiated directly.
def test_llm_client_is_abstract():

    # The base class should not be directly instantiable.
    with pytest.raises(TypeError):
        LLMClient()


# Test that the generate method exists on the interface.
def test_llm_client_has_generate_method():

    # Verify that the interface exposes generate().
    assert hasattr(
        LLMClient,
        "generate",
    )


# Test that a concrete implementation must
# implement generate().
def test_generate_must_be_implemented():

    # Create an incomplete implementation.
    class IncompleteLLMClient(LLMClient):
        pass

    # Instantiating it should fail because generate()
    # is abstract.
    with pytest.raises(TypeError):
        IncompleteLLMClient()