# Import the abstract base class functionality.
from abc import ABC, abstractmethod


# Define the common interface for every LLM implementation.
class LLMClient(ABC):

    # Generate a response from a prompt.
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a text response from the supplied prompt.

        Implementations should return only the generated
        assistant response.
        """

        # This method must be implemented by subclasses.
        raise NotImplementedError