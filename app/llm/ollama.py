# Import JSON so we can decode Ollama responses.
import json

# Import urllib for HTTP communication without
# requiring a third-party Python package.
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Import the local model backend interface.
from app.llm.local_backend import LocalModelBackend


# Create a backend that communicates with Ollama.
class OllamaBackend(LocalModelBackend):

    # Initialize the Ollama backend.
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):

        # Remove unnecessary whitespace from the model name.
        model = model.strip()

        # Reject an empty model name.
        if not model:
            raise ValueError(
                "model cannot be empty."
            )

        # Remove trailing slashes from the URL.
        base_url = base_url.rstrip("/")

        # Reject an empty URL.
        if not base_url:
            raise ValueError(
                "base_url cannot be empty."
            )

        # Make sure the timeout is positive.
        if timeout <= 0:
            raise ValueError(
                "timeout must be greater than 0."
            )

        # Store the model name.
        self.model = model

        # Store the Ollama base URL.
        self.base_url = base_url

        # Store the request timeout.
        self.timeout = timeout

    # Generate text using Ollama.
    def generate(
        self,
        prompt: str,
    ) -> str:

        # Remove unnecessary whitespace.
        prompt = prompt.strip()

        # Reject an empty prompt.
        if not prompt:
            raise ValueError(
                "prompt cannot be empty."
            )

        # Prepare the request payload.
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }
        ).encode("utf-8")

        # Build the HTTP request.
        request = Request(
            url=f"{self.base_url}/api/generate",
            data=payload,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        # Send the request to Ollama.
        try:

            with urlopen(
                request,
                timeout=self.timeout,
            ) as response:

                # Read and decode the response.
                raw_response = (
                    response
                    .read()
                    .decode("utf-8")
                )

        # IMPORTANT:
        # HTTPError is a subclass of URLError,
        # so HTTPError must be handled first.
        except HTTPError as error:

            # Try to read Ollama's error response.
            try:

                error_body = (
                    error
                    .read()
                    .decode("utf-8")
                    .strip()
                )

            except Exception:

                error_body = ""

            # Include Ollama's message when available.
            if error_body:

                raise RuntimeError(
                    f"Ollama request failed "
                    f"({error.code}): {error_body}"
                ) from error

            # Otherwise report the HTTP status.
            raise RuntimeError(
                f"Ollama request failed "
                f"with HTTP {error.code}."
            ) from error

        # Handle connection failures.
        except URLError as error:

            raise RuntimeError(
                f"Unable to connect to Ollama: {error}"
            ) from error

        # Decode the JSON response.
        try:

            result = json.loads(
                raw_response
            )

        except json.JSONDecodeError as error:

            raise RuntimeError(
                "Ollama returned invalid JSON."
            ) from error

        # Make sure the response is a JSON object.
        if not isinstance(result, dict):

            raise RuntimeError(
                "Ollama returned an invalid response."
            )

        # Extract the generated response.
        generated_text = result.get(
            "response"
        )

        # Make sure Ollama returned text.
        if not isinstance(
            generated_text,
            str,
        ):

            raise RuntimeError(
                "Ollama response did not contain "
                "a valid 'response' field."
            )

        # Clean the generated response.
        generated_text = (
            generated_text.strip()
        )

        # Reject an empty model response.
        if not generated_text:

            raise RuntimeError(
                "Ollama returned an empty response."
            )

        # Return the generated text.
        return generated_text