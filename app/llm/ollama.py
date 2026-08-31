# Import JSON so we can decode Ollama's response.
import json

# Import urllib tools for making HTTP requests.
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaBackend:
    """
    Local Ollama LLM backend.

    Communicates directly with the Ollama HTTP API
    without requiring a third-party Python client.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
    ) -> None:

        # Make sure the model name is valid.
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                "model must be a non-empty string"
            )

        # Make sure the base URL is valid.
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError(
                "base_url must be a non-empty string"
            )

        # Make sure timeout is positive.
        if timeout <= 0:
            raise ValueError(
                "timeout must be greater than zero"
            )

        # Store the Ollama model name.
        self.model = model.strip()

        # Remove unnecessary trailing slashes.
        self.base_url = base_url.rstrip("/")

        # Store the HTTP timeout.
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        """
        Generate a response from Ollama.
        """

        # Validate the prompt.
        if not isinstance(prompt, str):
            raise TypeError(
                "prompt must be a string"
            )

        # Remove unnecessary whitespace.
        prompt = prompt.strip()

        # Do not send an empty prompt to Ollama.
        if not prompt:
            raise ValueError(
                "prompt cannot be empty"
            )

        # Ollama generation endpoint.
        url = f"{self.base_url}/api/generate"

        # Build the request payload.
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        # Convert the payload to JSON bytes.
        data = json.dumps(payload).encode("utf-8")

        # Create the HTTP request.
        request = Request(
            url=url,
            data=data,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:

            # Send the request to Ollama.
            with urlopen(
                request,
                timeout=self.timeout,
            ) as response:

                # Read the response body.
                raw_response = response.read()

        except HTTPError as error:

            # Handle HTTP-level errors.
            raise RuntimeError(
                f"Ollama request failed: HTTP {error.code}"
            ) from error

        except URLError as error:

            # Handle connection and URL errors.
            raise RuntimeError(
                f"Unable to connect to Ollama: {error.reason}"
            ) from error

        except TimeoutError as error:

            # Handle request timeout.
            raise RuntimeError(
                "Ollama request timed out"
            ) from error

        except OSError as error:

            # Handle lower-level network errors.
            raise RuntimeError(
                f"Unable to connect to Ollama: {error}"
            ) from error

        # Decode the response as UTF-8.
        try:

            response_text = raw_response.decode(
                "utf-8"
            )

        except UnicodeDecodeError as error:

            raise RuntimeError(
                "Ollama returned invalid UTF-8 data"
            ) from error

        # Parse the JSON response.
        try:

            response_data = json.loads(
                response_text
            )

        except json.JSONDecodeError as error:

            raise RuntimeError(
                "Ollama returned invalid JSON"
            ) from error

        # Make sure Ollama returned an object.
        if not isinstance(response_data, dict):

            raise RuntimeError(
                "Ollama returned an invalid response type"
            )

        # Extract the generated response.
        response = response_data.get(
            "response"
        )

        # Make sure the response field exists.
        if response is None:

            raise RuntimeError(
                "Ollama response is missing the 'response' field"
            )

        # Make sure the response is a string.
        if not isinstance(response, str):

            raise RuntimeError(
                "Ollama response must be a string"
            )

        # Remove unnecessary whitespace.
        response = response.strip()

        # Reject an empty response.
        if not response:

            raise RuntimeError(
                "Ollama returned an empty response"
            )

        # Return the generated answer.
        return response