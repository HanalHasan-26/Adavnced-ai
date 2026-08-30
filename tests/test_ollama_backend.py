# Import JSON for creating fake HTTP responses.
import json

# Import pytest for validation and error tests.
import pytest

# Import the Ollama backend.
from app.llm.ollama import OllamaBackend


# Create a fake HTTP response.
class FakeResponse:

    # Initialize the fake response.
    def __init__(self, payload):

        # Convert the payload into JSON bytes.
        self.payload = json.dumps(
            payload
        ).encode("utf-8")

    # Support the context-manager protocol.
    def __enter__(self):

        return self

    # Support the context-manager protocol.
    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        return False

    # Return the fake response body.
    def read(self):

        return self.payload


# Test that Ollama responses are parsed correctly.
def test_ollama_generate(monkeypatch):

    # Create a backend.
    backend = OllamaBackend(
        model="qwen3:4b",
    )

    # Store the request received by the fake server.
    captured = {}

    # Create a fake HTTP request handler.
    def fake_urlopen(
        request,
        timeout,
    ):

        # Capture the request.
        captured["request"] = request
        captured["timeout"] = timeout

        # Return a successful Ollama response.
        return FakeResponse(
            {
                "response": "Hello from Qwen.",
            }
        )

    # Replace the real HTTP call.
    monkeypatch.setattr(
        "app.llm.ollama.urlopen",
        fake_urlopen,
    )

    # Generate text.
    result = backend.generate(
        "Hello",
    )

    # Verify the generated response.
    assert result == "Hello from Qwen."

    # Decode the request payload.
    payload = json.loads(
        captured["request"].data.decode("utf-8")
    )

    # Verify the model.
    assert payload["model"] == "qwen3:4b"

    # Verify the prompt.
    assert payload["prompt"] == "Hello"

    # Verify streaming is disabled.
    assert payload["stream"] is False


# Test that prompt whitespace is removed.
def test_ollama_strips_prompt(monkeypatch):

    # Create the backend.
    backend = OllamaBackend(
        model="qwen3:4b",
    )

    # Store the request.
    captured = {}

    # Create a fake HTTP handler.
    def fake_urlopen(
        request,
        timeout,
    ):

        captured["request"] = request

        return FakeResponse(
            {
                "response": "Answer",
            }
        )

    # Replace the real HTTP call.
    monkeypatch.setattr(
        "app.llm.ollama.urlopen",
        fake_urlopen,
    )

    # Generate with padded input.
    backend.generate(
        "   Hello   "
    )

    # Decode the request.
    payload = json.loads(
        captured["request"].data.decode("utf-8")
    )

    # Verify the prompt was cleaned.
    assert payload["prompt"] == "Hello"


# Test that an empty model name is rejected.
def test_ollama_empty_model():

    with pytest.raises(ValueError):

        OllamaBackend(
            model="",
        )


# Test that an empty URL is rejected.
def test_ollama_empty_base_url():

    with pytest.raises(ValueError):

        OllamaBackend(
            model="qwen3:4b",
            base_url="",
        )


# Test that an invalid timeout is rejected.
def test_ollama_invalid_timeout():

    with pytest.raises(ValueError):

        OllamaBackend(
            model="qwen3:4b",
            timeout=0,
        )


# Test that an empty prompt is rejected.
def test_ollama_empty_prompt():

    # Create the backend.
    backend = OllamaBackend(
        model="qwen3:4b",
    )

    # Verify empty input is rejected.
    with pytest.raises(ValueError):

        backend.generate("")


# Test that invalid JSON is rejected.
def test_ollama_invalid_json(monkeypatch):

    # Create the backend.
    backend = OllamaBackend(
        model="qwen3:4b",
    )

    # Create a fake response containing invalid JSON.
    class InvalidJSONResponse:

        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type,
            exc_value,
            traceback,
        ):
            return False

        def read(self):
            return b"not valid json"

    # Replace the HTTP call.
    monkeypatch.setattr(
        "app.llm.ollama.urlopen",
        lambda request, timeout: InvalidJSONResponse(),
    )

    # Verify the error is handled.
    with pytest.raises(RuntimeError):

        backend.generate("Hello")


# Test that a missing response field is rejected.
def test_ollama_missing_response(monkeypatch):

    # Create the backend.
    backend = OllamaBackend(
        model="qwen3:4b",
    )

    # Replace the HTTP call.
    monkeypatch.setattr(
        "app.llm.ollama.urlopen",
        lambda request, timeout: FakeResponse(
            {
                "model": "qwen3:4b",
            }
        ),
    )

    # Verify the malformed response is rejected.
    with pytest.raises(RuntimeError):

        backend.generate("Hello")


# Test that an empty response is rejected.
def test_ollama_empty_response(monkeypatch):

    # Create the backend.
    backend = OllamaBackend(
        model="qwen3:4b",
    )

    # Replace the HTTP call.
    monkeypatch.setattr(
        "app.llm.ollama.urlopen",
        lambda request, timeout: FakeResponse(
            {
                "response": "   ",
            }
        ),
    )

    # Verify the empty response is rejected.
    with pytest.raises(RuntimeError):

        backend.generate("Hello")

# Import URL errors for simulating connection failures.
from urllib.error import URLError


# Test that connection failures are converted
# into a RuntimeError.
def test_ollama_connection_error(monkeypatch):

    # Create the backend.
    backend = OllamaBackend(
        model="qwen3:4b",
    )

    # Create a fake HTTP handler that fails.
    def fake_urlopen(request, timeout):

        raise URLError(
            "Connection refused"
        )

    # Replace the real HTTP call.
    monkeypatch.setattr(
        "app.llm.ollama.urlopen",
        fake_urlopen,
    )

    # Verify the connection error is handled.
    with pytest.raises(RuntimeError, match="Unable to connect to Ollama"):

        backend.generate("Hello")


# Test that Ollama HTTP errors are converted
# into a RuntimeError.
def test_ollama_http_error(monkeypatch):

    # Create the backend.
    backend = OllamaBackend(
        model="qwen3:4b",
    )

    # Create a fake HTTP error.
    from urllib.error import HTTPError

    def fake_urlopen(request, timeout):

        raise HTTPError(
            url=request.full_url,
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=None,
        )

    # Replace the real HTTP call.
    monkeypatch.setattr(
        "app.llm.ollama.urlopen",
        fake_urlopen,
    )

    # Verify the HTTP error is handled.
    with pytest.raises(
        RuntimeError,
        match="Ollama request failed",
    ):

        backend.generate("Hello")


# Test that a non-object JSON response is rejected.
def test_ollama_invalid_response_type(monkeypatch):

    # Create the backend.
    backend = OllamaBackend(
        model="qwen3:4b",
    )

    # Create a fake response containing a JSON list.
    class InvalidResponse:

        def __enter__(self):

            return self

        def __exit__(
            self,
            exc_type,
            exc_value,
            traceback,
        ):

            return False

        def read(self):

            return json.dumps(
                ["invalid"]
            ).encode("utf-8")

    # Replace the HTTP call.
    monkeypatch.setattr(
        "app.llm.ollama.urlopen",
        lambda request, timeout: InvalidResponse(),
    )

    # Verify the malformed response is rejected.
    with pytest.raises(
        RuntimeError,
        match="invalid response",
    ):

        backend.generate("Hello")


# Test that a non-string response field is rejected.
def test_ollama_non_string_response(monkeypatch):

    # Create the backend.
    backend = OllamaBackend(
        model="qwen3:4b",
    )

    # Replace the HTTP call.
    monkeypatch.setattr(
        "app.llm.ollama.urlopen",
        lambda request, timeout: FakeResponse(
            {
                "response": 123,
            }
        ),
    )

    # Verify the malformed response is rejected.
    with pytest.raises(RuntimeError):

        backend.generate("Hello")