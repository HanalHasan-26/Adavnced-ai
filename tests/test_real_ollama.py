from app.llm.local import LocalLLMClient
from app.llm.ollama import OllamaBackend


def test_real_ollama():

    backend = OllamaBackend(
        model="qwen3:1.7b",
    )

    client = LocalLLMClient(
        backend=backend,
    )

    response = client.generate(
        "What is support in trading? Answer briefly."
    )

    assert isinstance(response, str)
    assert response.strip()