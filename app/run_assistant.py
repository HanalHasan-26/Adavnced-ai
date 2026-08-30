# Import the knowledge storage.
from app.knowledge.storage import KnowledgeStorage

# Import the chunk retriever.
from app.knowledge.retrieval.chunk_retriever import ChunkRetriever

# Import the retrieval pipeline.
from app.knowledge.retrieval.pipeline import KnowledgeRetrievalPipeline

# Import the main knowledge assistant.
from app.knowledge.assistant import KnowledgeAssistant

# Import the local LLM client.
from app.llm.local import LocalLLMClient

# Import the Ollama backend.
from app.llm.ollama import OllamaBackend


# Create the complete local AI assistant.
def create_assistant() -> KnowledgeAssistant:

    # Create persistent knowledge storage.
    storage = KnowledgeStorage()

    # Create the knowledge retriever.
    retriever = ChunkRetriever(
        storage=storage,
    )

    # Create the retrieval pipeline.
    pipeline = KnowledgeRetrievalPipeline(
        retriever=retriever,
    )

    # Create the Ollama backend.
    backend = OllamaBackend(
        model="qwen3:1.7b",
    )

    # Create the LLM client.
    llm = LocalLLMClient(
        backend=backend,
    )

    # Create the complete assistant.
    return KnowledgeAssistant(
        retrieval_pipeline=pipeline,
        llm=llm,
    )


# Run the interactive assistant.
def main() -> None:

    # Create the assistant.
    assistant = create_assistant()

    print()
    print("==============================")
    print("       Local AI Assistant")
    print("==============================")
    print("Model: Qwen3 1.7B")
    print("Type 'exit' to quit.")
    print()

    # Start the conversation loop.
    while True:

        # Read the user's question.
        query = input("You: ").strip()

        # Exit the application.
        if query.lower() in {"exit", "quit"}:

            print()
            print("Goodbye.")

            break

        # Ignore empty input.
        if not query:
            continue

        try:

            # Ask the complete AI system.
            answer = assistant.ask(
                query=query,
                limit=5,
            )

            # Display the answer.
            print()
            print("AI:")
            print(answer)
            print()

        except Exception as error:

            # Display a readable error.
            print()
            print(f"Error: {error}")
            print()


# Start the application.
if __name__ == "__main__":
    main()