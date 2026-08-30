# Import the knowledge chunk model.
from app.knowledge.chunking.chunk import KnowledgeChunk

# Import the chunk retriever.
from app.knowledge.retrieval.chunk_retriever import ChunkRetriever

# Import the context assembler.
from app.knowledge.retrieval.context import KnowledgeContext

# Import the retrieval result model.
from app.knowledge.retrieval.result import KnowledgeRetrievalResult


# Create the main knowledge retrieval pipeline.
class KnowledgeRetrievalPipeline:

    # Initialize the pipeline.
    def __init__(
        self,
        retriever: ChunkRetriever,
        context: KnowledgeContext | None = None,
    ):

        # Store the chunk retriever.
        self.retriever = retriever

        # Use the supplied context assembler when provided.
        # Otherwise create a default one.
        self.context = context or KnowledgeContext()

    # Retrieve relevant knowledge and return a complete result.
    def run(
        self,
        query: str,
        limit: int = 5,
    ) -> KnowledgeRetrievalResult:

        # Retrieve the most relevant chunks.
        chunks: list[KnowledgeChunk] = self.retriever.retrieve(
            query=query,
            limit=limit,
        )

        # Convert the retrieved chunks into AI-ready context.
        context = self.context.assemble(chunks)

        # Return both the chunks and assembled context.
        return KnowledgeRetrievalResult(
            query=query,
            chunks=chunks,
            context=context,
        )