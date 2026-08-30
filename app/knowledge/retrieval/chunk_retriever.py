# Import the knowledge chunk model.
from app.knowledge.chunking.chunk import KnowledgeChunk

# Import persistent knowledge storage.
from app.knowledge.storage import KnowledgeStorage


# Create a component responsible for retrieving knowledge chunks.
class ChunkRetriever:

    # Initialize the retriever.
    def __init__(self, storage: KnowledgeStorage):

        # Store the storage dependency.
        self.storage = storage

    # Retrieve chunks that contain the requested query.
    def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> list[KnowledgeChunk]:

        # Remove unnecessary whitespace.
        query = query.strip()

        # Return no results for an empty query.
        if not query:
            return []

        # Make sure the requested limit is valid.
        if limit <= 0:
            raise ValueError("limit must be greater than 0.")

        # Search through stored documents.
        documents = self.storage.search(query)

        # Store matching chunks.
        matching_chunks: list[KnowledgeChunk] = []

        # Examine every matching document.
        for document in documents:

            # Retrieve the chunks belonging to this document.
            chunks = self.storage.get_chunks(str(document.id))

            # Check every chunk.
            for chunk in chunks:

                # Match the query against chunk content.
                if query.lower() in chunk.content.lower():

                    # Add the matching chunk.
                    matching_chunks.append(chunk)

                    # Stop once we have enough results.
                    if len(matching_chunks) >= limit:
                        return matching_chunks

        # Return all matching chunks when fewer than the limit exist.
        return matching_chunks