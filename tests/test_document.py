# Import the KnowledgeDocument model we want to test.
from app.knowledge.document import KnowledgeDocument


# Define a test that checks whether a document is created correctly.
def test_create_knowledge_document():

    # Create a sample knowledge document.
    document = KnowledgeDocument.create(
        content="Python is a programming language.",
        source="example.txt",
        source_type="txt",
    )

    # Make sure the document has a unique ID.
    assert document.id is not None

    # Make sure the content was stored correctly.
    assert document.content == "Python is a programming language."

    # Make sure the source was stored correctly.
    assert document.source == "example.txt"

    # Make sure the source type was stored correctly.
    assert document.source_type == "txt"

    # Make sure a creation timestamp was generated.
    assert document.created_at is not None