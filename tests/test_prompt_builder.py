# Import pytest for validation tests.
import pytest

# Import the prompt builder.
from app.llm.prompt_builder import PromptBuilder


# Test that a prompt is created with knowledge context.
def test_build_prompt_with_context():

    # Create the prompt builder.
    builder = PromptBuilder()

    # Build a prompt.
    result = builder.build(
        query="What is support?",
        context=(
            "[Knowledge 1]\n"
            "Support is a price level where buying "
            "interest may appear."
        ),
    )

    # Verify that the question appears.
    assert "What is support?" in result

    # Verify that the knowledge appears.
    assert "Support is a price level" in result

    # Verify that the answer section exists.
    assert "Answer:" in result


# Test that the prompt identifies the knowledge section.
def test_prompt_contains_knowledge_section():

    # Create the builder.
    builder = PromptBuilder()

    # Build the prompt.
    result = builder.build(
        query="What is resistance?",
        context="Resistance can act as a price barrier.",
    )

    # Verify the section heading.
    assert "Knowledge:" in result


# Test that the prompt identifies the user question.
def test_prompt_contains_user_question_section():

    # Create the builder.
    builder = PromptBuilder()

    # Build the prompt.
    result = builder.build(
        query="What is resistance?",
        context="Resistance can act as a price barrier.",
    )

    # Verify the section heading.
    assert "User question:" in result


# Test that empty context is supported.
def test_build_prompt_without_context():

    # Create the builder.
    builder = PromptBuilder()

    # Build a prompt without retrieved knowledge.
    result = builder.build(
        query="What is artificial intelligence?",
        context="",
    )

    # Verify that the question appears.
    assert "What is artificial intelligence?" in result

    # Verify that an answer section still exists.
    assert "Answer:" in result


# Test that whitespace-only context is treated
# as empty context.
def test_whitespace_context():

    # Create the builder.
    builder = PromptBuilder()

    # Build the prompt.
    result = builder.build(
        query="What is AI?",
        context="   \n   ",
    )

    # Verify that the question remains.
    assert "What is AI?" in result

    # The prompt should still be valid.
    assert "Answer:" in result


# Test that query whitespace is removed.
def test_query_whitespace_is_removed():

    # Create the builder.
    builder = PromptBuilder()

    # Build using padded query text.
    result = builder.build(
        query="   What is AI?   ",
        context="AI means artificial intelligence.",
    )

    # Verify the cleaned question.
    assert "What is AI?" in result

    # Verify the padded version isn't present.
    assert "   What is AI?   " not in result


# Test that an empty query is rejected.
def test_empty_query_is_rejected():

    # Create the builder.
    builder = PromptBuilder()

    # Verify that empty input raises an error.
    with pytest.raises(ValueError):

        builder.build(
            query="",
            context="Some knowledge.",
        )


# Test that whitespace-only queries are rejected.
def test_whitespace_query_is_rejected():

    # Create the builder.
    builder = PromptBuilder()

    # Verify that whitespace-only input raises an error.
    with pytest.raises(ValueError):

        builder.build(
            query="   ",
            context="Some knowledge.",
        )