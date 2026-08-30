# Import pytest for validation tests.
import pytest

# Import Path for temporary database paths.
from pathlib import Path

# Import the memory system.
from app.memory.memory import Memory


# Create a temporary database for each test.
@pytest.fixture
def memory(tmp_path: Path):

    # Create a temporary database path.
    database_path = tmp_path / "memory.db"

    # Return a fresh memory system.
    return Memory(
        database_path=database_path
    )


# Test that a memory can be added.
def test_add_memory(memory):

    # Add a memory.
    memory_id = memory.add(
        "My name is Pirlo."
    )

    # Verify an ID was generated.
    assert memory_id

    # Retrieve the memory.
    result = memory.get(memory_id)

    # Verify the memory exists.
    assert result is not None

    # Verify the stored content.
    assert result["content"] == (
        "My name is Pirlo."
    )


# Test that multiple memories can be stored.
def test_multiple_memories(memory):

    # Add two memories.
    first_id = memory.add(
        "I trade forex."
    )

    second_id = memory.add(
        "I am learning AI."
    )

    # Retrieve all memories.
    memories = memory.list()

    # Verify both memories exist.
    assert len(memories) == 2

    # Verify both IDs exist.
    ids = {
        item["id"]
        for item in memories
    }

    assert first_id in ids
    assert second_id in ids


# Test that memory search works.
def test_search_memory(memory):

    # Store memories.
    memory.add(
        "I trade forex and gold."
    )

    memory.add(
        "I am learning Python."
    )

    # Search for forex-related memory.
    results = memory.search(
        "forex"
    )

    # Verify one matching result exists.
    assert len(results) == 1

    # Verify the correct memory was returned.
    assert (
        "forex"
        in results[0]["content"]
    )


# Test that searching with no matches
# returns an empty list.
def test_search_without_match(memory):

    # Store a memory.
    memory.add(
        "I am learning Python."
    )

    # Search for something unrelated.
    results = memory.search(
        "football"
    )

    # Verify no results were returned.
    assert results == []


# Test that an empty memory is rejected.
def test_empty_memory_is_rejected(memory):

    # Verify empty content raises an error.
    with pytest.raises(ValueError):

        memory.add("")


# Test that whitespace-only memory
# is rejected.
def test_whitespace_memory_is_rejected(memory):

    # Verify whitespace-only content raises an error.
    with pytest.raises(ValueError):

        memory.add("   ")


# Test that empty searches return no results.
def test_empty_search(memory):

    # Store a memory.
    memory.add(
        "I trade forex."
    )

    # Search using an empty query.
    results = memory.search("")

    # Verify no results are returned.
    assert results == []


# Test that invalid search limits are rejected.
def test_invalid_search_limit(memory):

    # Verify zero is rejected.
    with pytest.raises(ValueError):

        memory.search(
            "forex",
            limit=0,
        )

    # Verify negative values are rejected.
    with pytest.raises(ValueError):

        memory.search(
            "forex",
            limit=-1,
        )


# Test that a memory can be deleted.
def test_delete_memory(memory):

    # Add a memory.
    memory_id = memory.add(
        "Temporary memory."
    )

    # Verify it exists.
    assert memory.get(
        memory_id
    ) is not None

    # Delete it.
    deleted = memory.delete(
        memory_id
    )

    # Verify deletion succeeded.
    assert deleted is True

    # Verify it no longer exists.
    assert memory.get(
        memory_id
    ) is None


# Test deleting an unknown memory.
def test_delete_unknown_memory(memory):

    # Try deleting a memory that doesn't exist.
    deleted = memory.delete(
        "does-not-exist"
    )

    # Verify nothing was deleted.
    assert deleted is False


# Test that the memory survives
# creation of a new Memory instance.
def test_memory_persistence(tmp_path):

    # Create a database path.
    database_path = (
        tmp_path / "memory.db"
    )

    # Create the first memory instance.
    first_memory = Memory(
        database_path=database_path
    )

    # Store a memory.
    memory_id = first_memory.add(
        "Persistent memory."
    )

    # Create a completely new memory instance
    # using the same database.
    second_memory = Memory(
        database_path=database_path
    )

    # Retrieve the memory.
    result = second_memory.get(
        memory_id
    )

    # Verify the memory survived.
    assert result is not None

    # Verify the content survived.
    assert result["content"] == (
        "Persistent memory."
    )