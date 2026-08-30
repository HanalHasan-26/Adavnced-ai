# Import Path for temporary database paths.
from pathlib import Path

# Import the persistent memory system.
from app.memory.memory import Memory


def test_memory_persists_after_restart(tmp_path: Path):

    # Create a database path inside pytest's temporary directory.
    database_path = tmp_path / "memory.db"

    # Create the first memory instance.
    memory_1 = Memory(
        database_path=database_path
    )

    # Save a memory.
    memory_id = memory_1.add(
        "User: My name is Pirlo."
    )

    # Make sure the memory was created.
    assert memory_id is not None

    # Create a completely new Memory instance.
    #
    # This simulates restarting the application.
    memory_2 = Memory(
        database_path=database_path
    )

    # Search the new memory instance.
    results = memory_2.search(
        query="Pirlo",
        limit=5,
    )

    # Make sure the previous memory still exists.
    assert len(results) > 0

    # Verify that the original content survived
    # the restart.
    assert any(
        result["content"]
        == "User: My name is Pirlo."
        for result in results
    )