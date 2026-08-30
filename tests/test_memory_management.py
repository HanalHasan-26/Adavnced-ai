from pathlib import Path

import pytest

from app.memory.memory import Memory


def test_memory_get_returns_saved_memory(
    tmp_path: Path,
):

    database_path = tmp_path / "memory.db"

    memory = Memory(
        database_path=database_path
    )

    memory_id = memory.add(
        "User: My name is Pirlo."
    )

    result = memory.get(
        memory_id
    )

    assert result is not None
    assert result["id"] == memory_id
    assert result["content"] == (
        "User: My name is Pirlo."
    )


def test_memory_get_unknown_id_returns_none(
    tmp_path: Path,
):

    database_path = tmp_path / "memory.db"

    memory = Memory(
        database_path=database_path
    )

    result = memory.get(
        "does-not-exist"
    )

    assert result is None


def test_memory_list_returns_all_memories(
    tmp_path: Path,
):

    database_path = tmp_path / "memory.db"

    memory = Memory(
        database_path=database_path
    )

    first_id = memory.add(
        "User: My name is Pirlo."
    )

    second_id = memory.add(
        "User: I trade forex."
    )

    results = memory.list()

    result_ids = {
        result["id"]
        for result in results
    }

    assert first_id in result_ids
    assert second_id in result_ids


def test_memory_delete_removes_memory(
    tmp_path: Path,
):

    database_path = tmp_path / "memory.db"

    memory = Memory(
        database_path=database_path
    )

    memory_id = memory.add(
        "User: My name is Pirlo."
    )

    deleted = memory.delete(
        memory_id
    )

    assert deleted is True

    result = memory.get(
        memory_id
    )

    assert result is None


def test_memory_delete_unknown_id_returns_false(
    tmp_path: Path,
):

    database_path = tmp_path / "memory.db"

    memory = Memory(
        database_path=database_path
    )

    deleted = memory.delete(
        "does-not-exist"
    )

    assert deleted is False


def test_memory_delete_persists_after_restart(
    tmp_path: Path,
):

    database_path = tmp_path / "memory.db"

    memory_1 = Memory(
        database_path=database_path
    )

    memory_id = memory_1.add(
        "User: My name is Pirlo."
    )

    assert memory_1.delete(
        memory_id
    ) is True

    # Simulate application restart.
    memory_2 = Memory(
        database_path=database_path
    )

    assert memory_2.get(
        memory_id
    ) is None


def test_memory_search_finds_matching_memory(
    tmp_path: Path,
):

    database_path = tmp_path / "memory.db"

    memory = Memory(
        database_path=database_path
    )

    memory.add(
        "User: My name is Pirlo."
    )

    memory.add(
        "User: I like football."
    )

    results = memory.search(
        query="Pirlo"
    )

    assert len(results) == 1

    assert results[0]["content"] == (
        "User: My name is Pirlo."
    )


def test_memory_search_is_case_insensitive(
    tmp_path: Path,
):

    database_path = tmp_path / "memory.db"

    memory = Memory(
        database_path=database_path
    )

    memory.add(
        "User: My name is Pirlo."
    )

    results = memory.search(
        query="pirlo"
    )

    assert len(results) == 1


def test_memory_add_rejects_empty_content(
    tmp_path: Path,
):

    database_path = tmp_path / "memory.db"

    memory = Memory(
        database_path=database_path
    )

    with pytest.raises(ValueError):
        memory.add("")


def test_memory_search_rejects_invalid_limit(
    tmp_path: Path,
):

    database_path = tmp_path / "memory.db"

    memory = Memory(
        database_path=database_path
    )

    with pytest.raises(ValueError):
        memory.search(
            query="Pirlo",
            limit=0,
        )


def test_memory_get_rejects_empty_id(
    tmp_path: Path,
):

    database_path = tmp_path / "memory.db"

    memory = Memory(
        database_path=database_path
    )

    with pytest.raises(ValueError):
        memory.get("")


def test_memory_delete_rejects_empty_id(
    tmp_path: Path,
):

    database_path = tmp_path / "memory.db"

    memory = Memory(
        database_path=database_path
    )

    with pytest.raises(ValueError):
        memory.delete("")