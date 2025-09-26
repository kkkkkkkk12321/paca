import asyncio

import pytest

from paca.cognitive.memory.types import MemoryConfiguration
from paca.cognitive.memory.working import WorkingMemory
from paca.core.utils import portable_storage


@pytest.fixture
def temp_storage(tmp_path, monkeypatch):
    manager = portable_storage.PortableStorageManager(base_path=tmp_path)
    monkeypatch.setattr(portable_storage, "_storage_manager", manager)
    return manager


@pytest.mark.asyncio
async def test_working_memory_uses_external_configuration(temp_storage):
    memory = WorkingMemory(storage_manager=temp_storage)
    await asyncio.sleep(0)

    assert memory.capacity == 12
    assert memory.ttl_seconds == 900

    await memory.shutdown()


@pytest.mark.asyncio
async def test_working_memory_ttl_expiration_cleanup(temp_storage):
    config = MemoryConfiguration(
        working_memory_capacity=2,
        working_memory_ttl_seconds=1,
        cleanup_interval_seconds=1,
    )
    memory = WorkingMemory(config=config, storage_manager=temp_storage)

    await memory.initialize()
    await memory.clear()

    item_id = await memory.store({"message": "hello"})
    assert len(memory.items) == 1
    await asyncio.sleep(1.2)

    await memory.cleanup_expired()
    assert len(memory.items) == 0
    item = await memory.retrieve(item_id)
    assert item is None

    await memory.shutdown()


@pytest.mark.asyncio
async def test_working_memory_expired_item_not_returned_by_key(temp_storage):
    config = MemoryConfiguration(
        working_memory_capacity=2,
        working_memory_ttl_seconds=1,
        cleanup_interval_seconds=1,
    )
    memory = WorkingMemory(config=config, storage_manager=temp_storage)

    await memory.initialize()
    await memory.clear()

    await memory.store_kv("greeting", "안녕하세요")
    await asyncio.sleep(1.1)
    await memory.cleanup_expired()

    value = await memory.retrieve_by_key("greeting")
    assert value is None

    await memory.shutdown()


@pytest.mark.asyncio
async def test_working_memory_handles_ttl_disabled(temp_storage):
    config = MemoryConfiguration(
        working_memory_capacity=2,
        working_memory_ttl_seconds=None,
        cleanup_interval_seconds=1,
    )
    memory = WorkingMemory(config=config, storage_manager=temp_storage)

    await memory.initialize()
    await memory.clear()
    item_id = await memory.store({"message": "persistent"})
    await asyncio.sleep(0)

    retrieved = await memory.retrieve(item_id)
    assert retrieved is not None
    assert memory._cleanup_task is None

    await memory.shutdown()


@pytest.mark.asyncio
async def test_working_memory_shutdown_cancels_cleanup_loop(temp_storage):
    config = MemoryConfiguration(
        working_memory_capacity=2,
        working_memory_ttl_seconds=1,
        cleanup_interval_seconds=1,
    )
    memory = WorkingMemory(config=config, storage_manager=temp_storage)

    await memory.initialize()
    await asyncio.sleep(0)
    assert memory._cleanup_task is not None

    await memory.shutdown()
    assert memory._cleanup_task is None
