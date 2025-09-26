import pytest

from paca.cognitive.memory.longterm import LongTermMemory
from paca.cognitive.memory.types import LongTermMemorySettings
from paca.core.utils import portable_storage


@pytest.fixture
def temp_storage(tmp_path, monkeypatch):
    manager = portable_storage.PortableStorageManager(base_path=tmp_path)
    monkeypatch.setattr(portable_storage, "_storage_manager", manager)
    return manager


@pytest.mark.asyncio
async def test_longterm_memory_enforces_max_items(temp_storage):
    settings = LongTermMemorySettings(
        max_items=3,
        min_strength_threshold=None,
        max_idle_seconds=None,
        cleanup_batch_size=10,
        persistent_db=False,
    )
    memory = LongTermMemory(settings=settings, storage_manager=temp_storage)
    await memory.initialize()

    for idx in range(5):
        await memory.store({"value": idx})

    count = await memory.get_item_count()
    assert count <= 3

    await memory.shutdown()


@pytest.mark.asyncio
async def test_longterm_memory_strength_cleanup(temp_storage):
    settings = LongTermMemorySettings(
        max_items=10,
        min_strength_threshold=0.5,
        max_idle_seconds=None,
        cleanup_batch_size=10,
        persistent_db=False,
    )
    memory = LongTermMemory(settings=settings, storage_manager=temp_storage)
    await memory.initialize()

    strong_id = await memory.store({"value": "strong"}, strength=0.9)
    weak_id = await memory.store({"value": "weak"}, strength=0.2)

    count_before = await memory.get_item_count()
    assert count_before <= 2

    await memory.cleanup()
    count_after = await memory.get_item_count()

    assert count_after == 1
    assert await memory.retrieve(strong_id) is not None
    assert await memory.retrieve(weak_id) is None

    await memory.shutdown()


@pytest.mark.asyncio
async def test_longterm_memory_export_import_cycle(temp_storage, tmp_path):
    settings = LongTermMemorySettings(
        max_items=20,
        min_strength_threshold=None,
        max_idle_seconds=None,
        cleanup_batch_size=10,
        persistent_db=False,
    )
    memory = LongTermMemory(settings=settings, storage_manager=temp_storage)
    await memory.initialize()

    await memory.store({"topic": "alpha"}, strength=0.6)
    await memory.store({"topic": "beta"}, strength=0.8)

    export_path = tmp_path / "longterm_export.json"
    exported = await memory.export_items(path=export_path)
    assert exported.exists()

    await memory.shutdown()

    restored = LongTermMemory(settings=settings, storage_manager=temp_storage)
    await restored.initialize()
    imported = await restored.import_items(exported, upsert=True)

    assert imported == 2
    assert await restored.get_item_count() == 2

    await restored.shutdown()
