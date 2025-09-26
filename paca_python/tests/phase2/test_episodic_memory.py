import pytest

from paca.cognitive.memory.episodic import EpisodicMemory
from paca.cognitive.memory.types import EpisodicMemorySettings
from paca.core.utils import portable_storage


@pytest.fixture
def temp_storage(tmp_path, monkeypatch):
    manager = portable_storage.PortableStorageManager(base_path=tmp_path)
    monkeypatch.setattr(portable_storage, "_storage_manager", manager)
    return manager


@pytest.mark.asyncio
async def test_episodic_memory_retention_cleanup(temp_storage):
    settings = EpisodicMemorySettings(
        retention_days=1 / (24 * 60),  # 약 1분 유지
        snapshot_interval_seconds=0,
        enable_async_io=True,
        max_snapshot_items=10,
    )
    memory = EpisodicMemory(settings=settings, storage_manager=temp_storage)
    await memory.initialize()

    episode_id = await memory.store_episode({"event": "old"})
    assert episode_id
    # Retention 검증을 위해 시간이 지난 것처럼 조정
    memory.episodes[0].created_at -= 120  # 2분 전
    removed = await memory.enforce_retention()

    assert removed == 1
    assert memory.get_episode_count() == 0

    await memory.shutdown()


@pytest.mark.asyncio
async def test_episodic_memory_async_persistence(temp_storage):
    settings = EpisodicMemorySettings(
        retention_days=None,
        snapshot_interval_seconds=0,
        enable_async_io=True,
        max_snapshot_items=5,
    )

    primary = EpisodicMemory(settings=settings, storage_manager=temp_storage)
    await primary.initialize()
    await primary.store_episode({"event": "persist"})
    await primary.shutdown()

    secondary = EpisodicMemory(settings=settings, storage_manager=temp_storage)
    await secondary.initialize()

    assert secondary.get_episode_count() == 1

    await secondary.shutdown()


@pytest.mark.asyncio
async def test_episodic_memory_export_import_cycle(temp_storage, tmp_path):
    settings = EpisodicMemorySettings(
        retention_days=None,
        snapshot_interval_seconds=0,
        enable_async_io=False,  # export/import 경로만 검증
        max_snapshot_items=10,
    )

    memory = EpisodicMemory(settings=settings, storage_manager=temp_storage)
    await memory.initialize()
    await memory.store_episode({"event": "alpha"})
    await memory.store_episode({"event": "beta"})

    export_path = tmp_path / "episodic_export.json"
    exported_file = await memory.export_batch(path=export_path)
    assert exported_file.exists()

    # 저장소 초기화 후 가져오기
    await memory.shutdown()
    storage_file = temp_storage.get_memory_file_path("episodic", "episodic_memory.json")
    if storage_file.exists():
        storage_file.unlink()

    restored = EpisodicMemory(settings=settings, storage_manager=temp_storage)
    await restored.initialize()
    assert restored.get_episode_count() == 0

    imported = await restored.import_batch(exported_file, merge_strategy="replace")
    assert imported == 2
    assert restored.get_episode_count() == 2

    await restored.shutdown()
