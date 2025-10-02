import asyncio
import json
import threading
import time
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paca.learning.auto.engine import AutoLearningSystem
from paca.learning.auto.synchronizer import FileLearningDataSynchronizer, LearningDataSnapshot

from paca.learning.auto.types import (
    GeneratedTactic,
    GeneratedHeuristic,
    LearningCategory,
    LearningPoint,
)


class _StubDatabase:
    def __init__(self) -> None:
        self.experiences = []
        self.heuristics = []

    def add_experience(self, name: str, description: str) -> bool:
        self.experiences.append((name, description))
        return True

    def add_heuristic(self, rule: str) -> bool:
        self.heuristics.append(rule)
        return True

    def get_experiences(self, context=None):  # pragma: no cover - unused in test
        return []

    def get_heuristics(self, context=None):  # pragma: no cover - unused in test
        return []


class _StubConversationMemory:
    def get_recent_conversations(self, limit: int = 10):  # pragma: no cover - unused in test
        return []

    def get_conversation_context(self, conversation_id: str):  # pragma: no cover - unused in test
        return None

    def store_learning_point(self, learning_point):  # pragma: no cover - unused in test
        return True


@pytest.mark.asyncio
async def test_analyze_learning_opportunities_writes_artifacts_without_blocking(tmp_path: Path):
    system = AutoLearningSystem(
        database=_StubDatabase(),
        conversation_memory=_StubConversationMemory(),
        storage_path=str(tmp_path),
        enable_korean_nlp=False,
    )

    original_write = system._write_json_file
    write_started = threading.Event()

    def slow_write(path: Path, data):
        write_started.set()
        time.sleep(0.05)
        original_write(path, data)

    system._write_json_file = slow_write  # type: ignore[method-assign]

    user_message = "오류가 있었지만 지금은 완전히 해결됐어. 정말 훌륭해!"
    paca_response = "문제를 해결됐어 라고 보고하고 싶어요. 수정한 방법이 완벽해."

    task = asyncio.create_task(
        system.analyze_learning_opportunities(user_message, paca_response)
    )

    event_result = await asyncio.wait_for(
        asyncio.to_thread(write_started.wait, 1),
        timeout=1.5,
    )
    assert event_result, "background write was not triggered"

    await asyncio.wait_for(asyncio.sleep(0.01), timeout=0.05)

    result = await asyncio.wait_for(task, timeout=1)
    assert result.is_success

    artifacts = {
        "learning_points.json": list,
        "generated_tactics.json": list,
        "generated_heuristics.json": list,
        "learning_metrics.json": dict,
    }

    for filename, expected_type in artifacts.items():
        artifact_path = tmp_path / filename
        assert artifact_path.exists(), f"{filename} should be created"
        content = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert isinstance(content, expected_type), f"{filename} should contain {expected_type.__name__}"


@pytest.mark.asyncio
async def test_concurrent_saves_capture_mutations(tmp_path: Path):
    system = AutoLearningSystem(
        database=_StubDatabase(),
        conversation_memory=_StubConversationMemory(),
        storage_path=str(tmp_path),
        enable_korean_nlp=False,
    )

    tactic = GeneratedTactic(name="existing tactic", description="desc", context="ctx")
    heuristic = GeneratedHeuristic(pattern="pattern", avoidance_rule="avoid", context="ctx")
    system.generated_tactics.append(tactic)
    system.generated_heuristics.append(heuristic)

    original_write = system._write_json_file
    write_started = threading.Event()

    def slow_write(path: Path, data):
        if not write_started.is_set():
            write_started.set()
            time.sleep(0.05)
        original_write(path, data)

    system._write_json_file = slow_write  # type: ignore[method-assign]

    try:
        first_task = asyncio.create_task(
            system.analyze_learning_opportunities(
                "문제를 해결했어. 아주 좋아.",
                "도움이 되었다니 다행이야.",
            )
        )

        await asyncio.wait_for(asyncio.to_thread(write_started.wait, 1), timeout=1.5)

        tactic.metadata["note"] = "updated"
        heuristic.source_conversations.append("conversation-2")

        second_task = asyncio.create_task(
            system.analyze_learning_opportunities(
                "새로운 오류를 발견했지만 해결 가능해.",
                "다음 번에는 더 빠르게 대응하겠습니다.",
            )
        )

        results = await asyncio.gather(first_task, second_task)
    finally:
        system._write_json_file = original_write  # type: ignore[method-assign]

    assert all(result.is_success for result in results)

    tactics_data = json.loads((tmp_path / "generated_tactics.json").read_text(encoding="utf-8"))
    heuristics_data = json.loads((tmp_path / "generated_heuristics.json").read_text(encoding="utf-8"))

    assert any(entry.get("metadata", {}).get("note") == "updated" for entry in tactics_data)
    assert any("conversation-2" in entry.get("source_conversations", []) for entry in heuristics_data)


def test_auto_learning_system_initializes_without_event_loop(tmp_path: Path):
    system = AutoLearningSystem(
        database=_StubDatabase(),
        conversation_memory=_StubConversationMemory(),
        storage_path=str(tmp_path),
        enable_korean_nlp=False,
    )

    async def trigger_save() -> None:
        await system._save_learning_data()

    asyncio.run(trigger_save())

    for artifact in (
        "learning_points.json",
        "generated_tactics.json",
        "generated_heuristics.json",
        "learning_metrics.json",
    ):
        assert (tmp_path / artifact).exists(), f"{artifact} should be persisted without an active loop at init"


def test_auto_learning_system_survives_multiple_asyncio_run_calls(tmp_path: Path):
    system = AutoLearningSystem(
        database=_StubDatabase(),
        conversation_memory=_StubConversationMemory(),
        storage_path=str(tmp_path),
        enable_korean_nlp=False,
    )

    for _ in range(2):
        asyncio.run(system._save_learning_data())

    monitoring_snapshot = tmp_path / "monitoring" / "learning_snapshot.json"
    assert monitoring_snapshot.exists(), "default synchronizer should persist snapshot across event loops"


def test_file_learning_data_synchronizer_initializes_without_event_loop(tmp_path: Path):
    synchronizer = FileLearningDataSynchronizer(tmp_path / "snapshot.json")
    snapshot = LearningDataSnapshot(
        saved_at=time.time(),
        learning_points=[],
        generated_tactics=[],
        generated_heuristics=[],
        metrics={},
    )

    asyncio.run(synchronizer.sync(snapshot))

    assert (tmp_path / "snapshot.json").exists(), "snapshot export should succeed without pre-running loop"


def test_file_learning_data_synchronizer_survives_multiple_asyncio_run_calls(tmp_path: Path):
    synchronizer = FileLearningDataSynchronizer(tmp_path / "snapshot.json")
    snapshot = LearningDataSnapshot(
        saved_at=time.time(),
        learning_points=[],
        generated_tactics=[],
        generated_heuristics=[],
        metrics={},
    )

    for _ in range(2):
        asyncio.run(synchronizer.sync(snapshot))

    payload = json.loads((tmp_path / "snapshot.json").read_text(encoding="utf-8"))
    assert payload["learning_points"] == []


class _RecordingSynchronizer:
    def __init__(self) -> None:
        self.snapshots = []

    async def sync(self, snapshot) -> None:
        self.snapshots.append(snapshot)


@pytest.mark.asyncio
async def test_learning_snapshot_synchronizer_receives_data(tmp_path: Path):
    synchronizer = _RecordingSynchronizer()
    system = AutoLearningSystem(
        database=_StubDatabase(),
        conversation_memory=_StubConversationMemory(),
        storage_path=str(tmp_path),
        enable_korean_nlp=False,
        learning_synchronizer=synchronizer,
    )

    system.learning_points.append(
        LearningPoint(
            user_message="성공했어",
            paca_response="도움을 드릴 수 있어 기쁩니다.",
            context="테스트",
            category=LearningCategory.SUCCESS_PATTERN,
            confidence=0.9,
            extracted_knowledge="테스트 시나리오 학습",
        )
    )

    await system._save_learning_data()

    assert synchronizer.snapshots, "custom synchronizer should receive a snapshot"
    snapshot = synchronizer.snapshots[-1]
    assert snapshot.learning_points, "snapshot should include learning points"

    monitoring_snapshot = tmp_path / "monitoring" / "learning_snapshot.json"
    assert monitoring_snapshot.exists()
    exported = json.loads(monitoring_snapshot.read_text(encoding="utf-8"))
    assert exported["learning_points"], "default synchronizer should persist data for monitoring"
