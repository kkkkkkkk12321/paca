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
