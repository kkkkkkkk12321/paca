
import sys
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from paca.system import PacaSystem
from paca.learning.auto.types import LearningPattern, PatternType


LEARNING_STORAGE = PROJECT_ROOT / "data" / "memory" / "learning"


@pytest.fixture(autouse=True)
def clean_learning_storage():
    if LEARNING_STORAGE.exists():
        shutil.rmtree(LEARNING_STORAGE)
    yield
    if LEARNING_STORAGE.exists():
        shutil.rmtree(LEARNING_STORAGE)


@pytest.mark.asyncio
async def test_auto_learning_generates_learning_points():
    system = PacaSystem()

    try:
        await system.config_manager.initialize()
        system.config_manager.set_value("default", "llm.api_keys", [])

        init_result = await system.initialize()
        assert init_result.is_success
        assert system.auto_learning_system is not None

        result = await system.process_message(
            "오류가 있었는데 지금은 완전히 해결됐어 정말 고마워요",
            user_id="tester",
        )

        assert result.is_success
        analysis = result.data.get("analysis", {})
        learning = analysis.get("learning")
        assert learning, "학습 요약이 포함되어야 합니다"
        assert learning["detected_points"] >= 1
        assert system.recent_learning_points, "최근 학습 포인트 ID가 기록되어야 합니다"

        store = system.data_storage.get_store("learning")
        assert store is not None
        count_result = await store.count()
        assert count_result.is_success
        assert count_result.data >= 1

    finally:
        await system.cleanup()


@pytest.mark.asyncio
async def test_auto_learning_persists_between_sessions():
    system = PacaSystem()

    initial_point_ids = []
    initial_tactic_ids = []
    initial_heuristic_ids = []
    initial_metrics_snapshot = {}

    try:
        await system.config_manager.initialize()
        system.config_manager.set_value("default", "llm.api_keys", [])

        init_result = await system.initialize()
        assert init_result.is_success
        assert system.auto_learning_system is not None

        auto_learning = system.auto_learning_system
        auto_learning.learning_patterns = [
            LearningPattern(
                pattern_type=PatternType.SUCCESS,
                keywords=["success"],
                context_indicators=["context"],
                confidence_threshold=0.1,
                extraction_rule=""
            ),
            LearningPattern(
                pattern_type=PatternType.FAILURE,
                keywords=["failure"],
                context_indicators=["context"],
                confidence_threshold=0.1,
                extraction_rule=""
            ),
        ]

        success_message = "success context success message ensures successful handling of the task"
        failure_message = "failure context failure sequence describing the persistent problem"

        success_result = await system.process_message(success_message, user_id="tester")
        assert success_result.is_success

        failure_result = await system.process_message(failure_message, user_id="tester")
        assert failure_result.is_success

        for tactic in auto_learning.generated_tactics:
            tactic.apply(success=True)
        for heuristic in auto_learning.generated_heuristics:
            heuristic.trigger(avoided=True)
        await auto_learning._save_learning_data()

        initial_point_ids = [lp.id for lp in auto_learning.learning_points]
        initial_tactic_ids = [t.id for t in auto_learning.generated_tactics]
        initial_heuristic_ids = [h.id for h in auto_learning.generated_heuristics]
        initial_metrics_snapshot = asdict(auto_learning.metrics)

        assert initial_point_ids, "학습 포인트가 저장되어야 합니다"
        assert initial_tactic_ids, "자동 생성된 전술이 저장되어야 합니다"
        assert initial_heuristic_ids, "자동 생성된 휴리스틱이 저장되어야 합니다"

    finally:
        await system.cleanup()

    system_reloaded = PacaSystem()

    try:
        await system_reloaded.config_manager.initialize()
        system_reloaded.config_manager.set_value("default", "llm.api_keys", [])

        init_result = await system_reloaded.initialize()
        assert init_result.is_success
        assert system_reloaded.auto_learning_system is not None

        auto_learning = system_reloaded.auto_learning_system
        reloaded_point_ids = [lp.id for lp in auto_learning.learning_points]
        reloaded_tactic_ids = [t.id for t in auto_learning.generated_tactics]
        reloaded_heuristic_ids = [h.id for h in auto_learning.generated_heuristics]
        reloaded_metrics_snapshot = asdict(auto_learning.metrics)

        assert set(reloaded_point_ids) == set(initial_point_ids)
        assert len(reloaded_point_ids) == len(initial_point_ids)
        assert set(reloaded_tactic_ids) == set(initial_tactic_ids)
        assert len(reloaded_tactic_ids) == len(initial_tactic_ids)
        assert set(reloaded_heuristic_ids) == set(initial_heuristic_ids)
        assert len(reloaded_heuristic_ids) == len(initial_heuristic_ids)
        assert reloaded_metrics_snapshot == initial_metrics_snapshot

    finally:
        await system_reloaded.cleanup()
