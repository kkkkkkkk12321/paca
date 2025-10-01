import pytest

from paca.system import PacaSystem


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
