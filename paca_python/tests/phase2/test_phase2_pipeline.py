import pytest

from paca.system import PacaSystem


@pytest.mark.asyncio
async def test_process_message_complexity_pipeline():
    system = PacaSystem()

    init_result = await system.initialize()
    assert init_result.is_success, f"Initialization failed: {init_result.error}"

    message = (
        "만약 분산 시스템에서 데이터 일관성과 가용성을 동시에 확보하려면 어떤 설계 전략을 선택해야 하는지"
        " 단계별로 설명해 주고, 각 선택의 장단점도 비교해 주세요."
    )

    response = await system.process_message(message, user_id="pipeline_tester")
    assert response.is_success, f"process_message failed: {response.error}"

    payload = response.data

    analysis = payload.get("analysis", {})
    complexity = analysis.get("complexity")
    assert complexity, "Complexity analysis missing"
    assert complexity["reasoning_required"] is True
    assert complexity["score"] >= 40

    metacog = analysis.get("metacognition")
    assert metacog, "Metacognition summary missing"
    assert metacog["quality_grade"] is not None
    assert "quality_alerts" in metacog

    reasoning = analysis.get("reasoning", {})
    assert reasoning.get("used") in {True, False}

    assert payload["response"], "Assistant response should not be empty"
