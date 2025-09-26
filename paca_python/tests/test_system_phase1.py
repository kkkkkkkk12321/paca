import pytest

from paca.system import PacaSystem
from paca.core.types import Status


@pytest.mark.asyncio
async def test_basic_response_patterns():
    system = PacaSystem()

    scenarios = [
        ("안녕 PACA", "안녕하세요!"),
        ("정말 고마워요", "천만에요"),
        ("파이썬 공부 도와줘", "학습에 대해"),
    ]

    try:
        await system.config_manager.initialize()
        system.config_manager.set_value("default", "llm.api_keys", [])
        await system.initialize()

        for user_input, expected_phrase in scenarios:
            result = await system.process_message(user_input)
            assert result.is_success
            message = result.data["response"]
            assert expected_phrase in message
    finally:
        await system.cleanup()


@pytest.mark.asyncio
async def test_llm_api_key_management_cycle():
    system = PacaSystem()

    try:
        await system.config_manager.initialize()
        system.config_manager.set_value("default", "llm.api_keys", [])
        await system.initialize()
        assert system.status == Status.READY
        original_keys = list(system.config.gemini_api_keys)

        update_result = await system.update_llm_api_keys(["test-key-1", "test-key-2"], persist=False)
        assert update_result.is_success
        assert system.config.gemini_api_keys == ["test-key-1", "test-key-2"]

        add_result = await system.add_llm_api_key("additional-key", persist=False)
        assert add_result.is_success
        assert "additional-key" in system.config.gemini_api_keys

        remove_result = await system.remove_llm_api_key("test-key-1", persist=False)
        assert remove_result.is_success
        assert "test-key-1" not in system.config.gemini_api_keys

        # 원래 키 복구 (필요 시)
        if original_keys:
            await system.update_llm_api_keys(original_keys, persist=False)
    finally:
        await system.cleanup()
