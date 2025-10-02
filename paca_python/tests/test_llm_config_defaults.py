import pytest

from paca.config import ConfigManager
from paca.system import PacaSystem


@pytest.mark.asyncio
async def test_config_manager_includes_llm_defaults():
    manager = ConfigManager()
    result = await manager.initialize()
    assert result.is_success

    provider = manager.get_value("default", "llm.provider")
    assert provider == "gemini"

    api_keys = manager.get_value("default", "llm.api_keys")
    assert isinstance(api_keys, list) and len(api_keys) == 3
    assert api_keys == [
        "<<SET_GEMINI_API_KEY_1>>",
        "<<SET_GEMINI_API_KEY_2>>",
        "<<SET_GEMINI_API_KEY_3>>",
    ]

    redacted = manager.get_value("default", "llm.api_keys_redacted")
    assert redacted == api_keys
    assert manager.get_value("default", "llm.contains_placeholder_keys") is True

    models = manager.get_value("default", "llm.models.image")
    assert "gemini-2.5-flash-image-preview" in models
    assert "gemini-2.0-flash-preview-image-generation" in models

    rotation_strategy = manager.get_value("default", "llm.rotation.strategy")
    assert rotation_strategy == "round_robin"


@pytest.mark.asyncio
async def test_paca_system_applies_llm_defaults():
    system = PacaSystem()
    await system.config_manager.initialize()
    system._apply_llm_config()

    assert len(system.config.gemini_api_keys) == 3
    assert system.config.gemini_api_keys == [
        "<<SET_GEMINI_API_KEY_1>>",
        "<<SET_GEMINI_API_KEY_2>>",
        "<<SET_GEMINI_API_KEY_3>>",
    ]
    assert system.config.llm_model_preferences["conversation"][0] == "gemini-2.5-pro"
    assert system.config.llm_rotation_strategy == "round_robin"
    assert system.config.llm_rotation_min_interval == 1.0

    update_result = await system.update_llm_api_keys(["new-key-1", "new-key-2"], persist=False)
    assert update_result.is_success
    assert system.config.gemini_api_keys == ["new-key-1", "new-key-2"]
