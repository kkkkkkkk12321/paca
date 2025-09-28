import pytest

from paca.config import ConfigManager
from paca.system import PacaSystem


@pytest.mark.asyncio
async def test_config_manager_includes_llm_defaults(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    manager = ConfigManager()
    result = await manager.initialize()
    assert result.is_success

    provider = manager.get_value("default", "llm.provider")
    assert provider == "gemini"

    api_keys = manager.get_value("default", "llm.api_keys")
    assert isinstance(api_keys, list)
    assert api_keys == []

    models = manager.get_value("default", "llm.models.image")
    assert "gemini-2.5-flash-image-preview" in models
    assert "gemini-2.0-flash-preview-image-generation" in models

    rotation_strategy = manager.get_value("default", "llm.rotation.strategy")
    assert rotation_strategy == "round_robin"


@pytest.mark.asyncio
async def test_paca_system_applies_llm_defaults(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    system = PacaSystem()
    await system.config_manager.initialize()
    system._apply_llm_config()

    assert system.config.gemini_api_keys == []
    assert system.config.llm_model_preferences["conversation"][0] == "gemini-2.5-pro"
    assert system.config.llm_rotation_strategy == "round_robin"
    assert system.config.llm_rotation_min_interval == 1.0

    update_result = await system.update_llm_api_keys(["new-key-1", "new-key-2"], persist=False)
    assert update_result.is_success
    assert system.config.gemini_api_keys == ["new-key-1", "new-key-2"]


@pytest.mark.asyncio
async def test_config_manager_prefers_env_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", "env-key-1, env-key-2")
    manager = ConfigManager()
    result = await manager.initialize()
    assert result.is_success

    api_keys = manager.get_value("default", "llm.api_keys")
    assert api_keys == ["env-key-1", "env-key-2"]
