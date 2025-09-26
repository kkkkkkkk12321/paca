import pytest

from paca.api.llm.gemini_client import APIKeyManager


def test_api_key_manager_round_robin_rotation():
    manager = APIKeyManager(
        ["k1", "k2", "k3"],
        rotation_strategy="round_robin",
        min_interval_seconds=0.0
    )

    sequence = [manager.get_next_key() for _ in range(6)]
    assert sequence == ["k1", "k2", "k3", "k1", "k2", "k3"]


def test_api_key_manager_add_and_remove():
    manager = APIKeyManager(["k1"], rotation_strategy="round_robin", min_interval_seconds=0.0)
    manager.add_keys(["k2", "k3"])
    assert manager.get_keys() == ["k1", "k2", "k3"]

    manager.mark_key_failed("k1")
    next_key = manager.get_next_key()
    assert next_key in {"k2", "k3"}

    manager.mark_key_success("k1")
    manager.remove_key("k2")
    assert manager.get_keys() == ["k1", "k3"]
    round_trip = [manager.get_next_key() for _ in range(4)]
    assert "k1" in round_trip and "k3" in round_trip
