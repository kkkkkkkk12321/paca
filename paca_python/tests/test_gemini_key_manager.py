import pytest

import sys
import types
import importlib.metadata as importlib_metadata
from pathlib import Path

_email_validator = types.ModuleType("email_validator")


class _EmailInfo:
    def __init__(self, email: str):
        self.email = email


def _validate_email(address: str, **_kwargs):
    return _EmailInfo(address)


_email_validator.validate_email = _validate_email  # type: ignore[attr-defined]
_email_validator.EmailNotValidError = ValueError  # type: ignore[attr-defined]
sys.modules.setdefault("email_validator", _email_validator)

_sympy = types.ModuleType("sympy")
_sympy.sympify = lambda expr: expr  # type: ignore[attr-defined]
_sympy.Eq = lambda left, right: ("eq", left, right)  # type: ignore[attr-defined]
_sympy.solve = lambda *_args, **_kwargs: []  # type: ignore[attr-defined]
_sympy.Symbol = lambda name: name  # type: ignore[attr-defined]
_sympy.diff = lambda *_args, **_kwargs: 0  # type: ignore[attr-defined]
_sympy.integrate = lambda *_args, **_kwargs: 0  # type: ignore[attr-defined]
sys.modules.setdefault("sympy", _sympy)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_original_version = importlib_metadata.version


def _fake_version(name: str) -> str:
    if name == "email-validator":
        return "2.0.0"
    return _original_version(name)


importlib_metadata.version = _fake_version  # type: ignore[assignment]

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


def test_api_key_manager_balanced_distribution_many_keys():
    keys = [f"k{i:02d}" for i in range(1, 51)]
    manager = APIKeyManager(keys, rotation_strategy="round_robin", min_interval_seconds=0.0)

    selections = [manager.get_next_key() for _ in range(250)]
    counts = {key: selections.count(key) for key in keys}

    # 모든 키가 거의 동일하게 사용되며 최대 편차는 1을 넘지 않는다
    usage_values = counts.values()
    assert max(usage_values) - min(usage_values) <= 1

    stats = manager.get_usage_statistics()
    assert stats == counts
