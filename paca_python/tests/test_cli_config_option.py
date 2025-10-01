import asyncio
import json
import sys
import types
from pathlib import Path

import pytest
import importlib.metadata as importlib_metadata

# Optional dependencies stubbed so importing paca.__main__ does not fail in CI environments.
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

from paca.core.types.base import Result
from paca import __main__ as cli


class DummyPacaSystem:
    def __init__(self, config):
        self.config = config
        self.initialized = False
        self.cleaned_up = False
        self.seen_messages = []

    async def initialize(self):
        self.initialized = True
        return Result.success(True)

    async def process_message(self, message: str):
        self.seen_messages.append(message)
        return Result.success({
            "response": f"echo:{message}",
            "processing_time": 0.01,
            "confidence": getattr(self.config, "reasoning_confidence_threshold", 0.0),
        })

    async def cleanup(self):
        self.cleaned_up = True


@pytest.fixture
def captured_system(monkeypatch):
    instances = []

    def _build(config):
        instance = DummyPacaSystem(config)
        instances.append(instance)
        return instance

    monkeypatch.setattr(cli, "PacaSystem", _build)
    return instances


def test_cli_applies_config_overrides(tmp_path: Path, monkeypatch, captured_system):
    overrides = {
        "reasoning_confidence_threshold": 0.42,
        "backtrack_confidence_threshold": 0.35,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(overrides), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["paca", "--config", str(config_path), "--message", "ping"])

    asyncio.run(cli.main_async())

    assert captured_system, "CLI did not instantiate PacaSystem"
    cfg = captured_system[0].config
    assert cfg.reasoning_confidence_threshold == pytest.approx(0.42)
    assert cfg.backtrack_confidence_threshold == pytest.approx(0.35)


def test_policy_thresholds_are_respected(monkeypatch, captured_system):
    def _fake_policy():
        return {
            "reasoning_confidence_threshold": 0.55,
            "backtrack_confidence_threshold": 0.45,
            "strategy_switch_confidence_threshold": 0.65,
            "policy": {"low_confidence_threshold": 0.52},
            "escalation": {"min_confidence": 0.6},
        }

    monkeypatch.setattr("paca.cognitive._collab_policy_loader.load_policy", _fake_policy)
    monkeypatch.setattr(sys, "argv", ["paca", "--message", "hello"])

    asyncio.run(cli.main_async())

    assert captured_system, "CLI did not instantiate PacaSystem"
    cfg = captured_system[0].config
    assert cfg.reasoning_confidence_threshold == pytest.approx(0.55)
    assert cfg.backtrack_confidence_threshold == pytest.approx(0.45)
    assert cfg.strategy_switch_confidence_threshold == pytest.approx(0.65)
    assert cfg.policy["low_confidence_threshold"] == pytest.approx(0.52)
    assert cfg.escalation["min_confidence"] == pytest.approx(0.6)
