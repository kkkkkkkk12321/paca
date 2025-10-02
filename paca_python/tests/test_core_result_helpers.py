from pathlib import Path
import sys
import types
import importlib.util

import pytest
import importlib.metadata as importlib_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

_original_version = importlib_metadata.version


def _fake_version(name: str) -> str:
    if name == "email-validator":
        return "2.0.0"
    return _original_version(name)


importlib_metadata.version = _fake_version  # type: ignore[assignment]

from paca.core.types import Result, create_failure, create_result, create_success

_legacy_events_path = PROJECT_ROOT / "paca" / "core" / "events.py"

spec = importlib.util.spec_from_file_location(
    "paca.core.events_legacy", _legacy_events_path
)
events_legacy = importlib.util.module_from_spec(spec)
sys.modules.setdefault("paca.core.events_legacy", events_legacy)
assert spec.loader is not None
spec.loader.exec_module(events_legacy)

EventBus = events_legacy.EventBus


@pytest.mark.asyncio
async def test_event_bus_emit_returns_success_result():
    bus = EventBus()
    captured = {}

    async def handler(payload):
        captured["payload"] = payload

    bus.on("demo", handler)

    result = await bus.emit("demo", {"message": "hello"})

    assert result.is_success
    assert result.data == 1
    assert result.metadata == {}
    assert captured["payload"] == {"message": "hello"}


def test_result_helpers_accept_metadata_and_strings():
    failure = create_failure("nope", metadata={"code": "X"})
    assert failure.is_failure
    assert failure.error == "nope"
    assert failure.metadata == {"code": "X"}

    success = create_success({"ok": True}, metadata={"source": "unit"})
    assert success.is_success
    assert success.data == {"ok": True}
    assert success.metadata == {"source": "unit"}

    custom = create_result(True, data="hi", metadata={"note": "custom"})
    assert custom.is_success
    assert custom.metadata == {"note": "custom"}

    failure_from_helper = create_result(
        False, data=None, error="boom", metadata={"step": "emit"}
    )
    assert failure_from_helper.is_failure
    assert failure_from_helper.error == "boom"
    assert failure_from_helper.metadata == {"step": "emit"}

    exc = ValueError("bad")
    failure_from_cls = Result.failure(exc, metadata={"stage": "test"})
    assert failure_from_cls.is_failure
    assert failure_from_cls.error is exc
    assert failure_from_cls.metadata == {"stage": "test"}

    success_from_cls = Result.success("done", metadata={"stage": "test"})
    assert success_from_cls.is_success
    assert success_from_cls.data == "done"
    assert success_from_cls.metadata == {"stage": "test"}
