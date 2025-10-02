"""Tests for CLI exception logging behaviour."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paca.core.utils import logger as logger_module


def test_structured_logger_exception_uses_error_and_stack(monkeypatch):
    """StructuredLogger.exception should forward the active exception."""

    error_calls = []
    log_calls = []

    def fake_error(self, message: str, error: Exception = None, meta: Optional[Dict[str, Any]] = None):
        error_calls.append((message, error, meta))

    def fake_log(self, level, message: str, metadata: Optional[Dict[str, Any]] = None, stack: Optional[str] = None):
        log_calls.append((level, message, metadata, stack))

    monkeypatch.setattr(logger_module.PacaLogger, "error", fake_error, raising=False)
    monkeypatch.setattr(logger_module.PacaLogger, "_log", fake_log, raising=False)

    structured_logger = logger_module.StructuredLogger("TestLogger")

    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        structured_logger.exception("runtime failed", context="unit-test")

        assert error_calls, "Expected StructuredLogger to forward the exception to PacaLogger.error"
        message, error, meta = error_calls[0]
        assert message == "runtime failed"
        assert error is exc
        assert meta == {"context": "unit-test"}
        assert not log_calls, "Stack logging should not be used when an exception is active"

    error_calls.clear()
    log_calls.clear()

    structured_logger.exception("no active exception")

    assert not error_calls, "error() should not be invoked when there is no active exception"
    assert log_calls, "Expected StructuredLogger to emit stack information via _log"
    level, message, metadata, stack_trace = log_calls[0]
    assert level == logger_module.LogLevel.ERROR
    assert message == "no active exception"
    assert metadata is None
    assert isinstance(stack_trace, str) and stack_trace.strip()


@pytest.mark.asyncio
async def test_main_async_logs_exception_and_exits(monkeypatch):
    """Ensure CLI surfaces exceptions through the new logger helper."""

    from paca import __main__ as cli

    calls = []

    def fake_error(self, message: str, error: Exception = None, meta: Optional[Dict[str, Any]] = None):
        calls.append((message, error, meta))

    monkeypatch.setattr(logger_module.PacaLogger, "error", fake_error, raising=False)

    class DummyParser:
        def parse_args(self):
            return argparse.Namespace(
                message=None,
                interactive=False,
                gui=False,
                config=None,
                debug=False,
                log_level="INFO",
            )

        def print_help(self):  # pragma: no cover - not triggered in this test
            pass

        def error(self, msg: str) -> None:  # pragma: no cover - defensive
            raise SystemExit(msg)

    monkeypatch.setattr(cli, "create_parser", lambda: DummyParser())

    class Boom(RuntimeError):
        """Custom exception used to trigger the error path."""

    def raise_boom(*_args, **_kwargs):
        raise Boom("config failure")

    monkeypatch.setattr(cli, "_build_runtime_config", raise_boom)

    with pytest.raises(SystemExit) as exc_info:
        await cli.main_async()

    assert exc_info.value.code == 1
    assert calls, "Expected the logger to be invoked for the CLI exception"
    message, error, meta = calls[0]
    assert message == "메인 실행 중 오류"
    assert isinstance(error, Boom)
    assert str(error) == "config failure"
    assert meta is None
