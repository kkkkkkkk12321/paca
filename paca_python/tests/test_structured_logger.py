import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paca.core.utils.logger import StructuredLogger


@pytest.fixture
def dummy_logger(monkeypatch):
    logger = StructuredLogger("TestLogger")

    calls = []

    def _record(message, error, meta):
        calls.append((message, error, meta))

    stub = types.SimpleNamespace(error=_record)
    monkeypatch.setattr(logger, "_inner", stub, raising=False)
    return logger, calls


def test_exception_logs_with_explicit_error(dummy_logger):
    logger, calls = dummy_logger

    exc = RuntimeError("boom")
    logger.exception("failed", error=exc, context="cli")

    assert calls == [("failed", exc, {"context": "cli", "error": "boom"})]


def test_exception_infers_error_from_context(dummy_logger):
    logger, calls = dummy_logger

    try:
        raise ValueError("broken")
    except ValueError:
        logger.exception("caught")

    assert len(calls) == 1
    _, error, meta = calls[0]
    assert isinstance(error, ValueError)
    assert meta == {"error": "broken"}


@pytest.mark.asyncio
async def test_exception_async_alias(dummy_logger):
    logger, calls = dummy_logger

    await logger.exception_async("async", error=KeyError("missing"))

    assert len(calls) == 1
    message, error, meta = calls[0]
    assert message == "async"
    assert isinstance(error, KeyError)
    assert isinstance(meta, dict)
    assert "missing" in meta.get("error", "")
