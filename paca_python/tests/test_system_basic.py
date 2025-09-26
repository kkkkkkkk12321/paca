import pytest

from paca.system import PacaSystem
from paca.core.types import Status
from paca.core.errors import PacaError


@pytest.mark.asyncio
async def test_system_initializes_successfully():
    system = PacaSystem()

    try:
        result = await system.initialize()
        assert result.is_success
        assert system.is_initialized is True
        assert system.status == Status.READY
    finally:
        await system.cleanup()


@pytest.mark.asyncio
async def test_process_message_returns_basic_response():
    system = PacaSystem()

    try:
        await system.initialize()
        result = await system.process_message("안녕 PACA")

        assert result.is_success
        data = result.data
        assert data is not None
        assert "안녕하세요" in data["response"]
        assert data["confidence"] == pytest.approx(0.8)
    finally:
        await system.cleanup()


@pytest.mark.asyncio
async def test_process_message_rejects_empty_input():
    system = PacaSystem()

    try:
        await system.initialize()
        result = await system.process_message("   ")

        assert result.is_failure
        assert isinstance(result.error, PacaError)
        assert "Empty message" in str(result.error)
    finally:
        await system.cleanup()
