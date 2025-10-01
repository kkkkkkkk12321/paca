import asyncio
from types import SimpleNamespace

import pytest

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import importlib.metadata as importlib_metadata
import types

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

importlib_metadata.version = _fake_version

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paca.api.llm import gemini_client
from paca.api.llm.base import GenerationConfig, LLMRequest, ModelType
from paca.api.llm.gemini_client import GeminiClientManager, GeminiConfig
from paca.api.llm.response_processor import ContextManager


@pytest.fixture
def stubbed_genai(monkeypatch):
    """Patch google-genai client and asyncio.to_thread to capture payloads."""
    captured = {}

    class DummyModel:
        def generate_content(self, **kwargs):
            captured.clear()
            captured.update(kwargs)
            return SimpleNamespace(
                text="stub-response",
                usage_metadata=SimpleNamespace(total_tokens=42),
                candidates=[object()],
                finish_reason="STOP",
            )

    class DummyClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.models = DummyModel()

    monkeypatch.setattr(gemini_client, "genai", SimpleNamespace(Client=DummyClient))

    async def fake_to_thread(fn, /, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(gemini_client.asyncio, "to_thread", fake_to_thread)

    return captured


@pytest.mark.asyncio
async def test_generate_text_includes_system_prompt_and_context(stubbed_genai):
    manager = GeminiClientManager(
        GeminiConfig(
            api_keys=["test-key"],
            enable_caching=False,
            safety_settings={"HARM_CATEGORY_HARASSMENT": "BLOCK_NONE"},
        )
    )
    manager.is_initialized = True

    request = LLMRequest(
        prompt="현재 질문",
        system_prompt="시스템 지침",
        model=ModelType.GEMINI_FLASH,
        config=GenerationConfig(temperature=0.3, max_tokens=128),
        context={

            "prior_messages": [
                {"role": "user", "content": "사전 질문"},
                {"role": "assistant", "content": "사전 답변"},
            ],

            "recent_history": [
                {"user_input": "첫 번째 질문", "assistant_response": "첫 번째 답변"},
            ],
            "context_summary": "이전 대화의 핵심 요약",
            "session_context": {"topic": "과학"},
            "user_preferences": {"tone": "친절하게"},
            "long_term_summary": "오랜 대화 요약",

        },
    )

    result = await manager.generate_text(request)

    assert result.is_success
    payload = stubbed_genai
    assert payload["system_instruction"] == "시스템 지침"
    assert payload["config"] == request.config.to_dict()

    contents = payload["contents"]
    assert contents[0]["parts"][0]["text"] == "사전 질문"
    assert contents[1]["parts"][0]["text"] == "사전 답변"
    assert any(
        item["parts"][0]["text"] == "첫 번째 질문"
        for item in contents
    )

    assert any(
        item["role"] == "model" and item["parts"][0]["text"] == "첫 번째 답변"
        for item in contents
    )
    summary_entry = next(
        (item for item in contents if item["parts"][0]["text"].startswith("[대화 요약]")),
        None,
    )
    assert summary_entry is not None
    long_term_entry = next(
        (item for item in contents if item["parts"][0]["text"].startswith("[장기 요약]")),
        None,
    )
    assert long_term_entry is not None
    assert "오랜 대화 요약" in long_term_entry["parts"][0]["text"]

    assert contents[-1]["parts"][0]["text"] == "현재 질문"


@pytest.mark.asyncio
async def test_generate_with_context_uses_prior_messages(stubbed_genai):
    manager = GeminiClientManager(
        GeminiConfig(api_keys=["another-key"], enable_caching=False)
    )
    manager.is_initialized = True

    context_messages = [
        {"role": "user", "content": "과제를 도와줘"},
        {"role": "assistant", "content": "어떤 과제인지 알려줘"},
    ]

    result = await manager.generate_with_context(
        "수학 문제를 풀고 싶어",
        context_messages,
        model=ModelType.GEMINI_PRO,
        config=GenerationConfig(temperature=0.6, max_tokens=256),
    )

    assert result.is_success
    payload = stubbed_genai
    assert payload["contents"][0]["parts"][0]["text"] == "과제를 도와줘"
    assert payload["contents"][1]["parts"][0]["text"] == "어떤 과제인지 알려줘"
    assert payload["contents"][-1]["parts"][0]["text"] == "수학 문제를 풀고 싶어"


@pytest.mark.asyncio
async def test_context_summary_includes_assistant_responses():
    manager = ContextManager()

    await manager.add_exchange("사용자 질문", "어시스턴트 답변")
    summary = await manager._generate_context_summary()

    assert "사용자: 사용자 질문" in summary
    assert "PACA: 어시스턴트 답변" in summary
