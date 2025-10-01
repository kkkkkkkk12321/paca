import sys
import types
from pathlib import Path

import importlib.metadata as importlib_metadata
import pytest

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

from paca.system import PacaSystem, Message


def test_fallback_response_is_dynamic_and_records_observation():
    system = PacaSystem()
    system.conversation_history.append(Message("이전 대화는 프로젝트 계획이야", "user"))

    response = system._generate_fallback_response({}, "파이썬 비동기 처리 방법 알려줘?")

    assert "파이썬" in response
    assert "대응 계획" in response
    assert "정보/질문" in response
    observations = system.user_context.get("fallback_observations")
    assert observations, "fallback 관찰 메모가 기록되어야 합니다."
    latest = observations[-1]
    assert latest["keywords"], "핵심 키워드를 수집해야 합니다."
    assert "정보/질문" in latest["intents"]


def test_fallback_response_mentions_recent_context():
    system = PacaSystem()
    system.conversation_history.extend(
        [
            Message("첫 번째 요청: 데이터 분석 흐름을 설명해줘", "user"),
            Message("이전 답변", "assistant"),
            Message("두 번째 요청: 모델 정확도 높이는 법", "user"),
        ]
    )

    response = system._generate_fallback_response({}, "배포 자동화를 어떻게 진행해야 할까?")

    assert "이전 흐름" in response
    assert "배포" in response
    observations = system.user_context.get("fallback_observations")
    assert observations and len(observations) >= 1
