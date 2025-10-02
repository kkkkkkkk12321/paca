# tests/phase2/test_collaboration_policy.py

import pytest
from types import SimpleNamespace
from paca.cognitive.reasoning_chain import ReasoningChain

class _FakeEngine:
    def __init__(self):
        self.calls = []

    async def run(self, reasoning_type, context=None, meta=None):
        # 호출 기록만 남기고 "성공" 형태의 결과 객체 반환
        self.calls.append((reasoning_type, dict(meta or {})))
        return SimpleNamespace(
            success=True,
            status="completed",
            quality_assessment={"escalation_triggers": []},
        )

@pytest.mark.asyncio
async def test_collaboration_retries_are_invoked_via_helper():
    # 1) 에스컬레이션/정책 ON
    cfg = {
        "escalation": {
            "enabled": True,
            "after_attempts": 1,
            "min_confidence": 0.82,
            "trigger_reasons": ["forced_validation_failure"],
            "collaboration_policy": {
                "forced_validation_failure": {
                    "reasoning_types": ["abductive", "analogical"],
                    "max_attempts": 2,
                },
                "default": {
                    "reasoning_types": ["deductive"],
                    "max_attempts": 1,
                },
            },
        }
    }
    chain = ReasoningChain(config=cfg)

    # 2) 가짜 ReasoningEngine 장착
    fe = _FakeEngine()
    chain.reasoning_engine = fe

    # 3) 협업 재시도 헬퍼 직접 호출 (필수 인자: attempt_history)
    context = {"domain": "ANALYTICAL"}
    attempt_history = []

    # 강제 사유: forced_validation_failure → abductive/analogical 중 시도
    result = await chain._try_collaboration_retries(
        "forced_validation_failure",
        context,
        attempt_history=attempt_history,
    )

    # 4) 검증: 최소 1회 이상 호출되었고, 호출된 추론 타입이 정책에 포함됨
    assert len(fe.calls) >= 1
    rtype, meta = fe.calls[0]
    assert rtype in {"abductive", "analogical"}
    if "collab_reason" in meta:
        assert meta["collab_reason"] == "forced_validation_failure"
    if "attempt" in meta:
        assert isinstance(meta["attempt"], int)
    assert result is not None
    assert result.quality_assessment.get("unresolved_validation") is False

    # 5) 기본(default) 규칙도 확인
    fe.calls.clear()
    result_default = await chain._try_collaboration_retries(
        None,  # reason_key 없음 → default 규칙 사용
        context,
        attempt_history=attempt_history,
    )
    assert len(fe.calls) >= 1
    assert fe.calls[0][0] in {"deductive"}
    assert result_default is not None
