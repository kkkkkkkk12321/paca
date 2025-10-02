import pytest

from paca.cognitive.reasoning_chain import (
    ReasoningChain,
    ReasoningStatus,
    ReasoningStrategy,
)
from paca.cognitive.complexity_detector import DomainType
from paca.core.types import Result
from paca.reasoning.base import ReasoningResult as EngineReasoningResult, ReasoningType


class _StubReasoningEngine:
    async def initialize(self):
        return Result.success(True)

    async def reason(self, reasoning_type, premises, target_conclusion=None, **kwargs):
        outcome = EngineReasoningResult(
            context_id="stub",
            conclusion=f"{target_conclusion or 'stub-conclusion'} (escalated)",
            confidence=0.91,
            reasoning_steps=[],
            execution_time_ms=5.0,
            is_valid=True,
            metadata={}
        )
        return Result.success(outcome)


class _TrackingReasoningEngine(_StubReasoningEngine):
    def __init__(self):
        super().__init__()
        self.calls = []

    async def reason(self, reasoning_type, premises, target_conclusion=None, **kwargs):
        label = reasoning_type.value if isinstance(reasoning_type, ReasoningType) else str(reasoning_type)
        self.calls.append(label)
        return await super().reason(reasoning_type, premises, target_conclusion=target_conclusion, **kwargs)


@pytest.mark.asyncio
async def test_reasoning_chain_triggers_backtracking_on_validation_failure():
    chain = ReasoningChain({
        "enable_backtracking": True,
        "force_validation_failure": True,
        "max_backtrack_attempts": 2,
        "max_sequential_steps": 8,
        "strategy_fallback_order": [ReasoningStrategy.SEQUENTIAL.value],
        "max_strategy_attempts": 1,
    })

    result = await chain.execute_reasoning_chain(
        "데이터 분석 중 오류를 복구하는 절차를 단계별로 설명해줘",
        complexity_score=70,
        context={"domain": DomainType.ANALYTICAL},
    )

    assert result.status == ReasoningStatus.COMPLETED
    assert result.backtrack_attempts >= 1
    assert result.backtrack_attempts == result.backtrack_successes + result.backtrack_failures
    assert result.backtrack_summary, "백트래킹 요약이 비어 있습니다"
    assert result.alternative_solutions, "백트래킹 대안 경로가 없습니다"
    assert any(step.metadata.get("backtrack") for step in result.steps if step.metadata)


@pytest.mark.asyncio
async def test_reasoning_chain_skips_backtracking_when_not_needed():
    chain = ReasoningChain({
        "enable_backtracking": True,
        "max_sequential_steps": 6,
        "strategy_fallback_order": [ReasoningStrategy.SEQUENTIAL.value],
        "max_strategy_attempts": 1,
    })

    result = await chain.execute_reasoning_chain(
        "간단한 절차를 점검하는 테스트",
        complexity_score=35,
        context={"domain": DomainType.LOGICAL},
    )

    assert result.status == ReasoningStatus.COMPLETED
    assert result.backtrack_attempts == 0
    assert not result.backtrack_summary
    assert not result.alternative_solutions
    assert not any(step.metadata.get("backtrack") for step in result.steps if step.metadata)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "domain,score,expected_strategy",
    [
        (DomainType.CREATIVE, 70, ReasoningStrategy.PARALLEL),
        (DomainType.TECHNICAL, 85, ReasoningStrategy.HIERARCHICAL),
        (DomainType.CONVERSATIONAL, 45, ReasoningStrategy.ITERATIVE),
    ],
)
async def test_backtracking_triggers_for_non_sequential_strategies(domain, score, expected_strategy):
    chain = ReasoningChain({
        "enable_backtracking": True,
        "force_validation_failure": True,
        "max_backtrack_attempts": 2,
        "max_parallel_branches": 3,
        "max_iterations": 4,
        "strategy_fallback_order": [
            ReasoningStrategy.PARALLEL.value,
            ReasoningStrategy.HIERARCHICAL.value,
            ReasoningStrategy.ITERATIVE.value,
            ReasoningStrategy.SEQUENTIAL.value
        ],
        "max_strategy_attempts": 4,
    })

    result = await chain.execute_reasoning_chain(
        "복잡한 시나리오에 대한 해결 전략을 단계별로 설명해줘",
        complexity_score=score,
        context={"domain": domain},
    )

    history = result.quality_assessment.get("strategy_history", [])
    assert history, "전략 기록이 존재해야 합니다"
    assert history[0]["strategy"] == expected_strategy.value
    assert history[0]["backtrack_attempts"] >= 1
    assert any(entry["backtrack_attempts"] >= 1 for entry in history)
    assert history[0]["alternative_solutions"]


@pytest.mark.asyncio
async def test_backtracking_disabled_even_on_failure():
    chain = ReasoningChain({
        "enable_backtracking": False,
        "force_validation_failure": True,
        "max_sequential_steps": 8,
        "strategy_fallback_order": [ReasoningStrategy.SEQUENTIAL.value],
        "max_strategy_attempts": 1,
    })

    result = await chain.execute_reasoning_chain(
        "절차 점검 테스트",
        complexity_score=70,
        context={"domain": DomainType.ANALYTICAL},
    )

    history = result.quality_assessment.get("strategy_history", [])
    assert history
    assert history[0]["strategy"] == ReasoningStrategy.SEQUENTIAL.value
    assert history[0]["backtrack_attempts"] == 0
    assert all(entry["backtrack_attempts"] == 0 for entry in history)


@pytest.mark.asyncio
async def test_strategy_fallback_triggers_when_recovery_fails():
    chain = ReasoningChain({
        "enable_backtracking": True,
        "force_validation_failure": True,
        "max_backtrack_attempts": 1,
        "strategy_fallback_order": [
            ReasoningStrategy.SEQUENTIAL.value,
            ReasoningStrategy.HIERARCHICAL.value
        ],
        "max_strategy_attempts": 2,
        "strategy_switch_rules": {"require_successful_backtrack": True},
    })

    result = await chain.execute_reasoning_chain(
        "창의적 문제를 해결할 전략을 단계별로 설명해줘",
        complexity_score=80,
        context={"domain": DomainType.CREATIVE},
    )

    history = result.quality_assessment.get("strategy_history", [])

    assert result.strategy_used != ReasoningStrategy.PARALLEL
    assert len(history) >= 2
    assert history[0]["strategy"] == ReasoningStrategy.PARALLEL.value
    collab_attempts = result.quality_assessment.get("collaboration_attempts", [])
    if collab_attempts:
        assert history[0]["unresolved_validation"] is False
        assert any(entry["status"] == "collaboration_success" for entry in collab_attempts)
    else:
        assert history[0]["unresolved_validation"] is True
    assert any(entry["strategy"] == ReasoningStrategy.HIERARCHICAL.value for entry in history[1:])


@pytest.mark.asyncio
async def test_reasoning_engine_escalation_records_additional_attempt():
    chain = ReasoningChain({
        "enable_backtracking": True,
        "force_validation_failure": True,
        "max_backtrack_attempts": 1,
        "strategy_fallback_order": [ReasoningStrategy.SEQUENTIAL.value],
        "max_strategy_attempts": 1,
        "strategy_switch_rules": {
            "require_successful_backtrack": True,
            "max_unresolved_attempts": 1
        },
        "escalation": {
            "enabled": True,
            "after_attempts": 1,
            "min_confidence": 0.85,
            "reasoning_type": "deductive"
        }
    }, reasoning_engine=_StubReasoningEngine())

    result = await chain.execute_reasoning_chain(
        "복잡한 의사결정 전략을 설명해줘",
        complexity_score=85,
        context={"domain": DomainType.TECHNICAL},
    )

    history = result.quality_assessment.get("strategy_history", [])
    collab_attempts = result.quality_assessment.get("collaboration_attempts", [])
    assert collab_attempts or any(entry["strategy"] == "reasoning_engine" for entry in history)
    if collab_attempts:
        assert any(entry["status"] in {"collaboration_success", "collaboration_failed"} for entry in collab_attempts)
    else:
        assert result.quality_assessment.get("reasoning_engine")
    assert result.confidence_score >= 0.85


@pytest.mark.asyncio
async def test_collaboration_policy_retries_on_backtrack_failure():
    tracking_engine = _TrackingReasoningEngine()
    chain = ReasoningChain({
        "enable_backtracking": True,
        "force_validation_failure": True,
        "max_backtrack_attempts": 1,
        "strategy_fallback_order": [ReasoningStrategy.SEQUENTIAL.value],
        "max_strategy_attempts": 1,
        "escalation": {
            "enabled": True,
            "after_attempts": 1,
            "min_confidence": 0.8,
            "trigger_reasons": ["forced_validation_failure"],
            "collaboration_policy": {
                "forced_validation_failure": {
                    "reasoning_types": ["abductive", "analogical"],
                    "max_attempts": 2
                },
                "default": {
                    "reasoning_types": ["deductive"],
                    "max_attempts": 1
                }
            }
        }
    }, reasoning_engine=tracking_engine)

    result = await chain.execute_reasoning_chain(
        "검증 단계가 반복해서 실패하는 경우 어떻게 복구할 수 있을까?",
        complexity_score=75,
        context={"domain": DomainType.TECHNICAL},
    )

    collab_attempts = result.quality_assessment.get("collaboration_attempts", [])
    assert collab_attempts, "협업 재시도 기록이 품질 평가에 포함되어야 합니다"
    assert any(entry["status"] == "collaboration_success" for entry in collab_attempts)
    assert result.quality_assessment.get("unresolved_validation") is False
    assert tracking_engine.calls, "ReasoningEngine이 협업 재시도 동안 호출되어야 합니다"
    assert any(call in {"abductive", "analogical", "deductive"} for call in tracking_engine.calls)


@pytest.mark.asyncio
async def test_low_confidence_switches_strategy(monkeypatch):
    call_count = {"value": 0}

    async def low_confidence_conclusion(self, steps):
        call_count["value"] += 1
        return "저신뢰 결론", 0.5

    monkeypatch.setattr(ReasoningChain, "_synthesize_conclusion", low_confidence_conclusion)

    chain = ReasoningChain({
        "strategy_switch_policy": {
            "low_confidence_threshold": 0.9,
            "consecutive_low_confidence_limit": 1
        },
        "strategy_fallback_order": [ReasoningStrategy.SEQUENTIAL.value],
        "max_strategy_attempts": 2,
    })

    result = await chain.execute_reasoning_chain(
        "저신뢰 결과를 유도하는 테스트",
        complexity_score=65,
        context={"domain": DomainType.LOGICAL},
    )

    history = result.quality_assessment.get("strategy_history", [])
    assert len(history) >= 2
    assert 'low_confidence' in history[0].get("switch_reasons", [])
    assert history[0]["strategy"] == ReasoningStrategy.HIERARCHICAL.value
    assert history[1]["strategy"] == ReasoningStrategy.SEQUENTIAL.value
    assert call_count["value"] >= 2


@pytest.mark.asyncio
async def test_escalation_triggers_when_configured_reason(monkeypatch):
    chain = ReasoningChain({
        "enable_backtracking": True,
        "force_validation_failure": True,
        "max_backtrack_attempts": 1,
        "strategy_fallback_order": [ReasoningStrategy.SEQUENTIAL.value],
        "max_strategy_attempts": 1,
        "escalation": {
            "enabled": True,
            "after_attempts": 5,
            "min_confidence": 0.95,
            "reasoning_type": "deductive",
            "trigger_reasons": ["forced_validation_failure"]
        }
    }, reasoning_engine=_StubReasoningEngine())

    result = await chain.execute_reasoning_chain(
        "강제 검증 실패로 에스컬레이션을 유도",
        complexity_score=85,
        context={"domain": DomainType.TECHNICAL},
    )

    history = result.quality_assessment.get("strategy_history", [])
    assert any(entry["strategy"] == "reasoning_engine" for entry in history)
    triggers = result.quality_assessment.get("escalation_triggers", [])
    assert "forced_validation_failure" in triggers
    assert 'forced_validation_failure' in history[0].get("switch_reasons", [])
