"""
Reasoning Chains Module
추론 체인 관리 시스템
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass, field

from ...core.types.base import ID, Result
from ...core.errors.base import ValidationError, SystemError
from ...core.utils.async_utils import create_logger
from ...core.events.base import EventEmitter


class ReasoningStepType(Enum):
    """추론 단계 타입"""
    PREMISE = "premise"
    INFERENCE = "inference"
    CONCLUSION = "conclusion"
    ASSUMPTION = "assumption"
    EVIDENCE = "evidence"
    CONTRADICTION = "contradiction"


class ReasoningMethod(Enum):
    """추론 방법"""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"


@dataclass
class ReasoningStep:
    """추론 단계"""
    id: ID
    type: ReasoningStepType
    content: Any
    method: ReasoningMethod
    confidence: float
    premises: List[ID]
    conclusions: List[ID]
    created_at: int
    evidence: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = int(time.time() * 1000)


@dataclass
class SubGoal:
    """서브 골"""
    id: ID
    description: str
    status: str  # 'pending', 'in_progress', 'completed', 'failed'
    priority: int
    dependencies: List[ID]
    estimated_effort: int
    created_at: int
    updated_at: int
    actual_effort: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = int(time.time() * 1000)
        if not self.updated_at:
            self.updated_at = self.created_at


@dataclass
class ReasoningChain:
    """추론 체인"""
    id: ID
    name: str
    steps: List[ReasoningStep]
    start_premise: ID
    method: ReasoningMethod
    confidence: float
    is_valid: bool
    created_at: int
    updated_at: int
    description: Optional[str] = None
    final_conclusion: Optional[ID] = None
    alternatives: Optional[List['ReasoningChain']] = None
    total_estimated_steps: Optional[int] = None
    subgoals: Optional[List[SubGoal]] = None
    completed_subgoals: Optional[List[ID]] = None
    input_data: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = int(time.time() * 1000)
        if not self.updated_at:
            self.updated_at = self.created_at


@dataclass
class ValidationResult:
    """검증 결과"""
    step_id: ID
    is_valid: bool
    issues: List[str]
    suggestions: List[str]


@dataclass
class ReasoningChainResult:
    """추론 체인 결과"""
    chain_id: ID
    success: bool
    confidence: float
    step_count: int
    execution_time: int
    validation_results: List[ValidationResult]
    final_conclusion: Optional[Any] = None
    error: Optional[Exception] = None


class ReasoningChainManager(EventEmitter):
    """추론 체인 관리자"""

    def __init__(self):
        super().__init__()
        self.logger = create_logger("ReasoningChain")
        self.chains: Dict[ID, ReasoningChain] = {}
        self.steps: Dict[ID, ReasoningStep] = {}
        self.chain_results: Dict[ID, ReasoningChainResult] = {}

    async def create_chain(
        self,
        name: str,
        start_premise: Any,
        method: ReasoningMethod,
        options: Optional[Dict[str, Any]] = None
    ) -> Result:
        """새로운 추론 체인 생성"""
        try:
            options = options or {}
            chain_id = f"chain_{int(time.time() * 1000)}_{uuid.uuid4().hex[:9]}"
            now = int(time.time() * 1000)

            # 시작 전제 단계 생성
            start_step_id = f"step_{int(time.time() * 1000)}_{uuid.uuid4().hex[:9]}"
            start_step = ReasoningStep(
                id=start_step_id,
                type=ReasoningStepType.PREMISE,
                content=start_premise,
                method=method,
                confidence=1.0,
                premises=[],
                conclusions=[],
                metadata={},
                created_at=now
            )

            self.steps[start_step_id] = start_step

            chain = ReasoningChain(
                id=chain_id,
                name=name,
                description=options.get("description"),
                steps=[start_step],
                start_premise=start_step_id,
                method=method,
                confidence=1.0,
                is_valid=True,
                metadata=options.get("metadata", {}),
                created_at=now,
                updated_at=now
            )

            self.chains[chain_id] = chain

            await self.emit_event("chain:created", {
                "chain_id": chain_id,
                "name": name,
                "method": method.value
            })

            self.logger.info("Reasoning chain created", {
                "chain_id": chain_id,
                "name": name,
                "method": method.value
            })

            return Result(success=True, data=chain)

        except Exception as error:
            return Result(
                success=False,
                data=None,
                error=SystemError(
                    f"Failed to create reasoning chain: {str(error)}",
                    context={"name": name, "method": method, "error": str(error)}
                )
            )

    async def add_step(
        self,
        chain_id: ID,
        step_type: ReasoningStepType,
        content: Any,
        premises: List[ID],
        confidence: float,
        options: Optional[Dict[str, Any]] = None
    ) -> Result:
        """추론 단계 추가"""
        try:
            options = options or {}
            chain = self.chains.get(chain_id)
            if not chain:
                return Result(
                    success=False,
                    data=None,
                    error=ValidationError(
                        "Reasoning chain not found",
                        context={"chain_id": chain_id}
                    )
                )

            # 전제 단계들이 존재하는지 확인
            for premise_id in premises:
                if premise_id not in self.steps:
                    return Result(
                        success=False,
                        data=None,
                        error=ValidationError(
                            "Premise step not found",
                            context={"chain_id": chain_id, "premise_id": premise_id}
                        )
                    )

            step_id = f"step_{int(time.time() * 1000)}_{uuid.uuid4().hex[:9]}"
            step = ReasoningStep(
                id=step_id,
                type=step_type,
                content=content,
                method=chain.method,
                confidence=confidence,
                premises=premises,
                conclusions=[],
                evidence=options.get("evidence"),
                metadata=options.get("metadata", {}),
                created_at=int(time.time() * 1000)
            )

            self.steps[step_id] = step

            # 체인 업데이트
            updated_steps = chain.steps + [step]
            chain.steps = updated_steps
            chain.confidence = self._calculate_chain_confidence(updated_steps)
            chain.updated_at = int(time.time() * 1000)

            # 전제 단계들의 결론에 현재 단계 추가
            for premise_id in premises:
                premise_step = self.steps[premise_id]
                premise_step.conclusions.append(step_id)

            # 결론 단계인 경우 최종 결론으로 설정
            if step_type == ReasoningStepType.CONCLUSION:
                chain.final_conclusion = step_id

            await self.emit_event("chain:step_added", {
                "chain_id": chain_id,
                "step_id": step_id,
                "step_type": step_type.value,
                "confidence": confidence
            })

            self.logger.debug("Reasoning step added", {
                "chain_id": chain_id,
                "step_id": step_id,
                "step_type": step_type.value,
                "confidence": confidence
            })

            return Result(success=True, data=step)

        except Exception as error:
            return Result(
                success=False,
                data=None,
                error=ValidationError(
                    f"Failed to add reasoning step: {str(error)}",
                    context={"chain_id": chain_id, "step_type": step_type, "error": str(error)}
                )
            )

    async def execute_chain(self, chain_id: ID) -> Result:
        """추론 체인 실행"""
        try:
            chain = self.chains.get(chain_id)
            if not chain:
                return Result(
                    success=False,
                    data=None,
                    error=SystemError(
                        "Reasoning chain not found",
                        context={"chain_id": chain_id}
                    )
                )

            start_time = int(time.time() * 1000)

            # 체인 검증
            validation_results = await self._validate_chain(chain)
            is_valid = all(result.is_valid for result in validation_results)

            if not is_valid:
                result = ReasoningChainResult(
                    chain_id=chain_id,
                    success=False,
                    confidence=0.0,
                    step_count=len(chain.steps),
                    execution_time=int(time.time() * 1000) - start_time,
                    validation_results=validation_results,
                    error=Exception("Chain validation failed")
                )

                self.chain_results[chain_id] = result
                return Result(success=True, data=result)

            # 추론 실행
            final_conclusion = None
            if chain.final_conclusion:
                conclusion_step = self.steps.get(chain.final_conclusion)
                if conclusion_step:
                    final_conclusion = conclusion_step.content

            result = ReasoningChainResult(
                chain_id=chain_id,
                success=True,
                final_conclusion=final_conclusion,
                confidence=chain.confidence,
                step_count=len(chain.steps),
                execution_time=int(time.time() * 1000) - start_time,
                validation_results=validation_results
            )

            self.chain_results[chain_id] = result

            await self.emit_event("chain:executed", {
                "chain_id": chain_id,
                "success": True,
                "confidence": result.confidence
            })

            self.logger.info("Reasoning chain executed", {
                "chain_id": chain_id,
                "success": True,
                "confidence": result.confidence,
                "step_count": result.step_count
            })

            return Result(success=True, data=result)

        except Exception as error:
            result = ReasoningChainResult(
                chain_id=chain_id,
                success=False,
                confidence=0.0,
                step_count=0,
                execution_time=0,
                validation_results=[],
                error=error
            )

            self.chain_results[chain_id] = result

            return Result(
                success=False,
                data=None,
                error=SystemError(
                    f"Failed to execute reasoning chain: {str(error)}",
                    context={"chain_id": chain_id, "error": str(error)}
                )
            )

    async def _validate_chain(self, chain: ReasoningChain) -> List[ValidationResult]:
        """체인 검증"""
        validation_results = []

        for step in chain.steps:
            issues = []
            suggestions = []

            # 신뢰도 검증
            if step.confidence < 0 or step.confidence > 1:
                issues.append("Invalid confidence value")
                suggestions.append("Confidence should be between 0 and 1")

            # 전제 검증
            if step.type != ReasoningStepType.PREMISE and not step.premises:
                issues.append("Non-premise steps must have at least one premise")
                suggestions.append("Add appropriate premises for this reasoning step")

            # 내용 검증
            if not step.content:
                issues.append("Step content is empty")
                suggestions.append("Provide meaningful content for this step")

            validation_results.append(ValidationResult(
                step_id=step.id,
                is_valid=len(issues) == 0,
                issues=issues,
                suggestions=suggestions
            ))

        return validation_results

    def _calculate_chain_confidence(self, steps: List[ReasoningStep]) -> float:
        """체인 신뢰도 계산"""
        if not steps:
            return 0.0

        # 가장 낮은 신뢰도를 기반으로 전체 신뢰도 계산
        min_confidence = min(step.confidence for step in steps)
        avg_confidence = sum(step.confidence for step in steps) / len(steps)

        # 가중 평균 (최소값 70%, 평균값 30%)
        return min_confidence * 0.7 + avg_confidence * 0.3

    # 조회 메서드들
    def get_chain(self, chain_id: ID) -> Optional[ReasoningChain]:
        """체인 조회"""
        return self.chains.get(chain_id)

    def get_all_chains(self) -> List[ReasoningChain]:
        """모든 체인 조회"""
        return list(self.chains.values())

    def get_chain_result(self, chain_id: ID) -> Optional[ReasoningChainResult]:
        """체인 결과 조회"""
        return self.chain_results.get(chain_id)

    def get_step(self, step_id: ID) -> Optional[ReasoningStep]:
        """단계 조회"""
        return self.steps.get(step_id)

    async def create_reasoning_chain(
        self,
        name: str,
        method: ReasoningMethod,
        description: Optional[str] = None
    ) -> Result:
        """추론 체인 생성"""
        try:
            chain_id = f"chain_{int(time.time() * 1000)}_{uuid.uuid4().hex[:9]}"

            chain = ReasoningChain(
                id=chain_id,
                name=name,
                description=description,
                steps=[],
                start_premise="",
                method=method,
                confidence=0.0,
                is_valid=False,
                metadata={},
                created_at=int(time.time() * 1000),
                updated_at=int(time.time() * 1000)
            )

            self.chains[chain_id] = chain

            await self.emit_event("chain:created", {
                "chain_id": chain_id,
                "name": name,
                "method": method.value
            })

            self.logger.info("추론 체인 생성", {
                "chain_id": chain_id,
                "name": name,
                "method": method.value
            })

            return Result(success=True, data=chain)

        except Exception as error:
            return Result(
                success=False,
                data=None,
                error=ValidationError(
                    f"추론 체인 생성 실패: {str(error)}",
                    context={"name": name, "method": method}
                )
            )

    async def get_next_subgoal(self, chain_id: ID) -> Optional[Dict[str, Any]]:
        """다음 서브골 조회"""
        chain = self.chains.get(chain_id)
        if not chain:
            return None

        # 다음 예정된 단계를 반환
        completed_steps = len([
            step for step in chain.steps
            if step.type == ReasoningStepType.CONCLUSION
        ])
        next_step_index = completed_steps

        if next_step_index < len(chain.steps):
            step = chain.steps[next_step_index]
            return {
                "step_id": step.id,
                "step_type": step.type.value,
                "content": step.content,
                "method": step.method.value
            }

        return None

    def update_subgoal_status(
        self,
        chain_id: ID,
        subgoal_id: ID,
        status: str
    ) -> Result:
        """서브골 상태 업데이트"""
        try:
            chain = self.chains.get(chain_id)
            if not chain:
                return Result(
                    success=False,
                    data=None,
                    error=ValidationError(
                        "Reasoning chain not found",
                        context={"chain_id": chain_id}
                    )
                )

            # 서브골 찾기 및 업데이트
            if chain.subgoals:
                for subgoal in chain.subgoals:
                    if subgoal.id == subgoal_id:
                        subgoal.status = status
                        subgoal.updated_at = int(time.time() * 1000)

                        if status == "completed":
                            if not chain.completed_subgoals:
                                chain.completed_subgoals = []
                            if subgoal_id not in chain.completed_subgoals:
                                chain.completed_subgoals.append(subgoal_id)

                        asyncio.create_task(self.emit_event("subgoal:updated", {
                            "chain_id": chain_id,
                            "subgoal_id": subgoal_id,
                            "status": status
                        }))

                        return Result(success=True, data=None)

            return Result(
                success=False,
                data=None,
                error=ValidationError(
                    "Subgoal not found",
                    context={"chain_id": chain_id, "subgoal_id": subgoal_id}
                )
            )

        except Exception as error:
            return Result(
                success=False,
                data=None,
                error=ValidationError(
                    f"Failed to update subgoal status: {str(error)}",
                    context={"chain_id": chain_id, "subgoal_id": subgoal_id, "error": str(error)}
                )
            )

    def get_chain_statistics(self) -> Dict[str, Any]:
        """추론 체인 통계 조회"""
        total_chains = len(self.chains)
        total_steps = len(self.steps)
        average_steps_per_chain = total_steps / total_chains if total_chains > 0 else 0

        chains = list(self.chains.values())
        average_confidence = (
            sum(chain.confidence for chain in chains) / len(chains)
            if chains else 0.0
        )

        results = list(self.chain_results.values())
        successful_results = len([result for result in results if result.success])
        success_rate = successful_results / len(results) if results else 0.0

        return {
            "total_chains": total_chains,
            "total_steps": total_steps,
            "average_steps_per_chain": average_steps_per_chain,
            "average_confidence": average_confidence,
            "success_rate": success_rate
        }


# 호환성을 위한 익스포트
__all__ = [
    "ReasoningStepType",
    "ReasoningMethod",
    "ReasoningStep",
    "SubGoal",
    "ReasoningChain",
    "ValidationResult",
    "ReasoningChainResult",
    "ReasoningChainManager"
]