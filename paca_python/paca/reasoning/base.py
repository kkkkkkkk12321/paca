"""
Reasoning Base Module
추론 시스템의 기본 클래스들과 공통 인터페이스
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ..core.types.base import ID, Timestamp, KeyValuePair, Result, create_id, current_timestamp
from ..core.events.emitter import EventEmitter
from ..core.utils.logger import create_logger
from ..core.errors.reasoning import ReasoningError, DeductiveReasoningError, LogicalInconsistencyError


class ReasoningType(Enum):
    """추론 타입"""
    DEDUCTIVE = 'deductive'
    INDUCTIVE = 'inductive'
    ABDUCTIVE = 'abductive'
    ANALOGICAL = 'analogical'
    CAUSAL = 'causal'


class InferenceRule(Enum):
    """추론 규칙"""
    MODUS_PONENS = 'modus_ponens'
    MODUS_TOLLENS = 'modus_tollens'
    HYPOTHETICAL_SYLLOGISM = 'hypothetical_syllogism'
    DISJUNCTIVE_SYLLOGISM = 'disjunctive_syllogism'
    UNIVERSAL_INSTANTIATION = 'universal_instantiation'
    EXISTENTIAL_GENERALIZATION = 'existential_generalization'
    DIRECT_INFERENCE = 'direct_inference'


@dataclass(frozen=True)
class ReasoningStep:
    """추론 체인의 개별 단계"""
    id: ID
    premise: Any
    inference_rule: str
    conclusion: Any
    confidence: float
    evidence: List[Any] = field(default_factory=list)
    timestamp: Timestamp = field(default_factory=current_timestamp)


@dataclass
class ReasoningContext:
    """추론 컨텍스트"""
    id: ID
    reasoning_type: ReasoningType
    premises: List[Any]
    target_conclusion: Optional[Any] = None
    constraints: Optional[KeyValuePair] = None
    confidence_threshold: float = 0.7
    max_steps: int = 10
    timeout_seconds: float = 30.0


@dataclass
class ReasoningResult:
    """추론 결과"""
    context_id: ID
    conclusion: Any
    confidence: float
    reasoning_steps: List[ReasoningStep]
    execution_time_ms: float
    is_valid: bool
    metadata: KeyValuePair = field(default_factory=dict)


class BaseReasoningEngine(ABC):
    """추론 엔진 기본 클래스"""

    def __init__(self, name: str, events: Optional[EventEmitter] = None):
        self.name = name
        self.logger = create_logger(f"ReasoningEngine.{name}")
        self.events = events
        self.steps: List[ReasoningStep] = []
        self.confidence_threshold = 0.7

    async def reason(self, context: ReasoningContext) -> Result[ReasoningResult]:
        """추론 실행"""
        start_time = time.time()

        if self.events:
            await self.events.emit('reasoning.started', {
                'engine': self.name,
                'context_id': context.id,
                'reasoning_type': context.reasoning_type.value
            })

        try:
            # 입력 검증
            await self._validate_context(context)

            # 실제 추론 수행
            result = await self.perform_reasoning(context)

            # 결과 검증
            await self._validate_result(result, context)

            execution_time = (time.time() - start_time) * 1000

            if self.events:
                await self.events.emit('reasoning.completed', {
                    'engine': self.name,
                    'context_id': context.id,
                    'confidence': result.confidence,
                    'execution_time_ms': execution_time
                })

            return Result.success(result)

        except Exception as error:
            execution_time = (time.time() - start_time) * 1000

            reasoning_error = error if isinstance(error, ReasoningError) else ReasoningError(
                message=f"Reasoning failed: {str(error)}",
                reasoning_type=context.reasoning_type.value
            )

            self.logger.error(
                'Reasoning failed',
                error=reasoning_error,
                meta={
                    'context_id': context.id,
                    'reasoning_type': context.reasoning_type.value,
                    'execution_time_ms': execution_time
                }
            )

            if self.events:
                await self.events.emit('reasoning.failed', {
                    'engine': self.name,
                    'context_id': context.id,
                    'error': str(reasoning_error)
                })

            return Result.failure(reasoning_error)

    @abstractmethod
    async def perform_reasoning(self, context: ReasoningContext) -> ReasoningResult:
        """구체적인 추론 수행 (하위 클래스에서 구현)"""
        pass

    async def _validate_context(self, context: ReasoningContext) -> None:
        """컨텍스트 검증"""
        if not context.premises:
            raise ReasoningError("Premises are required for reasoning")

        if context.confidence_threshold < 0 or context.confidence_threshold > 1:
            raise ReasoningError("Confidence threshold must be between 0 and 1")

        if context.max_steps <= 0:
            raise ReasoningError("Max steps must be positive")

    async def _validate_result(self, result: ReasoningResult, context: ReasoningContext) -> None:
        """결과 검증"""
        if result.confidence < context.confidence_threshold:
            raise ReasoningError(
                f"Result confidence {result.confidence} below threshold {context.confidence_threshold}"
            )

        # 논리적 일관성 검증
        if not await self._check_logical_consistency(result.reasoning_steps):
            raise LogicalInconsistencyError(
                "Logical inconsistency detected in reasoning chain",
                reasoning_chain=[step.__dict__ for step in result.reasoning_steps]
            )

    async def _check_logical_consistency(self, steps: List[ReasoningStep]) -> bool:
        """논리적 일관성 검증"""
        # 간단한 일관성 검증 - 실제 구현에서는 더 복잡한 로직 사용
        for i, step in enumerate(steps):
            if step.confidence < 0 or step.confidence > 1:
                return False

            # 각 단계의 신뢰도가 전체적으로 감소하는지 확인
            if i > 0 and step.confidence < steps[i-1].confidence * 0.5:
                self.logger.warn(f"Significant confidence drop at step {i}")

        return True

    def add_step(self, premise: Any, rule: str, conclusion: Any, confidence: float, evidence: List[Any] = None) -> ReasoningStep:
        """추론 단계 추가"""
        step = ReasoningStep(
            id=create_id(),
            premise=premise,
            inference_rule=rule,
            conclusion=conclusion,
            confidence=confidence,
            evidence=evidence or []
        )
        self.steps.append(step)
        return step

    def clear_steps(self) -> None:
        """추론 단계 초기화"""
        self.steps.clear()


class DeductiveReasoningEngine(BaseReasoningEngine):
    """연역적 추론 엔진"""

    def __init__(self, events: Optional[EventEmitter] = None):
        super().__init__("DeductiveReasoning", events)

    async def perform_reasoning(self, context: ReasoningContext) -> ReasoningResult:
        """연역적 추론 실행"""
        start_time = time.time()
        self.clear_steps()

        try:
            premises = context.premises
            target_conclusion = context.target_conclusion

            # 1. 전제들 분석
            analyzed_premises = await self._analyze_premises(premises)

            # 2. 추론 규칙 적용
            reasoning_steps = await self._apply_deductive_rules(analyzed_premises, target_conclusion)

            fallback_used = False

            if not reasoning_steps:
                fallback_used = True
                fallback_conclusion = self._generate_fallback_conclusion(premises, target_conclusion)
                fallback_premise = str(premises[-1]) if premises else "insufficient premises"
                reasoning_steps.append(
                    ReasoningStep(
                        id=create_id(),
                        premise=fallback_premise,
                        inference_rule=InferenceRule.DIRECT_INFERENCE.value,
                        conclusion=fallback_conclusion,
                        confidence=0.7,
                        evidence=[fallback_premise] if premises else []
                    )
                )

            # 3. 결론 도출
            conclusion = await self._derive_conclusion(reasoning_steps)

            # 4. 신뢰도 계산
            confidence = await self._calculate_confidence(reasoning_steps)

            execution_time = (time.time() - start_time) * 1000

            return ReasoningResult(
                context_id=context.id,
                conclusion=conclusion,
                confidence=confidence,
                reasoning_steps=reasoning_steps,
                execution_time_ms=execution_time,
                is_valid=True,
                metadata={
                    'premises_count': len(premises),
                    'rules_applied': len(reasoning_steps),
                    'fallback_used': fallback_used
                }
            )

        except Exception as e:
            raise DeductiveReasoningError(
                message=f"Deductive reasoning failed: {str(e)}",
                premises=[str(p) for p in context.premises],
                conclusion=str(context.target_conclusion) if context.target_conclusion else None
            )

    async def _analyze_premises(self, premises: List[Any]) -> List[Dict[str, Any]]:
        """전제들 분석"""
        analyzed = []
        for i, premise in enumerate(premises):
            analyzed.append({
                'id': i,
                'content': premise,
                'type': self._classify_premise(premise),
                'variables': self._extract_variables(premise)
            })
        return analyzed

    async def _apply_deductive_rules(self, premises: List[Dict[str, Any]], target: Any) -> List[ReasoningStep]:
        """연역적 추론 규칙 적용"""
        steps = []

        # Modus Ponens 적용 예시
        for i, premise1 in enumerate(premises):
            for j, premise2 in enumerate(premises):
                if i != j:
                    step = await self._try_modus_ponens(premise1, premise2)
                    if step:
                        steps.append(step)

                    step = await self._try_modus_tollens(premise1, premise2)
                    if step:
                        steps.append(step)

                    step = await self._try_hypothetical_syllogism(premise1, premise2)
                    if step:
                        steps.append(step)

        return steps

    async def _try_modus_ponens(self, premise1: Dict[str, Any], premise2: Dict[str, Any]) -> Optional[ReasoningStep]:
        """Modus Ponens 규칙 적용 시도"""
        # 간단한 Modus Ponens 구현
        # "If A then B" + "A" => "B"

        p1_content = str(premise1['content']).lower()
        p2_content = str(premise2['content']).lower()

        # 조건문 패턴 검사
        if 'if' in p1_content and 'then' in p1_content:
            # 조건문 파싱
            parts = p1_content.split('then')
            if len(parts) == 2:
                condition = parts[0].replace('if', '').strip()
                conclusion = parts[1].strip()

                # 두 번째 전제가 조건과 일치하는지 확인
                if condition in p2_content:
                    return ReasoningStep(
                        id=create_id(),
                        premise=f"{premise1['content']} AND {premise2['content']}",
                        inference_rule=InferenceRule.MODUS_PONENS.value,
                        conclusion=conclusion,
                        confidence=0.9,
                        evidence=[premise1['content'], premise2['content']]
                    )

        return None

    async def _try_modus_tollens(self, premise1: Dict[str, Any], premise2: Dict[str, Any]) -> Optional[ReasoningStep]:
        """Modus Tollens 규칙 적용 시도"""
        p1_content = str(premise1['content']).lower()
        p2_content = str(premise2['content']).lower()

        if 'if' in p1_content and 'then' in p1_content and 'not' in p2_content:
            parts = p1_content.split('then')
            if len(parts) == 2:
                condition = parts[0].replace('if', '').strip()
                consequence = parts[1].strip()

                normalized_consequence = consequence.replace('not ', '').strip()
                if normalized_consequence and normalized_consequence in p2_content:
                    conclusion = f"not {condition}".strip()
                    return ReasoningStep(
                        id=create_id(),
                        premise=f"{premise1['content']} AND {premise2['content']}",
                        inference_rule=InferenceRule.MODUS_TOLLENS.value,
                        conclusion=conclusion,
                        confidence=0.85,
                        evidence=[premise1['content'], premise2['content']]
                    )

        return None

    async def _try_hypothetical_syllogism(self, premise1: Dict[str, Any], premise2: Dict[str, Any]) -> Optional[ReasoningStep]:
        """가설 삼단논법 적용 시도"""
        p1_content = str(premise1['content']).lower()
        p2_content = str(premise2['content']).lower()

        if 'if' in p1_content and 'then' in p1_content and 'if' in p2_content and 'then' in p2_content:
            parts1 = p1_content.split('then', 1)
            parts2 = p2_content.split('then', 1)
            if len(parts1) == 2 and len(parts2) == 2:
                condition1 = parts1[0].replace('if', '').strip()
                consequence1 = parts1[1].strip()
                condition2 = parts2[0].replace('if', '').strip()
                consequence2 = parts2[1].strip()

                if consequence1 == condition2 and condition1 and consequence2:
                    new_conclusion = f"If {condition1} then {consequence2}"
                    return ReasoningStep(
                        id=create_id(),
                        premise=f"{premise1['content']} AND {premise2['content']}",
                        inference_rule=InferenceRule.HYPOTHETICAL_SYLLOGISM.value,
                        conclusion=new_conclusion,
                        confidence=0.8,
                        evidence=[premise1['content'], premise2['content']]
                    )

        return None

    async def _derive_conclusion(self, steps: List[ReasoningStep]) -> Any:
        """결론 도출"""
        if not steps:
            return "No conclusion could be derived"

        # 가장 높은 신뢰도를 가진 단계의 결론 반환
        best_step = max(steps, key=lambda s: s.confidence)
        return best_step.conclusion

    async def _calculate_confidence(self, steps: List[ReasoningStep]) -> float:
        """신뢰도 계산"""
        if not steps:
            return 0.0

        # 단계별 신뢰도의 가중평균
        total_confidence = sum(step.confidence for step in steps)
        return total_confidence / len(steps)

    def _classify_premise(self, premise: Any) -> str:
        """전제 분류"""
        premise_str = str(premise).lower()

        if 'if' in premise_str and 'then' in premise_str:
            return 'conditional'
        elif 'all' in premise_str or 'every' in premise_str:
            return 'universal'
        elif 'some' in premise_str or 'exists' in premise_str:
            return 'existential'
        else:
            return 'atomic'

    def _extract_variables(self, premise: Any) -> List[str]:
        """변수 추출"""
        # 간단한 변수 추출 - 실제 구현에서는 더 정교한 파싱 필요
        import re
        premise_str = str(premise)
        variables = re.findall(r'\b[A-Z]\b', premise_str)
        return list(set(variables))

    def _generate_fallback_conclusion(self, premises: List[Any], target: Any) -> str:
        """규칙 적용 실패 시 기본 결론 생성"""
        if target is not None:
            return str(target)

        if premises:
            return str(premises[-1])

        return "추론을 진행할 충분한 정보가 없습니다."


class ReasoningEngine:
    """통합 추론 엔진"""

    def __init__(self, events: Optional[EventEmitter] = None):
        self.engines = {
            ReasoningType.DEDUCTIVE: DeductiveReasoningEngine(events)
        }
        self.events = events
        self.logger = create_logger('ReasoningEngine')
        self._is_initialized = False

    async def initialize(self) -> Result[bool]:
        """추론 엔진 초기화"""
        if self._is_initialized:
            return Result.success(True)

        try:
            self._is_initialized = True
            return Result.success(True)

        except Exception as error:
            return Result.failure(ReasoningError(f"Reasoning engine initialization failed: {str(error)}"))

    async def reason(
        self,
        reasoning_type: ReasoningType,
        premises: List[Any],
        target_conclusion: Any = None,
        **kwargs
    ) -> Result[ReasoningResult]:
        """추론 실행"""
        if reasoning_type not in self.engines:
            return Result.failure(ReasoningError(f"Reasoning type {reasoning_type.value} not supported"))

        context = ReasoningContext(
            id=create_id(),
            reasoning_type=reasoning_type,
            premises=premises,
            target_conclusion=target_conclusion,
            **kwargs
        )

        engine = self.engines[reasoning_type]
        return await engine.reason(context)

    def get_supported_types(self) -> List[ReasoningType]:
        """지원되는 추론 타입 목록"""
        return list(self.engines.keys())

    def add_engine(self, reasoning_type: ReasoningType, engine: BaseReasoningEngine) -> None:
        """추론 엔진 추가"""
        self.engines[reasoning_type] = engine
        self.logger.info(f"Added reasoning engine for {reasoning_type.value}")

    def remove_engine(self, reasoning_type: ReasoningType) -> bool:
        """추론 엔진 제거"""
        if reasoning_type in self.engines:
            del self.engines[reasoning_type]
            self.logger.info(f"Removed reasoning engine for {reasoning_type.value}")
            return True
        return False
