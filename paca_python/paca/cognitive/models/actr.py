"""
ACT-R Cognitive Model Implementation
ACT-R (Adaptive Control of Thought-Rational) 인지 모델 구현
"""

import asyncio
import time
import math
import random
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Mapping
from abc import ABC, abstractmethod

from .base import BaseCognitiveModel, CognitiveArchitecture, ModelState
from ..base import (
    CognitiveContext, CognitiveResult, ProcessingStep,
    QualityMetrics, ResourceUsage, create_processing_step
)
from ...core.types import ID, Timestamp, KeyValuePair, create_id, current_timestamp
from ...core.errors import CognitiveError, ModelError


class BufferState(Enum):
    """버퍼 상태"""
    EMPTY = 'empty'
    FULL = 'full'
    ERROR = 'error'
    BUSY = 'busy'


class ActionType(Enum):
    """프로덕션 액션 타입"""
    RETRIEVE = 'retrieve'
    MODIFY = 'modify'
    OUTPUT = 'output'
    CLEAR = 'clear'


@dataclass
class ACTRParameters:
    """ACT-R 특화 파라미터"""
    # 활성화 관련 파라미터
    base_level_learning: float = 0.5
    activation_noise: float = 0.25
    retrieval_threshold: float = 0.0
    latency_factor: float = 1.0

    # 절차적 학습 파라미터
    utility_noise: float = 0.25
    alpha: float = 0.2
    egs: float = 0.0

    # 선언적 메모리 파라미터
    maximum_associative_strength: float = 1.0
    spreading_activation: float = 1.0
    partial_matching_penalty: float = 1.0


@dataclass
class ACTRChunk:
    """ACT-R 청크 (지식 단위)"""
    name: str
    chunk_type: str
    slots: Dict[str, Any]
    base_level_activation: float
    creation_time: float
    references: List[float] = field(default_factory=list)
    associative_links: Dict[str, float] = field(default_factory=dict)


@dataclass
class ProductionCondition:
    """프로덕션 조건"""
    buffer_name: str
    pattern: Dict[str, Any]
    negated: bool = False


@dataclass
class ProductionAction:
    """프로덕션 액션"""
    action_type: ActionType
    buffer_name: str
    parameters: Dict[str, Any]


@dataclass
class ACTRProduction:
    """ACT-R 프로덕션 룰"""
    name: str
    conditions: List[ProductionCondition]
    actions: List[ProductionAction]
    utility: float
    conflict_resolution_value: float = 0.0
    creation_time: float = 0.0
    success_count: int = 0
    failure_count: int = 0


@dataclass
class ACTRBuffer:
    """ACT-R 버퍼"""
    name: str
    chunk: Optional[ACTRChunk]
    state: BufferState
    last_access_time: float


@dataclass
class ACTRReasoningResult:
    """ACT-R 추론 결과"""
    success: bool
    result: Any
    processing_time: float
    confidence: float
    steps: List[ProcessingStep]
    error: Optional[str] = None


@dataclass
class ACTRLearningResult:
    """ACT-R 학습 결과"""
    success: bool
    improvement: float
    learning_time: float
    knowledge_gained: Optional[Dict[str, Any]]
    error: Optional[str] = None


class ACTRModel(BaseCognitiveModel):
    """ACT-R 인지 모델"""

    def __init__(self, config_id: str, name: str, parameters: Optional[ACTRParameters] = None):
        # 기본 ACT-R 아키텍처 정의
        architecture = self._create_architecture()

        super().__init__(config_id, name, "ACT-R", architecture)

        self.parameters = parameters or ACTRParameters()
        self.declarative_memory: Dict[str, ACTRChunk] = {}
        self.procedural_memory: Dict[str, ACTRProduction] = {}
        self.buffers: Dict[str, ACTRBuffer] = {}
        self.current_time: float = 0.0
        self.conflict_set: List[ACTRProduction] = []

        self.logger = logging.getLogger(f"ACTRModel.{name}")

        # 모델 초기화
        self._initialize_model()

    def _create_architecture(self) -> CognitiveArchitecture:
        """ACT-R 아키텍처 생성"""
        return CognitiveArchitecture(
            name="ACT-R Architecture",
            description="Adaptive Control of Thought-Rational cognitive architecture",
            modules={
                "goal": {
                    "name": "Goal Module",
                    "type": "executive_control",
                    "capacity": 1,
                    "processing_time": 50
                },
                "retrieval": {
                    "name": "Retrieval Module",
                    "type": "long_term_memory",
                    "capacity": 1,
                    "processing_time": 50
                },
                "imaginal": {
                    "name": "Imaginal Module",
                    "type": "working_memory",
                    "capacity": 1,
                    "processing_time": 200
                }
            },
            connections=[
                ("goal", "retrieval", "control"),
                ("goal", "imaginal", "control")
            ]
        )

    def _initialize_model(self) -> None:
        """모델 초기화"""
        self._initialize_buffers()
        self._initialize_basic_knowledge()
        self._initialize_basic_productions()

    def _initialize_buffers(self) -> None:
        """버퍼 초기화"""
        buffer_names = ['goal', 'retrieval', 'imaginal', 'visual', 'manual']

        for name in buffer_names:
            self.buffers[name] = ACTRBuffer(
                name=name,
                chunk=None,
                state=BufferState.EMPTY,
                last_access_time=0.0
            )

    def _initialize_basic_knowledge(self) -> None:
        """기본 지식 초기화"""
        # 기본 청크들 추가
        basic_chunks = [
            ACTRChunk(
                name='start',
                chunk_type='goal',
                slots={'state': 'start'},
                base_level_activation=1.0,
                creation_time=0.0,
                references=[0.0]
            )
        ]

        for chunk in basic_chunks:
            self.declarative_memory[chunk.name] = chunk

    def _initialize_basic_productions(self) -> None:
        """기본 프로덕션 룰 초기화"""
        basic_productions = [
            ACTRProduction(
                name='start-reasoning',
                conditions=[
                    ProductionCondition(
                        buffer_name='goal',
                        pattern={'state': 'start'},
                        negated=False
                    )
                ],
                actions=[
                    ProductionAction(
                        action_type=ActionType.MODIFY,
                        buffer_name='goal',
                        parameters={'state': 'reasoning'}
                    )
                ],
                utility=10.0,
                creation_time=0.0
            )
        ]

        for production in basic_productions:
            self.procedural_memory[production.name] = production

    async def process_reasoning(self, input_data: Any, context: Optional[Any] = None) -> ACTRReasoningResult:
        """추론 과정 실행"""
        start_time = time.time()
        steps: List[ProcessingStep] = []

        try:
            # 목표 설정
            await self._set_goal(input_data)
            steps.append(create_processing_step(
                name='Goal setting',
                step_type='goal_initialization',
                input_data=input_data,
                output=self.buffers['goal'].chunk,
                confidence=0.9
            ))

            # 인지 사이클 실행
            cycle_count = 0
            max_cycles = 100

            while cycle_count < max_cycles and await self._should_continue_processing():
                cycle_count += 1

                # 충돌 해결
                await self._conflict_resolution()

                # 선택된 프로덕션 실행
                if self.conflict_set:
                    selected_production = self.conflict_set[0]
                    await self._execute_production(selected_production)

                    steps.append(create_processing_step(
                        name=f'Execute production: {selected_production.name}',
                        step_type='production_execution',
                        input_data=selected_production.conditions,
                        output=selected_production.actions,
                        confidence=self._calculate_production_confidence(selected_production)
                    ))

                self.current_time += 50  # 50ms per cycle

            processing_time = (time.time() - start_time) * 1000
            result = self.buffers.get('goal', {}).chunk if 'goal' in self.buffers else None

            return ACTRReasoningResult(
                success=True,
                result=result,
                processing_time=processing_time,
                confidence=self._calculate_overall_confidence(steps),
                steps=steps
            )

        except Exception as error:
            return ACTRReasoningResult(
                success=False,
                result=None,
                processing_time=(time.time() - start_time) * 1000,
                confidence=0.0,
                steps=steps,
                error=str(error)
            )

    async def process_learning(self, experience: Any, feedback: Optional[Any] = None) -> ACTRLearningResult:
        """학습 과정 실행"""
        start_time = time.time()

        try:
            # 기본 수준 학습
            await self._update_base_level_activations(experience)

            # 절차적 학습
            if feedback:
                await self._update_production_utilities(experience, feedback)

            # 연관적 학습
            await self._update_associative_strengths(experience)

            learning_time = (time.time() - start_time) * 1000

            return ACTRLearningResult(
                success=True,
                improvement=self._calculate_learning_improvement(experience, feedback),
                learning_time=learning_time,
                knowledge_gained={
                    'chunks_updated': len(self.declarative_memory),
                    'productions_updated': len(self.procedural_memory),
                    'associations_updated': self._count_associations()
                }
            )

        except Exception as error:
            return ACTRLearningResult(
                success=False,
                improvement=0.0,
                learning_time=(time.time() - start_time) * 1000,
                knowledge_gained=None,
                error=str(error)
            )

    async def _set_goal(self, goal: Any) -> None:
        """목표 설정"""
        goal_buffer = self.buffers.get('goal')
        if goal_buffer:
            goal_buffer.chunk = ACTRChunk(
                name='current_goal',
                chunk_type='goal',
                slots=goal if isinstance(goal, dict) else {'content': goal},
                base_level_activation=1.0,
                creation_time=self.current_time,
                references=[self.current_time]
            )
            goal_buffer.state = BufferState.FULL
            goal_buffer.last_access_time = self.current_time

    async def _should_continue_processing(self) -> bool:
        """처리 계속 여부 판단"""
        goal_buffer = self.buffers.get('goal')
        if not goal_buffer or not goal_buffer.chunk:
            return False

        goal_state = goal_buffer.chunk.slots.get('state')
        return goal_state not in ['completed', 'failed']

    async def _conflict_resolution(self) -> None:
        """충돌 해결"""
        self.conflict_set = []

        # 모든 프로덕션에 대해 조건 매칭 확인
        for production in self.procedural_memory.values():
            if await self._matches_conditions(production.conditions):
                self.conflict_set.append(production)

        # 유틸리티에 따른 정렬
        self.conflict_set.sort(key=lambda p: self._get_noisy_utility(p), reverse=True)

    def _get_noisy_utility(self, production: ACTRProduction) -> float:
        """노이즈가 포함된 유틸리티 계산"""
        noise = self.parameters.utility_noise * (random.random() - 0.5)
        return production.utility + noise

    async def _matches_conditions(self, conditions: List[ProductionCondition]) -> bool:
        """조건 매칭 확인"""
        for condition in conditions:
            buffer = self.buffers.get(condition.buffer_name)
            if not buffer or not buffer.chunk:
                return condition.negated

            matches = self._matches_pattern(buffer.chunk, condition.pattern)
            if matches == condition.negated:
                return False
        return True

    def _matches_pattern(self, chunk: ACTRChunk, pattern: Dict[str, Any]) -> bool:
        """패턴 매칭"""
        for slot, value in pattern.items():
            if chunk.slots.get(slot) != value:
                return False
        return True

    async def _execute_production(self, production: ACTRProduction) -> None:
        """프로덕션 실행"""
        for action in production.actions:
            await self._execute_action(action)

        production.success_count += 1
        production.conflict_resolution_value = self._calculate_crv(production)

    async def _execute_action(self, action: ProductionAction) -> None:
        """액션 실행"""
        buffer = self.buffers.get(action.buffer_name)
        if not buffer:
            return

        if action.action_type == ActionType.MODIFY:
            if buffer.chunk:
                for slot, value in action.parameters.items():
                    buffer.chunk.slots[slot] = value

        elif action.action_type == ActionType.CLEAR:
            buffer.chunk = None
            buffer.state = BufferState.EMPTY

        elif action.action_type == ActionType.RETRIEVE:
            await self._perform_retrieval(action.parameters)

        elif action.action_type == ActionType.OUTPUT:
            self.logger.info(f"ACT-R Output: {action.parameters}")

    async def _perform_retrieval(self, retrieval_spec: Dict[str, Any]) -> None:
        """검색 수행"""
        best_match = None
        highest_activation = self.parameters.retrieval_threshold

        for chunk in self.declarative_memory.values():
            if self._matches_pattern(chunk, retrieval_spec):
                activation = self._calculate_activation(chunk)
                if activation > highest_activation:
                    highest_activation = activation
                    best_match = chunk

        retrieval_buffer = self.buffers.get('retrieval')
        if retrieval_buffer:
            retrieval_buffer.chunk = best_match
            retrieval_buffer.state = BufferState.FULL if best_match else BufferState.ERROR
            retrieval_buffer.last_access_time = self.current_time

    def _calculate_activation(self, chunk: ACTRChunk) -> float:
        """활성화 계산"""
        # 기본 수준 활성화
        base_level = self._calculate_base_level_activation(chunk)

        # 확산 활성화
        spreading = self._calculate_spreading_activation(chunk)

        # 노이즈
        noise = self.parameters.activation_noise * (random.random() - 0.5)

        return base_level + spreading + noise

    def _calculate_base_level_activation(self, chunk: ACTRChunk) -> float:
        """기본 수준 활성화 계산"""
        if not chunk.references:
            return chunk.base_level_activation

        total = 0.0
        for reference_time in chunk.references:
            age = self.current_time - reference_time
            if age > 0:
                total += math.pow(age / 1000, -self.parameters.base_level_learning)

        return math.log(total) if total > 0 else chunk.base_level_activation

    def _calculate_spreading_activation(self, chunk: ACTRChunk) -> float:
        """확산 활성화 계산"""
        spreading = 0.0

        for linked_chunk, strength in chunk.associative_links.items():
            source_activation = self._get_chunk_activation(linked_chunk)
            spreading += strength * source_activation * self.parameters.spreading_activation

        return spreading

    def _get_chunk_activation(self, chunk_name: str) -> float:
        """청크 활성화 가져오기"""
        chunk = self.declarative_memory.get(chunk_name)
        return self._calculate_activation(chunk) if chunk else 0.0

    def _calculate_production_confidence(self, production: ACTRProduction) -> float:
        """프로덕션 신뢰도 계산"""
        total_attempts = production.success_count + production.failure_count
        if total_attempts == 0:
            return 0.5
        success_rate = production.success_count / total_attempts
        return max(0.1, min(0.9, success_rate))

    def _calculate_overall_confidence(self, steps: List[ProcessingStep]) -> float:
        """전체 신뢰도 계산"""
        if not steps:
            return 0.0
        return sum(step.confidence for step in steps) / len(steps)

    def _calculate_crv(self, production: ACTRProduction) -> float:
        """충돌 해결 값 계산"""
        return production.utility + self.parameters.egs

    async def _update_base_level_activations(self, experience: Any) -> None:
        """기본 수준 활성화 업데이트"""
        for chunk in self.declarative_memory.values():
            chunk.references.append(self.current_time)

    async def _update_production_utilities(self, experience: Any, feedback: Any) -> None:
        """프로덕션 유틸리티 업데이트"""
        for production in self.procedural_memory.values():
            if hasattr(feedback, 'success') and feedback.success:
                reward = getattr(feedback, 'reward', 1.0)
                production.utility += self.parameters.alpha * (reward - production.utility)

    async def _update_associative_strengths(self, experience: Any) -> None:
        """연관적 강도 업데이트"""
        # 연관적 학습 구현 (필요에 따라 상세 구현)
        pass

    def _calculate_learning_improvement(self, experience: Any, feedback: Optional[Any] = None) -> float:
        """학습 개선도 계산"""
        if feedback and hasattr(feedback, 'reward'):
            return getattr(feedback, 'reward', 0.1)
        return 0.1

    def _count_associations(self) -> int:
        """연관 관계 수 계산"""
        return sum(len(chunk.associative_links) for chunk in self.declarative_memory.values())

    async def perform_cognitive_processing(
        self,
        context: CognitiveContext,
        relevant_memory: Any
    ) -> CognitiveResult:
        """인지 처리 수행 (BaseCognitiveProcessor 인터페이스 구현)"""
        start_time = time.time()

        # ACT-R 추론 실행
        reasoning_result = await self.process_reasoning(context.input, context)

        # 결과 변환
        processing_time_ms = (time.time() - start_time) * 1000

        return CognitiveResult(
            context_id=context.id,
            output=reasoning_result.result,
            confidence=reasoning_result.confidence,
            quality_metrics=QualityMetrics(
                accuracy=reasoning_result.confidence,
                precision=reasoning_result.confidence,
                confidence=reasoning_result.confidence
            ),
            processing_steps=reasoning_result.steps,
            memory_updates=[],
            processing_time_ms=processing_time_ms,
            resource_usage=ResourceUsage(
                memory_mb=0.0,
                cpu_percent=0.0,
                processing_time_ms=processing_time_ms,
                api_calls=0,
                cache_hits=0,
                cache_misses=0
            )
        )

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            'name': self.name,
            'type': 'ACT-R',
            'parameters': {
                'base_level_learning': self.parameters.base_level_learning,
                'activation_noise': self.parameters.activation_noise,
                'retrieval_threshold': self.parameters.retrieval_threshold,
                'utility_noise': self.parameters.utility_noise
            },
            'memory_stats': {
                'declarative_chunks': len(self.declarative_memory),
                'procedural_rules': len(self.procedural_memory),
                'associations': self._count_associations()
            },
            'current_time': self.current_time
        }


# 편의 함수들
def create_actr_model(
    name: str,
    parameters: Optional[ACTRParameters] = None
) -> ACTRModel:
    """ACT-R 모델 생성 헬퍼"""
    return ACTRModel(create_id(), name, parameters)


def create_actr_chunk(
    name: str,
    chunk_type: str,
    slots: Dict[str, Any],
    activation: float = 1.0
) -> ACTRChunk:
    """ACT-R 청크 생성 헬퍼"""
    return ACTRChunk(
        name=name,
        chunk_type=chunk_type,
        slots=slots,
        base_level_activation=activation,
        creation_time=time.time(),
        references=[time.time()]
    )


def create_production_rule(
    name: str,
    conditions: List[ProductionCondition],
    actions: List[ProductionAction],
    utility: float = 1.0
) -> ACTRProduction:
    """프로덕션 룰 생성 헬퍼"""
    return ACTRProduction(
        name=name,
        conditions=conditions,
        actions=actions,
        utility=utility,
        creation_time=time.time()
    )