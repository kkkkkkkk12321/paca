"""
SOAR Cognitive Model Implementation
SOAR (State, Operator, And Result) 인지 모델 구현
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

from .base import BaseCognitiveModel, CognitiveArchitecture, PerformanceMetrics
from ..base import (
    CognitiveContext, CognitiveResult, ProcessingStep,
    QualityMetrics, ResourceUsage, create_processing_step
)
from ...core.types import ID, Timestamp, KeyValuePair, create_id, current_timestamp
from ...core.errors import CognitiveError, ModelError


class GoalType(Enum):
    """목표 타입"""
    ACHIEVE = 'achieve'
    MAINTAIN = 'maintain'
    AVOID = 'avoid'


class GoalStatus(Enum):
    """목표 상태"""
    ACTIVE = 'active'
    SATISFIED = 'satisfied'
    FAILED = 'failed'


class ActionType(Enum):
    """액션 타입"""
    ADD = 'add'
    REMOVE = 'remove'
    MODIFY = 'modify'


class PreferenceType(Enum):
    """선호도 타입"""
    BETTER = 'better'
    WORSE = 'worse'
    BEST = 'best'
    WORST = 'worst'
    INDIFFERENT = 'indifferent'
    REQUIRE = 'require'
    PROHIBIT = 'prohibit'


@dataclass
class SOARParameters:
    """SOAR 특화 파라미터"""
    learning_rate: float = 0.1
    chunking_threshold: float = 0.8
    working_memory_capacity: int = 7
    operator_selection_strategy: str = 'utility'  # 'utility', 'random', 'first'


@dataclass
class WorkingMemoryElement:
    """작업 메모리 요소"""
    identifier: str
    attribute: str
    value: Any
    acceptable: bool
    timestamp: float


@dataclass
class SOARGoal:
    """SOAR 목표"""
    id: str
    goal_type: GoalType
    conditions: Dict[str, Any]
    priority: int
    status: GoalStatus


@dataclass
class OperatorAction:
    """연산자 액션"""
    action_type: ActionType
    identifier: str
    attribute: str
    value: Any


@dataclass
class OperatorPreference:
    """연산자 선호도"""
    preference_type: PreferenceType
    operator: str
    value: Optional[Any] = None


@dataclass
class SOAROperator:
    """SOAR 연산자"""
    name: str
    conditions: Dict[str, Any]
    actions: List[OperatorAction]
    preferences: List[OperatorPreference]
    utility: float


@dataclass
class SOARState:
    """SOAR 상태"""
    id: str
    attributes: Dict[str, Any]
    subgoals: List[SOARGoal]
    working_memory_elements: List[WorkingMemoryElement]


@dataclass
class ProductionRule:
    """프로덕션 룰"""
    name: str
    conditions: Dict[str, Any]
    actions: List[OperatorAction]
    utility: float = 1.0


class SOARModel(BaseCognitiveModel):
    """SOAR 인지 모델"""

    def __init__(self, config_id: str, name: str, parameters: Optional[SOARParameters] = None):
        # 기본 SOAR 아키텍처 정의
        architecture = self._create_architecture()

        super().__init__(config_id, name, "SOAR", architecture)

        self.parameters = parameters or SOARParameters()
        self.soar_state: Optional[SOARState] = None
        self.operator_space: Dict[str, SOAROperator] = {}
        self.production_memory: Dict[str, ProductionRule] = {}
        self.working_memory: List[WorkingMemoryElement] = []
        self.decision_cycle_count = 0

        self.logger = logging.getLogger(f"SOARModel.{name}")

        # 모델 초기화
        self._initialize_model()

    def _create_architecture(self) -> CognitiveArchitecture:
        """SOAR 아키텍처 생성"""
        return CognitiveArchitecture(
            name="SOAR Architecture",
            description="State, Operator, And Result cognitive architecture",
            modules={
                "working_memory": {
                    "name": "Working Memory",
                    "type": "working_memory",
                    "capacity": 7,
                    "processing_time": 10
                },
                "production_memory": {
                    "name": "Production Memory",
                    "type": "long_term_memory",
                    "capacity": -1,  # unlimited
                    "processing_time": 50
                },
                "preference_memory": {
                    "name": "Preference Memory",
                    "type": "preference_system",
                    "capacity": -1,
                    "processing_time": 20
                }
            },
            connections=[
                ("working_memory", "production_memory", "activation"),
                ("production_memory", "working_memory", "modification"),
                ("working_memory", "preference_memory", "evaluation")
            ]
        )

    def _initialize_model(self) -> None:
        """모델 초기화"""
        self._initialize_state()
        self._initialize_basic_operators()
        self._initialize_basic_productions()

    def _initialize_state(self) -> None:
        """상태 초기화"""
        self.soar_state = SOARState(
            id="initial_state",
            attributes={"status": "ready"},
            subgoals=[],
            working_memory_elements=[]
        )

    def _initialize_basic_operators(self) -> None:
        """기본 연산자 초기화"""
        basic_operators = [
            SOAROperator(
                name="start_reasoning",
                conditions={"state": "ready"},
                actions=[
                    OperatorAction(
                        action_type=ActionType.MODIFY,
                        identifier="state",
                        attribute="status",
                        value="reasoning"
                    )
                ],
                preferences=[],
                utility=1.0
            )
        ]

        for operator in basic_operators:
            self.operator_space[operator.name] = operator

    def _initialize_basic_productions(self) -> None:
        """기본 프로덕션 룰 초기화"""
        basic_productions = [
            ProductionRule(
                name="detect_goal",
                conditions={"input": "goal"},
                actions=[
                    OperatorAction(
                        action_type=ActionType.ADD,
                        identifier="goal",
                        attribute="type",
                        value="achieve"
                    )
                ],
                utility=1.0
            )
        ]

        for production in basic_productions:
            self.production_memory[production.name] = production

    async def process_reasoning(self, input_data: Any, context: Optional[Any] = None) -> Dict[str, Any]:
        """추론 과정 실행"""
        start_time = time.time()
        steps: List[ProcessingStep] = []

        try:
            # 입력을 작업 메모리에 추가
            await self._add_to_working_memory("input", "content", input_data)

            steps.append(create_processing_step(
                name='Input to Working Memory',
                step_type='memory_input',
                input_data=input_data,
                output=self.working_memory[-1] if self.working_memory else None,
                confidence=0.9
            ))

            # 의사결정 사이클 실행
            max_cycles = 50
            cycle_count = 0

            while cycle_count < max_cycles and await self._should_continue_reasoning():
                cycle_count += 1
                self.decision_cycle_count += 1

                # 1. 정교화 단계 (Elaboration)
                await self._elaboration_phase()

                # 2. 의사결정 단계 (Decision)
                selected_operator = await self._decision_phase()

                if selected_operator:
                    # 3. 적용 단계 (Application)
                    await self._application_phase(selected_operator)

                    steps.append(create_processing_step(
                        name=f'Apply operator: {selected_operator.name}',
                        step_type='operator_application',
                        input_data=selected_operator.conditions,
                        output=selected_operator.actions,
                        confidence=self._calculate_operator_confidence(selected_operator)
                    ))

            processing_time = (time.time() - start_time) * 1000
            result = self._extract_result()

            return {
                'success': True,
                'result': result,
                'processing_time': processing_time,
                'confidence': self._calculate_overall_confidence(steps),
                'steps': steps,
                'decision_cycles': cycle_count
            }

        except Exception as error:
            return {
                'success': False,
                'result': None,
                'processing_time': (time.time() - start_time) * 1000,
                'confidence': 0.0,
                'steps': steps,
                'error': str(error)
            }

    async def process_learning(self, experience: Any, feedback: Optional[Any] = None) -> Dict[str, Any]:
        """학습 과정 실행 (청킹 메커니즘)"""
        start_time = time.time()

        try:
            # 청킹 - 유용한 연산자 시퀀스를 새로운 프로덕션으로 변환
            chunks_created = await self._chunking_learning(experience)

            # 유틸리티 학습
            if feedback:
                await self._utility_learning(experience, feedback)

            learning_time = (time.time() - start_time) * 1000

            return {
                'success': True,
                'improvement': self._calculate_learning_improvement(chunks_created, feedback),
                'learning_time': learning_time,
                'knowledge_gained': {
                    'chunks_created': chunks_created,
                    'productions_updated': len(self.production_memory),
                    'operators_updated': len(self.operator_space)
                }
            }

        except Exception as error:
            return {
                'success': False,
                'improvement': 0.0,
                'learning_time': (time.time() - start_time) * 1000,
                'knowledge_gained': None,
                'error': str(error)
            }

    async def _add_to_working_memory(self, identifier: str, attribute: str, value: Any) -> None:
        """작업 메모리에 요소 추가"""
        if len(self.working_memory) >= self.parameters.working_memory_capacity:
            # 용량 초과시 오래된 요소 제거
            self.working_memory.pop(0)

        element = WorkingMemoryElement(
            identifier=identifier,
            attribute=attribute,
            value=value,
            acceptable=True,
            timestamp=time.time()
        )

        self.working_memory.append(element)

    async def _should_continue_reasoning(self) -> bool:
        """추론 계속 여부 판단"""
        # 목표가 달성되었거나 실패한 경우 중지
        if self.soar_state:
            for goal in self.soar_state.subgoals:
                if goal.status in [GoalStatus.SATISFIED, GoalStatus.FAILED]:
                    return False

        # 작업 메모리에 처리할 내용이 있으면 계속
        return len(self.working_memory) > 0

    async def _elaboration_phase(self) -> None:
        """정교화 단계 - 프로덕션 매칭 및 실행"""
        matched_productions = []

        for production in self.production_memory.values():
            if self._matches_production_conditions(production):
                matched_productions.append(production)

        # 매칭된 프로덕션들 실행
        for production in matched_productions:
            await self._fire_production(production)

    async def _decision_phase(self) -> Optional[SOAROperator]:
        """의사결정 단계 - 연산자 선택"""
        candidate_operators = []

        for operator in self.operator_space.values():
            if self._matches_operator_conditions(operator):
                candidate_operators.append(operator)

        if not candidate_operators:
            return None

        # 선택 전략에 따라 연산자 선택
        if self.parameters.operator_selection_strategy == 'utility':
            return max(candidate_operators, key=lambda op: op.utility)
        elif self.parameters.operator_selection_strategy == 'random':
            import random
            return random.choice(candidate_operators)
        else:  # 'first'
            return candidate_operators[0]

    async def _application_phase(self, operator: SOAROperator) -> None:
        """적용 단계 - 연산자 실행"""
        for action in operator.actions:
            await self._execute_operator_action(action)

    def _matches_production_conditions(self, production: ProductionRule) -> bool:
        """프로덕션 조건 매칭"""
        for condition_attr, condition_value in production.conditions.items():
            # 작업 메모리에서 조건 확인
            found = False
            for element in self.working_memory:
                if element.attribute == condition_attr and element.value == condition_value:
                    found = True
                    break
            if not found:
                return False
        return True

    def _matches_operator_conditions(self, operator: SOAROperator) -> bool:
        """연산자 조건 매칭"""
        for condition_attr, condition_value in operator.conditions.items():
            # 현재 상태나 작업 메모리에서 조건 확인
            if self.soar_state and condition_attr in self.soar_state.attributes:
                if self.soar_state.attributes[condition_attr] != condition_value:
                    return False
            else:
                # 작업 메모리에서 확인
                found = False
                for element in self.working_memory:
                    if element.attribute == condition_attr and element.value == condition_value:
                        found = True
                        break
                if not found:
                    return False
        return True

    async def _fire_production(self, production: ProductionRule) -> None:
        """프로덕션 실행"""
        for action in production.actions:
            await self._execute_operator_action(action)

    async def _execute_operator_action(self, action: OperatorAction) -> None:
        """연산자 액션 실행"""
        if action.action_type == ActionType.ADD:
            await self._add_to_working_memory(action.identifier, action.attribute, action.value)

        elif action.action_type == ActionType.MODIFY:
            # 상태 수정
            if self.soar_state and action.identifier == "state":
                self.soar_state.attributes[action.attribute] = action.value

            # 작업 메모리 요소 수정
            for element in self.working_memory:
                if element.identifier == action.identifier and element.attribute == action.attribute:
                    element.value = action.value
                    break

        elif action.action_type == ActionType.REMOVE:
            # 작업 메모리에서 요소 제거
            self.working_memory = [
                element for element in self.working_memory
                if not (element.identifier == action.identifier and element.attribute == action.attribute)
            ]

    async def _chunking_learning(self, experience: Any) -> int:
        """청킹 학습 - 성공적인 시퀀스를 새로운 프로덕션으로 변환"""
        chunks_created = 0

        # 청킹 임계값을 넘는 성공적인 패턴 찾기
        if self.decision_cycle_count > self.parameters.chunking_threshold:
            # 새로운 청크 생성 로직 (간단한 구현)
            chunk_name = f"chunk_{len(self.production_memory)}"
            new_chunk = ProductionRule(
                name=chunk_name,
                conditions={"context": "learned"},
                actions=[
                    OperatorAction(
                        action_type=ActionType.ADD,
                        identifier="knowledge",
                        attribute="type",
                        value="chunk"
                    )
                ],
                utility=1.0
            )

            self.production_memory[chunk_name] = new_chunk
            chunks_created = 1

        return chunks_created

    async def _utility_learning(self, experience: Any, feedback: Any) -> None:
        """유틸리티 학습"""
        if hasattr(feedback, 'success') and feedback.success:
            # 성공한 연산자들의 유틸리티 증가
            for operator in self.operator_space.values():
                operator.utility += self.parameters.learning_rate * 0.1

    def _calculate_operator_confidence(self, operator: SOAROperator) -> float:
        """연산자 신뢰도 계산"""
        return min(0.9, max(0.1, operator.utility / 2.0))

    def _calculate_overall_confidence(self, steps: List[ProcessingStep]) -> float:
        """전체 신뢰도 계산"""
        if not steps:
            return 0.0
        return sum(step.confidence for step in steps) / len(steps)

    def _calculate_learning_improvement(self, chunks_created: int, feedback: Optional[Any] = None) -> float:
        """학습 개선도 계산"""
        improvement = chunks_created * 0.1
        if feedback and hasattr(feedback, 'reward'):
            improvement += getattr(feedback, 'reward', 0.0) * 0.1
        return improvement

    def _extract_result(self) -> Any:
        """결과 추출"""
        # 작업 메모리에서 결과 찾기
        for element in self.working_memory:
            if element.attribute == "result":
                return element.value

        # 상태에서 결과 찾기
        if self.soar_state:
            return self.soar_state.attributes.get("result", "No result found")

        return "Processing completed"

    async def perform_cognitive_processing(
        self,
        context: CognitiveContext,
        relevant_memory: Any
    ) -> CognitiveResult:
        """인지 처리 수행 (BaseCognitiveProcessor 인터페이스 구현)"""
        start_time = time.time()

        # SOAR 추론 실행
        reasoning_result = await self.process_reasoning(context.input, context)

        # 결과 변환
        processing_time_ms = (time.time() - start_time) * 1000

        return CognitiveResult(
            context_id=context.id,
            output=reasoning_result['result'],
            confidence=reasoning_result['confidence'],
            quality_metrics=QualityMetrics(
                accuracy=reasoning_result['confidence'],
                precision=reasoning_result['confidence'],
                confidence=reasoning_result['confidence']
            ),
            processing_steps=reasoning_result['steps'],
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
            'type': 'SOAR',
            'parameters': {
                'learning_rate': self.parameters.learning_rate,
                'chunking_threshold': self.parameters.chunking_threshold,
                'working_memory_capacity': self.parameters.working_memory_capacity,
                'operator_selection_strategy': self.parameters.operator_selection_strategy
            },
            'memory_stats': {
                'working_memory_elements': len(self.working_memory),
                'production_rules': len(self.production_memory),
                'operators': len(self.operator_space),
                'decision_cycles': self.decision_cycle_count
            },
            'current_state': self.soar_state.id if self.soar_state else None
        }


# 편의 함수들
def create_soar_model(
    name: str,
    parameters: Optional[SOARParameters] = None
) -> SOARModel:
    """SOAR 모델 생성 헬퍼"""
    return SOARModel(create_id(), name, parameters)


def create_soar_goal(
    goal_id: str,
    goal_type: GoalType,
    conditions: Dict[str, Any],
    priority: int = 1
) -> SOARGoal:
    """SOAR 목표 생성 헬퍼"""
    return SOARGoal(
        id=goal_id,
        goal_type=goal_type,
        conditions=conditions,
        priority=priority,
        status=GoalStatus.ACTIVE
    )


def create_soar_operator(
    name: str,
    conditions: Dict[str, Any],
    actions: List[OperatorAction],
    utility: float = 1.0
) -> SOAROperator:
    """SOAR 연산자 생성 헬퍼"""
    return SOAROperator(
        name=name,
        conditions=conditions,
        actions=actions,
        preferences=[],
        utility=utility
    )