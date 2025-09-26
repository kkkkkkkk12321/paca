"""
Cognitive Integrator

인지 프로세스 통합 관리자로, attention, perception, memory 시스템을
통합하여 일관된 인지 처리 파이프라인을 제공합니다.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Callable, Union
from uuid import uuid4, UUID

from ..base import BaseCognitiveProcessor
from ..memory import WorkingMemory, EpisodicMemory, LongTermMemory, MemoryItem
from .attention import (
    AttentionManager, AttentionTask, AttentionPriority,
    FocusController, FocusTarget, FocusLevel
)
from .perception import (
    PerceptionEngine, SensoryInput, ProcessingMode
)


class CognitiveState(Enum):
    """인지 상태"""
    IDLE = auto()              # 유휴 상태
    ATTENDING = auto()         # 주의 집중 상태
    PERCEIVING = auto()        # 지각 처리 상태
    REMEMBERING = auto()       # 기억 처리 상태
    INTEGRATING = auto()       # 통합 처리 상태
    LEARNING = auto()          # 학습 상태


class ProcessingPriority(Enum):
    """처리 우선순위"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class CognitiveRequest:
    """인지 처리 요청"""
    id: UUID = field(default_factory=uuid4)
    input_data: Any = None
    modality: str = "general"
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    require_attention: bool = True
    require_perception: bool = True
    require_memory: bool = True
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class CognitiveResult:
    """인지 처리 결과"""
    request_id: UUID
    attended_features: Dict[str, Any] = field(default_factory=dict)
    perceived_patterns: List[Dict[str, Any]] = field(default_factory=list)
    formed_concepts: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_memories: List[MemoryItem] = field(default_factory=list)
    stored_memories: List[UUID] = field(default_factory=list)
    processing_time_ms: float = 0.0
    confidence_score: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class CognitiveIntegrator(BaseCognitiveProcessor):
    """
    인지 프로세스 통합 관리자

    attention, perception, memory 시스템을 통합하여
    일관된 인지 처리를 수행합니다.
    """

    def __init__(self):
        super().__init__()

        # 인지 시스템 컴포넌트들
        self._attention_manager: Optional[AttentionManager] = None
        self._focus_controller: Optional[FocusController] = None
        self._perception_engine: Optional[PerceptionEngine] = None
        self._working_memory: Optional[WorkingMemory] = None
        self._episodic_memory: Optional[EpisodicMemory] = None
        self._longterm_memory: Optional[LongTermMemory] = None

        # 통합 상태
        self._state = CognitiveState.IDLE
        self._processing_queue: List[CognitiveRequest] = []
        self._active_requests: Dict[UUID, CognitiveRequest] = {}

        # 통합 메트릭
        self._total_processed = 0
        self._successful_processed = 0
        self._integration_efficiency = 0.0

        # 학습 및 적응
        self._learning_enabled = True
        self._adaptation_history: List[Dict[str, Any]] = []

    async def initialize(self) -> bool:
        """인지 통합 시스템 초기화"""
        try:
            self.logger.info("Initializing Cognitive Integrator...")

            # 서브 시스템 초기화
            await self._initialize_subsystems()

            # 백그라운드 프로세스 시작
            asyncio.create_task(self._process_request_queue())
            asyncio.create_task(self._monitor_cognitive_load())
            asyncio.create_task(self._optimize_integration())

            self._state = CognitiveState.IDLE
            self.logger.info("Cognitive Integrator initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Cognitive Integrator: {e}")
            return False

    async def process_cognitive_request(self, request: CognitiveRequest) -> CognitiveResult:
        """
        인지 처리 요청 수행

        Args:
            request: 인지 처리 요청

        Returns:
            인지 처리 결과
        """
        try:
            start_time = time.time()

            # 요청 전처리
            preprocessed_request = await self._preprocess_request(request)

            # 통합 처리 파이프라인 실행
            result = await self._execute_cognitive_pipeline(preprocessed_request)

            # 처리 시간 계산
            result.processing_time_ms = (time.time() - start_time) * 1000

            # 학습 및 적응
            if self._learning_enabled:
                await self._learn_from_processing(preprocessed_request, result)

            # 메트릭 업데이트
            self._update_metrics(result)

            return result

        except Exception as e:
            self.logger.error(f"Error processing cognitive request {request.id}: {e}")
            return CognitiveResult(
                request_id=request.id,
                success=False,
                error_message=str(e)
            )

    async def set_cognitive_focus(self, targets: List[str],
                                 attention_weights: Optional[Dict[str, float]] = None) -> bool:
        """
        인지 집중 설정

        Args:
            targets: 집중할 대상들
            attention_weights: 주의 가중치

        Returns:
            설정 성공 여부
        """
        try:
            # 주의 시스템 집중 설정
            if self._attention_manager and self._focus_controller:
                # 주의 자원 할당
                for target in targets:
                    attention_task = AttentionTask(
                        name=f"focus_{target}",
                        priority=AttentionPriority.HIGH,
                        resource_required=20.0
                    )
                    await self._attention_manager.allocate_attention(attention_task)

                # 집중 대상 설정
                focus_targets = []
                for target in targets:
                    focus_target = FocusTarget(
                        name=target,
                        importance=attention_weights.get(target, 0.8) if attention_weights else 0.8,
                        urgency=0.7
                    )
                    focus_targets.append(focus_target)

                # 집중 시작
                for focus_target in focus_targets:
                    await self._focus_controller.start_focus(focus_target, FocusLevel.HIGH)

            # 지각 시스템 집중 설정
            if self._perception_engine:
                await self._perception_engine.set_attention_focus(targets, attention_weights)

            self.logger.info(f"Set cognitive focus to targets: {targets}")
            return True

        except Exception as e:
            self.logger.error(f"Error setting cognitive focus: {e}")
            return False

    async def get_cognitive_state(self) -> Dict[str, Any]:
        """현재 인지 상태 조회"""
        state_info = {
            "cognitive_state": self._state.name,
            "active_requests": len(self._active_requests),
            "queued_requests": len(self._processing_queue),
            "total_processed": self._total_processed,
            "success_rate": (
                self._successful_processed / max(1, self._total_processed)
            ),
            "integration_efficiency": self._integration_efficiency
        }

        # 서브시스템 상태 추가
        if self._attention_manager:
            attention_state = await self._attention_manager.get_attention_status()
            state_info["attention"] = attention_state

        if self._focus_controller:
            focus_state = await self._focus_controller.get_current_focus_state()
            state_info["focus"] = focus_state

        if self._perception_engine:
            perception_state = await self._perception_engine.get_perception_state()
            state_info["perception"] = perception_state

        # 메모리 상태 추가
        memory_states = {}
        if self._working_memory:
            working_stats = await self._working_memory.get_stats()
            memory_states["working"] = working_stats

        if self._episodic_memory:
            episodic_stats = await self._episodic_memory.get_stats()
            memory_states["episodic"] = episodic_stats

        if self._longterm_memory:
            longterm_stats = await self._longterm_memory.get_stats()
            memory_states["longterm"] = longterm_stats

        state_info["memory"] = memory_states

        return state_info

    async def _initialize_subsystems(self) -> None:
        """서브 시스템 초기화"""
        try:
            # Attention 시스템 초기화
            from .attention import create_attention_manager, create_focus_controller
            self._attention_manager = await create_attention_manager()
            self._focus_controller = await create_focus_controller()

            # Perception 시스템 초기화
            from .perception import create_perception_engine
            self._perception_engine = await create_perception_engine()

            # Memory 시스템 초기화
            from ..memory.working import WorkingMemory
            from ..memory.episodic import EpisodicMemory
            from ..memory.longterm import LongTermMemory

            self._working_memory = WorkingMemory()
            await self._working_memory.initialize()

            self._episodic_memory = EpisodicMemory()
            await self._episodic_memory.initialize()

            self._longterm_memory = LongTermMemory()
            await self._longterm_memory.initialize()

            self.logger.info("All cognitive subsystems initialized")

        except Exception as e:
            self.logger.error(f"Error initializing subsystems: {e}")
            raise

    async def _preprocess_request(self, request: CognitiveRequest) -> CognitiveRequest:
        """요청 전처리"""
        try:
            # 우선순위 조정
            if request.priority == ProcessingPriority.CRITICAL:
                # 긴급 요청은 즉시 처리
                self._processing_queue.insert(0, request)
            elif len(self._processing_queue) > 10:
                # 대기열이 길면 우선순위 상향 조정
                if request.priority.value > ProcessingPriority.HIGH.value:
                    request.priority = ProcessingPriority.HIGH

            # 컨텍스트 보강
            if "timestamp" not in request.context:
                request.context["timestamp"] = time.time()

            if "processing_started" not in request.context:
                request.context["processing_started"] = time.time()

            return request

        except Exception as e:
            self.logger.error(f"Error preprocessing request: {e}")
            return request

    async def _execute_cognitive_pipeline(self, request: CognitiveRequest) -> CognitiveResult:
        """인지 처리 파이프라인 실행"""
        try:
            result = CognitiveResult(request_id=request.id)

            # 1. 주의 처리 (Attention)
            if request.require_attention and self._attention_manager:
                attended_features = await self._process_attention(request)
                result.attended_features = attended_features

            # 2. 지각 처리 (Perception)
            if request.require_perception and self._perception_engine:
                perception_result = await self._process_perception(request, result.attended_features)
                result.perceived_patterns = perception_result.get("patterns", [])
                result.formed_concepts = perception_result.get("concepts", [])

            # 3. 메모리 처리 (Memory)
            if request.require_memory:
                memory_result = await self._process_memory(request, result)
                result.retrieved_memories = memory_result.get("retrieved", [])
                result.stored_memories = memory_result.get("stored", [])

            # 4. 통합 및 일관성 검사
            await self._integrate_and_validate(result)

            # 신뢰도 계산
            result.confidence_score = await self._calculate_confidence(result)

            return result

        except Exception as e:
            self.logger.error(f"Error in cognitive pipeline: {e}")
            return CognitiveResult(
                request_id=request.id,
                success=False,
                error_message=str(e)
            )

    async def _process_attention(self, request: CognitiveRequest) -> Dict[str, Any]:
        """주의 처리"""
        try:
            attended_features = {}

            # 주의 자원 할당
            attention_task = AttentionTask(
                name=f"process_{request.id}",
                priority=self._convert_to_attention_priority(request.priority),
                resource_required=15.0,
                context=request.context
            )

            # 주의 할당 시도
            if await self._attention_manager.allocate_attention(attention_task):
                # 주의 집중된 특징 추출
                attended_features = {
                    "attention_allocated": True,
                    "focus_strength": 0.8,
                    "attended_modality": request.modality,
                    "attention_context": request.context
                }

                # 집중 제어
                if self._focus_controller:
                    focus_target = FocusTarget(
                        name=f"target_{request.id}",
                        importance=0.7,
                        urgency=0.6,
                        context=request.context
                    )

                    if await self._focus_controller.start_focus(focus_target, FocusLevel.MEDIUM):
                        attended_features["focus_active"] = True

            else:
                attended_features = {
                    "attention_allocated": False,
                    "reason": "insufficient_resources"
                }

            return attended_features

        except Exception as e:
            self.logger.error(f"Error in attention processing: {e}")
            return {"error": str(e)}

    async def _process_perception(self, request: CognitiveRequest,
                                 attended_features: Dict[str, Any]) -> Dict[str, Any]:
        """지각 처리"""
        try:
            # 감각 입력 생성
            sensory_input = SensoryInput(
                modality=request.modality,
                data=request.input_data,
                intensity=attended_features.get("focus_strength", 1.0),
                confidence=1.0,
                metadata=request.context
            )

            # 처리 모드 결정
            if attended_features.get("attention_allocated", False):
                processing_mode = ProcessingMode.INTERACTIVE
            else:
                processing_mode = ProcessingMode.BOTTOM_UP

            # 지각 처리 실행
            perception_result = await self._perception_engine.process_input(
                sensory_input, processing_mode
            )

            return {
                "patterns": perception_result.recognized_patterns,
                "concepts": perception_result.formed_concepts,
                "objects": perception_result.perceived_objects,
                "confidence": perception_result.confidence_score
            }

        except Exception as e:
            self.logger.error(f"Error in perception processing: {e}")
            return {"patterns": [], "concepts": [], "objects": []}

    async def _process_memory(self, request: CognitiveRequest,
                            result: CognitiveResult) -> Dict[str, Any]:
        """메모리 처리"""
        try:
            memory_result = {"retrieved": [], "stored": []}

            # 관련 기억 검색
            if result.perceived_patterns or result.formed_concepts:
                # 패턴 기반 기억 검색
                search_queries = []

                for pattern in result.perceived_patterns:
                    pattern_name = pattern.get("name", "")
                    if pattern_name:
                        search_queries.append(pattern_name)

                for concept in result.formed_concepts:
                    concept_name = concept.get("name", "")
                    if concept_name:
                        search_queries.append(concept_name)

                # 각 메모리 시스템에서 검색
                for query in search_queries[:5]:  # 최대 5개 쿼리
                    # Working Memory 검색
                    if self._working_memory:
                        working_memories = await self._working_memory.search(query)
                        memory_result["retrieved"].extend(working_memories)

                    # Episodic Memory 검색
                    if self._episodic_memory:
                        episodic_memories = await self._episodic_memory.search(query)
                        memory_result["retrieved"].extend(episodic_memories)

                    # Long-term Memory 검색
                    if self._longterm_memory:
                        longterm_memories = await self._longterm_memory.search(query)
                        memory_result["retrieved"].extend(longterm_memories)

            # 새로운 기억 저장
            if request.input_data and result.success:
                # Working Memory에 저장
                if self._working_memory:
                    working_item = MemoryItem(
                        content=request.input_data,
                        tags=[request.modality, "processed"],
                        metadata=request.context
                    )
                    if await self._working_memory.store(working_item):
                        memory_result["stored"].append(working_item.id)

                # 중요한 정보는 Episodic Memory에도 저장
                if (result.confidence_score > 0.8 and
                    request.priority.value <= ProcessingPriority.HIGH.value):

                    if self._episodic_memory:
                        episodic_item = MemoryItem(
                            content={
                                "input": request.input_data,
                                "patterns": result.perceived_patterns,
                                "concepts": result.formed_concepts
                            },
                            tags=[request.modality, "high_confidence"],
                            metadata=request.context
                        )
                        if await self._episodic_memory.store(episodic_item):
                            memory_result["stored"].append(episodic_item.id)

            return memory_result

        except Exception as e:
            self.logger.error(f"Error in memory processing: {e}")
            return {"retrieved": [], "stored": []}

    async def _integrate_and_validate(self, result: CognitiveResult) -> None:
        """통합 및 일관성 검사"""
        try:
            # 결과 간 일관성 확인
            consistency_score = await self._check_consistency(result)

            # 일관성이 낮으면 보정
            if consistency_score < 0.7:
                await self._reconcile_inconsistencies(result)

            # 통합 메타데이터 추가
            result.attended_features["integration_timestamp"] = time.time()
            result.attended_features["consistency_score"] = consistency_score

        except Exception as e:
            self.logger.error(f"Error in integration and validation: {e}")

    async def _check_consistency(self, result: CognitiveResult) -> float:
        """결과 일관성 확인"""
        try:
            consistency_scores = []

            # 지각된 패턴과 형성된 개념 간 일치도
            if result.perceived_patterns and result.formed_concepts:
                pattern_names = {p.get("name", "") for p in result.perceived_patterns}
                concept_names = {c.get("name", "") for c in result.formed_concepts}

                overlap = len(pattern_names & concept_names)
                total = len(pattern_names | concept_names)

                if total > 0:
                    consistency_scores.append(overlap / total)

            # 검색된 기억과 현재 처리 간 관련성
            if result.retrieved_memories and (result.perceived_patterns or result.formed_concepts):
                # 간단한 관련성 점수 계산
                relevance_score = min(len(result.retrieved_memories) / 5, 1.0)
                consistency_scores.append(relevance_score)

            # 전체 일관성 점수
            if consistency_scores:
                return sum(consistency_scores) / len(consistency_scores)
            else:
                return 0.8  # 기본값

        except Exception as e:
            self.logger.error(f"Error checking consistency: {e}")
            return 0.5

    async def _reconcile_inconsistencies(self, result: CognitiveResult) -> None:
        """일관성 문제 해결"""
        try:
            # 신뢰도 기반 결과 조정
            if result.confidence_score < 0.6:
                # 낮은 신뢰도 결과 필터링
                filtered_patterns = [
                    p for p in result.perceived_patterns
                    if p.get("confidence", 0) >= 0.7
                ]
                result.perceived_patterns = filtered_patterns

                filtered_concepts = [
                    c for c in result.formed_concepts
                    if c.get("confidence", 0) >= 0.7
                ]
                result.formed_concepts = filtered_concepts

        except Exception as e:
            self.logger.error(f"Error reconciling inconsistencies: {e}")

    async def _calculate_confidence(self, result: CognitiveResult) -> float:
        """전체 신뢰도 계산"""
        try:
            confidence_scores = []

            # 주의 처리 신뢰도
            if result.attended_features.get("attention_allocated", False):
                confidence_scores.append(0.8)
            else:
                confidence_scores.append(0.4)

            # 지각 처리 신뢰도
            if result.perceived_patterns:
                pattern_confidences = [
                    p.get("confidence", 0.5) for p in result.perceived_patterns
                ]
                if pattern_confidences:
                    confidence_scores.append(sum(pattern_confidences) / len(pattern_confidences))

            if result.formed_concepts:
                concept_confidences = [
                    c.get("confidence", 0.5) for c in result.formed_concepts
                ]
                if concept_confidences:
                    confidence_scores.append(sum(concept_confidences) / len(concept_confidences))

            # 메모리 처리 신뢰도
            if result.retrieved_memories:
                confidence_scores.append(0.7)  # 기억 검색 성공
            if result.stored_memories:
                confidence_scores.append(0.8)  # 기억 저장 성공

            # 전체 신뢰도
            if confidence_scores:
                return sum(confidence_scores) / len(confidence_scores)
            else:
                return 0.5

        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5

    def _convert_to_attention_priority(self, priority: ProcessingPriority) -> AttentionPriority:
        """처리 우선순위를 주의 우선순위로 변환"""
        mapping = {
            ProcessingPriority.CRITICAL: AttentionPriority.CRITICAL,
            ProcessingPriority.HIGH: AttentionPriority.HIGH,
            ProcessingPriority.NORMAL: AttentionPriority.NORMAL,
            ProcessingPriority.LOW: AttentionPriority.LOW,
            ProcessingPriority.BACKGROUND: AttentionPriority.BACKGROUND
        }
        return mapping.get(priority, AttentionPriority.NORMAL)

    async def _learn_from_processing(self, request: CognitiveRequest,
                                   result: CognitiveResult) -> None:
        """처리 결과로부터 학습"""
        try:
            learning_data = {
                "request_type": request.modality,
                "priority": request.priority.name,
                "success": result.success,
                "confidence": result.confidence_score,
                "processing_time": result.processing_time_ms,
                "timestamp": time.time()
            }

            self._adaptation_history.append(learning_data)

            # 히스토리 길이 제한
            if len(self._adaptation_history) > 100:
                self._adaptation_history = self._adaptation_history[-50:]

            # 성능 패턴 분석
            if len(self._adaptation_history) >= 10:
                await self._analyze_performance_patterns()

        except Exception as e:
            self.logger.error(f"Error learning from processing: {e}")

    async def _analyze_performance_patterns(self) -> None:
        """성능 패턴 분석"""
        try:
            recent_data = self._adaptation_history[-10:]

            # 성공률 분석
            success_rate = sum(1 for d in recent_data if d["success"]) / len(recent_data)

            # 평균 신뢰도 분석
            avg_confidence = sum(d["confidence"] for d in recent_data) / len(recent_data)

            # 평균 처리 시간 분석
            avg_time = sum(d["processing_time"] for d in recent_data) / len(recent_data)

            # 통합 효율성 업데이트
            self._integration_efficiency = (success_rate * 0.4 + avg_confidence * 0.4 +
                                          min(1000 / max(avg_time, 1), 1.0) * 0.2)

            self.logger.debug(f"Integration efficiency: {self._integration_efficiency:.3f}")

        except Exception as e:
            self.logger.error(f"Error analyzing performance patterns: {e}")

    def _update_metrics(self, result: CognitiveResult) -> None:
        """메트릭 업데이트"""
        self._total_processed += 1

        if result.success:
            self._successful_processed += 1

    async def _process_request_queue(self) -> None:
        """요청 큐 처리 백그라운드 프로세스"""
        while True:
            try:
                if self._processing_queue and len(self._active_requests) < 5:
                    # 우선순위 순으로 정렬
                    self._processing_queue.sort(key=lambda r: r.priority.value)

                    # 다음 요청 처리
                    next_request = self._processing_queue.pop(0)
                    self._active_requests[next_request.id] = next_request

                    # 비동기 처리 시작
                    asyncio.create_task(self._handle_request(next_request))

                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error processing request queue: {e}")
                await asyncio.sleep(1.0)

    async def _handle_request(self, request: CognitiveRequest) -> None:
        """개별 요청 처리"""
        try:
            result = await self.process_cognitive_request(request)

            # 처리 완료 정리
            if request.id in self._active_requests:
                del self._active_requests[request.id]

        except Exception as e:
            self.logger.error(f"Error handling request {request.id}: {e}")
            if request.id in self._active_requests:
                del self._active_requests[request.id]

    async def _monitor_cognitive_load(self) -> None:
        """인지 부하 모니터링"""
        while True:
            try:
                # 전체 시스템 부하 확인
                total_load = len(self._active_requests) + len(self._processing_queue)

                if total_load > 15:  # 과부하 상태
                    self._state = CognitiveState.INTEGRATING
                    # 낮은 우선순위 요청 지연
                    low_priority_requests = [
                        r for r in self._processing_queue
                        if r.priority.value >= ProcessingPriority.LOW.value
                    ]

                    for request in low_priority_requests[:5]:
                        self._processing_queue.remove(request)
                        # 나중에 다시 처리하도록 지연
                        await asyncio.sleep(0.1)
                        self._processing_queue.append(request)

                elif total_load == 0:
                    self._state = CognitiveState.IDLE
                else:
                    self._state = CognitiveState.INTEGRATING

                await asyncio.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Error monitoring cognitive load: {e}")
                await asyncio.sleep(5.0)

    async def _optimize_integration(self) -> None:
        """통합 최적화 백그라운드 프로세스"""
        while True:
            try:
                # 5분마다 최적화 수행
                if self._integration_efficiency < 0.7:
                    # 효율성이 낮으면 시스템 조정
                    await self._adjust_system_parameters()

                await asyncio.sleep(300)  # 5분

            except Exception as e:
                self.logger.error(f"Error in integration optimization: {e}")
                await asyncio.sleep(300)

    async def _adjust_system_parameters(self) -> None:
        """시스템 파라미터 조정"""
        try:
            # 주의 시스템 조정
            if self._attention_manager:
                attention_status = await self._attention_manager.get_attention_status()
                if attention_status.get("efficiency_score", 1.0) < 0.7:
                    # 주의 자원 한계 조정 등
                    pass

            # 지각 시스템 조정
            if self._perception_engine:
                perception_state = await self._perception_engine.get_perception_state()
                if perception_state.get("success_rate", 1.0) < 0.8:
                    # 처리 임계값 조정 등
                    pass

            self.logger.info("Adjusted system parameters for better integration")

        except Exception as e:
            self.logger.error(f"Error adjusting system parameters: {e}")


async def create_cognitive_integrator() -> CognitiveIntegrator:
    """CognitiveIntegrator 인스턴스 생성 및 초기화"""
    integrator = CognitiveIntegrator()
    await integrator.initialize()
    return integrator