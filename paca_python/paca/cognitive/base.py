"""
Cognitive Base Module
인지 시스템의 기본 클래스들과 공통 인터페이스
"""

import asyncio
import time
import os
import re
from typing import Any, Dict, List, Optional, Union

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - 환경에 따라 psutil이 없을 수 있음
    psutil = None
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..core.types.base import ID, Timestamp, KeyValuePair, Priority, Result, create_id, current_timestamp
from ..core.events.emitter import EventEmitter
from ..core.utils.logger import create_logger
from ..core.errors.cognitive import CognitiveError, MemoryError, ModelError


class CognitiveState(Enum):
    """인지 처리 상태"""
    IDLE = 'idle'
    INITIALIZING = 'initializing'
    PROCESSING = 'processing'
    ANALYZING = 'analyzing'
    LEARNING = 'learning'
    COMPLETED = 'completed'
    ERROR = 'error'


class CognitiveTaskType(Enum):
    """인지 작업 타입"""
    REASONING = 'reasoning'
    LEARNING = 'learning'
    ANALYSIS = 'analysis'
    SIMULATION = 'simulation'
    PREDICTION = 'prediction'
    CLASSIFICATION = 'classification'


@dataclass
class QualityMetrics:
    """품질 메트릭"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    consistency: float = 0.0
    completeness: float = 0.0
    efficiency: float = 0.0
    reliability: float = 0.0
    coherence: float = 0.0
    relevance: float = 0.0
    confidence: float = 0.0


@dataclass
class CognitiveModel:
    """인지 모델 정보"""
    name: str
    version: str
    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemorySystem:
    """메모리 시스템 정보"""
    type: str
    capacity: int
    persistence: bool = True
    configuration: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveConfig:
    """인지 처리 설정"""
    id: ID
    name: str
    description: Optional[str]
    model: CognitiveModel
    memory: MemorySystem
    quality_threshold: float
    max_processing_time: float
    enable_learning: bool
    enable_memory_persistence: bool
    debug_mode: bool = False


@dataclass(frozen=True)
class CognitiveContext:
    """인지 컨텍스트"""
    id: ID
    task_type: CognitiveTaskType
    timestamp: Timestamp
    input: Any
    expected_output: Optional[Any] = None
    metadata: KeyValuePair = field(default_factory=dict)
    quality_requirements: Optional[QualityMetrics] = None
    constraints: Optional[KeyValuePair] = None


@dataclass(frozen=True)
class ProcessingStep:
    """처리 단계"""
    id: ID
    name: str
    type: str
    start_time: Timestamp
    end_time: Timestamp
    input: Any
    output: Any
    confidence: float
    metadata: KeyValuePair


@dataclass(frozen=True)
class MemoryUpdate:
    """메모리 업데이트"""
    id: ID
    type: str  # 'store' | 'update' | 'delete' | 'consolidate'
    target: str
    data: Any
    timestamp: Timestamp
    confidence: float


@dataclass(frozen=True)
class LearningData:
    """학습 데이터"""
    id: ID
    type: str  # 'supervised' | 'unsupervised' | 'reinforcement'
    input: Any
    expected_output: Optional[Any]
    actual_output: Any
    feedback: Optional[Any]
    importance: float
    timestamp: Timestamp


@dataclass(frozen=True)
class ResourceUsage:
    """리소스 사용량"""
    memory_mb: float
    cpu_percent: float
    processing_time_ms: float
    api_calls: int
    cache_hits: int
    cache_misses: int


@dataclass(frozen=True)
class CognitiveResult:
    """인지 처리 결과"""
    context_id: ID
    output: Any
    confidence: float
    quality_metrics: QualityMetrics
    processing_steps: List[ProcessingStep]
    memory_updates: List[MemoryUpdate]
    learning_data: Optional[LearningData]
    processing_time_ms: float
    resource_usage: ResourceUsage


@dataclass
class CognitiveStatistics:
    """인지 프로세서 통계"""
    process_count: int = 0
    success_count: int = 0
    error_count: int = 0
    success_rate: float = 0.0
    average_processing_time_ms: float = 0.0
    last_processed_at: Optional[Timestamp] = None
    current_state: CognitiveState = CognitiveState.IDLE


class BaseCognitiveProcessor(ABC):
    """기본 인지 프로세서 추상 클래스"""

    def __init__(self, config: CognitiveConfig, events: Optional[EventEmitter] = None):
        self.config = config
        self.logger = create_logger(config.name)
        self.events = events
        self.state = CognitiveState.IDLE

        # 통계 관련 변수들
        self._statistics = CognitiveStatistics()
        self._total_processing_time = 0.0

        if config.debug_mode:
            self.logger.info(
                'Cognitive processor initialized',
                meta={
                    'id': config.id,
                    'name': config.name,
                    'model': config.model.name,
                    'memory_type': config.memory.type
                }
            )

    async def process(self, context: CognitiveContext) -> CognitiveResult:
        """인지 작업 실행"""
        start_time = time.time()
        self.state = CognitiveState.INITIALIZING
        self._statistics.process_count += 1

        if self.events:
            await self.events.emit(
                'cognitive.process.started',
                {
                    'processor_id': self.config.id,
                    'context_id': context.id,
                    'task_type': context.task_type.value
                }
            )

        try:
            # 1. 전처리 및 검증
            await self._validate_input(context)
            self.state = CognitiveState.PROCESSING

            # 2. 관련 메모리 조회
            relevant_memory = await self._retrieve_memory(context)

            # 3. 실제 인지 처리
            result = await self.perform_cognitive_processing(context, relevant_memory)

            # 4. 품질 검증
            await self._validate_quality(result, context.quality_requirements)

            # 5. 메모리 업데이트
            if self.config.enable_memory_persistence:
                await self._update_memory(result.memory_updates)

            # 6. 학습 데이터 저장
            if self.config.enable_learning and result.learning_data:
                await self._store_learning_data(result.learning_data)

            self.state = CognitiveState.COMPLETED
            self._statistics.success_count += 1
            self._statistics.last_processed_at = current_timestamp()

            processing_time = (time.time() - start_time) * 1000
            self._total_processing_time += processing_time

            if self.events:
                await self.events.emit(
                    'cognitive.process.completed',
                    {
                        'processor_id': self.config.id,
                        'context_id': context.id,
                        'confidence': result.confidence,
                        'processing_time_ms': result.processing_time_ms
                    }
                )

            return result

        except Exception as error:
            self.state = CognitiveState.ERROR
            self._statistics.error_count += 1

            cognitive_error = error if isinstance(error, CognitiveError) else ModelError(
                message=f"Cognitive processing failed: {str(error)}",
                model_name=self.config.model.name,
                model_state=self.state.value,
                input_data=context.input
            )

            self.logger.error(
                'Cognitive processing failed',
                error=cognitive_error,
                meta={
                    'context_id': context.id,
                    'task_type': context.task_type.value,
                    'processing_time_ms': (time.time() - start_time) * 1000
                }
            )

            if self.events:
                await self.events.emit(
                    'cognitive.process.failed',
                    {
                        'processor_id': self.config.id,
                        'context_id': context.id,
                        'error': str(cognitive_error)
                    }
                )

            raise cognitive_error

    def get_state(self) -> CognitiveState:
        """프로세서 상태 조회"""
        return self.state

    def get_statistics(self) -> CognitiveStatistics:
        """프로세서 통계 조회"""
        if self._statistics.process_count > 0:
            self._statistics.success_rate = self._statistics.success_count / self._statistics.process_count
            self._statistics.average_processing_time_ms = self._total_processing_time / self._statistics.process_count

        self._statistics.current_state = self.state
        return self._statistics

    def get_config(self) -> CognitiveConfig:
        """프로세서 설정 조회"""
        return self.config

    @abstractmethod
    async def perform_cognitive_processing(
        self,
        context: CognitiveContext,
        relevant_memory: Any
    ) -> CognitiveResult:
        """구체적인 인지 처리 (하위 클래스에서 구현)"""
        pass

    async def _validate_input(self, context: CognitiveContext) -> None:
        """입력 검증"""
        if context.input is None:
            raise ModelError(
                message="Input is required",
                model_name=self.config.model.name,
                model_state=self.state.value,
                input_data=context.input
            )

        if not context.task_type:
            raise ModelError(
                message="Task type is required",
                model_name=self.config.model.name,
                model_state=self.state.value,
                input_data=context.input
            )

    async def _retrieve_memory(self, context: CognitiveContext) -> Any:
        """관련 메모리 조회"""
        try:
            if not self.config.memory:
                return None

            # 메모리 검색 로직
            search_query = {
                'query': context.input,
                'task_type': context.task_type.value,
                'limit': 10,
                'threshold': 0.7
            }

            return await self.search_memory(search_query)

        except Exception as error:
            self.logger.warn(
                'Memory retrieval failed',
                meta={
                    'error': str(error),
                    'context_id': context.id
                }
            )
            return None

    async def search_memory(self, query: Dict[str, Any]) -> Any:
        """메모리 검색 (하위 클래스에서 구현 가능)"""
        return None

    async def _validate_quality(
        self,
        result: CognitiveResult,
        requirements: Optional[QualityMetrics]
    ) -> None:
        """품질 검증"""
        if not requirements:
            return

        metrics = result.quality_metrics
        threshold = self.config.quality_threshold

        # 전체적인 신뢰도 확인
        if result.confidence < threshold:
            raise CognitiveError(
                message=f"Confidence {result.confidence} below threshold {threshold}",
                component="quality_validation",
                metadata={
                    'context_id': result.context_id,
                    'confidence': result.confidence,
                    'threshold': threshold
                }
            )

        # 개별 품질 지표 확인
        quality_checks = [
            ('coherence', metrics.coherence, requirements.coherence),
            ('completeness', metrics.completeness, requirements.completeness),
            ('relevance', metrics.relevance, requirements.relevance)
        ]

        for metric_name, actual_value, required_value in quality_checks:
            if required_value and actual_value < required_value:
                raise CognitiveError(
                    message=f"{metric_name} {actual_value} below requirement {required_value}",
                    component="quality_validation",
                    metadata={
                        'context_id': result.context_id,
                        'metric': metric_name,
                        'actual': actual_value,
                        'required': required_value
                    }
                )

    async def _update_memory(self, updates: List[MemoryUpdate]) -> None:
        """메모리 업데이트"""
        try:
            for update in updates:
                await self.apply_memory_update(update)

            self.logger.debug(
                'Memory updates applied',
                meta={'update_count': len(updates)}
            )
        except Exception as error:
            raise MemoryError(
                message=f"Failed to update memory: {str(error)}",
                memory_type=self.config.memory.type,
                operation="batch_update"
            )

    async def apply_memory_update(self, update: MemoryUpdate) -> None:
        """단일 메모리 업데이트 적용 (하위 클래스에서 구현 가능)"""
        pass

    async def _store_learning_data(self, data: LearningData) -> None:
        """학습 데이터 저장"""
        try:
            self.logger.debug(
                'Learning data stored',
                meta={
                    'data_id': data.id,
                    'type': data.type,
                    'importance': data.importance
                }
            )
        except Exception as error:
            self.logger.warn(
                'Failed to store learning data',
                meta={
                    'error': str(error),
                    'data_id': data.id
                }
            )

    def _calculate_resource_usage(self, start_time: float) -> ResourceUsage:
        """리소스 사용량 계산"""
        processing_time_ms = (time.time() - start_time) * 1000

        # 메모리 사용량 (MB)
        if psutil is not None:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
        else:
            memory_mb = 0.0
            cpu_percent = 0.0

        return ResourceUsage(
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            processing_time_ms=processing_time_ms,
            api_calls=0,  # 실제 구현에서 추적
            cache_hits=0,  # 실제 구현에서 추적
            cache_misses=0  # 실제 구현에서 추적
        )

    def _calculate_quality_metrics(
        self,
        input_data: Any,
        output: Any,
        processing_steps: List[ProcessingStep]
    ) -> QualityMetrics:
        """품질 메트릭 계산"""
        return QualityMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            consistency=0.85,
            completeness=self._calculate_completeness(input_data, output),
            efficiency=0.80,
            reliability=0.85,
            coherence=self._calculate_coherence(output),
            relevance=self._calculate_relevance(input_data, output),
            confidence=self._calculate_confidence(processing_steps)
        )

    def _calculate_confidence(self, steps: List[ProcessingStep]) -> float:
        """신뢰도 계산"""
        if not steps:
            return 0.5
        return sum(step.confidence for step in steps) / len(steps)

    def _calculate_coherence(self, output: Any) -> float:
        """일관성 계산"""
        # 실제 구현에서는 더 복잡한 로직 사용
        return 0.8

    def _calculate_completeness(self, input_data: Any, output: Any) -> float:
        """완성도 계산"""
        # 실제 구현에서는 더 복잡한 로직 사용
        return 0.85

    def _calculate_relevance(self, input_data: Any, output: Any) -> float:
        """관련성 계산"""
        # 실제 구현에서는 더 복잡한 로직 사용
        return 0.9


class SimpleCognitiveProcessor(BaseCognitiveProcessor):
    """Phase 1용 기본 규칙 기반 인지 프로세서"""

    def __init__(self, config: CognitiveConfig, events: Optional[EventEmitter] = None):
        super().__init__(config, events)

    async def perform_cognitive_processing(
        self,
        context: CognitiveContext,
        relevant_memory: Any
    ) -> CognitiveResult:
        start_time = time.time()
        normalized_input = str(context.input).strip()
        keywords = self._extract_keywords(normalized_input)

        analysis_output = {
            'summary': normalized_input[:200],
            'keywords': keywords,
            'metadata': context.metadata or {},
            'memory_matches': relevant_memory or []
        }

        processing_step = create_processing_step(
            name='basic_analysis',
            step_type='analysis',
            input_data=context.input,
            output_data=analysis_output,
            confidence=0.75 if normalized_input else 0.5
        )

        processing_steps = [processing_step]
        quality_metrics = self._calculate_quality_metrics(context.input, analysis_output, processing_steps)
        resource_usage = self._calculate_resource_usage(start_time)

        return CognitiveResult(
            context_id=context.id,
            output=analysis_output,
            confidence=quality_metrics.confidence,
            quality_metrics=quality_metrics,
            processing_steps=processing_steps,
            memory_updates=[],
            learning_data=None,
            processing_time_ms=resource_usage.processing_time_ms,
            resource_usage=resource_usage
        )

    def _extract_keywords(self, text: str) -> List[str]:
        if not text:
            return []

        tokens = re.findall(r"[가-힣a-zA-Z0-9]+", text)
        filtered = [token.lower() for token in tokens if len(token) > 1]
        # 상위 5개 키워드만 추출 (등장 빈도 기준)
        frequency: Dict[str, int] = {}
        for token in filtered:
            frequency[token] = frequency.get(token, 0) + 1

        sorted_tokens = sorted(frequency.items(), key=lambda item: item[1], reverse=True)
        return [token for token, _ in sorted_tokens[:5]]


def create_default_cognitive_processor(events: Optional[EventEmitter] = None) -> BaseCognitiveProcessor:
    """기본 규칙 기반 인지 프로세서 생성"""
    default_config = CognitiveConfig(
        id=create_id(),
        name="DefaultCognitiveProcessor",
        description="기본 규칙 기반 인지 프로세서",
        model=CognitiveModel(
            name="RuleBasedModel",
            version="1.0.0",
            type="rule_based",
            parameters={'language': 'ko'}
        ),
        memory=MemorySystem(
            type="in_memory",
            capacity=1024,
            persistence=False,
            configuration={}
        ),
        quality_threshold=0.5,
        max_processing_time=1000.0,
        enable_learning=False,
        enable_memory_persistence=False,
        debug_mode=False
    )

    return SimpleCognitiveProcessor(default_config, events)


class CognitiveSystem:
    """통합 인지 시스템"""

    def __init__(self, processors: List[BaseCognitiveProcessor], events: Optional[EventEmitter] = None):
        self.processors = {processor.config.id: processor for processor in processors}
        self.events = events
        self.logger = create_logger('CognitiveSystem')
        self._is_initialized = False
        self.default_processor_id: Optional[ID] = next(iter(self.processors), None)

    async def initialize(self) -> Result[bool]:
        """인지 시스템 초기화"""
        if self._is_initialized:
            return Result.success(True)

        try:
            if not self.processors:
                default_processor = create_default_cognitive_processor(self.events)
                self.register_processor(default_processor)

            self._is_initialized = True
            return Result.success(True)

        except Exception as error:
            return Result.failure(CognitiveError(
                message=f"Failed to initialize cognitive system: {str(error)}",
                component="system"
            ))

    async def cleanup(self) -> Result[bool]:
        """인지 시스템 정리"""
        try:
            self._is_initialized = False
            return Result.success(True)

        except Exception as error:
            return Result.failure(CognitiveError(
                message=f"Failed to cleanup cognitive system: {str(error)}",
                component="system"
            ))

    def register_processor(self, processor: BaseCognitiveProcessor) -> None:
        """인지 프로세서 등록"""
        self.processors[processor.config.id] = processor
        if not self.default_processor_id:
            self.default_processor_id = processor.config.id

    async def process(self, *args, **kwargs) -> CognitiveResult:
        """인지 작업 실행 (기본 프로세서 지원)"""
        processor_id: Optional[ID] = kwargs.pop('processor_id', None)

        if len(args) == 1 and isinstance(args[0], CognitiveContext):
            context = args[0]
        elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], CognitiveContext):
            processor_id = args[0]
            context = args[1]
        else:
            raise CognitiveError(
                message="Invalid arguments for cognitive processing",
                component="system",
                metadata={'args': [type(arg).__name__ for arg in args]}
            )

        target_processor_id = processor_id or self.default_processor_id
        if target_processor_id is None:
            raise CognitiveError(
                message="No cognitive processors registered",
                component="system"
            )

        if target_processor_id not in self.processors:
            raise CognitiveError(
                message=f"Processor {target_processor_id} not found",
                component="system",
                metadata={'processor_id': target_processor_id}
            )

        processor = self.processors[target_processor_id]
        return await processor.process(context)

    def get_processor(self, processor_id: ID) -> Optional[BaseCognitiveProcessor]:
        """프로세서 조회"""
        return self.processors.get(processor_id)

    def list_processors(self) -> List[str]:
        """프로세서 목록 조회"""
        return list(self.processors.keys())

    def get_system_statistics(self) -> Dict[str, Any]:
        """시스템 전체 통계"""
        total_processes = 0
        total_successes = 0
        total_errors = 0

        processor_stats = {}
        for proc_id, processor in self.processors.items():
            stats = processor.get_statistics()
            processor_stats[proc_id] = stats
            total_processes += stats.process_count
            total_successes += stats.success_count
            total_errors += stats.error_count

        return {
            'total_processes': total_processes,
            'total_successes': total_successes,
            'total_errors': total_errors,
            'success_rate': total_successes / total_processes if total_processes > 0 else 0,
            'processor_count': len(self.processors),
            'processor_statistics': processor_stats
        }


# Helper functions
def create_cognitive_context(
    task_type: CognitiveTaskType,
    input_data: Any,
    metadata: Optional[Dict[str, Any]] = None,
    quality_requirements: Optional[QualityMetrics] = None
) -> CognitiveContext:
    """인지 컨텍스트 생성 헬퍼 함수"""
    return CognitiveContext(
        id=create_id(),
        task_type=task_type,
        input=input_data,
        metadata=metadata or {},
        quality_requirements=quality_requirements,
        timestamp=current_timestamp()
    )


def create_processing_step(
    name: str,
    step_type: str,
    input_data: Any,
    output_data: Any,
    confidence: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None
) -> ProcessingStep:
    """처리 단계 생성 헬퍼 함수"""
    current_time = current_timestamp()
    return ProcessingStep(
        id=create_id(),
        name=name,
        type=step_type,
        start_time=current_time,
        end_time=current_time,
        input=input_data,
        output=output_data,
        confidence=confidence,
        metadata=metadata or {}
    )
