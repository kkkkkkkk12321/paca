"""
Learning Module
기계학습 및 적응적 학습 시스템
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import time
import asyncio
import numpy as np
from datetime import datetime

from paca.cognitive.base import (
    BaseCognitiveProcessor,
    CognitiveContext,
    CognitiveResult,
    CognitiveConfig,
    LearningData,
    MemoryUpdate,
    ProcessingStep,
    CognitiveTaskType
)
from paca.core.types import ID, Timestamp, KeyValuePair
from paca.core.errors import CognitiveModelError
from paca.core.events import EventEmitter


class LearningMethod(Enum):
    """학습 방법론"""
    SUPERVISED = "supervised"           # 지도 학습
    UNSUPERVISED = "unsupervised"      # 비지도 학습
    REINFORCEMENT = "reinforcement"     # 강화 학습
    TRANSFER = "transfer"               # 전이 학습
    META = "meta"                       # 메타 학습
    INCREMENTAL = "incremental"         # 점진적 학습
    ACTIVE = "active"                   # 능동 학습


class LearningState(Enum):
    """학습 상태"""
    TRAINING = "training"
    VALIDATING = "validating"
    TESTING = "testing"
    ADAPTING = "adapting"
    CONVERGED = "converged"
    OVERFITTING = "overfitting"


@dataclass
class LearningConfig(CognitiveConfig):
    """학습 설정"""
    method: LearningMethod = LearningMethod.SUPERVISED
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    convergence_threshold: float = 0.001
    regularization: float = 0.01
    enable_early_stopping: bool = True
    enable_validation: bool = True
    validation_split: float = 0.2


@dataclass
class TrainingExample:
    """훈련 예제"""
    id: ID
    input: Any
    output: Optional[Any] = None
    label: Optional[Any] = None
    weight: float = 1.0
    metadata: KeyValuePair = field(default_factory=dict)
    timestamp: Timestamp = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class LearningContext(CognitiveContext):
    """학습 컨텍스트"""
    method: LearningMethod = LearningMethod.SUPERVISED
    training_data: List[TrainingExample] = field(default_factory=list)
    validation_data: Optional[List[TrainingExample]] = None
    test_data: Optional[List[TrainingExample]] = None
    hyperparameters: Optional[KeyValuePair] = None


@dataclass
class LearnedModel:
    """학습된 모델"""
    id: ID
    name: str
    method: LearningMethod
    parameters: KeyValuePair = field(default_factory=dict)
    weights: List[float] = field(default_factory=list)
    architecture: Optional[Any] = None
    training_time: float = 0.0
    created_at: Timestamp = field(default_factory=lambda: int(time.time() * 1000))
    version: str = "1.0.0"


@dataclass
class PerformanceMetrics:
    """성능 지표"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    loss: float = 0.0
    mae: Optional[float] = None
    mse: Optional[float] = None
    custom_metrics: KeyValuePair = field(default_factory=dict)


@dataclass
class TrainingEpoch:
    """훈련 에포크"""
    epoch: int
    training_loss: float
    validation_loss: Optional[float] = None
    metrics: Optional[PerformanceMetrics] = None
    learning_rate: float = 0.001
    duration: float = 0.0
    timestamp: Timestamp = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class ValidationResult:
    """검증 결과"""
    epoch: int
    metrics: PerformanceMetrics
    improved: bool = False
    best_so_far: bool = False


@dataclass
class LearningResult(CognitiveResult):
    """학습 결과"""
    method: LearningMethod = LearningMethod.SUPERVISED
    model: Optional[LearnedModel] = None
    performance: Optional[PerformanceMetrics] = None
    training_history: List[TrainingEpoch] = field(default_factory=list)
    validation_history: Optional[List[ValidationResult]] = None
    best_model: Optional[LearnedModel] = None
    convergence_reached: bool = False


class AdaptiveLearningProcessor(BaseCognitiveProcessor):
    """적응적 학습 프로세서"""

    def __init__(self, config: LearningConfig, events: Optional[EventEmitter] = None):
        super().__init__(config, events)

        # 기본 설정 병합
        self.learning_config = LearningConfig(
            learning_rate=0.001,
            batch_size=32,
            max_epochs=100,
            convergence_threshold=0.001,
            regularization=0.01,
            enable_early_stopping=True,
            enable_validation=True,
            validation_split=0.2,
            **config.__dict__
        )

        self.models: Dict[ID, LearnedModel] = {}
        self.training_history: Dict[ID, List[TrainingEpoch]] = {}

        self.current_model: Optional[LearnedModel] = None
        self.learning_state: LearningState = LearningState.TRAINING
        self.current_epoch: int = 0
        self.best_performance: float = 0.0
        self.no_improvement_count: int = 0

    async def perform_cognitive_processing(
        self,
        context: CognitiveContext,
        relevant_memory: Any
    ) -> CognitiveResult:
        """학습 처리 실행"""
        start_time = time.time() * 1000
        learning_context = context if isinstance(context, LearningContext) else LearningContext(**context.__dict__)

        # 학습 방법에 따른 처리
        method_handlers = {
            LearningMethod.SUPERVISED: self._perform_supervised_learning,
            LearningMethod.UNSUPERVISED: self._perform_unsupervised_learning,
            LearningMethod.REINFORCEMENT: self._perform_reinforcement_learning,
            LearningMethod.TRANSFER: self._perform_transfer_learning,
            LearningMethod.META: self._perform_meta_learning,
            LearningMethod.INCREMENTAL: self._perform_incremental_learning,
            LearningMethod.ACTIVE: self._perform_active_learning
        }

        handler = method_handlers.get(learning_context.method)
        if not handler:
            raise CognitiveModelError(
                component='CognitiveLearningProcessor',
                operation='performLearning',
                context=learning_context,
                expected='valid learning result',
                details=f'Unsupported learning method: {learning_context.method}'
            )

        return await handler(learning_context, relevant_memory, start_time)

    async def _perform_supervised_learning(
        self,
        context: LearningContext,
        relevant_memory: Any,
        start_time: float
    ) -> LearningResult:
        """지도 학습"""
        processing_steps: List[ProcessingStep] = []

        # 1. 데이터 전처리
        preprocessing_step = await self._create_processing_step('data_preprocessing', {
            'input': context.training_data,
            'output': await self._preprocess_data(context.training_data),
            'confidence': 0.9
        })
        processing_steps.append(preprocessing_step)

        # 2. 모델 초기화
        initial_model = await self._initialize_model(context.method, context.hyperparameters)
        initialization_step = await self._create_processing_step('model_initialization', {
            'input': context.hyperparameters,
            'output': initial_model,
            'confidence': 0.8
        })
        processing_steps.append(initialization_step)

        # 3. 훈련 실행
        training_result = await self._train_model(
            initial_model,
            context.training_data,
            context.validation_data
        )
        training_step = await self._create_processing_step('model_training', {
            'input': {'model': initial_model, 'data': context.training_data},
            'output': training_result,
            'confidence': 0.9 if training_result.get('convergence_reached', False) else 0.7
        })
        processing_steps.append(training_step)

        # 4. 모델 평가
        performance = await self._evaluate_model(
            training_result['model'],
            context.validation_data or context.training_data
        )
        evaluation_step = await self._create_processing_step('model_evaluation', {
            'input': training_result['model'],
            'output': performance,
            'confidence': performance.accuracy
        })
        processing_steps.append(evaluation_step)

        # 5. 모델 저장
        model = training_result['model']
        self.models[model.id] = model
        self.current_model = model

        return LearningResult(
            context_id=context.id,
            output=model,
            confidence=performance.accuracy,
            quality_metrics=self._calculate_quality_metrics(context.input, model, processing_steps),
            processing_steps=processing_steps,
            memory_updates=self._generate_memory_updates(context, model, performance),
            learning_data=self._generate_learning_data(context, model),
            processing_time_ms=int(time.time() * 1000 - start_time),
            resource_usage=self._calculate_resource_usage(start_time),
            method=context.method,
            model=model,
            performance=performance,
            training_history=training_result.get('history', []),
            validation_history=training_result.get('validation_history'),
            best_model=training_result.get('best_model'),
            convergence_reached=training_result.get('convergence_reached', True)
        )

    async def _perform_unsupervised_learning(
        self,
        context: LearningContext,
        relevant_memory: Any,
        start_time: float
    ) -> LearningResult:
        """비지도 학습"""
        processing_steps: List[ProcessingStep] = []

        # 1. 패턴 발견
        patterns = await self._discover_patterns(context.training_data)
        pattern_step = await self._create_processing_step('pattern_discovery', {
            'input': context.training_data,
            'output': patterns,
            'confidence': 0.8
        })
        processing_steps.append(pattern_step)

        # 2. 클러스터링/차원 축소 등
        clusters = await self._perform_clustering(context.training_data, patterns)
        clustering_step = await self._create_processing_step('clustering', {
            'input': {'data': context.training_data, 'patterns': patterns},
            'output': clusters,
            'confidence': 0.75
        })
        processing_steps.append(clustering_step)

        # 3. 모델 생성
        model = await self._create_unsupervised_model(patterns, clusters, context.method)

        return self._create_learning_result(context, model, processing_steps, start_time)

    async def _perform_reinforcement_learning(
        self,
        context: LearningContext,
        relevant_memory: Any,
        start_time: float
    ) -> LearningResult:
        """강화 학습"""
        processing_steps: List[ProcessingStep] = []

        # 1. 환경 설정
        environment = await self._setup_environment(context)
        env_step = await self._create_processing_step('environment_setup', {
            'input': context,
            'output': environment,
            'confidence': 0.9
        })
        processing_steps.append(env_step)

        # 2. 에이전트 초기화
        agent = await self._initialize_agent(context.hyperparameters)
        agent_step = await self._create_processing_step('agent_initialization', {
            'input': context.hyperparameters,
            'output': agent,
            'confidence': 0.8
        })
        processing_steps.append(agent_step)

        # 3. 에피소드 실행
        episodes = await self._run_episodes(agent, environment, context.training_data)
        episode_step = await self._create_processing_step('episode_execution', {
            'input': {'agent': agent, 'environment': environment},
            'output': episodes,
            'confidence': 0.7
        })
        processing_steps.append(episode_step)

        # 4. 정책 업데이트
        updated_agent = await self._update_policy(agent, episodes)

        model = await self._extract_model_from_agent(updated_agent)
        return self._create_learning_result(context, model, processing_steps, start_time)

    async def _perform_transfer_learning(
        self,
        context: LearningContext,
        relevant_memory: Any,
        start_time: float
    ) -> LearningResult:
        """전이 학습"""
        processing_steps: List[ProcessingStep] = []

        # 1. 사전 훈련된 모델 로드
        pretrained_model = await self._load_pretrained_model(relevant_memory)
        load_step = await self._create_processing_step('pretrained_model_loading', {
            'input': relevant_memory,
            'output': pretrained_model,
            'confidence': 0.9
        })
        processing_steps.append(load_step)

        # 2. 모델 적응
        adapted_model = await self._adapt_model(pretrained_model, context.training_data)
        adapt_step = await self._create_processing_step('model_adaptation', {
            'input': {'model': pretrained_model, 'data': context.training_data},
            'output': adapted_model,
            'confidence': 0.85
        })
        processing_steps.append(adapt_step)

        # 3. 미세 조정
        fine_tuned_model = await self._fine_tune_model(adapted_model, context.training_data)

        return self._create_learning_result(context, fine_tuned_model, processing_steps, start_time)

    async def _perform_meta_learning(
        self,
        context: LearningContext,
        relevant_memory: Any,
        start_time: float
    ) -> LearningResult:
        """메타 학습"""
        processing_steps: List[ProcessingStep] = []

        # 1. 태스크 분석
        tasks = await self._analyze_tasks(context.training_data)
        task_step = await self._create_processing_step('task_analysis', {
            'input': context.training_data,
            'output': tasks,
            'confidence': 0.8
        })
        processing_steps.append(task_step)

        # 2. 메타 모델 학습
        meta_model = await self._learn_meta_model(tasks, relevant_memory)
        meta_step = await self._create_processing_step('meta_model_learning', {
            'input': tasks,
            'output': meta_model,
            'confidence': 0.75
        })
        processing_steps.append(meta_step)

        return self._create_learning_result(context, meta_model, processing_steps, start_time)

    async def _perform_incremental_learning(
        self,
        context: LearningContext,
        relevant_memory: Any,
        start_time: float
    ) -> LearningResult:
        """점진적 학습"""
        processing_steps: List[ProcessingStep] = []

        # 1. 기존 모델 로드
        current_model = self.current_model or await self._load_existing_model(relevant_memory)

        # 2. 점진적 업데이트
        updated_model = await self._update_model_incrementally(current_model, context.training_data)
        update_step = await self._create_processing_step('incremental_update', {
            'input': {'model': current_model, 'data': context.training_data},
            'output': updated_model,
            'confidence': 0.8
        })
        processing_steps.append(update_step)

        return self._create_learning_result(context, updated_model, processing_steps, start_time)

    async def _perform_active_learning(
        self,
        context: LearningContext,
        relevant_memory: Any,
        start_time: float
    ) -> LearningResult:
        """능동 학습"""
        processing_steps: List[ProcessingStep] = []

        # 1. 불확실성 샘플링
        uncertain_samples = await self._select_uncertain_samples(context.training_data)
        sampling_step = await self._create_processing_step('uncertainty_sampling', {
            'input': context.training_data,
            'output': uncertain_samples,
            'confidence': 0.7
        })
        processing_steps.append(sampling_step)

        # 2. 능동적 쿼리
        labeled_samples = await self._query_labels(uncertain_samples)
        query_step = await self._create_processing_step('active_querying', {
            'input': uncertain_samples,
            'output': labeled_samples,
            'confidence': 0.8
        })
        processing_steps.append(query_step)

        # 3. 모델 재훈련
        retrained_model = await self._retrain_with_new_samples(
            self.current_model,
            labeled_samples
        )

        return self._create_learning_result(context, retrained_model, processing_steps, start_time)

    # ==========================================
    # 헬퍼 메소드들
    # ==========================================

    async def _create_processing_step(self, name: str, data: Dict[str, Any]) -> ProcessingStep:
        """처리 단계 생성"""
        current_time = int(time.time() * 1000)
        return ProcessingStep(
            id=f"step_{current_time}_{np.random.randint(100000, 999999)}",
            name=name,
            type='learning',
            start_time=current_time,
            end_time=current_time,
            input=data['input'],
            output=data['output'],
            confidence=data['confidence'],
            metadata=data.get('metadata', {})
        )

    def _create_learning_result(
        self,
        context: LearningContext,
        model: LearnedModel,
        steps: List[ProcessingStep],
        start_time: float
    ) -> LearningResult:
        """학습 결과 생성"""
        performance = PerformanceMetrics(
            accuracy=0.85,
            precision=0.8,
            recall=0.82,
            f1_score=0.81,
            loss=0.15,
            custom_metrics={}
        )

        return LearningResult(
            context_id=context.id,
            output=model,
            confidence=performance.accuracy,
            quality_metrics=self._calculate_quality_metrics(context.input, model, steps),
            processing_steps=steps,
            memory_updates=self._generate_memory_updates(context, model, performance),
            learning_data=self._generate_learning_data(context, model),
            processing_time_ms=int(time.time() * 1000 - start_time),
            resource_usage=self._calculate_resource_usage(start_time),
            method=context.method,
            model=model,
            performance=performance,
            training_history=[],
            convergence_reached=True
        )

    def _generate_memory_updates(
        self,
        context: LearningContext,
        model: LearnedModel,
        performance: PerformanceMetrics
    ) -> List[MemoryUpdate]:
        """메모리 업데이트 생성"""
        return [
            MemoryUpdate(
                id=f"update_{int(time.time() * 1000)}",
                type='store',
                target='learned_model',
                data={'model': model, 'performance': performance},
                timestamp=int(time.time() * 1000),
                confidence=performance.accuracy
            )
        ]

    def _generate_learning_data(
        self,
        context: LearningContext,
        model: LearnedModel
    ) -> LearningData:
        """학습 데이터 생성"""
        return LearningData(
            id=f"learning_{int(time.time() * 1000)}",
            type='supervised' if context.method == LearningMethod.SUPERVISED else 'unsupervised',
            input=context.training_data,
            actual_output=model,
            importance=0.8,
            timestamp=int(time.time() * 1000)
        )

    # ==========================================
    # 간단한 구현의 메소드들 (실제 구현에서는 더 복잡)
    # ==========================================

    async def _preprocess_data(self, data: List[TrainingExample]) -> Any:
        """데이터 전처리"""
        return data

    async def _initialize_model(
        self,
        method: LearningMethod,
        params: Optional[KeyValuePair] = None
    ) -> LearnedModel:
        """모델 초기화"""
        return LearnedModel(
            id=f"model_{int(time.time() * 1000)}",
            name=f"{method.value}_model",
            method=method,
            parameters=params or {},
            weights=[],
            training_time=0.0,
            created_at=int(time.time() * 1000),
            version='1.0.0'
        )

    async def _train_model(
        self,
        model: LearnedModel,
        training: List[TrainingExample],
        validation: Optional[List[TrainingExample]] = None
    ) -> Dict[str, Any]:
        """모델 훈련"""
        return {
            'model': model,
            'history': [],
            'convergence_reached': True
        }

    async def _evaluate_model(
        self,
        model: LearnedModel,
        data: List[TrainingExample]
    ) -> PerformanceMetrics:
        """모델 평가"""
        return PerformanceMetrics(
            accuracy=0.85,
            precision=0.8,
            recall=0.82,
            f1_score=0.81,
            loss=0.15,
            custom_metrics={}
        )

    async def _discover_patterns(self, data: List[TrainingExample]) -> Any:
        """패턴 발견"""
        return {}

    async def _perform_clustering(
        self,
        data: List[TrainingExample],
        patterns: Any
    ) -> Any:
        """클러스터링"""
        return {}

    async def _create_unsupervised_model(
        self,
        patterns: Any,
        clusters: Any,
        method: LearningMethod
    ) -> LearnedModel:
        """비지도 모델 생성"""
        return await self._initialize_model(method)

    async def _setup_environment(self, context: LearningContext) -> Any:
        """환경 설정"""
        return {}

    async def _initialize_agent(self, params: Optional[KeyValuePair] = None) -> Any:
        """에이전트 초기화"""
        return {}

    async def _run_episodes(
        self,
        agent: Any,
        env: Any,
        data: List[TrainingExample]
    ) -> Any:
        """에피소드 실행"""
        return []

    async def _update_policy(self, agent: Any, episodes: Any) -> Any:
        """정책 업데이트"""
        return agent

    async def _extract_model_from_agent(self, agent: Any) -> LearnedModel:
        """에이전트에서 모델 추출"""
        return await self._initialize_model(LearningMethod.REINFORCEMENT)

    async def _load_pretrained_model(self, memory: Any) -> LearnedModel:
        """사전 훈련된 모델 로드"""
        return await self._initialize_model(LearningMethod.TRANSFER)

    async def _adapt_model(
        self,
        model: LearnedModel,
        data: List[TrainingExample]
    ) -> LearnedModel:
        """모델 적응"""
        return model

    async def _fine_tune_model(
        self,
        model: LearnedModel,
        data: List[TrainingExample]
    ) -> LearnedModel:
        """모델 미세 조정"""
        return model

    async def _analyze_tasks(self, data: List[TrainingExample]) -> Any:
        """태스크 분석"""
        return []

    async def _learn_meta_model(self, tasks: Any, memory: Any) -> LearnedModel:
        """메타 모델 학습"""
        return await self._initialize_model(LearningMethod.META)

    async def _load_existing_model(self, memory: Any) -> LearnedModel:
        """기존 모델 로드"""
        return await self._initialize_model(LearningMethod.INCREMENTAL)

    async def _update_model_incrementally(
        self,
        model: Optional[LearnedModel],
        data: List[TrainingExample]
    ) -> LearnedModel:
        """점진적 모델 업데이트"""
        if model is None:
            return await self._initialize_model(LearningMethod.INCREMENTAL)
        return model

    async def _select_uncertain_samples(
        self,
        data: List[TrainingExample]
    ) -> List[TrainingExample]:
        """불확실성 샘플 선택"""
        return data[:10]  # 상위 10개 샘플

    async def _query_labels(self, samples: List[TrainingExample]) -> List[TrainingExample]:
        """라벨 쿼리"""
        return samples

    async def _retrain_with_new_samples(
        self,
        model: Optional[LearnedModel],
        samples: List[TrainingExample]
    ) -> LearnedModel:
        """새 샘플로 재훈련"""
        if model is None:
            return await self._initialize_model(LearningMethod.ACTIVE)
        return model


# ==========================================
# 내보내기
# ==========================================

__all__ = [
    # Enums
    'LearningMethod', 'LearningState',

    # Data Classes
    'LearningConfig', 'TrainingExample', 'LearningContext',
    'LearnedModel', 'PerformanceMetrics', 'TrainingEpoch',
    'ValidationResult', 'LearningResult',

    # Main Classes
    'AdaptiveLearningProcessor'
]