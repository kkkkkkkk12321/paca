"""
Perception Engine

지각 처리의 중앙 엔진으로, 다양한 감각 입력을 통합하고
의미 있는 지각적 표현으로 변환합니다.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Callable, Union
from uuid import uuid4, UUID

from ...base import BaseCognitiveProcessor


class PerceptionState(Enum):
    """지각 상태"""
    IDLE = auto()           # 유휴 상태
    ACTIVE = auto()         # 활성 처리 상태
    FOCUSED = auto()        # 집중 처리 상태
    INTEGRATING = auto()    # 통합 처리 상태
    OVERLOADED = auto()     # 과부하 상태


class ProcessingMode(Enum):
    """처리 모드"""
    BOTTOM_UP = auto()      # 상향식 처리 (데이터 주도)
    TOP_DOWN = auto()       # 하향식 처리 (기대 주도)
    INTERACTIVE = auto()    # 상호작용 처리
    PARALLEL = auto()       # 병렬 처리


@dataclass
class PerceptionConfig:
    """지각 시스템 설정"""
    max_concurrent_inputs: int = 10        # 최대 동시 입력 수
    processing_timeout_ms: int = 5000      # 처리 타임아웃
    integration_window_ms: int = 500       # 통합 시간 윈도우
    default_processing_mode: ProcessingMode = ProcessingMode.INTERACTIVE
    enable_predictive_processing: bool = True
    sensory_buffer_size: int = 100         # 감각 버퍼 크기
    pattern_matching_threshold: float = 0.7
    concept_formation_threshold: float = 0.8


@dataclass
class SensoryInput:
    """감각 입력 데이터"""
    id: UUID = field(default_factory=uuid4)
    modality: str = "general"              # visual, auditory, textual 등
    data: Any = None                       # 실제 입력 데이터
    timestamp: float = field(default_factory=time.time)
    intensity: float = 1.0                 # 입력 강도
    quality: float = 1.0                   # 입력 품질
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"                # 입력 소스
    confidence: float = 1.0                # 입력 신뢰도


@dataclass
class PerceptionResult:
    """지각 처리 결과"""
    input_id: UUID
    perceived_objects: List[Dict[str, Any]] = field(default_factory=list)
    recognized_patterns: List[Dict[str, Any]] = field(default_factory=list)
    formed_concepts: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_ms: float = 0.0
    confidence_score: float = 0.0
    processing_mode: ProcessingMode = ProcessingMode.INTERACTIVE
    attention_weights: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None
    success: bool = True


class PerceptionEngine(BaseCognitiveProcessor):
    """
    지각 처리 엔진

    다양한 감각 입력을 받아 지각적 표현으로 변환하고,
    패턴 인식과 개념 형성을 수행합니다.
    """

    def __init__(self, config: Optional[PerceptionConfig] = None):
        super().__init__()
        self.config = config or PerceptionConfig()

        # 처리 상태
        self._state = PerceptionState.IDLE
        self._current_inputs: Dict[UUID, SensoryInput] = {}
        self._processing_queue: List[SensoryInput] = []

        # 처리 컴포넌트들 (지연 초기화)
        self._pattern_recognizer = None
        self._concept_former = None
        self._sensory_processor = None

        # 통합 및 예측
        self._integration_buffer: List[PerceptionResult] = []
        self._prediction_cache: Dict[str, Any] = {}

        # 성능 메트릭
        self._total_processed = 0
        self._successful_processed = 0
        self._average_processing_time = 0.0

        # 주의 시스템 연동
        self._attention_weights: Dict[str, float] = {}
        self._focused_modalities: Set[str] = set()

    async def initialize(self) -> bool:
        """지각 엔진 초기화"""
        try:
            self.logger.info("Initializing Perception Engine...")

            # 서브 컴포넌트 초기화 (늦은 import로 순환 의존성 방지)
            from .pattern_recognizer import create_pattern_recognizer
            from .concept_former import create_concept_former
            from .sensory_processor import create_sensory_processor

            self._pattern_recognizer = await create_pattern_recognizer()
            self._concept_former = await create_concept_former()
            self._sensory_processor = await create_sensory_processor()

            # 백그라운드 프로세스 시작
            asyncio.create_task(self._process_input_queue())
            asyncio.create_task(self._manage_integration())
            asyncio.create_task(self._update_predictions())

            self._state = PerceptionState.IDLE
            self.logger.info("Perception Engine initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Perception Engine: {e}")
            return False

    async def process_input(self, sensory_input: SensoryInput,
                          processing_mode: Optional[ProcessingMode] = None) -> PerceptionResult:
        """
        감각 입력 처리

        Args:
            sensory_input: 처리할 감각 입력
            processing_mode: 처리 모드 (None이면 기본값 사용)

        Returns:
            지각 처리 결과
        """
        try:
            start_time = time.time()
            mode = processing_mode or self.config.default_processing_mode

            # 입력 전처리
            preprocessed_input = await self._preprocess_input(sensory_input)

            # 용량 관리
            if len(self._current_inputs) >= self.config.max_concurrent_inputs:
                self._processing_queue.append(preprocessed_input)
                return PerceptionResult(
                    input_id=sensory_input.id,
                    success=False,
                    error_message="Processing queue full"
                )

            # 현재 처리 중인 입력에 추가
            self._current_inputs[preprocessed_input.id] = preprocessed_input

            # 처리 모드에 따른 분기
            if mode == ProcessingMode.BOTTOM_UP:
                result = await self._bottom_up_processing(preprocessed_input)
            elif mode == ProcessingMode.TOP_DOWN:
                result = await self._top_down_processing(preprocessed_input)
            elif mode == ProcessingMode.PARALLEL:
                result = await self._parallel_processing(preprocessed_input)
            else:  # INTERACTIVE
                result = await self._interactive_processing(preprocessed_input)

            # 처리 시간 계산
            processing_time = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time
            result.processing_mode = mode

            # 통합 버퍼에 추가
            self._integration_buffer.append(result)

            # 메트릭 업데이트
            self._update_metrics(result)

            # 처리 완료 정리
            if preprocessed_input.id in self._current_inputs:
                del self._current_inputs[preprocessed_input.id]

            self.logger.debug(f"Processed input {sensory_input.id} in {processing_time:.2f}ms")
            return result

        except Exception as e:
            self.logger.error(f"Error processing input {sensory_input.id}: {e}")
            return PerceptionResult(
                input_id=sensory_input.id,
                success=False,
                error_message=str(e)
            )

    async def set_attention_focus(self, modalities: List[str],
                                weights: Optional[Dict[str, float]] = None) -> None:
        """
        주의 집중 설정

        Args:
            modalities: 집중할 감각 양상들
            weights: 각 양상별 가중치
        """
        try:
            self._focused_modalities = set(modalities)

            if weights:
                self._attention_weights.update(weights)
            else:
                # 균등 가중치 설정
                equal_weight = 1.0 / len(modalities) if modalities else 0.0
                for modality in modalities:
                    self._attention_weights[modality] = equal_weight

            self.logger.info(f"Set attention focus to modalities: {modalities}")

        except Exception as e:
            self.logger.error(f"Error setting attention focus: {e}")

    async def get_perception_state(self) -> Dict[str, Any]:
        """현재 지각 상태 조회"""
        return {
            "state": self._state.name,
            "current_inputs": len(self._current_inputs),
            "queued_inputs": len(self._processing_queue),
            "integration_buffer_size": len(self._integration_buffer),
            "focused_modalities": list(self._focused_modalities),
            "attention_weights": self._attention_weights.copy(),
            "total_processed": self._total_processed,
            "success_rate": (
                self._successful_processed / max(1, self._total_processed)
            ),
            "average_processing_time_ms": self._average_processing_time
        }

    async def predict_next_input(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """다음 입력 예측"""
        try:
            if not self.config.enable_predictive_processing:
                return {"prediction": None, "confidence": 0.0}

            # 간단한 예측 모델 (실제로는 더 복잡한 모델 사용)
            recent_patterns = self._analyze_recent_patterns()
            prediction = await self._generate_prediction(recent_patterns, context)

            return {
                "prediction": prediction,
                "confidence": prediction.get("confidence", 0.0),
                "based_on_patterns": len(recent_patterns)
            }

        except Exception as e:
            self.logger.error(f"Error predicting next input: {e}")
            return {"prediction": None, "confidence": 0.0, "error": str(e)}

    async def _preprocess_input(self, sensory_input: SensoryInput) -> SensoryInput:
        """입력 전처리"""
        try:
            # 감각 양상별 전처리
            if self._sensory_processor:
                processed_data = await self._sensory_processor.preprocess(
                    sensory_input.data, sensory_input.modality
                )
                sensory_input.data = processed_data

            # 주의 가중치 적용
            if sensory_input.modality in self._attention_weights:
                weight = self._attention_weights[sensory_input.modality]
                sensory_input.intensity *= weight

            return sensory_input

        except Exception as e:
            self.logger.error(f"Error preprocessing input: {e}")
            return sensory_input

    async def _bottom_up_processing(self, sensory_input: SensoryInput) -> PerceptionResult:
        """상향식 처리 (데이터 주도)"""
        try:
            result = PerceptionResult(input_id=sensory_input.id)

            # 1. 기본 특징 추출
            if self._sensory_processor:
                features = await self._sensory_processor.extract_features(
                    sensory_input.data, sensory_input.modality
                )
                result.perceived_objects.append({
                    "type": "features",
                    "data": features,
                    "confidence": sensory_input.confidence
                })

            # 2. 패턴 인식
            if self._pattern_recognizer:
                patterns = await self._pattern_recognizer.recognize(
                    sensory_input.data, sensory_input.modality
                )
                result.recognized_patterns.extend(patterns)

            # 3. 개념 형성 (패턴 기반)
            if self._concept_former and patterns:
                concepts = await self._concept_former.form_concepts(patterns)
                result.formed_concepts.extend(concepts)

            result.confidence_score = self._calculate_confidence(result)
            return result

        except Exception as e:
            self.logger.error(f"Error in bottom-up processing: {e}")
            return PerceptionResult(
                input_id=sensory_input.id,
                success=False,
                error_message=str(e)
            )

    async def _top_down_processing(self, sensory_input: SensoryInput) -> PerceptionResult:
        """하향식 처리 (기대 주도)"""
        try:
            result = PerceptionResult(input_id=sensory_input.id)

            # 1. 예측된 패턴 확인
            predicted_patterns = await self._get_predicted_patterns(sensory_input)

            # 2. 예측 기반 처리
            if predicted_patterns:
                verified_patterns = await self._verify_predictions(
                    sensory_input, predicted_patterns
                )
                result.recognized_patterns.extend(verified_patterns)

            # 3. 예측 실패 시 상향식 처리로 폴백
            if not result.recognized_patterns:
                fallback_result = await self._bottom_up_processing(sensory_input)
                result.recognized_patterns = fallback_result.recognized_patterns
                result.perceived_objects = fallback_result.perceived_objects

            # 4. 개념 업데이트
            if result.recognized_patterns and self._concept_former:
                concepts = await self._concept_former.update_concepts(
                    result.recognized_patterns
                )
                result.formed_concepts.extend(concepts)

            result.confidence_score = self._calculate_confidence(result)
            return result

        except Exception as e:
            self.logger.error(f"Error in top-down processing: {e}")
            return PerceptionResult(
                input_id=sensory_input.id,
                success=False,
                error_message=str(e)
            )

    async def _interactive_processing(self, sensory_input: SensoryInput) -> PerceptionResult:
        """상호작용 처리 (상향식 + 하향식)"""
        try:
            # 상향식과 하향식 처리를 병렬로 수행
            bottom_up_task = asyncio.create_task(
                self._bottom_up_processing(sensory_input)
            )
            top_down_task = asyncio.create_task(
                self._top_down_processing(sensory_input)
            )

            bottom_up_result, top_down_result = await asyncio.gather(
                bottom_up_task, top_down_task
            )

            # 결과 통합
            integrated_result = await self._integrate_processing_results(
                sensory_input.id, bottom_up_result, top_down_result
            )

            return integrated_result

        except Exception as e:
            self.logger.error(f"Error in interactive processing: {e}")
            return PerceptionResult(
                input_id=sensory_input.id,
                success=False,
                error_message=str(e)
            )

    async def _parallel_processing(self, sensory_input: SensoryInput) -> PerceptionResult:
        """병렬 처리"""
        try:
            result = PerceptionResult(input_id=sensory_input.id)

            # 여러 처리 경로를 병렬로 실행
            tasks = []

            if self._sensory_processor:
                tasks.append(
                    self._sensory_processor.process_parallel(
                        sensory_input.data, sensory_input.modality
                    )
                )

            if self._pattern_recognizer:
                tasks.append(
                    self._pattern_recognizer.recognize_parallel(
                        sensory_input.data, sensory_input.modality
                    )
                )

            # 병렬 실행 및 결과 수집
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for task_result in results:
                    if isinstance(task_result, Exception):
                        self.logger.warning(f"Parallel task failed: {task_result}")
                        continue

                    # 결과 통합
                    if isinstance(task_result, dict):
                        if "patterns" in task_result:
                            result.recognized_patterns.extend(task_result["patterns"])
                        if "objects" in task_result:
                            result.perceived_objects.extend(task_result["objects"])

            result.confidence_score = self._calculate_confidence(result)
            return result

        except Exception as e:
            self.logger.error(f"Error in parallel processing: {e}")
            return PerceptionResult(
                input_id=sensory_input.id,
                success=False,
                error_message=str(e)
            )

    async def _integrate_processing_results(self, input_id: UUID,
                                          bottom_up: PerceptionResult,
                                          top_down: PerceptionResult) -> PerceptionResult:
        """처리 결과 통합"""
        try:
            integrated = PerceptionResult(input_id=input_id)

            # 패턴 통합 (중복 제거 및 신뢰도 기반 선택)
            all_patterns = bottom_up.recognized_patterns + top_down.recognized_patterns
            integrated.recognized_patterns = await self._merge_patterns(all_patterns)

            # 객체 통합
            all_objects = bottom_up.perceived_objects + top_down.perceived_objects
            integrated.perceived_objects = await self._merge_objects(all_objects)

            # 개념 통합
            all_concepts = bottom_up.formed_concepts + top_down.formed_concepts
            integrated.formed_concepts = await self._merge_concepts(all_concepts)

            # 신뢰도 계산 (두 결과의 일치도 고려)
            consistency_score = await self._calculate_consistency(bottom_up, top_down)
            integrated.confidence_score = (
                bottom_up.confidence_score + top_down.confidence_score
            ) / 2 * consistency_score

            return integrated

        except Exception as e:
            self.logger.error(f"Error integrating results: {e}")
            return bottom_up  # 폴백으로 상향식 결과 반환

    def _calculate_confidence(self, result: PerceptionResult) -> float:
        """신뢰도 계산"""
        try:
            scores = []

            # 패턴 인식 신뢰도
            if result.recognized_patterns:
                pattern_scores = [
                    p.get("confidence", 0.5) for p in result.recognized_patterns
                ]
                scores.append(sum(pattern_scores) / len(pattern_scores))

            # 객체 인식 신뢰도
            if result.perceived_objects:
                object_scores = [
                    obj.get("confidence", 0.5) for obj in result.perceived_objects
                ]
                scores.append(sum(object_scores) / len(object_scores))

            # 개념 형성 신뢰도
            if result.formed_concepts:
                concept_scores = [
                    c.get("confidence", 0.5) for c in result.formed_concepts
                ]
                scores.append(sum(concept_scores) / len(concept_scores))

            return sum(scores) / len(scores) if scores else 0.5

        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5

    async def _merge_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """패턴 병합 (중복 제거)"""
        merged = {}

        for pattern in patterns:
            pattern_id = pattern.get("id") or pattern.get("name", "unknown")

            if pattern_id in merged:
                # 신뢰도가 높은 것으로 대체
                if pattern.get("confidence", 0) > merged[pattern_id].get("confidence", 0):
                    merged[pattern_id] = pattern
            else:
                merged[pattern_id] = pattern

        return list(merged.values())

    async def _merge_objects(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """객체 병합"""
        # 간단한 구현: 중복 제거 없이 모든 객체 포함
        return objects

    async def _merge_concepts(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """개념 병합"""
        merged = {}

        for concept in concepts:
            concept_id = concept.get("id") or concept.get("name", "unknown")

            if concept_id in merged:
                # 신뢰도 기반 병합
                existing = merged[concept_id]
                new_confidence = max(
                    existing.get("confidence", 0),
                    concept.get("confidence", 0)
                )
                existing["confidence"] = new_confidence
            else:
                merged[concept_id] = concept

        return list(merged.values())

    async def _calculate_consistency(self, result1: PerceptionResult,
                                   result2: PerceptionResult) -> float:
        """두 결과 간 일치도 계산"""
        try:
            consistency_scores = []

            # 패턴 일치도
            if result1.recognized_patterns and result2.recognized_patterns:
                pattern_overlap = len(set(
                    p.get("name", "") for p in result1.recognized_patterns
                ) & set(
                    p.get("name", "") for p in result2.recognized_patterns
                ))
                total_patterns = len(result1.recognized_patterns) + len(result2.recognized_patterns)
                if total_patterns > 0:
                    consistency_scores.append(2 * pattern_overlap / total_patterns)

            # 기본 일치도
            if not consistency_scores:
                consistency_scores.append(0.8)  # 기본값

            return sum(consistency_scores) / len(consistency_scores)

        except Exception as e:
            self.logger.error(f"Error calculating consistency: {e}")
            return 0.8

    def _update_metrics(self, result: PerceptionResult) -> None:
        """메트릭 업데이트"""
        self._total_processed += 1

        if result.success:
            self._successful_processed += 1

        # 평균 처리 시간 업데이트
        if result.processing_time_ms > 0:
            total_time = (
                self._average_processing_time * (self._total_processed - 1) +
                result.processing_time_ms
            )
            self._average_processing_time = total_time / self._total_processed

    def _analyze_recent_patterns(self) -> List[Dict[str, Any]]:
        """최근 패턴 분석"""
        recent_results = self._integration_buffer[-10:]  # 최근 10개
        patterns = []

        for result in recent_results:
            patterns.extend(result.recognized_patterns)

        return patterns

    async def _generate_prediction(self, patterns: List[Dict[str, Any]],
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """예측 생성"""
        try:
            # 간단한 예측 모델
            if not patterns:
                return {"confidence": 0.0}

            # 패턴 빈도 분석
            pattern_counts = {}
            for pattern in patterns:
                name = pattern.get("name", "unknown")
                pattern_counts[name] = pattern_counts.get(name, 0) + 1

            # 가장 빈번한 패턴 예측
            if pattern_counts:
                most_frequent = max(pattern_counts, key=pattern_counts.get)
                confidence = pattern_counts[most_frequent] / len(patterns)

                return {
                    "predicted_pattern": most_frequent,
                    "confidence": min(confidence, 1.0),
                    "based_on_frequency": pattern_counts[most_frequent]
                }

            return {"confidence": 0.0}

        except Exception as e:
            self.logger.error(f"Error generating prediction: {e}")
            return {"confidence": 0.0}

    async def _get_predicted_patterns(self, sensory_input: SensoryInput) -> List[Dict[str, Any]]:
        """예측된 패턴 조회"""
        cache_key = f"{sensory_input.modality}_{int(time.time() / 10)}"  # 10초 간격
        return self._prediction_cache.get(cache_key, [])

    async def _verify_predictions(self, sensory_input: SensoryInput,
                                predicted_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """예측 검증"""
        verified = []

        for pattern in predicted_patterns:
            # 간단한 검증 로직
            if self._pattern_recognizer:
                verification_result = await self._pattern_recognizer.verify_pattern(
                    sensory_input.data, pattern
                )
                if verification_result.get("verified", False):
                    verified.append(pattern)

        return verified

    async def _process_input_queue(self) -> None:
        """입력 큐 처리 백그라운드 프로세스"""
        while True:
            try:
                if (self._processing_queue and
                    len(self._current_inputs) < self.config.max_concurrent_inputs):

                    next_input = self._processing_queue.pop(0)
                    asyncio.create_task(self.process_input(next_input))

                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error processing input queue: {e}")
                await asyncio.sleep(1.0)

    async def _manage_integration(self) -> None:
        """통합 관리 백그라운드 프로세스"""
        while True:
            try:
                # 통합 버퍼 크기 관리
                if len(self._integration_buffer) > self.config.sensory_buffer_size:
                    # 오래된 결과 제거
                    self._integration_buffer = self._integration_buffer[-50:]

                await asyncio.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Error in integration management: {e}")
                await asyncio.sleep(5.0)

    async def _update_predictions(self) -> None:
        """예측 업데이트 백그라운드 프로세스"""
        while True:
            try:
                if self.config.enable_predictive_processing:
                    # 예측 캐시 업데이트
                    recent_patterns = self._analyze_recent_patterns()
                    current_time_key = f"general_{int(time.time() / 10)}"

                    if recent_patterns:
                        self._prediction_cache[current_time_key] = recent_patterns[-5:]

                # 오래된 예측 정리
                current_time = int(time.time() / 10)
                old_keys = [
                    key for key in self._prediction_cache.keys()
                    if int(key.split('_')[-1]) < current_time - 10
                ]
                for key in old_keys:
                    del self._prediction_cache[key]

                await asyncio.sleep(5.0)

            except Exception as e:
                self.logger.error(f"Error updating predictions: {e}")
                await asyncio.sleep(30.0)


async def create_perception_engine(config: Optional[PerceptionConfig] = None) -> PerceptionEngine:
    """PerceptionEngine 인스턴스 생성 및 초기화"""
    engine = PerceptionEngine(config)
    await engine.initialize()
    return engine