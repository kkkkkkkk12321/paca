"""
Selective Attention

선택적 주의 시스템으로, 여러 자극 중에서 중요한 것들을 선별하고
불필요한 정보를 필터링합니다.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Callable, Union
from uuid import uuid4, UUID

from ...base import BaseCognitiveProcessor


class AttentionPriority(Enum):
    """주의 우선순위"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    IGNORE = 5


class FilterType(Enum):
    """필터 유형"""
    FEATURE_BASED = auto()      # 특징 기반 필터
    SPATIAL_BASED = auto()      # 공간 기반 필터
    TEMPORAL_BASED = auto()     # 시간 기반 필터
    SEMANTIC_BASED = auto()     # 의미 기반 필터
    PRIORITY_BASED = auto()     # 우선순위 기반 필터


@dataclass
class SelectionCriteria:
    """선택 기준 정의"""
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    filter_type: FilterType = FilterType.FEATURE_BASED
    weight: float = 1.0                     # 기준 가중치
    threshold: float = 0.5                  # 선택 임계값
    conditions: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: float = field(default_factory=time.time)


@dataclass
class AttentionFilter:
    """주의 필터 정의"""
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    criteria: List[SelectionCriteria] = field(default_factory=list)
    combination_rule: str = "AND"          # AND, OR, WEIGHTED
    adaptive: bool = True                   # 적응적 필터링 여부
    learning_rate: float = 0.1             # 학습률
    performance_history: List[float] = field(default_factory=list)


@dataclass
class SelectionResult:
    """선택 결과"""
    stimulus_id: UUID
    selected: bool
    confidence_score: float = 0.0          # 선택 신뢰도
    priority: AttentionPriority = AttentionPriority.MEDIUM
    reasoning: List[str] = field(default_factory=list)
    filter_scores: Dict[UUID, float] = field(default_factory=dict)
    processing_time_ms: float = 0.0


@dataclass
class StimulusInfo:
    """자극 정보"""
    id: UUID = field(default_factory=uuid4)
    type: str = "general"
    features: Dict[str, Any] = field(default_factory=dict)
    spatial_location: Optional[Dict[str, float]] = None
    temporal_info: Dict[str, float] = field(default_factory=dict)
    semantic_tags: List[str] = field(default_factory=list)
    base_priority: AttentionPriority = AttentionPriority.MEDIUM
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class SelectiveAttention(BaseCognitiveProcessor):
    """
    선택적 주의 시스템

    다양한 자극 중에서 중요한 것들을 선별하고,
    주의를 집중할 대상을 결정합니다.
    """

    def __init__(self, max_concurrent_stimuli: int = 10):
        super().__init__()
        self.max_concurrent_stimuli = max_concurrent_stimuli

        # 필터 관리
        self._active_filters: Dict[UUID, AttentionFilter] = {}
        self._default_criteria: List[SelectionCriteria] = []

        # 자극 처리
        self._current_stimuli: Dict[UUID, StimulusInfo] = {}
        self._selection_history: List[SelectionResult] = []

        # 적응적 학습
        self._learning_enabled = True
        self._adaptation_interval = 10.0  # 10초마다 적응
        self._last_adaptation_time = time.time()

        # 성능 메트릭
        self._total_selections = 0
        self._correct_selections = 0
        self._false_positives = 0
        self._false_negatives = 0

    async def initialize(self) -> bool:
        """선택적 주의 시스템 초기화"""
        try:
            self.logger.info("Initializing Selective Attention system...")

            # 기본 선택 기준 설정
            await self._setup_default_criteria()

            # 백그라운드 프로세스 시작
            asyncio.create_task(self._monitor_stimuli())
            asyncio.create_task(self._adaptive_learning())

            self.logger.info("Selective Attention system initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Selective Attention: {e}")
            return False

    async def add_filter(self, attention_filter: AttentionFilter) -> bool:
        """주의 필터 추가"""
        try:
            self._active_filters[attention_filter.id] = attention_filter
            self.logger.info(f"Added attention filter: {attention_filter.name}")
            return True

        except Exception as e:
            self.logger.error(f"Error adding filter {attention_filter.name}: {e}")
            return False

    async def remove_filter(self, filter_id: UUID) -> bool:
        """주의 필터 제거"""
        try:
            if filter_id in self._active_filters:
                filter_name = self._active_filters[filter_id].name
                del self._active_filters[filter_id]
                self.logger.info(f"Removed attention filter: {filter_name}")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Error removing filter {filter_id}: {e}")
            return False

    async def process_stimulus(self, stimulus: StimulusInfo) -> SelectionResult:
        """
        자극 처리 및 선택 결정

        Args:
            stimulus: 처리할 자극 정보

        Returns:
            선택 결과
        """
        try:
            start_time = time.time()

            # 자극 등록
            self._current_stimuli[stimulus.id] = stimulus

            # 용량 관리
            await self._manage_stimulus_capacity()

            # 각 필터에 대해 평가
            filter_scores = {}
            selection_reasoning = []

            for filter_id, attention_filter in self._active_filters.items():
                score = await self._evaluate_stimulus_against_filter(
                    stimulus, attention_filter
                )
                filter_scores[filter_id] = score

                if score > 0.5:  # 임계값 초과
                    selection_reasoning.append(
                        f"Filter '{attention_filter.name}' score: {score:.3f}"
                    )

            # 최종 선택 결정
            final_score, selected = await self._make_selection_decision(
                stimulus, filter_scores
            )

            # 우선순위 결정
            priority = await self._determine_priority(stimulus, final_score)

            # 결과 생성
            processing_time = (time.time() - start_time) * 1000
            result = SelectionResult(
                stimulus_id=stimulus.id,
                selected=selected,
                confidence_score=final_score,
                priority=priority,
                reasoning=selection_reasoning,
                filter_scores=filter_scores,
                processing_time_ms=processing_time
            )

            # 선택 기록 저장
            self._selection_history.append(result)
            self._total_selections += 1

            # 선택된 자극 추가 처리
            if selected:
                await self._handle_selected_stimulus(stimulus, result)

            return result

        except Exception as e:
            self.logger.error(f"Error processing stimulus {stimulus.id}: {e}")
            return SelectionResult(
                stimulus_id=stimulus.id,
                selected=False,
                confidence_score=0.0
            )

    async def get_current_selections(self) -> List[Dict[str, Any]]:
        """현재 선택된 자극들 조회"""
        selected_stimuli = []

        for result in self._selection_history[-50:]:  # 최근 50개
            if result.selected:
                stimulus = self._current_stimuli.get(result.stimulus_id)
                if stimulus:
                    selected_stimuli.append({
                        "stimulus_id": str(result.stimulus_id),
                        "stimulus_type": stimulus.type,
                        "priority": result.priority.name,
                        "confidence": result.confidence_score,
                        "timestamp": stimulus.timestamp
                    })

        return selected_stimuli

    async def get_attention_metrics(self) -> Dict[str, Any]:
        """주의 성능 메트릭 조회"""
        if self._total_selections == 0:
            return {"message": "No selections made yet"}

        precision = (
            self._correct_selections /
            max(1, self._correct_selections + self._false_positives)
        )
        recall = (
            self._correct_selections /
            max(1, self._correct_selections + self._false_negatives)
        )

        return {
            "total_selections": self._total_selections,
            "correct_selections": self._correct_selections,
            "false_positives": self._false_positives,
            "false_negatives": self._false_negatives,
            "precision": precision,
            "recall": recall,
            "f1_score": 2 * (precision * recall) / max(0.001, precision + recall),
            "active_filters": len(self._active_filters),
            "current_stimuli": len(self._current_stimuli)
        }

    async def update_selection_feedback(self, stimulus_id: UUID,
                                      was_correct: bool) -> None:
        """선택 결과에 대한 피드백 업데이트"""
        try:
            # 해당 선택 결과 찾기
            for result in reversed(self._selection_history):
                if result.stimulus_id == stimulus_id:
                    if result.selected and was_correct:
                        self._correct_selections += 1
                    elif result.selected and not was_correct:
                        self._false_positives += 1
                    elif not result.selected and was_correct:
                        self._false_negatives += 1

                    break

            # 적응적 학습에 피드백 반영
            if self._learning_enabled:
                await self._learn_from_feedback(stimulus_id, was_correct)

        except Exception as e:
            self.logger.error(f"Error updating feedback for {stimulus_id}: {e}")

    async def _setup_default_criteria(self) -> None:
        """기본 선택 기준 설정"""
        # 우선순위 기반 기준
        priority_criteria = SelectionCriteria(
            name="Priority Filter",
            filter_type=FilterType.PRIORITY_BASED,
            weight=2.0,
            threshold=0.3,
            conditions={"min_priority": AttentionPriority.MEDIUM}
        )

        # 시간 기반 기준 (최신성)
        temporal_criteria = SelectionCriteria(
            name="Recency Filter",
            filter_type=FilterType.TEMPORAL_BASED,
            weight=1.0,
            threshold=0.5,
            conditions={"max_age_seconds": 30.0}
        )

        # 의미 기반 기준
        semantic_criteria = SelectionCriteria(
            name="Relevance Filter",
            filter_type=FilterType.SEMANTIC_BASED,
            weight=1.5,
            threshold=0.4,
            conditions={"required_tags": ["important", "urgent", "relevant"]}
        )

        self._default_criteria = [priority_criteria, temporal_criteria, semantic_criteria]

        # 기본 필터 생성
        default_filter = AttentionFilter(
            name="Default Selection Filter",
            criteria=self._default_criteria,
            combination_rule="WEIGHTED",
            adaptive=True
        )

        await self.add_filter(default_filter)

    async def _evaluate_stimulus_against_filter(self, stimulus: StimulusInfo,
                                              attention_filter: AttentionFilter) -> float:
        """특정 필터에 대해 자극 평가"""
        try:
            scores = []

            for criteria in attention_filter.criteria:
                if not criteria.is_active:
                    continue

                score = await self._evaluate_criteria(stimulus, criteria)
                weighted_score = score * criteria.weight
                scores.append(weighted_score)

            if not scores:
                return 0.0

            # 조합 규칙에 따른 최종 점수 계산
            if attention_filter.combination_rule == "AND":
                return min(scores)
            elif attention_filter.combination_rule == "OR":
                return max(scores)
            elif attention_filter.combination_rule == "WEIGHTED":
                total_weight = sum(c.weight for c in attention_filter.criteria if c.is_active)
                return sum(scores) / max(1.0, total_weight)
            else:
                return sum(scores) / len(scores)

        except Exception as e:
            self.logger.error(f"Error evaluating stimulus against filter: {e}")
            return 0.0

    async def _evaluate_criteria(self, stimulus: StimulusInfo,
                               criteria: SelectionCriteria) -> float:
        """개별 기준에 대해 자극 평가"""
        try:
            if criteria.filter_type == FilterType.PRIORITY_BASED:
                return self._evaluate_priority_criteria(stimulus, criteria)

            elif criteria.filter_type == FilterType.TEMPORAL_BASED:
                return self._evaluate_temporal_criteria(stimulus, criteria)

            elif criteria.filter_type == FilterType.SEMANTIC_BASED:
                return self._evaluate_semantic_criteria(stimulus, criteria)

            elif criteria.filter_type == FilterType.FEATURE_BASED:
                return self._evaluate_feature_criteria(stimulus, criteria)

            elif criteria.filter_type == FilterType.SPATIAL_BASED:
                return self._evaluate_spatial_criteria(stimulus, criteria)

            return 0.0

        except Exception as e:
            self.logger.error(f"Error evaluating criteria {criteria.name}: {e}")
            return 0.0

    def _evaluate_priority_criteria(self, stimulus: StimulusInfo,
                                  criteria: SelectionCriteria) -> float:
        """우선순위 기준 평가"""
        min_priority = criteria.conditions.get("min_priority", AttentionPriority.LOW)

        if stimulus.base_priority.value <= min_priority.value:
            return 1.0
        else:
            # 우선순위 차이에 따른 점진적 점수
            priority_diff = stimulus.base_priority.value - min_priority.value
            return max(0.0, 1.0 - priority_diff * 0.2)

    def _evaluate_temporal_criteria(self, stimulus: StimulusInfo,
                                  criteria: SelectionCriteria) -> float:
        """시간 기준 평가"""
        max_age = criteria.conditions.get("max_age_seconds", 60.0)
        current_time = time.time()
        age = current_time - stimulus.timestamp

        if age <= max_age:
            return 1.0 - (age / max_age) * 0.5  # 선형 감소
        else:
            return 0.0

    def _evaluate_semantic_criteria(self, stimulus: StimulusInfo,
                                  criteria: SelectionCriteria) -> float:
        """의미 기준 평가"""
        required_tags = criteria.conditions.get("required_tags", [])

        if not required_tags:
            return 1.0

        matching_tags = set(stimulus.semantic_tags) & set(required_tags)
        return len(matching_tags) / len(required_tags)

    def _evaluate_feature_criteria(self, stimulus: StimulusInfo,
                                 criteria: SelectionCriteria) -> float:
        """특징 기준 평가"""
        required_features = criteria.conditions.get("required_features", {})

        if not required_features:
            return 1.0

        score = 0.0
        for feature_name, expected_value in required_features.items():
            if feature_name in stimulus.features:
                actual_value = stimulus.features[feature_name]
                # 값의 유사도에 따른 점수 (간단한 구현)
                if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                    similarity = 1.0 - abs(expected_value - actual_value) / max(abs(expected_value), 1.0)
                    score += max(0.0, similarity)
                elif expected_value == actual_value:
                    score += 1.0

        return score / len(required_features) if required_features else 1.0

    def _evaluate_spatial_criteria(self, stimulus: StimulusInfo,
                                 criteria: SelectionCriteria) -> float:
        """공간 기준 평가"""
        if not stimulus.spatial_location:
            return 0.5  # 공간 정보가 없으면 중립적 점수

        target_location = criteria.conditions.get("target_location", {})
        max_distance = criteria.conditions.get("max_distance", float('inf'))

        if not target_location:
            return 1.0

        # 단순한 유클리드 거리 계산
        distance = 0.0
        for axis in ['x', 'y', 'z']:
            if axis in target_location and axis in stimulus.spatial_location:
                diff = target_location[axis] - stimulus.spatial_location[axis]
                distance += diff ** 2

        distance = distance ** 0.5

        if distance <= max_distance:
            return 1.0 - (distance / max_distance) * 0.5
        else:
            return 0.0

    async def _make_selection_decision(self, stimulus: StimulusInfo,
                                     filter_scores: Dict[UUID, float]) -> tuple[float, bool]:
        """최종 선택 결정"""
        if not filter_scores:
            return 0.0, False

        # 가중 평균 계산
        total_weight = 0.0
        weighted_sum = 0.0

        for filter_id, score in filter_scores.items():
            if filter_id in self._active_filters:
                # 필터의 성능 기반 가중치 계산
                filter_performance = self._calculate_filter_performance(filter_id)
                weight = filter_performance

                weighted_sum += score * weight
                total_weight += weight

        final_score = weighted_sum / max(total_weight, 1.0)

        # 적응적 임계값 결정
        adaptive_threshold = await self._calculate_adaptive_threshold()
        selected = final_score >= adaptive_threshold

        return final_score, selected

    async def _determine_priority(self, stimulus: StimulusInfo,
                                confidence_score: float) -> AttentionPriority:
        """우선순위 결정"""
        base_priority_value = stimulus.base_priority.value

        # 신뢰도에 따른 우선순위 조정
        if confidence_score >= 0.9:
            adjusted_priority = max(1, base_priority_value - 1)
        elif confidence_score >= 0.7:
            adjusted_priority = base_priority_value
        else:
            adjusted_priority = min(5, base_priority_value + 1)

        return AttentionPriority(adjusted_priority)

    async def _handle_selected_stimulus(self, stimulus: StimulusInfo,
                                      result: SelectionResult) -> None:
        """선택된 자극에 대한 추가 처리"""
        try:
            # 로깅
            self.logger.info(f"Selected stimulus: {stimulus.type} "
                           f"(confidence: {result.confidence_score:.3f}, "
                           f"priority: {result.priority.name})")

            # 컨텍스트 업데이트
            await self._update_selection_context(stimulus, result)

        except Exception as e:
            self.logger.error(f"Error handling selected stimulus: {e}")

    async def _manage_stimulus_capacity(self) -> None:
        """자극 용량 관리"""
        if len(self._current_stimuli) > self.max_concurrent_stimuli:
            # 오래된 자극 제거
            sorted_stimuli = sorted(
                self._current_stimuli.items(),
                key=lambda x: x[1].timestamp
            )

            # 가장 오래된 것들 제거
            remove_count = len(self._current_stimuli) - self.max_concurrent_stimuli
            for i in range(remove_count):
                stimulus_id, _ = sorted_stimuli[i]
                del self._current_stimuli[stimulus_id]

    def _calculate_filter_performance(self, filter_id: UUID) -> float:
        """필터 성능 계산"""
        if filter_id not in self._active_filters:
            return 1.0

        attention_filter = self._active_filters[filter_id]

        if not attention_filter.performance_history:
            return 1.0

        # 최근 성능의 평균
        recent_performance = attention_filter.performance_history[-10:]
        return sum(recent_performance) / len(recent_performance)

    async def _calculate_adaptive_threshold(self) -> float:
        """적응적 임계값 계산"""
        base_threshold = 0.5

        # 최근 성능에 따른 임계값 조정
        if self._total_selections > 10:
            recent_precision = self._correct_selections / max(1, self._total_selections)

            if recent_precision < 0.7:  # 정밀도가 낮으면 임계값 상승
                return min(0.8, base_threshold + 0.1)
            elif recent_precision > 0.9:  # 정밀도가 높으면 임계값 하강
                return max(0.3, base_threshold - 0.1)

        return base_threshold

    async def _update_selection_context(self, stimulus: StimulusInfo,
                                      result: SelectionResult) -> None:
        """선택 컨텍스트 업데이트"""
        # 추후 구현: 선택된 자극의 패턴 학습
        pass

    async def _learn_from_feedback(self, stimulus_id: UUID, was_correct: bool) -> None:
        """피드백으로부터 학습"""
        try:
            # 해당 자극의 선택 결과 찾기
            for result in reversed(self._selection_history):
                if result.stimulus_id == stimulus_id:
                    # 각 필터의 성능 업데이트
                    for filter_id, score in result.filter_scores.items():
                        if filter_id in self._active_filters:
                            attention_filter = self._active_filters[filter_id]

                            # 성능 점수 계산 (올바른 선택이면 1.0, 틀렸으면 0.0)
                            performance_score = 1.0 if was_correct else 0.0
                            attention_filter.performance_history.append(performance_score)

                            # 히스토리 길이 제한
                            if len(attention_filter.performance_history) > 100:
                                attention_filter.performance_history = \
                                    attention_filter.performance_history[-50:]

                    break

        except Exception as e:
            self.logger.error(f"Error learning from feedback: {e}")

    async def _monitor_stimuli(self) -> None:
        """자극 모니터링 백그라운드 프로세스"""
        while True:
            try:
                current_time = time.time()

                # 오래된 자극 정리 (5분 이상 된 것들)
                old_stimuli = [
                    stimulus_id for stimulus_id, stimulus in self._current_stimuli.items()
                    if current_time - stimulus.timestamp > 300.0
                ]

                for stimulus_id in old_stimuli:
                    del self._current_stimuli[stimulus_id]

                await asyncio.sleep(10.0)

            except Exception as e:
                self.logger.error(f"Error in stimuli monitoring: {e}")
                await asyncio.sleep(30.0)

    async def _adaptive_learning(self) -> None:
        """적응적 학습 백그라운드 프로세스"""
        while True:
            try:
                current_time = time.time()

                if (self._learning_enabled and
                    current_time - self._last_adaptation_time > self._adaptation_interval):

                    await self._adapt_filters()
                    self._last_adaptation_time = current_time

                await asyncio.sleep(self._adaptation_interval)

            except Exception as e:
                self.logger.error(f"Error in adaptive learning: {e}")
                await asyncio.sleep(60.0)

    async def _adapt_filters(self) -> None:
        """필터 적응적 조정"""
        try:
            for filter_id, attention_filter in self._active_filters.items():
                if not attention_filter.adaptive:
                    continue

                # 필터 성능 분석
                performance = self._calculate_filter_performance(filter_id)

                # 성능이 낮으면 기준 조정
                if performance < 0.6:
                    for criteria in attention_filter.criteria:
                        if criteria.threshold > 0.1:
                            criteria.threshold -= attention_filter.learning_rate * 0.1

                # 성능이 높으면 기준 강화
                elif performance > 0.9:
                    for criteria in attention_filter.criteria:
                        if criteria.threshold < 0.9:
                            criteria.threshold += attention_filter.learning_rate * 0.05

            self.logger.debug("Completed adaptive filter adjustment")

        except Exception as e:
            self.logger.error(f"Error adapting filters: {e}")


async def create_selective_attention(max_concurrent_stimuli: int = 10) -> SelectiveAttention:
    """SelectiveAttention 인스턴스 생성 및 초기화"""
    system = SelectiveAttention(max_concurrent_stimuli)
    await system.initialize()
    return system