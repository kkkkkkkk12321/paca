"""
IIS Calculator Module
지능적 지위성 점수(Intelligent Intelligence Quotient Score) 계산 시스템

IIS는 AI의 학습 수준을 0-100점으로 측정하는 종합 지표입니다.
- 학습된 전술 숙련도 (30%)
- 문제 해결 성공률 (25%)
- 추론 품질 점수 (20%)
- 학습 속도 (15%)
- 적응 능력 (10%)
"""

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import statistics
import math

# 조건부 임포트: 패키지 실행시와 직접 실행시 모두 지원
try:
    from ..core.types.base import (
        ID, Timestamp, Result, current_timestamp, generate_id, create_success, create_failure
    )
except ImportError:
    from paca.core.types.base import (
        ID, Timestamp, Result, current_timestamp, generate_id, create_success, create_failure
    )


class TrendType(Enum):
    """IIS 점수 추세 타입"""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


class IISComponent(Enum):
    """IIS 구성 요소"""
    TACTIC_MASTERY = "tactic_mastery"
    PROBLEM_SOLVING = "problem_solving"
    REASONING_QUALITY = "reasoning_quality"
    LEARNING_SPEED = "learning_speed"
    ADAPTATION_ABILITY = "adaptation_ability"


@dataclass(frozen=True)
class IISBreakdown:
    """IIS 점수 세부 분석"""
    tactic_mastery: float  # 전술 숙련도 (0-100)
    problem_solving: float  # 문제 해결 성공률 (0-100)
    reasoning_quality: float  # 추론 품질 (0-100)
    learning_speed: float  # 학습 속도 (0-100)
    adaptation_ability: float  # 적응 능력 (0-100)

    def to_dict(self) -> Dict[str, float]:
        """딕셔너리로 변환"""
        return asdict(self)


@dataclass(frozen=True)
class IISScore:
    """IIS 점수 결과"""
    current_score: int  # 현재 IIS 점수 (0-100)
    trend: TrendType  # 추세
    breakdown: IISBreakdown  # 세부 점수
    confidence: float  # 신뢰도 (0.0-1.0)
    calculation_timestamp: Timestamp  # 계산 시점

    def get_grade(self) -> str:
        """점수에 따른 등급 반환"""
        if self.current_score >= 90:
            return "S+"
        elif self.current_score >= 80:
            return "S"
        elif self.current_score >= 70:
            return "A"
        elif self.current_score >= 60:
            return "B"
        elif self.current_score >= 50:
            return "C"
        elif self.current_score >= 40:
            return "D"
        else:
            return "F"

    def get_strongest_area(self) -> Tuple[str, float]:
        """가장 강한 영역 반환"""
        breakdown_dict = self.breakdown.to_dict()
        strongest = max(breakdown_dict.items(), key=lambda x: x[1])
        return strongest

    def get_weakest_area(self) -> Tuple[str, float]:
        """가장 약한 영역 반환"""
        breakdown_dict = self.breakdown.to_dict()
        weakest = min(breakdown_dict.items(), key=lambda x: x[1])
        return weakest


@dataclass(frozen=True)
class LearningData:
    """학습 데이터 컨테이너"""
    interactions_count: int  # 총 상호작용 횟수
    successful_interactions: int  # 성공한 상호작용 횟수
    reasoning_sessions: List[Dict[str, Any]]  # 추론 세션 데이터
    tactic_usage: Dict[str, Dict[str, Any]]  # 전술 사용 데이터
    learning_events: List[Dict[str, Any]]  # 학습 이벤트 데이터
    adaptation_events: List[Dict[str, Any]]  # 적응 이벤트 데이터
    time_span_days: float  # 데이터 수집 기간 (일)


@dataclass(frozen=True)
class InteractionResult:
    """상호작용 결과"""
    interaction_id: ID
    timestamp: Timestamp
    success: bool
    complexity_score: int
    reasoning_quality: float
    response_time_ms: int
    tactics_used: List[str]
    adaptation_required: bool


class IISCalculator:
    """
    IIS (지능적 지위성 점수) 계산기

    AI의 학습 수준을 종합적으로 평가하여 0-100점으로 점수화합니다.
    """

    # 가중치 상수
    WEIGHTS = {
        IISComponent.TACTIC_MASTERY: 0.30,      # 30%
        IISComponent.PROBLEM_SOLVING: 0.25,     # 25%
        IISComponent.REASONING_QUALITY: 0.20,   # 20%
        IISComponent.LEARNING_SPEED: 0.15,      # 15%
        IISComponent.ADAPTATION_ABILITY: 0.10   # 10%
    }

    # 점수 임계값
    TREND_THRESHOLD = 5.0  # 추세 판단 임계값
    MIN_DATA_POINTS = 10   # 최소 필요 데이터 포인트

    def __init__(self):
        """IIS 계산기 초기화"""
        self._calculation_history: List[IISScore] = []
        self._cache: Dict[str, Tuple[IISScore, Timestamp]] = {}
        self._cache_duration = 300  # 5분 캐시

    async def calculate_iis_score(self, learning_data: LearningData) -> Result[IISScore]:
        """
        IIS 점수 계산

        Args:
            learning_data: 학습 데이터

        Returns:
            IIS 점수 결과
        """
        try:
            # 캐시 확인
            cache_key = self._generate_cache_key(learning_data)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return create_success(cached_result)

            # 데이터 유효성 검사
            validation_result = self._validate_learning_data(learning_data)
            if not validation_result.is_success:
                return create_failure(validation_result.error)

            # 각 구성 요소 점수 계산
            components = await self._calculate_all_components(learning_data)

            # 전체 IIS 점수 계산
            weighted_score = self._calculate_weighted_score(components)

            # 신뢰도 계산
            confidence = self._calculate_confidence(learning_data)

            # 추세 분석
            trend = self._analyze_trend(weighted_score)

            # IIS 점수 객체 생성
            iis_score = IISScore(
                current_score=int(round(weighted_score)),
                trend=trend,
                breakdown=components,
                confidence=confidence,
                calculation_timestamp=current_timestamp()
            )

            # 결과 캐싱 및 히스토리 저장
            self._cache_result(cache_key, iis_score)
            self._add_to_history(iis_score)

            return create_success(iis_score)

        except Exception as e:
            return create_failure(e)

    async def update_iis_from_interaction(self, interaction_result: InteractionResult) -> Result[None]:
        """
        대화 결과 기반 IIS 업데이트

        Args:
            interaction_result: 상호작용 결과

        Returns:
            업데이트 결과
        """
        try:
            # 상호작용 데이터를 학습 데이터에 반영
            # 실제 구현에서는 영구 저장소에서 기존 데이터를 가져와 업데이트
            # 여기서는 시뮬레이션

            # 간단한 증분 업데이트 시뮬레이션
            await asyncio.sleep(0.01)  # 비동기 처리 시뮬레이션

            # 캐시 무효화
            self._invalidate_cache()

            return create_success(None)

        except Exception as e:
            return create_failure(e)

    def get_recent_scores(self, count: int = 10) -> List[IISScore]:
        """
        최근 IIS 점수 조회

        Args:
            count: 조회할 점수 개수

        Returns:
            최근 IIS 점수 목록
        """
        return self._calculation_history[-count:] if self._calculation_history else []

    def get_improvement_suggestions(self, iis_score: IISScore) -> List[str]:
        """
        개선 제안 생성

        Args:
            iis_score: IIS 점수 객체

        Returns:
            개선 제안 목록
        """
        suggestions = []
        breakdown = iis_score.breakdown.to_dict()

        for component, score in breakdown.items():
            if score < 60:  # 60점 미만인 영역에 대한 제안
                suggestion = self._generate_improvement_suggestion(component, score)
                suggestions.append(suggestion)

        # 전체 점수가 낮은 경우 일반적인 제안
        if iis_score.current_score < 70:
            suggestions.append("전반적인 학습 활동을 늘려 다양한 문제 해결 경험을 쌓으세요.")

        return suggestions

    # === 내부 메서드들 ===

    async def _calculate_all_components(self, learning_data: LearningData) -> IISBreakdown:
        """모든 구성 요소 점수 계산"""
        # 병렬로 모든 구성 요소 계산
        tasks = [
            self._calculate_tactic_mastery(learning_data),
            self._calculate_problem_solving(learning_data),
            self._calculate_reasoning_quality(learning_data),
            self._calculate_learning_speed(learning_data),
            self._calculate_adaptation_ability(learning_data)
        ]

        results = await asyncio.gather(*tasks)

        return IISBreakdown(
            tactic_mastery=results[0],
            problem_solving=results[1],
            reasoning_quality=results[2],
            learning_speed=results[3],
            adaptation_ability=results[4]
        )

    async def _calculate_tactic_mastery(self, learning_data: LearningData) -> float:
        """전술 숙련도 계산 (30% 가중치)"""
        if not learning_data.tactic_usage:
            return 50.0  # 기본값

        total_tactics = len(learning_data.tactic_usage)
        mastered_tactics = 0
        average_proficiency = 0.0

        for tactic_id, tactic_data in learning_data.tactic_usage.items():
            proficiency = tactic_data.get('proficiency', 0.0)
            usage_count = tactic_data.get('usage_count', 0)
            success_rate = tactic_data.get('success_rate', 0.0)

            # 숙련도 계산 (사용 횟수, 성공률, 기본 숙련도 종합)
            normalized_usage = min(usage_count / 10, 1.0)  # 10회 사용시 최대값
            tactic_score = (proficiency * 0.5 + success_rate * 0.3 + normalized_usage * 0.2) * 100

            average_proficiency += tactic_score
            if tactic_score > 75:  # 75점 이상을 숙련으로 간주
                mastered_tactics += 1

        if total_tactics > 0:
            average_proficiency /= total_tactics
            mastery_ratio = mastered_tactics / total_tactics
            return min((average_proficiency * 0.7 + mastery_ratio * 100 * 0.3), 100.0)

        return 50.0

    async def _calculate_problem_solving(self, learning_data: LearningData) -> float:
        """문제 해결 성공률 계산 (25% 가중치)"""
        if learning_data.interactions_count == 0:
            return 50.0

        base_success_rate = (learning_data.successful_interactions / learning_data.interactions_count) * 100

        # 복잡도별 성공률 분석
        complexity_bonuses = 0.0
        if learning_data.reasoning_sessions:
            high_complexity_success = 0
            high_complexity_total = 0

            for session in learning_data.reasoning_sessions:
                complexity = session.get('complexity_score', 0)
                success = session.get('success', False)

                if complexity >= 70:  # 고복잡도 문제
                    high_complexity_total += 1
                    if success:
                        high_complexity_success += 1

            if high_complexity_total > 0:
                high_complexity_rate = high_complexity_success / high_complexity_total
                complexity_bonuses = high_complexity_rate * 20  # 최대 20점 보너스

        return min(base_success_rate + complexity_bonuses, 100.0)

    async def _calculate_reasoning_quality(self, learning_data: LearningData) -> float:
        """추론 품질 점수 계산 (20% 가중치)"""
        if not learning_data.reasoning_sessions:
            return 50.0

        quality_scores = []

        for session in learning_data.reasoning_sessions:
            # 추론 품질 메트릭들
            logical_consistency = session.get('logical_consistency', 0.5)
            step_clarity = session.get('step_clarity', 0.5)
            conclusion_validity = session.get('conclusion_validity', 0.5)
            completeness = session.get('completeness', 0.5)
            efficiency = session.get('efficiency', 0.5)

            # 가중 평균으로 세션 품질 계산
            session_quality = (
                logical_consistency * 0.25 +
                step_clarity * 0.20 +
                conclusion_validity * 0.25 +
                completeness * 0.15 +
                efficiency * 0.15
            ) * 100

            quality_scores.append(session_quality)

        # 최근 품질에 더 많은 가중치 부여
        if len(quality_scores) > 5:
            recent_scores = quality_scores[-5:]
            overall_score = statistics.mean(recent_scores) * 0.7 + statistics.mean(quality_scores) * 0.3
        else:
            overall_score = statistics.mean(quality_scores)

        return min(overall_score, 100.0)

    async def _calculate_learning_speed(self, learning_data: LearningData) -> float:
        """학습 속도 계산 (15% 가중치)"""
        if not learning_data.learning_events or learning_data.time_span_days <= 0:
            return 50.0

        # 학습 이벤트 분석
        learning_rate = len(learning_data.learning_events) / learning_data.time_span_days

        # 학습 곡선 분석 (시간에 따른 성능 향상)
        improvement_trend = 0.0
        if len(learning_data.learning_events) > 2:
            early_performance = []
            late_performance = []

            mid_point = len(learning_data.learning_events) // 2

            for i, event in enumerate(learning_data.learning_events):
                performance = event.get('performance_score', 0.5)
                if i < mid_point:
                    early_performance.append(performance)
                else:
                    late_performance.append(performance)

            if early_performance and late_performance:
                early_avg = statistics.mean(early_performance)
                late_avg = statistics.mean(late_performance)
                improvement_trend = (late_avg - early_avg) * 100

        # 학습률과 개선 추세를 조합
        normalized_rate = min(learning_rate * 10, 1.0) * 50  # 일일 1개 학습 이벤트를 기준으로 정규화
        trend_bonus = max(improvement_trend, 0) * 2  # 개선 추세 보너스

        return min(normalized_rate + trend_bonus + 25, 100.0)  # 기본 25점 + 성과

    async def _calculate_adaptation_ability(self, learning_data: LearningData) -> float:
        """적응 능력 계산 (10% 가중치)"""
        if not learning_data.adaptation_events:
            return 50.0

        successful_adaptations = 0
        adaptation_speed_scores = []

        for event in learning_data.adaptation_events:
            success = event.get('success', False)
            adaptation_time = event.get('adaptation_time_ms', 1000)
            complexity = event.get('situation_complexity', 0.5)

            if success:
                successful_adaptations += 1

                # 적응 속도 점수 (빠를수록 높은 점수)
                speed_score = max(0, 100 - (adaptation_time / 100))  # 1초당 10점 감점
                complexity_adjusted_score = speed_score * (0.5 + complexity * 0.5)
                adaptation_speed_scores.append(complexity_adjusted_score)

        # 성공률
        success_rate = (successful_adaptations / len(learning_data.adaptation_events)) * 100

        # 평균 적응 속도
        avg_speed_score = statistics.mean(adaptation_speed_scores) if adaptation_speed_scores else 50

        # 조합 점수
        return min((success_rate * 0.6 + avg_speed_score * 0.4), 100.0)

    def _calculate_weighted_score(self, breakdown: IISBreakdown) -> float:
        """가중치 적용한 최종 점수 계산"""
        breakdown_dict = breakdown.to_dict()

        total_score = 0.0
        for component_name, score in breakdown_dict.items():
            component_enum = IISComponent(component_name)
            weight = self.WEIGHTS[component_enum]
            total_score += score * weight

        return min(total_score, 100.0)

    def _calculate_confidence(self, learning_data: LearningData) -> float:
        """신뢰도 계산"""
        # 데이터 양에 따른 신뢰도
        data_confidence = min(learning_data.interactions_count / 100, 1.0)

        # 데이터 기간에 따른 신뢰도
        time_confidence = min(learning_data.time_span_days / 30, 1.0)

        # 데이터 다양성에 따른 신뢰도
        diversity_score = 0.0
        if learning_data.reasoning_sessions:
            complexity_variety = len(set(
                session.get('complexity_score', 0) // 10
                for session in learning_data.reasoning_sessions
            )) / 10  # 복잡도 구간의 다양성
            diversity_score += complexity_variety * 0.5

        if learning_data.tactic_usage:
            tactic_variety = min(len(learning_data.tactic_usage) / 20, 1.0)
            diversity_score += tactic_variety * 0.5

        # 전체 신뢰도 계산
        overall_confidence = (data_confidence * 0.4 + time_confidence * 0.3 + diversity_score * 0.3)
        return min(overall_confidence, 1.0)

    def _analyze_trend(self, current_score: float) -> TrendType:
        """점수 추세 분석"""
        if len(self._calculation_history) < 2:
            return TrendType.STABLE

        recent_scores = [score.current_score for score in self._calculation_history[-5:]]
        recent_scores.append(current_score)

        # 선형 회귀를 통한 추세 분석
        if len(recent_scores) >= 3:
            x_values = list(range(len(recent_scores)))
            slope = self._calculate_slope(x_values, recent_scores)

            if slope > self.TREND_THRESHOLD:
                return TrendType.IMPROVING
            elif slope < -self.TREND_THRESHOLD:
                return TrendType.DECLINING
            else:
                return TrendType.STABLE

        return TrendType.STABLE

    def _calculate_slope(self, x_values: List[int], y_values: List[float]) -> float:
        """선형 회귀 기울기 계산"""
        n = len(x_values)
        if n < 2:
            return 0.0

        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(y_values)

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        return numerator / denominator if denominator != 0 else 0.0

    def _validate_learning_data(self, learning_data: LearningData) -> Result[None]:
        """학습 데이터 유효성 검사"""
        try:
            if learning_data.interactions_count < 0:
                raise ValueError("상호작용 횟수는 0 이상이어야 합니다")

            if learning_data.successful_interactions > learning_data.interactions_count:
                raise ValueError("성공한 상호작용 횟수가 전체 횟수를 초과할 수 없습니다")

            if learning_data.time_span_days < 0:
                raise ValueError("데이터 수집 기간은 0 이상이어야 합니다")

            return create_success(None)

        except Exception as e:
            return create_failure(e)

    def _generate_cache_key(self, learning_data: LearningData) -> str:
        """캐시 키 생성"""
        # 학습 데이터의 핵심 요소들로 해시 생성
        key_data = {
            'interactions': learning_data.interactions_count,
            'successful': learning_data.successful_interactions,
            'reasoning_count': len(learning_data.reasoning_sessions),
            'tactics_count': len(learning_data.tactic_usage),
            'timespan': learning_data.time_span_days
        }
        return str(hash(json.dumps(key_data, sort_keys=True)))

    def _get_cached_result(self, cache_key: str) -> Optional[IISScore]:
        """캐시된 결과 조회"""
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if current_timestamp() - timestamp < self._cache_duration:
                return result
            else:
                del self._cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: IISScore) -> None:
        """결과 캐싱"""
        self._cache[cache_key] = (result, current_timestamp())

    def _invalidate_cache(self) -> None:
        """캐시 무효화"""
        self._cache.clear()

    def _add_to_history(self, iis_score: IISScore) -> None:
        """히스토리에 추가"""
        self._calculation_history.append(iis_score)
        # 최대 100개 히스토리 유지
        if len(self._calculation_history) > 100:
            self._calculation_history = self._calculation_history[-100:]

    def _generate_improvement_suggestion(self, component: str, score: float) -> str:
        """구성 요소별 개선 제안 생성"""
        suggestions = {
            "tactic_mastery": [
                "더 다양한 전술을 시도해보세요.",
                "기존 전술의 사용법을 개선해보세요.",
                "전술 조합을 실험해보세요."
            ],
            "problem_solving": [
                "더 어려운 문제에 도전해보세요.",
                "문제 해결 접근법을 다양화해보세요.",
                "실패한 문제를 다시 시도해보세요."
            ],
            "reasoning_quality": [
                "추론 과정을 더 체계적으로 구성해보세요.",
                "논리적 일관성을 점검해보세요.",
                "결론의 타당성을 검증해보세요."
            ],
            "learning_speed": [
                "학습 빈도를 늘려보세요.",
                "학습 방법을 개선해보세요.",
                "피드백을 적극적으로 반영해보세요."
            ],
            "adaptation_ability": [
                "새로운 상황에 더 빨리 적응해보세요.",
                "변화에 대한 대응 전략을 개발해보세요.",
                "유연성을 기르는 연습을 해보세요."
            ]
        }

        component_suggestions = suggestions.get(component, ["해당 영역의 학습을 강화해보세요."])
        # 점수에 따라 적절한 제안 선택
        if score < 30:
            return f"{component_suggestions[0]} (현재 점수: {score:.1f})"
        elif score < 50:
            return f"{component_suggestions[1]} (현재 점수: {score:.1f})"
        else:
            return f"{component_suggestions[2]} (현재 점수: {score:.1f})"


# Helper functions for creating sample data
def create_sample_learning_data() -> LearningData:
    """샘플 학습 데이터 생성 (테스트용)"""
    return LearningData(
        interactions_count=150,
        successful_interactions=120,
        reasoning_sessions=[
            {
                'complexity_score': 65,
                'success': True,
                'logical_consistency': 0.85,
                'step_clarity': 0.90,
                'conclusion_validity': 0.88,
                'completeness': 0.82,
                'efficiency': 0.75
            },
            {
                'complexity_score': 45,
                'success': True,
                'logical_consistency': 0.92,
                'step_clarity': 0.85,
                'conclusion_validity': 0.90,
                'completeness': 0.88,
                'efficiency': 0.80
            }
        ],
        tactic_usage={
            'analytical_reasoning': {
                'proficiency': 0.85,
                'usage_count': 25,
                'success_rate': 0.88
            },
            'creative_thinking': {
                'proficiency': 0.70,
                'usage_count': 15,
                'success_rate': 0.75
            }
        },
        learning_events=[
            {'performance_score': 0.65, 'timestamp': time.time() - 86400 * 10},
            {'performance_score': 0.70, 'timestamp': time.time() - 86400 * 5},
            {'performance_score': 0.78, 'timestamp': time.time() - 86400 * 1}
        ],
        adaptation_events=[
            {
                'success': True,
                'adaptation_time_ms': 500,
                'situation_complexity': 0.7
            },
            {
                'success': True,
                'adaptation_time_ms': 800,
                'situation_complexity': 0.6
            }
        ],
        time_span_days=30.0
    )


def create_sample_interaction_result() -> InteractionResult:
    """샘플 상호작용 결과 생성 (테스트용)"""
    return InteractionResult(
        interaction_id=generate_id("interaction_"),
        timestamp=current_timestamp(),
        success=True,
        complexity_score=65,
        reasoning_quality=0.85,
        response_time_ms=1200,
        tactics_used=['analytical_reasoning', 'systematic_approach'],
        adaptation_required=False
    )