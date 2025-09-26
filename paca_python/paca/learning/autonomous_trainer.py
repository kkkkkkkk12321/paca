"""
Autonomous Trainer Module
자율 훈련 시스템

AI가 스스로 약점을 분석하고 맞춤형 훈련을 수행하는 시스템입니다.
- 약점 영역 자동 감지
- 맞춤형 훈련 임무 생성
- 연속 자동 훈련 실행
- 자동 중단 조건 설정
"""

import asyncio
import json
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from enum import Enum
import statistics
import math

# 조건부 임포트: 패키지 실행시와 직접 실행시 모두 지원
try:
    from ..core.types.base import (
        ID, Timestamp, Result, Status, Priority, current_timestamp, generate_id,
        create_success, create_failure
    )
    from .iis_calculator import IISScore, IISBreakdown, LearningData, IISCalculator
except ImportError:
    from paca.core.types.base import (
        ID, Timestamp, Result, Status, Priority, current_timestamp, generate_id,
        create_success, create_failure
    )
    from paca.learning.iis_calculator import IISScore, IISBreakdown, LearningData, IISCalculator


class WeaknessType(Enum):
    """약점 유형"""
    TACTIC_PROFICIENCY = "tactic_proficiency"      # 전술 숙련도
    PROBLEM_COMPLEXITY = "problem_complexity"      # 문제 복잡도 처리
    REASONING_QUALITY = "reasoning_quality"        # 추론 품질
    LEARNING_EFFICIENCY = "learning_efficiency"    # 학습 효율성
    ADAPTATION_SPEED = "adaptation_speed"          # 적응 속도


class TrainingType(Enum):
    """훈련 유형"""
    TACTIC_PRACTICE = "tactic_practice"           # 전술 연습
    COMPLEX_PROBLEM = "complex_problem"           # 복잡한 문제 해결
    REASONING_DRILL = "reasoning_drill"           # 추론 훈련
    SPEED_TRAINING = "speed_training"             # 속도 훈련
    ADAPTATION_EXERCISE = "adaptation_exercise"   # 적응 연습


class TrainingStatus(Enum):
    """훈련 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class WeaknessArea:
    """약점 영역"""
    weakness_type: WeaknessType
    severity: float  # 심각도 (0.0-1.0)
    description: str
    current_score: float  # 현재 점수 (0-100)
    target_score: float   # 목표 점수 (0-100)
    priority: Priority
    identified_at: Timestamp


@dataclass(frozen=True)
class TrainingMission:
    """훈련 임무"""
    mission_id: str
    description: str
    training_type: TrainingType
    target_weakness: WeaknessType
    difficulty: int  # 난이도 (1-5)
    estimated_duration_minutes: int
    success_criteria: Dict[str, Any]
    created_at: Timestamp
    priority: Priority


@dataclass(frozen=True)
class TrainingResult:
    """훈련 결과"""
    mission_id: str
    status: TrainingStatus
    start_time: Timestamp
    end_time: Optional[Timestamp]
    success: bool
    performance_score: float  # 성과 점수 (0.0-1.0)
    improvement_metrics: Dict[str, float]
    feedback: str
    duration_minutes: float


@dataclass(frozen=True)
class TrainingSession:
    """훈련 세션"""
    session_id: str
    start_time: Timestamp
    end_time: Optional[Timestamp]
    total_missions: int
    completed_missions: int
    failed_missions: int
    skipped_missions: int
    overall_improvement: float
    session_summary: str


@dataclass(frozen=True)
class TrainingConfig:
    """훈련 설정"""
    max_cycles: int = 10
    max_session_duration_minutes: int = 60
    min_improvement_threshold: float = 0.05  # 최소 개선 임계값
    auto_difficulty_adjustment: bool = True
    enable_adaptive_breaks: bool = True
    break_duration_minutes: int = 5


class AutonomousTrainer:
    """
    자율 훈련 시스템

    AI가 스스로 약점을 분석하고 체계적인 훈련을 수행합니다.
    """

    # 훈련 임계값 상수
    WEAKNESS_THRESHOLD = 70.0  # 70점 미만을 약점으로 간주
    SEVERE_WEAKNESS_THRESHOLD = 50.0  # 50점 미만을 심각한 약점으로 간주
    IMPROVEMENT_TARGET_MULTIPLIER = 1.2  # 목표 점수 = 현재 점수 * 1.2

    def __init__(self, iis_calculator: Optional[IISCalculator] = None):
        """자율 훈련 시스템 초기화"""
        self.iis_calculator = iis_calculator or IISCalculator()
        self._training_history: List[TrainingSession] = []
        self._active_session: Optional[TrainingSession] = None
        self._stop_training_flag = False
        self._training_callbacks: List[Callable] = []

    async def analyze_weaknesses(self, learning_data: Optional[LearningData] = None) -> Result[List[WeaknessArea]]:
        """
        현재 약점 영역 분석

        Args:
            learning_data: 분석할 학습 데이터 (없으면 현재 상태 분석)

        Returns:
            약점 영역 목록
        """
        try:
            # IIS 점수 계산 또는 제공된 데이터 사용
            if learning_data:
                iis_result = await self.iis_calculator.calculate_iis_score(learning_data)
                if not iis_result.is_success:
                    return create_failure(iis_result.error)
                iis_score = iis_result.value
            else:
                # 현재 상태에서 가상의 학습 데이터 생성 (실제 구현에서는 실제 데이터 사용)
                sample_data = self._create_current_learning_data()
                iis_result = await self.iis_calculator.calculate_iis_score(sample_data)
                if not iis_result.is_success:
                    return create_failure(iis_result.error)
                iis_score = iis_result.value

            # 약점 영역 식별
            weaknesses = self._identify_weaknesses(iis_score.breakdown)

            return create_success(weaknesses)

        except Exception as e:
            return create_failure(e)

    async def generate_training_missions(self, weaknesses: List[WeaknessArea]) -> Result[List[TrainingMission]]:
        """
        약점 보완용 훈련 임무 생성

        Args:
            weaknesses: 약점 영역 목록

        Returns:
            훈련 임무 목록
        """
        try:
            missions = []

            # 우선순위순으로 정렬
            sorted_weaknesses = sorted(weaknesses, key=lambda w: (w.priority.value, w.severity), reverse=True)

            for weakness in sorted_weaknesses:
                # 각 약점에 대해 2-3개의 훈련 임무 생성
                weakness_missions = await self._generate_missions_for_weakness(weakness)
                missions.extend(weakness_missions)

                # 최대 임무 수 제한
                if len(missions) >= 15:
                    break

            return create_success(missions)

        except Exception as e:
            return create_failure(e)

    async def execute_continuous_training(self,
                                        config: Optional[TrainingConfig] = None,
                                        progress_callback: Optional[Callable] = None) -> Result[TrainingSession]:
        """
        연속 자동 훈련 실행

        Args:
            config: 훈련 설정
            progress_callback: 진행 상황 콜백 함수

        Returns:
            훈련 세션 결과
        """
        try:
            training_config = config or TrainingConfig()
            session_id = generate_id("training_session_")
            start_time = current_timestamp()

            if progress_callback:
                self._training_callbacks.append(progress_callback)

            # 훈련 세션 시작
            await self._notify_callbacks("session_started", {"session_id": session_id})

            # 1. 약점 분석
            await self._notify_callbacks("analyzing_weaknesses", {})
            weakness_result = await self.analyze_weaknesses()
            if not weakness_result.is_success:
                return create_failure(weakness_result.error)

            weaknesses = weakness_result.value
            if not weaknesses:
                session = TrainingSession(
                    session_id=session_id,
                    start_time=start_time,
                    end_time=current_timestamp(),
                    total_missions=0,
                    completed_missions=0,
                    failed_missions=0,
                    skipped_missions=0,
                    overall_improvement=0.0,
                    session_summary="약점이 발견되지 않았습니다. 현재 수준이 양호합니다."
                )
                return create_success(session)

            # 2. 훈련 임무 생성
            await self._notify_callbacks("generating_missions", {})
            mission_result = await self.generate_training_missions(weaknesses)
            if not mission_result.is_success:
                return create_failure(mission_result.error)

            missions = mission_result.value

            # 3. 훈련 실행
            training_results = []
            completed_count = 0
            failed_count = 0
            skipped_count = 0

            for cycle in range(training_config.max_cycles):
                if self._stop_training_flag:
                    break

                # 세션 시간 제한 확인
                elapsed_minutes = (current_timestamp() - start_time) / 60
                if elapsed_minutes > training_config.max_session_duration_minutes:
                    await self._notify_callbacks("session_timeout", {"elapsed_minutes": elapsed_minutes})
                    break

                # 미션별 훈련 실행
                for mission in missions:
                    if self._stop_training_flag:
                        break

                    await self._notify_callbacks("mission_started", {"mission": mission})

                    # 훈련 실행
                    result = await self._execute_training_mission(mission, training_config)
                    training_results.append(result)

                    # 결과 처리
                    if result.status == TrainingStatus.COMPLETED and result.success:
                        completed_count += 1
                    elif result.status == TrainingStatus.FAILED:
                        failed_count += 1
                    elif result.status == TrainingStatus.SKIPPED:
                        skipped_count += 1

                    await self._notify_callbacks("mission_completed", {"result": result})

                    # 적응형 휴식
                    if training_config.enable_adaptive_breaks and cycle > 0:
                        await asyncio.sleep(training_config.break_duration_minutes * 60 * 0.1)  # 시뮬레이션용 단축

                # 사이클별 개선도 평가
                improvement = await self._evaluate_cycle_improvement(cycle, training_results)
                await self._notify_callbacks("cycle_completed", {
                    "cycle": cycle,
                    "improvement": improvement
                })

                # 조기 종료 조건 확인
                if improvement < training_config.min_improvement_threshold:
                    await self._notify_callbacks("early_termination", {
                        "reason": "minimal_improvement",
                        "improvement": improvement
                    })
                    break

            # 4. 전체 개선도 계산
            overall_improvement = await self._calculate_overall_improvement(training_results)

            # 5. 세션 완료
            end_time = current_timestamp()
            session = TrainingSession(
                session_id=session_id,
                start_time=start_time,
                end_time=end_time,
                total_missions=len(missions),
                completed_missions=completed_count,
                failed_missions=failed_count,
                skipped_missions=skipped_count,
                overall_improvement=overall_improvement,
                session_summary=self._generate_session_summary(
                    completed_count, failed_count, skipped_count, overall_improvement
                )
            )

            self._training_history.append(session)
            self._stop_training_flag = False

            await self._notify_callbacks("session_completed", {"session": session})

            return create_success(session)

        except Exception as e:
            return create_failure(e)

    def stop_training(self) -> None:
        """훈련 중단"""
        self._stop_training_flag = True

    def get_training_history(self, limit: int = 10) -> List[TrainingSession]:
        """
        훈련 히스토리 조회

        Args:
            limit: 조회할 세션 수

        Returns:
            훈련 세션 목록
        """
        return self._training_history[-limit:] if self._training_history else []

    def get_training_statistics(self) -> Dict[str, Any]:
        """
        훈련 통계 조회

        Returns:
            훈련 통계 데이터
        """
        if not self._training_history:
            return {
                "total_sessions": 0,
                "total_missions": 0,
                "success_rate": 0.0,
                "average_improvement": 0.0
            }

        total_sessions = len(self._training_history)
        total_missions = sum(session.total_missions for session in self._training_history)
        total_completed = sum(session.completed_missions for session in self._training_history)
        success_rate = total_completed / total_missions if total_missions > 0 else 0.0
        average_improvement = statistics.mean(
            session.overall_improvement for session in self._training_history
        )

        return {
            "total_sessions": total_sessions,
            "total_missions": total_missions,
            "completed_missions": total_completed,
            "success_rate": success_rate,
            "average_improvement": average_improvement,
            "last_session_date": self._training_history[-1].start_time if self._training_history else None
        }

    # === 내부 메서드들 ===

    def _identify_weaknesses(self, breakdown: IISBreakdown) -> List[WeaknessArea]:
        """IIS 분석 결과에서 약점 영역 식별"""
        weaknesses = []
        breakdown_dict = breakdown.to_dict()
        current_time = current_timestamp()

        weakness_mapping = {
            "tactic_mastery": WeaknessType.TACTIC_PROFICIENCY,
            "problem_solving": WeaknessType.PROBLEM_COMPLEXITY,
            "reasoning_quality": WeaknessType.REASONING_QUALITY,
            "learning_speed": WeaknessType.LEARNING_EFFICIENCY,
            "adaptation_ability": WeaknessType.ADAPTATION_SPEED
        }

        for component_name, score in breakdown_dict.items():
            if score < self.WEAKNESS_THRESHOLD:
                weakness_type = weakness_mapping.get(component_name)
                if weakness_type:
                    # 심각도 계산 (점수가 낮을수록 높은 심각도)
                    severity = 1.0 - (score / 100.0)

                    # 우선순위 결정
                    if score < self.SEVERE_WEAKNESS_THRESHOLD:
                        priority = Priority.CRITICAL
                    elif score < 60:
                        priority = Priority.HIGH
                    else:
                        priority = Priority.NORMAL

                    # 목표 점수 계산
                    target_score = min(score * self.IMPROVEMENT_TARGET_MULTIPLIER, 100.0)

                    weakness = WeaknessArea(
                        weakness_type=weakness_type,
                        severity=severity,
                        description=self._generate_weakness_description(weakness_type, score),
                        current_score=score,
                        target_score=target_score,
                        priority=priority,
                        identified_at=current_time
                    )
                    weaknesses.append(weakness)

        return weaknesses

    async def _generate_missions_for_weakness(self, weakness: WeaknessArea) -> List[TrainingMission]:
        """특정 약점에 대한 훈련 임무 생성"""
        missions = []
        current_time = current_timestamp()

        # 약점 유형별 훈련 전략
        training_strategies = {
            WeaknessType.TACTIC_PROFICIENCY: [
                (TrainingType.TACTIC_PRACTICE, "기본 전술 연습", 2),
                (TrainingType.TACTIC_PRACTICE, "고급 전술 연습", 3),
                (TrainingType.COMPLEX_PROBLEM, "전술 적용 문제", 3)
            ],
            WeaknessType.PROBLEM_COMPLEXITY: [
                (TrainingType.COMPLEX_PROBLEM, "복잡도 점진적 증가 훈련", 3),
                (TrainingType.REASONING_DRILL, "복잡한 추론 연습", 4),
                (TrainingType.COMPLEX_PROBLEM, "다단계 문제 해결", 4)
            ],
            WeaknessType.REASONING_QUALITY: [
                (TrainingType.REASONING_DRILL, "논리적 일관성 훈련", 2),
                (TrainingType.REASONING_DRILL, "단계별 명확성 훈련", 3),
                (TrainingType.REASONING_DRILL, "결론 타당성 검증 훈련", 3)
            ],
            WeaknessType.LEARNING_EFFICIENCY: [
                (TrainingType.SPEED_TRAINING, "빠른 패턴 인식 훈련", 2),
                (TrainingType.SPEED_TRAINING, "효율적 학습 전략 훈련", 3),
                (TrainingType.ADAPTATION_EXERCISE, "학습 전이 연습", 4)
            ],
            WeaknessType.ADAPTATION_SPEED: [
                (TrainingType.ADAPTATION_EXERCISE, "상황 변화 대응 훈련", 3),
                (TrainingType.ADAPTATION_EXERCISE, "빠른 전략 전환 훈련", 4),
                (TrainingType.SPEED_TRAINING, "적응 속도 향상 훈련", 3)
            ]
        }

        strategies = training_strategies.get(weakness.weakness_type, [])

        for i, (training_type, base_description, base_difficulty) in enumerate(strategies):
            # 난이도 조정 (약점 심각도에 따라)
            adjusted_difficulty = min(base_difficulty + int(weakness.severity * 2), 5)

            # 예상 소요 시간 계산
            duration_minutes = base_difficulty * 10 + int(weakness.severity * 20)

            # 성공 기준 설정
            success_criteria = self._generate_success_criteria(training_type, weakness)

            mission = TrainingMission(
                mission_id=generate_id(f"mission_{weakness.weakness_type.value}_"),
                description=f"{base_description} (대상: {weakness.description})",
                training_type=training_type,
                target_weakness=weakness.weakness_type,
                difficulty=adjusted_difficulty,
                estimated_duration_minutes=duration_minutes,
                success_criteria=success_criteria,
                created_at=current_time,
                priority=weakness.priority
            )
            missions.append(mission)

        return missions

    async def _execute_training_mission(self, mission: TrainingMission, config: TrainingConfig) -> TrainingResult:
        """개별 훈련 임무 실행"""
        start_time = current_timestamp()

        try:
            # 실제 구현에서는 각 훈련 유형별로 실제 훈련을 수행
            # 여기서는 시뮬레이션

            # 훈련 시뮬레이션
            await asyncio.sleep(0.1)  # 시뮬레이션용 짧은 대기

            # 성과 계산 (난이도와 랜덤 요소 고려)
            base_success_chance = max(0.3, 1.0 - (mission.difficulty - 1) * 0.15)
            random_factor = random.uniform(0.8, 1.2)
            performance_score = min(base_success_chance * random_factor, 1.0)

            success = performance_score > 0.6
            status = TrainingStatus.COMPLETED if success else TrainingStatus.FAILED

            # 개선 메트릭 생성
            improvement_metrics = self._calculate_improvement_metrics(mission, performance_score)

            # 피드백 생성
            feedback = self._generate_training_feedback(mission, success, performance_score)

            end_time = current_timestamp()
            duration_minutes = (end_time - start_time) / 60

            return TrainingResult(
                mission_id=mission.mission_id,
                status=status,
                start_time=start_time,
                end_time=end_time,
                success=success,
                performance_score=performance_score,
                improvement_metrics=improvement_metrics,
                feedback=feedback,
                duration_minutes=duration_minutes
            )

        except Exception as e:
            end_time = current_timestamp()
            duration_minutes = (end_time - start_time) / 60

            return TrainingResult(
                mission_id=mission.mission_id,
                status=TrainingStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                success=False,
                performance_score=0.0,
                improvement_metrics={},
                feedback=f"훈련 중 오류 발생: {str(e)}",
                duration_minutes=duration_minutes
            )

    def _generate_success_criteria(self, training_type: TrainingType, weakness: WeaknessArea) -> Dict[str, Any]:
        """훈련 유형별 성공 기준 생성"""
        base_criteria = {
            "minimum_performance_score": 0.7,
            "target_improvement": weakness.target_score - weakness.current_score
        }

        type_specific_criteria = {
            TrainingType.TACTIC_PRACTICE: {
                "tactic_accuracy": 0.8,
                "application_consistency": 0.75
            },
            TrainingType.COMPLEX_PROBLEM: {
                "problem_completion_rate": 0.8,
                "solution_quality": 0.7
            },
            TrainingType.REASONING_DRILL: {
                "logical_consistency": 0.85,
                "step_clarity": 0.8
            },
            TrainingType.SPEED_TRAINING: {
                "response_time_improvement": 0.2,
                "accuracy_maintenance": 0.75
            },
            TrainingType.ADAPTATION_EXERCISE: {
                "adaptation_speed": 0.8,
                "flexibility_score": 0.75
            }
        }

        base_criteria.update(type_specific_criteria.get(training_type, {}))
        return base_criteria

    def _calculate_improvement_metrics(self, mission: TrainingMission, performance_score: float) -> Dict[str, float]:
        """개선 메트릭 계산"""
        # 훈련 유형별 개선 메트릭
        base_improvement = performance_score * 0.1  # 기본 10% 개선

        metrics = {
            "overall_improvement": base_improvement,
            "skill_specific_improvement": base_improvement * 1.2,
            "confidence_boost": performance_score * 0.05
        }

        # 훈련 유형별 특화 메트릭
        if mission.training_type == TrainingType.TACTIC_PRACTICE:
            metrics["tactic_proficiency_gain"] = base_improvement * 1.5
        elif mission.training_type == TrainingType.REASONING_DRILL:
            metrics["reasoning_quality_gain"] = base_improvement * 1.3
        elif mission.training_type == TrainingType.SPEED_TRAINING:
            metrics["processing_speed_gain"] = base_improvement * 1.4
        elif mission.training_type == TrainingType.ADAPTATION_EXERCISE:
            metrics["adaptation_ability_gain"] = base_improvement * 1.2

        return metrics

    def _generate_training_feedback(self, mission: TrainingMission, success: bool, performance_score: float) -> str:
        """훈련 피드백 생성"""
        if success:
            if performance_score > 0.9:
                return f"우수한 성과입니다! {mission.training_type.value} 훈련에서 탁월한 결과를 보였습니다."
            elif performance_score > 0.8:
                return f"좋은 성과입니다. {mission.training_type.value} 영역에서 개선이 확인됩니다."
            else:
                return f"목표를 달성했습니다. {mission.training_type.value} 기술이 향상되었습니다."
        else:
            if performance_score > 0.5:
                return f"아쉽게 목표에 도달하지 못했지만, {mission.training_type.value} 영역에서 부분적 개선이 있었습니다."
            else:
                return f"{mission.training_type.value} 훈련에서 어려움을 겪었습니다. 기초 훈련을 강화해보세요."

    async def _evaluate_cycle_improvement(self, cycle: int, training_results: List[TrainingResult]) -> float:
        """사이클별 개선도 평가"""
        if not training_results:
            return 0.0

        # 최근 결과들의 평균 성과
        cycle_results = [r for r in training_results if r.status == TrainingStatus.COMPLETED]
        if not cycle_results:
            return 0.0

        avg_performance = statistics.mean(r.performance_score for r in cycle_results)
        return avg_performance

    async def _calculate_overall_improvement(self, training_results: List[TrainingResult]) -> float:
        """전체 개선도 계산"""
        if not training_results:
            return 0.0

        successful_results = [r for r in training_results if r.success]
        if not successful_results:
            return 0.0

        # 가중 평균으로 전체 개선도 계산
        total_improvement = 0.0
        total_weight = 0.0

        for result in successful_results:
            weight = result.performance_score
            improvement = result.improvement_metrics.get("overall_improvement", 0.0)
            total_improvement += improvement * weight
            total_weight += weight

        return total_improvement / total_weight if total_weight > 0 else 0.0

    def _generate_session_summary(self, completed: int, failed: int, skipped: int, improvement: float) -> str:
        """세션 요약 생성"""
        total = completed + failed + skipped
        success_rate = (completed / total * 100) if total > 0 else 0

        summary_parts = []
        summary_parts.append(f"총 {total}개 임무 중 {completed}개 완료")
        summary_parts.append(f"성공률: {success_rate:.1f}%")
        summary_parts.append(f"전체 개선도: {improvement:.3f}")

        if improvement > 0.1:
            summary_parts.append("상당한 개선이 확인되었습니다.")
        elif improvement > 0.05:
            summary_parts.append("적절한 개선이 있었습니다.")
        else:
            summary_parts.append("추가 훈련이 필요합니다.")

        return " | ".join(summary_parts)

    def _generate_weakness_description(self, weakness_type: WeaknessType, score: float) -> str:
        """약점 설명 생성"""
        descriptions = {
            WeaknessType.TACTIC_PROFICIENCY: f"전술 숙련도가 부족합니다 (현재: {score:.1f}점)",
            WeaknessType.PROBLEM_COMPLEXITY: f"복잡한 문제 해결 능력이 부족합니다 (현재: {score:.1f}점)",
            WeaknessType.REASONING_QUALITY: f"추론 품질을 개선할 필요가 있습니다 (현재: {score:.1f}점)",
            WeaknessType.LEARNING_EFFICIENCY: f"학습 효율성을 높여야 합니다 (현재: {score:.1f}점)",
            WeaknessType.ADAPTATION_SPEED: f"적응 속도가 느립니다 (현재: {score:.1f}점)"
        }
        return descriptions.get(weakness_type, f"개선이 필요한 영역입니다 (현재: {score:.1f}점)")

    def _create_current_learning_data(self) -> LearningData:
        """현재 상태의 가상 학습 데이터 생성 (실제 구현에서는 실제 데이터 사용)"""
        from .iis_calculator import create_sample_learning_data
        return create_sample_learning_data()

    async def _notify_callbacks(self, event_type: str, data: Dict[str, Any]) -> None:
        """콜백 함수들에 알림 전송"""
        for callback in self._training_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except Exception as e:
                # 콜백 오류는 무시하고 계속 진행
                pass


# Helper functions for testing
async def create_sample_training_session() -> TrainingSession:
    """샘플 훈련 세션 생성 (테스트용)"""
    trainer = AutonomousTrainer()

    # 샘플 약점 생성
    weakness = WeaknessArea(
        weakness_type=WeaknessType.TACTIC_PROFICIENCY,
        severity=0.6,
        description="전술 숙련도 부족",
        current_score=65.0,
        target_score=78.0,
        priority=Priority.HIGH,
        identified_at=current_timestamp()
    )

    # 훈련 실행
    config = TrainingConfig(max_cycles=3, max_session_duration_minutes=30)
    result = await trainer.execute_continuous_training(config)

    if result.is_success:
        return result.value
    else:
        raise result.error


def create_sample_weakness_area() -> WeaknessArea:
    """샘플 약점 영역 생성 (테스트용)"""
    return WeaknessArea(
        weakness_type=WeaknessType.REASONING_QUALITY,
        severity=0.7,
        description="추론 품질 개선 필요",
        current_score=58.0,
        target_score=70.0,
        priority=Priority.HIGH,
        identified_at=current_timestamp()
    )