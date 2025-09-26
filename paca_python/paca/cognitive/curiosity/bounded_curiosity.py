"""
Bounded Curiosity System
제한된 호기심 시스템 - 궁극적 사명에 부합하는 경우에만 자율적 탐구 수행
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import logging
import time
from datetime import datetime, timedelta

# 조건부 임포트
try:
    from ..core.types.base import (
        ID, Timestamp, Result, current_timestamp, generate_id, create_success, create_failure
    )
    from .mission_aligner import MissionAligner, AlignmentCheck, AlignmentScore
except ImportError:
    from paca.core.types.base import (
        ID, Timestamp, Result, current_timestamp, generate_id, create_success, create_failure
    )
    from paca.cognitive.curiosity.mission_aligner import MissionAligner, AlignmentCheck, AlignmentScore


class CuriosityLevel(Enum):
    """호기심 수준"""
    DORMANT = "dormant"           # 비활성
    LOW = "low"                   # 낮음
    MODERATE = "moderate"         # 보통
    HIGH = "high"                 # 높음
    INTENSE = "intense"           # 강함


class ExplorationStatus(Enum):
    """탐구 상태"""
    PENDING = "pending"           # 대기 중
    APPROVED = "approved"         # 승인됨
    REJECTED = "rejected"         # 거부됨
    ACTIVE = "active"             # 진행 중
    COMPLETED = "completed"       # 완료됨
    SUSPENDED = "suspended"       # 일시 중단
    FAILED = "failed"             # 실패


class ResourceType(Enum):
    """자원 유형"""
    CPU_TIME = "cpu_time"         # CPU 시간
    MEMORY = "memory"             # 메모리
    NETWORK = "network"           # 네트워크
    STORAGE = "storage"           # 저장소
    API_CALLS = "api_calls"       # API 호출
    USER_ATTENTION = "user_attention"  # 사용자 관심


@dataclass
class CuriosityBudget:
    """호기심 예산"""
    budget_id: ID
    resource_type: ResourceType
    total_allocation: float
    consumed: float = 0.0
    reserved: float = 0.0
    reset_interval: int = 3600  # seconds
    last_reset: Timestamp = field(default_factory=current_timestamp)

    @property
    def available(self) -> float:
        """사용 가능한 예산"""
        return max(0.0, self.total_allocation - self.consumed - self.reserved)

    @property
    def utilization_rate(self) -> float:
        """사용률"""
        if self.total_allocation == 0:
            return 0.0
        return (self.consumed + self.reserved) / self.total_allocation

    def can_allocate(self, amount: float) -> bool:
        """할당 가능 여부"""
        return self.available >= amount

    def allocate(self, amount: float) -> bool:
        """예산 할당"""
        if self.can_allocate(amount):
            self.reserved += amount
            return True
        return False

    def consume(self, amount: float) -> None:
        """예산 소비"""
        actual_consumption = min(amount, self.reserved)
        self.reserved -= actual_consumption
        self.consumed += actual_consumption

    def release_reservation(self, amount: float) -> None:
        """예약 해제"""
        self.reserved = max(0.0, self.reserved - amount)

    def reset_if_needed(self) -> bool:
        """필요시 예산 리셋"""
        if current_timestamp() - self.last_reset >= self.reset_interval:
            self.consumed = 0.0
            self.reserved = 0.0
            self.last_reset = current_timestamp()
            return True
        return False


@dataclass
class ExplorationRequest:
    """탐구 요청"""
    request_id: ID
    trigger_reason: str
    exploration_objective: str
    predicted_value: float  # 0-1
    complexity_estimate: float  # 0-1
    resource_requirements: Dict[ResourceType, float]
    time_estimate: float  # seconds
    priority_score: float  # 0-1
    context_data: Dict[str, Any] = field(default_factory=dict)
    created_at: Timestamp = field(default_factory=current_timestamp)


@dataclass
class ExplorationExecution:
    """탐구 실행"""
    execution_id: ID
    request: ExplorationRequest
    alignment_check: AlignmentCheck
    status: ExplorationStatus
    allocated_resources: Dict[ResourceType, float]
    actual_consumption: Dict[ResourceType, float] = field(default_factory=dict)
    start_time: Optional[Timestamp] = None
    end_time: Optional[Timestamp] = None
    results: Dict[str, Any] = field(default_factory=dict)
    lessons_learned: List[str] = field(default_factory=list)
    success_rating: Optional[float] = None  # 0-1


@dataclass
class CuriosityInsight:
    """호기심으로 얻은 통찰"""
    insight_id: ID
    source_exploration: ID
    insight_type: str
    content: str
    confidence_level: float  # 0-1
    practical_value: float  # 0-1
    follow_up_suggestions: List[str]
    discovered_at: Timestamp = field(default_factory=current_timestamp)


class MissionAlignmentChecker:
    """사명 부합성 검증자"""

    def __init__(self, mission_aligner: MissionAligner):
        self.mission_aligner = mission_aligner
        self.alignment_cache: Dict[str, Tuple[AlignmentCheck, Timestamp]] = {}
        self.cache_ttl = 300  # 5분 캐시
        self.logger = logging.getLogger(__name__)

    async def check_mission_compatibility(self, exploration_request: ExplorationRequest) -> AlignmentCheck:
        """사명 호환성 검사"""
        try:
            # 캐시 확인
            cache_key = self._generate_cache_key(exploration_request)
            if cache_key in self.alignment_cache:
                cached_check, timestamp = self.alignment_cache[cache_key]
                if current_timestamp() - timestamp < self.cache_ttl:
                    return cached_check

            # 새로운 정렬 검사 수행
            alignment_check = await self.mission_aligner.check_exploration_alignment(
                exploration_objective=exploration_request.exploration_objective,
                exploration_context={
                    'trigger_reason': exploration_request.trigger_reason,
                    'predicted_value': exploration_request.predicted_value,
                    'complexity': exploration_request.complexity_estimate,
                    'priority': exploration_request.priority_score,
                    'context': exploration_request.context_data
                }
            )

            # 캐시에 저장
            self.alignment_cache[cache_key] = (alignment_check, current_timestamp())

            return alignment_check

        except Exception as e:
            self.logger.error(f"사명 호환성 검사 실패: {e}")
            # 기본 거부 응답
            return AlignmentCheck(
                check_id=generate_id("check_error_"),
                exploration_objective=exploration_request.exploration_objective,
                mission_references=[],
                alignment_score=0.0,
                alignment_category=AlignmentScore.STRONG_MISALIGNMENT,
                recommendations=["호환성 검사 실패로 인한 거부"],
                approval_required=True
            )

    def _generate_cache_key(self, request: ExplorationRequest) -> str:
        """캐시 키 생성"""
        # 요청의 핵심 요소를 해시하여 캐시 키 생성
        key_components = [
            request.exploration_objective,
            request.trigger_reason,
            str(request.predicted_value),
            str(request.complexity_estimate)
        ]
        return "_".join(key_components)

    def clear_cache(self) -> None:
        """캐시 클리어"""
        self.alignment_cache.clear()


class CuriosityBudgetManager:
    """호기심 자원 관리"""

    def __init__(self, initial_budgets: Optional[Dict[ResourceType, float]] = None):
        self.budgets: Dict[ResourceType, CuriosityBudget] = {}
        self.usage_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

        # 기본 예산 설정
        default_budgets = {
            ResourceType.CPU_TIME: 300.0,      # 5분
            ResourceType.MEMORY: 1024.0,       # 1GB
            ResourceType.NETWORK: 100.0,       # 100 요청
            ResourceType.STORAGE: 100.0,       # 100MB
            ResourceType.API_CALLS: 50.0,      # 50 호출
            ResourceType.USER_ATTENTION: 5.0   # 5 알림
        }

        budgets_to_use = initial_budgets if initial_budgets else default_budgets

        for resource_type, allocation in budgets_to_use.items():
            self.budgets[resource_type] = CuriosityBudget(
                budget_id=generate_id("budget_"),
                resource_type=resource_type,
                total_allocation=allocation
            )

    async def check_resource_availability(self, requirements: Dict[ResourceType, float]) -> Tuple[bool, Dict[ResourceType, str]]:
        """자원 가용성 확인"""
        availability_status = {}
        overall_available = True

        for resource_type, required_amount in requirements.items():
            if resource_type not in self.budgets:
                availability_status[resource_type] = f"알 수 없는 자원 유형: {resource_type}"
                overall_available = False
                continue

            budget = self.budgets[resource_type]
            budget.reset_if_needed()

            if budget.can_allocate(required_amount):
                availability_status[resource_type] = "사용 가능"
            else:
                available = budget.available
                availability_status[resource_type] = f"부족 (필요: {required_amount}, 사용가능: {available})"
                overall_available = False

        return overall_available, availability_status

    async def reserve_resources(self, requirements: Dict[ResourceType, float]) -> Tuple[bool, Dict[ResourceType, bool]]:
        """자원 예약"""
        reservation_results = {}
        all_reserved = True

        for resource_type, amount in requirements.items():
            if resource_type in self.budgets:
                success = self.budgets[resource_type].allocate(amount)
                reservation_results[resource_type] = success
                if not success:
                    all_reserved = False
            else:
                reservation_results[resource_type] = False
                all_reserved = False

        # 일부 예약 실패시 모든 예약 롤백
        if not all_reserved:
            await self.release_reservations(requirements)

        return all_reserved, reservation_results

    async def consume_resources(self, consumption: Dict[ResourceType, float]) -> None:
        """자원 소비"""
        for resource_type, amount in consumption.items():
            if resource_type in self.budgets:
                self.budgets[resource_type].consume(amount)

        # 사용 기록
        self.usage_history.append({
            'consumption': consumption,
            'timestamp': current_timestamp()
        })

        # 기록 크기 제한
        if len(self.usage_history) > 1000:
            self.usage_history = self.usage_history[-1000:]

    async def release_reservations(self, reservations: Dict[ResourceType, float]) -> None:
        """예약 해제"""
        for resource_type, amount in reservations.items():
            if resource_type in self.budgets:
                self.budgets[resource_type].release_reservation(amount)

    def get_budget_status(self) -> Dict[ResourceType, Dict[str, Any]]:
        """예산 상태 조회"""
        status = {}
        for resource_type, budget in self.budgets.items():
            budget.reset_if_needed()
            status[resource_type] = {
                'total_allocation': budget.total_allocation,
                'consumed': budget.consumed,
                'reserved': budget.reserved,
                'available': budget.available,
                'utilization_rate': budget.utilization_rate,
                'last_reset': budget.last_reset
            }
        return status

    def adjust_budget(self, resource_type: ResourceType, new_allocation: float) -> bool:
        """예산 조정"""
        if resource_type in self.budgets:
            self.budgets[resource_type].total_allocation = new_allocation
            return True
        return False


class AutonomousInquiry:
    """자율적 탐구 실행"""

    def __init__(self, budget_manager: CuriosityBudgetManager):
        self.budget_manager = budget_manager
        self.active_explorations: Dict[ID, ExplorationExecution] = {}
        self.completed_explorations: List[ExplorationExecution] = []
        self.insights: List[CuriosityInsight] = []
        self.logger = logging.getLogger(__name__)

    async def execute_exploration(self, request: ExplorationRequest, alignment_check: AlignmentCheck) -> ExplorationExecution:
        """탐구 실행"""
        execution = ExplorationExecution(
            execution_id=generate_id("exec_"),
            request=request,
            alignment_check=alignment_check,
            status=ExplorationStatus.PENDING,
            allocated_resources={}
        )

        try:
            # 자원 예약
            reserved, reservation_results = await self.budget_manager.reserve_resources(
                request.resource_requirements
            )

            if not reserved:
                execution.status = ExplorationStatus.REJECTED
                execution.results['rejection_reason'] = "자원 부족"
                execution.results['reservation_results'] = reservation_results
                return execution

            execution.allocated_resources = request.resource_requirements.copy()
            execution.status = ExplorationStatus.APPROVED

            # 활성 탐구 목록에 추가
            self.active_explorations[execution.execution_id] = execution

            # 실제 탐구 실행
            await self._perform_exploration(execution)

            return execution

        except Exception as e:
            self.logger.error(f"탐구 실행 실패: {e}")
            execution.status = ExplorationStatus.FAILED
            execution.results['error'] = str(e)
            await self._cleanup_execution(execution)
            return execution

    async def _perform_exploration(self, execution: ExplorationExecution) -> None:
        """실제 탐구 수행"""
        execution.status = ExplorationStatus.ACTIVE
        execution.start_time = current_timestamp()

        try:
            # 탐구 시뮬레이션 (실제 구현에서는 구체적인 탐구 로직)
            request = execution.request
            exploration_time = min(request.time_estimate, 30.0)  # 최대 30초 제한

            # 비동기 탐구 시뮬레이션
            await asyncio.sleep(min(exploration_time, 2.0))  # 시뮬레이션용 짧은 시간

            # 결과 생성
            success_probability = max(0.3, request.predicted_value * 0.8)
            success = time.time() % 1.0 < success_probability

            if success:
                # 성공적인 탐구
                execution.status = ExplorationStatus.COMPLETED
                execution.success_rating = min(1.0, request.predicted_value + 0.2)

                # 통찰 생성
                insight = await self._generate_insight(execution)
                if insight:
                    self.insights.append(insight)

                execution.results.update({
                    'success': True,
                    'insight_generated': insight is not None,
                    'value_realized': execution.success_rating,
                    'exploration_summary': f"탐구 목표 '{request.exploration_objective}' 달성"
                })

                execution.lessons_learned.append("성공적인 탐구를 통해 새로운 통찰 획득")

            else:
                # 실패한 탐구
                execution.status = ExplorationStatus.FAILED
                execution.success_rating = 0.1

                execution.results.update({
                    'success': False,
                    'failure_reason': '예상된 결과를 얻지 못함',
                    'partial_results': '부분적 정보 수집됨'
                })

                execution.lessons_learned.append("탐구 실패를 통한 학습 - 접근 방법 개선 필요")

            # 실제 자원 소비 기록
            actual_consumption = {}
            for resource_type, allocated in execution.allocated_resources.items():
                # 성공률에 따른 소비량 조정
                consumption_factor = 1.0 if success else 0.7
                actual_consumption[resource_type] = allocated * consumption_factor

            execution.actual_consumption = actual_consumption
            await self.budget_manager.consume_resources(actual_consumption)

        except Exception as e:
            execution.status = ExplorationStatus.FAILED
            execution.results['error'] = str(e)
            execution.success_rating = 0.0

        finally:
            execution.end_time = current_timestamp()
            await self._cleanup_execution(execution)

    async def _generate_insight(self, execution: ExplorationExecution) -> Optional[CuriosityInsight]:
        """통찰 생성"""
        try:
            request = execution.request

            # 통찰 유형 결정
            insight_types = [
                "pattern_discovery",
                "knowledge_gap_identification",
                "optimization_opportunity",
                "relationship_understanding",
                "anomaly_detection"
            ]

            insight_type = insight_types[int(time.time()) % len(insight_types)]

            # 통찰 내용 생성
            insight_content = f"'{request.exploration_objective}' 탐구를 통해 발견: {insight_type}와 관련된 새로운 이해 획득"

            # 후속 제안 생성
            follow_up_suggestions = [
                f"{request.exploration_objective} 관련 추가 탐구",
                "발견된 패턴의 더 깊은 분석",
                "관련 영역으로 탐구 범위 확장"
            ]

            insight = CuriosityInsight(
                insight_id=generate_id("insight_"),
                source_exploration=execution.execution_id,
                insight_type=insight_type,
                content=insight_content,
                confidence_level=min(1.0, execution.success_rating + 0.1),
                practical_value=request.predicted_value,
                follow_up_suggestions=follow_up_suggestions
            )

            return insight

        except Exception as e:
            self.logger.error(f"통찰 생성 실패: {e}")
            return None

    async def _cleanup_execution(self, execution: ExplorationExecution) -> None:
        """실행 정리"""
        # 활성 목록에서 제거
        if execution.execution_id in self.active_explorations:
            del self.active_explorations[execution.execution_id]

        # 완료 목록에 추가
        self.completed_explorations.append(execution)

        # 완료 목록 크기 제한
        if len(self.completed_explorations) > 500:
            self.completed_explorations = self.completed_explorations[-500:]

        # 미사용 자원 해제
        unused_resources = {}
        for resource_type, allocated in execution.allocated_resources.items():
            consumed = execution.actual_consumption.get(resource_type, 0)
            unused = allocated - consumed
            if unused > 0:
                unused_resources[resource_type] = unused

        if unused_resources:
            await self.budget_manager.release_reservations(unused_resources)

    def get_active_explorations(self) -> List[ExplorationExecution]:
        """활성 탐구 목록"""
        return list(self.active_explorations.values())

    def get_exploration_statistics(self) -> Dict[str, Any]:
        """탐구 통계"""
        total_explorations = len(self.completed_explorations)
        if total_explorations == 0:
            return {'total_explorations': 0}

        successful = sum(1 for e in self.completed_explorations
                        if e.status == ExplorationStatus.COMPLETED)
        avg_success_rating = sum(e.success_rating or 0 for e in self.completed_explorations) / total_explorations
        total_insights = len(self.insights)

        return {
            'total_explorations': total_explorations,
            'successful_explorations': successful,
            'success_rate': successful / total_explorations,
            'average_success_rating': avg_success_rating,
            'total_insights_generated': total_insights,
            'active_explorations': len(self.active_explorations),
            'recent_completions': [
                {
                    'execution_id': e.execution_id,
                    'objective': e.request.exploration_objective,
                    'status': e.status.value,
                    'success_rating': e.success_rating,
                    'completion_time': e.end_time
                }
                for e in self.completed_explorations[-5:]
            ]
        }


class BoundedCuriositySystem:
    """제한된 호기심 메인 시스템"""

    def __init__(self, mission_aligner: MissionAligner,
                 initial_budgets: Optional[Dict[ResourceType, float]] = None):
        self.mission_aligner = mission_aligner
        self.alignment_checker = MissionAlignmentChecker(mission_aligner)
        self.budget_manager = CuriosityBudgetManager(initial_budgets)
        self.inquiry_executor = AutonomousInquiry(self.budget_manager)

        # 시스템 설정
        self.curiosity_enabled = True
        self.minimum_alignment_score = 0.5
        self.exploration_queue: List[ExplorationRequest] = []
        self.processing_active = False

        self.logger = logging.getLogger(__name__)

    async def submit_exploration_request(self, trigger_reason: str,
                                       exploration_objective: str,
                                       predicted_value: float = 0.5,
                                       complexity_estimate: float = 0.5,
                                       resource_requirements: Optional[Dict[ResourceType, float]] = None,
                                       time_estimate: float = 60.0,
                                       priority_score: float = 0.5,
                                       context_data: Optional[Dict[str, Any]] = None) -> Result[ID]:
        """탐구 요청 제출"""
        try:
            if not self.curiosity_enabled:
                return create_failure("호기심 시스템이 비활성화되어 있습니다")

            # 기본 자원 요구사항 설정
            if resource_requirements is None:
                resource_requirements = {
                    ResourceType.CPU_TIME: min(time_estimate, 30.0),
                    ResourceType.MEMORY: 50.0,
                    ResourceType.API_CALLS: 5.0
                }

            if context_data is None:
                context_data = {}

            # 탐구 요청 생성
            request = ExplorationRequest(
                request_id=generate_id("req_"),
                trigger_reason=trigger_reason,
                exploration_objective=exploration_objective,
                predicted_value=predicted_value,
                complexity_estimate=complexity_estimate,
                resource_requirements=resource_requirements,
                time_estimate=time_estimate,
                priority_score=priority_score,
                context_data=context_data
            )

            # 큐에 추가
            self.exploration_queue.append(request)

            # 처리 시작 (비동기)
            if not self.processing_active:
                asyncio.create_task(self._process_exploration_queue())

            self.logger.info(f"탐구 요청 제출: {exploration_objective} (ID: {request.request_id})")
            return create_success(request.request_id)

        except Exception as e:
            self.logger.error(f"탐구 요청 제출 실패: {e}")
            return create_failure(e)

    async def _process_exploration_queue(self) -> None:
        """탐구 큐 처리"""
        if self.processing_active:
            return

        self.processing_active = True

        try:
            while self.exploration_queue and self.curiosity_enabled:
                # 우선순위에 따라 정렬
                self.exploration_queue.sort(key=lambda r: r.priority_score, reverse=True)

                request = self.exploration_queue.pop(0)

                # 사명 부합성 검사
                alignment_check = await self.alignment_checker.check_mission_compatibility(request)

                if alignment_check.alignment_score < self.minimum_alignment_score:
                    self.logger.info(f"탐구 거부 - 사명 부합성 부족: {request.exploration_objective}")
                    continue

                if alignment_check.approval_required:
                    # 사용자 승인 필요 (현재 구현에서는 로그만)
                    self.logger.warning(f"탐구 승인 필요: {request.exploration_objective}")
                    # 실제 구현에서는 사용자에게 승인 요청
                    continue

                # 자원 가용성 확인
                available, status = await self.budget_manager.check_resource_availability(
                    request.resource_requirements
                )

                if not available:
                    self.logger.info(f"탐구 연기 - 자원 부족: {request.exploration_objective}")
                    # 잠시 대기 후 다시 큐에 추가
                    await asyncio.sleep(10)
                    self.exploration_queue.append(request)
                    continue

                # 탐구 실행
                execution = await self.inquiry_executor.execute_exploration(request, alignment_check)

                self.logger.info(f"탐구 완료: {request.exploration_objective}, 상태: {execution.status.value}")

                # 짧은 대기 (시스템 부하 방지)
                await asyncio.sleep(1)

        except Exception as e:
            self.logger.error(f"탐구 큐 처리 오류: {e}")

        finally:
            self.processing_active = False

    def enable_curiosity(self, enabled: bool = True) -> None:
        """호기심 활성화/비활성화"""
        self.curiosity_enabled = enabled
        self.logger.info(f"호기심 시스템 {'활성화' if enabled else '비활성화'}")

    def set_minimum_alignment_score(self, score: float) -> None:
        """최소 사명 부합 점수 설정"""
        self.minimum_alignment_score = max(0.0, min(1.0, score))
        self.logger.info(f"최소 사명 부합 점수 설정: {self.minimum_alignment_score}")

    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        return {
            'curiosity_enabled': self.curiosity_enabled,
            'minimum_alignment_score': self.minimum_alignment_score,
            'queue_size': len(self.exploration_queue),
            'processing_active': self.processing_active,
            'budget_status': self.budget_manager.get_budget_status(),
            'exploration_statistics': self.inquiry_executor.get_exploration_statistics(),
            'recent_insights': [
                {
                    'insight_id': insight.insight_id,
                    'type': insight.insight_type,
                    'content': insight.content[:100] + "..." if len(insight.content) > 100 else insight.content,
                    'confidence': insight.confidence_level,
                    'practical_value': insight.practical_value,
                    'discovered_at': insight.discovered_at
                }
                for insight in self.inquiry_executor.insights[-5:]
            ]
        }

    def adjust_resource_budget(self, resource_type: ResourceType, new_allocation: float) -> bool:
        """자원 예산 조정"""
        return self.budget_manager.adjust_budget(resource_type, new_allocation)

    def clear_exploration_queue(self) -> int:
        """탐구 큐 클리어"""
        cleared_count = len(self.exploration_queue)
        self.exploration_queue.clear()
        self.logger.info(f"탐구 큐 클리어: {cleared_count}개 요청 제거")
        return cleared_count


# 전역 인스턴스
_bounded_curiosity_instance = None


def get_bounded_curiosity_system(mission_aligner: Optional[MissionAligner] = None) -> BoundedCuriositySystem:
    """제한된 호기심 시스템 싱글톤 인스턴스 획득"""
    global _bounded_curiosity_instance
    if _bounded_curiosity_instance is None:
        if mission_aligner is None:
            mission_aligner = MissionAligner()
        _bounded_curiosity_instance = BoundedCuriositySystem(mission_aligner)
    return _bounded_curiosity_instance