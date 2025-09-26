"""
Attention Resource Allocator

주의 자원 할당자로, 제한된 주의 자원을 다양한 작업과 대상에
효율적으로 분배합니다.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Callable
from uuid import uuid4, UUID

from ...base import BaseCognitiveProcessor


class AllocationStrategy(Enum):
    """자원 할당 전략"""
    ROUND_ROBIN = auto()        # 순환 할당
    PRIORITY_BASED = auto()     # 우선순위 기반
    WEIGHTED_FAIR = auto()      # 가중 공정 할당
    DYNAMIC_ADAPTIVE = auto()   # 동적 적응 할당
    EMERGENCY_OVERRIDE = auto() # 긴급 상황 우선 할당


@dataclass
class ResourcePool:
    """자원 풀 정의"""
    total_capacity: float = 100.0       # 전체 용량
    available_capacity: float = 100.0   # 사용 가능한 용량
    reserved_capacity: float = 0.0      # 예약된 용량
    emergency_reserve: float = 10.0     # 긴급 상황용 예비 자원
    fragmentation_threshold: float = 5.0 # 파편화 임계값


@dataclass
class ResourceRequest:
    """자원 요청"""
    id: UUID = field(default_factory=uuid4)
    requester_id: str = ""
    amount: float = 0.0                 # 요청 자원량
    priority: int = 5                   # 우선순위 (1=highest, 10=lowest)
    duration_estimate_ms: int = 1000    # 예상 사용 시간
    minimum_required: float = 0.0       # 최소 필요량
    can_be_preempted: bool = True       # 선점 가능 여부
    deadline_ms: Optional[int] = None   # 마감 시간 (ms)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class ResourceAllocation:
    """자원 할당 결과"""
    request_id: UUID
    allocated_amount: float = 0.0
    allocation_ratio: float = 0.0       # 요청 대비 할당 비율
    start_time: float = field(default_factory=time.time)
    estimated_end_time: Optional[float] = None
    actual_end_time: Optional[float] = None
    was_preempted: bool = False
    efficiency_score: float = 1.0


class AttentionResourceAllocator(BaseCognitiveProcessor):
    """
    주의 자원 할당자

    제한된 주의 자원을 여러 요청자에게 효율적으로 분배하고,
    동적으로 재할당을 수행합니다.
    """

    def __init__(self, total_capacity: float = 100.0,
                 strategy: AllocationStrategy = AllocationStrategy.PRIORITY_BASED):
        super().__init__()

        # 자원 풀 초기화
        self.resource_pool = ResourcePool(total_capacity=total_capacity)
        self.allocation_strategy = strategy

        # 할당 상태 관리
        self._active_allocations: Dict[UUID, ResourceAllocation] = {}
        self._pending_requests: List[ResourceRequest] = []
        self._allocation_history: List[ResourceAllocation] = []

        # 성능 메트릭
        self._total_requests = 0
        self._successful_allocations = 0
        self._preemption_count = 0
        self._fragmentation_events = 0

        # 적응적 제어
        self._allocation_weights: Dict[str, float] = {}
        self._performance_history: List[float] = []
        self._last_optimization_time = time.time()

    async def initialize(self) -> bool:
        """자원 할당자 초기화"""
        try:
            self.logger.info("Initializing Attention Resource Allocator...")

            # 백그라운드 프로세스 시작
            asyncio.create_task(self._monitor_allocations())
            asyncio.create_task(self._optimize_allocation_strategy())
            asyncio.create_task(self._manage_fragmentation())

            self.logger.info("Attention Resource Allocator initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Resource Allocator: {e}")
            return False

    async def request_resources(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
        """
        자원 요청 처리

        Args:
            request: 자원 요청 객체

        Returns:
            할당 결과 (할당 실패 시 None)
        """
        try:
            self._total_requests += 1

            # 즉시 할당 가능한지 확인
            if self._can_allocate_immediately(request):
                allocation = await self._allocate_resources(request)
                if allocation:
                    self._successful_allocations += 1
                    return allocation

            # 대기열에 추가
            self._pending_requests.append(request)
            self._sort_pending_requests()

            # 선점을 통한 할당 시도
            if request.priority <= 3:  # 높은 우선순위 요청
                await self._attempt_preemption(request)

            return None

        except Exception as e:
            self.logger.error(f"Error processing resource request {request.id}: {e}")
            return None

    async def release_resources(self, allocation_id: UUID) -> bool:
        """
        자원 해제

        Args:
            allocation_id: 해제할 할당 ID

        Returns:
            해제 성공 여부
        """
        try:
            if allocation_id not in self._active_allocations:
                return False

            allocation = self._active_allocations[allocation_id]
            allocation.actual_end_time = time.time()

            # 자원 반환
            self.resource_pool.available_capacity += allocation.allocated_amount

            # 할당 완료 처리
            del self._active_allocations[allocation_id]
            self._allocation_history.append(allocation)

            # 대기 중인 요청 처리
            await self._process_pending_requests()

            self.logger.debug(f"Released resources for allocation {allocation_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error releasing resources for {allocation_id}: {e}")
            return False

    async def adjust_allocation(self, allocation_id: UUID,
                              new_amount: float) -> bool:
        """
        기존 할당량 조정

        Args:
            allocation_id: 조정할 할당 ID
            new_amount: 새로운 할당량

        Returns:
            조정 성공 여부
        """
        try:
            if allocation_id not in self._active_allocations:
                return False

            allocation = self._active_allocations[allocation_id]
            current_amount = allocation.allocated_amount
            difference = new_amount - current_amount

            # 자원 가용성 확인
            if difference > 0 and difference > self.resource_pool.available_capacity:
                return False

            # 할당량 조정
            allocation.allocated_amount = new_amount
            self.resource_pool.available_capacity -= difference

            self.logger.debug(f"Adjusted allocation {allocation_id} by {difference}")
            return True

        except Exception as e:
            self.logger.error(f"Error adjusting allocation {allocation_id}: {e}")
            return False

    async def get_allocation_status(self) -> Dict[str, Any]:
        """현재 할당 상태 조회"""
        utilization_rate = (
            (self.resource_pool.total_capacity - self.resource_pool.available_capacity) /
            self.resource_pool.total_capacity * 100
        )

        return {
            "total_capacity": self.resource_pool.total_capacity,
            "available_capacity": self.resource_pool.available_capacity,
            "utilization_rate_percent": utilization_rate,
            "active_allocations": len(self._active_allocations),
            "pending_requests": len(self._pending_requests),
            "allocation_strategy": self.allocation_strategy.name,
            "success_rate_percent": (
                self._successful_allocations / max(1, self._total_requests) * 100
            ),
            "preemption_count": self._preemption_count,
            "fragmentation_events": self._fragmentation_events
        }

    async def optimize_strategy(self) -> None:
        """할당 전략 최적화"""
        try:
            # 성능 기반 전략 조정
            current_performance = await self._calculate_performance_score()
            self._performance_history.append(current_performance)

            # 최근 성능이 저하되면 전략 변경
            if len(self._performance_history) >= 10:
                recent_avg = sum(self._performance_history[-5:]) / 5
                older_avg = sum(self._performance_history[-10:-5]) / 5

                if recent_avg < older_avg * 0.9:  # 10% 이상 성능 저하
                    await self._switch_allocation_strategy()

            self.logger.debug(f"Current allocation performance: {current_performance:.3f}")

        except Exception as e:
            self.logger.error(f"Error optimizing allocation strategy: {e}")

    def _can_allocate_immediately(self, request: ResourceRequest) -> bool:
        """즉시 할당 가능 여부 확인"""
        required_amount = max(request.amount, request.minimum_required)
        return self.resource_pool.available_capacity >= required_amount

    async def _allocate_resources(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
        """실제 자원 할당 수행"""
        try:
            # 할당량 결정
            allocated_amount = min(request.amount, self.resource_pool.available_capacity)

            # 최소 요구량 미충족 시 실패
            if allocated_amount < request.minimum_required:
                return None

            # 자원 할당
            allocation = ResourceAllocation(
                request_id=request.id,
                allocated_amount=allocated_amount,
                allocation_ratio=allocated_amount / request.amount,
                estimated_end_time=(
                    time.time() + request.duration_estimate_ms / 1000.0
                    if request.duration_estimate_ms else None
                )
            )

            # 자원 차감
            self.resource_pool.available_capacity -= allocated_amount

            # 활성 할당에 추가
            self._active_allocations[allocation.request_id] = allocation

            self.logger.debug(f"Allocated {allocated_amount} resources to request {request.id}")
            return allocation

        except Exception as e:
            self.logger.error(f"Error allocating resources for request {request.id}: {e}")
            return None

    async def _attempt_preemption(self, urgent_request: ResourceRequest) -> bool:
        """선점을 통한 자원 확보 시도"""
        try:
            # 선점 가능한 할당 찾기
            preemptable_allocations = [
                alloc for alloc in self._active_allocations.values()
                if self._is_preemptable(alloc, urgent_request)
            ]

            if not preemptable_allocations:
                return False

            # 우선순위가 낮은 할당을 선점
            target_allocation = min(
                preemptable_allocations,
                key=lambda a: self._get_allocation_priority(a)
            )

            # 선점 실행
            freed_amount = target_allocation.allocated_amount
            target_allocation.was_preempted = True
            target_allocation.actual_end_time = time.time()

            # 자원 회수
            del self._active_allocations[target_allocation.request_id]
            self.resource_pool.available_capacity += freed_amount

            self._preemption_count += 1
            self.logger.info(f"Preempted allocation {target_allocation.request_id} "
                           f"to free {freed_amount} resources")

            return True

        except Exception as e:
            self.logger.error(f"Error during preemption attempt: {e}")
            return False

    def _is_preemptable(self, allocation: ResourceAllocation,
                       urgent_request: ResourceRequest) -> bool:
        """할당이 선점 가능한지 확인"""
        # 선점 불가능하도록 명시된 경우
        request_id = allocation.request_id
        for req in self._pending_requests:
            if req.id == request_id and not req.can_be_preempted:
                return False

        # 우선순위 비교 (낮은 숫자가 높은 우선순위)
        current_priority = self._get_allocation_priority(allocation)
        return urgent_request.priority < current_priority

    def _get_allocation_priority(self, allocation: ResourceAllocation) -> int:
        """할당의 우선순위 추정"""
        # 실제 구현에서는 원본 요청 정보를 저장해야 함
        # 여기서는 할당 시간 기반으로 추정
        allocation_age = time.time() - allocation.start_time

        # 오래된 할당일수록 우선순위가 낮다고 가정
        return int(5 + allocation_age / 60)  # 1분당 +1 우선순위

    def _sort_pending_requests(self) -> None:
        """대기 요청을 우선순위에 따라 정렬"""
        if self.allocation_strategy == AllocationStrategy.PRIORITY_BASED:
            self._pending_requests.sort(key=lambda req: (req.priority, req.created_at))
        elif self.allocation_strategy == AllocationStrategy.ROUND_ROBIN:
            # Round Robin의 경우 생성 시간 순서 유지
            self._pending_requests.sort(key=lambda req: req.created_at)

    async def _process_pending_requests(self) -> None:
        """대기 중인 요청 처리"""
        processed_requests = []

        for request in self._pending_requests:
            allocation = await self._allocate_resources(request)
            if allocation:
                processed_requests.append(request)
                self._successful_allocations += 1

        # 처리된 요청 제거
        for request in processed_requests:
            self._pending_requests.remove(request)

    async def _monitor_allocations(self) -> None:
        """할당 모니터링 백그라운드 프로세스"""
        while True:
            try:
                current_time = time.time()
                expired_allocations = []

                # 만료된 할당 확인
                for allocation_id, allocation in self._active_allocations.items():
                    if (allocation.estimated_end_time and
                        current_time > allocation.estimated_end_time):
                        expired_allocations.append(allocation_id)

                # 만료된 할당 정리
                for allocation_id in expired_allocations:
                    await self.release_resources(allocation_id)

                await asyncio.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Error in allocation monitoring: {e}")
                await asyncio.sleep(5.0)

    async def _optimize_allocation_strategy(self) -> None:
        """할당 전략 최적화 백그라운드 프로세스"""
        while True:
            try:
                current_time = time.time()

                # 5분마다 최적화 수행
                if current_time - self._last_optimization_time > 300:
                    await self.optimize_strategy()
                    self._last_optimization_time = current_time

                await asyncio.sleep(30.0)

            except Exception as e:
                self.logger.error(f"Error in strategy optimization: {e}")
                await asyncio.sleep(60.0)

    async def _manage_fragmentation(self) -> None:
        """자원 파편화 관리"""
        while True:
            try:
                # 파편화 검출
                if await self._detect_fragmentation():
                    await self._defragment_resources()

                await asyncio.sleep(10.0)

            except Exception as e:
                self.logger.error(f"Error in fragmentation management: {e}")
                await asyncio.sleep(30.0)

    async def _detect_fragmentation(self) -> bool:
        """자원 파편화 검출"""
        # 사용 가능한 자원이 있지만 요청을 만족할 수 없는 상황
        if (self.resource_pool.available_capacity > self.resource_pool.fragmentation_threshold and
            self._pending_requests):

            largest_request = max(
                (req.minimum_required for req in self._pending_requests),
                default=0
            )

            return largest_request > self.resource_pool.available_capacity

        return False

    async def _defragment_resources(self) -> None:
        """자원 파편화 해결"""
        self._fragmentation_events += 1
        self.logger.info("Detected resource fragmentation, attempting defragmentation")

        # 작은 할당들을 일시적으로 재조정하여 공간 확보
        small_allocations = [
            alloc for alloc in self._active_allocations.values()
            if alloc.allocated_amount < 10.0  # 작은 할당 기준
        ]

        # 일부 작은 할당을 일시 중단하고 재할당
        for allocation in small_allocations[:2]:  # 최대 2개까지만
            if allocation.was_preempted:
                continue

            await self.release_resources(allocation.request_id)

        # 대기 중인 요청 다시 처리
        await self._process_pending_requests()

    async def _calculate_performance_score(self) -> float:
        """성능 점수 계산"""
        if self._total_requests == 0:
            return 1.0

        success_rate = self._successful_allocations / self._total_requests
        utilization = (
            (self.resource_pool.total_capacity - self.resource_pool.available_capacity) /
            self.resource_pool.total_capacity
        )

        # 성공률과 자원 활용률의 조화 평균
        if success_rate == 0 or utilization == 0:
            return 0.0

        return 2 * (success_rate * utilization) / (success_rate + utilization)

    async def _switch_allocation_strategy(self) -> None:
        """할당 전략 변경"""
        strategies = list(AllocationStrategy)
        current_index = strategies.index(self.allocation_strategy)

        # 다음 전략으로 순환
        next_strategy = strategies[(current_index + 1) % len(strategies)]
        self.allocation_strategy = next_strategy

        self.logger.info(f"Switched allocation strategy to {next_strategy.name}")


async def create_resource_allocator(
    total_capacity: float = 100.0,
    strategy: AllocationStrategy = AllocationStrategy.PRIORITY_BASED
) -> AttentionResourceAllocator:
    """AttentionResourceAllocator 인스턴스 생성 및 초기화"""
    allocator = AttentionResourceAllocator(total_capacity, strategy)
    await allocator.initialize()
    return allocator