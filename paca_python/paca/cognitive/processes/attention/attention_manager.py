"""
Attention Manager

주의 메커니즘의 중앙 관리자로, 전체 주의 자원을 관리하고
다양한 주의 작업들을 조율합니다.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Callable, Union
from uuid import uuid4, UUID

from ...base import BaseCognitiveProcessor


class AttentionState(Enum):
    """주의 상태 정의"""
    IDLE = auto()          # 유휴 상태
    FOCUSED = auto()       # 집중 상태
    DIVIDED = auto()       # 분산 주의 상태
    SWITCHING = auto()     # 주의 전환 상태
    OVERLOADED = auto()    # 과부하 상태


class AttentionPriority(Enum):
    """주의 우선순위"""
    CRITICAL = 1    # 긴급한 주의 필요
    HIGH = 2        # 높은 우선순위
    NORMAL = 3      # 일반 우선순위
    LOW = 4         # 낮은 우선순위
    BACKGROUND = 5  # 배경 처리


@dataclass
class AttentionConfig:
    """주의 시스템 설정"""
    max_concurrent_tasks: int = 5           # 최대 동시 처리 작업 수
    task_timeout_ms: int = 5000            # 작업 타임아웃 (ms)
    resource_limit: float = 100.0         # 주의 자원 한계
    switching_cost_ms: int = 50            # 주의 전환 비용 (ms)
    focus_decay_rate: float = 0.1          # 집중도 감소율 (초당)
    overload_threshold: float = 0.9        # 과부하 임계점
    enable_adaptive_allocation: bool = True # 적응적 자원 할당


@dataclass
class AttentionTask:
    """주의 작업 정의"""
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    priority: AttentionPriority = AttentionPriority.NORMAL
    resource_required: float = 10.0       # 필요한 주의 자원량
    duration_estimate_ms: int = 1000      # 예상 처리 시간
    context: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None   # 완료 콜백
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class AttentionResult:
    """주의 처리 결과"""
    task_id: UUID
    success: bool
    result_data: Any = None
    processing_time_ms: float = 0.0
    resource_used: float = 0.0
    error_message: Optional[str] = None
    quality_score: float = 1.0            # 처리 품질 점수


@dataclass
class AttentionMetrics:
    """주의 시스템 메트릭"""
    total_tasks_processed: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_processing_time_ms: float = 0.0
    current_resource_usage: float = 0.0
    peak_resource_usage: float = 0.0
    attention_switching_count: int = 0
    current_focus_level: float = 1.0
    overload_incidents: int = 0
    efficiency_score: float = 1.0


class AttentionManager(BaseCognitiveProcessor):
    """
    주의 메커니즘 중앙 관리자

    주의 자원을 효율적으로 관리하고, 여러 작업 간 주의를 적절히 분배합니다.
    인지 부하를 모니터링하고 성능을 최적화합니다.
    """

    def __init__(self, config: Optional[AttentionConfig] = None):
        super().__init__()
        self.config = config or AttentionConfig()

        # 주의 상태 관리
        self._state = AttentionState.IDLE
        self._current_tasks: Dict[UUID, AttentionTask] = {}
        self._task_queue: List[AttentionTask] = []
        self._resource_pool = self.config.resource_limit

        # 메트릭 추적
        self._metrics = AttentionMetrics()
        self._performance_history: List[float] = []

        # 이벤트 시스템
        self._event_handlers: Dict[str, List[Callable]] = {}

        # 적응적 제어
        self._adaptation_enabled = self.config.enable_adaptive_allocation
        self._last_adaptation_time = time.time()

    async def initialize(self) -> bool:
        """주의 시스템 초기화"""
        try:
            self.logger.info("Initializing Attention Manager...")

            # 백그라운드 작업 시작
            asyncio.create_task(self._monitor_tasks())
            asyncio.create_task(self._resource_manager())
            asyncio.create_task(self._adaptive_controller())

            self._state = AttentionState.IDLE
            self.logger.info("Attention Manager initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Attention Manager: {e}")
            return False

    async def shutdown(self) -> bool:
        """주의 시스템 종료"""
        try:
            self.logger.info("Shutting down Attention Manager...")

            # 진행 중인 작업 완료 대기
            await self._complete_pending_tasks()

            self._state = AttentionState.IDLE
            self.logger.info("Attention Manager shut down successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error during Attention Manager shutdown: {e}")
            return False

    async def allocate_attention(self, task: AttentionTask) -> bool:
        """
        주의 자원을 작업에 할당

        Args:
            task: 처리할 주의 작업

        Returns:
            할당 성공 여부
        """
        try:
            # 자원 가용성 확인
            if not self._can_allocate_resources(task.resource_required):
                # 대기열에 추가
                self._task_queue.append(task)
                await self._emit_event("task_queued", {"task_id": task.id})
                return False

            # 자원 할당
            await self._allocate_resources(task)

            # 작업 시작
            task.started_at = time.time()
            self._current_tasks[task.id] = task

            # 상태 업데이트
            self._update_attention_state()

            await self._emit_event("attention_allocated", {
                "task_id": task.id,
                "resource_allocated": task.resource_required
            })

            return True

        except Exception as e:
            self.logger.error(f"Error allocating attention for task {task.id}: {e}")
            return False

    async def complete_task(self, task_id: UUID, result: Any = None,
                          success: bool = True) -> Optional[AttentionResult]:
        """
        작업 완료 처리

        Args:
            task_id: 완료된 작업 ID
            result: 작업 결과
            success: 성공 여부

        Returns:
            처리 결과 객체
        """
        try:
            if task_id not in self._current_tasks:
                return None

            task = self._current_tasks[task_id]
            task.completed_at = time.time()

            # 결과 생성
            processing_time = (task.completed_at - task.started_at) * 1000
            attention_result = AttentionResult(
                task_id=task_id,
                success=success,
                result_data=result,
                processing_time_ms=processing_time,
                resource_used=task.resource_required
            )

            # 자원 해제
            await self._release_resources(task)

            # 작업 제거
            del self._current_tasks[task_id]

            # 메트릭 업데이트
            self._update_metrics(attention_result)

            # 대기 중인 작업 처리
            await self._process_queued_tasks()

            # 상태 업데이트
            self._update_attention_state()

            await self._emit_event("task_completed", {
                "task_id": task_id,
                "success": success,
                "processing_time_ms": processing_time
            })

            return attention_result

        except Exception as e:
            self.logger.error(f"Error completing task {task_id}: {e}")
            return None

    async def get_attention_status(self) -> Dict[str, Any]:
        """현재 주의 상태 조회"""
        return {
            "state": self._state.name,
            "active_tasks": len(self._current_tasks),
            "queued_tasks": len(self._task_queue),
            "resource_usage_percent": (
                (self.config.resource_limit - self._resource_pool) /
                self.config.resource_limit * 100
            ),
            "current_focus_level": self._metrics.current_focus_level,
            "efficiency_score": self._metrics.efficiency_score
        }

    async def get_metrics(self) -> AttentionMetrics:
        """주의 시스템 메트릭 반환"""
        return self._metrics

    def _can_allocate_resources(self, required: float) -> bool:
        """자원 할당 가능 여부 확인"""
        return self._resource_pool >= required

    async def _allocate_resources(self, task: AttentionTask) -> None:
        """자원 할당 실행"""
        self._resource_pool -= task.resource_required
        self._metrics.current_resource_usage = (
            self.config.resource_limit - self._resource_pool
        )

    async def _release_resources(self, task: AttentionTask) -> None:
        """자원 해제 실행"""
        self._resource_pool += task.resource_required
        self._metrics.current_resource_usage = (
            self.config.resource_limit - self._resource_pool
        )

    def _update_attention_state(self) -> None:
        """주의 상태 업데이트"""
        active_count = len(self._current_tasks)
        resource_usage_ratio = (
            self._metrics.current_resource_usage / self.config.resource_limit
        )

        if resource_usage_ratio > self.config.overload_threshold:
            self._state = AttentionState.OVERLOADED
        elif active_count == 0:
            self._state = AttentionState.IDLE
        elif active_count == 1:
            self._state = AttentionState.FOCUSED
        else:
            self._state = AttentionState.DIVIDED

    def _update_metrics(self, result: AttentionResult) -> None:
        """메트릭 업데이트"""
        self._metrics.total_tasks_processed += 1

        if result.success:
            self._metrics.successful_tasks += 1
        else:
            self._metrics.failed_tasks += 1

        # 평균 처리 시간 업데이트
        total_time = (
            self._metrics.average_processing_time_ms *
            (self._metrics.total_tasks_processed - 1) +
            result.processing_time_ms
        )
        self._metrics.average_processing_time_ms = (
            total_time / self._metrics.total_tasks_processed
        )

        # 효율성 점수 계산
        success_rate = (
            self._metrics.successful_tasks / self._metrics.total_tasks_processed
        )
        self._metrics.efficiency_score = min(success_rate * result.quality_score, 1.0)

    async def _process_queued_tasks(self) -> None:
        """대기 중인 작업 처리"""
        processed = []

        for task in self._task_queue:
            if self._can_allocate_resources(task.resource_required):
                await self.allocate_attention(task)
                processed.append(task)
            else:
                break  # 우선순위 순서로 정렬된 상태에서 중단

        # 처리된 작업들을 대기열에서 제거
        for task in processed:
            self._task_queue.remove(task)

    async def _monitor_tasks(self) -> None:
        """작업 모니터링 백그라운드 프로세스"""
        while True:
            try:
                current_time = time.time()
                timeout_tasks = []

                # 타임아웃된 작업 확인
                for task_id, task in self._current_tasks.items():
                    if task.started_at:
                        elapsed_ms = (current_time - task.started_at) * 1000
                        if elapsed_ms > self.config.task_timeout_ms:
                            timeout_tasks.append(task_id)

                # 타임아웃된 작업 처리
                for task_id in timeout_tasks:
                    await self.complete_task(task_id, success=False)
                    self.logger.warning(f"Task {task_id} timed out")

                await asyncio.sleep(0.1)  # 100ms 간격으로 모니터링

            except Exception as e:
                self.logger.error(f"Error in task monitoring: {e}")
                await asyncio.sleep(1.0)

    async def _resource_manager(self) -> None:
        """자원 관리 백그라운드 프로세스"""
        while True:
            try:
                # 자원 사용량 체크
                usage_ratio = (
                    self._metrics.current_resource_usage / self.config.resource_limit
                )

                if usage_ratio > self.config.overload_threshold:
                    self._metrics.overload_incidents += 1
                    await self._handle_overload()

                # 집중도 감소 적용
                if self._metrics.current_focus_level > 0:
                    decay = self.config.focus_decay_rate / 10  # 100ms당 감소
                    self._metrics.current_focus_level = max(
                        0, self._metrics.current_focus_level - decay
                    )

                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error in resource management: {e}")
                await asyncio.sleep(1.0)

    async def _adaptive_controller(self) -> None:
        """적응적 제어 백그라운드 프로세스"""
        if not self._adaptation_enabled:
            return

        while True:
            try:
                current_time = time.time()

                # 5초마다 적응적 조정
                if current_time - self._last_adaptation_time > 5.0:
                    await self._adapt_parameters()
                    self._last_adaptation_time = current_time

                await asyncio.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Error in adaptive control: {e}")
                await asyncio.sleep(5.0)

    async def _handle_overload(self) -> None:
        """과부하 상황 처리"""
        # 낮은 우선순위 작업 일시 중단
        low_priority_tasks = [
            task for task in self._current_tasks.values()
            if task.priority in [AttentionPriority.LOW, AttentionPriority.BACKGROUND]
        ]

        for task in low_priority_tasks[:2]:  # 최대 2개까지만 중단
            await self.complete_task(task.id, success=False)
            self.logger.info(f"Suspended low priority task {task.id} due to overload")

    async def _adapt_parameters(self) -> None:
        """시스템 파라미터 적응적 조정"""
        # 성능 기반 자동 조정 로직
        if self._metrics.efficiency_score < 0.7:
            # 효율성이 낮으면 동시 작업 수 감소
            self.config.max_concurrent_tasks = max(2, self.config.max_concurrent_tasks - 1)
        elif self._metrics.efficiency_score > 0.9:
            # 효율성이 높으면 동시 작업 수 증가
            self.config.max_concurrent_tasks = min(10, self.config.max_concurrent_tasks + 1)

    async def _complete_pending_tasks(self) -> None:
        """진행 중인 모든 작업 완료 대기"""
        while self._current_tasks:
            await asyncio.sleep(0.1)

    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """이벤트 발생"""
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    await handler(data)
                except Exception as e:
                    self.logger.error(f"Error in event handler for {event_type}: {e}")

    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """이벤트 핸들러 추가"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)


async def create_attention_manager(config: Optional[AttentionConfig] = None) -> AttentionManager:
    """AttentionManager 인스턴스 생성 및 초기화"""
    manager = AttentionManager(config)
    await manager.initialize()
    return manager