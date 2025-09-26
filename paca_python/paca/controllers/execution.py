"""
Execution Control System
PACA v5 실행 제어 관리 - 작업 실행, 리소스 관리, 동시성 제어
"""

import asyncio
import time
import psutil
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Awaitable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from ..core.types.base import ID, Timestamp, create_id, current_timestamp
from ..core.utils.logger import create_logger
from ..core.errors.base import PacaError

class ExecutionState(Enum):
    """실행 상태"""
    PENDING = 'pending'         # 대기 중
    RUNNING = 'running'         # 실행 중
    PAUSED = 'paused'          # 일시 정지
    COMPLETED = 'completed'     # 완료
    FAILED = 'failed'          # 실패
    CANCELLED = 'cancelled'     # 취소
    TIMEOUT = 'timeout'        # 타임아웃

class ExecutionPriority(Enum):
    """실행 우선순위"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ExecutionConfig:
    """실행 설정"""
    max_concurrent_tasks: int = 10
    default_timeout: float = 30.0
    enable_resource_monitoring: bool = True
    enable_auto_scaling: bool = False
    memory_limit_mb: int = 1024
    cpu_limit_percent: float = 80.0
    task_retry_count: int = 3
    retry_delay: float = 1.0

@dataclass
class ExecutionContext:
    """실행 컨텍스트"""
    task_id: ID
    user_id: Optional[str]
    priority: ExecutionPriority
    timeout: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[ID] = field(default_factory=list)
    created_at: Timestamp = field(default_factory=current_timestamp)

@dataclass
class ExecutionResult:
    """실행 결과"""
    task_id: ID
    state: ExecutionState
    result_data: Any
    error_message: Optional[str] = None
    execution_time: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    retry_count: int = 0
    completed_at: Optional[Timestamp] = None

@dataclass
class TaskDefinition:
    """작업 정의"""
    task_id: ID
    name: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    context: ExecutionContext = field(default_factory=lambda: ExecutionContext(
        task_id=create_id(),
        user_id=None,
        priority=ExecutionPriority.NORMAL,
        timeout=30.0
    ))

class ExecutionPolicy:
    """실행 정책"""

    def __init__(self):
        self.max_memory_usage = 80.0  # 80%
        self.max_cpu_usage = 85.0     # 85%
        self.emergency_stop_threshold = 95.0  # 95%

    def should_execute(self, resource_usage: Dict[str, float]) -> bool:
        """실행 가능 여부 판단"""
        memory_usage = resource_usage.get('memory_percent', 0.0)
        cpu_usage = resource_usage.get('cpu_percent', 0.0)

        return (memory_usage < self.max_memory_usage and
                cpu_usage < self.max_cpu_usage)

    def should_emergency_stop(self, resource_usage: Dict[str, float]) -> bool:
        """비상 정지 여부 판단"""
        memory_usage = resource_usage.get('memory_percent', 0.0)
        cpu_usage = resource_usage.get('cpu_percent', 0.0)

        return (memory_usage > self.emergency_stop_threshold or
                cpu_usage > self.emergency_stop_threshold)

class ResourceManager:
    """리소스 관리자"""

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.logger = create_logger(__name__)

    def get_system_resources(self) -> Dict[str, float]:
        """시스템 리소스 사용량 조회"""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # 메모리 사용률
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # 디스크 사용률
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100

            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_used_mb': memory.used / (1024 * 1024),
                'memory_available_mb': memory.available / (1024 * 1024),
                'disk_percent': disk_percent
            }

        except Exception as e:
            self.logger.error(f"리소스 조회 실패: {e}")
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'memory_used_mb': 0.0,
                'memory_available_mb': 0.0,
                'disk_percent': 0.0
            }

    def is_resource_available(self, required_memory_mb: float = 0) -> bool:
        """리소스 사용 가능 여부 확인"""
        resources = self.get_system_resources()

        # 메모리 체크
        available_memory = resources['memory_available_mb']
        if required_memory_mb > 0 and available_memory < required_memory_mb:
            return False

        # CPU 체크
        if resources['cpu_percent'] > self.config.cpu_limit_percent:
            return False

        # 메모리 체크
        memory_limit = (self.config.memory_limit_mb / psutil.virtual_memory().total) * 100
        if resources['memory_percent'] > memory_limit:
            return False

        return True

    def estimate_task_resources(self, task: TaskDefinition) -> Dict[str, float]:
        """작업 리소스 사용량 추정"""
        # 간단한 추정 로직 (실제로는 더 정교할 수 있음)
        base_memory = 50.0  # MB
        base_cpu = 10.0     # %

        # 작업 유형에 따른 조정
        if hasattr(task.function, '__name__'):
            func_name = task.function.__name__
            if 'analyze' in func_name or 'process' in func_name:
                base_memory *= 2
                base_cpu *= 1.5
            elif 'generate' in func_name or 'create' in func_name:
                base_memory *= 1.5
                base_cpu *= 1.2

        return {
            'estimated_memory_mb': base_memory,
            'estimated_cpu_percent': base_cpu
        }

class TaskExecutor:
    """작업 실행기"""

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.logger = create_logger(__name__)
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_concurrent_tasks)

    async def execute_task(self, task: TaskDefinition) -> ExecutionResult:
        """작업 실행"""
        start_time = time.time()
        task_id = task.task_id

        try:
            self.logger.info(f"작업 실행 시작: {task_id} ({task.name})")

            # 함수가 비동기인지 확인
            if asyncio.iscoroutinefunction(task.function):
                # 비동기 함수 실행
                result_data = await asyncio.wait_for(
                    task.function(*task.args, **task.kwargs),
                    timeout=task.context.timeout
                )
            else:
                # 동기 함수를 스레드 풀에서 실행
                loop = asyncio.get_event_loop()
                result_data = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.thread_pool,
                        lambda: task.function(*task.args, **task.kwargs)
                    ),
                    timeout=task.context.timeout
                )

            execution_time = time.time() - start_time
            self.logger.info(f"작업 실행 완료: {task_id} ({execution_time:.3f}s)")

            return ExecutionResult(
                task_id=task_id,
                state=ExecutionState.COMPLETED,
                result_data=result_data,
                execution_time=execution_time,
                completed_at=current_timestamp()
            )

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            self.logger.warning(f"작업 타임아웃: {task_id} ({execution_time:.3f}s)")

            return ExecutionResult(
                task_id=task_id,
                state=ExecutionState.TIMEOUT,
                result_data=None,
                error_message=f"작업이 {task.context.timeout}초 내에 완료되지 않았습니다",
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"작업 실행 실패: {task_id} - {e}")

            return ExecutionResult(
                task_id=task_id,
                state=ExecutionState.FAILED,
                result_data=None,
                error_message=str(e),
                execution_time=execution_time
            )

    def shutdown(self):
        """실행기 종료"""
        self.thread_pool.shutdown(wait=True)
        self.logger.info("작업 실행기 종료")

class ExecutionController:
    """실행 제어기"""

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.logger = create_logger(__name__)

        # 컴포넌트 초기화
        self.resource_manager = ResourceManager(self.config)
        self.task_executor = TaskExecutor(self.config)
        self.execution_policy = ExecutionPolicy()

        # 작업 관리
        self.pending_tasks: List[TaskDefinition] = []
        self.running_tasks: Dict[ID, TaskDefinition] = {}
        self.completed_tasks: Dict[ID, ExecutionResult] = {}

        # 실행 상태
        self.is_running = False
        self.executor_loop_task: Optional[asyncio.Task] = None

        # 통계
        self.total_executed = 0
        self.total_failed = 0
        self.total_cancelled = 0

    async def start(self):
        """실행 제어기 시작"""
        self.is_running = True
        self.executor_loop_task = asyncio.create_task(self._executor_loop())
        self.logger.info("실행 제어기 시작")

    async def stop(self):
        """실행 제어기 정지"""
        self.is_running = False

        if self.executor_loop_task:
            self.executor_loop_task.cancel()
            try:
                await self.executor_loop_task
            except asyncio.CancelledError:
                pass

        # 실행 중인 작업 완료 대기
        await self._wait_for_running_tasks()

        # 실행기 종료
        self.task_executor.shutdown()

        self.logger.info("실행 제어기 정지")

    async def submit_task(
        self,
        name: str,
        function: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        priority: ExecutionPriority = ExecutionPriority.NORMAL,
        timeout: float = None,
        user_id: Optional[str] = None
    ) -> ID:
        """작업 제출"""
        task_id = create_id()
        timeout = timeout or self.config.default_timeout

        context = ExecutionContext(
            task_id=task_id,
            user_id=user_id,
            priority=priority,
            timeout=timeout
        )

        task = TaskDefinition(
            task_id=task_id,
            name=name,
            function=function,
            args=args,
            kwargs=kwargs or {},
            context=context
        )

        # 우선순위에 따라 삽입
        self._insert_task_by_priority(task)

        self.logger.info(f"작업 제출: {task_id} ({name})")
        return task_id

    def _insert_task_by_priority(self, task: TaskDefinition):
        """우선순위에 따라 작업 삽입"""
        inserted = False
        for i, pending_task in enumerate(self.pending_tasks):
            if task.context.priority.value > pending_task.context.priority.value:
                self.pending_tasks.insert(i, task)
                inserted = True
                break

        if not inserted:
            self.pending_tasks.append(task)

    async def cancel_task(self, task_id: ID) -> bool:
        """작업 취소"""
        # 대기 중인 작업에서 제거
        for task in self.pending_tasks:
            if task.task_id == task_id:
                self.pending_tasks.remove(task)
                self.total_cancelled += 1
                self.logger.info(f"대기 중인 작업 취소: {task_id}")
                return True

        # 실행 중인 작업은 취소할 수 없음 (현재 구현에서는)
        if task_id in self.running_tasks:
            self.logger.warning(f"실행 중인 작업은 취소할 수 없음: {task_id}")
            return False

        return False

    async def get_task_result(self, task_id: ID) -> Optional[ExecutionResult]:
        """작업 결과 조회"""
        return self.completed_tasks.get(task_id)

    async def get_task_status(self, task_id: ID) -> ExecutionState:
        """작업 상태 조회"""
        # 완료된 작업 확인
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].state

        # 실행 중인 작업 확인
        if task_id in self.running_tasks:
            return ExecutionState.RUNNING

        # 대기 중인 작업 확인
        for task in self.pending_tasks:
            if task.task_id == task_id:
                return ExecutionState.PENDING

        return ExecutionState.FAILED  # 작업을 찾을 수 없음

    async def _executor_loop(self):
        """실행기 메인 루프"""
        while self.is_running:
            try:
                await self._process_pending_tasks()
                await asyncio.sleep(0.1)  # 100ms 대기
            except Exception as e:
                self.logger.error(f"실행기 루프 오류: {e}")
                await asyncio.sleep(1.0)

    async def _process_pending_tasks(self):
        """대기 중인 작업 처리"""
        if not self.pending_tasks:
            return

        # 리소스 사용량 확인
        resource_usage = self.resource_manager.get_system_resources()

        # 비상 정지 체크
        if self.execution_policy.should_emergency_stop(resource_usage):
            self.logger.warning("시스템 리소스 부족으로 작업 실행 중단")
            return

        # 동시 실행 제한 체크
        if len(self.running_tasks) >= self.config.max_concurrent_tasks:
            return

        # 실행 가능한 작업 찾기
        if not self.execution_policy.should_execute(resource_usage):
            return

        # 가장 높은 우선순위 작업 실행
        task = self.pending_tasks.pop(0)
        await self._execute_task_async(task)

    async def _execute_task_async(self, task: TaskDefinition):
        """비동기 작업 실행"""
        task_id = task.task_id
        self.running_tasks[task_id] = task

        try:
            # 작업 실행
            result = await self.task_executor.execute_task(task)

            # 결과 저장
            self.completed_tasks[task_id] = result

            # 통계 업데이트
            if result.state == ExecutionState.COMPLETED:
                self.total_executed += 1
            else:
                self.total_failed += 1

        except Exception as e:
            self.logger.error(f"작업 실행 중 예외 발생: {task_id} - {e}")

            # 실패 결과 저장
            self.completed_tasks[task_id] = ExecutionResult(
                task_id=task_id,
                state=ExecutionState.FAILED,
                result_data=None,
                error_message=str(e)
            )
            self.total_failed += 1

        finally:
            # 실행 중인 작업에서 제거
            self.running_tasks.pop(task_id, None)

    async def _wait_for_running_tasks(self, timeout: float = 10.0):
        """실행 중인 작업 완료 대기"""
        start_time = time.time()

        while self.running_tasks and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)

        if self.running_tasks:
            self.logger.warning(f"타임아웃으로 {len(self.running_tasks)}개 작업 강제 종료")

    def get_execution_statistics(self) -> Dict[str, Any]:
        """실행 통계 반환"""
        total_tasks = self.total_executed + self.total_failed + self.total_cancelled

        return {
            'total_submitted': total_tasks + len(self.pending_tasks) + len(self.running_tasks),
            'total_executed': self.total_executed,
            'total_failed': self.total_failed,
            'total_cancelled': self.total_cancelled,
            'pending_count': len(self.pending_tasks),
            'running_count': len(self.running_tasks),
            'success_rate': self.total_executed / max(total_tasks, 1),
            'is_running': self.is_running
        }

    def get_resource_status(self) -> Dict[str, Any]:
        """리소스 상태 반환"""
        resource_usage = self.resource_manager.get_system_resources()

        return {
            'resource_usage': resource_usage,
            'resource_available': self.resource_manager.is_resource_available(),
            'can_execute': self.execution_policy.should_execute(resource_usage),
            'emergency_stop_needed': self.execution_policy.should_emergency_stop(resource_usage)
        }

    async def execute_function(
        self,
        function: Callable,
        *args,
        timeout: float = None,
        priority: ExecutionPriority = ExecutionPriority.NORMAL,
        **kwargs
    ) -> Any:
        """함수 직접 실행 (동기식)"""
        task_id = await self.submit_task(
            name=getattr(function, '__name__', 'anonymous'),
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout
        )

        # 작업 완료 대기
        while True:
            status = await self.get_task_status(task_id)
            if status in [ExecutionState.COMPLETED, ExecutionState.FAILED,
                         ExecutionState.TIMEOUT, ExecutionState.CANCELLED]:
                break
            await asyncio.sleep(0.1)

        # 결과 반환
        result = await self.get_task_result(task_id)
        if result.state == ExecutionState.COMPLETED:
            return result.result_data
        else:
            raise RuntimeError(f"작업 실행 실패: {result.error_message}")

# 사용 예시 함수들
async def example_async_task(data: str, delay: float = 1.0) -> str:
    """예시 비동기 작업"""
    await asyncio.sleep(delay)
    return f"처리된 데이터: {data}"

def example_sync_task(x: int, y: int) -> int:
    """예시 동기 작업"""
    time.sleep(0.5)  # 무거운 작업 시뮬레이션
    return x + y

def example_heavy_task(size: int) -> List[int]:
    """예시 무거운 작업"""
    return [i * i for i in range(size)]