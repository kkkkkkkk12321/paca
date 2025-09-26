"""
Resource Monitor Module
자원 인식 시스템 - CPU/RAM 사용량 실시간 모니터링 및 동적 우선순위 조절
"""

import asyncio
import psutil
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import threading
from queue import Queue, PriorityQueue
import logging

# 조건부 임포트
try:
    from ..core.types.base import (
        ID, Timestamp, Result, current_timestamp, generate_id, create_success, create_failure
    )
except ImportError:
    from paca.core.types.base import (
        ID, Timestamp, Result, current_timestamp, generate_id, create_success, create_failure
    )


class ResourceStatus(Enum):
    """자원 상태"""
    NORMAL = "normal"        # 정상 (0-60%)
    WARNING = "warning"      # 경고 (60-80%)
    CRITICAL = "critical"    # 위험 (80-95%)
    EMERGENCY = "emergency"  # 비상 (95%+)


class TaskPriority(Enum):
    """작업 우선순위"""
    EMERGENCY = 0    # 사용자 응답
    HIGH = 1        # 핵심 시스템
    MEDIUM = 2      # 일반 작업
    LOW = 3         # 호기심 탐구
    BACKGROUND = 4  # 백그라운드 작업


@dataclass
class ResourceMetrics:
    """자원 사용량 메트릭"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    timestamp: Timestamp = field(default_factory=current_timestamp)

    @property
    def overall_usage(self) -> float:
        """전체 자원 사용률"""
        return (self.cpu_percent + self.memory_percent + self.disk_percent) / 3.0

    @property
    def status(self) -> ResourceStatus:
        """자원 상태 평가"""
        usage = self.overall_usage
        if usage >= 95:
            return ResourceStatus.EMERGENCY
        elif usage >= 80:
            return ResourceStatus.CRITICAL
        elif usage >= 60:
            return ResourceStatus.WARNING
        else:
            return ResourceStatus.NORMAL


@dataclass
class ResourceAlert:
    """자원 알림"""
    alert_id: ID
    alert_type: ResourceStatus
    message: str
    metrics: ResourceMetrics
    timestamp: Timestamp = field(default_factory=current_timestamp)
    acknowledged: bool = False


@dataclass
class BackgroundTask:
    """백그라운드 작업"""
    task_id: ID
    name: str
    priority: TaskPriority
    function: Callable
    args: tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: float = 0.0
    resource_requirement: float = 0.1  # 0-1 scale
    created_at: Timestamp = field(default_factory=current_timestamp)
    started_at: Optional[Timestamp] = None
    completed_at: Optional[Timestamp] = None
    suspended: bool = False


class SystemResourceMonitor:
    """시스템 자원 실시간 모니터링"""

    def __init__(self,
                 polling_interval: float = 1.0,
                 warning_threshold: float = 60.0,
                 critical_threshold: float = 80.0,
                 emergency_threshold: float = 95.0):
        self.polling_interval = polling_interval
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.emergency_threshold = emergency_threshold

        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics_history: List[ResourceMetrics] = []
        self.alerts: List[ResourceAlert] = []
        self.alert_callbacks: List[Callable[[ResourceAlert], None]] = []

        self.logger = logging.getLogger(__name__)

        # 최대 기록 보관 수
        self.max_history_size = 1000
        self.max_alerts_size = 100

    def start_monitoring(self) -> None:
        """모니터링 시작"""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("자원 모니터링이 시작되었습니다")

    def stop_monitoring(self) -> None:
        """모니터링 중지"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("자원 모니터링이 중지되었습니다")

    def _monitor_loop(self) -> None:
        """모니터링 루프"""
        while self.running:
            try:
                metrics = self._collect_metrics()
                self._record_metrics(metrics)
                self._check_alerts(metrics)

                time.sleep(self.polling_interval)

            except Exception as e:
                self.logger.error(f"모니터링 오류: {e}")
                time.sleep(self.polling_interval * 2)  # 오류시 더 오래 대기

    def _collect_metrics(self) -> ResourceMetrics:
        """자원 사용량 수집"""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # 메모리 사용률
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # 디스크 사용률
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100

            # 네트워크 I/O
            network_io = psutil.net_io_counters()._asdict()

            return ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_io=network_io
            )

        except Exception as e:
            self.logger.error(f"메트릭 수집 실패: {e}")
            # 기본값 반환
            return ResourceMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                network_io={}
            )

    def _record_metrics(self, metrics: ResourceMetrics) -> None:
        """메트릭 기록"""
        self.metrics_history.append(metrics)

        # 히스토리 크기 제한
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]

    def _check_alerts(self, metrics: ResourceMetrics) -> None:
        """알림 확인 및 생성"""
        previous_status = None
        if self.metrics_history and len(self.metrics_history) > 1:
            previous_status = self.metrics_history[-2].status

        current_status = metrics.status

        # 상태 변화가 있거나 위험 상태일 때 알림
        if (previous_status != current_status or
            current_status in [ResourceStatus.CRITICAL, ResourceStatus.EMERGENCY]):

            alert = self._create_alert(current_status, metrics)
            self.alerts.append(alert)

            # 알림 크기 제한
            if len(self.alerts) > self.max_alerts_size:
                self.alerts = self.alerts[-self.max_alerts_size:]

            # 콜백 실행
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"알림 콜백 실행 실패: {e}")

    def _create_alert(self, status: ResourceStatus, metrics: ResourceMetrics) -> ResourceAlert:
        """알림 생성"""
        status_messages = {
            ResourceStatus.NORMAL: "자원 사용량이 정상 수준입니다",
            ResourceStatus.WARNING: f"자원 사용량 경고 (CPU: {metrics.cpu_percent:.1f}%, RAM: {metrics.memory_percent:.1f}%, Disk: {metrics.disk_percent:.1f}%)",
            ResourceStatus.CRITICAL: f"자원 사용량 위험 (CPU: {metrics.cpu_percent:.1f}%, RAM: {metrics.memory_percent:.1f}%, Disk: {metrics.disk_percent:.1f}%)",
            ResourceStatus.EMERGENCY: f"자원 사용량 비상 (CPU: {metrics.cpu_percent:.1f}%, RAM: {metrics.memory_percent:.1f}%, Disk: {metrics.disk_percent:.1f}%)"
        }

        return ResourceAlert(
            alert_id=generate_id("alert_"),
            alert_type=status,
            message=status_messages[status],
            metrics=metrics
        )

    def add_alert_callback(self, callback: Callable[[ResourceAlert], None]) -> None:
        """알림 콜백 추가"""
        self.alert_callbacks.append(callback)

    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """현재 메트릭 조회"""
        return self.metrics_history[-1] if self.metrics_history else None

    def get_metrics_history(self, hours: int = 1) -> List[ResourceMetrics]:
        """메트릭 히스토리 조회"""
        cutoff_time = current_timestamp() - (hours * 3600)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]

    def get_resource_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """자원 사용 통계"""
        recent_metrics = self.get_metrics_history(hours)

        if not recent_metrics:
            return {'error': '데이터 없음'}

        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        disk_values = [m.disk_percent for m in recent_metrics]

        return {
            'period_hours': hours,
            'data_points': len(recent_metrics),
            'cpu_stats': {
                'average': sum(cpu_values) / len(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values)
            },
            'memory_stats': {
                'average': sum(memory_values) / len(memory_values),
                'min': min(memory_values),
                'max': max(memory_values)
            },
            'disk_stats': {
                'average': sum(disk_values) / len(disk_values),
                'min': min(disk_values),
                'max': max(disk_values)
            },
            'alert_count': len([a for a in self.alerts
                              if a.timestamp >= current_timestamp() - (hours * 3600)])
        }


class PriorityManager:
    """작업 우선순위 동적 관리"""

    def __init__(self, resource_monitor: SystemResourceMonitor):
        self.resource_monitor = resource_monitor
        self.priority_rules: Dict[ResourceStatus, Dict[TaskPriority, bool]] = {
            ResourceStatus.NORMAL: {
                TaskPriority.EMERGENCY: True,
                TaskPriority.HIGH: True,
                TaskPriority.MEDIUM: True,
                TaskPriority.LOW: True,
                TaskPriority.BACKGROUND: True
            },
            ResourceStatus.WARNING: {
                TaskPriority.EMERGENCY: True,
                TaskPriority.HIGH: True,
                TaskPriority.MEDIUM: True,
                TaskPriority.LOW: False,  # 호기심 탐구 일시 중단
                TaskPriority.BACKGROUND: False
            },
            ResourceStatus.CRITICAL: {
                TaskPriority.EMERGENCY: True,
                TaskPriority.HIGH: True,
                TaskPriority.MEDIUM: False,
                TaskPriority.LOW: False,
                TaskPriority.BACKGROUND: False
            },
            ResourceStatus.EMERGENCY: {
                TaskPriority.EMERGENCY: True,  # 사용자 응답만
                TaskPriority.HIGH: False,
                TaskPriority.MEDIUM: False,
                TaskPriority.LOW: False,
                TaskPriority.BACKGROUND: False
            }
        }

        self.logger = logging.getLogger(__name__)

    def is_task_allowed(self, priority: TaskPriority) -> bool:
        """작업 실행 허용 여부 확인"""
        current_metrics = self.resource_monitor.get_current_metrics()
        if not current_metrics:
            return True  # 메트릭이 없으면 허용

        resource_status = current_metrics.status
        return self.priority_rules[resource_status].get(priority, False)

    def get_allowed_priorities(self) -> List[TaskPriority]:
        """현재 허용된 우선순위 목록"""
        current_metrics = self.resource_monitor.get_current_metrics()
        if not current_metrics:
            return list(TaskPriority)  # 모든 우선순위 허용

        resource_status = current_metrics.status
        allowed_rules = self.priority_rules[resource_status]

        return [priority for priority, allowed in allowed_rules.items() if allowed]

    def should_suspend_background_tasks(self) -> bool:
        """백그라운드 작업 중단 여부"""
        return not self.is_task_allowed(TaskPriority.BACKGROUND)

    def get_priority_adjustment_recommendation(self) -> Dict[str, Any]:
        """우선순위 조정 권고사항"""
        current_metrics = self.resource_monitor.get_current_metrics()
        if not current_metrics:
            return {'recommendation': 'normal_operation'}

        status = current_metrics.status
        allowed_priorities = self.get_allowed_priorities()

        recommendations = {
            ResourceStatus.NORMAL: {
                'action': 'normal_operation',
                'message': '모든 작업 정상 실행 가능'
            },
            ResourceStatus.WARNING: {
                'action': 'suspend_low_priority',
                'message': '호기심 탐구 및 백그라운드 작업 일시 중단',
                'suspended_priorities': ['LOW', 'BACKGROUND']
            },
            ResourceStatus.CRITICAL: {
                'action': 'essential_only',
                'message': '필수 작업만 실행, 일반 작업 일시 중단',
                'suspended_priorities': ['MEDIUM', 'LOW', 'BACKGROUND']
            },
            ResourceStatus.EMERGENCY: {
                'action': 'emergency_mode',
                'message': '사용자 응답만 처리, 모든 백그라운드 작업 중단',
                'suspended_priorities': ['HIGH', 'MEDIUM', 'LOW', 'BACKGROUND']
            }
        }

        recommendation = recommendations[status].copy()
        recommendation['current_status'] = status.value
        recommendation['allowed_priorities'] = [p.value for p in allowed_priorities]
        recommendation['resource_usage'] = {
            'cpu': current_metrics.cpu_percent,
            'memory': current_metrics.memory_percent,
            'disk': current_metrics.disk_percent
        }

        return recommendation


class BackgroundTaskScheduler:
    """백그라운드 작업 스케줄러"""

    def __init__(self, priority_manager: PriorityManager):
        self.priority_manager = priority_manager
        self.task_queue = PriorityQueue()
        self.active_tasks: Dict[ID, BackgroundTask] = {}
        self.completed_tasks: List[BackgroundTask] = []
        self.running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.max_concurrent_tasks = 3

        self.logger = logging.getLogger(__name__)

    def start_scheduler(self) -> None:
        """스케줄러 시작"""
        if self.running:
            return

        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        self.logger.info("백그라운드 작업 스케줄러가 시작되었습니다")

    def stop_scheduler(self) -> None:
        """스케줄러 중지"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        self.logger.info("백그라운드 작업 스케줄러가 중지되었습니다")

    def schedule_task(self,
                     name: str,
                     function: Callable,
                     priority: TaskPriority = TaskPriority.BACKGROUND,
                     args: tuple = (),
                     kwargs: Dict[str, Any] = None,
                     estimated_duration: float = 0.0,
                     resource_requirement: float = 0.1) -> ID:
        """작업 스케줄링"""
        if kwargs is None:
            kwargs = {}

        task = BackgroundTask(
            task_id=generate_id("task_"),
            name=name,
            priority=priority,
            function=function,
            args=args,
            kwargs=kwargs,
            estimated_duration=estimated_duration,
            resource_requirement=resource_requirement
        )

        # 우선순위 큐에 추가 (우선순위 값이 낮을수록 높은 우선순위)
        self.task_queue.put((priority.value, task.created_at, task))

        self.logger.info(f"작업 스케줄링: {name} (우선순위: {priority.value})")
        return task.task_id

    def _scheduler_loop(self) -> None:
        """스케줄러 루프"""
        while self.running:
            try:
                # 동시 실행 작업 수 확인
                if len(self.active_tasks) >= self.max_concurrent_tasks:
                    time.sleep(0.5)
                    continue

                # 큐에서 작업 가져오기
                if self.task_queue.empty():
                    time.sleep(0.5)
                    continue

                _, _, task = self.task_queue.get(timeout=1.0)

                # 작업 실행 가능 여부 확인
                if not self.priority_manager.is_task_allowed(task.priority):
                    # 다시 큐에 넣기 (나중에 다시 시도)
                    self.task_queue.put((task.priority.value, task.created_at, task))
                    time.sleep(1.0)
                    continue

                # 작업 실행
                self._execute_task(task)

            except Exception as e:
                self.logger.error(f"스케줄러 루프 오류: {e}")
                time.sleep(1.0)

    def _execute_task(self, task: BackgroundTask) -> None:
        """작업 실행"""
        task.started_at = current_timestamp()
        self.active_tasks[task.task_id] = task

        def task_wrapper():
            try:
                self.logger.info(f"작업 실행 시작: {task.name}")

                # 실제 작업 함수 실행
                result = task.function(*task.args, **task.kwargs)

                task.completed_at = current_timestamp()
                self.logger.info(f"작업 완료: {task.name}")

                # 완료된 작업을 활성 목록에서 제거하고 완료 목록에 추가
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                self.completed_tasks.append(task)

                # 완료 목록 크기 제한
                if len(self.completed_tasks) > 100:
                    self.completed_tasks = self.completed_tasks[-100:]

            except Exception as e:
                self.logger.error(f"작업 실행 실패: {task.name}, 오류: {e}")
                task.completed_at = current_timestamp()
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]

        # 별도 스레드에서 작업 실행
        task_thread = threading.Thread(target=task_wrapper, daemon=True)
        task_thread.start()

    def suspend_low_priority_tasks(self) -> List[ID]:
        """낮은 우선순위 작업 일시 중단"""
        suspended_task_ids = []

        for task_id, task in list(self.active_tasks.items()):
            if not self.priority_manager.is_task_allowed(task.priority):
                task.suspended = True
                suspended_task_ids.append(task_id)
                self.logger.info(f"작업 일시 중단: {task.name}")

        return suspended_task_ids

    def resume_suspended_tasks(self) -> List[ID]:
        """중단된 작업 재개"""
        resumed_task_ids = []

        for task_id, task in self.active_tasks.items():
            if task.suspended and self.priority_manager.is_task_allowed(task.priority):
                task.suspended = False
                resumed_task_ids.append(task_id)
                self.logger.info(f"작업 재개: {task.name}")

        return resumed_task_ids

    def get_scheduler_statistics(self) -> Dict[str, Any]:
        """스케줄러 통계"""
        return {
            'queue_size': self.task_queue.qsize(),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'suspended_tasks': len([t for t in self.active_tasks.values() if t.suspended]),
            'active_task_details': [
                {
                    'task_id': task.task_id,
                    'name': task.name,
                    'priority': task.priority.value,
                    'started_at': task.started_at,
                    'suspended': task.suspended
                }
                for task in self.active_tasks.values()
            ],
            'recent_completed': [
                {
                    'task_id': task.task_id,
                    'name': task.name,
                    'priority': task.priority.value,
                    'duration': (task.completed_at or current_timestamp()) - task.started_at if task.started_at else 0,
                    'completed_at': task.completed_at
                }
                for task in self.completed_tasks[-5:]
            ]
        }


# 전역 인스턴스 (싱글톤 패턴)
_resource_monitor_instance = None
_priority_manager_instance = None
_task_scheduler_instance = None


def get_resource_monitor() -> SystemResourceMonitor:
    """자원 모니터 싱글톤 인스턴스 획득"""
    global _resource_monitor_instance
    if _resource_monitor_instance is None:
        _resource_monitor_instance = SystemResourceMonitor()
        _resource_monitor_instance.start_monitoring()
    return _resource_monitor_instance


def get_priority_manager() -> PriorityManager:
    """우선순위 관리자 싱글톤 인스턴스 획득"""
    global _priority_manager_instance
    if _priority_manager_instance is None:
        _priority_manager_instance = PriorityManager(get_resource_monitor())
    return _priority_manager_instance


def get_task_scheduler() -> BackgroundTaskScheduler:
    """작업 스케줄러 싱글톤 인스턴스 획득"""
    global _task_scheduler_instance
    if _task_scheduler_instance is None:
        _task_scheduler_instance = BackgroundTaskScheduler(get_priority_manager())
        _task_scheduler_instance.start_scheduler()
    return _task_scheduler_instance


def shutdown_monitoring_system():
    """모니터링 시스템 종료"""
    global _resource_monitor_instance, _task_scheduler_instance

    if _task_scheduler_instance:
        _task_scheduler_instance.stop_scheduler()
        _task_scheduler_instance = None

    if _resource_monitor_instance:
        _resource_monitor_instance.stop_monitoring()
        _resource_monitor_instance = None