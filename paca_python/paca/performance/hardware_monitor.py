"""
Hardware Monitor Module - PACA Python v5
실시간 하드웨어 모니터링 시스템

CPU/메모리 사용량을 실시간으로 추적하고
시스템 상태에 따라 성능 프로파일을 제안
"""

import asyncio
import psutil
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor

# 조건부 임포트: 패키지 실행시와 직접 실행시 모두 지원
try:
    from ..core.types.base import (
        ID, Timestamp, Result, current_timestamp, generate_id,
        create_success, create_failure
    )
except ImportError:
    from paca.core.types.base import (
        ID, Timestamp, Result, current_timestamp, generate_id,
        create_success, create_failure
    )


class AlertLevel(Enum):
    """성능 알림 레벨"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MonitoringState(Enum):
    """모니터링 상태"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class ResourceUsage:
    """리소스 사용량 정보"""
    timestamp: Timestamp
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0

    @property
    def is_high_usage(self) -> bool:
        """고사용량 상태 판정"""
        return (self.cpu_percent > 80.0 or
                self.memory_percent > 85.0 or
                self.disk_usage_percent > 90.0)

    @property
    def is_critical_usage(self) -> bool:
        """임계 사용량 상태 판정"""
        return (self.cpu_percent > 95.0 or
                self.memory_percent > 95.0 or
                self.disk_usage_percent > 98.0)


@dataclass
class PerformanceMetrics:
    """성능 메트릭 정보"""
    process_id: int
    process_cpu_percent: float
    process_memory_mb: float
    process_threads: int
    process_fds: int = 0  # File descriptors (Linux/Mac)
    gc_collections: Dict[str, int] = field(default_factory=dict)
    response_times: List[float] = field(default_factory=list)

    @property
    def avg_response_time(self) -> float:
        """평균 응답 시간 (ms)"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)


@dataclass
class PerformanceAlert:
    """성능 알림 정보"""
    alert_id: ID
    level: AlertLevel
    message: str
    timestamp: Timestamp
    metrics: Optional[Dict[str, Any]] = None
    suggested_action: Optional[str] = None

    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = generate_id("alert")


@dataclass
class SystemStatus:
    """전체 시스템 상태"""
    timestamp: Timestamp
    resource_usage: ResourceUsage
    performance_metrics: Optional[PerformanceMetrics]
    recommended_profile: str
    alerts: List[PerformanceAlert] = field(default_factory=list)
    monitoring_duration_seconds: float = 0.0

    @property
    def overall_health_score(self) -> float:
        """전체 건강도 점수 (0-100)"""
        cpu_score = max(0, 100 - self.resource_usage.cpu_percent)
        memory_score = max(0, 100 - self.resource_usage.memory_percent)
        disk_score = max(0, 100 - self.resource_usage.disk_usage_percent)

        # 가중평균 계산
        return (cpu_score * 0.4 + memory_score * 0.4 + disk_score * 0.2)

    @property
    def is_healthy(self) -> bool:
        """시스템 건강 상태 판정"""
        return (self.overall_health_score >= 70.0 and
                not any(alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]
                       for alert in self.alerts))


class HardwareMonitor:
    """
    실시간 하드웨어 모니터링 클래스

    CPU/메모리 사용량을 추적하고 성능 프로파일을 제안하며
    임계값 초과시 알림을 생성합니다.
    """

    def __init__(self,
                 monitoring_interval: float = 1.0,
                 history_size: int = 100,
                 alert_thresholds: Optional[Dict[str, float]] = None):
        """
        하드웨어 모니터 초기화

        Args:
            monitoring_interval: 모니터링 간격 (초)
            history_size: 히스토리 보관 개수
            alert_thresholds: 알림 임계값 설정
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.alert_thresholds = alert_thresholds or {
            'cpu_warning': 70.0,
            'cpu_critical': 90.0,
            'memory_warning': 75.0,
            'memory_critical': 90.0,
            'disk_warning': 80.0,
            'disk_critical': 95.0
        }

        self._state = MonitoringState.STOPPED
        self._monitoring_task: Optional[asyncio.Task] = None
        self._callbacks: List[Callable[[SystemStatus], None]] = []
        self._history: List[SystemStatus] = []
        self._start_time: Optional[float] = None
        self._executor = ThreadPoolExecutor(max_workers=2)

        # 로깅 설정
        self.logger = logging.getLogger(__name__)

    @property
    def state(self) -> MonitoringState:
        """현재 모니터링 상태"""
        return self._state

    @property
    def is_running(self) -> bool:
        """모니터링 실행 중 여부"""
        return self._state == MonitoringState.RUNNING

    @property
    def history(self) -> List[SystemStatus]:
        """시스템 상태 히스토리 (읽기 전용)"""
        return self._history.copy()

    def add_callback(self, callback: Callable[[SystemStatus], None]) -> None:
        """상태 변경 콜백 함수 추가"""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[SystemStatus], None]) -> None:
        """상태 변경 콜백 함수 제거"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _collect_resource_usage(self) -> ResourceUsage:
        """시스템 리소스 사용량 수집"""
        try:
            # CPU 사용량 (1초 간격으로 측정)
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # 메모리 사용량
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / (1024 * 1024)

            # 디스크 사용량 (루트 파티션)
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent

            # 네트워크 사용량
            net_io = psutil.net_io_counters()
            network_bytes_sent = net_io.bytes_sent if net_io else 0
            network_bytes_recv = net_io.bytes_recv if net_io else 0

            return ResourceUsage(
                timestamp=current_timestamp(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv
            )
        except Exception as e:
            self.logger.error(f"리소스 사용량 수집 실패: {e}")
            # 기본값 반환
            return ResourceUsage(
                timestamp=current_timestamp(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0
            )

    def _collect_performance_metrics(self) -> Optional[PerformanceMetrics]:
        """PACA 프로세스 성능 메트릭 수집"""
        try:
            import os
            import gc

            process = psutil.Process(os.getpid())

            # 프로세스 정보
            process_cpu_percent = process.cpu_percent()
            process_memory_mb = process.memory_info().rss / (1024 * 1024)
            process_threads = process.num_threads()

            # 파일 디스크립터 (Unix 계열에서만)
            process_fds = 0
            try:
                process_fds = process.num_fds()
            except AttributeError:
                pass  # Windows에서는 지원하지 않음

            # 가비지 컬렉션 통계
            gc_collections = {
                f"generation_{i}": gc.get_count()[i]
                for i in range(len(gc.get_count()))
            }

            return PerformanceMetrics(
                process_id=os.getpid(),
                process_cpu_percent=process_cpu_percent,
                process_memory_mb=process_memory_mb,
                process_threads=process_threads,
                process_fds=process_fds,
                gc_collections=gc_collections
            )
        except Exception as e:
            self.logger.warning(f"성능 메트릭 수집 실패: {e}")
            return None

    def _analyze_alerts(self, resource_usage: ResourceUsage) -> List[PerformanceAlert]:
        """리소스 사용량 기반 알림 분석"""
        alerts = []

        # CPU 사용량 체크
        if resource_usage.cpu_percent >= self.alert_thresholds['cpu_critical']:
            alerts.append(PerformanceAlert(
                alert_id="",  # __post_init__에서 자동 생성
                level=AlertLevel.CRITICAL,
                message=f"CPU 사용량이 임계값을 초과했습니다: {resource_usage.cpu_percent:.1f}%",
                timestamp=current_timestamp(),
                metrics={'cpu_percent': resource_usage.cpu_percent},
                suggested_action="conservative 프로파일로 전환하거나 불필요한 프로세스를 종료하세요"
            ))
        elif resource_usage.cpu_percent >= self.alert_thresholds['cpu_warning']:
            alerts.append(PerformanceAlert(
                alert_id="",
                level=AlertLevel.WARNING,
                message=f"CPU 사용량이 높습니다: {resource_usage.cpu_percent:.1f}%",
                timestamp=current_timestamp(),
                metrics={'cpu_percent': resource_usage.cpu_percent},
                suggested_action="low-end 또는 mid-range 프로파일로 전환을 고려하세요"
            ))

        # 메모리 사용량 체크
        if resource_usage.memory_percent >= self.alert_thresholds['memory_critical']:
            alerts.append(PerformanceAlert(
                alert_id="",
                level=AlertLevel.CRITICAL,
                message=f"메모리 사용량이 임계값을 초과했습니다: {resource_usage.memory_percent:.1f}%",
                timestamp=current_timestamp(),
                metrics={'memory_percent': resource_usage.memory_percent},
                suggested_action="conservative 프로파일로 전환하고 메모리를 정리하세요"
            ))
        elif resource_usage.memory_percent >= self.alert_thresholds['memory_warning']:
            alerts.append(PerformanceAlert(
                alert_id="",
                level=AlertLevel.WARNING,
                message=f"메모리 사용량이 높습니다: {resource_usage.memory_percent:.1f}%",
                timestamp=current_timestamp(),
                metrics={'memory_percent': resource_usage.memory_percent},
                suggested_action="메모리 사용량을 모니터링하고 필요시 프로파일을 조정하세요"
            ))

        # 디스크 사용량 체크
        if resource_usage.disk_usage_percent >= self.alert_thresholds['disk_critical']:
            alerts.append(PerformanceAlert(
                alert_id="",
                level=AlertLevel.CRITICAL,
                message=f"디스크 사용량이 임계값을 초과했습니다: {resource_usage.disk_usage_percent:.1f}%",
                timestamp=current_timestamp(),
                metrics={'disk_usage_percent': resource_usage.disk_usage_percent},
                suggested_action="디스크 공간을 확보하세요"
            ))
        elif resource_usage.disk_usage_percent >= self.alert_thresholds['disk_warning']:
            alerts.append(PerformanceAlert(
                alert_id="",
                level=AlertLevel.WARNING,
                message=f"디스크 사용량이 높습니다: {resource_usage.disk_usage_percent:.1f}%",
                timestamp=current_timestamp(),
                metrics={'disk_usage_percent': resource_usage.disk_usage_percent},
                suggested_action="디스크 정리를 고려하세요"
            ))

        return alerts

    def _suggest_performance_profile(self, resource_usage: ResourceUsage) -> str:
        """리소스 사용량 기반 성능 프로파일 제안"""
        # 임계 상황
        if resource_usage.is_critical_usage:
            return "conservative"

        # 고사용량 상황
        if resource_usage.is_high_usage:
            return "low-end"

        # CPU와 메모리 사용량 기반 판정
        avg_usage = (resource_usage.cpu_percent + resource_usage.memory_percent) / 2

        if avg_usage < 30:
            return "high-end"
        elif avg_usage < 60:
            return "mid-range"
        else:
            return "low-end"

    def get_system_status(self) -> Result[SystemStatus]:
        """현재 시스템 상태 즉시 조회"""
        try:
            # 스레드 풀에서 동기 작업 실행
            loop = asyncio.get_event_loop()

            # 리소스 사용량 수집
            resource_usage = self._collect_resource_usage()

            # 성능 메트릭 수집
            performance_metrics = self._collect_performance_metrics()

            # 알림 분석
            alerts = self._analyze_alerts(resource_usage)

            # 프로파일 제안
            recommended_profile = self._suggest_performance_profile(resource_usage)

            # 모니터링 지속 시간 계산
            duration = 0.0
            if self._start_time:
                duration = current_timestamp() - self._start_time

            status = SystemStatus(
                timestamp=current_timestamp(),
                resource_usage=resource_usage,
                performance_metrics=performance_metrics,
                recommended_profile=recommended_profile,
                alerts=alerts,
                monitoring_duration_seconds=duration
            )

            return create_success(status)

        except Exception as e:
            self.logger.error(f"시스템 상태 조회 실패: {e}")
            return create_failure(Exception(f"시스템 상태 조회 실패: {e}"))

    async def _monitoring_loop(self) -> None:
        """모니터링 루프 (비동기)"""
        self.logger.info("하드웨어 모니터링 시작")
        self._start_time = current_timestamp()

        while self._state == MonitoringState.RUNNING:
            try:
                # 시스템 상태 수집
                status_result = self.get_system_status()

                if status_result.is_success:
                    status = status_result.value

                    # 히스토리에 추가 (크기 제한)
                    self._history.append(status)
                    if len(self._history) > self.history_size:
                        self._history.pop(0)

                    # 콜백 함수 호출
                    for callback in self._callbacks:
                        try:
                            callback(status)
                        except Exception as e:
                            self.logger.warning(f"콜백 함수 실행 실패: {e}")

                    # 임계 상황시 추가 로깅
                    if status.resource_usage.is_critical_usage:
                        self.logger.warning(f"임계 리소스 사용량 감지: "
                                          f"CPU {status.resource_usage.cpu_percent:.1f}%, "
                                          f"Memory {status.resource_usage.memory_percent:.1f}%")

                # 다음 주기까지 대기
                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"모니터링 루프 오류: {e}")
                self._state = MonitoringState.ERROR
                break

        self.logger.info("하드웨어 모니터링 종료")

    async def start_monitoring(self) -> Result[str]:
        """모니터링 시작"""
        try:
            if self._state == MonitoringState.RUNNING:
                return create_failure(Exception("이미 모니터링이 실행 중입니다"))

            self._state = MonitoringState.RUNNING
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

            monitoring_id = generate_id("monitor")
            self.logger.info(f"하드웨어 모니터링 시작: {monitoring_id}")

            return create_success(monitoring_id)

        except Exception as e:
            self._state = MonitoringState.ERROR
            return create_failure(Exception(f"모니터링 시작 실패: {e}"))

    async def stop_monitoring(self) -> Result[bool]:
        """모니터링 중지"""
        try:
            if self._state != MonitoringState.RUNNING:
                return create_failure(Exception("모니터링이 실행 중이 아닙니다"))

            self._state = MonitoringState.STOPPED

            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
                self._monitoring_task = None

            self.logger.info("하드웨어 모니터링 중지")
            return create_success(True)

        except Exception as e:
            self._state = MonitoringState.ERROR
            return create_failure(Exception(f"모니터링 중지 실패: {e}"))

    def get_resource_history(self,
                           last_n: Optional[int] = None,
                           time_range_seconds: Optional[float] = None) -> List[ResourceUsage]:
        """리소스 사용량 히스토리 조회"""
        history_data = [status.resource_usage for status in self._history]

        # 시간 범위 필터링
        if time_range_seconds:
            cutoff_time = current_timestamp() - time_range_seconds
            history_data = [usage for usage in history_data
                          if usage.timestamp >= cutoff_time]

        # 개수 제한
        if last_n:
            history_data = history_data[-last_n:]

        return history_data

    def get_performance_trends(self) -> Dict[str, Any]:
        """성능 트렌드 분석"""
        if len(self._history) < 2:
            return {'error': '분석을 위한 충분한 데이터가 없습니다'}

        # 최근 데이터 추출
        recent_data = self._history[-10:]  # 최근 10개 데이터

        cpu_values = [s.resource_usage.cpu_percent for s in recent_data]
        memory_values = [s.resource_usage.memory_percent for s in recent_data]

        return {
            'cpu_trend': {
                'average': sum(cpu_values) / len(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values),
                'trend': 'increasing' if cpu_values[-1] > cpu_values[0] else 'decreasing'
            },
            'memory_trend': {
                'average': sum(memory_values) / len(memory_values),
                'min': min(memory_values),
                'max': max(memory_values),
                'trend': 'increasing' if memory_values[-1] > memory_values[0] else 'decreasing'
            },
            'sample_count': len(recent_data),
            'time_span_seconds': recent_data[-1].timestamp - recent_data[0].timestamp if recent_data else 0
        }

    def __del__(self):
        """소멸자 - 리소스 정리"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


# 편의 함수들
async def quick_system_check() -> SystemStatus:
    """빠른 시스템 상태 체크"""
    monitor = HardwareMonitor()
    result = monitor.get_system_status()

    if result.is_success:
        return result.value
    else:
        raise result.error


def get_current_resource_usage() -> ResourceUsage:
    """현재 리소스 사용량 동기 조회"""
    monitor = HardwareMonitor()
    return monitor._collect_resource_usage()


if __name__ == "__main__":
    # 테스트 실행
    async def main():
        print("=== PACA v5 하드웨어 모니터 테스트 ===")

        monitor = HardwareMonitor(monitoring_interval=2.0)

        # 즉시 상태 체크
        status_result = monitor.get_system_status()
        if status_result.is_success:
            status = status_result.value
            print(f"CPU: {status.resource_usage.cpu_percent:.1f}%")
            print(f"Memory: {status.resource_usage.memory_percent:.1f}%")
            print(f"추천 프로파일: {status.recommended_profile}")
            print(f"건강도 점수: {status.overall_health_score:.1f}")
            print(f"알림 개수: {len(status.alerts)}")

        # 짧은 모니터링 테스트
        print("\n10초간 모니터링 테스트...")

        def status_callback(status: SystemStatus):
            print(f"[{status.timestamp:.1f}] CPU: {status.resource_usage.cpu_percent:.1f}%, "
                  f"Memory: {status.resource_usage.memory_percent:.1f}%, "
                  f"프로파일: {status.recommended_profile}")

        monitor.add_callback(status_callback)

        await monitor.start_monitoring()
        await asyncio.sleep(10)
        await monitor.stop_monitoring()

        print("테스트 완료!")

    asyncio.run(main())