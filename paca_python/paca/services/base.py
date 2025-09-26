"""
Services Base Module
서비스 시스템의 기본 클래스들과 공통 인터페이스
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..core.types.base import ID, Timestamp, KeyValuePair, Result, create_id, current_timestamp
from ..core.events.emitter import EventEmitter
from ..core.utils.logger import create_logger
from ..core.errors.base import PacaError, ExternalServiceError


class ServiceStatus(Enum):
    """서비스 상태"""
    UNKNOWN = 'unknown'
    STARTING = 'starting'
    RUNNING = 'running'
    STOPPING = 'stopping'
    STOPPED = 'stopped'
    ERROR = 'error'
    MAINTENANCE = 'maintenance'


class ServiceType(Enum):
    """서비스 타입"""
    WEB_API = 'web_api'
    BACKGROUND_TASK = 'background_task'
    DATA_PROCESSOR = 'data_processor'
    MESSAGE_HANDLER = 'message_handler'
    SCHEDULER = 'scheduler'
    GATEWAY = 'gateway'


class ServicePriority(Enum):
    """서비스 우선순위"""
    LOW = 'low'
    NORMAL = 'normal'
    HIGH = 'high'
    CRITICAL = 'critical'


@dataclass
class ServiceConfig:
    """서비스 설정"""
    id: ID
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    type: ServiceType = ServiceType.DATA_PROCESSOR
    enabled: bool = True
    auto_start: bool = True
    restart_on_failure: bool = True
    max_retries: int = 3
    retry_delay: int = 1000  # milliseconds
    timeout: int = 60000  # milliseconds
    dependencies: List[str] = field(default_factory=list)
    enable_metrics: bool = True
    enable_events: bool = True
    priority: ServicePriority = ServicePriority.NORMAL
    retry_delay_seconds: float = 5.0
    health_check_interval: float = 30.0
    timeout_seconds: float = 60.0
    configuration: KeyValuePair = field(default_factory=dict)


@dataclass
class ServiceHealth:
    """서비스 건강 상태"""
    service_id: ID
    status: ServiceStatus
    last_check: Timestamp
    uptime_seconds: float
    response_time_ms: float
    error_count: int
    restart_count: int
    metadata: KeyValuePair = field(default_factory=dict)


@dataclass
class ServiceMetrics:
    """서비스 메트릭"""
    requests_total: int = 0
    requests_successful: int = 0
    requests_failed: int = 0
    average_response_time_ms: float = 0.0
    peak_response_time_ms: float = 0.0
    last_request_at: Optional[Timestamp] = None
    uptime_percentage: float = 0.0


class BaseService(ABC):
    """기본 서비스 추상 클래스"""

    def __init__(self, config: ServiceConfig, events: Optional[EventEmitter] = None):
        self.config = config
        self.logger = create_logger(f"Service.{config.name}")
        self.events = events

        self._status = ServiceStatus.STOPPED
        self._start_time: Optional[float] = None
        self._metrics = ServiceMetrics()
        self._health = ServiceHealth(
            service_id=config.id,
            status=ServiceStatus.STOPPED,
            last_check=current_timestamp(),
            uptime_seconds=0.0,
            response_time_ms=0.0,
            error_count=0,
            restart_count=0
        )

        self._retry_count = 0
        self._last_error: Optional[Exception] = None

    async def start(self) -> Result[bool]:
        """서비스 시작"""
        if self._status == ServiceStatus.RUNNING:
            return Result.success(True)

        self._status = ServiceStatus.STARTING
        self.logger.info(f"Starting service {self.config.name}")

        if self.events:
            await self.events.emit('service.starting', {
                'service_id': self.config.id,
                'service_name': self.config.name
            })

        try:
            await self.initialize()
            self._start_time = time.time()
            self._status = ServiceStatus.RUNNING
            self._retry_count = 0

            if self.events:
                await self.events.emit('service.started', {
                    'service_id': self.config.id,
                    'service_name': self.config.name
                })

            self.logger.info(f"Service {self.config.name} started successfully")
            return Result.success(True)

        except Exception as error:
            self._status = ServiceStatus.ERROR
            self._last_error = error

            self.logger.error(
                f"Failed to start service {self.config.name}",
                error=error
            )

            if self.events:
                await self.events.emit('service.start_failed', {
                    'service_id': self.config.id,
                    'service_name': self.config.name,
                    'error': str(error)
                })

            return Result.failure(ExternalServiceError(
                message=f"Service start failed: {str(error)}",
                service_name=self.config.name,
                operation="start"
            ))

    async def stop(self) -> Result[bool]:
        """서비스 중지"""
        if self._status == ServiceStatus.STOPPED:
            return Result.success(True)

        self._status = ServiceStatus.STOPPING
        self.logger.info(f"Stopping service {self.config.name}")

        if self.events:
            await self.events.emit('service.stopping', {
                'service_id': self.config.id,
                'service_name': self.config.name
            })

        try:
            await self.cleanup()
            self._status = ServiceStatus.STOPPED
            self._start_time = None

            if self.events:
                await self.events.emit('service.stopped', {
                    'service_id': self.config.id,
                    'service_name': self.config.name
                })

            self.logger.info(f"Service {self.config.name} stopped successfully")
            return Result.success(True)

        except Exception as error:
            self._status = ServiceStatus.ERROR
            self._last_error = error

            self.logger.error(
                f"Failed to stop service {self.config.name}",
                error=error
            )

            return Result.failure(ExternalServiceError(
                message=f"Service stop failed: {str(error)}",
                service_name=self.config.name,
                operation="stop"
            ))

    async def restart(self) -> Result[bool]:
        """서비스 재시작"""
        self.logger.info(f"Restarting service {self.config.name}")

        stop_result = await self.stop()
        if not stop_result.is_success:
            return stop_result

        # 잠시 대기
        await asyncio.sleep(1.0)

        start_result = await self.start()
        if start_result.is_success:
            self._health.restart_count += 1

        return start_result

    async def health_check(self) -> ServiceHealth:
        """건강 상태 확인"""
        check_start = time.time()

        try:
            # 서비스별 건강 상태 확인
            is_healthy = await self.check_health()

            response_time_ms = (time.time() - check_start) * 1000
            uptime_seconds = time.time() - self._start_time if self._start_time else 0

            self._health = ServiceHealth(
                service_id=self.config.id,
                status=self._status if is_healthy else ServiceStatus.ERROR,
                last_check=current_timestamp(),
                uptime_seconds=uptime_seconds,
                response_time_ms=response_time_ms,
                error_count=self._health.error_count,
                restart_count=self._health.restart_count,
                metadata=await self.get_health_metadata()
            )

        except Exception as error:
            self._health.error_count += 1
            self._last_error = error

            self.logger.warn(
                f"Health check failed for service {self.config.name}",
                error=error
            )

        return self._health

    def get_status(self) -> ServiceStatus:
        """서비스 상태 조회"""
        return self._status

    def get_metrics(self) -> ServiceMetrics:
        """서비스 메트릭 조회"""
        return self._metrics

    def get_config(self) -> ServiceConfig:
        """서비스 설정 조회"""
        return self.config

    def get_last_error(self) -> Optional[Exception]:
        """마지막 오류 조회"""
        return self._last_error

    @abstractmethod
    async def initialize(self) -> None:
        """서비스 초기화 (하위 클래스에서 구현)"""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """서비스 정리 (하위 클래스에서 구현)"""
        pass

    async def check_health(self) -> bool:
        """건강 상태 확인 (하위 클래스에서 구현 가능)"""
        return self._status == ServiceStatus.RUNNING

    async def get_health_metadata(self) -> KeyValuePair:
        """건강 상태 메타데이터 (하위 클래스에서 구현 가능)"""
        return {}

    def _update_metrics(self, success: bool, response_time_ms: float) -> None:
        """메트릭 업데이트"""
        self._metrics.requests_total += 1
        self._metrics.last_request_at = current_timestamp()

        if success:
            self._metrics.requests_successful += 1
        else:
            self._metrics.requests_failed += 1

        # 평균 응답 시간 업데이트
        total_requests = self._metrics.requests_total
        current_avg = self._metrics.average_response_time_ms
        self._metrics.average_response_time_ms = (
            (current_avg * (total_requests - 1) + response_time_ms) / total_requests
        )

        # 최대 응답 시간 업데이트
        if response_time_ms > self._metrics.peak_response_time_ms:
            self._metrics.peak_response_time_ms = response_time_ms

        # 가동률 계산
        if self._start_time:
            uptime = time.time() - self._start_time
            total_time = uptime + (self._health.restart_count * 60)  # 재시작 시간 추정
            self._metrics.uptime_percentage = (uptime / total_time) * 100 if total_time > 0 else 0


class ServiceManager:
    """서비스 관리자"""

    def __init__(self, events: Optional[EventEmitter] = None):
        self.services: Dict[ID, BaseService] = {}
        self.events = events
        self.logger = create_logger('ServiceManager')
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_initialized = False

    async def initialize(self) -> Result[bool]:
        """서비스 관리자 초기화"""
        if self._is_initialized:
            return Result.success(True)

        try:
            self._is_initialized = True
            return Result.success(True)

        except Exception as error:
            return Result.failure(PacaError(f"Service manager initialization failed: {str(error)}"))

    def register_service(self, service: BaseService) -> None:
        """서비스 등록"""
        self.services[service.config.id] = service
        self.logger.info(f"Registered service: {service.config.name}")

    def unregister_service(self, service_id: ID) -> bool:
        """서비스 등록 해제"""
        if service_id in self.services:
            service = self.services[service_id]
            del self.services[service_id]
            self.logger.info(f"Unregistered service: {service.config.name}")
            return True
        return False

    async def start_service(self, service_id: ID) -> Result[bool]:
        """특정 서비스 시작"""
        if service_id not in self.services:
            return Result.failure(PacaError(f"Service {service_id} not found"))

        service = self.services[service_id]
        return await service.start()

    async def stop_service(self, service_id: ID) -> Result[bool]:
        """특정 서비스 중지"""
        if service_id not in self.services:
            return Result.failure(PacaError(f"Service {service_id} not found"))

        service = self.services[service_id]
        return await service.stop()

    async def start_all(self) -> Dict[ID, Result[bool]]:
        """모든 서비스 시작"""
        results = {}
        for service_id, service in self.services.items():
            if service.config.auto_start:
                results[service_id] = await service.start()
        return results

    async def stop_all(self) -> Dict[ID, Result[bool]]:
        """모든 서비스 중지"""
        results = {}
        for service_id, service in self.services.items():
            results[service_id] = await service.stop()
        return results

    async def health_check_all(self) -> Dict[ID, ServiceHealth]:
        """모든 서비스 건강 상태 확인"""
        health_status = {}
        for service_id, service in self.services.items():
            health_status[service_id] = await service.health_check()
        return health_status

    def get_service_status(self) -> Dict[ID, ServiceStatus]:
        """모든 서비스 상태 조회"""
        return {
            service_id: service.get_status()
            for service_id, service in self.services.items()
        }

    def start_health_monitoring(self, interval_seconds: float = 30.0) -> None:
        """건강 상태 모니터링 시작"""
        if self._health_check_task and not self._health_check_task.done():
            return

        async def health_monitor():
            while True:
                try:
                    health_results = await self.health_check_all()

                    # 건강하지 않은 서비스 재시작 시도
                    for service_id, health in health_results.items():
                        service = self.services[service_id]

                        if (health.status == ServiceStatus.ERROR and
                            service.config.restart_on_failure and
                            service._retry_count < service.config.max_retries):

                            self.logger.warn(
                                f"Service {service.config.name} is unhealthy, attempting restart"
                            )

                            service._retry_count += 1
                            await asyncio.sleep(service.config.retry_delay_seconds)
                            await service.restart()

                    await asyncio.sleep(interval_seconds)

                except asyncio.CancelledError:
                    break
                except Exception as error:
                    self.logger.error("Health monitoring error", error=error)
                    await asyncio.sleep(interval_seconds)

        self._health_check_task = asyncio.create_task(health_monitor())
        self.logger.info("Health monitoring started")

    def stop_health_monitoring(self) -> None:
        """건강 상태 모니터링 중지"""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            self.logger.info("Health monitoring stopped")

    async def shutdown(self) -> Result[bool]:
        """서비스 관리자 종료"""
        try:
            if self._health_check_task and not self._health_check_task.done():
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass

            await self.stop_all()
            self._is_initialized = False
            return Result.success(True)

        except Exception as error:
            return Result.failure(PacaError(f"Service manager shutdown failed: {str(error)}"))


# Additional service classes for enhanced functionality
class ServicePriority(Enum):
    """서비스 우선순위"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ServiceContext:
    """서비스 요청 컨텍스트"""
    request_id: ID = field(default_factory=create_id)
    operation: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[ID] = None
    session_id: Optional[ID] = None
    priority: ServicePriority = ServicePriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Timestamp = field(default_factory=current_timestamp)


@dataclass
class ServiceResult:
    """서비스 처리 결과"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Timestamp = field(default_factory=current_timestamp)
