"""
Database Monitor
데이터베이스 성능 모니터링 및 헬스체크
"""

import asyncio
import time
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum

from ...core.types import Result
from ...core.utils.logger import PacaLogger
from .sql_connector import SQLConnector
from .nosql_connector import NoSQLConnector
from .connection_pool import ConnectionPool


class AlertLevel(Enum):
    """알림 레벨"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """성능 메트릭"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None


@dataclass
class HealthCheckResult:
    """헬스체크 결과"""
    database_name: str
    status: str
    response_time: float
    error: Optional[str] = None
    metrics: List[PerformanceMetric] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Alert:
    """알림"""
    id: str
    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class DatabaseMonitor:
    """데이터베이스 모니터"""

    def __init__(
        self,
        check_interval: float = 60.0,
        retention_hours: int = 24
    ):
        self.check_interval = check_interval
        self.retention_hours = retention_hours
        self.logger = PacaLogger("DatabaseMonitor")

        # 모니터링 대상들
        self.monitored_databases: Dict[str, Union[SQLConnector, NoSQLConnector, ConnectionPool]] = {}

        # 메트릭 저장소
        self.metrics_history: Dict[str, List[PerformanceMetric]] = {}
        self.health_history: Dict[str, List[HealthCheckResult]] = {}

        # 알림 시스템
        self.alerts: Dict[str, Alert] = {}
        self.alert_callbacks: List[callable] = []

        # 임계값 설정
        self.thresholds = {
            "response_time": {"warning": 1.0, "critical": 5.0},
            "cpu_usage": {"warning": 70.0, "critical": 90.0},
            "memory_usage": {"warning": 80.0, "critical": 95.0},
            "disk_usage": {"warning": 85.0, "critical": 95.0},
            "connection_usage": {"warning": 80.0, "critical": 95.0},
            "error_rate": {"warning": 5.0, "critical": 10.0}
        }

        # 모니터링 태스크
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False

    def add_database(
        self,
        name: str,
        database: Union[SQLConnector, NoSQLConnector, ConnectionPool]
    ) -> None:
        """모니터링 대상 데이터베이스 추가"""
        self.monitored_databases[name] = database
        self.metrics_history[name] = []
        self.health_history[name] = []
        self.logger.info(f"Added database to monitoring: {name}")

    def remove_database(self, name: str) -> None:
        """모니터링 대상 데이터베이스 제거"""
        if name in self.monitored_databases:
            del self.monitored_databases[name]
            if name in self.metrics_history:
                del self.metrics_history[name]
            if name in self.health_history:
                del self.health_history[name]
            self.logger.info(f"Removed database from monitoring: {name}")

    def add_alert_callback(self, callback: callable) -> None:
        """알림 콜백 추가"""
        self.alert_callbacks.append(callback)

    def set_threshold(self, metric_name: str, warning: float, critical: float) -> None:
        """임계값 설정"""
        self.thresholds[metric_name] = {
            "warning": warning,
            "critical": critical
        }

    async def start_monitoring(self) -> Result[bool]:
        """모니터링 시작"""
        try:
            if self.is_monitoring:
                return Result(False, False, "Monitoring already started")

            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())

            self.logger.info("Database monitoring started")
            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"Failed to start monitoring: {str(e)}")

    async def stop_monitoring(self) -> Result[bool]:
        """모니터링 중지"""
        try:
            self.is_monitoring = False

            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass

            self.logger.info("Database monitoring stopped")
            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"Failed to stop monitoring: {str(e)}")

    async def _monitoring_loop(self) -> None:
        """모니터링 루프"""
        while self.is_monitoring:
            try:
                # 모든 데이터베이스 헬스체크
                for db_name, database in self.monitored_databases.items():
                    await self._perform_health_check(db_name, database)

                # 시스템 메트릭 수집
                await self._collect_system_metrics()

                # 알림 검사
                await self._check_alerts()

                # 데이터 정리
                await self._cleanup_old_data()

                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {str(e)}")
                await asyncio.sleep(self.check_interval)

    async def _perform_health_check(
        self,
        db_name: str,
        database: Union[SQLConnector, NoSQLConnector, ConnectionPool]
    ) -> None:
        """개별 데이터베이스 헬스체크"""
        try:
            start_time = time.time()

            # 헬스체크 실행
            if isinstance(database, ConnectionPool):
                # 연결 풀의 경우 통계 정보 활용
                pool_stats = database.get_pool_stats()
                response_time = 0.1  # 연결 풀은 즉시 응답
                status = "healthy" if pool_stats["is_initialized"] else "unhealthy"
                error = None

                # 연결 풀 메트릭 추가
                metrics = [
                    PerformanceMetric(
                        name="pool_utilization",
                        value=pool_stats["pool_utilization"] * 100,
                        unit="%",
                        timestamp=datetime.now(),
                        threshold_warning=self.thresholds.get("connection_usage", {}).get("warning"),
                        threshold_critical=self.thresholds.get("connection_usage", {}).get("critical")
                    ),
                    PerformanceMetric(
                        name="active_connections",
                        value=pool_stats["active_connections"],
                        unit="count",
                        timestamp=datetime.now()
                    ),
                    PerformanceMetric(
                        name="available_connections",
                        value=pool_stats["available_connections"],
                        unit="count",
                        timestamp=datetime.now()
                    )
                ]

            else:
                # 일반 커넥터의 경우 헬스체크 실행
                health_result = await database.health_check()
                response_time = time.time() - start_time

                if health_result.is_success:
                    status = health_result.data.get("status", "unknown")
                    error = None
                else:
                    status = "unhealthy"
                    error = health_result.error

                # 기본 메트릭들
                metrics = [
                    PerformanceMetric(
                        name="response_time",
                        value=response_time * 1000,  # ms로 변환
                        unit="ms",
                        timestamp=datetime.now(),
                        threshold_warning=self.thresholds.get("response_time", {}).get("warning", 1000),
                        threshold_critical=self.thresholds.get("response_time", {}).get("critical", 5000)
                    )
                ]

                # 커넥터별 추가 메트릭
                if hasattr(database, 'get_stats'):
                    stats = database.get_stats()
                    if 'success_rate' in stats:
                        error_rate = (1 - stats['success_rate']) * 100
                        metrics.append(
                            PerformanceMetric(
                                name="error_rate",
                                value=error_rate,
                                unit="%",
                                timestamp=datetime.now(),
                                threshold_warning=self.thresholds.get("error_rate", {}).get("warning"),
                                threshold_critical=self.thresholds.get("error_rate", {}).get("critical")
                            )
                        )

            # 헬스체크 결과 저장
            health_check = HealthCheckResult(
                database_name=db_name,
                status=status,
                response_time=response_time,
                error=error,
                metrics=metrics
            )

            self.health_history[db_name].append(health_check)

            # 메트릭 히스토리에 추가
            for metric in metrics:
                self.metrics_history[db_name].append(metric)

        except Exception as e:
            self.logger.error(f"Health check failed for {db_name}: {str(e)}")

            # 실패한 헬스체크도 기록
            health_check = HealthCheckResult(
                database_name=db_name,
                status="error",
                response_time=0.0,
                error=str(e)
            )
            self.health_history[db_name].append(health_check)

    async def _collect_system_metrics(self) -> None:
        """시스템 메트릭 수집"""
        try:
            current_time = datetime.now()

            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_metric = PerformanceMetric(
                name="cpu_usage",
                value=cpu_percent,
                unit="%",
                timestamp=current_time,
                threshold_warning=self.thresholds.get("cpu_usage", {}).get("warning"),
                threshold_critical=self.thresholds.get("cpu_usage", {}).get("critical")
            )

            # 메모리 사용률
            memory = psutil.virtual_memory()
            memory_metric = PerformanceMetric(
                name="memory_usage",
                value=memory.percent,
                unit="%",
                timestamp=current_time,
                threshold_warning=self.thresholds.get("memory_usage", {}).get("warning"),
                threshold_critical=self.thresholds.get("memory_usage", {}).get("critical")
            )

            # 디스크 사용률
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_metric = PerformanceMetric(
                name="disk_usage",
                value=disk_percent,
                unit="%",
                timestamp=current_time,
                threshold_warning=self.thresholds.get("disk_usage", {}).get("warning"),
                threshold_critical=self.thresholds.get("disk_usage", {}).get("critical")
            )

            # 시스템 메트릭 저장
            system_metrics = [cpu_metric, memory_metric, disk_metric]
            if "system" not in self.metrics_history:
                self.metrics_history["system"] = []

            self.metrics_history["system"].extend(system_metrics)

        except Exception as e:
            self.logger.error(f"System metrics collection failed: {str(e)}")

    async def _check_alerts(self) -> None:
        """알림 검사"""
        try:
            current_time = datetime.now()

            # 모든 메트릭에 대해 임계값 검사
            for db_name, metrics in self.metrics_history.items():
                # 최근 메트릭만 검사 (마지막 5분)
                recent_metrics = [
                    m for m in metrics
                    if current_time - m.timestamp < timedelta(minutes=5)
                ]

                for metric in recent_metrics:
                    await self._check_metric_threshold(db_name, metric)

        except Exception as e:
            self.logger.error(f"Alert checking failed: {str(e)}")

    async def _check_metric_threshold(self, db_name: str, metric: PerformanceMetric) -> None:
        """개별 메트릭 임계값 검사"""
        alert_id = f"{db_name}_{metric.name}"

        # 기존 알림이 있는지 확인
        existing_alert = self.alerts.get(alert_id)

        # 임계값 확인
        level = None
        threshold_value = None

        if metric.threshold_critical and metric.value >= metric.threshold_critical:
            level = AlertLevel.CRITICAL
            threshold_value = metric.threshold_critical
        elif metric.threshold_warning and metric.value >= metric.threshold_warning:
            level = AlertLevel.WARNING
            threshold_value = metric.threshold_warning

        if level:
            # 새로운 알림 또는 레벨 상승
            if not existing_alert or existing_alert.level.value != level.value:
                alert = Alert(
                    id=alert_id,
                    level=level,
                    message=f"{db_name} {metric.name} is {metric.value}{metric.unit} (threshold: {threshold_value}{metric.unit})",
                    metric_name=metric.name,
                    current_value=metric.value,
                    threshold_value=threshold_value,
                    timestamp=datetime.now()
                )

                self.alerts[alert_id] = alert
                await self._trigger_alert(alert)

        else:
            # 임계값 아래로 떨어짐 - 알림 해제
            if existing_alert and not existing_alert.resolved:
                existing_alert.resolved = True
                existing_alert.resolved_at = datetime.now()
                await self._resolve_alert(existing_alert)

    async def _trigger_alert(self, alert: Alert) -> None:
        """알림 발생"""
        self.logger.log(
            level=alert.level.value.upper(),
            msg=f"ALERT: {alert.message}"
        )

        # 등록된 콜백 호출
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {str(e)}")

    async def _resolve_alert(self, alert: Alert) -> None:
        """알림 해제"""
        self.logger.info(f"RESOLVED: {alert.message}")

        # 해제 콜백도 호출
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                self.logger.error(f"Alert resolution callback failed: {str(e)}")

    async def _cleanup_old_data(self) -> None:
        """오래된 데이터 정리"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)

            # 메트릭 히스토리 정리
            for db_name in self.metrics_history:
                self.metrics_history[db_name] = [
                    m for m in self.metrics_history[db_name]
                    if m.timestamp > cutoff_time
                ]

            # 헬스체크 히스토리 정리
            for db_name in self.health_history:
                self.health_history[db_name] = [
                    h for h in self.health_history[db_name]
                    if h.timestamp > cutoff_time
                ]

            # 해결된 알림 정리 (24시간 후)
            alert_cutoff = datetime.now() - timedelta(hours=24)
            resolved_alerts = [
                alert_id for alert_id, alert in self.alerts.items()
                if alert.resolved and alert.resolved_at and alert.resolved_at < alert_cutoff
            ]

            for alert_id in resolved_alerts:
                del self.alerts[alert_id]

        except Exception as e:
            self.logger.error(f"Data cleanup failed: {str(e)}")

    def get_database_status(self, db_name: str) -> Optional[Dict[str, Any]]:
        """데이터베이스 상태 조회"""
        if db_name not in self.health_history:
            return None

        recent_checks = self.health_history[db_name][-10:]  # 최근 10개
        if not recent_checks:
            return None

        latest_check = recent_checks[-1]

        # 가용성 계산
        healthy_count = sum(1 for check in recent_checks if check.status == "healthy")
        availability = (healthy_count / len(recent_checks)) * 100

        # 평균 응답 시간
        avg_response_time = sum(check.response_time for check in recent_checks) / len(recent_checks)

        return {
            "database_name": db_name,
            "current_status": latest_check.status,
            "availability": availability,
            "average_response_time": avg_response_time,
            "last_check": latest_check.timestamp.isoformat(),
            "error": latest_check.error
        }

    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """모든 데이터베이스 상태"""
        return {
            db_name: self.get_database_status(db_name)
            for db_name in self.monitored_databases.keys()
        }

    def get_alerts(self, include_resolved: bool = False) -> List[Alert]:
        """알림 목록 조회"""
        alerts = list(self.alerts.values())
        if not include_resolved:
            alerts = [alert for alert in alerts if not alert.resolved]
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def get_metrics(
        self,
        db_name: str,
        metric_name: Optional[str] = None,
        hours: int = 1
    ) -> List[PerformanceMetric]:
        """메트릭 데이터 조회"""
        if db_name not in self.metrics_history:
            return []

        cutoff_time = datetime.now() - timedelta(hours=hours)
        metrics = [
            m for m in self.metrics_history[db_name]
            if m.timestamp > cutoff_time
        ]

        if metric_name:
            metrics = [m for m in metrics if m.name == metric_name]

        return sorted(metrics, key=lambda m: m.timestamp)

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """모니터링 요약 정보"""
        total_databases = len(self.monitored_databases)
        healthy_databases = sum(
            1 for status in self.get_all_statuses().values()
            if status and status["current_status"] == "healthy"
        )

        active_alerts = len([a for a in self.alerts.values() if not a.resolved])
        critical_alerts = len([
            a for a in self.alerts.values()
            if not a.resolved and a.level == AlertLevel.CRITICAL
        ])

        return {
            "is_monitoring": self.is_monitoring,
            "check_interval": self.check_interval,
            "retention_hours": self.retention_hours,
            "total_databases": total_databases,
            "healthy_databases": healthy_databases,
            "active_alerts": active_alerts,
            "critical_alerts": critical_alerts,
            "monitored_databases": list(self.monitored_databases.keys()),
            "last_update": datetime.now().isoformat()
        }


# 팩토리 함수
def create_database_monitor(
    check_interval: float = 60.0,
    retention_hours: int = 24
) -> DatabaseMonitor:
    """데이터베이스 모니터 생성"""
    return DatabaseMonitor(
        check_interval=check_interval,
        retention_hours=retention_hours
    )


# 알림 콜백 예시
async def log_alert_callback(alert: Alert) -> None:
    """로그 알림 콜백"""
    logger = PacaLogger("AlertCallback")
    if alert.resolved:
        logger.info(f"Alert resolved: {alert.message}")
    else:
        logger.warning(f"Alert triggered: {alert.message}")


async def email_alert_callback(alert: Alert) -> None:
    """이메일 알림 콜백 (예시)"""
    # 실제 구현에서는 이메일 전송 로직 추가
    if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
        print(f"EMAIL ALERT: {alert.message}")  # 실제로는 이메일 전송