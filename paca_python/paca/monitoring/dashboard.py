"""
PACA 모니터링 대시보드
실시간 시스템 상태 모니터링 및 시각화
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from .logger import PACALogger, LogLevel
from ..feedback.storage import FeedbackStorage
from ..tools.tool_manager import PACAToolManager


@dataclass
class DashboardMetrics:
    """대시보드 메트릭"""
    timestamp: datetime

    # 시스템 메트릭
    total_sessions: int = 0
    active_sessions: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0

    # 도구 메트릭
    tool_executions: int = 0
    tool_success_rate: float = 0.0
    most_used_tool: Optional[str] = None

    # 피드백 메트릭
    total_feedback: int = 0
    average_rating: float = 0.0
    critical_issues: int = 0

    # 리소스 메트릭
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0

    # 로그 메트릭
    total_logs: int = 0
    error_rate: float = 0.0
    warning_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class HealthStatus:
    """헬스 상태"""
    component: str
    status: str  # healthy, warning, critical, unknown
    message: str
    last_check: datetime
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['last_check'] = self.last_check.isoformat()
        return data


class MonitoringDashboard:
    """모니터링 대시보드"""

    def __init__(
        self,
        logger: PACALogger,
        feedback_storage: Optional[FeedbackStorage] = None,
        tool_manager: Optional[PACAToolManager] = None,
        update_interval: int = 30
    ):
        """
        초기화

        Args:
            logger: PACA 로거
            feedback_storage: 피드백 저장소
            tool_manager: 도구 관리자
            update_interval: 업데이트 간격 (초)
        """
        self.logger = logger
        self.feedback_storage = feedback_storage
        self.tool_manager = tool_manager
        self.update_interval = update_interval

        # 메트릭 저장소
        self.current_metrics: Optional[DashboardMetrics] = None
        self.metrics_history: List[DashboardMetrics] = []
        self.max_history = 1440  # 24시간 (30초 간격)

        # 헬스 상태
        self.health_status: Dict[str, HealthStatus] = {}

        # 백그라운드 작업
        self._running = False
        self._update_task: Optional[asyncio.Task] = None

        # 알림 임계값
        self.alert_thresholds = {
            'error_rate': 5.0,  # 5% 이상
            'response_time': 10.0,  # 10초 이상
            'memory_usage': 80.0,  # 80% 이상
            'cpu_usage': 90.0,  # 90% 이상
            'success_rate': 95.0,  # 95% 미만
        }

    async def start(self):
        """대시보드 시작"""
        if self._running:
            return

        self._running = True
        await self.logger.info("Monitoring dashboard started", component="dashboard")

        # 백그라운드 업데이트 시작
        self._update_task = asyncio.create_task(self._update_loop())

    async def stop(self):
        """대시보드 정지"""
        if not self._running:
            return

        await self.logger.info("Monitoring dashboard stopping", component="dashboard")
        self._running = False

        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

    async def _update_loop(self):
        """백그라운드 업데이트 루프"""
        try:
            while self._running:
                await self.update_metrics()
                await self.check_health()
                await asyncio.sleep(self.update_interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            await self.logger.error(
                f"Error in dashboard update loop: {e}",
                component="dashboard"
            )

    async def update_metrics(self):
        """메트릭 업데이트"""
        try:
            now = datetime.now()
            metrics = DashboardMetrics(timestamp=now)

            # 로그 메트릭 수집
            await self._collect_log_metrics(metrics)

            # 피드백 메트릭 수집
            if self.feedback_storage:
                await self._collect_feedback_metrics(metrics)

            # 도구 메트릭 수집
            if self.tool_manager:
                await self._collect_tool_metrics(metrics)

            # 시스템 리소스 메트릭
            await self._collect_system_metrics(metrics)

            # 현재 메트릭 업데이트
            self.current_metrics = metrics

            # 히스토리에 추가
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history:
                self.metrics_history.pop(0)

            await self.logger.debug(
                "Dashboard metrics updated",
                component="dashboard",
                metadata={'metrics_count': len(self.metrics_history)}
            )

        except Exception as e:
            await self.logger.error(
                f"Failed to update metrics: {e}",
                component="dashboard"
            )

    async def _collect_log_metrics(self, metrics: DashboardMetrics):
        """로그 메트릭 수집"""
        try:
            # 최근 1시간 로그 통계
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)

            log_stats = await self.logger.get_log_stats(start_time, end_time)

            metrics.total_logs = log_stats.get('total_logs', 0)
            metrics.error_rate = log_stats.get('error_rate', 0.0)

            # 경고 수 계산
            by_level = log_stats.get('by_level', {})
            metrics.warning_count = by_level.get('WARNING', 0)

        except Exception as e:
            await self.logger.error(
                f"Failed to collect log metrics: {e}",
                component="dashboard"
            )

    async def _collect_feedback_metrics(self, metrics: DashboardMetrics):
        """피드백 메트릭 수집"""
        try:
            # 최근 24시간 피드백 통계
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)

            stats = await self.feedback_storage.get_feedback_stats(start_time, end_time)

            metrics.total_feedback = stats.total_feedback
            metrics.average_rating = stats.average_rating or 0.0

            # 중요 이슈 카운트 (평점 2점 이하)
            feedback_list = await self.feedback_storage.list_feedback(
                start_date=start_time,
                end_date=end_time,
                limit=1000
            )

            metrics.critical_issues = len([
                f for f in feedback_list
                if f.rating and f.rating <= 2
            ])

        except Exception as e:
            await self.logger.error(
                f"Failed to collect feedback metrics: {e}",
                component="dashboard"
            )

    async def _collect_tool_metrics(self, metrics: DashboardMetrics):
        """도구 메트릭 수집"""
        try:
            # 도구 관리자에서 통계 수집
            if hasattr(self.tool_manager, 'get_execution_stats'):
                stats = await self.tool_manager.get_execution_stats()

                metrics.tool_executions = stats.get('total_executions', 0)
                metrics.tool_success_rate = stats.get('success_rate', 0.0) * 100
                metrics.most_used_tool = stats.get('most_used_tool')

                # 요청 메트릭 계산
                metrics.total_requests = metrics.tool_executions
                successful = int(metrics.tool_executions * metrics.tool_success_rate / 100)
                metrics.successful_requests = successful
                metrics.failed_requests = metrics.tool_executions - successful

        except Exception as e:
            await self.logger.error(
                f"Failed to collect tool metrics: {e}",
                component="dashboard"
            )

    async def _collect_system_metrics(self, metrics: DashboardMetrics):
        """시스템 리소스 메트릭 수집"""
        try:
            import psutil

            # 메모리 사용량
            memory = psutil.virtual_memory()
            metrics.memory_usage_mb = memory.used / (1024 * 1024)

            # CPU 사용률
            metrics.cpu_usage_percent = psutil.cpu_percent(interval=1)

            # 디스크 사용률
            disk = psutil.disk_usage('/')
            metrics.disk_usage_percent = (disk.used / disk.total) * 100

        except Exception as e:
            await self.logger.error(
                f"Failed to collect system metrics: {e}",
                component="dashboard"
            )

    async def check_health(self):
        """헬스 체크"""
        try:
            now = datetime.now()

            # 로거 헬스 체크
            await self._check_logger_health(now)

            # 피드백 시스템 헬스 체크
            if self.feedback_storage:
                await self._check_feedback_health(now)

            # 도구 관리자 헬스 체크
            if self.tool_manager:
                await self._check_tool_manager_health(now)

            # 시스템 리소스 헬스 체크
            await self._check_system_health(now)

        except Exception as e:
            await self.logger.error(
                f"Failed to check health: {e}",
                component="dashboard"
            )

    async def _check_logger_health(self, now: datetime):
        """로거 헬스 체크"""
        try:
            # 최근 로그 활동 확인
            recent_logs = await self.logger.get_logs(
                start_time=now - timedelta(minutes=5),
                limit=1
            )

            if recent_logs:
                status = "healthy"
                message = "Logger is active"
            else:
                status = "warning"
                message = "No recent log activity"

            self.health_status['logger'] = HealthStatus(
                component='logger',
                status=status,
                message=message,
                last_check=now,
                details={'recent_logs_count': len(recent_logs)}
            )

        except Exception as e:
            self.health_status['logger'] = HealthStatus(
                component='logger',
                status='critical',
                message=f"Logger health check failed: {e}",
                last_check=now,
                details={}
            )

    async def _check_feedback_health(self, now: datetime):
        """피드백 시스템 헬스 체크"""
        try:
            # 최근 피드백 확인
            recent_feedback = await self.feedback_storage.list_feedback(
                start_date=now - timedelta(hours=1),
                limit=1
            )

            # 데이터베이스 연결 테스트
            stats = await self.feedback_storage.get_feedback_stats()

            status = "healthy"
            message = "Feedback system is operational"

            self.health_status['feedback'] = HealthStatus(
                component='feedback',
                status=status,
                message=message,
                last_check=now,
                details={
                    'recent_feedback_count': len(recent_feedback),
                    'total_feedback': stats.total_feedback
                }
            )

        except Exception as e:
            self.health_status['feedback'] = HealthStatus(
                component='feedback',
                status='critical',
                message=f"Feedback system health check failed: {e}",
                last_check=now,
                details={}
            )

    async def _check_tool_manager_health(self, now: datetime):
        """도구 관리자 헬스 체크"""
        try:
            # 등록된 도구 확인
            tools_count = len(self.tool_manager.tools)

            if tools_count > 0:
                status = "healthy"
                message = f"{tools_count} tools registered"
            else:
                status = "warning"
                message = "No tools registered"

            self.health_status['tool_manager'] = HealthStatus(
                component='tool_manager',
                status=status,
                message=message,
                last_check=now,
                details={'tools_count': tools_count}
            )

        except Exception as e:
            self.health_status['tool_manager'] = HealthStatus(
                component='tool_manager',
                status='critical',
                message=f"Tool manager health check failed: {e}",
                last_check=now,
                details={}
            )

    async def _check_system_health(self, now: datetime):
        """시스템 헬스 체크"""
        try:
            import psutil

            # 메모리 체크
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # CPU 체크
            cpu_percent = psutil.cpu_percent(interval=1)

            # 디스크 체크
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100

            # 상태 결정
            if (memory_percent > 90 or cpu_percent > 95 or disk_percent > 95):
                status = "critical"
                message = "System resources critically low"
            elif (memory_percent > 80 or cpu_percent > 90 or disk_percent > 90):
                status = "warning"
                message = "System resources running high"
            else:
                status = "healthy"
                message = "System resources normal"

            self.health_status['system'] = HealthStatus(
                component='system',
                status=status,
                message=message,
                last_check=now,
                details={
                    'memory_percent': memory_percent,
                    'cpu_percent': cpu_percent,
                    'disk_percent': disk_percent
                }
            )

        except Exception as e:
            self.health_status['system'] = HealthStatus(
                component='system',
                status='critical',
                message=f"System health check failed: {e}",
                last_check=now,
                details={}
            )

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """대시보드 데이터 조회"""
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'current_metrics': self.current_metrics.to_dict() if self.current_metrics else None,
                'health_status': {
                    name: status.to_dict()
                    for name, status in self.health_status.items()
                },
                'alerts': await self._get_active_alerts(),
                'trends': self._calculate_trends(),
                'system_info': await self._get_system_info()
            }

            return data

        except Exception as e:
            await self.logger.error(
                f"Failed to get dashboard data: {e}",
                component="dashboard"
            )
            return {}

    async def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """활성 알림 조회"""
        alerts = []

        try:
            if not self.current_metrics:
                return alerts

            metrics = self.current_metrics

            # 오류율 체크
            if metrics.error_rate > self.alert_thresholds['error_rate']:
                alerts.append({
                    'type': 'error_rate',
                    'severity': 'warning',
                    'message': f"Error rate is {metrics.error_rate:.1f}% (threshold: {self.alert_thresholds['error_rate']}%)",
                    'value': metrics.error_rate,
                    'threshold': self.alert_thresholds['error_rate']
                })

            # 메모리 사용량 체크
            memory_percent = (metrics.memory_usage_mb / (8 * 1024)) * 100  # 8GB 기준
            if memory_percent > self.alert_thresholds['memory_usage']:
                alerts.append({
                    'type': 'memory_usage',
                    'severity': 'warning',
                    'message': f"Memory usage is {memory_percent:.1f}% (threshold: {self.alert_thresholds['memory_usage']}%)",
                    'value': memory_percent,
                    'threshold': self.alert_thresholds['memory_usage']
                })

            # CPU 사용률 체크
            if metrics.cpu_usage_percent > self.alert_thresholds['cpu_usage']:
                alerts.append({
                    'type': 'cpu_usage',
                    'severity': 'warning',
                    'message': f"CPU usage is {metrics.cpu_usage_percent:.1f}% (threshold: {self.alert_thresholds['cpu_usage']}%)",
                    'value': metrics.cpu_usage_percent,
                    'threshold': self.alert_thresholds['cpu_usage']
                })

            # 성공률 체크
            if metrics.total_requests > 0:
                success_rate = (metrics.successful_requests / metrics.total_requests) * 100
                if success_rate < self.alert_thresholds['success_rate']:
                    alerts.append({
                        'type': 'success_rate',
                        'severity': 'critical',
                        'message': f"Success rate is {success_rate:.1f}% (threshold: {self.alert_thresholds['success_rate']}%)",
                        'value': success_rate,
                        'threshold': self.alert_thresholds['success_rate']
                    })

        except Exception as e:
            await self.logger.error(
                f"Failed to get active alerts: {e}",
                component="dashboard"
            )

        return alerts

    def _calculate_trends(self) -> Dict[str, Any]:
        """트렌드 계산"""
        trends = {}

        try:
            if len(self.metrics_history) < 2:
                return trends

            # 최근 1시간과 이전 1시간 비교
            now = datetime.now()
            one_hour_ago = now - timedelta(hours=1)
            two_hours_ago = now - timedelta(hours=2)

            recent_metrics = [
                m for m in self.metrics_history
                if m.timestamp > one_hour_ago
            ]

            previous_metrics = [
                m for m in self.metrics_history
                if two_hours_ago < m.timestamp <= one_hour_ago
            ]

            if recent_metrics and previous_metrics:
                # 오류율 트렌드
                recent_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
                previous_error_rate = sum(m.error_rate for m in previous_metrics) / len(previous_metrics)

                trends['error_rate'] = {
                    'current': recent_error_rate,
                    'previous': previous_error_rate,
                    'change': recent_error_rate - previous_error_rate,
                    'direction': 'up' if recent_error_rate > previous_error_rate else 'down'
                }

                # 요청 수 트렌드
                recent_requests = sum(m.total_requests for m in recent_metrics)
                previous_requests = sum(m.total_requests for m in previous_metrics)

                trends['request_volume'] = {
                    'current': recent_requests,
                    'previous': previous_requests,
                    'change': recent_requests - previous_requests,
                    'direction': 'up' if recent_requests > previous_requests else 'down'
                }

        except Exception as e:
            # 트렌드 계산 오류는 로깅만 하고 빈 객체 반환
            pass

        return trends

    async def _get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 조회"""
        try:
            import psutil
            import platform

            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
                'uptime_seconds': (datetime.now() - datetime.fromtimestamp(psutil.boot_time())).total_seconds()
            }

        except Exception as e:
            await self.logger.error(
                f"Failed to get system info: {e}",
                component="dashboard"
            )
            return {}

    async def export_metrics(self, file_path: str, hours: int = 24):
        """메트릭 데이터 내보내기"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            export_data = [
                metrics.to_dict()
                for metrics in self.metrics_history
                if metrics.timestamp > cutoff_time
            ]

            export_path = Path(file_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            await self.logger.info(
                f"Metrics exported to {file_path}",
                component="dashboard",
                metadata={
                    'records_count': len(export_data),
                    'hours': hours
                }
            )

        except Exception as e:
            await self.logger.error(
                f"Failed to export metrics: {e}",
                component="dashboard"
            )