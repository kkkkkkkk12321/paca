"""
Analytics Service Module
사용자 행동 분석 및 성과 추적 서비스
TypeScript → Python 완전 변환
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union

from ..core.types import (
    ID, Timestamp, Result, KeyValuePair,
    create_success, create_failure, generate_id, current_timestamp
)
from ..core.errors import ValidationError, ApplicationError
from ..core.events import EventEmitter
from ..core.utils.logger import PacaLogger
from .base import (
    BaseService, ServiceConfig, ServiceContext, ServiceResult, ServicePriority
)


class AnalyticsEventType(Enum):
    """이벤트 타입"""
    USER_LOGIN = 'user_login'
    USER_LOGOUT = 'user_logout'
    SESSION_START = 'session_start'
    SESSION_END = 'session_end'
    QUESTION_ASKED = 'question_asked'
    ANSWER_RECEIVED = 'answer_received'
    KNOWLEDGE_CREATED = 'knowledge_created'
    KNOWLEDGE_REVIEWED = 'knowledge_reviewed'
    LEARNING_GOAL_SET = 'learning_goal_set'
    LEARNING_GOAL_ACHIEVED = 'learning_goal_achieved'
    FEATURE_USED = 'feature_used'
    ERROR_OCCURRED = 'error_occurred'


@dataclass(frozen=True)
class EventContext:
    """이벤트 컨텍스트"""
    user_agent: Optional[str] = None
    ip: Optional[str] = None
    referrer: Optional[str] = None
    platform: Optional[str] = None
    app_version: Optional[str] = None
    device_id: Optional[str] = None
    screen_resolution: Optional[str] = None
    timezone: Optional[str] = None


@dataclass(frozen=True)
class AnalyticsEvent:
    """분석 이벤트"""
    id: ID
    user_id: ID
    type: AnalyticsEventType
    timestamp: Timestamp
    properties: KeyValuePair
    context: EventContext
    session_id: Optional[ID] = None


class MetricType(Enum):
    """메트릭 타입"""
    COUNTER = 'counter'
    GAUGE = 'gauge'
    HISTOGRAM = 'histogram'
    TIMER = 'timer'


@dataclass(frozen=True)
class Metric:
    """메트릭"""
    id: ID
    name: str
    type: MetricType
    value: Union[int, float]
    timestamp: Timestamp
    tags: KeyValuePair = field(default_factory=dict)
    description: Optional[str] = None


@dataclass(frozen=True)
class UserMetrics:
    """사용자 메트릭"""
    user_id: ID
    session_count: int
    total_time_spent: float  # seconds
    questions_asked: int
    knowledge_items_created: int
    learning_goals_achieved: int
    last_activity: Timestamp
    retention_rate: float
    engagement_score: float


@dataclass(frozen=True)
class SystemMetrics:
    """시스템 메트릭"""
    timestamp: Timestamp
    active_users: int
    total_sessions: int
    average_session_duration: float
    total_questions: int
    total_knowledge_items: int
    error_rate: float
    response_time_avg: float
    memory_usage: float
    cpu_usage: float


@dataclass
class AnalyticsRequest:
    """분석 요청"""
    event_type: AnalyticsEventType
    user_id: ID
    properties: KeyValuePair
    context: Optional[EventContext] = None
    session_id: Optional[ID] = None


@dataclass
class AnalyticsQuery:
    """분석 쿼리"""
    user_ids: Optional[List[ID]] = None
    event_types: Optional[List[AnalyticsEventType]] = None
    start_time: Optional[Timestamp] = None
    end_time: Optional[Timestamp] = None
    limit: int = 100
    offset: int = 0
    include_context: bool = False


@dataclass
class AnalyticsReport:
    """분석 리포트"""
    id: ID
    title: str
    generated_at: Timestamp
    time_range: Dict[str, Timestamp]  # start, end
    user_metrics: List[UserMetrics]
    system_metrics: SystemMetrics
    events_analyzed: int
    insights: List[str]
    recommendations: List[str]


class AnalyticsService(BaseService):
    """분석 서비스"""

    def __init__(self, config: Optional[ServiceConfig] = None):
        super().__init__(config or ServiceConfig(
            name="analytics_service",
            version="1.0.0",
            description="사용자 행동 분석 및 성과 추적 서비스"
        ))

        self.logger = PacaLogger("AnalyticsService")
        self.events: List[AnalyticsEvent] = []
        self.metrics: List[Metric] = []
        self.user_sessions: Dict[ID, Dict[str, Any]] = {}

        # 설정
        self.retention_window_days = 30
        self.session_timeout_minutes = 30
        self.max_events_in_memory = 10000

    async def initialize(self) -> Result[bool]:
        """서비스 초기화"""
        try:
            self.logger.info("Analytics Service 초기화 중...")

            # 이벤트 리스너 설정
            self.event_emitter.on('user_action', self._handle_user_action)
            self.event_emitter.on('system_metric', self._handle_system_metric)

            self.logger.info("Analytics Service 초기화 완료")
            return create_success(True)

        except Exception as e:
            self.logger.error(f"Analytics Service 초기화 실패: {e}")
            return create_failure(ApplicationError(f"서비스 초기화 실패: {e}"))

    async def track_event(self, request: AnalyticsRequest) -> Result[AnalyticsEvent]:
        """이벤트 추적"""
        try:
            # 이벤트 생성
            event = AnalyticsEvent(
                id=generate_id(),
                user_id=request.user_id,
                type=request.event_type,
                timestamp=current_timestamp(),
                properties=request.properties,
                context=request.context or EventContext(),
                session_id=request.session_id
            )

            # 이벤트 저장
            self.events.append(event)

            # 메모리 관리
            if len(self.events) > self.max_events_in_memory:
                self.events = self.events[-self.max_events_in_memory//2:]

            # 세션 업데이트
            await self._update_user_session(event)

            # 이벤트 발생
            self.event_emitter.emit('analytics_event_tracked', {
                'event': event,
                'user_id': request.user_id
            })

            self.logger.debug(f"이벤트 추적됨: {event.type.value}")
            return create_success(event)

        except Exception as e:
            self.logger.error(f"이벤트 추적 실패: {e}")
            return create_failure(ApplicationError(f"이벤트 추적 실패: {e}"))

    async def query_events(self, query: AnalyticsQuery) -> Result[List[AnalyticsEvent]]:
        """이벤트 조회"""
        try:
            filtered_events = self.events.copy()

            # 필터링
            if query.user_ids:
                filtered_events = [e for e in filtered_events if e.user_id in query.user_ids]

            if query.event_types:
                filtered_events = [e for e in filtered_events if e.type in query.event_types]

            if query.start_time:
                filtered_events = [e for e in filtered_events if e.timestamp >= query.start_time]

            if query.end_time:
                filtered_events = [e for e in filtered_events if e.timestamp <= query.end_time]

            # 정렬 (최신 순)
            filtered_events.sort(key=lambda e: e.timestamp, reverse=True)

            # 페이지네이션
            start_idx = query.offset
            end_idx = start_idx + query.limit
            result_events = filtered_events[start_idx:end_idx]

            return create_success(result_events)

        except Exception as e:
            self.logger.error(f"이벤트 조회 실패: {e}")
            return create_failure(ApplicationError(f"이벤트 조회 실패: {e}"))

    async def get_user_metrics(self, user_id: ID) -> Result[UserMetrics]:
        """사용자 메트릭 조회"""
        try:
            user_events = [e for e in self.events if e.user_id == user_id]

            if not user_events:
                return create_failure(ValidationError(f"사용자 {user_id}의 이벤트가 없습니다"))

            # 메트릭 계산
            session_count = len(set(e.session_id for e in user_events if e.session_id))
            questions_asked = len([e for e in user_events if e.type == AnalyticsEventType.QUESTION_ASKED])
            knowledge_created = len([e for e in user_events if e.type == AnalyticsEventType.KNOWLEDGE_CREATED])
            goals_achieved = len([e for e in user_events if e.type == AnalyticsEventType.LEARNING_GOAL_ACHIEVED])

            # 총 사용 시간 계산 (세션 기반)
            total_time = 0.0
            for session_id in set(e.session_id for e in user_events if e.session_id):
                session_events = [e for e in user_events if e.session_id == session_id]
                if len(session_events) > 1:
                    session_events.sort(key=lambda e: e.timestamp)
                    total_time += session_events[-1].timestamp - session_events[0].timestamp

            # 마지막 활동
            last_activity = max(e.timestamp for e in user_events)

            # 유지율 계산 (30일 기준)
            thirty_days_ago = current_timestamp() - (30 * 24 * 3600)
            recent_events = [e for e in user_events if e.timestamp >= thirty_days_ago]
            retention_rate = len(recent_events) / max(len(user_events), 1)

            # 참여도 점수 계산
            engagement_score = min(1.0, (questions_asked * 0.3 + knowledge_created * 0.4 + goals_achieved * 0.3) / 10)

            metrics = UserMetrics(
                user_id=user_id,
                session_count=session_count,
                total_time_spent=total_time,
                questions_asked=questions_asked,
                knowledge_items_created=knowledge_created,
                learning_goals_achieved=goals_achieved,
                last_activity=last_activity,
                retention_rate=retention_rate,
                engagement_score=engagement_score
            )

            return create_success(metrics)

        except Exception as e:
            self.logger.error(f"사용자 메트릭 조회 실패: {e}")
            return create_failure(ApplicationError(f"사용자 메트릭 조회 실패: {e}"))

    async def get_system_metrics(self) -> Result[SystemMetrics]:
        """시스템 메트릭 조회"""
        try:
            current_time = current_timestamp()

            # 기본 통계
            total_events = len(self.events)
            unique_users = len(set(e.user_id for e in self.events))
            unique_sessions = len(set(e.session_id for e in self.events if e.session_id))

            # 평균 세션 시간 계산
            session_durations = []
            for session_id in set(e.session_id for e in self.events if e.session_id):
                session_events = [e for e in self.events if e.session_id == session_id]
                if len(session_events) > 1:
                    session_events.sort(key=lambda e: e.timestamp)
                    duration = session_events[-1].timestamp - session_events[0].timestamp
                    session_durations.append(duration)

            avg_session_duration = sum(session_durations) / len(session_durations) if session_durations else 0

            # 질문 수
            total_questions = len([e for e in self.events if e.type == AnalyticsEventType.QUESTION_ASKED])

            # 지식 항목 수
            total_knowledge = len([e for e in self.events if e.type == AnalyticsEventType.KNOWLEDGE_CREATED])

            # 에러율
            error_events = len([e for e in self.events if e.type == AnalyticsEventType.ERROR_OCCURRED])
            error_rate = error_events / max(total_events, 1)

            # 활성 사용자 (지난 24시간)
            day_ago = current_time - (24 * 3600)
            active_users = len(set(e.user_id for e in self.events if e.timestamp >= day_ago))

            # 시스템 메트릭 (모의 값)
            import psutil
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()

            metrics = SystemMetrics(
                timestamp=current_time,
                active_users=active_users,
                total_sessions=unique_sessions,
                average_session_duration=avg_session_duration,
                total_questions=total_questions,
                total_knowledge_items=total_knowledge,
                error_rate=error_rate,
                response_time_avg=0.15,  # 모의 값
                memory_usage=memory_usage,
                cpu_usage=cpu_usage
            )

            return create_success(metrics)

        except Exception as e:
            self.logger.error(f"시스템 메트릭 조회 실패: {e}")
            return create_failure(ApplicationError(f"시스템 메트릭 조회 실패: {e}"))

    async def generate_report(
        self,
        user_ids: Optional[List[ID]] = None,
        days: int = 7
    ) -> Result[AnalyticsReport]:
        """분석 리포트 생성"""
        try:
            current_time = current_timestamp()
            start_time = current_time - (days * 24 * 3600)

            # 기간 내 이벤트 필터링
            period_events = [e for e in self.events if e.timestamp >= start_time]

            if user_ids:
                period_events = [e for e in period_events if e.user_id in user_ids]

            # 사용자 메트릭 수집
            user_metrics = []
            unique_users = set(e.user_id for e in period_events)

            for user_id in unique_users:
                metrics_result = await self.get_user_metrics(user_id)
                if metrics_result.is_success:
                    user_metrics.append(metrics_result.value)

            # 시스템 메트릭
            system_metrics_result = await self.get_system_metrics()
            system_metrics = system_metrics_result.value if system_metrics_result.is_success else None

            # 인사이트 생성
            insights = self._generate_insights(period_events, user_metrics)

            # 추천사항 생성
            recommendations = self._generate_recommendations(user_metrics, system_metrics)

            report = AnalyticsReport(
                id=generate_id(),
                title=f"{days}일간 분석 리포트",
                generated_at=current_time,
                time_range={"start": start_time, "end": current_time},
                user_metrics=user_metrics,
                system_metrics=system_metrics,
                events_analyzed=len(period_events),
                insights=insights,
                recommendations=recommendations
            )

            return create_success(report)

        except Exception as e:
            self.logger.error(f"리포트 생성 실패: {e}")
            return create_failure(ApplicationError(f"리포트 생성 실패: {e}"))

    async def _update_user_session(self, event: AnalyticsEvent) -> None:
        """사용자 세션 업데이트"""
        if not event.session_id:
            return

        session_key = f"{event.user_id}:{event.session_id}"

        if session_key not in self.user_sessions:
            self.user_sessions[session_key] = {
                'start_time': event.timestamp,
                'last_activity': event.timestamp,
                'event_count': 0
            }

        self.user_sessions[session_key]['last_activity'] = event.timestamp
        self.user_sessions[session_key]['event_count'] += 1

    def _generate_insights(self, events: List[AnalyticsEvent], user_metrics: List[UserMetrics]) -> List[str]:
        """인사이트 생성"""
        insights = []

        if not events:
            return ["분석할 데이터가 없습니다."]

        # 가장 활발한 시간대
        hour_counts = {}
        for event in events:
            hour = datetime.fromtimestamp(event.timestamp).hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1

        if hour_counts:
            peak_hour = max(hour_counts, key=hour_counts.get)
            insights.append(f"가장 활발한 시간대는 {peak_hour}시입니다.")

        # 평균 참여도
        if user_metrics:
            avg_engagement = sum(m.engagement_score for m in user_metrics) / len(user_metrics)
            insights.append(f"평균 사용자 참여도는 {avg_engagement:.2f}입니다.")

        # 질문 패턴
        question_events = [e for e in events if e.type == AnalyticsEventType.QUESTION_ASKED]
        if question_events:
            insights.append(f"분석 기간 동안 총 {len(question_events)}개의 질문이 있었습니다.")

        return insights

    def _generate_recommendations(self, user_metrics: List[UserMetrics], system_metrics: Optional[SystemMetrics]) -> List[str]:
        """추천사항 생성"""
        recommendations = []

        if not user_metrics:
            return ["사용자 데이터가 부족합니다."]

        # 참여도 기반 추천
        low_engagement_users = [m for m in user_metrics if m.engagement_score < 0.3]
        if len(low_engagement_users) > len(user_metrics) * 0.3:
            recommendations.append("참여도가 낮은 사용자들을 위한 온보딩 개선이 필요합니다.")

        # 시스템 성능 기반 추천
        if system_metrics:
            if system_metrics.error_rate > 0.05:
                recommendations.append("에러율이 높습니다. 시스템 안정성 개선이 필요합니다.")

            if system_metrics.response_time_avg > 1.0:
                recommendations.append("응답 시간이 느립니다. 성능 최적화를 고려해보세요.")

        return recommendations

    async def _handle_user_action(self, data: Dict[str, Any]) -> None:
        """사용자 액션 처리"""
        try:
            event_type_map = {
                'login': AnalyticsEventType.USER_LOGIN,
                'logout': AnalyticsEventType.USER_LOGOUT,
                'question': AnalyticsEventType.QUESTION_ASKED,
                'answer': AnalyticsEventType.ANSWER_RECEIVED
            }

            action = data.get('action')
            if action in event_type_map:
                request = AnalyticsRequest(
                    event_type=event_type_map[action],
                    user_id=data.get('user_id'),
                    properties=data.get('properties', {}),
                    session_id=data.get('session_id')
                )
                await self.track_event(request)

        except Exception as e:
            self.logger.error(f"사용자 액션 처리 실패: {e}")

    async def _handle_system_metric(self, data: Dict[str, Any]) -> None:
        """시스템 메트릭 처리"""
        try:
            metric = Metric(
                id=generate_id(),
                name=data.get('name', 'unknown'),
                type=MetricType(data.get('type', 'gauge')),
                value=data.get('value', 0),
                timestamp=current_timestamp(),
                tags=data.get('tags', {}),
                description=data.get('description')
            )

            self.metrics.append(metric)

            # 메모리 관리
            if len(self.metrics) > self.max_events_in_memory:
                self.metrics = self.metrics[-self.max_events_in_memory//2:]

        except Exception as e:
            self.logger.error(f"시스템 메트릭 처리 실패: {e}")

    async def cleanup(self) -> Result[bool]:
        """서비스 정리"""
        try:
            self.logger.info("Analytics Service 정리 중...")

            # 데이터 정리
            self.events.clear()
            self.metrics.clear()
            self.user_sessions.clear()

            self.logger.info("Analytics Service 정리 완료")
            return create_success(True)

        except Exception as e:
            self.logger.error(f"Analytics Service 정리 실패: {e}")
            return create_failure(ApplicationError(f"서비스 정리 실패: {e}"))