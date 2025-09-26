"""
Notification Service Module
이메일, 푸시, 인앱 알림을 관리하는 서비스
TypeScript → Python 완전 변환
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

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


class NotificationType(Enum):
    """알림 타입"""
    EMAIL = 'email'
    PUSH = 'push'
    IN_APP = 'in_app'
    SMS = 'sms'


class NotificationPriority(Enum):
    """알림 우선순위"""
    LOW = 'low'
    NORMAL = 'normal'
    HIGH = 'high'
    URGENT = 'urgent'


class NotificationStatus(Enum):
    """알림 상태"""
    PENDING = 'pending'
    SENT = 'sent'
    DELIVERED = 'delivered'
    READ = 'read'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


@dataclass(frozen=True)
class NotificationTemplate:
    """알림 템플릿"""
    id: ID
    name: str
    type: NotificationType
    subject_template: str
    content_template: str
    variables: List[str] = field(default_factory=list)
    is_active: bool = True


@dataclass(frozen=True)
class NotificationChannel:
    """알림 채널"""
    id: ID
    type: NotificationType
    name: str
    config: KeyValuePair = field(default_factory=dict)
    is_enabled: bool = True
    rate_limit: Optional[int] = None  # per hour


@dataclass
class Notification:
    """알림"""
    id: ID
    user_id: ID
    type: NotificationType
    priority: NotificationPriority
    title: str
    content: str
    created_at: Timestamp
    scheduled_at: Optional[Timestamp] = None
    sent_at: Optional[Timestamp] = None
    delivered_at: Optional[Timestamp] = None
    read_at: Optional[Timestamp] = None
    status: NotificationStatus = NotificationStatus.PENDING
    metadata: KeyValuePair = field(default_factory=dict)
    template_id: Optional[ID] = None
    channel_id: Optional[ID] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class NotificationRequest:
    """알림 요청"""
    user_id: ID
    type: NotificationType
    title: str
    content: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    scheduled_at: Optional[Timestamp] = None
    metadata: KeyValuePair = field(default_factory=dict)
    template_id: Optional[ID] = None
    template_variables: KeyValuePair = field(default_factory=dict)


@dataclass
class NotificationQuery:
    """알림 쿼리"""
    user_ids: Optional[List[ID]] = None
    types: Optional[List[NotificationType]] = None
    statuses: Optional[List[NotificationStatus]] = None
    priorities: Optional[List[NotificationPriority]] = None
    start_date: Optional[Timestamp] = None
    end_date: Optional[Timestamp] = None
    limit: int = 50
    offset: int = 0
    include_read: bool = True


@dataclass
class NotificationStats:
    """알림 통계"""
    total_sent: int
    total_delivered: int
    total_read: int
    total_failed: int
    delivery_rate: float
    read_rate: float
    by_type: Dict[str, int] = field(default_factory=dict)
    by_priority: Dict[str, int] = field(default_factory=dict)


class NotificationService(BaseService):
    """알림 서비스"""

    def __init__(self, config: Optional[ServiceConfig] = None):
        super().__init__(config or ServiceConfig(
            name="notification_service",
            version="1.0.0",
            description="이메일, 푸시, 인앱 알림 관리 서비스"
        ))

        self.logger = PacaLogger("NotificationService")
        self.notifications: Dict[ID, Notification] = {}
        self.templates: Dict[ID, NotificationTemplate] = {}
        self.channels: Dict[ID, NotificationChannel] = {}
        self.user_preferences: Dict[ID, Dict[str, Any]] = {}

        # 발송 대기열
        self.pending_queue: List[ID] = []
        self.scheduled_queue: List[ID] = []

        # 이벤트 핸들러
        self.delivery_handlers: Dict[NotificationType, Callable] = {}

        # 설정
        self.batch_size = 50
        self.processing_interval = 30  # seconds
        self.retry_delay_base = 60  # seconds

    async def initialize(self) -> Result[bool]:
        """서비스 초기화"""
        try:
            self.logger.info("Notification Service 초기화 중...")

            # 기본 채널 설정
            await self._setup_default_channels()

            # 기본 템플릿 설정
            await self._setup_default_templates()

            # 발송 핸들러 등록
            self._register_delivery_handlers()

            # 백그라운드 작업 시작
            asyncio.create_task(self._process_notifications())
            asyncio.create_task(self._process_scheduled_notifications())

            # 이벤트 리스너 설정
            self.event_emitter.on('user_action', self._handle_user_action)

            self.logger.info("Notification Service 초기화 완료")
            return create_success(True)

        except Exception as e:
            self.logger.error(f"Notification Service 초기화 실패: {e}")
            return create_failure(ApplicationError(f"서비스 초기화 실패: {e}"))

    async def send_notification(self, request: NotificationRequest) -> Result[Notification]:
        """알림 발송"""
        try:
            # 템플릿 적용
            title, content = await self._apply_template(request)

            # 알림 생성
            notification = Notification(
                id=generate_id(),
                user_id=request.user_id,
                type=request.type,
                priority=request.priority,
                title=title,
                content=content,
                created_at=current_timestamp(),
                scheduled_at=request.scheduled_at,
                metadata=request.metadata,
                template_id=request.template_id
            )

            # 사용자 설정 확인
            if not await self._check_user_preferences(notification):
                notification.status = NotificationStatus.CANCELLED
                self.logger.info(f"알림 취소됨 (사용자 설정): {notification.id}")
                return create_success(notification)

            # 저장
            self.notifications[notification.id] = notification

            # 스케줄 처리
            if notification.scheduled_at and notification.scheduled_at > current_timestamp():
                self.scheduled_queue.append(notification.id)
                self.logger.info(f"알림 스케줄됨: {notification.id}")
            else:
                self.pending_queue.append(notification.id)
                self.logger.info(f"알림 대기열 추가됨: {notification.id}")

            # 이벤트 발생
            self.event_emitter.emit('notification_created', {
                'notification_id': notification.id,
                'user_id': notification.user_id,
                'type': notification.type.value
            })

            return create_success(notification)

        except Exception as e:
            self.logger.error(f"알림 발송 실패: {e}")
            return create_failure(ApplicationError(f"알림 발송 실패: {e}"))

    async def get_notifications(self, query: NotificationQuery) -> Result[List[Notification]]:
        """알림 조회"""
        try:
            notifications = list(self.notifications.values())

            # 필터링
            if query.user_ids:
                notifications = [n for n in notifications if n.user_id in query.user_ids]

            if query.types:
                notifications = [n for n in notifications if n.type in query.types]

            if query.statuses:
                notifications = [n for n in notifications if n.status in query.statuses]

            if query.priorities:
                notifications = [n for n in notifications if n.priority in query.priorities]

            if query.start_date:
                notifications = [n for n in notifications if n.created_at >= query.start_date]

            if query.end_date:
                notifications = [n for n in notifications if n.created_at <= query.end_date]

            if not query.include_read:
                notifications = [n for n in notifications if n.status != NotificationStatus.READ]

            # 정렬 (최신 순)
            notifications.sort(key=lambda n: n.created_at, reverse=True)

            # 페이지네이션
            start_idx = query.offset
            end_idx = start_idx + query.limit
            result = notifications[start_idx:end_idx]

            return create_success(result)

        except Exception as e:
            self.logger.error(f"알림 조회 실패: {e}")
            return create_failure(ApplicationError(f"알림 조회 실패: {e}"))

    async def mark_as_read(self, notification_id: ID, user_id: ID) -> Result[bool]:
        """알림 읽음 처리"""
        try:
            if notification_id not in self.notifications:
                return create_failure(ValidationError("알림을 찾을 수 없습니다"))

            notification = self.notifications[notification_id]

            if notification.user_id != user_id:
                return create_failure(ValidationError("권한이 없습니다"))

            if notification.status == NotificationStatus.READ:
                return create_success(True)  # 이미 읽음

            # 상태 업데이트
            notification.status = NotificationStatus.READ
            notification.read_at = current_timestamp()

            # 이벤트 발생
            self.event_emitter.emit('notification_read', {
                'notification_id': notification_id,
                'user_id': user_id
            })

            self.logger.debug(f"알림 읽음 처리됨: {notification_id}")
            return create_success(True)

        except Exception as e:
            self.logger.error(f"알림 읽음 처리 실패: {e}")
            return create_failure(ApplicationError(f"알림 읽음 처리 실패: {e}"))

    async def get_unread_count(self, user_id: ID) -> Result[int]:
        """읽지 않은 알림 수 조회"""
        try:
            unread_notifications = [
                n for n in self.notifications.values()
                if n.user_id == user_id and n.status != NotificationStatus.READ
            ]

            return create_success(len(unread_notifications))

        except Exception as e:
            self.logger.error(f"읽지 않은 알림 수 조회 실패: {e}")
            return create_failure(ApplicationError(f"읽지 않은 알림 수 조회 실패: {e}"))

    async def get_statistics(self, user_id: Optional[ID] = None) -> Result[NotificationStats]:
        """알림 통계 조회"""
        try:
            notifications = list(self.notifications.values())

            if user_id:
                notifications = [n for n in notifications if n.user_id == user_id]

            total_sent = len([n for n in notifications if n.status != NotificationStatus.PENDING])
            total_delivered = len([n for n in notifications if n.status in [
                NotificationStatus.DELIVERED, NotificationStatus.READ
            ]])
            total_read = len([n for n in notifications if n.status == NotificationStatus.READ])
            total_failed = len([n for n in notifications if n.status == NotificationStatus.FAILED])

            delivery_rate = total_delivered / max(total_sent, 1)
            read_rate = total_read / max(total_delivered, 1)

            # 타입별 통계
            by_type = {}
            for notification_type in NotificationType:
                by_type[notification_type.value] = len([
                    n for n in notifications if n.type == notification_type
                ])

            # 우선순위별 통계
            by_priority = {}
            for priority in NotificationPriority:
                by_priority[priority.value] = len([
                    n for n in notifications if n.priority == priority
                ])

            stats = NotificationStats(
                total_sent=total_sent,
                total_delivered=total_delivered,
                total_read=total_read,
                total_failed=total_failed,
                delivery_rate=delivery_rate,
                read_rate=read_rate,
                by_type=by_type,
                by_priority=by_priority
            )

            return create_success(stats)

        except Exception as e:
            self.logger.error(f"알림 통계 조회 실패: {e}")
            return create_failure(ApplicationError(f"알림 통계 조회 실패: {e}"))

    async def _process_notifications(self) -> None:
        """알림 처리 백그라운드 작업"""
        while True:
            try:
                if not self.pending_queue:
                    await asyncio.sleep(self.processing_interval)
                    continue

                # 배치 처리
                batch = self.pending_queue[:self.batch_size]
                self.pending_queue = self.pending_queue[self.batch_size:]

                for notification_id in batch:
                    if notification_id in self.notifications:
                        await self._deliver_notification(self.notifications[notification_id])

                await asyncio.sleep(1)  # 짧은 대기

            except Exception as e:
                self.logger.error(f"알림 처리 오류: {e}")
                await asyncio.sleep(self.processing_interval)

    async def _process_scheduled_notifications(self) -> None:
        """스케줄된 알림 처리"""
        while True:
            try:
                current_time = current_timestamp()
                ready_notifications = []

                for notification_id in self.scheduled_queue.copy():
                    if notification_id in self.notifications:
                        notification = self.notifications[notification_id]
                        if notification.scheduled_at and notification.scheduled_at <= current_time:
                            ready_notifications.append(notification_id)
                            self.scheduled_queue.remove(notification_id)

                # 준비된 알림을 대기열에 추가
                self.pending_queue.extend(ready_notifications)

                await asyncio.sleep(60)  # 1분마다 확인

            except Exception as e:
                self.logger.error(f"스케줄된 알림 처리 오류: {e}")
                await asyncio.sleep(60)

    async def _deliver_notification(self, notification: Notification) -> bool:
        """알림 발송"""
        try:
            handler = self.delivery_handlers.get(notification.type)
            if not handler:
                self.logger.warning(f"지원되지 않는 알림 타입: {notification.type}")
                notification.status = NotificationStatus.FAILED
                return False

            # 발송 시도
            success = await handler(notification)

            if success:
                notification.status = NotificationStatus.SENT
                notification.sent_at = current_timestamp()

                # 타입에 따라 즉시 전달됨으로 처리
                if notification.type == NotificationType.IN_APP:
                    notification.status = NotificationStatus.DELIVERED
                    notification.delivered_at = current_timestamp()

                self.logger.info(f"알림 발송 성공: {notification.id}")

                # 이벤트 발생
                self.event_emitter.emit('notification_sent', {
                    'notification_id': notification.id,
                    'user_id': notification.user_id,
                    'type': notification.type.value
                })

                return True
            else:
                # 재시도 로직
                notification.retry_count += 1
                if notification.retry_count < notification.max_retries:
                    # 지수 백오프로 재시도 스케줄
                    delay = self.retry_delay_base * (2 ** notification.retry_count)
                    notification.scheduled_at = current_timestamp() + delay
                    self.scheduled_queue.append(notification.id)
                    self.logger.info(f"알림 재시도 스케줄됨: {notification.id}")
                else:
                    notification.status = NotificationStatus.FAILED
                    self.logger.error(f"알림 발송 최종 실패: {notification.id}")

                return False

        except Exception as e:
            self.logger.error(f"알림 발송 오류: {e}")
            notification.status = NotificationStatus.FAILED
            return False

    async def _apply_template(self, request: NotificationRequest) -> tuple[str, str]:
        """템플릿 적용"""
        if not request.template_id or request.template_id not in self.templates:
            return request.title, request.content

        template = self.templates[request.template_id]

        title = template.subject_template
        content = template.content_template

        # 변수 치환
        for variable, value in request.template_variables.items():
            title = title.replace(f"{{{variable}}}", str(value))
            content = content.replace(f"{{{variable}}}", str(value))

        return title, content

    async def _check_user_preferences(self, notification: Notification) -> bool:
        """사용자 설정 확인"""
        if notification.user_id not in self.user_preferences:
            return True  # 기본적으로 허용

        prefs = self.user_preferences[notification.user_id]

        # 타입별 설정 확인
        type_enabled = prefs.get(f"{notification.type.value}_enabled", True)
        if not type_enabled:
            return False

        # 우선순위별 설정 확인
        priority_enabled = prefs.get(f"{notification.priority.value}_enabled", True)
        if not priority_enabled:
            return False

        return True

    def _register_delivery_handlers(self) -> None:
        """발송 핸들러 등록"""
        self.delivery_handlers[NotificationType.EMAIL] = self._send_email
        self.delivery_handlers[NotificationType.IN_APP] = self._send_in_app
        self.delivery_handlers[NotificationType.PUSH] = self._send_push
        self.delivery_handlers[NotificationType.SMS] = self._send_sms

    async def _send_email(self, notification: Notification) -> bool:
        """이메일 발송"""
        try:
            # 모의 이메일 발송 (실제로는 SMTP 설정 필요)
            self.logger.info(f"이메일 발송 (모의): {notification.title}")
            return True

        except Exception as e:
            self.logger.error(f"이메일 발송 실패: {e}")
            return False

    async def _send_in_app(self, notification: Notification) -> bool:
        """인앱 알림 발송"""
        try:
            # 인앱 알림은 즉시 전달됨
            self.logger.info(f"인앱 알림 발송: {notification.title}")
            return True

        except Exception as e:
            self.logger.error(f"인앱 알림 발송 실패: {e}")
            return False

    async def _send_push(self, notification: Notification) -> bool:
        """푸시 알림 발송"""
        try:
            # 모의 푸시 알림 발송
            self.logger.info(f"푸시 알림 발송 (모의): {notification.title}")
            return True

        except Exception as e:
            self.logger.error(f"푸시 알림 발송 실패: {e}")
            return False

    async def _send_sms(self, notification: Notification) -> bool:
        """SMS 발송"""
        try:
            # 모의 SMS 발송
            self.logger.info(f"SMS 발송 (모의): {notification.title}")
            return True

        except Exception as e:
            self.logger.error(f"SMS 발송 실패: {e}")
            return False

    async def _setup_default_channels(self) -> None:
        """기본 채널 설정"""
        default_channels = [
            NotificationChannel(
                id=generate_id(),
                type=NotificationType.EMAIL,
                name="기본 이메일",
                config={},
                is_enabled=True
            ),
            NotificationChannel(
                id=generate_id(),
                type=NotificationType.IN_APP,
                name="인앱 알림",
                config={},
                is_enabled=True
            )
        ]

        for channel in default_channels:
            self.channels[channel.id] = channel

    async def _setup_default_templates(self) -> None:
        """기본 템플릿 설정"""
        default_templates = [
            NotificationTemplate(
                id=generate_id(),
                name="welcome",
                type=NotificationType.EMAIL,
                subject_template="안녕하세요, {username}님!",
                content_template="PACA에 오신 것을 환영합니다!",
                variables=["username"]
            ),
            NotificationTemplate(
                id=generate_id(),
                name="learning_reminder",
                type=NotificationType.IN_APP,
                subject_template="학습 시간입니다!",
                content_template="오늘의 학습을 시작해보세요.",
                variables=[]
            )
        ]

        for template in default_templates:
            self.templates[template.id] = template

    async def _handle_user_action(self, data: Dict[str, Any]) -> None:
        """사용자 액션 처리"""
        try:
            action = data.get('action')
            user_id = data.get('user_id')

            if action == 'goal_achieved' and user_id:
                await self.send_notification(NotificationRequest(
                    user_id=user_id,
                    type=NotificationType.IN_APP,
                    title="목표 달성!",
                    content="축하합니다! 학습 목표를 달성했습니다.",
                    priority=NotificationPriority.HIGH
                ))

        except Exception as e:
            self.logger.error(f"사용자 액션 처리 실패: {e}")

    async def cleanup(self) -> Result[bool]:
        """서비스 정리"""
        try:
            self.logger.info("Notification Service 정리 중...")

            # 데이터 정리
            self.notifications.clear()
            self.templates.clear()
            self.channels.clear()
            self.user_preferences.clear()
            self.pending_queue.clear()
            self.scheduled_queue.clear()

            self.logger.info("Notification Service 정리 완료")
            return create_success(True)

        except Exception as e:
            self.logger.error(f"Notification Service 정리 실패: {e}")
            return create_failure(ApplicationError(f"서비스 정리 실패: {e}"))