"""
Event Emitter Module
이벤트 발행 및 구독 관리 시스템
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Union
import time
from copy import deepcopy

from .base import (
    BaseEvent, EventListener, EventFilter, EventPublishResult,
    EventPriority, PacaEvent
)
from ..types.base import ID, KeyValuePair


logger = logging.getLogger(__name__)


@dataclass
class SubscriptionOptions:
    """구독 옵션"""
    once: bool = False
    max_events: Optional[int] = None
    timeout: Optional[float] = None
    error_handler: Optional[Callable[[Exception, BaseEvent], None]] = None


@dataclass
class EventSubscription:
    """이벤트 구독 정보"""
    listener: EventListener
    filters: List[EventFilter] = field(default_factory=list)
    options: SubscriptionOptions = field(default_factory=SubscriptionOptions)
    subscribed_at: float = field(default_factory=time.time)
    last_event_at: Optional[float] = None
    event_count: int = 0


@dataclass
class EventStatistics:
    """이벤트 통계"""
    total_events_published: int = 0
    total_listeners_notified: int = 0
    average_processing_time_ms: float = 0.0
    events_by_type: Dict[str, int] = field(default_factory=dict)
    error_count: int = 0
    last_event_at: Optional[float] = None


class EventEmitter:
    """PACA 이벤트 발행자"""

    def __init__(self, max_history_size: int = 1000):
        self._subscriptions: Dict[ID, EventSubscription] = {}
        self._global_filters: List[EventFilter] = []
        self._event_history: List[BaseEvent] = []
        self._statistics = EventStatistics()
        self._max_history_size = max_history_size
        self._timeout_tasks: Dict[ID, asyncio.Task] = {}

    async def subscribe(
        self,
        listener: EventListener,
        filters: List[EventFilter] = None,
        options: SubscriptionOptions = None
    ) -> ID:
        """이벤트 리스너 구독"""
        subscription = EventSubscription(
            listener=listener,
            filters=filters or [],
            options=options or SubscriptionOptions()
        )

        self._subscriptions[listener.id] = subscription

        logger.debug(
            "Event listener subscribed",
            extra={
                'listener_id': listener.id,
                'event_types': listener.event_types,
                'filter_count': len(subscription.filters)
            }
        )

        # 타임아웃 설정
        if subscription.options.timeout:
            task = asyncio.create_task(self._handle_timeout(listener.id, subscription.options.timeout))
            self._timeout_tasks[listener.id] = task

        return listener.id

    async def unsubscribe(self, listener_id: ID) -> bool:
        """이벤트 리스너 구독 해제"""
        removed = self._subscriptions.pop(listener_id, None) is not None

        if removed:
            # 타임아웃 태스크 정리
            if listener_id in self._timeout_tasks:
                self._timeout_tasks[listener_id].cancel()
                del self._timeout_tasks[listener_id]

            logger.debug("Event listener unsubscribed", extra={'listener_id': listener_id})

        return removed

    async def unsubscribe_all(self) -> None:
        """모든 구독 해제"""
        count = len(self._subscriptions)
        self._subscriptions.clear()

        # 모든 타임아웃 태스크 정리
        for task in self._timeout_tasks.values():
            task.cancel()
        self._timeout_tasks.clear()

        logger.info("All event listeners unsubscribed", extra={'count': count})

    def on(self, event_type: str, handler: Callable[[BaseEvent], Union[None, asyncio.Future]]) -> 'EventEmitter':
        """이벤트 리스너 등록 (Node.js 스타일 별명)"""
        from uuid import uuid4

        class SimpleListener(EventListener):
            def __init__(self, event_type: str, handler_func: Callable):
                super().__init__([event_type])
                self.handler_func = handler_func
                self.id = str(uuid4())

            async def handle(self, event: BaseEvent) -> None:
                result = self.handler_func(event)
                if asyncio.iscoroutine(result):
                    await result

        listener = SimpleListener(event_type, handler)
        asyncio.create_task(self.subscribe(listener))
        return self

    def add_global_filter(self, filter_obj: EventFilter) -> None:
        """글로벌 필터 추가"""
        self._global_filters.append(filter_obj)
        logger.debug(
            "Global filter added",
            extra={'filter_id': filter_obj.id, 'filter_name': filter_obj.name}
        )

    def remove_global_filter(self, filter_id: ID) -> bool:
        """글로벌 필터 제거"""
        for i, filter_obj in enumerate(self._global_filters):
            if filter_obj.id == filter_id:
                removed = self._global_filters.pop(i)
                logger.debug(
                    "Global filter removed",
                    extra={'filter_id': removed.id, 'filter_name': removed.name}
                )
                return True
        return False

    async def publish(self, event: BaseEvent) -> EventPublishResult:
        """이벤트 발행"""
        start_time = time.time()
        errors: List[Exception] = []
        listeners_notified = 0

        # 글로벌 필터 적용
        if not self._passes_global_filters(event):
            logger.debug(
                "Event blocked by global filters",
                extra={'event_id': event.id, 'event_type': event.type}
            )
            return EventPublishResult(
                event_id=event.id,
                published_at=start_time,
                listeners_notified=0,
                errors=[],
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # 이벤트 히스토리에 추가
        self._add_to_history(event)

        # 통계 업데이트
        self._update_statistics(event)

        # 구독자들에게 이벤트 전달
        eligible_subscriptions = self._get_eligible_subscriptions(event)

        for subscription in eligible_subscriptions:
            try:
                await self._notify_listener(subscription, event)
                listeners_notified += 1

                # 구독 정보 업데이트
                subscription.event_count += 1
                subscription.last_event_at = time.time()

                # once 옵션 처리
                if subscription.options.once:
                    await self.unsubscribe(subscription.listener.id)

                # max_events 옵션 처리
                if (subscription.options.max_events and
                    subscription.event_count >= subscription.options.max_events):
                    await self.unsubscribe(subscription.listener.id)

            except Exception as error:
                errors.append(error)
                self._statistics.error_count += 1

                # 에러 핸들러 호출
                if subscription.options.error_handler:
                    try:
                        subscription.options.error_handler(error, event)
                    except Exception as handler_error:
                        logger.error("Error handler failed", exc_info=handler_error)

                logger.error(
                    "Event listener failed",
                    exc_info=error,
                    extra={
                        'event_id': event.id,
                        'listener_id': subscription.listener.id
                    }
                )

        processing_time_ms = (time.time() - start_time) * 1000

        # 통계 업데이트
        self._statistics.total_listeners_notified += listeners_notified
        self._update_average_processing_time(processing_time_ms)

        logger.debug(
            "Event published",
            extra={
                'event_id': event.id,
                'event_type': event.type,
                'listeners_notified': listeners_notified,
                'processing_time_ms': processing_time_ms,
                'error_count': len(errors)
            }
        )

        return EventPublishResult(
            event_id=event.id,
            published_at=start_time,
            listeners_notified=listeners_notified,
            errors=errors,
            processing_time_ms=processing_time_ms
        )

    async def emit(
        self,
        event_type: str,
        data: KeyValuePair = None,
        priority: EventPriority = EventPriority.MEDIUM,
        source: str = "EventEmitter",
        metadata: KeyValuePair = None
    ) -> EventPublishResult:
        """이벤트 생성 및 발행"""
        event = PacaEvent(
            event_type=event_type,
            data=data or {},
            priority=priority,
            source=source,
            metadata=metadata
        )

        return await self.publish(event)

    def get_subscriptions(self) -> List[EventSubscription]:
        """구독 정보 조회"""
        return list(self._subscriptions.values())

    def get_statistics(self) -> EventStatistics:
        """이벤트 통계 조회"""
        return deepcopy(self._statistics)

    def get_event_history(self, limit: Optional[int] = None) -> List[BaseEvent]:
        """이벤트 히스토리 조회"""
        if limit:
            return self._event_history[-limit:]
        return self._event_history.copy()

    def clear_history(self) -> None:
        """히스토리 초기화"""
        self._event_history.clear()
        logger.info("Event history cleared")

    async def _handle_timeout(self, listener_id: ID, timeout: float) -> None:
        """타임아웃 처리"""
        await asyncio.sleep(timeout)
        await self.unsubscribe(listener_id)
        logger.info(
            "Event listener unsubscribed due to timeout",
            extra={'listener_id': listener_id, 'timeout_ms': timeout * 1000}
        )

    def _passes_global_filters(self, event: BaseEvent) -> bool:
        """글로벌 필터 통과 여부 확인"""
        return all(filter_obj.match(event) for filter_obj in self._global_filters)

    def _add_to_history(self, event: BaseEvent) -> None:
        """히스토리에 이벤트 추가"""
        self._event_history.append(event)

        if len(self._event_history) > self._max_history_size:
            self._event_history.pop(0)

    def _update_statistics(self, event: BaseEvent) -> None:
        """통계 업데이트"""
        self._statistics.total_events_published += 1
        self._statistics.last_event_at = event.timestamp

        # 타입별 통계
        type_count = self._statistics.events_by_type.get(event.type, 0)
        self._statistics.events_by_type[event.type] = type_count + 1

    def _update_average_processing_time(self, processing_time_ms: float) -> None:
        """평균 처리 시간 업데이트"""
        total = self._statistics.total_events_published
        current_avg = self._statistics.average_processing_time_ms
        self._statistics.average_processing_time_ms = (
            (current_avg * (total - 1) + processing_time_ms) / total
        )

    def _get_eligible_subscriptions(self, event: BaseEvent) -> List[EventSubscription]:
        """적격 구독자 목록 조회"""
        eligible = []

        for subscription in self._subscriptions.values():
            # 리스너가 이벤트를 처리할 수 있는지 확인
            if not subscription.listener.can_handle(event):
                continue

            # 구독별 필터 적용
            if not all(filter_obj.match(event) for filter_obj in subscription.filters):
                continue

            eligible.append(subscription)

        # 우선순위 순서로 정렬
        priority_order = [EventPriority.CRITICAL, EventPriority.HIGH, EventPriority.MEDIUM, EventPriority.LOW]
        eligible.sort(
            key=lambda s: priority_order.index(s.listener.priority)
        )

        return eligible

    async def _notify_listener(self, subscription: EventSubscription, event: BaseEvent) -> None:
        """리스너에게 이벤트 전달"""
        await subscription.listener.handle(event)


# EventBus 별칭
EventBus = EventEmitter