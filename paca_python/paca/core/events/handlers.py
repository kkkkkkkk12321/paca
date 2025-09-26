"""
Event Handlers Module
공통 이벤트 핸들러들과 유틸리티 함수들
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable, Any, Union
from collections import defaultdict

from .base import BaseEvent, AbstractEventListener, EventPriority, EventCategory
from ..types.base import ID, KeyValuePair

# 로거 설정
logger = logging.getLogger("EventHandlers")


class LoggingEventListener(AbstractEventListener):
    """로깅 이벤트 리스너"""

    def __init__(
        self,
        event_types: Union[str, List[str]] = ["*"],
        log_level: str = "info"
    ):
        super().__init__(event_types, EventPriority.LOW)
        self.log_level = log_level.lower()

    async def handle(self, event: BaseEvent) -> None:
        """이벤트 로깅 처리"""
        log_data = {
            "event_id": event.id,
            "event_type": event.type,
            "category": event.category.value,
            "priority": event.priority.value,
            "source": event.source,
            "timestamp": event.timestamp,
            "data": event.data
        }

        message = f"Event processed: {event.type}"

        if self.log_level == "debug":
            logger.debug(message, extra=log_data)
        elif self.log_level == "info":
            logger.info(message, extra=log_data)
        elif self.log_level == "warn":
            logger.warning(message, extra=log_data)
        elif self.log_level == "error":
            logger.error(message, extra=log_data)


class MetricsEventListener(AbstractEventListener):
    """메트릭 수집 이벤트 리스너"""

    def __init__(self, event_types: Union[str, List[str]] = ["*"]):
        super().__init__(event_types, EventPriority.LOW)
        self.metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "last_event_at": 0.0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "error_count": 0
        })

    async def handle(self, event: BaseEvent) -> None:
        """메트릭 수집 처리"""
        start_time = time.time()

        try:
            self._update_metrics(event.type, start_time)
        except Exception as error:
            self._update_error_metrics(event.type)
            logger.error(
                f"Metrics collection failed for event {event.id}",
                exc_info=error
            )

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """전체 메트릭 반환"""
        return dict(self.metrics)

    def get_metrics_for_type(self, event_type: str) -> Optional[Dict[str, Any]]:
        """특정 이벤트 타입의 메트릭 반환"""
        return self.metrics.get(event_type)

    def clear_metrics(self) -> None:
        """메트릭 초기화"""
        self.metrics.clear()
        logger.info("Event metrics cleared")

    def _update_metrics(self, event_type: str, start_time: float) -> None:
        """메트릭 업데이트"""
        processing_time = (time.time() - start_time) * 1000  # ms 단위
        metrics = self.metrics[event_type]

        metrics["count"] += 1
        metrics["last_event_at"] = time.time()
        metrics["total_processing_time"] += processing_time
        metrics["average_processing_time"] = (
            metrics["total_processing_time"] / metrics["count"]
        )

    def _update_error_metrics(self, event_type: str) -> None:
        """에러 메트릭 업데이트"""
        self.metrics[event_type]["error_count"] += 1


class FilteringEventListener(AbstractEventListener):
    """이벤트 필터링 리스너"""

    def __init__(
        self,
        wrapped_listener: AbstractEventListener,
        filter_fn: Callable[[BaseEvent], bool]
    ):
        super().__init__(wrapped_listener.event_types, wrapped_listener.priority)
        self.wrapped_listener = wrapped_listener
        self.filter_fn = filter_fn

    def can_handle(self, event: BaseEvent) -> bool:
        """필터링된 이벤트 처리 가능 여부 확인"""
        return (
            self.wrapped_listener.can_handle(event) and
            self.filter_fn(event)
        )

    async def handle(self, event: BaseEvent) -> None:
        """필터링된 이벤트 처리"""
        if self.can_handle(event):
            await self.wrapped_listener.handle(event)


class TransformingEventListener(AbstractEventListener):
    """이벤트 변환 리스너"""

    def __init__(
        self,
        wrapped_listener: AbstractEventListener,
        transform_fn: Callable[[BaseEvent], BaseEvent]
    ):
        super().__init__(wrapped_listener.event_types, wrapped_listener.priority)
        self.wrapped_listener = wrapped_listener
        self.transform_fn = transform_fn

    async def handle(self, event: BaseEvent) -> None:
        """변환된 이벤트 처리"""
        transformed_event = self.transform_fn(event)
        await self.wrapped_listener.handle(transformed_event)


class ConditionalEventListener(AbstractEventListener):
    """조건부 이벤트 리스너"""

    def __init__(
        self,
        event_types: Union[str, List[str]],
        condition: Callable[[BaseEvent], bool],
        on_true: AbstractEventListener,
        on_false: Optional[AbstractEventListener] = None,
        priority: EventPriority = EventPriority.MEDIUM
    ):
        super().__init__(event_types, priority)
        self.condition = condition
        self.on_true = on_true
        self.on_false = on_false

    async def handle(self, event: BaseEvent) -> None:
        """조건에 따른 이벤트 처리"""
        if self.condition(event):
            await self.on_true.handle(event)
        elif self.on_false:
            await self.on_false.handle(event)


class BatchEventListener(AbstractEventListener):
    """배치 처리 이벤트 리스너"""

    def __init__(
        self,
        event_types: Union[str, List[str]],
        batch_size: int,
        flush_interval_ms: float,
        process_batch: Callable[[List[BaseEvent]], Any],
        priority: EventPriority = EventPriority.MEDIUM
    ):
        super().__init__(event_types, priority)
        self.batch_size = batch_size
        self.flush_interval_ms = flush_interval_ms / 1000.0  # 초 단위로 변환
        self.process_batch = process_batch
        self.batch: List[BaseEvent] = []
        self._flush_task: Optional[asyncio.Task] = None
        self._schedule_flush()

    async def handle(self, event: BaseEvent) -> None:
        """배치에 이벤트 추가"""
        self.batch.append(event)

        if len(self.batch) >= self.batch_size:
            await self.flush()

    async def flush(self) -> None:
        """배치 처리 및 플러시"""
        if not self.batch:
            return

        current_batch = self.batch.copy()
        self.batch.clear()

        try:
            result = self.process_batch(current_batch)
            if asyncio.iscoroutine(result):
                await result
            logger.debug(f"Batch processed: {len(current_batch)} events")
        except Exception as error:
            logger.error(
                f"Batch processing failed: {len(current_batch)} events",
                exc_info=error
            )

        self._schedule_flush()

    def _schedule_flush(self) -> None:
        """플러시 스케줄링"""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()

        self._flush_task = asyncio.create_task(
            self._delayed_flush()
        )

    async def _delayed_flush(self) -> None:
        """지연된 플러시"""
        await asyncio.sleep(self.flush_interval_ms)
        await self.flush()


class RetryEventListener(AbstractEventListener):
    """재시도 이벤트 리스너"""

    def __init__(
        self,
        wrapped_listener: AbstractEventListener,
        max_retries: int = 3,
        retry_delay_ms: float = 1000.0
    ):
        super().__init__(wrapped_listener.event_types, wrapped_listener.priority)
        self.wrapped_listener = wrapped_listener
        self.max_retries = max_retries
        self.retry_delay_ms = retry_delay_ms / 1000.0  # 초 단위로 변환
        self.retry_count: Dict[ID, int] = {}

    async def handle(self, event: BaseEvent) -> None:
        """재시도 로직이 포함된 이벤트 처리"""
        current_retries = self.retry_count.get(event.id, 0)

        try:
            await self.wrapped_listener.handle(event)
            # 성공시 재시도 카운트 제거
            self.retry_count.pop(event.id, None)
        except Exception as error:
            if current_retries < self.max_retries:
                self.retry_count[event.id] = current_retries + 1

                logger.warning(
                    f"Event handler failed, scheduling retry {current_retries + 1}/{self.max_retries}",
                    extra={
                        "event_id": event.id,
                        "retry_count": current_retries + 1,
                        "max_retries": self.max_retries
                    }
                )

                # 지연 후 재시도
                await asyncio.sleep(self.retry_delay_ms)
                await self.handle(event)
            else:
                # 최대 재시도 횟수 초과
                self.retry_count.pop(event.id, None)
                logger.error(
                    f"Event handler failed permanently after {current_retries} retries",
                    exc_info=error,
                    extra={"event_id": event.id}
                )
                raise error


class ThrottlingEventListener(AbstractEventListener):
    """스로틀링 이벤트 리스너"""

    def __init__(
        self,
        wrapped_listener: AbstractEventListener,
        max_events_per_second: float
    ):
        super().__init__(wrapped_listener.event_types, wrapped_listener.priority)
        self.wrapped_listener = wrapped_listener
        self.min_interval = 1.0 / max_events_per_second
        self.last_processed = 0.0

    async def handle(self, event: BaseEvent) -> None:
        """스로틀링된 이벤트 처리"""
        current_time = time.time()
        time_since_last = current_time - self.last_processed

        if time_since_last < self.min_interval:
            await asyncio.sleep(self.min_interval - time_since_last)

        await self.wrapped_listener.handle(event)
        self.last_processed = time.time()


class DebounceEventListener(AbstractEventListener):
    """디바운스 이벤트 리스너"""

    def __init__(
        self,
        wrapped_listener: AbstractEventListener,
        delay_ms: float,
        key_fn: Optional[Callable[[BaseEvent], str]] = None
    ):
        super().__init__(wrapped_listener.event_types, wrapped_listener.priority)
        self.wrapped_listener = wrapped_listener
        self.delay_ms = delay_ms / 1000.0  # 초 단위로 변환
        self.key_fn = key_fn or (lambda event: event.type)
        self.pending_tasks: Dict[str, asyncio.Task] = {}

    async def handle(self, event: BaseEvent) -> None:
        """디바운스된 이벤트 처리"""
        key = self.key_fn(event)

        # 기존 작업이 있으면 취소
        if key in self.pending_tasks:
            self.pending_tasks[key].cancel()

        # 새로운 지연된 작업 스케줄
        self.pending_tasks[key] = asyncio.create_task(
            self._delayed_handle(event, key)
        )

    async def _delayed_handle(self, event: BaseEvent, key: str) -> None:
        """지연된 이벤트 처리"""
        try:
            await asyncio.sleep(self.delay_ms)
            await self.wrapped_listener.handle(event)
        finally:
            self.pending_tasks.pop(key, None)


# 팩토리 함수들
def create_logging_handler(
    event_types: Union[str, List[str]] = ["*"],
    log_level: str = "info"
) -> LoggingEventListener:
    """로깅 핸들러 생성"""
    return LoggingEventListener(event_types, log_level)


def create_metrics_handler(
    event_types: Union[str, List[str]] = ["*"]
) -> MetricsEventListener:
    """메트릭 핸들러 생성"""
    return MetricsEventListener(event_types)


def create_filtering_handler(
    wrapped_listener: AbstractEventListener,
    filter_fn: Callable[[BaseEvent], bool]
) -> FilteringEventListener:
    """필터링 핸들러 생성"""
    return FilteringEventListener(wrapped_listener, filter_fn)


def create_transforming_handler(
    wrapped_listener: AbstractEventListener,
    transform_fn: Callable[[BaseEvent], BaseEvent]
) -> TransformingEventListener:
    """변환 핸들러 생성"""
    return TransformingEventListener(wrapped_listener, transform_fn)


def create_conditional_handler(
    event_types: Union[str, List[str]],
    condition: Callable[[BaseEvent], bool],
    on_true: AbstractEventListener,
    on_false: Optional[AbstractEventListener] = None,
    priority: EventPriority = EventPriority.MEDIUM
) -> ConditionalEventListener:
    """조건부 핸들러 생성"""
    return ConditionalEventListener(event_types, condition, on_true, on_false, priority)


def create_batch_handler(
    event_types: Union[str, List[str]],
    batch_size: int,
    flush_interval_ms: float,
    process_batch: Callable[[List[BaseEvent]], Any],
    priority: EventPriority = EventPriority.MEDIUM
) -> BatchEventListener:
    """배치 핸들러 생성"""
    return BatchEventListener(event_types, batch_size, flush_interval_ms, process_batch, priority)


def create_retry_handler(
    wrapped_listener: AbstractEventListener,
    max_retries: int = 3,
    retry_delay_ms: float = 1000.0
) -> RetryEventListener:
    """재시도 핸들러 생성"""
    return RetryEventListener(wrapped_listener, max_retries, retry_delay_ms)


def create_throttling_handler(
    wrapped_listener: AbstractEventListener,
    max_events_per_second: float
) -> ThrottlingEventListener:
    """스로틀링 핸들러 생성"""
    return ThrottlingEventListener(wrapped_listener, max_events_per_second)


def create_debounce_handler(
    wrapped_listener: AbstractEventListener,
    delay_ms: float,
    key_fn: Optional[Callable[[BaseEvent], str]] = None
) -> DebounceEventListener:
    """디바운스 핸들러 생성"""
    return DebounceEventListener(wrapped_listener, delay_ms, key_fn)