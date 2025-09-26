"""
Event Queue Module
이벤트 큐 관리 및 배치 처리 시스템
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from .base import BaseEvent, EventPriority, PacaEvent
from ..types.base import ID, KeyValuePair

logger = logging.getLogger("EventQueue")


@dataclass
class QueuedEvent:
    """큐 이벤트 아이템"""
    event: BaseEvent
    queued_at: float
    retry_count: int = 0
    scheduled_at: Optional[float] = None
    last_attempt_at: Optional[float] = None
    max_retries: int = 3
    retry_delay_ms: float = 1000.0


@dataclass
class BatchProcessingOptions:
    """배치 처리 옵션"""
    batch_size: int = 50
    max_wait_time_ms: float = 1000.0
    max_batch_processing_time_ms: float = 5000.0
    retry_failed_batches: bool = True


@dataclass
class QueueStatistics:
    """큐 통계"""
    total_enqueued: int = 0
    total_processed: int = 0
    total_failed: int = 0
    current_queue_size: int = 0
    average_processing_time_ms: float = 0.0
    peak_queue_size: int = 0
    batches_processed: int = 0
    last_processed_at: Optional[float] = None


class QueueStatus(str, Enum):
    """이벤트 큐 상태"""
    IDLE = "idle"
    PROCESSING = "processing"
    PAUSED = "paused"
    ERROR = "error"


class EventQueue:
    """PACA 이벤트 큐"""

    def __init__(
        self,
        emitter,  # EventEmitter를 순환 참조 방지를 위해 타입 힌트 생략
        max_queue_size: int = 10000,
        batch_options: Optional[BatchProcessingOptions] = None
    ):
        self.emitter = emitter
        self.max_queue_size = max_queue_size
        self.batch_options = batch_options or BatchProcessingOptions()

        # 큐 관리
        self.queue: List[QueuedEvent] = []
        self.priority_queues: Dict[EventPriority, List[QueuedEvent]] = {
            priority: [] for priority in EventPriority
        }

        # 상태 관리
        self.status = QueueStatus.IDLE
        self.statistics = QueueStatistics()
        self._processing_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start_processing(self) -> None:
        """큐 처리 시작"""
        if self.status == QueueStatus.PROCESSING:
            return

        self.status = QueueStatus.PROCESSING
        self._stop_event.clear()
        self._processing_task = asyncio.create_task(self._process_loop())

        logger.info(
            "Queue processing started",
            extra={
                "batch_size": self.batch_options.batch_size,
                "max_wait_time_ms": self.batch_options.max_wait_time_ms
            }
        )

    async def stop_processing(self) -> None:
        """큐 처리 중지"""
        self.status = QueueStatus.PAUSED
        self._stop_event.set()

        if self._processing_task and not self._processing_task.done():
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        logger.info("Queue processing stopped")

    def enqueue(
        self,
        event: BaseEvent,
        max_retries: int = 3,
        retry_delay_ms: float = 1000.0,
        scheduled_at: Optional[float] = None
    ) -> bool:
        """이벤트를 큐에 추가"""
        if len(self.queue) >= self.max_queue_size:
            logger.warning(
                "Queue is full, dropping event",
                extra={
                    "event_id": event.id,
                    "queue_size": len(self.queue),
                    "max_queue_size": self.max_queue_size
                }
            )
            return False

        queued_event = QueuedEvent(
            event=event,
            queued_at=time.time(),
            retry_count=0,
            scheduled_at=scheduled_at,
            max_retries=max_retries,
            retry_delay_ms=retry_delay_ms
        )

        # 우선순위별 큐에 추가
        self.priority_queues[event.priority].append(queued_event)
        self.queue.append(queued_event)

        # 통계 업데이트
        self.statistics.total_enqueued += 1
        self.statistics.current_queue_size = len(self.queue)
        if len(self.queue) > self.statistics.peak_queue_size:
            self.statistics.peak_queue_size = len(self.queue)

        logger.debug(
            "Event enqueued",
            extra={
                "event_id": event.id,
                "event_type": event.type,
                "priority": event.priority.value,
                "queue_size": len(self.queue),
                "scheduled_at": scheduled_at
            }
        )

        return True

    def enqueue_event(
        self,
        event_type: str,
        data: Optional[KeyValuePair] = None,
        priority: EventPriority = EventPriority.MEDIUM,
        source: str = "EventQueue",
        metadata: Optional[KeyValuePair] = None,
        max_retries: int = 3,
        retry_delay_ms: float = 1000.0,
        scheduled_at: Optional[float] = None
    ) -> bool:
        """이벤트 생성 및 큐 추가"""
        event = PacaEvent(
            event_type=event_type,
            data=data,
            priority=priority,
            source=source,
            metadata=metadata
        )

        return self.enqueue(
            event.to_base_event(),
            max_retries=max_retries,
            retry_delay_ms=retry_delay_ms,
            scheduled_at=scheduled_at
        )

    def clear(self) -> None:
        """큐 비우기"""
        cleared_count = len(self.queue)
        self.queue.clear()
        for priority_queue in self.priority_queues.values():
            priority_queue.clear()

        self.statistics.current_queue_size = 0
        logger.info("Queue cleared", extra={"cleared_count": cleared_count})

    def get_status(self) -> QueueStatus:
        """큐 상태 조회"""
        return self.status

    def get_statistics(self) -> QueueStatistics:
        """큐 통계 조회"""
        return QueueStatistics(
            total_enqueued=self.statistics.total_enqueued,
            total_processed=self.statistics.total_processed,
            total_failed=self.statistics.total_failed,
            current_queue_size=self.statistics.current_queue_size,
            average_processing_time_ms=self.statistics.average_processing_time_ms,
            peak_queue_size=self.statistics.peak_queue_size,
            batches_processed=self.statistics.batches_processed,
            last_processed_at=self.statistics.last_processed_at
        )

    def get_pending_events(self) -> List[BaseEvent]:
        """대기 중인 이벤트 조회"""
        now = time.time()
        return [
            qe.event for qe in self.queue
            if not qe.scheduled_at or qe.scheduled_at <= now
        ]

    def get_scheduled_events(self) -> List[BaseEvent]:
        """예약된 이벤트 조회"""
        now = time.time()
        return [
            qe.event for qe in self.queue
            if qe.scheduled_at and qe.scheduled_at > now
        ]

    async def _process_loop(self) -> None:
        """이벤트 처리 루프"""
        while not self._stop_event.is_set():
            try:
                # 다음 배치가 준비될 때까지 대기
                await asyncio.wait_for(
                    self._wait_for_batch(),
                    timeout=self.batch_options.max_wait_time_ms / 1000.0
                )
            except asyncio.TimeoutError:
                # 타임아웃으로 배치 처리 시작
                pass

            if self._stop_event.is_set():
                break

            await self._process_batch()

    async def _wait_for_batch(self) -> None:
        """배치가 준비될 때까지 대기"""
        while len(self.queue) < self.batch_options.batch_size:
            await asyncio.sleep(0.1)
            if self._stop_event.is_set():
                break

    async def _process_batch(self) -> None:
        """배치 처리"""
        if self.status != QueueStatus.PROCESSING:
            return

        start_time = time.time()
        batch = self._get_next_batch()

        if not batch:
            return

        logger.debug(
            "Processing batch",
            extra={
                "batch_size": len(batch),
                "queue_size": len(self.queue)
            }
        )

        try:
            await self._process_batch_events(batch)
            self.statistics.batches_processed += 1
            self.statistics.last_processed_at = time.time()

        except Exception as error:
            logger.error("Batch processing failed", exc_info=error)
            self.status = QueueStatus.ERROR

            if self.batch_options.retry_failed_batches:
                await self._requeue_failed_events(batch)

        processing_time = (time.time() - start_time) * 1000  # ms 단위
        self._update_average_processing_time(processing_time)

    def _get_next_batch(self) -> List[QueuedEvent]:
        """다음 배치 가져오기 (우선순위 기준)"""
        batch = []
        now = time.time()

        # 우선순위 순서로 이벤트 선택
        priority_order = [
            EventPriority.CRITICAL,
            EventPriority.HIGH,
            EventPriority.MEDIUM,
            EventPriority.LOW
        ]

        for priority in priority_order:
            priority_queue = self.priority_queues[priority]

            while (
                len(batch) < self.batch_options.batch_size and
                priority_queue
            ):
                queued_event = priority_queue[0]

                # 예약 시간 확인
                if queued_event.scheduled_at and queued_event.scheduled_at > now:
                    break

                # 큐에서 제거
                priority_queue.pop(0)
                if queued_event in self.queue:
                    self.queue.remove(queued_event)

                batch.append(queued_event)

            if len(batch) >= self.batch_options.batch_size:
                break

        self.statistics.current_queue_size = len(self.queue)
        return batch

    async def _process_batch_events(self, batch: List[QueuedEvent]) -> None:
        """배치 이벤트 처리"""
        tasks = []

        for queued_event in batch:
            task = asyncio.create_task(
                self._process_single_event(queued_event)
            )
            tasks.append(task)

        # 모든 이벤트를 병렬로 처리
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 처리
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                await self._handle_event_failure(batch[i], result)

    async def _process_single_event(self, queued_event: QueuedEvent) -> None:
        """단일 이벤트 처리"""
        queued_event.last_attempt_at = time.time()

        # EventEmitter를 통해 이벤트 발행
        if hasattr(self.emitter, 'publish'):
            await self.emitter.publish(queued_event.event)
        else:
            # 동기식 emitter인 경우
            self.emitter.emit(queued_event.event.type, queued_event.event)

        self.statistics.total_processed += 1

    async def _handle_event_failure(
        self,
        queued_event: QueuedEvent,
        error: Exception
    ) -> None:
        """이벤트 처리 실패 처리"""
        queued_event.retry_count += 1

        if queued_event.retry_count <= queued_event.max_retries:
            # 재시도를 위해 다시 큐에 추가
            queued_event.scheduled_at = (
                time.time() + queued_event.retry_delay_ms / 1000.0
            )
            self.queue.append(queued_event)
            self.priority_queues[queued_event.event.priority].append(queued_event)

            logger.warning(
                "Event processing failed, scheduled for retry",
                extra={
                    "event_id": queued_event.event.id,
                    "retry_count": queued_event.retry_count,
                    "max_retries": queued_event.max_retries,
                    "retry_delay_ms": queued_event.retry_delay_ms
                }
            )
        else:
            # 최대 재시도 횟수 초과
            self.statistics.total_failed += 1
            logger.error(
                "Event processing failed permanently",
                exc_info=error,
                extra={
                    "event_id": queued_event.event.id,
                    "retry_count": queued_event.retry_count
                }
            )

    async def _requeue_failed_events(self, batch: List[QueuedEvent]) -> None:
        """실패한 이벤트 재큐잉"""
        for queued_event in batch:
            if queued_event.retry_count <= queued_event.max_retries:
                queued_event.scheduled_at = (
                    time.time() + queued_event.retry_delay_ms / 1000.0
                )
                self.queue.append(queued_event)
                self.priority_queues[queued_event.event.priority].append(queued_event)

    def _update_average_processing_time(self, processing_time_ms: float) -> None:
        """평균 처리 시간 업데이트"""
        batches_processed = self.statistics.batches_processed
        if batches_processed == 0:
            self.statistics.average_processing_time_ms = processing_time_ms
        else:
            current_avg = self.statistics.average_processing_time_ms
            self.statistics.average_processing_time_ms = (
                (current_avg * (batches_processed - 1) + processing_time_ms) /
                batches_processed
            )


class PriorityEventQueue(EventQueue):
    """우선순위 기반 이벤트 큐"""

    def __init__(self, emitter, **kwargs):
        super().__init__(emitter, **kwargs)
        # 우선순위 가중치 설정
        self.priority_weights = {
            EventPriority.CRITICAL: 4,
            EventPriority.HIGH: 3,
            EventPriority.MEDIUM: 2,
            EventPriority.LOW: 1
        }

    def _get_next_batch(self) -> List[QueuedEvent]:
        """우선순위 가중치를 고려한 배치 가져오기"""
        batch = []
        now = time.time()

        # 가중치 기반으로 각 우선순위에서 이벤트 선택
        for priority in EventPriority:
            priority_queue = self.priority_queues[priority]
            weight = self.priority_weights[priority]

            # 가중치에 비례하여 이벤트 선택
            max_events_from_priority = min(
                len(priority_queue),
                max(1, self.batch_options.batch_size * weight // 10)
            )

            selected = 0
            while (
                selected < max_events_from_priority and
                len(batch) < self.batch_options.batch_size and
                priority_queue
            ):
                queued_event = priority_queue[0]

                # 예약 시간 확인
                if queued_event.scheduled_at and queued_event.scheduled_at > now:
                    break

                # 큐에서 제거
                priority_queue.pop(0)
                if queued_event in self.queue:
                    self.queue.remove(queued_event)

                batch.append(queued_event)
                selected += 1

        self.statistics.current_queue_size = len(self.queue)
        return batch


# 편의 함수들
def create_event_queue(
    emitter,
    max_queue_size: int = 10000,
    batch_size: int = 50,
    max_wait_time_ms: float = 1000.0
) -> EventQueue:
    """기본 이벤트 큐 생성"""
    batch_options = BatchProcessingOptions(
        batch_size=batch_size,
        max_wait_time_ms=max_wait_time_ms
    )
    return EventQueue(emitter, max_queue_size, batch_options)


def create_priority_queue(
    emitter,
    max_queue_size: int = 10000,
    batch_size: int = 50,
    max_wait_time_ms: float = 1000.0
) -> PriorityEventQueue:
    """우선순위 이벤트 큐 생성"""
    batch_options = BatchProcessingOptions(
        batch_size=batch_size,
        max_wait_time_ms=max_wait_time_ms
    )
    return PriorityEventQueue(emitter, max_queue_size, batch_options)