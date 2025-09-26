"""
Event System Module
TypeScript EventEmitter3을 대체하는 Python 이벤트 시스템
asyncio 기반 비동기 이벤트 처리 지원
"""

import asyncio
import time
import weakref
from typing import Dict, List, Callable, Any, Optional, Union, Set, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import logging

from .types import ID, Timestamp, KeyValuePair, BaseEvent, create_result, Result
from .errors import PacaError, ErrorSeverity

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """이벤트 우선순위"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5


class EventCategory(Enum):
    """이벤트 카테고리"""
    SYSTEM = "system"
    USER = "user"
    COGNITIVE = "cognitive"
    LEARNING = "learning"
    REASONING = "reasoning"
    ERROR = "error"
    PERFORMANCE = "performance"


@dataclass
class EventListener:
    """이벤트 리스너 정보"""
    callback: Union[Callable, Callable[..., Awaitable]]
    once: bool = False
    priority: EventPriority = EventPriority.NORMAL
    category: EventCategory = EventCategory.SYSTEM
    created_at: Timestamp = field(default_factory=time.time)
    call_count: int = 0
    last_called: Optional[Timestamp] = None


@dataclass
class EventStats:
    """이벤트 통계"""
    total_events: int = 0
    events_by_type: Dict[str, int] = field(default_factory=dict)
    events_by_priority: Dict[EventPriority, int] = field(default_factory=dict)
    listeners_count: int = 0
    last_event_at: Optional[Timestamp] = None
    errors_count: int = 0


class EventBus:
    """
    TypeScript EventEmitter3을 대체하는 Python 이벤트 시스템

    Features:
    - 동기/비동기 이벤트 리스너 지원
    - 이벤트 우선순위 처리
    - 한 번만 실행되는 리스너 (once)
    - 이벤트 통계 및 모니터링
    - 메모리 누수 방지 (약한 참조)
    - 에러 처리 및 복구
    """

    def __init__(self, max_listeners: int = 100, enable_stats: bool = True):
        self._listeners: Dict[str, List[EventListener]] = {}
        self._max_listeners = max_listeners
        self._enable_stats = enable_stats
        self._stats = EventStats()
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._paused = False

    def on(self, event_type: str, callback: Union[Callable, Callable[..., Awaitable]],
           priority: EventPriority = EventPriority.NORMAL,
           category: EventCategory = EventCategory.SYSTEM) -> 'EventBus':
        """이벤트 리스너 등록"""
        return self._add_listener(event_type, callback, once=False, priority=priority, category=category)

    def once(self, event_type: str, callback: Union[Callable, Callable[..., Awaitable]],
             priority: EventPriority = EventPriority.NORMAL,
             category: EventCategory = EventCategory.SYSTEM) -> 'EventBus':
        """한 번만 실행되는 이벤트 리스너 등록"""
        return self._add_listener(event_type, callback, once=True, priority=priority, category=category)

    def off(self, event_type: str, callback: Optional[Callable] = None) -> 'EventBus':
        """이벤트 리스너 제거"""
        if event_type not in self._listeners:
            return self

        if callback is None:
            # 모든 리스너 제거
            del self._listeners[event_type]
        else:
            # 특정 리스너만 제거
            self._listeners[event_type] = [
                listener for listener in self._listeners[event_type]
                if listener.callback != callback
            ]

            # 빈 리스트면 삭제
            if not self._listeners[event_type]:
                del self._listeners[event_type]

        self._update_listener_count()
        return self

    def remove_all_listeners(self, event_type: Optional[str] = None) -> 'EventBus':
        """모든 리스너 제거"""
        if event_type is None:
            self._listeners.clear()
        else:
            self._listeners.pop(event_type, None)

        self._update_listener_count()
        return self

    async def emit(self, event_type: str, *args, **kwargs) -> Result[int]:
        """이벤트 발생 - 모든 리스너 호출"""
        if self._paused:
            return create_result(False, 0, "EventBus is paused")

        if event_type not in self._listeners:
            return create_result(True, 0, None)

        listeners = self._listeners[event_type].copy()

        # 우선순위로 정렬 (높은 우선순위부터)
        listeners.sort(key=lambda x: x.priority.value, reverse=True)

        success_count = 0
        errors = []

        # 이벤트 데이터 생성
        event_data = {
            'type': event_type,
            'timestamp': time.time(),
            'args': args,
            'kwargs': kwargs
        }

        for listener in listeners:
            try:
                await self._call_listener(listener, event_data, *args, **kwargs)
                success_count += 1

                # once 리스너는 제거
                if listener.once:
                    self._listeners[event_type].remove(listener)

            except Exception as e:
                error_msg = f"Error in listener for {event_type}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

                if self._enable_stats:
                    self._stats.errors_count += 1

        # 빈 리스트 정리
        if not self._listeners[event_type]:
            del self._listeners[event_type]

        # 통계 업데이트
        if self._enable_stats:
            self._stats.total_events += 1
            self._stats.events_by_type[event_type] = self._stats.events_by_type.get(event_type, 0) + 1
            self._stats.last_event_at = time.time()

        self._update_listener_count()

        if errors:
            return create_result(False, success_count, f"Errors occurred: {'; '.join(errors)}")

        return create_result(True, success_count, None)

    def emit_sync(self, event_type: str, *args, **kwargs) -> Result[int]:
        """동기적 이벤트 발생"""
        try:
            if self._loop is None:
                # 새 이벤트 루프에서 실행
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(self.emit(event_type, *args, **kwargs))
                finally:
                    loop.close()
            else:
                # 기존 루프에서 태스크 생성
                task = asyncio.create_task(self.emit(event_type, *args, **kwargs))
                return asyncio.run_coroutine_threadsafe(task, self._loop).result()

        except Exception as e:
            return create_result(False, 0, f"Sync emit failed: {str(e)}")

    def pause(self) -> None:
        """이벤트 처리 일시정지"""
        self._paused = True

    def resume(self) -> None:
        """이벤트 처리 재개"""
        self._paused = False

    def is_paused(self) -> bool:
        """일시정지 상태 확인"""
        return self._paused

    def get_listener_count(self, event_type: Optional[str] = None) -> int:
        """리스너 개수 반환"""
        if event_type is None:
            return sum(len(listeners) for listeners in self._listeners.values())
        return len(self._listeners.get(event_type, []))

    def get_event_types(self) -> List[str]:
        """등록된 이벤트 타입 목록"""
        return list(self._listeners.keys())

    def has_listeners(self, event_type: str) -> bool:
        """특정 이벤트에 리스너가 있는지 확인"""
        return event_type in self._listeners and len(self._listeners[event_type]) > 0

    def get_stats(self) -> EventStats:
        """이벤트 통계 반환"""
        return self._stats

    def clear_stats(self) -> None:
        """통계 초기화"""
        self._stats = EventStats()
        self._update_listener_count()

    async def wait_for(self, event_type: str, timeout: Optional[float] = None,
                      condition: Optional[Callable[[Any], bool]] = None) -> Result[Any]:
        """특정 이벤트 대기"""
        future = asyncio.Future()

        def handler(*args, **kwargs):
            event_data = {'args': args, 'kwargs': kwargs}
            if condition is None or condition(event_data):
                if not future.done():
                    future.set_result(event_data)

        self.once(event_type, handler)

        try:
            if timeout is None:
                result = await future
            else:
                result = await asyncio.wait_for(future, timeout=timeout)

            return create_result(True, result, None)

        except asyncio.TimeoutError:
            self.off(event_type, handler)
            return create_result(False, None, f"Timeout waiting for {event_type}")
        except Exception as e:
            self.off(event_type, handler)
            return create_result(False, None, f"Error waiting for {event_type}: {str(e)}")

    def _add_listener(self, event_type: str, callback: Union[Callable, Callable[..., Awaitable]],
                     once: bool, priority: EventPriority, category: EventCategory) -> 'EventBus':
        """리스너 추가 내부 메서드"""
        if event_type not in self._listeners:
            self._listeners[event_type] = []

        current_count = self.get_listener_count(event_type)
        if current_count >= self._max_listeners:
            logger.warning(f"Max listeners ({self._max_listeners}) exceeded for event: {event_type}")

        listener = EventListener(
            callback=callback,
            once=once,
            priority=priority,
            category=category
        )

        self._listeners[event_type].append(listener)
        self._update_listener_count()

        return self

    async def _call_listener(self, listener: EventListener, event_data: Dict[str, Any],
                           *args, **kwargs) -> None:
        """리스너 호출 내부 메서드"""
        listener.call_count += 1
        listener.last_called = time.time()

        callback = listener.callback

        try:
            if asyncio.iscoroutinefunction(callback):
                # 비동기 함수
                await callback(*args, **kwargs)
            elif callable(callback):
                # 동기 함수 - 스레드 풀에서 실행
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self._thread_pool, callback, *args, **kwargs)
            else:
                raise TypeError(f"Invalid callback type: {type(callback)}")

        except Exception as e:
            raise PacaError(
                f"Event listener error: {str(e)}",
                ErrorSeverity.MEDIUM,
                "EVENT_LISTENER_ERROR",
                {"event_data": event_data, "listener": str(listener)}
            )

    def _update_listener_count(self) -> None:
        """리스너 개수 통계 업데이트"""
        if self._enable_stats:
            self._stats.listeners_count = self.get_listener_count()

    def __enter__(self) -> 'EventBus':
        """컨텍스트 매니저 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """컨텍스트 매니저 종료"""
        self.remove_all_listeners()
        self._thread_pool.shutdown(wait=True)


# 전역 이벤트 버스 인스턴스
_default_event_bus: Optional[EventBus] = None


def get_default_event_bus() -> EventBus:
    """기본 이벤트 버스 인스턴스 반환"""
    global _default_event_bus
    if _default_event_bus is None:
        _default_event_bus = EventBus()
    return _default_event_bus


def create_event_bus(max_listeners: int = 100, enable_stats: bool = True) -> EventBus:
    """새 이벤트 버스 인스턴스 생성"""
    return EventBus(max_listeners=max_listeners, enable_stats=enable_stats)


# 편의 함수들
def on(event_type: str, callback: Union[Callable, Callable[..., Awaitable]],
       priority: EventPriority = EventPriority.NORMAL) -> None:
    """기본 이벤트 버스에 리스너 등록"""
    get_default_event_bus().on(event_type, callback, priority)


def once(event_type: str, callback: Union[Callable, Callable[..., Awaitable]],
         priority: EventPriority = EventPriority.NORMAL) -> None:
    """기본 이벤트 버스에 일회성 리스너 등록"""
    get_default_event_bus().once(event_type, callback, priority)


def off(event_type: str, callback: Optional[Callable] = None) -> None:
    """기본 이벤트 버스에서 리스너 제거"""
    get_default_event_bus().off(event_type, callback)


async def emit(event_type: str, *args, **kwargs) -> Result[int]:
    """기본 이벤트 버스에서 이벤트 발생"""
    return await get_default_event_bus().emit(event_type, *args, **kwargs)