"""
Base Event System Module
기본 이벤트 시스템과 공통 인터페이스들
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Set, Optional, Any, Union
import time
import uuid
import json
from datetime import datetime

from ..types.base import ID, Timestamp, KeyValuePair, create_id, current_timestamp


class EventPriority(Enum):
    """이벤트 우선순위 레벨"""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'


class EventCategory(Enum):
    """이벤트 카테고리"""
    SYSTEM = 'system'
    USER = 'user'
    COGNITIVE = 'cognitive'
    PERFORMANCE = 'performance'
    ERROR = 'error'
    LIFECYCLE = 'lifecycle'


class EventStatus(Enum):
    """이벤트 상태"""
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


@dataclass(frozen=True)
class BaseEvent:
    """기본 이벤트 인터페이스"""
    id: ID
    type: str
    timestamp: Timestamp
    priority: EventPriority
    category: EventCategory
    source: str
    data: KeyValuePair
    metadata: KeyValuePair
    correlation_id: Optional[ID] = None
    causation_id: Optional[ID] = None


class EventListener(ABC):
    """이벤트 리스너 인터페이스"""

    def __init__(self, event_types: Union[str, List[str]], priority: EventPriority = EventPriority.MEDIUM):
        self.id = f"listener_{int(time.time() * 1000)}_{uuid.uuid4().hex[:9]}"
        self.event_types = [event_types] if isinstance(event_types, str) else event_types
        self.priority = priority

    @abstractmethod
    async def handle(self, event: BaseEvent) -> None:
        """이벤트 처리"""
        pass

    def can_handle(self, event: BaseEvent) -> bool:
        """이벤트 처리 가능 여부 확인"""
        return event.type in self.event_types or '*' in self.event_types


class EventFilter(ABC):
    """이벤트 필터 인터페이스"""

    def __init__(self, name: str):
        self.id = f"filter_{int(time.time() * 1000)}_{uuid.uuid4().hex[:9]}"
        self.name = name

    @abstractmethod
    def match(self, event: BaseEvent) -> bool:
        """이벤트 매칭 여부 확인"""
        pass


@dataclass(frozen=True)
class EventPublishResult:
    """이벤트 발행 결과"""
    event_id: ID
    published_at: Timestamp
    listeners_notified: int
    errors: List[Exception]
    processing_time_ms: float


class PacaEvent:
    """PACA 기본 이벤트 클래스"""

    def __init__(
        self,
        event_type: str,
        data: KeyValuePair = None,
        priority: EventPriority = EventPriority.MEDIUM,
        category: EventCategory = EventCategory.SYSTEM,
        source: str = "unknown",
        metadata: KeyValuePair = None,
        correlation_id: Optional[ID] = None,
        causation_id: Optional[ID] = None
    ):
        self.id = self._generate_event_id()
        self.type = event_type
        self.timestamp = current_timestamp()
        self.priority = priority
        self.category = category
        self.source = source
        self.data = data or {}
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'version': '1.0',
            **(metadata or {})
        }
        self.correlation_id = correlation_id
        self.causation_id = causation_id

    @staticmethod
    def _generate_event_id() -> ID:
        """이벤트 ID 생성"""
        return f"evt_{int(time.time() * 1000)}_{uuid.uuid4().hex[:9]}"

    def clone(self, **overrides) -> 'PacaEvent':
        """이벤트 복제"""
        return PacaEvent(
            event_type=overrides.get('type', self.type),
            data={**self.data, **overrides.get('data', {})},
            priority=overrides.get('priority', self.priority),
            category=overrides.get('category', self.category),
            source=overrides.get('source', self.source),
            metadata={**self.metadata, **overrides.get('metadata', {})},
            correlation_id=overrides.get('correlation_id', self.correlation_id),
            causation_id=overrides.get('causation_id', self.causation_id)
        )

    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            'id': self.id,
            'type': self.type,
            'timestamp': self.timestamp,
            'priority': self.priority.value,
            'category': self.category.value,
            'source': self.source,
            'data': self.data,
            'metadata': self.metadata,
            'correlation_id': self.correlation_id,
            'causation_id': self.causation_id
        }

    def to_json(self) -> str:
        """JSON 문자열로 변환"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def __str__(self) -> str:
        """문자열 표현"""
        return f"[{self.category.value}:{self.type}] {self.id} from {self.source} at {datetime.fromtimestamp(self.timestamp).isoformat()}"


class AbstractEventListener(EventListener):
    """기본 이벤트 리스너 추상 클래스"""

    def __init__(self, event_types: Union[str, List[str]], priority: EventPriority = EventPriority.MEDIUM):
        super().__init__(event_types, priority)


class EventTypeFilter(EventFilter):
    """이벤트 타입 필터"""

    def __init__(self, name: str, allowed_types: List[str]):
        super().__init__(name)
        self.allowed_types = set(allowed_types)

    def match(self, event: BaseEvent) -> bool:
        """타입 매칭 확인"""
        return event.type in self.allowed_types or '*' in self.allowed_types


class EventCategoryFilter(EventFilter):
    """이벤트 카테고리 필터"""

    def __init__(self, name: str, allowed_categories: List[EventCategory]):
        super().__init__(name)
        self.allowed_categories = set(allowed_categories)

    def match(self, event: BaseEvent) -> bool:
        """카테고리 매칭 확인"""
        return event.category in self.allowed_categories


class EventPriorityFilter(EventFilter):
    """이벤트 우선순위 필터"""

    PRIORITY_ORDER = [
        EventPriority.LOW,
        EventPriority.MEDIUM,
        EventPriority.HIGH,
        EventPriority.CRITICAL
    ]

    def __init__(self, name: str, min_priority: EventPriority):
        super().__init__(name)
        self.min_priority = min_priority

    def match(self, event: BaseEvent) -> bool:
        """우선순위 매칭 확인"""
        event_priority_index = self.PRIORITY_ORDER.index(event.priority)
        min_priority_index = self.PRIORITY_ORDER.index(self.min_priority)
        return event_priority_index >= min_priority_index


class EventEmitter:
    """기본 이벤트 에미터 클래스"""

    def __init__(self):
        self._listeners = {}

    async def emit_event(self, event_type: str, data: Any = None):
        """이벤트 발생"""
        if event_type in self._listeners:
            for listener in self._listeners[event_type]:
                try:
                    if asyncio.iscoroutinefunction(listener):
                        await listener(data)
                    else:
                        listener(data)
                except Exception as e:
                    # 에러를 무시하고 계속 진행
                    pass

    def on(self, event_type: str, listener):
        """이벤트 리스너 등록"""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(listener)

    def off(self, event_type: str, listener):
        """이벤트 리스너 제거"""
        if event_type in self._listeners:
            if listener in self._listeners[event_type]:
                self._listeners[event_type].remove(listener)

    def emit(self, event_type: str, data: Any = None):
        """동기 이벤트 발생 (호환성)"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 이미 실행 중인 루프에서는 태스크로 실행
                asyncio.create_task(self.emit_event(event_type, data))
            else:
                # 새 루프에서 실행
                loop.run_until_complete(self.emit_event(event_type, data))
        except RuntimeError:
            # 루프가 없으면 새로 생성
            asyncio.run(self.emit_event(event_type, data))


# Event 별명 (호환성)
Event = PacaEvent