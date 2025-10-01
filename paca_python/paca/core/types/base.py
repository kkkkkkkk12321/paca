"""Base type definitions that are shared across the PACA runtime."""

from typing import TypeVar, Generic, Union, Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import time

# 시스템 내 고유 식별자
ID = str

# 타임스탬프 (초 단위 float)
Timestamp = float

# 제네릭 타입 변수
T = TypeVar('T')
E = TypeVar('E', bound=Exception)

# 키-값 쌍 객체 타입
KeyValuePair = Dict[str, Any]


@dataclass(frozen=True)
class Result(Generic[T]):
    """Container describing whether an operation succeeded and any payload."""

    is_success: bool
    data: Optional[T] = None
    error: Optional[Union[str, Exception]] = None
    metadata: KeyValuePair = field(default_factory=dict)

    @property
    def is_failure(self) -> bool:
        return not self.is_success

    @property
    def value(self) -> T:
        if self.is_success and self.data is not None:
            return self.data
        raise ValueError("Cannot access value of failed result")

    @classmethod
    def success(
        cls, data: T, *, metadata: Optional[KeyValuePair] = None
    ) -> "Result[T]":
        return cls(is_success=True, data=data, error=None, metadata=metadata or {})

    @classmethod
    def failure(
        cls, error: Union[str, Exception], *, metadata: Optional[KeyValuePair] = None
    ) -> "Result[T]":
        return cls(is_success=False, error=error, metadata=metadata or {})


class LogLevel(Enum):
    """로그 레벨 열거형"""
    DEBUG = 'debug'
    INFO = 'info'
    WARN = 'warn'
    ERROR = 'error'
    FATAL = 'fatal'


class Status(Enum):
    """상태 열거형"""
    IDLE = 'idle'
    PENDING = 'pending'
    RUNNING = 'running'
    PROCESSING = 'processing'
    SUCCESS = 'success'
    FAILED = 'failed'
    CANCELLED = 'cancelled'
    # PACA 시스템 전용 상태
    INITIALIZING = 'initializing'
    ERROR = 'error'
    READY = 'ready'
    SHUTTING_DOWN = 'shutting_down'
    SHUTDOWN = 'shutdown'


class Priority(Enum):
    """우선순위 열거형"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5


@dataclass(frozen=True)
class BaseConfig:
    """설정 객체 기본 인터페이스"""
    id: ID
    name: str
    version: str
    enabled: bool
    created_at: Timestamp
    updated_at: Timestamp


@dataclass(frozen=True)
class BaseEntity:
    """기본 엔티티 인터페이스"""
    id: ID
    created_at: Timestamp
    updated_at: Timestamp
    metadata: Optional[KeyValuePair] = None


@dataclass(frozen=True)
class BaseEvent:
    """기본 이벤트 인터페이스"""
    id: ID
    type: str
    timestamp: Timestamp
    source: str
    data: KeyValuePair


@dataclass(frozen=True)
class PaginationRequest:
    """페이지네이션 요청 인터페이스"""
    page: int
    limit: int
    sort_by: Optional[str] = None
    sort_order: Optional[str] = None  # 'asc' | 'desc'


@dataclass(frozen=True)
class PaginationResponse(Generic[T]):
    """페이지네이션 응답 인터페이스"""
    items: List[T]
    total_items: int
    total_pages: int
    current_page: int
    has_next_page: bool
    has_previous_page: bool


@dataclass(frozen=True)
class BaseFilter:
    """검색 필터 기본 인터페이스"""
    query: Optional[str] = None
    start_date: Optional[Timestamp] = None
    end_date: Optional[Timestamp] = None
    status: Optional[List[Status]] = None
    tags: Optional[List[str]] = None


@dataclass(frozen=True)
class BaseStats:
    """기본 통계 인터페이스"""
    count: int
    average: float
    minimum: float
    maximum: float
    standard_deviation: float


def current_timestamp() -> Timestamp:
    """현재 타임스탬프 반환"""
    return time.time()


def create_id() -> ID:
    """고유 ID 생성"""
    import uuid
    return str(uuid.uuid4())


# 별칭 및 확장 함수들
def generate_id(prefix: str = "") -> ID:
    """접두사를 포함한 고유 ID 생성"""
    import uuid
    base_id = str(uuid.uuid4()).replace('-', '')[:8]
    return f"{prefix}{base_id}" if prefix else base_id


# Result helper functions
def create_success(
    data: T, metadata: Optional[KeyValuePair] = None
) -> Result[T]:
    """성공 결과 생성"""

    return Result(is_success=True, data=data, error=None, metadata=metadata or {})


def create_failure(
    error: Union[str, Exception], metadata: Optional[KeyValuePair] = None
) -> Result[None]:
    """실패 결과 생성"""

    return Result(is_success=False, data=None, error=error, metadata=metadata or {})


def create_result(
    success: bool,
    data: Optional[T] = None,
    error: Optional[Union[str, Exception]] = None,
    metadata: Optional[KeyValuePair] = None,
) -> Result[T]:
    """Result 객체 생성 헬퍼"""
    return Result(
        is_success=success,
        data=data,
        error=error,
        metadata=metadata or {}
    )
