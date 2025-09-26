"""
Core Types Module
시스템 전반에서 사용되는 기본 공통 타입 정의
TypeScript Result 타입과 완전 호환되는 Python 구현
"""

import time
import uuid
from typing import TypeVar, Generic, Union, Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# Type aliases
ID = str
Timestamp = float  # Unix timestamp in seconds
KeyValuePair = Dict[str, Any]

T = TypeVar('T')
E = TypeVar('E', bound=Exception)


@dataclass(frozen=True)
class Result(Generic[T]):
    """
    TypeScript Result 타입의 Python 구현
    성공/실패 결과를 명확하게 구분하여 처리
    """
    is_success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    metadata: KeyValuePair = field(default_factory=dict)

    @property
    def is_failure(self) -> bool:
        """실패 여부 확인"""
        return not self.is_success

    @property
    def value(self) -> T:
        """성공시 데이터 반환, 실패시 예외 발생"""
        if self.is_success and self.data is not None:
            return self.data
        raise ValueError(f"Result is failure: {self.error}")

    def unwrap(self) -> T:
        """값 언래핑 (value의 별칭)"""
        return self.value

    def unwrap_or(self, default: T) -> T:
        """실패시 기본값 반환"""
        return self.data if self.is_success and self.data is not None else default

    def map(self, func: Callable[[T], Any]) -> 'Result[Any]':
        """성공 결과에 함수 적용"""
        if self.is_success and self.data is not None:
            try:
                new_data = func(self.data)
                return Result(True, new_data, None, self.metadata)
            except Exception as e:
                return Result(False, None, str(e), self.metadata)
        return Result(False, None, self.error, self.metadata)

    def __bool__(self) -> bool:
        """불린 컨텍스트에서 성공 여부 반환"""
        return self.is_success


class LogLevel(Enum):
    """로그 레벨 열거형"""
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    FATAL = "fatal"


class Status(Enum):
    """상태 열거형"""
    IDLE = "idle"
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(Enum):
    """우선순위 열거형"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5


class CognitiveState(Enum):
    """인지 시스템의 현재 상태"""
    IDLE = "idle"
    PROCESSING = "processing"
    LEARNING = "learning"
    REASONING = "reasoning"
    ERROR = "error"


@dataclass
class BaseConfig:
    """설정 객체 기본 클래스"""
    id: ID
    name: str
    version: str
    enabled: bool = True
    created_at: Timestamp = field(default_factory=time.time)
    updated_at: Timestamp = field(default_factory=time.time)


@dataclass
class BaseEntity:
    """기본 엔티티 클래스"""
    id: ID = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: Timestamp = field(default_factory=time.time)
    updated_at: Timestamp = field(default_factory=time.time)
    metadata: KeyValuePair = field(default_factory=dict)

    def update_timestamp(self) -> None:
        """업데이트 시간 갱신"""
        self.updated_at = time.time()


@dataclass
class BaseEvent:
    """기본 이벤트 클래스"""
    id: ID = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    timestamp: Timestamp = field(default_factory=time.time)
    source: str = ""
    data: KeyValuePair = field(default_factory=dict)


@dataclass
class PaginationRequest:
    """페이지네이션 요청"""
    page: int
    limit: int
    sort_by: Optional[str] = None
    sort_order: str = "asc"  # "asc" | "desc"

    def __post_init__(self):
        if self.page < 1:
            raise ValueError("Page must be >= 1")
        if self.limit < 1:
            raise ValueError("Limit must be >= 1")
        if self.sort_order not in ("asc", "desc"):
            raise ValueError("Sort order must be 'asc' or 'desc'")


@dataclass
class PaginationResponse(Generic[T]):
    """페이지네이션 응답"""
    items: List[T]
    total_items: int
    total_pages: int
    current_page: int
    has_next_page: bool
    has_previous_page: bool

    @classmethod
    def create(cls, items: List[T], total_items: int, page: int, limit: int) -> 'PaginationResponse[T]':
        """페이지네이션 응답 생성 헬퍼"""
        total_pages = (total_items + limit - 1) // limit  # Ceiling division
        return cls(
            items=items,
            total_items=total_items,
            total_pages=total_pages,
            current_page=page,
            has_next_page=page < total_pages,
            has_previous_page=page > 1
        )


@dataclass
class BaseFilter:
    """검색 필터 기본 클래스"""
    query: Optional[str] = None
    start_date: Optional[Timestamp] = None
    end_date: Optional[Timestamp] = None
    status: Optional[List[Status]] = None
    tags: Optional[List[str]] = None


@dataclass
class BaseStats:
    """기본 통계 클래스"""
    count: int
    average: float
    minimum: float
    maximum: float
    standard_deviation: float

    @classmethod
    def from_values(cls, values: List[float]) -> 'BaseStats':
        """값 리스트로부터 통계 생성"""
        if not values:
            return cls(0, 0.0, 0.0, 0.0, 0.0)

        count = len(values)
        average = sum(values) / count
        minimum = min(values)
        maximum = max(values)

        # 표준편차 계산
        variance = sum((x - average) ** 2 for x in values) / count
        std_dev = variance ** 0.5

        return cls(count, average, minimum, maximum, std_dev)


@dataclass
class LearningPoint:
    """학습 포인트 데이터 클래스"""
    id: ID = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    confidence: float = 0.0
    timestamp: Timestamp = field(default_factory=time.time)
    metadata: KeyValuePair = field(default_factory=dict)
    user_context: Optional[str] = None
    importance_score: float = 0.5

    def __post_init__(self):
        """유효성 검사"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not 0.0 <= self.importance_score <= 1.0:
            raise ValueError("Importance score must be between 0.0 and 1.0")


# Helper functions for Result type
def create_result(success: bool, data: Optional[T] = None, error: Optional[str] = None,
                 metadata: Optional[KeyValuePair] = None) -> Result[T]:
    """Result 객체 생성 헬퍼"""
    return Result(
        is_success=success,
        data=data,
        error=error,
        metadata=metadata or {}
    )


def create_success(data: T, metadata: Optional[KeyValuePair] = None) -> Result[T]:
    """성공 Result 생성"""
    return Result(
        is_success=True,
        data=data,
        error=None,
        metadata=metadata or {}
    )


def create_failure(error: str, metadata: Optional[KeyValuePair] = None) -> Result[Any]:
    """실패 Result 생성"""
    return Result(
        is_success=False,
        data=None,
        error=error,
        metadata=metadata or {}
    )


# Type validation functions
def is_valid_id(value: Any) -> bool:
    """ID 유효성 검사"""
    return isinstance(value, str) and len(value) > 0


def is_valid_timestamp(value: Any) -> bool:
    """타임스탬프 유효성 검사"""
    return isinstance(value, (int, float)) and value >= 0