"""
Memory Types
메모리 관련 공통 타입 정의
"""

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from ...core.types import ID, Timestamp, KeyValuePair


class MemoryType(Enum):
    """메모리 타입"""
    WORKING = 'working'
    EPISODIC = 'episodic'
    SEMANTIC = 'semantic'
    PROCEDURAL = 'procedural'
    LONG_TERM = 'long_term'
    SHORT_TERM = 'short_term'


class MemoryOperation(Enum):
    """메모리 연산"""
    STORE = 'store'
    RETRIEVE = 'retrieve'
    UPDATE = 'update'
    DELETE = 'delete'
    SEARCH = 'search'
    CONSOLIDATE = 'consolidate'


@dataclass
class MemoryConfiguration:
    """작업 메모리 및 공통 설정"""
    working_memory_capacity: int = 7
    long_term_retention: bool = True
    episodic_context_tracking: bool = True
    consolidation_threshold: float = 0.8
    decay_rate: float = 0.1
    interference_factor: float = 0.05
    working_memory_ttl_seconds: Optional[int] = 600
    cleanup_interval_seconds: int = 60
    ttl_disabled_fallback: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryConfiguration":
        """사전 데이터를 이용해 설정 인스턴스 생성"""
        kwargs: Dict[str, Any] = {}
        for field_def in fields(cls):
            if field_def.name in data:
                kwargs[field_def.name] = data[field_def.name]
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """설정을 사전 형태로 변환"""
        return {field_def.name: getattr(self, field_def.name) for field_def in fields(self)}


@dataclass
class EpisodicMemorySettings:
    """일화 메모리 전용 설정"""
    retention_days: Optional[float] = 30.0
    snapshot_interval_seconds: Optional[int] = 1800
    enable_async_io: bool = True
    max_snapshot_items: Optional[int] = 200

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodicMemorySettings":
        kwargs: Dict[str, Any] = {}
        for field_def in fields(cls):
            if field_def.name in data:
                kwargs[field_def.name] = data[field_def.name]
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {field_def.name: getattr(self, field_def.name) for field_def in fields(self)}


@dataclass
class LongTermMemorySettings:
    """장기 메모리 전용 설정"""
    cleanup_policy: str = "priority"
    max_items: Optional[int] = 5000
    min_strength_threshold: Optional[float] = None
    max_idle_seconds: Optional[int] = 30 * 24 * 3600
    cleanup_batch_size: int = 100
    persistent_db: bool = True
    database_name: str = "longterm_memory.db"
    storage_adapter: str = "sqlite"
    connection_uri: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LongTermMemorySettings":
        kwargs: Dict[str, Any] = {}
        for field_def in fields(cls):
            if field_def.name in data:
                kwargs[field_def.name] = data[field_def.name]
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {field_def.name: getattr(self, field_def.name) for field_def in fields(self)}


@dataclass
class MemoryMetrics:
    """메모리 성능 지표"""
    total_stored_items: int = 0
    average_retrieval_time: float = 0.0
    hit_rate: float = 0.0
    memory_efficiency: float = 0.0
    consolidation_rate: float = 0.0


@dataclass
class MemoryItem:
    """메모리 아이템"""
    id: ID
    content: Any
    memory_type: MemoryType
    created_at: Timestamp
    accessed_at: Timestamp
    access_count: int = 0
    strength: float = 1.0
    context: KeyValuePair = field(default_factory=dict)
    associations: List[ID] = field(default_factory=list)
    metadata: KeyValuePair = field(default_factory=dict)


@dataclass
class SearchQuery:
    """검색 쿼리"""
    query: Union[str, Dict[str, Any]]
    memory_types: List[MemoryType] = field(default_factory=lambda: [MemoryType.LONG_TERM])
    limit: int = 10
    threshold: float = 0.7
    include_context: bool = True
    sort_by: str = 'relevance'  # 'relevance', 'recency', 'strength'


@dataclass
class SearchResult:
    """검색 결과"""
    items: List[MemoryItem]
    total_found: int
    query_time: float
    relevance_scores: Dict[ID, float] = field(default_factory=dict)


@dataclass
class ConsolidationRequest:
    """통합 요청"""
    source_items: List[ID]
    target_memory_type: MemoryType
    consolidation_strategy: str = 'similarity'  # 'similarity', 'frequency', 'importance'
    metadata: KeyValuePair = field(default_factory=dict)
