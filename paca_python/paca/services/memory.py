"""
Memory Service Module
메모리 관리 서비스
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass, field
import weakref

from ..core.types.base import ID, Result, Priority
from ..core.errors.base import ValidationError, SystemError
from ..core.utils.async_utils import create_logger
from ..core.events.base import EventEmitter, Event


class MemoryType(Enum):
    """메모리 항목 타입"""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryPriority(Enum):
    """메모리 우선순위"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class MemoryStatus(Enum):
    """메모리 항목 상태"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    EXPIRED = "expired"


@dataclass
class MemoryItem:
    """메모리 항목"""
    id: ID
    type: MemoryType
    content: Any
    tags: List[str]
    priority: MemoryPriority
    created_at: int
    last_accessed_at: int
    status: MemoryStatus = MemoryStatus.ACTIVE
    access_count: int = 0
    associations: List[ID] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[int] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = int(time.time() * 1000)
        if not self.last_accessed_at:
            self.last_accessed_at = self.created_at


@dataclass
class MemoryQuery:
    """메모리 쿼리"""
    type: Optional[MemoryType] = None
    tags: Optional[List[str]] = None
    priority: Optional[MemoryPriority] = None
    status: Optional[MemoryStatus] = None
    content_filter: Optional[str] = None
    time_range: Optional[Dict[str, int]] = None  # {"start": timestamp, "end": timestamp}
    limit: Optional[int] = None
    sort_by: str = "created"  # "created", "accessed", "priority"
    sort_order: str = "desc"  # "asc", "desc"


@dataclass
class MemoryStatistics:
    """메모리 통계"""
    total_items: int
    items_by_type: Dict[MemoryType, int]
    items_by_status: Dict[MemoryStatus, int]
    average_access_count: float
    memory_usage_bytes: int
    oldest_item: Optional[MemoryItem] = None
    most_accessed_item: Optional[MemoryItem] = None


class MemoryService(EventEmitter):
    """메모리 서비스"""

    def __init__(self, max_memory_size: int = 10000):
        super().__init__()
        self.logger = create_logger("MemoryService")
        self.memories: Dict[ID, MemoryItem] = {}
        self.memory_index: Dict[str, Set[ID]] = {}  # tag -> memory IDs
        self.max_memory_size = max_memory_size
        self._cleanup_task: Optional[asyncio.Task] = None
        # 생성자에서는 cleanup 프로세스를 시작하지 않음

    async def store(
        self,
        memory_type: MemoryType,
        content: Any,
        options: Optional[Dict[str, Any]] = None
    ) -> Result:
        """메모리 저장"""
        try:
            options = options or {}
            memory_id = f"memory_{int(time.time() * 1000)}_{uuid.uuid4().hex[:9]}"
            now = int(time.time() * 1000)

            memory_item = MemoryItem(
                id=memory_id,
                type=memory_type,
                content=content,
                tags=options.get("tags", []),
                priority=options.get("priority", MemoryPriority.NORMAL),
                created_at=now,
                last_accessed_at=now,
                expires_at=options.get("expires_at"),
                status=MemoryStatus.ACTIVE,
                access_count=0,
                associations=options.get("associations", []),
                metadata=options.get("metadata", {})
            )

            # 메모리 크기 제한 확인
            if len(self.memories) >= self.max_memory_size:
                await self._evict_old_memories()

            self.memories[memory_id] = memory_item

            # 인덱스 업데이트
            self._update_index(memory_item)

            await self.emit_event("memory:stored", {
                "memory_id": memory_id,
                "type": memory_type.value,
                "tags": memory_item.tags
            })

            self.logger.debug("Memory stored", {
                "memory_id": memory_id,
                "type": memory_type.value,
                "tags": memory_item.tags,
                "priority": memory_item.priority.value
            })

            return Result(success=True, data=memory_item)

        except Exception as error:
            return Result(
                success=False,
                data=None,
                error=SystemError(
                    f"Failed to store memory: {str(error)}",
                    context={"type": memory_type, "error": str(error)}
                )
            )

    async def retrieve(self, query: MemoryQuery) -> Result:
        """메모리 검색"""
        try:
            candidates = list(self.memories.values())

            # 필터 적용
            if query.type:
                candidates = [item for item in candidates if item.type == query.type]

            if query.status:
                candidates = [item for item in candidates if item.status == query.status]
            else:
                # 기본적으로 활성 메모리만 반환
                candidates = [item for item in candidates if item.status == MemoryStatus.ACTIVE]

            if query.priority:
                candidates = [item for item in candidates if item.priority.value >= query.priority.value]

            if query.tags:
                candidates = [
                    item for item in candidates
                    if any(tag in item.tags for tag in query.tags)
                ]

            if query.content_filter:
                filter_text = query.content_filter.lower()
                candidates = [
                    item for item in candidates
                    if filter_text in json.dumps(item.content).lower()
                ]

            if query.time_range:
                start_time = query.time_range.get("start", 0)
                end_time = query.time_range.get("end", int(time.time() * 1000))
                candidates = [
                    item for item in candidates
                    if start_time <= item.created_at <= end_time
                ]

            # 정렬
            sort_key_map = {
                "created": lambda x: x.created_at,
                "accessed": lambda x: x.last_accessed_at,
                "priority": lambda x: x.priority.value
            }

            sort_key = sort_key_map.get(query.sort_by, sort_key_map["created"])
            reverse = query.sort_order == "desc"

            candidates.sort(key=sort_key, reverse=reverse)

            # 제한 적용
            if query.limit:
                candidates = candidates[:query.limit]

            # 접근 기록 업데이트
            now = int(time.time() * 1000)
            for item in candidates:
                item.last_accessed_at = now
                item.access_count += 1

            self.logger.debug("Memory retrieved", {
                "query": query.__dict__,
                "result_count": len(candidates)
            })

            return Result(success=True, data=candidates)

        except Exception as error:
            return Result(
                success=False,
                data=None,
                error=SystemError(
                    f"Failed to retrieve memory: {str(error)}",
                    context={"query": query.__dict__, "error": str(error)}
                )
            )

    async def get_by_id(self, memory_id: ID) -> Result:
        """특정 메모리 조회"""
        try:
            memory_item = self.memories.get(memory_id)

            if memory_item and memory_item.status == MemoryStatus.ACTIVE:
                # 접근 기록 업데이트
                memory_item.last_accessed_at = int(time.time() * 1000)
                memory_item.access_count += 1

            return Result(success=True, data=memory_item)

        except Exception as error:
            return Result(
                success=False,
                data=None,
                error=SystemError(
                    f"Failed to get memory by ID: {str(error)}",
                    context={"memory_id": memory_id, "error": str(error)}
                )
            )

    async def update(self, memory_id: ID, updates: Dict[str, Any]) -> Result:
        """메모리 업데이트"""
        try:
            memory_item = self.memories.get(memory_id)

            if not memory_item:
                return Result(
                    success=False,
                    data=None,
                    error=ValidationError(
                        "Memory item not found",
                        context={"memory_id": memory_id}
                    )
                )

            if memory_item.status != MemoryStatus.ACTIVE:
                return Result(
                    success=False,
                    data=None,
                    error=ValidationError(
                        "Cannot update inactive memory item",
                        context={"memory_id": memory_id, "status": memory_item.status.value}
                    )
                )

            # 기존 인덱스에서 제거
            self._remove_from_index(memory_item)

            # 업데이트 적용
            allowed_fields = ["content", "tags", "priority", "metadata", "associations"]
            for field, value in updates.items():
                if field in allowed_fields and hasattr(memory_item, field):
                    setattr(memory_item, field, value)

            memory_item.last_accessed_at = int(time.time() * 1000)

            # 인덱스 재구축
            self._update_index(memory_item)

            await self.emit_event("memory:updated", {
                "memory_id": memory_id,
                "updates": updates
            })

            self.logger.debug("Memory updated", {"memory_id": memory_id, "updates": updates})

            return Result(success=True, data=memory_item)

        except Exception as error:
            return Result(
                success=False,
                data=None,
                error=ValidationError(
                    f"Failed to update memory: {str(error)}",
                    context={"memory_id": memory_id, "error": str(error)}
                )
            )

    async def delete(self, memory_id: ID) -> Result:
        """메모리 삭제"""
        try:
            memory_item = self.memories.get(memory_id)

            if not memory_item:
                return Result(
                    success=False,
                    data=None,
                    error=ValidationError(
                        "Memory item not found",
                        context={"memory_id": memory_id}
                    )
                )

            # 상태를 삭제됨으로 변경 (물리적 삭제는 정리 과정에서 수행)
            memory_item.status = MemoryStatus.DELETED

            self._remove_from_index(memory_item)

            await self.emit_event("memory:deleted", {"memory_id": memory_id})

            self.logger.debug("Memory marked for deletion", {"memory_id": memory_id})

            return Result(success=True, data=None)

        except Exception as error:
            return Result(
                success=False,
                data=None,
                error=ValidationError(
                    f"Failed to delete memory: {str(error)}",
                    context={"memory_id": memory_id, "error": str(error)}
                )
            )

    def get_memory_statistics(self) -> MemoryStatistics:
        """메모리 통계 조회"""
        active_memories = [
            item for item in self.memories.values()
            if item.status == MemoryStatus.ACTIVE
        ]

        items_by_type = {}
        items_by_status = {}
        total_access_count = 0
        memory_usage_bytes = 0
        oldest_item = None
        most_accessed_item = None
        max_access_count = 0

        for item in self.memories.values():
            # 타입별 통계
            items_by_type[item.type] = items_by_type.get(item.type, 0) + 1
            items_by_status[item.status] = items_by_status.get(item.status, 0) + 1

            if item.status == MemoryStatus.ACTIVE:
                total_access_count += item.access_count
                memory_usage_bytes += len(json.dumps(item.__dict__))

                # 최고 접근 항목
                if item.access_count > max_access_count:
                    max_access_count = item.access_count
                    most_accessed_item = item

                # 최오래된 항목
                if not oldest_item or item.created_at < oldest_item.created_at:
                    oldest_item = item

        average_access_count = (
            total_access_count / len(active_memories) if active_memories else 0.0
        )

        return MemoryStatistics(
            total_items=len(self.memories),
            items_by_type=items_by_type,
            items_by_status=items_by_status,
            average_access_count=average_access_count,
            memory_usage_bytes=memory_usage_bytes,
            oldest_item=oldest_item,
            most_accessed_item=most_accessed_item
        )

    def _update_index(self, memory_item: MemoryItem) -> None:
        """인덱스 업데이트"""
        for tag in memory_item.tags:
            if tag not in self.memory_index:
                self.memory_index[tag] = set()
            self.memory_index[tag].add(memory_item.id)

    def _remove_from_index(self, memory_item: MemoryItem) -> None:
        """인덱스에서 제거"""
        for tag in memory_item.tags:
            if tag in self.memory_index:
                self.memory_index[tag].discard(memory_item.id)
                if not self.memory_index[tag]:
                    del self.memory_index[tag]

    async def _evict_old_memories(self) -> None:
        """오래된 메모리 제거"""
        items = [
            item for item in self.memories.values()
            if item.status == MemoryStatus.ACTIVE and item.priority != MemoryPriority.CRITICAL
        ]

        # 마지막 접근 시간 기준으로 정렬
        items.sort(key=lambda x: x.last_accessed_at)

        evict_count = min(len(items), int(self.max_memory_size * 0.1))

        for i in range(evict_count):
            item = items[i]
            item.status = MemoryStatus.ARCHIVED
            self._remove_from_index(item)

        self.logger.info("Memory eviction completed", {"evict_count": evict_count})

    def start_cleanup_process(self) -> None:
        """정리 프로세스 시작"""
        try:
            loop = asyncio.get_running_loop()
            self._cleanup_task = loop.create_task(self._periodic_cleanup())
        except RuntimeError:
            # 이벤트 루프가 실행 중이 아닌 경우 무시
            pass

    async def _periodic_cleanup(self) -> None:
        """주기적 정리 작업"""
        while True:
            try:
                await asyncio.sleep(60)  # 1분마다 실행
                await self._perform_cleanup()
            except asyncio.CancelledError:
                break
            except Exception as error:
                self.logger.error(f"Cleanup failed: {error}")

    async def _perform_cleanup(self) -> None:
        """정리 작업 수행"""
        now = int(time.time() * 1000)
        cleanup_count = 0

        items_to_remove = []

        for memory_id, item in self.memories.items():
            # 만료된 메모리 제거
            if item.expires_at and now > item.expires_at:
                if item.status == MemoryStatus.ACTIVE:
                    item.status = MemoryStatus.EXPIRED
                    self._remove_from_index(item)
                    cleanup_count += 1

            # 삭제 표시된 메모리 물리적 제거
            if item.status == MemoryStatus.DELETED:
                items_to_remove.append(memory_id)
                cleanup_count += 1

        # 물리적 제거
        for memory_id in items_to_remove:
            del self.memories[memory_id]

        if cleanup_count > 0:
            self.logger.debug("Memory cleanup completed", {"cleanup_count": cleanup_count})

    async def shutdown(self) -> None:
        """서비스 종료"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Memory service shut down")


# 호환성을 위한 별명
ConversationMemory = MemoryService