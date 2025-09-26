"""
Working Memory Implementation
작업 메모리 구현
"""

import asyncio
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

CONFIG_FILENAME = "memory_settings.json"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "config" / CONFIG_FILENAME

from .types import (
    MemoryType, MemoryOperation, MemoryConfiguration, MemoryMetrics,
    MemoryItem, SearchQuery, SearchResult
)
from ...core.types import ID, Timestamp, KeyValuePair, create_id, current_timestamp
from ...core.errors import MemoryError
from ...core.utils.portable_storage import get_storage_manager


class WorkingMemory:
    """작업 메모리 시스템"""

    def __init__(
        self,
        config: Optional[MemoryConfiguration] = None,
        storage_manager: Optional[Any] = None,
    ):
        self.config = config or self._load_configuration()
        self.items: List[MemoryItem] = []
        self.capacity = self.config.working_memory_capacity
        self.metrics = MemoryMetrics()
        self.ttl_seconds = max(0, self.config.working_memory_ttl_seconds or 0)
        self.cleanup_interval = max(1, int(self.config.cleanup_interval_seconds or 60))
        self._cleanup_task: Optional[asyncio.Task] = None
        self._load_task: Optional[asyncio.Task] = None
        self._initialized = False

        # 포터블 저장소 설정
        self.storage_manager = storage_manager or get_storage_manager()
        self.memory_file_path = self.storage_manager.get_memory_file_path(
            "working", "working_memory.json"
        )

        self.logger = logging.getLogger("WorkingMemory")

        # 기존 데이터 로드는 initialize 단계에서 처리
        self._schedule_initial_load()

    async def store(self, content: Any, context: Optional[KeyValuePair] = None) -> ID:
        """작업 메모리에 항목 저장"""
        start_time = time.time()

        try:
            await self._ensure_initialized()
            await self._remove_expired()

            # 용량 확인 및 필요시 제거
            if len(self.items) >= self.capacity:
                await self._evict_least_recent()

            # 새 항목 생성
            item = MemoryItem(
                id=create_id(),
                content=content,
                memory_type=MemoryType.WORKING,
                created_at=current_timestamp(),
                accessed_at=current_timestamp(),
                context=context or {},
                access_count=1,
                metadata={}
            )

            if self.ttl_seconds:
                item.metadata['expires_at'] = item.created_at + self.ttl_seconds

            self.items.append(item)

            # 메트릭 업데이트
            self.metrics.total_stored_items += 1
            self._update_retrieval_time(time.time() - start_time)

            # 포터블 저장소에 저장
            await self._save_to_storage()

            self.logger.debug(f"Stored item {item.id} in working memory")
            return item.id

        except Exception as error:
            raise MemoryError(
                f"작업 메모리 저장 실패: {str(error)}",
                memory_type="working",
                operation="store"
            )

    async def retrieve(self, item_id: ID) -> Optional[MemoryItem]:
        """항목 검색"""
        start_time = time.time()

        try:
            await self._ensure_initialized()
            await self._remove_expired()
            for item in self.items:
                if item.id == item_id:
                    # 접근 정보 업데이트
                    item.accessed_at = current_timestamp()
                    item.access_count += 1

                    self._update_retrieval_time(time.time() - start_time)
                    self.metrics.hit_rate = self._calculate_hit_rate()

                    return item

            self._update_retrieval_time(time.time() - start_time)
            return None

        except Exception as error:
            raise MemoryError(
                f"작업 메모리 검색 실패: {str(error)}",
                memory_type="working",
                operation="retrieve"
            )

    async def search(self, query: SearchQuery) -> SearchResult:
        """작업 메모리 검색"""
        start_time = time.time()

        try:
            await self._remove_expired()
            results = []
            relevance_scores = {}

            for item in self.items:
                score = self._calculate_relevance(item, query.query)
                if score >= query.threshold:
                    results.append(item)
                    relevance_scores[item.id] = score

                    # 접근 정보 업데이트
                    item.accessed_at = current_timestamp()
                    item.access_count += 1

            # 정렬
            if query.sort_by == 'relevance':
                results.sort(key=lambda x: relevance_scores[x.id], reverse=True)
            elif query.sort_by == 'recency':
                results.sort(key=lambda x: x.accessed_at, reverse=True)
            elif query.sort_by == 'strength':
                results.sort(key=lambda x: x.strength, reverse=True)

            # 제한 적용
            results = results[:query.limit]

            query_time = time.time() - start_time

            return SearchResult(
                items=results,
                total_found=len(results),
                query_time=query_time,
                relevance_scores=relevance_scores
            )

        except Exception as error:
            raise MemoryError(
                f"작업 메모리 검색 실패: {str(error)}",
                memory_type="working",
                operation="search"
            )

    async def update(self, item_id: ID, content: Any, context: Optional[KeyValuePair] = None) -> bool:
        """항목 업데이트"""
        try:
            await self._ensure_initialized()
            await self._remove_expired()
            for item in self.items:
                if item.id == item_id:
                    item.content = content
                    if context:
                        item.context.update(context)
                    item.accessed_at = current_timestamp()
                    item.access_count += 1

                    await self._save_to_storage()
                    self.logger.debug(f"Updated item {item_id} in working memory")
                    return True

            return False

        except Exception as error:
            raise MemoryError(
                f"작업 메모리 업데이트 실패: {str(error)}",
                memory_type="working",
                operation="update"
            )

    async def delete(self, item_id: ID) -> bool:
        """항목 삭제"""
        try:
            await self._ensure_initialized()
            await self._remove_expired()
            for i, item in enumerate(self.items):
                if item.id == item_id:
                    del self.items[i]
                    await self._save_to_storage()
                    self.logger.debug(f"Deleted item {item_id} from working memory")
                    return True

            return False

        except Exception as error:
            raise MemoryError(
                f"작업 메모리 삭제 실패: {str(error)}",
                memory_type="working",
                operation="delete"
            )

    async def clear(self) -> None:
        """작업 메모리 비우기"""
        try:
            await self._ensure_initialized()
            self.items.clear()
            await self._save_to_storage()
            self.logger.debug("Cleared working memory")

        except Exception as error:
            raise MemoryError(
                f"작업 메모리 초기화 실패: {str(error)}",
                memory_type="working",
                operation="clear"
            )

    async def cleanup_expired(self) -> int:
        """만료된 항목을 즉시 정리"""
        await self._ensure_initialized()
        removed = await self._remove_expired()
        if removed:
            self.logger.debug(f"Removed {removed} expired working-memory items")
        return removed

    def get_metrics(self) -> MemoryMetrics:
        """성능 지표 반환"""
        self.metrics.memory_efficiency = self._calculate_efficiency()
        return self.metrics

    def get_capacity_info(self) -> Dict[str, Any]:
        """용량 정보 반환"""
        return {
            'current_size': len(self.items),
            'capacity': self.capacity,
            'utilization': len(self.items) / self.capacity,
            'available_slots': self.capacity - len(self.items)
        }

    def get_all_items(self) -> List[MemoryItem]:
        """모든 항목 반환"""
        return self.items.copy()

    async def _evict_least_recent(self) -> None:
        """가장 오래된 항목 제거"""
        if not self.items:
            return

        # 가장 오래 전에 접근된 항목 찾기
        oldest_item = min(self.items, key=lambda x: x.accessed_at)
        await self.delete(oldest_item.id)

    def _calculate_relevance(self, item: MemoryItem, query: Any) -> float:
        """관련성 점수 계산"""
        if isinstance(query, str):
            # 텍스트 기반 유사도 (간단한 구현)
            content_str = str(item.content).lower()
            query_str = query.lower()

            if query_str in content_str:
                return 1.0

            # 키워드 기반 부분 일치
            query_words = set(query_str.split())
            content_words = set(content_str.split())
            intersection = query_words.intersection(content_words)

            if len(query_words) > 0:
                return len(intersection) / len(query_words)

        elif isinstance(query, dict):
            # 컨텍스트 기반 매칭
            matches = 0
            total = len(query)

            for key, value in query.items():
                if key in item.context and item.context[key] == value:
                    matches += 1

            return matches / total if total > 0 else 0.0

        return 0.0

    def _update_retrieval_time(self, time_taken: float) -> None:
        """검색 시간 업데이트"""
        if self.metrics.average_retrieval_time == 0:
            self.metrics.average_retrieval_time = time_taken
        else:
            # 지수 이동 평균
            alpha = 0.1
            self.metrics.average_retrieval_time = (
                alpha * time_taken +
                (1 - alpha) * self.metrics.average_retrieval_time
            )

    def _calculate_hit_rate(self) -> float:
        """적중률 계산"""
        # 간단한 구현 - 실제로는 더 정교한 추적 필요
        if len(self.items) == 0:
            return 0.0

        total_accesses = sum(item.access_count for item in self.items)
        if total_accesses == 0:
            return 0.0

        return len(self.items) / total_accesses

    def _calculate_efficiency(self) -> float:
        """메모리 효율성 계산"""
        if len(self.items) == 0:
            return 0.0

        # 접근 빈도와 최신성을 고려한 효율성
        current_time = current_timestamp()
        efficiency_sum = 0.0

        for item in self.items:
            recency_factor = 1.0 / (current_time - item.accessed_at + 1)
            frequency_factor = item.access_count
            efficiency_sum += recency_factor * frequency_factor

        return efficiency_sum / len(self.items)

    # 키-값 저장을 위한 편의 메서드들
    async def store_kv(self, key: str, value: Any) -> ID:
        """키-값 쌍 저장"""
        return await self.store(value, {'key': key})

    async def retrieve_by_key(self, key: str) -> Optional[Any]:
        """키로 값 검색"""
        await self._ensure_initialized()
        await self._remove_expired()
        for item in self.items:
            if item.context.get('key') == key:
                # 접근 정보 업데이트
                item.accessed_at = current_timestamp()
                item.access_count += 1
                return item.content
        return None

    async def _save_to_storage(self) -> None:
        """메모리 데이터를 포터블 저장소에 저장"""
        try:
            # MemoryItem을 직렬화 가능한 형태로 변환
            serializable_items = []
            for item in self.items:
                item_dict = {
                    'id': item.id,
                    'content': item.content,
                    'memory_type': item.memory_type.value,
                    'created_at': item.created_at,
                    'accessed_at': item.accessed_at,
                    'access_count': item.access_count,
                    'strength': item.strength,
                    'context': item.context,
                    'associations': item.associations,
                    'metadata': item.metadata
                }
                serializable_items.append(item_dict)

            # 메트릭 정보도 포함
            data = {
                'items': serializable_items,
                'metrics': {
                    'total_stored_items': self.metrics.total_stored_items,
                    'average_retrieval_time': self.metrics.average_retrieval_time,
                    'hit_rate': self.metrics.hit_rate,
                    'memory_efficiency': self.metrics.memory_efficiency,
                    'consolidation_rate': self.metrics.consolidation_rate
                },
                'saved_at': datetime.now().isoformat()
            }

            self.storage_manager.save_json_data(self.memory_file_path, data)

        except Exception as e:
            self.logger.error(f"Failed to save working memory to storage: {e}")

    async def _load_from_storage(self) -> None:
        """포터블 저장소에서 메모리 데이터 로드"""
        try:
            data = self.storage_manager.load_json_data(self.memory_file_path)
            if not data:
                return

            # 아이템 복원
            self.items = []
            for item_dict in data.get('items', []):
                item = MemoryItem(
                    id=item_dict['id'],
                    content=item_dict['content'],
                    memory_type=MemoryType(item_dict['memory_type']),
                    created_at=item_dict['created_at'],
                    accessed_at=item_dict['accessed_at'],
                    access_count=item_dict['access_count'],
                    strength=item_dict['strength'],
                    context=item_dict['context'],
                    associations=item_dict['associations'],
                    metadata=item_dict['metadata']
                )
                self.items.append(item)

            # 메트릭 복원
            if 'metrics' in data:
                metrics_data = data['metrics']
                self.metrics.total_stored_items = metrics_data.get('total_stored_items', 0)
                self.metrics.average_retrieval_time = metrics_data.get('average_retrieval_time', 0.0)
                self.metrics.hit_rate = metrics_data.get('hit_rate', 0.0)
                self.metrics.memory_efficiency = metrics_data.get('memory_efficiency', 0.0)
                self.metrics.consolidation_rate = metrics_data.get('consolidation_rate', 0.0)

            self.logger.debug(f"Loaded {len(self.items)} items from working memory storage")
            await self._remove_expired(persist=False)

        except Exception as e:
            self.logger.error(f"Failed to load working memory from storage: {e}")
            # 로드 실패해도 빈 상태로 시작

    def _load_configuration(self) -> MemoryConfiguration:
        """외부 설정에서 작업 메모리 구성을 로드"""
        if DEFAULT_CONFIG_PATH.exists():
            try:
                with DEFAULT_CONFIG_PATH.open(encoding='utf-8') as handle:
                    data = json.load(handle)
                working_config = data.get('working_memory', {})
                if working_config:
                    return MemoryConfiguration.from_dict(working_config)
            except Exception as error:
                self.logger.warning(
                    "Failed to load working memory configuration, using defaults",
                    extra={"error": str(error)}
                )
        return MemoryConfiguration()

    def _ensure_cleanup_task(self) -> None:
        if not self.ttl_seconds or self._cleanup_task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        self._cleanup_task = loop.create_task(self._expiration_loop())

    def _schedule_initial_load(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        if self._load_task is None:
            self._load_task = loop.create_task(self._load_from_storage())

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        if self._load_task is not None:
            try:
                await self._load_task
            finally:
                self._load_task = None
        else:
            await self._load_from_storage()

        self._ensure_cleanup_task()
        self._initialized = True

    async def initialize(self) -> bool:
        """외부 시스템에서 호출하는 초기화 헬퍼"""
        await self._ensure_initialized()
        return True

    async def _remove_expired(self, *, persist: bool = True) -> int:
        if not self.ttl_seconds:
            return 0

        current_time = current_timestamp()
        remaining_items: List[MemoryItem] = []
        removed = 0

        for item in self.items:
            expires_at = item.metadata.get('expires_at') if item.metadata else None
            if expires_at and current_time >= expires_at:
                removed += 1
            else:
                remaining_items.append(item)

        if removed:
            self.items = remaining_items
            if persist:
                await self._save_to_storage()

        return removed

    async def _expiration_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.cleanup_interval)
                removed = await self._remove_expired()
                if removed:
                    self.logger.debug(
                        "Periodic cleanup removed %d expired working-memory items",
                        removed
                    )
        except asyncio.CancelledError:
            self.logger.debug("Working memory expiration loop cancelled")

    async def shutdown(self) -> None:
        """백그라운드 작업 정리"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            finally:
                self._cleanup_task = None

        if self._initialized:
            await self._save_to_storage()
            self._initialized = False
