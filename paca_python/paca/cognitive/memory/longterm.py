"""
Long Term Memory Implementation
장기 메모리 구현 - 의미적 지식과 절차적 지식 저장
"""

import asyncio
import time
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass

from .types import (
    MemoryType,
    MemoryOperation,
    MemoryConfiguration,
    MemoryMetrics,
    MemoryItem,
    SearchQuery,
    SearchResult,
    ConsolidationRequest,
    LongTermMemorySettings,
)
from ...core.types import ID, Timestamp, KeyValuePair, create_id, current_timestamp
from ...core.errors import MemoryError
from ...core.utils.portable_storage import get_storage_manager
from .longterm_storage import SQLiteStorageAdapter


CONFIG_FILENAME = "memory_settings.json"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "config" / CONFIG_FILENAME


@dataclass
class SemanticKnowledge:
    """의미적 지식"""
    concept: str
    definition: str
    properties: Dict[str, Any]
    relationships: Dict[str, List[str]]  # 다른 개념과의 관계


@dataclass
class ProceduralKnowledge:
    """절차적 지식"""
    skill_name: str
    steps: List[str]
    conditions: Dict[str, Any]
    success_rate: float


class LongTermMemory:
    """장기 메모리 시스템"""

    def __init__(
        self,
        config: Optional[MemoryConfiguration] = None,
        db_path: str = ":memory:",
        settings: Optional[LongTermMemorySettings] = None,
        storage_manager: Optional[Any] = None,
    ):
        self.config = config or MemoryConfiguration()
        self.settings = settings or self._load_settings()
        self.storage_manager = storage_manager or get_storage_manager()
        self.logger = logging.getLogger("LongTermMemory")

        adapter_name = (self.settings.storage_adapter or "sqlite").lower()
        if adapter_name != "sqlite":
            self.logger.warning(
                "Unsupported long-term storage adapter '%s'; falling back to sqlite",
                self.settings.storage_adapter,
            )
            adapter_name = "sqlite"

        self.storage_adapter = SQLiteStorageAdapter(
            self.settings,
            self.storage_manager,
            explicit_path=db_path if adapter_name == "sqlite" else ":memory:",
        )
        self.conn = self.storage_adapter.connect()
        self.semantic_knowledge: Dict[str, SemanticKnowledge] = {}
        self.procedural_knowledge: Dict[str, ProceduralKnowledge] = {}
        self.metrics = MemoryMetrics()
        self._initialized = False

        # 데이터베이스 초기화
        self._init_database()
        self._initialized = True

    def _init_database(self) -> None:
        """SQLite 데이터베이스 초기화"""
        try:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    memory_type TEXT,
                    created_at REAL,
                    accessed_at REAL,
                    access_count INTEGER,
                    strength REAL,
                    context TEXT,
                    associations TEXT,
                    metadata TEXT
                )
            ''')

            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS semantic_knowledge (
                    id TEXT PRIMARY KEY,
                    concept TEXT,
                    definition TEXT,
                    properties TEXT,
                    relationships TEXT
                )
            ''')

            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS procedural_knowledge (
                    id TEXT PRIMARY KEY,
                    skill_name TEXT,
                    steps TEXT,
                    conditions TEXT,
                    success_rate REAL
                )
            ''')

            # 인덱스 생성
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_items(memory_type)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON memory_items(created_at)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_concept ON semantic_knowledge(concept)')

            self.conn.commit()

        except Exception as error:
            raise MemoryError(
                f"장기 메모리 데이터베이스 초기화 실패: {str(error)}",
                memory_type="long_term",
                operation="init"
            )

    async def store(self, content: Any, memory_type: MemoryType = MemoryType.LONG_TERM,
                   context: Optional[KeyValuePair] = None, *, strength: float = 1.0) -> ID:
        """장기 메모리에 항목 저장"""
        start_time = time.time()

        try:
            item = MemoryItem(
                id=create_id(),
                content=content,
                memory_type=memory_type,
                created_at=current_timestamp(),
                accessed_at=current_timestamp(),
                context=context or {},
                access_count=1,
                strength=strength,
            )

            # 데이터베이스에 저장
            await self._store_to_db(item)

            # 메트릭 업데이트
            self.metrics.total_stored_items += 1
            self._update_retrieval_time(time.time() - start_time)

            # 보존 정책 적용
            await asyncio.to_thread(self._apply_cleanup_policy_sync)

            self.logger.debug(f"Stored item {item.id} in long-term memory")
            return item.id

        except Exception as error:
            raise MemoryError(
                f"장기 메모리 저장 실패: {str(error)}",
                memory_type="long_term",
                operation="store"
            )

    async def store_semantic_knowledge(self, knowledge: SemanticKnowledge) -> ID:
        """의미적 지식 저장"""
        try:
            knowledge_id = create_id()

            self.conn.execute('''
                INSERT INTO semantic_knowledge
                (id, concept, definition, properties, relationships)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                knowledge_id,
                knowledge.concept,
                knowledge.definition,
                json.dumps(knowledge.properties),
                json.dumps(knowledge.relationships)
            ))

            self.conn.commit()
            self.semantic_knowledge[knowledge_id] = knowledge

            self.logger.debug(f"Stored semantic knowledge: {knowledge.concept}")
            return knowledge_id

        except Exception as error:
            raise MemoryError(
                f"의미적 지식 저장 실패: {str(error)}",
                memory_type="semantic",
                operation="store"
            )

    async def store_procedural_knowledge(self, knowledge: ProceduralKnowledge) -> ID:
        """절차적 지식 저장"""
        try:
            knowledge_id = create_id()

            self.conn.execute('''
                INSERT INTO procedural_knowledge
                (id, skill_name, steps, conditions, success_rate)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                knowledge_id,
                knowledge.skill_name,
                json.dumps(knowledge.steps),
                json.dumps(knowledge.conditions),
                knowledge.success_rate
            ))

            self.conn.commit()
            self.procedural_knowledge[knowledge_id] = knowledge

            self.logger.debug(f"Stored procedural knowledge: {knowledge.skill_name}")
            return knowledge_id

        except Exception as error:
            raise MemoryError(
                f"절차적 지식 저장 실패: {str(error)}",
                memory_type="procedural",
                operation="store"
            )

    async def retrieve(self, item_id: ID) -> Optional[MemoryItem]:
        """항목 검색"""
        start_time = time.time()

        try:
            cursor = self.conn.execute('''
                SELECT * FROM memory_items WHERE id = ?
            ''', (item_id,))

            row = cursor.fetchone()
            if row:
                item = self._row_to_memory_item(row)

                # 접근 정보 업데이트
                item.accessed_at = current_timestamp()
                item.access_count += 1

                await self._update_access_info(item)
                self._update_retrieval_time(time.time() - start_time)

                return item

            return None

        except Exception as error:
            raise MemoryError(
                f"장기 메모리 검색 실패: {str(error)}",
                memory_type="long_term",
                operation="retrieve"
            )

    async def search(self, query: SearchQuery) -> SearchResult:
        """장기 메모리 검색"""
        start_time = time.time()

        try:
            results: List[MemoryItem] = []
            relevance_scores: Dict[str, float] = {}

            if isinstance(query.query, str):
                # 텍스트 기반 검색
                cursor = self.conn.execute('''
                    SELECT * FROM memory_items
                    WHERE content LIKE ?
                    ORDER BY accessed_at DESC
                    LIMIT ?
                ''', (f'%{query.query}%', query.limit))

                for row in cursor.fetchall():
                    item = self._row_to_memory_item(row)
                    score = self._calculate_text_relevance(str(item.content), query.query)

                    if score >= query.threshold:
                        results.append(item)
                        relevance_scores[item.id] = score

                        # 접근 정보 업데이트
                        item.accessed_at = current_timestamp()
                        item.access_count += 1
                        await self._update_access_info(item)

            # 정렬
            if query.sort_by == 'relevance' and relevance_scores:
                results.sort(key=lambda x: relevance_scores[x.id], reverse=True)
            elif query.sort_by == 'recency':
                results.sort(key=lambda x: x.accessed_at, reverse=True)
            elif query.sort_by == 'strength':
                results.sort(key=lambda x: x.strength, reverse=True)

            query_time = time.time() - start_time

            return SearchResult(
                items=results[:query.limit],
                total_found=len(results),
                query_time=query_time,
                relevance_scores=relevance_scores
            )

        except Exception as error:
            raise MemoryError(
                f"장기 메모리 검색 실패: {str(error)}",
                memory_type="long_term",
                operation="search"
            )

    async def search_semantic_knowledge(self, concept: str) -> List[SemanticKnowledge]:
        """의미적 지식 검색"""
        try:
            cursor = self.conn.execute('''
                SELECT * FROM semantic_knowledge
                WHERE concept LIKE ? OR definition LIKE ?
            ''', (f'%{concept}%', f'%{concept}%'))

            results = []
            for row in cursor.fetchall():
                knowledge = SemanticKnowledge(
                    concept=row[1],
                    definition=row[2],
                    properties=json.loads(row[3]),
                    relationships=json.loads(row[4])
                )
                results.append(knowledge)

            return results

        except Exception as error:
            raise MemoryError(
                f"의미적 지식 검색 실패: {str(error)}",
                memory_type="semantic",
                operation="search"
            )

    async def search_procedural_knowledge(self, skill_name: str) -> List[ProceduralKnowledge]:
        """절차적 지식 검색"""
        try:
            cursor = self.conn.execute('''
                SELECT * FROM procedural_knowledge
                WHERE skill_name LIKE ?
            ''', (f'%{skill_name}%',))

            results = []
            for row in cursor.fetchall():
                knowledge = ProceduralKnowledge(
                    skill_name=row[1],
                    steps=json.loads(row[2]),
                    conditions=json.loads(row[3]),
                    success_rate=row[4]
                )
                results.append(knowledge)

            return results

        except Exception as error:
            raise MemoryError(
                f"절차적 지식 검색 실패: {str(error)}",
                memory_type="procedural",
                operation="search"
            )

    async def consolidate(self, request: ConsolidationRequest) -> bool:
        """메모리 통합"""
        try:
            # 소스 항목들 조회
            source_items = []
            for item_id in request.source_items:
                item = await self.retrieve(item_id)
                if item:
                    source_items.append(item)

            if not source_items:
                return False

            # 통합 전략에 따라 처리
            if request.consolidation_strategy == 'similarity':
                consolidated_item = await self._consolidate_by_similarity(source_items)
            elif request.consolidation_strategy == 'frequency':
                consolidated_item = await self._consolidate_by_frequency(source_items)
            else:  # importance
                consolidated_item = await self._consolidate_by_importance(source_items)

            # 새로운 통합 항목 저장
            consolidated_item.memory_type = request.target_memory_type
            await self._store_to_db(consolidated_item)

            # 원본 항목들 삭제
            for item in source_items:
                await self.delete(item.id)

            self.logger.debug(f"Consolidated {len(source_items)} items")
            return True

        except Exception as error:
            raise MemoryError(
                f"메모리 통합 실패: {str(error)}",
                memory_type="long_term",
                operation="consolidate"
            )

    async def delete(self, item_id: ID) -> bool:
        """항목 삭제"""
        try:
            cursor = self.conn.execute('DELETE FROM memory_items WHERE id = ?', (item_id,))
            self.conn.commit()

            return cursor.rowcount > 0

        except Exception as error:
            raise MemoryError(
                f"장기 메모리 삭제 실패: {str(error)}",
                memory_type="long_term",
                operation="delete"
            )

    async def cleanup(self) -> int:
        """외부에서 호출 가능한 정리 메서드"""
        removed = await asyncio.to_thread(self._apply_cleanup_policy_sync)
        if removed:
            self.logger.debug("Long-term cleanup removed %d items", removed)
        return removed

    async def export_items(
        self,
        *,
        memory_type: Optional[MemoryType] = None,
        limit: Optional[int] = None,
        path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """장기 메모리 데이터를 JSON 파일로 내보냅니다."""

        def fetch() -> List[Dict[str, Any]]:
            params: List[Any] = []
            sql = "SELECT * FROM memory_items"
            if memory_type:
                sql += " WHERE memory_type = ?"
                params.append(memory_type.value)
            sql += " ORDER BY created_at DESC"
            if limit:
                sql += " LIMIT ?"
                params.append(limit)
            cursor = self.conn.execute(sql, params)
            rows = cursor.fetchall()
            return [self._memory_item_to_dict(self._row_to_memory_item(row)) for row in rows]

        records = await asyncio.to_thread(fetch)

        export_payload = {
            'exported_at': datetime.now().isoformat(),
            'count': len(records),
            'items': records,
        }

        target_path = Path(path) if path else self.storage_manager.get_memory_file_path(
            "long_term", "longterm_export.json"
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(export_payload, ensure_ascii=False, indent=2, default=str)
        await asyncio.to_thread(target_path.write_text, serialized, 'utf-8')
        return target_path

    async def import_items(
        self,
        source: Optional[Union[str, Path, Dict[str, Any], List[Dict[str, Any]]]] = None,
        *,
        upsert: bool = True,
    ) -> int:
        """JSON 데이터로부터 장기 메모리를 채웁니다."""

        if source is None:
            source = self.storage_manager.get_memory_file_path("long_term", "longterm_export.json")

        if isinstance(source, (str, Path)):
            load_path = Path(source)
            raw = await asyncio.to_thread(load_path.read_text, 'utf-8')
            data_obj = json.loads(raw) if raw else {}
        else:
            data_obj = source or {}

        if isinstance(data_obj, dict) and 'items' in data_obj:
            records = data_obj.get('items', [])
        else:
            records = data_obj if isinstance(data_obj, list) else []

        inserted = 0

        for record in records:
            try:
                item = MemoryItem(
                    id=record.get('id', create_id()),
                    content=record.get('content'),
                    memory_type=MemoryType(record.get('memory_type', MemoryType.LONG_TERM.value)),
                    created_at=record.get('created_at', current_timestamp()),
                    accessed_at=record.get('accessed_at', current_timestamp()),
                    access_count=record.get('access_count', 0),
                    strength=record.get('strength', 1.0),
                    context=record.get('context', {}),
                    associations=record.get('associations', []),
                    metadata=record.get('metadata', {}),
                )

                if upsert:
                    await self._store_to_db(item)
                else:
                    await asyncio.to_thread(self._insert_if_absent_sync, item)

                inserted += 1
            except Exception as error:
                self.logger.warning("Failed to import long-term memory item", extra={"error": str(error)})

        if inserted:
            self.metrics.total_stored_items += inserted
            await asyncio.to_thread(self._apply_cleanup_policy_sync)

        return inserted

    async def initialize(self) -> bool:
        """외부 초기화 헬퍼"""
        self._initialized = True
        return True

    async def shutdown(self) -> None:
        if hasattr(self, 'conn'):
            await asyncio.to_thread(self.conn.commit)
            self.close()

    async def _store_to_db(self, item: MemoryItem) -> None:
        """데이터베이스에 항목 저장"""
        self.conn.execute('''
            INSERT OR REPLACE INTO memory_items
            (id, content, memory_type, created_at, accessed_at, access_count, strength, context, associations, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item.id,
            str(item.content),
            item.memory_type.value,
            item.created_at,
            item.accessed_at,
            item.access_count,
            item.strength,
            json.dumps(item.context),
            json.dumps(item.associations),
            json.dumps(item.metadata)
        ))

        self.conn.commit()

    def _insert_if_absent_sync(self, item: MemoryItem) -> None:
        cursor = self.conn.execute(
            'SELECT 1 FROM memory_items WHERE id = ?',
            (item.id,)
        )
        if cursor.fetchone():
            return

        self.conn.execute('''
            INSERT INTO memory_items
            (id, content, memory_type, created_at, accessed_at, access_count, strength, context, associations, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item.id,
            str(item.content),
            item.memory_type.value,
            item.created_at,
            item.accessed_at,
            item.access_count,
            item.strength,
            json.dumps(item.context),
            json.dumps(item.associations),
            json.dumps(item.metadata)
        ))
        self.conn.commit()

    async def _update_access_info(self, item: MemoryItem) -> None:
        """접근 정보 업데이트"""
        self.conn.execute('''
            UPDATE memory_items
            SET accessed_at = ?, access_count = ?
            WHERE id = ?
        ''', (item.accessed_at, item.access_count, item.id))

        self.conn.commit()

    def _row_to_memory_item(self, row: Tuple) -> MemoryItem:
        """데이터베이스 행을 MemoryItem으로 변환"""
        import json

        return MemoryItem(
            id=row[0],
            content=row[1],
            memory_type=MemoryType(row[2]),
            created_at=row[3],
            accessed_at=row[4],
            access_count=row[5],
            strength=row[6],
            context=json.loads(row[7]) if row[7] else {},
            associations=json.loads(row[8]) if row[8] else [],
            metadata=json.loads(row[9]) if row[9] else {}
        )

    def _memory_item_to_dict(self, item: MemoryItem) -> Dict[str, Any]:
        return {
            'id': item.id,
            'content': item.content,
            'memory_type': item.memory_type.value,
            'created_at': item.created_at,
            'accessed_at': item.accessed_at,
            'access_count': item.access_count,
            'strength': item.strength,
            'context': item.context,
            'associations': item.associations,
            'metadata': item.metadata,
        }

    def _calculate_text_relevance(self, content: str, query: str) -> float:
        """텍스트 관련성 계산"""
        content_lower = content.lower()
        query_lower = query.lower()

        if query_lower in content_lower:
            return 1.0

        # 키워드 기반 부분 일치
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        intersection = query_words.intersection(content_words)

        return len(intersection) / len(query_words) if query_words else 0.0

    async def _consolidate_by_similarity(self, items: List[MemoryItem]) -> MemoryItem:
        """유사도 기반 통합"""
        # 가장 강한 항목을 기준으로 통합
        base_item = max(items, key=lambda x: x.strength)

        # 내용 통합
        consolidated_content = {
            'primary': base_item.content,
            'related': [item.content for item in items if item.id != base_item.id]
        }

        # 강도 계산 (평균)
        avg_strength = sum(item.strength for item in items) / len(items)

        # 접근 횟수 합산
        total_access = sum(item.access_count for item in items)

        return MemoryItem(
            id=create_id(),
            content=consolidated_content,
            memory_type=base_item.memory_type,
            created_at=current_timestamp(),
            accessed_at=current_timestamp(),
            access_count=total_access,
            strength=avg_strength,
            context=base_item.context,
            associations=list(set(sum([item.associations for item in items], []))),
            metadata={'consolidation_type': 'similarity', 'source_count': len(items)}
        )

    async def _consolidate_by_frequency(self, items: List[MemoryItem]) -> MemoryItem:
        """빈도 기반 통합"""
        # 가장 많이 접근된 항목을 기준으로 통합
        base_item = max(items, key=lambda x: x.access_count)
        return await self._consolidate_by_similarity(items)  # 유사한 로직 재사용

    async def _consolidate_by_importance(self, items: List[MemoryItem]) -> MemoryItem:
        """중요도 기반 통합"""
        # 가장 강한 항목을 기준으로 통합
        base_item = max(items, key=lambda x: x.strength)
        return await self._consolidate_by_similarity(items)  # 유사한 로직 재사용

    def _update_retrieval_time(self, time_taken: float) -> None:
        """검색 시간 업데이트"""
        if self.metrics.average_retrieval_time == 0:
            self.metrics.average_retrieval_time = time_taken
        else:
            alpha = 0.1
            self.metrics.average_retrieval_time = (
                alpha * time_taken +
                (1 - alpha) * self.metrics.average_retrieval_time
            )

    def get_metrics(self) -> MemoryMetrics:
        """성능 지표 반환"""
        return self.metrics

    def close(self) -> None:
        """데이터베이스 연결 종료"""
        if hasattr(self, 'conn'):
            self.conn.close()

    async def get_item_count(self) -> int:
        return await asyncio.to_thread(self._count_items_sync)

    def _count_items_sync(self) -> int:
        cursor = self.conn.execute('SELECT COUNT(*) FROM memory_items')
        row = cursor.fetchone()
        return row[0] if row else 0

    def _apply_cleanup_policy_sync(self) -> int:
        if not self.settings:
            return 0

        removed = 0
        conn = self.conn
        if conn is None:
            return 0

        try:
            if self.settings.max_items is not None:
                cursor = conn.execute('SELECT COUNT(*) FROM memory_items')
                total = cursor.fetchone()[0]
                if total > self.settings.max_items:
                    overflow = total - self.settings.max_items
                    cursor = conn.execute('''
                        SELECT id FROM memory_items
                        ORDER BY strength ASC, access_count ASC, accessed_at ASC
                        LIMIT ?
                    ''', (overflow,))
                    ids = [row[0] for row in cursor.fetchall()]
                    if ids:
                        conn.executemany('DELETE FROM memory_items WHERE id = ?', ((item_id,) for item_id in ids))
                        removed += len(ids)

            if self.settings.min_strength_threshold is not None:
                cursor = conn.execute(
                    'DELETE FROM memory_items WHERE strength < ?',
                    (self.settings.min_strength_threshold,)
                )
                removed += cursor.rowcount or 0

            if self.settings.max_idle_seconds:
                cutoff = current_timestamp() - self.settings.max_idle_seconds
                cursor = conn.execute(
                    'DELETE FROM memory_items WHERE accessed_at < ?',
                    (cutoff,)
                )
                removed += cursor.rowcount or 0

            if removed:
                conn.commit()
            return removed

        except Exception as error:
            self.logger.warning(
                "Failed to apply long-term cleanup policy",
                extra={"error": str(error)}
            )
            return 0

    def _load_settings(self) -> LongTermMemorySettings:
        if DEFAULT_CONFIG_PATH.exists():
            try:
                with DEFAULT_CONFIG_PATH.open(encoding='utf-8') as handle:
                    data = json.load(handle)
                config_data = data.get('long_term_memory', {})
                if config_data:
                    return LongTermMemorySettings.from_dict(config_data)
            except Exception as error:
                self.logger.warning(
                    "Failed to load long-term memory configuration, using defaults",
                    extra={"error": str(error)}
                )
        return LongTermMemorySettings()
