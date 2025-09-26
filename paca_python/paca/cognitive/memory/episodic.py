"""
Episodic Memory Implementation
일화 메모리 구현 - 경험과 사건을 시간순으로 저장
"""

import asyncio
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

try:  # optional dependency
    import aiofiles  # type: ignore
except ImportError:  # pragma: no cover - fallback handled at runtime
    aiofiles = None  # type: ignore[assignment]

from .types import (
    MemoryType,
    MemoryOperation,
    MemoryConfiguration,
    MemoryMetrics,
    MemoryItem,
    SearchQuery,
    SearchResult,
    EpisodicMemorySettings,
)
from ...core.types import ID, Timestamp, KeyValuePair, create_id, current_timestamp
from ...core.errors import MemoryError
from ...core.utils.portable_storage import get_storage_manager


CONFIG_FILENAME = "memory_settings.json"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "config" / CONFIG_FILENAME


@dataclass
class EpisodicContext:
    """일화적 맥락"""
    temporal_context: Dict[str, Any]  # 시간적 맥락
    spatial_context: Dict[str, Any]   # 공간적 맥락
    emotional_context: Dict[str, Any] # 감정적 맥락
    social_context: Dict[str, Any]    # 사회적 맥락


class EpisodicMemory:
    """일화 메모리 시스템"""

    def __init__(
        self,
        config: Optional[MemoryConfiguration] = None,
        settings: Optional[EpisodicMemorySettings] = None,
        storage_manager: Optional[Any] = None,
    ):
        self.config = config or MemoryConfiguration()
        self.settings = settings or self._load_settings()
        self.episodes: List[MemoryItem] = []
        self.temporal_index: Dict[str, List[ID]] = {}  # 시간 기반 인덱스
        self.context_index: Dict[str, List[ID]] = {}   # 맥락 기반 인덱스
        self.metrics = MemoryMetrics()
        self.retention_seconds: Optional[float] = (
            self.settings.retention_days * 24 * 3600
            if self.settings.retention_days is not None
            else None
        )
        self.snapshot_interval_seconds: int = max(
            0,
            int(self.settings.snapshot_interval_seconds or 0)
        )
        self._last_snapshot_at: float = 0.0
        self._load_task: Optional[asyncio.Task] = None
        self._initialized = False

        # 포터블 저장소 설정
        self.storage_manager = storage_manager or get_storage_manager()
        self.memory_file_path = self.storage_manager.get_memory_file_path(
            "episodic", "episodic_memory.json"
        )
        self.snapshot_file_path = self.storage_manager.get_memory_file_path(
            "episodic", "episodic_context_snapshot.json"
        )

        self.logger = logging.getLogger("EpisodicMemory")
        if self.settings.enable_async_io and aiofiles is None:
            self.logger.warning(
                "aiofiles not available; falling back to synchronous file IO for episodic memory"
            )
            self.settings.enable_async_io = False

        # 초기 데이터 로드는 initialize 단계에서 처리
        self._schedule_initial_load()

    async def store_episode(
        self,
        content: Any,
        episodic_context: EpisodicContext = None,
        importance: float = 1.0
    ) -> ID:
        """일화 저장"""
        start_time = time.time()

        try:
            await self._ensure_initialized()

            # 기본 맥락 설정
            if episodic_context is None:
                episodic_context = EpisodicContext(
                    temporal_context={'timestamp': current_timestamp()},
                    spatial_context={},
                    emotional_context={},
                    social_context={}
                )

            # 일화 항목 생성
            episode = MemoryItem(
                id=create_id(),
                content=content,
                memory_type=MemoryType.EPISODIC,
                created_at=current_timestamp(),
                accessed_at=current_timestamp(),
                strength=importance,
                context={
                    'temporal': episodic_context.temporal_context,
                    'spatial': episodic_context.spatial_context,
                    'emotional': episodic_context.emotional_context,
                    'social': episodic_context.social_context
                }
            )

            self.episodes.append(episode)

            # 인덱스 업데이트
            await self._update_temporal_index(episode)
            await self._update_context_index(episode)

            # 메트릭 업데이트
            self.metrics.total_stored_items += 1
            self._update_retrieval_time(time.time() - start_time)

            # 보존 기간 적용
            await self.enforce_retention(persist=False)

            # 포터블 저장소에 저장
            await self._save_to_storage()
            await self._maybe_snapshot()

            self.logger.debug(f"Stored episode {episode.id}")
            return episode.id

        except Exception as error:
            raise MemoryError(
                f"일화 메모리 저장 실패: {str(error)}",
                memory_type="episodic",
                operation="store"
            )

    async def store_simple_episode(self, episode_id: str, episode_data: Dict[str, Any]) -> ID:
        """간단한 일화 저장 (테스트용)"""
        return await self.store_episode(episode_data)

    async def retrieve_by_time(
        self,
        start_time: Optional[Timestamp] = None,
        end_time: Optional[Timestamp] = None,
        limit: int = 10
    ) -> List[MemoryItem]:
        """시간 범위로 일화 검색"""
        try:
            await self._ensure_initialized()
            results = []

            for episode in self.episodes:
                # 시간 범위 확인
                if start_time and episode.created_at < start_time:
                    continue
                if end_time and episode.created_at > end_time:
                    continue

                results.append(episode)
                episode.accessed_at = current_timestamp()
                episode.access_count += 1

            # 시간순 정렬 (최신 순)
            results.sort(key=lambda x: x.created_at, reverse=True)
            return results[:limit]

        except Exception as error:
            raise MemoryError(
                f"시간 기반 일화 검색 실패: {str(error)}",
                memory_type="episodic",
                operation="retrieve"
            )

    async def retrieve_by_context(
        self,
        context_query: Dict[str, Any],
        similarity_threshold: float = 0.7
    ) -> List[MemoryItem]:
        """맥락으로 일화 검색"""
        try:
            await self._ensure_initialized()
            results = []

            for episode in self.episodes:
                similarity = self._calculate_context_similarity(
                    episode.context,
                    context_query
                )

                if similarity >= similarity_threshold:
                    results.append((episode, similarity))
                    episode.accessed_at = current_timestamp()
                    episode.access_count += 1

            # 유사도순 정렬
            results.sort(key=lambda x: x[1], reverse=True)
            return [episode for episode, _ in results]

        except Exception as error:
            raise MemoryError(
                f"맥락 기반 일화 검색 실패: {str(error)}",
                memory_type="episodic",
                operation="retrieve"
            )

    async def search(self, query: SearchQuery) -> SearchResult:
        """일화 검색"""
        start_time = time.time()

        try:
            await self._ensure_initialized()
            results = []
            relevance_scores = {}

            for episode in self.episodes:
                # 내용 관련성 계산
                content_score = self._calculate_content_relevance(episode, query.query)

                # 맥락 관련성 계산
                context_score = 0.0
                if isinstance(query.query, dict) and 'context' in query.query:
                    context_score = self._calculate_context_similarity(
                        episode.context,
                        query.query['context']
                    )

                # 종합 점수
                total_score = (content_score + context_score) / 2

                if total_score >= query.threshold:
                    results.append(episode)
                    relevance_scores[episode.id] = total_score

                    episode.accessed_at = current_timestamp()
                    episode.access_count += 1

            # 정렬
            if query.sort_by == 'relevance':
                results.sort(key=lambda x: relevance_scores[x.id], reverse=True)
            elif query.sort_by == 'recency':
                results.sort(key=lambda x: x.created_at, reverse=True)

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
                f"일화 메모리 검색 실패: {str(error)}",
                memory_type="episodic",
                operation="search"
            )

    async def consolidate_similar_episodes(
        self,
        similarity_threshold: float = 0.8
    ) -> int:
        """유사한 일화들 통합"""
        try:
            await self._ensure_initialized()
            consolidated_count = 0
            episodes_to_remove = set()

            for i, episode1 in enumerate(self.episodes):
                if episode1.id in episodes_to_remove:
                    continue

                similar_episodes = []
                for j, episode2 in enumerate(self.episodes[i+1:], i+1):
                    if episode2.id in episodes_to_remove:
                        continue

                    similarity = self._calculate_episode_similarity(episode1, episode2)
                    if similarity >= similarity_threshold:
                        similar_episodes.append(episode2)

                if similar_episodes:
                    # 통합된 일화 생성
                    await self._merge_episodes(episode1, similar_episodes)

                    # 통합된 일화들을 제거 목록에 추가
                    for ep in similar_episodes:
                        episodes_to_remove.add(ep.id)

                    consolidated_count += len(similar_episodes)

            # 제거할 일화들 삭제
            self.episodes = [ep for ep in self.episodes if ep.id not in episodes_to_remove]

            # 포터블 저장소에 저장
            await self._save_to_storage()

            self.logger.debug(f"Consolidated {consolidated_count} episodes")
            return consolidated_count

        except Exception as error:
            raise MemoryError(
                f"일화 통합 실패: {str(error)}",
                memory_type="episodic",
                operation="consolidate"
            )

    async def enforce_retention(self, *, persist: bool = True) -> int:
        """보존 기간을 초과한 일화 정리"""
        if not self.retention_seconds:
            return 0

        cutoff = current_timestamp() - self.retention_seconds
        remaining: List[MemoryItem] = []
        removed = 0

        for episode in self.episodes:
            if episode.created_at < cutoff:
                removed += 1
            else:
                remaining.append(episode)

        if not removed:
            return 0

        self.episodes = remaining
        await self._rebuild_indexes()

        if persist:
            await self._save_to_storage()

        self.logger.debug("Episodic retention removed %d episodes", removed)
        return removed

    async def export_batch(
        self,
        *,
        limit: Optional[int] = None,
        since_timestamp: Optional[Timestamp] = None,
        path: Optional[Union[str, Path]] = None,
        include_context: bool = True,
    ) -> Path:
        """일화 메모리를 JSON 파일로 내보내기."""
        await self._ensure_initialized()
        await self.enforce_retention(persist=False)

        episodes_sorted = sorted(self.episodes, key=lambda ep: ep.created_at, reverse=True)
        export_items: List[Dict[str, Any]] = []
        for episode in episodes_sorted:
            if since_timestamp and episode.created_at < since_timestamp:
                continue

            payload: Dict[str, Any] = {
                'id': episode.id,
                'content': episode.content,
                'memory_type': episode.memory_type.value,
                'created_at': episode.created_at,
                'accessed_at': episode.accessed_at,
                'access_count': episode.access_count,
                'strength': episode.strength,
                'metadata': episode.metadata,
            }
            if include_context:
                payload['context'] = episode.context
            export_items.append(payload)

            if limit and len(export_items) >= limit:
                break

        export_payload = {
            'exported_at': datetime.now().isoformat(),
            'count': len(export_items),
            'episodes': export_items,
        }

        target_path = Path(path) if path else self.storage_manager.get_memory_file_path(
            "episodic", "episodic_export.json"
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)

        serialized = json.dumps(export_payload, ensure_ascii=False, indent=2, default=str)
        if self.settings.enable_async_io and aiofiles is not None:
            async with aiofiles.open(target_path, 'w', encoding='utf-8') as handle:
                await handle.write(serialized)
        else:
            await asyncio.to_thread(target_path.write_text, serialized, 'utf-8')

        return target_path

    async def import_batch(
        self,
        source: Optional[Union[str, Path, Dict[str, Any], List[Dict[str, Any]]]] = None,
        *,
        merge_strategy: str = "append",
    ) -> int:
        """JSON 파일이나 객체로부터 일화 데이터를 불러오기."""
        await self._ensure_initialized()

        if source is None:
            source = self.storage_manager.get_memory_file_path("episodic", "episodic_export.json")

        if isinstance(source, (str, Path)):
            load_path = Path(source)
            if self.settings.enable_async_io and aiofiles is not None:
                async with aiofiles.open(load_path, 'r', encoding='utf-8') as handle:
                    raw = await handle.read()
            else:
                raw = await asyncio.to_thread(load_path.read_text, 'utf-8')
            data_obj = json.loads(raw) if raw else {}
        else:
            data_obj = source or {}

        if isinstance(data_obj, dict) and 'episodes' in data_obj:
            records = data_obj.get('episodes', [])
        else:
            records = data_obj if isinstance(data_obj, list) else []

        if merge_strategy == "replace":
            self.episodes = []
            self.metrics = MemoryMetrics()
            self.temporal_index = {}
            self.context_index = {}

        existing_ids = {episode.id for episode in self.episodes}
        inserted = 0

        for record in records:
            episode_id = record.get('id')
            if not episode_id or episode_id in existing_ids:
                continue

            episode = MemoryItem(
                id=episode_id,
                content=record.get('content'),
                memory_type=MemoryType(record.get('memory_type', MemoryType.EPISODIC.value)),
                created_at=record.get('created_at', current_timestamp()),
                accessed_at=record.get('accessed_at', current_timestamp()),
                access_count=record.get('access_count', 0),
                strength=record.get('strength', 1.0),
                context=record.get('context', {}),
                metadata=record.get('metadata', {}),
            )

            self.episodes.append(episode)
            existing_ids.add(episode.id)
            inserted += 1

        if inserted:
            await self._rebuild_indexes()
            self.metrics.total_stored_items += inserted
            await self.enforce_retention(persist=False)
            await self._save_to_storage()
            await self._maybe_snapshot(force=True)
        elif merge_strategy == "replace":
            await self._save_to_storage()

        return inserted

    async def _update_temporal_index(self, episode: MemoryItem) -> None:
        """시간 인덱스 업데이트"""
        time_key = f"{int(episode.created_at // 3600)}"  # 시간당 그룹핑
        if time_key not in self.temporal_index:
            self.temporal_index[time_key] = []
        self.temporal_index[time_key].append(episode.id)

    async def _update_context_index(self, episode: MemoryItem) -> None:
        """맥락 인덱스 업데이트"""
        for context_type, context_data in episode.context.items():
            if isinstance(context_data, dict):
                for key, value in context_data.items():
                    index_key = f"{context_type}:{key}:{value}"
                    if index_key not in self.context_index:
                        self.context_index[index_key] = []
                    self.context_index[index_key].append(episode.id)

    def _calculate_context_similarity(
        self,
        context1: Dict[str, Any],
        context2: Dict[str, Any]
    ) -> float:
        """맥락 유사도 계산"""
        if not context1 or not context2:
            return 0.0

        total_weight = 0.0
        similarity_sum = 0.0

        for context_type in ['temporal', 'spatial', 'emotional', 'social']:
            if context_type in context1 and context_type in context2:
                weight = 1.0
                similarity = self._calculate_dict_similarity(
                    context1[context_type],
                    context2[context_type]
                )
                similarity_sum += weight * similarity
                total_weight += weight

        return similarity_sum / total_weight if total_weight > 0 else 0.0

    def _calculate_dict_similarity(self, dict1: Dict, dict2: Dict) -> float:
        """딕셔너리 유사도 계산"""
        if not dict1 or not dict2:
            return 0.0

        common_keys = set(dict1.keys()).intersection(set(dict2.keys()))
        if not common_keys:
            return 0.0

        matches = sum(1 for key in common_keys if dict1[key] == dict2[key])
        return matches / len(common_keys)

    def _calculate_content_relevance(self, episode: MemoryItem, query: Any) -> float:
        """내용 관련성 계산"""
        if isinstance(query, str):
            content_str = str(episode.content).lower()
            query_str = query.lower()

            if query_str in content_str:
                return 1.0

            # 키워드 기반 부분 일치
            query_words = set(query_str.split())
            content_words = set(content_str.split())
            intersection = query_words.intersection(content_words)

            return len(intersection) / len(query_words) if query_words else 0.0

        return 0.0

    def _calculate_episode_similarity(self, episode1: MemoryItem, episode2: MemoryItem) -> float:
        """일화 유사도 계산"""
        # 내용 유사도
        content_similarity = self._calculate_content_relevance(episode1, str(episode2.content))

        # 맥락 유사도
        context_similarity = self._calculate_context_similarity(
            episode1.context,
            episode2.context
        )

        # 시간 유사도 (시간이 가까울수록 높은 점수)
        time_diff = abs(episode1.created_at - episode2.created_at)
        time_similarity = 1.0 / (1.0 + time_diff / 3600)  # 시간 단위

        # 종합 유사도
        return (content_similarity * 0.5 + context_similarity * 0.3 + time_similarity * 0.2)

    async def _merge_episodes(self, primary: MemoryItem, similar: List[MemoryItem]) -> None:
        """일화들 병합"""
        # 접근 횟수 합산
        primary.access_count += sum(ep.access_count for ep in similar)

        # 강도 가중 평균
        total_strength = primary.strength + sum(ep.strength for ep in similar)
        primary.strength = total_strength / (len(similar) + 1)

        # 연관 관계 병합
        for episode in similar:
            primary.associations.extend(episode.associations)
        primary.associations = list(set(primary.associations))  # 중복 제거

        # 변경사항 저장
        await self._save_to_storage()

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

    def get_episode_count(self) -> int:
        """저장된 일화 수 반환"""
        return len(self.episodes)

    async def initialize(self) -> bool:
        """외부 초기화 헬퍼"""
        await self._ensure_initialized()
        return True

    async def shutdown(self) -> None:
        """스냅샷을 저장하고 상태 플래그 초기화"""
        if not self._initialized:
            return
        await self._save_to_storage()
        if self.snapshot_interval_seconds:
            await self._maybe_snapshot(force=True)
        self._initialized = False

    async def _save_to_storage(self) -> None:
        """메모리 데이터를 포터블 저장소에 저장"""
        try:
            # MemoryItem을 직렬화 가능한 형태로 변환
            serializable_episodes = []
            for episode in self.episodes:
                episode_dict = {
                    'id': episode.id,
                    'content': episode.content,
                    'memory_type': episode.memory_type.value,
                    'created_at': episode.created_at,
                    'accessed_at': episode.accessed_at,
                    'access_count': episode.access_count,
                    'strength': episode.strength,
                    'context': episode.context,
                    'associations': episode.associations,
                    'metadata': episode.metadata
                }
                serializable_episodes.append(episode_dict)

            # 인덱스와 메트릭 정보도 포함
            data = {
                'episodes': serializable_episodes,
                'temporal_index': self.temporal_index,
                'context_index': self.context_index,
                'metrics': {
                    'total_stored_items': self.metrics.total_stored_items,
                    'average_retrieval_time': self.metrics.average_retrieval_time,
                    'hit_rate': self.metrics.hit_rate,
                    'memory_efficiency': self.metrics.memory_efficiency,
                    'consolidation_rate': self.metrics.consolidation_rate
                },
                'saved_at': datetime.now().isoformat()
            }

            serialized = json.dumps(data, ensure_ascii=False, indent=2, default=str)
            self.memory_file_path.parent.mkdir(parents=True, exist_ok=True)

            if self.settings.enable_async_io and aiofiles is not None:
                async with aiofiles.open(self.memory_file_path, 'w', encoding='utf-8') as handle:
                    await handle.write(serialized)
            else:
                await asyncio.to_thread(
                    self.storage_manager.save_json_data,
                    self.memory_file_path,
                    data,
                )

        except Exception as e:
            self.logger.error(f"Failed to save episodic memory to storage: {e}")

    async def _load_from_storage(self) -> None:
        """포터블 저장소에서 메모리 데이터 로드"""
        try:
            if not self.memory_file_path.exists():
                return

            if self.settings.enable_async_io and aiofiles is not None:
                async with aiofiles.open(self.memory_file_path, 'r', encoding='utf-8') as handle:
                    raw = await handle.read()
                data = json.loads(raw) if raw else None
            else:
                data = await asyncio.to_thread(
                    self.storage_manager.load_json_data,
                    self.memory_file_path,
                )

            if not data:
                return

            # 일화 복원
            self.episodes = []
            for episode_dict in data.get('episodes', []):
                episode = MemoryItem(
                    id=episode_dict['id'],
                    content=episode_dict['content'],
                    memory_type=MemoryType(episode_dict['memory_type']),
                    created_at=episode_dict['created_at'],
                    accessed_at=episode_dict['accessed_at'],
                    access_count=episode_dict['access_count'],
                    strength=episode_dict['strength'],
                    context=episode_dict['context'],
                    associations=episode_dict['associations'],
                    metadata=episode_dict['metadata']
                )
                self.episodes.append(episode)

            # 인덱스 복원
            self.temporal_index = data.get('temporal_index', {})
            self.context_index = data.get('context_index', {})

            # 메트릭 복원
            if 'metrics' in data:
                metrics_data = data['metrics']
                self.metrics.total_stored_items = metrics_data.get('total_stored_items', 0)
                self.metrics.average_retrieval_time = metrics_data.get('average_retrieval_time', 0.0)
                self.metrics.hit_rate = metrics_data.get('hit_rate', 0.0)
                self.metrics.memory_efficiency = metrics_data.get('memory_efficiency', 0.0)
                self.metrics.consolidation_rate = metrics_data.get('consolidation_rate', 0.0)

            self.logger.debug(f"Loaded {len(self.episodes)} episodes from episodic memory storage")
            await self.enforce_retention(persist=False)

        except Exception as e:
            self.logger.error(f"Failed to load episodic memory from storage: {e}")
            # 로드 실패해도 빈 상태로 시작

    async def _maybe_snapshot(self, force: bool = False) -> None:
        if not self.snapshot_interval_seconds and not force:
            return

        now = time.time()
        if not force and (now - self._last_snapshot_at) < self.snapshot_interval_seconds:
            return

        snapshot_limit = self.settings.max_snapshot_items or len(self.episodes)
        sample_items = [
            {
                'id': episode.id,
                'created_at': episode.created_at,
                'strength': episode.strength,
                'context': episode.context,
            }
            for episode in self.episodes[-snapshot_limit:]
        ]

        snapshot_data = {
            'generated_at': datetime.now().isoformat(),
            'episode_count': len(self.episodes),
            'temporal_index_keys': list(self.temporal_index.keys()),
            'context_index_keys': list(self.context_index.keys()),
            'sample_items': sample_items,
        }

        serialized = json.dumps(snapshot_data, ensure_ascii=False, indent=2, default=str)
        self.snapshot_file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if self.settings.enable_async_io and aiofiles is not None:
                async with aiofiles.open(self.snapshot_file_path, 'w', encoding='utf-8') as handle:
                    await handle.write(serialized)
            else:
                await asyncio.to_thread(
                    self.storage_manager.save_json_data,
                    self.snapshot_file_path,
                    snapshot_data,
                )
            self._last_snapshot_at = now
        except Exception as error:
            self.logger.warning("Failed to write episodic snapshot", extra={"error": str(error)})

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

        self._initialized = True

    def _load_settings(self) -> EpisodicMemorySettings:
        if DEFAULT_CONFIG_PATH.exists():
            try:
                with DEFAULT_CONFIG_PATH.open(encoding='utf-8') as handle:
                    data = json.load(handle)
                episodic_config = data.get('episodic_memory', {})
                if episodic_config:
                    return EpisodicMemorySettings.from_dict(episodic_config)
            except Exception as error:
                self.logger.warning(
                    "Failed to load episodic memory configuration, using defaults",
                    extra={"error": str(error)}
                )
        return EpisodicMemorySettings()

    async def _rebuild_indexes(self) -> None:
        self.temporal_index.clear()
        self.context_index.clear()
        for episode in self.episodes:
            await self._update_temporal_index(episode)
            await self._update_context_index(episode)
