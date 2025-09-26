"""
메모리 통합기 (Memory Consolidator)

장기 기억 공고화와 메모리 정리를 담당합니다.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
import hashlib
import json


class MemoryType(Enum):
    """메모리 유형"""
    EPISODIC = "episodic"        # 에피소드 기억
    SEMANTIC = "semantic"        # 의미 기억
    PROCEDURAL = "procedural"    # 절차 기억
    WORKING = "working"          # 작업 기억
    EMOTIONAL = "emotional"      # 감정 기억


class ConsolidationStrategy(Enum):
    """통합 전략"""
    FREQUENCY_BASED = "frequency"      # 빈도 기반
    RECENCY_BASED = "recency"          # 최신성 기반
    IMPORTANCE_BASED = "importance"    # 중요도 기반
    SIMILARITY_BASED = "similarity"    # 유사성 기반
    MIXED = "mixed"                    # 혼합 전략


@dataclass
class MemoryItem:
    """메모리 항목"""
    id: str
    content: Any
    memory_type: MemoryType
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    importance_score: float = 0.5
    emotional_weight: float = 0.0
    tags: List[str] = field(default_factory=list)
    connections: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'content': self.content,
            'memory_type': self.memory_type.value,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'importance_score': self.importance_score,
            'emotional_weight': self.emotional_weight,
            'tags': self.tags,
            'connections': self.connections,
            'metadata': self.metadata
        }

    def calculate_retention_score(self) -> float:
        """보존 점수 계산"""
        # 시간 감쇠 요소
        days_since_creation = (datetime.now() - self.created_at).days
        days_since_access = (datetime.now() - self.last_accessed).days

        time_decay = max(0.1, 1.0 - (days_since_access * 0.1))
        age_factor = max(0.1, 1.0 - (days_since_creation * 0.05))

        # 접근 빈도 요소
        frequency_factor = min(1.0, self.access_count * 0.1)

        # 종합 점수
        retention_score = (
            self.importance_score * 0.4 +
            frequency_factor * 0.3 +
            time_decay * 0.2 +
            age_factor * 0.1
        )

        return min(1.0, max(0.0, retention_score))


@dataclass
class ConsolidationResult:
    """통합 결과"""
    processed_count: int
    consolidated_count: int
    archived_count: int
    deleted_count: int
    processing_time: float
    strategy_used: ConsolidationStrategy
    quality_metrics: Dict[str, float]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'processed_count': self.processed_count,
            'consolidated_count': self.consolidated_count,
            'archived_count': self.archived_count,
            'deleted_count': self.deleted_count,
            'processing_time': self.processing_time,
            'strategy_used': self.strategy_used.value,
            'quality_metrics': self.quality_metrics,
            'recommendations': self.recommendations
        }


class MemoryConsolidator:
    """메모리 통합기 클래스"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # 기본 설정
        self.retention_threshold = 0.3
        self.consolidation_similarity_threshold = 0.8
        self.archive_threshold = 0.1
        self.max_consolidation_group_size = 10

        # 메모리 저장소 (시뮬레이션)
        self.memory_store: Dict[str, MemoryItem] = {}
        self.consolidated_memories: Dict[str, List[str]] = {}
        self.archived_memories: Dict[str, MemoryItem] = {}

        # 통계
        self.consolidation_stats = {
            'total_consolidations': 0,
            'successful_consolidations': 0,
            'total_processing_time': 0.0,
            'memory_efficiency_improvement': 0.0
        }

    async def consolidate_memories(self, memory_count: int, batch_size: int = 100,
                                 parallel: bool = True) -> Dict[str, Any]:
        """메모리 통합 실행"""
        start_time = datetime.now()
        self.logger.info(f"메모리 통합 시작: {memory_count}개 메모리 처리")

        try:
            # 시뮬레이션된 메모리 생성
            memories = self._generate_simulation_memories(memory_count)

            # 전략 선택
            strategy = self._select_consolidation_strategy(memories)

            # 배치 처리
            results = []
            for i in range(0, len(memories), batch_size):
                batch = memories[i:i + batch_size]

                if parallel:
                    batch_result = await self._process_batch_parallel(batch, strategy)
                else:
                    batch_result = await self._process_batch_sequential(batch, strategy)

                results.append(batch_result)

            # 결과 집계
            final_result = self._aggregate_results(results, strategy)
            final_result.processing_time = (datetime.now() - start_time).total_seconds()

            # 통계 업데이트
            self._update_consolidation_stats(final_result)

            self.logger.info(f"메모리 통합 완료: {final_result.processed_count}개 처리됨")
            return final_result.to_dict()

        except Exception as e:
            self.logger.error(f"메모리 통합 오류: {str(e)}")
            raise RuntimeError(f"메모리 통합 실패: {str(e)}")

    def _generate_simulation_memories(self, count: int) -> List[MemoryItem]:
        """시뮬레이션 메모리 생성"""
        memories = []

        for i in range(count):
            memory_types = list(MemoryType)
            memory_type = memory_types[i % len(memory_types)]

            created_time = datetime.now() - timedelta(days=i % 100, hours=i % 24)
            last_accessed = created_time + timedelta(days=(i % 50))

            memory = MemoryItem(
                id=f"mem_{i:06d}",
                content=f"메모리 내용 {i}: {memory_type.value} 정보",
                memory_type=memory_type,
                created_at=created_time,
                last_accessed=last_accessed,
                access_count=max(1, (count - i) // 100),
                importance_score=min(1.0, (count - i) / count + 0.1),
                emotional_weight=(i % 10) / 10.0,
                tags=[f"tag_{i % 10}", f"category_{i % 5}"],
                connections=[f"mem_{(i-1):06d}", f"mem_{(i+1):06d}"] if i > 0 else []
            )

            memories.append(memory)

        return memories

    def _select_consolidation_strategy(self, memories: List[MemoryItem]) -> ConsolidationStrategy:
        """통합 전략 선택"""
        # 메모리 특성 분석
        total_memories = len(memories)
        avg_age = sum((datetime.now() - m.created_at).days for m in memories) / total_memories
        avg_access_count = sum(m.access_count for m in memories) / total_memories

        # 전략 결정
        if avg_access_count > 10:
            return ConsolidationStrategy.FREQUENCY_BASED
        elif avg_age < 7:  # 7일 미만
            return ConsolidationStrategy.RECENCY_BASED
        elif sum(m.importance_score > 0.7 for m in memories) / total_memories > 0.3:
            return ConsolidationStrategy.IMPORTANCE_BASED
        else:
            return ConsolidationStrategy.MIXED

    async def _process_batch_parallel(self, batch: List[MemoryItem],
                                    strategy: ConsolidationStrategy) -> ConsolidationResult:
        """병렬 배치 처리"""
        tasks = []

        for memory in batch:
            task = asyncio.create_task(self._process_single_memory(memory, strategy))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 집계
        processed = 0
        consolidated = 0
        archived = 0
        deleted = 0

        for result in results:
            if isinstance(result, Exception):
                continue

            processed += 1
            if result['action'] == 'consolidated':
                consolidated += 1
            elif result['action'] == 'archived':
                archived += 1
            elif result['action'] == 'deleted':
                deleted += 1

        return ConsolidationResult(
            processed_count=processed,
            consolidated_count=consolidated,
            archived_count=archived,
            deleted_count=deleted,
            processing_time=0.0,  # 배치 레벨에서는 0
            strategy_used=strategy,
            quality_metrics={'batch_efficiency': processed / len(batch)},
            recommendations=[]
        )

    async def _process_batch_sequential(self, batch: List[MemoryItem],
                                      strategy: ConsolidationStrategy) -> ConsolidationResult:
        """순차 배치 처리"""
        processed = 0
        consolidated = 0
        archived = 0
        deleted = 0

        for memory in batch:
            try:
                result = await self._process_single_memory(memory, strategy)

                processed += 1
                if result['action'] == 'consolidated':
                    consolidated += 1
                elif result['action'] == 'archived':
                    archived += 1
                elif result['action'] == 'deleted':
                    deleted += 1

            except Exception as e:
                self.logger.warning(f"메모리 처리 실패: {memory.id} - {str(e)}")

        return ConsolidationResult(
            processed_count=processed,
            consolidated_count=consolidated,
            archived_count=archived,
            deleted_count=deleted,
            processing_time=0.0,
            strategy_used=strategy,
            quality_metrics={'batch_efficiency': processed / len(batch)},
            recommendations=[]
        )

    async def _process_single_memory(self, memory: MemoryItem,
                                   strategy: ConsolidationStrategy) -> Dict[str, Any]:
        """단일 메모리 처리"""
        # 보존 점수 계산
        retention_score = memory.calculate_retention_score()

        # 처리 방식 결정
        if retention_score < self.archive_threshold:
            action = 'deleted'
            await self._delete_memory(memory)
        elif retention_score < self.retention_threshold:
            action = 'archived'
            await self._archive_memory(memory)
        else:
            # 통합 가능한 메모리 찾기
            similar_memories = await self._find_similar_memories(memory, strategy)
            if similar_memories:
                action = 'consolidated'
                await self._consolidate_with_similar(memory, similar_memories)
            else:
                action = 'retained'
                await self._retain_memory(memory)

        return {
            'memory_id': memory.id,
            'action': action,
            'retention_score': retention_score
        }

    async def _find_similar_memories(self, memory: MemoryItem,
                                   strategy: ConsolidationStrategy) -> List[MemoryItem]:
        """유사한 메모리 찾기"""
        similar_memories = []

        # 전략에 따른 유사성 기준
        if strategy == ConsolidationStrategy.SIMILARITY_BASED:
            # 태그 기반 유사성
            for stored_memory in self.memory_store.values():
                if stored_memory.id == memory.id:
                    continue

                # 태그 일치율 계산
                common_tags = set(memory.tags) & set(stored_memory.tags)
                tag_similarity = len(common_tags) / max(len(memory.tags), len(stored_memory.tags), 1)

                if tag_similarity >= self.consolidation_similarity_threshold:
                    similar_memories.append(stored_memory)

        elif strategy == ConsolidationStrategy.FREQUENCY_BASED:
            # 접근 빈도가 비슷한 메모리
            for stored_memory in self.memory_store.values():
                if stored_memory.id == memory.id:
                    continue

                access_ratio = min(memory.access_count, stored_memory.access_count) / \
                             max(memory.access_count, stored_memory.access_count, 1)

                if access_ratio >= 0.7:  # 접근 빈도 70% 이상 유사
                    similar_memories.append(stored_memory)

        return similar_memories[:self.max_consolidation_group_size]

    async def _consolidate_with_similar(self, memory: MemoryItem, similar_memories: List[MemoryItem]):
        """유사한 메모리와 통합"""
        # 통합 그룹 ID 생성
        group_id = f"consolidated_{memory.id}"

        # 통합 대상 메모리 ID 목록
        consolidated_ids = [memory.id] + [m.id for m in similar_memories]

        # 통합 메모리 저장
        self.consolidated_memories[group_id] = consolidated_ids

        # 개별 메모리는 원본 저장소에서 제거하지 않고 참조만 유지
        # 실제 구현에서는 메모리 압축이나 요약 기법 사용

        await asyncio.sleep(0.01)  # 시뮬레이션 지연

    async def _archive_memory(self, memory: MemoryItem):
        """메모리 아카이브"""
        self.archived_memories[memory.id] = memory
        # 원본에서는 제거하지 않고 아카이브 플래그만 설정

        await asyncio.sleep(0.005)  # 시뮬레이션 지연

    async def _delete_memory(self, memory: MemoryItem):
        """메모리 삭제"""
        # 실제로는 완전 삭제하지 않고 삭제 플래그 설정
        memory.metadata['deleted'] = True
        memory.metadata['deleted_at'] = datetime.now().isoformat()

        await asyncio.sleep(0.002)  # 시뮬레이션 지연

    async def _retain_memory(self, memory: MemoryItem):
        """메모리 보존"""
        # 메모리를 활성 저장소에 유지
        self.memory_store[memory.id] = memory

        await asyncio.sleep(0.001)  # 시뮬레이션 지연

    def _aggregate_results(self, results: List[ConsolidationResult],
                          strategy: ConsolidationStrategy) -> ConsolidationResult:
        """결과 집계"""
        total_processed = sum(r.processed_count for r in results)
        total_consolidated = sum(r.consolidated_count for r in results)
        total_archived = sum(r.archived_count for r in results)
        total_deleted = sum(r.deleted_count for r in results)

        # 품질 메트릭 계산
        quality_metrics = {
            'consolidation_ratio': total_consolidated / max(total_processed, 1),
            'retention_ratio': (total_processed - total_deleted) / max(total_processed, 1),
            'efficiency_score': total_consolidated / max(total_processed - total_deleted, 1)
        }

        # 권장사항 생성
        recommendations = self._generate_recommendations(quality_metrics)

        return ConsolidationResult(
            processed_count=total_processed,
            consolidated_count=total_consolidated,
            archived_count=total_archived,
            deleted_count=total_deleted,
            processing_time=0.0,  # 상위에서 설정
            strategy_used=strategy,
            quality_metrics=quality_metrics,
            recommendations=recommendations
        )

    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """권장사항 생성"""
        recommendations = []

        if metrics['consolidation_ratio'] < 0.1:
            recommendations.append("통합률이 낮습니다. 유사성 임계값을 낮춰보세요.")

        if metrics['retention_ratio'] > 0.9:
            recommendations.append("보존율이 높습니다. 보존 임계값을 높여 메모리를 더 적극적으로 정리하세요.")

        if metrics['efficiency_score'] < 0.2:
            recommendations.append("효율성이 낮습니다. 통합 전략을 변경해보세요.")

        return recommendations

    def _update_consolidation_stats(self, result: ConsolidationResult):
        """통합 통계 업데이트"""
        self.consolidation_stats['total_consolidations'] += 1

        if result.processed_count > 0:
            self.consolidation_stats['successful_consolidations'] += 1

        self.consolidation_stats['total_processing_time'] += result.processing_time

        # 효율성 개선 계산
        efficiency_improvement = result.quality_metrics.get('efficiency_score', 0.0)
        self.consolidation_stats['memory_efficiency_improvement'] = (
            self.consolidation_stats['memory_efficiency_improvement'] * 0.9 +
            efficiency_improvement * 0.1
        )

    def get_consolidation_statistics(self) -> Dict[str, Any]:
        """통합 통계 조회"""
        return {
            'stats': self.consolidation_stats.copy(),
            'memory_store_size': len(self.memory_store),
            'consolidated_groups': len(self.consolidated_memories),
            'archived_count': len(self.archived_memories),
            'average_processing_time': (
                self.consolidation_stats['total_processing_time'] /
                max(self.consolidation_stats['total_consolidations'], 1)
            )
        }

    async def optimize_memory_layout(self) -> Dict[str, Any]:
        """메모리 레이아웃 최적화"""
        self.logger.info("메모리 레이아웃 최적화 시작")

        optimization_result = {
            'memory_defragmentation': await self._defragment_memory(),
            'connection_optimization': await self._optimize_connections(),
            'access_pattern_optimization': await self._optimize_access_patterns()
        }

        self.logger.info("메모리 레이아웃 최적화 완료")
        return optimization_result

    async def _defragment_memory(self) -> Dict[str, Any]:
        """메모리 조각 모음"""
        # 메모리 조각화 해결 시뮬레이션
        await asyncio.sleep(0.5)

        return {
            'fragmentation_before': 0.3,
            'fragmentation_after': 0.1,
            'space_saved': '20MB',
            'processing_time': 0.5
        }

    async def _optimize_connections(self) -> Dict[str, Any]:
        """연결 최적화"""
        # 메모리 간 연결 최적화 시뮬레이션
        await asyncio.sleep(0.3)

        return {
            'connections_optimized': 150,
            'redundant_connections_removed': 45,
            'connection_strength_improved': 0.15
        }

    async def _optimize_access_patterns(self) -> Dict[str, Any]:
        """접근 패턴 최적화"""
        # 접근 패턴 기반 메모리 재배치 시뮬레이션
        await asyncio.sleep(0.4)

        return {
            'memories_relocated': 80,
            'access_speed_improvement': 0.25,
            'cache_hit_rate_improvement': 0.12
        }