"""
Knowledge Management Service
사용자의 지식 항목과 관계를 관리하는 서비스
TypeScript → Python 완전 변환
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union
from enum import Enum

from ..core.types import (
    ID, Timestamp, Result, Priority, Status,
    create_success, create_failure, generate_id, current_timestamp
)
from ..core.errors import ValidationError, ApplicationError
from ..core.events import EventEmitter
from ..cognitive import CognitiveSystem, CognitiveContext, CognitiveTaskType
from .base import (
    BaseService, ServiceConfig, ServiceContext, ServiceResult, ServicePriority
)


class KnowledgeType(Enum):
    """지식 항목 유형"""
    FACT = "fact"
    CONCEPT = "concept"
    PROCEDURE = "procedure"
    SKILL = "skill"
    EXPERIENCE = "experience"
    INSIGHT = "insight"


class RelationshipType(Enum):
    """지식 관계 유형"""
    PREREQUISITE = "prerequisite"
    RELATED = "related"
    CONTRADICTION = "contradiction"
    EXAMPLE = "example"
    DEPENDENCY = "dependency"
    ALTERNATIVE = "alternative"


@dataclass
class KnowledgeMetadata:
    """지식 메타데이터"""
    source: str
    verified: bool
    importance: float
    complexity: int
    access_count: int = 0
    last_accessed: Optional[Timestamp] = None
    keywords: List[str] = field(default_factory=list)
    context_tags: List[str] = field(default_factory=list)


@dataclass
class KnowledgeRelationship:
    """지식 관계"""
    target_id: ID
    type: RelationshipType
    strength: float
    description: Optional[str] = None
    bidirectional: bool = True
    created_at: Timestamp = field(default_factory=current_timestamp)


@dataclass
class KnowledgeItem:
    """지식 항목"""
    id: ID
    user_id: ID
    title: str
    content: str
    tags: List[str]
    category: str
    knowledge_type: KnowledgeType
    difficulty: int  # 1-10 scale
    confidence: float  # 0.0-1.0 scale
    last_reviewed: Optional[Timestamp] = None
    review_count: int = 0
    relationships: List[KnowledgeRelationship] = field(default_factory=list)
    metadata: KnowledgeMetadata = field(default_factory=lambda: KnowledgeMetadata(
        source="user_input", verified=False, importance=0.5, complexity=1
    ))
    created_at: Timestamp = field(default_factory=current_timestamp)
    updated_at: Timestamp = field(default_factory=current_timestamp)

    def add_relationship(self, target_id: ID, relationship_type: RelationshipType,
                        strength: float, description: Optional[str] = None) -> None:
        """관계 추가"""
        relationship = KnowledgeRelationship(
            target_id=target_id,
            type=relationship_type,
            strength=strength,
            description=description
        )
        self.relationships.append(relationship)
        self.updated_at = current_timestamp()

    def update_confidence(self, new_confidence: float) -> None:
        """신뢰도 업데이트"""
        self.confidence = max(0.0, min(1.0, new_confidence))
        self.updated_at = current_timestamp()

    def mark_reviewed(self) -> None:
        """복습 기록"""
        self.last_reviewed = current_timestamp()
        self.review_count += 1
        self.updated_at = current_timestamp()


@dataclass
class KnowledgeSearchQuery:
    """지식 검색 쿼리"""
    query: str
    categories: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    knowledge_types: Optional[List[KnowledgeType]] = None
    min_confidence: float = 0.0
    max_difficulty: int = 10
    limit: int = 50
    include_relationships: bool = True


@dataclass
class KnowledgeSearchResult:
    """지식 검색 결과"""
    items: List[KnowledgeItem]
    total_count: int
    search_time_ms: float
    relevance_scores: Dict[ID, float]
    related_items: Dict[ID, List[KnowledgeItem]] = field(default_factory=dict)


class KnowledgeService(BaseService):
    """
    지식 관리 서비스

    Features:
    - 지식 항목 CRUD 작업
    - 지식 관계 관리
    - 의미적 검색
    - 지식 그래프 분석
    - 개인화된 추천
    """

    def __init__(
        self,
        config: ServiceConfig,
        events: Optional[EventEmitter] = None,
        cognitive_system: Optional[CognitiveSystem] = None
    ):
        super().__init__(config, events)
        self.cognitive_system = cognitive_system

        # In-memory storage (실제 구현에서는 데이터베이스 사용)
        self.knowledge_items: Dict[ID, KnowledgeItem] = {}
        self.user_knowledge: Dict[ID, Set[ID]] = {}  # user_id -> knowledge_item_ids
        self.category_index: Dict[str, Set[ID]] = {}
        self.tag_index: Dict[str, Set[ID]] = {}

        # 캐시
        self.search_cache: Dict[str, KnowledgeSearchResult] = {}
        self.relationship_cache: Dict[ID, List[KnowledgeRelationship]] = {}

    async def startup(self) -> None:
        """서비스 시작"""
        await super().startup()
        await self._load_knowledge_data()

        if self.events:
            await self.events.emit('knowledge.service.started', {
                'service_id': self.config.id,
                'total_items': len(self.knowledge_items)
            })

    async def shutdown(self) -> None:
        """서비스 종료"""
        await self._save_knowledge_data()
        await super().shutdown()

    async def create_knowledge_item(
        self,
        user_id: ID,
        title: str,
        content: str,
        category: str,
        knowledge_type: KnowledgeType,
        tags: Optional[List[str]] = None,
        difficulty: int = 1,
        metadata: Optional[KnowledgeMetadata] = None
    ) -> Result[KnowledgeItem]:
        """지식 항목 생성"""
        try:
            # 입력 검증
            if not title.strip():
                return create_failure(ValidationError(
                    message="제목은 필수입니다",
                    field="title",
                    value=title
                ))

            if not content.strip():
                return create_failure(ValidationError(
                    message="내용은 필수입니다",
                    field="content",
                    value=content
                ))

            if not 1 <= difficulty <= 10:
                return create_failure(ValidationError(
                    message="난이도는 1-10 사이여야 합니다",
                    field="difficulty",
                    value=difficulty
                ))

            # 지식 항목 생성
            knowledge_item = KnowledgeItem(
                id=generate_id(),
                user_id=user_id,
                title=title.strip(),
                content=content.strip(),
                tags=tags or [],
                category=category,
                knowledge_type=knowledge_type,
                difficulty=difficulty,
                confidence=0.8,  # 기본 신뢰도
                metadata=metadata or KnowledgeMetadata(
                    source="user_input",
                    verified=False,
                    importance=0.5,
                    complexity=difficulty
                )
            )

            # 저장
            self.knowledge_items[knowledge_item.id] = knowledge_item

            # 인덱스 업데이트
            if user_id not in self.user_knowledge:
                self.user_knowledge[user_id] = set()
            self.user_knowledge[user_id].add(knowledge_item.id)

            if category not in self.category_index:
                self.category_index[category] = set()
            self.category_index[category].add(knowledge_item.id)

            for tag in knowledge_item.tags:
                if tag not in self.tag_index:
                    self.tag_index[tag] = set()
                self.tag_index[tag].add(knowledge_item.id)

            # 관련 지식 자동 추천
            if self.cognitive_system:
                asyncio.create_task(self._suggest_relationships(knowledge_item))

            # 이벤트 발생
            if self.events:
                await self.events.emit('knowledge.item.created', {
                    'item_id': knowledge_item.id,
                    'user_id': user_id,
                    'category': category,
                    'type': knowledge_type.value
                })

            return create_success(knowledge_item)

        except Exception as error:
            return create_failure(ApplicationError(
                message=f"지식 항목 생성 실패: {str(error)}",
                service_name="KnowledgeService",
                operation="create_knowledge_item"
            ))

    async def get_knowledge_item(
        self,
        item_id: ID,
        user_id: Optional[ID] = None
    ) -> Result[KnowledgeItem]:
        """지식 항목 조회"""
        try:
            if item_id not in self.knowledge_items:
                return create_failure(ValidationError(
                    message="지식 항목을 찾을 수 없습니다",
                    field="item_id",
                    value=item_id
                ))

            item = self.knowledge_items[item_id]

            # 권한 확인
            if user_id and item.user_id != user_id:
                return create_failure(ApplicationError(
                    message="접근 권한이 없습니다",
                    service_name="KnowledgeService",
                    operation="get_knowledge_item"
                ))

            # 접근 기록
            item.metadata.access_count += 1
            item.metadata.last_accessed = current_timestamp()

            return create_success(item)

        except Exception as error:
            return create_failure(ApplicationError(
                message=f"지식 항목 조회 실패: {str(error)}",
                service_name="KnowledgeService",
                operation="get_knowledge_item"
            ))

    async def search_knowledge(
        self,
        query: KnowledgeSearchQuery,
        user_id: Optional[ID] = None
    ) -> Result[KnowledgeSearchResult]:
        """지식 검색"""
        try:
            start_time = time.time()

            # 캐시 확인
            cache_key = self._generate_search_cache_key(query, user_id)
            if cache_key in self.search_cache:
                cached_result = self.search_cache[cache_key]
                cached_result.search_time_ms = (time.time() - start_time) * 1000
                return create_success(cached_result)

            # 기본 필터링
            candidate_items = []

            if user_id:
                # 사용자별 지식으로 제한
                user_item_ids = self.user_knowledge.get(user_id, set())
                candidate_items = [
                    self.knowledge_items[item_id]
                    for item_id in user_item_ids
                    if item_id in self.knowledge_items
                ]
            else:
                candidate_items = list(self.knowledge_items.values())

            # 필터 적용
            filtered_items = []
            for item in candidate_items:
                # 카테고리 필터
                if query.categories and item.category not in query.categories:
                    continue

                # 태그 필터
                if query.tags and not any(tag in item.tags for tag in query.tags):
                    continue

                # 지식 유형 필터
                if query.knowledge_types and item.knowledge_type not in query.knowledge_types:
                    continue

                # 신뢰도 필터
                if item.confidence < query.min_confidence:
                    continue

                # 난이도 필터
                if item.difficulty > query.max_difficulty:
                    continue

                filtered_items.append(item)

            # 텍스트 매칭 및 관련성 점수 계산
            scored_items = []
            for item in filtered_items:
                score = self._calculate_relevance_score(query.query, item)
                if score > 0.1:  # 최소 관련성 임계값
                    scored_items.append((item, score))

            # 점수순 정렬
            scored_items.sort(key=lambda x: x[1], reverse=True)

            # 결과 제한
            result_items = [item for item, _ in scored_items[:query.limit]]
            relevance_scores = {item.id: score for item, score in scored_items[:query.limit]}

            # 관련 항목 검색
            related_items = {}
            if query.include_relationships:
                for item in result_items:
                    related = await self._get_related_items(item.id, limit=5)
                    if related:
                        related_items[item.id] = related

            search_time_ms = (time.time() - start_time) * 1000

            result = KnowledgeSearchResult(
                items=result_items,
                total_count=len(scored_items),
                search_time_ms=search_time_ms,
                relevance_scores=relevance_scores,
                related_items=related_items
            )

            # 캐시 저장
            self.search_cache[cache_key] = result

            return create_success(result)

        except Exception as error:
            return create_failure(ApplicationError(
                message=f"지식 검색 실패: {str(error)}",
                service_name="KnowledgeService",
                operation="search_knowledge"
            ))

    async def add_relationship(
        self,
        source_id: ID,
        target_id: ID,
        relationship_type: RelationshipType,
        strength: float,
        description: Optional[str] = None
    ) -> Result[bool]:
        """지식 관계 추가"""
        try:
            if source_id not in self.knowledge_items:
                return create_failure(ValidationError(
                    message="소스 지식 항목을 찾을 수 없습니다",
                    field="source_id",
                    value=source_id
                ))

            if target_id not in self.knowledge_items:
                return create_failure(ValidationError(
                    message="대상 지식 항목을 찾을 수 없습니다",
                    field="target_id",
                    value=target_id
                ))

            if not 0.0 <= strength <= 1.0:
                return create_failure(ValidationError(
                    message="관계 강도는 0.0-1.0 사이여야 합니다",
                    field="strength",
                    value=strength
                ))

            source_item = self.knowledge_items[source_id]
            source_item.add_relationship(target_id, relationship_type, strength, description)

            # 양방향 관계인 경우 역방향도 추가
            if relationship_type in [RelationshipType.RELATED, RelationshipType.CONTRADICTION]:
                target_item = self.knowledge_items[target_id]
                target_item.add_relationship(source_id, relationship_type, strength, description)

            # 캐시 무효화
            self._invalidate_relationship_cache(source_id)
            if target_id != source_id:
                self._invalidate_relationship_cache(target_id)

            return create_success(True)

        except Exception as error:
            return create_failure(ApplicationError(
                message=f"관계 추가 실패: {str(error)}",
                service_name="KnowledgeService",
                operation="add_relationship"
            ))

    def _calculate_relevance_score(self, query: str, item: KnowledgeItem) -> float:
        """관련성 점수 계산"""
        score = 0.0
        query_lower = query.lower()

        # 제목 매칭 (가중치 높음)
        if query_lower in item.title.lower():
            score += 0.8

        # 내용 매칭
        if query_lower in item.content.lower():
            score += 0.5

        # 태그 매칭
        for tag in item.tags:
            if query_lower in tag.lower():
                score += 0.3

        # 카테고리 매칭
        if query_lower in item.category.lower():
            score += 0.2

        # 키워드 매칭
        for keyword in item.metadata.keywords:
            if query_lower in keyword.lower():
                score += 0.3

        # 단어 기반 매칭
        query_words = set(query_lower.split())
        title_words = set(item.title.lower().split())
        content_words = set(item.content.lower().split())

        title_overlap = len(query_words & title_words) / max(len(query_words), 1)
        content_overlap = len(query_words & content_words) / max(len(query_words), 1)

        score += title_overlap * 0.6
        score += content_overlap * 0.3

        # 신뢰도 및 중요도 보정
        score *= item.confidence
        score *= item.metadata.importance

        return min(score, 1.0)

    async def _get_related_items(self, item_id: ID, limit: int = 5) -> List[KnowledgeItem]:
        """관련 항목 조회"""
        if item_id not in self.knowledge_items:
            return []

        item = self.knowledge_items[item_id]
        related_items = []

        for relationship in item.relationships:
            if relationship.target_id in self.knowledge_items:
                target_item = self.knowledge_items[relationship.target_id]
                related_items.append((target_item, relationship.strength))

        # 강도순 정렬
        related_items.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in related_items[:limit]]

    async def _suggest_relationships(self, item: KnowledgeItem) -> None:
        """관계 자동 추천"""
        if not self.cognitive_system:
            return

        try:
            # 인지 시스템을 통해 관련 항목 분석
            context = {
                'title': item.title,
                'content': item.content,
                'category': item.category,
                'tags': item.tags
            }

            # 유사한 항목 찾기
            similar_items = []
            for other_id, other_item in self.knowledge_items.items():
                if other_id == item.id:
                    continue

                similarity = self._calculate_similarity(item, other_item)
                if similarity > 0.6:
                    similar_items.append((other_item, similarity))

            # 자동 관계 생성 (임계값 이상인 경우)
            for similar_item, similarity in similar_items[:3]:
                if similarity > 0.8:
                    item.add_relationship(
                        similar_item.id,
                        RelationshipType.RELATED,
                        similarity,
                        "자동 감지된 관련성"
                    )

        except Exception as error:
            # 관계 추천 실패는 치명적이지 않음
            pass

    def _calculate_similarity(self, item1: KnowledgeItem, item2: KnowledgeItem) -> float:
        """항목 간 유사도 계산"""
        similarity = 0.0

        # 카테고리 유사도
        if item1.category == item2.category:
            similarity += 0.3

        # 태그 유사도
        common_tags = set(item1.tags) & set(item2.tags)
        if item1.tags and item2.tags:
            tag_similarity = len(common_tags) / max(len(set(item1.tags) | set(item2.tags)), 1)
            similarity += tag_similarity * 0.4

        # 텍스트 유사도 (간단한 버전)
        title_words1 = set(item1.title.lower().split())
        title_words2 = set(item2.title.lower().split())
        title_overlap = len(title_words1 & title_words2) / max(len(title_words1 | title_words2), 1)
        similarity += title_overlap * 0.3

        return min(similarity, 1.0)

    def _generate_search_cache_key(
        self,
        query: KnowledgeSearchQuery,
        user_id: Optional[ID]
    ) -> str:
        """검색 캐시 키 생성"""
        key_parts = [
            query.query,
            str(query.categories),
            str(query.tags),
            str([t.value for t in query.knowledge_types] if query.knowledge_types else None),
            str(query.min_confidence),
            str(query.max_difficulty),
            str(query.limit),
            str(user_id)
        ]
        return "|".join(key_parts)

    def _invalidate_relationship_cache(self, item_id: ID) -> None:
        """관계 캐시 무효화"""
        if item_id in self.relationship_cache:
            del self.relationship_cache[item_id]

    async def _load_knowledge_data(self) -> None:
        """지식 데이터 로드 (실제 구현에서는 데이터베이스에서)"""
        # 실제 구현에서는 데이터베이스에서 로드
        pass

    async def _save_knowledge_data(self) -> None:
        """지식 데이터 저장 (실제 구현에서는 데이터베이스에)"""
        # 실제 구현에서는 데이터베이스에 저장
        pass

    async def process_request(self, context: ServiceContext) -> ServiceResult:
        """서비스 요청 처리"""
        operation = context.operation

        if operation == "create":
            return await self._handle_create_request(context)
        elif operation == "search":
            return await self._handle_search_request(context)
        elif operation == "get":
            return await self._handle_get_request(context)
        elif operation == "add_relationship":
            return await self._handle_add_relationship_request(context)
        else:
            return ServiceResult(
                success=False,
                data=None,
                error=f"Unknown operation: {operation}",
                processing_time_ms=0
            )

    async def _handle_create_request(self, context: ServiceContext) -> ServiceResult:
        """생성 요청 처리"""
        start_time = time.time()

        try:
            params = context.parameters
            result = await self.create_knowledge_item(
                user_id=params.get('user_id'),
                title=params.get('title', ''),
                content=params.get('content', ''),
                category=params.get('category', 'general'),
                knowledge_type=KnowledgeType(params.get('knowledge_type', 'fact')),
                tags=params.get('tags', []),
                difficulty=params.get('difficulty', 1)
            )

            processing_time_ms = (time.time() - start_time) * 1000

            if result.success:
                return ServiceResult(
                    success=True,
                    data=result.data,
                    processing_time_ms=processing_time_ms
                )
            else:
                return ServiceResult(
                    success=False,
                    data=None,
                    error=str(result.error),
                    processing_time_ms=processing_time_ms
                )

        except Exception as error:
            processing_time_ms = (time.time() - start_time) * 1000
            return ServiceResult(
                success=False,
                data=None,
                error=str(error),
                processing_time_ms=processing_time_ms
            )

    async def _handle_search_request(self, context: ServiceContext) -> ServiceResult:
        """검색 요청 처리"""
        start_time = time.time()

        try:
            params = context.parameters
            query = KnowledgeSearchQuery(
                query=params.get('query', ''),
                categories=params.get('categories'),
                tags=params.get('tags'),
                knowledge_types=[KnowledgeType(t) for t in params.get('knowledge_types', [])] if params.get('knowledge_types') else None,
                min_confidence=params.get('min_confidence', 0.0),
                max_difficulty=params.get('max_difficulty', 10),
                limit=params.get('limit', 50),
                include_relationships=params.get('include_relationships', True)
            )

            result = await self.search_knowledge(query, params.get('user_id'))

            processing_time_ms = (time.time() - start_time) * 1000

            if result.success:
                return ServiceResult(
                    success=True,
                    data=result.data,
                    processing_time_ms=processing_time_ms
                )
            else:
                return ServiceResult(
                    success=False,
                    data=None,
                    error=str(result.error),
                    processing_time_ms=processing_time_ms
                )

        except Exception as error:
            processing_time_ms = (time.time() - start_time) * 1000
            return ServiceResult(
                success=False,
                data=None,
                error=str(error),
                processing_time_ms=processing_time_ms
            )

    async def _handle_get_request(self, context: ServiceContext) -> ServiceResult:
        """조회 요청 처리"""
        start_time = time.time()

        try:
            params = context.parameters
            result = await self.get_knowledge_item(
                item_id=params.get('item_id'),
                user_id=params.get('user_id')
            )

            processing_time_ms = (time.time() - start_time) * 1000

            if result.success:
                return ServiceResult(
                    success=True,
                    data=result.data,
                    processing_time_ms=processing_time_ms
                )
            else:
                return ServiceResult(
                    success=False,
                    data=None,
                    error=str(result.error),
                    processing_time_ms=processing_time_ms
                )

        except Exception as error:
            processing_time_ms = (time.time() - start_time) * 1000
            return ServiceResult(
                success=False,
                data=None,
                error=str(error),
                processing_time_ms=processing_time_ms
            )

    async def _handle_add_relationship_request(self, context: ServiceContext) -> ServiceResult:
        """관계 추가 요청 처리"""
        start_time = time.time()

        try:
            params = context.parameters
            result = await self.add_relationship(
                source_id=params.get('source_id'),
                target_id=params.get('target_id'),
                relationship_type=RelationshipType(params.get('relationship_type')),
                strength=params.get('strength', 1.0),
                description=params.get('description')
            )

            processing_time_ms = (time.time() - start_time) * 1000

            if result.success:
                return ServiceResult(
                    success=True,
                    data=result.data,
                    processing_time_ms=processing_time_ms
                )
            else:
                return ServiceResult(
                    success=False,
                    data=None,
                    error=str(result.error),
                    processing_time_ms=processing_time_ms
                )

        except Exception as error:
            processing_time_ms = (time.time() - start_time) * 1000
            return ServiceResult(
                success=False,
                data=None,
                error=str(error),
                processing_time_ms=processing_time_ms
            )

    def get_service_info(self) -> Dict[str, Any]:
        """서비스 정보 조회"""
        return {
            'name': 'KnowledgeService',
            'version': '1.0.0',
            'description': 'Knowledge management service with semantic search and relationship management',
            'total_items': len(self.knowledge_items),
            'total_users': len(self.user_knowledge),
            'total_categories': len(self.category_index),
            'total_tags': len(self.tag_index),
            'cache_size': len(self.search_cache),
            'supported_operations': ['create', 'search', 'get', 'add_relationship']
        }