"""
Concept Former

개념 형성 시스템으로, 인식된 패턴들로부터 추상적 개념을 형성하고
개념 간의 관계를 학습합니다.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Callable, Union
from uuid import uuid4, UUID
import json

from ...base import BaseCognitiveProcessor


class ConceptType(Enum):
    """개념 유형"""
    CONCRETE = auto()       # 구체적 개념
    ABSTRACT = auto()       # 추상적 개념
    RELATIONAL = auto()     # 관계적 개념
    PROCEDURAL = auto()     # 절차적 개념
    CATEGORICAL = auto()    # 범주적 개념
    TEMPORAL = auto()       # 시간적 개념


class AbstractionLevel(Enum):
    """추상화 수준"""
    INSTANCE = 1           # 개별 인스턴스
    BASIC = 2              # 기본 수준
    SUPERORDINATE = 3      # 상위 수준
    ABSTRACT = 4           # 추상 수준
    META = 5               # 메타 수준


@dataclass
class Concept:
    """개념 정의"""
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    type: ConceptType = ConceptType.CONCRETE
    abstraction_level: AbstractionLevel = AbstractionLevel.BASIC

    # 개념 내용
    features: Dict[str, Any] = field(default_factory=dict)
    examples: List[Any] = field(default_factory=list)
    counter_examples: List[Any] = field(default_factory=list)

    # 관계
    parent_concepts: Set[UUID] = field(default_factory=set)
    child_concepts: Set[UUID] = field(default_factory=set)
    related_concepts: Dict[UUID, str] = field(default_factory=dict)  # concept_id -> relation_type

    # 통계
    activation_count: int = 0
    confidence: float = 0.5
    stability: float = 0.5          # 개념의 안정성
    coherence: float = 0.5          # 개념의 일관성

    # 메타데이터
    created_at: float = field(default_factory=time.time)
    last_activated: float = field(default_factory=time.time)
    source_patterns: List[str] = field(default_factory=list)


@dataclass
class ConceptFormationResult:
    """개념 형성 결과"""
    formed_concepts: List[Concept] = field(default_factory=list)
    updated_concepts: List[Concept] = field(default_factory=list)
    new_relations: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class ConceptFormer(BaseCognitiveProcessor):
    """
    개념 형성 시스템

    패턴들로부터 개념을 형성하고, 개념 간의 관계를 학습하며,
    계층적 개념 구조를 구축합니다.
    """

    def __init__(self, max_concepts: int = 5000):
        super().__init__()
        self.max_concepts = max_concepts

        # 개념 저장소
        self._concepts: Dict[UUID, Concept] = {}
        self._concept_index: Dict[str, Set[UUID]] = {}  # 빠른 검색용

        # 형성 규칙
        self._formation_rules: List[Callable] = []
        self._abstraction_rules: List[Callable] = []

        # 학습 파라미터
        self._formation_threshold = 0.7
        self._similarity_threshold = 0.8
        self._abstraction_threshold = 3  # 최소 예시 수

        # 통계
        self._total_formations = 0
        self._successful_formations = 0

    async def initialize(self) -> bool:
        """개념 형성 시스템 초기화"""
        try:
            self.logger.info("Initializing Concept Former...")

            # 기본 형성 규칙 설정
            await self._setup_formation_rules()

            # 기본 개념 로드
            await self._load_basic_concepts()

            # 백그라운드 프로세스 시작
            asyncio.create_task(self._concept_maintenance())
            asyncio.create_task(self._hierarchy_optimization())

            self.logger.info("Concept Former initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Concept Former: {e}")
            return False

    async def form_concepts(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        패턴들로부터 개념 형성

        Args:
            patterns: 입력 패턴 목록

        Returns:
            형성된 개념들의 정보
        """
        try:
            start_time = time.time()
            result = ConceptFormationResult()

            if not patterns:
                return []

            # 1. 기존 개념과 매칭 시도
            matched_concepts = await self._match_existing_concepts(patterns)

            # 2. 새로운 개념 형성 후보 생성
            formation_candidates = await self._generate_formation_candidates(patterns)

            # 3. 개념 형성 실행
            for candidate in formation_candidates:
                if await self._should_form_concept(candidate):
                    new_concept = await self._create_concept_from_patterns(candidate["patterns"])
                    if new_concept:
                        result.formed_concepts.append(new_concept)
                        self._concepts[new_concept.id] = new_concept
                        await self._update_concept_index(new_concept)

            # 4. 기존 개념 업데이트
            for concept_id in matched_concepts:
                if concept_id in self._concepts:
                    concept = self._concepts[concept_id]
                    updated = await self._update_concept_with_patterns(concept, patterns)
                    if updated:
                        result.updated_concepts.append(concept)

            # 5. 관계 형성
            new_relations = await self._form_concept_relations(
                result.formed_concepts + result.updated_concepts
            )
            result.new_relations = new_relations

            # 6. 추상화 수행
            await self._perform_abstraction(result.formed_concepts)

            result.processing_time_ms = (time.time() - start_time) * 1000
            self._update_formation_metrics(result)

            # 반환 형식 변환
            return self._convert_concepts_to_dict(result.formed_concepts)

        except Exception as e:
            self.logger.error(f"Error forming concepts: {e}")
            return []

    async def update_concepts(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """기존 개념들 업데이트"""
        try:
            updated_concepts = []

            # 관련 개념 찾기
            relevant_concepts = await self._find_relevant_concepts(patterns)

            for concept_id in relevant_concepts:
                if concept_id in self._concepts:
                    concept = self._concepts[concept_id]

                    # 개념 업데이트
                    if await self._update_concept_with_patterns(concept, patterns):
                        updated_concepts.append(concept)

            return self._convert_concepts_to_dict(updated_concepts)

        except Exception as e:
            self.logger.error(f"Error updating concepts: {e}")
            return []

    async def get_concept_hierarchy(self) -> Dict[str, Any]:
        """개념 계층 구조 조회"""
        try:
            hierarchy = {
                "root_concepts": [],
                "total_concepts": len(self._concepts),
                "abstraction_levels": {}
            }

            # 추상화 수준별 분류
            for level in AbstractionLevel:
                hierarchy["abstraction_levels"][level.name] = []

            # 루트 개념 및 수준별 분류
            for concept in self._concepts.values():
                concept_info = {
                    "id": str(concept.id),
                    "name": concept.name,
                    "type": concept.type.name,
                    "confidence": concept.confidence,
                    "child_count": len(concept.child_concepts)
                }

                # 루트 개념 확인 (부모가 없는 개념)
                if not concept.parent_concepts:
                    hierarchy["root_concepts"].append(concept_info)

                # 추상화 수준별 분류
                level_name = concept.abstraction_level.name
                hierarchy["abstraction_levels"][level_name].append(concept_info)

            return hierarchy

        except Exception as e:
            self.logger.error(f"Error getting concept hierarchy: {e}")
            return {"error": str(e)}

    async def find_concept_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """이름으로 개념 찾기"""
        try:
            name_lower = name.lower()
            if name_lower in self._concept_index:
                concept_ids = self._concept_index[name_lower]
                if concept_ids:
                    concept_id = next(iter(concept_ids))
                    concept = self._concepts[concept_id]
                    return self._convert_concept_to_dict(concept)

            return None

        except Exception as e:
            self.logger.error(f"Error finding concept by name {name}: {e}")
            return None

    async def get_concept_relations(self, concept_id: UUID) -> Dict[str, Any]:
        """개념의 관계 정보 조회"""
        try:
            if concept_id not in self._concepts:
                return {"error": "Concept not found"}

            concept = self._concepts[concept_id]
            relations = {
                "concept_id": str(concept_id),
                "concept_name": concept.name,
                "parents": [],
                "children": [],
                "related": []
            }

            # 부모 개념들
            for parent_id in concept.parent_concepts:
                if parent_id in self._concepts:
                    parent = self._concepts[parent_id]
                    relations["parents"].append({
                        "id": str(parent_id),
                        "name": parent.name,
                        "type": parent.type.name
                    })

            # 자식 개념들
            for child_id in concept.child_concepts:
                if child_id in self._concepts:
                    child = self._concepts[child_id]
                    relations["children"].append({
                        "id": str(child_id),
                        "name": child.name,
                        "type": child.type.name
                    })

            # 관련 개념들
            for related_id, relation_type in concept.related_concepts.items():
                if related_id in self._concepts:
                    related = self._concepts[related_id]
                    relations["related"].append({
                        "id": str(related_id),
                        "name": related.name,
                        "relation_type": relation_type,
                        "type": related.type.name
                    })

            return relations

        except Exception as e:
            self.logger.error(f"Error getting concept relations: {e}")
            return {"error": str(e)}

    async def _match_existing_concepts(self, patterns: List[Dict[str, Any]]) -> Set[UUID]:
        """기존 개념과 매칭"""
        matched_concepts = set()

        try:
            for pattern in patterns:
                pattern_name = pattern.get("name", "").lower()
                pattern_features = pattern.get("features", {})

                # 이름 기반 매칭
                if pattern_name in self._concept_index:
                    matched_concepts.update(self._concept_index[pattern_name])

                # 특징 기반 매칭
                for concept in self._concepts.values():
                    similarity = await self._calculate_pattern_concept_similarity(
                        pattern, concept
                    )
                    if similarity >= self._similarity_threshold:
                        matched_concepts.add(concept.id)

        except Exception as e:
            self.logger.error(f"Error matching existing concepts: {e}")

        return matched_concepts

    async def _generate_formation_candidates(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """개념 형성 후보 생성"""
        candidates = []

        try:
            # 1. 유사한 패턴들을 그룹화
            pattern_groups = await self._group_similar_patterns(patterns)

            # 2. 각 그룹을 개념 형성 후보로 변환
            for group in pattern_groups:
                if len(group) >= 2:  # 최소 2개 패턴 필요
                    candidate = {
                        "patterns": group,
                        "commonalities": await self._extract_commonalities(group),
                        "formation_confidence": await self._calculate_formation_confidence(group)
                    }
                    candidates.append(candidate)

            # 형성 신뢰도 순으로 정렬
            candidates.sort(key=lambda x: x["formation_confidence"], reverse=True)

        except Exception as e:
            self.logger.error(f"Error generating formation candidates: {e}")

        return candidates

    async def _group_similar_patterns(self, patterns: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """유사한 패턴들 그룹화"""
        groups = []
        used_patterns = set()

        try:
            for i, pattern1 in enumerate(patterns):
                if i in used_patterns:
                    continue

                group = [pattern1]
                used_patterns.add(i)

                for j, pattern2 in enumerate(patterns[i+1:], i+1):
                    if j in used_patterns:
                        continue

                    similarity = await self._calculate_pattern_similarity(pattern1, pattern2)
                    if similarity >= self._similarity_threshold:
                        group.append(pattern2)
                        used_patterns.add(j)

                if len(group) > 1:
                    groups.append(group)

        except Exception as e:
            self.logger.error(f"Error grouping similar patterns: {e}")

        return groups

    async def _calculate_pattern_similarity(self, pattern1: Dict[str, Any],
                                          pattern2: Dict[str, Any]) -> float:
        """패턴 간 유사도 계산"""
        try:
            similarities = []

            # 이름 유사도
            name1 = pattern1.get("name", "").lower()
            name2 = pattern2.get("name", "").lower()
            if name1 and name2:
                name_sim = len(set(name1.split()) & set(name2.split())) / len(set(name1.split()) | set(name2.split()))
                similarities.append(name_sim)

            # 타입 유사도
            type1 = pattern1.get("type", "")
            type2 = pattern2.get("type", "")
            if type1 == type2:
                similarities.append(1.0)
            else:
                similarities.append(0.0)

            # 특징 유사도
            features1 = pattern1.get("features", {})
            features2 = pattern2.get("features", {})
            if features1 and features2:
                feature_sim = await self._calculate_feature_similarity(features1, features2)
                similarities.append(feature_sim)

            return sum(similarities) / len(similarities) if similarities else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating pattern similarity: {e}")
            return 0.0

    async def _calculate_feature_similarity(self, features1: Dict[str, Any],
                                          features2: Dict[str, Any]) -> float:
        """특징 간 유사도 계산"""
        try:
            if not features1 or not features2:
                return 0.0

            common_keys = set(features1.keys()) & set(features2.keys())
            all_keys = set(features1.keys()) | set(features2.keys())

            if not all_keys:
                return 1.0

            # 키 일치도
            key_similarity = len(common_keys) / len(all_keys)

            # 값 일치도
            value_similarities = []
            for key in common_keys:
                val1, val2 = features1[key], features2[key]
                if val1 == val2:
                    value_similarities.append(1.0)
                elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # 숫자 값의 유사도
                    max_val = max(abs(val1), abs(val2), 1)
                    diff_ratio = abs(val1 - val2) / max_val
                    value_similarities.append(max(0, 1 - diff_ratio))
                else:
                    value_similarities.append(0.0)

            value_similarity = sum(value_similarities) / len(value_similarities) if value_similarities else 0.0

            return (key_similarity + value_similarity) / 2

        except Exception as e:
            self.logger.error(f"Error calculating feature similarity: {e}")
            return 0.0

    async def _extract_commonalities(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """패턴들의 공통점 추출"""
        commonalities = {}

        try:
            if not patterns:
                return commonalities

            # 공통 특징 추출
            all_features = [p.get("features", {}) for p in patterns]
            if all_features:
                common_feature_keys = set(all_features[0].keys())
                for features in all_features[1:]:
                    common_feature_keys &= set(features.keys())

                common_features = {}
                for key in common_feature_keys:
                    values = [features[key] for features in all_features]
                    # 모든 값이 같으면 공통 특징으로 추가
                    if len(set(str(v) for v in values)) == 1:
                        common_features[key] = values[0]

                commonalities["features"] = common_features

            # 공통 타입
            types = [p.get("type") for p in patterns]
            if len(set(types)) == 1 and types[0]:
                commonalities["type"] = types[0]

            # 평균 신뢰도
            confidences = [p.get("confidence", 0.5) for p in patterns]
            commonalities["average_confidence"] = sum(confidences) / len(confidences)

        except Exception as e:
            self.logger.error(f"Error extracting commonalities: {e}")

        return commonalities

    async def _calculate_formation_confidence(self, patterns: List[Dict[str, Any]]) -> float:
        """개념 형성 신뢰도 계산"""
        try:
            if len(patterns) < 2:
                return 0.0

            # 패턴 수에 따른 기본 신뢰도
            base_confidence = min(len(patterns) / 5, 1.0)

            # 패턴 간 일관성
            consistency_scores = []
            for i in range(len(patterns)):
                for j in range(i+1, len(patterns)):
                    similarity = await self._calculate_pattern_similarity(patterns[i], patterns[j])
                    consistency_scores.append(similarity)

            consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0

            # 패턴 품질 (평균 신뢰도)
            quality_scores = [p.get("confidence", 0.5) for p in patterns]
            quality = sum(quality_scores) / len(quality_scores)

            # 가중 평균
            formation_confidence = (base_confidence * 0.3 + consistency * 0.4 + quality * 0.3)

            return formation_confidence

        except Exception as e:
            self.logger.error(f"Error calculating formation confidence: {e}")
            return 0.0

    async def _should_form_concept(self, candidate: Dict[str, Any]) -> bool:
        """개념 형성 여부 결정"""
        try:
            formation_confidence = candidate.get("formation_confidence", 0.0)

            # 신뢰도 임계값 확인
            if formation_confidence < self._formation_threshold:
                return False

            # 패턴 수 확인
            patterns = candidate.get("patterns", [])
            if len(patterns) < 2:
                return False

            # 중복 개념 확인
            commonalities = candidate.get("commonalities", {})
            if await self._is_duplicate_concept(commonalities):
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error deciding concept formation: {e}")
            return False

    async def _is_duplicate_concept(self, commonalities: Dict[str, Any]) -> bool:
        """중복 개념 확인"""
        try:
            # 기존 개념들과 유사도 확인
            for concept in self._concepts.values():
                similarity = await self._calculate_commonality_concept_similarity(
                    commonalities, concept
                )
                if similarity >= 0.9:  # 매우 높은 유사도
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking duplicate concept: {e}")
            return False

    async def _calculate_commonality_concept_similarity(self, commonalities: Dict[str, Any],
                                                       concept: Concept) -> float:
        """공통점과 기존 개념 간 유사도"""
        try:
            similarities = []

            # 특징 유사도
            common_features = commonalities.get("features", {})
            if common_features and concept.features:
                feature_sim = await self._calculate_feature_similarity(
                    common_features, concept.features
                )
                similarities.append(feature_sim)

            # 타입 유사도
            common_type = commonalities.get("type")
            if common_type:
                type_sim = 1.0 if common_type == concept.type.name else 0.0
                similarities.append(type_sim)

            return sum(similarities) / len(similarities) if similarities else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating commonality-concept similarity: {e}")
            return 0.0

    async def _create_concept_from_patterns(self, patterns: List[Dict[str, Any]]) -> Optional[Concept]:
        """패턴들로부터 개념 생성"""
        try:
            if not patterns:
                return None

            commonalities = await self._extract_commonalities(patterns)

            # 개념 이름 생성
            concept_name = await self._generate_concept_name(patterns, commonalities)

            # 개념 타입 결정
            concept_type = await self._determine_concept_type(patterns, commonalities)

            # 추상화 수준 결정
            abstraction_level = await self._determine_abstraction_level(patterns)

            # 개념 생성
            concept = Concept(
                name=concept_name,
                type=concept_type,
                abstraction_level=abstraction_level,
                features=commonalities.get("features", {}),
                examples=[p.get("data") for p in patterns if p.get("data")],
                confidence=commonalities.get("average_confidence", 0.5),
                source_patterns=[p.get("name", "") for p in patterns]
            )

            # 초기 통계 설정
            concept.activation_count = 1
            concept.stability = 0.7  # 새 개념은 중간 안정성
            concept.coherence = await self._calculate_formation_confidence(patterns)

            return concept

        except Exception as e:
            self.logger.error(f"Error creating concept from patterns: {e}")
            return None

    async def _generate_concept_name(self, patterns: List[Dict[str, Any]],
                                   commonalities: Dict[str, Any]) -> str:
        """개념 이름 생성"""
        try:
            # 공통 키워드 추출
            all_names = [p.get("name", "") for p in patterns if p.get("name")]

            if not all_names:
                return f"concept_{int(time.time())}"

            # 가장 빈번한 단어들 찾기
            word_counts = {}
            for name in all_names:
                words = name.lower().split()
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1

            if word_counts:
                # 가장 빈번한 단어 선택
                most_common_word = max(word_counts, key=word_counts.get)
                return f"{most_common_word}_concept"

            return f"pattern_concept_{int(time.time())}"

        except Exception as e:
            self.logger.error(f"Error generating concept name: {e}")
            return f"concept_{int(time.time())}"

    async def _determine_concept_type(self, patterns: List[Dict[str, Any]],
                                    commonalities: Dict[str, Any]) -> ConceptType:
        """개념 타입 결정"""
        try:
            # 패턴 타입 기반 결정
            pattern_types = [p.get("type", "") for p in patterns]

            if "SEQUENTIAL" in pattern_types:
                return ConceptType.PROCEDURAL
            elif "SEMANTIC" in pattern_types:
                return ConceptType.ABSTRACT
            elif "SPATIAL" in pattern_types:
                return ConceptType.CONCRETE
            elif "STRUCTURAL" in pattern_types:
                return ConceptType.CATEGORICAL
            else:
                return ConceptType.CONCRETE

        except Exception as e:
            self.logger.error(f"Error determining concept type: {e}")
            return ConceptType.CONCRETE

    async def _determine_abstraction_level(self, patterns: List[Dict[str, Any]]) -> AbstractionLevel:
        """추상화 수준 결정"""
        try:
            # 패턴 수와 복잡도에 따라 결정
            pattern_count = len(patterns)

            if pattern_count >= 10:
                return AbstractionLevel.ABSTRACT
            elif pattern_count >= 5:
                return AbstractionLevel.SUPERORDINATE
            else:
                return AbstractionLevel.BASIC

        except Exception as e:
            self.logger.error(f"Error determining abstraction level: {e}")
            return AbstractionLevel.BASIC

    async def _update_concept_with_patterns(self, concept: Concept,
                                          patterns: List[Dict[str, Any]]) -> bool:
        """패턴들로 기존 개념 업데이트"""
        try:
            updated = False

            # 새로운 예시 추가
            for pattern in patterns:
                pattern_data = pattern.get("data")
                if pattern_data and pattern_data not in concept.examples:
                    concept.examples.append(pattern_data)
                    updated = True

            # 특징 업데이트
            for pattern in patterns:
                pattern_features = pattern.get("features", {})
                for key, value in pattern_features.items():
                    if key not in concept.features:
                        concept.features[key] = value
                        updated = True

            if updated:
                # 통계 업데이트
                concept.activation_count += 1
                concept.last_activated = time.time()

                # 신뢰도 업데이트 (새 패턴의 품질 고려)
                pattern_confidences = [p.get("confidence", 0.5) for p in patterns]
                avg_new_confidence = sum(pattern_confidences) / len(pattern_confidences)
                concept.confidence = (concept.confidence * 0.8 + avg_new_confidence * 0.2)

                # 안정성 증가
                concept.stability = min(1.0, concept.stability + 0.1)

            return updated

        except Exception as e:
            self.logger.error(f"Error updating concept with patterns: {e}")
            return False

    async def _update_concept_index(self, concept: Concept) -> None:
        """개념 인덱스 업데이트"""
        try:
            # 이름 기반 인덱스
            name_key = concept.name.lower()
            if name_key not in self._concept_index:
                self._concept_index[name_key] = set()
            self._concept_index[name_key].add(concept.id)

            # 타입 기반 인덱스
            type_key = concept.type.name.lower()
            if type_key not in self._concept_index:
                self._concept_index[type_key] = set()
            self._concept_index[type_key].add(concept.id)

        except Exception as e:
            self.logger.error(f"Error updating concept index: {e}")

    def _convert_concepts_to_dict(self, concepts: List[Concept]) -> List[Dict[str, Any]]:
        """개념들을 딕셔너리 형태로 변환"""
        return [self._convert_concept_to_dict(concept) for concept in concepts]

    def _convert_concept_to_dict(self, concept: Concept) -> Dict[str, Any]:
        """개념을 딕셔너리로 변환"""
        return {
            "id": str(concept.id),
            "name": concept.name,
            "type": concept.type.name,
            "abstraction_level": concept.abstraction_level.name,
            "confidence": concept.confidence,
            "stability": concept.stability,
            "coherence": concept.coherence,
            "activation_count": concept.activation_count,
            "features": concept.features,
            "example_count": len(concept.examples),
            "parent_count": len(concept.parent_concepts),
            "child_count": len(concept.child_concepts),
            "created_at": concept.created_at,
            "last_activated": concept.last_activated
        }

    async def _calculate_pattern_concept_similarity(self, pattern: Dict[str, Any],
                                                   concept: Concept) -> float:
        """패턴과 개념 간 유사도 계산"""
        try:
            similarities = []

            # 이름 유사도
            pattern_name = pattern.get("name", "").lower()
            if pattern_name and concept.name:
                name_words = set(pattern_name.split())
                concept_words = set(concept.name.lower().split())
                if name_words and concept_words:
                    name_sim = len(name_words & concept_words) / len(name_words | concept_words)
                    similarities.append(name_sim)

            # 특징 유사도
            pattern_features = pattern.get("features", {})
            if pattern_features and concept.features:
                feature_sim = await self._calculate_feature_similarity(
                    pattern_features, concept.features
                )
                similarities.append(feature_sim)

            # 타입 유사도
            pattern_type = pattern.get("type", "")
            if pattern_type == concept.type.name:
                similarities.append(1.0)
            else:
                similarities.append(0.0)

            return sum(similarities) / len(similarities) if similarities else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating pattern-concept similarity: {e}")
            return 0.0

    async def _find_relevant_concepts(self, patterns: List[Dict[str, Any]]) -> Set[UUID]:
        """관련 개념들 찾기"""
        relevant_concepts = set()

        try:
            for pattern in patterns:
                # 이름 기반 검색
                pattern_name = pattern.get("name", "").lower()
                if pattern_name in self._concept_index:
                    relevant_concepts.update(self._concept_index[pattern_name])

                # 특징 기반 검색
                pattern_features = pattern.get("features", {})
                for concept in self._concepts.values():
                    if await self._has_overlapping_features(pattern_features, concept.features):
                        relevant_concepts.add(concept.id)

        except Exception as e:
            self.logger.error(f"Error finding relevant concepts: {e}")

        return relevant_concepts

    async def _has_overlapping_features(self, features1: Dict[str, Any],
                                       features2: Dict[str, Any]) -> bool:
        """특징 겹침 여부 확인"""
        try:
            if not features1 or not features2:
                return False

            common_keys = set(features1.keys()) & set(features2.keys())
            return len(common_keys) > 0

        except Exception as e:
            self.logger.error(f"Error checking overlapping features: {e}")
            return False

    async def _form_concept_relations(self, concepts: List[Concept]) -> List[Dict[str, Any]]:
        """개념 간 관계 형성"""
        new_relations = []

        try:
            for i, concept1 in enumerate(concepts):
                for j, concept2 in enumerate(concepts[i+1:], i+1):
                    relation = await self._determine_concept_relation(concept1, concept2)
                    if relation:
                        new_relations.append(relation)
                        await self._add_concept_relation(concept1, concept2, relation["type"])

        except Exception as e:
            self.logger.error(f"Error forming concept relations: {e}")

        return new_relations

    async def _determine_concept_relation(self, concept1: Concept, concept2: Concept) -> Optional[Dict[str, Any]]:
        """두 개념 간 관계 결정"""
        try:
            # 추상화 수준 기반 관계
            if concept1.abstraction_level.value > concept2.abstraction_level.value:
                return {
                    "type": "parent_child",
                    "parent": concept1.id,
                    "child": concept2.id,
                    "confidence": 0.8
                }
            elif concept2.abstraction_level.value > concept1.abstraction_level.value:
                return {
                    "type": "parent_child",
                    "parent": concept2.id,
                    "child": concept1.id,
                    "confidence": 0.8
                }

            # 특징 유사도 기반 관계
            feature_sim = await self._calculate_feature_similarity(
                concept1.features, concept2.features
            )
            if feature_sim > 0.7:
                return {
                    "type": "similar",
                    "concept1": concept1.id,
                    "concept2": concept2.id,
                    "confidence": feature_sim
                }

            return None

        except Exception as e:
            self.logger.error(f"Error determining concept relation: {e}")
            return None

    async def _add_concept_relation(self, concept1: Concept, concept2: Concept, relation_type: str) -> None:
        """개념 간 관계 추가"""
        try:
            if relation_type == "parent_child":
                # 추상화 수준에 따라 부모-자식 관계 설정
                if concept1.abstraction_level.value > concept2.abstraction_level.value:
                    concept1.child_concepts.add(concept2.id)
                    concept2.parent_concepts.add(concept1.id)
                else:
                    concept2.child_concepts.add(concept1.id)
                    concept1.parent_concepts.add(concept2.id)
            else:
                # 일반적인 관계
                concept1.related_concepts[concept2.id] = relation_type
                concept2.related_concepts[concept1.id] = relation_type

        except Exception as e:
            self.logger.error(f"Error adding concept relation: {e}")

    async def _perform_abstraction(self, concepts: List[Concept]) -> None:
        """추상화 수행"""
        try:
            for concept in concepts:
                if len(concept.examples) >= self._abstraction_threshold:
                    await self._attempt_abstraction(concept)

        except Exception as e:
            self.logger.error(f"Error performing abstraction: {e}")

    async def _attempt_abstraction(self, concept: Concept) -> None:
        """개념의 추상화 시도"""
        try:
            # 현재 추상화 수준이 최고가 아닌 경우에만 추상화
            if concept.abstraction_level != AbstractionLevel.META:
                # 추상화 조건 확인
                if (concept.activation_count >= 10 and
                    concept.stability >= 0.8 and
                    len(concept.examples) >= 5):

                    # 추상화 수준 상승
                    current_level = concept.abstraction_level.value
                    new_level = min(current_level + 1, AbstractionLevel.META.value)
                    concept.abstraction_level = AbstractionLevel(new_level)

                    self.logger.info(f"Abstracted concept {concept.name} to level {concept.abstraction_level.name}")

        except Exception as e:
            self.logger.error(f"Error attempting abstraction for concept {concept.name}: {e}")

    def _update_formation_metrics(self, result: ConceptFormationResult) -> None:
        """형성 메트릭 업데이트"""
        self._total_formations += 1

        if result.success and (result.formed_concepts or result.updated_concepts):
            self._successful_formations += 1

    async def _setup_formation_rules(self) -> None:
        """형성 규칙 설정"""
        # 추후 구현: 복잡한 형성 규칙들
        pass

    async def _load_basic_concepts(self) -> None:
        """기본 개념들 로드"""
        try:
            basic_concepts = [
                {
                    "name": "object",
                    "type": ConceptType.CONCRETE,
                    "level": AbstractionLevel.SUPERORDINATE,
                    "features": {"physical": True}
                },
                {
                    "name": "action",
                    "type": ConceptType.PROCEDURAL,
                    "level": AbstractionLevel.BASIC,
                    "features": {"temporal": True}
                },
                {
                    "name": "property",
                    "type": ConceptType.ABSTRACT,
                    "level": AbstractionLevel.BASIC,
                    "features": {"descriptive": True}
                }
            ]

            for concept_data in basic_concepts:
                concept = Concept(
                    name=concept_data["name"],
                    type=concept_data["type"],
                    abstraction_level=concept_data["level"],
                    features=concept_data["features"],
                    confidence=1.0,
                    stability=1.0,
                    coherence=1.0
                )

                self._concepts[concept.id] = concept
                await self._update_concept_index(concept)

        except Exception as e:
            self.logger.error(f"Error loading basic concepts: {e}")

    async def _concept_maintenance(self) -> None:
        """개념 유지보수 백그라운드 프로세스"""
        while True:
            try:
                current_time = time.time()

                # 오래 사용되지 않은 개념 정리
                concepts_to_remove = []
                for concept_id, concept in self._concepts.items():
                    # 30일 이상 사용되지 않고 안정성이 낮은 개념
                    if (current_time - concept.last_activated > 30 * 24 * 3600 and
                        concept.stability < 0.3 and
                        concept.activation_count < 5):
                        concepts_to_remove.append(concept_id)

                # 개념 제거
                for concept_id in concepts_to_remove:
                    if concept_id in self._concepts:
                        removed_concept = self._concepts[concept_id]
                        del self._concepts[concept_id]
                        self.logger.info(f"Removed unused concept: {removed_concept.name}")

                # 개념 수 제한
                if len(self._concepts) > self.max_concepts:
                    # 안정성과 활성화 빈도가 낮은 개념부터 제거
                    sorted_concepts = sorted(
                        self._concepts.values(),
                        key=lambda c: c.stability * c.activation_count
                    )

                    remove_count = len(self._concepts) - self.max_concepts
                    for i in range(remove_count):
                        concept_to_remove = sorted_concepts[i]
                        del self._concepts[concept_to_remove.id]

                await asyncio.sleep(3600)  # 1시간마다 실행

            except Exception as e:
                self.logger.error(f"Error in concept maintenance: {e}")
                await asyncio.sleep(3600)

    async def _hierarchy_optimization(self) -> None:
        """계층 구조 최적화 백그라운드 프로세스"""
        while True:
            try:
                # 주기적으로 개념 간 관계 재평가
                await self._optimize_concept_hierarchy()

                await asyncio.sleep(7200)  # 2시간마다 실행

            except Exception as e:
                self.logger.error(f"Error in hierarchy optimization: {e}")
                await asyncio.sleep(7200)

    async def _optimize_concept_hierarchy(self) -> None:
        """개념 계층 구조 최적화"""
        try:
            # 모든 개념 쌍에 대해 관계 재평가
            concepts = list(self._concepts.values())

            for i, concept1 in enumerate(concepts):
                for concept2 in concepts[i+1:]:
                    # 기존 관계 확인
                    has_relation = (
                        concept2.id in concept1.parent_concepts or
                        concept2.id in concept1.child_concepts or
                        concept2.id in concept1.related_concepts
                    )

                    if not has_relation:
                        # 새로운 관계 가능성 확인
                        relation = await self._determine_concept_relation(concept1, concept2)
                        if relation and relation.get("confidence", 0) > 0.8:
                            await self._add_concept_relation(concept1, concept2, relation["type"])

        except Exception as e:
            self.logger.error(f"Error optimizing concept hierarchy: {e}")


async def create_concept_former(max_concepts: int = 5000) -> ConceptFormer:
    """ConceptFormer 인스턴스 생성 및 초기화"""
    former = ConceptFormer(max_concepts)
    await former.initialize()
    return former