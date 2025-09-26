"""
Tactic Generator Module
전술/휴리스틱 자동 생성 시스템

성공/실패 패턴에서 자동으로 학습하여 전술과 휴리스틱을 생성하는 시스템입니다.
- 성공적인 상호작용에서 전술 추출
- 실패 사례에서 회피 휴리스틱 생성
- 전술 숙련도 관리 (learning → mastered)
- 자동 정제 및 개선
"""

import asyncio
import json
import time
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum
import statistics
import math
from collections import defaultdict, Counter

# 조건부 임포트: 패키지 실행시와 직접 실행시 모두 지원
try:
    from ..core.types.base import (
        ID, Timestamp, Result, Priority, current_timestamp, generate_id,
        create_success, create_failure
    )
except ImportError:
    from paca.core.types.base import (
        ID, Timestamp, Result, Priority, current_timestamp, generate_id,
        create_success, create_failure
    )


class TacticType(Enum):
    """전술 유형"""
    ANALYTICAL = "analytical"           # 분석적 접근
    CREATIVE = "creative"              # 창의적 접근
    SYSTEMATIC = "systematic"          # 체계적 접근
    INTUITIVE = "intuitive"           # 직관적 접근
    COLLABORATIVE = "collaborative"    # 협력적 접근
    DEFENSIVE = "defensive"           # 방어적 접근


class TacticStatus(Enum):
    """전술 상태"""
    LEARNING = "learning"      # 학습 중
    PRACTICING = "practicing"  # 연습 중
    COMPETENT = "competent"    # 숙련
    MASTERED = "mastered"      # 마스터


class HeuristicType(Enum):
    """휴리스틱 유형"""
    AVOIDANCE = "avoidance"           # 회피 규칙
    OPTIMIZATION = "optimization"     # 최적화 규칙
    FALLBACK = "fallback"            # 대안 규칙
    VALIDATION = "validation"        # 검증 규칙
    PRIORITIZATION = "prioritization" # 우선순위 규칙


class PatternConfidence(Enum):
    """패턴 신뢰도"""
    LOW = "low"        # 낮음 (1-3회 관찰)
    MEDIUM = "medium"  # 보통 (4-9회 관찰)
    HIGH = "high"      # 높음 (10-19회 관찰)
    VERY_HIGH = "very_high"  # 매우 높음 (20회+ 관찰)


@dataclass(frozen=True)
class InteractionPattern:
    """상호작용 패턴"""
    pattern_id: str
    description: str
    context_keywords: List[str]
    action_sequence: List[str]
    success_rate: float  # 성공률 (0.0-1.0)
    frequency: int       # 발생 빈도
    confidence: PatternConfidence
    discovered_at: Timestamp
    last_observed: Timestamp


@dataclass(frozen=True)
class Tactic:
    """전술"""
    tactic_id: str
    name: str
    description: str
    tactic_type: TacticType
    status: TacticStatus
    proficiency: float    # 숙련도 (0.0-1.0)
    usage_count: int      # 사용 횟수
    success_rate: float   # 성공률 (0.0-1.0)
    applicable_contexts: List[str]
    prerequisite_skills: List[str]
    effectiveness_score: float  # 효과성 점수 (0.0-1.0)
    created_at: Timestamp
    last_used: Optional[Timestamp] = None
    mastery_progress: float = 0.0  # 숙련 진행도 (0.0-1.0)


@dataclass(frozen=True)
class Heuristic:
    """휴리스틱 (경험 규칙)"""
    heuristic_id: str
    name: str
    description: str
    heuristic_type: HeuristicType
    condition: str        # 적용 조건
    action: str          # 수행 동작
    confidence: float    # 신뢰도 (0.0-1.0)
    validation_count: int  # 검증 횟수
    effectiveness: float  # 효과성 (0.0-1.0)
    applicable_domains: List[str]
    created_at: Timestamp
    last_validated: Optional[Timestamp] = None


@dataclass(frozen=True)
class TacticUsageRecord:
    """전술 사용 기록"""
    record_id: str
    tactic_id: str
    context: str
    success: bool
    performance_score: float  # 성과 점수 (0.0-1.0)
    execution_time_ms: int
    feedback: Optional[str] = None
    timestamp: Timestamp = None

    def __post_init__(self):
        if self.timestamp is None:
            object.__setattr__(self, 'timestamp', current_timestamp())


@dataclass(frozen=True)
class LearningSnapshot:
    """학습 스냅샷"""
    snapshot_id: str
    total_tactics: int
    mastered_tactics: int
    active_heuristics: int
    overall_proficiency: float
    learning_velocity: float  # 학습 속도
    timestamp: Timestamp


class TacticGenerator:
    """
    전술/휴리스틱 자동 생성 시스템

    상호작용 패턴을 분석하여 효과적인 전술과 휴리스틱을 자동으로 생성합니다.
    """

    # 임계값 상수
    MIN_PATTERN_FREQUENCY = 3      # 최소 패턴 발생 빈도
    HIGH_SUCCESS_THRESHOLD = 0.8   # 높은 성공률 임계값
    MASTERY_THRESHOLD = 0.9        # 숙련도 임계값
    USAGE_COUNT_FOR_MASTERY = 20   # 마스터리를 위한 최소 사용 횟수

    def __init__(self):
        """전술 생성기 초기화"""
        self._tactics: Dict[str, Tactic] = {}
        self._heuristics: Dict[str, Heuristic] = {}
        self._usage_records: List[TacticUsageRecord] = []
        self._interaction_patterns: Dict[str, InteractionPattern] = {}
        self._learning_snapshots: List[LearningSnapshot] = []

    async def extract_successful_patterns(self, interactions: List[Dict[str, Any]]) -> Result[List[Tactic]]:
        """
        성공적인 상호작용에서 전술 추출

        Args:
            interactions: 상호작용 데이터 목록

        Returns:
            추출된 전술 목록
        """
        try:
            if not interactions:
                return create_success([])

            # 1. 성공적인 상호작용 필터링
            successful_interactions = [
                interaction for interaction in interactions
                if interaction.get('success', False) and interaction.get('success_score', 0) > 0.7
            ]

            if not successful_interactions:
                return create_success([])

            # 2. 패턴 분석
            patterns = await self._analyze_interaction_patterns(successful_interactions, success_focus=True)

            # 3. 패턴에서 전술 생성
            new_tactics = []
            for pattern in patterns:
                if pattern.success_rate >= self.HIGH_SUCCESS_THRESHOLD:
                    tactics = await self._generate_tactics_from_pattern(pattern)
                    new_tactics.extend(tactics)

            # 4. 기존 전술과 중복 제거 및 통합
            refined_tactics = await self._refine_and_merge_tactics(new_tactics)

            # 5. 전술 저장
            for tactic in refined_tactics:
                self._tactics[tactic.tactic_id] = tactic

            return create_success(refined_tactics)

        except Exception as e:
            return create_failure(e)

    async def generate_heuristics(self, failures: List[Dict[str, Any]]) -> Result[List[Heuristic]]:
        """
        실패 사례에서 회피 휴리스틱 생성

        Args:
            failures: 실패 사례 데이터 목록

        Returns:
            생성된 휴리스틱 목록
        """
        try:
            if not failures:
                return create_success([])

            # 1. 실패 패턴 분석
            failure_patterns = await self._analyze_failure_patterns(failures)

            # 2. 회피 휴리스틱 생성
            avoidance_heuristics = []
            for pattern in failure_patterns:
                heuristics = await self._generate_avoidance_heuristics(pattern)
                avoidance_heuristics.extend(heuristics)

            # 3. 최적화 휴리스틱 생성 (실패에서 배운 개선점)
            optimization_heuristics = await self._generate_optimization_heuristics(failure_patterns)

            # 4. 모든 휴리스틱 통합
            all_heuristics = avoidance_heuristics + optimization_heuristics

            # 5. 휴리스틱 검증 및 정제
            validated_heuristics = await self._validate_heuristics(all_heuristics)

            # 6. 휴리스틱 저장
            for heuristic in validated_heuristics:
                self._heuristics[heuristic.heuristic_id] = heuristic

            return create_success(validated_heuristics)

        except Exception as e:
            return create_failure(e)

    async def update_mastery_levels(self, tactic_usage: Dict[str, Dict[str, Any]]) -> Result[List[str]]:
        """
        전술 숙련도 업데이트 (learning → mastered)

        Args:
            tactic_usage: 전술 사용 데이터

        Returns:
            숙련도가 업데이트된 전술 ID 목록
        """
        try:
            updated_tactics = []

            for tactic_id, usage_data in tactic_usage.items():
                if tactic_id in self._tactics:
                    current_tactic = self._tactics[tactic_id]
                    updated_tactic = await self._update_tactic_mastery(current_tactic, usage_data)

                    if updated_tactic.status != current_tactic.status:
                        self._tactics[tactic_id] = updated_tactic
                        updated_tactics.append(tactic_id)

            # 학습 스냅샷 생성
            await self._create_learning_snapshot()

            return create_success(updated_tactics)

        except Exception as e:
            return create_failure(e)

    async def get_recommended_tactics(self, context: str, difficulty: float = 0.5) -> Result[List[Tactic]]:
        """
        컨텍스트에 맞는 추천 전술 조회

        Args:
            context: 현재 컨텍스트
            difficulty: 문제 난이도 (0.0-1.0)

        Returns:
            추천 전술 목록
        """
        try:
            context_keywords = self._extract_keywords(context)
            suitable_tactics = []

            for tactic in self._tactics.values():
                # 컨텍스트 적합성 계산
                context_match = self._calculate_context_match(tactic, context_keywords)

                # 숙련도 적합성 (너무 어렵거나 쉬운 전술 제외)
                proficiency_match = self._calculate_proficiency_match(tactic, difficulty)

                # 최근 성과 고려
                recent_performance = await self._get_recent_performance(tactic.tactic_id)

                # 종합 점수 계산
                overall_score = (
                    context_match * 0.4 +
                    proficiency_match * 0.3 +
                    recent_performance * 0.2 +
                    tactic.effectiveness_score * 0.1
                )

                if overall_score > 0.6:  # 임계값 이상만 추천
                    suitable_tactics.append((tactic, overall_score))

            # 점수순으로 정렬하여 상위 5개 반환
            suitable_tactics.sort(key=lambda x: x[1], reverse=True)
            recommended = [tactic for tactic, _ in suitable_tactics[:5]]

            return create_success(recommended)

        except Exception as e:
            return create_failure(e)

    async def optimize_tactics(self) -> Result[Dict[str, Any]]:
        """
        전술 자동 정제 및 개선

        Returns:
            최적화 결과 요약
        """
        try:
            optimization_results = {
                "removed_tactics": [],
                "merged_tactics": [],
                "improved_tactics": [],
                "new_combinations": []
            }

            # 1. 비효율적인 전술 제거
            removed = await self._remove_ineffective_tactics()
            optimization_results["removed_tactics"] = removed

            # 2. 유사한 전술 병합
            merged = await self._merge_similar_tactics()
            optimization_results["merged_tactics"] = merged

            # 3. 전술 조합 발견
            combinations = await self._discover_tactic_combinations()
            optimization_results["new_combinations"] = combinations

            # 4. 효과성 재계산
            improved = await self._recalculate_effectiveness()
            optimization_results["improved_tactics"] = improved

            return create_success(optimization_results)

        except Exception as e:
            return create_failure(e)

    def get_learning_statistics(self) -> Dict[str, Any]:
        """
        학습 통계 조회

        Returns:
            학습 통계 데이터
        """
        total_tactics = len(self._tactics)
        mastered_tactics = len([t for t in self._tactics.values() if t.status == TacticStatus.MASTERED])
        active_heuristics = len([h for h in self._heuristics.values() if h.effectiveness > 0.7])

        avg_proficiency = statistics.mean([t.proficiency for t in self._tactics.values()]) if self._tactics else 0.0
        avg_success_rate = statistics.mean([t.success_rate for t in self._tactics.values()]) if self._tactics else 0.0

        return {
            "total_tactics": total_tactics,
            "mastered_tactics": mastered_tactics,
            "mastery_rate": mastered_tactics / total_tactics if total_tactics > 0 else 0.0,
            "active_heuristics": active_heuristics,
            "average_proficiency": avg_proficiency,
            "average_success_rate": avg_success_rate,
            "total_usage_records": len(self._usage_records),
            "learning_snapshots": len(self._learning_snapshots)
        }

    def get_tactic_details(self, tactic_id: str) -> Optional[Tactic]:
        """특정 전술 상세 정보 조회"""
        return self._tactics.get(tactic_id)

    def get_all_tactics(self) -> List[Tactic]:
        """모든 전술 목록 조회"""
        return list(self._tactics.values())

    def get_all_heuristics(self) -> List[Heuristic]:
        """모든 휴리스틱 목록 조회"""
        return list(self._heuristics.values())

    # === 내부 메서드들 ===

    async def _analyze_interaction_patterns(self,
                                          interactions: List[Dict[str, Any]],
                                          success_focus: bool = True) -> List[InteractionPattern]:
        """상호작용 패턴 분석"""
        patterns = []
        pattern_clusters = defaultdict(list)

        # 상호작용을 컨텍스트별로 클러스터링
        for interaction in interactions:
            context = interaction.get('context', '')
            actions = interaction.get('actions', [])
            success = interaction.get('success', False)

            # 키워드 추출
            keywords = self._extract_keywords(context)
            key = tuple(sorted(keywords[:3]))  # 상위 3개 키워드로 클러스터 키 생성

            pattern_clusters[key].append({
                'context': context,
                'actions': actions,
                'success': success,
                'timestamp': interaction.get('timestamp', current_timestamp())
            })

        # 각 클러스터에서 패턴 추출
        for cluster_key, cluster_interactions in pattern_clusters.items():
            if len(cluster_interactions) >= self.MIN_PATTERN_FREQUENCY:
                pattern = await self._extract_pattern_from_cluster(cluster_key, cluster_interactions)
                if pattern:
                    patterns.append(pattern)

        return patterns

    async def _extract_pattern_from_cluster(self,
                                          cluster_key: Tuple[str, ...],
                                          interactions: List[Dict[str, Any]]) -> Optional[InteractionPattern]:
        """클러스터에서 패턴 추출"""
        if not interactions:
            return None

        # 성공률 계산
        successful_count = sum(1 for i in interactions if i['success'])
        success_rate = successful_count / len(interactions)

        # 공통 액션 시퀀스 추출
        action_sequences = [i['actions'] for i in interactions if i['actions']]
        common_actions = self._find_common_action_sequence(action_sequences)

        # 신뢰도 결정
        frequency = len(interactions)
        if frequency >= 20:
            confidence = PatternConfidence.VERY_HIGH
        elif frequency >= 10:
            confidence = PatternConfidence.HIGH
        elif frequency >= 4:
            confidence = PatternConfidence.MEDIUM
        else:
            confidence = PatternConfidence.LOW

        # 패턴 생성
        pattern_id = generate_id("pattern_")
        description = f"패턴: {', '.join(cluster_key)} 컨텍스트에서의 행동"

        return InteractionPattern(
            pattern_id=pattern_id,
            description=description,
            context_keywords=list(cluster_key),
            action_sequence=common_actions,
            success_rate=success_rate,
            frequency=frequency,
            confidence=confidence,
            discovered_at=current_timestamp(),
            last_observed=max(i['timestamp'] for i in interactions)
        )

    def _find_common_action_sequence(self, action_sequences: List[List[str]]) -> List[str]:
        """공통 액션 시퀀스 찾기"""
        if not action_sequences:
            return []

        # 모든 액션의 빈도 계산
        action_counter = Counter()
        for sequence in action_sequences:
            action_counter.update(sequence)

        # 가장 빈번한 액션들 추출 (50% 이상 등장)
        min_frequency = len(action_sequences) * 0.5
        common_actions = [
            action for action, count in action_counter.items()
            if count >= min_frequency
        ]

        # 원래 시퀀스에서의 평균 순서로 정렬
        action_positions = defaultdict(list)
        for sequence in action_sequences:
            for i, action in enumerate(sequence):
                if action in common_actions:
                    action_positions[action].append(i)

        # 평균 위치로 정렬
        common_actions.sort(key=lambda a: statistics.mean(action_positions[a]))

        return common_actions

    async def _generate_tactics_from_pattern(self, pattern: InteractionPattern) -> List[Tactic]:
        """패턴에서 전술 생성"""
        tactics = []

        # 전술 유형 결정
        tactic_type = self._determine_tactic_type(pattern)

        # 기본 전술 생성
        tactic_id = generate_id("tactic_")
        tactic_name = f"{tactic_type.value.title()} 접근법"
        description = f"{pattern.description}에 기반한 {tactic_type.value} 전술"

        tactic = Tactic(
            tactic_id=tactic_id,
            name=tactic_name,
            description=description,
            tactic_type=tactic_type,
            status=TacticStatus.LEARNING,
            proficiency=0.3,  # 초기 숙련도
            usage_count=pattern.frequency,
            success_rate=pattern.success_rate,
            applicable_contexts=pattern.context_keywords,
            prerequisite_skills=[],
            effectiveness_score=pattern.success_rate * (1.0 if pattern.confidence == PatternConfidence.VERY_HIGH else 0.8),
            created_at=current_timestamp()
        )

        tactics.append(tactic)

        return tactics

    def _determine_tactic_type(self, pattern: InteractionPattern) -> TacticType:
        """패턴 특성에 따른 전술 유형 결정"""
        keywords = pattern.context_keywords
        actions = pattern.action_sequence

        # 키워드 기반 분류
        if any(word in keywords for word in ['analysis', 'analyze', 'data', 'logic']):
            return TacticType.ANALYTICAL
        elif any(word in keywords for word in ['create', 'design', 'innovative', 'creative']):
            return TacticType.CREATIVE
        elif any(word in keywords for word in ['step', 'systematic', 'process', 'method']):
            return TacticType.SYSTEMATIC
        elif any(word in keywords for word in ['quick', 'intuitive', 'feeling', 'guess']):
            return TacticType.INTUITIVE
        elif any(word in keywords for word in ['team', 'collaborate', 'group', 'together']):
            return TacticType.COLLABORATIVE
        else:
            return TacticType.DEFENSIVE

    async def _analyze_failure_patterns(self, failures: List[Dict[str, Any]]) -> List[InteractionPattern]:
        """실패 패턴 분석"""
        # 실패 상호작용에 대해서도 패턴 분석 수행
        return await self._analyze_interaction_patterns(failures, success_focus=False)

    async def _generate_avoidance_heuristics(self, failure_pattern: InteractionPattern) -> List[Heuristic]:
        """실패 패턴에서 회피 휴리스틱 생성"""
        heuristics = []

        # 회피 조건 생성
        avoid_conditions = []
        for keyword in failure_pattern.context_keywords:
            avoid_conditions.append(f"컨텍스트에 '{keyword}'가 포함된 경우")

        for action in failure_pattern.action_sequence:
            avoid_conditions.append(f"'{action}' 액션을 수행하려는 경우")

        # 각 조건에 대해 휴리스틱 생성
        for i, condition in enumerate(avoid_conditions[:3]):  # 최대 3개
            heuristic_id = generate_id("heuristic_avoid_")
            name = f"패턴 회피 #{i+1}"
            description = f"실패 패턴을 회피하기 위한 휴리스틱"

            # 대안 액션 제안
            alternative_action = self._suggest_alternative_action(failure_pattern)

            heuristic = Heuristic(
                heuristic_id=heuristic_id,
                name=name,
                description=description,
                heuristic_type=HeuristicType.AVOIDANCE,
                condition=condition,
                action=alternative_action,
                confidence=1.0 - failure_pattern.success_rate,  # 실패율 기반 신뢰도
                validation_count=failure_pattern.frequency,
                effectiveness=0.7,  # 초기 효과성
                applicable_domains=failure_pattern.context_keywords,
                created_at=current_timestamp()
            )
            heuristics.append(heuristic)

        return heuristics

    async def _generate_optimization_heuristics(self, failure_patterns: List[InteractionPattern]) -> List[Heuristic]:
        """실패 패턴에서 최적화 휴리스틱 생성"""
        heuristics = []

        # 공통 실패 요인 분석
        common_failure_keywords = self._find_common_failure_keywords(failure_patterns)

        for keyword in common_failure_keywords[:3]:  # 상위 3개
            heuristic_id = generate_id("heuristic_opt_")
            name = f"'{keyword}' 최적화"
            description = f"'{keyword}' 관련 상황에서의 최적화 규칙"

            condition = f"'{keyword}' 관련 작업 수행시"
            action = f"추가 검증 및 신중한 접근 수행"

            heuristic = Heuristic(
                heuristic_id=heuristic_id,
                name=name,
                description=description,
                heuristic_type=HeuristicType.OPTIMIZATION,
                condition=condition,
                action=action,
                confidence=0.8,
                validation_count=len(failure_patterns),
                effectiveness=0.6,
                applicable_domains=[keyword],
                created_at=current_timestamp()
            )
            heuristics.append(heuristic)

        return heuristics

    def _suggest_alternative_action(self, failure_pattern: InteractionPattern) -> str:
        """실패 패턴에 대한 대안 액션 제안"""
        # 실패한 액션의 반대 또는 대안 제안
        alternatives = {
            "rush": "신중하게 검토",
            "assume": "명시적으로 확인",
            "skip": "단계별로 수행",
            "ignore": "면밀히 검토",
            "guess": "정확한 정보 수집"
        }

        for action in failure_pattern.action_sequence:
            for fail_action, alternative in alternatives.items():
                if fail_action in action.lower():
                    return f"{alternative} 수행"

        return "더 신중한 접근 방법 선택"

    def _find_common_failure_keywords(self, failure_patterns: List[InteractionPattern]) -> List[str]:
        """실패 패턴에서 공통 키워드 찾기"""
        keyword_counter = Counter()

        for pattern in failure_patterns:
            # 실패율로 가중치 부여
            weight = 1.0 - pattern.success_rate
            for keyword in pattern.context_keywords:
                keyword_counter[keyword] += weight

        # 가중치 순으로 정렬하여 반환
        return [keyword for keyword, _ in keyword_counter.most_common()]

    async def _validate_heuristics(self, heuristics: List[Heuristic]) -> List[Heuristic]:
        """휴리스틱 검증 및 정제"""
        validated = []

        for heuristic in heuristics:
            # 기본 검증 (중복 제거, 유효성 검사 등)
            if self._is_valid_heuristic(heuristic):
                validated.append(heuristic)

        return validated

    def _is_valid_heuristic(self, heuristic: Heuristic) -> bool:
        """휴리스틱 유효성 검사"""
        # 기본적인 유효성 검사
        if not heuristic.condition or not heuristic.action:
            return False

        # 중복 검사
        for existing_heuristic in self._heuristics.values():
            if (existing_heuristic.condition == heuristic.condition and
                existing_heuristic.action == heuristic.action):
                return False

        return True

    async def _update_tactic_mastery(self, tactic: Tactic, usage_data: Dict[str, Any]) -> Tactic:
        """전술 숙련도 업데이트"""
        new_usage_count = usage_data.get('usage_count', tactic.usage_count)
        new_success_rate = usage_data.get('success_rate', tactic.success_rate)
        new_proficiency = usage_data.get('proficiency', tactic.proficiency)

        # 숙련도 진행도 계산
        progress_factors = [
            min(new_usage_count / self.USAGE_COUNT_FOR_MASTERY, 1.0),  # 사용 횟수 기반
            new_success_rate,  # 성공률 기반
            new_proficiency   # 기본 숙련도 기반
        ]
        mastery_progress = statistics.mean(progress_factors)

        # 상태 결정
        if mastery_progress >= self.MASTERY_THRESHOLD and new_usage_count >= self.USAGE_COUNT_FOR_MASTERY:
            new_status = TacticStatus.MASTERED
        elif mastery_progress >= 0.7 and new_usage_count >= 10:
            new_status = TacticStatus.COMPETENT
        elif mastery_progress >= 0.4 and new_usage_count >= 5:
            new_status = TacticStatus.PRACTICING
        else:
            new_status = TacticStatus.LEARNING

        # 효과성 점수 재계산
        effectiveness = (new_success_rate * 0.6 + new_proficiency * 0.4) * (
            1.0 if new_status == TacticStatus.MASTERED else 0.8
        )

        # 새로운 전술 객체 생성
        return Tactic(
            tactic_id=tactic.tactic_id,
            name=tactic.name,
            description=tactic.description,
            tactic_type=tactic.tactic_type,
            status=new_status,
            proficiency=new_proficiency,
            usage_count=new_usage_count,
            success_rate=new_success_rate,
            applicable_contexts=tactic.applicable_contexts,
            prerequisite_skills=tactic.prerequisite_skills,
            effectiveness_score=effectiveness,
            created_at=tactic.created_at,
            last_used=current_timestamp(),
            mastery_progress=mastery_progress
        )

    async def _refine_and_merge_tactics(self, new_tactics: List[Tactic]) -> List[Tactic]:
        """새로운 전술들을 정제하고 기존 전술과 병합"""
        refined = []

        for new_tactic in new_tactics:
            # 기존 전술과 유사성 검사
            similar_tactic = self._find_similar_tactic(new_tactic)

            if similar_tactic:
                # 기존 전술과 통합
                merged = await self._merge_tactics(similar_tactic, new_tactic)
                self._tactics[similar_tactic.tactic_id] = merged
            else:
                # 새로운 전술로 추가
                refined.append(new_tactic)

        return refined

    def _find_similar_tactic(self, new_tactic: Tactic) -> Optional[Tactic]:
        """유사한 기존 전술 찾기"""
        for existing_tactic in self._tactics.values():
            # 유형과 컨텍스트가 유사한지 확인
            if existing_tactic.tactic_type == new_tactic.tactic_type:
                # 컨텍스트 키워드 유사도 계산
                common_contexts = set(existing_tactic.applicable_contexts) & set(new_tactic.applicable_contexts)
                if len(common_contexts) >= 2:  # 2개 이상 공통 컨텍스트
                    return existing_tactic

        return None

    async def _merge_tactics(self, existing: Tactic, new: Tactic) -> Tactic:
        """두 전술 병합"""
        # 가중 평균으로 메트릭 계산
        total_usage = existing.usage_count + new.usage_count
        if total_usage > 0:
            weighted_success_rate = (
                (existing.success_rate * existing.usage_count + new.success_rate * new.usage_count) /
                total_usage
            )
            weighted_proficiency = (
                (existing.proficiency * existing.usage_count + new.proficiency * new.usage_count) /
                total_usage
            )
        else:
            weighted_success_rate = max(existing.success_rate, new.success_rate)
            weighted_proficiency = max(existing.proficiency, new.proficiency)

        # 컨텍스트 통합
        merged_contexts = list(set(existing.applicable_contexts + new.applicable_contexts))

        # 병합된 전술 생성
        return Tactic(
            tactic_id=existing.tactic_id,
            name=existing.name,
            description=f"{existing.description} (개선됨)",
            tactic_type=existing.tactic_type,
            status=existing.status,
            proficiency=weighted_proficiency,
            usage_count=total_usage,
            success_rate=weighted_success_rate,
            applicable_contexts=merged_contexts,
            prerequisite_skills=existing.prerequisite_skills,
            effectiveness_score=max(existing.effectiveness_score, new.effectiveness_score),
            created_at=existing.created_at,
            last_used=current_timestamp(),
            mastery_progress=existing.mastery_progress
        )

    def _extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 키워드 추출"""
        if not text:
            return []

        # 간단한 키워드 추출 (실제 구현에서는 더 정교한 NLP 사용)
        text = text.lower()
        # 불용어 제거 및 키워드 추출
        words = re.findall(r'\b[a-zA-Z가-힣]{3,}\b', text)
        stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word not in stopwords]
        return keywords[:10]  # 상위 10개만 반환

    def _calculate_context_match(self, tactic: Tactic, context_keywords: List[str]) -> float:
        """전술과 컨텍스트의 적합성 계산"""
        if not context_keywords or not tactic.applicable_contexts:
            return 0.0

        common_keywords = set(tactic.applicable_contexts) & set(context_keywords)
        return len(common_keywords) / len(set(tactic.applicable_contexts) | set(context_keywords))

    def _calculate_proficiency_match(self, tactic: Tactic, difficulty: float) -> float:
        """전술 숙련도와 문제 난이도의 적합성 계산"""
        # 숙련도가 난이도와 비슷할 때 높은 점수
        proficiency_gap = abs(tactic.proficiency - difficulty)
        return max(0.0, 1.0 - proficiency_gap * 2)

    async def _get_recent_performance(self, tactic_id: str) -> float:
        """전술의 최근 성과 조회"""
        recent_records = [
            record for record in self._usage_records
            if record.tactic_id == tactic_id and
            current_timestamp() - record.timestamp < 86400 * 7  # 최근 7일
        ]

        if not recent_records:
            return 0.5  # 기본값

        return statistics.mean(record.performance_score for record in recent_records)

    async def _create_learning_snapshot(self) -> None:
        """학습 스냅샷 생성"""
        total_tactics = len(self._tactics)
        mastered_tactics = len([t for t in self._tactics.values() if t.status == TacticStatus.MASTERED])
        active_heuristics = len([h for h in self._heuristics.values() if h.effectiveness > 0.7])

        overall_proficiency = statistics.mean([t.proficiency for t in self._tactics.values()]) if self._tactics else 0.0

        # 학습 속도 계산 (최근 스냅샷과 비교)
        learning_velocity = 0.0
        if len(self._learning_snapshots) > 0:
            last_snapshot = self._learning_snapshots[-1]
            time_diff = current_timestamp() - last_snapshot.timestamp
            proficiency_diff = overall_proficiency - last_snapshot.overall_proficiency
            if time_diff > 0:
                learning_velocity = proficiency_diff / (time_diff / 86400)  # 일당 학습 속도

        snapshot = LearningSnapshot(
            snapshot_id=generate_id("snapshot_"),
            total_tactics=total_tactics,
            mastered_tactics=mastered_tactics,
            active_heuristics=active_heuristics,
            overall_proficiency=overall_proficiency,
            learning_velocity=learning_velocity,
            timestamp=current_timestamp()
        )

        self._learning_snapshots.append(snapshot)

        # 최대 100개 스냅샷 유지
        if len(self._learning_snapshots) > 100:
            self._learning_snapshots = self._learning_snapshots[-100:]

    async def _remove_ineffective_tactics(self) -> List[str]:
        """비효율적인 전술 제거"""
        removed_tactics = []

        tactics_to_remove = [
            tactic_id for tactic_id, tactic in self._tactics.items()
            if tactic.effectiveness_score < 0.3 and tactic.usage_count > 10
        ]

        for tactic_id in tactics_to_remove:
            del self._tactics[tactic_id]
            removed_tactics.append(tactic_id)

        return removed_tactics

    async def _merge_similar_tactics(self) -> List[str]:
        """유사한 전술들 병합"""
        merged_tactics = []
        # 구현 생략 (복잡한 유사도 계산 및 병합 로직)
        return merged_tactics

    async def _discover_tactic_combinations(self) -> List[Dict[str, Any]]:
        """전술 조합 발견"""
        combinations = []
        # 구현 생략 (전술 조합 패턴 분석)
        return combinations

    async def _recalculate_effectiveness(self) -> List[str]:
        """효과성 재계산"""
        improved_tactics = []

        for tactic_id, tactic in self._tactics.items():
            # 최근 성과 기반으로 효과성 재계산
            recent_performance = await self._get_recent_performance(tactic_id)
            new_effectiveness = (tactic.success_rate * 0.6 + recent_performance * 0.4)

            if abs(new_effectiveness - tactic.effectiveness_score) > 0.1:
                # 효과성이 크게 변했다면 업데이트
                updated_tactic = Tactic(
                    tactic_id=tactic.tactic_id,
                    name=tactic.name,
                    description=tactic.description,
                    tactic_type=tactic.tactic_type,
                    status=tactic.status,
                    proficiency=tactic.proficiency,
                    usage_count=tactic.usage_count,
                    success_rate=tactic.success_rate,
                    applicable_contexts=tactic.applicable_contexts,
                    prerequisite_skills=tactic.prerequisite_skills,
                    effectiveness_score=new_effectiveness,
                    created_at=tactic.created_at,
                    last_used=tactic.last_used,
                    mastery_progress=tactic.mastery_progress
                )
                self._tactics[tactic_id] = updated_tactic
                improved_tactics.append(tactic_id)

        return improved_tactics


# Helper functions for testing
def create_sample_interaction_data() -> List[Dict[str, Any]]:
    """샘플 상호작용 데이터 생성 (테스트용)"""
    return [
        {
            'context': 'analytical problem solving task',
            'actions': ['analyze', 'break_down', 'systematic_approach'],
            'success': True,
            'success_score': 0.85,
            'timestamp': current_timestamp() - 3600
        },
        {
            'context': 'creative design challenge',
            'actions': ['brainstorm', 'iterate', 'refine'],
            'success': True,
            'success_score': 0.90,
            'timestamp': current_timestamp() - 1800
        },
        {
            'context': 'complex reasoning task',
            'actions': ['logical_analysis', 'step_by_step'],
            'success': False,
            'success_score': 0.40,
            'timestamp': current_timestamp() - 900
        }
    ]


def create_sample_tactic() -> Tactic:
    """샘플 전술 생성 (테스트용)"""
    return Tactic(
        tactic_id=generate_id("tactic_"),
        name="분석적 문제 해결",
        description="복잡한 문제를 체계적으로 분석하여 해결하는 전술",
        tactic_type=TacticType.ANALYTICAL,
        status=TacticStatus.PRACTICING,
        proficiency=0.75,
        usage_count=15,
        success_rate=0.85,
        applicable_contexts=['analysis', 'problem_solving', 'systematic'],
        prerequisite_skills=['logical_thinking', 'pattern_recognition'],
        effectiveness_score=0.82,
        created_at=current_timestamp() - 86400 * 7,
        last_used=current_timestamp() - 3600,
        mastery_progress=0.7
    )