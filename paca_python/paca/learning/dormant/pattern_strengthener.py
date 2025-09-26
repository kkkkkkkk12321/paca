"""
패턴 강화기 (Pattern Strengthener)

패턴 발견 및 강화를 담당합니다.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import logging
import math


class PatternType(Enum):
    """패턴 유형"""
    TEMPORAL = "temporal"        # 시간적 패턴
    CAUSAL = "causal"           # 인과 관계 패턴
    SIMILARITY = "similarity"    # 유사성 패턴
    SEQUENCE = "sequence"       # 순서 패턴
    FREQUENCY = "frequency"     # 빈도 패턴
    CORRELATION = "correlation" # 상관관계 패턴


class PatternStrength(Enum):
    """패턴 강도"""
    VERY_WEAK = "very_weak"     # 0.0 - 0.2
    WEAK = "weak"               # 0.2 - 0.4
    MEDIUM = "medium"           # 0.4 - 0.6
    STRONG = "strong"           # 0.6 - 0.8
    VERY_STRONG = "very_strong" # 0.8 - 1.0


@dataclass
class Pattern:
    """패턴 정보"""
    id: str
    pattern_type: PatternType
    strength: float
    confidence: float
    occurrences: int
    last_seen: datetime
    created_at: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    connections: List[str] = field(default_factory=list)

    def get_strength_category(self) -> PatternStrength:
        """강도 카테고리 반환"""
        if self.strength < 0.2:
            return PatternStrength.VERY_WEAK
        elif self.strength < 0.4:
            return PatternStrength.WEAK
        elif self.strength < 0.6:
            return PatternStrength.MEDIUM
        elif self.strength < 0.8:
            return PatternStrength.STRONG
        else:
            return PatternStrength.VERY_STRONG

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'pattern_type': self.pattern_type.value,
            'strength': self.strength,
            'confidence': self.confidence,
            'occurrences': self.occurrences,
            'last_seen': self.last_seen.isoformat(),
            'created_at': self.created_at.isoformat(),
            'strength_category': self.get_strength_category().value,
            'data': self.data,
            'connections': self.connections
        }


@dataclass
class StrengtheningResult:
    """강화 결과"""
    strengthened_count: int
    weakened_count: int
    new_patterns_discovered: int
    removed_patterns: int
    total_processed: int
    average_strength_improvement: float
    quality_metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'strengthened_count': self.strengthened_count,
            'weakened_count': self.weakened_count,
            'new_patterns_discovered': self.new_patterns_discovered,
            'removed_patterns': self.removed_patterns,
            'total_processed': self.total_processed,
            'average_strength_improvement': self.average_strength_improvement,
            'quality_metrics': self.quality_metrics
        }


class PatternStrengthener:
    """패턴 강화기 클래스"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # 설정
        self.min_occurrences_for_strengthening = 3
        self.strength_increment = 0.1
        self.confidence_threshold = 0.5
        self.max_patterns_per_type = 100

        # 패턴 저장소 (시뮬레이션)
        self.patterns: Dict[str, Pattern] = {}

        # 통계
        self.strengthening_stats = {
            'total_strengthenings': 0,
            'patterns_created': 0,
            'patterns_removed': 0,
            'average_strength': 0.0
        }

    async def strengthen_patterns(self, pattern_data: Dict[str, Any],
                                confidence_threshold: float) -> Dict[str, Any]:
        """패턴 강화 실행"""
        self.logger.info("패턴 강화 시작")

        try:
            # 시뮬레이션된 패턴 생성
            patterns = self._generate_simulation_patterns(pattern_data)

            # 패턴 강화 처리
            result = await self._process_pattern_strengthening(patterns, confidence_threshold)

            # 통계 업데이트
            self._update_strengthening_stats(result)

            self.logger.info(f"패턴 강화 완료: {result.strengthened_count}개 강화됨")
            return result.to_dict()

        except Exception as e:
            self.logger.error(f"패턴 강화 오류: {str(e)}")
            raise RuntimeError(f"패턴 강화 실패: {str(e)}")

    def _generate_simulation_patterns(self, pattern_data: Dict[str, Any]) -> List[Pattern]:
        """시뮬레이션 패턴 생성"""
        patterns = []

        # 패턴 강도 분포를 기반으로 시뮬레이션 패턴 생성
        strength_distribution = pattern_data.get('pattern_strength_distribution', {})

        pattern_types = list(PatternType)
        pattern_id = 0

        for strength_category, count in strength_distribution.items():
            # 강도 카테고리를 숫자로 변환
            strength_ranges = {
                'very_weak': (0.0, 0.2),
                'weak': (0.2, 0.4),
                'medium': (0.4, 0.6),
                'strong': (0.6, 0.8),
                'very_strong': (0.8, 1.0)
            }

            strength_range = strength_ranges.get(strength_category, (0.0, 1.0))

            for i in range(count):
                pattern_type = pattern_types[pattern_id % len(pattern_types)]
                strength = strength_range[0] + (i / count) * (strength_range[1] - strength_range[0])

                pattern = Pattern(
                    id=f"pattern_{pattern_id:06d}",
                    pattern_type=pattern_type,
                    strength=strength,
                    confidence=min(1.0, strength + 0.1),
                    occurrences=max(1, int(strength * 20)),
                    last_seen=datetime.now(),
                    created_at=datetime.now(),
                    data={
                        'description': f'{pattern_type.value} 패턴 #{pattern_id}',
                        'context': f'simulation_pattern_{strength_category}'
                    }
                )

                patterns.append(pattern)
                pattern_id += 1

        return patterns

    async def _process_pattern_strengthening(self, patterns: List[Pattern],
                                           confidence_threshold: float) -> StrengtheningResult:
        """패턴 강화 처리"""
        strengthened_count = 0
        weakened_count = 0
        new_patterns = 0
        removed_patterns = 0
        total_strength_change = 0.0

        for pattern in patterns:
            # 신뢰도 임계값 확인
            if pattern.confidence < confidence_threshold:
                continue

            # 강화 여부 결정
            if await self._should_strengthen_pattern(pattern):
                old_strength = pattern.strength
                pattern.strength = min(1.0, pattern.strength + self.strength_increment)
                strength_change = pattern.strength - old_strength
                total_strength_change += strength_change
                strengthened_count += 1

                # 패턴 저장소에 추가/업데이트
                self.patterns[pattern.id] = pattern

            elif await self._should_weaken_pattern(pattern):
                old_strength = pattern.strength
                pattern.strength = max(0.0, pattern.strength - self.strength_increment)
                strength_change = old_strength - pattern.strength
                total_strength_change -= strength_change
                weakened_count += 1

                # 너무 약해진 패턴은 제거
                if pattern.strength < 0.1:
                    if pattern.id in self.patterns:
                        del self.patterns[pattern.id]
                    removed_patterns += 1

            # 새로운 패턴 발견 시뮬레이션
            if pattern.strength > 0.8 and pattern.occurrences > 10:
                if await self._discover_related_pattern(pattern):
                    new_patterns += 1

        # 결과 계산
        average_improvement = total_strength_change / max(len(patterns), 1)

        quality_metrics = {
            'strengthening_efficiency': strengthened_count / max(len(patterns), 1),
            'pattern_stability': 1.0 - (removed_patterns / max(len(patterns), 1)),
            'discovery_rate': new_patterns / max(len(patterns), 1)
        }

        return StrengtheningResult(
            strengthened_count=strengthened_count,
            weakened_count=weakened_count,
            new_patterns_discovered=new_patterns,
            removed_patterns=removed_patterns,
            total_processed=len(patterns),
            average_strength_improvement=average_improvement,
            quality_metrics=quality_metrics
        )

    async def _should_strengthen_pattern(self, pattern: Pattern) -> bool:
        """패턴 강화 여부 판단"""
        # 발생 빈도가 충분한가?
        if pattern.occurrences < self.min_occurrences_for_strengthening:
            return False

        # 최근에 관찰되었는가?
        time_since_last_seen = (datetime.now() - pattern.last_seen).days
        if time_since_last_seen > 30:  # 30일 이상 관찰되지 않음
            return False

        # 현재 강도가 개선 가능한가?
        if pattern.strength >= 0.95:  # 이미 매우 강함
            return False

        # 시뮬레이션 지연
        await asyncio.sleep(0.001)

        return True

    async def _should_weaken_pattern(self, pattern: Pattern) -> bool:
        """패턴 약화 여부 판단"""
        # 오래 관찰되지 않았는가?
        time_since_last_seen = (datetime.now() - pattern.last_seen).days
        if time_since_last_seen > 60:  # 60일 이상 관찰되지 않음
            return True

        # 발생 빈도가 매우 낮은가?
        if pattern.occurrences == 1 and pattern.confidence < 0.3:
            return True

        # 시뮬레이션 지연
        await asyncio.sleep(0.001)

        return False

    async def _discover_related_pattern(self, strong_pattern: Pattern) -> bool:
        """관련 패턴 발견"""
        # 강한 패턴을 기반으로 새로운 패턴 발견 시뮬레이션
        discovery_probability = strong_pattern.strength * strong_pattern.confidence * 0.1

        # 시뮬레이션 지연
        await asyncio.sleep(0.002)

        # 확률적 발견
        import random
        return random.random() < discovery_probability

    def _update_strengthening_stats(self, result: StrengtheningResult):
        """강화 통계 업데이트"""
        self.strengthening_stats['total_strengthenings'] += result.strengthened_count
        self.strengthening_stats['patterns_created'] += result.new_patterns_discovered
        self.strengthening_stats['patterns_removed'] += result.removed_patterns

        # 평균 강도 계산
        if self.patterns:
            total_strength = sum(p.strength for p in self.patterns.values())
            self.strengthening_stats['average_strength'] = total_strength / len(self.patterns)

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """패턴 통계 조회"""
        if not self.patterns:
            return {
                'total_patterns': 0,
                'average_strength': 0.0,
                'pattern_distribution': {},
                'strengthening_stats': self.strengthening_stats.copy()
            }

        # 패턴 유형별 분포
        type_distribution = {}
        for pattern in self.patterns.values():
            pattern_type = pattern.pattern_type.value
            if pattern_type not in type_distribution:
                type_distribution[pattern_type] = 0
            type_distribution[pattern_type] += 1

        # 강도별 분포
        strength_distribution = {}
        for pattern in self.patterns.values():
            strength_category = pattern.get_strength_category().value
            if strength_category not in strength_distribution:
                strength_distribution[strength_category] = 0
            strength_distribution[strength_category] += 1

        return {
            'total_patterns': len(self.patterns),
            'average_strength': sum(p.strength for p in self.patterns.values()) / len(self.patterns),
            'average_confidence': sum(p.confidence for p in self.patterns.values()) / len(self.patterns),
            'type_distribution': type_distribution,
            'strength_distribution': strength_distribution,
            'strengthening_stats': self.strengthening_stats.copy()
        }

    async def analyze_pattern_trends(self) -> Dict[str, Any]:
        """패턴 트렌드 분석"""
        self.logger.info("패턴 트렌드 분석 시작")

        if not self.patterns:
            return {'error': '분석할 패턴이 없습니다.'}

        # 트렌드 분석 시뮬레이션
        await asyncio.sleep(0.5)

        trends = {
            'emerging_patterns': [],
            'declining_patterns': [],
            'stable_patterns': [],
            'pattern_correlations': {}
        }

        for pattern in self.patterns.values():
            # 패턴 트렌드 분류 (시뮬레이션)
            if pattern.strength > 0.7 and pattern.occurrences > 10:
                trends['emerging_patterns'].append(pattern.id)
            elif pattern.strength < 0.3 and pattern.occurrences < 5:
                trends['declining_patterns'].append(pattern.id)
            else:
                trends['stable_patterns'].append(pattern.id)

        self.logger.info("패턴 트렌드 분석 완료")
        return trends

    async def optimize_pattern_network(self) -> Dict[str, Any]:
        """패턴 네트워크 최적화"""
        self.logger.info("패턴 네트워크 최적화 시작")

        optimization_result = {
            'connections_optimized': 0,
            'redundant_patterns_merged': 0,
            'network_efficiency_improvement': 0.0
        }

        # 패턴 네트워크 최적화 시뮬레이션
        await asyncio.sleep(0.3)

        # 중복 패턴 병합
        merged_patterns = await self._merge_similar_patterns()
        optimization_result['redundant_patterns_merged'] = merged_patterns

        # 연결 최적화
        optimized_connections = await self._optimize_pattern_connections()
        optimization_result['connections_optimized'] = optimized_connections

        # 효율성 개선 계산
        optimization_result['network_efficiency_improvement'] = 0.15

        self.logger.info("패턴 네트워크 최적화 완료")
        return optimization_result

    async def _merge_similar_patterns(self) -> int:
        """유사한 패턴 병합"""
        # 유사한 패턴을 찾아 병합하는 시뮬레이션
        await asyncio.sleep(0.1)

        # 시뮬레이션된 병합 수
        return len(self.patterns) // 20  # 패턴의 5% 정도 병합

    async def _optimize_pattern_connections(self) -> int:
        """패턴 연결 최적화"""
        # 패턴 간 연결을 최적화하는 시뮬레이션
        await asyncio.sleep(0.1)

        # 시뮬레이션된 최적화된 연결 수
        return len(self.patterns) * 2  # 패턴당 평균 2개 연결