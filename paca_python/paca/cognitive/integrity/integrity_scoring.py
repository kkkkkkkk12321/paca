"""
Intellectual Integrity Scoring (IIS) System
지적 무결성 점수 시스템 - Phase 2.3 구현

PACA의 지적 무결성을 정량적으로 측정하고 보상/패널티 시스템을 제공합니다.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
import json
import statistics

from ...core.types import Result, create_success, create_failure
from ...core.utils.logger import PacaLogger


class IntegrityDimension(Enum):
    """무결성 차원"""
    HONESTY = auto()           # 정직성
    ACCURACY = auto()          # 정확성
    TRANSPARENCY = auto()      # 투명성
    CONSISTENCY = auto()       # 일관성
    HUMILITY = auto()         # 겸손함
    VERIFICATION = auto()      # 검증 행동


class BehaviorType(Enum):
    """행동 유형"""
    TRUTH_SEEKING = auto()     # 진실 탐구
    ERROR_CORRECTION = auto()  # 오류 수정
    UNCERTAINTY_ADMISSION = auto()  # 불확실성 인정
    SOURCE_CITING = auto()     # 출처 인용
    BIAS_RECOGNITION = auto()  # 편향 인식
    KNOWLEDGE_UPDATE = auto()  # 지식 업데이트

    # 부정적 행동
    MISINFORMATION = auto()    # 잘못된 정보 제공
    OVERCONFIDENCE = auto()    # 과도한 확신
    SOURCE_OMISSION = auto()   # 출처 누락
    BIAS_REINFORCEMENT = auto()  # 편향 강화
    ERROR_PERSISTENCE = auto() # 오류 지속


@dataclass
class IntegrityAction:
    """무결성 행동 기록"""
    action_id: str
    behavior_type: BehaviorType
    dimension: IntegrityDimension
    score_impact: float  # -1.0 to +1.0
    confidence: float    # 행동 신뢰도
    context: Dict[str, Any]
    evidence: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    auto_detected: bool = True


@dataclass
class IntegrityMetrics:
    """무결성 메트릭"""
    overall_score: float  # 0.0-100.0
    dimension_scores: Dict[IntegrityDimension, float]
    trend: str  # "improving", "stable", "declining"
    recent_actions: int
    positive_actions: int
    negative_actions: int
    consistency_rating: float
    reliability_score: float


@dataclass
class IntegrityReward:
    """무결성 보상"""
    reward_id: str
    reward_type: str  # "score_boost", "recognition", "privilege"
    value: float
    reason: str
    criteria_met: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class IntegrityPenalty:
    """무결성 패널티"""
    penalty_id: str
    penalty_type: str  # "score_reduction", "warning", "restriction"
    value: float
    reason: str
    violations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class IntegrityScoring:
    """
    지적 무결성 점수 시스템 - Phase 2.3 메인 클래스

    정직성 평가 메트릭, 검증 행동 보상, 거짓말 탐지 및 패널티, 신뢰도 점수 관리
    """

    def __init__(self):
        self.logger = PacaLogger("IntegrityScoring")

        # 점수 시스템 초기화
        self.base_score = 50.0  # 기준 점수
        self.max_score = 100.0
        self.min_score = 0.0

        # 행동 기록
        self.action_history: List[IntegrityAction] = []
        self.rewards: List[IntegrityReward] = []
        self.penalties: List[IntegrityPenalty] = []

        # 현재 점수 상태
        self.current_metrics = IntegrityMetrics(
            overall_score=self.base_score,
            dimension_scores={dim: self.base_score for dim in IntegrityDimension},
            trend="stable",
            recent_actions=0,
            positive_actions=0,
            negative_actions=0,
            consistency_rating=0.5,
            reliability_score=0.5
        )

        # 점수 계산 가중치
        self.dimension_weights = {
            IntegrityDimension.HONESTY: 0.25,
            IntegrityDimension.ACCURACY: 0.25,
            IntegrityDimension.TRANSPARENCY: 0.15,
            IntegrityDimension.CONSISTENCY: 0.15,
            IntegrityDimension.HUMILITY: 0.1,
            IntegrityDimension.VERIFICATION: 0.1
        }

        # 행동별 점수 영향
        self.behavior_impacts = self._initialize_behavior_impacts()

        # 보상/패널티 기준
        self.reward_thresholds = self._initialize_reward_thresholds()
        self.penalty_thresholds = self._initialize_penalty_thresholds()

        self.logger.info("IntegrityScoring system initialized")

    def _initialize_behavior_impacts(self) -> Dict[BehaviorType, Dict[str, float]]:
        """행동별 점수 영향 초기화"""
        return {
            # 긍정적 행동
            BehaviorType.TRUTH_SEEKING: {
                'base_impact': +2.0,
                'dimensions': {
                    IntegrityDimension.HONESTY: +0.8,
                    IntegrityDimension.VERIFICATION: +1.0
                }
            },
            BehaviorType.ERROR_CORRECTION: {
                'base_impact': +1.5,
                'dimensions': {
                    IntegrityDimension.HONESTY: +1.0,
                    IntegrityDimension.ACCURACY: +0.8
                }
            },
            BehaviorType.UNCERTAINTY_ADMISSION: {
                'base_impact': +1.0,
                'dimensions': {
                    IntegrityDimension.HUMILITY: +1.0,
                    IntegrityDimension.HONESTY: +0.5
                }
            },
            BehaviorType.SOURCE_CITING: {
                'base_impact': +0.8,
                'dimensions': {
                    IntegrityDimension.TRANSPARENCY: +1.0,
                    IntegrityDimension.VERIFICATION: +0.6
                }
            },
            BehaviorType.BIAS_RECOGNITION: {
                'base_impact': +1.2,
                'dimensions': {
                    IntegrityDimension.HONESTY: +0.7,
                    IntegrityDimension.HUMILITY: +0.8
                }
            },
            BehaviorType.KNOWLEDGE_UPDATE: {
                'base_impact': +0.6,
                'dimensions': {
                    IntegrityDimension.ACCURACY: +0.8,
                    IntegrityDimension.CONSISTENCY: +0.4
                }
            },

            # 부정적 행동
            BehaviorType.MISINFORMATION: {
                'base_impact': -3.0,
                'dimensions': {
                    IntegrityDimension.ACCURACY: -1.5,
                    IntegrityDimension.HONESTY: -1.0
                }
            },
            BehaviorType.OVERCONFIDENCE: {
                'base_impact': -1.5,
                'dimensions': {
                    IntegrityDimension.HUMILITY: -1.2,
                    IntegrityDimension.ACCURACY: -0.8
                }
            },
            BehaviorType.SOURCE_OMISSION: {
                'base_impact': -1.0,
                'dimensions': {
                    IntegrityDimension.TRANSPARENCY: -1.0,
                    IntegrityDimension.VERIFICATION: -0.6
                }
            },
            BehaviorType.BIAS_REINFORCEMENT: {
                'base_impact': -2.0,
                'dimensions': {
                    IntegrityDimension.HONESTY: -1.0,
                    IntegrityDimension.ACCURACY: -0.8
                }
            },
            BehaviorType.ERROR_PERSISTENCE: {
                'base_impact': -2.5,
                'dimensions': {
                    IntegrityDimension.ACCURACY: -1.2,
                    IntegrityDimension.CONSISTENCY: -1.0
                }
            }
        }

    def _initialize_reward_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """보상 기준 초기화"""
        return {
            'excellence_milestone': {
                'score_threshold': 85.0,
                'reward_type': 'recognition',
                'value': 5.0,
                'description': 'Excellence in intellectual integrity'
            },
            'consistent_verification': {
                'consecutive_truth_seeking': 5,
                'reward_type': 'score_boost',
                'value': 3.0,
                'description': 'Consistent verification behavior'
            },
            'error_correction_habit': {
                'error_corrections': 3,
                'time_window_hours': 24,
                'reward_type': 'recognition',
                'value': 2.0,
                'description': 'Active error correction'
            },
            'transparency_champion': {
                'source_citations': 10,
                'time_window_hours': 168,  # 1주일
                'reward_type': 'privilege',
                'value': 4.0,
                'description': 'Exceptional transparency'
            }
        }

    def _initialize_penalty_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """패널티 기준 초기화"""
        return {
            'misinformation_spread': {
                'score_threshold': 30.0,
                'penalty_type': 'restriction',
                'value': -10.0,
                'description': 'Spreading misinformation'
            },
            'persistent_overconfidence': {
                'overconfidence_count': 3,
                'time_window_hours': 24,
                'penalty_type': 'warning',
                'value': -5.0,
                'description': 'Persistent overconfidence'
            },
            'verification_neglect': {
                'source_omissions': 5,
                'time_window_hours': 48,
                'penalty_type': 'score_reduction',
                'value': -8.0,
                'description': 'Neglecting verification responsibilities'
            }
        }

    async def record_behavior(
        self,
        behavior_type: BehaviorType,
        context: Dict[str, Any],
        evidence: Optional[List[str]] = None,
        confidence: float = 0.8
    ) -> Result[IntegrityAction]:
        """
        행동 기록 및 점수 업데이트

        Args:
            behavior_type: 행동 유형
            context: 행동 컨텍스트
            evidence: 증거 리스트
            confidence: 행동 감지 신뢰도

        Returns:
            Result[IntegrityAction]: 기록된 행동
        """
        try:
            # 주요 차원 결정
            primary_dimension = self._get_primary_dimension(behavior_type)

            # 점수 영향 계산
            score_impact = self._calculate_score_impact(behavior_type, context, confidence)

            # 행동 기록 생성
            action = IntegrityAction(
                action_id=f"action_{datetime.now().isoformat()}",
                behavior_type=behavior_type,
                dimension=primary_dimension,
                score_impact=score_impact,
                confidence=confidence,
                context=context,
                evidence=evidence or []
            )

            # 행동 히스토리에 추가
            self.action_history.append(action)

            # 점수 업데이트
            await self._update_scores(action)

            # 보상/패널티 확인
            await self._check_rewards_and_penalties()

            # 메트릭 재계산
            await self._recalculate_metrics()

            self.logger.info(f"Recorded behavior: {behavior_type.name}, Score impact: {score_impact:.2f}")

            return create_success(action)

        except Exception as e:
            self.logger.error(f"Error recording behavior: {str(e)}")
            return create_failure(f"Behavior recording failed: {str(e)}")

    def _get_primary_dimension(self, behavior_type: BehaviorType) -> IntegrityDimension:
        """행동의 주요 차원 결정"""
        dimension_mapping = {
            BehaviorType.TRUTH_SEEKING: IntegrityDimension.HONESTY,
            BehaviorType.ERROR_CORRECTION: IntegrityDimension.ACCURACY,
            BehaviorType.UNCERTAINTY_ADMISSION: IntegrityDimension.HUMILITY,
            BehaviorType.SOURCE_CITING: IntegrityDimension.TRANSPARENCY,
            BehaviorType.BIAS_RECOGNITION: IntegrityDimension.HONESTY,
            BehaviorType.KNOWLEDGE_UPDATE: IntegrityDimension.ACCURACY,
            BehaviorType.MISINFORMATION: IntegrityDimension.ACCURACY,
            BehaviorType.OVERCONFIDENCE: IntegrityDimension.HUMILITY,
            BehaviorType.SOURCE_OMISSION: IntegrityDimension.TRANSPARENCY,
            BehaviorType.BIAS_REINFORCEMENT: IntegrityDimension.HONESTY,
            BehaviorType.ERROR_PERSISTENCE: IntegrityDimension.ACCURACY
        }
        return dimension_mapping.get(behavior_type, IntegrityDimension.HONESTY)

    def _calculate_score_impact(
        self,
        behavior_type: BehaviorType,
        context: Dict[str, Any],
        confidence: float
    ) -> float:
        """점수 영향 계산"""
        behavior_config = self.behavior_impacts.get(behavior_type, {'base_impact': 0.0})
        base_impact = behavior_config['base_impact']

        # 신뢰도로 조정
        adjusted_impact = base_impact * confidence

        # 컨텍스트 기반 조정
        severity = context.get('severity', 'normal')
        severity_multipliers = {
            'low': 0.5,
            'normal': 1.0,
            'high': 1.5,
            'critical': 2.0
        }
        adjusted_impact *= severity_multipliers.get(severity, 1.0)

        # 반복 행동에 대한 감쇠/증폭
        recent_similar = self._count_recent_behavior(behavior_type, hours=24)
        if recent_similar > 0:
            if base_impact > 0:  # 긍정적 행동은 감쇠
                adjusted_impact *= (0.8 ** recent_similar)
            else:  # 부정적 행동은 증폭
                adjusted_impact *= (1.2 ** min(recent_similar, 3))

        return max(-10.0, min(10.0, adjusted_impact))

    def _count_recent_behavior(self, behavior_type: BehaviorType, hours: int = 24) -> int:
        """최근 유사 행동 수 계산"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return sum(1 for action in self.action_history
                  if action.behavior_type == behavior_type and action.timestamp >= cutoff_time)

    async def _update_scores(self, action: IntegrityAction):
        """점수 업데이트"""
        behavior_config = self.behavior_impacts.get(action.behavior_type, {})

        # 차원별 점수 업데이트
        dimension_impacts = behavior_config.get('dimensions', {})
        for dimension, impact in dimension_impacts.items():
            current_score = self.current_metrics.dimension_scores[dimension]
            adjusted_impact = impact * action.confidence

            new_score = max(self.min_score,
                          min(self.max_score, current_score + adjusted_impact))
            self.current_metrics.dimension_scores[dimension] = new_score

        # 전체 점수 재계산
        weighted_sum = sum(
            score * weight
            for dim, score in self.current_metrics.dimension_scores.items()
            for weight in [self.dimension_weights[dim]]
        )
        self.current_metrics.overall_score = weighted_sum

    async def _check_rewards_and_penalties(self):
        """보상 및 패널티 확인"""
        # 보상 확인
        await self._check_rewards()

        # 패널티 확인
        await self._check_penalties()

    async def _check_rewards(self):
        """보상 확인"""
        for reward_name, criteria in self.reward_thresholds.items():
            if await self._meets_reward_criteria(reward_name, criteria):
                reward = IntegrityReward(
                    reward_id=f"reward_{datetime.now().isoformat()}",
                    reward_type=criteria['reward_type'],
                    value=criteria['value'],
                    reason=criteria['description'],
                    criteria_met=[reward_name]
                )
                self.rewards.append(reward)

                # 점수 부스트 적용
                if criteria['reward_type'] == 'score_boost':
                    self.current_metrics.overall_score = min(
                        self.max_score,
                        self.current_metrics.overall_score + criteria['value']
                    )

                self.logger.info(f"Integrity reward granted: {reward_name}")

    async def _check_penalties(self):
        """패널티 확인"""
        for penalty_name, criteria in self.penalty_thresholds.items():
            if await self._meets_penalty_criteria(penalty_name, criteria):
                penalty = IntegrityPenalty(
                    penalty_id=f"penalty_{datetime.now().isoformat()}",
                    penalty_type=criteria['penalty_type'],
                    value=criteria['value'],
                    reason=criteria['description'],
                    violations=[penalty_name]
                )
                self.penalties.append(penalty)

                # 점수 차감 적용
                if criteria['penalty_type'] in ['score_reduction', 'restriction']:
                    self.current_metrics.overall_score = max(
                        self.min_score,
                        self.current_metrics.overall_score + criteria['value']  # value는 음수
                    )

                self.logger.warning(f"Integrity penalty applied: {penalty_name}")

    async def _meets_reward_criteria(self, reward_name: str, criteria: Dict[str, Any]) -> bool:
        """보상 기준 충족 여부 확인"""
        if reward_name == 'excellence_milestone':
            return self.current_metrics.overall_score >= criteria['score_threshold']

        elif reward_name == 'consistent_verification':
            recent_truth_seeking = [
                action for action in self.action_history[-10:]
                if action.behavior_type == BehaviorType.TRUTH_SEEKING
            ]
            return len(recent_truth_seeking) >= criteria['consecutive_truth_seeking']

        elif reward_name == 'error_correction_habit':
            cutoff_time = datetime.now() - timedelta(hours=criteria['time_window_hours'])
            recent_corrections = sum(
                1 for action in self.action_history
                if (action.behavior_type == BehaviorType.ERROR_CORRECTION and
                    action.timestamp >= cutoff_time)
            )
            return recent_corrections >= criteria['error_corrections']

        elif reward_name == 'transparency_champion':
            cutoff_time = datetime.now() - timedelta(hours=criteria['time_window_hours'])
            recent_citations = sum(
                1 for action in self.action_history
                if (action.behavior_type == BehaviorType.SOURCE_CITING and
                    action.timestamp >= cutoff_time)
            )
            return recent_citations >= criteria['source_citations']

        return False

    async def _meets_penalty_criteria(self, penalty_name: str, criteria: Dict[str, Any]) -> bool:
        """패널티 기준 충족 여부 확인"""
        if penalty_name == 'misinformation_spread':
            return self.current_metrics.overall_score <= criteria['score_threshold']

        elif penalty_name == 'persistent_overconfidence':
            cutoff_time = datetime.now() - timedelta(hours=criteria['time_window_hours'])
            recent_overconfidence = sum(
                1 for action in self.action_history
                if (action.behavior_type == BehaviorType.OVERCONFIDENCE and
                    action.timestamp >= cutoff_time)
            )
            return recent_overconfidence >= criteria['overconfidence_count']

        elif penalty_name == 'verification_neglect':
            cutoff_time = datetime.now() - timedelta(hours=criteria['time_window_hours'])
            recent_omissions = sum(
                1 for action in self.action_history
                if (action.behavior_type == BehaviorType.SOURCE_OMISSION and
                    action.timestamp >= cutoff_time)
            )
            return recent_omissions >= criteria['source_omissions']

        return False

    async def _recalculate_metrics(self):
        """메트릭 재계산"""
        # 최근 행동 통계
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_actions = [a for a in self.action_history if a.timestamp >= recent_cutoff]

        self.current_metrics.recent_actions = len(recent_actions)
        self.current_metrics.positive_actions = sum(
            1 for a in recent_actions if a.score_impact > 0
        )
        self.current_metrics.negative_actions = sum(
            1 for a in recent_actions if a.score_impact < 0
        )

        # 일관성 평가
        if len(self.action_history) > 5:
            recent_scores = [a.score_impact for a in self.action_history[-10:]]
            consistency = 1.0 - (statistics.stdev(recent_scores) / 10.0)
            self.current_metrics.consistency_rating = max(0.0, min(1.0, consistency))

        # 신뢰도 점수
        if recent_actions:
            avg_confidence = sum(a.confidence for a in recent_actions) / len(recent_actions)
            self.current_metrics.reliability_score = avg_confidence
        else:
            self.current_metrics.reliability_score = 0.5

        # 트렌드 계산
        if len(self.action_history) >= 10:
            old_actions = self.action_history[-20:-10]
            new_actions = self.action_history[-10:]

            old_avg = sum(a.score_impact for a in old_actions) / len(old_actions)
            new_avg = sum(a.score_impact for a in new_actions) / len(new_actions)

            if new_avg > old_avg + 0.1:
                self.current_metrics.trend = "improving"
            elif new_avg < old_avg - 0.1:
                self.current_metrics.trend = "declining"
            else:
                self.current_metrics.trend = "stable"

    def get_integrity_report(self) -> Dict[str, Any]:
        """무결성 보고서 생성"""
        return {
            'overall_metrics': {
                'score': self.current_metrics.overall_score,
                'level': self._get_integrity_level(self.current_metrics.overall_score),
                'trend': self.current_metrics.trend,
                'reliability': self.current_metrics.reliability_score,
                'consistency': self.current_metrics.consistency_rating
            },
            'dimension_scores': {
                dim.name: score for dim, score in self.current_metrics.dimension_scores.items()
            },
            'recent_activity': {
                'total_actions': self.current_metrics.recent_actions,
                'positive_actions': self.current_metrics.positive_actions,
                'negative_actions': self.current_metrics.negative_actions
            },
            'rewards_earned': len(self.rewards),
            'penalties_incurred': len(self.penalties),
            'action_history_length': len(self.action_history),
            'timestamp': datetime.now().isoformat()
        }

    def _get_integrity_level(self, score: float) -> str:
        """무결성 수준 문자열"""
        if score >= 90:
            return "EXCEPTIONAL"
        elif score >= 80:
            return "EXCELLENT"
        elif score >= 70:
            return "GOOD"
        elif score >= 60:
            return "FAIR"
        elif score >= 50:
            return "AVERAGE"
        elif score >= 40:
            return "BELOW_AVERAGE"
        elif score >= 30:
            return "POOR"
        else:
            return "CRITICAL"

    async def detect_dishonesty(self, content: str, context: Dict[str, Any]) -> Result[List[str]]:
        """거짓말/부정직 탐지"""
        try:
            dishonesty_indicators = []

            # 과도한 확신 표현 탐지
            overconfidence_patterns = [
                r'(?i)absolutely certain',
                r'(?i)definitely true',
                r'(?i)no doubt whatsoever',
                r'(?i)100% sure'
            ]

            import re
            for pattern in overconfidence_patterns:
                if re.search(pattern, content):
                    dishonesty_indicators.append(f"Overconfidence detected: {pattern}")

            # 모순 탐지
            contradiction_patterns = [
                r'(?i)always.*never',
                r'(?i)impossible.*possible',
                r'(?i)certain.*uncertain'
            ]

            for pattern in contradiction_patterns:
                if re.search(pattern, content):
                    dishonesty_indicators.append(f"Contradiction detected: {pattern}")

            # 출처 없는 사실 주장 탐지
            fact_claim_patterns = [
                r'(?i)studies show',
                r'(?i)research proves',
                r'(?i)statistics indicate'
            ]

            source_patterns = [
                r'(?i)according to',
                r'(?i)source:',
                r'(?i)cited in'
            ]

            has_fact_claims = any(re.search(pattern, content) for pattern in fact_claim_patterns)
            has_sources = any(re.search(pattern, content) for pattern in source_patterns)

            if has_fact_claims and not has_sources:
                dishonesty_indicators.append("Unsourced factual claims detected")

            return create_success(dishonesty_indicators)

        except Exception as e:
            self.logger.error(f"Error detecting dishonesty: {str(e)}")
            return create_failure(f"Dishonesty detection failed: {str(e)}")

    async def get_trust_score(self) -> float:
        """신뢰도 점수 반환 (0.0-1.0)"""
        return self.current_metrics.overall_score / 100.0

    async def cleanup(self):
        """리소스 정리"""
        self.logger.info("IntegrityScoring system cleanup completed")