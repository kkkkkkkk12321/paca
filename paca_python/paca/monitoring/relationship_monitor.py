"""
Relationship Monitor Module
관계적 항상성 시스템 - 사용자와의 관계 건강도 측정 및 관계 회복 시도
"""

import asyncio
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import statistics
import logging
from datetime import datetime, timedelta

# 조건부 임포트
try:
    from ..core.types.base import (
        ID, Timestamp, Result, current_timestamp, generate_id, create_success, create_failure
    )
except ImportError:
    from paca.core.types.base import (
        ID, Timestamp, Result, current_timestamp, generate_id, create_success, create_failure
    )


class HealthStatus(Enum):
    """관계 건강도 상태"""
    EXCELLENT = "excellent"     # 90-100%
    GOOD = "good"              # 70-89%
    FAIR = "fair"              # 50-69%
    POOR = "poor"              # 30-49%
    CRITICAL = "critical"      # 0-29%


class ConversationTone(Enum):
    """대화 톤"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


class InteractionType(Enum):
    """상호작용 유형"""
    QUESTION = "question"
    FEEDBACK = "feedback"
    CORRECTION = "correction"
    PRAISE = "praise"
    COMPLAINT = "complaint"
    REQUEST = "request"
    CASUAL = "casual"


@dataclass
class ConversationPattern:
    """대화 패턴"""
    pattern_id: ID
    user_message: str
    ai_response: str
    response_time: float
    user_satisfaction_score: Optional[float] = None  # 0-1
    tone: ConversationTone = ConversationTone.NEUTRAL
    interaction_type: InteractionType = InteractionType.CASUAL
    timestamp: Timestamp = field(default_factory=current_timestamp)

    @property
    def message_length_ratio(self) -> float:
        """사용자 메시지 대비 AI 응답 길이 비율"""
        if len(self.user_message) == 0:
            return 0.0
        return len(self.ai_response) / len(self.user_message)


@dataclass
class RelationshipMetrics:
    """관계 지표"""
    metric_id: ID
    period_start: Timestamp
    period_end: Timestamp

    # 대화 패턴 지표
    total_interactions: int
    avg_response_time: float
    avg_satisfaction_score: float
    tone_distribution: Dict[str, int]
    interaction_type_distribution: Dict[str, int]

    # 관계 건강도 지표
    engagement_score: float          # 참여도 (0-1)
    trust_score: float              # 신뢰도 (0-1)
    satisfaction_score: float       # 만족도 (0-1)
    communication_quality: float    # 소통 품질 (0-1)

    # 종합 건강도
    overall_health_score: float     # 전체 건강도 (0-1)
    health_status: HealthStatus

    timestamp: Timestamp = field(default_factory=current_timestamp)

    @property
    def health_percentage(self) -> float:
        """건강도 퍼센티지"""
        return self.overall_health_score * 100


@dataclass
class RelationshipAlert:
    """관계 알림"""
    alert_id: ID
    alert_type: str
    severity: str
    message: str
    recommendations: List[str]
    metrics: RelationshipMetrics
    timestamp: Timestamp = field(default_factory=current_timestamp)
    acknowledged: bool = False


@dataclass
class RecoveryAction:
    """관계 회복 행동"""
    action_id: ID
    action_type: str
    message_template: str
    trigger_conditions: List[str]
    success_rate: float = 0.0
    usage_count: int = 0
    created_at: Timestamp = field(default_factory=current_timestamp)


class RelationshipHealthAnalyzer:
    """관계 건강도 분석기"""

    def __init__(self):
        self.conversation_history: List[ConversationPattern] = []
        self.metrics_history: List[RelationshipMetrics] = []
        self.alerts: List[RelationshipAlert] = []

        # 감정 분석용 키워드
        self.tone_keywords = {
            ConversationTone.VERY_POSITIVE: [
                'excellent', 'amazing', 'perfect', 'wonderful', 'fantastic',
                '훌륭', '완벽', '최고', '대단', '감사'
            ],
            ConversationTone.POSITIVE: [
                'good', 'nice', 'helpful', 'useful', 'thanks',
                '좋', '도움', '고마', '유용', '만족'
            ],
            ConversationTone.NEGATIVE: [
                'wrong', 'bad', 'unhelpful', 'confusing', 'frustrated',
                '틀렸', '나쁘', '이상', '혼란', '답답'
            ],
            ConversationTone.VERY_NEGATIVE: [
                'terrible', 'awful', 'useless', 'hate', 'angry',
                '끔찍', '최악', '쓸모없', '화나', '짜증'
            ]
        }

        # 상호작용 유형 키워드
        self.interaction_keywords = {
            InteractionType.QUESTION: ['?', '뭐', '무엇', '어떻게', 'how', 'what', 'why'],
            InteractionType.FEEDBACK: ['피드백', 'feedback', '의견', '생각'],
            InteractionType.CORRECTION: ['틀렸', 'wrong', '수정', '고쳐', 'correct'],
            InteractionType.PRAISE: ['잘했', 'good job', '훌륭', '대단'],
            InteractionType.COMPLAINT: ['불만', 'complaint', '문제', 'problem'],
            InteractionType.REQUEST: ['해줘', 'please', '부탁', '요청', 'request']
        }

        self.logger = logging.getLogger(__name__)

        # 히스토리 크기 제한
        self.max_conversation_history = 1000
        self.max_metrics_history = 100
        self.max_alerts = 50

    def record_conversation(self,
                          user_message: str,
                          ai_response: str,
                          response_time: float,
                          user_satisfaction_score: Optional[float] = None) -> ID:
        """대화 기록"""
        try:
            # 톤 분석
            tone = self._analyze_tone(user_message)

            # 상호작용 유형 분석
            interaction_type = self._analyze_interaction_type(user_message)

            pattern = ConversationPattern(
                pattern_id=generate_id("conv_"),
                user_message=user_message,
                ai_response=ai_response,
                response_time=response_time,
                user_satisfaction_score=user_satisfaction_score,
                tone=tone,
                interaction_type=interaction_type
            )

            self.conversation_history.append(pattern)

            # 히스토리 크기 제한
            if len(self.conversation_history) > self.max_conversation_history:
                self.conversation_history = self.conversation_history[-self.max_conversation_history:]

            self.logger.debug(f"대화 기록됨: {pattern.pattern_id}, 톤: {tone.value}, 유형: {interaction_type.value}")
            return pattern.pattern_id

        except Exception as e:
            self.logger.error(f"대화 기록 실패: {e}")
            return generate_id("conv_error_")

    def _analyze_tone(self, message: str) -> ConversationTone:
        """메시지 톤 분석"""
        message_lower = message.lower()

        # 각 톤별 점수 계산
        tone_scores = {}
        for tone, keywords in self.tone_keywords.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            tone_scores[tone] = score

        # 가장 높은 점수의 톤 반환
        if any(tone_scores.values()):
            return max(tone_scores, key=tone_scores.get)

        return ConversationTone.NEUTRAL

    def _analyze_interaction_type(self, message: str) -> InteractionType:
        """상호작용 유형 분석"""
        message_lower = message.lower()

        # 각 유형별 점수 계산
        type_scores = {}
        for interaction_type, keywords in self.interaction_keywords.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            type_scores[interaction_type] = score

        # 가장 높은 점수의 유형 반환
        if any(type_scores.values()):
            return max(type_scores, key=type_scores.get)

        return InteractionType.CASUAL

    def analyze_relationship_health(self, hours: int = 24) -> RelationshipMetrics:
        """관계 건강도 분석"""
        try:
            end_time = current_timestamp()
            start_time = end_time - (hours * 3600)

            # 기간 내 대화 필터링
            recent_conversations = [
                conv for conv in self.conversation_history
                if start_time <= conv.timestamp <= end_time
            ]

            if not recent_conversations:
                # 기본 메트릭 반환
                return self._create_default_metrics(start_time, end_time)

            # 기본 통계
            total_interactions = len(recent_conversations)
            avg_response_time = statistics.mean([conv.response_time for conv in recent_conversations])

            # 만족도 점수 (만족도가 기록된 대화만)
            satisfaction_scores = [conv.user_satisfaction_score for conv in recent_conversations
                                 if conv.user_satisfaction_score is not None]
            avg_satisfaction_score = statistics.mean(satisfaction_scores) if satisfaction_scores else 0.5

            # 톤 분포
            tone_distribution = {}
            for tone in ConversationTone:
                tone_distribution[tone.value] = sum(1 for conv in recent_conversations if conv.tone == tone)

            # 상호작용 유형 분포
            interaction_type_distribution = {}
            for itype in InteractionType:
                interaction_type_distribution[itype.value] = sum(1 for conv in recent_conversations if conv.interaction_type == itype)

            # 관계 건강도 지표 계산
            engagement_score = self._calculate_engagement_score(recent_conversations)
            trust_score = self._calculate_trust_score(recent_conversations)
            satisfaction_score = avg_satisfaction_score
            communication_quality = self._calculate_communication_quality(recent_conversations)

            # 종합 건강도 (가중 평균)
            overall_health_score = (
                engagement_score * 0.25 +
                trust_score * 0.25 +
                satisfaction_score * 0.25 +
                communication_quality * 0.25
            )

            # 건강 상태 결정
            health_status = self._determine_health_status(overall_health_score)

            metrics = RelationshipMetrics(
                metric_id=generate_id("metrics_"),
                period_start=start_time,
                period_end=end_time,
                total_interactions=total_interactions,
                avg_response_time=avg_response_time,
                avg_satisfaction_score=avg_satisfaction_score,
                tone_distribution=tone_distribution,
                interaction_type_distribution=interaction_type_distribution,
                engagement_score=engagement_score,
                trust_score=trust_score,
                satisfaction_score=satisfaction_score,
                communication_quality=communication_quality,
                overall_health_score=overall_health_score,
                health_status=health_status
            )

            self.metrics_history.append(metrics)

            # 메트릭 히스토리 크기 제한
            if len(self.metrics_history) > self.max_metrics_history:
                self.metrics_history = self.metrics_history[-self.max_metrics_history:]

            # 알림 생성 여부 확인
            self._check_relationship_alerts(metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"관계 건강도 분석 실패: {e}")
            return self._create_default_metrics(start_time, end_time)

    def _create_default_metrics(self, start_time: Timestamp, end_time: Timestamp) -> RelationshipMetrics:
        """기본 메트릭 생성"""
        return RelationshipMetrics(
            metric_id=generate_id("metrics_default_"),
            period_start=start_time,
            period_end=end_time,
            total_interactions=0,
            avg_response_time=0.0,
            avg_satisfaction_score=0.5,
            tone_distribution={tone.value: 0 for tone in ConversationTone},
            interaction_type_distribution={itype.value: 0 for itype in InteractionType},
            engagement_score=0.5,
            trust_score=0.5,
            satisfaction_score=0.5,
            communication_quality=0.5,
            overall_health_score=0.5,
            health_status=HealthStatus.FAIR
        )

    def _calculate_engagement_score(self, conversations: List[ConversationPattern]) -> float:
        """참여도 점수 계산"""
        if not conversations:
            return 0.5

        # 대화 빈도 점수 (하루 기준)
        frequency_score = min(len(conversations) / 10, 1.0)  # 하루 10회를 100%로 설정

        # 메시지 길이 다양성
        message_lengths = [len(conv.user_message) for conv in conversations]
        length_variety = statistics.stdev(message_lengths) / statistics.mean(message_lengths) if len(message_lengths) > 1 else 0
        variety_score = min(length_variety, 1.0)

        # 상호작용 유형 다양성
        unique_types = len(set(conv.interaction_type for conv in conversations))
        type_variety_score = min(unique_types / len(InteractionType), 1.0)

        return (frequency_score * 0.5 + variety_score * 0.25 + type_variety_score * 0.25)

    def _calculate_trust_score(self, conversations: List[ConversationPattern]) -> float:
        """신뢰도 점수 계산"""
        if not conversations:
            return 0.5

        # 부정적 피드백 비율
        negative_tones = [ConversationTone.NEGATIVE, ConversationTone.VERY_NEGATIVE]
        negative_count = sum(1 for conv in conversations if conv.tone in negative_tones)
        negative_ratio = negative_count / len(conversations)

        # 수정 요청 비율
        correction_count = sum(1 for conv in conversations if conv.interaction_type == InteractionType.CORRECTION)
        correction_ratio = correction_count / len(conversations)

        # 신뢰도 = 1 - (부정적 비율 + 수정 비율) / 2
        trust_score = 1.0 - (negative_ratio + correction_ratio) / 2
        return max(0.0, min(trust_score, 1.0))

    def _calculate_communication_quality(self, conversations: List[ConversationPattern]) -> float:
        """소통 품질 점수 계산"""
        if not conversations:
            return 0.5

        # 응답 시간 점수 (빠를수록 좋음)
        avg_response_time = statistics.mean([conv.response_time for conv in conversations])
        response_time_score = max(0.0, 1.0 - (avg_response_time / 10.0))  # 10초를 기준으로 설정

        # 응답 길이 적절성 점수
        length_ratios = [conv.message_length_ratio for conv in conversations]
        avg_length_ratio = statistics.mean(length_ratios)
        # 1.0-3.0 비율을 적절하다고 가정
        if 1.0 <= avg_length_ratio <= 3.0:
            length_score = 1.0
        else:
            length_score = max(0.0, 1.0 - abs(avg_length_ratio - 2.0) / 2.0)

        # 톤 긍정성 점수
        positive_tones = [ConversationTone.POSITIVE, ConversationTone.VERY_POSITIVE]
        positive_count = sum(1 for conv in conversations if conv.tone in positive_tones)
        tone_score = positive_count / len(conversations)

        return (response_time_score * 0.4 + length_score * 0.3 + tone_score * 0.3)

    def _determine_health_status(self, overall_score: float) -> HealthStatus:
        """건강 상태 결정"""
        percentage = overall_score * 100

        if percentage >= 90:
            return HealthStatus.EXCELLENT
        elif percentage >= 70:
            return HealthStatus.GOOD
        elif percentage >= 50:
            return HealthStatus.FAIR
        elif percentage >= 30:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    def _check_relationship_alerts(self, metrics: RelationshipMetrics) -> None:
        """관계 알림 확인"""
        alerts_to_create = []

        # 건강도 임계치 알림
        if metrics.health_status == HealthStatus.CRITICAL:
            alerts_to_create.append({
                'type': 'critical_health',
                'severity': 'high',
                'message': f"관계 건강도가 위험 수준입니다 ({metrics.health_percentage:.1f}%)",
                'recommendations': [
                    "사용자와 메타 대화 시작을 고려하세요",
                    "더 친근하고 개인화된 응답을 시도하세요",
                    "사용자의 피드백을 더 적극적으로 요청하세요"
                ]
            })

        elif metrics.health_status == HealthStatus.POOR:
            alerts_to_create.append({
                'type': 'poor_health',
                'severity': 'medium',
                'message': f"관계 건강도가 낮습니다 ({metrics.health_percentage:.1f}%)",
                'recommendations': [
                    "사용자 만족도를 개선할 방법을 모색하세요",
                    "응답 품질과 관련성을 점검하세요"
                ]
            })

        # 신뢰도 알림
        if metrics.trust_score < 0.4:
            alerts_to_create.append({
                'type': 'low_trust',
                'severity': 'high',
                'message': f"사용자 신뢰도가 낮습니다 ({metrics.trust_score:.2f})",
                'recommendations': [
                    "더 정확하고 신뢰할 수 있는 정보를 제공하세요",
                    "불확실한 내용에 대해서는 솔직히 인정하세요"
                ]
            })

        # 참여도 알림
        if metrics.engagement_score < 0.3:
            alerts_to_create.append({
                'type': 'low_engagement',
                'severity': 'medium',
                'message': f"사용자 참여도가 낮습니다 ({metrics.engagement_score:.2f})",
                'recommendations': [
                    "더 흥미롭고 매력적인 대화를 유도하세요",
                    "사용자의 관심사에 맞는 질문을 해보세요"
                ]
            })

        # 알림 생성
        for alert_data in alerts_to_create:
            alert = RelationshipAlert(
                alert_id=generate_id("alert_"),
                alert_type=alert_data['type'],
                severity=alert_data['severity'],
                message=alert_data['message'],
                recommendations=alert_data['recommendations'],
                metrics=metrics
            )
            self.alerts.append(alert)

        # 알림 크기 제한
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]

    def get_recent_metrics(self, hours: int = 24) -> List[RelationshipMetrics]:
        """최근 메트릭 조회"""
        cutoff_time = current_timestamp() - (hours * 3600)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]

    def get_unacknowledged_alerts(self) -> List[RelationshipAlert]:
        """미확인 알림 조회"""
        return [alert for alert in self.alerts if not alert.acknowledged]

    def acknowledge_alert(self, alert_id: ID) -> bool:
        """알림 확인 처리"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False


class RelationshipRecovery:
    """관계 회복 시스템"""

    def __init__(self, analyzer: RelationshipHealthAnalyzer):
        self.analyzer = analyzer
        self.recovery_actions: List[RecoveryAction] = []
        self.logger = logging.getLogger(__name__)

        # 기본 회복 행동 정의
        self._initialize_recovery_actions()

    def _initialize_recovery_actions(self) -> None:
        """기본 회복 행동 초기화"""
        default_actions = [
            {
                'type': 'meta_conversation',
                'template': "제가 더 개선해야 할 점이 있을까요? 사용자님의 피드백을 듣고 싶습니다.",
                'triggers': ['critical_health', 'poor_health', 'low_trust']
            },
            {
                'type': 'acknowledge_mistake',
                'template': "죄송합니다. 제가 부족했던 부분이 있었나 봅니다. 더 나은 도움을 드리겠습니다.",
                'triggers': ['low_trust', 'negative_feedback']
            },
            {
                'type': 'engagement_boost',
                'template': "혹시 더 궁금하신 것이나 도움이 필요한 일이 있으신가요?",
                'triggers': ['low_engagement']
            },
            {
                'type': 'clarification_request',
                'template': "제가 정확히 이해했는지 확인하고 싶습니다. 혹시 제 답변에서 명확하지 않은 부분이 있나요?",
                'triggers': ['communication_issues']
            },
            {
                'type': 'appreciation',
                'template': "사용자님과 대화할 수 있어서 감사합니다. 더 도움이 되는 AI가 되도록 노력하겠습니다.",
                'triggers': ['relationship_maintenance']
            }
        ]

        for action_data in default_actions:
            action = RecoveryAction(
                action_id=generate_id("recovery_"),
                action_type=action_data['type'],
                message_template=action_data['template'],
                trigger_conditions=action_data['triggers']
            )
            self.recovery_actions.append(action)

    async def assess_recovery_need(self) -> Dict[str, Any]:
        """회복 필요성 평가"""
        try:
            # 최근 24시간 메트릭 분석
            recent_metrics = self.analyzer.analyze_relationship_health(24)

            # 미확인 알림 확인
            unacknowledged_alerts = self.analyzer.get_unacknowledged_alerts()

            recovery_needed = False
            recovery_urgency = "low"
            recommended_actions = []

            # 건강도 기반 평가
            if recent_metrics.health_status in [HealthStatus.CRITICAL, HealthStatus.POOR]:
                recovery_needed = True
                recovery_urgency = "high" if recent_metrics.health_status == HealthStatus.CRITICAL else "medium"

            # 알림 기반 평가
            if unacknowledged_alerts:
                recovery_needed = True
                for alert in unacknowledged_alerts:
                    if alert.severity == "high" and recovery_urgency != "high":
                        recovery_urgency = "high"
                    elif alert.severity == "medium" and recovery_urgency == "low":
                        recovery_urgency = "medium"

            # 권장 행동 선택
            if recovery_needed:
                recommended_actions = self._select_recovery_actions(recent_metrics, unacknowledged_alerts)

            return {
                'recovery_needed': recovery_needed,
                'urgency': recovery_urgency,
                'health_score': recent_metrics.overall_health_score,
                'health_status': recent_metrics.health_status.value,
                'active_alerts': len(unacknowledged_alerts),
                'recommended_actions': recommended_actions,
                'assessment_timestamp': current_timestamp()
            }

        except Exception as e:
            self.logger.error(f"회복 필요성 평가 실패: {e}")
            return {
                'recovery_needed': False,
                'urgency': 'low',
                'error': str(e)
            }

    def _select_recovery_actions(self,
                               metrics: RelationshipMetrics,
                               alerts: List[RelationshipAlert]) -> List[Dict[str, Any]]:
        """회복 행동 선택"""
        relevant_actions = []

        # 알림 기반 행동 선택
        alert_types = [alert.alert_type for alert in alerts]

        for action in self.recovery_actions:
            if any(trigger in alert_types for trigger in action.trigger_conditions):
                relevant_actions.append({
                    'action_id': action.action_id,
                    'type': action.action_type,
                    'message': action.message_template,
                    'success_rate': action.success_rate,
                    'usage_count': action.usage_count,
                    'recommended': True
                })

        # 건강도 기반 추가 행동
        if metrics.health_status == HealthStatus.CRITICAL:
            for action in self.recovery_actions:
                if 'critical_health' in action.trigger_conditions and action.action_id not in [a['action_id'] for a in relevant_actions]:
                    relevant_actions.append({
                        'action_id': action.action_id,
                        'type': action.action_type,
                        'message': action.message_template,
                        'success_rate': action.success_rate,
                        'usage_count': action.usage_count,
                        'recommended': True
                    })

        # 우선순위 정렬 (성공률과 사용 빈도 고려)
        relevant_actions.sort(key=lambda x: (x['success_rate'], -x['usage_count']), reverse=True)

        return relevant_actions[:3]  # 상위 3개만 반환

    async def execute_recovery_action(self, action_id: ID) -> Result[str]:
        """회복 행동 실행"""
        try:
            # 행동 찾기
            action = None
            for a in self.recovery_actions:
                if a.action_id == action_id:
                    action = a
                    break

            if not action:
                return create_failure(f"회복 행동을 찾을 수 없습니다: {action_id}")

            # 사용 횟수 증가
            action.usage_count += 1

            # 메시지 반환
            recovery_message = action.message_template

            self.logger.info(f"회복 행동 실행: {action.action_type}")
            return create_success(recovery_message)

        except Exception as e:
            self.logger.error(f"회복 행동 실행 실패: {e}")
            return create_failure(e)

    def record_action_success(self, action_id: ID, successful: bool) -> None:
        """회복 행동 성공 여부 기록"""
        for action in self.recovery_actions:
            if action.action_id == action_id:
                # 성공률 업데이트 (지수 이동 평균)
                if action.usage_count == 1:
                    action.success_rate = 1.0 if successful else 0.0
                else:
                    alpha = 0.1  # 학습률
                    new_value = 1.0 if successful else 0.0
                    action.success_rate = (1 - alpha) * action.success_rate + alpha * new_value
                break

    def get_recovery_statistics(self) -> Dict[str, Any]:
        """회복 시스템 통계"""
        total_actions = len(self.recovery_actions)
        total_usage = sum(action.usage_count for action in self.recovery_actions)
        avg_success_rate = statistics.mean([action.success_rate for action in self.recovery_actions]) if self.recovery_actions else 0.0

        return {
            'total_recovery_actions': total_actions,
            'total_usage_count': total_usage,
            'average_success_rate': avg_success_rate,
            'most_used_actions': [
                {
                    'action_id': action.action_id,
                    'type': action.action_type,
                    'usage_count': action.usage_count,
                    'success_rate': action.success_rate
                }
                for action in sorted(self.recovery_actions, key=lambda x: x.usage_count, reverse=True)[:5]
            ],
            'highest_success_actions': [
                {
                    'action_id': action.action_id,
                    'type': action.action_type,
                    'usage_count': action.usage_count,
                    'success_rate': action.success_rate
                }
                for action in sorted(self.recovery_actions, key=lambda x: x.success_rate, reverse=True)[:5]
            ]
        }


# 전역 인스턴스 (싱글톤 패턴)
_relationship_analyzer_instance = None
_relationship_recovery_instance = None


def get_relationship_analyzer() -> RelationshipHealthAnalyzer:
    """관계 분석기 싱글톤 인스턴스 획득"""
    global _relationship_analyzer_instance
    if _relationship_analyzer_instance is None:
        _relationship_analyzer_instance = RelationshipHealthAnalyzer()
    return _relationship_analyzer_instance


def get_relationship_recovery() -> RelationshipRecovery:
    """관계 회복 시스템 싱글톤 인스턴스 획득"""
    global _relationship_recovery_instance
    if _relationship_recovery_instance is None:
        analyzer = get_relationship_analyzer()
        _relationship_recovery_instance = RelationshipRecovery(analyzer)
    return _relationship_recovery_instance