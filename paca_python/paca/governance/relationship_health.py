"""
Relationship Health Module
관계적 항상성 시스템

이 모듈은 사용자와 PACA 간의 관계 건강도를 모니터링하고 관리합니다:
- 대화 톤 분석
- 상호작용 패턴 추적
- 관계 만족도 평가
- 갈등 감지 및 해결 제안
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import statistics
import re
import time

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
    """관계 건강 상태"""
    CRITICAL = "critical"     # 관계 위기
    POOR = "poor"            # 관계 악화
    FAIR = "fair"            # 보통 관계
    GOOD = "good"            # 좋은 관계
    EXCELLENT = "excellent"   # 탁월한 관계


class ToneType(Enum):
    """대화 톤 타입"""
    HOSTILE = "hostile"           # 적대적
    FRUSTRATED = "frustrated"    # 좌절감
    NEUTRAL = "neutral"          # 중립적
    FRIENDLY = "friendly"        # 친근한
    APPRECIATIVE = "appreciative" # 감사하는


class InteractionType(Enum):
    """상호작용 타입"""
    QUESTION = "question"         # 질문
    REQUEST = "request"           # 요청
    COMPLAINT = "complaint"       # 불만
    FEEDBACK = "feedback"         # 피드백
    APPRECIATION = "appreciation" # 감사 표현
    CASUAL = "casual"            # 일상 대화


class ConflictType(Enum):
    """갈등 타입"""
    MISUNDERSTANDING = "misunderstanding"     # 오해
    EXPECTATION_MISMATCH = "expectation_mismatch"  # 기대 불일치
    TECHNICAL_ISSUE = "technical_issue"       # 기술적 문제
    COMMUNICATION_GAP = "communication_gap"   # 소통 문제
    VALUE_DIFFERENCE = "value_difference"     # 가치관 차이


@dataclass(frozen=True)
class ToneAnalysis:
    """톤 분석 결과"""
    analysis_id: ID
    tone_type: ToneType
    confidence: float  # 0.0-1.0
    intensity: float   # 0.0-1.0 (톤의 강도)
    detected_emotions: List[str]
    key_indicators: List[str]
    timestamp: Timestamp


@dataclass(frozen=True)
class InteractionRecord:
    """상호작용 기록"""
    interaction_id: ID
    interaction_type: InteractionType
    user_message: str
    ai_response: str
    tone_analysis: ToneAnalysis
    satisfaction_score: Optional[float] = None  # 사용자 만족도 (0.0-1.0)
    duration_seconds: Optional[int] = None
    timestamp: Timestamp = field(default_factory=current_timestamp)


@dataclass(frozen=True)
class ConflictDetection:
    """갈등 감지 결과"""
    detection_id: ID
    conflict_type: ConflictType
    severity: float  # 0.0-1.0
    trigger_content: str
    context: Dict[str, Any]
    resolution_suggestions: List[str]
    timestamp: Timestamp


@dataclass(frozen=True)
class HealthMetric:
    """건강도 지표"""
    metric_name: str
    current_value: float  # 0.0-1.0
    trend: str  # 'improving', 'stable', 'declining'
    measurement_period: int  # 측정 기간 (시간)
    timestamp: Timestamp


@dataclass(frozen=True)
class HealthReport:
    """관계 건강 보고서"""
    report_id: ID
    overall_status: HealthStatus
    overall_score: float  # 0.0-1.0
    metrics: Dict[str, HealthMetric]
    tone_distribution: Dict[ToneType, float]
    interaction_patterns: Dict[str, Any]
    conflict_summary: Dict[str, Any]
    improvement_suggestions: List[str]
    timestamp: Timestamp


class ToneAnalyzer:
    """대화 톤 분석기"""

    def __init__(self):
        self.tone_indicators = {
            ToneType.HOSTILE: {
                'keywords': ['stupid', 'useless', 'terrible', 'hate', 'awful', 'worst'],
                'patterns': [r'(?i)you.*wrong', r'(?i)completely.*useless', r'(?i)total.*garbage']
            },
            ToneType.FRUSTRATED: {
                'keywords': ['frustrated', 'annoying', 'confusing', 'difficult', 'problem'],
                'patterns': [r'(?i)why.*not.*work', r'(?i)this.*makes.*no.*sense', r'(?i)so.*frustrating']
            },
            ToneType.NEUTRAL: {
                'keywords': ['okay', 'fine', 'understand', 'see', 'thanks'],
                'patterns': [r'(?i)^(ok|okay|fine)$', r'(?i)i.*see', r'(?i)understood']
            },
            ToneType.FRIENDLY: {
                'keywords': ['great', 'good', 'nice', 'helpful', 'wonderful', 'excellent'],
                'patterns': [r'(?i)that.*great', r'(?i)really.*helpful', r'(?i)love.*this']
            },
            ToneType.APPRECIATIVE: {
                'keywords': ['thank', 'appreciate', 'grateful', 'amazing', 'perfect', 'brilliant'],
                'patterns': [r'(?i)thank.*you.*much', r'(?i)really.*appreciate', r'(?i)so.*helpful']
            }
        }

        self.emotion_indicators = {
            'happy': ['happy', 'joy', 'glad', 'pleased', 'delighted'],
            'sad': ['sad', 'disappointed', 'upset', 'down'],
            'angry': ['angry', 'mad', 'furious', 'irritated'],
            'excited': ['excited', 'thrilled', 'amazing', 'awesome'],
            'confused': ['confused', 'unclear', 'lost', 'puzzled'],
            'satisfied': ['satisfied', 'content', 'pleased', 'good']
        }

    async def analyze_tone(self, message: str, context: Dict[str, Any]) -> Result[ToneAnalysis]:
        """메시지 톤 분석"""
        try:
            analysis_id = generate_id("tone_")

            # 각 톤 타입별 점수 계산
            tone_scores = {}
            detected_indicators = {}

            for tone_type, indicators in self.tone_indicators.items():
                score, indicators_found = await self._calculate_tone_score(message, indicators)
                tone_scores[tone_type] = score
                detected_indicators[tone_type] = indicators_found

            # 가장 높은 점수의 톤 선택
            dominant_tone = max(tone_scores.items(), key=lambda x: x[1])
            tone_type = dominant_tone[0]
            confidence = dominant_tone[1]

            # 강도 계산 (대문자, 느낌표 등)
            intensity = await self._calculate_intensity(message)

            # 감정 감지
            detected_emotions = await self._detect_emotions(message)

            # 주요 지표 추출
            key_indicators = detected_indicators[tone_type]

            analysis = ToneAnalysis(
                analysis_id=analysis_id,
                tone_type=tone_type,
                confidence=confidence,
                intensity=intensity,
                detected_emotions=detected_emotions,
                key_indicators=key_indicators,
                timestamp=current_timestamp()
            )

            return create_success(analysis)

        except Exception as e:
            return create_failure(e)

    async def _calculate_tone_score(self, message: str,
                                  indicators: Dict[str, List[str]]) -> Tuple[float, List[str]]:
        """톤 점수 계산"""
        message_lower = message.lower()
        score = 0.0
        found_indicators = []

        # 키워드 기반 점수
        for keyword in indicators['keywords']:
            if keyword in message_lower:
                score += 0.1
                found_indicators.append(keyword)

        # 패턴 기반 점수
        for pattern in indicators['patterns']:
            matches = re.findall(pattern, message)
            if matches:
                score += 0.2
                found_indicators.extend(matches)

        # 정규화
        score = min(score, 1.0)
        return score, found_indicators

    async def _calculate_intensity(self, message: str) -> float:
        """강도 계산"""
        intensity_score = 0.0

        # 대문자 비율
        if message:
            uppercase_ratio = sum(1 for c in message if c.isupper()) / len(message)
            intensity_score += uppercase_ratio * 0.3

        # 느낌표 개수
        exclamation_count = message.count('!')
        intensity_score += min(exclamation_count * 0.1, 0.3)

        # 반복된 문자
        repeated_chars = len(re.findall(r'(.)\1{2,}', message))
        intensity_score += min(repeated_chars * 0.1, 0.2)

        # 전체 길이 고려 (긴 메시지는 강도가 낮아질 수 있음)
        length_factor = min(len(message) / 100, 1.0)
        intensity_score *= (0.5 + length_factor * 0.5)

        return min(intensity_score, 1.0)

    async def _detect_emotions(self, message: str) -> List[str]:
        """감정 감지"""
        message_lower = message.lower()
        detected_emotions = []

        for emotion, keywords in self.emotion_indicators.items():
            if any(keyword in message_lower for keyword in keywords):
                detected_emotions.append(emotion)

        return detected_emotions


class InteractionPatternAnalyzer:
    """상호작용 패턴 분석기"""

    def __init__(self):
        self.interaction_classifiers = {
            InteractionType.QUESTION: [r'\?', r'(?i)what.*is', r'(?i)how.*do', r'(?i)why.*'],
            InteractionType.REQUEST: [r'(?i)please.*', r'(?i)can.*you', r'(?i)would.*you'],
            InteractionType.COMPLAINT: [r'(?i)problem.*with', r'(?i)not.*working', r'(?i)issue.*'],
            InteractionType.FEEDBACK: [r'(?i)feedback', r'(?i)suggestion', r'(?i)improvement'],
            InteractionType.APPRECIATION: [r'(?i)thank.*you', r'(?i)appreciate', r'(?i)great.*job'],
            InteractionType.CASUAL: [r'(?i)hello', r'(?i)hi', r'(?i)good.*morning']
        }

    async def classify_interaction(self, message: str) -> Result[InteractionType]:
        """상호작용 분류"""
        try:
            scores = {}

            for interaction_type, patterns in self.interaction_classifiers.items():
                score = 0.0
                for pattern in patterns:
                    matches = len(re.findall(pattern, message))
                    score += matches * 0.2

                scores[interaction_type] = score

            # 가장 높은 점수의 타입 선택
            if max(scores.values()) == 0:
                return create_success(InteractionType.CASUAL)

            best_type = max(scores.items(), key=lambda x: x[1])[0]
            return create_success(best_type)

        except Exception as e:
            return create_failure(e)

    async def analyze_patterns(self, interactions: List[InteractionRecord],
                             time_window_hours: int = 24) -> Result[Dict[str, Any]]:
        """상호작용 패턴 분석"""
        try:
            if not interactions:
                return create_success({'pattern': 'no_data'})

            cutoff_time = current_timestamp() - (time_window_hours * 3600)
            recent_interactions = [i for i in interactions if i.timestamp >= cutoff_time]

            if not recent_interactions:
                return create_success({'pattern': 'no_recent_data'})

            # 타입별 분포
            type_counts = {}
            for itype in InteractionType:
                type_counts[itype.value] = sum(1 for i in recent_interactions
                                             if i.interaction_type == itype)

            # 톤 분포
            tone_counts = {}
            for tone in ToneType:
                tone_counts[tone.value] = sum(1 for i in recent_interactions
                                            if i.tone_analysis.tone_type == tone)

            # 시간대별 분포 (간단화)
            hour_counts = {}
            for interaction in recent_interactions:
                hour = time.localtime(interaction.timestamp).tm_hour
                hour_range = f"{hour//6*6}-{(hour//6+1)*6}"  # 6시간 단위
                hour_counts[hour_range] = hour_counts.get(hour_range, 0) + 1

            # 평균 만족도
            satisfaction_scores = [i.satisfaction_score for i in recent_interactions
                                 if i.satisfaction_score is not None]
            avg_satisfaction = statistics.mean(satisfaction_scores) if satisfaction_scores else None

            patterns = {
                'total_interactions': len(recent_interactions),
                'type_distribution': type_counts,
                'tone_distribution': tone_counts,
                'time_distribution': hour_counts,
                'average_satisfaction': avg_satisfaction,
                'dominant_type': max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None,
                'dominant_tone': max(tone_counts.items(), key=lambda x: x[1])[0] if tone_counts else None
            }

            return create_success(patterns)

        except Exception as e:
            return create_failure(e)


class ConflictDetector:
    """갈등 감지기"""

    def __init__(self):
        self.conflict_indicators = {
            ConflictType.MISUNDERSTANDING: {
                'keywords': ['misunderstand', 'confused', 'unclear', 'wrong'],
                'patterns': [r'(?i)that.*not.*what.*meant', r'(?i)you.*misunderstood']
            },
            ConflictType.EXPECTATION_MISMATCH: {
                'keywords': ['expected', 'thought', 'supposed', 'should'],
                'patterns': [r'(?i)expected.*different', r'(?i)thought.*would', r'(?i)should.*be']
            },
            ConflictType.TECHNICAL_ISSUE: {
                'keywords': ['error', 'bug', 'broken', 'not working', 'failed'],
                'patterns': [r'(?i)technical.*problem', r'(?i)system.*error']
            },
            ConflictType.COMMUNICATION_GAP: {
                'keywords': ['communicate', 'explain', 'understand', 'clear'],
                'patterns': [r'(?i)need.*better.*explanation', r'(?i)communication.*problem']
            },
            ConflictType.VALUE_DIFFERENCE: {
                'keywords': ['disagree', 'wrong', 'inappropriate', 'unacceptable'],
                'patterns': [r'(?i)fundamentally.*disagree', r'(?i)values.*different']
            }
        }

    async def detect_conflict(self, interaction: InteractionRecord,
                            context: Dict[str, Any]) -> Result[Optional[ConflictDetection]]:
        """갈등 감지"""
        try:
            # 부정적 톤과 높은 강도가 갈등의 기본 지표
            tone_analysis = interaction.tone_analysis
            if tone_analysis.tone_type not in [ToneType.HOSTILE, ToneType.FRUSTRATED]:
                return create_success(None)

            if tone_analysis.intensity < 0.3:
                return create_success(None)

            # 갈등 타입 분석
            conflict_scores = {}
            for conflict_type, indicators in self.conflict_indicators.items():
                score = await self._calculate_conflict_score(interaction.user_message, indicators)
                conflict_scores[conflict_type] = score

            # 가장 높은 점수의 갈등 타입
            max_score = max(conflict_scores.values())
            if max_score < 0.3:  # 임계값 미달
                return create_success(None)

            conflict_type = max(conflict_scores.items(), key=lambda x: x[1])[0]

            # 심각도 계산
            severity = min((tone_analysis.intensity + max_score) / 2, 1.0)

            # 해결 제안 생성
            resolution_suggestions = await self._generate_resolution_suggestions(conflict_type, interaction)

            conflict_detection = ConflictDetection(
                detection_id=generate_id("conflict_"),
                conflict_type=conflict_type,
                severity=severity,
                trigger_content=interaction.user_message,
                context=context,
                resolution_suggestions=resolution_suggestions,
                timestamp=current_timestamp()
            )

            return create_success(conflict_detection)

        except Exception as e:
            return create_failure(e)

    async def _calculate_conflict_score(self, message: str,
                                      indicators: Dict[str, List[str]]) -> float:
        """갈등 점수 계산"""
        message_lower = message.lower()
        score = 0.0

        # 키워드 기반 점수
        for keyword in indicators['keywords']:
            if keyword in message_lower:
                score += 0.1

        # 패턴 기반 점수
        for pattern in indicators['patterns']:
            if re.search(pattern, message):
                score += 0.2

        return min(score, 1.0)

    async def _generate_resolution_suggestions(self, conflict_type: ConflictType,
                                             interaction: InteractionRecord) -> List[str]:
        """해결 제안 생성"""
        base_suggestions = {
            ConflictType.MISUNDERSTANDING: [
                "사용자의 의도를 명확히 확인하고 재설명을 제공하세요",
                "구체적인 예시를 들어 이해를 도우세요"
            ],
            ConflictType.EXPECTATION_MISMATCH: [
                "기능과 한계를 명확히 설명하세요",
                "현실적인 기대치를 설정하도록 도우세요"
            ],
            ConflictType.TECHNICAL_ISSUE: [
                "기술적 문제를 확인하고 해결 방법을 제시하세요",
                "대안적 접근 방법을 제안하세요"
            ],
            ConflictType.COMMUNICATION_GAP: [
                "더 명확하고 단계적인 설명을 제공하세요",
                "사용자의 배경 지식 수준에 맞춰 소통하세요"
            ],
            ConflictType.VALUE_DIFFERENCE: [
                "다양한 관점을 존중하며 대화하세요",
                "공통점을 찾아 합의점을 모색하세요"
            ]
        }

        suggestions = base_suggestions.get(conflict_type, ["상황을 재평가하고 건설적인 해결책을 모색하세요"])

        # 상황별 추가 제안
        if interaction.tone_analysis.intensity > 0.7:
            suggestions.append("감정이 진정될 때까지 시간을 두고 대화를 계속하세요")

        return suggestions


class RelationshipHealth:
    """관계 건강도 관리자"""

    def __init__(self):
        self.tone_analyzer = ToneAnalyzer()
        self.pattern_analyzer = InteractionPatternAnalyzer()
        self.conflict_detector = ConflictDetector()

        self.interactions: List[InteractionRecord] = []
        self.conflicts: List[ConflictDetection] = []
        self.health_reports: List[HealthReport] = []

        # 건강도 가중치
        self.metric_weights = {
            'tone_positivity': 0.3,      # 톤 긍정성
            'interaction_variety': 0.2,   # 상호작용 다양성
            'conflict_frequency': 0.25,   # 갈등 빈도 (역수)
            'satisfaction_level': 0.25    # 만족도 수준
        }

    async def record_interaction(self, user_message: str, ai_response: str,
                               context: Dict[str, Any]) -> Result[InteractionRecord]:
        """상호작용 기록"""
        try:
            # 톤 분석
            tone_result = await self.tone_analyzer.analyze_tone(user_message, context)
            if not tone_result.is_success:
                return create_failure(tone_result.error)

            # 상호작용 타입 분류
            type_result = await self.pattern_analyzer.classify_interaction(user_message)
            if not type_result.is_success:
                return create_failure(type_result.error)

            # 상호작용 기록 생성
            interaction = InteractionRecord(
                interaction_id=generate_id("interaction_"),
                interaction_type=type_result.value,
                user_message=user_message,
                ai_response=ai_response,
                tone_analysis=tone_result.value,
                duration_seconds=context.get('duration_seconds'),
                timestamp=current_timestamp()
            )

            self.interactions.append(interaction)

            # 갈등 감지
            conflict_result = await self.conflict_detector.detect_conflict(interaction, context)
            if conflict_result.is_success and conflict_result.value:
                self.conflicts.append(conflict_result.value)

            return create_success(interaction)

        except Exception as e:
            return create_failure(e)

    async def assess_health(self, time_window_hours: int = 24) -> Result[HealthReport]:
        """관계 건강도 평가"""
        try:
            report_id = generate_id("health_report_")

            # 최근 상호작용 필터링
            cutoff_time = current_timestamp() - (time_window_hours * 3600)
            recent_interactions = [i for i in self.interactions if i.timestamp >= cutoff_time]
            recent_conflicts = [c for c in self.conflicts if c.timestamp >= cutoff_time]

            # 각 지표 계산
            metrics = {}

            # 톤 긍정성 계산
            tone_positivity = await self._calculate_tone_positivity(recent_interactions)
            metrics['tone_positivity'] = HealthMetric(
                metric_name='tone_positivity',
                current_value=tone_positivity,
                trend=await self._calculate_trend('tone_positivity', time_window_hours),
                measurement_period=time_window_hours,
                timestamp=current_timestamp()
            )

            # 상호작용 다양성 계산
            interaction_variety = await self._calculate_interaction_variety(recent_interactions)
            metrics['interaction_variety'] = HealthMetric(
                metric_name='interaction_variety',
                current_value=interaction_variety,
                trend=await self._calculate_trend('interaction_variety', time_window_hours),
                measurement_period=time_window_hours,
                timestamp=current_timestamp()
            )

            # 갈등 빈도 (역수) 계산
            conflict_frequency = await self._calculate_conflict_frequency(recent_conflicts, recent_interactions)
            metrics['conflict_frequency'] = HealthMetric(
                metric_name='conflict_frequency',
                current_value=conflict_frequency,
                trend=await self._calculate_trend('conflict_frequency', time_window_hours),
                measurement_period=time_window_hours,
                timestamp=current_timestamp()
            )

            # 만족도 수준 계산
            satisfaction_level = await self._calculate_satisfaction_level(recent_interactions)
            metrics['satisfaction_level'] = HealthMetric(
                metric_name='satisfaction_level',
                current_value=satisfaction_level,
                trend=await self._calculate_trend('satisfaction_level', time_window_hours),
                measurement_period=time_window_hours,
                timestamp=current_timestamp()
            )

            # 전체 건강도 점수 계산
            overall_score = sum(metric.current_value * self.metric_weights[name]
                              for name, metric in metrics.items())

            # 건강 상태 결정
            overall_status = self._determine_health_status(overall_score)

            # 톤 분포 계산
            tone_distribution = await self._calculate_tone_distribution(recent_interactions)

            # 상호작용 패턴 분석
            interaction_patterns_result = await self.pattern_analyzer.analyze_patterns(
                recent_interactions, time_window_hours
            )
            interaction_patterns = interaction_patterns_result.value if interaction_patterns_result.is_success else {}

            # 갈등 요약
            conflict_summary = await self._summarize_conflicts(recent_conflicts)

            # 개선 제안 생성
            improvement_suggestions = await self._generate_improvement_suggestions(
                metrics, recent_conflicts, overall_score
            )

            # 보고서 생성
            report = HealthReport(
                report_id=report_id,
                overall_status=overall_status,
                overall_score=overall_score,
                metrics=metrics,
                tone_distribution=tone_distribution,
                interaction_patterns=interaction_patterns,
                conflict_summary=conflict_summary,
                improvement_suggestions=improvement_suggestions,
                timestamp=current_timestamp()
            )

            self.health_reports.append(report)
            return create_success(report)

        except Exception as e:
            return create_failure(e)

    async def _calculate_tone_positivity(self, interactions: List[InteractionRecord]) -> float:
        """톤 긍정성 계산"""
        if not interactions:
            return 0.5  # 중립

        positive_tones = [ToneType.FRIENDLY, ToneType.APPRECIATIVE]
        negative_tones = [ToneType.HOSTILE, ToneType.FRUSTRATED]

        positive_count = sum(1 for i in interactions
                           if i.tone_analysis.tone_type in positive_tones)
        negative_count = sum(1 for i in interactions
                           if i.tone_analysis.tone_type in negative_tones)
        neutral_count = len(interactions) - positive_count - negative_count

        # 가중 점수 계산
        weighted_score = (positive_count * 1.0 + neutral_count * 0.5 + negative_count * 0.0) / len(interactions)
        return weighted_score

    async def _calculate_interaction_variety(self, interactions: List[InteractionRecord]) -> float:
        """상호작용 다양성 계산"""
        if not interactions:
            return 0.0

        # 고유한 상호작용 타입 수
        unique_types = set(i.interaction_type for i in interactions)
        max_possible_types = len(InteractionType)

        variety_score = len(unique_types) / max_possible_types
        return variety_score

    async def _calculate_conflict_frequency(self, conflicts: List[ConflictDetection],
                                         interactions: List[InteractionRecord]) -> float:
        """갈등 빈도 계산 (높을수록 좋음 - 갈등이 적을수록)"""
        if not interactions:
            return 1.0  # 상호작용이 없으면 갈등도 없음

        conflict_ratio = len(conflicts) / len(interactions)
        # 갈등 빈도의 역수로 계산 (갈등이 적을수록 점수가 높음)
        frequency_score = max(0.0, 1.0 - conflict_ratio * 2)  # 50% 갈등시 0점
        return frequency_score

    async def _calculate_satisfaction_level(self, interactions: List[InteractionRecord]) -> float:
        """만족도 수준 계산"""
        satisfaction_scores = [i.satisfaction_score for i in interactions
                             if i.satisfaction_score is not None]

        if not satisfaction_scores:
            # 만족도 점수가 없으면 톤을 기반으로 추정
            if not interactions:
                return 0.5

            positive_interactions = sum(1 for i in interactions
                                      if i.tone_analysis.tone_type in [ToneType.FRIENDLY, ToneType.APPRECIATIVE])
            estimated_satisfaction = positive_interactions / len(interactions)
            return estimated_satisfaction

        return statistics.mean(satisfaction_scores)

    async def _calculate_trend(self, metric_name: str, time_window_hours: int) -> str:
        """지표 추세 계산"""
        if len(self.health_reports) < 2:
            return 'stable'

        # 최근 두 보고서 비교
        recent_reports = self.health_reports[-2:]
        if len(recent_reports) < 2:
            return 'stable'

        current_value = recent_reports[-1].metrics.get(metric_name)
        previous_value = recent_reports[-2].metrics.get(metric_name)

        if not current_value or not previous_value:
            return 'stable'

        diff = current_value.current_value - previous_value.current_value

        if diff > 0.05:
            return 'improving'
        elif diff < -0.05:
            return 'declining'
        else:
            return 'stable'

    def _determine_health_status(self, overall_score: float) -> HealthStatus:
        """건강 상태 결정"""
        if overall_score >= 0.8:
            return HealthStatus.EXCELLENT
        elif overall_score >= 0.65:
            return HealthStatus.GOOD
        elif overall_score >= 0.5:
            return HealthStatus.FAIR
        elif overall_score >= 0.3:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    async def _calculate_tone_distribution(self, interactions: List[InteractionRecord]) -> Dict[ToneType, float]:
        """톤 분포 계산"""
        if not interactions:
            return {tone: 0.0 for tone in ToneType}

        tone_counts = {}
        for tone in ToneType:
            count = sum(1 for i in interactions if i.tone_analysis.tone_type == tone)
            tone_counts[tone] = count / len(interactions)

        return tone_counts

    async def _summarize_conflicts(self, conflicts: List[ConflictDetection]) -> Dict[str, Any]:
        """갈등 요약"""
        if not conflicts:
            return {'total_conflicts': 0}

        # 타입별 분포
        type_counts = {}
        for ctype in ConflictType:
            type_counts[ctype.value] = sum(1 for c in conflicts if c.conflict_type == ctype)

        # 평균 심각도
        avg_severity = statistics.mean(c.severity for c in conflicts)

        # 최근 갈등
        recent_conflicts = [
            {
                'type': c.conflict_type.value,
                'severity': c.severity,
                'timestamp': c.timestamp
            }
            for c in conflicts[-3:]  # 최근 3개
        ]

        return {
            'total_conflicts': len(conflicts),
            'type_distribution': type_counts,
            'average_severity': avg_severity,
            'recent_conflicts': recent_conflicts
        }

    async def _generate_improvement_suggestions(self, metrics: Dict[str, HealthMetric],
                                              conflicts: List[ConflictDetection],
                                              overall_score: float) -> List[str]:
        """개선 제안 생성"""
        suggestions = []

        # 지표별 제안
        for name, metric in metrics.items():
            if metric.current_value < 0.6:
                if name == 'tone_positivity':
                    suggestions.append("사용자와의 소통에서 더 긍정적이고 친근한 톤을 유지하세요")
                elif name == 'interaction_variety':
                    suggestions.append("다양한 유형의 상호작용을 장려하여 관계를 풍부하게 만드세요")
                elif name == 'conflict_frequency':
                    suggestions.append("갈등 예방과 조기 해결에 더 집중하세요")
                elif name == 'satisfaction_level':
                    suggestions.append("사용자 만족도 향상을 위해 더 나은 서비스를 제공하세요")

        # 갈등별 제안
        if conflicts:
            common_conflict_types = {}
            for conflict in conflicts:
                ctype = conflict.conflict_type
                common_conflict_types[ctype] = common_conflict_types.get(ctype, 0) + 1

            most_common = max(common_conflict_types.items(), key=lambda x: x[1])[0]
            if most_common == ConflictType.MISUNDERSTANDING:
                suggestions.append("의사소통 명확성을 개선하여 오해를 줄이세요")
            elif most_common == ConflictType.TECHNICAL_ISSUE:
                suggestions.append("기술적 문제 해결 능력을 강화하세요")

        # 전반적 건강도에 따른 제안
        if overall_score < 0.5:
            suggestions.append("관계 회복을 위한 종합적인 접근이 필요합니다")
        elif overall_score < 0.7:
            suggestions.append("현재 관계를 유지하면서 점진적인 개선을 추진하세요")

        if not suggestions:
            suggestions.append("현재 건강한 관계를 유지하고 있습니다. 지속적인 모니터링을 권장합니다")

        return suggestions

    def get_relationship_statistics(self) -> Dict[str, Any]:
        """관계 통계"""
        if not self.interactions:
            return {'total_interactions': 0}

        total_interactions = len(self.interactions)
        total_conflicts = len(self.conflicts)

        # 최근 보고서
        latest_report = self.health_reports[-1] if self.health_reports else None

        # 전체 톤 분포
        overall_tone_dist = {}
        for tone in ToneType:
            count = sum(1 for i in self.interactions if i.tone_analysis.tone_type == tone)
            overall_tone_dist[tone.value] = count / total_interactions if total_interactions > 0 else 0

        return {
            'total_interactions': total_interactions,
            'total_conflicts': total_conflicts,
            'conflict_rate': total_conflicts / total_interactions if total_interactions > 0 else 0,
            'overall_tone_distribution': overall_tone_dist,
            'latest_health_status': latest_report.overall_status.value if latest_report else None,
            'latest_health_score': latest_report.overall_score if latest_report else None,
            'total_reports': len(self.health_reports)
        }