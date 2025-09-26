"""
Pattern Analyzer
감지된 패턴 분석 및 의미 추출
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from collections import Counter

from ...core.types import Result, create_success, create_failure
from ..auto.types import LearningPattern, PatternType, LearningCategory
from .detector import DetectionResult

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """분석 결과"""
    semantic_meaning: str
    importance_score: float
    actionable_insights: List[str]
    recommended_actions: List[str]
    confidence: float
    pattern_relationships: List[str]


class PatternAnalyzer:
    """
    패턴 분석 시스템
    감지된 패턴의 의미를 분석하고 실행 가능한 인사이트 생성
    """

    def __init__(self):
        self.semantic_rules = self._initialize_semantic_rules()
        self.action_templates = self._initialize_action_templates()

    def analyze_patterns(
        self,
        detection_results: List[DetectionResult],
        context: Optional[str] = None
    ) -> Result[List[AnalysisResult]]:
        """패턴 분석"""
        try:
            analysis_results = []

            for detection_result in detection_results:
                analysis = self._analyze_single_pattern(detection_result, context)
                if analysis:
                    analysis_results.append(analysis)

            # 패턴 간 관계 분석
            self._analyze_pattern_relationships(analysis_results)

            return create_success(analysis_results)

        except Exception as e:
            return create_failure(f"Pattern analysis failed: {str(e)}")

    def _analyze_single_pattern(
        self,
        detection_result: DetectionResult,
        context: Optional[str]
    ) -> Optional[AnalysisResult]:
        """단일 패턴 분석"""
        pattern = detection_result.pattern
        pattern_type = pattern.pattern_type

        # 의미 추출
        semantic_meaning = self._extract_semantic_meaning(detection_result, context)

        # 중요도 계산
        importance_score = self._calculate_importance_score(detection_result, context)

        # 실행 가능한 인사이트 생성
        actionable_insights = self._generate_actionable_insights(detection_result, context)

        # 권장 액션 생성
        recommended_actions = self._generate_recommended_actions(detection_result, context)

        # 분석 신뢰도 계산
        analysis_confidence = self._calculate_analysis_confidence(detection_result)

        return AnalysisResult(
            semantic_meaning=semantic_meaning,
            importance_score=importance_score,
            actionable_insights=actionable_insights,
            recommended_actions=recommended_actions,
            confidence=analysis_confidence,
            pattern_relationships=[]  # 나중에 채워짐
        )

    def _extract_semantic_meaning(
        self,
        detection_result: DetectionResult,
        context: Optional[str]
    ) -> str:
        """의미론적 의미 추출"""
        pattern_type = detection_result.pattern.pattern_type
        matched_keywords = detection_result.matched_keywords
        matched_contexts = detection_result.matched_contexts

        # 패턴 타입별 기본 의미
        base_meanings = {
            PatternType.SUCCESS: "성공적인 해결 방법이 발견됨",
            PatternType.FAILURE: "실패 패턴이 식별됨",
            PatternType.PREFERENCE: "사용자 선호도가 표현됨",
            PatternType.KNOWLEDGE: "새로운 지식이 습득됨",
            PatternType.PERFORMANCE: "성능 관련 이슈가 발견됨"
        }

        base_meaning = base_meanings.get(pattern_type, "패턴이 감지됨")

        # 키워드와 컨텍스트를 활용해 구체화
        specific_meaning = self._generate_specific_meaning(
            base_meaning, matched_keywords, matched_contexts, context
        )

        return specific_meaning

    def _generate_specific_meaning(
        self,
        base_meaning: str,
        keywords: List[str],
        contexts: List[str],
        global_context: Optional[str]
    ) -> str:
        """구체적인 의미 생성"""
        meaning_parts = [base_meaning]

        # 주요 키워드 추가
        if keywords:
            top_keywords = keywords[:3]  # 상위 3개만
            meaning_parts.append(f"핵심 요소: {', '.join(top_keywords)}")

        # 컨텍스트 정보 추가
        if contexts:
            primary_context = contexts[0]
            meaning_parts.append(f"분야: {primary_context}")

        # 글로벌 컨텍스트 고려
        if global_context:
            meaning_parts.append(f"상황: {global_context}")

        return " | ".join(meaning_parts)

    def _calculate_importance_score(
        self,
        detection_result: DetectionResult,
        context: Optional[str]
    ) -> float:
        """중요도 점수 계산"""
        base_score = detection_result.confidence

        # 패턴 타입별 가중치
        type_weights = {
            PatternType.SUCCESS: 0.9,
            PatternType.FAILURE: 0.8,
            PatternType.KNOWLEDGE: 0.7,
            PatternType.PREFERENCE: 0.6,
            PatternType.PERFORMANCE: 0.8
        }

        pattern_weight = type_weights.get(detection_result.pattern.pattern_type, 0.5)
        weighted_score = base_score * pattern_weight

        # 매칭된 키워드 수에 따른 보너스
        keyword_bonus = min(len(detection_result.matched_keywords) * 0.1, 0.3)

        # 컨텍스트 보너스
        context_bonus = 0.1 if context else 0.0

        final_score = weighted_score + keyword_bonus + context_bonus

        return min(final_score, 1.0)

    def _generate_actionable_insights(
        self,
        detection_result: DetectionResult,
        context: Optional[str]
    ) -> List[str]:
        """실행 가능한 인사이트 생성"""
        pattern_type = detection_result.pattern.pattern_type
        insights = []

        insight_generators = {
            PatternType.SUCCESS: self._generate_success_insights,
            PatternType.FAILURE: self._generate_failure_insights,
            PatternType.PREFERENCE: self._generate_preference_insights,
            PatternType.KNOWLEDGE: self._generate_knowledge_insights,
            PatternType.PERFORMANCE: self._generate_performance_insights
        }

        generator = insight_generators.get(pattern_type)
        if generator:
            insights = generator(detection_result, context)

        return insights

    def _generate_success_insights(
        self,
        detection_result: DetectionResult,
        context: Optional[str]
    ) -> List[str]:
        """성공 패턴 인사이트"""
        insights = [
            "성공한 방법을 향후 유사한 상황에서 재사용할 수 있습니다",
            "이 해결책을 지식 베이스에 저장하여 학습 효과를 높일 수 있습니다"
        ]

        keywords = detection_result.matched_keywords
        if "해결" in keywords:
            insights.append("문제 해결 능력이 향상되었음을 나타냅니다")
        if "성공" in keywords:
            insights.append("성공 경험을 통해 자신감이 증가했을 가능성이 높습니다")

        return insights

    def _generate_failure_insights(
        self,
        detection_result: DetectionResult,
        context: Optional[str]
    ) -> List[str]:
        """실패 패턴 인사이트"""
        insights = [
            "실패 원인을 분석하여 향후 동일한 실수를 방지할 수 있습니다",
            "이 실패 패턴을 회피 전략으로 활용할 수 있습니다"
        ]

        keywords = detection_result.matched_keywords
        if "오류" in keywords or "에러" in keywords:
            insights.append("기술적 오류 패턴이므로 시스템적 개선이 필요합니다")
        if "문제" in keywords:
            insights.append("근본적인 문제 해결 접근법을 재검토해야 합니다")

        return insights

    def _generate_preference_insights(
        self,
        detection_result: DetectionResult,
        context: Optional[str]
    ) -> List[str]:
        """선호도 패턴 인사이트"""
        insights = [
            "사용자의 작업 스타일과 선호도를 파악할 수 있습니다",
            "개인화된 추천과 제안을 제공할 수 있습니다"
        ]

        keywords = detection_result.matched_keywords
        if "좋아" in keywords:
            insights.append("긍정적 선호도이므로 해당 방식을 우선적으로 제안해야 합니다")
        if "싫어" in keywords:
            insights.append("부정적 선호도이므로 대안적 접근법을 찾아야 합니다")

        return insights

    def _generate_knowledge_insights(
        self,
        detection_result: DetectionResult,
        context: Optional[str]
    ) -> List[str]:
        """지식 패턴 인사이트"""
        insights = [
            "새로운 학습 기회가 식별되었습니다",
            "지식 격차를 메우기 위한 추가 학습이 필요합니다"
        ]

        keywords = detection_result.matched_keywords
        if "몰랐" in keywords:
            insights.append("지식 부족 영역이 발견되어 체계적 학습이 필요합니다")
        if "배웠" in keywords or "이해" in keywords:
            insights.append("학습 효과가 확인되어 관련 내용을 심화 학습할 수 있습니다")

        return insights

    def _generate_performance_insights(
        self,
        detection_result: DetectionResult,
        context: Optional[str]
    ) -> List[str]:
        """성능 패턴 인사이트"""
        insights = [
            "성능 최적화 기회가 발견되었습니다",
            "시스템 효율성 개선이 필요합니다"
        ]

        keywords = detection_result.matched_keywords
        if "느려" in keywords:
            insights.append("성능 병목이 식별되어 최적화 작업이 시급합니다")
        if "빨라" in keywords:
            insights.append("성능 개선이 확인되어 해당 방법을 다른 영역에도 적용할 수 있습니다")

        return insights

    def _generate_recommended_actions(
        self,
        detection_result: DetectionResult,
        context: Optional[str]
    ) -> List[str]:
        """권장 액션 생성"""
        pattern_type = detection_result.pattern.pattern_type
        actions = []

        action_generators = {
            PatternType.SUCCESS: lambda: [
                "성공 사례를 문서화하고 재사용 가능한 템플릿으로 만들기",
                "팀원들과 성공 방법 공유하기",
                "유사한 상황에서 이 방법을 우선적으로 고려하기"
            ],
            PatternType.FAILURE: lambda: [
                "실패 원인을 상세히 분석하고 문서화하기",
                "동일한 실수 방지를 위한 체크리스트 만들기",
                "대안적 접근법 연구하기"
            ],
            PatternType.PREFERENCE: lambda: [
                "선호도 정보를 사용자 프로필에 저장하기",
                "개인화된 추천 시스템에 반영하기",
                "선호도 기반 워크플로우 최적화하기"
            ],
            PatternType.KNOWLEDGE: lambda: [
                "관련 학습 자료 수집하기",
                "체계적인 학습 계획 수립하기",
                "전문가나 멘토에게 조언 구하기"
            ],
            PatternType.PERFORMANCE: lambda: [
                "성능 메트릭 모니터링 설정하기",
                "병목 지점 상세 분석하기",
                "최적화 우선순위 결정하기"
            ]
        }

        generator = action_generators.get(pattern_type)
        if generator:
            actions = generator()

        return actions

    def _calculate_analysis_confidence(self, detection_result: DetectionResult) -> float:
        """분석 신뢰도 계산"""
        base_confidence = detection_result.confidence

        # 매칭된 요소 수에 따른 보너스
        element_count = len(detection_result.matched_keywords) + len(detection_result.matched_contexts)
        element_bonus = min(element_count * 0.05, 0.2)

        # 패턴 신뢰도 임계값 대비 점수
        threshold_ratio = detection_result.confidence / detection_result.pattern.confidence_threshold
        threshold_bonus = min((threshold_ratio - 1.0) * 0.1, 0.1)

        final_confidence = base_confidence + element_bonus + threshold_bonus

        return min(final_confidence, 1.0)

    def _analyze_pattern_relationships(self, analysis_results: List[AnalysisResult]) -> None:
        """패턴 간 관계 분석"""
        if len(analysis_results) < 2:
            return

        for i, result1 in enumerate(analysis_results):
            relationships = []
            for j, result2 in enumerate(analysis_results):
                if i != j:
                    relationship = self._find_pattern_relationship(result1, result2)
                    if relationship:
                        relationships.append(relationship)

            result1.pattern_relationships = relationships

    def _find_pattern_relationship(
        self,
        result1: AnalysisResult,
        result2: AnalysisResult
    ) -> Optional[str]:
        """두 패턴 간의 관계 찾기"""
        # 의미적 유사성 검사
        if self._are_semantically_related(result1.semantic_meaning, result2.semantic_meaning):
            return f"의미적으로 연관됨: {result2.semantic_meaning[:50]}..."

        # 중요도 기반 관계
        if abs(result1.importance_score - result2.importance_score) < 0.1:
            return "유사한 중요도를 가짐"

        # 인사이트 중복성 검사
        common_insights = set(result1.actionable_insights) & set(result2.actionable_insights)
        if common_insights:
            return f"공통 인사이트 존재: {len(common_insights)}개"

        return None

    def _are_semantically_related(self, meaning1: str, meaning2: str) -> bool:
        """의미적 연관성 검사"""
        # 간단한 키워드 기반 유사성 검사
        words1 = set(meaning1.lower().split())
        words2 = set(meaning2.lower().split())

        common_words = words1 & words2
        total_words = words1 | words2

        similarity = len(common_words) / len(total_words) if total_words else 0.0

        return similarity > 0.3

    def _initialize_semantic_rules(self) -> Dict[str, Any]:
        """의미론적 규칙 초기화"""
        return {
            "success_indicators": ["해결", "성공", "완료", "좋아"],
            "failure_indicators": ["실패", "오류", "문제", "안됨"],
            "preference_indicators": ["좋아", "싫어", "선호", "원해"],
            "knowledge_indicators": ["몰랐", "배웠", "이해", "알았"],
            "performance_indicators": ["느려", "빨라", "성능", "효율"]
        }

    def _initialize_action_templates(self) -> Dict[str, List[str]]:
        """액션 템플릿 초기화"""
        return {
            "documentation": [
                "관련 내용을 문서화하기",
                "지식 베이스에 추가하기",
                "팀과 공유하기"
            ],
            "prevention": [
                "재발 방지 계획 수립하기",
                "체크리스트 만들기",
                "모니터링 설정하기"
            ],
            "optimization": [
                "개선 방안 연구하기",
                "최적화 우선순위 정하기",
                "성능 측정하기"
            ]
        }