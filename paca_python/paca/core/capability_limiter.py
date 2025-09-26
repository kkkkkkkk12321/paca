"""
Capability Limiter Module
점진적 기능 저하 시스템 - 능력 한계 도달시 환각 대신 정직한 한계 인정
"""

import asyncio
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import math

# 조건부 임포트
try:
    from .types.base import (
        ID, Timestamp, Result, current_timestamp, generate_id, create_success, create_failure
    )
except ImportError:
    from paca.core.types.base import (
        ID, Timestamp, Result, current_timestamp, generate_id, create_success, create_failure
    )


class CapabilityLevel(Enum):
    """능력 수준"""
    FULL = "full"               # 완전한 능력 (90-100%)
    HIGH = "high"               # 높은 능력 (70-89%)
    MEDIUM = "medium"           # 중간 능력 (50-69%)
    LOW = "low"                 # 낮은 능력 (30-49%)
    MINIMAL = "minimal"         # 최소 능력 (10-29%)
    INSUFFICIENT = "insufficient"  # 능력 부족 (0-9%)


class DegradationStrategy(Enum):
    """기능 저하 전략"""
    HONEST_ADMISSION = "honest_admission"      # 정직한 한계 인정
    PARTIAL_HELP = "partial_help"             # 부분적 도움 제공
    RESOURCE_PROVISION = "resource_provision"  # 자료 제공
    GUIDANCE_ONLY = "guidance_only"           # 안내만 제공
    GRACEFUL_DECLINE = "graceful_decline"     # 정중한 거절


class ComplexityDomain(Enum):
    """복잡도 영역"""
    MATHEMATICAL = "mathematical"          # 수학적 추론
    SCIENTIFIC = "scientific"             # 과학적 분석
    TECHNICAL = "technical"               # 기술적 문제
    CREATIVE = "creative"                 # 창작 작업
    LINGUISTIC = "linguistic"             # 언어 처리
    LOGICAL = "logical"                   # 논리적 추론
    SOCIAL = "social"                     # 사회적 상황
    ETHICAL = "ethical"                   # 윤리적 판단


@dataclass
class CapabilityAssessment:
    """능력 평가"""
    assessment_id: ID
    query: str
    domain: ComplexityDomain
    complexity_score: float  # 0-1
    required_capability: CapabilityLevel
    available_capability: CapabilityLevel
    confidence_level: float  # 0-1
    assessment_reasons: List[str]
    timestamp: Timestamp = field(default_factory=current_timestamp)

    @property
    def capability_gap(self) -> float:
        """능력 격차"""
        required_value = self._capability_to_value(self.required_capability)
        available_value = self._capability_to_value(self.available_capability)
        return max(0, required_value - available_value)

    @property
    def can_handle(self) -> bool:
        """처리 가능 여부"""
        return self.available_capability.value != CapabilityLevel.INSUFFICIENT.value

    def _capability_to_value(self, capability: CapabilityLevel) -> float:
        """능력 수준을 수치로 변환"""
        mapping = {
            CapabilityLevel.INSUFFICIENT: 0.05,
            CapabilityLevel.MINIMAL: 0.2,
            CapabilityLevel.LOW: 0.4,
            CapabilityLevel.MEDIUM: 0.6,
            CapabilityLevel.HIGH: 0.8,
            CapabilityLevel.FULL: 0.95
        }
        return mapping.get(capability, 0.0)


@dataclass
class DegradationResponse:
    """기능 저하 응답"""
    response_id: ID
    original_query: str
    assessment: CapabilityAssessment
    strategy: DegradationStrategy
    response_message: str
    alternative_suggestions: List[str]
    resource_links: List[str]
    confidence_disclaimer: str
    timestamp: Timestamp = field(default_factory=current_timestamp)


@dataclass
class CapabilityRecord:
    """능력 기록"""
    record_id: ID
    domain: ComplexityDomain
    successful_complexity: float
    failed_complexity: float
    total_attempts: int
    success_rate: float
    last_updated: Timestamp = field(default_factory=current_timestamp)


class CapabilityEvaluator:
    """능력 범위 평가 시스템"""

    def __init__(self):
        self.capability_records: Dict[ComplexityDomain, CapabilityRecord] = {}
        self.assessment_history: List[CapabilityAssessment] = []

        # 복잡도 분석 패턴
        self.complexity_indicators = {
            ComplexityDomain.MATHEMATICAL: {
                'keywords': ['계산', '수학', '공식', '방정식', '미적분', '통계', '확률'],
                'patterns': [r'\d+[\+\-\*/]\d+', r'[xy]\s*=', r'∫|∑|∂'],
                'base_complexity': 0.3
            },
            ComplexityDomain.SCIENTIFIC: {
                'keywords': ['실험', '가설', '이론', '분석', '연구', '과학'],
                'patterns': [r'실험\s*설계', r'가설\s*검증'],
                'base_complexity': 0.4
            },
            ComplexityDomain.TECHNICAL: {
                'keywords': ['프로그래밍', '코딩', '알고리즘', '시스템', '아키텍처'],
                'patterns': [r'def\s+\w+', r'class\s+\w+', r'import\s+\w+'],
                'base_complexity': 0.5
            },
            ComplexityDomain.CREATIVE: {
                'keywords': ['창작', '소설', '시', '디자인', '아이디어'],
                'patterns': [r'창의적', r'독창적'],
                'base_complexity': 0.6
            },
            ComplexityDomain.LINGUISTIC: {
                'keywords': ['번역', '문법', '언어', '의미'],
                'patterns': [r'번역해', r'문법\s*검사'],
                'base_complexity': 0.3
            },
            ComplexityDomain.LOGICAL: {
                'keywords': ['논리', '추론', '증명', '결론'],
                'patterns': [r'따라서', r'그러므로', r'∴'],
                'base_complexity': 0.4
            },
            ComplexityDomain.SOCIAL: {
                'keywords': ['관계', '소통', '갈등', '협상'],
                'patterns': [r'사람들과', r'인간관계'],
                'base_complexity': 0.7
            },
            ComplexityDomain.ETHICAL: {
                'keywords': ['윤리', '도덕', '옳고', '그름'],
                'patterns': [r'윤리적', r'도덕적'],
                'base_complexity': 0.8
            }
        }

        # 기본 능력 수준 설정
        self.base_capabilities = {
            ComplexityDomain.MATHEMATICAL: CapabilityLevel.MEDIUM,
            ComplexityDomain.SCIENTIFIC: CapabilityLevel.HIGH,
            ComplexityDomain.TECHNICAL: CapabilityLevel.HIGH,
            ComplexityDomain.CREATIVE: CapabilityLevel.MEDIUM,
            ComplexityDomain.LINGUISTIC: CapabilityLevel.HIGH,
            ComplexityDomain.LOGICAL: CapabilityLevel.HIGH,
            ComplexityDomain.SOCIAL: CapabilityLevel.MEDIUM,
            ComplexityDomain.ETHICAL: CapabilityLevel.LOW
        }

        self.logger = logging.getLogger(__name__)

    async def assess_capability(self, query: str) -> CapabilityAssessment:
        """쿼리에 대한 능력 평가"""
        try:
            # 1. 도메인 식별
            domain = self._identify_domain(query)

            # 2. 복잡도 평가
            complexity_score = self._calculate_complexity(query, domain)

            # 3. 필요 능력 수준 결정
            required_capability = self._determine_required_capability(complexity_score)

            # 4. 사용 가능한 능력 평가
            available_capability = self._assess_available_capability(domain, complexity_score)

            # 5. 신뢰도 계산
            confidence_level = self._calculate_confidence(domain, complexity_score, available_capability)

            # 6. 평가 이유 생성
            assessment_reasons = self._generate_assessment_reasons(
                domain, complexity_score, required_capability, available_capability
            )

            assessment = CapabilityAssessment(
                assessment_id=generate_id("assess_"),
                query=query,
                domain=domain,
                complexity_score=complexity_score,
                required_capability=required_capability,
                available_capability=available_capability,
                confidence_level=confidence_level,
                assessment_reasons=assessment_reasons
            )

            # 평가 기록 저장
            self.assessment_history.append(assessment)
            if len(self.assessment_history) > 1000:
                self.assessment_history = self.assessment_history[-1000:]

            return assessment

        except Exception as e:
            self.logger.error(f"능력 평가 실패: {e}")
            # 기본 평가 반환
            return CapabilityAssessment(
                assessment_id=generate_id("assess_error_"),
                query=query,
                domain=ComplexityDomain.LOGICAL,
                complexity_score=0.5,
                required_capability=CapabilityLevel.MEDIUM,
                available_capability=CapabilityLevel.LOW,
                confidence_level=0.1,
                assessment_reasons=["평가 중 오류 발생"]
            )

    def _identify_domain(self, query: str) -> ComplexityDomain:
        """도메인 식별"""
        query_lower = query.lower()
        domain_scores = {}

        for domain, indicators in self.complexity_indicators.items():
            score = 0

            # 키워드 매칭
            for keyword in indicators['keywords']:
                if keyword in query_lower:
                    score += 1

            # 패턴 매칭
            for pattern in indicators['patterns']:
                if re.search(pattern, query_lower):
                    score += 2

            domain_scores[domain] = score

        # 가장 높은 점수의 도메인 반환
        if any(domain_scores.values()):
            return max(domain_scores, key=domain_scores.get)

        # 기본값
        return ComplexityDomain.LOGICAL

    def _calculate_complexity(self, query: str, domain: ComplexityDomain) -> float:
        """복잡도 계산"""
        base_complexity = self.complexity_indicators[domain]['base_complexity']

        # 길이 기반 복잡도
        length_complexity = min(len(query) / 1000, 0.3)

        # 특수 패턴 복잡도
        pattern_complexity = 0
        for pattern in self.complexity_indicators[domain]['patterns']:
            if re.search(pattern, query):
                pattern_complexity += 0.1

        # 복합 질문 복잡도
        question_marks = query.count('?')
        compound_complexity = min(question_marks * 0.1, 0.2)

        # 전문 용어 복잡도
        technical_terms = len([word for word in query.split() if len(word) > 8])
        technical_complexity = min(technical_terms * 0.05, 0.2)

        total_complexity = (
            base_complexity +
            length_complexity +
            pattern_complexity +
            compound_complexity +
            technical_complexity
        )

        return min(total_complexity, 1.0)

    def _determine_required_capability(self, complexity_score: float) -> CapabilityLevel:
        """필요 능력 수준 결정"""
        if complexity_score >= 0.9:
            return CapabilityLevel.FULL
        elif complexity_score >= 0.7:
            return CapabilityLevel.HIGH
        elif complexity_score >= 0.5:
            return CapabilityLevel.MEDIUM
        elif complexity_score >= 0.3:
            return CapabilityLevel.LOW
        elif complexity_score >= 0.1:
            return CapabilityLevel.MINIMAL
        else:
            return CapabilityLevel.INSUFFICIENT

    def _assess_available_capability(self, domain: ComplexityDomain, complexity_score: float) -> CapabilityLevel:
        """사용 가능한 능력 평가"""
        base_capability = self.base_capabilities.get(domain, CapabilityLevel.LOW)
        base_value = self._capability_to_value(base_capability)

        # 과거 성과 기반 조정
        if domain in self.capability_records:
            record = self.capability_records[domain]
            performance_adjustment = (record.success_rate - 0.5) * 0.2
            adjusted_value = base_value + performance_adjustment
        else:
            adjusted_value = base_value

        # 복잡도 기반 능력 감소
        complexity_penalty = complexity_score * 0.3
        final_value = max(0, adjusted_value - complexity_penalty)

        return self._value_to_capability(final_value)

    def _calculate_confidence(self, domain: ComplexityDomain, complexity_score: float,
                           available_capability: CapabilityLevel) -> float:
        """신뢰도 계산"""
        # 기본 신뢰도
        base_confidence = 0.7

        # 도메인별 신뢰도 조정
        domain_confidence_modifiers = {
            ComplexityDomain.MATHEMATICAL: -0.2,
            ComplexityDomain.SCIENTIFIC: 0.1,
            ComplexityDomain.TECHNICAL: 0.1,
            ComplexityDomain.CREATIVE: -0.1,
            ComplexityDomain.LINGUISTIC: 0.2,
            ComplexityDomain.LOGICAL: 0.1,
            ComplexityDomain.SOCIAL: -0.2,
            ComplexityDomain.ETHICAL: -0.3
        }

        domain_modifier = domain_confidence_modifiers.get(domain, 0)

        # 복잡도 기반 신뢰도 감소
        complexity_penalty = complexity_score * 0.4

        # 능력 수준 기반 조정
        capability_value = self._capability_to_value(available_capability)
        capability_bonus = capability_value * 0.3

        final_confidence = base_confidence + domain_modifier - complexity_penalty + capability_bonus
        return max(0.01, min(final_confidence, 0.99))

    def _generate_assessment_reasons(self, domain: ComplexityDomain, complexity_score: float,
                                   required_capability: CapabilityLevel,
                                   available_capability: CapabilityLevel) -> List[str]:
        """평가 이유 생성"""
        reasons = []

        # 도메인 관련
        reasons.append(f"질문이 {domain.value} 영역에 속함")

        # 복잡도 관련
        if complexity_score >= 0.7:
            reasons.append("높은 복잡도로 인해 처리가 어려움")
        elif complexity_score >= 0.5:
            reasons.append("중간 수준의 복잡도")
        else:
            reasons.append("비교적 단순한 문제")

        # 능력 격차 관련
        required_value = self._capability_to_value(required_capability)
        available_value = self._capability_to_value(available_capability)
        gap = required_value - available_value

        if gap > 0.3:
            reasons.append("필요한 능력 수준이 현재 능력을 크게 초과함")
        elif gap > 0.1:
            reasons.append("필요한 능력 수준이 현재 능력을 다소 초과함")
        elif gap < -0.1:
            reasons.append("현재 능력으로 충분히 처리 가능")

        # 도메인별 특수 사항
        if domain == ComplexityDomain.ETHICAL:
            reasons.append("윤리적 판단은 주관적이며 신중한 접근이 필요")
        elif domain == ComplexityDomain.CREATIVE:
            reasons.append("창작 작업은 개인의 취향과 창의성이 중요")

        return reasons

    def _capability_to_value(self, capability: CapabilityLevel) -> float:
        """능력 수준을 수치로 변환"""
        mapping = {
            CapabilityLevel.INSUFFICIENT: 0.05,
            CapabilityLevel.MINIMAL: 0.2,
            CapabilityLevel.LOW: 0.4,
            CapabilityLevel.MEDIUM: 0.6,
            CapabilityLevel.HIGH: 0.8,
            CapabilityLevel.FULL: 0.95
        }
        return mapping.get(capability, 0.0)

    def _value_to_capability(self, value: float) -> CapabilityLevel:
        """수치를 능력 수준으로 변환"""
        if value >= 0.9:
            return CapabilityLevel.FULL
        elif value >= 0.7:
            return CapabilityLevel.HIGH
        elif value >= 0.5:
            return CapabilityLevel.MEDIUM
        elif value >= 0.3:
            return CapabilityLevel.LOW
        elif value >= 0.1:
            return CapabilityLevel.MINIMAL
        else:
            return CapabilityLevel.INSUFFICIENT

    def update_capability_record(self, domain: ComplexityDomain, complexity: float, success: bool) -> None:
        """능력 기록 업데이트"""
        if domain not in self.capability_records:
            self.capability_records[domain] = CapabilityRecord(
                record_id=generate_id("record_"),
                domain=domain,
                successful_complexity=0.0,
                failed_complexity=0.0,
                total_attempts=0,
                success_rate=0.5
            )

        record = self.capability_records[domain]
        record.total_attempts += 1

        if success:
            record.successful_complexity = max(record.successful_complexity, complexity)
        else:
            record.failed_complexity = max(record.failed_complexity, complexity)

        # 성공률 업데이트 (지수 이동 평균)
        alpha = 0.1
        new_value = 1.0 if success else 0.0
        record.success_rate = (1 - alpha) * record.success_rate + alpha * new_value
        record.last_updated = current_timestamp()

    def get_capability_statistics(self) -> Dict[str, Any]:
        """능력 통계"""
        return {
            'total_assessments': len(self.assessment_history),
            'domain_records': {
                domain.value: {
                    'successful_complexity': record.successful_complexity,
                    'failed_complexity': record.failed_complexity,
                    'total_attempts': record.total_attempts,
                    'success_rate': record.success_rate
                }
                for domain, record in self.capability_records.items()
            },
            'recent_assessments': [
                {
                    'domain': assessment.domain.value,
                    'complexity': assessment.complexity_score,
                    'required_capability': assessment.required_capability.value,
                    'available_capability': assessment.available_capability.value,
                    'confidence': assessment.confidence_level,
                    'can_handle': assessment.can_handle
                }
                for assessment in self.assessment_history[-10:]
            ]
        }


class HumilityResponse:
    """겸손한 한계 인정 응답 생성"""

    def __init__(self):
        self.response_templates = {
            DegradationStrategy.HONEST_ADMISSION: [
                "죄송하지만 이 문제는 제 능력을 벗어납니다.",
                "이 질문은 제가 자신 있게 답변하기에는 너무 복잡합니다.",
                "제 현재 능력으로는 이 문제를 완전히 해결할 수 없습니다."
            ],
            DegradationStrategy.PARTIAL_HELP: [
                "완전한 답변은 어렵지만, 다음과 같은 부분적인 도움을 드릴 수 있습니다:",
                "전체 해결책은 제공하기 어렵지만, 접근 방법에 대해서는 말씀드릴 수 있습니다:",
                "완전한 분석은 어렵지만, 기본적인 관점에서 도움을 드릴 수 있습니다:"
            ],
            DegradationStrategy.RESOURCE_PROVISION: [
                "제가 직접 답변하기는 어렵지만, 다음 자료들이 도움이 될 것 같습니다:",
                "이 문제에 대해서는 전문가의 도움이 필요할 것 같습니다. 다음 리소스를 참고해보세요:",
                "제 능력 밖의 문제이지만, 관련 정보를 찾을 수 있는 곳을 안내해드릴 수 있습니다:"
            ],
            DegradationStrategy.GUIDANCE_ONLY: [
                "직접적인 답변은 어렵지만, 문제 해결을 위한 방향을 제시해드릴 수 있습니다:",
                "구체적인 해답은 제공하기 어렵지만, 접근 방법에 대한 안내는 가능합니다:",
                "정확한 답변 대신, 문제를 풀어나가는 과정에 대해 설명드릴 수 있습니다:"
            ],
            DegradationStrategy.GRACEFUL_DECLINE: [
                "이 문제는 전문가의 도움이 필요한 영역입니다.",
                "제 전문성을 벗어나는 분야이므로, 해당 전문가에게 문의하시는 것이 좋겠습니다.",
                "이런 복잡한 문제는 저보다는 관련 전문가가 더 적절한 도움을 줄 수 있을 것입니다."
            ]
        }

        self.confidence_disclaimers = [
            "이 답변의 정확성에 대해서는 확신할 수 없습니다.",
            "제한된 정보를 바탕으로 한 답변이므로 추가 확인이 필요합니다.",
            "이는 일반적인 관점에서의 의견이며, 전문적 조언은 아닙니다.",
            "불확실한 부분이 있으므로 다른 출처로도 확인해보시기 바랍니다."
        ]

        self.alternative_suggestions = {
            ComplexityDomain.MATHEMATICAL: [
                "수학 전문가나 교수에게 문의해보세요",
                "울프램 알파나 매스매티카 같은 수학 소프트웨어를 사용해보세요",
                "수학 포럼이나 커뮤니티에서 도움을 요청해보세요"
            ],
            ComplexityDomain.SCIENTIFIC: [
                "해당 분야의 연구자나 전문가에게 문의해보세요",
                "관련 학술 논문을 찾아보세요",
                "과학 전문 포럼이나 학회에 질문해보세요"
            ],
            ComplexityDomain.TECHNICAL: [
                "해당 기술의 공식 문서를 참조해보세요",
                "기술 커뮤니티나 포럼에서 도움을 요청해보세요",
                "관련 전문가나 개발자에게 문의해보세요"
            ],
            ComplexityDomain.CREATIVE: [
                "창작 전문가나 아티스트에게 조언을 구해보세요",
                "관련 창작 커뮤니티에서 아이디어를 얻어보세요",
                "다양한 작품들을 참고하여 영감을 얻어보세요"
            ],
            ComplexityDomain.ETHICAL: [
                "윤리학 전문가나 철학자에게 문의해보세요",
                "관련 윤리 위원회나 기관에 상담을 요청해보세요",
                "다양한 윤리적 관점을 제시하는 자료들을 참고해보세요"
            ]
        }

    def generate_degraded_response(self, assessment: CapabilityAssessment) -> DegradationResponse:
        """기능 저하 응답 생성"""
        try:
            # 전략 결정
            strategy = self._select_strategy(assessment)

            # 응답 메시지 생성
            response_message = self._generate_response_message(assessment, strategy)

            # 대안 제안 생성
            alternative_suggestions = self._generate_alternatives(assessment)

            # 자료 링크 (현재는 placeholder)
            resource_links = []

            # 신뢰도 면책 조항
            confidence_disclaimer = self._select_disclaimer(assessment.confidence_level)

            return DegradationResponse(
                response_id=generate_id("response_"),
                original_query=assessment.query,
                assessment=assessment,
                strategy=strategy,
                response_message=response_message,
                alternative_suggestions=alternative_suggestions,
                resource_links=resource_links,
                confidence_disclaimer=confidence_disclaimer
            )

        except Exception as e:
            # 기본 응답 생성
            return DegradationResponse(
                response_id=generate_id("response_error_"),
                original_query=assessment.query,
                assessment=assessment,
                strategy=DegradationStrategy.HONEST_ADMISSION,
                response_message="죄송하지만 이 문제를 처리하는 중 오류가 발생했습니다.",
                alternative_suggestions=["다시 시도해보시거나 다른 방법으로 질문해주세요."],
                resource_links=[],
                confidence_disclaimer="답변 생성 중 기술적 문제가 발생했습니다."
            )

    def _select_strategy(self, assessment: CapabilityAssessment) -> DegradationStrategy:
        """전략 선택"""
        capability_gap = assessment.capability_gap

        if capability_gap >= 0.5:
            # 큰 격차: 정중한 거절
            return DegradationStrategy.GRACEFUL_DECLINE
        elif capability_gap >= 0.3:
            # 중간 격차: 자료 제공
            return DegradationStrategy.RESOURCE_PROVISION
        elif capability_gap >= 0.2:
            # 작은 격차: 안내만 제공
            return DegradationStrategy.GUIDANCE_ONLY
        elif capability_gap >= 0.1:
            # 매우 작은 격차: 부분적 도움
            return DegradationStrategy.PARTIAL_HELP
        else:
            # 격차 없음: 정직한 인정 (신뢰도가 낮은 경우)
            if assessment.confidence_level < 0.5:
                return DegradationStrategy.HONEST_ADMISSION
            else:
                return DegradationStrategy.PARTIAL_HELP

    def _generate_response_message(self, assessment: CapabilityAssessment,
                                 strategy: DegradationStrategy) -> str:
        """응답 메시지 생성"""
        templates = self.response_templates.get(strategy, self.response_templates[DegradationStrategy.HONEST_ADMISSION])
        base_message = templates[0]  # 첫 번째 템플릿 사용

        # 맥락화된 메시지 추가
        context_additions = []

        if assessment.domain == ComplexityDomain.MATHEMATICAL:
            context_additions.append("수학적 계산이나 증명")
        elif assessment.domain == ComplexityDomain.SCIENTIFIC:
            context_additions.append("과학적 분석이나 연구")
        elif assessment.domain == ComplexityDomain.ETHICAL:
            context_additions.append("윤리적 판단이나 도덕적 딜레마")

        if assessment.complexity_score >= 0.8:
            context_additions.append("매우 복잡한 문제")
        elif assessment.complexity_score >= 0.6:
            context_additions.append("상당히 복잡한 문제")

        if context_additions:
            context_info = f" 특히 {', '.join(context_additions)}에 관한 질문은 제 전문성을 벗어납니다."
            base_message += context_info

        return base_message

    def _generate_alternatives(self, assessment: CapabilityAssessment) -> List[str]:
        """대안 제안 생성"""
        suggestions = []

        # 도메인별 기본 제안
        domain_suggestions = self.alternative_suggestions.get(assessment.domain, [
            "해당 분야의 전문가에게 문의해보세요",
            "관련 전문 자료나 문서를 참고해보세요"
        ])
        suggestions.extend(domain_suggestions)

        # 복잡도별 추가 제안
        if assessment.complexity_score >= 0.7:
            suggestions.append("문제를 더 작은 단위로 나누어 접근해보세요")
            suggestions.append("단계별로 차근차근 해결해나가는 것이 좋겠습니다")

        # 일반적인 제안
        suggestions.extend([
            "다른 AI 도구나 전문 소프트웨어를 활용해보세요",
            "관련 커뮤니티나 포럼에서 도움을 요청해보세요"
        ])

        return suggestions[:4]  # 최대 4개까지

    def _select_disclaimer(self, confidence_level: float) -> str:
        """면책 조항 선택"""
        if confidence_level < 0.3:
            return self.confidence_disclaimers[0]
        elif confidence_level < 0.5:
            return self.confidence_disclaimers[1]
        elif confidence_level < 0.7:
            return self.confidence_disclaimers[2]
        else:
            return self.confidence_disclaimers[3]


class GracefulDegradation:
    """점진적 기능 저하 메인 시스템"""

    def __init__(self):
        self.capability_assessor = CapabilityEvaluator()
        self.humility_responder = HumilityResponse()
        self.degradation_history: List[DegradationResponse] = []
        self.logger = logging.getLogger(__name__)

    async def process_query(self, query: str) -> Tuple[bool, Optional[DegradationResponse]]:
        """쿼리 처리 - 기능 저하 필요성 판단"""
        try:
            # 능력 평가
            assessment = await self.capability_assessor.assess_capability(query)

            # 처리 가능 여부 판단
            if assessment.can_handle and assessment.confidence_level >= 0.5:
                # 정상 처리 가능
                return True, None

            # 기능 저하 응답 생성
            degraded_response = self.humility_responder.generate_degraded_response(assessment)

            # 기록 저장
            self.degradation_history.append(degraded_response)
            if len(self.degradation_history) > 500:
                self.degradation_history = self.degradation_history[-500:]

            self.logger.info(f"기능 저하 응답 생성: {degraded_response.strategy.value}")
            return False, degraded_response

        except Exception as e:
            self.logger.error(f"쿼리 처리 실패: {e}")
            # 안전한 기본 응답
            return False, None

    async def update_performance(self, query: str, success: bool) -> None:
        """성능 기록 업데이트"""
        try:
            assessment = await self.capability_assessor.assess_capability(query)
            self.capability_assessor.update_capability_record(
                assessment.domain, assessment.complexity_score, success
            )
        except Exception as e:
            self.logger.error(f"성능 기록 업데이트 실패: {e}")

    def get_degradation_statistics(self) -> Dict[str, Any]:
        """기능 저하 통계"""
        if not self.degradation_history:
            return {'total_degradations': 0}

        # 전략별 분포
        strategy_counts = {}
        for strategy in DegradationStrategy:
            strategy_counts[strategy.value] = sum(
                1 for response in self.degradation_history
                if response.strategy == strategy
            )

        # 도메인별 분포
        domain_counts = {}
        for domain in ComplexityDomain:
            domain_counts[domain.value] = sum(
                1 for response in self.degradation_history
                if response.assessment.domain == domain
            )

        # 평균 복잡도
        avg_complexity = sum(r.assessment.complexity_score for r in self.degradation_history) / len(self.degradation_history)

        return {
            'total_degradations': len(self.degradation_history),
            'strategy_distribution': strategy_counts,
            'domain_distribution': domain_counts,
            'average_complexity': avg_complexity,
            'capability_statistics': self.capability_assessor.get_capability_statistics(),
            'recent_degradations': [
                {
                    'query': response.original_query[:100] + "..." if len(response.original_query) > 100 else response.original_query,
                    'domain': response.assessment.domain.value,
                    'complexity': response.assessment.complexity_score,
                    'strategy': response.strategy.value,
                    'timestamp': response.timestamp
                }
                for response in self.degradation_history[-10:]
            ]
        }


# 전역 인스턴스 (싱글톤 패턴)
_graceful_degradation_instance = None


def get_graceful_degradation() -> GracefulDegradation:
    """점진적 기능 저하 시스템 싱글톤 인스턴스 획득"""
    global _graceful_degradation_instance
    if _graceful_degradation_instance is None:
        _graceful_degradation_instance = GracefulDegradation()
    return _graceful_degradation_instance