"""
Integrity Monitor Module
지적 무결성 모니터링 시스템

이 모듈은 PACA의 지적 무결성을 실시간으로 모니터링하고 평가합니다:
- 논리적 일관성 검사
- 사실 정확성 검증
- 편향 감지 및 보정
- 투명성 및 설명가능성 평가
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
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


class IntegrityLevel(Enum):
    """무결성 수준"""
    CRITICAL = "critical"     # 심각한 무결성 위반
    LOW = "low"              # 낮은 무결성
    MODERATE = "moderate"     # 보통 무결성
    HIGH = "high"            # 높은 무결성
    EXCELLENT = "excellent"   # 탁월한 무결성


class ViolationType(Enum):
    """위반 타입"""
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    FACTUAL_ERROR = "factual_error"
    BIAS_DETECTED = "bias_detected"
    TRANSPARENCY_VIOLATION = "transparency_violation"
    OVERCONFIDENCE = "overconfidence"
    UNDERCONFIDENCE = "underconfidence"
    CIRCULAR_REASONING = "circular_reasoning"
    FALSE_CORRELATION = "false_correlation"


class BiasType(Enum):
    """편향 타입"""
    CONFIRMATION_BIAS = "confirmation_bias"
    AVAILABILITY_BIAS = "availability_bias"
    ANCHORING_BIAS = "anchoring_bias"
    RECENCY_BIAS = "recency_bias"
    AUTHORITY_BIAS = "authority_bias"
    CULTURAL_BIAS = "cultural_bias"


@dataclass(frozen=True)
class IntegrityViolation:
    """무결성 위반"""
    violation_id: ID
    violation_type: ViolationType
    severity: float  # 0.0-1.0
    description: str
    context: Dict[str, Any]
    evidence: List[str]
    timestamp: Timestamp
    auto_corrected: bool = False
    correction_suggestion: Optional[str] = None


@dataclass(frozen=True)
class BiasDetection:
    """편향 감지 결과"""
    detection_id: ID
    bias_type: BiasType
    confidence: float  # 0.0-1.0
    affected_content: str
    evidence_markers: List[str]
    mitigation_suggestion: str
    timestamp: Timestamp


@dataclass(frozen=True)
class IntegrityReport:
    """무결성 보고서"""
    report_id: ID
    overall_score: float  # 0.0-1.0
    integrity_level: IntegrityLevel
    component_scores: Dict[str, float]
    violations: List[IntegrityViolation]
    bias_detections: List[BiasDetection]
    recommendations: List[str]
    timestamp: Timestamp


class LogicalConsistencyChecker:
    """논리적 일관성 검사기"""

    def __init__(self):
        self.contradiction_patterns = [
            r'(?i)always.*never',
            r'(?i)impossible.*possible',
            r'(?i)certain.*uncertain',
            r'(?i)true.*false.*same.*context'
        ]
        self.logical_fallacy_patterns = {
            'circular_reasoning': r'(?i)because.*because|due to.*due to',
            'false_dichotomy': r'(?i)either.*or.*only|must be.*or.*nothing else',
            'ad_hominem': r'(?i)you are wrong because you',
            'straw_man': r'(?i)you claim that.*but that means'
        }

    async def check_consistency(self, content: str, context: Dict[str, Any]) -> Result[List[IntegrityViolation]]:
        """논리적 일관성 검사"""
        try:
            violations = []

            # 모순 패턴 검사
            contradictions = await self._detect_contradictions(content)
            violations.extend(contradictions)

            # 논리적 오류 검사
            fallacies = await self._detect_logical_fallacies(content)
            violations.extend(fallacies)

            # 추론 체인 검사
            reasoning_violations = await self._check_reasoning_chain(content, context)
            violations.extend(reasoning_violations)

            return create_success(violations)

        except Exception as e:
            return create_failure(e)

    async def _detect_contradictions(self, content: str) -> List[IntegrityViolation]:
        """모순 감지"""
        violations = []

        for pattern in self.contradiction_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                violation = IntegrityViolation(
                    violation_id=generate_id("violation_"),
                    violation_type=ViolationType.LOGICAL_INCONSISTENCY,
                    severity=0.7,
                    description=f"논리적 모순 감지: {match.group()}",
                    context={'pattern': pattern, 'match': match.group(), 'position': match.span()},
                    evidence=[match.group()],
                    timestamp=current_timestamp(),
                    correction_suggestion="모순되는 표현을 제거하거나 조건을 명확히 하세요"
                )
                violations.append(violation)

        return violations

    async def _detect_logical_fallacies(self, content: str) -> List[IntegrityViolation]:
        """논리적 오류 감지"""
        violations = []

        for fallacy_type, pattern in self.logical_fallacy_patterns.items():
            matches = re.finditer(pattern, content)
            for match in matches:
                violation = IntegrityViolation(
                    violation_id=generate_id("violation_"),
                    violation_type=ViolationType.LOGICAL_INCONSISTENCY,
                    severity=0.6,
                    description=f"논리적 오류 감지: {fallacy_type}",
                    context={'fallacy_type': fallacy_type, 'match': match.group()},
                    evidence=[match.group()],
                    timestamp=current_timestamp(),
                    correction_suggestion=f"{fallacy_type} 오류를 피하고 더 객관적인 논증을 사용하세요"
                )
                violations.append(violation)

        return violations

    async def _check_reasoning_chain(self, content: str, context: Dict[str, Any]) -> List[IntegrityViolation]:
        """추론 체인 검사"""
        violations = []

        # 결론 지시어 찾기
        conclusion_markers = ['therefore', 'thus', 'hence', 'so', 'consequently']
        premise_markers = ['because', 'since', 'given that', 'due to']

        conclusions = []
        premises = []

        content_lower = content.lower()
        for marker in conclusion_markers:
            if marker in content_lower:
                conclusions.append(marker)

        for marker in premise_markers:
            if marker in content_lower:
                premises.append(marker)

        # 결론이 있지만 전제가 부족한 경우
        if conclusions and len(premises) < len(conclusions):
            violation = IntegrityViolation(
                violation_id=generate_id("violation_"),
                violation_type=ViolationType.LOGICAL_INCONSISTENCY,
                severity=0.5,
                description="결론에 비해 부족한 전제",
                context={'conclusions': conclusions, 'premises': premises},
                evidence=[f"결론 표시어: {conclusions}, 전제 표시어: {premises}"],
                timestamp=current_timestamp(),
                correction_suggestion="결론을 뒷받침하는 충분한 전제와 근거를 제시하세요"
            )
            violations.append(violation)

        return violations


class FactualAccuracyChecker:
    """사실 정확성 검사기"""

    def __init__(self):
        self.uncertainty_indicators = {
            'high': ['definitely', 'certainly', 'absolutely', 'guaranteed'],
            'medium': ['likely', 'probably', 'usually', 'generally'],
            'low': ['maybe', 'possibly', 'might', 'could be']
        }
        self.fact_claim_patterns = [
            r'(?i)studies show',
            r'(?i)research indicates',
            r'(?i)according to.*statistics',
            r'(?i)\d+% of.*'
        ]

    async def check_factual_accuracy(self, content: str, context: Dict[str, Any]) -> Result[List[IntegrityViolation]]:
        """사실 정확성 검사"""
        try:
            violations = []

            # 과도한 확신 표현 검사
            overconfidence_violations = await self._detect_overconfidence(content)
            violations.extend(overconfidence_violations)

            # 사실 주장 검증 필요 감지
            fact_claim_violations = await self._detect_unverified_claims(content, context)
            violations.extend(fact_claim_violations)

            # 수치 정확성 검사
            numerical_violations = await self._check_numerical_claims(content)
            violations.extend(numerical_violations)

            return create_success(violations)

        except Exception as e:
            return create_failure(e)

    async def _detect_overconfidence(self, content: str) -> List[IntegrityViolation]:
        """과도한 확신 감지"""
        violations = []
        content_lower = content.lower()

        high_confidence_words = self.uncertainty_indicators['high']
        high_confidence_count = sum(content_lower.count(word) for word in high_confidence_words)

        # 과도한 확신 표현이 많은 경우
        if high_confidence_count > 3:
            violation = IntegrityViolation(
                violation_id=generate_id("violation_"),
                violation_type=ViolationType.OVERCONFIDENCE,
                severity=min(high_confidence_count * 0.1, 0.8),
                description=f"과도한 확신 표현 사용 ({high_confidence_count}회)",
                context={'confidence_words': high_confidence_words, 'count': high_confidence_count},
                evidence=[word for word in high_confidence_words if word in content_lower],
                timestamp=current_timestamp(),
                correction_suggestion="불확실성을 인정하는 더 겸손한 표현을 사용하세요"
            )
            violations.append(violation)

        return violations

    async def _detect_unverified_claims(self, content: str, context: Dict[str, Any]) -> List[IntegrityViolation]:
        """검증되지 않은 주장 감지"""
        violations = []

        for pattern in self.fact_claim_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                # 출처나 참조가 없는 사실 주장
                if not self._has_source_reference(content, match):
                    violation = IntegrityViolation(
                        violation_id=generate_id("violation_"),
                        violation_type=ViolationType.FACTUAL_ERROR,
                        severity=0.6,
                        description="출처 없는 사실 주장",
                        context={'claim': match.group(), 'position': match.span()},
                        evidence=[match.group()],
                        timestamp=current_timestamp(),
                        correction_suggestion="사실 주장에는 신뢰할 수 있는 출처를 제시하세요"
                    )
                    violations.append(violation)

        return violations

    def _has_source_reference(self, content: str, match: re.Match) -> bool:
        """출처 참조 여부 확인"""
        source_indicators = ['source:', 'according to', 'cited in', 'reference:', 'study by']

        # 매치 주변 텍스트에서 출처 지시어 찾기
        start = max(0, match.start() - 100)
        end = min(len(content), match.end() + 100)
        surrounding_text = content[start:end].lower()

        return any(indicator in surrounding_text for indicator in source_indicators)

    async def _check_numerical_claims(self, content: str) -> List[IntegrityViolation]:
        """수치 주장 검사"""
        violations = []

        # 퍼센트 패턴 찾기
        percent_pattern = r'(\d+(?:\.\d+)?)\s*%'
        percent_matches = re.finditer(percent_pattern, content)

        for match in percent_matches:
            percentage = float(match.group(1))
            if percentage > 100:
                violation = IntegrityViolation(
                    violation_id=generate_id("violation_"),
                    violation_type=ViolationType.FACTUAL_ERROR,
                    severity=0.8,
                    description=f"불가능한 퍼센트 값: {percentage}%",
                    context={'percentage': percentage, 'position': match.span()},
                    evidence=[match.group()],
                    timestamp=current_timestamp(),
                    correction_suggestion="퍼센트 값이 100%를 초과할 수 없습니다"
                )
                violations.append(violation)

        return violations


class BiasDetector:
    """편향 감지기"""

    def __init__(self):
        self.bias_patterns = {
            BiasType.CONFIRMATION_BIAS: [
                r'(?i)this proves that.*I.*was right',
                r'(?i)as I suspected',
                r'(?i)this confirms.*my.*belief'
            ],
            BiasType.AUTHORITY_BIAS: [
                r'(?i)expert.*says.*therefore.*true',
                r'(?i)famous.*person.*believes.*so.*correct',
                r'(?i)because.*authority.*said'
            ],
            BiasType.RECENCY_BIAS: [
                r'(?i)recent.*events.*show.*always',
                r'(?i)latest.*news.*proves.*permanent',
                r'(?i)just.*happened.*means.*forever'
            ],
            BiasType.CULTURAL_BIAS: [
                r'(?i)everyone.*knows',
                r'(?i)obviously.*all.*people',
                r'(?i)common.*sense.*tells.*us'
            ]
        }

    async def detect_bias(self, content: str, context: Dict[str, Any]) -> Result[List[BiasDetection]]:
        """편향 감지"""
        try:
            detections = []

            for bias_type, patterns in self.bias_patterns.items():
                bias_detections = await self._detect_specific_bias(content, bias_type, patterns)
                detections.extend(bias_detections)

            # 언어 편향 검사
            language_bias = await self._detect_language_bias(content)
            detections.extend(language_bias)

            return create_success(detections)

        except Exception as e:
            return create_failure(e)

    async def _detect_specific_bias(self, content: str, bias_type: BiasType,
                                  patterns: List[str]) -> List[BiasDetection]:
        """특정 편향 감지"""
        detections = []

        for pattern in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                detection = BiasDetection(
                    detection_id=generate_id("bias_"),
                    bias_type=bias_type,
                    confidence=0.7,
                    affected_content=match.group(),
                    evidence_markers=[match.group()],
                    mitigation_suggestion=self._get_bias_mitigation_suggestion(bias_type),
                    timestamp=current_timestamp()
                )
                detections.append(detection)

        return detections

    async def _detect_language_bias(self, content: str) -> List[BiasDetection]:
        """언어 편향 감지"""
        detections = []

        # 성별 편향 검사
        gendered_assumptions = [
            r'(?i)he.*must.*be.*engineer',
            r'(?i)she.*probably.*nurse',
            r'(?i)men.*are.*better.*at',
            r'(?i)women.*naturally.*good.*at'
        ]

        for pattern in gendered_assumptions:
            matches = re.finditer(pattern, content)
            for match in matches:
                detection = BiasDetection(
                    detection_id=generate_id("bias_"),
                    bias_type=BiasType.CULTURAL_BIAS,
                    confidence=0.8,
                    affected_content=match.group(),
                    evidence_markers=[match.group()],
                    mitigation_suggestion="성별에 대한 고정관념을 피하고 중립적 언어를 사용하세요",
                    timestamp=current_timestamp()
                )
                detections.append(detection)

        return detections

    def _get_bias_mitigation_suggestion(self, bias_type: BiasType) -> str:
        """편향별 완화 제안"""
        suggestions = {
            BiasType.CONFIRMATION_BIAS: "반대 증거도 고려하고 다양한 관점을 탐색하세요",
            BiasType.AUTHORITY_BIAS: "권위자의 주장도 비판적으로 검토하고 근거를 평가하세요",
            BiasType.RECENCY_BIAS: "최근 사건뿐만 아니라 장기적 패턴도 고려하세요",
            BiasType.CULTURAL_BIAS: "문화적 다양성을 인정하고 보편적 가정을 피하세요",
            BiasType.AVAILABILITY_BIAS: "쉽게 떠오르는 예시 외에도 전체적 데이터를 고려하세요",
            BiasType.ANCHORING_BIAS: "초기 정보에 과도하게 의존하지 말고 추가 정보를 수집하세요"
        }
        return suggestions.get(bias_type, "객관적이고 균형잡힌 관점을 유지하세요")


class TransparencyEvaluator:
    """투명성 평가기"""

    def __init__(self):
        self.explanation_indicators = [
            'because', 'due to', 'since', 'given that', 'as a result of',
            'this is why', 'the reason is', 'explanation', 'therefore'
        ]
        self.uncertainty_expressions = [
            'uncertain', 'unclear', 'unknown', 'possibly', 'might be',
            'not sure', 'difficult to determine', 'ambiguous'
        ]

    async def evaluate_transparency(self, content: str, context: Dict[str, Any]) -> Result[float]:
        """투명성 평가 (0.0-1.0)"""
        try:
            scores = []

            # 설명 제공 정도
            explanation_score = await self._evaluate_explanation_quality(content)
            scores.append(explanation_score)

            # 불확실성 명시 정도
            uncertainty_score = await self._evaluate_uncertainty_disclosure(content)
            scores.append(uncertainty_score)

            # 한계 인정 정도
            limitation_score = await self._evaluate_limitation_acknowledgment(content)
            scores.append(limitation_score)

            # 출처 명시 정도
            source_score = await self._evaluate_source_transparency(content)
            scores.append(source_score)

            overall_score = statistics.mean(scores)
            return create_success(overall_score)

        except Exception as e:
            return create_failure(e)

    async def _evaluate_explanation_quality(self, content: str) -> float:
        """설명 품질 평가"""
        content_lower = content.lower()
        explanation_count = sum(content_lower.count(indicator) for indicator in self.explanation_indicators)

        # 내용 길이 대비 설명 비율
        words = len(content.split())
        if words == 0:
            return 0.0

        explanation_ratio = explanation_count / max(words // 20, 1)  # 20단어당 1개 설명 기준
        return min(explanation_ratio, 1.0)

    async def _evaluate_uncertainty_disclosure(self, content: str) -> float:
        """불확실성 공개 평가"""
        content_lower = content.lower()
        uncertainty_count = sum(content_lower.count(expr) for expr in self.uncertainty_expressions)

        # 확신 표현과 불확실성 표현의 균형
        confidence_words = ['definitely', 'certainly', 'absolutely', 'guaranteed']
        confidence_count = sum(content_lower.count(word) for word in confidence_words)

        if confidence_count + uncertainty_count == 0:
            return 0.5  # 중립

        uncertainty_ratio = uncertainty_count / (confidence_count + uncertainty_count)
        return uncertainty_ratio

    async def _evaluate_limitation_acknowledgment(self, content: str) -> float:
        """한계 인정 평가"""
        limitation_indicators = [
            'limitation', 'constraint', 'caveat', 'however', 'but',
            'limitation of', 'may not', 'cannot guarantee', 'limited to'
        ]

        content_lower = content.lower()
        limitation_mentions = sum(content_lower.count(indicator) for indicator in limitation_indicators)

        # 내용 길이 대비 한계 인정 비율
        words = len(content.split())
        if words == 0:
            return 0.0

        limitation_ratio = limitation_mentions / max(words // 30, 1)  # 30단어당 1개 한계 인정
        return min(limitation_ratio, 1.0)

    async def _evaluate_source_transparency(self, content: str) -> float:
        """출처 투명성 평가"""
        source_indicators = [
            'source:', 'according to', 'based on', 'cited in', 'reference:',
            'study by', 'research from', 'data from', 'reported by'
        ]

        content_lower = content.lower()
        source_mentions = sum(content_lower.count(indicator) for indicator in source_indicators)

        # 사실 주장 대비 출처 제시 비율
        fact_claim_count = len(re.findall(r'(?i)studies show|research indicates|statistics|data shows', content))

        if fact_claim_count == 0:
            return 1.0 if source_mentions == 0 else 1.0  # 사실 주장이 없으면 만점

        source_ratio = source_mentions / fact_claim_count
        return min(source_ratio, 1.0)


class IntegrityMonitor:
    """지적 무결성 모니터"""

    def __init__(self):
        self.logical_checker = LogicalConsistencyChecker()
        self.factual_checker = FactualAccuracyChecker()
        self.bias_detector = BiasDetector()
        self.transparency_evaluator = TransparencyEvaluator()

        self.monitoring_history: List[IntegrityReport] = []
        self.violation_thresholds = {
            ViolationType.LOGICAL_INCONSISTENCY: 0.7,
            ViolationType.FACTUAL_ERROR: 0.8,
            ViolationType.BIAS_DETECTED: 0.6,
            ViolationType.TRANSPARENCY_VIOLATION: 0.5,
            ViolationType.OVERCONFIDENCE: 0.6
        }

    async def assess_integrity(self, content: str, context: Dict[str, Any]) -> Result[IntegrityReport]:
        """종합 무결성 평가"""
        try:
            report_id = generate_id("integrity_")

            # 병렬로 모든 검사 수행
            tasks = [
                self.logical_checker.check_consistency(content, context),
                self.factual_checker.check_factual_accuracy(content, context),
                self.bias_detector.detect_bias(content, context),
                self.transparency_evaluator.evaluate_transparency(content, context)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 결과 처리
            logical_violations = results[0].value if results[0].is_success else []
            factual_violations = results[1].value if results[1].is_success else []
            bias_detections = results[2].value if results[2].is_success else []
            transparency_score = results[3].value if results[3].is_success else 0.5

            # 모든 위반 통합
            all_violations = logical_violations + factual_violations

            # 편향을 위반으로 변환
            bias_violations = [self._convert_bias_to_violation(bias) for bias in bias_detections]
            all_violations.extend(bias_violations)

            # 투명성 위반 검사
            if transparency_score < 0.5:
                transparency_violation = IntegrityViolation(
                    violation_id=generate_id("violation_"),
                    violation_type=ViolationType.TRANSPARENCY_VIOLATION,
                    severity=1.0 - transparency_score,
                    description=f"낮은 투명성 점수: {transparency_score:.2f}",
                    context={'transparency_score': transparency_score},
                    evidence=["투명성 부족"],
                    timestamp=current_timestamp(),
                    correction_suggestion="더 명확한 설명과 출처 제시가 필요합니다"
                )
                all_violations.append(transparency_violation)

            # 구성 요소별 점수 계산
            component_scores = await self._calculate_component_scores(
                logical_violations, factual_violations, bias_detections, transparency_score
            )

            # 전체 점수 계산
            overall_score = await self._calculate_overall_score(component_scores, all_violations)

            # 무결성 수준 결정
            integrity_level = self._determine_integrity_level(overall_score)

            # 권장사항 생성
            recommendations = await self._generate_recommendations(all_violations, bias_detections, transparency_score)

            # 보고서 생성
            report = IntegrityReport(
                report_id=report_id,
                overall_score=overall_score,
                integrity_level=integrity_level,
                component_scores=component_scores,
                violations=all_violations,
                bias_detections=bias_detections,
                recommendations=recommendations,
                timestamp=current_timestamp()
            )

            # 히스토리에 추가
            self.monitoring_history.append(report)

            return create_success(report)

        except Exception as e:
            return create_failure(e)

    async def _calculate_component_scores(self, logical_violations: List[IntegrityViolation],
                                        factual_violations: List[IntegrityViolation],
                                        bias_detections: List[BiasDetection],
                                        transparency_score: float) -> Dict[str, float]:
        """구성 요소별 점수 계산"""

        # 논리적 일관성 점수
        logical_penalty = sum(v.severity for v in logical_violations) * 0.1
        logical_score = max(0.0, 1.0 - logical_penalty)

        # 사실 정확성 점수
        factual_penalty = sum(v.severity for v in factual_violations) * 0.15
        factual_score = max(0.0, 1.0 - factual_penalty)

        # 편향 부재 점수
        bias_penalty = sum(b.confidence for b in bias_detections) * 0.1
        bias_score = max(0.0, 1.0 - bias_penalty)

        return {
            'logical_consistency': logical_score,
            'factual_accuracy': factual_score,
            'bias_absence': bias_score,
            'transparency': transparency_score
        }

    async def _calculate_overall_score(self, component_scores: Dict[str, float],
                                     violations: List[IntegrityViolation]) -> float:
        """전체 점수 계산"""
        # 가중 평균
        weights = {
            'logical_consistency': 0.3,
            'factual_accuracy': 0.3,
            'bias_absence': 0.2,
            'transparency': 0.2
        }

        weighted_score = sum(score * weights[component]
                           for component, score in component_scores.items())

        # 심각한 위반에 대한 추가 페널티
        severe_violations = [v for v in violations if v.severity > 0.8]
        severe_penalty = len(severe_violations) * 0.1

        final_score = max(0.0, weighted_score - severe_penalty)
        return min(final_score, 1.0)

    def _determine_integrity_level(self, overall_score: float) -> IntegrityLevel:
        """무결성 수준 결정"""
        if overall_score >= 0.9:
            return IntegrityLevel.EXCELLENT
        elif overall_score >= 0.75:
            return IntegrityLevel.HIGH
        elif overall_score >= 0.5:
            return IntegrityLevel.MODERATE
        elif overall_score >= 0.3:
            return IntegrityLevel.LOW
        else:
            return IntegrityLevel.CRITICAL

    def _convert_bias_to_violation(self, bias_detection: BiasDetection) -> IntegrityViolation:
        """편향 감지를 위반으로 변환"""
        return IntegrityViolation(
            violation_id=generate_id("violation_"),
            violation_type=ViolationType.BIAS_DETECTED,
            severity=bias_detection.confidence,
            description=f"편향 감지: {bias_detection.bias_type.value}",
            context={'bias_type': bias_detection.bias_type.value, 'confidence': bias_detection.confidence},
            evidence=bias_detection.evidence_markers,
            timestamp=bias_detection.timestamp,
            correction_suggestion=bias_detection.mitigation_suggestion
        )

    async def _generate_recommendations(self, violations: List[IntegrityViolation],
                                      bias_detections: List[BiasDetection],
                                      transparency_score: float) -> List[str]:
        """권장사항 생성"""
        recommendations = []

        # 위반별 권장사항
        violation_types = set(v.violation_type for v in violations)

        if ViolationType.LOGICAL_INCONSISTENCY in violation_types:
            recommendations.append("논리적 일관성을 개선하고 모순되는 표현을 제거하세요")

        if ViolationType.FACTUAL_ERROR in violation_types:
            recommendations.append("사실 주장에 대한 검증과 출처 제시를 강화하세요")

        if ViolationType.OVERCONFIDENCE in violation_types:
            recommendations.append("불확실성을 인정하는 더 겸손한 표현을 사용하세요")

        # 편향별 권장사항
        if bias_detections:
            recommendations.append("감지된 편향을 줄이고 더 객관적인 관점을 유지하세요")

        # 투명성 권장사항
        if transparency_score < 0.7:
            recommendations.append("설명의 명확성과 출처 투명성을 개선하세요")

        # 일반적 권장사항
        if not recommendations:
            recommendations.append("현재 높은 무결성을 유지하고 있습니다. 지속적인 모니터링을 권장합니다")

        return recommendations

    def get_integrity_trends(self, hours: int = 24) -> Dict[str, Any]:
        """무결성 추세 분석"""
        cutoff_time = current_timestamp() - (hours * 3600)
        recent_reports = [r for r in self.monitoring_history if r.timestamp >= cutoff_time]

        if not recent_reports:
            return {'trend': 'no_data'}

        # 점수 추세
        scores = [r.overall_score for r in recent_reports]
        avg_score = statistics.mean(scores)

        if len(scores) > 1:
            trend = "improving" if scores[-1] > scores[0] else "declining" if scores[-1] < scores[0] else "stable"
        else:
            trend = "stable"

        # 위반 통계
        all_violations = [v for r in recent_reports for v in r.violations]
        violation_counts = {}
        for vtype in ViolationType:
            violation_counts[vtype.value] = sum(1 for v in all_violations if v.violation_type == vtype)

        return {
            'trend': trend,
            'average_score': avg_score,
            'report_count': len(recent_reports),
            'violation_counts': violation_counts,
            'latest_level': recent_reports[-1].integrity_level.value if recent_reports else None
        }

    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """모니터링 통계"""
        if not self.monitoring_history:
            return {'total_reports': 0}

        total_reports = len(self.monitoring_history)

        # 수준별 분포
        level_counts = {}
        for level in IntegrityLevel:
            level_counts[level.value] = sum(1 for r in self.monitoring_history
                                          if r.integrity_level == level)

        # 평균 점수
        avg_overall_score = statistics.mean(r.overall_score for r in self.monitoring_history)

        # 구성 요소별 평균 점수
        avg_component_scores = {}
        for component in ['logical_consistency', 'factual_accuracy', 'bias_absence', 'transparency']:
            scores = [r.component_scores.get(component, 0) for r in self.monitoring_history]
            avg_component_scores[component] = statistics.mean(scores) if scores else 0

        return {
            'total_reports': total_reports,
            'level_distribution': level_counts,
            'average_overall_score': avg_overall_score,
            'average_component_scores': avg_component_scores,
            'recent_trends': self.get_integrity_trends(24)
        }