"""
Truth Assessment Module
진실 평가 통합 시스템

이 모듈은 모든 진실 추구 도구를 통합하여 종합적인 진실성 평가를 제공합니다.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Tuple, Union, TYPE_CHECKING
from datetime import datetime, timedelta
import asyncio
import json
import statistics

try:
    from ...core.types import ID, Result, ErrorType
    from ...core.utils import generate_id, current_timestamp
except ImportError:
    # 직접 실행 시 대체
    ID = str
    Result = dict
    ErrorType = str
    def generate_id() -> str: return f"id_{datetime.now().isoformat()}"
    def current_timestamp() -> datetime: return datetime.now()

# 런타임에는 실제 클래스 import, 타입 체크 시에는 TYPE_CHECKING 사용
if TYPE_CHECKING:
    from .evidence_evaluator import EvidenceEvaluator, EvidenceAssessment, EvidenceQuality
    from .source_validator import SourceValidator, ValidationResult, SourceProfile
    from .fact_checker import FactChecker, FactCheckResult, FactStatus
else:
    # 런타임 import
    try:
        from .evidence_evaluator import EvidenceEvaluator, EvidenceAssessment, EvidenceQuality
        from .source_validator import SourceValidator, ValidationResult, SourceProfile
        from .fact_checker import FactChecker, FactCheckResult, FactStatus
    except ImportError:
        # 순환 import 방지를 위해 늦은 import 사용
        EvidenceEvaluator = None
        SourceValidator = None
        FactChecker = None


class ConfidenceLevel(Enum):
    """신뢰도 수준"""
    VERY_HIGH = auto()    # 95%+ (매우 높음)
    HIGH = auto()         # 80-94% (높음)
    MODERATE = auto()     # 60-79% (보통)
    LOW = auto()          # 40-59% (낮음)
    VERY_LOW = auto()     # <40% (매우 낮음)


@dataclass
class TruthScore:
    """진실성 점수"""
    overall_score: float           # 전체 진실성 점수 (0.0-1.0)
    confidence_level: ConfidenceLevel  # 신뢰도 수준

    # 세부 점수
    evidence_score: float         # 증거 점수
    source_score: float          # 정보원 점수
    consistency_score: float     # 일관성 점수
    verifiability_score: float   # 검증가능성 점수

    # 위험 요소
    bias_risk: float             # 편향 위험도 (0.0-1.0)
    misinformation_risk: float   # 잘못된 정보 위험도 (0.0-1.0)
    uncertainty_level: float     # 불확실성 수준 (0.0-1.0)

    def to_percentage(self) -> float:
        """백분율로 변환"""
        return self.overall_score * 100

    def get_risk_assessment(self) -> str:
        """위험 평가 문자열"""
        avg_risk = (self.bias_risk + self.misinformation_risk + self.uncertainty_level) / 3

        if avg_risk < 0.2:
            return "LOW_RISK"
        elif avg_risk < 0.4:
            return "MODERATE_RISK"
        elif avg_risk < 0.6:
            return "HIGH_RISK"
        else:
            return "CRITICAL_RISK"


@dataclass
class UncertaintyMetrics:
    """불확실성 지표"""
    epistemic_uncertainty: float  # 인식론적 불확실성 (지식 부족)
    aleatory_uncertainty: float   # 우연적 불확실성 (본질적 가변성)
    model_uncertainty: float      # 모델 불확실성 (방법론적 한계)

    # 불확실성 원인
    data_gaps: List[str] = field(default_factory=list)
    conflicting_evidence: List[str] = field(default_factory=list)
    methodological_limitations: List[str] = field(default_factory=list)

    def total_uncertainty(self) -> float:
        """총 불확실성"""
        return (self.epistemic_uncertainty + self.aleatory_uncertainty + self.model_uncertainty) / 3


@dataclass
class AssessmentReport:
    """평가 보고서"""
    assessment_id: ID
    query: str
    truth_score: TruthScore
    uncertainty_metrics: UncertaintyMetrics

    # 세부 분석
    fact_check_results: List[FactCheckResult]
    evidence_assessments: List[EvidenceAssessment]
    source_validations: List[ValidationResult]

    # 종합 분석
    key_findings: List[str]
    evidence_summary: str
    consensus_level: float        # 합의 수준 (0.0-1.0)

    # 권장사항
    recommendations: List[str]
    further_research_needed: List[str]
    quality_improvements: List[str]

    # 메타데이터
    assessment_timestamp: datetime
    assessor_id: Optional[ID] = None
    methodology_version: str = "1.0.0"

    # 추가 정보
    related_assessments: List[ID] = field(default_factory=list)
    external_validations: List[Dict[str, Any]] = field(default_factory=list)


class TruthAssessment:
    """진실 평가 엔진"""

    def __init__(self):
        self.assessments: Dict[ID, 'AssessmentReport'] = {}

        # 통합 구성 요소 (늦은 import)
        self.evidence_evaluator = None
        self.source_validator = None
        self.fact_checker = None
        self._initialize_components()

        # 평가 기준 및 가중치
        self.scoring_weights = self._initialize_scoring_weights()
        self.confidence_thresholds = self._initialize_confidence_thresholds()
        self.uncertainty_factors = self._initialize_uncertainty_factors()

    def _initialize_components(self):
        """구성 요소 초기화 (늦은 import)"""
        if self.evidence_evaluator is None:
            from .evidence_evaluator import EvidenceEvaluator
            self.evidence_evaluator = EvidenceEvaluator()

        if self.source_validator is None:
            from .source_validator import SourceValidator
            self.source_validator = SourceValidator()

        if self.fact_checker is None:
            from .fact_checker import FactChecker
            self.fact_checker = FactChecker()

    def _initialize_scoring_weights(self) -> Dict[str, float]:
        """점수 가중치 초기화"""
        return {
            'evidence_quality': 0.3,      # 증거 품질
            'source_credibility': 0.25,   # 정보원 신뢰성
            'fact_verification': 0.25,    # 사실 검증
            'consistency': 0.1,           # 일관성
            'verifiability': 0.1          # 검증가능성
        }

    def _initialize_confidence_thresholds(self) -> Dict[ConfidenceLevel, Tuple[float, float]]:
        """신뢰도 임계값 초기화"""
        return {
            ConfidenceLevel.VERY_HIGH: (0.95, 1.0),
            ConfidenceLevel.HIGH: (0.8, 0.94),
            ConfidenceLevel.MODERATE: (0.6, 0.79),
            ConfidenceLevel.LOW: (0.4, 0.59),
            ConfidenceLevel.VERY_LOW: (0.0, 0.39)
        }

    def _initialize_uncertainty_factors(self) -> Dict[str, Dict[str, float]]:
        """불확실성 요소 초기화"""
        return {
            'data_quality': {
                'missing_data': 0.3,
                'incomplete_data': 0.2,
                'outdated_data': 0.15,
                'inconsistent_data': 0.25,
                'biased_data': 0.1
            },
            'methodological': {
                'sampling_bias': 0.2,
                'measurement_error': 0.15,
                'model_limitations': 0.25,
                'validation_gaps': 0.2,
                'interpretation_variance': 0.2
            },
            'external_factors': {
                'temporal_changes': 0.3,
                'contextual_differences': 0.25,
                'cultural_variations': 0.15,
                'technological_evolution': 0.2,
                'regulatory_changes': 0.1
            }
        }

    async def comprehensive_assessment(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Result['AssessmentReport']:
        """종합적인 진실성 평가"""
        try:
            # 구성 요소가 초기화되었는지 확인
            self._initialize_components()

            assessment_id = generate_id()

            # 1. 사실 확인
            fact_check_result = await self.fact_checker.check_fact(query, context)
            if not fact_check_result["success"]:
                return {"success": False, "error": f"Fact checking failed: {fact_check_result['error']}"}

            fact_check = fact_check_result["data"]

            # 2. 증거 평가 (사실 확인에서 사용된 증거들)
            evidence_assessments = await self._evaluate_evidence_from_fact_check(fact_check)

            # 3. 정보원 검증 (사실 확인에서 사용된 정보원들)
            source_validations = await self._validate_sources_from_fact_check(fact_check)

            # 4. 종합 점수 계산
            truth_score = await self._calculate_truth_score(
                fact_check, evidence_assessments, source_validations
            )

            # 5. 불확실성 분석
            uncertainty_metrics = await self._analyze_uncertainty(
                fact_check, evidence_assessments, source_validations
            )

            # 6. 핵심 발견사항 추출
            key_findings = await self._extract_key_findings(
                fact_check, evidence_assessments, source_validations
            )

            # 7. 증거 요약
            evidence_summary = await self._summarize_evidence(evidence_assessments)

            # 8. 합의 수준 계산
            consensus_level = await self._calculate_consensus_level(
                fact_check, evidence_assessments
            )

            # 9. 권장사항 생성
            recommendations, research_needed, quality_improvements = await self._generate_recommendations(
                truth_score, uncertainty_metrics, fact_check
            )

            # 10. 평가 보고서 생성
            assessment_report = AssessmentReport(
                assessment_id=assessment_id,
                query=query,
                truth_score=truth_score,
                uncertainty_metrics=uncertainty_metrics,
                fact_check_results=[fact_check],
                evidence_assessments=evidence_assessments,
                source_validations=source_validations,
                key_findings=key_findings,
                evidence_summary=evidence_summary,
                consensus_level=consensus_level,
                recommendations=recommendations,
                further_research_needed=research_needed,
                quality_improvements=quality_improvements,
                assessment_timestamp=current_timestamp()
            )

            self.assessments[assessment_id] = assessment_report

            return {"success": True, "data": assessment_report}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _evaluate_evidence_from_fact_check(
        self,
        fact_check: FactCheckResult
    ) -> List[EvidenceAssessment]:
        """사실 확인에서 증거 평가"""
        assessments = []

        # 지지 증거 평가
        for ref in fact_check.supporting_evidence:
            evidence_data = {
                "content": ref.supporting_text,
                "reliability": ref.confidence_level,
                "relevance": ref.relevance_score,
                "verification_method": ref.verification_method.name
            }

            assessment_result = await self.evidence_evaluator.evaluate_evidence(
                evidence_data, self._map_verification_to_evidence_type(ref.verification_method)
            )

            if assessment_result["success"]:
                assessments.append(assessment_result["data"])

        # 반박 증거 평가
        for ref in fact_check.contradicting_evidence:
            evidence_data = {
                "content": ref.supporting_text or ref.contradiction_text,
                "reliability": ref.confidence_level,
                "relevance": ref.relevance_score,
                "verification_method": ref.verification_method.name,
                "contradicts_claim": True
            }

            assessment_result = await self.evidence_evaluator.evaluate_evidence(
                evidence_data, self._map_verification_to_evidence_type(ref.verification_method)
            )

            if assessment_result["success"]:
                assessments.append(assessment_result["data"])

        return assessments

    async def _validate_sources_from_fact_check(
        self,
        fact_check: FactCheckResult
    ) -> List[ValidationResult]:
        """사실 확인에서 정보원 검증"""
        validations = []

        # 실제 구현에서는 fact_check.sources_checked를 통해 정보원 검증
        for source_id in fact_check.sources_checked:
            # 임시 정보원 프로필 생성 (실제로는 DB에서 조회)
            source_data = {
                "name": f"Source {source_id}",
                "url": f"https://example.com/{source_id}",
                "type": "general_website"
            }

            profile_result = await self.source_validator.create_source_profile(
                source_data, self._infer_source_type(source_data)
            )

            if profile_result["success"]:
                profile = profile_result["data"]
                validation_result = await self.source_validator.validate_source(profile.source_id)

                if validation_result["success"]:
                    validations.append(validation_result["data"])

        return validations

    async def _calculate_truth_score(
        self,
        fact_check: FactCheckResult,
        evidence_assessments: List[EvidenceAssessment],
        source_validations: List[ValidationResult]
    ) -> TruthScore:
        """진실성 점수 계산"""

        # 증거 점수 계산
        evidence_score = self._calculate_evidence_score(evidence_assessments)

        # 정보원 점수 계산
        source_score = self._calculate_source_score(source_validations)

        # 사실 검증 점수
        fact_score = self._map_fact_status_to_score(fact_check.fact_status)

        # 일관성 점수
        consistency_score = self._calculate_consistency_score(
            fact_check, evidence_assessments
        )

        # 검증가능성 점수
        verifiability_score = fact_check.claim_analysis.verifiability

        # 가중 평균으로 전체 점수 계산
        overall_score = (
            evidence_score * self.scoring_weights['evidence_quality'] +
            source_score * self.scoring_weights['source_credibility'] +
            fact_score * self.scoring_weights['fact_verification'] +
            consistency_score * self.scoring_weights['consistency'] +
            verifiability_score * self.scoring_weights['verifiability']
        )

        # 신뢰도 수준 결정
        confidence_level = self._determine_confidence_level(overall_score)

        # 위험 요소 계산
        bias_risk = self._calculate_bias_risk(evidence_assessments, source_validations)
        misinformation_risk = self._calculate_misinformation_risk(fact_check, evidence_assessments)
        uncertainty_level = self._calculate_uncertainty_level(fact_check, evidence_assessments)

        return TruthScore(
            overall_score=overall_score,
            confidence_level=confidence_level,
            evidence_score=evidence_score,
            source_score=source_score,
            consistency_score=consistency_score,
            verifiability_score=verifiability_score,
            bias_risk=bias_risk,
            misinformation_risk=misinformation_risk,
            uncertainty_level=uncertainty_level
        )

    async def _analyze_uncertainty(
        self,
        fact_check: FactCheckResult,
        evidence_assessments: List[EvidenceAssessment],
        source_validations: List[ValidationResult]
    ) -> UncertaintyMetrics:
        """불확실성 분석"""

        # 인식론적 불확실성 (지식 부족)
        epistemic_uncertainty = self._calculate_epistemic_uncertainty(
            evidence_assessments, source_validations
        )

        # 우연적 불확실성 (본질적 가변성)
        aleatory_uncertainty = self._calculate_aleatory_uncertainty(fact_check)

        # 모델 불확실성 (방법론적 한계)
        model_uncertainty = self._calculate_model_uncertainty(fact_check)

        # 불확실성 원인 식별
        data_gaps = self._identify_data_gaps(evidence_assessments)
        conflicting_evidence = self._identify_conflicting_evidence(fact_check)
        methodological_limitations = self._identify_methodological_limitations(fact_check)

        return UncertaintyMetrics(
            epistemic_uncertainty=epistemic_uncertainty,
            aleatory_uncertainty=aleatory_uncertainty,
            model_uncertainty=model_uncertainty,
            data_gaps=data_gaps,
            conflicting_evidence=conflicting_evidence,
            methodological_limitations=methodological_limitations
        )

    def _calculate_evidence_score(self, assessments: List[EvidenceAssessment]) -> float:
        """증거 점수 계산"""
        if not assessments:
            return 0.0

        # 가중 평균 (신뢰도로 가중)
        total_weighted_score = sum(
            assessment.get_weighted_score() * assessment.confidence_level
            for assessment in assessments
        )
        total_weight = sum(assessment.confidence_level for assessment in assessments)

        return total_weighted_score / total_weight if total_weight > 0 else 0.0

    def _calculate_source_score(self, validations: List[ValidationResult]) -> float:
        """정보원 점수 계산"""
        if not validations:
            return 0.0

        return sum(validation.overall_credibility for validation in validations) / len(validations)

    def _map_fact_status_to_score(self, fact_status: FactStatus) -> float:
        """사실 상태를 점수로 매핑"""
        mapping = {
            FactStatus.TRUE: 1.0,
            FactStatus.PARTIALLY_TRUE: 0.7,
            FactStatus.MISLEADING: 0.4,
            FactStatus.DISPUTED: 0.3,
            FactStatus.UNVERIFIABLE: 0.2,
            FactStatus.INSUFFICIENT_EVIDENCE: 0.1,
            FactStatus.FALSE: 0.0,
            FactStatus.OUTDATED: 0.1
        }
        return mapping.get(fact_status, 0.5)

    def _calculate_consistency_score(
        self,
        fact_check: FactCheckResult,
        evidence_assessments: List[EvidenceAssessment]
    ) -> float:
        """일관성 점수 계산"""
        # 증거 간 일관성
        evidence_balance = fact_check.get_evidence_balance()
        consistency = max(evidence_balance['supporting'], evidence_balance['contradicting'])

        # 정보원 간 일관성
        if evidence_assessments:
            source_scores = [assessment.credibility_score.overall_score for assessment in evidence_assessments]
            if len(source_scores) > 1:
                # 표준편차가 낮을수록 일관성이 높음
                std_dev = statistics.stdev(source_scores)
                source_consistency = max(0, 1 - std_dev)
                consistency = (consistency + source_consistency) / 2

        return consistency

    def _determine_confidence_level(self, overall_score: float) -> ConfidenceLevel:
        """신뢰도 수준 결정"""
        for level, (min_score, max_score) in self.confidence_thresholds.items():
            if min_score <= overall_score <= max_score:
                return level
        return ConfidenceLevel.VERY_LOW

    def _calculate_bias_risk(
        self,
        evidence_assessments: List[EvidenceAssessment],
        source_validations: List[ValidationResult]
    ) -> float:
        """편향 위험도 계산"""
        risk_factors = []

        # 증거의 편향 요소
        for assessment in evidence_assessments:
            risk_factors.append(assessment.credibility_score.bias_factor)

        # 정보원의 편향 위험
        for validation in source_validations:
            if 'bias' in validation.recommendation.lower():
                risk_factors.append(0.7)
            elif validation.risk_level == "HIGH":
                risk_factors.append(0.6)

        return sum(risk_factors) / len(risk_factors) if risk_factors else 0.0

    def _calculate_misinformation_risk(
        self,
        fact_check: FactCheckResult,
        evidence_assessments: List[EvidenceAssessment]
    ) -> float:
        """잘못된 정보 위험도 계산"""
        risk = 0.0

        # 사실 확인 결과에 따른 위험도
        if fact_check.fact_status == FactStatus.FALSE:
            risk += 0.8
        elif fact_check.fact_status == FactStatus.MISLEADING:
            risk += 0.6
        elif fact_check.fact_status == FactStatus.DISPUTED:
            risk += 0.4

        # 낮은 품질 증거
        low_quality_count = sum(1 for assessment in evidence_assessments
                               if assessment.quality_rating in [EvidenceQuality.POOR, EvidenceQuality.VERY_POOR])
        if evidence_assessments:
            risk += (low_quality_count / len(evidence_assessments)) * 0.3

        return min(risk, 1.0)

    def _calculate_uncertainty_level(
        self,
        fact_check: FactCheckResult,
        evidence_assessments: List[EvidenceAssessment]
    ) -> float:
        """불확실성 수준 계산"""
        uncertainty = 0.0

        # 신뢰도 기반 불확실성
        uncertainty += (1.0 - fact_check.confidence_score) * 0.4

        # 증거 강도 기반 불확실성
        uncertainty += (1.0 - fact_check.evidence_strength) * 0.3

        # 검증가능성 기반 불확실성
        uncertainty += (1.0 - fact_check.claim_analysis.verifiability) * 0.3

        return min(uncertainty, 1.0)

    def _calculate_epistemic_uncertainty(
        self,
        evidence_assessments: List[EvidenceAssessment],
        source_validations: List[ValidationResult]
    ) -> float:
        """인식론적 불확실성 계산"""
        uncertainty = 0.0

        # 증거 부족
        if len(evidence_assessments) < 3:
            uncertainty += 0.3

        # 낮은 품질 정보원
        if source_validations:
            low_credibility_count = sum(1 for validation in source_validations
                                       if validation.overall_credibility < 0.5)
            uncertainty += (low_credibility_count / len(source_validations)) * 0.3

        # 검증 한계
        uncertainty += 0.2  # 기본 방법론적 한계

        return min(uncertainty, 1.0)

    def _calculate_aleatory_uncertainty(self, fact_check: FactCheckResult) -> float:
        """우연적 불확실성 계산"""
        # 주장의 복잡성에 따른 본질적 불확실성
        complexity = fact_check.claim_analysis.complexity
        return min(complexity * 0.5, 0.5)

    def _calculate_model_uncertainty(self, fact_check: FactCheckResult) -> float:
        """모델 불확실성 계산"""
        # 자동화된 평가의 한계
        base_uncertainty = 0.15

        # 복잡한 주장일수록 모델 불확실성 증가
        complexity_factor = fact_check.claim_analysis.complexity * 0.1

        return min(base_uncertainty + complexity_factor, 0.3)

    def _identify_data_gaps(self, evidence_assessments: List[EvidenceAssessment]) -> List[str]:
        """데이터 격차 식별"""
        gaps = []

        if len(evidence_assessments) < 3:
            gaps.append("Insufficient number of evidence sources")

        low_relevance_count = sum(1 for assessment in evidence_assessments
                                 if assessment.relevance_score < 0.6)
        if low_relevance_count > len(evidence_assessments) / 2:
            gaps.append("Low relevance of available evidence")

        outdated_count = sum(1 for assessment in evidence_assessments
                            if assessment.recency_score < 0.5)
        if outdated_count > 0:
            gaps.append("Outdated evidence sources")

        return gaps

    def _identify_conflicting_evidence(self, fact_check: FactCheckResult) -> List[str]:
        """상충하는 증거 식별"""
        conflicts = []

        if len(fact_check.supporting_evidence) > 0 and len(fact_check.contradicting_evidence) > 0:
            conflicts.append("Conflicting evidence from different sources")

        evidence_balance = fact_check.get_evidence_balance()
        if 0.3 <= evidence_balance['contradicting'] <= 0.7:
            conflicts.append("Significant disagreement among sources")

        return conflicts

    def _identify_methodological_limitations(self, fact_check: FactCheckResult) -> List[str]:
        """방법론적 한계 식별"""
        limitations = []

        if fact_check.claim_analysis.verifiability < 0.5:
            limitations.append("Limited verifiability of the claim")

        if len(fact_check.verification_methods) < 2:
            limitations.append("Limited verification methods used")

        limitations.extend(fact_check.limitations)

        return limitations

    async def _extract_key_findings(
        self,
        fact_check: FactCheckResult,
        evidence_assessments: List[EvidenceAssessment],
        source_validations: List[ValidationResult]
    ) -> List[str]:
        """핵심 발견사항 추출"""
        findings = []

        # 사실 확인 결과
        findings.append(f"Fact check status: {fact_check.fact_status.name}")
        findings.append(f"Overall confidence: {fact_check.confidence_score:.1%}")

        # 증거 품질
        if evidence_assessments:
            high_quality_count = sum(1 for assessment in evidence_assessments
                                   if assessment.quality_rating in [EvidenceQuality.EXCELLENT, EvidenceQuality.GOOD])
            findings.append(f"High-quality evidence sources: {high_quality_count}/{len(evidence_assessments)}")

        # 정보원 신뢰성
        if source_validations:
            reliable_count = sum(1 for validation in source_validations
                               if validation.overall_credibility >= 0.7)
            findings.append(f"Reliable sources: {reliable_count}/{len(source_validations)}")

        # 합의 수준
        evidence_balance = fact_check.get_evidence_balance()
        if evidence_balance['supporting'] > 0.7:
            findings.append("Strong consensus supporting the claim")
        elif evidence_balance['contradicting'] > 0.7:
            findings.append("Strong consensus contradicting the claim")
        else:
            findings.append("Mixed or limited consensus on the claim")

        return findings

    async def _summarize_evidence(self, evidence_assessments: List[EvidenceAssessment]) -> str:
        """증거 요약"""
        if not evidence_assessments:
            return "No evidence sources were evaluated."

        total = len(evidence_assessments)
        excellent_count = sum(1 for a in evidence_assessments if a.quality_rating == EvidenceQuality.EXCELLENT)
        good_count = sum(1 for a in evidence_assessments if a.quality_rating == EvidenceQuality.GOOD)
        avg_credibility = sum(a.credibility_score.overall_score for a in evidence_assessments) / total

        summary = f"Evaluated {total} evidence sources. "
        summary += f"Quality distribution: {excellent_count} excellent, {good_count} good. "
        summary += f"Average credibility: {avg_credibility:.1%}."

        return summary

    async def _calculate_consensus_level(
        self,
        fact_check: FactCheckResult,
        evidence_assessments: List[EvidenceAssessment]
    ) -> float:
        """합의 수준 계산"""
        evidence_balance = fact_check.get_evidence_balance()

        # 지지 또는 반박 증거가 압도적일 때 높은 합의
        consensus = max(evidence_balance['supporting'], evidence_balance['contradicting'])

        # 증거 품질로 가중
        if evidence_assessments:
            avg_quality = sum(a.credibility_score.overall_score for a in evidence_assessments) / len(evidence_assessments)
            consensus *= avg_quality

        return consensus

    async def _generate_recommendations(
        self,
        truth_score: TruthScore,
        uncertainty_metrics: UncertaintyMetrics,
        fact_check: FactCheckResult
    ) -> Tuple[List[str], List[str], List[str]]:
        """권장사항 생성"""
        recommendations = []
        research_needed = []
        quality_improvements = []

        # 전체 점수에 따른 권장사항
        if truth_score.overall_score >= 0.8:
            recommendations.append("This claim is well-supported and can be considered reliable.")
        elif truth_score.overall_score >= 0.6:
            recommendations.append("This claim has moderate support but should be verified with additional sources.")
        elif truth_score.overall_score >= 0.4:
            recommendations.append("This claim has limited support and should be treated with caution.")
        else:
            recommendations.append("This claim lacks sufficient support and should not be relied upon.")

        # 위험 수준에 따른 권장사항
        risk_level = truth_score.get_risk_assessment()
        if risk_level in ["HIGH_RISK", "CRITICAL_RISK"]:
            recommendations.append("High risk of misinformation or bias detected. Seek alternative sources.")

        # 불확실성에 따른 추가 연구 필요
        if uncertainty_metrics.total_uncertainty() > 0.6:
            research_needed.extend([
                "Additional primary sources needed",
                "Expert consultation recommended",
                "Longitudinal validation required"
            ])

        # 데이터 격차 해결
        research_needed.extend(uncertainty_metrics.data_gaps)

        # 품질 개선 사항
        if truth_score.evidence_score < 0.6:
            quality_improvements.append("Seek higher quality evidence sources")

        if truth_score.source_score < 0.6:
            quality_improvements.append("Validate with more credible sources")

        if truth_score.bias_risk > 0.5:
            quality_improvements.append("Address potential bias in sources")

        return recommendations, research_needed, quality_improvements

    # 유틸리티 메서드들
    def _map_verification_to_evidence_type(self, verification_method):
        """검증 방법을 증거 유형으로 매핑"""
        # 실제 구현에서는 더 정교한 매핑 필요
        # 늦은 import로 순환 import 방지
        from .evidence_evaluator import EvidenceType
        return EvidenceType.EMPIRICAL

    def _infer_source_type(self, source_data: Dict[str, Any]):
        """정보원 데이터에서 유형 추론"""
        # 늦은 import로 순환 import 방지
        from .source_validator import SourceType
        return SourceType.WEBSITE_GENERAL

    async def get_assessment_report(self, assessment_id: ID) -> Optional[AssessmentReport]:
        """평가 보고서 조회"""
        return self.assessments.get(assessment_id)

    async def search_assessments(
        self,
        query: str,
        min_truth_score: float = 0.0,
        confidence_level: Optional[ConfidenceLevel] = None
    ) -> List[AssessmentReport]:
        """평가 결과 검색"""
        results = []

        for assessment in self.assessments.values():
            # 쿼리 매칭
            if query.lower() in assessment.query.lower():
                # 점수 필터
                if assessment.truth_score.overall_score < min_truth_score:
                    continue

                # 신뢰도 필터
                if confidence_level and assessment.truth_score.confidence_level != confidence_level:
                    continue

                results.append(assessment)

        # 진실성 점수 순으로 정렬
        results.sort(key=lambda x: x.truth_score.overall_score, reverse=True)
        return results

    async def get_assessment_statistics(self) -> Dict[str, Any]:
        """평가 통계"""
        if not self.assessments:
            return {"total_assessments": 0}

        total = len(self.assessments)
        avg_truth_score = sum(a.truth_score.overall_score for a in self.assessments.values()) / total

        confidence_distribution = {}
        for level in ConfidenceLevel:
            count = sum(1 for a in self.assessments.values()
                       if a.truth_score.confidence_level == level)
            confidence_distribution[level.name] = count

        high_quality_percentage = sum(
            1 for a in self.assessments.values()
            if a.truth_score.overall_score >= 0.7
        ) / total * 100

        return {
            "total_assessments": total,
            "average_truth_score": avg_truth_score,
            "confidence_distribution": confidence_distribution,
            "high_quality_percentage": high_quality_percentage,
            "average_consensus_level": sum(a.consensus_level for a in self.assessments.values()) / total
        }


class TruthSeekingEngine:
    """진실 추구 통합 엔진"""

    def __init__(self):
        self.truth_assessment = TruthAssessment()

    async def seek_truth(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Result[AssessmentReport]:
        """진실 추구 메인 인터페이스"""
        return await self.truth_assessment.comprehensive_assessment(query, context)

    async def verify_claim(
        self,
        claim: str,
        required_confidence: ConfidenceLevel = ConfidenceLevel.MODERATE
    ) -> Result[bool]:
        """주장 검증 (단순 True/False)"""
        assessment_result = await self.seek_truth(claim)

        if not assessment_result["success"]:
            return assessment_result

        assessment = assessment_result["data"]
        truth_score = assessment.truth_score

        # 요구되는 신뢰도 수준을 충족하는지 확인
        required_levels = {
            ConfidenceLevel.VERY_HIGH: 0.95,
            ConfidenceLevel.HIGH: 0.8,
            ConfidenceLevel.MODERATE: 0.6,
            ConfidenceLevel.LOW: 0.4,
            ConfidenceLevel.VERY_LOW: 0.0
        }

        required_score = required_levels[required_confidence]
        is_verified = truth_score.overall_score >= required_score

        return {
            "success": True,
            "data": is_verified,
            "metadata": {
                "truth_score": truth_score.overall_score,
                "confidence_level": truth_score.confidence_level.name,
                "risk_assessment": truth_score.get_risk_assessment()
            }
        }

    async def batch_assessment(
        self,
        queries: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Result[AssessmentReport]]:
        """배치 평가"""
        results = []
        for query in queries:
            result = await self.seek_truth(query, context)
            results.append(result)
        return results

    async def comparative_assessment(
        self,
        claims: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Result[Dict[str, Any]]:
        """비교 평가"""
        try:
            assessments = []
            for claim in claims:
                result = await self.seek_truth(claim, context)
                if result["success"]:
                    assessments.append(result["data"])

            if not assessments:
                return {"success": False, "error": "No successful assessments"}

            # 비교 분석
            truth_scores = [a.truth_score.overall_score for a in assessments]
            best_supported_idx = truth_scores.index(max(truth_scores))
            worst_supported_idx = truth_scores.index(min(truth_scores))

            comparison = {
                "claims": claims,
                "assessments": assessments,
                "best_supported": {
                    "claim": claims[best_supported_idx],
                    "score": truth_scores[best_supported_idx]
                },
                "worst_supported": {
                    "claim": claims[worst_supported_idx],
                    "score": truth_scores[worst_supported_idx]
                },
                "average_score": sum(truth_scores) / len(truth_scores),
                "score_variance": statistics.variance(truth_scores) if len(truth_scores) > 1 else 0
            }

            return {"success": True, "data": comparison}

        except Exception as e:
            return {"success": False, "error": str(e)}


# 사용 예시
if __name__ == "__main__":
    async def test_truth_seeking_engine():
        engine = TruthSeekingEngine()

        # 테스트 주장들
        test_claims = [
            "Drinking 8 glasses of water daily is necessary for good health.",
            "Vaccines cause autism.",
            "Climate change is primarily caused by human activities.",
            "The Earth is approximately 4.5 billion years old.",
            "Organic foods are significantly more nutritious than conventional foods."
        ]

        print("=" * 80)
        print("TRUTH SEEKING ENGINE TEST")
        print("=" * 80)

        for i, claim in enumerate(test_claims, 1):
            print(f"\n{i}. CLAIM: {claim}")
            print("-" * 60)

            # 종합 평가
            result = await engine.seek_truth(claim)

            if result["success"]:
                assessment = result["data"]
                truth_score = assessment.truth_score

                print(f"Truth Score: {truth_score.to_percentage():.1f}%")
                print(f"Confidence Level: {truth_score.confidence_level.name}")
                print(f"Risk Assessment: {truth_score.get_risk_assessment()}")

                print(f"\nDetailed Scores:")
                print(f"  Evidence Quality: {truth_score.evidence_score:.3f}")
                print(f"  Source Credibility: {truth_score.source_score:.3f}")
                print(f"  Consistency: {truth_score.consistency_score:.3f}")
                print(f"  Verifiability: {truth_score.verifiability_score:.3f}")

                print(f"\nRisk Factors:")
                print(f"  Bias Risk: {truth_score.bias_risk:.3f}")
                print(f"  Misinformation Risk: {truth_score.misinformation_risk:.3f}")
                print(f"  Uncertainty Level: {truth_score.uncertainty_level:.3f}")

                print(f"\nKey Findings:")
                for finding in assessment.key_findings:
                    print(f"  • {finding}")

                print(f"\nRecommendations:")
                for rec in assessment.recommendations:
                    print(f"  • {rec}")

                if assessment.further_research_needed:
                    print(f"\nFurther Research Needed:")
                    for research in assessment.further_research_needed:
                        print(f"  • {research}")

            else:
                print(f"Assessment failed: {result['error']}")

        # 간단한 검증 테스트
        print(f"\n\n" + "=" * 80)
        print("SIMPLE VERIFICATION TEST")
        print("=" * 80)

        simple_claims = [
            "Water boils at 100°C at sea level.",
            "The moon is made of cheese."
        ]

        for claim in simple_claims:
            verification_result = await engine.verify_claim(claim, ConfidenceLevel.MODERATE)
            if verification_result["success"]:
                is_verified = verification_result["data"]
                metadata = verification_result["metadata"]

                print(f"\nClaim: {claim}")
                print(f"Verified: {is_verified}")
                print(f"Truth Score: {metadata['truth_score']:.1%}")
                print(f"Confidence: {metadata['confidence_level']}")
                print(f"Risk: {metadata['risk_assessment']}")

        # 통계 조회
        stats = await engine.truth_assessment.get_assessment_statistics()
        print(f"\n\n" + "=" * 80)
        print("ASSESSMENT STATISTICS")
        print("=" * 80)
        print(f"Total Assessments: {stats['total_assessments']}")
        print(f"Average Truth Score: {stats['average_truth_score']:.1%}")
        print(f"High Quality Percentage: {stats['high_quality_percentage']:.1f}%")
        print(f"Average Consensus Level: {stats['average_consensus_level']:.1%}")
        print(f"Confidence Distribution: {stats['confidence_distribution']}")

    # 테스트 실행
    asyncio.run(test_truth_seeking_engine())