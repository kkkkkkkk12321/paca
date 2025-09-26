"""
Evidence Evaluator Module
증거 평가 시스템

이 모듈은 다양한 증거의 품질과 신뢰성을 평가합니다.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
import asyncio
import json

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


class EvidenceType(Enum):
    """증거 유형"""
    EMPIRICAL = auto()          # 경험적 증거
    STATISTICAL = auto()        # 통계적 증거
    TESTIMONIAL = auto()        # 증언/목격 증거
    DOCUMENTARY = auto()        # 문서 증거
    EXPERT_OPINION = auto()     # 전문가 의견
    SCIENTIFIC_STUDY = auto()   # 과학적 연구
    LOGICAL_REASONING = auto()  # 논리적 추론
    ANECDOTAL = auto()         # 일화적 증거
    CIRCUMSTANTIAL = auto()    # 정황 증거


class EvidenceQuality(Enum):
    """증거 품질 등급"""
    EXCELLENT = auto()      # 매우 높음 (90-100%)
    GOOD = auto()          # 좋음 (70-89%)
    FAIR = auto()          # 보통 (50-69%)
    POOR = auto()          # 낮음 (30-49%)
    VERY_POOR = auto()     # 매우 낮음 (0-29%)


class SourceReliability(Enum):
    """정보원 신뢰성"""
    HIGHLY_RELIABLE = auto()    # 매우 신뢰할 만함
    RELIABLE = auto()          # 신뢰할 만함
    MODERATELY_RELIABLE = auto() # 보통 신뢰함
    QUESTIONABLE = auto()      # 의문스러움
    UNRELIABLE = auto()        # 신뢰할 수 없음


@dataclass
class CredibilityScore:
    """신뢰성 점수"""
    overall_score: float        # 전체 점수 (0.0-1.0)
    source_credibility: float   # 정보원 신뢰성
    evidence_strength: float    # 증거 강도
    methodology_quality: float  # 방법론 품질
    consistency_score: float    # 일관성 점수
    bias_factor: float         # 편향 요소 (낮을수록 좋음)

    def to_percentage(self) -> float:
        """백분율로 변환"""
        return self.overall_score * 100


@dataclass
class EvidenceAssessment:
    """증거 평가 결과"""
    evidence_id: ID
    evidence_type: EvidenceType
    quality_rating: EvidenceQuality
    credibility_score: CredibilityScore
    source_reliability: SourceReliability

    # 세부 평가 항목
    relevance_score: float      # 관련성 (0.0-1.0)
    recency_score: float        # 최신성 (0.0-1.0)
    independence_score: float   # 독립성 (0.0-1.0)
    verifiability_score: float  # 검증가능성 (0.0-1.0)

    # 평가 메타데이터
    assessment_timestamp: datetime
    assessor_id: Optional[ID] = None
    assessment_method: str = "automated"
    confidence_level: float = 0.0

    # 추가 정보
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    cross_references: List[ID] = field(default_factory=list)

    def get_weighted_score(self) -> float:
        """가중 평균 점수 계산"""
        weights = {
            'credibility': 0.3,
            'relevance': 0.25,
            'recency': 0.15,
            'independence': 0.15,
            'verifiability': 0.15
        }

        return (
            self.credibility_score.overall_score * weights['credibility'] +
            self.relevance_score * weights['relevance'] +
            self.recency_score * weights['recency'] +
            self.independence_score * weights['independence'] +
            self.verifiability_score * weights['verifiability']
        )


class EvidenceEvaluator:
    """증거 평가 시스템"""

    def __init__(self):
        self.assessments: Dict[ID, EvidenceAssessment] = {}
        self.evaluation_criteria = self._initialize_evaluation_criteria()
        self.quality_thresholds = self._initialize_quality_thresholds()
        self.bias_indicators = self._initialize_bias_indicators()

    def _initialize_evaluation_criteria(self) -> Dict[str, Dict[str, float]]:
        """평가 기준 초기화"""
        return {
            EvidenceType.SCIENTIFIC_STUDY.name: {
                'peer_review_weight': 0.3,
                'sample_size_weight': 0.2,
                'methodology_weight': 0.25,
                'replication_weight': 0.15,
                'publication_weight': 0.1
            },
            EvidenceType.STATISTICAL.name: {
                'sample_size_weight': 0.3,
                'methodology_weight': 0.25,
                'margin_of_error_weight': 0.2,
                'source_weight': 0.15,
                'currency_weight': 0.1
            },
            EvidenceType.EXPERT_OPINION.name: {
                'expertise_weight': 0.35,
                'credentials_weight': 0.25,
                'consensus_weight': 0.2,
                'independence_weight': 0.2
            },
            EvidenceType.TESTIMONIAL.name: {
                'credibility_weight': 0.3,
                'consistency_weight': 0.25,
                'corroboration_weight': 0.25,
                'bias_weight': 0.2
            }
        }

    def _initialize_quality_thresholds(self) -> Dict[EvidenceQuality, Tuple[float, float]]:
        """품질 임계값 초기화"""
        return {
            EvidenceQuality.EXCELLENT: (0.9, 1.0),
            EvidenceQuality.GOOD: (0.7, 0.89),
            EvidenceQuality.FAIR: (0.5, 0.69),
            EvidenceQuality.POOR: (0.3, 0.49),
            EvidenceQuality.VERY_POOR: (0.0, 0.29)
        }

    def _initialize_bias_indicators(self) -> Dict[str, List[str]]:
        """편향 지표 초기화"""
        return {
            'confirmation_bias': [
                'selective_data_presentation',
                'cherry_picking_evidence',
                'ignoring_contradictory_evidence'
            ],
            'source_bias': [
                'financial_interest',
                'political_affiliation',
                'ideological_stance'
            ],
            'methodology_bias': [
                'leading_questions',
                'small_sample_size',
                'self_selection_bias'
            ]
        }

    async def evaluate_evidence(
        self,
        evidence_data: Dict[str, Any],
        evidence_type: EvidenceType,
        source_info: Optional[Dict[str, Any]] = None
    ) -> Result[EvidenceAssessment]:
        """증거 평가 수행"""
        try:
            evidence_id = generate_id()

            # 기본 평가 수행
            credibility_score = await self._calculate_credibility_score(
                evidence_data, evidence_type, source_info
            )

            source_reliability = self._assess_source_reliability(source_info)
            quality_rating = self._determine_quality_rating(credibility_score.overall_score)

            # 세부 평가 항목 계산
            relevance_score = self._calculate_relevance_score(evidence_data)
            recency_score = self._calculate_recency_score(evidence_data)
            independence_score = self._calculate_independence_score(evidence_data, source_info)
            verifiability_score = self._calculate_verifiability_score(evidence_data)

            # 강점, 약점, 권장사항 분석
            strengths, weaknesses = self._analyze_strengths_weaknesses(
                evidence_data, evidence_type, credibility_score
            )
            recommendations = self._generate_recommendations(weaknesses, evidence_type)

            assessment = EvidenceAssessment(
                evidence_id=evidence_id,
                evidence_type=evidence_type,
                quality_rating=quality_rating,
                credibility_score=credibility_score,
                source_reliability=source_reliability,
                relevance_score=relevance_score,
                recency_score=recency_score,
                independence_score=independence_score,
                verifiability_score=verifiability_score,
                assessment_timestamp=current_timestamp(),
                confidence_level=self._calculate_confidence_level(credibility_score),
                strengths=strengths,
                weaknesses=weaknesses,
                recommendations=recommendations
            )

            self.assessments[evidence_id] = assessment

            return {"success": True, "data": assessment}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _calculate_credibility_score(
        self,
        evidence_data: Dict[str, Any],
        evidence_type: EvidenceType,
        source_info: Optional[Dict[str, Any]]
    ) -> CredibilityScore:
        """신뢰성 점수 계산"""

        # 정보원 신뢰성 평가
        source_credibility = self._evaluate_source_credibility(source_info)

        # 증거 강도 평가
        evidence_strength = self._evaluate_evidence_strength(evidence_data, evidence_type)

        # 방법론 품질 평가
        methodology_quality = self._evaluate_methodology_quality(evidence_data, evidence_type)

        # 일관성 점수 계산
        consistency_score = self._evaluate_consistency(evidence_data)

        # 편향 요소 평가
        bias_factor = self._evaluate_bias_factor(evidence_data, source_info)

        # 전체 점수 계산 (가중 평균)
        overall_score = (
            source_credibility * 0.25 +
            evidence_strength * 0.3 +
            methodology_quality * 0.2 +
            consistency_score * 0.15 +
            (1.0 - bias_factor) * 0.1  # 편향이 낮을수록 좋음
        )

        return CredibilityScore(
            overall_score=min(max(overall_score, 0.0), 1.0),
            source_credibility=source_credibility,
            evidence_strength=evidence_strength,
            methodology_quality=methodology_quality,
            consistency_score=consistency_score,
            bias_factor=bias_factor
        )

    def _evaluate_source_credibility(self, source_info: Optional[Dict[str, Any]]) -> float:
        """정보원 신뢰성 평가"""
        if not source_info:
            return 0.3  # 기본 낮은 점수

        score = 0.5  # 기본 점수

        # 출처 유형에 따른 점수 조정
        source_type = source_info.get('type', '').lower()
        if source_type in ['academic_journal', 'peer_reviewed']:
            score += 0.3
        elif source_type in ['government', 'official_statistics']:
            score += 0.25
        elif source_type in ['news_media', 'established_publication']:
            score += 0.15
        elif source_type in ['blog', 'social_media']:
            score -= 0.2

        # 기관 신뢰도
        institution_score = source_info.get('institution_credibility', 0.5)
        score = (score + institution_score) / 2

        # 저자 전문성
        author_expertise = source_info.get('author_expertise', 0.5)
        score = (score * 0.7) + (author_expertise * 0.3)

        return min(max(score, 0.0), 1.0)

    def _evaluate_evidence_strength(self, evidence_data: Dict[str, Any], evidence_type: EvidenceType) -> float:
        """증거 강도 평가"""
        base_strength = {
            EvidenceType.SCIENTIFIC_STUDY: 0.8,
            EvidenceType.STATISTICAL: 0.75,
            EvidenceType.EMPIRICAL: 0.7,
            EvidenceType.EXPERT_OPINION: 0.6,
            EvidenceType.DOCUMENTARY: 0.55,
            EvidenceType.TESTIMONIAL: 0.4,
            EvidenceType.LOGICAL_REASONING: 0.5,
            EvidenceType.CIRCUMSTANTIAL: 0.3,
            EvidenceType.ANECDOTAL: 0.2
        }.get(evidence_type, 0.5)

        # 증거 특성에 따른 조정
        sample_size = evidence_data.get('sample_size', 0)
        if sample_size > 1000:
            base_strength += 0.1
        elif sample_size < 30:
            base_strength -= 0.15

        # 복제 연구 여부
        if evidence_data.get('replicated', False):
            base_strength += 0.15

        # 메타분석 여부
        if evidence_data.get('meta_analysis', False):
            base_strength += 0.2

        return min(max(base_strength, 0.0), 1.0)

    def _evaluate_methodology_quality(self, evidence_data: Dict[str, Any], evidence_type: EvidenceType) -> float:
        """방법론 품질 평가"""
        base_quality = 0.5

        # 연구 설계 품질
        study_design = evidence_data.get('study_design', '').lower()
        if study_design in ['randomized_controlled_trial', 'rct']:
            base_quality += 0.3
        elif study_design in ['cohort_study', 'case_control']:
            base_quality += 0.2
        elif study_design in ['cross_sectional', 'survey']:
            base_quality += 0.1
        elif study_design in ['case_study', 'anecdotal']:
            base_quality -= 0.2

        # 통계적 유의성
        if evidence_data.get('statistical_significance', False):
            base_quality += 0.15

        # 신뢰구간 제공 여부
        if evidence_data.get('confidence_interval', False):
            base_quality += 0.1

        # 효과 크기 보고 여부
        if evidence_data.get('effect_size', False):
            base_quality += 0.1

        return min(max(base_quality, 0.0), 1.0)

    def _evaluate_consistency(self, evidence_data: Dict[str, Any]) -> float:
        """일관성 평가"""
        consistency_score = 0.5

        # 내부 일관성
        internal_consistency = evidence_data.get('internal_consistency', 0.5)
        consistency_score = (consistency_score + internal_consistency) / 2

        # 다른 연구와의 일관성
        external_consistency = evidence_data.get('external_consistency', 0.5)
        consistency_score = (consistency_score + external_consistency) / 2

        return min(max(consistency_score, 0.0), 1.0)

    def _evaluate_bias_factor(self, evidence_data: Dict[str, Any], source_info: Optional[Dict[str, Any]]) -> float:
        """편향 요소 평가 (높을수록 편향이 많음)"""
        bias_score = 0.0

        # 이해관계 충돌
        if evidence_data.get('conflict_of_interest', False):
            bias_score += 0.3

        # 선택적 데이터 제시
        if evidence_data.get('selective_reporting', False):
            bias_score += 0.25

        # 정치적/이념적 편향
        if source_info and source_info.get('political_bias', False):
            bias_score += 0.2

        # 확증 편향 지표
        if evidence_data.get('confirmation_bias_indicators', 0) > 2:
            bias_score += 0.15

        return min(bias_score, 1.0)

    def _assess_source_reliability(self, source_info: Optional[Dict[str, Any]]) -> SourceReliability:
        """정보원 신뢰성 등급 결정"""
        if not source_info:
            return SourceReliability.QUESTIONABLE

        credibility = self._evaluate_source_credibility(source_info)

        if credibility >= 0.8:
            return SourceReliability.HIGHLY_RELIABLE
        elif credibility >= 0.65:
            return SourceReliability.RELIABLE
        elif credibility >= 0.45:
            return SourceReliability.MODERATELY_RELIABLE
        elif credibility >= 0.25:
            return SourceReliability.QUESTIONABLE
        else:
            return SourceReliability.UNRELIABLE

    def _determine_quality_rating(self, overall_score: float) -> EvidenceQuality:
        """품질 등급 결정"""
        for quality, (min_score, max_score) in self.quality_thresholds.items():
            if min_score <= overall_score <= max_score:
                return quality
        return EvidenceQuality.VERY_POOR

    def _calculate_relevance_score(self, evidence_data: Dict[str, Any]) -> float:
        """관련성 점수 계산"""
        # 주제 관련성
        topic_relevance = evidence_data.get('topic_relevance', 0.5)

        # 맥락 적합성
        context_relevance = evidence_data.get('context_relevance', 0.5)

        # 직접성 vs 간접성
        directness = evidence_data.get('directness', 0.5)

        return (topic_relevance * 0.4 + context_relevance * 0.35 + directness * 0.25)

    def _calculate_recency_score(self, evidence_data: Dict[str, Any]) -> float:
        """최신성 점수 계산"""
        publication_date = evidence_data.get('publication_date')
        if not publication_date:
            return 0.3

        if isinstance(publication_date, str):
            try:
                publication_date = datetime.fromisoformat(publication_date.replace('Z', '+00:00'))
            except:
                return 0.3

        now = datetime.now()
        age_days = (now - publication_date).days

        # 1년 이내: 최고 점수
        if age_days <= 365:
            return 1.0
        # 3년 이내: 높은 점수
        elif age_days <= 1095:
            return 0.8
        # 5년 이내: 보통 점수
        elif age_days <= 1825:
            return 0.6
        # 10년 이내: 낮은 점수
        elif age_days <= 3650:
            return 0.4
        else:
            return 0.2

    def _calculate_independence_score(self, evidence_data: Dict[str, Any], source_info: Optional[Dict[str, Any]]) -> float:
        """독립성 점수 계산"""
        independence = 0.7  # 기본값

        # 이해관계 충돌
        if evidence_data.get('conflict_of_interest', False):
            independence -= 0.3

        # 자금 지원 독립성
        funding_independence = evidence_data.get('funding_independence', 0.7)
        independence = (independence + funding_independence) / 2

        # 기관 독립성
        if source_info:
            institutional_independence = source_info.get('institutional_independence', 0.7)
            independence = (independence + institutional_independence) / 2

        return min(max(independence, 0.0), 1.0)

    def _calculate_verifiability_score(self, evidence_data: Dict[str, Any]) -> float:
        """검증가능성 점수 계산"""
        verifiability = 0.5

        # 원본 데이터 접근 가능성
        if evidence_data.get('raw_data_available', False):
            verifiability += 0.2

        # 방법론 상세 설명
        if evidence_data.get('detailed_methodology', False):
            verifiability += 0.15

        # 재현 가능성
        if evidence_data.get('reproducible', False):
            verifiability += 0.2

        # 동료 검토
        if evidence_data.get('peer_reviewed', False):
            verifiability += 0.15

        return min(verifiability, 1.0)

    def _calculate_confidence_level(self, credibility_score: CredibilityScore) -> float:
        """신뢰도 계산"""
        # 전체 점수와 편향 요소를 고려한 신뢰도
        base_confidence = credibility_score.overall_score
        bias_penalty = credibility_score.bias_factor * 0.3

        return min(max(base_confidence - bias_penalty, 0.0), 1.0)

    def _analyze_strengths_weaknesses(
        self,
        evidence_data: Dict[str, Any],
        evidence_type: EvidenceType,
        credibility_score: CredibilityScore
    ) -> Tuple[List[str], List[str]]:
        """강점과 약점 분석"""
        strengths = []
        weaknesses = []

        # 신뢰성 점수 기반 분석
        if credibility_score.source_credibility > 0.7:
            strengths.append("신뢰할 만한 정보원")
        elif credibility_score.source_credibility < 0.4:
            weaknesses.append("정보원의 신뢰성이 낮음")

        if credibility_score.evidence_strength > 0.7:
            strengths.append("강력한 증거 강도")
        elif credibility_score.evidence_strength < 0.4:
            weaknesses.append("증거 강도가 약함")

        if credibility_score.methodology_quality > 0.7:
            strengths.append("우수한 방법론")
        elif credibility_score.methodology_quality < 0.4:
            weaknesses.append("방법론적 한계")

        if credibility_score.bias_factor < 0.3:
            strengths.append("편향 요소가 적음")
        elif credibility_score.bias_factor > 0.6:
            weaknesses.append("상당한 편향 가능성")

        # 증거 유형별 특성 분석
        if evidence_type == EvidenceType.SCIENTIFIC_STUDY:
            if evidence_data.get('peer_reviewed', False):
                strengths.append("동료 검토를 거친 연구")
            if evidence_data.get('large_sample_size', False):
                strengths.append("충분한 표본 크기")

        return strengths, weaknesses

    def _generate_recommendations(self, weaknesses: List[str], evidence_type: EvidenceType) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []

        for weakness in weaknesses:
            if "신뢰성이 낮음" in weakness:
                recommendations.append("더 신뢰할 만한 정보원을 찾아 교차 검증")
            elif "증거 강도가 약함" in weakness:
                recommendations.append("추가적인 증거나 더 강력한 연구 결과 확인")
            elif "방법론적 한계" in weakness:
                recommendations.append("방법론이 우수한 연구나 메타분석 참조")
            elif "편향 가능성" in weakness:
                recommendations.append("독립적인 출처에서 정보 교차 확인")

        # 증거 유형별 권장사항
        if evidence_type == EvidenceType.ANECDOTAL:
            recommendations.append("일화적 증거이므로 더 체계적인 연구 결과로 보완 필요")
        elif evidence_type == EvidenceType.TESTIMONIAL:
            recommendations.append("증언 증거이므로 객관적 데이터로 뒷받침 필요")

        return recommendations

    async def get_assessment(self, evidence_id: ID) -> Optional[EvidenceAssessment]:
        """증거 평가 결과 조회"""
        return self.assessments.get(evidence_id)

    async def list_assessments_by_quality(self, min_quality: EvidenceQuality) -> List[EvidenceAssessment]:
        """품질 기준으로 평가 결과 목록 조회"""
        quality_order = {
            EvidenceQuality.EXCELLENT: 5,
            EvidenceQuality.GOOD: 4,
            EvidenceQuality.FAIR: 3,
            EvidenceQuality.POOR: 2,
            EvidenceQuality.VERY_POOR: 1
        }

        min_level = quality_order[min_quality]

        return [
            assessment for assessment in self.assessments.values()
            if quality_order[assessment.quality_rating] >= min_level
        ]

    async def get_assessment_summary(self) -> Dict[str, Any]:
        """평가 결과 요약 통계"""
        if not self.assessments:
            return {"total_assessments": 0}

        total = len(self.assessments)
        quality_counts = {}
        avg_score = sum(a.credibility_score.overall_score for a in self.assessments.values()) / total

        for quality in EvidenceQuality:
            count = sum(1 for a in self.assessments.values() if a.quality_rating == quality)
            quality_counts[quality.name] = count

        return {
            "total_assessments": total,
            "average_credibility_score": avg_score,
            "quality_distribution": quality_counts,
            "high_quality_percentage": (
                quality_counts.get("EXCELLENT", 0) + quality_counts.get("GOOD", 0)
            ) / total * 100 if total > 0 else 0
        }


# 사용 예시
if __name__ == "__main__":
    async def test_evidence_evaluator():
        evaluator = EvidenceEvaluator()

        # 테스트 증거 데이터
        evidence_data = {
            "sample_size": 1500,
            "study_design": "randomized_controlled_trial",
            "peer_reviewed": True,
            "statistical_significance": True,
            "effect_size": True,
            "confidence_interval": True,
            "publication_date": "2023-01-15",
            "topic_relevance": 0.9,
            "context_relevance": 0.85,
            "directness": 0.8,
            "conflict_of_interest": False,
            "replicated": True
        }

        source_info = {
            "type": "academic_journal",
            "institution_credibility": 0.9,
            "author_expertise": 0.85,
            "institutional_independence": 0.8,
            "funding_independence": 0.7
        }

        # 증거 평가
        result = await evaluator.evaluate_evidence(
            evidence_data,
            EvidenceType.SCIENTIFIC_STUDY,
            source_info
        )

        if result["success"]:
            assessment = result["data"]
            print(f"증거 평가 완료: {assessment.evidence_id}")
            print(f"품질 등급: {assessment.quality_rating.name}")
            print(f"신뢰성 점수: {assessment.credibility_score.to_percentage():.1f}%")
            print(f"정보원 신뢰성: {assessment.source_reliability.name}")
            print(f"가중 점수: {assessment.get_weighted_score():.3f}")
            print(f"강점: {', '.join(assessment.strengths)}")
            if assessment.weaknesses:
                print(f"약점: {', '.join(assessment.weaknesses)}")
            if assessment.recommendations:
                print(f"권장사항: {', '.join(assessment.recommendations)}")
        else:
            print(f"평가 실패: {result['error']}")

    # 테스트 실행
    asyncio.run(test_evidence_evaluator())