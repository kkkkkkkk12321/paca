"""
Source Validator Module
정보원 검증 시스템

이 모듈은 정보원의 신뢰성과 품질을 종합적으로 검증합니다.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from datetime import datetime, timedelta
import asyncio
import json
import re

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


class SourceType(Enum):
    """정보원 유형"""
    ACADEMIC_JOURNAL = auto()      # 학술지
    PEER_REVIEWED = auto()         # 동료 검토 자료
    GOVERNMENT_OFFICIAL = auto()   # 정부 공식 자료
    NEWS_MEDIA = auto()           # 언론 매체
    EXPERT_TESTIMONY = auto()     # 전문가 증언
    STATISTICAL_AGENCY = auto()   # 통계 기관
    RESEARCH_INSTITUTE = auto()   # 연구 기관
    CORPORATE_REPORT = auto()     # 기업 보고서
    BLOG_PERSONAL = auto()        # 개인 블로그
    SOCIAL_MEDIA = auto()         # 소셜 미디어
    WIKIPEDIA = auto()            # 위키피디아
    DOCUMENTARY = auto()          # 다큐멘터리
    BOOK_PUBLISHED = auto()       # 출판 도서
    WEBSITE_GENERAL = auto()      # 일반 웹사이트


class ValidationCriteria(Enum):
    """검증 기준"""
    AUTHORSHIP = auto()           # 저자 신뢰성
    INSTITUTIONAL_BACKING = auto() # 기관 지원
    EDITORIAL_OVERSIGHT = auto()   # 편집 감독
    TRANSPARENCY = auto()         # 투명성
    METHODOLOGY = auto()          # 방법론
    CURRENCY = auto()            # 최신성
    OBJECTIVITY = auto()         # 객관성
    VERIFIABILITY = auto()       # 검증 가능성


@dataclass
class CredibilityMetrics:
    """신뢰성 지표"""
    authority_score: float         # 권위 점수 (0.0-1.0)
    accuracy_score: float          # 정확성 점수 (0.0-1.0)
    objectivity_score: float       # 객관성 점수 (0.0-1.0)
    transparency_score: float      # 투명성 점수 (0.0-1.0)
    currency_score: float          # 최신성 점수 (0.0-1.0)
    coverage_score: float          # 포괄성 점수 (0.0-1.0)

    def overall_score(self) -> float:
        """전체 신뢰성 점수 계산"""
        weights = {
            'authority': 0.25,
            'accuracy': 0.25,
            'objectivity': 0.2,
            'transparency': 0.15,
            'currency': 0.1,
            'coverage': 0.05
        }

        return (
            self.authority_score * weights['authority'] +
            self.accuracy_score * weights['accuracy'] +
            self.objectivity_score * weights['objectivity'] +
            self.transparency_score * weights['transparency'] +
            self.currency_score * weights['currency'] +
            self.coverage_score * weights['coverage']
        )


@dataclass
class SourceProfile:
    """정보원 프로필"""
    source_id: ID
    source_type: SourceType
    name: str
    url: Optional[str] = None

    # 기본 정보
    publisher: Optional[str] = None
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    # 기관 정보
    institution: Optional[str] = None
    institutional_affiliation: Optional[str] = None
    editorial_board: Optional[List[str]] = field(default_factory=list)

    # 품질 지표
    peer_reviewed: bool = False
    impact_factor: Optional[float] = None
    citation_count: Optional[int] = None

    # 편향 및 신뢰성 지표
    political_bias: Optional[str] = None
    commercial_interest: bool = False
    funding_source: Optional[str] = None

    # 기술적 정보
    domain_age: Optional[timedelta] = None
    ssl_certificate: bool = False
    contact_information: bool = False

    # 메타데이터
    profile_created: datetime = field(default_factory=current_timestamp)
    last_validated: Optional[datetime] = None
    validation_count: int = 0


@dataclass
class ValidationResult:
    """검증 결과"""
    source_id: ID
    validation_id: ID

    # 검증 점수
    credibility_metrics: CredibilityMetrics
    overall_credibility: float
    confidence_level: float

    # 검증 세부사항
    passed_criteria: List[ValidationCriteria]
    failed_criteria: List[ValidationCriteria]
    warning_flags: List[str]

    # 추천 사항
    recommendation: str  # "HIGHLY_RECOMMENDED", "RECOMMENDED", "CAUTION", "NOT_RECOMMENDED"
    risk_level: str     # "LOW", "MEDIUM", "HIGH", "CRITICAL"

    # 메타데이터
    validation_timestamp: datetime
    validator_version: str = "1.0.0"
    validation_method: str = "automated"

    # 상세 분석
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    alternative_sources: List[str] = field(default_factory=list)


class SourceValidator:
    """정보원 검증 시스템"""

    def __init__(self):
        self.source_profiles: Dict[ID, SourceProfile] = {}
        self.validation_results: Dict[ID, ValidationResult] = {}
        self.validation_criteria_weights = self._initialize_criteria_weights()
        self.source_type_baselines = self._initialize_source_type_baselines()
        self.red_flag_patterns = self._initialize_red_flag_patterns()
        self.trusted_domains = self._initialize_trusted_domains()

    def _initialize_criteria_weights(self) -> Dict[ValidationCriteria, float]:
        """검증 기준별 가중치 초기화"""
        return {
            ValidationCriteria.AUTHORSHIP: 0.25,
            ValidationCriteria.INSTITUTIONAL_BACKING: 0.20,
            ValidationCriteria.EDITORIAL_OVERSIGHT: 0.15,
            ValidationCriteria.TRANSPARENCY: 0.15,
            ValidationCriteria.METHODOLOGY: 0.10,
            ValidationCriteria.CURRENCY: 0.08,
            ValidationCriteria.OBJECTIVITY: 0.04,
            ValidationCriteria.VERIFIABILITY: 0.03
        }

    def _initialize_source_type_baselines(self) -> Dict[SourceType, Dict[str, float]]:
        """정보원 유형별 기준점 설정"""
        return {
            SourceType.ACADEMIC_JOURNAL: {
                'authority': 0.9, 'accuracy': 0.85, 'objectivity': 0.8,
                'transparency': 0.75, 'currency': 0.7, 'coverage': 0.8
            },
            SourceType.PEER_REVIEWED: {
                'authority': 0.85, 'accuracy': 0.9, 'objectivity': 0.85,
                'transparency': 0.8, 'currency': 0.75, 'coverage': 0.75
            },
            SourceType.GOVERNMENT_OFFICIAL: {
                'authority': 0.8, 'accuracy': 0.75, 'objectivity': 0.6,
                'transparency': 0.7, 'currency': 0.8, 'coverage': 0.85
            },
            SourceType.NEWS_MEDIA: {
                'authority': 0.6, 'accuracy': 0.65, 'objectivity': 0.5,
                'transparency': 0.6, 'currency': 0.9, 'coverage': 0.7
            },
            SourceType.RESEARCH_INSTITUTE: {
                'authority': 0.75, 'accuracy': 0.8, 'objectivity': 0.7,
                'transparency': 0.7, 'currency': 0.75, 'coverage': 0.75
            },
            SourceType.BLOG_PERSONAL: {
                'authority': 0.3, 'accuracy': 0.4, 'objectivity': 0.4,
                'transparency': 0.5, 'currency': 0.8, 'coverage': 0.4
            },
            SourceType.SOCIAL_MEDIA: {
                'authority': 0.2, 'accuracy': 0.3, 'objectivity': 0.3,
                'transparency': 0.4, 'currency': 0.95, 'coverage': 0.3
            },
            SourceType.WIKIPEDIA: {
                'authority': 0.5, 'accuracy': 0.7, 'objectivity': 0.65,
                'transparency': 0.9, 'currency': 0.8, 'coverage': 0.85
            }
        }

    def _initialize_red_flag_patterns(self) -> Dict[str, List[str]]:
        """주의 패턴 초기화"""
        return {
            'unreliable_language': [
                'secret', 'they don\'t want you to know',
                'miracle cure', 'guaranteed', 'instant',
                'revolutionary breakthrough', 'exclusive'
            ],
            'bias_indicators': [
                'always', 'never', 'all experts agree',
                'proven fact', 'undeniable truth',
                'everyone knows', 'obviously'
            ],
            'commercial_bias': [
                'buy now', 'limited time', 'special offer',
                'sponsored content', 'affiliate link',
                'advertisement', 'promotional'
            ],
            'conspiracy_markers': [
                'cover-up', 'conspiracy', 'hidden agenda',
                'mainstream media lies', 'wake up',
                'sheeple', 'deep state'
            ]
        }

    def _initialize_trusted_domains(self) -> Dict[str, float]:
        """신뢰할 만한 도메인 목록"""
        return {
            # 학술 기관
            'pubmed.ncbi.nlm.nih.gov': 0.95,
            'scholar.google.com': 0.9,
            'arxiv.org': 0.85,
            'jstor.org': 0.9,

            # 정부 기관
            'gov': 0.8,  # 정부 도메인 일반
            'who.int': 0.9,
            'cdc.gov': 0.85,
            'nih.gov': 0.9,

            # 신뢰할 만한 언론
            'reuters.com': 0.75,
            'ap.org': 0.75,
            'bbc.com': 0.7,

            # 국제 기구
            'un.org': 0.8,
            'worldbank.org': 0.75,
            'oecd.org': 0.8,

            # 위험한 도메인 (낮은 점수)
            'naturalnews.com': 0.1,
            'infowars.com': 0.05,
        }

    async def create_source_profile(
        self,
        source_data: Dict[str, Any],
        source_type: SourceType
    ) -> Result[SourceProfile]:
        """정보원 프로필 생성"""
        try:
            source_id = generate_id()

            # 날짜 파싱
            publication_date = self._parse_date(source_data.get('publication_date'))
            last_updated = self._parse_date(source_data.get('last_updated'))

            profile = SourceProfile(
                source_id=source_id,
                source_type=source_type,
                name=source_data.get('name', ''),
                url=source_data.get('url'),
                publisher=source_data.get('publisher'),
                author=source_data.get('author'),
                publication_date=publication_date,
                last_updated=last_updated,
                institution=source_data.get('institution'),
                institutional_affiliation=source_data.get('institutional_affiliation'),
                editorial_board=source_data.get('editorial_board', []),
                peer_reviewed=source_data.get('peer_reviewed', False),
                impact_factor=source_data.get('impact_factor'),
                citation_count=source_data.get('citation_count'),
                political_bias=source_data.get('political_bias'),
                commercial_interest=source_data.get('commercial_interest', False),
                funding_source=source_data.get('funding_source'),
                domain_age=self._parse_timedelta(source_data.get('domain_age')),
                ssl_certificate=source_data.get('ssl_certificate', False),
                contact_information=source_data.get('contact_information', False)
            )

            self.source_profiles[source_id] = profile

            return {"success": True, "data": profile}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def validate_source(
        self,
        source_id: ID,
        validation_context: Optional[Dict[str, Any]] = None
    ) -> Result[ValidationResult]:
        """정보원 검증 수행"""
        try:
            profile = self.source_profiles.get(source_id)
            if not profile:
                return {"success": False, "error": "Source profile not found"}

            validation_id = generate_id()

            # 신뢰성 지표 계산
            credibility_metrics = await self._calculate_credibility_metrics(
                profile, validation_context
            )

            # 검증 기준 평가
            passed_criteria, failed_criteria = await self._evaluate_validation_criteria(
                profile, credibility_metrics
            )

            # 경고 플래그 확인
            warning_flags = await self._check_warning_flags(profile, validation_context)

            # 전체 신뢰성 및 신뢰도 계산
            overall_credibility = credibility_metrics.overall_score()
            confidence_level = self._calculate_confidence_level(
                profile, credibility_metrics, len(warning_flags)
            )

            # 추천 및 위험 수준 결정
            recommendation = self._determine_recommendation(overall_credibility, warning_flags)
            risk_level = self._assess_risk_level(overall_credibility, warning_flags)

            # 강점, 약점, 개선 사항 분석
            strengths, weaknesses, improvements = await self._analyze_source_quality(
                profile, credibility_metrics, warning_flags
            )

            # 대안 정보원 제안
            alternative_sources = await self._suggest_alternative_sources(
                profile, overall_credibility
            )

            validation_result = ValidationResult(
                source_id=source_id,
                validation_id=validation_id,
                credibility_metrics=credibility_metrics,
                overall_credibility=overall_credibility,
                confidence_level=confidence_level,
                passed_criteria=passed_criteria,
                failed_criteria=failed_criteria,
                warning_flags=warning_flags,
                recommendation=recommendation,
                risk_level=risk_level,
                validation_timestamp=current_timestamp(),
                strengths=strengths,
                weaknesses=weaknesses,
                improvement_suggestions=improvements,
                alternative_sources=alternative_sources
            )

            self.validation_results[validation_id] = validation_result

            # 프로필 업데이트
            profile.last_validated = current_timestamp()
            profile.validation_count += 1

            return {"success": True, "data": validation_result}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _calculate_credibility_metrics(
        self,
        profile: SourceProfile,
        context: Optional[Dict[str, Any]]
    ) -> CredibilityMetrics:
        """신뢰성 지표 계산"""

        # 기준 점수 가져오기
        baseline = self.source_type_baselines.get(profile.source_type, {
            'authority': 0.5, 'accuracy': 0.5, 'objectivity': 0.5,
            'transparency': 0.5, 'currency': 0.5, 'coverage': 0.5
        })

        # 권위 점수 계산
        authority_score = await self._calculate_authority_score(profile, baseline['authority'])

        # 정확성 점수 계산
        accuracy_score = await self._calculate_accuracy_score(profile, baseline['accuracy'])

        # 객관성 점수 계산
        objectivity_score = await self._calculate_objectivity_score(profile, baseline['objectivity'])

        # 투명성 점수 계산
        transparency_score = await self._calculate_transparency_score(profile, baseline['transparency'])

        # 최신성 점수 계산
        currency_score = await self._calculate_currency_score(profile, baseline['currency'])

        # 포괄성 점수 계산
        coverage_score = await self._calculate_coverage_score(profile, baseline['coverage'])

        return CredibilityMetrics(
            authority_score=authority_score,
            accuracy_score=accuracy_score,
            objectivity_score=objectivity_score,
            transparency_score=transparency_score,
            currency_score=currency_score,
            coverage_score=coverage_score
        )

    async def _calculate_authority_score(self, profile: SourceProfile, baseline: float) -> float:
        """권위 점수 계산"""
        score = baseline

        # 기관 지원
        if profile.institution:
            score += 0.1
            if any(keyword in profile.institution.lower() for keyword in
                   ['university', 'institute', 'academy', 'research']):
                score += 0.1

        # 동료 검토
        if profile.peer_reviewed:
            score += 0.15

        # 영향 지수
        if profile.impact_factor:
            if profile.impact_factor > 5.0:
                score += 0.1
            elif profile.impact_factor > 2.0:
                score += 0.05

        # 인용 횟수
        if profile.citation_count:
            if profile.citation_count > 100:
                score += 0.1
            elif profile.citation_count > 50:
                score += 0.05

        # 편집위원회
        if profile.editorial_board and len(profile.editorial_board) > 0:
            score += 0.05

        # 도메인 신뢰성
        if profile.url:
            domain_trust = self._get_domain_trust_score(profile.url)
            score = (score + domain_trust) / 2

        return min(max(score, 0.0), 1.0)

    async def _calculate_accuracy_score(self, profile: SourceProfile, baseline: float) -> float:
        """정확성 점수 계산"""
        score = baseline

        # 동료 검토 = 높은 정확성
        if profile.peer_reviewed:
            score += 0.2

        # 방법론 투명성
        if profile.source_type in [SourceType.ACADEMIC_JOURNAL, SourceType.RESEARCH_INSTITUTE]:
            score += 0.1

        # 편집 감독
        if profile.editorial_board:
            score += 0.1

        # 정정/업데이트 정책 (URL에서 추정)
        if profile.url and any(indicator in profile.url.lower() for indicator in
                              ['correction', 'retraction', 'update']):
            score += 0.05

        return min(max(score, 0.0), 1.0)

    async def _calculate_objectivity_score(self, profile: SourceProfile, baseline: float) -> float:
        """객관성 점수 계산"""
        score = baseline

        # 상업적 이해관계가 있으면 감점
        if profile.commercial_interest:
            score -= 0.2

        # 정치적 편향이 있으면 감점
        if profile.political_bias:
            bias_level = profile.political_bias.lower()
            if bias_level in ['strong', 'extreme']:
                score -= 0.3
            elif bias_level in ['moderate', 'slight']:
                score -= 0.15

        # 자금 출처 투명성
        if profile.funding_source:
            if 'government' in profile.funding_source.lower():
                score += 0.05
            elif 'corporate' in profile.funding_source.lower():
                score -= 0.1

        # 독립적 기관이면 가점
        if profile.institution and any(keyword in profile.institution.lower() for keyword in
                                      ['independent', 'non-profit', 'academic']):
            score += 0.1

        return min(max(score, 0.0), 1.0)

    async def _calculate_transparency_score(self, profile: SourceProfile, baseline: float) -> float:
        """투명성 점수 계산"""
        score = baseline

        # 저자 정보 제공
        if profile.author:
            score += 0.1

        # 기관 정보 제공
        if profile.institution:
            score += 0.1

        # 연락처 정보
        if profile.contact_information:
            score += 0.1

        # 자금 출처 공개
        if profile.funding_source:
            score += 0.1

        # 편집위원회 공개
        if profile.editorial_board:
            score += 0.05

        # SSL 인증서 (웹 보안)
        if profile.ssl_certificate:
            score += 0.05

        return min(max(score, 0.0), 1.0)

    async def _calculate_currency_score(self, profile: SourceProfile, baseline: float) -> float:
        """최신성 점수 계산"""
        score = baseline

        now = datetime.now()

        # 출판일 기준 최신성
        if profile.publication_date:
            days_old = (now - profile.publication_date).days
            if days_old <= 30:
                score += 0.2
            elif days_old <= 365:
                score += 0.1
            elif days_old <= 1825:  # 5년
                score += 0.05
            elif days_old > 3650:  # 10년
                score -= 0.2

        # 최근 업데이트
        if profile.last_updated:
            days_since_update = (now - profile.last_updated).days
            if days_since_update <= 30:
                score += 0.1
            elif days_since_update <= 365:
                score += 0.05

        # 도메인 연령 (오래된 도메인이 더 신뢰할 만함)
        if profile.domain_age:
            if profile.domain_age.days > 1825:  # 5년 이상
                score += 0.1
            elif profile.domain_age.days > 365:  # 1년 이상
                score += 0.05

        return min(max(score, 0.0), 1.0)

    async def _calculate_coverage_score(self, profile: SourceProfile, baseline: float) -> float:
        """포괄성 점수 계산"""
        score = baseline

        # 기관 규모/범위 추정
        if profile.institution:
            if any(keyword in profile.institution.lower() for keyword in
                   ['international', 'global', 'world', 'national']):
                score += 0.15
            elif any(keyword in profile.institution.lower() for keyword in
                     ['university', 'institute', 'center']):
                score += 0.1

        # 편집위원회 다양성
        if profile.editorial_board and len(profile.editorial_board) > 5:
            score += 0.1

        # 정보원 유형별 조정
        if profile.source_type in [SourceType.STATISTICAL_AGENCY, SourceType.GOVERNMENT_OFFICIAL]:
            score += 0.1  # 포괄적 데이터 제공 가능성

        return min(max(score, 0.0), 1.0)

    async def _evaluate_validation_criteria(
        self,
        profile: SourceProfile,
        metrics: CredibilityMetrics
    ) -> Tuple[List[ValidationCriteria], List[ValidationCriteria]]:
        """검증 기준 평가"""
        passed = []
        failed = []

        # 저자 신뢰성
        if profile.author and metrics.authority_score > 0.6:
            passed.append(ValidationCriteria.AUTHORSHIP)
        else:
            failed.append(ValidationCriteria.AUTHORSHIP)

        # 기관 지원
        if profile.institution and metrics.authority_score > 0.5:
            passed.append(ValidationCriteria.INSTITUTIONAL_BACKING)
        else:
            failed.append(ValidationCriteria.INSTITUTIONAL_BACKING)

        # 편집 감독
        if profile.peer_reviewed or profile.editorial_board:
            passed.append(ValidationCriteria.EDITORIAL_OVERSIGHT)
        else:
            failed.append(ValidationCriteria.EDITORIAL_OVERSIGHT)

        # 투명성
        if metrics.transparency_score > 0.6:
            passed.append(ValidationCriteria.TRANSPARENCY)
        else:
            failed.append(ValidationCriteria.TRANSPARENCY)

        # 방법론
        if profile.source_type in [SourceType.ACADEMIC_JOURNAL, SourceType.PEER_REVIEWED]:
            passed.append(ValidationCriteria.METHODOLOGY)
        elif metrics.accuracy_score < 0.5:
            failed.append(ValidationCriteria.METHODOLOGY)

        # 최신성
        if metrics.currency_score > 0.5:
            passed.append(ValidationCriteria.CURRENCY)
        else:
            failed.append(ValidationCriteria.CURRENCY)

        # 객관성
        if metrics.objectivity_score > 0.6:
            passed.append(ValidationCriteria.OBJECTIVITY)
        else:
            failed.append(ValidationCriteria.OBJECTIVITY)

        # 검증 가능성
        if profile.url and metrics.transparency_score > 0.5:
            passed.append(ValidationCriteria.VERIFIABILITY)
        else:
            failed.append(ValidationCriteria.VERIFIABILITY)

        return passed, failed

    async def _check_warning_flags(
        self,
        profile: SourceProfile,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """경고 플래그 확인"""
        warnings = []

        # 상업적 이해관계
        if profile.commercial_interest:
            warnings.append("Commercial interest detected")

        # 정치적 편향
        if profile.political_bias:
            warnings.append(f"Political bias: {profile.political_bias}")

        # 도메인 신뢰성
        if profile.url:
            domain_trust = self._get_domain_trust_score(profile.url)
            if domain_trust < 0.3:
                warnings.append("Low domain trustworthiness")

        # 최신성 문제
        if profile.publication_date:
            days_old = (datetime.now() - profile.publication_date).days
            if days_old > 3650:  # 10년 이상
                warnings.append("Very old publication date")

        # SSL 인증서 부재
        if profile.url and not profile.ssl_certificate:
            warnings.append("No SSL certificate")

        # 연락처 정보 부재
        if not profile.contact_information:
            warnings.append("No contact information provided")

        # 맥락 기반 경고
        if context:
            content = context.get('content', '').lower()
            for category, patterns in self.red_flag_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in content:
                        warnings.append(f"Red flag pattern detected: {pattern}")
                        break

        return warnings

    def _calculate_confidence_level(
        self,
        profile: SourceProfile,
        metrics: CredibilityMetrics,
        warning_count: int
    ) -> float:
        """신뢰도 계산"""
        base_confidence = metrics.overall_score()

        # 검증 횟수에 따른 신뢰도 증가
        validation_bonus = min(profile.validation_count * 0.02, 0.1)

        # 경고 플래그에 따른 신뢰도 감소
        warning_penalty = min(warning_count * 0.05, 0.3)

        # 동료 검토 보너스
        peer_review_bonus = 0.1 if profile.peer_reviewed else 0.0

        confidence = base_confidence + validation_bonus + peer_review_bonus - warning_penalty

        return min(max(confidence, 0.0), 1.0)

    def _determine_recommendation(self, credibility: float, warnings: List[str]) -> str:
        """추천 수준 결정"""
        if credibility >= 0.8 and len(warnings) <= 1:
            return "HIGHLY_RECOMMENDED"
        elif credibility >= 0.6 and len(warnings) <= 3:
            return "RECOMMENDED"
        elif credibility >= 0.4 or len(warnings) <= 5:
            return "CAUTION"
        else:
            return "NOT_RECOMMENDED"

    def _assess_risk_level(self, credibility: float, warnings: List[str]) -> str:
        """위험 수준 평가"""
        critical_warnings = [w for w in warnings if any(keyword in w.lower()
                           for keyword in ['conspiracy', 'bias', 'commercial'])]

        if len(critical_warnings) >= 3 or credibility < 0.2:
            return "CRITICAL"
        elif len(critical_warnings) >= 2 or credibility < 0.4:
            return "HIGH"
        elif len(warnings) >= 3 or credibility < 0.6:
            return "MEDIUM"
        else:
            return "LOW"

    async def _analyze_source_quality(
        self,
        profile: SourceProfile,
        metrics: CredibilityMetrics,
        warnings: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """정보원 품질 분석"""
        strengths = []
        weaknesses = []
        improvements = []

        # 강점 분석
        if metrics.authority_score > 0.7:
            strengths.append("High authority and expertise")
        if metrics.accuracy_score > 0.7:
            strengths.append("Strong accuracy record")
        if metrics.transparency_score > 0.7:
            strengths.append("Good transparency practices")
        if profile.peer_reviewed:
            strengths.append("Peer-reviewed content")
        if profile.impact_factor and profile.impact_factor > 2.0:
            strengths.append("High impact factor")

        # 약점 분석
        if metrics.objectivity_score < 0.5:
            weaknesses.append("Potential objectivity concerns")
        if metrics.currency_score < 0.5:
            weaknesses.append("Outdated information")
        if len(warnings) > 3:
            weaknesses.append("Multiple warning flags")
        if not profile.author:
            weaknesses.append("No author information")
        if not profile.contact_information:
            weaknesses.append("Limited contact information")

        # 개선 제안
        if not profile.peer_reviewed and profile.source_type == SourceType.ACADEMIC_JOURNAL:
            improvements.append("Seek peer-reviewed alternatives")
        if metrics.transparency_score < 0.6:
            improvements.append("Look for sources with better transparency")
        if len(warnings) > 2:
            improvements.append("Cross-reference with other sources")
        if metrics.currency_score < 0.5:
            improvements.append("Find more recent publications")

        return strengths, weaknesses, improvements

    async def _suggest_alternative_sources(
        self,
        profile: SourceProfile,
        credibility: float
    ) -> List[str]:
        """대안 정보원 제안"""
        alternatives = []

        if credibility < 0.6:
            # 정보원 유형별 대안 제안
            if profile.source_type == SourceType.NEWS_MEDIA:
                alternatives.extend([
                    "Academic research papers",
                    "Government official statistics",
                    "Peer-reviewed journals"
                ])
            elif profile.source_type == SourceType.BLOG_PERSONAL:
                alternatives.extend([
                    "Expert testimonies",
                    "Institutional reports",
                    "Peer-reviewed research"
                ])
            elif profile.source_type == SourceType.SOCIAL_MEDIA:
                alternatives.extend([
                    "Official institutional sources",
                    "Academic publications",
                    "Government databases"
                ])

        return alternatives

    def _get_domain_trust_score(self, url: str) -> float:
        """도메인 신뢰도 점수 조회"""
        if not url:
            return 0.5

        url_lower = url.lower()

        # 직접 매칭
        for domain, score in self.trusted_domains.items():
            if domain in url_lower:
                return score

        # 일반적인 패턴 매칭
        if '.gov' in url_lower:
            return 0.8
        elif '.edu' in url_lower:
            return 0.75
        elif '.org' in url_lower:
            return 0.6
        elif any(tld in url_lower for tld in ['.com', '.net', '.info']):
            return 0.5
        else:
            return 0.4

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """날짜 문자열 파싱"""
        if not date_str:
            return None

        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            return None

    def _parse_timedelta(self, delta_str: Optional[str]) -> Optional[timedelta]:
        """시간 차이 문자열 파싱"""
        if not delta_str:
            return None

        try:
            # 간단한 일수 파싱 (예: "365 days")
            match = re.search(r'(\d+)\s*days?', delta_str)
            if match:
                return timedelta(days=int(match.group(1)))
        except:
            pass

        return None

    async def get_validation_result(self, validation_id: ID) -> Optional[ValidationResult]:
        """검증 결과 조회"""
        return self.validation_results.get(validation_id)

    async def get_source_profile(self, source_id: ID) -> Optional[SourceProfile]:
        """정보원 프로필 조회"""
        return self.source_profiles.get(source_id)

    async def list_validated_sources(
        self,
        min_credibility: float = 0.0,
        source_type: Optional[SourceType] = None
    ) -> List[Tuple[SourceProfile, ValidationResult]]:
        """검증된 정보원 목록 조회"""
        results = []

        for validation in self.validation_results.values():
            if validation.overall_credibility >= min_credibility:
                profile = self.source_profiles.get(validation.source_id)
                if profile and (not source_type or profile.source_type == source_type):
                    results.append((profile, validation))

        # 신뢰성 순으로 정렬
        results.sort(key=lambda x: x[1].overall_credibility, reverse=True)
        return results

    async def get_validation_statistics(self) -> Dict[str, Any]:
        """검증 통계 조회"""
        if not self.validation_results:
            return {"total_validations": 0}

        total = len(self.validation_results)
        avg_credibility = sum(v.overall_credibility for v in self.validation_results.values()) / total

        recommendations = {}
        risk_levels = {}

        for validation in self.validation_results.values():
            rec = validation.recommendation
            risk = validation.risk_level

            recommendations[rec] = recommendations.get(rec, 0) + 1
            risk_levels[risk] = risk_levels.get(risk, 0) + 1

        return {
            "total_validations": total,
            "average_credibility": avg_credibility,
            "recommendation_distribution": recommendations,
            "risk_level_distribution": risk_levels,
            "high_credibility_percentage": sum(
                1 for v in self.validation_results.values() if v.overall_credibility >= 0.7
            ) / total * 100
        }


# 사용 예시
if __name__ == "__main__":
    async def test_source_validator():
        validator = SourceValidator()

        # 테스트 정보원 데이터
        source_data = {
            "name": "Nature Medicine",
            "url": "https://www.nature.com/nm/",
            "publisher": "Nature Publishing Group",
            "author": "Dr. Jane Smith",
            "publication_date": "2023-06-15",
            "institution": "Harvard Medical School",
            "peer_reviewed": True,
            "impact_factor": 36.13,
            "citation_count": 245,
            "ssl_certificate": True,
            "contact_information": True,
            "funding_source": "government research grant"
        }

        # 정보원 프로필 생성
        profile_result = await validator.create_source_profile(
            source_data, SourceType.ACADEMIC_JOURNAL
        )

        if profile_result["success"]:
            profile = profile_result["data"]
            print(f"정보원 프로필 생성: {profile.source_id}")

            # 정보원 검증
            validation_result = await validator.validate_source(profile.source_id)

            if validation_result["success"]:
                validation = validation_result["data"]
                print(f"\n검증 결과:")
                print(f"전체 신뢰성: {validation.overall_credibility:.3f}")
                print(f"신뢰도: {validation.confidence_level:.3f}")
                print(f"추천 수준: {validation.recommendation}")
                print(f"위험 수준: {validation.risk_level}")
                print(f"통과한 기준: {len(validation.passed_criteria)}개")
                print(f"실패한 기준: {len(validation.failed_criteria)}개")
                print(f"경고 플래그: {len(validation.warning_flags)}개")

                if validation.strengths:
                    print(f"강점: {', '.join(validation.strengths)}")
                if validation.weaknesses:
                    print(f"약점: {', '.join(validation.weaknesses)}")
                if validation.improvement_suggestions:
                    print(f"개선 제안: {', '.join(validation.improvement_suggestions)}")

            else:
                print(f"검증 실패: {validation_result['error']}")
        else:
            print(f"프로필 생성 실패: {profile_result['error']}")

        # 통계 조회
        stats = await validator.get_validation_statistics()
        print(f"\n검증 통계: {stats}")

    # 테스트 실행
    asyncio.run(test_source_validator())