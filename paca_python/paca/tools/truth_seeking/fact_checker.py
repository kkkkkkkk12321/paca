"""
Fact Checker Module
사실 확인 시스템

이 모듈은 주장과 사실의 정확성을 체계적으로 검증합니다.
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


class FactStatus(Enum):
    """사실 확인 상태"""
    TRUE = auto()              # 참
    FALSE = auto()             # 거짓
    PARTIALLY_TRUE = auto()    # 부분적으로 참
    MISLEADING = auto()        # 오해의 소지
    UNVERIFIABLE = auto()      # 검증 불가
    INSUFFICIENT_EVIDENCE = auto() # 증거 불충분
    DISPUTED = auto()          # 논쟁 중
    OUTDATED = auto()         # 더 이상 유효하지 않음


class VerificationMethod(Enum):
    """검증 방법"""
    PRIMARY_SOURCE = auto()     # 1차 자료 확인
    EXPERT_CONSENSUS = auto()   # 전문가 합의
    CROSS_REFERENCE = auto()    # 교차 검증
    STATISTICAL_ANALYSIS = auto() # 통계적 분석
    LOGICAL_REASONING = auto()  # 논리적 추론
    EMPIRICAL_EVIDENCE = auto() # 경험적 증거
    HISTORICAL_RECORD = auto()  # 역사적 기록
    SCIENTIFIC_METHOD = auto()  # 과학적 방법


@dataclass
class ClaimAnalysis:
    """주장 분석"""
    claim_id: ID
    original_claim: str
    parsed_claim: str

    # 주장 분류
    claim_type: str             # factual, opinion, prediction, etc.
    verifiability: float        # 검증가능성 (0.0-1.0)
    specificity: float          # 구체성 (0.0-1.0)
    complexity: float           # 복잡성 (0.0-1.0)

    # 핵심 요소 추출
    key_entities: List[str]     # 핵심 개체
    key_claims: List[str]       # 세부 주장들
    temporal_context: Optional[str] # 시간적 맥락
    geographical_context: Optional[str] # 지리적 맥락

    # 분석 메타데이터
    analysis_timestamp: datetime
    analysis_confidence: float = 0.0


@dataclass
class CrossReference:
    """교차 참조"""
    reference_id: ID
    source_id: ID
    claim_support: float        # 주장 지지도 (-1.0 to 1.0)
    relevance_score: float      # 관련성 점수 (0.0-1.0)
    confidence_level: float     # 신뢰도 (0.0-1.0)

    # 참조 내용
    supporting_text: str
    reference_date: datetime
    verification_method: VerificationMethod

    # 선택적 필드
    contradiction_text: Optional[str] = None
    context_information: Optional[str] = None


@dataclass
class FactCheckResult:
    """사실 확인 결과"""
    check_id: ID
    claim_analysis: ClaimAnalysis

    # 검증 결과
    fact_status: FactStatus
    confidence_score: float     # 신뢰도 (0.0-1.0)
    evidence_strength: float    # 증거 강도 (0.0-1.0)

    # 세부 분석
    supporting_evidence: List[CrossReference]
    contradicting_evidence: List[CrossReference]
    neutral_evidence: List[CrossReference]

    # 검증 과정
    verification_methods: List[VerificationMethod]
    sources_checked: List[ID]
    expert_opinions: List[Dict[str, Any]]

    # 결론 및 설명
    conclusion_summary: str
    detailed_explanation: str
    check_timestamp: datetime

    # 선택적 필드
    caveats: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    checker_id: Optional[ID] = None
    last_updated: Optional[datetime] = None

    def get_evidence_balance(self) -> Dict[str, float]:
        """증거 균형 분석"""
        total_supporting = len(self.supporting_evidence)
        total_contradicting = len(self.contradicting_evidence)
        total_neutral = len(self.neutral_evidence)
        total_evidence = total_supporting + total_contradicting + total_neutral

        if total_evidence == 0:
            return {"supporting": 0.0, "contradicting": 0.0, "neutral": 0.0}

        return {
            "supporting": total_supporting / total_evidence,
            "contradicting": total_contradicting / total_evidence,
            "neutral": total_neutral / total_evidence
        }

    def get_weighted_evidence_score(self) -> float:
        """가중 증거 점수 계산"""
        supporting_weight = sum(ref.confidence_level * ref.claim_support
                              for ref in self.supporting_evidence if ref.claim_support > 0)
        contradicting_weight = sum(ref.confidence_level * abs(ref.claim_support)
                                 for ref in self.contradicting_evidence if ref.claim_support < 0)

        total_weight = supporting_weight + contradicting_weight
        if total_weight == 0:
            return 0.5  # 중립

        return supporting_weight / total_weight


class FactChecker:
    """사실 확인 시스템"""

    def __init__(self):
        self.fact_checks: Dict[ID, FactCheckResult] = {}
        self.claim_analyses: Dict[ID, ClaimAnalysis] = {}

        # 늦은 import를 사용하여 순환 import 방지
        self.evidence_evaluator = None
        self.source_validator = None
        self._initialize_components()

        # 검증 규칙 및 패턴
        self.claim_patterns = self._initialize_claim_patterns()
        self.verification_strategies = self._initialize_verification_strategies()
        self.credibility_thresholds = self._initialize_credibility_thresholds()
        self.factual_indicators = self._initialize_factual_indicators()

    def _initialize_components(self):
        """구성 요소 초기화 (늦은 import)"""
        if self.evidence_evaluator is None:
            from .evidence_evaluator import EvidenceEvaluator
            self.evidence_evaluator = EvidenceEvaluator()

        if self.source_validator is None:
            from .source_validator import SourceValidator
            self.source_validator = SourceValidator()

    def _initialize_claim_patterns(self) -> Dict[str, Dict[str, Any]]:
        """주장 패턴 초기화"""
        return {
            'numerical_claim': {
                'pattern': r'\b\d+(\.\d+)?(%|percent|million|billion|trillion)?\b',
                'verifiability': 0.8,
                'verification_methods': [VerificationMethod.STATISTICAL_ANALYSIS, VerificationMethod.PRIMARY_SOURCE]
            },
            'causal_claim': {
                'pattern': r'\b(causes?|leads? to|results? in|due to|because of)\b',
                'verifiability': 0.6,
                'verification_methods': [VerificationMethod.SCIENTIFIC_METHOD, VerificationMethod.EXPERT_CONSENSUS]
            },
            'temporal_claim': {
                'pattern': r'\b(\d{4}|yesterday|today|tomorrow|last|next|ago|since)\b',
                'verifiability': 0.7,
                'verification_methods': [VerificationMethod.HISTORICAL_RECORD, VerificationMethod.PRIMARY_SOURCE]
            },
            'comparative_claim': {
                'pattern': r'\b(more|less|better|worse|higher|lower|than|compared to)\b',
                'verifiability': 0.7,
                'verification_methods': [VerificationMethod.STATISTICAL_ANALYSIS, VerificationMethod.CROSS_REFERENCE]
            },
            'absolute_claim': {
                'pattern': r'\b(all|none|never|always|every|no one|everyone)\b',
                'verifiability': 0.9,
                'verification_methods': [VerificationMethod.LOGICAL_REASONING, VerificationMethod.CROSS_REFERENCE]
            }
        }

    def _initialize_verification_strategies(self) -> Dict[VerificationMethod, Dict[str, Any]]:
        """검증 전략 초기화"""
        return {
            VerificationMethod.PRIMARY_SOURCE: {
                'priority': 1,
                'confidence_weight': 0.9,
                'required_sources': 1,
                'source_types': ['government_official', 'academic_journal']
            },
            VerificationMethod.EXPERT_CONSENSUS: {
                'priority': 2,
                'confidence_weight': 0.8,
                'required_sources': 3,
                'source_types': ['expert_testimony', 'research_institute']
            },
            VerificationMethod.CROSS_REFERENCE: {
                'priority': 3,
                'confidence_weight': 0.7,
                'required_sources': 2,
                'source_types': ['news_media', 'academic_journal']
            },
            VerificationMethod.STATISTICAL_ANALYSIS: {
                'priority': 2,
                'confidence_weight': 0.85,
                'required_sources': 2,
                'source_types': ['statistical_agency', 'research_institute']
            }
        }

    def _initialize_credibility_thresholds(self) -> Dict[str, float]:
        """신뢰성 임계값 초기화"""
        return {
            'high_confidence': 0.8,
            'medium_confidence': 0.6,
            'low_confidence': 0.4,
            'insufficient_evidence': 0.3
        }

    def _initialize_factual_indicators(self) -> Dict[str, List[str]]:
        """사실성 지표 초기화"""
        return {
            'high_certainty': [
                'scientific study shows', 'data indicates', 'research confirms',
                'statistics show', 'according to official records'
            ],
            'medium_certainty': [
                'reports suggest', 'experts believe', 'studies indicate',
                'evidence suggests', 'analysis shows'
            ],
            'low_certainty': [
                'some say', 'it is believed', 'allegedly', 'reportedly',
                'claims have been made', 'rumors suggest'
            ],
            'opinion_markers': [
                'in my opinion', 'i think', 'i believe', 'personally',
                'seems to me', 'my view is'
            ]
        }

    async def analyze_claim(self, claim_text: str) -> Result[ClaimAnalysis]:
        """주장 분석"""
        try:
            # 구성 요소 확인
            self._initialize_components()

            claim_id = generate_id()

            # 텍스트 전처리
            parsed_claim = self._preprocess_claim(claim_text)

            # 주장 분류
            claim_type = self._classify_claim(parsed_claim)

            # 검증가능성 평가
            verifiability = self._assess_verifiability(parsed_claim)

            # 구체성 평가
            specificity = self._assess_specificity(parsed_claim)

            # 복잡성 평가
            complexity = self._assess_complexity(parsed_claim)

            # 핵심 요소 추출
            key_entities = self._extract_key_entities(parsed_claim)
            key_claims = self._extract_sub_claims(parsed_claim)
            temporal_context = self._extract_temporal_context(parsed_claim)
            geographical_context = self._extract_geographical_context(parsed_claim)

            # 분석 신뢰도 계산
            analysis_confidence = self._calculate_analysis_confidence(
                verifiability, specificity, complexity
            )

            analysis = ClaimAnalysis(
                claim_id=claim_id,
                original_claim=claim_text,
                parsed_claim=parsed_claim,
                claim_type=claim_type,
                verifiability=verifiability,
                specificity=specificity,
                complexity=complexity,
                key_entities=key_entities,
                key_claims=key_claims,
                temporal_context=temporal_context,
                geographical_context=geographical_context,
                analysis_timestamp=current_timestamp(),
                analysis_confidence=analysis_confidence
            )

            self.claim_analyses[claim_id] = analysis

            return {"success": True, "data": analysis}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def check_fact(
        self,
        claim_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Result[FactCheckResult]:
        """사실 확인 수행"""
        try:
            # 구성 요소 확인
            self._initialize_components()

            # 주장 분석
            analysis_result = await self.analyze_claim(claim_text)
            if not analysis_result["success"]:
                return analysis_result

            claim_analysis = analysis_result["data"]
            check_id = generate_id()

            # 검증 방법 결정
            verification_methods = self._determine_verification_methods(claim_analysis)

            # 증거 수집
            evidence_results = await self._collect_evidence(
                claim_analysis, verification_methods, context
            )

            # 교차 참조 생성
            cross_references = await self._create_cross_references(
                claim_analysis, evidence_results
            )

            # 증거 분류
            supporting_evidence = [ref for ref in cross_references if ref.claim_support > 0.1]
            contradicting_evidence = [ref for ref in cross_references if ref.claim_support < -0.1]
            neutral_evidence = [ref for ref in cross_references if -0.1 <= ref.claim_support <= 0.1]

            # 사실 상태 결정
            fact_status = self._determine_fact_status(
                supporting_evidence, contradicting_evidence, neutral_evidence
            )

            # 신뢰도 계산
            confidence_score = self._calculate_confidence_score(
                supporting_evidence, contradicting_evidence, claim_analysis
            )

            # 증거 강도 계산
            evidence_strength = self._calculate_evidence_strength(cross_references)

            # 전문가 의견 수집
            expert_opinions = await self._collect_expert_opinions(claim_analysis)

            # 결론 생성
            conclusion_summary, detailed_explanation = self._generate_conclusion(
                fact_status, supporting_evidence, contradicting_evidence, claim_analysis
            )

            # 주의사항 및 한계 식별
            caveats = self._identify_caveats(claim_analysis, cross_references)
            limitations = self._identify_limitations(verification_methods, evidence_results)

            fact_check_result = FactCheckResult(
                check_id=check_id,
                claim_analysis=claim_analysis,
                fact_status=fact_status,
                confidence_score=confidence_score,
                evidence_strength=evidence_strength,
                supporting_evidence=supporting_evidence,
                contradicting_evidence=contradicting_evidence,
                neutral_evidence=neutral_evidence,
                verification_methods=verification_methods,
                sources_checked=[ref.source_id for ref in cross_references],
                expert_opinions=expert_opinions,
                conclusion_summary=conclusion_summary,
                detailed_explanation=detailed_explanation,
                caveats=caveats,
                limitations=limitations,
                check_timestamp=current_timestamp()
            )

            self.fact_checks[check_id] = fact_check_result

            return {"success": True, "data": fact_check_result}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _preprocess_claim(self, claim_text: str) -> str:
        """주장 전처리"""
        # 기본 정리
        cleaned = claim_text.strip()

        # 여러 공백을 하나로
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # 특수 문자 정리 (필요시)
        cleaned = re.sub(r'[^\w\s\.,!?;:-]', '', cleaned)

        return cleaned

    def _classify_claim(self, claim_text: str) -> str:
        """주장 분류"""
        claim_lower = claim_text.lower()

        # 의견 지표 확인
        for indicator in self.factual_indicators['opinion_markers']:
            if indicator in claim_lower:
                return "opinion"

        # 예측/미래 관련
        if any(word in claim_lower for word in ['will', 'predict', 'forecast', 'expect']):
            return "prediction"

        # 수치적 주장
        if re.search(self.claim_patterns['numerical_claim']['pattern'], claim_text):
            return "numerical"

        # 인과관계 주장
        if re.search(self.claim_patterns['causal_claim']['pattern'], claim_lower):
            return "causal"

        # 비교 주장
        if re.search(self.claim_patterns['comparative_claim']['pattern'], claim_lower):
            return "comparative"

        # 기본적으로 사실적 주장
        return "factual"

    def _assess_verifiability(self, claim_text: str) -> float:
        """검증가능성 평가"""
        verifiability = 0.5  # 기본값

        # 패턴 기반 점수 조정
        for pattern_name, pattern_info in self.claim_patterns.items():
            if re.search(pattern_info['pattern'], claim_text, re.IGNORECASE):
                verifiability = max(verifiability, pattern_info['verifiability'])

        # 구체적 정보 존재 시 가점
        if re.search(r'\b\d{4}\b', claim_text):  # 연도
            verifiability += 0.1
        if re.search(r'\b\d+(\.\d+)?%\b', claim_text):  # 백분율
            verifiability += 0.1
        if re.search(r'\b(according to|study|research|report)\b', claim_text.lower()):
            verifiability += 0.1

        # 모호한 표현 시 감점
        if any(word in claim_text.lower() for word in ['some', 'many', 'often', 'usually']):
            verifiability -= 0.1

        return min(max(verifiability, 0.0), 1.0)

    def _assess_specificity(self, claim_text: str) -> float:
        """구체성 평가"""
        specificity = 0.3  # 기본값

        # 숫자 포함
        if re.search(r'\d+', claim_text):
            specificity += 0.2

        # 날짜/시간 정보
        if re.search(r'\b(\d{4}|\d{1,2}/\d{1,2}|\w+day)\b', claim_text):
            specificity += 0.2

        # 고유명사 (대문자로 시작)
        proper_nouns = len(re.findall(r'\b[A-Z][a-z]+\b', claim_text))
        specificity += min(proper_nouns * 0.1, 0.3)

        # 정확한 출처 언급
        if re.search(r'\b(study|research|report|according to)\b', claim_text.lower()):
            specificity += 0.2

        return min(specificity, 1.0)

    def _assess_complexity(self, claim_text: str) -> float:
        """복잡성 평가"""
        # 단어 수
        word_count = len(claim_text.split())
        complexity = min(word_count / 50, 0.4)  # 최대 0.4

        # 절의 수 (콤마, 접속사 기준)
        clause_count = len(re.findall(r'[,;]|\b(and|but|or|because|since|although)\b', claim_text.lower()))
        complexity += min(clause_count * 0.1, 0.3)

        # 인과관계/조건문
        if re.search(r'\b(if|when|because|since|due to|leads to)\b', claim_text.lower()):
            complexity += 0.2

        # 수식어 많음
        adjective_count = len(re.findall(r'\b\w+ly\b|\b(very|quite|extremely|highly)\b', claim_text.lower()))
        complexity += min(adjective_count * 0.05, 0.1)

        return min(complexity, 1.0)

    def _extract_key_entities(self, claim_text: str) -> List[str]:
        """핵심 개체 추출"""
        entities = []

        # 고유명사 (간단한 휴리스틱)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', claim_text)
        entities.extend(proper_nouns)

        # 숫자와 단위
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:\s*%|percent|million|billion|trillion)?\b', claim_text)
        entities.extend(numbers)

        # 날짜
        dates = re.findall(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b', claim_text)
        entities.extend(dates)

        return list(set(entities))  # 중복 제거

    def _extract_sub_claims(self, claim_text: str) -> List[str]:
        """세부 주장 추출"""
        # 간단한 문장 분리
        sentences = re.split(r'[.!?;]', claim_text)

        # 접속사로 분리된 절
        clauses = []
        for sentence in sentences:
            parts = re.split(r'\b(and|but|or|because|since|although|however)\b', sentence.strip())
            clauses.extend([part.strip() for part in parts if part.strip() and
                          part.lower() not in ['and', 'but', 'or', 'because', 'since', 'although', 'however']])

        return [clause for clause in clauses if len(clause) > 10]  # 너무 짧은 것 제외

    def _extract_temporal_context(self, claim_text: str) -> Optional[str]:
        """시간적 맥락 추출"""
        temporal_patterns = [
            r'\b\d{4}\b',  # 연도
            r'\b(yesterday|today|tomorrow|last week|next month)\b',
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # 날짜
            r'\b(since|until|during|before|after)\s+\d{4}\b'
        ]

        for pattern in temporal_patterns:
            match = re.search(pattern, claim_text, re.IGNORECASE)
            if match:
                return match.group()

        return None

    def _extract_geographical_context(self, claim_text: str) -> Optional[str]:
        """지리적 맥락 추출"""
        # 간단한 지명 패턴 (실제로는 더 정교한 NER 필요)
        geo_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:City|State|Country|Province|County))\b',
            r'\b(?:in|at|from)\s+[A-Z][a-z]+\b'
        ]

        for pattern in geo_patterns:
            match = re.search(pattern, claim_text)
            if match:
                return match.group()

        return None

    def _calculate_analysis_confidence(
        self,
        verifiability: float,
        specificity: float,
        complexity: float
    ) -> float:
        """분석 신뢰도 계산"""
        # 검증가능성과 구체성이 높고, 복잡성이 적당할 때 높은 신뢰도
        confidence = (verifiability * 0.4 + specificity * 0.4 +
                     (1.0 - min(complexity, 0.8)) * 0.2)

        return min(max(confidence, 0.0), 1.0)

    def _determine_verification_methods(self, claim_analysis: ClaimAnalysis) -> List[VerificationMethod]:
        """검증 방법 결정"""
        methods = []

        # 주장 유형별 검증 방법
        if claim_analysis.claim_type == "numerical":
            methods.extend([VerificationMethod.STATISTICAL_ANALYSIS, VerificationMethod.PRIMARY_SOURCE])
        elif claim_analysis.claim_type == "causal":
            methods.extend([VerificationMethod.SCIENTIFIC_METHOD, VerificationMethod.EXPERT_CONSENSUS])
        elif claim_analysis.claim_type == "factual":
            methods.extend([VerificationMethod.PRIMARY_SOURCE, VerificationMethod.CROSS_REFERENCE])

        # 항상 교차 검증 포함
        if VerificationMethod.CROSS_REFERENCE not in methods:
            methods.append(VerificationMethod.CROSS_REFERENCE)

        # 검증가능성이 높으면 1차 자료 확인
        if claim_analysis.verifiability > 0.7 and VerificationMethod.PRIMARY_SOURCE not in methods:
            methods.insert(0, VerificationMethod.PRIMARY_SOURCE)

        return methods

    async def _collect_evidence(
        self,
        claim_analysis: ClaimAnalysis,
        methods: List[VerificationMethod],
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """증거 수집 (시뮬레이션)"""
        # 실제 구현에서는 외부 API, 데이터베이스 등을 활용
        evidence_results = []

        for method in methods:
            # 방법별 증거 수집 시뮬레이션
            if method == VerificationMethod.PRIMARY_SOURCE:
                evidence_results.extend(await self._search_primary_sources(claim_analysis))
            elif method == VerificationMethod.EXPERT_CONSENSUS:
                evidence_results.extend(await self._search_expert_opinions(claim_analysis))
            elif method == VerificationMethod.CROSS_REFERENCE:
                evidence_results.extend(await self._search_cross_references(claim_analysis))
            elif method == VerificationMethod.STATISTICAL_ANALYSIS:
                evidence_results.extend(await self._search_statistical_data(claim_analysis))

        return evidence_results

    async def _search_primary_sources(self, claim_analysis: ClaimAnalysis) -> List[Dict[str, Any]]:
        """1차 자료 검색 (시뮬레이션)"""
        # 실제로는 정부 데이터베이스, 공식 문서 등을 검색
        return [
            {
                "source_type": "government_database",
                "content": f"Official data regarding {claim_analysis.key_entities[0] if claim_analysis.key_entities else 'topic'}",
                "reliability": 0.9,
                "support_level": 0.7,
                "url": "https://example.gov/data"
            }
        ]

    async def _search_expert_opinions(self, claim_analysis: ClaimAnalysis) -> List[Dict[str, Any]]:
        """전문가 의견 검색 (시뮬레이션)"""
        return [
            {
                "source_type": "expert_testimony",
                "content": f"Expert analysis of {claim_analysis.parsed_claim[:50]}...",
                "reliability": 0.8,
                "support_level": 0.6,
                "expert_credentials": "Ph.D. in relevant field"
            }
        ]

    async def _search_cross_references(self, claim_analysis: ClaimAnalysis) -> List[Dict[str, Any]]:
        """교차 참조 검색 (시뮬레이션)"""
        return [
            {
                "source_type": "news_media",
                "content": f"Multiple sources report on {claim_analysis.parsed_claim[:30]}...",
                "reliability": 0.7,
                "support_level": 0.5,
                "source_count": 3
            }
        ]

    async def _search_statistical_data(self, claim_analysis: ClaimAnalysis) -> List[Dict[str, Any]]:
        """통계 데이터 검색 (시뮬레이션)"""
        return [
            {
                "source_type": "statistical_agency",
                "content": f"Statistical data for {claim_analysis.key_entities[0] if claim_analysis.key_entities else 'topic'}",
                "reliability": 0.85,
                "support_level": 0.8,
                "methodology": "Random sampling, n=1000"
            }
        ]

    async def _create_cross_references(
        self,
        claim_analysis: ClaimAnalysis,
        evidence_results: List[Dict[str, Any]]
    ) -> List[CrossReference]:
        """교차 참조 생성"""
        cross_references = []

        for i, evidence in enumerate(evidence_results):
            # 임시 소스 ID 생성 (실제로는 source_validator를 통해 생성)
            source_id = f"source_{i}"

            # 지지도 계산 (실제로는 더 정교한 분석 필요)
            claim_support = evidence.get('support_level', 0.0)
            if evidence.get('contradicts', False):
                claim_support = -abs(claim_support)

            cross_ref = CrossReference(
                reference_id=generate_id(),
                source_id=source_id,
                claim_support=claim_support,
                relevance_score=evidence.get('relevance', 0.8),
                confidence_level=evidence.get('reliability', 0.7),
                supporting_text=evidence.get('content', ''),
                reference_date=current_timestamp(),
                verification_method=VerificationMethod.CROSS_REFERENCE
            )

            cross_references.append(cross_ref)

        return cross_references

    def _determine_fact_status(
        self,
        supporting: List[CrossReference],
        contradicting: List[CrossReference],
        neutral: List[CrossReference]
    ) -> FactStatus:
        """사실 상태 결정"""
        total_support = sum(ref.claim_support * ref.confidence_level for ref in supporting)
        total_contradiction = sum(abs(ref.claim_support) * ref.confidence_level for ref in contradicting)
        total_evidence = len(supporting) + len(contradicting) + len(neutral)

        if total_evidence == 0:
            return FactStatus.INSUFFICIENT_EVIDENCE

        # 강한 지지 증거
        if total_support > total_contradiction * 2 and total_support > 2.0:
            return FactStatus.TRUE

        # 강한 반박 증거
        elif total_contradiction > total_support * 2 and total_contradiction > 2.0:
            return FactStatus.FALSE

        # 부분적 지지
        elif total_support > total_contradiction and total_support > 1.0:
            return FactStatus.PARTIALLY_TRUE

        # 상충하는 증거
        elif abs(total_support - total_contradiction) < 0.5:
            return FactStatus.DISPUTED

        # 증거 부족
        elif total_evidence < 3:
            return FactStatus.INSUFFICIENT_EVIDENCE

        # 오해의 소지 (약한 지지/반박)
        else:
            return FactStatus.MISLEADING

    def _calculate_confidence_score(
        self,
        supporting: List[CrossReference],
        contradicting: List[CrossReference],
        claim_analysis: ClaimAnalysis
    ) -> float:
        """신뢰도 점수 계산"""
        # 기본 신뢰도
        base_confidence = claim_analysis.analysis_confidence

        # 증거 품질
        if supporting or contradicting:
            avg_evidence_confidence = sum(ref.confidence_level for ref in supporting + contradicting) / len(supporting + contradicting)
            base_confidence = (base_confidence + avg_evidence_confidence) / 2

        # 증거 일관성
        if len(supporting) > 0 and len(contradicting) == 0:
            base_confidence += 0.2  # 일관된 지지
        elif len(contradicting) > 0 and len(supporting) == 0:
            base_confidence += 0.1  # 일관된 반박
        elif len(supporting) > 0 and len(contradicting) > 0:
            base_confidence -= 0.1  # 상충하는 증거

        return min(max(base_confidence, 0.0), 1.0)

    def _calculate_evidence_strength(self, cross_references: List[CrossReference]) -> float:
        """증거 강도 계산"""
        if not cross_references:
            return 0.0

        # 평균 신뢰도와 관련성
        avg_confidence = sum(ref.confidence_level for ref in cross_references) / len(cross_references)
        avg_relevance = sum(ref.relevance_score for ref in cross_references) / len(cross_references)

        # 증거 수량 보너스
        quantity_bonus = min(len(cross_references) * 0.1, 0.3)

        strength = (avg_confidence * 0.5 + avg_relevance * 0.3 + quantity_bonus * 0.2)

        return min(max(strength, 0.0), 1.0)

    async def _collect_expert_opinions(self, claim_analysis: ClaimAnalysis) -> List[Dict[str, Any]]:
        """전문가 의견 수집 (시뮬레이션)"""
        # 실제로는 전문가 데이터베이스나 API 활용
        return [
            {
                "expert_id": "expert_1",
                "credentials": "Ph.D. in relevant field, 15 years experience",
                "opinion": f"Analysis of claim: {claim_analysis.parsed_claim[:50]}...",
                "confidence": 0.8,
                "supporting": True
            }
        ]

    def _generate_conclusion(
        self,
        fact_status: FactStatus,
        supporting: List[CrossReference],
        contradicting: List[CrossReference],
        claim_analysis: ClaimAnalysis
    ) -> Tuple[str, str]:
        """결론 생성"""

        # 요약 결론
        status_messages = {
            FactStatus.TRUE: "This claim is supported by available evidence.",
            FactStatus.FALSE: "This claim is contradicted by available evidence.",
            FactStatus.PARTIALLY_TRUE: "This claim is partially supported by evidence.",
            FactStatus.MISLEADING: "This claim may be misleading or requires clarification.",
            FactStatus.UNVERIFIABLE: "This claim cannot be verified with available sources.",
            FactStatus.INSUFFICIENT_EVIDENCE: "There is insufficient evidence to verify this claim.",
            FactStatus.DISPUTED: "This claim is disputed with conflicting evidence.",
            FactStatus.OUTDATED: "This claim may have been true but is now outdated."
        }

        summary = status_messages.get(fact_status, "Unable to determine the veracity of this claim.")

        # 상세 설명
        explanation_parts = [
            f"Claim analysis: {claim_analysis.claim_type} claim with {claim_analysis.verifiability:.1%} verifiability.",
            f"Evidence review: {len(supporting)} supporting, {len(contradicting)} contradicting sources found."
        ]

        if supporting:
            explanation_parts.append(f"Supporting evidence includes sources with average confidence of {sum(ref.confidence_level for ref in supporting) / len(supporting):.1%}.")

        if contradicting:
            explanation_parts.append(f"Contradicting evidence includes sources with average confidence of {sum(ref.confidence_level for ref in contradicting) / len(contradicting):.1%}.")

        detailed_explanation = " ".join(explanation_parts)

        return summary, detailed_explanation

    def _identify_caveats(self, claim_analysis: ClaimAnalysis, cross_references: List[CrossReference]) -> List[str]:
        """주의사항 식별"""
        caveats = []

        if claim_analysis.complexity > 0.7:
            caveats.append("This is a complex claim that may require expert interpretation.")

        if claim_analysis.verifiability < 0.5:
            caveats.append("The verifiability of this claim is limited.")

        if len(cross_references) < 3:
            caveats.append("Limited number of sources available for verification.")

        if any(ref.confidence_level < 0.5 for ref in cross_references):
            caveats.append("Some sources have low confidence levels.")

        return caveats

    def _identify_limitations(self, methods: List[VerificationMethod], evidence_results: List[Dict[str, Any]]) -> List[str]:
        """제한사항 식별"""
        limitations = []

        if VerificationMethod.PRIMARY_SOURCE not in methods:
            limitations.append("Primary sources were not accessible for verification.")

        if len(evidence_results) < 5:
            limitations.append("Limited evidence sources available.")

        if all(result.get('source_type') == 'news_media' for result in evidence_results):
            limitations.append("Verification relied primarily on media sources.")

        limitations.append("Automated fact-checking has inherent limitations and may require human review.")

        return limitations

    async def get_fact_check_result(self, check_id: ID) -> Optional[FactCheckResult]:
        """사실 확인 결과 조회"""
        return self.fact_checks.get(check_id)

    async def get_claim_analysis(self, claim_id: ID) -> Optional[ClaimAnalysis]:
        """주장 분석 결과 조회"""
        return self.claim_analyses.get(claim_id)

    async def search_fact_checks(
        self,
        query: str,
        fact_status: Optional[FactStatus] = None,
        min_confidence: float = 0.0
    ) -> List[FactCheckResult]:
        """사실 확인 검색"""
        results = []

        for fact_check in self.fact_checks.values():
            # 쿼리 매칭 (간단한 키워드 검색)
            if query.lower() in fact_check.claim_analysis.parsed_claim.lower():
                # 상태 필터
                if fact_status and fact_check.fact_status != fact_status:
                    continue

                # 신뢰도 필터
                if fact_check.confidence_score < min_confidence:
                    continue

                results.append(fact_check)

        # 신뢰도 순으로 정렬
        results.sort(key=lambda x: x.confidence_score, reverse=True)
        return results

    async def get_fact_check_statistics(self) -> Dict[str, Any]:
        """사실 확인 통계"""
        if not self.fact_checks:
            return {"total_checks": 0}

        total = len(self.fact_checks)
        status_counts = {}
        avg_confidence = sum(fc.confidence_score for fc in self.fact_checks.values()) / total

        for status in FactStatus:
            count = sum(1 for fc in self.fact_checks.values() if fc.fact_status == status)
            status_counts[status.name] = count

        return {
            "total_checks": total,
            "average_confidence": avg_confidence,
            "status_distribution": status_counts,
            "high_confidence_percentage": sum(
                1 for fc in self.fact_checks.values() if fc.confidence_score >= 0.7
            ) / total * 100
        }


# 사용 예시
if __name__ == "__main__":
    async def test_fact_checker():
        fact_checker = FactChecker()

        # 테스트 주장
        test_claims = [
            "COVID-19 vaccines are 95% effective against severe illness.",
            "Climate change is caused by human activities.",
            "The population of Tokyo is over 13 million people.",
            "Vitamin C prevents the common cold.",
            "Electric cars produce zero emissions."
        ]

        for claim in test_claims:
            print(f"\n{'='*60}")
            print(f"주장: {claim}")
            print(f"{'='*60}")

            # 사실 확인
            result = await fact_checker.check_fact(claim)

            if result["success"]:
                fact_check = result["data"]

                print(f"사실 상태: {fact_check.fact_status.name}")
                print(f"신뢰도: {fact_check.confidence_score:.1%}")
                print(f"증거 강도: {fact_check.evidence_strength:.1%}")
                print(f"검증 방법: {[method.name for method in fact_check.verification_methods]}")

                print(f"\n결론: {fact_check.conclusion_summary}")
                print(f"상세 설명: {fact_check.detailed_explanation}")

                evidence_balance = fact_check.get_evidence_balance()
                print(f"증거 균형: 지지 {evidence_balance['supporting']:.1%}, "
                      f"반박 {evidence_balance['contradicting']:.1%}, "
                      f"중립 {evidence_balance['neutral']:.1%}")

                if fact_check.caveats:
                    print(f"주의사항: {', '.join(fact_check.caveats)}")

                if fact_check.limitations:
                    print(f"제한사항: {', '.join(fact_check.limitations)}")

            else:
                print(f"사실 확인 실패: {result['error']}")

        # 통계 출력
        stats = await fact_checker.get_fact_check_statistics()
        print(f"\n\n전체 통계:")
        print(f"총 검증: {stats['total_checks']}건")
        print(f"평균 신뢰도: {stats['average_confidence']:.1%}")
        print(f"고신뢰도 비율: {stats['high_confidence_percentage']:.1f}%")
        print(f"상태 분포: {stats['status_distribution']}")

    # 테스트 실행
    asyncio.run(test_fact_checker())