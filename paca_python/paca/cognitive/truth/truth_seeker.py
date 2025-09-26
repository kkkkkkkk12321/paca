"""
Truth Seeker Module
진실 탐구자 - Phase 2.2 구현

PACA의 핵심 인지 프로세스 중 하나로, 불확실한 정보에 대한 진실 탐구를 담당합니다.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
import json

from ...core.types import Result, create_success, create_failure
from ...core.utils.logger import PacaLogger
from ...tools.truth_seeking.truth_assessment import TruthSeekingEngine, AssessmentReport, ConfidenceLevel


class UncertaintyType(Enum):
    """불확실성 유형"""
    EPISTEMIC = auto()      # 지식 부족으로 인한 불확실성
    ALEATORY = auto()       # 본질적 임의성
    MODEL = auto()          # 모델 한계
    TEMPORAL = auto()       # 시간에 따른 변화
    CONTEXTUAL = auto()     # 상황 의존적


@dataclass
class UncertaintyDetection:
    """불확실성 감지 결과"""
    detection_id: str
    uncertainty_type: UncertaintyType
    confidence_level: float  # 0.0-1.0
    uncertain_content: str
    evidence_markers: List[str]
    verification_needed: bool
    priority_score: float  # 검증 우선순위
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TruthSeekingRequest:
    """진실 탐구 요청"""
    request_id: str
    query: str
    context: Dict[str, Any]
    uncertainty_detections: List[UncertaintyDetection]
    priority: float
    requester_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TruthSeekingResult:
    """진실 탐구 결과"""
    request_id: str
    original_query: str
    truth_assessment: AssessmentReport
    verification_actions_taken: List[str]
    knowledge_updates: List[Dict[str, Any]]
    remaining_uncertainties: List[UncertaintyDetection]
    confidence_improvement: float  # 개선된 신뢰도
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)


class TruthSeeker:
    """
    진실 탐구자 - Phase 2.2 메인 클래스

    불확실한 정보를 감지하고, 외부 검증 소스를 조회하여,
    교차 검증 및 신뢰도 평가를 통해 지식 베이스를 업데이트합니다.
    """

    def __init__(self):
        self.logger = PacaLogger("TruthSeeker")

        # 진실 탐구 엔진 초기화
        self.truth_engine = TruthSeekingEngine()

        # 불확실성 감지 패턴
        self.uncertainty_patterns = self._initialize_uncertainty_patterns()

        # 검증 소스 관리
        self.verification_sources = self._initialize_verification_sources()

        # 진실 탐구 기록
        self.seeking_history: List[TruthSeekingResult] = []
        self.knowledge_updates: Dict[str, Any] = {}

        # 설정
        self.config = {
            'uncertainty_threshold': 0.6,      # 불확실성 감지 임계값
            'verification_threshold': 0.7,     # 검증 필요 임계값
            'confidence_improvement_threshold': 0.1,  # 최소 신뢰도 개선
            'max_verification_sources': 3,     # 최대 검증 소스 수
            'timeout_seconds': 30.0           # 검증 타임아웃
        }

        self.logger.info("TruthSeeker initialized with truth seeking engine")

    def _initialize_uncertainty_patterns(self) -> Dict[UncertaintyType, List[str]]:
        """불확실성 패턴 초기화"""
        return {
            UncertaintyType.EPISTEMIC: [
                r'(?i)I\'m not sure',
                r'(?i)uncertain about',
                r'(?i)don\'t know if',
                r'(?i)unclear whether',
                r'(?i)need to verify',
                r'(?i)requires confirmation'
            ],
            UncertaintyType.ALEATORY: [
                r'(?i)random',
                r'(?i)unpredictable',
                r'(?i)varies randomly',
                r'(?i)stochastic'
            ],
            UncertaintyType.MODEL: [
                r'(?i)model limitation',
                r'(?i)approximation',
                r'(?i)simplified assumption',
                r'(?i)beyond my knowledge'
            ],
            UncertaintyType.TEMPORAL: [
                r'(?i)may change over time',
                r'(?i)current situation',
                r'(?i)as of now',
                r'(?i)recently changed'
            ],
            UncertaintyType.CONTEXTUAL: [
                r'(?i)depends on context',
                r'(?i)situation-specific',
                r'(?i)varies by case',
                r'(?i)context-dependent'
            ]
        }

    def _initialize_verification_sources(self) -> List[Dict[str, Any]]:
        """검증 소스 초기화"""
        return [
            {
                'name': 'Academic Databases',
                'type': 'scholarly',
                'reliability': 0.9,
                'coverage': ['scientific', 'academic', 'research'],
                'api_available': False
            },
            {
                'name': 'Government Data',
                'type': 'official',
                'reliability': 0.85,
                'coverage': ['statistics', 'policy', 'official'],
                'api_available': False
            },
            {
                'name': 'Fact-checking Sites',
                'type': 'fact_check',
                'reliability': 0.8,
                'coverage': ['claims', 'news', 'public_statements'],
                'api_available': False
            },
            {
                'name': 'Expert Consensus',
                'type': 'expert',
                'reliability': 0.75,
                'coverage': ['professional', 'technical', 'specialized'],
                'api_available': False
            }
        ]

    async def detect_uncertainty(self, content: str, context: Dict[str, Any]) -> Result[List[UncertaintyDetection]]:
        """
        불확실한 정보 감지

        Args:
            content: 분석할 내용
            context: 추가 컨텍스트

        Returns:
            Result[List[UncertaintyDetection]]: 감지된 불확실성 목록
        """
        try:
            detections = []

            for uncertainty_type, patterns in self.uncertainty_patterns.items():
                type_detections = await self._detect_uncertainty_type(
                    content, uncertainty_type, patterns
                )
                detections.extend(type_detections)

            # 우선순위 계산
            for detection in detections:
                detection.priority_score = await self._calculate_priority(detection, context)

            # 우선순위 순으로 정렬
            detections.sort(key=lambda x: x.priority_score, reverse=True)

            self.logger.info(f"Detected {len(detections)} uncertainty markers")
            return create_success(detections)

        except Exception as e:
            self.logger.error(f"Error detecting uncertainty: {str(e)}")
            return create_failure(f"Uncertainty detection failed: {str(e)}")

    async def _detect_uncertainty_type(
        self,
        content: str,
        uncertainty_type: UncertaintyType,
        patterns: List[str]
    ) -> List[UncertaintyDetection]:
        """특정 유형의 불확실성 감지"""
        import re

        detections = []

        for pattern in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                detection = UncertaintyDetection(
                    detection_id=f"uncertainty_{datetime.now().isoformat()}_{len(detections)}",
                    uncertainty_type=uncertainty_type,
                    confidence_level=0.7,  # 기본 신뢰도
                    uncertain_content=match.group(),
                    evidence_markers=[match.group()],
                    verification_needed=True,
                    priority_score=0.0  # 나중에 계산
                )
                detections.append(detection)

        return detections

    async def _calculate_priority(
        self,
        detection: UncertaintyDetection,
        context: Dict[str, Any]
    ) -> float:
        """검증 우선순위 계산"""
        priority = detection.confidence_level

        # 불확실성 유형별 가중치
        type_weights = {
            UncertaintyType.EPISTEMIC: 0.8,    # 지식 부족 - 높은 우선순위
            UncertaintyType.MODEL: 0.6,        # 모델 한계 - 중간 우선순위
            UncertaintyType.TEMPORAL: 0.7,     # 시간적 변화 - 중상 우선순위
            UncertaintyType.CONTEXTUAL: 0.5,   # 상황 의존 - 중간 우선순위
            UncertaintyType.ALEATORY: 0.3      # 본질적 임의성 - 낮은 우선순위
        }

        priority *= type_weights.get(detection.uncertainty_type, 0.5)

        # 컨텍스트 기반 조정
        if context.get('importance', 'normal') == 'high':
            priority *= 1.3
        elif context.get('importance', 'normal') == 'critical':
            priority *= 1.5

        return min(priority, 1.0)

    async def seek_truth(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Result[TruthSeekingResult]:
        """
        진실 탐구 메인 메서드

        Args:
            query: 검증할 쿼리
            context: 추가 컨텍스트

        Returns:
            Result[TruthSeekingResult]: 진실 탐구 결과
        """
        start_time = datetime.now()
        request_id = f"truth_seek_{start_time.isoformat()}"

        try:
            self.logger.info(f"Starting truth seeking for query: {query[:100]}...")

            if context is None:
                context = {}

            # 1. 불확실성 감지
            uncertainty_result = await self.detect_uncertainty(query, context)
            if not uncertainty_result.is_success:
                return create_failure(f"Uncertainty detection failed: {uncertainty_result.error}")

            uncertainties = uncertainty_result.data

            # 2. 진실 탐구 요청 생성
            truth_request = TruthSeekingRequest(
                request_id=request_id,
                query=query,
                context=context,
                uncertainty_detections=uncertainties,
                priority=max([u.priority_score for u in uncertainties], default=0.5)
            )

            # 3. 외부 검증 수행
            verification_actions = []
            truth_assessment = None

            if uncertainties and any(u.verification_needed for u in uncertainties):
                # 진실 평가 엔진을 통한 검증
                assessment_result = await self.truth_engine.seek_truth(query, context)

                if assessment_result["success"]:
                    truth_assessment = assessment_result["data"]
                    verification_actions.append("Comprehensive truth assessment completed")
                else:
                    self.logger.warning(f"Truth assessment failed: {assessment_result['error']}")
                    verification_actions.append("Truth assessment failed - using fallback verification")

                    # 폴백 검증
                    truth_assessment = await self._fallback_verification(query, context)
            else:
                # 불확실성이 감지되지 않은 경우 기본 평가
                assessment_result = await self.truth_engine.seek_truth(query, context)
                if assessment_result["success"]:
                    truth_assessment = assessment_result["data"]
                    verification_actions.append("Basic truth assessment completed")

            # 4. 지식 베이스 업데이트
            knowledge_updates = []
            if truth_assessment:
                updates = await self._update_knowledge_base(truth_assessment, context)
                knowledge_updates.extend(updates)

            # 5. 남은 불확실성 분석
            remaining_uncertainties = await self._analyze_remaining_uncertainties(
                uncertainties, truth_assessment
            )

            # 6. 신뢰도 개선 계산
            confidence_improvement = await self._calculate_confidence_improvement(
                uncertainties, truth_assessment
            )

            # 7. 결과 생성
            processing_time = (datetime.now() - start_time).total_seconds()

            result = TruthSeekingResult(
                request_id=request_id,
                original_query=query,
                truth_assessment=truth_assessment,
                verification_actions_taken=verification_actions,
                knowledge_updates=knowledge_updates,
                remaining_uncertainties=remaining_uncertainties,
                confidence_improvement=confidence_improvement,
                processing_time=processing_time
            )

            # 히스토리에 추가
            self.seeking_history.append(result)

            self.logger.info(f"Truth seeking completed in {processing_time:.2f}s")
            self.logger.info(f"Confidence improvement: {confidence_improvement:.2f}")

            return create_success(result)

        except Exception as e:
            self.logger.error(f"Truth seeking failed: {str(e)}")
            return create_failure(f"Truth seeking failed: {str(e)}")

    async def _fallback_verification(self, query: str, context: Dict[str, Any]) -> AssessmentReport:
        """폴백 검증 (진실 엔진 실패 시)"""
        from ...tools.truth_seeking.truth_assessment import TruthScore, ConfidenceLevel, UncertaintyMetrics

        # 기본 평가 생성
        truth_score = TruthScore(
            overall_score=0.5,
            confidence_level=ConfidenceLevel.MODERATE,
            evidence_score=0.5,
            source_score=0.5,
            consistency_score=0.5,
            verifiability_score=0.5,
            bias_risk=0.3,
            misinformation_risk=0.3,
            uncertainty_level=0.5
        )

        uncertainty_metrics = UncertaintyMetrics(
            epistemic_uncertainty=0.5,
            aleatory_uncertainty=0.3,
            model_uncertainty=0.4,
            data_gaps=["Limited verification sources available"],
            conflicting_evidence=[],
            methodological_limitations=["Fallback verification used"]
        )

        fallback_assessment = AssessmentReport(
            assessment_id=f"fallback_{datetime.now().isoformat()}",
            query=query,
            truth_score=truth_score,
            uncertainty_metrics=uncertainty_metrics,
            fact_check_results=[],
            evidence_assessments=[],
            source_validations=[],
            key_findings=["Fallback verification performed due to primary system failure"],
            evidence_summary="Limited verification performed",
            consensus_level=0.5,
            recommendations=["Manual verification recommended", "Use multiple sources"],
            further_research_needed=["Comprehensive fact-checking needed"],
            quality_improvements=["Implement additional verification sources"],
            assessment_timestamp=datetime.now()
        )

        return fallback_assessment

    async def _update_knowledge_base(
        self,
        assessment: AssessmentReport,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """지식 베이스 업데이트"""
        updates = []

        try:
            # 높은 신뢰도의 정보만 지식 베이스에 추가
            if assessment.truth_score.overall_score >= self.config['verification_threshold']:
                update = {
                    'type': 'verified_fact',
                    'query': assessment.query,
                    'truth_score': assessment.truth_score.overall_score,
                    'confidence_level': assessment.truth_score.confidence_level.name,
                    'key_findings': assessment.key_findings,
                    'timestamp': assessment.assessment_timestamp.isoformat(),
                    'sources': len(assessment.source_validations)
                }

                self.knowledge_updates[assessment.assessment_id] = update
                updates.append(update)

                self.logger.info(f"Knowledge base updated with verified information: {assessment.query[:50]}...")

            # 불확실한 정보는 별도 관리
            elif assessment.truth_score.uncertainty_level > 0.5:
                update = {
                    'type': 'uncertain_information',
                    'query': assessment.query,
                    'uncertainty_level': assessment.truth_score.uncertainty_level,
                    'further_research_needed': assessment.further_research_needed,
                    'timestamp': assessment.assessment_timestamp.isoformat()
                }
                updates.append(update)

        except Exception as e:
            self.logger.error(f"Error updating knowledge base: {str(e)}")

        return updates

    async def _analyze_remaining_uncertainties(
        self,
        original_uncertainties: List[UncertaintyDetection],
        assessment: Optional[AssessmentReport]
    ) -> List[UncertaintyDetection]:
        """남은 불확실성 분석"""
        if not assessment:
            return original_uncertainties

        remaining = []

        for uncertainty in original_uncertainties:
            # 검증을 통해 해결되지 않은 불확실성 식별
            if assessment.truth_score.overall_score < self.config['verification_threshold']:
                # 여전히 불확실함
                uncertainty.confidence_level = min(
                    uncertainty.confidence_level * 1.1,  # 약간 증가
                    0.9
                )
                remaining.append(uncertainty)
            elif assessment.truth_score.uncertainty_level > 0.3:
                # 부분적으로 해결됨
                uncertainty.confidence_level *= 0.7  # 감소
                if uncertainty.confidence_level > 0.3:
                    remaining.append(uncertainty)

        return remaining

    async def _calculate_confidence_improvement(
        self,
        uncertainties: List[UncertaintyDetection],
        assessment: Optional[AssessmentReport]
    ) -> float:
        """신뢰도 개선 계산"""
        if not assessment or not uncertainties:
            return 0.0

        # 초기 불확실성 평균
        initial_uncertainty = sum(u.confidence_level for u in uncertainties) / len(uncertainties)

        # 검증 후 신뢰도
        final_confidence = assessment.truth_score.overall_score

        # 개선 정도 계산
        improvement = final_confidence - (1.0 - initial_uncertainty)

        return max(improvement, 0.0)

    async def verify_claim(
        self,
        claim: str,
        required_confidence: ConfidenceLevel = ConfidenceLevel.MODERATE
    ) -> Result[bool]:
        """
        주장 검증 (단순 참/거짓 판단)

        Args:
            claim: 검증할 주장
            required_confidence: 요구되는 신뢰도 수준

        Returns:
            Result[bool]: 검증 결과
        """
        try:
            # 진실 탐구 수행
            seeking_result = await self.seek_truth(claim)

            if not seeking_result.is_success:
                return create_failure(f"Truth seeking failed: {seeking_result.error}")

            result = seeking_result.data

            if not result.truth_assessment:
                return create_failure("No truth assessment available")

            # 요구 신뢰도와 비교
            assessment_confidence = result.truth_assessment.truth_score.confidence_level
            confidence_levels = [level for level in ConfidenceLevel]

            # 신뢰도 수준 비교
            required_index = confidence_levels.index(required_confidence)
            assessment_index = confidence_levels.index(assessment_confidence)

            is_verified = assessment_index >= required_index

            return create_success(is_verified)

        except Exception as e:
            self.logger.error(f"Claim verification failed: {str(e)}")
            return create_failure(f"Claim verification failed: {str(e)}")

    def get_seeking_history(self, limit: int = 10) -> List[TruthSeekingResult]:
        """진실 탐구 이력 조회"""
        return sorted(
            self.seeking_history,
            key=lambda x: x.timestamp,
            reverse=True
        )[:limit]

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """지식 베이스 통계"""
        if not self.knowledge_updates:
            return {'total_entries': 0}

        verified_count = sum(1 for update in self.knowledge_updates.values()
                           if update['type'] == 'verified_fact')
        uncertain_count = sum(1 for update in self.knowledge_updates.values()
                            if update['type'] == 'uncertain_information')

        return {
            'total_entries': len(self.knowledge_updates),
            'verified_facts': verified_count,
            'uncertain_information': uncertain_count,
            'average_confidence': sum(
                update.get('truth_score', 0.5)
                for update in self.knowledge_updates.values()
            ) / len(self.knowledge_updates) if self.knowledge_updates else 0.0
        }

    async def cleanup(self):
        """리소스 정리"""
        self.logger.info("TruthSeeker cleanup completed")