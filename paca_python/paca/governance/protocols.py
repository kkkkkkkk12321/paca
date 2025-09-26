"""
Governance Protocols Module
3대 거버넌스 프로토콜 구현

1. 모순 수용 프로토콜 (Contradiction Acceptance)
2. 최종 판단 유보 프로토콜 (Final Judgment Reservation)
3. 신뢰-검증-롤백 프로토콜 (Trust-Verify-Rollback)
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
import json
import hashlib

# 조건부 임포트
try:
    from ..core.types.base import (
        ID, Timestamp, Result, current_timestamp, generate_id, create_success, create_failure
    )
except ImportError:
    from paca.core.types.base import (
        ID, Timestamp, Result, current_timestamp, generate_id, create_success, create_failure
    )


class ProtocolStatus(Enum):
    """프로토콜 실행 상태"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLBACK = "rollback"


class ContradictionLevel(Enum):
    """모순 수준"""
    MINOR = "minor"           # 사소한 불일치
    MODERATE = "moderate"     # 중간 수준 모순
    MAJOR = "major"          # 심각한 모순
    FUNDAMENTAL = "fundamental"  # 근본적 모순


class JudgmentType(Enum):
    """판단 유형"""
    FACTUAL = "factual"           # 사실적 판단
    EVALUATIVE = "evaluative"     # 평가적 판단
    PRESCRIPTIVE = "prescriptive" # 처방적 판단
    PREDICTIVE = "predictive"     # 예측적 판단


class TrustLevel(Enum):
    """신뢰 수준"""
    NONE = "none"         # 신뢰 없음
    LOW = "low"           # 낮은 신뢰
    MEDIUM = "medium"     # 보통 신뢰
    HIGH = "high"         # 높은 신뢰
    VERIFIED = "verified" # 검증된 신뢰


@dataclass(frozen=True)
class ProtocolResult:
    """프로토콜 실행 결과"""
    protocol_id: ID
    status: ProtocolStatus
    result_data: Dict[str, Any]
    timestamp: Timestamp
    success: bool
    error_message: Optional[str] = None


@dataclass
class ContradictionCase:
    """모순 사례"""
    case_id: ID
    statement_a: str
    statement_b: str
    level: ContradictionLevel
    context: Dict[str, Any]
    resolution_strategy: Optional[str] = None
    accepted: bool = False
    timestamp: Timestamp = field(default_factory=current_timestamp)


@dataclass
class JudgmentReservation:
    """판단 유보 사례"""
    reservation_id: ID
    subject: str
    judgment_type: JudgmentType
    uncertainty_level: float  # 0.0-1.0
    evidence_strength: float  # 0.0-1.0
    reservation_reason: str
    revisit_conditions: List[str]
    timestamp: Timestamp = field(default_factory=current_timestamp)


@dataclass
class TrustRecord:
    """신뢰 기록"""
    record_id: ID
    source: str
    trust_level: TrustLevel
    verification_method: str
    verification_result: bool
    evidence: Dict[str, Any]
    rollback_conditions: List[str]
    timestamp: Timestamp = field(default_factory=current_timestamp)


class GovernanceProtocol(ABC):
    """거버넌스 프로토콜 기본 클래스"""

    def __init__(self, protocol_name: str):
        self.protocol_name = protocol_name
        self.active = True
        self.execution_history: List[ProtocolResult] = []
        self.configuration: Dict[str, Any] = {}

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Result[ProtocolResult]:
        """프로토콜 실행"""
        pass

    @abstractmethod
    async def validate_input(self, input_data: Dict[str, Any]) -> Result[bool]:
        """입력 데이터 검증"""
        pass

    def record_execution(self, result: ProtocolResult) -> None:
        """실행 결과 기록"""
        self.execution_history.append(result)

    def get_success_rate(self, hours: int = 24) -> float:
        """성공률 계산"""
        cutoff_time = current_timestamp() - (hours * 3600)
        recent_executions = [e for e in self.execution_history if e.timestamp >= cutoff_time]

        if not recent_executions:
            return 1.0

        successful = sum(1 for e in recent_executions if e.success)
        return successful / len(recent_executions)


class ContradictionAcceptance(GovernanceProtocol):
    """
    모순 수용 프로토콜
    서로 모순되는 정보나 관점을 수용하고 관리하는 시스템
    """

    def __init__(self):
        super().__init__("contradiction_acceptance")
        self.contradiction_cases: List[ContradictionCase] = []
        self.resolution_strategies = {
            ContradictionLevel.MINOR: self._handle_minor_contradiction,
            ContradictionLevel.MODERATE: self._handle_moderate_contradiction,
            ContradictionLevel.MAJOR: self._handle_major_contradiction,
            ContradictionLevel.FUNDAMENTAL: self._handle_fundamental_contradiction
        }

    async def execute(self, input_data: Dict[str, Any]) -> Result[ProtocolResult]:
        """모순 수용 프로토콜 실행"""
        try:
            protocol_id = generate_id("contradiction_")

            # 입력 검증
            validation_result = await self.validate_input(input_data)
            if not validation_result.is_success:
                return create_failure(validation_result.error)

            # 모순 감지 및 분석
            contradiction_result = await self._detect_contradiction(input_data)
            if not contradiction_result.is_success:
                return create_failure(contradiction_result.error)

            contradiction_case = contradiction_result.value

            # 모순 수준별 처리
            resolution_strategy = self.resolution_strategies[contradiction_case.level]
            strategy_result = await resolution_strategy(contradiction_case)

            if strategy_result.is_success:
                contradiction_case.accepted = True
                contradiction_case.resolution_strategy = strategy_result.value
                self.contradiction_cases.append(contradiction_case)

                result = ProtocolResult(
                    protocol_id=protocol_id,
                    status=ProtocolStatus.COMPLETED,
                    result_data={
                        'contradiction_case': {
                            'case_id': contradiction_case.case_id,
                            'level': contradiction_case.level.value,
                            'resolution_strategy': contradiction_case.resolution_strategy,
                            'accepted': contradiction_case.accepted
                        }
                    },
                    timestamp=current_timestamp(),
                    success=True
                )
            else:
                result = ProtocolResult(
                    protocol_id=protocol_id,
                    status=ProtocolStatus.FAILED,
                    result_data={'error': strategy_result.error},
                    timestamp=current_timestamp(),
                    success=False,
                    error_message=str(strategy_result.error)
                )

            self.record_execution(result)
            return create_success(result)

        except Exception as e:
            result = ProtocolResult(
                protocol_id=generate_id("contradiction_"),
                status=ProtocolStatus.FAILED,
                result_data={'error': str(e)},
                timestamp=current_timestamp(),
                success=False,
                error_message=str(e)
            )
            self.record_execution(result)
            return create_failure(e)

    async def validate_input(self, input_data: Dict[str, Any]) -> Result[bool]:
        """입력 데이터 검증"""
        try:
            required_fields = ['statement_a', 'statement_b', 'context']
            for field in required_fields:
                if field not in input_data:
                    return create_failure(f"필수 필드 누락: {field}")

            if not isinstance(input_data.get('context'), dict):
                return create_failure("context는 딕셔너리 형태여야 합니다")

            return create_success(True)

        except Exception as e:
            return create_failure(e)

    async def _detect_contradiction(self, input_data: Dict[str, Any]) -> Result[ContradictionCase]:
        """모순 감지 및 분석"""
        try:
            statement_a = input_data['statement_a']
            statement_b = input_data['statement_b']
            context = input_data['context']

            # 모순 수준 평가
            level = await self._assess_contradiction_level(statement_a, statement_b, context)

            contradiction_case = ContradictionCase(
                case_id=generate_id("case_"),
                statement_a=statement_a,
                statement_b=statement_b,
                level=level,
                context=context
            )

            return create_success(contradiction_case)

        except Exception as e:
            return create_failure(e)

    async def _assess_contradiction_level(self, statement_a: str, statement_b: str,
                                        context: Dict[str, Any]) -> ContradictionLevel:
        """모순 수준 평가"""
        # 단순화된 모순 수준 평가 알고리즘
        # 실제 구현에서는 더 정교한 NLP 분석이 필요

        # 키워드 기반 분석
        strong_negation_words = {'not', 'never', 'impossible', 'cannot', 'contradicts'}
        moderate_negation_words = {'disagree', 'different', 'opposite', 'unlike'}

        statement_a_lower = statement_a.lower()
        statement_b_lower = statement_b.lower()

        # 강한 부정어 포함 시
        if any(word in statement_a_lower or word in statement_b_lower
               for word in strong_negation_words):
            return ContradictionLevel.MAJOR

        # 중간 수준 부정어 포함 시
        if any(word in statement_a_lower or word in statement_b_lower
               for word in moderate_negation_words):
            return ContradictionLevel.MODERATE

        # 문맥상 중요도 평가
        context_importance = context.get('importance_level', 0.5)
        if context_importance > 0.8:
            return ContradictionLevel.MAJOR
        elif context_importance > 0.5:
            return ContradictionLevel.MODERATE
        else:
            return ContradictionLevel.MINOR

    async def _handle_minor_contradiction(self, case: ContradictionCase) -> Result[str]:
        """사소한 모순 처리"""
        strategy = f"사소한 차이로 인정하고 두 관점을 병존시킵니다: '{case.statement_a}'와 '{case.statement_b}' 모두 유효한 관점으로 간주"
        return create_success(strategy)

    async def _handle_moderate_contradiction(self, case: ContradictionCase) -> Result[str]:
        """중간 수준 모순 처리"""
        strategy = f"추가 맥락을 통해 조화점을 찾습니다: {case.statement_a}와 {case.statement_b}가 서로 다른 조건이나 관점에서 성립할 수 있음을 인정"
        return create_success(strategy)

    async def _handle_major_contradiction(self, case: ContradictionCase) -> Result[str]:
        """심각한 모순 처리"""
        strategy = f"명시적 모순 수용: '{case.statement_a}'와 '{case.statement_b}'의 모순을 인정하고, 추가 정보나 증거 수집을 통해 해결 모색"
        return create_success(strategy)

    async def _handle_fundamental_contradiction(self, case: ContradictionCase) -> Result[str]:
        """근본적 모순 처리"""
        strategy = f"근본적 불확실성 인정: '{case.statement_a}'와 '{case.statement_b}' 간의 근본적 모순을 받아들이고, 현재로선 결정적 답을 내릴 수 없음을 솔직히 인정"
        return create_success(strategy)

    def get_contradiction_statistics(self) -> Dict[str, Any]:
        """모순 통계 조회"""
        total_cases = len(self.contradiction_cases)
        if total_cases == 0:
            return {'total_cases': 0}

        level_counts = {}
        for level in ContradictionLevel:
            level_counts[level.value] = sum(1 for case in self.contradiction_cases
                                          if case.level == level)

        accepted_rate = sum(1 for case in self.contradiction_cases
                          if case.accepted) / total_cases

        return {
            'total_cases': total_cases,
            'level_distribution': level_counts,
            'acceptance_rate': accepted_rate,
            'recent_cases': [
                {
                    'case_id': case.case_id,
                    'level': case.level.value,
                    'accepted': case.accepted,
                    'timestamp': case.timestamp
                }
                for case in self.contradiction_cases[-5:]  # 최근 5개
            ]
        }


class FinalJudgmentReservation(GovernanceProtocol):
    """
    최종 판단 유보 프로토콜
    불확실한 상황에서 성급한 판단을 피하고 유보하는 시스템
    """

    def __init__(self):
        super().__init__("final_judgment_reservation")
        self.reservations: List[JudgmentReservation] = []
        self.uncertainty_threshold = 0.3
        self.evidence_threshold = 0.7

    async def execute(self, input_data: Dict[str, Any]) -> Result[ProtocolResult]:
        """최종 판단 유보 프로토콜 실행"""
        try:
            protocol_id = generate_id("judgment_")

            # 입력 검증
            validation_result = await self.validate_input(input_data)
            if not validation_result.is_success:
                return create_failure(validation_result.error)

            # 판단 유보 필요성 평가
            should_reserve = await self._should_reserve_judgment(input_data)

            if should_reserve:
                reservation = await self._create_reservation(input_data)
                self.reservations.append(reservation)

                result = ProtocolResult(
                    protocol_id=protocol_id,
                    status=ProtocolStatus.COMPLETED,
                    result_data={
                        'judgment_reserved': True,
                        'reservation_id': reservation.reservation_id,
                        'reason': reservation.reservation_reason,
                        'revisit_conditions': reservation.revisit_conditions
                    },
                    timestamp=current_timestamp(),
                    success=True
                )
            else:
                result = ProtocolResult(
                    protocol_id=protocol_id,
                    status=ProtocolStatus.COMPLETED,
                    result_data={
                        'judgment_reserved': False,
                        'proceed_with_judgment': True
                    },
                    timestamp=current_timestamp(),
                    success=True
                )

            self.record_execution(result)
            return create_success(result)

        except Exception as e:
            result = ProtocolResult(
                protocol_id=protocol_id,
                status=ProtocolStatus.FAILED,
                result_data={'error': str(e)},
                timestamp=current_timestamp(),
                success=False,
                error_message=str(e)
            )
            self.record_execution(result)
            return create_failure(e)

    async def validate_input(self, input_data: Dict[str, Any]) -> Result[bool]:
        """입력 데이터 검증"""
        try:
            required_fields = ['subject', 'judgment_type', 'uncertainty_level', 'evidence_strength']
            for field in required_fields:
                if field not in input_data:
                    return create_failure(f"필수 필드 누락: {field}")

            # 판단 유형 검증
            judgment_type = input_data.get('judgment_type')
            if judgment_type not in [jt.value for jt in JudgmentType]:
                return create_failure(f"유효하지 않은 판단 유형: {judgment_type}")

            # 수치 범위 검증
            uncertainty = input_data.get('uncertainty_level', 0)
            evidence = input_data.get('evidence_strength', 0)

            if not (0 <= uncertainty <= 1):
                return create_failure("uncertainty_level은 0-1 범위여야 합니다")

            if not (0 <= evidence <= 1):
                return create_failure("evidence_strength는 0-1 범위여야 합니다")

            return create_success(True)

        except Exception as e:
            return create_failure(e)

    async def _should_reserve_judgment(self, input_data: Dict[str, Any]) -> bool:
        """판단 유보 필요성 평가"""
        uncertainty_level = input_data.get('uncertainty_level', 0)
        evidence_strength = input_data.get('evidence_strength', 1)
        judgment_type = input_data.get('judgment_type')

        # 불확실성이 높거나 증거가 부족한 경우
        if uncertainty_level > self.uncertainty_threshold:
            return True

        if evidence_strength < self.evidence_threshold:
            return True

        # 판단 유형별 특별 조건
        if judgment_type == JudgmentType.PREDICTIVE.value and uncertainty_level > 0.2:
            return True

        if judgment_type == JudgmentType.PRESCRIPTIVE.value and evidence_strength < 0.8:
            return True

        return False

    async def _create_reservation(self, input_data: Dict[str, Any]) -> JudgmentReservation:
        """판단 유보 객체 생성"""
        uncertainty_level = input_data.get('uncertainty_level', 0)
        evidence_strength = input_data.get('evidence_strength', 1)

        # 유보 이유 생성
        reasons = []
        if uncertainty_level > self.uncertainty_threshold:
            reasons.append(f"높은 불확실성 (수준: {uncertainty_level:.2f})")
        if evidence_strength < self.evidence_threshold:
            reasons.append(f"불충분한 증거 (강도: {evidence_strength:.2f})")

        reservation_reason = "; ".join(reasons) if reasons else "신중한 검토 필요"

        # 재검토 조건 생성
        revisit_conditions = []
        if uncertainty_level > self.uncertainty_threshold:
            revisit_conditions.append("추가 정보나 명확한 증거 확보")
        if evidence_strength < self.evidence_threshold:
            revisit_conditions.append("더 강력한 증거나 다수의 출처 확인")

        revisit_conditions.append("시간 경과에 따른 상황 변화 관찰")

        return JudgmentReservation(
            reservation_id=generate_id("reservation_"),
            subject=input_data['subject'],
            judgment_type=JudgmentType(input_data['judgment_type']),
            uncertainty_level=uncertainty_level,
            evidence_strength=evidence_strength,
            reservation_reason=reservation_reason,
            revisit_conditions=revisit_conditions
        )

    def get_reservation_statistics(self) -> Dict[str, Any]:
        """판단 유보 통계"""
        total_reservations = len(self.reservations)
        if total_reservations == 0:
            return {'total_reservations': 0}

        # 유형별 분포
        type_counts = {}
        for jtype in JudgmentType:
            type_counts[jtype.value] = sum(1 for r in self.reservations
                                         if r.judgment_type == jtype)

        # 평균 불확실성 및 증거 강도
        avg_uncertainty = sum(r.uncertainty_level for r in self.reservations) / total_reservations
        avg_evidence = sum(r.evidence_strength for r in self.reservations) / total_reservations

        return {
            'total_reservations': total_reservations,
            'type_distribution': type_counts,
            'average_uncertainty': avg_uncertainty,
            'average_evidence_strength': avg_evidence,
            'recent_reservations': [
                {
                    'reservation_id': r.reservation_id,
                    'subject': r.subject,
                    'type': r.judgment_type.value,
                    'reason': r.reservation_reason,
                    'timestamp': r.timestamp
                }
                for r in self.reservations[-5:]
            ]
        }


class TrustVerifyRollback(GovernanceProtocol):
    """
    신뢰-검증-롤백 프로토콜
    정보나 행동에 대해 신뢰하되 검증하고, 필요시 롤백하는 시스템
    """

    def __init__(self):
        super().__init__("trust_verify_rollback")
        self.trust_records: List[TrustRecord] = []
        self.verification_methods = {
            'source_credibility': self._verify_source_credibility,
            'cross_reference': self._verify_cross_reference,
            'logical_consistency': self._verify_logical_consistency,
            'empirical_evidence': self._verify_empirical_evidence
        }

    async def execute(self, input_data: Dict[str, Any]) -> Result[ProtocolResult]:
        """신뢰-검증-롤백 프로토콜 실행"""
        try:
            protocol_id = generate_id("trust_")

            # 입력 검증
            validation_result = await self.validate_input(input_data)
            if not validation_result.is_success:
                return create_failure(validation_result.error)

            # 1단계: 신뢰 수준 평가
            trust_level = await self._assess_trust_level(input_data)

            # 2단계: 검증 실행
            verification_result = await self._perform_verification(input_data, trust_level)

            # 3단계: 신뢰 기록 생성
            trust_record = TrustRecord(
                record_id=generate_id("trust_record_"),
                source=input_data['source'],
                trust_level=trust_level,
                verification_method=input_data.get('verification_method', 'source_credibility'),
                verification_result=verification_result.is_success,
                evidence=verification_result.value if verification_result.is_success else {},
                rollback_conditions=input_data.get('rollback_conditions', [])
            )

            self.trust_records.append(trust_record)

            # 4단계: 롤백 필요성 평가
            rollback_needed = await self._assess_rollback_need(trust_record, input_data)

            result_data = {
                'trust_record_id': trust_record.record_id,
                'trust_level': trust_level.value,
                'verification_passed': verification_result.is_success,
                'rollback_needed': rollback_needed
            }

            if rollback_needed:
                rollback_result = await self._perform_rollback(trust_record, input_data)
                result_data['rollback_result'] = rollback_result.value if rollback_result.is_success else None

            result = ProtocolResult(
                protocol_id=protocol_id,
                status=ProtocolStatus.COMPLETED,
                result_data=result_data,
                timestamp=current_timestamp(),
                success=True
            )

            self.record_execution(result)
            return create_success(result)

        except Exception as e:
            result = ProtocolResult(
                protocol_id=protocol_id,
                status=ProtocolStatus.FAILED,
                result_data={'error': str(e)},
                timestamp=current_timestamp(),
                success=False,
                error_message=str(e)
            )
            self.record_execution(result)
            return create_failure(e)

    async def validate_input(self, input_data: Dict[str, Any]) -> Result[bool]:
        """입력 데이터 검증"""
        try:
            required_fields = ['source', 'content']
            for field in required_fields:
                if field not in input_data:
                    return create_failure(f"필수 필드 누락: {field}")

            verification_method = input_data.get('verification_method')
            if verification_method and verification_method not in self.verification_methods:
                return create_failure(f"지원하지 않는 검증 방법: {verification_method}")

            return create_success(True)

        except Exception as e:
            return create_failure(e)

    async def _assess_trust_level(self, input_data: Dict[str, Any]) -> TrustLevel:
        """신뢰 수준 평가"""
        source = input_data['source']
        content = input_data['content']

        # 소스 기반 초기 신뢰도 평가
        trusted_sources = {'academic_paper', 'government_official', 'verified_expert'}
        suspicious_sources = {'anonymous', 'unverified', 'social_media'}

        if source in trusted_sources:
            base_trust = TrustLevel.HIGH
        elif source in suspicious_sources:
            base_trust = TrustLevel.LOW
        else:
            base_trust = TrustLevel.MEDIUM

        # 내용 기반 조정
        content_lower = content.lower()
        confidence_indicators = ['proven', 'verified', 'confirmed', 'established']
        uncertainty_indicators = ['maybe', 'possibly', 'might', 'unclear']

        if any(indicator in content_lower for indicator in confidence_indicators):
            if base_trust == TrustLevel.LOW:
                return TrustLevel.MEDIUM
            elif base_trust == TrustLevel.MEDIUM:
                return TrustLevel.HIGH

        if any(indicator in content_lower for indicator in uncertainty_indicators):
            if base_trust == TrustLevel.HIGH:
                return TrustLevel.MEDIUM
            elif base_trust == TrustLevel.MEDIUM:
                return TrustLevel.LOW

        return base_trust

    async def _perform_verification(self, input_data: Dict[str, Any],
                                  trust_level: TrustLevel) -> Result[Dict[str, Any]]:
        """검증 수행"""
        verification_method = input_data.get('verification_method', 'source_credibility')

        if verification_method in self.verification_methods:
            verifier = self.verification_methods[verification_method]
            return await verifier(input_data, trust_level)
        else:
            return create_failure(f"알 수 없는 검증 방법: {verification_method}")

    async def _verify_source_credibility(self, input_data: Dict[str, Any],
                                       trust_level: TrustLevel) -> Result[Dict[str, Any]]:
        """소스 신뢰성 검증"""
        source = input_data['source']

        # 간단한 신뢰성 평가 로직
        credibility_score = 0.5

        if trust_level == TrustLevel.HIGH:
            credibility_score = 0.9
        elif trust_level == TrustLevel.MEDIUM:
            credibility_score = 0.7
        elif trust_level == TrustLevel.LOW:
            credibility_score = 0.3

        evidence = {
            'credibility_score': credibility_score,
            'verification_method': 'source_credibility',
            'source_assessment': f"소스 '{source}'의 신뢰성 점수: {credibility_score}"
        }

        return create_success(evidence)

    async def _verify_cross_reference(self, input_data: Dict[str, Any],
                                    trust_level: TrustLevel) -> Result[Dict[str, Any]]:
        """교차 참조 검증"""
        # 교차 참조 시뮬레이션
        content = input_data['content']

        # 가상의 교차 참조 결과
        references_found = min(len(content) // 50, 5)  # 내용 길이에 비례한 참조 수
        consistency_score = 0.8 if trust_level != TrustLevel.LOW else 0.4

        evidence = {
            'references_found': references_found,
            'consistency_score': consistency_score,
            'verification_method': 'cross_reference'
        }

        return create_success(evidence)

    async def _verify_logical_consistency(self, input_data: Dict[str, Any],
                                        trust_level: TrustLevel) -> Result[Dict[str, Any]]:
        """논리적 일관성 검증"""
        content = input_data['content']

        # 간단한 논리 일관성 평가
        contradiction_keywords = ['but', 'however', 'although', 'despite']
        logical_keywords = ['therefore', 'thus', 'hence', 'consequently']

        contradictions = sum(1 for keyword in contradiction_keywords if keyword in content.lower())
        logical_connections = sum(1 for keyword in logical_keywords if keyword in content.lower())

        consistency_score = max(0.1, (logical_connections - contradictions * 0.3) / max(1, len(content.split()) // 10))
        consistency_score = min(consistency_score, 1.0)

        evidence = {
            'logical_consistency_score': consistency_score,
            'contradictions_detected': contradictions,
            'logical_connections_found': logical_connections,
            'verification_method': 'logical_consistency'
        }

        return create_success(evidence)

    async def _verify_empirical_evidence(self, input_data: Dict[str, Any],
                                       trust_level: TrustLevel) -> Result[Dict[str, Any]]:
        """경험적 증거 검증"""
        content = input_data['content']

        # 경험적 증거 지표 검색
        evidence_keywords = ['study', 'research', 'data', 'experiment', 'observation', 'measurement']
        evidence_count = sum(1 for keyword in evidence_keywords if keyword in content.lower())

        evidence_strength = min(evidence_count * 0.2, 1.0)

        evidence = {
            'empirical_evidence_strength': evidence_strength,
            'evidence_indicators_found': evidence_count,
            'verification_method': 'empirical_evidence'
        }

        return create_success(evidence)

    async def _assess_rollback_need(self, trust_record: TrustRecord,
                                  input_data: Dict[str, Any]) -> bool:
        """롤백 필요성 평가"""
        # 검증 실패시 롤백 필요
        if not trust_record.verification_result:
            return True

        # 신뢰 수준이 낮고 검증 증거가 약한 경우
        if (trust_record.trust_level == TrustLevel.LOW and
            trust_record.evidence.get('credibility_score', 1.0) < 0.5):
            return True

        # 롤백 조건 확인
        for condition in trust_record.rollback_conditions:
            if self._check_rollback_condition(condition, input_data):
                return True

        return False

    def _check_rollback_condition(self, condition: str, input_data: Dict[str, Any]) -> bool:
        """롤백 조건 확인"""
        # 간단한 조건 확인 로직
        if 'verification_failed' in condition and not input_data.get('verification_passed', True):
            return True

        if 'low_confidence' in condition and input_data.get('confidence_level', 1.0) < 0.5:
            return True

        return False

    async def _perform_rollback(self, trust_record: TrustRecord,
                              input_data: Dict[str, Any]) -> Result[Dict[str, Any]]:
        """롤백 수행"""
        rollback_actions = []

        # 신뢰 수준 하향 조정
        if trust_record.trust_level != TrustLevel.NONE:
            rollback_actions.append("신뢰 수준 하향 조정")

        # 정보 재검증 요청
        rollback_actions.append("추가 검증 필요")

        # 사용자 경고
        rollback_actions.append("사용자에게 불확실성 경고")

        rollback_result = {
            'rollback_actions': rollback_actions,
            'rollback_reason': "검증 실패 또는 신뢰성 부족",
            'timestamp': current_timestamp()
        }

        return create_success(rollback_result)

    def get_trust_statistics(self) -> Dict[str, Any]:
        """신뢰 통계"""
        total_records = len(self.trust_records)
        if total_records == 0:
            return {'total_records': 0}

        # 신뢰 수준별 분포
        trust_level_counts = {}
        for level in TrustLevel:
            trust_level_counts[level.value] = sum(1 for r in self.trust_records
                                                if r.trust_level == level)

        # 검증 성공률
        verification_success_rate = sum(1 for r in self.trust_records
                                      if r.verification_result) / total_records

        return {
            'total_records': total_records,
            'trust_level_distribution': trust_level_counts,
            'verification_success_rate': verification_success_rate,
            'recent_records': [
                {
                    'record_id': r.record_id,
                    'source': r.source,
                    'trust_level': r.trust_level.value,
                    'verification_result': r.verification_result,
                    'timestamp': r.timestamp
                }
                for r in self.trust_records[-5:]
            ]
        }


class GovernanceProtocolManager:
    """거버넌스 프로토콜 통합 관리자"""

    def __init__(self):
        self.protocols = {
            'contradiction_acceptance': ContradictionAcceptance(),
            'final_judgment_reservation': FinalJudgmentReservation(),
            'trust_verify_rollback': TrustVerifyRollback()
        }
        self.protocol_coordination_history: List[Dict[str, Any]] = []

    async def execute_protocol(self, protocol_name: str,
                             input_data: Dict[str, Any]) -> Result[ProtocolResult]:
        """특정 프로토콜 실행"""
        if protocol_name not in self.protocols:
            return create_failure(f"알 수 없는 프로토콜: {protocol_name}")

        protocol = self.protocols[protocol_name]
        return await protocol.execute(input_data)

    async def execute_coordinated_protocols(self,
                                          protocol_sequence: List[Tuple[str, Dict[str, Any]]]) -> Result[List[ProtocolResult]]:
        """여러 프로토콜 순차 실행"""
        results = []
        coordination_id = generate_id("coordination_")

        try:
            for protocol_name, input_data in protocol_sequence:
                result = await self.execute_protocol(protocol_name, input_data)
                results.append(result.value if result.is_success else None)

                # 이전 결과를 다음 프로토콜 입력에 반영
                if result.is_success and len(protocol_sequence) > len(results):
                    next_protocol_data = protocol_sequence[len(results)][1]
                    next_protocol_data['previous_result'] = result.value

            # 조정 이력 기록
            self.protocol_coordination_history.append({
                'coordination_id': coordination_id,
                'protocols_executed': [name for name, _ in protocol_sequence],
                'results_count': len([r for r in results if r is not None]),
                'timestamp': current_timestamp()
            })

            return create_success(results)

        except Exception as e:
            return create_failure(e)

    def get_overall_statistics(self) -> Dict[str, Any]:
        """전체 프로토콜 통계"""
        stats = {}

        for name, protocol in self.protocols.items():
            if hasattr(protocol, 'get_success_rate'):
                stats[f"{name}_success_rate"] = protocol.get_success_rate()

            if hasattr(protocol, 'get_contradiction_statistics'):
                stats[f"{name}_stats"] = protocol.get_contradiction_statistics()
            elif hasattr(protocol, 'get_reservation_statistics'):
                stats[f"{name}_stats"] = protocol.get_reservation_statistics()
            elif hasattr(protocol, 'get_trust_statistics'):
                stats[f"{name}_stats"] = protocol.get_trust_statistics()

        stats['coordination_count'] = len(self.protocol_coordination_history)
        return stats