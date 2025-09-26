"""
Core Principles Module
4대 핵심 원칙 구현

1. 사용자 주권 (User Sovereignty)
2. 인식론적 겸손 (Epistemic Humility)
3. 수용적 태세 (Receptive Stance)
4. 건설적 이의 제기 (Constructive Objection)
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
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


class PrincipleType(Enum):
    """원칙 타입"""
    USER_SOVEREIGNTY = "user_sovereignty"
    EPISTEMIC_HUMILITY = "epistemic_humility"
    RECEPTIVE_STANCE = "receptive_stance"
    CONSTRUCTIVE_OBJECTION = "constructive_objection"


class ViolationSeverity(Enum):
    """위반 심각도"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class PrincipleViolation:
    """원칙 위반 정보"""
    violation_id: ID
    principle_type: PrincipleType
    severity: ViolationSeverity
    description: str
    context: Dict[str, Any]
    timestamp: Timestamp
    auto_corrected: bool = False
    correction_action: Optional[str] = None


class CorePrinciple(ABC):
    """핵심 원칙 기본 클래스"""

    def __init__(self, principle_type: PrincipleType):
        self.principle_type = principle_type
        self.violation_history: List[PrincipleViolation] = []
        self.active = True

    @abstractmethod
    async def validate_action(self, action: Dict[str, Any]) -> Result[bool]:
        """액션이 원칙에 부합하는지 검증"""
        pass

    @abstractmethod
    async def suggest_correction(self, violation: PrincipleViolation) -> Result[str]:
        """위반에 대한 수정 제안"""
        pass

    def record_violation(self, violation: PrincipleViolation) -> None:
        """위반 기록"""
        self.violation_history.append(violation)

    def get_violation_count(self, hours: int = 24) -> int:
        """특정 시간 내 위반 횟수"""
        cutoff_time = current_timestamp() - (hours * 3600)
        return sum(1 for v in self.violation_history if v.timestamp >= cutoff_time)


class UserSovereignty(CorePrinciple):
    """
    사용자 주권 원칙
    사용자의 의도와 가치를 최우선으로 하는 원칙
    """

    def __init__(self):
        super().__init__(PrincipleType.USER_SOVEREIGNTY)
        self.user_preferences: Dict[str, Any] = {}
        self.consent_required_actions = {
            'data_sharing', 'external_communication', 'system_modification',
            'privacy_sensitive_operation', 'user_data_analysis'
        }

    async def validate_action(self, action: Dict[str, Any]) -> Result[bool]:
        """사용자 주권 원칙 검증"""
        try:
            action_type = action.get('type', '')
            user_consent = action.get('user_consent', False)

            # 동의 필요 액션 확인
            if action_type in self.consent_required_actions:
                if not user_consent:
                    violation = PrincipleViolation(
                        violation_id=generate_id("violation_"),
                        principle_type=self.principle_type,
                        severity=ViolationSeverity.HIGH,
                        description=f"사용자 동의 없이 {action_type} 실행 시도",
                        context=action,
                        timestamp=current_timestamp()
                    )
                    self.record_violation(violation)
                    return create_failure(f"사용자 동의가 필요한 액션입니다: {action_type}")

            # 사용자 선호도와 일치성 확인
            user_intent = action.get('user_intent', {})
            if user_intent and not self._aligns_with_user_intent(user_intent):
                violation = PrincipleViolation(
                    violation_id=generate_id("violation_"),
                    principle_type=self.principle_type,
                    severity=ViolationSeverity.MEDIUM,
                    description="사용자 의도와 불일치하는 액션",
                    context=action,
                    timestamp=current_timestamp()
                )
                self.record_violation(violation)
                return create_failure("사용자 의도와 일치하지 않는 액션입니다")

            return create_success(True)

        except Exception as e:
            return create_failure(e)

    async def suggest_correction(self, violation: PrincipleViolation) -> Result[str]:
        """사용자 주권 위반 수정 제안"""
        try:
            if "동의" in violation.description:
                suggestion = "사용자에게 명시적 동의를 요청하고 승인 후 진행하세요."
            elif "의도" in violation.description:
                suggestion = "사용자의 명확한 의도를 확인하고 그에 맞게 조정하세요."
            else:
                suggestion = "사용자의 권한과 선호도를 존중하는 방향으로 수정하세요."

            return create_success(suggestion)

        except Exception as e:
            return create_failure(e)

    def _aligns_with_user_intent(self, user_intent: Dict[str, Any]) -> bool:
        """사용자 의도와 일치성 확인"""
        # 사용자 선호도와 비교
        for key, value in user_intent.items():
            if key in self.user_preferences:
                if self.user_preferences[key] != value:
                    return False
        return True

    def update_user_preferences(self, preferences: Dict[str, Any]) -> None:
        """사용자 선호도 업데이트"""
        self.user_preferences.update(preferences)


class EpistemicHumility(CorePrinciple):
    """
    인식론적 겸손 원칙
    지식의 한계를 인정하고 불확실성을 받아들이는 원칙
    """

    def __init__(self):
        super().__init__(PrincipleType.EPISTEMIC_HUMILITY)
        self.confidence_threshold = 0.8  # 확신 임계값
        self.uncertainty_keywords = {
            'certain', 'definitely', 'absolutely', 'always', 'never',
            'impossible', 'guaranteed', 'perfect', 'complete'
        }

    async def validate_action(self, action: Dict[str, Any]) -> Result[bool]:
        """인식론적 겸손 원칙 검증"""
        try:
            response_text = action.get('response_text', '')
            confidence_level = action.get('confidence_level', 0.5)

            # 과도한 확신 표현 검사
            if self._contains_overconfident_language(response_text):
                violation = PrincipleViolation(
                    violation_id=generate_id("violation_"),
                    principle_type=self.principle_type,
                    severity=ViolationSeverity.MEDIUM,
                    description="과도한 확신 표현 사용",
                    context={'response_text': response_text},
                    timestamp=current_timestamp()
                )
                self.record_violation(violation)
                return create_failure("과도한 확신 표현이 감지되었습니다")

            # 확신도와 실제 지식 수준 불일치 검사
            knowledge_uncertainty = action.get('knowledge_uncertainty', 0.0)
            if confidence_level > self.confidence_threshold and knowledge_uncertainty > 0.3:
                violation = PrincipleViolation(
                    violation_id=generate_id("violation_"),
                    principle_type=self.principle_type,
                    severity=ViolationSeverity.HIGH,
                    description="불확실한 지식에 대한 과도한 확신",
                    context={'confidence': confidence_level, 'uncertainty': knowledge_uncertainty},
                    timestamp=current_timestamp()
                )
                self.record_violation(violation)
                return create_failure("불확실한 정보에 대해 과도한 확신을 표현했습니다")

            return create_success(True)

        except Exception as e:
            return create_failure(e)

    async def suggest_correction(self, violation: PrincipleViolation) -> Result[str]:
        """인식론적 겸손 위반 수정 제안"""
        try:
            if "확신 표현" in violation.description:
                suggestion = "불확실성을 인정하는 표현을 사용하세요 (예: '~일 가능성이 있습니다', '~로 보입니다')"
            elif "과도한 확신" in violation.description:
                suggestion = "지식의 한계를 명시하고 추가 확인이 필요함을 언급하세요"
            else:
                suggestion = "겸손한 표현을 사용하고 불확실성을 솔직히 인정하세요"

            return create_success(suggestion)

        except Exception as e:
            return create_failure(e)

    def _contains_overconfident_language(self, text: str) -> bool:
        """과도한 확신 표현 검사"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.uncertainty_keywords)


class ReceptiveStance(CorePrinciple):
    """
    수용적 태세 원칙
    사용자의 피드백과 새로운 정보에 열린 자세를 유지하는 원칙
    """

    def __init__(self):
        super().__init__(PrincipleType.RECEPTIVE_STANCE)
        self.feedback_response_rate = 0.0
        self.adaptation_attempts = 0
        self.rejection_patterns = {
            'dismissive', 'ignore', 'reject', 'dismiss', 'wrong', 'incorrect'
        }

    async def validate_action(self, action: Dict[str, Any]) -> Result[bool]:
        """수용적 태세 원칙 검증"""
        try:
            response_to_feedback = action.get('response_to_feedback', '')
            feedback_acknowledged = action.get('feedback_acknowledged', True)
            adaptation_attempted = action.get('adaptation_attempted', False)

            # 피드백 무시 또는 거부 검사
            if response_to_feedback and self._contains_dismissive_language(response_to_feedback):
                violation = PrincipleViolation(
                    violation_id=generate_id("violation_"),
                    principle_type=self.principle_type,
                    severity=ViolationSeverity.HIGH,
                    description="사용자 피드백에 대한 거부적 반응",
                    context={'response': response_to_feedback},
                    timestamp=current_timestamp()
                )
                self.record_violation(violation)
                return create_failure("피드백에 대한 거부적 반응이 감지되었습니다")

            # 피드백 수용 여부 확인
            if not feedback_acknowledged and response_to_feedback:
                violation = PrincipleViolation(
                    violation_id=generate_id("violation_"),
                    principle_type=self.principle_type,
                    severity=ViolationSeverity.MEDIUM,
                    description="피드백 미수용",
                    context=action,
                    timestamp=current_timestamp()
                )
                self.record_violation(violation)
                return create_failure("피드백을 적절히 수용하지 않았습니다")

            # 적응 시도 추적
            if adaptation_attempted:
                self.adaptation_attempts += 1

            return create_success(True)

        except Exception as e:
            return create_failure(e)

    async def suggest_correction(self, violation: PrincipleViolation) -> Result[str]:
        """수용적 태세 위반 수정 제안"""
        try:
            if "거부적 반응" in violation.description:
                suggestion = "피드백을 감사히 받아들이고 건설적으로 반영하는 방안을 모색하세요"
            elif "미수용" in violation.description:
                suggestion = "피드백의 가치를 인정하고 적용 가능한 부분을 찾아 반영하세요"
            else:
                suggestion = "열린 마음으로 피드백을 수용하고 개선 방향을 제시하세요"

            return create_success(suggestion)

        except Exception as e:
            return create_failure(e)

    def _contains_dismissive_language(self, text: str) -> bool:
        """거부적 언어 패턴 검사"""
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in self.rejection_patterns)

    def get_feedback_responsiveness(self) -> float:
        """피드백 반응성 점수"""
        total_interactions = len(self.violation_history) + self.adaptation_attempts
        if total_interactions == 0:
            return 1.0
        return self.adaptation_attempts / total_interactions


class ConstructiveObjection(CorePrinciple):
    """
    건설적 이의 제기 원칙
    사용자의 요청이 해로울 수 있을 때 건설적으로 이의를 제기하는 원칙
    """

    def __init__(self):
        super().__init__(PrincipleType.CONSTRUCTIVE_OBJECTION)
        self.harm_indicators = {
            'security_risk', 'privacy_violation', 'unethical_request',
            'harmful_content', 'illegal_activity', 'dangerous_action'
        }
        self.objection_history: List[Dict[str, Any]] = []

    async def validate_action(self, action: Dict[str, Any]) -> Result[bool]:
        """건설적 이의 제기 원칙 검증"""
        try:
            user_request = action.get('user_request', '')
            harm_level = action.get('harm_level', 0)
            objection_made = action.get('objection_made', False)
            alternative_provided = action.get('alternative_provided', False)

            # 해로운 요청에 대한 이의 제기 확인
            if harm_level > 0.6 and not objection_made:
                violation = PrincipleViolation(
                    violation_id=generate_id("violation_"),
                    principle_type=self.principle_type,
                    severity=ViolationSeverity.HIGH,
                    description="해로운 요청에 대한 이의 제기 미실시",
                    context={'request': user_request, 'harm_level': harm_level},
                    timestamp=current_timestamp()
                )
                self.record_violation(violation)
                return create_failure("해로운 요청에 대해 적절한 이의를 제기하지 않았습니다")

            # 이의 제기 시 대안 제시 확인
            if objection_made and not alternative_provided:
                violation = PrincipleViolation(
                    violation_id=generate_id("violation_"),
                    principle_type=self.principle_type,
                    severity=ViolationSeverity.MEDIUM,
                    description="이의 제기 시 건설적 대안 미제시",
                    context=action,
                    timestamp=current_timestamp()
                )
                self.record_violation(violation)
                return create_failure("이의 제기 시 건설적 대안을 제시해야 합니다")

            # 성공적인 이의 제기 기록
            if objection_made and alternative_provided:
                self.objection_history.append({
                    'timestamp': current_timestamp(),
                    'request': user_request,
                    'harm_level': harm_level,
                    'alternative_provided': True
                })

            return create_success(True)

        except Exception as e:
            return create_failure(e)

    async def suggest_correction(self, violation: PrincipleViolation) -> Result[str]:
        """건설적 이의 제기 위반 수정 제안"""
        try:
            if "이의 제기 미실시" in violation.description:
                suggestion = "요청의 잠재적 위험을 설명하고 신중한 검토를 제안하세요"
            elif "대안 미제시" in violation.description:
                suggestion = "문제점을 지적할 때는 반드시 건설적인 대안이나 개선방안을 함께 제시하세요"
            else:
                suggestion = "우려사항을 정중하게 표현하고 더 나은 방향을 제안하세요"

            return create_success(suggestion)

        except Exception as e:
            return create_failure(e)

    def assess_harm_level(self, request: str) -> float:
        """요청의 해로움 수준 평가"""
        harm_score = 0.0
        request_lower = request.lower()

        for indicator in self.harm_indicators:
            if indicator.replace('_', ' ') in request_lower:
                harm_score += 0.2

        return min(harm_score, 1.0)

    def get_objection_effectiveness(self) -> float:
        """이의 제기 효과성 점수"""
        if not self.objection_history:
            return 1.0

        successful_objections = sum(1 for obj in self.objection_history
                                  if obj.get('alternative_provided', False))
        return successful_objections / len(self.objection_history)


class CorePrinciples:
    """
    4대 핵심 원칙 통합 관리 시스템
    """

    def __init__(self):
        self.principles = {
            PrincipleType.USER_SOVEREIGNTY: UserSovereignty(),
            PrincipleType.EPISTEMIC_HUMILITY: EpistemicHumility(),
            PrincipleType.RECEPTIVE_STANCE: ReceptiveStance(),
            PrincipleType.CONSTRUCTIVE_OBJECTION: ConstructiveObjection()
        }
        self.global_violations: List[PrincipleViolation] = []
        self.monitoring_active = True

    async def validate_all_principles(self, action: Dict[str, Any]) -> Result[Dict[PrincipleType, bool]]:
        """모든 원칙에 대해 액션 검증"""
        try:
            results = {}

            # 병렬로 모든 원칙 검증
            tasks = []
            for principle_type, principle in self.principles.items():
                if principle.active:
                    tasks.append(principle.validate_action(action))
                else:
                    results[principle_type] = True

            if tasks:
                validation_results = await asyncio.gather(*tasks, return_exceptions=True)

                active_principles = [p for p in self.principles.values() if p.active]
                for i, result in enumerate(validation_results):
                    principle = active_principles[i]
                    if isinstance(result, Exception):
                        results[principle.principle_type] = False
                    else:
                        results[principle.principle_type] = result.is_success

            return create_success(results)

        except Exception as e:
            return create_failure(e)

    async def get_violation_summary(self, hours: int = 24) -> Dict[PrincipleType, int]:
        """원칙별 위반 요약"""
        summary = {}
        for principle_type, principle in self.principles.items():
            summary[principle_type] = principle.get_violation_count(hours)
        return summary

    async def get_improvement_suggestions(self) -> Dict[PrincipleType, List[str]]:
        """원칙별 개선 제안"""
        suggestions = {}

        for principle_type, principle in self.principles.items():
            principle_suggestions = []
            recent_violations = [v for v in principle.violation_history
                               if current_timestamp() - v.timestamp < 86400]  # 24시간

            for violation in recent_violations[-3:]:  # 최근 3개 위반
                suggestion_result = await principle.suggest_correction(violation)
                if suggestion_result.is_success:
                    principle_suggestions.append(suggestion_result.value)

            suggestions[principle_type] = principle_suggestions

        return suggestions

    def get_principle_health_score(self) -> Dict[PrincipleType, float]:
        """원칙별 건강도 점수 (0.0-1.0)"""
        scores = {}

        for principle_type, principle in self.principles.items():
            violations_24h = principle.get_violation_count(24)

            # 기본 점수에서 위반에 따라 감점
            base_score = 1.0
            penalty = min(violations_24h * 0.1, 0.8)  # 최대 80% 감점

            # 특별 보너스 계산
            bonus = 0.0
            if isinstance(principle, ReceptiveStance):
                bonus = principle.get_feedback_responsiveness() * 0.1
            elif isinstance(principle, ConstructiveObjection):
                bonus = principle.get_objection_effectiveness() * 0.1

            scores[principle_type] = max(base_score - penalty + bonus, 0.0)

        return scores

    def toggle_principle(self, principle_type: PrincipleType, active: bool) -> None:
        """원칙 활성화/비활성화"""
        if principle_type in self.principles:
            self.principles[principle_type].active = active

    def update_user_preferences(self, preferences: Dict[str, Any]) -> None:
        """사용자 선호도 업데이트 (사용자 주권 원칙에 반영)"""
        sovereignty = self.principles[PrincipleType.USER_SOVEREIGNTY]
        if isinstance(sovereignty, UserSovereignty):
            sovereignty.update_user_preferences(preferences)