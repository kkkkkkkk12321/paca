"""
Reasoning Error Classes Module
추론 시스템 관련 에러 클래스들
"""

from typing import List, Any, Dict
from .base import PacaError, ErrorSeverity, ErrorCategory, KeyValuePair


class ReasoningError(PacaError):
    """추론 시스템 기본 에러"""

    def __init__(
        self,
        message: str,
        reasoning_type: str = None,
        step_number: int = None,
        metadata: KeyValuePair = None,
        recovery_hints: List[str] = None
    ):
        enhanced_metadata = metadata or {}
        enhanced_hints = recovery_hints or ["Check reasoning parameters", "Verify logical consistency"]

        if reasoning_type:
            enhanced_metadata['reasoning_type'] = reasoning_type
            enhanced_hints.append(f"Review {reasoning_type} reasoning rules")

        if step_number is not None:
            enhanced_metadata['step_number'] = step_number
            enhanced_hints.append(f"Check step {step_number} in reasoning chain")

        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.BUSINESS_LOGIC,
            metadata=enhanced_metadata,
            recovery_hints=enhanced_hints
        )


class DeductiveReasoningError(ReasoningError):
    """연역적 추론 에러"""

    def __init__(
        self,
        message: str,
        premises: List[str] = None,
        conclusion: str = None,
        invalid_rule: str = None
    ):
        metadata = {}
        hints = ["Check premises validity", "Verify logical rules"]

        if premises:
            metadata['premises'] = premises
            hints.append(f"Review {len(premises)} premises")

        if conclusion:
            metadata['conclusion'] = conclusion

        if invalid_rule:
            metadata['invalid_rule'] = invalid_rule
            hints.append(f"Fix rule: {invalid_rule}")

        super().__init__(
            message=message,
            reasoning_type="deductive",
            metadata=metadata,
            recovery_hints=hints
        )


class InductiveReasoningError(ReasoningError):
    """귀납적 추론 에러"""

    def __init__(
        self,
        message: str,
        sample_size: int = None,
        confidence_level: float = None,
        pattern: str = None
    ):
        metadata = {}
        hints = ["Check sample size", "Verify pattern consistency"]

        if sample_size is not None:
            metadata['sample_size'] = sample_size
            if sample_size < 10:
                hints.append("Consider increasing sample size")

        if confidence_level is not None:
            metadata['confidence_level'] = confidence_level
            if confidence_level < 0.5:
                hints.append("Low confidence level detected")

        if pattern:
            metadata['pattern'] = pattern

        super().__init__(
            message=message,
            reasoning_type="inductive",
            metadata=metadata,
            recovery_hints=hints
        )


class AbductiveReasoningError(ReasoningError):
    """가추적 추론 에러"""

    def __init__(
        self,
        message: str,
        observations: List[str] = None,
        hypotheses: List[str] = None,
        best_explanation: str = None
    ):
        metadata = {}
        hints = ["Check observation quality", "Verify hypothesis generation"]

        if observations:
            metadata['observations'] = observations
            hints.append(f"Review {len(observations)} observations")

        if hypotheses:
            metadata['hypotheses'] = hypotheses
            hints.append(f"Evaluate {len(hypotheses)} hypotheses")

        if best_explanation:
            metadata['best_explanation'] = best_explanation

        super().__init__(
            message=message,
            reasoning_type="abductive",
            metadata=metadata,
            recovery_hints=hints
        )


class LogicalInconsistencyError(ReasoningError):
    """논리적 비일관성 에러"""

    def __init__(
        self,
        message: str,
        conflicting_statements: List[str] = None,
        reasoning_chain: List[Dict[str, Any]] = None
    ):
        metadata = {}
        hints = ["Resolve logical conflicts", "Check reasoning chain"]

        if conflicting_statements:
            metadata['conflicting_statements'] = conflicting_statements
            hints.append(f"Resolve {len(conflicting_statements)} conflicts")

        if reasoning_chain:
            metadata['reasoning_chain_length'] = len(reasoning_chain)
            # 민감한 정보를 피하기 위해 체인의 길이만 저장

        super().__init__(
            message=message,
            reasoning_type="consistency_check",
            metadata=metadata,
            recovery_hints=hints
        )


class ChainOfThoughtError(ReasoningError):
    """사고 체인 에러"""

    def __init__(
        self,
        message: str,
        chain_id: str = None,
        broken_link: int = None,
        chain_length: int = None
    ):
        metadata = {}
        hints = ["Check reasoning chain links", "Verify logical flow"]

        if chain_id:
            metadata['chain_id'] = chain_id

        if broken_link is not None:
            metadata['broken_link'] = broken_link
            hints.append(f"Fix link at position {broken_link}")

        if chain_length is not None:
            metadata['chain_length'] = chain_length

        super().__init__(
            message=message,
            reasoning_type="chain_of_thought",
            metadata=metadata,
            recovery_hints=hints
        )


class MetacognitionError(ReasoningError):
    """메타인지 에러"""

    def __init__(
        self,
        message: str,
        metacognitive_process: str = None,
        confidence_estimate: float = None,
        actual_performance: float = None
    ):
        metadata = {}
        hints = ["Check metacognitive awareness", "Calibrate confidence estimates"]

        if metacognitive_process:
            metadata['metacognitive_process'] = metacognitive_process

        if confidence_estimate is not None:
            metadata['confidence_estimate'] = confidence_estimate

        if actual_performance is not None:
            metadata['actual_performance'] = actual_performance

        # 신뢰도와 실제 성능 간의 차이 확인
        if confidence_estimate is not None and actual_performance is not None:
            calibration_error = abs(confidence_estimate - actual_performance)
            metadata['calibration_error'] = calibration_error
            if calibration_error > 0.3:
                hints.append("High calibration error detected")

        super().__init__(
            message=message,
            reasoning_type="metacognition",
            metadata=metadata,
            recovery_hints=hints
        )


class ParallelReasoningError(ReasoningError):
    """병렬 추론 에러"""

    def __init__(
        self,
        message: str,
        parallel_threads: int = None,
        failed_thread_id: str = None,
        synchronization_point: str = None
    ):
        metadata = {}
        hints = ["Check parallel processing", "Verify thread synchronization"]

        if parallel_threads is not None:
            metadata['parallel_threads'] = parallel_threads

        if failed_thread_id:
            metadata['failed_thread_id'] = failed_thread_id
            hints.append(f"Restart thread: {failed_thread_id}")

        if synchronization_point:
            metadata['synchronization_point'] = synchronization_point

        super().__init__(
            message=message,
            reasoning_type="parallel",
            metadata=metadata,
            recovery_hints=hints
        )


class ReasoningTimeoutError(ReasoningError):
    """추론 타임아웃 에러"""

    def __init__(
        self,
        message: str,
        timeout_duration: float = None,
        completed_steps: int = None,
        total_steps: int = None
    ):
        metadata = {}
        hints = ["Increase timeout duration", "Optimize reasoning complexity"]

        if timeout_duration is not None:
            metadata['timeout_duration'] = timeout_duration

        if completed_steps is not None:
            metadata['completed_steps'] = completed_steps

        if total_steps is not None:
            metadata['total_steps'] = total_steps
            if completed_steps is not None:
                progress = completed_steps / total_steps
                metadata['progress'] = progress
                hints.append(f"Progress: {progress:.1%}")

        super().__init__(
            message=message,
            reasoning_type="timeout",
            metadata=metadata,
            recovery_hints=hints
        )