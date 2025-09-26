"""
Cognitive Error Classes Module
인지 시스템 관련 에러 클래스들
"""

from typing import List, Any
from .base import PacaError, ErrorSeverity, ErrorCategory, KeyValuePair


class CognitiveError(PacaError):
    """인지 시스템 기본 에러"""

    def __init__(
        self,
        message: str,
        component: str = None,
        metadata: KeyValuePair = None,
        recovery_hints: List[str] = None
    ):
        enhanced_metadata = metadata or {}
        enhanced_hints = recovery_hints or ["Restart cognitive processing", "Check system resources"]

        if component:
            enhanced_metadata['cognitive_component'] = component
            enhanced_hints.append(f"Check {component} component status")

        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SYSTEM,
            metadata=enhanced_metadata,
            recovery_hints=enhanced_hints
        )


class MemoryError(CognitiveError):
    """메모리 시스템 에러"""

    def __init__(
        self,
        message: str,
        memory_type: str = None,
        memory_id: str = None,
        operation: str = None
    ):
        metadata = {}
        hints = ["Check memory system integrity", "Verify memory capacity"]

        if memory_type:
            metadata['memory_type'] = memory_type
            hints.append(f"Check {memory_type} memory")

        if memory_id:
            metadata['memory_id'] = memory_id

        if operation:
            metadata['operation'] = operation
            hints.append(f"Verify {operation} operation")

        super().__init__(
            message=message,
            component="memory",
            metadata=metadata,
            recovery_hints=hints
        )


class AttentionError(CognitiveError):
    """주의 집중 시스템 에러"""

    def __init__(
        self,
        message: str,
        focus_target: str = None,
        attention_level: float = None
    ):
        metadata = {}
        hints = ["Check attention mechanisms", "Verify focus targets"]

        if focus_target:
            metadata['focus_target'] = focus_target

        if attention_level is not None:
            metadata['attention_level'] = attention_level

        super().__init__(
            message=message,
            component="attention",
            metadata=metadata,
            recovery_hints=hints
        )


class PerceptionError(CognitiveError):
    """지각 처리 에러"""

    def __init__(
        self,
        message: str,
        sensory_input_type: str = None,
        processing_stage: str = None
    ):
        metadata = {}
        hints = ["Check sensory input", "Verify perception pipeline"]

        if sensory_input_type:
            metadata['sensory_input_type'] = sensory_input_type
            hints.append(f"Validate {sensory_input_type} input")

        if processing_stage:
            metadata['processing_stage'] = processing_stage

        super().__init__(
            message=message,
            component="perception",
            metadata=metadata,
            recovery_hints=hints
        )


class DecisionError(CognitiveError):
    """의사결정 시스템 에러"""

    def __init__(
        self,
        message: str,
        decision_context: str = None,
        available_options: List[str] = None,
        confidence_threshold: float = None
    ):
        metadata = {}
        hints = ["Check decision criteria", "Verify available options"]

        if decision_context:
            metadata['decision_context'] = decision_context

        if available_options:
            metadata['available_options'] = available_options
            hints.append(f"Review {len(available_options)} available options")

        if confidence_threshold is not None:
            metadata['confidence_threshold'] = confidence_threshold

        super().__init__(
            message=message,
            component="decision",
            metadata=metadata,
            recovery_hints=hints
        )


class ModelError(CognitiveError):
    """인지 모델 에러"""

    def __init__(
        self,
        message: str,
        model_name: str = None,
        model_state: str = None,
        input_data: Any = None
    ):
        metadata = {}
        hints = ["Check model parameters", "Verify model training"]

        if model_name:
            metadata['model_name'] = model_name
            hints.append(f"Restart {model_name} model")

        if model_state:
            metadata['model_state'] = model_state

        if input_data is not None:
            metadata['input_data_type'] = type(input_data).__name__

        super().__init__(
            message=message,
            component="model",
            metadata=metadata,
            recovery_hints=hints
        )


class ACTRError(ModelError):
    """ACT-R 모델 에러"""

    def __init__(
        self,
        message: str,
        production_rule: str = None,
        chunk_id: str = None,
        activation_level: float = None
    ):
        metadata = {}
        hints = ["Check ACT-R parameters", "Verify production rules"]

        if production_rule:
            metadata['production_rule'] = production_rule

        if chunk_id:
            metadata['chunk_id'] = chunk_id

        if activation_level is not None:
            metadata['activation_level'] = activation_level

        super().__init__(
            message=message,
            model_name="ACT-R",
            metadata=metadata
        )
        self.recovery_hints.extend(hints)


class SOARError(ModelError):
    """SOAR 모델 에러"""

    def __init__(
        self,
        message: str,
        goal_stack: List[str] = None,
        operator: str = None,
        state_representation: str = None
    ):
        metadata = {}
        hints = ["Check SOAR parameters", "Verify goal stack"]

        if goal_stack:
            metadata['goal_stack'] = goal_stack
            hints.append(f"Review {len(goal_stack)} goals in stack")

        if operator:
            metadata['operator'] = operator

        if state_representation:
            metadata['state_representation'] = state_representation

        super().__init__(
            message=message,
            model_name="SOAR",
            metadata=metadata
        )
        self.recovery_hints.extend(hints)


class SimulationError(CognitiveError):
    """인지 시뮬레이션 에러"""

    def __init__(
        self,
        message: str,
        simulation_step: int = None,
        simulation_state: str = None,
        environment_data: Any = None
    ):
        metadata = {}
        hints = ["Check simulation parameters", "Verify environment state"]

        if simulation_step is not None:
            metadata['simulation_step'] = simulation_step

        if simulation_state:
            metadata['simulation_state'] = simulation_state

        if environment_data is not None:
            metadata['environment_data_type'] = type(environment_data).__name__

        super().__init__(
            message=message,
            component="simulator",
            metadata=metadata,
            recovery_hints=hints
        )