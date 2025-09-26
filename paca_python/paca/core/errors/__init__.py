"""
Errors Module
PACA 시스템의 모든 에러 클래스들
"""

from .base import (
    ErrorSeverity,
    ErrorCategory,
    ErrorContext,
    PacaError,
    ApplicationError,
    InfrastructureError,
    ConfigurationError,
    ValidationError,
    NetworkError,
    AuthenticationError,
    AuthorizationError,
    ExternalServiceError
)

from .cognitive import (
    CognitiveError,
    MemoryError,
    AttentionError,
    PerceptionError,
    DecisionError,
    ModelError,
    ACTRError,
    SOARError,
    SimulationError
)

from .reasoning import (
    ReasoningError,
    DeductiveReasoningError,
    InductiveReasoningError,
    AbductiveReasoningError,
    LogicalInconsistencyError,
    ChainOfThoughtError,
    MetacognitionError,
    ParallelReasoningError,
    ReasoningTimeoutError
)

__all__ = [
    # Base errors
    'ErrorSeverity',
    'ErrorCategory',
    'ErrorContext',
    'PacaError',
    'ApplicationError',
    'InfrastructureError',
    'ConfigurationError',
    'ValidationError',
    'NetworkError',
    'AuthenticationError',
    'AuthorizationError',
    'ExternalServiceError',

    # Cognitive errors
    'CognitiveError',
    'MemoryError',
    'AttentionError',
    'PerceptionError',
    'DecisionError',
    'ModelError',
    'ACTRError',
    'SOARError',
    'SimulationError',

    # Reasoning errors
    'ReasoningError',
    'DeductiveReasoningError',
    'InductiveReasoningError',
    'AbductiveReasoningError',
    'LogicalInconsistencyError',
    'ChainOfThoughtError',
    'MetacognitionError',
    'ParallelReasoningError',
    'ReasoningTimeoutError'
]