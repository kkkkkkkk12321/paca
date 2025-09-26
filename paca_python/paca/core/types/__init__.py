"""
Core Types Module
핵심 타입 정의 및 기본 인터페이스들
"""

from .base import (
    ID,
    Timestamp,
    Result,
    KeyValuePair,
    LogLevel,
    Status,
    Priority,
    BaseConfig,
    BaseEntity,
    BaseEvent,
    PaginationRequest,
    PaginationResponse,
    BaseFilter,
    BaseStats,
    current_timestamp,
    create_id,
    generate_id,
    create_success,
    create_failure,
    create_result
)

# Import from types.py (backward compatibility)
try:
    from ..types import CognitiveState
except ImportError:
    from enum import Enum

    class CognitiveState(Enum):
        """인지 시스템의 현재 상태"""
        IDLE = "idle"
        PROCESSING = "processing"
        LEARNING = "learning"
        REASONING = "reasoning"
        ERROR = "error"

# Import learning types for convenience
try:
    from ..learning.auto.types import LearningPoint
except ImportError:
    # Define a placeholder if learning module is not available
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class LearningPoint:
        user_message: str = ""
        paca_response: str = ""
        context: str = ""
        confidence: float = 0.0
        extracted_knowledge: str = ""

__all__ = [
    'ID',
    'Timestamp',
    'Result',
    'KeyValuePair',
    'LogLevel',
    'Status',
    'Priority',
    'BaseConfig',
    'BaseEntity',
    'BaseEvent',
    'PaginationRequest',
    'PaginationResponse',
    'BaseFilter',
    'BaseStats',
    'current_timestamp',
    'create_id',
    'generate_id',
    'create_success',
    'create_failure',
    'create_result',
    'LearningPoint',
    'CognitiveState'
]