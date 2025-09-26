"""
PACA v5 Python Edition
Personal Adaptive Cognitive Assistant v5

인간 유사 인지 처리 시스템으로 ACT-R, SOAR 기반 하이브리드 아키텍처를 통해
추론, 기억, 학습을 통합 처리하는 개인화된 적응형 인지 어시스턴트입니다.
TypeScript 원본을 Python으로 완전 변환하여 한국어 자연어 처리 기능을 강화하였습니다.
"""

__version__ = "5.0.0"
__author__ = "PACA Development Team"
__description__ = "Personal Adaptive Cognitive Assistant v5 - Python Edition"

# Core imports for easy access
from .core.types import Result, Status, Priority, LogLevel
from .core.events import EventBus, EventEmitter
from .core.errors import PacaError, CognitiveError, ReasoningError
from .cognitive import CognitiveSystem, BaseCognitiveProcessor
from .reasoning import ReasoningEngine, ReasoningType
from .mathematics import Calculator, StatisticalAnalyzer
from .system import PacaSystem, PacaConfig, Message

__all__ = [
    # Core types
    "Result",
    "Status",
    "Priority",
    "LogLevel",

    # Events
    "EventBus",
    "EventEmitter",

    # Errors
    "PacaError",
    "CognitiveError",
    "ReasoningError",

    # Main systems
    "CognitiveSystem",
    "BaseCognitiveProcessor",
    "ReasoningEngine",
    "ReasoningType",
    "Calculator",
    "StatisticalAnalyzer",

    # Integrated system
    "PacaSystem",
    "PacaConfig",
    "Message"
]