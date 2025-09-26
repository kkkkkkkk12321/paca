"""
Reasoning Module - Entry Point
추론 시스템 모듈 진입점
"""

from .base import (
    ReasoningType,
    InferenceRule,
    ReasoningStep,
    ReasoningContext,
    ReasoningResult,
    BaseReasoningEngine,
    DeductiveReasoningEngine,
    ReasoningEngine
)

# 체인 추론 모듈
from .chains import (
    ReasoningStepType,
    ReasoningMethod,
    ReasoningStep as ChainReasoningStep,
    SubGoal,
    ReasoningChain,
    ValidationResult,
    ReasoningChainResult,
    ReasoningChainManager
)

# 공통 타입들
from dataclasses import dataclass
from typing import List

@dataclass
class ReasoningCapability:
    """추론 능력"""
    name: str
    description: str
    supported: bool
    version: str

@dataclass
class ReasoningConfiguration:
    """추론 설정"""
    default_strategy: str
    max_concurrency: int
    timeout: int
    enable_chaining: bool
    enable_parallel: bool
    enable_metacognition: bool

# 추론 능력 정의
REASONING_CAPABILITIES: List[ReasoningCapability] = [
    ReasoningCapability(
        name="chain_reasoning",
        description="Sequential step-by-step reasoning",
        supported=True,
        version="1.0.0"
    ),
    ReasoningCapability(
        name="parallel_reasoning",
        description="Concurrent reasoning processes",
        supported=True,
        version="1.0.0"
    ),
    ReasoningCapability(
        name="metacognitive_reasoning",
        description="Self-aware reasoning monitoring",
        supported=True,
        version="1.0.0"
    )
]

# 기본 추론 설정
DEFAULT_REASONING_CONFIG = ReasoningConfiguration(
    default_strategy="chain_reasoning",
    max_concurrency=4,
    timeout=30000,
    enable_chaining=True,
    enable_parallel=True,
    enable_metacognition=True
)

__all__ = [
    # Base reasoning
    'ReasoningType',
    'InferenceRule',
    'ReasoningStep',
    'ReasoningContext',
    'ReasoningResult',
    'BaseReasoningEngine',
    'DeductiveReasoningEngine',
    'ReasoningEngine',
    # Chains module
    "ReasoningStepType",
    "ReasoningMethod",
    "ChainReasoningStep",
    "SubGoal",
    "ReasoningChain",
    "ValidationResult",
    "ReasoningChainResult",
    "ReasoningChainManager",
    # Configuration
    "ReasoningCapability",
    "ReasoningConfiguration",
    "REASONING_CAPABILITIES",
    "DEFAULT_REASONING_CONFIG"
]