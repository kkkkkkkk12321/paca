"""
Cognitive Reflection Module
자기 성찰 시스템 - Phase 2 LLM 통합

이 모듈은 PACA의 자기 성찰 능력을 구현합니다:
- 1차 답변 생성 후 즉시 비평적 검토
- 논리적 허점 및 반론 검토
- 결론 개선 및 다듬기 시스템
- 성찰 기반 학습 및 개선

Phase 2 추가:
- LLM 기반 자기 성찰 루프
- 다차원 응답 품질 분석
- 반복적 개선 메커니즘
"""

# Phase 2 LLM 기반 자기 성찰 시스템 (우선)
from .base import (
    ReflectionResult as LLMReflectionResult,
    CritiqueResult,
    Weakness,
    Improvement,
    ReflectionConfig,
    ReflectionLevel,
    WeaknessType,
    ImprovementType,
    create_reflection_config,
    calculate_overall_quality
)

from .processor import SelfReflectionProcessor
from .critique import CritiqueAnalyzer
from .improvement import IterativeImprover

# 기존 시스템 (호환성 유지)
try:
    from .self_reflection import (
        SelfReflection,
        ReflectionType,
        ReflectionResult,
        ReflectionCycle,
        ReflectionSession
    )
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False

__all__ = [
    # Phase 2 LLM 기반 시스템 (메인)
    'SelfReflectionProcessor',
    'CritiqueAnalyzer',
    'IterativeImprover',
    'LLMReflectionResult',
    'CritiqueResult',
    'Weakness',
    'Improvement',
    'ReflectionConfig',
    'ReflectionLevel',
    'WeaknessType',
    'ImprovementType',
    'create_reflection_config',
    'calculate_overall_quality',
]

# 기존 시스템 (호환성)
if LEGACY_AVAILABLE:
    __all__.extend([
        'SelfReflection',
        'ReflectionType',
        'ReflectionResult',
        'ReflectionCycle',
        'ReflectionSession',
    ])

__version__ = "1.0.0"