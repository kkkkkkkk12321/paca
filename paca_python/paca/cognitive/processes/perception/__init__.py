"""
Perception Module

지각 시스템과 관련된 모든 컴포넌트들을 제공합니다.
감각 데이터 처리, 패턴 인식, 개념 형성 등의 기능을 포함합니다.
"""

from .perception_engine import (
    PerceptionEngine,
    PerceptionConfig,
    PerceptionResult,
    PerceptionState,
    SensoryInput,
    ProcessingMode,
    create_perception_engine
)

from .pattern_recognizer import (
    PatternRecognizer,
    Pattern,
    PatternType,
    PatternMatchResult,
    RecognitionConfig,
    create_pattern_recognizer
)

from .concept_former import (
    ConceptFormer,
    Concept,
    ConceptType,
    ConceptFormationResult,
    AbstractionLevel,
    create_concept_former
)

from .sensory_processor import (
    SensoryProcessor,
    SensoryModality,
    SensoryData,
    ProcessingStage,
    SensoryResult,
    create_sensory_processor
)

__all__ = [
    # Perception Engine
    'PerceptionEngine',
    'PerceptionConfig',
    'PerceptionResult',
    'PerceptionState',
    'SensoryInput',
    'ProcessingMode',
    'create_perception_engine',

    # Pattern Recognizer
    'PatternRecognizer',
    'Pattern',
    'PatternType',
    'PatternMatchResult',
    'RecognitionConfig',
    'create_pattern_recognizer',

    # Concept Former
    'ConceptFormer',
    'Concept',
    'ConceptType',
    'ConceptFormationResult',
    'AbstractionLevel',
    'create_concept_former',

    # Sensory Processor
    'SensoryProcessor',
    'SensoryModality',
    'SensoryData',
    'ProcessingStage',
    'SensoryResult',
    'create_sensory_processor'
]