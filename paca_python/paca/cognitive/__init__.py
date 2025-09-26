"""
Cognitive Module
인지 시스템과 관련된 모든 컴포넌트들
"""

# Base cognitive components
from .base import (
    CognitiveState,
    CognitiveTaskType,
    CognitiveConfig,
    CognitiveContext,
    CognitiveResult,
    ProcessingStep,
    MemoryUpdate,
    LearningData,
    ResourceUsage,
    QualityMetrics,
    CognitiveModel,
    MemorySystem,
    BaseCognitiveProcessor,
    SimpleCognitiveProcessor,
    CognitiveStatistics,
    CognitiveSystem,
    create_cognitive_context,
    create_processing_step,
    create_default_cognitive_processor
)

# Cognitive models
from .models import (
    BaseCognitiveModel,
    ModelState,
    PerformanceMetrics,
    CognitiveArchitecture,
    ValidationResult,
    ACTRModel,
    ACTRParameters,
    SOARModel,
    SOARParameters,
    create_actr_model,
    create_soar_model
)

# Memory systems
from .memory import (
    MemoryType,
    MemoryOperation,
    MemoryConfiguration,
    MemoryMetrics,
    MemoryItem,
    SearchQuery,
    SearchResult,
    WorkingMemory,
    EpisodicMemory,
    LongTermMemory
)

# Complexity Detection System (PACA v5 핵심 기능)
from .complexity_detector import (
    ComplexityDetector,
    ComplexityResult,
    ComplexityMetrics,
    ComplexityFeatureSet,
    ComplexityLevel,
    DomainType,
    detect_complexity,
    create_complexity_detector
)

# Metacognition Engine (PACA v5 핵심 기능)
from .metacognition_engine import (
    MetacognitionEngine,
    QualityMetrics,
    QualityAssessment,
    ReasoningStep as MetaReasoningStep,
    MonitoringSession,
    SelfReflectionResult,
    MonitoringPhase,
    ReasoningQuality,
    QualityLevel,
    create_metacognition_engine,
    monitor_reasoning_session
)

# Reasoning Chain System (PACA v5 핵심 기능)
from .reasoning_chain import (
    ReasoningChain,
    ReasoningResult,
    ReasoningStep,
    ReasoningStrategy,
    StepType,
    ReasoningStatus,
    BacktrackPoint,
    execute_reasoning,
    create_reasoning_chain
)

# Self-Reflection System (Phase 2.1 - 완료됨)
from .reflection import (
    SelfReflectionProcessor,
    CritiqueAnalyzer,
    IterativeImprover,
    ReflectionResult,
    CritiqueResult,
    ReflectionConfig,
    ReflectionLevel
)

# Truth Seeking System (Phase 2.2 - 새로 구현됨)
from .truth import (
    TruthSeeker,
    UncertaintyType,
    UncertaintyDetection,
    TruthSeekingRequest,
    TruthSeekingResult
)

# Intellectual Integrity System (Phase 2.3 - 새로 구현됨)
from .integrity import (
    IntegrityScoring,
    IntegrityDimension,
    BehaviorType,
    IntegrityAction,
    IntegrityMetrics,
    IntegrityReward,
    IntegrityPenalty
)

__all__ = [
    # Base cognitive components
    'CognitiveState',
    'CognitiveTaskType',
    'CognitiveConfig',
    'CognitiveContext',
    'CognitiveResult',
    'ProcessingStep',
    'MemoryUpdate',
    'LearningData',
    'ResourceUsage',
    'QualityMetrics',
    'CognitiveModel',
    'MemorySystem',
    'BaseCognitiveProcessor',
    'SimpleCognitiveProcessor',
    'CognitiveStatistics',
    'CognitiveSystem',
    'create_cognitive_context',
    'create_processing_step',
    'create_default_cognitive_processor',

    # Cognitive models
    'BaseCognitiveModel',
    'ModelState',
    'PerformanceMetrics',
    'CognitiveArchitecture',
    'ValidationResult',
    'ACTRModel',
    'ACTRParameters',
    'SOARModel',
    'SOARParameters',
    'create_actr_model',
    'create_soar_model',

    # Memory systems
    'MemoryType',
    'MemoryOperation',
    'MemoryConfiguration',
    'MemoryMetrics',
    'MemoryItem',
    'SearchQuery',
    'SearchResult',
    'WorkingMemory',
    'EpisodicMemory',
    'LongTermMemory',

    # Complexity Detection System
    'ComplexityDetector',
    'ComplexityResult',
    'ComplexityMetrics',
    'ComplexityFeatureSet',
    'ComplexityLevel',
    'DomainType',
    'detect_complexity',
    'create_complexity_detector',

    # Metacognition Engine
    'MetacognitionEngine',
    'QualityMetrics',
    'QualityAssessment',
    'QualityLevel',
    'MetaReasoningStep',
    'MonitoringSession',
    'SelfReflectionResult',
    'MonitoringPhase',
    'ReasoningQuality',
    'create_metacognition_engine',
    'monitor_reasoning_session',

    # Reasoning Chain System
    'ReasoningChain',
    'ReasoningResult',
    'ReasoningStep',
    'ReasoningStrategy',
    'StepType',
    'ReasoningStatus',
    'BacktrackPoint',
    'execute_reasoning',
    'create_reasoning_chain',

    # Self-Reflection System (Phase 2.1)
    'SelfReflectionProcessor',
    'CritiqueAnalyzer',
    'IterativeImprover',
    'ReflectionResult',
    'CritiqueResult',
    'ReflectionConfig',
    'ReflectionLevel',

    # Truth Seeking System (Phase 2.2)
    'TruthSeeker',
    'UncertaintyType',
    'UncertaintyDetection',
    'TruthSeekingRequest',
    'TruthSeekingResult',

    # Intellectual Integrity System (Phase 2.3)
    'IntegrityScoring',
    'IntegrityDimension',
    'BehaviorType',
    'IntegrityAction',
    'IntegrityMetrics',
    'IntegrityReward',
    'IntegrityPenalty'
]
