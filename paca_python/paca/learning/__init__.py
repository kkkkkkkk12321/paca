"""
PACA Learning Module
자율학습 및 적응형 AI 시스템

Phase 2 추가 기능:
- IIS 점수 계산 시스템
- 자율 훈련 시스템
- 전술/휴리스틱 자동 생성
"""

# Phase 1 기존 시스템
from .auto.engine import AutoLearningSystem
from .auto.types import (
    LearningPoint, LearningPattern, LearningStatus,
    GeneratedTactic, GeneratedHeuristic, GeneratedKnowledge
)
from .patterns.detector import PatternDetector
from .patterns.analyzer import PatternAnalyzer
from .memory.storage import LearningMemory

# Phase 2 새로운 시스템
from .iis_calculator import (
    IISCalculator, IISScore, IISBreakdown, LearningData,
    InteractionResult, TrendType, create_sample_learning_data,
    create_sample_interaction_result
)
from .autonomous_trainer import (
    AutonomousTrainer, WeaknessArea, TrainingMission, TrainingResult,
    TrainingSession, TrainingConfig, WeaknessType, TrainingType,
    create_sample_training_session, create_sample_weakness_area
)
from .tactic_generator import (
    TacticGenerator, Tactic, Heuristic, TacticUsageRecord,
    LearningSnapshot, TacticType, TacticStatus, HeuristicType,
    create_sample_interaction_data, create_sample_tactic
)

__all__ = [
    # Phase 1 기존 exports
    "AutoLearningSystem",
    "LearningPoint",
    "LearningPattern",
    "LearningStatus",
    "GeneratedTactic",
    "GeneratedHeuristic",
    "GeneratedKnowledge",
    "PatternDetector",
    "PatternAnalyzer",
    "LearningMemory",

    # Phase 2 새로운 exports
    # IIS Calculator
    "IISCalculator",
    "IISScore",
    "IISBreakdown",
    "LearningData",
    "InteractionResult",
    "TrendType",
    "create_sample_learning_data",
    "create_sample_interaction_result",

    # Autonomous Trainer
    "AutonomousTrainer",
    "WeaknessArea",
    "TrainingMission",
    "TrainingResult",
    "TrainingSession",
    "TrainingConfig",
    "WeaknessType",
    "TrainingType",
    "create_sample_training_session",
    "create_sample_weakness_area",

    # Tactic Generator
    "TacticGenerator",
    "Tactic",
    "Heuristic",
    "TacticUsageRecord",
    "LearningSnapshot",
    "TacticType",
    "TacticStatus",
    "HeuristicType",
    "create_sample_interaction_data",
    "create_sample_tactic"
]