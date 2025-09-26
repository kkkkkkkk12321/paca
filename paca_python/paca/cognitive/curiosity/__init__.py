"""
PACA Curiosity Engine - Limited Curiosity System

This module implements a controlled curiosity system that generates autonomous
exploration drives while maintaining alignment with user missions and values.

The curiosity engine is designed with intentional limitations to prevent
runaway autonomous behavior while still enabling meaningful exploration
and learning within defined boundaries.
"""

from .curiosity_engine import (
    CuriosityEngine,
    CuriosityLevel,
    ExplorationFocus,
    CuriosityConfig
)

from .gap_detector import (
    GapDetector,
    LogicalGap,
    GapType,
    GapSeverity,
    CausalChain,
    GapAnalysis
)

from .exploration_planner import (
    ExplorationPlanner,
    ExplorationPlan,
    ExplorationStrategy,
    ExplorationScope,
    ExplorationResource
)

from .mission_aligner import (
    MissionAligner,
    MissionAlignment,
    AlignmentScore,
    AlignmentCheck,
    ValueConflict
)

__all__ = [
    'CuriosityEngine',
    'CuriosityLevel',
    'ExplorationFocus',
    'CuriosityConfig',
    'GapDetector',
    'LogicalGap',
    'GapType',
    'GapSeverity',
    'CausalChain',
    'GapAnalysis',
    'ExplorationPlanner',
    'ExplorationPlan',
    'ExplorationStrategy',
    'ExplorationScope',
    'ExplorationResource',
    'MissionAligner',
    'MissionAlignment',
    'AlignmentScore',
    'AlignmentCheck',
    'ValueConflict'
]

__version__ = "1.0.0"
__author__ = "PACA Development Team"