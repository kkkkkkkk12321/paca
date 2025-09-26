"""
Intellectual Integrity Module
지적 무결성 모듈

Phase 2.3 구현 - 지적 무결성 점수 (IIS) 시스템
"""

from .integrity_scoring import (
    IntegrityScoring,
    IntegrityDimension,
    BehaviorType,
    IntegrityAction,
    IntegrityMetrics,
    IntegrityReward,
    IntegrityPenalty
)

__all__ = [
    'IntegrityScoring',
    'IntegrityDimension',
    'BehaviorType',
    'IntegrityAction',
    'IntegrityMetrics',
    'IntegrityReward',
    'IntegrityPenalty'
]