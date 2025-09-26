"""
휴면기 통합 시스템 (DormantIntegration)

대화 기록 정리, 약한 연결 제거, 패턴 발견 및 강화, 장기 기억 공고화를 담당합니다.
"""

from .dormant_integration import DormantIntegration
from .memory_consolidator import MemoryConsolidator
from .pattern_strengthener import PatternStrengthener
from .weak_connection_pruner import WeakConnectionPruner

__all__ = [
    'DormantIntegration',
    'MemoryConsolidator',
    'PatternStrengthener',
    'WeakConnectionPruner'
]