"""
PACA Governance System
거버넌스 시스템 - 4대 핵심 원칙과 3대 프로토콜 구현

이 모듈은 PACA의 철학적 기반을 구현합니다:
- 4대 핵심 원칙 (사용자 주권, 인식론적 겸손, 수용적 태세, 건설적 이의 제기)
- 3대 거버넌스 프로토콜 (모순 수용, 최종 판단 유보, 신뢰-검증-롤백)
- 지적 무결성 모니터링 시스템
- 관계적 항상성 유지 시스템
"""

from .core_principles import (
    CorePrinciples,
    PrincipleType,
    PrincipleViolation,
    UserSovereignty,
    EpistemicHumility,
    ReceptiveStance,
    ConstructiveObjection
)

from .protocols import (
    GovernanceProtocol,
    ContradictionAcceptance,
    FinalJudgmentReservation,
    TrustVerifyRollback,
    ProtocolResult,
    ProtocolStatus
)

from .integrity_monitor import (
    IntegrityMonitor,
    IntegrityViolation,
    IntegrityLevel,
    IntegrityReport
)

from .relationship_health import (
    RelationshipHealth,
    HealthMetric,
    HealthStatus,
    HealthReport
)

__all__ = [
    # Core Principles
    'CorePrinciples',
    'PrincipleType',
    'PrincipleViolation',
    'UserSovereignty',
    'EpistemicHumility',
    'ReceptiveStance',
    'ConstructiveObjection',

    # Protocols
    'GovernanceProtocol',
    'ContradictionAcceptance',
    'FinalJudgmentReservation',
    'TrustVerifyRollback',
    'ProtocolResult',
    'ProtocolStatus',

    # Integrity Monitor
    'IntegrityMonitor',
    'IntegrityViolation',
    'IntegrityLevel',
    'IntegrityReport',

    # Relationship Health
    'RelationshipHealth',
    'HealthMetric',
    'HealthStatus',
    'HealthReport'
]

__version__ = "1.0.0"