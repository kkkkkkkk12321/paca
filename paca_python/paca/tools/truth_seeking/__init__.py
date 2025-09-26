"""
Truth Seeking Tools Module
진실 추구 도구 시스템

이 모듈은 PACA의 진실 추구 능력을 구현합니다:
- 증거 기반 추론 및 평가
- 정보원 신뢰성 검증
- 사실 확인 및 검증 프로세스
- 진실성 평가 및 불확실성 관리
"""

from .evidence_evaluator import (
    EvidenceEvaluator,
    EvidenceType,
    EvidenceQuality,
    EvidenceAssessment,
    CredibilityScore,
    SourceReliability
)

from .source_validator import (
    SourceValidator,
    SourceType,
    ValidationResult,
    CredibilityMetrics,
    SourceProfile,
    ValidationCriteria
)

from .fact_checker import (
    FactChecker,
    FactCheckResult,
    VerificationMethod,
    FactStatus,
    ClaimAnalysis,
    CrossReference
)

from .truth_assessment import (
    TruthAssessment,
    TruthScore,
    UncertaintyMetrics,
    ConfidenceLevel,
    AssessmentReport,
    TruthSeekingEngine
)

__all__ = [
    # Evidence Evaluation
    'EvidenceEvaluator',
    'EvidenceType',
    'EvidenceQuality',
    'EvidenceAssessment',
    'CredibilityScore',
    'SourceReliability',

    # Source Validation
    'SourceValidator',
    'SourceType',
    'ValidationResult',
    'CredibilityMetrics',
    'SourceProfile',
    'ValidationCriteria',

    # Fact Checking
    'FactChecker',
    'FactCheckResult',
    'VerificationMethod',
    'FactStatus',
    'ClaimAnalysis',
    'CrossReference',

    # Truth Assessment
    'TruthAssessment',
    'TruthScore',
    'UncertaintyMetrics',
    'ConfidenceLevel',
    'AssessmentReport',
    'TruthSeekingEngine'
]

__version__ = "1.0.0"