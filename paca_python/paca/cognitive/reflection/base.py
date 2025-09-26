"""
Self-Reflection Base Types
자기 성찰 시스템의 기본 타입과 인터페이스 정의
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime

from ...core.types import ID, Timestamp, Result, create_id, current_timestamp


class ReflectionLevel(Enum):
    """성찰 수준"""
    BASIC = "basic"          # 기본적인 문법/구조 검토
    MODERATE = "moderate"    # 논리적 일관성 검토
    DEEP = "deep"           # 깊은 메타인지적 분석
    CRITICAL = "critical"    # 비판적 사고 적용


class WeaknessType(Enum):
    """약점 유형"""
    LOGICAL_INCONSISTENCY = "logical_inconsistency"    # 논리적 불일치
    FACTUAL_ERROR = "factual_error"                   # 사실 오류
    INCOMPLETE_RESPONSE = "incomplete_response"        # 불완전한 응답
    IRRELEVANT_CONTENT = "irrelevant_content"         # 관련성 부족
    UNCLEAR_EXPRESSION = "unclear_expression"         # 불명확한 표현
    OVERCONFIDENCE = "overconfidence"                 # 과신
    UNDERCONFIDENCE = "underconfidence"               # 과소신


class ImprovementType(Enum):
    """개선 유형"""
    CLARIFICATION = "clarification"           # 명확화
    ELABORATION = "elaboration"              # 상세화
    CORRECTION = "correction"                # 정정
    RESTRUCTURING = "restructuring"          # 재구성
    EVIDENCE_ADDITION = "evidence_addition"  # 근거 추가
    TONE_ADJUSTMENT = "tone_adjustment"      # 어조 조정


@dataclass
class Weakness:
    """식별된 약점"""
    type: WeaknessType
    description: str
    location: str  # 약점이 발견된 위치 (예: "paragraph 2")
    severity: float  # 0.0-1.0, 심각도
    confidence: float  # 0.0-1.0, 탐지 신뢰도
    id: ID = field(default_factory=create_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "description": self.description,
            "location": self.location,
            "severity": self.severity,
            "confidence": self.confidence
        }


@dataclass
class Improvement:
    """제안된 개선사항"""
    type: ImprovementType
    description: str
    suggestion: str  # 구체적인 개선 제안
    priority: float  # 0.0-1.0, 우선순위
    estimated_impact: float  # 0.0-1.0, 예상 개선 효과
    weakness_id: Optional[ID] = None  # 연관된 약점 ID
    id: ID = field(default_factory=create_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "weakness_id": self.weakness_id,
            "type": self.type.value,
            "description": self.description,
            "suggestion": self.suggestion,
            "priority": self.priority,
            "estimated_impact": self.estimated_impact
        }


@dataclass
class CritiqueResult:
    """비평 분석 결과"""
    # 전체 평가 (필수 필드)
    overall_quality_score: float  # 0.0-100.0
    needs_improvement: bool

    # 세부 평가 (필수 필드)
    logical_consistency: float  # 0.0-100.0
    factual_accuracy: float    # 0.0-100.0
    completeness: float        # 0.0-100.0
    relevance: float          # 0.0-100.0
    clarity: float            # 0.0-100.0

    # 선택적 필드 (기본값 있음)
    id: ID = field(default_factory=create_id)
    timestamp: Timestamp = field(default_factory=current_timestamp)
    weaknesses: List[Weakness] = field(default_factory=list)
    improvements: List[Improvement] = field(default_factory=list)
    reflection_level: ReflectionLevel = ReflectionLevel.MODERATE
    processing_time: float = 0.0
    critique_reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "overall_quality_score": self.overall_quality_score,
            "needs_improvement": self.needs_improvement,
            "logical_consistency": self.logical_consistency,
            "factual_accuracy": self.factual_accuracy,
            "completeness": self.completeness,
            "relevance": self.relevance,
            "clarity": self.clarity,
            "weaknesses": [w.to_dict() for w in self.weaknesses],
            "improvements": [i.to_dict() for i in self.improvements],
            "reflection_level": self.reflection_level.value,
            "processing_time": self.processing_time,
            "critique_reasoning": self.critique_reasoning
        }


@dataclass
class ReflectionResult:
    """전체 성찰 처리 결과"""
    # 입력과 출력 (필수 필드)
    user_input: str
    initial_response: str
    final_response: str

    # 성찰 과정 (필수 필드)
    critique: CritiqueResult
    improvement_applied: bool
    iterations_performed: int

    # 선택적 필드 (기본값 있음)
    id: ID = field(default_factory=create_id)
    timestamp: Timestamp = field(default_factory=current_timestamp)
    total_processing_time: float = 0.0
    quality_improvement: float = 0.0  # 초기 응답 대비 개선 정도
    config_used: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "user_input": self.user_input,
            "initial_response": self.initial_response,
            "final_response": self.final_response,
            "critique": self.critique.to_dict(),
            "improvement_applied": self.improvement_applied,
            "iterations_performed": self.iterations_performed,
            "total_processing_time": self.total_processing_time,
            "quality_improvement": self.quality_improvement,
            "config_used": self.config_used
        }


@dataclass
class ReflectionConfig:
    """자기 성찰 설정"""

    # 성찰 수준 설정
    reflection_level: ReflectionLevel = ReflectionLevel.MODERATE
    quality_threshold: float = 85.0  # 품질 임계값
    max_iterations: int = 3

    # 시간 제한
    max_processing_time: float = 30.0  # 초
    critique_timeout: float = 10.0
    improvement_timeout: float = 15.0

    # 개선 기준
    min_improvement_threshold: float = 5.0  # 최소 개선 정도
    enable_iterative_improvement: bool = True

    # 약점 탐지 민감도
    weakness_detection_threshold: float = 0.6  # 0.0-1.0
    severity_threshold: float = 0.3  # 이 이상의 심각도만 처리

    # 모델 설정
    critique_model_temperature: float = 0.3  # 비평 시 낮은 창의성
    improvement_model_temperature: float = 0.7  # 개선 시 높은 창의성

    # 로깅 및 디버깅
    enable_detailed_logging: bool = True
    save_intermediate_results: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reflection_level": self.reflection_level.value,
            "quality_threshold": self.quality_threshold,
            "max_iterations": self.max_iterations,
            "max_processing_time": self.max_processing_time,
            "critique_timeout": self.critique_timeout,
            "improvement_timeout": self.improvement_timeout,
            "min_improvement_threshold": self.min_improvement_threshold,
            "enable_iterative_improvement": self.enable_iterative_improvement,
            "weakness_detection_threshold": self.weakness_detection_threshold,
            "severity_threshold": self.severity_threshold,
            "critique_model_temperature": self.critique_model_temperature,
            "improvement_model_temperature": self.improvement_model_temperature,
            "enable_detailed_logging": self.enable_detailed_logging,
            "save_intermediate_results": self.save_intermediate_results
        }


# 헬퍼 함수들
def create_reflection_config(
    level: ReflectionLevel = ReflectionLevel.MODERATE,
    quality_threshold: float = 85.0,
    max_iterations: int = 3
) -> ReflectionConfig:
    """기본 설정으로 ReflectionConfig 생성"""
    return ReflectionConfig(
        reflection_level=level,
        quality_threshold=quality_threshold,
        max_iterations=max_iterations
    )


def calculate_overall_quality(
    logical_consistency: float,
    factual_accuracy: float,
    completeness: float,
    relevance: float,
    clarity: float,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """개별 품질 점수들로부터 전체 품질 점수 계산"""
    if weights is None:
        weights = {
            "logical_consistency": 0.25,
            "factual_accuracy": 0.25,
            "completeness": 0.20,
            "relevance": 0.15,
            "clarity": 0.15
        }

    total_score = (
        logical_consistency * weights["logical_consistency"] +
        factual_accuracy * weights["factual_accuracy"] +
        completeness * weights["completeness"] +
        relevance * weights["relevance"] +
        clarity * weights["clarity"]
    )

    return min(100.0, max(0.0, total_score))