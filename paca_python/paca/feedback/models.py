"""
피드백 시스템 데이터 모델
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field


class FeedbackType(Enum):
    """피드백 유형"""
    TOOL_EXECUTION = "tool_execution"
    REASONING_QUALITY = "reasoning_quality"
    RESPONSE_ACCURACY = "response_accuracy"
    PERFORMANCE = "performance"
    USER_EXPERIENCE = "user_experience"
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"
    GENERAL = "general"


class FeedbackStatus(Enum):
    """피드백 상태"""
    PENDING = "pending"
    REVIEWED = "reviewed"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class SentimentScore(Enum):
    """감정 점수"""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


@dataclass
class UserSession:
    """사용자 세션 정보"""
    session_id: str
    user_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_interactions: int = 0
    successful_interactions: int = 0
    failed_interactions: int = 0
    tools_used: List[str] = field(default_factory=list)
    average_response_time: float = 0.0
    user_satisfaction: Optional[float] = None  # 1.0 - 5.0 scale


@dataclass
class FeedbackContext:
    """피드백 컨텍스트 정보"""
    session_id: str
    step_id: Optional[str] = None
    tool_name: Optional[str] = None
    action_type: Optional[str] = None
    execution_time: Optional[float] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None
    user_query: Optional[str] = None
    system_response: Optional[str] = None


class FeedbackModel(BaseModel):
    """피드백 모델"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    feedback_type: FeedbackType
    status: FeedbackStatus = FeedbackStatus.PENDING

    # 피드백 내용
    rating: Optional[int] = Field(None, ge=1, le=5, description="1-5점 평가")
    text_feedback: Optional[str] = Field(None, description="텍스트 피드백")
    sentiment_score: Optional[SentimentScore] = Field(None, description="감정 분석 결과")

    # 컨텍스트 정보
    context: Optional[FeedbackContext] = Field(None, description="피드백 컨텍스트")

    # 메타데이터
    user_id: Optional[str] = Field(None, description="사용자 ID")
    session_id: str = Field(..., description="세션 ID")
    ip_address: Optional[str] = Field(None, description="IP 주소")
    user_agent: Optional[str] = Field(None, description="User Agent")

    # 처리 정보
    reviewed_by: Optional[str] = Field(None, description="검토자")
    reviewed_at: Optional[datetime] = Field(None, description="검토 시간")
    resolution_notes: Optional[str] = Field(None, description="해결 노트")

    # 추가 데이터
    metadata: Dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터")
    tags: List[str] = Field(default_factory=list, description="태그")

    class Config:
        use_enum_values = True


class FeedbackStats(BaseModel):
    """피드백 통계"""
    total_feedback: int = 0
    feedback_by_type: Dict[str, int] = Field(default_factory=dict)
    feedback_by_status: Dict[str, int] = Field(default_factory=dict)
    average_rating: Optional[float] = None
    sentiment_distribution: Dict[str, int] = Field(default_factory=dict)

    # 시간별 통계
    feedback_by_hour: Dict[str, int] = Field(default_factory=dict)
    feedback_by_day: Dict[str, int] = Field(default_factory=dict)

    # 성능 관련
    average_response_time: Optional[float] = None
    success_rate: Optional[float] = None

    # 트렌드
    rating_trend: List[Dict[str, Any]] = Field(default_factory=list)
    volume_trend: List[Dict[str, Any]] = Field(default_factory=list)


class FeedbackAnalysis(BaseModel):
    """피드백 분석 결과"""
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    period_start: datetime
    period_end: datetime

    # 기본 통계
    stats: FeedbackStats

    # 인사이트
    key_insights: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)
    critical_issues: List[str] = Field(default_factory=list)

    # 카테고리별 분석
    tool_performance: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    common_complaints: List[Dict[str, Any]] = Field(default_factory=list)
    positive_highlights: List[Dict[str, Any]] = Field(default_factory=list)

    # 예측
    trend_predictions: Dict[str, Any] = Field(default_factory=dict)
    recommended_actions: List[Dict[str, str]] = Field(default_factory=list)


# API 요청/응답 모델들
class CreateFeedbackRequest(BaseModel):
    """피드백 생성 요청"""
    feedback_type: FeedbackType
    rating: Optional[int] = Field(None, ge=1, le=5)
    text_feedback: Optional[str] = None
    session_id: str
    context: Optional[Dict[str, Any]] = None
    tags: List[str] = Field(default_factory=list)


class FeedbackResponse(BaseModel):
    """피드백 응답"""
    success: bool
    feedback_id: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)


class FeedbackListResponse(BaseModel):
    """피드백 목록 응답"""
    feedback_list: List[FeedbackModel]
    total_count: int
    page: int
    page_size: int
    has_next: bool


class FeedbackStatsResponse(BaseModel):
    """피드백 통계 응답"""
    stats: FeedbackStats
    period_start: datetime
    period_end: datetime
    generated_at: datetime = Field(default_factory=datetime.now)