"""
Auto Learning Types
자동 학습 시스템의 타입 정의
TypeScript 인터페이스의 Python 변환
"""

from typing import List, Dict, Any, Optional, Protocol
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid

from ...core.types import ID, Timestamp, BaseEntity


class LearningCategory(Enum):
    """학습 카테고리"""
    SUCCESS_PATTERN = "success_pattern"
    ERROR_PATTERN = "error_pattern"
    USER_PREFERENCE = "user_preference"
    KNOWLEDGE_GAP = "knowledge_gap"
    PERFORMANCE_ISSUE = "performance_issue"
    OPTIMIZATION = "optimization"


class PatternType(Enum):
    """패턴 타입"""
    SUCCESS = "success"
    FAILURE = "failure"
    PREFERENCE = "preference"
    KNOWLEDGE = "knowledge"
    BEHAVIOR = "behavior"
    PERFORMANCE = "performance"


@dataclass
class LearningPattern:
    """학습 패턴 정의"""
    pattern_type: PatternType
    keywords: List[str]
    context_indicators: List[str]
    confidence_threshold: float
    extraction_rule: str
    weight: float = 1.0
    language: str = "ko"  # 언어 코드 (ko=한국어, en=영어)

    def __post_init__(self):
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")


@dataclass
class LearningPoint:
    """학습 포인트 데이터"""
    user_message: str
    paca_response: str
    context: str
    category: LearningCategory
    confidence: float
    extracted_knowledge: str
    conversation_id: Optional[str] = None
    source_pattern: Optional[str] = None
    effectiveness_score: float = 0.0
    usage_count: int = 0
    last_used: Optional[Timestamp] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: Timestamp = field(default_factory=time.time)
    updated_at: Timestamp = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not 0.0 <= self.effectiveness_score <= 1.0:
            raise ValueError("Effectiveness score must be between 0.0 and 1.0")

    def update_timestamp(self) -> None:
        """업데이트 시간 갱신"""
        self.updated_at = time.time()

    def use(self) -> None:
        """사용 기록 업데이트"""
        self.usage_count += 1
        self.last_used = time.time()
        self.update_timestamp()


@dataclass
class GeneratedTactic:
    """자동 생성된 전술"""
    name: str
    description: str
    context: str
    category: str = "auto_generated"
    success_count: int = 0
    total_applications: int = 0
    effectiveness: float = 0.0
    last_used: Optional[Timestamp] = None
    source_conversations: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: Timestamp = field(default_factory=time.time)
    updated_at: Timestamp = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_timestamp(self) -> None:
        """업데이트 시간 갱신"""
        self.updated_at = time.time()

    @property
    def success_rate(self) -> float:
        """성공률 계산"""
        if self.total_applications == 0:
            return 0.0
        return self.success_count / self.total_applications

    def apply(self, success: bool = True) -> None:
        """전술 적용 기록"""
        self.total_applications += 1
        if success:
            self.success_count += 1
        self.effectiveness = self.success_rate
        self.last_used = time.time()
        self.update_timestamp()


@dataclass
class GeneratedHeuristic:
    """자동 생성된 휴리스틱"""
    pattern: str
    avoidance_rule: str
    context: str
    category: str = "auto_generated"
    triggered_count: int = 0
    avoided_count: int = 0
    effectiveness: float = 0.0
    last_triggered: Optional[Timestamp] = None
    source_conversations: List[str] = field(default_factory=list)
    severity: str = "medium"  # low, medium, high, critical
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: Timestamp = field(default_factory=time.time)
    updated_at: Timestamp = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_timestamp(self) -> None:
        """업데이트 시간 갱신"""
        self.updated_at = time.time()

    @property
    def avoidance_rate(self) -> float:
        """회피율 계산"""
        if self.triggered_count == 0:
            return 0.0
        return self.avoided_count / self.triggered_count

    def trigger(self, avoided: bool = True) -> None:
        """휴리스틱 트리거 기록"""
        self.triggered_count += 1
        if avoided:
            self.avoided_count += 1
        self.effectiveness = self.avoidance_rate
        self.last_triggered = time.time()
        self.update_timestamp()


@dataclass
class LearningStatus:
    """학습 시스템 상태"""
    learning_points: int
    generated_tactics: int
    generated_heuristics: int
    recent_learning: int
    total_conversations_analyzed: int = 0
    average_confidence: float = 0.0
    last_learning_at: Optional[Timestamp] = None
    active_patterns: int = 0

    @classmethod
    def create_empty(cls) -> 'LearningStatus':
        """빈 상태 생성"""
        return cls(
            learning_points=0,
            generated_tactics=0,
            generated_heuristics=0,
            recent_learning=0
        )


@dataclass
class GeneratedKnowledge:
    """생성된 지식 요약"""
    tactics: List[Dict[str, Any]]
    heuristics: List[Dict[str, Any]]
    total_knowledge_items: int = 0
    creation_date: str = ""
    effectiveness_summary: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.total_knowledge_items = len(self.tactics) + len(self.heuristics)
        if not self.creation_date:
            self.creation_date = time.strftime("%Y-%m-%d %H:%M:%S")


# Protocol definitions for dependency injection
class DatabaseInterface(Protocol):
    """데이터베이스 인터페이스"""

    def add_experience(self, name: str, description: str) -> bool:
        """경험 추가"""
        ...

    def add_heuristic(self, rule: str) -> bool:
        """휴리스틱 추가"""
        ...

    def get_experiences(self, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """경험 조회"""
        ...

    def get_heuristics(self, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """휴리스틱 조회"""
        ...


class ConversationMemoryInterface(Protocol):
    """대화 메모리 인터페이스"""

    def get_recent_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """최근 대화 조회"""
        ...

    def get_conversation_context(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """대화 컨텍스트 조회"""
        ...

    def store_learning_point(self, learning_point: LearningPoint) -> bool:
        """학습 포인트 저장"""
        ...


# Korean NLP specific types
@dataclass
class KoreanAnalysisResult:
    """한국어 분석 결과"""
    morphs: List[str]  # 형태소
    pos_tags: List[tuple]  # 품사 태그
    nouns: List[str]  # 명사
    verbs: List[str]  # 동사
    adjectives: List[str]  # 형용사
    sentiment_score: float  # 감정 점수 (-1.0 ~ 1.0)
    confidence: float  # 분석 신뢰도

    @property
    def positive_sentiment(self) -> bool:
        """긍정적 감정 여부"""
        return self.sentiment_score > 0.1

    @property
    def negative_sentiment(self) -> bool:
        """부정적 감정 여부"""
        return self.sentiment_score < -0.1


@dataclass
class LearningMetrics:
    """학습 메트릭"""
    total_learning_points: int = 0
    successful_patterns: int = 0
    failed_patterns: int = 0
    average_confidence: float = 0.0
    learning_rate: float = 0.0  # 시간당 학습 포인트 수
    knowledge_retention: float = 1.0  # 지식 보존율
    adaptation_speed: float = 0.0  # 적응 속도

    @property
    def success_rate(self) -> float:
        """패턴 성공률"""
        total = self.successful_patterns + self.failed_patterns
        if total == 0:
            return 0.0
        return self.successful_patterns / total

    def update_metrics(self, new_learning_point: LearningPoint) -> None:
        """새 학습 포인트로 메트릭 업데이트"""
        self.total_learning_points += 1

        # 평균 신뢰도 업데이트
        old_avg = self.average_confidence
        n = self.total_learning_points
        self.average_confidence = ((n - 1) * old_avg + new_learning_point.confidence) / n