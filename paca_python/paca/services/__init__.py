"""
Services Module
서비스 관리 및 실행 시스템
"""

from .base import (
    ServiceStatus,
    ServiceType,
    ServiceConfig,
    ServiceHealth,
    ServiceMetrics,
    ServiceContext,
    ServiceResult,
    ServicePriority,
    BaseService,
    ServiceManager
)

# Specific services
from .auth import (
    AuthenticationService,
    AuthToken,
    AuthSession,
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    PasswordResetRequest,
    User,
    UserProfile,
    UserRole,
    DeviceInfo,
    TokenType
)

from .knowledge import (
    KnowledgeService,
    KnowledgeItem,
    KnowledgeRelationship,
    KnowledgeSearchQuery,
    KnowledgeSearchResult,
    KnowledgeMetadata,
    KnowledgeType,
    RelationshipType
)

from .analytics import (
    AnalyticsService,
    AnalyticsEvent,
    AnalyticsEventType,
    AnalyticsRequest,
    AnalyticsQuery,
    AnalyticsReport,
    UserMetrics,
    SystemMetrics,
    Metric,
    MetricType,
    EventContext
)

from .notification import (
    NotificationService,
    Notification,
    NotificationType,
    NotificationPriority,
    NotificationStatus,
    NotificationRequest,
    NotificationQuery,
    NotificationStats,
    NotificationTemplate,
    NotificationChannel
)

from .learning import (
    LearningService,
    LearningSession,
    LearningSessionState,
    SessionType,
    SessionStatus,
    LearningAction,
    LearningResult,
    LearningQuestion,
    LearningAnswer,
    LearningGoal,
    CreateSessionRequest,
    SubmitAnswerRequest,
    LearningProgress,
    LearningStatistics,
    LearningRecommendation
)

from .memory import (
    MemoryService,
    MemoryItem,
    MemoryType,
    MemoryPriority,
    MemoryStatus,
    MemoryQuery,
    MemoryStatistics,
    ConversationMemory
)

__all__ = [
    # Base service components
    'ServiceStatus',
    'ServiceType',
    'ServiceConfig',
    'ServiceHealth',
    'ServiceMetrics',
    'ServiceContext',
    'ServiceResult',
    'ServicePriority',
    'BaseService',
    'ServiceManager',

    # Authentication service
    'AuthenticationService',
    'AuthToken',
    'AuthSession',
    'LoginRequest',
    'LoginResponse',
    'RegisterRequest',
    'PasswordResetRequest',
    'User',
    'UserProfile',
    'UserRole',
    'DeviceInfo',
    'TokenType',

    # Knowledge service
    'KnowledgeService',
    'KnowledgeItem',
    'KnowledgeRelationship',
    'KnowledgeSearchQuery',
    'KnowledgeSearchResult',
    'KnowledgeMetadata',
    'KnowledgeType',
    'RelationshipType',

    # Analytics service
    'AnalyticsService',
    'AnalyticsEvent',
    'AnalyticsEventType',
    'AnalyticsRequest',
    'AnalyticsQuery',
    'AnalyticsReport',
    'UserMetrics',
    'SystemMetrics',
    'Metric',
    'MetricType',
    'EventContext',

    # Notification service
    'NotificationService',
    'Notification',
    'NotificationType',
    'NotificationPriority',
    'NotificationStatus',
    'NotificationRequest',
    'NotificationQuery',
    'NotificationStats',
    'NotificationTemplate',
    'NotificationChannel',

    # Learning service
    'LearningService',
    'LearningSession',
    'LearningSessionState',
    'SessionType',
    'SessionStatus',
    'LearningAction',
    'LearningResult',
    'LearningQuestion',
    'LearningAnswer',
    'LearningGoal',
    'CreateSessionRequest',
    'SubmitAnswerRequest',
    'LearningProgress',
    'LearningStatistics',
    'LearningRecommendation',

    # Memory service
    'MemoryService',
    'MemoryItem',
    'MemoryType',
    'MemoryPriority',
    'MemoryStatus',
    'MemoryQuery',
    'MemoryStatistics',
    'ConversationMemory'
]