"""
Limits Constants Module
시스템 제한값 및 임계값 상수들
"""

from typing import Final


class MemoryLimits:
    """메모리 제한값 (바이트)"""
    MAX_HEAP_SIZE: Final[int] = 1024 * 1024 * 1024  # 1GB
    MAX_BUFFER_SIZE: Final[int] = 64 * 1024 * 1024  # 64MB
    GC_THRESHOLD: Final[int] = 512 * 1024 * 1024  # 512MB
    CRITICAL_THRESHOLD: Final[float] = 0.9  # 90%


class CpuLimits:
    """CPU 사용률 제한값"""
    MAX_USAGE: Final[float] = 0.8  # 80%
    WARNING_THRESHOLD: Final[float] = 0.7  # 70%
    CRITICAL_THRESHOLD: Final[float] = 0.9  # 90%
    THROTTLE_THRESHOLD: Final[float] = 0.95  # 95%


class FileSizeLimits:
    """파일 크기 제한 (바이트)"""
    MAX_IMAGE: Final[int] = 5 * 1024 * 1024  # 5MB
    MAX_DOCUMENT: Final[int] = 10 * 1024 * 1024  # 10MB
    MAX_AUDIO: Final[int] = 50 * 1024 * 1024  # 50MB
    MAX_VIDEO: Final[int] = 100 * 1024 * 1024  # 100MB
    MAX_DATABASE: Final[int] = 500 * 1024 * 1024  # 500MB


class RateLimits:
    """API 호출 제한"""
    REQUESTS_PER_MINUTE: Final[int] = 60
    REQUESTS_PER_HOUR: Final[int] = 1000
    REQUESTS_PER_DAY: Final[int] = 10000
    BURST_LIMIT: Final[int] = 10
    WINDOW_SIZE: Final[int] = 60000  # 1분 (ms)


class DatabaseLimits:
    """데이터베이스 제한값"""
    MAX_QUERY_TIME: Final[int] = 30000  # 30초 (ms)
    MAX_RESULTS: Final[int] = 10000
    MAX_PAGE_SIZE: Final[int] = 100
    MAX_BATCH_SIZE: Final[int] = 1000
    MAX_CONNECTION_POOL: Final[int] = 20


class NetworkLimits:
    """네트워크 제한값"""
    MAX_PAYLOAD_SIZE: Final[int] = 16 * 1024 * 1024  # 16MB
    MAX_CONNECTIONS: Final[int] = 1000
    CONNECTION_TIMEOUT: Final[int] = 30000  # ms
    SOCKET_TIMEOUT: Final[int] = 60000  # ms
    MAX_REDIRECTS: Final[int] = 5


class ReasoningLimits:
    """추론 시스템 제한값"""
    MAX_CHAIN_LENGTH: Final[int] = 100
    MAX_SUBGOALS: Final[int] = 50
    MAX_ALTERNATIVES: Final[int] = 10
    MAX_REASONING_TIME: Final[int] = 300000  # 5분 (ms)
    MAX_PARALLEL_CHAINS: Final[int] = 5


class LearningLimits:
    """학습 시스템 제한값"""
    MAX_MEMORY_SIZE: Final[int] = 10000
    MAX_EPISODE_LENGTH: Final[int] = 1000
    MAX_TRAINING_TIME: Final[int] = 3600000  # 1시간 (ms)
    MAX_BATCH_SIZE: Final[int] = 32
    MAX_EPOCHS: Final[int] = 100


class UILimits:
    """사용자 인터페이스 제한값"""
    MAX_MESSAGE_LENGTH: Final[int] = 10000
    MAX_HISTORY_SIZE: Final[int] = 1000
    MAX_CONCURRENT_REQUESTS: Final[int] = 5
    TYPING_DELAY: Final[int] = 50  # ms
    ANIMATION_DURATION: Final[int] = 300  # ms


# 편의를 위한 상수 그룹핑
MEMORY_LIMITS = MemoryLimits()
CPU_LIMITS = CpuLimits()
FILE_SIZE_LIMITS = FileSizeLimits()
RATE_LIMITS = RateLimits()
DATABASE_LIMITS = DatabaseLimits()
NETWORK_LIMITS = NetworkLimits()
REASONING_LIMITS = ReasoningLimits()
LEARNING_LIMITS = LearningLimits()
UI_LIMITS = UILimits()