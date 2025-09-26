"""
Configuration Constants Module
시스템 기본 설정 상수들
"""

import os
from pathlib import Path
from typing import Dict, List, Final, Any

# 기본 타임아웃 설정 (초 단위)
DEFAULT_TIMEOUT: Final[float] = 30.0

# API 호출 타임아웃 (초 단위)
API_TIMEOUT: Final[float] = 60.0

# 데이터베이스 연결 타임아웃 (초 단위)
DB_TIMEOUT: Final[float] = 10.0

# 최대 재시도 횟수
MAX_RETRY_COUNT: Final[int] = 3

# 재시도 간격 (초 단위)
RETRY_DELAY: Final[float] = 1.0

# 캐시 생존 시간 (초 단위)
CACHE_TTL: Final[float] = 300.0  # 5분

# 세션 만료 시간 (초 단위)
SESSION_TIMEOUT: Final[float] = 1800.0  # 30분

# 환경 변수 기반 설정
ENVIRONMENT: Final[str] = os.getenv("PYTHON_ENV", "development")
DEBUG_MODE: Final[bool] = ENVIRONMENT == "development"
LOG_LEVEL: Final[str] = os.getenv("LOG_LEVEL", "info")

# 포트 설정
DEFAULT_PORT: Final[int] = int(os.getenv("PORT", "8000"))

# 포터블 경로 계산
def _get_portable_data_path() -> Path:
    """포터블 데이터 디렉토리 경로 반환"""
    current_file = Path(__file__).resolve()
    paca_python_root = current_file.parents[3]  # paca_python 폴더
    return paca_python_root / "data"

# 데이터베이스 설정
class DatabaseConfig:
    """데이터베이스 설정 상수"""
    MAX_CONNECTIONS: Final[int] = 10
    IDLE_TIMEOUT: Final[float] = 30.0
    CONNECTION_TIMEOUT: Final[float] = 10.0
    POOL_SIZE: Final[int] = 5

    # SQLite 설정 (포터블 경로 사용)
    _portable_db_path = _get_portable_data_path() / "database" / "paca.db"
    SQLITE_DB_PATH: Final[str] = os.getenv("SQLITE_DB_PATH", str(_portable_db_path))
    SQLITE_TIMEOUT: Final[float] = 30.0
    SQLITE_CHECK_SAME_THREAD: Final[bool] = False


# AI 모델 기본 설정
class AIModelConfig:
    """AI 모델 설정 상수"""
    DEFAULT_MODEL: Final[str] = "gemini-1.5-pro"
    MAX_TOKENS: Final[int] = 8192
    TEMPERATURE: Final[float] = 0.7
    TOP_P: Final[float] = 0.9
    TOP_K: Final[int] = 40

    # Korean NLP 설정
    KOREAN_MODEL: Final[str] = "ko"
    MECAB_DICPATH: Final[str] = os.getenv("MECAB_DICPATH", "")

    # API 키 설정
    GOOGLE_API_KEY: Final[str] = os.getenv("GOOGLE_API_KEY", "")
    OPENAI_API_KEY: Final[str] = os.getenv("OPENAI_API_KEY", "")


# 성능 모니터링 설정
class MonitoringConfig:
    """성능 모니터링 설정 상수"""
    COLLECT_INTERVAL: Final[float] = 5.0  # 5초
    RETENTION_PERIOD: Final[float] = 86400.0  # 24시간
    BATCH_SIZE: Final[int] = 100
    ALERT_THRESHOLD: Final[float] = 0.8

    # 메모리 모니터링
    MEMORY_WARNING_THRESHOLD: Final[int] = 500 * 1024 * 1024  # 500MB
    MEMORY_CRITICAL_THRESHOLD: Final[int] = 1024 * 1024 * 1024  # 1GB

    # CPU 모니터링
    CPU_WARNING_THRESHOLD: Final[float] = 80.0  # 80%
    CPU_CRITICAL_THRESHOLD: Final[float] = 95.0  # 95%


# 보안 설정
class SecurityConfig:
    """보안 설정 상수"""
    HASH_ROUNDS: Final[int] = 12
    TOKEN_EXPIRY: Final[float] = 3600.0  # 1시간
    MAX_LOGIN_ATTEMPTS: Final[int] = 5
    LOCKOUT_DURATION: Final[float] = 900.0  # 15분

    # 암호화 설정
    SECRET_KEY: Final[str] = os.getenv("SECRET_KEY", "paca-secret-key-change-in-production")
    ALGORITHM: Final[str] = "HS256"

    # CORS 설정
    ALLOWED_ORIGINS: Final[List[str]] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000"
    ]


# 파일 업로드 설정
class UploadConfig:
    """파일 업로드 설정 상수"""
    MAX_FILE_SIZE: Final[int] = 10 * 1024 * 1024  # 10MB
    ALLOWED_TYPES: Final[List[str]] = [
        "image/jpeg",
        "image/png",
        "text/plain",
        "application/json",
        "text/markdown"
    ]
    # 포터블 경로 사용
    _portable_data_path = _get_portable_data_path()
    UPLOAD_DIR: Final[str] = str(_portable_data_path / "uploads")
    TEMP_DIR: Final[str] = str(_portable_data_path / "temp")

    # 이미지 처리 설정
    MAX_IMAGE_WIDTH: Final[int] = 2048
    MAX_IMAGE_HEIGHT: Final[int] = 2048
    IMAGE_QUALITY: Final[int] = 85


# 로깅 설정
class LoggingConfig:
    """로깅 설정 상수"""
    LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # 포터블 경로 사용
    _portable_log_path = _get_portable_data_path() / "logs" / "paca.log"
    LOG_FILE_PATH: Final[str] = str(_portable_log_path)
    LOG_MAX_SIZE: Final[int] = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: Final[int] = 5

    # 로그 레벨 매핑
    LOG_LEVELS: Final[Dict[str, int]] = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50
    }


# 이벤트 시스템 설정
class EventConfig:
    """이벤트 시스템 설정 상수"""
    MAX_QUEUE_SIZE: Final[int] = 10000
    BATCH_SIZE: Final[int] = 50
    MAX_WAIT_TIME: Final[float] = 1.0  # 1초
    MAX_BATCH_PROCESSING_TIME: Final[float] = 5.0  # 5초
    DEFAULT_RETRY_DELAY: Final[float] = 1.0  # 1초
    MAX_EVENT_AGE: Final[float] = 3600.0  # 1시간


# 인지 시스템 설정
class CognitiveConfig:
    """인지 시스템 설정 상수"""
    # ACT-R 설정
    ACTR_ACTIVATION_THRESHOLD: Final[float] = -2.0
    ACTR_DECAY_RATE: Final[float] = 0.5
    ACTR_RETRIEVAL_THRESHOLD: Final[float] = 0.0

    # SOAR 설정
    SOAR_MAX_ELABORATIONS: Final[int] = 100
    SOAR_MAX_DECISION_CYCLES: Final[int] = 1000

    # 메모리 설정 (포터블 저장소 사용)
    WORKING_MEMORY_SIZE: Final[int] = 7  # Miller's law
    LONG_TERM_MEMORY_CAPACITY: Final[int] = 10000
    EPISODIC_MEMORY_RETENTION: Final[float] = 86400.0  # 24시간

    # 포터블 메모리 경로
    _portable_memory_path = _get_portable_data_path() / "memory"
    MEMORY_STORAGE_PATH: Final[str] = str(_portable_memory_path)


# 학습 시스템 설정
class LearningConfig:
    """학습 시스템 설정 상수"""
    # 강화학습 설정
    LEARNING_RATE: Final[float] = 0.01
    DISCOUNT_FACTOR: Final[float] = 0.95
    EXPLORATION_RATE: Final[float] = 0.1

    # 패턴 인식 설정
    MIN_PATTERN_FREQUENCY: Final[int] = 3
    PATTERN_CONFIDENCE_THRESHOLD: Final[float] = 0.7
    MAX_PATTERN_LENGTH: Final[int] = 10

    # 자율학습 설정
    AUTO_LEARNING_INTERVAL: Final[float] = 300.0  # 5분
    LEARNING_BATCH_SIZE: Final[int] = 20
    MAX_LEARNING_ITERATIONS: Final[int] = 1000


# 네트워크 설정
class NetworkConfig:
    """네트워크 설정 상수"""
    # HTTP 설정
    MAX_CONNECTIONS: Final[int] = 100
    KEEPALIVE_TIMEOUT: Final[float] = 75.0
    REQUEST_TIMEOUT: Final[float] = 30.0

    # 압축 설정
    COMPRESSION_LEVEL: Final[int] = 6
    MIN_COMPRESSION_SIZE: Final[int] = 1024

    # SSL/TLS 설정
    SSL_VERIFY: Final[bool] = True
    SSL_CERT_PATH: Final[str] = ""
    SSL_KEY_PATH: Final[str] = ""


# 개발 모드 전용 설정 오버라이드는 삭제 (Final 변수 재할당 방지)


# 환경별 설정 오버라이드
def get_env_config() -> Dict[str, Any]:
    """환경별 설정 반환"""
    if ENVIRONMENT == "production":
        return {
            "debug": False,
            "log_level": "warning",
            "cache_ttl": 600.0,  # 10분
            "max_connections": 50
        }
    elif ENVIRONMENT == "testing":
        return {
            "debug": True,
            "log_level": "debug",
            "cache_ttl": 10.0,  # 10초
            "database_url": ":memory:"
        }
    else:  # development
        return {
            "debug": True,
            "log_level": "debug",
            "cache_ttl": 60.0,  # 1분
            "auto_reload": True
        }


# 설정 검증 함수
def validate_config() -> bool:
    """설정 유효성 검사"""
    errors = []

    # 필수 환경 변수 검사
    if not AIModelConfig.GOOGLE_API_KEY and ENVIRONMENT == "production":
        errors.append("GOOGLE_API_KEY is required in production")

    # 포트 범위 검사
    if not (1 <= DEFAULT_PORT <= 65535):
        errors.append(f"Invalid port number: {DEFAULT_PORT}")

    # 타임아웃 값 검사
    if DEFAULT_TIMEOUT <= 0:
        errors.append("DEFAULT_TIMEOUT must be positive")

    if errors:
        raise ValueError(f"Configuration validation failed: {', '.join(errors)}")

    return True


# 모듈 로드 시 설정 검증
try:
    validate_config()
except ValueError as e:
    print(f"Warning: {e}")
    # 개발 환경에서는 경고만 출력하고 계속 진행
    if ENVIRONMENT == "production":
        raise