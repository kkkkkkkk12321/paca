"""
Constants Module
PACA 시스템 전반에서 사용되는 상수 정의
"""

from typing import Dict, List
from .types import LogLevel

# 시스템 기본값
DEFAULT_TIMEOUT = 30.0  # 30초
DEFAULT_RETRY_COUNT = 3
DEFAULT_BATCH_SIZE = 50
DEFAULT_CACHE_SIZE = 1000
DEFAULT_MAX_LISTENERS = 100

# 이벤트 관련 상수
SYSTEM_EVENTS = {
    "CORE_INITIALIZED": "system.core.initialized",
    "CORE_SHUTDOWN": "system.core.shutdown",
    "ERROR_OCCURRED": "system.error.occurred",
    "PERFORMANCE_WARNING": "system.performance.warning",
    "MEMORY_WARNING": "system.memory.warning",
    "COGNITIVE_STATE_CHANGED": "cognitive.state.changed",
    "LEARNING_POINT_ADDED": "learning.point.added",
    "REASONING_COMPLETED": "reasoning.completed"
}

# 로그 레벨 매핑
LOG_LEVELS = {
    LogLevel.DEBUG: 10,
    LogLevel.INFO: 20,
    LogLevel.WARN: 30,
    LogLevel.ERROR: 40,
    LogLevel.FATAL: 50
}

# 성능 임계값
PERFORMANCE_THRESHOLDS = {
    "RESPONSE_TIME_WARNING": 1.0,  # 1초
    "RESPONSE_TIME_CRITICAL": 5.0,  # 5초
    "MEMORY_WARNING": 500 * 1024 * 1024,  # 500MB
    "MEMORY_CRITICAL": 1024 * 1024 * 1024,  # 1GB
    "CPU_WARNING": 80.0,  # 80%
    "CPU_CRITICAL": 95.0  # 95%
}

# 한국어 NLP 관련 상수
KOREAN_NLP_CONFIG = {
    "TOKENIZER": "okt",  # Open Korean Text
    "MAX_SENTENCE_LENGTH": 512,
    "MIN_CONFIDENCE": 0.7,
    "SUPPORTED_POS_TAGS": [
        "Noun", "Verb", "Adjective", "Adverb", "Alpha", "Punctuation",
        "Hashtag", "ScreenName", "Email", "URL", "KoreanParticle",
        "JapaneseParticle", "Foreign", "Number", "Unknown"
    ]
}

# 학습 시스템 상수
LEARNING_CONFIG = {
    "AUTO_SAVE_INTERVAL": 300,  # 5분
    "MAX_LEARNING_POINTS": 10000,
    "MIN_IMPORTANCE_SCORE": 0.1,
    "DEFAULT_CONFIDENCE": 0.5,
    "BATCH_LEARNING_SIZE": 20
}

# 인지 시스템 상수
COGNITIVE_CONFIG = {
    "ACT_R_CHUNK_SIZE": 7,  # Miller's magic number
    "SOAR_DECISION_CYCLE_LIMIT": 1000,
    "MEMORY_DECAY_RATE": 0.5,
    "ACTIVATION_THRESHOLD": 0.0,
    "DEFAULT_RETRIEVAL_THRESHOLD": -1.0
}

# 추론 시스템 상수
REASONING_CONFIG = {
    "MAX_CHAIN_LENGTH": 50,
    "DEFAULT_REASONING_TIMEOUT": 10.0,
    "METACOGNITION_INTERVAL": 5,
    "CONFIDENCE_THRESHOLD": 0.8
}

# 파일 및 디렉토리 상수
FILE_PATHS = {
    "DATA_DIR": "data",
    "LOG_DIR": "logs",
    "CONFIG_DIR": "config",
    "TEMP_DIR": "temp",
    "BACKUP_DIR": "backup"
}

# 데이터베이스 관련 상수
DATABASE_CONFIG = {
    "DEFAULT_DB_NAME": "paca.db",
    "CONNECTION_TIMEOUT": 30.0,
    "QUERY_TIMEOUT": 10.0,
    "MAX_CONNECTIONS": 10,
    "BACKUP_INTERVAL": 3600  # 1시간
}

# API 관련 상수
API_CONFIG = {
    "DEFAULT_HOST": "localhost",
    "DEFAULT_PORT": 8000,
    "MAX_REQUEST_SIZE": 10 * 1024 * 1024,  # 10MB
    "RATE_LIMIT_REQUESTS": 100,
    "RATE_LIMIT_WINDOW": 60  # 1분
}

# 보안 관련 상수
SECURITY_CONFIG = {
    "SESSION_TIMEOUT": 3600,  # 1시간
    "MAX_LOGIN_ATTEMPTS": 5,
    "PASSWORD_MIN_LENGTH": 8,
    "TOKEN_EXPIRY": 86400,  # 24시간
    "ENCRYPTION_ALGORITHM": "AES-256-GCM"
}

# 캐시 관련 상수
CACHE_CONFIG = {
    "DEFAULT_TTL": 3600,  # 1시간
    "MAX_CACHE_SIZE": 1000,
    "CLEANUP_INTERVAL": 300,  # 5분
    "MAX_MEMORY_USAGE": 100 * 1024 * 1024  # 100MB
}

# GUI 관련 상수 (CustomTkinter)
GUI_CONFIG = {
    "DEFAULT_THEME": "dark",
    "DEFAULT_COLOR_THEME": "blue",
    "WINDOW_WIDTH": 1200,
    "WINDOW_HEIGHT": 800,
    "MIN_WIDTH": 800,
    "MIN_HEIGHT": 600,
    "FONT_FAMILY": "Segoe UI",
    "FONT_SIZE": 12
}

# 에러 코드 상수
ERROR_CODES = {
    "VALIDATION_ERROR": "E001",
    "AUTHENTICATION_ERROR": "E002",
    "AUTHORIZATION_ERROR": "E003",
    "NETWORK_ERROR": "E004",
    "DATABASE_ERROR": "E005",
    "COGNITIVE_ERROR": "E006",
    "LEARNING_ERROR": "E007",
    "REASONING_ERROR": "E008",
    "PERFORMANCE_ERROR": "E009",
    "CONFIGURATION_ERROR": "E010",
    "EXTERNAL_API_ERROR": "E011"
}

# 정규 표현식 패턴
REGEX_PATTERNS = {
    "EMAIL": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
    "URL": r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$',
    "UUID": r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    "KOREAN": r'[가-힣]+',
    "ENGLISH": r'[a-zA-Z]+',
    "NUMBER": r'\d+',
    "PHONE": r'^01[016789]-?\d{3,4}-?\d{4}$'
}

# 메시지 템플릿
MESSAGE_TEMPLATES = {
    "WELCOME": "PACA 시스템에 오신 것을 환영합니다.",
    "SYSTEM_READY": "시스템이 준비되었습니다.",
    "PROCESSING": "요청을 처리하고 있습니다...",
    "COMPLETED": "작업이 완료되었습니다.",
    "ERROR_OCCURRED": "오류가 발생했습니다: {error}",
    "LEARNING_UPDATED": "학습 데이터가 업데이트되었습니다.",
    "COGNITIVE_STATE_CHANGED": "인지 상태가 {state}로 변경되었습니다."
}