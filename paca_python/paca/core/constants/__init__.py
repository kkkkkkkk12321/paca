"""
Core Constants Module - Entry Point
모든 시스템 상수를 외부로 노출하는 진입점
"""

import os
from typing import Union, Dict, Any, Optional

# 설정 상수들
from .config import *

# 제한값 상수들
from .limits import *

# 메시지 상수들
from .messages import *

# 경로 상수들
from .paths import *

__all__ = [
    # 설정 관련
    'DEFAULT_TIMEOUT', 'API_TIMEOUT', 'DB_TIMEOUT', 'MAX_RETRY_COUNT', 'RETRY_DELAY',
    'CACHE_TTL', 'SESSION_TIMEOUT', 'ENVIRONMENT', 'DEBUG_MODE', 'LOG_LEVEL', 'DEFAULT_PORT',
    'DatabaseConfig', 'AIModelConfig', 'MonitoringConfig', 'SecurityConfig', 'UploadConfig',
    'LoggingConfig', 'EventConfig', 'CognitiveConfig', 'LearningConfig', 'NetworkConfig',
    'get_env_config', 'validate_config',

    # 제한값 관련
    'MEMORY_LIMITS', 'CPU_LIMITS', 'FILE_SIZE_LIMITS', 'RATE_LIMITS', 'DATABASE_LIMITS',
    'NETWORK_LIMITS', 'REASONING_LIMITS', 'LEARNING_LIMITS', 'UI_LIMITS',
    'MemoryLimits', 'CpuLimits', 'FileSizeLimits', 'RateLimits', 'DatabaseLimits',
    'NetworkLimits', 'ReasoningLimits', 'LearningLimits', 'UILimits',

    # 메시지 관련
    'ERROR_MESSAGES', 'SUCCESS_MESSAGES', 'STATUS_MESSAGES', 'NOTIFICATION_MESSAGES', 'LOG_TEMPLATES',
    'ErrorMessages', 'SuccessMessages', 'StatusMessages', 'NotificationMessages', 'LogTemplates',
    'format_message', 'get_error_message', 'get_success_message', 'get_status_message',

    # 경로 관련
    'API_ENDPOINTS', 'FILE_PATHS', 'CACHE_KEYS', 'DATABASE_TABLES', 'WEBSOCKET_EVENTS', 'EXTERNAL_URLS',
    'ApiEndpoints', 'FilePaths', 'CacheKeys', 'DatabaseTables', 'WebSocketEvents', 'ExternalUrls',
    'get_cache_key', 'ensure_directory_exists', 'get_absolute_path',
    'get_data_file_path', 'get_log_file_path', 'get_cache_file_path', 'get_temp_file_path',

    # 유틸리티 함수들
    'format_template', 'validate_limit', 'is_production', 'is_development', 'is_test'
]


def format_template(template: str, params: Dict[str, Any]) -> str:
    """템플릿을 매개변수로 포맷팅"""
    try:
        result = template
        for key, value in params.items():
            placeholder = '{' + str(key) + '}'
            result = result.replace(placeholder, str(value))
        return result
    except Exception as e:
        return f"템플릿 포맷 오류: {template} - {e}"


def validate_limit(value: Union[int, float], limit: Union[int, float], error_message: Optional[str] = None) -> bool:
    """값이 제한값을 초과하는지 검증"""
    if value > limit:
        if error_message:
            raise ValueError(error_message)
        return False
    return True


def is_production() -> bool:
    """프로덕션 환경 여부 확인"""
    return os.getenv("PYTHON_ENV", "development") == "production"


def is_development() -> bool:
    """개발 환경 여부 확인"""
    return os.getenv("PYTHON_ENV", "development") == "development"


def is_test() -> bool:
    """테스트 환경 여부 확인"""
    return os.getenv("PYTHON_ENV", "development") == "testing"