"""
Paths Constants Module
경로, URL, 엔드포인트 상수들
"""

import os
from typing import Final, Dict
from ..utils.portable_storage import get_storage_manager


class ApiEndpoints:
    """API 엔드포인트 경로"""

    class Auth:
        """인증 관련 엔드포인트"""
        LOGIN: Final[str] = '/api/auth/login'
        LOGOUT: Final[str] = '/api/auth/logout'
        REFRESH: Final[str] = '/api/auth/refresh'
        VERIFY: Final[str] = '/api/auth/verify'

    class Users:
        """사용자 관리 엔드포인트"""
        BASE: Final[str] = '/api/users'
        PROFILE: Final[str] = '/api/users/profile'
        PREFERENCES: Final[str] = '/api/users/preferences'
        HISTORY: Final[str] = '/api/users/history'

    class Models:
        """AI 모델 관리 엔드포인트"""
        BASE: Final[str] = '/api/models'
        EXECUTE: Final[str] = '/api/models/execute'
        STATUS: Final[str] = '/api/models/status'
        METRICS: Final[str] = '/api/models/metrics'

    class Reasoning:
        """추론 시스템 엔드포인트"""
        BASE: Final[str] = '/api/reasoning'
        CHAINS: Final[str] = '/api/reasoning/chains'
        STRATEGIES: Final[str] = '/api/reasoning/strategies'
        EXECUTE: Final[str] = '/api/reasoning/execute'

    class Learning:
        """학습 시스템 엔드포인트"""
        BASE: Final[str] = '/api/learning'
        TRAIN: Final[str] = '/api/learning/train'
        EVALUATE: Final[str] = '/api/learning/evaluate'
        MODELS: Final[str] = '/api/learning/models'

    class Performance:
        """성능 모니터링 엔드포인트"""
        BASE: Final[str] = '/api/performance'
        METRICS: Final[str] = '/api/performance/metrics'
        BENCHMARKS: Final[str] = '/api/performance/benchmarks'
        ALERTS: Final[str] = '/api/performance/alerts'


class FilePaths:
    """파일 시스템 경로 (포터블 버전)"""

    @staticmethod
    def _get_storage_manager():
        """Storage manager 인스턴스 반환"""
        return get_storage_manager()

    @classmethod
    def get_data_dir(cls) -> str:
        return str(cls._get_storage_manager().data_path)

    @classmethod
    def get_logs_dir(cls) -> str:
        return str(cls._get_storage_manager().logs_path)

    @classmethod
    def get_cache_dir(cls) -> str:
        return str(cls._get_storage_manager().cache_path)

    @classmethod
    def get_temp_dir(cls) -> str:
        return str(cls._get_storage_manager().data_path / "temp")

    @classmethod
    def get_uploads_dir(cls) -> str:
        return str(cls._get_storage_manager().data_path / "uploads")

    # 설정 파일 (포터블)
    @classmethod
    def get_config_file(cls) -> str:
        return str(cls._get_storage_manager().get_config_file_path("app.json"))

    @classmethod
    def get_database_config(cls) -> str:
        return str(cls._get_storage_manager().get_config_file_path("database.json"))

    @classmethod
    def get_model_config(cls) -> str:
        return str(cls._get_storage_manager().get_config_file_path("models.json"))

    # 데이터베이스 파일 (포터블)
    @classmethod
    def get_main_db(cls) -> str:
        return str(cls._get_storage_manager().get_database_path("paca.db"))

    @classmethod
    def get_performance_db(cls) -> str:
        return str(cls._get_storage_manager().get_database_path("performance.db"))

    @classmethod
    def get_learning_db(cls) -> str:
        return str(cls._get_storage_manager().get_database_path("learning.db"))

    # 로그 파일 (포터블)
    @classmethod
    def get_error_log(cls) -> str:
        return str(cls._get_storage_manager().get_log_file_path("error.log"))

    @classmethod
    def get_access_log(cls) -> str:
        return str(cls._get_storage_manager().get_log_file_path("access.log"))

    @classmethod
    def get_performance_log(cls) -> str:
        return str(cls._get_storage_manager().get_log_file_path("performance.log"))

    @classmethod
    def get_debug_log(cls) -> str:
        return str(cls._get_storage_manager().get_log_file_path("debug.log"))

    # 호환성을 위한 정적 속성(기존 코드와의 호환성)
    DATA_DIR = property(lambda self: self.get_data_dir())
    LOGS_DIR = property(lambda self: self.get_logs_dir())
    CACHE_DIR = property(lambda self: self.get_cache_dir())
    TEMP_DIR = property(lambda self: self.get_temp_dir())
    UPLOADS_DIR = property(lambda self: self.get_uploads_dir())
    CONFIG_FILE = property(lambda self: self.get_config_file())
    DATABASE_CONFIG = property(lambda self: self.get_database_config())
    MODEL_CONFIG = property(lambda self: self.get_model_config())
    MAIN_DB = property(lambda self: self.get_main_db())
    PERFORMANCE_DB = property(lambda self: self.get_performance_db())
    LEARNING_DB = property(lambda self: self.get_learning_db())
    ERROR_LOG = property(lambda self: self.get_error_log())
    ACCESS_LOG = property(lambda self: self.get_access_log())
    PERFORMANCE_LOG = property(lambda self: self.get_performance_log())
    DEBUG_LOG = property(lambda self: self.get_debug_log())


class CacheKeys:
    """캐시 키 패턴"""
    USER_PROFILE: Final[str] = 'user:profile:{user_id}'
    USER_PREFERENCES: Final[str] = 'user:preferences:{user_id}'
    MODEL_CACHE: Final[str] = 'model:cache:{model_id}'
    REASONING_RESULT: Final[str] = 'reasoning:result:{chain_id}'
    PERFORMANCE_METRICS: Final[str] = 'performance:metrics:{timestamp}'
    LEARNING_MODEL: Final[str] = 'learning:model:{model_id}'
    SESSION_DATA: Final[str] = 'session:{session_id}'
    API_RESPONSE: Final[str] = 'api:response:{endpoint}:{hash}'


class DatabaseTables:
    """데이터베이스 테이블명"""
    USERS: Final[str] = 'users'
    SESSIONS: Final[str] = 'sessions'
    MODELS: Final[str] = 'ai_models'
    REASONING_CHAINS: Final[str] = 'reasoning_chains'
    SUBGOALS: Final[str] = 'subgoals'
    PERFORMANCE_METRICS: Final[str] = 'performance_metrics'
    LEARNING_DATA: Final[str] = 'learning_data'
    CONVERSATIONS: Final[str] = 'conversations'
    INTERACTIONS: Final[str] = 'interactions'
    PREFERENCES: Final[str] = 'user_preferences'
    LOGS: Final[str] = 'system_logs'
    CACHE: Final[str] = 'cache_entries'


class WebSocketEvents:
    """WebSocket 이벤트 타입"""
    CONNECTION: Final[str] = 'connection'
    DISCONNECT: Final[str] = 'disconnect'
    ERROR: Final[str] = 'error'

    # 실시간 업데이트
    STATUS_UPDATE: Final[str] = 'status_update'
    PROGRESS_UPDATE: Final[str] = 'progress_update'
    RESULT_UPDATE: Final[str] = 'result_update'

    # 성능 모니터링
    PERFORMANCE_ALERT: Final[str] = 'performance_alert'
    METRIC_UPDATE: Final[str] = 'metric_update'

    # 추론 시스템
    REASONING_START: Final[str] = 'reasoning_start'
    REASONING_PROGRESS: Final[str] = 'reasoning_progress'
    REASONING_COMPLETE: Final[str] = 'reasoning_complete'


class ExternalUrls:
    """외부 서비스 URL"""
    GEMINI_API: Final[str] = 'https://generativelanguage.googleapis.com/v1beta'
    OPENAI_API: Final[str] = 'https://api.openai.com/v1'
    CLAUDE_API: Final[str] = 'https://api.anthropic.com/v1'

    # 문서화
    DOCS_BASE: Final[str] = 'https://docs.paca.ai'
    API_DOCS: Final[str] = 'https://docs.paca.ai/api'

    # 지원
    SUPPORT_URL: Final[str] = 'https://support.paca.ai'
    FEEDBACK_URL: Final[str] = 'https://feedback.paca.ai'


# 편의를 위한 상수 그룹핑
API_ENDPOINTS = ApiEndpoints()
FILE_PATHS = FilePaths()
CACHE_KEYS = CacheKeys()
DATABASE_TABLES = DatabaseTables()
WEBSOCKET_EVENTS = WebSocketEvents()
EXTERNAL_URLS = ExternalUrls()


def get_cache_key(pattern: str, **params) -> str:
    """캐시 키 패턴을 매개변수로 포맷팅"""
    try:
        return pattern.format(**params)
    except KeyError as e:
        raise ValueError(f"캐시 키 포맷 오류: {pattern} - 누락된 매개변수: {e}")


def ensure_directory_exists(path: str) -> str:
    """디렉토리가 존재하지 않으면 생성"""
    os.makedirs(path, exist_ok=True)
    return path


def get_absolute_path(relative_path: str) -> str:
    """상대 경로를 절대 경로로 변환"""
    return os.path.abspath(relative_path)


def get_data_file_path(filename: str) -> str:
    """데이터 파일의 전체 경로 반환 (포터블)"""
    storage_manager = get_storage_manager()
    return str(storage_manager.data_path / filename)


def get_log_file_path(filename: str) -> str:
    """로그 파일의 전체 경로 반환 (포터블)"""
    storage_manager = get_storage_manager()
    return str(storage_manager.get_log_file_path(filename))


def get_cache_file_path(filename: str) -> str:
    """캐시 파일의 전체 경로 반환 (포터블)"""
    storage_manager = get_storage_manager()
    return str(storage_manager.get_cache_file_path(filename))


def get_temp_file_path(filename: str) -> str:
    """임시 파일의 전체 경로 반환 (포터블)"""
    storage_manager = get_storage_manager()
    return str(storage_manager.data_path / "temp" / filename)


def get_database_file_path(filename: str) -> str:
    """데이터베이스 파일의 전체 경로 반환 (포터블)"""
    storage_manager = get_storage_manager()
    return str(storage_manager.get_database_path(filename))


def get_config_file_path(filename: str) -> str:
    """설정 파일의 전체 경로 반환 (포터블)"""
    storage_manager = get_storage_manager()
    return str(storage_manager.get_config_file_path(filename))