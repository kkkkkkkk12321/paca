"""
Messages Constants Module
시스템 메시지 및 텍스트 상수들
"""

from typing import Final, Dict


class ErrorMessages:
    """에러 메시지 템플릿"""
    # 일반 에러
    UNKNOWN_ERROR: Final[str] = '알 수 없는 오류가 발생했습니다.'
    TIMEOUT: Final[str] = '요청 시간이 초과되었습니다. (제한시간: {timeout}ms)'
    NETWORK_ERROR: Final[str] = '네트워크 연결에 문제가 발생했습니다.'

    # 인증 에러
    UNAUTHORIZED: Final[str] = '인증이 필요합니다.'
    FORBIDDEN: Final[str] = '접근 권한이 없습니다.'
    TOKEN_EXPIRED: Final[str] = '인증 토큰이 만료되었습니다.'

    # 입력 검증 에러
    INVALID_INPUT: Final[str] = '입력값이 올바르지 않습니다.'
    REQUIRED_FIELD: Final[str] = '{field} 필드는 필수입니다.'
    FIELD_TOO_LONG: Final[str] = '{field} 길이가 너무 깁니다. (최대: {max}자)'
    FIELD_TOO_SHORT: Final[str] = '{field} 길이가 너무 짧습니다. (최소: {min}자)'

    # 리소스 에러
    NOT_FOUND: Final[str] = '요청한 리소스를 찾을 수 없습니다.'
    ALREADY_EXISTS: Final[str] = '이미 존재하는 리소스입니다.'
    RESOURCE_LIMIT: Final[str] = '리소스 제한을 초과했습니다.'

    # 시스템 에러
    DATABASE_ERROR: Final[str] = '데이터베이스 오류가 발생했습니다.'
    MEMORY_ERROR: Final[str] = '메모리 부족으로 작업을 완료할 수 없습니다.'
    CPU_OVERLOAD: Final[str] = 'CPU 사용률이 임계치를 초과했습니다.'

    # AI 관련 에러
    MODEL_ERROR: Final[str] = 'AI 모델 처리 중 오류가 발생했습니다.'
    REASONING_FAILED: Final[str] = '추론 과정에서 오류가 발생했습니다.'
    LEARNING_ERROR: Final[str] = '학습 과정에서 오류가 발생했습니다.'


class SuccessMessages:
    """성공 메시지 템플릿"""
    OPERATION_COMPLETED: Final[str] = '작업이 성공적으로 완료되었습니다.'
    SAVED_SUCCESSFULLY: Final[str] = '저장이 완료되었습니다.'
    DELETED_SUCCESSFULLY: Final[str] = '삭제가 완료되었습니다.'
    UPDATED_SUCCESSFULLY: Final[str] = '업데이트가 완료되었습니다.'
    LOGIN_SUCCESS: Final[str] = '로그인이 완료되었습니다.'
    LOGOUT_SUCCESS: Final[str] = '로그아웃이 완료되었습니다.'

    MODEL_TRAINED: Final[str] = '모델 훈련이 완료되었습니다.'
    REASONING_COMPLETED: Final[str] = '추론이 성공적으로 완료되었습니다.'
    OPTIMIZATION_COMPLETED: Final[str] = '최적화가 완료되었습니다.'


class StatusMessages:
    """상태 메시지 템플릿"""
    INITIALIZING: Final[str] = '시스템을 초기화하는 중...'
    LOADING: Final[str] = '데이터를 불러오는 중...'
    PROCESSING: Final[str] = '요청을 처리하는 중...'
    SAVING: Final[str] = '저장하는 중...'
    ANALYZING: Final[str] = '분석하는 중...'
    TRAINING: Final[str] = '학습하는 중...'
    REASONING: Final[str] = '추론하는 중...'
    OPTIMIZING: Final[str] = '최적화하는 중...'

    IDLE: Final[str] = '대기 중'
    READY: Final[str] = '준비 완료'
    BUSY: Final[str] = '작업 중'
    MAINTENANCE: Final[str] = '유지보수 중'


class NotificationMessages:
    """사용자 알림 메시지"""
    WELCOME: Final[str] = 'PACA v5에 오신 것을 환영합니다!'
    SESSION_EXPIRING: Final[str] = '세션이 곧 만료됩니다. 계속 사용하시겠습니까?'
    UNSAVED_CHANGES: Final[str] = '저장하지 않은 변경사항이 있습니다.'
    PERFORMANCE_WARNING: Final[str] = '시스템 성능이 저하되었습니다.'
    MEMORY_WARNING: Final[str] = '메모리 사용량이 높습니다.'

    NEW_VERSION: Final[str] = '새 버전이 출시되었습니다.'
    UPDATE_REQUIRED: Final[str] = '필수 업데이트가 있습니다.'
    MAINTENANCE_SCHEDULED: Final[str] = '예정된 유지보수가 있습니다.'


class LogTemplates:
    """로그 메시지 템플릿"""
    REQUEST_RECEIVED: Final[str] = '[{timestamp}] 요청 수신: {method} {path}'
    REQUEST_COMPLETED: Final[str] = '[{timestamp}] 요청 완료: {status} ({duration}ms)'
    ERROR_OCCURRED: Final[str] = '[{timestamp}] 에러 발생: {error} - {details}'
    PERFORMANCE_METRIC: Final[str] = '[{timestamp}] 성능 지표: {metric}={value}'

    SYSTEM_STARTED: Final[str] = 'PACA v5 시스템이 시작되었습니다.'
    SYSTEM_STOPPED: Final[str] = 'PACA v5 시스템이 중지되었습니다.'
    MODULE_LOADED: Final[str] = '모듈 로드 완료: {module}'
    MODULE_FAILED: Final[str] = '모듈 로드 실패: {module} - {error}'


# 편의를 위한 상수 그룹핑
ERROR_MESSAGES = ErrorMessages()
SUCCESS_MESSAGES = SuccessMessages()
STATUS_MESSAGES = StatusMessages()
NOTIFICATION_MESSAGES = NotificationMessages()
LOG_TEMPLATES = LogTemplates()


def format_message(template: str, **params) -> str:
    """메시지 템플릿을 매개변수로 포맷팅"""
    try:
        return template.format(**params)
    except KeyError as e:
        return f"메시지 포맷 오류: {template} - 누락된 매개변수: {e}"


def get_error_message(error_type: str, **params) -> str:
    """에러 타입에 따른 포맷된 메시지 반환"""
    error_attr = getattr(ERROR_MESSAGES, error_type.upper(), None)
    if error_attr is None:
        return ERROR_MESSAGES.UNKNOWN_ERROR
    return format_message(error_attr, **params)


def get_success_message(success_type: str, **params) -> str:
    """성공 타입에 따른 포맷된 메시지 반환"""
    success_attr = getattr(SUCCESS_MESSAGES, success_type.upper(), None)
    if success_attr is None:
        return SUCCESS_MESSAGES.OPERATION_COMPLETED
    return format_message(success_attr, **params)


def get_status_message(status_type: str, **params) -> str:
    """상태 타입에 따른 포맷된 메시지 반환"""
    status_attr = getattr(STATUS_MESSAGES, status_type.upper(), None)
    if status_attr is None:
        return STATUS_MESSAGES.PROCESSING
    return format_message(status_attr, **params)