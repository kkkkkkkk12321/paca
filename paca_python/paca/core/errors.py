"""
Error Handling Module
PACA 시스템의 포괄적인 에러 처리 시스템
TypeScript 에러 시스템의 Python 변환
"""

import traceback
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import time

from .types import ID, Timestamp, KeyValuePair


class ErrorSeverity(Enum):
    """에러 심각도 레벨"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """에러 카테고리"""
    SYSTEM = "system"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NETWORK = "network"
    DATABASE = "database"
    COGNITIVE = "cognitive"
    LEARNING = "learning"
    REASONING = "reasoning"
    PERFORMANCE = "performance"
    CONFIGURATION = "configuration"
    EXTERNAL_API = "external_api"


@dataclass
class ErrorContext:
    """에러 컨텍스트 정보"""
    module: str
    function: str
    line_number: Optional[int] = None
    file_path: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_data: KeyValuePair = field(default_factory=dict)


@dataclass
class ErrorDetail:
    """상세 에러 정보"""
    error_id: ID
    timestamp: Timestamp
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    code: Optional[str] = None
    context: Optional[ErrorContext] = None
    stack_trace: Optional[str] = None
    cause: Optional['ErrorDetail'] = None
    suggested_actions: List[str] = field(default_factory=list)
    metadata: KeyValuePair = field(default_factory=dict)


class PacaError(Exception):
    """
    PACA 시스템의 기본 예외 클래스
    모든 커스텀 예외의 베이스 클래스
    """

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        code: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        metadata: Optional[KeyValuePair] = None
    ):
        super().__init__(message)
        self.detail = ErrorDetail(
            error_id=self._generate_error_id(),
            timestamp=time.time(),
            severity=severity,
            category=ErrorCategory.SYSTEM,
            message=message,
            code=code,
            context=context,
            stack_trace=traceback.format_exc(),
            cause=self._create_cause_detail(cause) if cause else None,
            metadata=metadata or {}
        )

    @staticmethod
    def _generate_error_id() -> ID:
        """에러 ID 생성"""
        import uuid
        return str(uuid.uuid4())

    def _create_cause_detail(self, cause: Exception) -> Optional[ErrorDetail]:
        """원인 예외의 상세 정보 생성"""
        if isinstance(cause, PacaError):
            return cause.detail

        return ErrorDetail(
            error_id=self._generate_error_id(),
            timestamp=time.time(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.SYSTEM,
            message=str(cause),
            stack_trace=traceback.format_exception(type(cause), cause, cause.__traceback__)
        )

    def get_severity(self) -> ErrorSeverity:
        """에러 심각도 반환"""
        return self.detail.severity

    def get_category(self) -> ErrorCategory:
        """에러 카테고리 반환"""
        return self.detail.category

    def get_error_id(self) -> ID:
        """에러 ID 반환"""
        return self.detail.error_id

    def get_context(self) -> Optional[ErrorContext]:
        """에러 컨텍스트 반환"""
        return self.detail.context

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'error_id': self.detail.error_id,
            'timestamp': self.detail.timestamp,
            'severity': self.detail.severity.value,
            'category': self.detail.category.value,
            'message': self.detail.message,
            'code': self.detail.code,
            'context': self._context_to_dict(),
            'stack_trace': self.detail.stack_trace,
            'cause': self.detail.cause.error_id if self.detail.cause else None,
            'suggested_actions': self.detail.suggested_actions,
            'metadata': self.detail.metadata
        }

    def _context_to_dict(self) -> Optional[Dict[str, Any]]:
        """컨텍스트를 딕셔너리로 변환"""
        if not self.detail.context:
            return None

        return {
            'module': self.detail.context.module,
            'function': self.detail.context.function,
            'line_number': self.detail.context.line_number,
            'file_path': self.detail.context.file_path,
            'user_id': self.detail.context.user_id,
            'session_id': self.detail.context.session_id,
            'request_id': self.detail.context.request_id,
            'additional_data': self.detail.context.additional_data
        }

    def add_suggested_action(self, action: str) -> 'PacaError':
        """권장 해결 방법 추가"""
        self.detail.suggested_actions.append(action)
        return self

    def __str__(self) -> str:
        """문자열 표현"""
        return f"[{self.detail.severity.value.upper()}] {self.detail.message} (ID: {self.detail.error_id})"

    def __repr__(self) -> str:
        """디버그 표현"""
        return (f"PacaError(message='{self.detail.message}', "
                f"severity={self.detail.severity.value}, "
                f"code={self.detail.code}, "
                f"error_id='{self.detail.error_id}')")


class ValidationError(PacaError):
    """데이터 검증 에러"""

    def __init__(self, message: str, field: Optional[str] = None,
                 value: Optional[Any] = None, **kwargs):
        super().__init__(message, ErrorSeverity.MEDIUM, **kwargs)
        self.detail.category = ErrorCategory.VALIDATION
        if field:
            self.detail.metadata['field'] = field
        if value is not None:
            self.detail.metadata['value'] = str(value)


class AuthenticationError(PacaError):
    """인증 에러"""

    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, ErrorSeverity.HIGH, **kwargs)
        self.detail.category = ErrorCategory.AUTHENTICATION


class AuthorizationError(PacaError):
    """권한 에러"""

    def __init__(self, message: str = "Authorization failed", **kwargs):
        super().__init__(message, ErrorSeverity.HIGH, **kwargs)
        self.detail.category = ErrorCategory.AUTHORIZATION


class NetworkError(PacaError):
    """네트워크 에러"""

    def __init__(self, message: str, status_code: Optional[int] = None,
                 endpoint: Optional[str] = None, **kwargs):
        super().__init__(message, ErrorSeverity.MEDIUM, **kwargs)
        self.detail.category = ErrorCategory.NETWORK
        if status_code:
            self.detail.metadata['status_code'] = status_code
        if endpoint:
            self.detail.metadata['endpoint'] = endpoint


class DatabaseError(PacaError):
    """데이터베이스 에러"""

    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        super().__init__(message, ErrorSeverity.HIGH, **kwargs)
        self.detail.category = ErrorCategory.DATABASE
        if query:
            self.detail.metadata['query'] = query


class CognitiveError(PacaError):
    """인지 시스템 에러"""

    def __init__(self, message: str, model: Optional[str] = None,
                 state: Optional[str] = None, **kwargs):
        super().__init__(message, ErrorSeverity.MEDIUM, **kwargs)
        self.detail.category = ErrorCategory.COGNITIVE
        if model:
            self.detail.metadata['model'] = model
        if state:
            self.detail.metadata['state'] = state


class LearningError(PacaError):
    """학습 시스템 에러"""

    def __init__(self, message: str, learning_type: Optional[str] = None,
                 data_source: Optional[str] = None, **kwargs):
        super().__init__(message, ErrorSeverity.MEDIUM, **kwargs)
        self.detail.category = ErrorCategory.LEARNING
        if learning_type:
            self.detail.metadata['learning_type'] = learning_type
        if data_source:
            self.detail.metadata['data_source'] = data_source


class ReasoningError(PacaError):
    """추론 시스템 에러"""

    def __init__(self, message: str, reasoning_type: Optional[str] = None,
                 chain_step: Optional[int] = None, **kwargs):
        super().__init__(message, ErrorSeverity.MEDIUM, **kwargs)
        self.detail.category = ErrorCategory.REASONING
        if reasoning_type:
            self.detail.metadata['reasoning_type'] = reasoning_type
        if chain_step:
            self.detail.metadata['chain_step'] = chain_step


class PerformanceError(PacaError):
    """성능 관련 에러"""

    def __init__(self, message: str, metric: Optional[str] = None,
                 threshold: Optional[float] = None, actual: Optional[float] = None, **kwargs):
        super().__init__(message, ErrorSeverity.MEDIUM, **kwargs)
        self.detail.category = ErrorCategory.PERFORMANCE
        if metric:
            self.detail.metadata['metric'] = metric
        if threshold:
            self.detail.metadata['threshold'] = threshold
        if actual:
            self.detail.metadata['actual'] = actual


class ConfigurationError(PacaError):
    """설정 에러"""

    def __init__(self, message: str, config_key: Optional[str] = None,
                 config_value: Optional[Any] = None, **kwargs):
        super().__init__(message, ErrorSeverity.HIGH, **kwargs)
        self.detail.category = ErrorCategory.CONFIGURATION
        if config_key:
            self.detail.metadata['config_key'] = config_key
        if config_value is not None:
            self.detail.metadata['config_value'] = str(config_value)


class ExternalApiError(PacaError):
    """외부 API 에러"""

    def __init__(self, message: str, api_name: Optional[str] = None,
                 response_code: Optional[int] = None, **kwargs):
        super().__init__(message, ErrorSeverity.MEDIUM, **kwargs)
        self.detail.category = ErrorCategory.EXTERNAL_API
        if api_name:
            self.detail.metadata['api_name'] = api_name
        if response_code:
            self.detail.metadata['response_code'] = response_code


# 에러 핸들링 헬퍼 함수들
def create_error_context(
    module: str,
    function: str,
    line_number: Optional[int] = None,
    file_path: Optional[str] = None,
    **kwargs
) -> ErrorContext:
    """에러 컨텍스트 생성 헬퍼"""
    return ErrorContext(
        module=module,
        function=function,
        line_number=line_number,
        file_path=file_path,
        additional_data=kwargs
    )


def handle_exception(
    func_name: str,
    exception: Exception,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    context: Optional[ErrorContext] = None
) -> PacaError:
    """예외를 PacaError로 변환"""
    if isinstance(exception, PacaError):
        return exception

    message = f"Exception in {func_name}: {str(exception)}"
    return PacaError(
        message=message,
        severity=severity,
        context=context,
        cause=exception
    )


def is_recoverable_error(error: PacaError) -> bool:
    """복구 가능한 에러인지 확인"""
    non_recoverable_categories = {
        ErrorCategory.AUTHENTICATION,
        ErrorCategory.AUTHORIZATION,
        ErrorCategory.CONFIGURATION
    }

    critical_severity = {ErrorSeverity.CRITICAL}

    return (error.get_category() not in non_recoverable_categories and
            error.get_severity() not in critical_severity)


def get_retry_delay(error: PacaError, attempt: int, base_delay: float = 1.0) -> float:
    """에러 타입에 따른 재시도 지연 시간 계산"""
    if error.get_category() == ErrorCategory.NETWORK:
        # 네트워크 에러는 지수 백오프
        return base_delay * (2 ** attempt)
    elif error.get_category() == ErrorCategory.EXTERNAL_API:
        # API 에러는 선형 증가
        return base_delay * (1 + attempt)
    else:
        # 기타 에러는 기본 지연
        return base_delay