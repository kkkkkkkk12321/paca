"""
Base Error Classes Module
모든 PACA 에러의 기본 클래스들과 공통 인터페이스
"""

import json
import time
import uuid
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

from ..types.base import ID, Timestamp, KeyValuePair, current_timestamp


class ErrorSeverity(Enum):
    """에러 심각도 레벨"""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'


class ErrorCategory(Enum):
    """에러 카테고리"""
    VALIDATION = 'validation'
    SYSTEM = 'system'
    NETWORK = 'network'
    AUTHENTICATION = 'authentication'
    AUTHORIZATION = 'authorization'
    BUSINESS_LOGIC = 'business_logic'
    EXTERNAL_SERVICE = 'external_service'


@dataclass(frozen=True)
class ErrorContext:
    """에러 컨텍스트"""
    error_id: ID
    timestamp: Timestamp
    severity: ErrorSeverity
    category: ErrorCategory
    metadata: KeyValuePair
    recovery_hints: Optional[List[str]] = None
    related_errors: Optional[List[ID]] = None


class PacaError(Exception):
    """PACA 기본 에러 클래스"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        metadata: Optional[KeyValuePair] = None,
        recovery_hints: Optional[List[str]] = None
    ):
        super().__init__(message)

        self.error_id = self._generate_error_id()
        self.timestamp = current_timestamp()
        self._severity = severity
        self.category = category
        self.metadata = metadata or {}
        self._recovery_hints = recovery_hints or []
        self.related_errors: List[ID] = []

    @property
    def severity(self) -> ErrorSeverity:
        """심각도 레벨"""
        return self._severity

    @severity.setter
    def severity(self, value: ErrorSeverity) -> None:
        """심각도 레벨 설정"""
        self._severity = value

    @property
    def recovery_hints(self) -> List[str]:
        """복구 힌트"""
        return self._recovery_hints.copy()

    @recovery_hints.setter
    def recovery_hints(self, value: List[str]) -> None:
        """복구 힌트 설정"""
        self._recovery_hints = value.copy()

    @staticmethod
    def _generate_error_id() -> ID:
        """에러 ID 생성"""
        return f"err_{int(time.time() * 1000)}_{uuid.uuid4().hex[:9]}"

    def get_context(self) -> ErrorContext:
        """에러 컨텍스트 반환"""
        return ErrorContext(
            error_id=self.error_id,
            timestamp=self.timestamp,
            severity=self.severity,
            category=self.category,
            metadata=self.metadata,
            recovery_hints=self.recovery_hints,
            related_errors=self.related_errors
        )

    def get_full_context(self) -> str:
        """전체 컨텍스트 JSON 문자열 반환"""
        context = self.get_context()
        return json.dumps({
            'name': self.__class__.__name__,
            'message': str(self),
            'error_id': context.error_id,
            'timestamp': context.timestamp,
            'severity': context.severity.value,
            'category': context.category.value,
            'metadata': context.metadata,
            'recovery_hints': context.recovery_hints,
            'related_errors': context.related_errors,
            'stack': traceback.format_exc()
        }, ensure_ascii=False, indent=2)

    def add_metadata(self, key: str, value: Any) -> None:
        """메타데이터 추가"""
        self.metadata[key] = value

    def add_recovery_hint(self, hint: str) -> None:
        """복구 힌트 추가"""
        self._recovery_hints.append(hint)

    def add_related_error(self, error_id: ID) -> None:
        """관련 에러 추가"""
        self.related_errors.append(error_id)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'name': self.__class__.__name__,
            'message': str(self),
            'error_id': self.error_id,
            'timestamp': self.timestamp,
            'severity': self.severity.value,
            'category': self.category.value,
            'metadata': self.metadata,
            'recovery_hints': self.recovery_hints,
            'related_errors': self.related_errors
        }

    def to_json(self) -> str:
        """JSON 문자열로 변환"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def __str__(self) -> str:
        """문자열 표현"""
        return f"[{self.severity.value.upper()}:{self.category.value}] {super().__str__()}"


class ApplicationError(PacaError):
    """애플리케이션 레벨 에러"""

    def __init__(
        self,
        message: str,
        metadata: Optional[KeyValuePair] = None,
        recovery_hints: Optional[List[str]] = None
    ):
        super().__init__(
            message=message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.BUSINESS_LOGIC,
            metadata=metadata,
            recovery_hints=recovery_hints
        )


class InfrastructureError(PacaError):
    """인프라 레벨 에러"""

    def __init__(
        self,
        message: str,
        metadata: Optional[KeyValuePair] = None,
        recovery_hints: Optional[List[str]] = None
    ):
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SYSTEM,
            metadata=metadata,
            recovery_hints=recovery_hints
        )


class ConfigurationError(InfrastructureError):
    """설정 오류"""

    def __init__(self, config_key: str, expected_format: Optional[str] = None):
        message = f"Invalid configuration: {config_key}"
        metadata = {'config_key': config_key}
        hints = [f"Check configuration file for key: {config_key}"]

        if expected_format:
            metadata['expected_format'] = expected_format
            hints.append(f"Expected format: {expected_format}")

        super().__init__(
            message=message,
            metadata=metadata,
            recovery_hints=hints
        )


class ValidationError(PacaError):
    """검증 에러"""

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        constraints: Optional[List[str]] = None
    ):
        metadata = {}
        hints = []

        if field_name:
            metadata['field_name'] = field_name
            hints.append(f"Check field: {field_name}")

        if field_value is not None:
            metadata['field_value'] = str(field_value)

        if constraints:
            metadata['constraints'] = constraints
            hints.extend([f"Constraint: {constraint}" for constraint in constraints])

        super().__init__(
            message=message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
            metadata=metadata,
            recovery_hints=hints
        )


class NetworkError(PacaError):
    """네트워크 에러"""

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None
    ):
        metadata = {}
        hints = ["Check network connectivity", "Verify endpoint availability"]

        if url:
            metadata['url'] = url
            hints.append(f"Verify URL: {url}")

        if status_code:
            metadata['status_code'] = status_code
            hints.append(f"HTTP Status: {status_code}")

        if response_body:
            metadata['response_body'] = response_body

        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.NETWORK,
            metadata=metadata,
            recovery_hints=hints
        )


class AuthenticationError(PacaError):
    """인증 에러"""

    def __init__(self, message: str = "Authentication failed", user_id: Optional[str] = None):
        metadata = {}
        hints = ["Check credentials", "Verify authentication token"]

        if user_id:
            metadata['user_id'] = user_id

        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AUTHENTICATION,
            metadata=metadata,
            recovery_hints=hints
        )


class AuthorizationError(PacaError):
    """인가 에러"""

    def __init__(
        self,
        message: str = "Access denied",
        user_id: Optional[str] = None,
        required_permission: Optional[str] = None
    ):
        metadata = {}
        hints = ["Check user permissions", "Verify access rights"]

        if user_id:
            metadata['user_id'] = user_id

        if required_permission:
            metadata['required_permission'] = required_permission
            hints.append(f"Required permission: {required_permission}")

        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AUTHORIZATION,
            metadata=metadata,
            recovery_hints=hints
        )


class SystemError(PacaError):
    """시스템 에러"""

    def __init__(
        self,
        message: str,
        metadata: Optional[KeyValuePair] = None,
        recovery_hints: Optional[List[str]] = None,
        context: Optional[KeyValuePair] = None
    ):
        # 기본 복구 힌트 설정
        default_hints = [
            "시스템 로그를 확인하세요",
            "시스템 관리자에게 문의하세요",
            "잠시 후 다시 시도하세요"
        ]

        if recovery_hints:
            default_hints.extend(recovery_hints)

        # 컨텍스트를 메타데이터에 추가
        if context:
            if metadata is None:
                metadata = {}
            metadata.update(context)

        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SYSTEM,
            metadata=metadata,
            recovery_hints=default_hints
        )


class ExternalServiceError(PacaError):
    """외부 서비스 에러"""

    def __init__(
        self,
        message: str,
        service_name: str,
        operation: Optional[str] = None,
        response_code: Optional[str] = None
    ):
        metadata = {'service_name': service_name}
        hints = [f"Check {service_name} service status", "Verify service connectivity"]

        if operation:
            metadata['operation'] = operation

        if response_code:
            metadata['response_code'] = response_code

        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.EXTERNAL_SERVICE,
            metadata=metadata,
            recovery_hints=hints
        )