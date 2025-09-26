"""
Test cases for core error classes
"""

import pytest
import json
from typing import Dict, List, Any

from paca.core.errors.base import (
    ErrorSeverity,
    ErrorCategory,
    ErrorContext,
    PacaError,
    ApplicationError,
    InfrastructureError,
    ConfigurationError,
    ValidationError,
    NetworkError,
    AuthenticationError,
    AuthorizationError,
    SystemError,
    ExternalServiceError
)


class TestErrorSeverity:
    """ErrorSeverity 테스트"""

    def test_error_severity_values(self):
        """에러 심각도 값 테스트"""
        assert ErrorSeverity.LOW.value == 'low'
        assert ErrorSeverity.MEDIUM.value == 'medium'
        assert ErrorSeverity.HIGH.value == 'high'
        assert ErrorSeverity.CRITICAL.value == 'critical'


class TestErrorCategory:
    """ErrorCategory 테스트"""

    def test_error_category_values(self):
        """에러 카테고리 값 테스트"""
        assert ErrorCategory.VALIDATION.value == 'validation'
        assert ErrorCategory.SYSTEM.value == 'system'
        assert ErrorCategory.NETWORK.value == 'network'
        assert ErrorCategory.AUTHENTICATION.value == 'authentication'
        assert ErrorCategory.AUTHORIZATION.value == 'authorization'
        assert ErrorCategory.BUSINESS_LOGIC.value == 'business_logic'
        assert ErrorCategory.EXTERNAL_SERVICE.value == 'external_service'


class TestPacaError:
    """PacaError 기본 클래스 테스트"""

    def test_basic_error_creation(self):
        """기본 에러 생성 테스트"""
        error = PacaError("Test error message")

        assert str(error) == "[MEDIUM:system] Test error message"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.category == ErrorCategory.SYSTEM
        assert error.metadata == {}
        assert error.recovery_hints == []
        assert error.related_errors == []
        assert error.error_id is not None
        assert error.timestamp is not None

    def test_error_with_metadata(self):
        """메타데이터가 있는 에러 테스트"""
        metadata = {"key": "value", "number": 42}
        recovery_hints = ["Hint 1", "Hint 2"]

        error = PacaError(
            "Test error with metadata",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.VALIDATION,
            metadata=metadata,
            recovery_hints=recovery_hints
        )

        assert error.severity == ErrorSeverity.HIGH
        assert error.category == ErrorCategory.VALIDATION
        assert error.metadata == metadata
        assert error.recovery_hints == recovery_hints

    def test_error_severity_setter(self):
        """심각도 설정 테스트"""
        error = PacaError("Test error")
        error.severity = ErrorSeverity.CRITICAL

        assert error.severity == ErrorSeverity.CRITICAL

    def test_recovery_hints_setter(self):
        """복구 힌트 설정 테스트"""
        error = PacaError("Test error")
        hints = ["New hint 1", "New hint 2"]
        error.recovery_hints = hints

        assert error.recovery_hints == hints
        assert error.recovery_hints is not hints  # 복사본인지 확인

    def test_error_context(self):
        """에러 컨텍스트 테스트"""
        error = PacaError("Test error")
        context = error.get_context()

        assert isinstance(context, ErrorContext)
        assert context.error_id == error.error_id
        assert context.timestamp == error.timestamp
        assert context.severity == error.severity
        assert context.category == error.category
        assert context.metadata == error.metadata

    def test_error_to_dict(self):
        """에러 딕셔너리 변환 테스트"""
        metadata = {"key": "value"}
        error = PacaError("Test error", metadata=metadata)
        error_dict = error.to_dict()

        assert error_dict['error_id'] == error.error_id
        assert error_dict['message'] == "[MEDIUM:system] Test error"  # 포맷된 메시지
        assert error_dict['severity'] == 'medium'
        assert error_dict['category'] == 'system'
        assert error_dict['metadata'] == metadata
        assert 'timestamp' in error_dict

    def test_error_to_json(self):
        """에러 JSON 변환 테스트"""
        error = PacaError("Test error")
        json_str = error.to_json()

        # JSON 파싱 가능한지 확인
        parsed = json.loads(json_str)
        assert parsed['message'] == "[MEDIUM:system] Test error"  # 포맷된 메시지
        assert parsed['severity'] == 'medium'


class TestApplicationError:
    """ApplicationError 테스트"""

    def test_application_error_creation(self):
        """애플리케이션 에러 생성 테스트"""
        error = ApplicationError("Business logic error")

        assert error.severity == ErrorSeverity.MEDIUM
        assert error.category == ErrorCategory.BUSINESS_LOGIC
        assert "Business logic error" in str(error)

    def test_application_error_with_metadata(self):
        """메타데이터가 있는 애플리케이션 에러 테스트"""
        metadata = {"business_rule": "validation_failed"}
        hints = ["Check business rules"]

        error = ApplicationError(
            "Validation failed",
            metadata=metadata,
            recovery_hints=hints
        )

        assert error.metadata == metadata
        assert error.recovery_hints == hints


class TestInfrastructureError:
    """InfrastructureError 테스트"""

    def test_infrastructure_error_creation(self):
        """인프라 에러 생성 테스트"""
        error = InfrastructureError("Database connection failed")

        assert error.severity == ErrorSeverity.HIGH
        assert error.category == ErrorCategory.SYSTEM
        assert "Database connection failed" in str(error)


class TestConfigurationError:
    """ConfigurationError 테스트"""

    def test_configuration_error_basic(self):
        """기본 설정 에러 테스트"""
        error = ConfigurationError("database.host")

        assert "Invalid configuration: database.host" in str(error)
        assert error.metadata['config_key'] == "database.host"
        assert any("Check configuration file" in hint for hint in error.recovery_hints)

    def test_configuration_error_with_format(self):
        """포맷이 있는 설정 에러 테스트"""
        error = ConfigurationError("database.port", "integer")

        assert error.metadata['config_key'] == "database.port"
        assert error.metadata['expected_format'] == "integer"


class TestValidationError:
    """ValidationError 테스트"""

    def test_validation_error_basic(self):
        """기본 검증 에러 테스트"""
        error = ValidationError("Field validation failed")

        assert error.severity == ErrorSeverity.MEDIUM
        assert error.category == ErrorCategory.VALIDATION

    def test_validation_error_with_field(self):
        """필드 정보가 있는 검증 에러 테스트"""
        error = ValidationError(
            "Email format invalid",
            field_name="email",
            field_value="invalid-email",
            constraints=["must contain @", "must be valid format"]
        )

        assert error.metadata['field_name'] == "email"
        assert error.metadata['field_value'] == "invalid-email"
        assert error.metadata['constraints'] == ["must contain @", "must be valid format"]
        assert any("Check field: email" in hint for hint in error.recovery_hints)


class TestNetworkError:
    """NetworkError 테스트"""

    def test_network_error_basic(self):
        """기본 네트워크 에러 테스트"""
        error = NetworkError("Connection timeout")

        assert error.severity == ErrorSeverity.HIGH
        assert error.category == ErrorCategory.NETWORK
        assert any("Check network connectivity" in hint for hint in error.recovery_hints)

    def test_network_error_with_details(self):
        """상세 정보가 있는 네트워크 에러 테스트"""
        error = NetworkError(
            "HTTP request failed",
            url="https://api.example.com/data",
            status_code=404,
            response_body="Not Found"
        )

        assert error.metadata['url'] == "https://api.example.com/data"
        assert error.metadata['status_code'] == 404
        assert error.metadata['response_body'] == "Not Found"
        assert any("Verify URL" in hint for hint in error.recovery_hints)
        assert any("HTTP Status: 404" in hint for hint in error.recovery_hints)


class TestAuthenticationError:
    """AuthenticationError 테스트"""

    def test_authentication_error_default(self):
        """기본 인증 에러 테스트"""
        error = AuthenticationError()

        assert "Authentication failed" in str(error)
        assert error.severity == ErrorSeverity.HIGH
        assert error.category == ErrorCategory.AUTHENTICATION

    def test_authentication_error_with_user(self):
        """사용자 정보가 있는 인증 에러 테스트"""
        error = AuthenticationError("Token expired", user_id="user123")

        assert "Token expired" in str(error)
        assert error.metadata['user_id'] == "user123"
        assert any("Check credentials" in hint for hint in error.recovery_hints)


class TestAuthorizationError:
    """AuthorizationError 테스트"""

    def test_authorization_error_default(self):
        """기본 인가 에러 테스트"""
        error = AuthorizationError()

        assert "Access denied" in str(error)
        assert error.severity == ErrorSeverity.HIGH
        assert error.category == ErrorCategory.AUTHORIZATION

    def test_authorization_error_with_permission(self):
        """권한 정보가 있는 인가 에러 테스트"""
        error = AuthorizationError(
            "Insufficient permissions",
            user_id="user123",
            required_permission="admin.write"
        )

        assert "Insufficient permissions" in str(error)
        assert error.metadata['user_id'] == "user123"
        assert error.metadata['required_permission'] == "admin.write"
        assert any("Required permission: admin.write" in hint for hint in error.recovery_hints)


class TestSystemError:
    """SystemError 테스트"""

    def test_system_error_basic(self):
        """기본 시스템 에러 테스트"""
        error = SystemError("System malfunction")

        assert error.severity == ErrorSeverity.HIGH
        assert error.category == ErrorCategory.SYSTEM
        assert any("시스템 로그를 확인하세요" in hint for hint in error.recovery_hints)

    def test_system_error_with_context(self):
        """컨텍스트가 있는 시스템 에러 테스트"""
        context = {"module": "database", "operation": "connect"}
        metadata = {"error_code": "DB001"}

        error = SystemError(
            "Database connection failed",
            metadata=metadata,
            context=context
        )

        # 컨텍스트가 메타데이터에 병합되는지 확인
        assert error.metadata['error_code'] == "DB001"
        assert error.metadata['module'] == "database"
        assert error.metadata['operation'] == "connect"


class TestExternalServiceError:
    """ExternalServiceError 테스트"""

    def test_external_service_error_basic(self):
        """기본 외부 서비스 에러 테스트"""
        error = ExternalServiceError("API call failed", "OpenAI API")

        assert error.severity == ErrorSeverity.HIGH
        assert error.category == ErrorCategory.EXTERNAL_SERVICE
        assert error.metadata['service_name'] == "OpenAI API"
        assert any("Check OpenAI API service status" in hint for hint in error.recovery_hints)

    def test_external_service_error_with_details(self):
        """상세 정보가 있는 외부 서비스 에러 테스트"""
        error = ExternalServiceError(
            "Rate limit exceeded",
            service_name="OpenAI API",
            operation="chat_completion",
            response_code="429"
        )

        assert error.metadata['service_name'] == "OpenAI API"
        assert error.metadata['operation'] == "chat_completion"
        assert error.metadata['response_code'] == "429"


if __name__ == "__main__":
    pytest.main([__file__])