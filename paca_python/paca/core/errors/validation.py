"""
Validation Error Classes Module
유효성 검증 관련 에러 클래스들
"""

from typing import Any, List, Optional, Union
from .base import PacaError, ErrorSeverity, ErrorCategory
from ..types.base import KeyValuePair


class ValidationError(PacaError):
    """일반적인 검증 에러"""

    def __init__(
        self,
        message: str,
        metadata: Optional[KeyValuePair] = None,
        recovery_hints: Optional[List[str]] = None
    ):
        super().__init__(
            message=message,
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.VALIDATION,
            metadata=metadata,
            recovery_hints=recovery_hints
        )


class RequiredFieldError(ValidationError):
    """필수 필드 누락 에러"""

    def __init__(self, field_name: str, object_type: Optional[str] = None):
        message = f"Required field '{field_name}' is missing"
        if object_type:
            message += f" in {object_type}"

        metadata = {"field_name": field_name, "object_type": object_type}
        hints = [f"Provide a value for field: {field_name}"]

        super().__init__(
            message=message,
            metadata=metadata,
            recovery_hints=hints
        )


class InvalidFormatError(ValidationError):
    """잘못된 형식 에러"""

    def __init__(
        self,
        field_name: str,
        actual_value: Any,
        expected_format: str,
        examples: Optional[List[str]] = None
    ):
        message = f"Field '{field_name}' has invalid format. Expected: {expected_format}"
        metadata = {
            "field_name": field_name,
            "actual_value": str(actual_value),
            "expected_format": expected_format,
            "examples": examples
        }
        hints = [f"Correct the format of field: {field_name}"]

        if examples:
            hints.append(f"Examples: {', '.join(examples)}")

        super().__init__(
            message=message,
            metadata=metadata,
            recovery_hints=hints
        )


class OutOfRangeError(ValidationError):
    """범위 벗어남 에러"""

    def __init__(
        self,
        field_name: str,
        actual_value: Union[int, float],
        min_value: Union[int, float],
        max_value: Union[int, float]
    ):
        message = f"Field '{field_name}' value {actual_value} is out of range [{min_value}, {max_value}]"
        metadata = {
            "field_name": field_name,
            "actual_value": actual_value,
            "min_value": min_value,
            "max_value": max_value
        }
        hints = [f"Provide a value between {min_value} and {max_value} for field: {field_name}"]

        super().__init__(
            message=message,
            metadata=metadata,
            recovery_hints=hints
        )


class InvalidLengthError(ValidationError):
    """문자열 길이 에러"""

    def __init__(
        self,
        field_name: str,
        actual_length: int,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ):
        message = f"Field '{field_name}' has invalid length: {actual_length}"
        metadata = {
            "field_name": field_name,
            "actual_length": actual_length,
            "min_length": min_length,
            "max_length": max_length
        }
        hints = []

        if min_length is not None and max_length is not None:
            message += f". Expected length between {min_length} and {max_length}"
            hints.append(f"Provide a string with length between {min_length} and {max_length} characters")
        elif min_length is not None:
            message += f". Minimum length: {min_length}"
            hints.append(f"Provide a string with at least {min_length} characters")
        elif max_length is not None:
            message += f". Maximum length: {max_length}"
            hints.append(f"Provide a string with at most {max_length} characters")

        super().__init__(
            message=message,
            metadata=metadata,
            recovery_hints=hints
        )


class DuplicateValueError(ValidationError):
    """중복 값 에러"""

    def __init__(
        self,
        field_name: str,
        value: Any,
        resource_type: Optional[str] = None
    ):
        message = f"Duplicate value for field '{field_name}': {value}"
        if resource_type:
            message += f" in {resource_type}"

        metadata = {
            "field_name": field_name,
            "value": str(value),
            "resource_type": resource_type
        }
        hints = [f"Provide a unique value for field: {field_name}"]

        super().__init__(
            message=message,
            metadata=metadata,
            recovery_hints=hints
        )


class TypeMismatchError(ValidationError):
    """타입 불일치 에러"""

    def __init__(
        self,
        field_name: str,
        actual_type: str,
        expected_type: str,
        actual_value: Optional[Any] = None
    ):
        message = f"Field '{field_name}' has wrong type. Expected: {expected_type}, got: {actual_type}"
        metadata = {
            "field_name": field_name,
            "actual_type": actual_type,
            "expected_type": expected_type
        }

        if actual_value is not None:
            metadata["actual_value"] = str(actual_value)

        hints = [f"Provide a {expected_type} value for field: {field_name}"]

        super().__init__(
            message=message,
            metadata=metadata,
            recovery_hints=hints
        )


class PatternMismatchError(ValidationError):
    """패턴 불일치 에러"""

    def __init__(
        self,
        field_name: str,
        actual_value: str,
        pattern: str,
        pattern_description: Optional[str] = None
    ):
        description = pattern_description or pattern
        message = f"Field '{field_name}' does not match required pattern: {description}"
        metadata = {
            "field_name": field_name,
            "actual_value": actual_value,
            "pattern": pattern,
            "pattern_description": pattern_description
        }
        hints = [f"Provide a value matching pattern: {description}"]

        super().__init__(
            message=message,
            metadata=metadata,
            recovery_hints=hints
        )


class DependencyValidationError(ValidationError):
    """의존성 검증 에러"""

    def __init__(
        self,
        field_name: str,
        dependent_field: str,
        condition: str
    ):
        message = f"Field '{field_name}' validation failed due to dependency on '{dependent_field}': {condition}"
        metadata = {
            "field_name": field_name,
            "dependent_field": dependent_field,
            "condition": condition
        }
        hints = [f"Ensure {condition} for field: {dependent_field}"]

        super().__init__(
            message=message,
            metadata=metadata,
            recovery_hints=hints
        )


class ConstraintViolationError(ValidationError):
    """제약 조건 위반 에러"""

    def __init__(
        self,
        field_name: str,
        constraint_name: str,
        constraint_description: str,
        actual_value: Any
    ):
        message = f"Field '{field_name}' violates constraint '{constraint_name}': {constraint_description}"
        metadata = {
            "field_name": field_name,
            "constraint_name": constraint_name,
            "constraint_description": constraint_description,
            "actual_value": str(actual_value)
        }
        hints = [f"Modify field '{field_name}' to satisfy: {constraint_description}"]

        super().__init__(
            message=message,
            metadata=metadata,
            recovery_hints=hints
        )


class SchemaValidationError(ValidationError):
    """스키마 검증 에러"""

    def __init__(
        self,
        schema_name: str,
        validation_errors: List[str],
        data_path: Optional[str] = None
    ):
        message = f"Schema validation failed for '{schema_name}'"
        if data_path:
            message += f" at path: {data_path}"

        metadata = {
            "schema_name": schema_name,
            "validation_errors": validation_errors,
            "data_path": data_path,
            "error_count": len(validation_errors)
        }
        hints = ["Fix the following validation errors:"]
        hints.extend([f"- {error}" for error in validation_errors[:5]])  # 처음 5개만 표시

        if len(validation_errors) > 5:
            hints.append(f"... and {len(validation_errors) - 5} more errors")

        super().__init__(
            message=message,
            metadata=metadata,
            recovery_hints=hints
        )