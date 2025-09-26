"""
Input Validation System
PACA v5 입력 검증 시스템 - 데이터 검증, 스키마 검증, 입력 살균
"""

import re
import json
import html
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable, Pattern
from datetime import datetime

from ..core.types.base import ID, create_id, current_timestamp
from ..core.utils.logger import create_logger
from ..core.errors.base import PacaError

class ValidationType(Enum):
    """검증 타입"""
    REQUIRED = 'required'
    TYPE = 'type'
    LENGTH = 'length'
    RANGE = 'range'
    PATTERN = 'pattern'
    FORMAT = 'format'
    CUSTOM = 'custom'
    SANITIZATION = 'sanitization'

class ValidationSeverity(Enum):
    """검증 심각도"""
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'

@dataclass
class ValidationRule:
    """검증 규칙"""
    name: str
    validation_type: ValidationType
    severity: ValidationSeverity = ValidationSeverity.ERROR
    error_message: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    validator_function: Optional[Callable] = None

@dataclass
class ValidationError:
    """검증 오류"""
    field_name: str
    rule_name: str
    error_message: str
    severity: ValidationSeverity
    value: Any = None
    suggestion: Optional[str] = None

@dataclass
class ValidationResult:
    """검증 결과"""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    sanitized_data: Optional[Any] = None
    validation_id: ID = field(default_factory=create_id)
    timestamp: float = field(default_factory=current_timestamp)

    def add_error(self, error: ValidationError):
        """오류 추가"""
        if error.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.errors.append(error)
            self.is_valid = False
        else:
            self.warnings.append(error)

    def get_error_summary(self) -> str:
        """오류 요약 반환"""
        if self.is_valid:
            return "검증 성공"

        error_count = len(self.errors)
        warning_count = len(self.warnings)

        summary = f"검증 실패: {error_count}개 오류"
        if warning_count > 0:
            summary += f", {warning_count}개 경고"

        return summary

@dataclass
class ValidationContext:
    """검증 컨텍스트"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_source: str = 'unknown'
    security_level: str = 'normal'  # 'low', 'normal', 'high', 'critical'
    custom_params: Dict[str, Any] = field(default_factory=dict)

class InputSanitizer:
    """입력 살균기"""

    def __init__(self):
        self.logger = create_logger(__name__)

        # 위험한 패턴들
        self.dangerous_patterns = {
            'sql_injection': [
                re.compile(r"('|(\\')|(;|\\x00|\\n|\\r|\\x1a))", re.IGNORECASE),
                re.compile(r"(union|select|insert|update|delete|drop|create|alter)", re.IGNORECASE),
            ],
            'xss': [
                re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
                re.compile(r"javascript:", re.IGNORECASE),
                re.compile(r"on\w+\s*=", re.IGNORECASE),
            ],
            'path_traversal': [
                re.compile(r"\.\./"),
                re.compile(r"\.\\"),
                re.compile(r"%2e%2e%2f", re.IGNORECASE),
            ],
            'command_injection': [
                re.compile(r"[;&|`]"),
                re.compile(r"(\$\()|(\`)|(\${)"),
            ]
        }

    def sanitize_text(self, text: str, security_level: str = 'normal') -> str:
        """텍스트 살균"""
        if not isinstance(text, str):
            return str(text)

        sanitized = text

        # HTML 인코딩
        sanitized = html.escape(sanitized)

        # 보안 수준에 따른 추가 처리
        if security_level in ['high', 'critical']:
            # 위험한 패턴 제거
            for pattern_type, patterns in self.dangerous_patterns.items():
                for pattern in patterns:
                    sanitized = pattern.sub('', sanitized)

            # 특수 문자 제한
            if security_level == 'critical':
                # 영숫자와 일부 안전한 문자만 허용
                sanitized = re.sub(r'[^a-zA-Z0-9\s가-힣.,!?-]', '', sanitized)

        # 제어 문자 제거
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)

        # 연속된 공백 정리
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()

        return sanitized

    def detect_threats(self, text: str) -> List[str]:
        """위협 탐지"""
        threats = []

        for threat_type, patterns in self.dangerous_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    threats.append(threat_type)
                    break

        return threats

    def is_safe_filename(self, filename: str) -> bool:
        """안전한 파일명 검사"""
        if not filename:
            return False

        # 위험한 문자 체크
        dangerous_chars = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*']
        for char in dangerous_chars:
            if char in filename:
                return False

        # 예약된 파일명 체크 (Windows)
        reserved_names = ['CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
                         'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2',
                         'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9']

        if filename.upper().split('.')[0] in reserved_names:
            return False

        return True

class SchemaValidator:
    """스키마 검증기"""

    def __init__(self):
        self.logger = create_logger(__name__)

    def validate_schema(self, data: Any, schema: Dict[str, Any]) -> ValidationResult:
        """스키마 검증"""
        result = ValidationResult(is_valid=True)

        try:
            self._validate_recursive(data, schema, result, "")
        except Exception as e:
            error = ValidationError(
                field_name="schema",
                rule_name="schema_validation",
                error_message=f"스키마 검증 중 오류: {str(e)}",
                severity=ValidationSeverity.ERROR,
                value=data
            )
            result.add_error(error)

        return result

    def _validate_recursive(self, data: Any, schema: Dict[str, Any],
                          result: ValidationResult, path: str):
        """재귀적 검증"""
        schema_type = schema.get('type', 'any')

        # 타입 검증
        if not self._validate_type(data, schema_type):
            error = ValidationError(
                field_name=path,
                rule_name="type_validation",
                error_message=f"타입 불일치: 예상 {schema_type}, 실제 {type(data).__name__}",
                severity=ValidationSeverity.ERROR,
                value=data
            )
            result.add_error(error)
            return

        # 필수 필드 검증 (객체인 경우)
        if schema_type == 'object' and isinstance(data, dict):
            required_fields = schema.get('required', [])
            for field in required_fields:
                if field not in data:
                    error = ValidationError(
                        field_name=f"{path}.{field}" if path else field,
                        rule_name="required_field",
                        error_message=f"필수 필드 누락: {field}",
                        severity=ValidationSeverity.ERROR
                    )
                    result.add_error(error)

            # 프로퍼티 검증
            properties = schema.get('properties', {})
            for field_name, field_schema in properties.items():
                if field_name in data:
                    field_path = f"{path}.{field_name}" if path else field_name
                    self._validate_recursive(data[field_name], field_schema, result, field_path)

        # 배열 검증
        elif schema_type == 'array' and isinstance(data, list):
            items_schema = schema.get('items', {})
            for i, item in enumerate(data):
                item_path = f"{path}[{i}]" if path else f"[{i}]"
                self._validate_recursive(item, items_schema, result, item_path)

        # 값 제약 검증
        self._validate_constraints(data, schema, result, path)

    def _validate_type(self, data: Any, expected_type: str) -> bool:
        """타입 검증"""
        type_mapping = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict,
            'null': type(None),
            'any': object
        }

        expected_python_type = type_mapping.get(expected_type, object)

        if expected_type == 'any':
            return True

        return isinstance(data, expected_python_type)

    def _validate_constraints(self, data: Any, schema: Dict[str, Any],
                            result: ValidationResult, path: str):
        """제약 조건 검증"""
        # 최소/최대 길이 (문자열, 배열)
        if isinstance(data, (str, list)):
            min_length = schema.get('minLength')
            max_length = schema.get('maxLength')

            if min_length is not None and len(data) < min_length:
                error = ValidationError(
                    field_name=path,
                    rule_name="min_length",
                    error_message=f"최소 길이 {min_length} 미만: 현재 {len(data)}",
                    severity=ValidationSeverity.ERROR,
                    value=data
                )
                result.add_error(error)

            if max_length is not None and len(data) > max_length:
                error = ValidationError(
                    field_name=path,
                    rule_name="max_length",
                    error_message=f"최대 길이 {max_length} 초과: 현재 {len(data)}",
                    severity=ValidationSeverity.ERROR,
                    value=data
                )
                result.add_error(error)

        # 최소/최대 값 (숫자)
        if isinstance(data, (int, float)):
            minimum = schema.get('minimum')
            maximum = schema.get('maximum')

            if minimum is not None and data < minimum:
                error = ValidationError(
                    field_name=path,
                    rule_name="minimum",
                    error_message=f"최소값 {minimum} 미만: 현재 {data}",
                    severity=ValidationSeverity.ERROR,
                    value=data
                )
                result.add_error(error)

            if maximum is not None and data > maximum:
                error = ValidationError(
                    field_name=path,
                    rule_name="maximum",
                    error_message=f"최대값 {maximum} 초과: 현재 {data}",
                    severity=ValidationSeverity.ERROR,
                    value=data
                )
                result.add_error(error)

        # 패턴 검증 (문자열)
        if isinstance(data, str):
            pattern = schema.get('pattern')
            if pattern:
                try:
                    if not re.match(pattern, data):
                        error = ValidationError(
                            field_name=path,
                            rule_name="pattern",
                            error_message=f"패턴 불일치: {pattern}",
                            severity=ValidationSeverity.ERROR,
                            value=data
                        )
                        result.add_error(error)
                except re.error:
                    self.logger.warning(f"잘못된 정규식 패턴: {pattern}")

        # 열거형 검증
        enum_values = schema.get('enum')
        if enum_values and data not in enum_values:
            error = ValidationError(
                field_name=path,
                rule_name="enum",
                error_message=f"허용되지 않은 값: {data}, 허용값: {enum_values}",
                severity=ValidationSeverity.ERROR,
                value=data
            )
            result.add_error(error)

class DataValidator:
    """데이터 검증기"""

    def __init__(self):
        self.logger = create_logger(__name__)

        # 기본 검증 규칙들
        self.built_in_validators = {
            'email': self._validate_email,
            'url': self._validate_url,
            'phone': self._validate_phone,
            'korean_name': self._validate_korean_name,
            'korean_text': self._validate_korean_text,
            'safe_text': self._validate_safe_text,
            'alphanumeric': self._validate_alphanumeric,
            'json': self._validate_json
        }

    def validate_field(self, value: Any, rules: List[ValidationRule],
                      field_name: str = "field") -> ValidationResult:
        """필드 검증"""
        result = ValidationResult(is_valid=True)

        for rule in rules:
            try:
                if not self._apply_rule(value, rule):
                    error = ValidationError(
                        field_name=field_name,
                        rule_name=rule.name,
                        error_message=rule.error_message or f"{rule.name} 검증 실패",
                        severity=rule.severity,
                        value=value
                    )
                    result.add_error(error)

            except Exception as e:
                error = ValidationError(
                    field_name=field_name,
                    rule_name=rule.name,
                    error_message=f"검증 중 오류: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    value=value
                )
                result.add_error(error)

        return result

    def _apply_rule(self, value: Any, rule: ValidationRule) -> bool:
        """규칙 적용"""
        if rule.validation_type == ValidationType.REQUIRED:
            return value is not None and value != ""

        elif rule.validation_type == ValidationType.TYPE:
            expected_type = rule.parameters.get('type')
            if expected_type == 'string':
                return isinstance(value, str)
            elif expected_type == 'integer':
                return isinstance(value, int)
            elif expected_type == 'float':
                return isinstance(value, (int, float))
            elif expected_type == 'boolean':
                return isinstance(value, bool)
            elif expected_type == 'list':
                return isinstance(value, list)
            elif expected_type == 'dict':
                return isinstance(value, dict)

        elif rule.validation_type == ValidationType.LENGTH:
            if not isinstance(value, (str, list)):
                return False

            min_len = rule.parameters.get('min', 0)
            max_len = rule.parameters.get('max', float('inf'))
            return min_len <= len(value) <= max_len

        elif rule.validation_type == ValidationType.RANGE:
            if not isinstance(value, (int, float)):
                return False

            min_val = rule.parameters.get('min', float('-inf'))
            max_val = rule.parameters.get('max', float('inf'))
            return min_val <= value <= max_val

        elif rule.validation_type == ValidationType.PATTERN:
            if not isinstance(value, str):
                return False

            pattern = rule.parameters.get('pattern')
            if isinstance(pattern, str):
                pattern = re.compile(pattern)
            return bool(pattern.match(value))

        elif rule.validation_type == ValidationType.FORMAT:
            format_name = rule.parameters.get('format')
            validator = self.built_in_validators.get(format_name)
            if validator:
                return validator(value)

        elif rule.validation_type == ValidationType.CUSTOM:
            if rule.validator_function:
                return rule.validator_function(value)

        return True

    def _validate_email(self, value: str) -> bool:
        """이메일 검증"""
        if not isinstance(value, str):
            return False

        pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        return bool(pattern.match(value))

    def _validate_url(self, value: str) -> bool:
        """URL 검증"""
        if not isinstance(value, str):
            return False

        pattern = re.compile(
            r'^https?://'  # http:// 또는 https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # 도메인
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # 포트
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return bool(pattern.match(value))

    def _validate_phone(self, value: str) -> bool:
        """전화번호 검증 (한국)"""
        if not isinstance(value, str):
            return False

        # 한국 전화번호 패턴
        patterns = [
            re.compile(r'^010-\d{4}-\d{4}$'),  # 010-1234-5678
            re.compile(r'^01[016789]-\d{3,4}-\d{4}$'),  # 기타 휴대폰
            re.compile(r'^0\d{1,2}-\d{3,4}-\d{4}$'),  # 지역번호
            re.compile(r'^01\d{9}$'),  # 01012345678
        ]

        return any(pattern.match(value) for pattern in patterns)

    def _validate_korean_name(self, value: str) -> bool:
        """한국어 이름 검증"""
        if not isinstance(value, str):
            return False

        # 한글 2-5자, 영문 2-20자
        korean_pattern = re.compile(r'^[가-힣]{2,5}$')
        english_pattern = re.compile(r'^[a-zA-Z\s]{2,20}$')

        return bool(korean_pattern.match(value) or english_pattern.match(value))

    def _validate_korean_text(self, value: str) -> bool:
        """한국어 텍스트 검증"""
        if not isinstance(value, str):
            return False

        # 한글, 영문, 숫자, 기본 특수문자 허용
        pattern = re.compile(r'^[가-힣a-zA-Z0-9\s.,!?()-]*$')
        return bool(pattern.match(value))

    def _validate_safe_text(self, value: str) -> bool:
        """안전한 텍스트 검증"""
        if not isinstance(value, str):
            return False

        # HTML 태그, 스크립트 등 위험한 요소 차단
        dangerous_patterns = [
            r'<[^>]*>',  # HTML 태그
            r'javascript:',  # JavaScript 프로토콜
            r'on\w+\s*=',  # 이벤트 핸들러
            r'[;&|`]',  # 명령어 주입
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False

        return True

    def _validate_alphanumeric(self, value: str) -> bool:
        """영숫자 검증"""
        if not isinstance(value, str):
            return False

        return value.isalnum()

    def _validate_json(self, value: str) -> bool:
        """JSON 형식 검증"""
        if not isinstance(value, str):
            return False

        try:
            json.loads(value)
            return True
        except (json.JSONDecodeError, ValueError):
            return False

class InputValidator:
    """입력 검증기 메인 클래스"""

    def __init__(self):
        self.logger = create_logger(__name__)
        self.sanitizer = InputSanitizer()
        self.schema_validator = SchemaValidator()
        self.data_validator = DataValidator()

        # 검증 통계
        self.validation_count = 0
        self.error_count = 0
        self.warning_count = 0

    async def validate(self, data: Any, context: Optional[ValidationContext] = None) -> ValidationResult:
        """메인 검증 메서드"""
        self.validation_count += 1
        context = context or ValidationContext()

        result = ValidationResult(is_valid=True)

        try:
            # 1. 기본 안전성 검사
            if isinstance(data, str):
                threats = self.sanitizer.detect_threats(data)
                if threats:
                    for threat in threats:
                        error = ValidationError(
                            field_name="input",
                            rule_name="threat_detection",
                            error_message=f"보안 위협 탐지: {threat}",
                            severity=ValidationSeverity.CRITICAL,
                            value=data
                        )
                        result.add_error(error)

                # 살균된 데이터 생성
                result.sanitized_data = self.sanitizer.sanitize_text(
                    data, context.security_level
                )

            # 2. 타입별 기본 검증
            type_result = self._validate_by_type(data, context)
            self._merge_results(result, type_result)

            # 3. 보안 수준별 추가 검증
            security_result = self._validate_security_level(data, context)
            self._merge_results(result, security_result)

            # 통계 업데이트
            if result.errors:
                self.error_count += len(result.errors)
            if result.warnings:
                self.warning_count += len(result.warnings)

        except Exception as e:
            self.logger.error(f"검증 중 예외 발생: {e}")
            error = ValidationError(
                field_name="system",
                rule_name="validation_error",
                error_message=f"검증 시스템 오류: {str(e)}",
                severity=ValidationSeverity.ERROR,
                value=data
            )
            result.add_error(error)

        return result

    def _validate_by_type(self, data: Any, context: ValidationContext) -> ValidationResult:
        """타입별 검증"""
        result = ValidationResult(is_valid=True)

        if isinstance(data, str):
            # 문자열 길이 검증
            if len(data) > 10000:  # 10KB 제한
                error = ValidationError(
                    field_name="input",
                    rule_name="max_length",
                    error_message="입력 텍스트가 너무 깁니다 (최대 10,000자)",
                    severity=ValidationSeverity.ERROR,
                    value=len(data)
                )
                result.add_error(error)

        elif isinstance(data, dict):
            # 딕셔너리 크기 제한
            if len(data) > 100:
                error = ValidationError(
                    field_name="input",
                    rule_name="max_keys",
                    error_message="딕셔너리 키가 너무 많습니다 (최대 100개)",
                    severity=ValidationSeverity.WARNING,
                    value=len(data)
                )
                result.add_error(error)

        elif isinstance(data, list):
            # 리스트 크기 제한
            if len(data) > 1000:
                error = ValidationError(
                    field_name="input",
                    rule_name="max_items",
                    error_message="리스트 항목이 너무 많습니다 (최대 1,000개)",
                    severity=ValidationSeverity.WARNING,
                    value=len(data)
                )
                result.add_error(error)

        return result

    def _validate_security_level(self, data: Any, context: ValidationContext) -> ValidationResult:
        """보안 수준별 검증"""
        result = ValidationResult(is_valid=True)

        if context.security_level == 'critical':
            # 매우 엄격한 검증
            if isinstance(data, str):
                # 특수문자 제한
                if re.search(r'[<>"`\'&;]', data):
                    error = ValidationError(
                        field_name="input",
                        rule_name="special_chars",
                        error_message="위험한 특수문자가 포함되어 있습니다",
                        severity=ValidationSeverity.ERROR,
                        value=data
                    )
                    result.add_error(error)

        elif context.security_level == 'high':
            # 높은 수준 검증
            if isinstance(data, str) and len(data) > 5000:
                error = ValidationError(
                    field_name="input",
                    rule_name="high_security_length",
                    error_message="높은 보안 수준에서는 5,000자를 초과할 수 없습니다",
                    severity=ValidationSeverity.ERROR,
                    value=len(data)
                )
                result.add_error(error)

        return result

    def _merge_results(self, target: ValidationResult, source: ValidationResult):
        """검증 결과 병합"""
        target.errors.extend(source.errors)
        target.warnings.extend(source.warnings)

        if source.errors:
            target.is_valid = False

    def validate_with_schema(self, data: Any, schema: Dict[str, Any]) -> ValidationResult:
        """스키마 기반 검증"""
        return self.schema_validator.validate_schema(data, schema)

    def validate_with_rules(self, data: Any, rules: List[ValidationRule],
                          field_name: str = "field") -> ValidationResult:
        """규칙 기반 검증"""
        return self.data_validator.validate_field(data, rules, field_name)

    def get_validation_statistics(self) -> Dict[str, Any]:
        """검증 통계 반환"""
        return {
            'total_validations': self.validation_count,
            'total_errors': self.error_count,
            'total_warnings': self.warning_count,
            'error_rate': self.error_count / max(self.validation_count, 1),
            'warning_rate': self.warning_count / max(self.validation_count, 1)
        }

# 사전 정의된 검증 규칙들
COMMON_VALIDATION_RULES = {
    'required_string': ValidationRule(
        name='required_string',
        validation_type=ValidationType.REQUIRED,
        error_message='필수 입력 항목입니다'
    ),

    'safe_text': ValidationRule(
        name='safe_text',
        validation_type=ValidationType.FORMAT,
        parameters={'format': 'safe_text'},
        error_message='안전하지 않은 텍스트입니다'
    ),

    'korean_text': ValidationRule(
        name='korean_text',
        validation_type=ValidationType.FORMAT,
        parameters={'format': 'korean_text'},
        error_message='한국어 텍스트만 허용됩니다'
    ),

    'email_format': ValidationRule(
        name='email_format',
        validation_type=ValidationType.FORMAT,
        parameters={'format': 'email'},
        error_message='올바른 이메일 형식이 아닙니다'
    ),

    'short_text': ValidationRule(
        name='short_text',
        validation_type=ValidationType.LENGTH,
        parameters={'min': 1, 'max': 100},
        error_message='1-100자 사이여야 합니다'
    ),

    'medium_text': ValidationRule(
        name='medium_text',
        validation_type=ValidationType.LENGTH,
        parameters={'min': 1, 'max': 1000},
        error_message='1-1000자 사이여야 합니다'
    )
}