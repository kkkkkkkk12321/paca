"""
Validator Utilities Module
유효성 검증 및 타입 가드 함수들
"""

import re
import json
from typing import Any, Optional, Dict, List, TypeVar, Union, Type
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

from paca.core.types import ID, KeyValuePair
from paca.core.constants import MEMORY_LIMITS, FILE_SIZE_LIMITS, RATE_LIMITS

T = TypeVar('T')

# ==========================================
# Pydantic 기반 검증 모델들
# ==========================================

class IdValidator(BaseModel):
    """ID 형식 검증을 위한 Pydantic 모델"""
    id: str = Field(..., min_length=1, max_length=255, description="유효한 ID")

EMAIL_REGEX = re.compile(
    r"^(?P<local>[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+)@"
    r"(?P<domain>[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*)$"
)


class EmailValidator(BaseModel):
    """이메일 형식 검증을 위한 Pydantic 모델"""

    email: str = Field(..., max_length=254, description="유효한 이메일 주소")

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("이메일 주소는 문자열이어야 합니다")

        if len(value) > 254:
            raise ValueError("이메일 주소가 너무 깁니다")

        if not EMAIL_REGEX.match(value):
            raise ValueError("유효하지 않은 이메일 주소 형식입니다")

        return value

class UrlValidator(BaseModel):
    """URL 형식 검증을 위한 Pydantic 모델"""
    url: str = Field(..., description="유효한 URL")

    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        try:
            result = urlparse(v)
            if not all([result.scheme, result.netloc]):
                raise ValueError('유효하지 않은 URL 형식입니다')
            return v
        except Exception:
            raise ValueError('URL 파싱 실패')

class RangeValidator(BaseModel):
    """숫자 범위 검증을 위한 Pydantic 모델"""
    value: float = Field(..., description="검증할 숫자 값")
    min_value: float = Field(..., description="최소값")
    max_value: float = Field(..., description="최대값")

    @model_validator(mode='after')
    def validate_range(self):
        if not (self.min_value <= self.value <= self.max_value):
            raise ValueError(f'값이 범위를 벗어났습니다: {self.min_value} <= {self.value} <= {self.max_value}')
        return self

class StringLengthValidator(BaseModel):
    """문자열 길이 검증을 위한 Pydantic 모델"""
    text: str = Field(..., description="검증할 문자열")
    min_length: int = Field(0, ge=0, description="최소 길이")
    max_length: int = Field(10000, ge=0, description="최대 길이")

    @model_validator(mode='after')
    def validate_length(self):
        if not (self.min_length <= len(self.text) <= self.max_length):
            raise ValueError(f'문자열 길이가 범위를 벗어났습니다: {self.min_length} <= {len(self.text)} <= {self.max_length}')
        return self

# ==========================================
# 기본 유효성 검증 함수들
# ==========================================

def is_valid_id(id_value: Any) -> bool:
    """기본 ID 형식 유효성 검증"""
    try:
        IdValidator(id=str(id_value))
        return True
    except Exception:
        return False

def is_valid_email(email: str) -> bool:
    """이메일 형식 유효성 검증"""
    try:
        EmailValidator(email=email)
        return True
    except Exception:
        return False

def is_valid_url(url: str) -> bool:
    """URL 형식 유효성 검증"""
    try:
        UrlValidator(url=url)
        return True
    except Exception:
        return False

def validate_range(value: Union[int, float], min_value: Union[int, float], max_value: Union[int, float]) -> bool:
    """숫자 범위 유효성 검증"""
    try:
        RangeValidator(value=value, min_value=min_value, max_value=max_value)
        return True
    except Exception:
        return False

def validate_string_length(text: str, min_length: int, max_length: int) -> bool:
    """문자열 길이 유효성 검증"""
    try:
        StringLengthValidator(text=text, min_length=min_length, max_length=max_length)
        return True
    except Exception:
        return False

def validate_file_size(size: int, file_type: str) -> bool:
    """파일 크기 유효성 검증"""
    if file_type not in FILE_SIZE_LIMITS:
        return False
    limit = FILE_SIZE_LIMITS[file_type]
    return 0 < size <= limit

def validate_memory_usage(usage: Union[int, float]) -> bool:
    """메모리 사용량 유효성 검증"""
    return validate_range(usage, 0, MEMORY_LIMITS["MAX_HEAP_SIZE"])

def validate_ratio(ratio: Union[int, float]) -> bool:
    """비율값 유효성 검증 (0-1)"""
    return validate_range(ratio, 0, 1)

def validate_percentage(percentage: Union[int, float]) -> bool:
    """퍼센트값 유효성 검증 (0-100)"""
    return validate_range(percentage, 0, 100)

def is_valid_timestamp(timestamp: Any) -> bool:
    """타임스탬프 유효성 검증"""
    try:
        import time
        ts = float(timestamp)
        current_time = time.time()
        # 미래 24시간까지 허용 (86400초)
        return 0 < ts <= current_time + 86400
    except (ValueError, TypeError):
        return False

def is_valid_json(json_string: str) -> bool:
    """JSON 문자열 유효성 검증"""
    try:
        json.loads(json_string)
        return True
    except (json.JSONDecodeError, TypeError):
        return False

def has_required_properties(obj: Any, required_props: List[str]) -> bool:
    """객체의 필수 속성 존재 확인"""
    if not isinstance(obj, dict):
        return False
    return all(prop in obj for prop in required_props)

# ==========================================
# 입력값 정제 함수들
# ==========================================

def sanitize_input(input_text: str) -> str:
    """입력값 정제 함수"""
    if not isinstance(input_text, str):
        input_text = str(input_text)

    # 기본 정제
    cleaned = input_text.strip()

    # HTML 태그 제거
    cleaned = re.sub(r'[<>]', '', cleaned)

    # 따옴표 제거
    cleaned = re.sub(r'[\'"]', '', cleaned)

    # 최대 길이 제한
    return cleaned[:10000]

def strip_html(html: str) -> str:
    """HTML 태그 제거"""
    if not isinstance(html, str):
        return str(html)
    return re.sub(r'<[^>]*>', '', html)

def escape_sql(text: str) -> str:
    """SQL 인젝션 방지를 위한 문자열 이스케이프"""
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r'[\'";\\\\]', r'\\\\&', text)

# ==========================================
# 고급 검증 함수들
# ==========================================

def is_valid_array(arr: Any, validator_func: Optional[callable] = None) -> bool:
    """배열 유효성 검증"""
    if not isinstance(arr, list):
        return False

    if validator_func is None:
        return True

    return all(validator_func(item) for item in arr)

def is_empty(value: Any) -> bool:
    """빈 값 체크"""
    if value is None:
        return True

    if isinstance(value, str):
        return len(value.strip()) == 0

    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0

    return False

def is_defined(value: Any) -> bool:
    """값이 정의되어 있는지 확인하는 함수"""
    return value is not None

def validate_api_response(response: Any) -> bool:
    """API 응답 유효성 검증"""
    return (isinstance(response, dict) and
            response is not None and
            not isinstance(response, list))

def validate_performance_metric(metric: Any) -> bool:
    """성능 지표 유효성 검증"""
    if not isinstance(metric, dict):
        return False

    required_fields = ['timestamp', 'value']
    if not has_required_properties(metric, required_fields):
        return False

    return (is_valid_timestamp(metric['timestamp']) and
            isinstance(metric['value'], (int, float)))

# ==========================================
# 한국어 특화 검증 함수들
# ==========================================

def is_valid_korean_text(text: str) -> bool:
    """한국어 텍스트 유효성 검증"""
    if not isinstance(text, str) or not text.strip():
        return False

    # 한글 유니코드 범위:
    # - 완성된 한글: 가-힣 (0xAC00-0xD7A3)
    # - 자모음: ㄱ-ㅎ (0x3131-0x318E), ㅏ-ㅣ (0x314F-0x3163)
    korean_pattern = re.compile(r'[가-힣ㄱ-ㅎㅏ-ㅣ]')
    return bool(korean_pattern.search(text))

def validate_korean_name(name: str) -> bool:
    """한국어 이름 유효성 검증"""
    if not isinstance(name, str):
        return False

    # 한글만 허용, 2-10자 제한
    korean_name_pattern = re.compile(r'^[가-힣]{2,10}$')
    return bool(korean_name_pattern.match(name.strip()))

def validate_korean_phone(phone: str) -> bool:
    """한국 전화번호 유효성 검증 (휴대폰 및 일반전화)"""
    if not isinstance(phone, str) or not phone.strip():
        return False

    phone = phone.strip()

    # 하이픈, 공백 제거 후 검증용 패턴
    clean_phone = re.sub(r'[-\s]', '', phone)

    # 패턴들:
    # 1. 휴대폰: 010, 011, 016, 017, 018, 019
    mobile_patterns = [
        r'^01[0-9]-\d{3,4}-\d{4}$',  # 하이픈 포함
        r'^01[0-9]\s\d{3,4}\s\d{4}$',  # 공백 포함
        r'^01[0-9]\d{7,8}$'  # 하이픈/공백 없음
    ]

    # 2. 일반전화: 02(서울), 031-9(지역번호)
    landline_patterns = [
        r'^0[2-9][0-9]?-\d{3,4}-\d{4}$',  # 지역번호-국번-번호
        r'^0[2-9][0-9]?\s\d{3,4}\s\d{4}$',  # 공백 포함
        r'^0[2-9][0-9]?\d{7,8}$'  # 하이픈/공백 없음
    ]

    # 각 패턴에 대해 검증
    all_patterns = mobile_patterns + landline_patterns

    for pattern in all_patterns:
        if re.match(pattern, phone):
            return True

    return False

# ==========================================
# 커스텀 Pydantic 검증기들
# ==========================================

class KoreanTextValidator(BaseModel):
    """한국어 텍스트 검증을 위한 Pydantic 모델"""
    text: str = Field(..., description="한국어 텍스트")

    @field_validator('text')
    @classmethod
    def validate_korean(cls, v):
        if not is_valid_korean_text(v):
            raise ValueError('한국어 텍스트가 포함되어야 합니다')
        return v

class PerformanceMetricValidator(BaseModel):
    """성능 지표 검증을 위한 Pydantic 모델"""
    timestamp: float = Field(..., description="타임스탬프")
    value: Union[int, float] = Field(..., description="지표 값")
    metric_type: Optional[str] = Field(None, description="지표 유형")

    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v):
        if not is_valid_timestamp(v):
            raise ValueError('유효하지 않은 타임스탬프입니다')
        return v

# ==========================================
# 내보내기
# ==========================================

__all__ = [
    # Pydantic 모델들
    'IdValidator', 'EmailValidator', 'UrlValidator', 'RangeValidator',
    'StringLengthValidator', 'KoreanTextValidator', 'PerformanceMetricValidator',

    # 기본 검증 함수들
    'is_valid_id', 'is_valid_email', 'is_valid_url', 'validate_range',
    'validate_string_length', 'validate_file_size', 'validate_memory_usage',
    'validate_ratio', 'validate_percentage', 'is_valid_timestamp',
    'is_valid_json', 'has_required_properties',

    # 정제 함수들
    'sanitize_input', 'strip_html', 'escape_sql',

    # 고급 검증 함수들
    'is_valid_array', 'is_empty', 'is_defined', 'validate_api_response',
    'validate_performance_metric',

    # 한국어 특화 함수들
    'is_valid_korean_text', 'validate_korean_name', 'validate_korean_phone'
]