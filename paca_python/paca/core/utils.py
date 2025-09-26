"""
Utility Functions Module
PACA 시스템 전반에서 사용되는 공통 유틸리티 함수들
TypeScript lodash 함수들의 Python 네이티브 구현
"""

import asyncio
import functools
import time
import uuid
import json
import hashlib
import re
from typing import (
    Any, Optional, Dict, List, Callable, TypeVar, Union, Awaitable,
    Tuple, Set, Iterable, Generator
)
from datetime import datetime, timezone
from pathlib import Path
import logging

from .types import ID, Timestamp, Result, create_success, create_failure
from .errors import PacaError, ErrorSeverity, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


# ID 및 시간 관련 유틸리티
def generate_id(prefix: str = "", length: int = 8) -> ID:
    """고유 ID 생성"""
    unique_id = str(uuid.uuid4()).replace('-', '')[:length]
    return f"{prefix}{unique_id}" if prefix else unique_id


def current_timestamp() -> Timestamp:
    """현재 타임스탬프 반환 (초)"""
    return time.time()


def format_timestamp(timestamp: Timestamp, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """타임스탬프를 포맷된 문자열로 변환"""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(format_str)


def timestamp_to_iso(timestamp: Timestamp) -> str:
    """타임스탬프를 ISO 8601 형식으로 변환"""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def iso_to_timestamp(iso_string: str) -> Timestamp:
    """ISO 8601 문자열을 타임스탬프로 변환"""
    return datetime.fromisoformat(iso_string.replace('Z', '+00:00')).timestamp()


# 데이터 접근 및 조작 유틸리티 (lodash 스타일)
def safe_get(obj: Dict[str, Any], path: str, default: Any = None) -> Any:
    """안전한 중첩 딕셔너리 접근 (lodash get 구현)"""
    try:
        keys = path.split('.')
        current = obj
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    except (AttributeError, TypeError, KeyError):
        return default


def safe_set(obj: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:
    """안전한 중첩 딕셔너리 설정 (lodash set 구현)"""
    keys = path.split('.')
    current = obj
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    return obj


def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """딕셔너리 깊은 병합 (lodash merge 구현)"""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def pick(obj: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """특정 키들만 선택 (lodash pick 구현)"""
    return {key: obj[key] for key in keys if key in obj}


def omit(obj: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """특정 키들 제외 (lodash omit 구현)"""
    return {key: value for key, value in obj.items() if key not in keys}


def group_by(items: List[T], key_func: Callable[[T], K]) -> Dict[K, List[T]]:
    """항목들을 그룹화 (lodash groupBy 구현)"""
    groups: Dict[K, List[T]] = {}
    for item in items:
        key = key_func(item)
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
    return groups


def unique_by(items: List[T], key_func: Callable[[T], Any]) -> List[T]:
    """키 함수로 중복 제거 (lodash uniqBy 구현)"""
    seen = set()
    result = []
    for item in items:
        key = key_func(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def flatten(nested_list: List[Union[T, List[T]]]) -> List[T]:
    """중첩 리스트 평면화 (lodash flatten 구현)"""
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result


def chunk(items: List[T], size: int) -> List[List[T]]:
    """리스트를 청크로 분할 (lodash chunk 구현)"""
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    return [items[i:i + size] for i in range(0, len(items), size)]


# 함수 조작 유틸리티
def debounce(delay: float):
    """디바운스 데코레이터"""
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        last_called = [0.0]
        timer = [None]

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            nonlocal timer
            current_time = time.time()

            if timer[0]:
                timer[0].cancel()

            def delayed_call():
                last_called[0] = time.time()
                return func(*args, **kwargs)

            timer[0] = None
            if current_time - last_called[0] >= delay:
                return delayed_call()
            else:
                # 실제 구현에서는 타이머를 사용해야 하지만, 여기서는 간단히 구현
                return None

        return wrapper
    return decorator


def throttle(delay: float):
    """스로틀 데코레이터"""
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        last_called = [0.0]

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            current_time = time.time()
            if current_time - last_called[0] >= delay:
                last_called[0] = current_time
                return func(*args, **kwargs)
            return None

        return wrapper
    return decorator


def memoize(maxsize: int = 128):
    """메모이제이션 데코레이터"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        return functools.lru_cache(maxsize=maxsize)(func)
    return decorator


async def retry_async(
    func: Callable[..., Awaitable[T]],
    max_attempts: int = 3,
    delay: float = 1.0,
    exponential_backoff: bool = True,
    exceptions: Tuple[type, ...] = (Exception,)
) -> Result[T]:
    """비동기 함수 재시도"""
    last_exception = None

    for attempt in range(max_attempts):
        try:
            result = await func()
            return create_success(result)
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:
                wait_time = delay * (2 ** attempt) if exponential_backoff else delay
                await asyncio.sleep(wait_time)
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")

    return create_failure(f"Failed after {max_attempts} attempts: {str(last_exception)}")


def safe_call(func: Callable[..., T], *args, default: Optional[T] = None, **kwargs) -> T:
    """안전한 함수 호출"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Safe call failed for {func.__name__}: {str(e)}")
        return default


# 검증 유틸리티
def validate_id(value: Any) -> bool:
    """ID 유효성 검사"""
    return isinstance(value, str) and len(value.strip()) > 0


def validate_email(email: str) -> bool:
    """이메일 유효성 검사"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_url(url: str) -> bool:
    """URL 유효성 검사"""
    pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
    return bool(re.match(pattern, url))


def validate_json(json_str: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """JSON 문자열 유효성 검사"""
    try:
        data = json.loads(json_str)
        return True, data
    except json.JSONDecodeError:
        return False, None


# 해시 및 암호화 유틸리티
def hash_string(text: str, algorithm: str = 'sha256') -> str:
    """문자열 해시"""
    if algorithm == 'md5':
        return hashlib.md5(text.encode()).hexdigest()
    elif algorithm == 'sha1':
        return hashlib.sha1(text.encode()).hexdigest()
    elif algorithm == 'sha256':
        return hashlib.sha256(text.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def generate_hash_id(data: Union[str, Dict[str, Any]], length: int = 16) -> str:
    """데이터 기반 해시 ID 생성"""
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True)
    return hash_string(data)[:length]


# 파일 및 경로 유틸리티
def ensure_directory(path: Union[str, Path]) -> Path:
    """디렉토리 존재 확인 및 생성"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(file_path: Union[str, Path]) -> int:
    """파일 크기 반환 (바이트)"""
    return Path(file_path).stat().st_size


def get_file_extension(file_path: Union[str, Path]) -> str:
    """파일 확장자 반환"""
    return Path(file_path).suffix.lower()


def is_file_newer(file1: Union[str, Path], file2: Union[str, Path]) -> bool:
    """file1이 file2보다 새로운 파일인지 확인"""
    path1, path2 = Path(file1), Path(file2)
    if not path2.exists():
        return True
    if not path1.exists():
        return False
    return path1.stat().st_mtime > path2.stat().st_mtime


# 성능 측정 유틸리티
class PerformanceTimer:
    """성능 측정 타이머"""

    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self) -> 'PerformanceTimer':
        """타이머 시작"""
        self.start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        """타이머 정지 및 경과 시간 반환"""
        if self.start_time is None:
            raise ValueError("Timer not started")
        self.end_time = time.perf_counter()
        return self.elapsed()

    def elapsed(self) -> float:
        """경과 시간 반환"""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.perf_counter()
        return end - self.start_time

    def __enter__(self) -> 'PerformanceTimer':
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed_time = self.stop()
        logger.debug(f"{self.name} completed in {elapsed_time:.4f} seconds")


def measure_time(func: Callable[..., T]) -> Callable[..., T]:
    """함수 실행 시간 측정 데코레이터"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        with PerformanceTimer(func.__name__):
            return func(*args, **kwargs)
    return wrapper


async def measure_time_async(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """비동기 함수 실행 시간 측정 데코레이터"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        with PerformanceTimer(func.__name__):
            return await func(*args, **kwargs)
    return wrapper


# 배치 처리 유틸리티
async def process_in_batches(
    items: List[T],
    processor: Callable[[List[T]], Awaitable[List[Any]]],
    batch_size: int = 10,
    max_concurrent: int = 3
) -> List[Any]:
    """항목들을 배치로 나누어 병렬 처리"""
    batches = chunk(items, batch_size)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_batch(batch: List[T]) -> List[Any]:
        async with semaphore:
            return await processor(batch)

    tasks = [process_batch(batch) for batch in batches]
    batch_results = await asyncio.gather(*tasks)
    return flatten(batch_results)


# 데이터 변환 유틸리티
def camel_to_snake(camel_str: str) -> str:
    """CamelCase를 snake_case로 변환"""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def snake_to_camel(snake_str: str) -> str:
    """snake_case를 CamelCase로 변환"""
    components = snake_str.split('_')
    return components[0] + ''.join(x.capitalize() for x in components[1:])


def convert_keys(obj: Any, converter: Callable[[str], str]) -> Any:
    """객체의 키를 변환"""
    if isinstance(obj, dict):
        return {converter(key): convert_keys(value, converter) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys(item, converter) for item in obj]
    else:
        return obj


# 로깅 유틸리티
def create_logger(name: str, level: str = "INFO") -> logging.Logger:
    """로거 생성"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger