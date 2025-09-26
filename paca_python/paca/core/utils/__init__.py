"""
Utils Module
유틸리티 함수들과 도구들
"""

from .async_utils import (
    RetryOptions,
    AsyncBatchProcessor,
    AsyncPool,
    retry_async,
    batch_process,
    debounce_async,
    throttle_async,
    with_timeout,
    gather_with_concurrency,
    schedule_task,
    delay,
    AsyncCacheManager,
    AsyncLRUCache
)

from .math_utils import (
    calculate_mean,
    calculate_median,
    calculate_mode,
    calculate_std_dev,
    calculate_variance,
    calculate_correlation,
    is_outlier,
    normalize,
    interpolate,
    MathUtilsError
)

# Import core utility functions from utils.py
try:
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent
    utils_file = parent_dir / '../utils.py'
    if utils_file.exists():
        # Import common functions from utils.py
        from ..utils import (
            generate_id,
            current_timestamp,
            safe_get,
            safe_set,
            deep_merge,
            pick,
            omit,
            group_by,
            unique_by,
            flatten,
            chunk
        )
    else:
        # Fallback definitions if utils.py is not available
        import time
        import uuid

        def generate_id(prefix: str = "", length: int = 8) -> str:
            unique_id = str(uuid.uuid4()).replace('-', '')[:length]
            return f"{prefix}{unique_id}" if prefix else unique_id

        def current_timestamp() -> float:
            return time.time()

        def safe_get(obj: dict, path: str, default=None):
            keys = path.split('.')
            current = obj
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default
            return current

        def safe_set(obj: dict, path: str, value):
            keys = path.split('.')
            current = obj
            for key in keys[:-1]:
                if key not in current or not isinstance(current[key], dict):
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
            return obj

        def deep_merge(dict1: dict, dict2: dict) -> dict:
            result = dict1.copy()
            for key, value in dict2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        def pick(obj: dict, keys: list) -> dict:
            return {key: obj[key] for key in keys if key in obj}

        def omit(obj: dict, keys: list) -> dict:
            return {key: value for key, value in obj.items() if key not in keys}

        def group_by(items: list, key_func) -> dict:
            groups = {}
            for item in items:
                key = key_func(item)
                if key not in groups:
                    groups[key] = []
                groups[key].append(item)
            return groups

        def unique_by(items: list, key_func) -> list:
            seen = set()
            result = []
            for item in items:
                key = key_func(item)
                if key not in seen:
                    seen.add(key)
                    result.append(item)
            return result

        def flatten(nested_list: list) -> list:
            result = []
            for item in nested_list:
                if isinstance(item, list):
                    result.extend(item)
                else:
                    result.append(item)
            return result

        def chunk(items: list, size: int) -> list:
            if size <= 0:
                raise ValueError("Chunk size must be positive")
            return [items[i:i + size] for i in range(0, len(items), size)]

except ImportError:
    # Fallback definitions
    import time
    import uuid

    def generate_id(prefix: str = "", length: int = 8) -> str:
        unique_id = str(uuid.uuid4()).replace('-', '')[:length]
        return f"{prefix}{unique_id}" if prefix else unique_id

    def current_timestamp() -> float:
        return time.time()

    def safe_get(obj: dict, path: str, default=None):
        return default

    def safe_set(obj: dict, path: str, value):
        return obj

    def deep_merge(dict1: dict, dict2: dict) -> dict:
        return {**dict1, **dict2}

    def pick(obj: dict, keys: list) -> dict:
        return {key: obj[key] for key in keys if key in obj}

    def omit(obj: dict, keys: list) -> dict:
        return {key: value for key, value in obj.items() if key not in keys}

    def group_by(items: list, key_func) -> dict:
        return {}

    def unique_by(items: list, key_func) -> list:
        return items

    def flatten(nested_list: list) -> list:
        return nested_list

    def chunk(items: list, size: int) -> list:
        return [items]

__all__ = [
    # Async utilities
    'RetryOptions',
    'AsyncBatchProcessor',
    'AsyncPool',
    'retry_async',
    'batch_process',
    'debounce_async',
    'throttle_async',
    'with_timeout',
    'gather_with_concurrency',
    'schedule_task',
    'delay',
    'AsyncCacheManager',
    'AsyncLRUCache',

    # Math utilities
    'calculate_mean',
    'calculate_median',
    'calculate_mode',
    'calculate_std_dev',
    'calculate_variance',
    'calculate_correlation',
    'is_outlier',
    'normalize',
    'interpolate',
    'MathUtilsError',

    # Core utilities
    'generate_id',
    'current_timestamp',
    'safe_get',
    'safe_set',
    'deep_merge',
    'pick',
    'omit',
    'group_by',
    'unique_by',
    'flatten',
    'chunk'
]