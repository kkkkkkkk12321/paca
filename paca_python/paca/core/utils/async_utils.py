"""
Async Utilities Module
비동기 처리 관련 유틸리티 함수들
"""

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, List, Optional, TypeVar, Union, Dict
from dataclasses import dataclass

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


def create_logger(name: str) -> logging.Logger:
    """로거 생성 함수"""
    logger = logging.getLogger(name)

    if not logger.handlers:
        # 핸들러가 없는 경우에만 설정
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


@dataclass
class RetryOptions:
    """재시도 옵션"""
    max_attempts: int = 3
    delay: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay: float = 30.0
    should_retry: Optional[Callable[[Exception], bool]] = None


@dataclass
class DebounceOptions:
    """디바운스 옵션"""
    leading: bool = False
    trailing: bool = True


async def retry_async(
    func: Callable[[], Awaitable[T]],
    options: Optional[RetryOptions] = None
) -> T:
    """재시도 로직이 포함된 비동기 함수 실행"""
    if options is None:
        options = RetryOptions()

    last_error: Optional[Exception] = None

    for attempt in range(1, options.max_attempts + 1):
        try:
            return await func()
        except Exception as error:
            last_error = error

            # 마지막 시도이거나 재시도하지 않을 조건이면 예외 발생
            if attempt == options.max_attempts:
                raise last_error

            if options.should_retry and not options.should_retry(error):
                raise last_error

            # 지연 시간 계산 (백오프)
            current_delay = min(
                options.delay * (options.backoff_multiplier ** (attempt - 1)),
                options.max_delay
            )
            await asyncio.sleep(current_delay)

    # 이 지점에 도달하면 안 되지만, 타입 체커를 위해
    if last_error:
        raise last_error
    raise RuntimeError("Unexpected error in retry logic")


async def timeout_async(coro: Awaitable[T], timeout_seconds: float) -> T:
    """Promise에 타임아웃 적용"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")


async def sleep_async(seconds: float) -> None:
    """지정된 시간만큼 대기"""
    await asyncio.sleep(seconds)


def debounce_async(delay: float, options: Optional[DebounceOptions] = None):
    """비동기 함수용 디바운스 데코레이터"""
    if options is None:
        options = DebounceOptions()

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        task: Optional[asyncio.Task] = None
        last_call_time = 0.0

        async def wrapper(*args, **kwargs) -> T:
            nonlocal task, last_call_time

            current_time = time.time()

            # leading 옵션이 True이고 처음 호출인 경우 즉시 실행
            if options.leading and (current_time - last_call_time) > delay:
                last_call_time = current_time
                return await func(*args, **kwargs)

            # 이전 작업이 있으면 취소
            if task and not task.done():
                task.cancel()

            # trailing 옵션이 True인 경우 지연 후 실행
            if options.trailing:
                async def delayed_call():
                    await asyncio.sleep(delay)
                    last_call_time = time.time()
                    return await func(*args, **kwargs)

                task = asyncio.create_task(delayed_call())
                return await task

            # leading만 True인 경우는 이미 위에서 처리됨
            raise RuntimeError("Invalid debounce configuration")

        return wrapper
    return decorator


def throttle_async(limit: float):
    """비동기 함수용 스로틀 데코레이터"""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        last_call_time = 0.0
        in_throttle = False

        async def wrapper(*args, **kwargs) -> T:
            nonlocal last_call_time, in_throttle

            current_time = time.time()

            if not in_throttle:
                in_throttle = True
                last_call_time = current_time

                try:
                    result = await func(*args, **kwargs)

                    # 스로틀 해제 스케줄
                    async def release_throttle():
                        await asyncio.sleep(limit)
                        nonlocal in_throttle
                        in_throttle = False

                    asyncio.create_task(release_throttle())
                    return result
                except Exception:
                    in_throttle = False
                    raise
            else:
                # 스로틀 중이면 대기 후 재시도
                wait_time = limit - (current_time - last_call_time)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                return await wrapper(*args, **kwargs)

        return wrapper
    return decorator


async def batch_process(
    items: List[T],
    processor: Callable[[T], Awaitable[Any]],
    batch_size: int = 10,
    delay_between_batches: float = 0.0
) -> List[Any]:
    """여러 항목을 배치로 비동기 처리"""
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_tasks = [processor(item) for item in batch]

        # 배치 처리 결과 수집
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        for result in batch_results:
            if isinstance(result, Exception):
                print(f"Batch processing error: {result}")
            else:
                results.append(result)

        # 배치 간 지연
        if delay_between_batches > 0 and i + batch_size < len(items):
            await asyncio.sleep(delay_between_batches)

    return results


async def process_batch_async(
    items: List[T],
    processor: Callable[[T], Awaitable[Any]],
    batch_size: int = 10,
    delay_between_batches: float = 0.0
) -> List[Any]:
    """여러 항목을 배치로 비동기 처리"""
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_tasks = [processor(item) for item in batch]

        # 배치 처리 결과 수집
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        for result in batch_results:
            if isinstance(result, Exception):
                print(f"Batch processing error: {result}")
                # 에러는 결과에 포함하지 않음
            else:
                results.append(result)

        # 배치 간 지연
        if delay_between_batches > 0 and i + batch_size < len(items):
            await asyncio.sleep(delay_between_batches)

    return results


async def safe_execute_async(
    coro: Awaitable[T],
    default_value: Optional[T] = None
) -> Dict[str, Any]:
    """비동기 작업을 안전하게 실행하고 결과 반환"""
    try:
        data = await coro
        return {"success": True, "data": data}
    except Exception as error:
        return {
            "success": False,
            "error": error,
            "data": default_value
        }


async def race_with_success_async(coros: List[Awaitable[T]]) -> T:
    """여러 코루틴 중 가장 먼저 성공한 것 반환"""
    if not coros:
        raise ValueError("No coroutines provided")

    done, pending = await asyncio.wait(
        coros,
        return_when=asyncio.FIRST_COMPLETED
    )

    # 완료된 작업들 확인
    for task in done:
        try:
            result = await task
            # 성공한 경우 나머지 작업들 취소
            for p in pending:
                p.cancel()
            return result
        except Exception:
            continue

    # 남은 작업들이 있으면 재귀적으로 처리
    if pending:
        return await race_with_success_async(list(pending))

    raise RuntimeError("All coroutines failed")


async def wait_until_async(
    condition: Callable[[], Union[bool, Awaitable[bool]]],
    timeout: float = 10.0,
    interval: float = 0.1
) -> None:
    """조건이 만족될 때까지 대기"""
    start_time = time.time()

    while time.time() - start_time < timeout:
        # 조건 함수가 비동기인지 확인
        result = condition()
        if asyncio.iscoroutine(result):
            result = await result

        if result:
            return

        await asyncio.sleep(interval)

    raise TimeoutError(f"Condition not met within {timeout} seconds")


class AsyncLock:
    """비동기 잠금 관리자"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._locked_by: Optional[str] = None

    async def acquire(self, identifier: str = "unknown"):
        """잠금 획득"""
        await self._lock.acquire()
        self._locked_by = identifier

    def release(self):
        """잠금 해제"""
        self._locked_by = None
        self._lock.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()

    @property
    def locked(self) -> bool:
        """잠금 상태 확인"""
        return self._lock.locked()

    @property
    def locked_by(self) -> Optional[str]:
        """잠금을 획득한 식별자"""
        return self._locked_by


class AsyncBatchProcessor:
    """비동기 배치 처리기"""

    def __init__(self, batch_size: int = 100, process_delay: float = 0.1):
        self.batch_size = batch_size
        self.process_delay = process_delay
        self._queue: List[Any] = []
        self._lock = asyncio.Lock()

    async def add_item(self, item: Any) -> None:
        """배치에 항목 추가"""
        async with self._lock:
            self._queue.append(item)
            if len(self._queue) >= self.batch_size:
                await self._process_batch()

    async def _process_batch(self) -> None:
        """배치 처리"""
        if not self._queue:
            return

        batch = self._queue[:]
        self._queue.clear()

        # 실제 처리 로직 (서브클래스에서 오버라이드)
        await self.process_items(batch)

    async def process_items(self, items: List[Any]) -> None:
        """배치 항목들을 처리 (서브클래스에서 구현)"""
        await asyncio.sleep(self.process_delay)

    async def flush(self) -> None:
        """남은 항목들을 강제 처리"""
        async with self._lock:
            await self._process_batch()


class AsyncPool:
    """비동기 풀 관리자"""

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.semaphore = asyncio.Semaphore(max_size)
        self.active_tasks: List[asyncio.Task] = []

    async def submit(self, coro: Awaitable[T]) -> T:
        """풀에 태스크 제출"""
        async with self.semaphore:
            task = asyncio.create_task(coro)
            self.active_tasks.append(task)
            try:
                result = await task
                return result
            finally:
                self.active_tasks.remove(task)

    async def shutdown(self, wait: bool = True) -> None:
        """풀 종료"""
        if wait:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
        else:
            for task in self.active_tasks:
                task.cancel()


class AsyncCacheManager:
    """비동기 캐시 관리자"""

    def __init__(self, default_ttl: float = 300.0):
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 가져오기"""
        async with self._lock:
            if key not in self._cache:
                return None

            item = self._cache[key]
            if time.time() > item['expires']:
                del self._cache[key]
                return None

            return item['value']

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """캐시에 값 설정"""
        ttl = ttl or self.default_ttl
        async with self._lock:
            self._cache[key] = {
                'value': value,
                'expires': time.time() + ttl
            }

    async def delete(self, key: str) -> bool:
        """캐시에서 값 삭제"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        """캐시 전체 삭제"""
        async with self._lock:
            self._cache.clear()


class AsyncLRUCache:
    """비동기 LRU 캐시"""

    def __init__(self, max_size: int = 100, default_ttl: float = 300.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 가져오기"""
        async with self._lock:
            if key not in self._cache:
                return None

            item = self._cache[key]
            if time.time() > item['expires']:
                self._remove_key(key)
                return None

            # LRU 순서 업데이트
            self._access_order.remove(key)
            self._access_order.append(key)

            return item['value']

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """캐시에 값 설정"""
        ttl = ttl or self.default_ttl
        async with self._lock:
            if key in self._cache:
                self._access_order.remove(key)
            elif len(self._cache) >= self.max_size:
                # 가장 오래된 항목 제거
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]

            self._cache[key] = {
                'value': value,
                'expires': time.time() + ttl
            }
            self._access_order.append(key)

    def _remove_key(self, key: str) -> None:
        """키 제거"""
        if key in self._cache:
            del self._cache[key]
            self._access_order.remove(key)


class AsyncQueue:
    """비동기 큐 래퍼"""

    def __init__(self, maxsize: int = 0):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)

    async def put(self, item: T) -> None:
        """큐에 항목 추가"""
        await self._queue.put(item)

    async def get(self) -> T:
        """큐에서 항목 가져오기"""
        return await self._queue.get()

    def put_nowait(self, item: T) -> None:
        """즉시 큐에 항목 추가 (블로킹 없음)"""
        self._queue.put_nowait(item)

    def get_nowait(self) -> T:
        """즉시 큐에서 항목 가져오기 (블로킹 없음)"""
        return self._queue.get_nowait()

    def task_done(self) -> None:
        """작업 완료 표시"""
        self._queue.task_done()

    async def join(self) -> None:
        """모든 작업이 완료될 때까지 대기"""
        await self._queue.join()

    def qsize(self) -> int:
        """큐 크기 반환"""
        return self._queue.qsize()

    def empty(self) -> bool:
        """큐가 비어있는지 확인"""
        return self._queue.empty()

    def full(self) -> bool:
        """큐가 가득 찼는지 확인"""
        return self._queue.full()


# 추가 유틸리티 함수들
async def with_timeout(coro: Awaitable[T], timeout_seconds: float) -> T:
    """비동기 작업에 타임아웃 적용"""
    return await timeout_async(coro, timeout_seconds)


async def gather_with_concurrency(
    coros: List[Awaitable[T]],
    concurrency: int = 10
) -> List[T]:
    """동시 실행 수를 제한하여 여러 코루틴 실행"""
    semaphore = asyncio.Semaphore(concurrency)

    async def limited_coro(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    limited_coros = [limited_coro(coro) for coro in coros]
    return await asyncio.gather(*limited_coros)


async def schedule_task(coro: Awaitable[T], delay_seconds: float = 0.0) -> asyncio.Task[T]:
    """지연된 태스크 스케줄링"""
    async def delayed_task():
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        return await coro

    return asyncio.create_task(delayed_task())


async def delay(seconds: float) -> None:
    """지정된 시간만큼 지연"""
    await asyncio.sleep(seconds)


# 편의 함수들
def run_sync(coro: Awaitable[T]) -> T:
    """비동기 함수를 동기적으로 실행"""
    try:
        loop = asyncio.get_running_loop()
        # 이미 이벤트 루프가 실행 중인 경우
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # 이벤트 루프가 실행 중이 아닌 경우
        return asyncio.run(coro)


def create_task_with_timeout(coro: Awaitable[T], timeout: float) -> asyncio.Task:
    """타임아웃이 있는 태스크 생성"""
    async def timeout_wrapper():
        return await timeout_async(coro, timeout)

    return asyncio.create_task(timeout_wrapper())