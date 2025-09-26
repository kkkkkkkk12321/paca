"""
Rate Limiter
API 호출 속도 제한 관리
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict, deque

from ...core.utils.logger import PacaLogger


@dataclass
class RateLimit:
    """속도 제한 설정"""
    requests_per_minute: int
    requests_per_hour: Optional[int] = None
    requests_per_day: Optional[int] = None
    burst_limit: Optional[int] = None
    window_size: int = 60  # 윈도우 크기 (초)


@dataclass
class RateLimitStatus:
    """속도 제한 상태"""
    endpoint: str
    remaining_requests: int
    reset_time: datetime
    current_requests: int
    limit: int


class TokenBucket:
    """토큰 버킷 알고리즘 구현"""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()
        self.lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> bool:
        """토큰 소비"""
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_refill

            # 토큰 보충
            self.tokens = min(
                self.capacity,
                self.tokens + time_passed * self.refill_rate
            )
            self.last_refill = now

            # 토큰 소비 가능 여부 확인
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def get_wait_time(self, tokens: int = 1) -> float:
        """필요한 대기 시간 계산"""
        if self.tokens >= tokens:
            return 0.0

        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


class SlidingWindowCounter:
    """슬라이딩 윈도우 카운터"""

    def __init__(self, window_size: int, limit: int):
        self.window_size = window_size
        self.limit = limit
        self.requests = deque()
        self.lock = asyncio.Lock()

    async def allow_request(self) -> bool:
        """요청 허용 여부 확인"""
        async with self.lock:
            now = time.time()
            window_start = now - self.window_size

            # 윈도우 밖의 요청 제거
            while self.requests and self.requests[0] < window_start:
                self.requests.popleft()

            # 제한 확인
            if len(self.requests) < self.limit:
                self.requests.append(now)
                return True
            return False

    def get_current_count(self) -> int:
        """현재 윈도우의 요청 수"""
        now = time.time()
        window_start = now - self.window_size

        # 유효한 요청만 카운트
        valid_requests = [req for req in self.requests if req >= window_start]
        return len(valid_requests)

    def get_reset_time(self) -> float:
        """윈도우 리셋 시간"""
        if not self.requests:
            return 0.0
        return self.requests[0] + self.window_size


class RateLimiter:
    """종합 속도 제한 관리자"""

    def __init__(self):
        self.logger = PacaLogger("RateLimiter")

        # 엔드포인트별 제한 설정
        self.limits: Dict[str, RateLimit] = {}

        # 토큰 버킷들
        self.token_buckets: Dict[str, TokenBucket] = {}

        # 슬라이딩 윈도우 카운터들
        self.sliding_counters: Dict[str, Dict[str, SlidingWindowCounter]] = defaultdict(dict)

        # 글로벌 제한
        self.global_limit: Optional[RateLimit] = None
        self.global_bucket: Optional[TokenBucket] = None

        # 통계
        self.stats = {
            "total_requests": 0,
            "accepted_requests": 0,
            "rejected_requests": 0,
            "total_wait_time": 0.0
        }

    def set_rate_limit(self, endpoint: str, rate_limit: RateLimit) -> None:
        """엔드포인트별 속도 제한 설정"""
        self.limits[endpoint] = rate_limit

        # 토큰 버킷 생성 (분당 요청 기준)
        refill_rate = rate_limit.requests_per_minute / 60.0  # per second
        capacity = rate_limit.burst_limit or rate_limit.requests_per_minute
        self.token_buckets[endpoint] = TokenBucket(capacity, refill_rate)

        # 슬라이딩 윈도우 카운터 생성
        if rate_limit.requests_per_minute:
            self.sliding_counters[endpoint]["minute"] = SlidingWindowCounter(
                60, rate_limit.requests_per_minute
            )

        if rate_limit.requests_per_hour:
            self.sliding_counters[endpoint]["hour"] = SlidingWindowCounter(
                3600, rate_limit.requests_per_hour
            )

        if rate_limit.requests_per_day:
            self.sliding_counters[endpoint]["day"] = SlidingWindowCounter(
                86400, rate_limit.requests_per_day
            )

        self.logger.info(f"Rate limit set for endpoint: {endpoint}")

    def set_global_rate_limit(self, rate_limit: RateLimit) -> None:
        """글로벌 속도 제한 설정"""
        self.global_limit = rate_limit

        refill_rate = rate_limit.requests_per_minute / 60.0
        capacity = rate_limit.burst_limit or rate_limit.requests_per_minute
        self.global_bucket = TokenBucket(capacity, refill_rate)

        self.logger.info("Global rate limit set")

    async def acquire(self, endpoint: str, requests: int = 1) -> bool:
        """속도 제한 획득 (블로킹)"""
        self.stats["total_requests"] += 1

        # 글로벌 제한 확인
        if self.global_bucket:
            wait_time = self.global_bucket.get_wait_time(requests)
            if wait_time > 0:
                self.logger.debug(f"Global rate limit wait: {wait_time:.2f}s")
                self.stats["total_wait_time"] += wait_time
                await asyncio.sleep(wait_time)

            if not await self.global_bucket.consume(requests):
                self.stats["rejected_requests"] += 1
                return False

        # 엔드포인트별 제한 확인
        if endpoint in self.limits:
            # 토큰 버킷 확인
            bucket = self.token_buckets.get(endpoint)
            if bucket:
                wait_time = bucket.get_wait_time(requests)
                if wait_time > 0:
                    self.logger.debug(f"Endpoint {endpoint} rate limit wait: {wait_time:.2f}s")
                    self.stats["total_wait_time"] += wait_time
                    await asyncio.sleep(wait_time)

                if not await bucket.consume(requests):
                    self.stats["rejected_requests"] += 1
                    return False

            # 슬라이딩 윈도우 확인
            counters = self.sliding_counters.get(endpoint, {})
            for window_type, counter in counters.items():
                if not await counter.allow_request():
                    self.logger.warning(f"Rate limit exceeded for {endpoint} ({window_type})")
                    self.stats["rejected_requests"] += 1
                    return False

        self.stats["accepted_requests"] += 1
        return True

    async def try_acquire(self, endpoint: str, requests: int = 1) -> bool:
        """속도 제한 확인 (논블로킹)"""
        self.stats["total_requests"] += 1

        # 글로벌 제한 확인
        if self.global_bucket:
            if not await self.global_bucket.consume(requests):
                self.stats["rejected_requests"] += 1
                return False

        # 엔드포인트별 제한 확인
        if endpoint in self.limits:
            # 토큰 버킷 확인
            bucket = self.token_buckets.get(endpoint)
            if bucket and not await bucket.consume(requests):
                self.stats["rejected_requests"] += 1
                return False

            # 슬라이딩 윈도우 확인
            counters = self.sliding_counters.get(endpoint, {})
            for counter in counters.values():
                if not await counter.allow_request():
                    self.stats["rejected_requests"] += 1
                    return False

        self.stats["accepted_requests"] += 1
        return True

    def get_rate_limit_status(self, endpoint: str) -> Optional[RateLimitStatus]:
        """속도 제한 상태 조회"""
        if endpoint not in self.limits:
            return None

        rate_limit = self.limits[endpoint]
        bucket = self.token_buckets.get(endpoint)
        counters = self.sliding_counters.get(endpoint, {})

        # 분단위 카운터 기준으로 상태 계산
        minute_counter = counters.get("minute")
        if minute_counter:
            current_requests = minute_counter.get_current_count()
            remaining = max(0, rate_limit.requests_per_minute - current_requests)
            reset_time = datetime.fromtimestamp(minute_counter.get_reset_time())
        else:
            current_requests = 0
            remaining = rate_limit.requests_per_minute
            reset_time = datetime.now() + timedelta(minutes=1)

        return RateLimitStatus(
            endpoint=endpoint,
            remaining_requests=remaining,
            reset_time=reset_time,
            current_requests=current_requests,
            limit=rate_limit.requests_per_minute
        )

    def get_all_statuses(self) -> Dict[str, RateLimitStatus]:
        """모든 엔드포인트의 속도 제한 상태"""
        statuses = {}
        for endpoint in self.limits.keys():
            status = self.get_rate_limit_status(endpoint)
            if status:
                statuses[endpoint] = status
        return statuses

    def get_wait_time(self, endpoint: str, requests: int = 1) -> float:
        """예상 대기 시간 계산"""
        max_wait_time = 0.0

        # 글로벌 대기 시간
        if self.global_bucket:
            max_wait_time = max(max_wait_time, self.global_bucket.get_wait_time(requests))

        # 엔드포인트별 대기 시간
        if endpoint in self.token_buckets:
            bucket = self.token_buckets[endpoint]
            max_wait_time = max(max_wait_time, bucket.get_wait_time(requests))

        return max_wait_time

    def reset_endpoint(self, endpoint: str) -> None:
        """엔드포인트 속도 제한 리셋"""
        if endpoint in self.token_buckets:
            bucket = self.token_buckets[endpoint]
            bucket.tokens = float(bucket.capacity)
            bucket.last_refill = time.time()

        if endpoint in self.sliding_counters:
            for counter in self.sliding_counters[endpoint].values():
                counter.requests.clear()

        self.logger.info(f"Rate limit reset for endpoint: {endpoint}")

    def reset_all(self) -> None:
        """모든 속도 제한 리셋"""
        for endpoint in self.limits.keys():
            self.reset_endpoint(endpoint)

        if self.global_bucket:
            self.global_bucket.tokens = float(self.global_bucket.capacity)
            self.global_bucket.last_refill = time.time()

        self.logger.info("All rate limits reset")

    def get_stats(self) -> Dict[str, any]:
        """통계 정보 반환"""
        success_rate = (
            self.stats["accepted_requests"] / self.stats["total_requests"]
            if self.stats["total_requests"] > 0 else 0
        )

        avg_wait_time = (
            self.stats["total_wait_time"] / self.stats["total_requests"]
            if self.stats["total_requests"] > 0 else 0
        )

        return {
            **self.stats,
            "success_rate": success_rate,
            "average_wait_time": avg_wait_time,
            "configured_endpoints": len(self.limits),
            "active_buckets": len(self.token_buckets),
            "has_global_limit": self.global_bucket is not None
        }


# 팩토리 함수들
def create_rate_limit(
    requests_per_minute: int,
    requests_per_hour: Optional[int] = None,
    requests_per_day: Optional[int] = None,
    burst_limit: Optional[int] = None
) -> RateLimit:
    """속도 제한 설정 생성 헬퍼"""
    return RateLimit(
        requests_per_minute=requests_per_minute,
        requests_per_hour=requests_per_hour,
        requests_per_day=requests_per_day,
        burst_limit=burst_limit
    )


def create_common_rate_limits() -> Dict[str, RateLimit]:
    """일반적인 속도 제한 설정들"""
    return {
        "strict": create_rate_limit(10, 100, 1000),
        "moderate": create_rate_limit(60, 1000, 10000),
        "generous": create_rate_limit(300, 5000, 50000),
        "development": create_rate_limit(1000, 10000, 100000)
    }