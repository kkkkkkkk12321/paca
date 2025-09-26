"""
Circuit Breaker Pattern Implementation
서비스 장애 전파 방지를 위한 회로 차단기 패턴
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any
from datetime import datetime, timedelta
from enum import Enum

from ...core.utils.logger import PacaLogger


class CircuitState(Enum):
    """회로 차단기 상태"""
    CLOSED = "closed"      # 정상 상태
    OPEN = "open"          # 차단 상태
    HALF_OPEN = "half_open"  # 반개방 상태


@dataclass
class CircuitBreakerConfig:
    """회로 차단기 설정"""
    failure_threshold: int = 5           # 실패 임계값
    success_threshold: int = 3           # 반개방 상태에서 성공 임계값
    timeout: float = 60.0               # 개방 상태 유지 시간 (초)
    window_size: int = 10               # 슬라이딩 윈도우 크기
    half_open_max_calls: int = 3        # 반개방 상태에서 최대 호출 수


@dataclass
class CircuitBreakerStats:
    """회로 차단기 통계"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    state_changes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None


class CircuitBreakerError(Exception):
    """회로 차단기 예외"""
    pass


class CircuitBreaker:
    """회로 차단기 구현"""

    def __init__(self):
        self.logger = PacaLogger("CircuitBreaker")

        # 서비스별 회로 차단기
        self.breakers: Dict[str, Dict] = {}

        # 글로벌 통계
        self.global_stats = CircuitBreakerStats()

    def create_breaker(
        self,
        service_name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> None:
        """회로 차단기 생성"""
        if config is None:
            config = CircuitBreakerConfig()

        self.breakers[service_name] = {
            "config": config,
            "state": CircuitState.CLOSED,
            "stats": CircuitBreakerStats(),
            "failure_count": 0,
            "success_count": 0,
            "last_failure_time": None,
            "last_success_time": None,
            "failure_window": [],  # 최근 실패 시간들
            "half_open_calls": 0,
            "lock": asyncio.Lock()
        }

        self.logger.info(f"Circuit breaker created for service: {service_name}")

    def get_breaker(self, service_name: str) -> Optional[Dict]:
        """회로 차단기 조회"""
        return self.breakers.get(service_name)

    async def can_execute(self, service_name: str) -> bool:
        """실행 가능 여부 확인"""
        if service_name not in self.breakers:
            self.create_breaker(service_name)

        breaker = self.breakers[service_name]
        config = breaker["config"]

        async with breaker["lock"]:
            current_time = time.time()
            current_state = breaker["state"]

            if current_state == CircuitState.CLOSED:
                # 정상 상태: 항상 실행 가능
                return True

            elif current_state == CircuitState.OPEN:
                # 차단 상태: 타임아웃 확인
                if breaker["last_failure_time"]:
                    time_since_failure = current_time - breaker["last_failure_time"]
                    if time_since_failure >= config.timeout:
                        # 반개방 상태로 전환
                        await self._change_state(service_name, CircuitState.HALF_OPEN)
                        return True
                return False

            elif current_state == CircuitState.HALF_OPEN:
                # 반개방 상태: 제한된 호출만 허용
                if breaker["half_open_calls"] < config.half_open_max_calls:
                    breaker["half_open_calls"] += 1
                    return True
                return False

        return False

    async def record_success(self, service_name: str) -> None:
        """성공 기록"""
        if service_name not in self.breakers:
            return

        breaker = self.breakers[service_name]
        config = breaker["config"]

        async with breaker["lock"]:
            current_time = time.time()
            breaker["last_success_time"] = current_time
            breaker["stats"].successful_requests += 1
            breaker["stats"].last_success_time = datetime.now()

            current_state = breaker["state"]

            if current_state == CircuitState.HALF_OPEN:
                breaker["success_count"] += 1
                if breaker["success_count"] >= config.success_threshold:
                    # 정상 상태로 복구
                    await self._change_state(service_name, CircuitState.CLOSED)
                    breaker["failure_count"] = 0
                    breaker["success_count"] = 0
                    breaker["half_open_calls"] = 0
                    breaker["failure_window"].clear()

            elif current_state == CircuitState.CLOSED:
                # 실패 카운트 감소 (성공 시 부분적 복구)
                if breaker["failure_count"] > 0:
                    breaker["failure_count"] = max(0, breaker["failure_count"] - 1)

        self.global_stats.successful_requests += 1
        self.global_stats.last_success_time = datetime.now()

    async def record_failure(self, service_name: str) -> None:
        """실패 기록"""
        if service_name not in self.breakers:
            self.create_breaker(service_name)

        breaker = self.breakers[service_name]
        config = breaker["config"]

        async with breaker["lock"]:
            current_time = time.time()
            breaker["last_failure_time"] = current_time
            breaker["stats"].failed_requests += 1
            breaker["stats"].last_failure_time = datetime.now()

            # 슬라이딩 윈도우 업데이트
            breaker["failure_window"].append(current_time)
            window_start = current_time - 60.0  # 1분 윈도우
            breaker["failure_window"] = [
                t for t in breaker["failure_window"]
                if t >= window_start
            ]

            current_state = breaker["state"]

            if current_state == CircuitState.CLOSED:
                breaker["failure_count"] += 1
                if breaker["failure_count"] >= config.failure_threshold:
                    # 차단 상태로 전환
                    await self._change_state(service_name, CircuitState.OPEN)

            elif current_state == CircuitState.HALF_OPEN:
                # 반개방 상태에서 실패 시 즉시 차단 상태로
                await self._change_state(service_name, CircuitState.OPEN)
                breaker["success_count"] = 0
                breaker["half_open_calls"] = 0

        self.global_stats.failed_requests += 1
        self.global_stats.last_failure_time = datetime.now()

    async def record_rejection(self, service_name: str) -> None:
        """거부 기록"""
        if service_name not in self.breakers:
            return

        breaker = self.breakers[service_name]
        breaker["stats"].rejected_requests += 1
        self.global_stats.rejected_requests += 1

    async def _change_state(self, service_name: str, new_state: CircuitState) -> None:
        """상태 변경"""
        breaker = self.breakers[service_name]
        old_state = breaker["state"]

        if old_state != new_state:
            breaker["state"] = new_state
            breaker["stats"].state_changes += 1
            self.global_stats.state_changes += 1

            self.logger.info(
                f"Circuit breaker state changed for {service_name}: "
                f"{old_state.value} -> {new_state.value}"
            )

    async def execute_with_breaker(
        self,
        service_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """회로 차단기와 함께 함수 실행"""
        # 실행 가능 여부 확인
        if not await self.can_execute(service_name):
            await self.record_rejection(service_name)
            raise CircuitBreakerError(f"Circuit breaker is open for service: {service_name}")

        breaker = self.breakers[service_name]
        breaker["stats"].total_requests += 1
        self.global_stats.total_requests += 1

        try:
            # 함수 실행
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # 성공 기록
            await self.record_success(service_name)
            return result

        except Exception as e:
            # 실패 기록
            await self.record_failure(service_name)
            raise e

    def get_state(self, service_name: str) -> Optional[CircuitState]:
        """현재 상태 조회"""
        breaker = self.get_breaker(service_name)
        if breaker:
            return breaker["state"]
        return None

    def get_stats(self, service_name: str) -> Optional[Dict[str, Any]]:
        """통계 정보 조회"""
        breaker = self.get_breaker(service_name)
        if not breaker:
            return None

        stats = breaker["stats"]
        config = breaker["config"]

        # 성공률 계산
        total_requests = stats.total_requests
        success_rate = (
            stats.successful_requests / total_requests
            if total_requests > 0 else 0
        )

        # 실패율 계산 (최근 윈도우 기준)
        current_time = time.time()
        recent_failures = len([
            t for t in breaker["failure_window"]
            if current_time - t <= 60.0
        ])

        return {
            "service_name": service_name,
            "state": breaker["state"].value,
            "total_requests": stats.total_requests,
            "successful_requests": stats.successful_requests,
            "failed_requests": stats.failed_requests,
            "rejected_requests": stats.rejected_requests,
            "success_rate": success_rate,
            "failure_count": breaker["failure_count"],
            "recent_failures": recent_failures,
            "state_changes": stats.state_changes,
            "last_failure_time": stats.last_failure_time.isoformat() if stats.last_failure_time else None,
            "last_success_time": stats.last_success_time.isoformat() if stats.last_success_time else None,
            "config": {
                "failure_threshold": config.failure_threshold,
                "success_threshold": config.success_threshold,
                "timeout": config.timeout,
                "window_size": config.window_size,
                "half_open_max_calls": config.half_open_max_calls
            }
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """모든 회로 차단기 통계"""
        service_stats = {}
        for service_name in self.breakers.keys():
            service_stats[service_name] = self.get_stats(service_name)

        # 글로벌 통계
        total_requests = self.global_stats.total_requests
        global_success_rate = (
            self.global_stats.successful_requests / total_requests
            if total_requests > 0 else 0
        )

        return {
            "global_stats": {
                "total_requests": self.global_stats.total_requests,
                "successful_requests": self.global_stats.successful_requests,
                "failed_requests": self.global_stats.failed_requests,
                "rejected_requests": self.global_stats.rejected_requests,
                "success_rate": global_success_rate,
                "state_changes": self.global_stats.state_changes,
                "last_failure_time": self.global_stats.last_failure_time.isoformat() if self.global_stats.last_failure_time else None,
                "last_success_time": self.global_stats.last_success_time.isoformat() if self.global_stats.last_success_time else None,
                "active_services": len(self.breakers)
            },
            "services": service_stats
        }

    async def reset_breaker(self, service_name: str) -> bool:
        """회로 차단기 리셋"""
        if service_name not in self.breakers:
            return False

        breaker = self.breakers[service_name]

        async with breaker["lock"]:
            breaker["state"] = CircuitState.CLOSED
            breaker["failure_count"] = 0
            breaker["success_count"] = 0
            breaker["half_open_calls"] = 0
            breaker["failure_window"].clear()
            breaker["last_failure_time"] = None
            breaker["stats"] = CircuitBreakerStats()

        self.logger.info(f"Circuit breaker reset for service: {service_name}")
        return True

    async def reset_all_breakers(self) -> None:
        """모든 회로 차단기 리셋"""
        for service_name in list(self.breakers.keys()):
            await self.reset_breaker(service_name)

        self.global_stats = CircuitBreakerStats()
        self.logger.info("All circuit breakers reset")

    def remove_breaker(self, service_name: str) -> bool:
        """회로 차단기 제거"""
        if service_name in self.breakers:
            del self.breakers[service_name]
            self.logger.info(f"Circuit breaker removed for service: {service_name}")
            return True
        return False

    async def force_open(self, service_name: str) -> bool:
        """강제로 차단 상태로 변경"""
        if service_name not in self.breakers:
            return False

        await self._change_state(service_name, CircuitState.OPEN)
        return True

    async def force_close(self, service_name: str) -> bool:
        """강제로 정상 상태로 변경"""
        if service_name not in self.breakers:
            return False

        breaker = self.breakers[service_name]
        async with breaker["lock"]:
            await self._change_state(service_name, CircuitState.CLOSED)
            breaker["failure_count"] = 0
            breaker["success_count"] = 0
            breaker["half_open_calls"] = 0

        return True


# 팩토리 함수들
def create_circuit_breaker_config(
    failure_threshold: int = 5,
    success_threshold: int = 3,
    timeout: float = 60.0,
    window_size: int = 10,
    half_open_max_calls: int = 3
) -> CircuitBreakerConfig:
    """회로 차단기 설정 생성"""
    return CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        success_threshold=success_threshold,
        timeout=timeout,
        window_size=window_size,
        half_open_max_calls=half_open_max_calls
    )


def create_common_configs() -> Dict[str, CircuitBreakerConfig]:
    """일반적인 회로 차단기 설정들"""
    return {
        "strict": create_circuit_breaker_config(3, 2, 30.0, 5, 2),
        "moderate": create_circuit_breaker_config(5, 3, 60.0, 10, 3),
        "lenient": create_circuit_breaker_config(10, 5, 120.0, 20, 5),
        "development": create_circuit_breaker_config(20, 10, 300.0, 50, 10)
    }