"""
Connection Pool
데이터베이스 연결 풀 관리
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, AsyncContextManager
from datetime import datetime, timedelta
from enum import Enum
import weakref

from ...core.types import Result
from ...core.utils.logger import PacaLogger
from .sql_connector import SQLConnector, DatabaseConfig
from .nosql_connector import NoSQLConnector, NoSQLConfig


class ConnectionStatus(Enum):
    """연결 상태"""
    AVAILABLE = "available"
    IN_USE = "in_use"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class PoolConfig:
    """연결 풀 설정"""
    min_connections: int = 5
    max_connections: int = 20
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0  # 5분
    max_lifetime: float = 3600.0  # 1시간
    health_check_interval: float = 60.0  # 1분
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class ConnectionInfo:
    """연결 정보"""
    connection: Union[SQLConnector, NoSQLConnector]
    status: ConnectionStatus
    created_at: datetime
    last_used: datetime
    use_count: int = 0
    pool_id: Optional[str] = None


class ConnectionPool:
    """데이터베이스 연결 풀"""

    def __init__(
        self,
        db_config: Union[DatabaseConfig, NoSQLConfig],
        pool_config: Optional[PoolConfig] = None
    ):
        self.db_config = db_config
        self.pool_config = pool_config or PoolConfig()
        self.logger = PacaLogger("ConnectionPool")

        # 연결 풀
        self.connections: Dict[str, ConnectionInfo] = {}
        self.available_connections: asyncio.Queue = asyncio.Queue()
        self.connection_semaphore = asyncio.Semaphore(self.pool_config.max_connections)

        # 풀 상태
        self.is_initialized = False
        self.is_closing = False

        # 통계
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "available_connections": 0,
            "failed_connections": 0,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_wait_time": 0.0,
            "total_wait_time": 0.0
        }

        # 헬스체크 태스크
        self.health_check_task: Optional[asyncio.Task] = None

        # 연결 ID 생성기
        self._connection_counter = 0

    async def initialize(self) -> Result[bool]:
        """연결 풀 초기화"""
        try:
            # 최소 연결 수만큼 미리 생성
            for _ in range(self.pool_config.min_connections):
                connection_result = await self._create_connection()
                if not connection_result.is_success:
                    self.logger.warning(f"Failed to create initial connection: {connection_result.error}")

            # 헬스체크 시작
            self.health_check_task = asyncio.create_task(self._health_check_loop())

            self.is_initialized = True
            self.logger.info(f"Connection pool initialized with {len(self.connections)} connections")
            return Result(True, True)

        except Exception as e:
            self.logger.error(f"Pool initialization failed: {str(e)}")
            return Result(False, False, str(e))

    async def _create_connection(self) -> Result[ConnectionInfo]:
        """새 연결 생성"""
        try:
            self._connection_counter += 1
            connection_id = f"conn_{self._connection_counter}"

            # SQL vs NoSQL에 따라 적절한 커넥터 생성
            if isinstance(self.db_config, DatabaseConfig):
                connector = SQLConnector(self.db_config)
            else:
                connector = NoSQLConnector(self.db_config)

            # 연결 시도
            connect_result = await connector.connect()
            if not connect_result.is_success:
                return Result(False, None, connect_result.error)

            # 연결 정보 생성
            connection_info = ConnectionInfo(
                connection=connector,
                status=ConnectionStatus.AVAILABLE,
                created_at=datetime.now(),
                last_used=datetime.now(),
                pool_id=connection_id
            )

            # 풀에 추가
            self.connections[connection_id] = connection_info
            await self.available_connections.put(connection_id)

            self.stats["total_connections"] += 1
            self._update_connection_stats()

            self.logger.debug(f"Created new connection: {connection_id}")
            return Result(True, connection_info)

        except Exception as e:
            return Result(False, None, f"Connection creation failed: {str(e)}")

    async def get_connection(self, timeout: Optional[float] = None) -> AsyncContextManager[Union[SQLConnector, NoSQLConnector]]:
        """연결 획득 (컨텍스트 매니저)"""
        return ConnectionContextManager(self, timeout or self.pool_config.connection_timeout)

    async def _acquire_connection(self, timeout: float) -> Result[str]:
        """연결 획득 (내부용)"""
        if not self.is_initialized:
            return Result(False, None, "Pool not initialized")

        if self.is_closing:
            return Result(False, None, "Pool is closing")

        start_time = time.time()
        self.stats["total_requests"] += 1

        try:
            # 세마포어로 최대 연결 수 제한
            async with self.connection_semaphore:
                # 사용 가능한 연결 대기
                try:
                    connection_id = await asyncio.wait_for(
                        self.available_connections.get(),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    # 타임아웃 시 새 연결 생성 시도
                    if len(self.connections) < self.pool_config.max_connections:
                        connection_result = await self._create_connection()
                        if connection_result.is_success:
                            connection_id = connection_result.data.pool_id
                        else:
                            self.stats["failed_requests"] += 1
                            return Result(False, None, "Connection timeout and creation failed")
                    else:
                        self.stats["failed_requests"] += 1
                        return Result(False, None, "Connection timeout")

                # 연결 상태 확인
                connection_info = self.connections.get(connection_id)
                if not connection_info:
                    return Result(False, None, "Connection not found")

                # 연결 상태 업데이트
                connection_info.status = ConnectionStatus.IN_USE
                connection_info.last_used = datetime.now()
                connection_info.use_count += 1

                wait_time = time.time() - start_time
                self.stats["total_wait_time"] += wait_time
                self.stats["average_wait_time"] = (
                    self.stats["total_wait_time"] / self.stats["total_requests"]
                )
                self.stats["successful_requests"] += 1

                self._update_connection_stats()
                return Result(True, connection_id)

        except Exception as e:
            self.stats["failed_requests"] += 1
            return Result(False, None, f"Connection acquisition failed: {str(e)}")

    async def _release_connection(self, connection_id: str) -> None:
        """연결 반환 (내부용)"""
        try:
            connection_info = self.connections.get(connection_id)
            if not connection_info:
                return

            # 연결 상태 확인
            if await self._is_connection_valid(connection_info):
                connection_info.status = ConnectionStatus.AVAILABLE
                await self.available_connections.put(connection_id)
            else:
                # 유효하지 않은 연결은 제거
                await self._remove_connection(connection_id)

            self._update_connection_stats()

        except Exception as e:
            self.logger.error(f"Connection release failed: {str(e)}")

    async def _is_connection_valid(self, connection_info: ConnectionInfo) -> bool:
        """연결 유효성 확인"""
        try:
            # 수명 확인
            age = datetime.now() - connection_info.created_at
            if age.total_seconds() > self.pool_config.max_lifetime:
                return False

            # 유휴 시간 확인
            idle_time = datetime.now() - connection_info.last_used
            if idle_time.total_seconds() > self.pool_config.idle_timeout:
                return False

            # 헬스체크
            health_result = await connection_info.connection.health_check()
            return health_result.is_success

        except Exception:
            return False

    async def _remove_connection(self, connection_id: str) -> None:
        """연결 제거"""
        try:
            connection_info = self.connections.get(connection_id)
            if connection_info:
                await connection_info.connection.disconnect()
                connection_info.status = ConnectionStatus.FAILED
                del self.connections[connection_id]

                self.stats["failed_connections"] += 1
                self._update_connection_stats()

                self.logger.debug(f"Removed connection: {connection_id}")

        except Exception as e:
            self.logger.error(f"Connection removal failed: {str(e)}")

    def _update_connection_stats(self) -> None:
        """연결 통계 업데이트"""
        active_count = sum(
            1 for conn in self.connections.values()
            if conn.status == ConnectionStatus.IN_USE
        )
        available_count = sum(
            1 for conn in self.connections.values()
            if conn.status == ConnectionStatus.AVAILABLE
        )

        self.stats["active_connections"] = active_count
        self.stats["available_connections"] = available_count

    async def _health_check_loop(self) -> None:
        """헬스체크 루프"""
        while not self.is_closing:
            try:
                await asyncio.sleep(self.pool_config.health_check_interval)

                # 모든 연결 헬스체크
                invalid_connections = []
                for connection_id, connection_info in self.connections.items():
                    if connection_info.status == ConnectionStatus.AVAILABLE:
                        if not await self._is_connection_valid(connection_info):
                            invalid_connections.append(connection_id)

                # 유효하지 않은 연결 제거
                for connection_id in invalid_connections:
                    await self._remove_connection(connection_id)

                # 최소 연결 수 유지
                current_count = len(self.connections)
                if current_count < self.pool_config.min_connections:
                    needed = self.pool_config.min_connections - current_count
                    for _ in range(needed):
                        await self._create_connection()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {str(e)}")

    async def execute_query(
        self,
        query: str,
        params: Optional[Any] = None,
        timeout: Optional[float] = None
    ) -> Result[Any]:
        """연결 풀을 사용한 쿼리 실행"""
        async with self.get_connection(timeout) as connection:
            if isinstance(connection, SQLConnector):
                return await connection.execute_query(query, params)
            else:
                # NoSQL의 경우 query를 operation으로 해석
                operation_parts = query.split(':', 2)
                if len(operation_parts) >= 2:
                    operation = operation_parts[0]
                    collection_or_key = operation_parts[1]
                    filter_query = params if isinstance(params, dict) else None
                    return await connection.execute_operation(operation, collection_or_key, params, filter_query)
                else:
                    return Result(False, None, "Invalid NoSQL query format")

    def get_pool_stats(self) -> Dict[str, Any]:
        """풀 통계 정보"""
        total_connections = len(self.connections)

        status_distribution = {}
        for status in ConnectionStatus:
            count = sum(1 for conn in self.connections.values() if conn.status == status)
            status_distribution[status.value] = count

        return {
            **self.stats,
            "pool_config": {
                "min_connections": self.pool_config.min_connections,
                "max_connections": self.pool_config.max_connections,
                "connection_timeout": self.pool_config.connection_timeout,
                "idle_timeout": self.pool_config.idle_timeout,
                "max_lifetime": self.pool_config.max_lifetime
            },
            "status_distribution": status_distribution,
            "pool_utilization": (
                self.stats["active_connections"] / self.pool_config.max_connections
                if self.pool_config.max_connections > 0 else 0
            ),
            "is_initialized": self.is_initialized,
            "is_closing": self.is_closing
        }

    async def close(self) -> Result[bool]:
        """연결 풀 종료"""
        try:
            self.is_closing = True

            # 헬스체크 태스크 종료
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass

            # 모든 연결 종료
            for connection_id in list(self.connections.keys()):
                await self._remove_connection(connection_id)

            self.connections.clear()

            # 큐 정리
            while not self.available_connections.empty():
                try:
                    self.available_connections.get_nowait()
                except asyncio.QueueEmpty:
                    break

            self.is_initialized = False
            self.logger.info("Connection pool closed")
            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"Pool close failed: {str(e)}")


class ConnectionContextManager:
    """연결 컨텍스트 매니저"""

    def __init__(self, pool: ConnectionPool, timeout: float):
        self.pool = pool
        self.timeout = timeout
        self.connection_id: Optional[str] = None
        self.connection: Optional[Union[SQLConnector, NoSQLConnector]] = None

    async def __aenter__(self) -> Union[SQLConnector, NoSQLConnector]:
        result = await self.pool._acquire_connection(self.timeout)
        if not result.is_success:
            raise Exception(f"Failed to acquire connection: {result.error}")

        self.connection_id = result.data
        connection_info = self.pool.connections[self.connection_id]
        self.connection = connection_info.connection

        return self.connection

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.connection_id:
            await self.pool._release_connection(self.connection_id)


# 팩토리 함수들
def create_pool_config(
    min_connections: int = 5,
    max_connections: int = 20,
    connection_timeout: float = 30.0,
    idle_timeout: float = 300.0,
    max_lifetime: float = 3600.0,
    **kwargs
) -> PoolConfig:
    """연결 풀 설정 생성"""
    return PoolConfig(
        min_connections=min_connections,
        max_connections=max_connections,
        connection_timeout=connection_timeout,
        idle_timeout=idle_timeout,
        max_lifetime=max_lifetime,
        **kwargs
    )


async def create_sql_pool(
    db_config: DatabaseConfig,
    pool_config: Optional[PoolConfig] = None
) -> ConnectionPool:
    """SQL 연결 풀 생성"""
    pool = ConnectionPool(db_config, pool_config)
    await pool.initialize()
    return pool


async def create_nosql_pool(
    db_config: NoSQLConfig,
    pool_config: Optional[PoolConfig] = None
) -> ConnectionPool:
    """NoSQL 연결 풀 생성"""
    pool = ConnectionPool(db_config, pool_config)
    await pool.initialize()
    return pool