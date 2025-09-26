"""
SQL Database Connector
SQLite, PostgreSQL, MySQL 등 SQL 데이터베이스 연동
"""

import asyncio
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from enum import Enum
import json

try:
    import aiosqlite
except ImportError:
    aiosqlite = None

try:
    import asyncpg
except ImportError:
    asyncpg = None

try:
    import aiomysql
except ImportError:
    aiomysql = None

from ...core.types import Result
from ...core.utils.logger import PacaLogger


class DatabaseType(Enum):
    """데이터베이스 타입"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


@dataclass
class DatabaseConfig:
    """데이터베이스 설정"""
    db_type: DatabaseType
    database: str
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """쿼리 결과"""
    rows: List[Dict[str, Any]]
    affected_rows: int
    execution_time: float
    query: str
    error: Optional[str] = None


class SQLConnector:
    """SQL 데이터베이스 커넥터"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.logger = PacaLogger("SQLConnector")
        self.connection = None
        self.is_connected = False

        # 통계
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_execution_time": 0.0,
            "connection_count": 0,
            "last_query_time": None
        }

    async def connect(self) -> Result[bool]:
        """데이터베이스 연결"""
        try:
            if self.config.db_type == DatabaseType.SQLITE:
                return await self._connect_sqlite()
            elif self.config.db_type == DatabaseType.POSTGRESQL:
                return await self._connect_postgresql()
            elif self.config.db_type == DatabaseType.MYSQL:
                return await self._connect_mysql()
            else:
                return Result(False, False, f"Unsupported database type: {self.config.db_type}")

        except Exception as e:
            self.logger.error(f"Database connection failed: {str(e)}")
            return Result(False, False, str(e))

    async def _connect_sqlite(self) -> Result[bool]:
        """SQLite 연결"""
        if aiosqlite is None:
            return Result(False, False, "aiosqlite not installed. Run: pip install aiosqlite")

        try:
            self.connection = await aiosqlite.connect(self.config.database)
            self.connection.row_factory = aiosqlite.Row
            self.is_connected = True
            self.stats["connection_count"] += 1

            self.logger.info(f"Connected to SQLite database: {self.config.database}")
            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"SQLite connection failed: {str(e)}")

    async def _connect_postgresql(self) -> Result[bool]:
        """PostgreSQL 연결"""
        if asyncpg is None:
            return Result(False, False, "asyncpg not installed. Run: pip install asyncpg")

        try:
            connection_params = {
                "database": self.config.database,
                "host": self.config.host or "localhost",
                "port": self.config.port or 5432,
                "user": self.config.username,
                "password": self.config.password
            }

            self.connection = await asyncpg.connect(**connection_params)
            self.is_connected = True
            self.stats["connection_count"] += 1

            self.logger.info(f"Connected to PostgreSQL database: {self.config.database}")
            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"PostgreSQL connection failed: {str(e)}")

    async def _connect_mysql(self) -> Result[bool]:
        """MySQL 연결"""
        if aiomysql is None:
            return Result(False, False, "aiomysql not installed. Run: pip install aiomysql")

        try:
            connection_params = {
                "db": self.config.database,
                "host": self.config.host or "localhost",
                "port": self.config.port or 3306,
                "user": self.config.username,
                "password": self.config.password,
                "autocommit": True
            }

            self.connection = await aiomysql.connect(**connection_params)
            self.is_connected = True
            self.stats["connection_count"] += 1

            self.logger.info(f"Connected to MySQL database: {self.config.database}")
            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"MySQL connection failed: {str(e)}")

    async def disconnect(self) -> Result[bool]:
        """데이터베이스 연결 해제"""
        try:
            if self.connection:
                if self.config.db_type == DatabaseType.SQLITE:
                    await self.connection.close()
                elif self.config.db_type == DatabaseType.POSTGRESQL:
                    await self.connection.close()
                elif self.config.db_type == DatabaseType.MYSQL:
                    self.connection.close()

                self.connection = None
                self.is_connected = False

            self.logger.info("Database disconnected")
            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"Disconnect failed: {str(e)}")

    async def execute_query(
        self,
        query: str,
        params: Optional[Union[List, Dict]] = None
    ) -> Result[QueryResult]:
        """쿼리 실행"""
        if not self.is_connected:
            connect_result = await self.connect()
            if not connect_result.is_success:
                return Result(False, None, connect_result.error)

        start_time = asyncio.get_event_loop().time()
        self.stats["total_queries"] += 1
        self.stats["last_query_time"] = datetime.now()

        try:
            if self.config.db_type == DatabaseType.SQLITE:
                result = await self._execute_sqlite_query(query, params)
            elif self.config.db_type == DatabaseType.POSTGRESQL:
                result = await self._execute_postgresql_query(query, params)
            elif self.config.db_type == DatabaseType.MYSQL:
                result = await self._execute_mysql_query(query, params)
            else:
                return Result(False, None, f"Unsupported database type: {self.config.db_type}")

            execution_time = asyncio.get_event_loop().time() - start_time
            result.execution_time = execution_time
            result.query = query

            self.stats["successful_queries"] += 1
            self.stats["total_execution_time"] += execution_time

            return Result(True, result)

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            self.stats["failed_queries"] += 1

            error_result = QueryResult(
                rows=[],
                affected_rows=0,
                execution_time=execution_time,
                query=query,
                error=str(e)
            )

            self.logger.error(f"Query execution failed: {str(e)}")
            return Result(False, error_result, str(e))

    async def _execute_sqlite_query(
        self,
        query: str,
        params: Optional[Union[List, Dict]] = None
    ) -> QueryResult:
        """SQLite 쿼리 실행"""
        cursor = await self.connection.execute(query, params or [])

        # SELECT 쿼리인지 확인
        is_select = query.strip().upper().startswith("SELECT")

        if is_select:
            rows = await cursor.fetchall()
            result_rows = [dict(row) for row in rows]
            affected_rows = len(result_rows)
        else:
            await self.connection.commit()
            result_rows = []
            affected_rows = cursor.rowcount

        await cursor.close()

        return QueryResult(
            rows=result_rows,
            affected_rows=affected_rows,
            execution_time=0.0,  # 실행 시간은 상위에서 계산
            query=query
        )

    async def _execute_postgresql_query(
        self,
        query: str,
        params: Optional[Union[List, Dict]] = None
    ) -> QueryResult:
        """PostgreSQL 쿼리 실행"""
        is_select = query.strip().upper().startswith("SELECT")

        if is_select:
            rows = await self.connection.fetch(query, *(params or []))
            result_rows = [dict(row) for row in rows]
            affected_rows = len(result_rows)
        else:
            result = await self.connection.execute(query, *(params or []))
            result_rows = []
            # PostgreSQL execute 결과에서 affected rows 추출
            if isinstance(result, str) and result.startswith(('INSERT', 'UPDATE', 'DELETE')):
                affected_rows = int(result.split()[-1]) if result.split()[-1].isdigit() else 0
            else:
                affected_rows = 0

        return QueryResult(
            rows=result_rows,
            affected_rows=affected_rows,
            execution_time=0.0,
            query=query
        )

    async def _execute_mysql_query(
        self,
        query: str,
        params: Optional[Union[List, Dict]] = None
    ) -> QueryResult:
        """MySQL 쿼리 실행"""
        async with self.connection.cursor() as cursor:
            await cursor.execute(query, params or [])

            is_select = query.strip().upper().startswith("SELECT")

            if is_select:
                rows = await cursor.fetchall()
                # 컬럼 이름 가져오기
                columns = [desc[0] for desc in cursor.description]
                result_rows = [dict(zip(columns, row)) for row in rows]
                affected_rows = len(result_rows)
            else:
                result_rows = []
                affected_rows = cursor.rowcount

        return QueryResult(
            rows=result_rows,
            affected_rows=affected_rows,
            execution_time=0.0,
            query=query
        )

    async def execute_many(
        self,
        query: str,
        params_list: List[Union[List, Dict]]
    ) -> Result[QueryResult]:
        """배치 쿼리 실행"""
        if not self.is_connected:
            connect_result = await self.connect()
            if not connect_result.is_success:
                return Result(False, None, connect_result.error)

        start_time = asyncio.get_event_loop().time()
        self.stats["total_queries"] += len(params_list)

        try:
            total_affected = 0

            if self.config.db_type == DatabaseType.SQLITE:
                cursor = await self.connection.executemany(query, params_list)
                total_affected = cursor.rowcount
                await self.connection.commit()
                await cursor.close()

            elif self.config.db_type == DatabaseType.POSTGRESQL:
                for params in params_list:
                    await self.connection.execute(query, *params)
                total_affected = len(params_list)

            elif self.config.db_type == DatabaseType.MYSQL:
                async with self.connection.cursor() as cursor:
                    await cursor.executemany(query, params_list)
                    total_affected = cursor.rowcount

            execution_time = asyncio.get_event_loop().time() - start_time
            self.stats["successful_queries"] += len(params_list)
            self.stats["total_execution_time"] += execution_time

            result = QueryResult(
                rows=[],
                affected_rows=total_affected,
                execution_time=execution_time,
                query=query
            )

            return Result(True, result)

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            self.stats["failed_queries"] += len(params_list)

            error_result = QueryResult(
                rows=[],
                affected_rows=0,
                execution_time=execution_time,
                query=query,
                error=str(e)
            )

            return Result(False, error_result, str(e))

    async def begin_transaction(self) -> Result[bool]:
        """트랜잭션 시작"""
        try:
            if self.config.db_type == DatabaseType.SQLITE:
                await self.connection.execute("BEGIN")
            elif self.config.db_type == DatabaseType.POSTGRESQL:
                await self.connection.execute("BEGIN")
            elif self.config.db_type == DatabaseType.MYSQL:
                await self.connection.begin()

            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"Transaction begin failed: {str(e)}")

    async def commit_transaction(self) -> Result[bool]:
        """트랜잭션 커밋"""
        try:
            if self.config.db_type == DatabaseType.SQLITE:
                await self.connection.commit()
            elif self.config.db_type == DatabaseType.POSTGRESQL:
                await self.connection.execute("COMMIT")
            elif self.config.db_type == DatabaseType.MYSQL:
                await self.connection.commit()

            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"Transaction commit failed: {str(e)}")

    async def rollback_transaction(self) -> Result[bool]:
        """트랜잭션 롤백"""
        try:
            if self.config.db_type == DatabaseType.SQLITE:
                await self.connection.rollback()
            elif self.config.db_type == DatabaseType.POSTGRESQL:
                await self.connection.execute("ROLLBACK")
            elif self.config.db_type == DatabaseType.MYSQL:
                await self.connection.rollback()

            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"Transaction rollback failed: {str(e)}")

    async def get_table_info(self, table_name: str) -> Result[List[Dict[str, Any]]]:
        """테이블 정보 조회"""
        try:
            if self.config.db_type == DatabaseType.SQLITE:
                query = f"PRAGMA table_info({table_name})"
            elif self.config.db_type == DatabaseType.POSTGRESQL:
                query = """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = $1
                ORDER BY ordinal_position
                """
            elif self.config.db_type == DatabaseType.MYSQL:
                query = f"DESCRIBE {table_name}"
            else:
                return Result(False, None, "Unsupported database type")

            params = [table_name] if self.config.db_type == DatabaseType.POSTGRESQL else None
            result = await self.execute_query(query, params)

            if result.is_success:
                return Result(True, result.data.rows)
            else:
                return Result(False, None, result.error)

        except Exception as e:
            return Result(False, None, f"Get table info failed: {str(e)}")

    async def list_tables(self) -> Result[List[str]]:
        """테이블 목록 조회"""
        try:
            if self.config.db_type == DatabaseType.SQLITE:
                query = "SELECT name FROM sqlite_master WHERE type='table'"
            elif self.config.db_type == DatabaseType.POSTGRESQL:
                query = """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                """
            elif self.config.db_type == DatabaseType.MYSQL:
                query = "SHOW TABLES"
            else:
                return Result(False, None, "Unsupported database type")

            result = await self.execute_query(query)

            if result.is_success:
                # 테이블 이름만 추출
                table_names = []
                for row in result.data.rows:
                    if self.config.db_type == DatabaseType.SQLITE:
                        table_names.append(row['name'])
                    elif self.config.db_type == DatabaseType.POSTGRESQL:
                        table_names.append(row['table_name'])
                    elif self.config.db_type == DatabaseType.MYSQL:
                        # MySQL SHOW TABLES의 첫 번째 컬럼
                        table_names.append(list(row.values())[0])

                return Result(True, table_names)
            else:
                return Result(False, None, result.error)

        except Exception as e:
            return Result(False, None, f"List tables failed: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        avg_execution_time = (
            self.stats["total_execution_time"] / self.stats["successful_queries"]
            if self.stats["successful_queries"] > 0 else 0
        )

        success_rate = (
            self.stats["successful_queries"] / self.stats["total_queries"]
            if self.stats["total_queries"] > 0 else 0
        )

        return {
            **self.stats,
            "average_execution_time": avg_execution_time,
            "success_rate": success_rate,
            "database_type": self.config.db_type.value,
            "is_connected": self.is_connected,
            "last_query_time": self.stats["last_query_time"].isoformat() if self.stats["last_query_time"] else None
        }

    async def health_check(self) -> Result[Dict[str, Any]]:
        """헬스 체크"""
        try:
            # 간단한 쿼리 실행
            if self.config.db_type == DatabaseType.SQLITE:
                test_query = "SELECT 1"
            elif self.config.db_type == DatabaseType.POSTGRESQL:
                test_query = "SELECT 1"
            elif self.config.db_type == DatabaseType.MYSQL:
                test_query = "SELECT 1"

            start_time = asyncio.get_event_loop().time()
            result = await self.execute_query(test_query)
            response_time = asyncio.get_event_loop().time() - start_time

            health_data = {
                "status": "healthy" if result.is_success else "unhealthy",
                "response_time": response_time,
                "is_connected": self.is_connected,
                "database_type": self.config.db_type.value,
                "stats": self.get_stats(),
                "timestamp": datetime.now().isoformat()
            }

            return Result(True, health_data)

        except Exception as e:
            return Result(False, None, f"Health check failed: {str(e)}")


# 팩토리 함수들
def create_sqlite_config(database: str, **options) -> DatabaseConfig:
    """SQLite 설정 생성"""
    return DatabaseConfig(
        db_type=DatabaseType.SQLITE,
        database=database,
        options=options
    )


def create_postgresql_config(
    database: str,
    host: str = "localhost",
    port: int = 5432,
    username: str = "",
    password: str = "",
    **options
) -> DatabaseConfig:
    """PostgreSQL 설정 생성"""
    return DatabaseConfig(
        db_type=DatabaseType.POSTGRESQL,
        database=database,
        host=host,
        port=port,
        username=username,
        password=password,
        options=options
    )


def create_mysql_config(
    database: str,
    host: str = "localhost",
    port: int = 3306,
    username: str = "",
    password: str = "",
    **options
) -> DatabaseConfig:
    """MySQL 설정 생성"""
    return DatabaseConfig(
        db_type=DatabaseType.MYSQL,
        database=database,
        host=host,
        port=port,
        username=username,
        password=password,
        options=options
    )