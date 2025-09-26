"""
NoSQL Database Connector
MongoDB, Redis 등 NoSQL 데이터베이스 연동
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

try:
    import motor.motor_asyncio
    from pymongo import errors as pymongo_errors
except ImportError:
    motor = None
    pymongo_errors = None

try:
    import aioredis
except ImportError:
    aioredis = None

from ...core.types import Result
from ...core.utils.logger import PacaLogger


class NoSQLType(Enum):
    """NoSQL 데이터베이스 타입"""
    MONGODB = "mongodb"
    REDIS = "redis"


@dataclass
class NoSQLConfig:
    """NoSQL 데이터베이스 설정"""
    db_type: NoSQLType
    host: str = "localhost"
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    connection_string: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NoSQLResult:
    """NoSQL 쿼리 결과"""
    data: Any
    affected_count: int
    execution_time: float
    operation: str
    error: Optional[str] = None


class NoSQLConnector:
    """NoSQL 데이터베이스 커넥터"""

    def __init__(self, config: NoSQLConfig):
        self.config = config
        self.logger = PacaLogger("NoSQLConnector")
        self.client = None
        self.database = None
        self.is_connected = False

        # 통계
        self.stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_execution_time": 0.0,
            "connection_count": 0,
            "last_operation_time": None
        }

    async def connect(self) -> Result[bool]:
        """데이터베이스 연결"""
        try:
            if self.config.db_type == NoSQLType.MONGODB:
                return await self._connect_mongodb()
            elif self.config.db_type == NoSQLType.REDIS:
                return await self._connect_redis()
            else:
                return Result(False, False, f"Unsupported NoSQL type: {self.config.db_type}")

        except Exception as e:
            self.logger.error(f"NoSQL connection failed: {str(e)}")
            return Result(False, False, str(e))

    async def _connect_mongodb(self) -> Result[bool]:
        """MongoDB 연결"""
        if motor is None:
            return Result(False, False, "motor not installed. Run: pip install motor")

        try:
            if self.config.connection_string:
                self.client = motor.motor_asyncio.AsyncIOMotorClient(self.config.connection_string)
            else:
                port = self.config.port or 27017
                if self.config.username and self.config.password:
                    uri = f"mongodb://{self.config.username}:{self.config.password}@{self.config.host}:{port}/"
                else:
                    uri = f"mongodb://{self.config.host}:{port}/"

                self.client = motor.motor_asyncio.AsyncIOMotorClient(uri, **self.config.options)

            # 연결 테스트
            await self.client.admin.command('ping')

            if self.config.database:
                self.database = self.client[self.config.database]

            self.is_connected = True
            self.stats["connection_count"] += 1

            self.logger.info(f"Connected to MongoDB: {self.config.host}:{self.config.port or 27017}")
            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"MongoDB connection failed: {str(e)}")

    async def _connect_redis(self) -> Result[bool]:
        """Redis 연결"""
        if aioredis is None:
            return Result(False, False, "aioredis not installed. Run: pip install aioredis")

        try:
            if self.config.connection_string:
                self.client = await aioredis.from_url(self.config.connection_string)
            else:
                port = self.config.port or 6379
                self.client = await aioredis.Redis(
                    host=self.config.host,
                    port=port,
                    password=self.config.password,
                    db=int(self.config.database) if self.config.database else 0,
                    **self.config.options
                )

            # 연결 테스트
            await self.client.ping()

            self.is_connected = True
            self.stats["connection_count"] += 1

            self.logger.info(f"Connected to Redis: {self.config.host}:{self.config.port or 6379}")
            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"Redis connection failed: {str(e)}")

    async def disconnect(self) -> Result[bool]:
        """데이터베이스 연결 해제"""
        try:
            if self.client:
                if self.config.db_type == NoSQLType.MONGODB:
                    self.client.close()
                elif self.config.db_type == NoSQLType.REDIS:
                    await self.client.close()

                self.client = None
                self.database = None
                self.is_connected = False

            self.logger.info("NoSQL database disconnected")
            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"Disconnect failed: {str(e)}")

    async def execute_operation(
        self,
        operation: str,
        collection_or_key: str,
        data: Optional[Any] = None,
        filter_query: Optional[Dict] = None,
        **kwargs
    ) -> Result[NoSQLResult]:
        """NoSQL 연산 실행"""
        if not self.is_connected:
            connect_result = await self.connect()
            if not connect_result.is_success:
                return Result(False, None, connect_result.error)

        start_time = asyncio.get_event_loop().time()
        self.stats["total_operations"] += 1
        self.stats["last_operation_time"] = datetime.now()

        try:
            if self.config.db_type == NoSQLType.MONGODB:
                result = await self._execute_mongodb_operation(
                    operation, collection_or_key, data, filter_query, **kwargs
                )
            elif self.config.db_type == NoSQLType.REDIS:
                result = await self._execute_redis_operation(
                    operation, collection_or_key, data, **kwargs
                )
            else:
                return Result(False, None, f"Unsupported NoSQL type: {self.config.db_type}")

            execution_time = asyncio.get_event_loop().time() - start_time
            result.execution_time = execution_time
            result.operation = operation

            self.stats["successful_operations"] += 1
            self.stats["total_execution_time"] += execution_time

            return Result(True, result)

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            self.stats["failed_operations"] += 1

            error_result = NoSQLResult(
                data=None,
                affected_count=0,
                execution_time=execution_time,
                operation=operation,
                error=str(e)
            )

            self.logger.error(f"NoSQL operation failed: {str(e)}")
            return Result(False, error_result, str(e))

    async def _execute_mongodb_operation(
        self,
        operation: str,
        collection_name: str,
        data: Optional[Any] = None,
        filter_query: Optional[Dict] = None,
        **kwargs
    ) -> NoSQLResult:
        """MongoDB 연산 실행"""
        if not self.database:
            raise Exception("No database selected")

        collection = self.database[collection_name]

        if operation == "find":
            cursor = collection.find(filter_query or {}, **kwargs)
            docs = await cursor.to_list(length=kwargs.get('limit', 1000))
            return NoSQLResult(
                data=docs,
                affected_count=len(docs),
                execution_time=0.0,
                operation=operation
            )

        elif operation == "find_one":
            doc = await collection.find_one(filter_query or {}, **kwargs)
            return NoSQLResult(
                data=doc,
                affected_count=1 if doc else 0,
                execution_time=0.0,
                operation=operation
            )

        elif operation == "insert_one":
            result = await collection.insert_one(data, **kwargs)
            return NoSQLResult(
                data={"inserted_id": str(result.inserted_id)},
                affected_count=1,
                execution_time=0.0,
                operation=operation
            )

        elif operation == "insert_many":
            result = await collection.insert_many(data, **kwargs)
            return NoSQLResult(
                data={"inserted_ids": [str(id) for id in result.inserted_ids]},
                affected_count=len(result.inserted_ids),
                execution_time=0.0,
                operation=operation
            )

        elif operation == "update_one":
            result = await collection.update_one(filter_query or {}, data, **kwargs)
            return NoSQLResult(
                data={
                    "matched_count": result.matched_count,
                    "modified_count": result.modified_count,
                    "upserted_id": str(result.upserted_id) if result.upserted_id else None
                },
                affected_count=result.modified_count,
                execution_time=0.0,
                operation=operation
            )

        elif operation == "update_many":
            result = await collection.update_many(filter_query or {}, data, **kwargs)
            return NoSQLResult(
                data={
                    "matched_count": result.matched_count,
                    "modified_count": result.modified_count
                },
                affected_count=result.modified_count,
                execution_time=0.0,
                operation=operation
            )

        elif operation == "delete_one":
            result = await collection.delete_one(filter_query or {}, **kwargs)
            return NoSQLResult(
                data={"deleted_count": result.deleted_count},
                affected_count=result.deleted_count,
                execution_time=0.0,
                operation=operation
            )

        elif operation == "delete_many":
            result = await collection.delete_many(filter_query or {}, **kwargs)
            return NoSQLResult(
                data={"deleted_count": result.deleted_count},
                affected_count=result.deleted_count,
                execution_time=0.0,
                operation=operation
            )

        elif operation == "count_documents":
            count = await collection.count_documents(filter_query or {}, **kwargs)
            return NoSQLResult(
                data={"count": count},
                affected_count=count,
                execution_time=0.0,
                operation=operation
            )

        elif operation == "aggregate":
            cursor = collection.aggregate(data, **kwargs)
            docs = await cursor.to_list(length=None)
            return NoSQLResult(
                data=docs,
                affected_count=len(docs),
                execution_time=0.0,
                operation=operation
            )

        else:
            raise Exception(f"Unsupported MongoDB operation: {operation}")

    async def _execute_redis_operation(
        self,
        operation: str,
        key: str,
        data: Optional[Any] = None,
        **kwargs
    ) -> NoSQLResult:
        """Redis 연산 실행"""
        if operation == "get":
            value = await self.client.get(key)
            if value:
                try:
                    # JSON 파싱 시도
                    value = json.loads(value)
                except:
                    # JSON이 아니면 문자열로 처리
                    value = value.decode() if isinstance(value, bytes) else value

            return NoSQLResult(
                data=value,
                affected_count=1 if value is not None else 0,
                execution_time=0.0,
                operation=operation
            )

        elif operation == "set":
            result = await self.client.set(key, json.dumps(data) if isinstance(data, (dict, list)) else data, **kwargs)
            return NoSQLResult(
                data={"success": result},
                affected_count=1 if result else 0,
                execution_time=0.0,
                operation=operation
            )

        elif operation == "delete":
            result = await self.client.delete(key)
            return NoSQLResult(
                data={"deleted_count": result},
                affected_count=result,
                execution_time=0.0,
                operation=operation
            )

        elif operation == "exists":
            result = await self.client.exists(key)
            return NoSQLResult(
                data={"exists": bool(result)},
                affected_count=result,
                execution_time=0.0,
                operation=operation
            )

        elif operation == "keys":
            keys = await self.client.keys(key)  # key는 패턴으로 사용
            return NoSQLResult(
                data=[k.decode() if isinstance(k, bytes) else k for k in keys],
                affected_count=len(keys),
                execution_time=0.0,
                operation=operation
            )

        elif operation == "hget":
            field = data
            value = await self.client.hget(key, field)
            if value:
                try:
                    value = json.loads(value)
                except:
                    value = value.decode() if isinstance(value, bytes) else value

            return NoSQLResult(
                data=value,
                affected_count=1 if value is not None else 0,
                execution_time=0.0,
                operation=operation
            )

        elif operation == "hset":
            field, value = data
            result = await self.client.hset(key, field, json.dumps(value) if isinstance(value, (dict, list)) else value)
            return NoSQLResult(
                data={"success": result},
                affected_count=1 if result else 0,
                execution_time=0.0,
                operation=operation
            )

        elif operation == "hgetall":
            hash_data = await self.client.hgetall(key)
            decoded_data = {}
            for k, v in hash_data.items():
                k = k.decode() if isinstance(k, bytes) else k
                try:
                    v = json.loads(v)
                except:
                    v = v.decode() if isinstance(v, bytes) else v
                decoded_data[k] = v

            return NoSQLResult(
                data=decoded_data,
                affected_count=len(decoded_data),
                execution_time=0.0,
                operation=operation
            )

        elif operation == "incr":
            result = await self.client.incr(key)
            return NoSQLResult(
                data={"value": result},
                affected_count=1,
                execution_time=0.0,
                operation=operation
            )

        elif operation == "expire":
            ttl = data
            result = await self.client.expire(key, ttl)
            return NoSQLResult(
                data={"success": result},
                affected_count=1 if result else 0,
                execution_time=0.0,
                operation=operation
            )

        else:
            raise Exception(f"Unsupported Redis operation: {operation}")

    # MongoDB 편의 메서드들
    async def find_documents(
        self,
        collection: str,
        filter_query: Optional[Dict] = None,
        limit: int = 1000,
        **kwargs
    ) -> Result[List[Dict]]:
        """문서 검색"""
        result = await self.execute_operation("find", collection, filter_query=filter_query, limit=limit, **kwargs)
        if result.is_success:
            return Result(True, result.data.data)
        return Result(False, None, result.error)

    async def insert_document(self, collection: str, document: Dict, **kwargs) -> Result[str]:
        """문서 삽입"""
        result = await self.execute_operation("insert_one", collection, data=document, **kwargs)
        if result.is_success:
            return Result(True, result.data.data["inserted_id"])
        return Result(False, None, result.error)

    async def update_document(
        self,
        collection: str,
        filter_query: Dict,
        update_data: Dict,
        **kwargs
    ) -> Result[int]:
        """문서 업데이트"""
        result = await self.execute_operation("update_one", collection, data=update_data, filter_query=filter_query, **kwargs)
        if result.is_success:
            return Result(True, result.data.affected_count)
        return Result(False, None, result.error)

    async def delete_document(self, collection: str, filter_query: Dict, **kwargs) -> Result[int]:
        """문서 삭제"""
        result = await self.execute_operation("delete_one", collection, filter_query=filter_query, **kwargs)
        if result.is_success:
            return Result(True, result.data.affected_count)
        return Result(False, None, result.error)

    # Redis 편의 메서드들
    async def cache_get(self, key: str) -> Result[Any]:
        """캐시 조회"""
        result = await self.execute_operation("get", key)
        if result.is_success:
            return Result(True, result.data.data)
        return Result(False, None, result.error)

    async def cache_set(self, key: str, value: Any, ttl: Optional[int] = None) -> Result[bool]:
        """캐시 설정"""
        kwargs = {"ex": ttl} if ttl else {}
        result = await self.execute_operation("set", key, data=value, **kwargs)
        if result.is_success:
            return Result(True, result.data.data["success"])
        return Result(False, None, result.error)

    async def cache_delete(self, key: str) -> Result[int]:
        """캐시 삭제"""
        result = await self.execute_operation("delete", key)
        if result.is_success:
            return Result(True, result.data.affected_count)
        return Result(False, None, result.error)

    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        avg_execution_time = (
            self.stats["total_execution_time"] / self.stats["successful_operations"]
            if self.stats["successful_operations"] > 0 else 0
        )

        success_rate = (
            self.stats["successful_operations"] / self.stats["total_operations"]
            if self.stats["total_operations"] > 0 else 0
        )

        return {
            **self.stats,
            "average_execution_time": avg_execution_time,
            "success_rate": success_rate,
            "database_type": self.config.db_type.value,
            "is_connected": self.is_connected,
            "last_operation_time": self.stats["last_operation_time"].isoformat() if self.stats["last_operation_time"] else None
        }

    async def health_check(self) -> Result[Dict[str, Any]]:
        """헬스 체크"""
        try:
            start_time = asyncio.get_event_loop().time()

            if self.config.db_type == NoSQLType.MONGODB:
                await self.client.admin.command('ping')
            elif self.config.db_type == NoSQLType.REDIS:
                await self.client.ping()

            response_time = asyncio.get_event_loop().time() - start_time

            health_data = {
                "status": "healthy",
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
def create_mongodb_config(
    host: str = "localhost",
    port: int = 27017,
    database: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    connection_string: Optional[str] = None,
    **options
) -> NoSQLConfig:
    """MongoDB 설정 생성"""
    return NoSQLConfig(
        db_type=NoSQLType.MONGODB,
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
        connection_string=connection_string,
        options=options
    )


def create_redis_config(
    host: str = "localhost",
    port: int = 6379,
    database: str = "0",
    password: Optional[str] = None,
    connection_string: Optional[str] = None,
    **options
) -> NoSQLConfig:
    """Redis 설정 생성"""
    return NoSQLConfig(
        db_type=NoSQLType.REDIS,
        host=host,
        port=port,
        database=database,
        password=password,
        connection_string=connection_string,
        options=options
    )