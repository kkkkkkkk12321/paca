"""
Data Base Module
데이터 관리 및 저장소 기본 클래스들
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ..core.types.base import ID, Timestamp, KeyValuePair, Result
from ..core.errors.base import PacaError


class DataType(Enum):
    """데이터 타입"""
    STRING = 'string'
    INTEGER = 'integer'
    FLOAT = 'float'
    BOOLEAN = 'boolean'
    DATETIME = 'datetime'
    JSON = 'json'
    BINARY = 'binary'


class StorageType(Enum):
    """저장소 타입"""
    MEMORY = 'memory'
    FILE = 'file'
    DATABASE = 'database'
    CACHE = 'cache'
    REMOTE = 'remote'


@dataclass
class DataRecord:
    """데이터 레코드"""
    id: ID
    data: Any
    data_type: DataType
    created_at: Timestamp
    updated_at: Timestamp
    metadata: KeyValuePair


@dataclass
class QueryFilter:
    """쿼리 필터"""
    field: str
    operator: str  # 'eq', 'ne', 'gt', 'lt', 'gte', 'lte', 'in', 'like'
    value: Any


@dataclass
class QueryOptions:
    """쿼리 옵션"""
    filters: List[QueryFilter]
    sort_by: Optional[str] = None
    sort_order: str = 'asc'  # 'asc' or 'desc'
    limit: Optional[int] = None
    offset: int = 0


class BaseDataStore(ABC):
    """기본 데이터 저장소 추상 클래스"""

    def __init__(self, name: str, storage_type: StorageType):
        self.name = name
        self.storage_type = storage_type

    @abstractmethod
    async def store(self, key: str, data: Any, metadata: KeyValuePair = None) -> Result[bool]:
        """데이터 저장"""
        pass

    @abstractmethod
    async def retrieve(self, key: str) -> Result[Optional[DataRecord]]:
        """데이터 조회"""
        pass

    @abstractmethod
    async def update(self, key: str, data: Any, metadata: KeyValuePair = None) -> Result[bool]:
        """데이터 업데이트"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> Result[bool]:
        """데이터 삭제"""
        pass

    @abstractmethod
    async def query(self, options: QueryOptions) -> Result[List[DataRecord]]:
        """데이터 쿼리"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> Result[bool]:
        """데이터 존재 여부 확인"""
        pass

    @abstractmethod
    async def count(self, filters: List[QueryFilter] = None) -> Result[int]:
        """데이터 개수 조회"""
        pass


class MemoryDataStore(BaseDataStore):
    """메모리 데이터 저장소"""

    def __init__(self, name: str):
        super().__init__(name, StorageType.MEMORY)
        self._data: Dict[str, DataRecord] = {}

    async def store(self, key: str, data: Any, metadata: KeyValuePair = None) -> Result[bool]:
        """데이터 저장"""
        try:
            from ..core.types.base import create_id, current_timestamp

            record = DataRecord(
                id=create_id(),
                data=data,
                data_type=self._infer_data_type(data),
                created_at=current_timestamp(),
                updated_at=current_timestamp(),
                metadata=metadata or {}
            )

            self._data[key] = record
            return Result.success(True)

        except Exception as e:
            return Result.failure(PacaError(f"Failed to store data: {str(e)}"))

    async def retrieve(self, key: str) -> Result[Optional[DataRecord]]:
        """데이터 조회"""
        try:
            record = self._data.get(key)
            return Result.success(record)

        except Exception as e:
            return Result.failure(PacaError(f"Failed to retrieve data: {str(e)}"))

    async def update(self, key: str, data: Any, metadata: KeyValuePair = None) -> Result[bool]:
        """데이터 업데이트"""
        try:
            if key not in self._data:
                return Result.failure(PacaError(f"Key {key} not found"))

            from ..core.types.base import current_timestamp

            record = self._data[key]
            updated_record = DataRecord(
                id=record.id,
                data=data,
                data_type=self._infer_data_type(data),
                created_at=record.created_at,
                updated_at=current_timestamp(),
                metadata=metadata or record.metadata
            )

            self._data[key] = updated_record
            return Result.success(True)

        except Exception as e:
            return Result.failure(PacaError(f"Failed to update data: {str(e)}"))

    async def delete(self, key: str) -> Result[bool]:
        """데이터 삭제"""
        try:
            if key in self._data:
                del self._data[key]
                return Result.success(True)
            else:
                return Result.failure(PacaError(f"Key {key} not found"))

        except Exception as e:
            return Result.failure(PacaError(f"Failed to delete data: {str(e)}"))

    async def query(self, options: QueryOptions) -> Result[List[DataRecord]]:
        """데이터 쿼리"""
        try:
            records = list(self._data.values())

            # 필터 적용
            for filter_item in options.filters:
                records = [r for r in records if self._apply_filter(r, filter_item)]

            # 정렬
            if options.sort_by:
                reverse = options.sort_order == 'desc'
                records.sort(
                    key=lambda r: getattr(r, options.sort_by, None),
                    reverse=reverse
                )

            # 페이징
            if options.offset:
                records = records[options.offset:]

            if options.limit:
                records = records[:options.limit]

            return Result.success(records)

        except Exception as e:
            return Result.failure(PacaError(f"Failed to query data: {str(e)}"))

    async def exists(self, key: str) -> Result[bool]:
        """데이터 존재 여부 확인"""
        return Result.success(key in self._data)

    async def count(self, filters: List[QueryFilter] = None) -> Result[int]:
        """데이터 개수 조회"""
        try:
            if not filters:
                return Result.success(len(self._data))

            records = list(self._data.values())
            for filter_item in filters:
                records = [r for r in records if self._apply_filter(r, filter_item)]

            return Result.success(len(records))

        except Exception as e:
            return Result.failure(PacaError(f"Failed to count data: {str(e)}"))

    def _infer_data_type(self, data: Any) -> DataType:
        """데이터 타입 추론"""
        if isinstance(data, str):
            return DataType.STRING
        elif isinstance(data, int):
            return DataType.INTEGER
        elif isinstance(data, float):
            return DataType.FLOAT
        elif isinstance(data, bool):
            return DataType.BOOLEAN
        elif isinstance(data, (dict, list)):
            return DataType.JSON
        else:
            return DataType.STRING

    def _apply_filter(self, record: DataRecord, filter_item: QueryFilter) -> bool:
        """필터 적용"""
        try:
            # 필드 값 추출
            if hasattr(record, filter_item.field):
                field_value = getattr(record, filter_item.field)
            elif filter_item.field in record.metadata:
                field_value = record.metadata[filter_item.field]
            else:
                return False

            # 연산자별 처리
            if filter_item.operator == 'eq':
                return field_value == filter_item.value
            elif filter_item.operator == 'ne':
                return field_value != filter_item.value
            elif filter_item.operator == 'gt':
                return field_value > filter_item.value
            elif filter_item.operator == 'lt':
                return field_value < filter_item.value
            elif filter_item.operator == 'gte':
                return field_value >= filter_item.value
            elif filter_item.operator == 'lte':
                return field_value <= filter_item.value
            elif filter_item.operator == 'in':
                return field_value in filter_item.value
            elif filter_item.operator == 'like':
                return str(filter_item.value).lower() in str(field_value).lower()
            else:
                return False

        except Exception:
            return False


class DataManager:
    """데이터 관리자"""

    def __init__(self):
        self.stores: Dict[str, BaseDataStore] = {}
        self._is_initialized = False

    async def initialize(self) -> Result[bool]:
        """데이터 관리자 초기화"""
        if self._is_initialized:
            return Result.success(True)

        try:
            # 기본 메모리 저장소 등록
            self.register_store("memory", MemoryDataStore("memory"))
            self.register_store("conversations", MemoryDataStore("conversations"))
            self.register_store("learning", MemoryDataStore("learning"))

            self._is_initialized = True
            return Result.success(True)

        except Exception as error:
            return Result.failure(PacaError(f"Data manager initialization failed: {str(error)}"))

    async def cleanup(self) -> Result[bool]:
        """데이터 관리자 정리"""
        try:
            for store in self.stores.values():
                if hasattr(store, '_data') and isinstance(store._data, dict):
                    store._data.clear()

            self.stores.clear()
            self._is_initialized = False
            return Result.success(True)

        except Exception as error:
            return Result.failure(PacaError(f"Data manager cleanup failed: {str(error)}"))

    def register_store(self, name: str, store: BaseDataStore) -> None:
        """데이터 저장소 등록"""
        self.stores[name] = store

    def get_store(self, name: str) -> Optional[BaseDataStore]:
        """데이터 저장소 조회"""
        return self.stores.get(name)

    def list_stores(self) -> List[str]:
        """데이터 저장소 목록"""
        return list(self.stores.keys())

    async def store_data(
        self,
        store_name: str,
        key: str,
        data: Any,
        metadata: KeyValuePair = None
    ) -> Result[bool]:
        """데이터 저장"""
        store = self.get_store(store_name)
        if not store:
            return Result.failure(PacaError(f"Store {store_name} not found"))

        return await store.store(key, data, metadata)

    async def retrieve_data(self, store_name: str, key: str) -> Result[Optional[DataRecord]]:
        """데이터 조회"""
        store = self.get_store(store_name)
        if not store:
            return Result.failure(PacaError(f"Store {store_name} not found"))

        return await store.retrieve(key)
