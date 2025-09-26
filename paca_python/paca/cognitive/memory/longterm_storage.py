"""Long-term memory storage adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import sqlite3

from .types import LongTermMemorySettings
from ...core.utils.portable_storage import PortableStorageManager


class LongTermStorageAdapter(ABC):
    """추상 스토리지 어댑터."""

    def __init__(self, settings: LongTermMemorySettings, storage_manager: PortableStorageManager):
        self.settings = settings
        self.storage_manager = storage_manager

    @abstractmethod
    def connect(self) -> sqlite3.Connection:
        """스토리지 연결을 생성하여 반환합니다."""

    @abstractmethod
    def describe(self) -> str:
        """현재 어댑터 구성을 문자열로 설명."""


class SQLiteStorageAdapter(LongTermStorageAdapter):
    """기존 SQLite 기반 어댑터."""

    def __init__(
        self,
        settings: LongTermMemorySettings,
        storage_manager: PortableStorageManager,
        explicit_path: str = ":memory:",
    ):
        super().__init__(settings, storage_manager)
        self._explicit_path = explicit_path
        self._resolved_path: Optional[str] = None

    def _resolve_path(self) -> str:
        if self._explicit_path != ":memory:" or not self.settings.persistent_db:
            return self._explicit_path

        persistent_path = self.storage_manager.get_database_path(self.settings.database_name)
        return str(persistent_path)

    def connect(self) -> sqlite3.Connection:
        db_path = self._resolve_path()
        self._resolved_path = db_path
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def describe(self) -> str:
        if self._resolved_path:
            return f"sqlite://{self._resolved_path}"
        return "sqlite://(unresolved)"

    @property
    def resolved_path(self) -> Optional[str]:
        return self._resolved_path
