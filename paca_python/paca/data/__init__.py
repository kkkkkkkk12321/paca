"""
PACA v5 Data Management System
백업, 복원, 데이터 관리 모듈
"""

from .base import (
    DataType,
    StorageType,
    DataRecord,
    QueryFilter,
    QueryOptions,
    BaseDataStore,
    MemoryDataStore,
    DataManager
)

from .backup_system import (
    BackupSystem, BackupManager, BackupMetadata,
    BackupType, BackupStatus, RestoreResult
)

__version__ = "5.0.0"
__author__ = "PACA Development Team"

__all__ = [
    'DataType',
    'StorageType',
    'DataRecord',
    'QueryFilter',
    'QueryOptions',
    'BaseDataStore',
    'MemoryDataStore',
    'DataManager',
    "BackupSystem",
    "BackupManager",
    "BackupMetadata",
    "BackupType",
    "BackupStatus",
    "RestoreResult"
]