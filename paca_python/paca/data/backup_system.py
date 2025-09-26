#!/usr/bin/env python3
"""
PACA v5 Backup/Restore System
학습 데이터 자동 백업 및 복원 시스템
"""

import os
import json
import shutil
import hashlib
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import zipfile
import tempfile

# 조건부 임포트: 패키지 실행시와 직접 실행시 모두 지원
try:
    from ..core.types.base import (
        ID, Timestamp, Result, current_timestamp, generate_id, create_success, create_failure
    )
    from ..core.errors import ApplicationError
    from ..core.utils.portable_storage import get_storage_manager
except ImportError:
    from paca.core.types.base import (
        ID, Timestamp, Result, current_timestamp, generate_id, create_success, create_failure
    )
    from paca.core.errors import ApplicationError
    from paca.core.utils.portable_storage import get_storage_manager


class BackupType(Enum):
    """백업 유형"""
    AUTOMATIC = "automatic"      # 자동 백업
    MANUAL = "manual"           # 수동 백업
    SCHEDULED = "scheduled"     # 스케줄 백업
    EMERGENCY = "emergency"     # 응급 백업
    INCREMENTAL = "incremental" # 증분 백업
    FULL = "full"              # 전체 백업


class BackupStatus(Enum):
    """백업 상태"""
    PENDING = "pending"         # 대기 중
    IN_PROGRESS = "in_progress" # 진행 중
    COMPLETED = "completed"     # 완료
    FAILED = "failed"          # 실패
    CORRUPTED = "corrupted"    # 손상됨
    EXPIRED = "expired"        # 만료됨


@dataclass
class BackupMetadata:
    """백업 메타데이터"""
    backup_id: str
    backup_type: BackupType
    status: BackupStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    file_path: Optional[str] = None
    file_size: int = 0
    checksum: Optional[str] = None
    compressed_size: int = 0
    compression_ratio: float = 0.0
    data_types: List[str] = None
    source_paths: List[str] = None
    description: Optional[str] = None
    tags: List[str] = None
    version: str = "5.0.0"

    def __post_init__(self):
        if self.data_types is None:
            self.data_types = []
        if self.source_paths is None:
            self.source_paths = []
        if self.tags is None:
            self.tags = []


@dataclass
class RestoreResult:
    """복원 결과"""
    success: bool
    backup_id: str
    restored_files: List[str]
    skipped_files: List[str]
    error_files: List[str]
    restore_time: float
    message: str


class BackupSystem:
    """자동 백업 시스템 (핵심 클래스)"""

    def __init__(self, backup_root: str = None):
        """
        백업 시스템 초기화

        Args:
            backup_root: 백업 루트 디렉토리 (기본값: 포터블 경로)
        """
        if backup_root is None:
            # 포터블 저장소 사용
            storage_manager = get_storage_manager()
            self.backup_root = storage_manager.data_path / "backups"
        else:
            self.backup_root = Path(backup_root)
        self.backup_root.mkdir(exist_ok=True)

        # 백업 설정
        self.max_backups = 50
        self.retention_days = 30
        self.compression_enabled = True
        self.auto_cleanup = True

        # 백업 메타데이터 저장소
        self.metadata_file = self.backup_root / "backup_metadata.json"
        self.metadata_cache: Dict[str, BackupMetadata] = {}

        # 비동기 작업 관리
        self.backup_tasks: Dict[str, asyncio.Task] = {}

        # 백업 일정 관리
        self.scheduled_backups: Dict[str, Dict[str, Any]] = {}

        # 통계 정보
        self.stats = {
            "total_backups": 0,
            "successful_backups": 0,
            "failed_backups": 0,
            "total_size": 0,
            "total_compressed_size": 0,
        }

        # 메타데이터 로드
        self._load_metadata()

    def create_auto_backup(self, trigger_event: str, data_paths: List[str] = None) -> str:
        """
        자동 백업 생성 (문서 명세 함수)

        Args:
            trigger_event: 백업 트리거 이벤트
            data_paths: 백업할 데이터 경로 목록

        Returns:
            str: 백업 ID
        """
        backup_id = f"PACA_BACKUP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 기본 데이터 경로 설정 (포터블 기준)
        if data_paths is None:
            storage_manager = get_storage_manager()
            data_paths = [
                str(storage_manager.memory_path),
                str(storage_manager.db_path),
                str(storage_manager.config_path),
                str(storage_manager.logs_path)
            ]

        # 백업 메타데이터 생성
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=BackupType.AUTOMATIC,
            status=BackupStatus.PENDING,
            created_at=datetime.now(),
            source_paths=data_paths,
            description=f"Auto backup triggered by: {trigger_event}",
            tags=["auto", trigger_event.lower()]
        )

        # 백업 시작
        asyncio.create_task(self._perform_backup_async(metadata))

        return backup_id

    def restore_from_backup(self, backup_id: str, target_path: str = None) -> bool:
        """
        지정 백업으로 복원 (문서 명세 함수)

        Args:
            backup_id: 백업 ID
            target_path: 복원 대상 경로

        Returns:
            bool: 복원 성공 여부
        """
        try:
            result = asyncio.run(self.restore_backup_async(backup_id, target_path))
            return result.success
        except Exception as e:
            print(f"Restore failed: {e}")
            return False

    async def create_backup_async(self,
                                backup_type: BackupType = BackupType.MANUAL,
                                source_paths: List[str] = None,
                                description: str = None,
                                tags: List[str] = None) -> Result[str]:
        """
        비동기 백업 생성

        Args:
            backup_type: 백업 유형
            source_paths: 소스 경로 목록
            description: 백업 설명
            tags: 백업 태그

        Returns:
            Result[str]: 백업 ID 결과
        """
        try:
            backup_id = generate_id()

            # 기본 소스 경로 설정 (포터블 기준)
            if source_paths is None:
                storage_manager = get_storage_manager()
                source_paths = [
                    str(storage_manager.memory_path),
                    str(storage_manager.db_path),
                    str(storage_manager.config_path),
                    str(storage_manager.logs_path),
                    str(storage_manager.cache_path)
                ]

            # 백업 메타데이터 생성
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=backup_type,
                status=BackupStatus.PENDING,
                created_at=datetime.now(),
                source_paths=source_paths,
                description=description or f"{backup_type.value} backup",
                tags=tags or [backup_type.value]
            )

            # 백업 수행
            success = await self._perform_backup_async(metadata)

            if success:
                self.stats["successful_backups"] += 1
                return create_success(backup_id)
            else:
                self.stats["failed_backups"] += 1
                return create_failure(f"Backup {backup_id} failed")

        except Exception as e:
            self.stats["failed_backups"] += 1
            return create_failure(f"Backup creation failed: {str(e)}")

    async def _perform_backup_async(self, metadata: BackupMetadata) -> bool:
        """
        비동기 백업 수행

        Args:
            metadata: 백업 메타데이터

        Returns:
            bool: 백업 성공 여부
        """
        try:
            # 백업 상태 업데이트
            metadata.status = BackupStatus.IN_PROGRESS
            self._save_metadata(metadata)

            # 백업 파일 경로
            backup_filename = f"{metadata.backup_id}.zip"
            backup_path = self.backup_root / backup_filename

            # 백업 디렉토리 생성
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            # 압축 백업 생성
            total_size = 0
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for source_path in metadata.source_paths:
                    source_path_obj = Path(source_path)

                    if source_path_obj.exists():
                        if source_path_obj.is_file():
                            # 단일 파일 백업
                            zip_file.write(source_path_obj, source_path_obj.name)
                            total_size += source_path_obj.stat().st_size
                        elif source_path_obj.is_dir():
                            # 디렉토리 재귀 백업
                            for file_path in source_path_obj.rglob('*'):
                                if file_path.is_file():
                                    relative_path = file_path.relative_to(source_path_obj.parent)
                                    zip_file.write(file_path, str(relative_path))
                                    total_size += file_path.stat().st_size

            # 백업 완료 처리
            compressed_size = backup_path.stat().st_size
            checksum = self._calculate_checksum(backup_path)
            compression_ratio = (1 - compressed_size / total_size) * 100 if total_size > 0 else 0

            # 메타데이터 업데이트
            metadata.status = BackupStatus.COMPLETED
            metadata.completed_at = datetime.now()
            metadata.file_path = str(backup_path)
            metadata.file_size = total_size
            metadata.compressed_size = compressed_size
            metadata.compression_ratio = compression_ratio
            metadata.checksum = checksum

            # 통계 업데이트
            self.stats["total_backups"] += 1
            self.stats["total_size"] += total_size
            self.stats["total_compressed_size"] += compressed_size

            # 메타데이터 저장
            self._save_metadata(metadata)

            # 자동 정리
            if self.auto_cleanup:
                await self._cleanup_old_backups()

            return True

        except Exception as e:
            # 백업 실패 처리
            metadata.status = BackupStatus.FAILED
            metadata.completed_at = datetime.now()
            self._save_metadata(metadata)

            print(f"Backup failed: {e}")
            return False

    async def restore_backup_async(self, backup_id: str, target_path: str = None) -> RestoreResult:
        """
        비동기 백업 복원

        Args:
            backup_id: 백업 ID
            target_path: 복원 대상 경로

        Returns:
            RestoreResult: 복원 결과
        """
        start_time = datetime.now()
        restored_files = []
        skipped_files = []
        error_files = []

        try:
            # 백업 메타데이터 조회
            metadata = self.get_backup_metadata(backup_id)
            if not metadata:
                return RestoreResult(
                    success=False,
                    backup_id=backup_id,
                    restored_files=[],
                    skipped_files=[],
                    error_files=[],
                    restore_time=0,
                    message=f"Backup {backup_id} not found"
                )

            # 백업 파일 확인
            backup_path = Path(metadata.file_path)
            if not backup_path.exists():
                return RestoreResult(
                    success=False,
                    backup_id=backup_id,
                    restored_files=[],
                    skipped_files=[],
                    error_files=[],
                    restore_time=0,
                    message=f"Backup file not found: {backup_path}"
                )

            # 체크섬 검증
            if metadata.checksum and self._calculate_checksum(backup_path) != metadata.checksum:
                return RestoreResult(
                    success=False,
                    backup_id=backup_id,
                    restored_files=[],
                    skipped_files=[],
                    error_files=[],
                    restore_time=0,
                    message="Backup file corrupted (checksum mismatch)"
                )

            # 복원 대상 경로 설정
            if target_path is None:
                target_path = "."
            target_path_obj = Path(target_path)
            target_path_obj.mkdir(parents=True, exist_ok=True)

            # 백업 파일 압축 해제
            with zipfile.ZipFile(backup_path, 'r') as zip_file:
                for file_info in zip_file.filelist:
                    try:
                        # 파일 경로 설정
                        extract_path = target_path_obj / file_info.filename

                        # 디렉토리 생성
                        extract_path.parent.mkdir(parents=True, exist_ok=True)

                        # 파일 추출
                        with zip_file.open(file_info) as source_file:
                            with open(extract_path, 'wb') as target_file:
                                target_file.write(source_file.read())

                        restored_files.append(str(extract_path))

                    except Exception as e:
                        error_files.append(f"{file_info.filename}: {str(e)}")

            # 복원 시간 계산
            restore_time = (datetime.now() - start_time).total_seconds()

            return RestoreResult(
                success=True,
                backup_id=backup_id,
                restored_files=restored_files,
                skipped_files=skipped_files,
                error_files=error_files,
                restore_time=restore_time,
                message=f"Restored {len(restored_files)} files successfully"
            )

        except Exception as e:
            restore_time = (datetime.now() - start_time).total_seconds()
            return RestoreResult(
                success=False,
                backup_id=backup_id,
                restored_files=restored_files,
                skipped_files=skipped_files,
                error_files=error_files,
                restore_time=restore_time,
                message=f"Restore failed: {str(e)}"
            )

    def get_backup_metadata(self, backup_id: str) -> Optional[BackupMetadata]:
        """
        백업 메타데이터 조회

        Args:
            backup_id: 백업 ID

        Returns:
            Optional[BackupMetadata]: 백업 메타데이터
        """
        return self.metadata_cache.get(backup_id)

    def list_backups(self,
                    backup_type: BackupType = None,
                    status: BackupStatus = None,
                    limit: int = None) -> List[BackupMetadata]:
        """
        백업 목록 조회

        Args:
            backup_type: 백업 유형 필터
            status: 상태 필터
            limit: 결과 제한

        Returns:
            List[BackupMetadata]: 백업 메타데이터 목록
        """
        backups = list(self.metadata_cache.values())

        # 필터 적용
        if backup_type:
            backups = [b for b in backups if b.backup_type == backup_type]

        if status:
            backups = [b for b in backups if b.status == status]

        # 생성일 기준 정렬 (최신순)
        backups.sort(key=lambda x: x.created_at, reverse=True)

        # 제한 적용
        if limit:
            backups = backups[:limit]

        return backups

    def delete_backup(self, backup_id: str) -> bool:
        """
        백업 삭제

        Args:
            backup_id: 백업 ID

        Returns:
            bool: 삭제 성공 여부
        """
        try:
            metadata = self.get_backup_metadata(backup_id)
            if not metadata:
                return False

            # 백업 파일 삭제
            if metadata.file_path:
                backup_path = Path(metadata.file_path)
                if backup_path.exists():
                    backup_path.unlink()

            # 메타데이터에서 제거
            if backup_id in self.metadata_cache:
                del self.metadata_cache[backup_id]

            # 메타데이터 파일 업데이트
            self._save_all_metadata()

            return True

        except Exception as e:
            print(f"Failed to delete backup {backup_id}: {e}")
            return False

    async def _cleanup_old_backups(self):
        """오래된 백업 자동 정리"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)

            # 만료된 백업 찾기
            expired_backups = [
                backup for backup in self.metadata_cache.values()
                if backup.created_at < cutoff_date
            ]

            # 백업 수 제한 적용
            if len(self.metadata_cache) > self.max_backups:
                # 가장 오래된 백업부터 삭제
                all_backups = sorted(
                    self.metadata_cache.values(),
                    key=lambda x: x.created_at
                )
                excess_count = len(all_backups) - self.max_backups
                expired_backups.extend(all_backups[:excess_count])

            # 중복 제거
            unique_expired = {backup.backup_id: backup for backup in expired_backups}

            # 삭제 실행
            for backup_id in unique_expired:
                self.delete_backup(backup_id)

        except Exception as e:
            print(f"Cleanup failed: {e}")

    def _calculate_checksum(self, file_path: Path) -> str:
        """파일 체크섬 계산"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _save_metadata(self, metadata: BackupMetadata):
        """메타데이터 저장"""
        self.metadata_cache[metadata.backup_id] = metadata
        self._save_all_metadata()

    def _save_all_metadata(self):
        """모든 메타데이터 저장"""
        try:
            # 메타데이터 디렉토리 생성
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)

            # 메타데이터를 JSON 직렬화 가능한 형태로 변환
            serializable_data = {}
            for backup_id, metadata in self.metadata_cache.items():
                data = asdict(metadata)
                # datetime 객체를 문자열로 변환
                data['created_at'] = data['created_at'].isoformat()
                if data['completed_at']:
                    data['completed_at'] = data['completed_at'].isoformat()
                # Enum을 문자열로 변환
                data['backup_type'] = data['backup_type'].value
                data['status'] = data['status'].value

                serializable_data[backup_id] = data

            # JSON 파일로 저장
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Failed to save metadata: {e}")

    def _load_metadata(self):
        """메타데이터 로드"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for backup_id, metadata_dict in data.items():
                    # 문자열을 datetime으로 변환
                    metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
                    if metadata_dict['completed_at']:
                        metadata_dict['completed_at'] = datetime.fromisoformat(metadata_dict['completed_at'])

                    # 문자열을 Enum으로 변환
                    metadata_dict['backup_type'] = BackupType(metadata_dict['backup_type'])
                    metadata_dict['status'] = BackupStatus(metadata_dict['status'])

                    # BackupMetadata 객체 생성
                    metadata = BackupMetadata(**metadata_dict)
                    self.metadata_cache[backup_id] = metadata

                # 통계 재계산
                self._recalculate_stats()

        except Exception as e:
            print(f"Failed to load metadata: {e}")
            self.metadata_cache = {}

    def _recalculate_stats(self):
        """통계 재계산"""
        self.stats = {
            "total_backups": len(self.metadata_cache),
            "successful_backups": len([b for b in self.metadata_cache.values() if b.status == BackupStatus.COMPLETED]),
            "failed_backups": len([b for b in self.metadata_cache.values() if b.status == BackupStatus.FAILED]),
            "total_size": sum(b.file_size for b in self.metadata_cache.values()),
            "total_compressed_size": sum(b.compressed_size for b in self.metadata_cache.values()),
        }

    def get_stats(self) -> Dict[str, Any]:
        """백업 통계 조회"""
        return self.stats.copy()


class BackupManager:
    """백업 관리 시스템 (고급 기능)"""

    def __init__(self, backup_systems: Dict[str, BackupSystem] = None):
        """
        백업 매니저 초기화

        Args:
            backup_systems: 백업 시스템 딕셔너리 (환경별)
        """
        if backup_systems is None:
            # 포터블 저장소 기준으로 백업 시스템 생성
            storage_manager = get_storage_manager()
            backup_root = storage_manager.data_path / "backups"
            self.backup_systems = {
                "local": BackupSystem(str(backup_root / "local")),
                "archive": BackupSystem(str(backup_root / "archive"))
            }
        else:
            self.backup_systems = backup_systems

        # 백업 스케줄러
        self.scheduler_tasks: Dict[str, asyncio.Task] = {}
        self.scheduled_jobs: Dict[str, Dict[str, Any]] = {}

        # 백업 정책 (포터블 경로 기준)
        storage_manager = get_storage_manager()
        self.backup_policies = {
            "learning_data": {
                "schedule": "0 */6 * * *",  # 6시간마다
                "retention_days": 7,
                "backup_type": BackupType.AUTOMATIC,
                "paths": [str(storage_manager.memory_path)]
            },
            "full_system": {
                "schedule": "0 2 * * 0",  # 매주 일요일 2시
                "retention_days": 30,
                "backup_type": BackupType.SCHEDULED,
                "paths": [str(storage_manager.data_path)]
            }
        }

    async def schedule_backup(self,
                            policy_name: str,
                            schedule: str,
                            backup_system: str = "local",
                            **kwargs) -> Result[str]:
        """
        백업 스케줄 등록

        Args:
            policy_name: 정책 이름
            schedule: 크론 표현식
            backup_system: 백업 시스템 이름
            **kwargs: 추가 백업 옵션

        Returns:
            Result[str]: 스케줄 ID
        """
        try:
            if backup_system not in self.backup_systems:
                return create_failure(f"Backup system '{backup_system}' not found")

            schedule_id = generate_id()

            # 스케줄 정보 저장
            self.scheduled_jobs[schedule_id] = {
                "policy_name": policy_name,
                "schedule": schedule,
                "backup_system": backup_system,
                "created_at": datetime.now(),
                "last_run": None,
                "next_run": None,
                "options": kwargs
            }

            # 스케줄러 태스크 시작 (실제 구현에서는 cron 라이브러리 사용)
            # 여기서는 간소화된 버전으로 구현

            return create_success(schedule_id)

        except Exception as e:
            return create_failure(f"Failed to schedule backup: {str(e)}")

    async def perform_emergency_backup(self, reason: str) -> Result[str]:
        """
        응급 백업 수행

        Args:
            reason: 응급 백업 사유

        Returns:
            Result[str]: 백업 ID
        """
        try:
            # 모든 백업 시스템에 응급 백업 생성
            backup_ids = []

            for system_name, backup_system in self.backup_systems.items():
                result = await backup_system.create_backup_async(
                    backup_type=BackupType.EMERGENCY,
                    description=f"Emergency backup: {reason}",
                    tags=["emergency", reason.lower().replace(" ", "_")]
                )

                if result.is_success():
                    backup_ids.append(result.data)

            if backup_ids:
                return create_success(f"Emergency backups created: {', '.join(backup_ids)}")
            else:
                return create_failure("Failed to create emergency backups")

        except Exception as e:
            return create_failure(f"Emergency backup failed: {str(e)}")

    def get_backup_system(self, system_name: str) -> Optional[BackupSystem]:
        """백업 시스템 조회"""
        return self.backup_systems.get(system_name)

    def add_backup_system(self, system_name: str, backup_system: BackupSystem):
        """백업 시스템 추가"""
        self.backup_systems[system_name] = backup_system

    def get_global_stats(self) -> Dict[str, Any]:
        """전체 백업 통계"""
        global_stats = {
            "systems": {},
            "total_backups": 0,
            "total_size": 0,
            "total_compressed_size": 0
        }

        for system_name, backup_system in self.backup_systems.items():
            stats = backup_system.get_stats()
            global_stats["systems"][system_name] = stats
            global_stats["total_backups"] += stats["total_backups"]
            global_stats["total_size"] += stats["total_size"]
            global_stats["total_compressed_size"] += stats["total_compressed_size"]

        return global_stats


# 백업 트리거 이벤트 정의
class BackupTrigger:
    """백업 트리거 이벤트"""

    TRAINING_START = "training_start"
    TRAINING_COMPLETE = "training_complete"
    CONFIG_CHANGE = "config_change"
    SYSTEM_SHUTDOWN = "system_shutdown"
    DATA_CORRUPTION = "data_corruption"
    MANUAL_REQUEST = "manual_request"
    SCHEDULED = "scheduled"


def create_default_backup_system() -> BackupSystem:
    """기본 백업 시스템 생성 (포터블)"""
    return BackupSystem()  # 기본값으로 포터블 경로 사용


def create_backup_manager() -> BackupManager:
    """기본 백업 매니저 생성 (포터블)"""
    storage_manager = get_storage_manager()
    backup_root = storage_manager.data_path / "backups"
    return BackupManager({
        "local": BackupSystem(str(backup_root / "local")),
        "archive": BackupSystem(str(backup_root / "archive")),
        "emergency": BackupSystem(str(backup_root / "emergency"))
    })


# 테스트 함수들
async def test_backup_system():
    """백업 시스템 테스트"""
    print("=== Backup System Test ===")

    # 백업 시스템 생성
    backup_system = create_default_backup_system()

    # 자동 백업 테스트
    backup_id = backup_system.create_auto_backup("test_trigger", ["paca/cognitive"])
    print(f"Auto backup ID: {backup_id}")

    # 수동 백업 테스트
    result = await backup_system.create_backup_async(
        backup_type=BackupType.MANUAL,
        source_paths=["paca/learning"],
        description="Manual test backup"
    )
    print(f"Manual backup result: {result.is_success()}")

    # 백업 목록 조회
    backups = backup_system.list_backups(limit=5)
    print(f"Total backups: {len(backups)}")

    # 통계 조회
    stats = backup_system.get_stats()
    print(f"Backup stats: {stats}")

    return True


if __name__ == "__main__":
    # 직접 실행 테스트
    asyncio.run(test_backup_system())