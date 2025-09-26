"""
Portable Storage Module
포터블 애플리케이션을 위한 데이터 저장 관리
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import logging


class PortableStorageManager:
    """포터블 저장소 관리자"""

    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        초기화

        Args:
            base_path: 기본 저장 경로 (None이면 프로그램 폴더 기준)
        """
        if base_path is None:
            # 현재 스크립트의 위치에서 프로젝트 루트 찾기
            current_file = Path(__file__).resolve()
            paca_python_root = current_file.parents[3]  # paca_python 폴더
            self.base_path = paca_python_root / "data"
        else:
            self.base_path = Path(base_path)

        self.data_path = self.base_path
        self.memory_path = self.data_path / "memory"
        self.logs_path = self.data_path / "logs"
        self.db_path = self.data_path / "database"
        self.config_path = self.data_path / "config"
        self.cache_path = self.data_path / "cache"

        self.logger = logging.getLogger("PortableStorage")

        # 초기화 시 디렉토리 생성
        self._create_directory_structure()

    def _create_directory_structure(self) -> None:
        """디렉토리 구조 생성"""
        directories = [
            self.data_path,
            self.memory_path,
            self.memory_path / "working",
            self.memory_path / "episodic",
            self.memory_path / "semantic",
            self.memory_path / "long_term",
            self.logs_path,
            self.db_path,
            self.config_path,
            self.cache_path
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")

    def get_memory_file_path(self, memory_type: str, file_name: str) -> Path:
        """메모리 파일 경로 반환"""
        return self.memory_path / memory_type / file_name

    def get_database_path(self, db_name: str = "paca.db") -> Path:
        """데이터베이스 파일 경로 반환"""
        return self.db_path / db_name

    def get_log_file_path(self, log_name: str = "paca.log") -> Path:
        """로그 파일 경로 반환"""
        return self.logs_path / log_name

    def get_config_file_path(self, config_name: str) -> Path:
        """설정 파일 경로 반환"""
        return self.config_path / config_name

    def get_cache_file_path(self, cache_name: str) -> Path:
        """캐시 파일 경로 반환"""
        return self.cache_path / cache_name

    def save_json_data(self, file_path: Union[str, Path], data: Dict[str, Any]) -> bool:
        """JSON 데이터 저장"""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)

            self.logger.debug(f"Saved JSON data to {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save JSON data to {file_path}: {e}")
            return False

    def load_json_data(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """JSON 데이터 로드"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.logger.debug(f"Loaded JSON data from {file_path}")
            return data

        except Exception as e:
            self.logger.error(f"Failed to load JSON data from {file_path}: {e}")
            return None

    def create_sqlite_connection(self, db_name: str = "paca.db") -> sqlite3.Connection:
        """SQLite 연결 생성"""
        db_path = self.get_database_path(db_name)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row  # 딕셔너리 형태로 결과 반환

        self.logger.debug(f"Created SQLite connection to {db_path}")
        return conn

    def get_memory_storage_path(self, memory_type: str) -> Path:
        """메모리 타입별 저장 경로 반환"""
        return self.memory_path / memory_type.lower()

    def list_memory_files(self, memory_type: str) -> List[Path]:
        """메모리 타입별 파일 목록 반환"""
        memory_dir = self.get_memory_storage_path(memory_type)
        if not memory_dir.exists():
            return []

        return list(memory_dir.glob("*.json"))

    def cleanup_old_files(self, memory_type: str, days_old: int = 30) -> int:
        """오래된 파일 정리"""
        cleanup_count = 0
        memory_dir = self.get_memory_storage_path(memory_type)

        if not memory_dir.exists():
            return 0

        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 3600)

        for file_path in memory_dir.glob("*.json"):
            try:
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    cleanup_count += 1
                    self.logger.debug(f"Cleaned up old file: {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to cleanup file {file_path}: {e}")

        return cleanup_count

    def get_storage_info(self) -> Dict[str, Any]:
        """저장소 정보 반환"""
        info = {
            "base_path": str(self.base_path),
            "directories": {
                "data": str(self.data_path),
                "memory": str(self.memory_path),
                "logs": str(self.logs_path),
                "database": str(self.db_path),
                "config": str(self.config_path),
                "cache": str(self.cache_path)
            },
            "memory_types": {},
            "total_files": 0,
            "total_size_mb": 0.0
        }

        # 메모리 타입별 정보 수집
        for memory_type in ["working", "episodic", "semantic", "long_term"]:
            memory_dir = self.get_memory_storage_path(memory_type)
            if memory_dir.exists():
                files = list(memory_dir.glob("*.json"))
                total_size = sum(f.stat().st_size for f in files if f.is_file())

                info["memory_types"][memory_type] = {
                    "file_count": len(files),
                    "size_mb": total_size / (1024 * 1024)
                }

                info["total_files"] += len(files)
                info["total_size_mb"] += total_size / (1024 * 1024)

        return info

    def export_all_data(self, export_path: Union[str, Path]) -> bool:
        """모든 데이터 내보내기"""
        try:
            export_path = Path(export_path)
            export_path.mkdir(parents=True, exist_ok=True)

            # 메모리 데이터 복사
            import shutil
            if self.memory_path.exists():
                shutil.copytree(self.memory_path, export_path / "memory", dirs_exist_ok=True)

            # 데이터베이스 복사
            if self.db_path.exists():
                shutil.copytree(self.db_path, export_path / "database", dirs_exist_ok=True)

            # 설정 파일 복사
            if self.config_path.exists():
                shutil.copytree(self.config_path, export_path / "config", dirs_exist_ok=True)

            self.logger.info(f"Exported all data to {export_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export data: {e}")
            return False

    def import_all_data(self, import_path: Union[str, Path]) -> bool:
        """모든 데이터 가져오기"""
        try:
            import_path = Path(import_path)
            if not import_path.exists():
                self.logger.error(f"Import path does not exist: {import_path}")
                return False

            import shutil

            # 메모리 데이터 복사
            memory_import = import_path / "memory"
            if memory_import.exists():
                shutil.copytree(memory_import, self.memory_path, dirs_exist_ok=True)

            # 데이터베이스 복사
            db_import = import_path / "database"
            if db_import.exists():
                shutil.copytree(db_import, self.db_path, dirs_exist_ok=True)

            # 설정 파일 복사
            config_import = import_path / "config"
            if config_import.exists():
                shutil.copytree(config_import, self.config_path, dirs_exist_ok=True)

            self.logger.info(f"Imported all data from {import_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to import data: {e}")
            return False


# 전역 저장소 관리자 인스턴스
_storage_manager: Optional[PortableStorageManager] = None


def get_storage_manager() -> PortableStorageManager:
    """전역 저장소 관리자 반환"""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = PortableStorageManager()
    return _storage_manager


def get_portable_memory_path(memory_type: str) -> Path:
    """포터블 메모리 경로 반환"""
    storage_manager = get_storage_manager()
    return storage_manager.get_memory_storage_path(memory_type)


def get_portable_database_path(db_name: str = "paca.db") -> Path:
    """포터블 데이터베이스 경로 반환"""
    storage_manager = get_storage_manager()
    return storage_manager.get_database_path(db_name)


def get_portable_log_path(log_name: str = "paca.log") -> Path:
    """포터블 로그 경로 반환"""
    storage_manager = get_storage_manager()
    return storage_manager.get_log_file_path(log_name)