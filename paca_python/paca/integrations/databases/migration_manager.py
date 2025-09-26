"""
Migration Manager
데이터베이스 스키마 마이그레이션 관리
"""

import os
import re
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

from ...core.types import Result
from ...core.utils.logger import PacaLogger
from .sql_connector import SQLConnector


@dataclass
class Migration:
    """마이그레이션 정보"""
    id: str
    version: str
    name: str
    description: str
    up_sql: str
    down_sql: str
    checksum: str
    created_at: datetime
    applied_at: Optional[datetime] = None
    applied_by: Optional[str] = None


@dataclass
class MigrationResult:
    """마이그레이션 실행 결과"""
    migration_id: str
    success: bool
    execution_time: float
    error: Optional[str] = None
    rollback_performed: bool = False


class MigrationManager:
    """데이터베이스 마이그레이션 관리자"""

    def __init__(
        self,
        connector: SQLConnector,
        migrations_path: str = "migrations",
        migration_table: str = "schema_migrations"
    ):
        self.connector = connector
        self.migrations_path = Path(migrations_path)
        self.migration_table = migration_table
        self.logger = PacaLogger("MigrationManager")

        # 로드된 마이그레이션들
        self.migrations: Dict[str, Migration] = {}

    async def initialize(self) -> Result[bool]:
        """마이그레이션 시스템 초기화"""
        try:
            # 마이그레이션 테이블 생성
            await self._create_migration_table()

            # 마이그레이션 파일들 로드
            load_result = await self._load_migrations()
            if not load_result.is_success:
                return load_result

            self.logger.info("Migration manager initialized")
            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"Migration initialization failed: {str(e)}")

    async def _create_migration_table(self) -> None:
        """마이그레이션 테이블 생성"""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.migration_table} (
            id VARCHAR(255) PRIMARY KEY,
            version VARCHAR(255) NOT NULL,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            checksum VARCHAR(255) NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            applied_by VARCHAR(255),
            execution_time REAL
        )
        """

        result = await self.connector.execute_query(create_table_sql)
        if not result.is_success:
            raise Exception(f"Failed to create migration table: {result.error}")

    async def _load_migrations(self) -> Result[bool]:
        """마이그레이션 파일들 로드"""
        try:
            if not self.migrations_path.exists():
                self.migrations_path.mkdir(parents=True, exist_ok=True)
                return Result(True, True)

            migration_files = sorted(self.migrations_path.glob("*.sql"))

            for file_path in migration_files:
                migration = await self._parse_migration_file(file_path)
                if migration:
                    self.migrations[migration.id] = migration

            self.logger.info(f"Loaded {len(self.migrations)} migrations")
            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"Migration loading failed: {str(e)}")

    async def _parse_migration_file(self, file_path: Path) -> Optional[Migration]:
        """마이그레이션 파일 파싱"""
        try:
            content = file_path.read_text(encoding='utf-8')

            # 파일명에서 버전과 이름 추출 (예: 001_create_users_table.sql)
            filename = file_path.stem
            match = re.match(r'^(\d+)_(.+)$', filename)
            if not match:
                self.logger.warning(f"Invalid migration filename: {filename}")
                return None

            version = match.group(1)
            name = match.group(2).replace('_', ' ').title()

            # 마이그레이션 ID 생성
            migration_id = f"{version}_{match.group(2)}"

            # UP과 DOWN SQL 분리
            up_sql, down_sql, description = self._parse_migration_content(content)

            # 체크섬 계산
            checksum = hashlib.md5(content.encode()).hexdigest()

            migration = Migration(
                id=migration_id,
                version=version,
                name=name,
                description=description,
                up_sql=up_sql,
                down_sql=down_sql,
                checksum=checksum,
                created_at=datetime.fromtimestamp(file_path.stat().st_mtime)
            )

            return migration

        except Exception as e:
            self.logger.error(f"Failed to parse migration file {file_path}: {str(e)}")
            return None

    def _parse_migration_content(self, content: str) -> tuple[str, str, str]:
        """마이그레이션 내용 파싱"""
        lines = content.split('\n')

        description = ""
        up_sql = ""
        down_sql = ""
        current_section = None

        for line in lines:
            line = line.strip()

            if line.startswith('-- Description:'):
                description = line[15:].strip()
            elif line.startswith('-- UP'):
                current_section = "up"
            elif line.startswith('-- DOWN'):
                current_section = "down"
            elif line and not line.startswith('--'):
                if current_section == "up":
                    up_sql += line + "\n"
                elif current_section == "down":
                    down_sql += line + "\n"

        return up_sql.strip(), down_sql.strip(), description

    async def get_applied_migrations(self) -> Result[List[Migration]]:
        """적용된 마이그레이션 목록 조회"""
        try:
            query = f"SELECT * FROM {self.migration_table} ORDER BY version"
            result = await self.connector.execute_query(query)

            if not result.is_success:
                return Result(False, None, result.error)

            applied_migrations = []
            for row in result.data.rows:
                migration = Migration(
                    id=row['id'],
                    version=row['version'],
                    name=row['name'],
                    description=row.get('description', ''),
                    up_sql="",  # 적용된 마이그레이션은 SQL 내용 불필요
                    down_sql="",
                    checksum=row['checksum'],
                    created_at=datetime.now(),  # 실제로는 파일에서 읽어와야 함
                    applied_at=datetime.fromisoformat(row['applied_at']) if row['applied_at'] else None,
                    applied_by=row.get('applied_by')
                )
                applied_migrations.append(migration)

            return Result(True, applied_migrations)

        except Exception as e:
            return Result(False, None, f"Failed to get applied migrations: {str(e)}")

    async def get_pending_migrations(self) -> Result[List[Migration]]:
        """적용 대기 중인 마이그레이션 목록"""
        try:
            applied_result = await self.get_applied_migrations()
            if not applied_result.is_success:
                return applied_result

            applied_ids = {m.id for m in applied_result.data}
            pending_migrations = [
                migration for migration_id, migration in self.migrations.items()
                if migration_id not in applied_ids
            ]

            # 버전 순으로 정렬
            pending_migrations.sort(key=lambda m: m.version)
            return Result(True, pending_migrations)

        except Exception as e:
            return Result(False, None, f"Failed to get pending migrations: {str(e)}")

    async def migrate_up(self, target_version: Optional[str] = None) -> Result[List[MigrationResult]]:
        """마이그레이션 적용"""
        try:
            pending_result = await self.get_pending_migrations()
            if not pending_result.is_success:
                return pending_result

            pending_migrations = pending_result.data
            if not pending_migrations:
                return Result(True, [])

            # 대상 버전까지만 필터링
            if target_version:
                pending_migrations = [
                    m for m in pending_migrations
                    if m.version <= target_version
                ]

            results = []
            for migration in pending_migrations:
                result = await self._apply_migration(migration)
                results.append(result)

                if not result.success:
                    self.logger.error(f"Migration failed: {migration.id}")
                    break

            return Result(True, results)

        except Exception as e:
            return Result(False, None, f"Migration up failed: {str(e)}")

    async def migrate_down(self, target_version: str) -> Result[List[MigrationResult]]:
        """마이그레이션 롤백"""
        try:
            applied_result = await self.get_applied_migrations()
            if not applied_result.is_success:
                return applied_result

            # 대상 버전보다 높은 버전들을 역순으로 롤백
            applied_migrations = applied_result.data
            rollback_migrations = [
                m for m in applied_migrations
                if m.version > target_version
            ]
            rollback_migrations.sort(key=lambda m: m.version, reverse=True)

            results = []
            for migration_record in rollback_migrations:
                # 파일에서 실제 마이그레이션 정보 로드
                migration = self.migrations.get(migration_record.id)
                if not migration:
                    self.logger.error(f"Migration file not found: {migration_record.id}")
                    continue

                result = await self._rollback_migration(migration)
                results.append(result)

                if not result.success:
                    self.logger.error(f"Rollback failed: {migration.id}")
                    break

            return Result(True, results)

        except Exception as e:
            return Result(False, None, f"Migration down failed: {str(e)}")

    async def _apply_migration(self, migration: Migration) -> MigrationResult:
        """단일 마이그레이션 적용"""
        start_time = datetime.now()

        try:
            # 트랜잭션 시작
            await self.connector.begin_transaction()

            # UP SQL 실행
            if migration.up_sql:
                result = await self.connector.execute_query(migration.up_sql)
                if not result.is_success:
                    await self.connector.rollback_transaction()
                    return MigrationResult(
                        migration_id=migration.id,
                        success=False,
                        execution_time=(datetime.now() - start_time).total_seconds(),
                        error=result.error
                    )

            # 마이그레이션 기록 저장
            record_sql = f"""
            INSERT INTO {self.migration_table}
            (id, version, name, description, checksum, applied_at, applied_by, execution_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """

            execution_time = (datetime.now() - start_time).total_seconds()
            params = [
                migration.id,
                migration.version,
                migration.name,
                migration.description,
                migration.checksum,
                datetime.now().isoformat(),
                "system",  # 실제로는 사용자 정보
                execution_time
            ]

            record_result = await self.connector.execute_query(record_sql, params)
            if not record_result.is_success:
                await self.connector.rollback_transaction()
                return MigrationResult(
                    migration_id=migration.id,
                    success=False,
                    execution_time=execution_time,
                    error=record_result.error
                )

            # 트랜잭션 커밋
            await self.connector.commit_transaction()

            self.logger.info(f"Applied migration: {migration.id}")
            return MigrationResult(
                migration_id=migration.id,
                success=True,
                execution_time=execution_time
            )

        except Exception as e:
            await self.connector.rollback_transaction()
            return MigrationResult(
                migration_id=migration.id,
                success=False,
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=str(e)
            )

    async def _rollback_migration(self, migration: Migration) -> MigrationResult:
        """단일 마이그레이션 롤백"""
        start_time = datetime.now()

        try:
            # 트랜잭션 시작
            await self.connector.begin_transaction()

            # DOWN SQL 실행
            if migration.down_sql:
                result = await self.connector.execute_query(migration.down_sql)
                if not result.is_success:
                    await self.connector.rollback_transaction()
                    return MigrationResult(
                        migration_id=migration.id,
                        success=False,
                        execution_time=(datetime.now() - start_time).total_seconds(),
                        error=result.error,
                        rollback_performed=True
                    )

            # 마이그레이션 기록 삭제
            delete_sql = f"DELETE FROM {self.migration_table} WHERE id = ?"
            delete_result = await self.connector.execute_query(delete_sql, [migration.id])
            if not delete_result.is_success:
                await self.connector.rollback_transaction()
                return MigrationResult(
                    migration_id=migration.id,
                    success=False,
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    error=delete_result.error,
                    rollback_performed=True
                )

            # 트랜잭션 커밋
            await self.connector.commit_transaction()

            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Rolled back migration: {migration.id}")

            return MigrationResult(
                migration_id=migration.id,
                success=True,
                execution_time=execution_time,
                rollback_performed=True
            )

        except Exception as e:
            await self.connector.rollback_transaction()
            return MigrationResult(
                migration_id=migration.id,
                success=False,
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=str(e),
                rollback_performed=True
            )

    async def create_migration(
        self,
        name: str,
        description: str = "",
        up_sql: str = "",
        down_sql: str = ""
    ) -> Result[str]:
        """새 마이그레이션 파일 생성"""
        try:
            # 다음 버전 번호 계산
            existing_versions = [int(m.version) for m in self.migrations.values()]
            next_version = str(max(existing_versions) + 1).zfill(3) if existing_versions else "001"

            # 파일명 생성
            filename = f"{next_version}_{name.lower().replace(' ', '_')}.sql"
            file_path = self.migrations_path / filename

            # 마이그레이션 파일 내용 생성
            content = f"""-- Description: {description}
-- Created: {datetime.now().isoformat()}

-- UP
{up_sql or '-- Add your UP migration SQL here'}

-- DOWN
{down_sql or '-- Add your DOWN migration SQL here'}
"""

            # 파일 저장
            file_path.write_text(content, encoding='utf-8')

            self.logger.info(f"Created migration file: {filename}")
            return Result(True, str(file_path))

        except Exception as e:
            return Result(False, None, f"Failed to create migration: {str(e)}")

    async def validate_migrations(self) -> Result[List[str]]:
        """마이그레이션 무결성 검증"""
        try:
            issues = []

            # 적용된 마이그레이션 체크섬 검증
            applied_result = await self.get_applied_migrations()
            if not applied_result.is_success:
                return applied_result

            for applied_migration in applied_result.data:
                file_migration = self.migrations.get(applied_migration.id)
                if not file_migration:
                    issues.append(f"Applied migration file not found: {applied_migration.id}")
                elif file_migration.checksum != applied_migration.checksum:
                    issues.append(f"Checksum mismatch for migration: {applied_migration.id}")

            # 버전 번호 중복 검사
            versions = [m.version for m in self.migrations.values()]
            if len(versions) != len(set(versions)):
                duplicates = [v for v in set(versions) if versions.count(v) > 1]
                issues.append(f"Duplicate version numbers: {duplicates}")

            return Result(True, issues)

        except Exception as e:
            return Result(False, None, f"Migration validation failed: {str(e)}")

    def get_migration_status(self) -> Dict[str, Any]:
        """마이그레이션 상태 정보"""
        total_migrations = len(self.migrations)

        # 비동기 메서드이므로 실제로는 별도 호출 필요
        return {
            "total_migrations": total_migrations,
            "migrations_path": str(self.migrations_path),
            "migration_table": self.migration_table,
            "available_migrations": list(self.migrations.keys())
        }


# 마이그레이션 파일 예시 생성 함수
def create_example_migration_files(migrations_path: str):
    """예시 마이그레이션 파일들 생성"""
    path = Path(migrations_path)
    path.mkdir(parents=True, exist_ok=True)

    # 001_create_users_table.sql
    users_migration = """-- Description: Create users table
-- Created: 2024-01-01T00:00:00

-- UP
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_active ON users(active);

-- DOWN
DROP INDEX IF EXISTS idx_users_active;
DROP INDEX IF EXISTS idx_users_email;
DROP TABLE IF EXISTS users;
"""

    # 002_create_posts_table.sql
    posts_migration = """-- Description: Create posts table
-- Created: 2024-01-02T00:00:00

-- UP
CREATE TABLE posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    title VARCHAR(255) NOT NULL,
    content TEXT,
    published BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX idx_posts_user_id ON posts(user_id);
CREATE INDEX idx_posts_published ON posts(published);

-- DOWN
DROP INDEX IF EXISTS idx_posts_published;
DROP INDEX IF EXISTS idx_posts_user_id;
DROP TABLE IF EXISTS posts;
"""

    (path / "001_create_users_table.sql").write_text(users_migration)
    (path / "002_create_posts_table.sql").write_text(posts_migration)