"""
PACA 구조화된 로깅 시스템
실시간 로그 수집, 처리, 저장을 위한 로거
"""

import asyncio
import json
import logging
import traceback
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import structlog
import aiosqlite


class LogLevel(Enum):
    """로그 레벨"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """로그 엔트리"""
    timestamp: datetime = field(default_factory=datetime.now)
    level: LogLevel = LogLevel.INFO
    message: str = ""
    component: str = "unknown"
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    tool_name: Optional[str] = None
    action: Optional[str] = None
    duration: Optional[float] = None
    success: Optional[bool] = None
    error_code: Optional[str] = None
    error_details: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['level'] = self.level.value
        return data

    def to_json(self) -> str:
        """JSON 문자열로 변환"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class PACALogger:
    """PACA 구조화 로거"""

    def __init__(
        self,
        name: str = "paca",
        log_file: Optional[str] = None,
        log_level: LogLevel = LogLevel.INFO,
        enable_console: bool = True,
        enable_database: bool = True,
        db_path: str = "logs.db"
    ):
        """
        초기화

        Args:
            name: 로거 이름
            log_file: 로그 파일 경로
            log_level: 최소 로그 레벨
            enable_console: 콘솔 출력 활성화
            enable_database: 데이터베이스 저장 활성화
            db_path: 데이터베이스 경로
        """
        self.name = name
        self.log_file = log_file
        self.log_level = log_level
        self.enable_console = enable_console
        self.enable_database = enable_database
        self.db_path = db_path

        # 로그 버퍼 (배치 처리용)
        self.log_buffer: List[LogEntry] = []
        self.buffer_size = 100
        self.flush_interval = 30  # 30초마다 플러시

        # 내부 로거 설정
        self._setup_logger()
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None

    def _setup_logger(self):
        """내부 로거 설정"""
        # Structlog 설정
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # 표준 로거 설정
        self.logger = structlog.get_logger(self.name)

        # 파일 핸들러 설정
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setLevel(getattr(logging, self.log_level.value))
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)

            # 루트 로거에 핸들러 추가
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
            root_logger.setLevel(getattr(logging, self.log_level.value))

    async def start(self):
        """로거 시작"""
        if self._running:
            return

        self._running = True

        # 데이터베이스 초기화
        if self.enable_database:
            await self._init_database()

        # 백그라운드 플러시 태스크 시작
        self._flush_task = asyncio.create_task(self._flush_loop())

        await self.info("PACA Logger started", component="logger")

    async def stop(self):
        """로거 정지"""
        if not self._running:
            return

        await self.info("PACA Logger stopping", component="logger")

        self._running = False

        # 플러시 태스크 취소
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # 남은 로그 플러시
        await self._flush_logs()

    async def _init_database(self):
        """로그 데이터베이스 초기화"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        level TEXT NOT NULL,
                        message TEXT NOT NULL,
                        component TEXT,
                        session_id TEXT,
                        user_id TEXT,
                        tool_name TEXT,
                        action TEXT,
                        duration REAL,
                        success BOOLEAN,
                        error_code TEXT,
                        error_details TEXT,
                        metadata TEXT,  -- JSON
                        trace_id TEXT
                    )
                """)

                # 인덱스 생성
                indices = [
                    "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level)",
                    "CREATE INDEX IF NOT EXISTS idx_logs_component ON logs(component)",
                    "CREATE INDEX IF NOT EXISTS idx_logs_session ON logs(session_id)",
                    "CREATE INDEX IF NOT EXISTS idx_logs_tool ON logs(tool_name)",
                ]

                for index_sql in indices:
                    await db.execute(index_sql)

                await db.commit()

        except Exception as e:
            print(f"Failed to initialize log database: {e}")

    async def _flush_loop(self):
        """백그라운드 로그 플러시 루프"""
        try:
            while self._running:
                await asyncio.sleep(self.flush_interval)
                await self._flush_logs()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error in flush loop: {e}")

    async def _flush_logs(self):
        """로그 버퍼 플러시"""
        if not self.log_buffer:
            return

        logs_to_flush = self.log_buffer.copy()
        self.log_buffer.clear()

        try:
            # 데이터베이스에 저장
            if self.enable_database:
                await self._save_logs_to_db(logs_to_flush)

            # 파일에 저장
            if self.log_file:
                await self._save_logs_to_file(logs_to_flush)

        except Exception as e:
            print(f"Failed to flush logs: {e}")
            # 실패한 로그들을 다시 버퍼에 추가
            self.log_buffer.extend(logs_to_flush)

    async def _save_logs_to_db(self, logs: List[LogEntry]):
        """데이터베이스에 로그 저장"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                for log_entry in logs:
                    await db.execute("""
                        INSERT INTO logs (
                            timestamp, level, message, component, session_id,
                            user_id, tool_name, action, duration, success,
                            error_code, error_details, metadata, trace_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        log_entry.timestamp.isoformat(),
                        log_entry.level.value,
                        log_entry.message,
                        log_entry.component,
                        log_entry.session_id,
                        log_entry.user_id,
                        log_entry.tool_name,
                        log_entry.action,
                        log_entry.duration,
                        log_entry.success,
                        log_entry.error_code,
                        log_entry.error_details,
                        json.dumps(log_entry.metadata) if log_entry.metadata else None,
                        log_entry.trace_id
                    ))

                await db.commit()

        except Exception as e:
            print(f"Failed to save logs to database: {e}")
            raise

    async def _save_logs_to_file(self, logs: List[LogEntry]):
        """파일에 로그 저장"""
        try:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.log_file, 'a', encoding='utf-8') as f:
                for log_entry in logs:
                    f.write(log_entry.to_json() + '\n')

        except Exception as e:
            print(f"Failed to save logs to file: {e}")
            raise

    async def _log(self, level: LogLevel, message: str, **kwargs):
        """로그 기록"""
        try:
            # LogEntry 생성
            log_entry = LogEntry(
                level=level,
                message=message,
                component=kwargs.get('component', 'unknown'),
                session_id=kwargs.get('session_id'),
                user_id=kwargs.get('user_id'),
                tool_name=kwargs.get('tool_name'),
                action=kwargs.get('action'),
                duration=kwargs.get('duration'),
                success=kwargs.get('success'),
                error_code=kwargs.get('error_code'),
                error_details=kwargs.get('error_details'),
                metadata=kwargs.get('metadata', {}),
                trace_id=kwargs.get('trace_id')
            )

            # 콘솔 출력
            if self.enable_console:
                log_method = getattr(self.logger, level.value.lower())
                log_method(message, **kwargs)

            # 버퍼에 추가
            self.log_buffer.append(log_entry)

            # 버퍼가 가득 차면 즉시 플러시
            if len(self.log_buffer) >= self.buffer_size:
                await self._flush_logs()

        except Exception as e:
            print(f"Failed to log message: {e}")

    async def debug(self, message: str, **kwargs):
        """디버그 로그"""
        await self._log(LogLevel.DEBUG, message, **kwargs)

    async def info(self, message: str, **kwargs):
        """정보 로그"""
        await self._log(LogLevel.INFO, message, **kwargs)

    async def warning(self, message: str, **kwargs):
        """경고 로그"""
        await self._log(LogLevel.WARNING, message, **kwargs)

    async def error(self, message: str, **kwargs):
        """오류 로그"""
        await self._log(LogLevel.ERROR, message, **kwargs)

    async def critical(self, message: str, **kwargs):
        """중요 로그"""
        await self._log(LogLevel.CRITICAL, message, **kwargs)

    async def log_tool_execution(
        self,
        tool_name: str,
        action: str,
        duration: float,
        success: bool,
        session_id: Optional[str] = None,
        error_details: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """도구 실행 로그"""
        level = LogLevel.INFO if success else LogLevel.ERROR
        message = f"Tool '{tool_name}' {action} {'succeeded' if success else 'failed'}"

        await self._log(
            level,
            message,
            component="tool_manager",
            tool_name=tool_name,
            action=action,
            duration=duration,
            success=success,
            session_id=session_id,
            error_details=error_details,
            metadata=metadata or {}
        )

    async def log_user_action(
        self,
        action: str,
        session_id: str,
        user_id: Optional[str] = None,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """사용자 액션 로그"""
        level = LogLevel.INFO if success else LogLevel.WARNING
        message = f"User action: {action}"

        await self._log(
            level,
            message,
            component="user_interface",
            action=action,
            session_id=session_id,
            user_id=user_id,
            success=success,
            metadata=metadata or {}
        )

    async def log_system_event(
        self,
        event: str,
        component: str,
        level: LogLevel = LogLevel.INFO,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """시스템 이벤트 로그"""
        message = f"System event: {event}"

        await self._log(
            level,
            message,
            component=component,
            metadata=metadata or {}
        )

    async def log_exception(
        self,
        exception: Exception,
        component: str,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """예외 로그"""
        error_details = ''.join(traceback.format_exception(
            type(exception), exception, exception.__traceback__
        ))

        await self._log(
            LogLevel.ERROR,
            f"Exception in {component}: {str(exception)}",
            component=component,
            session_id=session_id,
            error_code=type(exception).__name__,
            error_details=error_details,
            metadata=context or {}
        )

    async def get_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[LogLevel] = None,
        component: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """로그 조회"""
        if not self.enable_database:
            return []

        try:
            query = "SELECT * FROM logs WHERE 1=1"
            params = []

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())

            if level:
                query += " AND level = ?"
                params.append(level.value)

            if component:
                query += " AND component = ?"
                params.append(component)

            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]

        except Exception as e:
            await self.error(f"Failed to get logs: {e}", component="logger")
            return []

    async def get_log_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """로그 통계 조회"""
        if not self.enable_database:
            return {}

        try:
            if not start_time:
                start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            if not end_time:
                end_time = datetime.now()

            stats = {}

            async with aiosqlite.connect(self.db_path) as db:
                # 총 로그 수
                async with db.execute(
                    "SELECT COUNT(*) as count FROM logs WHERE timestamp BETWEEN ? AND ?",
                    (start_time.isoformat(), end_time.isoformat())
                ) as cursor:
                    row = await cursor.fetchone()
                    stats['total_logs'] = row[0] if row else 0

                # 레벨별 통계
                async with db.execute("""
                    SELECT level, COUNT(*) as count
                    FROM logs
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY level
                """, (start_time.isoformat(), end_time.isoformat())) as cursor:
                    level_stats = {}
                    async for row in cursor:
                        level_stats[row[0]] = row[1]
                    stats['by_level'] = level_stats

                # 컴포넌트별 통계
                async with db.execute("""
                    SELECT component, COUNT(*) as count
                    FROM logs
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY component
                    ORDER BY count DESC
                    LIMIT 10
                """, (start_time.isoformat(), end_time.isoformat())) as cursor:
                    component_stats = {}
                    async for row in cursor:
                        component_stats[row[0]] = row[1]
                    stats['by_component'] = component_stats

                # 오류율
                async with db.execute("""
                    SELECT
                        COUNT(CASE WHEN level IN ('ERROR', 'CRITICAL') THEN 1 END) * 100.0 / COUNT(*) as error_rate
                    FROM logs
                    WHERE timestamp BETWEEN ? AND ?
                """, (start_time.isoformat(), end_time.isoformat())) as cursor:
                    row = await cursor.fetchone()
                    stats['error_rate'] = row[0] if row and row[0] else 0.0

            return stats

        except Exception as e:
            await self.error(f"Failed to get log stats: {e}", component="logger")
            return {}

    async def cleanup_old_logs(self, days_to_keep: int = 30) -> int:
        """오래된 로그 정리"""
        if not self.enable_database:
            return 0

        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    "DELETE FROM logs WHERE timestamp < ?",
                    (cutoff_date.isoformat(),)
                ) as cursor:
                    deleted_count = cursor.rowcount

                await db.commit()

                await self.info(
                    f"Cleaned up {deleted_count} old log entries",
                    component="logger",
                    metadata={'days_to_keep': days_to_keep}
                )

                return deleted_count

        except Exception as e:
            await self.error(f"Failed to cleanup old logs: {e}", component="logger")
            return 0