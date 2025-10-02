"""
Logger Utilities Module
구조화된 로깅 시스템 유틸리티
"""

import json
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
import traceback

from ..types.base import LogLevel, Timestamp, KeyValuePair, current_timestamp


@dataclass(frozen=True)
class LogEntry:
    """로그 엔트리"""
    timestamp: Timestamp
    level: LogLevel
    namespace: str
    message: str
    metadata: Optional[KeyValuePair] = None
    stack: Optional[str] = None


class LogAppender(ABC):
    """로그 출력기 인터페이스"""

    @abstractmethod
    def write(self, entry: LogEntry) -> None:
        """로그 엔트리 출력"""
        pass


class ConsoleAppender(LogAppender):
    """콘솔 로그 출력기"""

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode

    def write(self, entry: LogEntry) -> None:
        """콘솔에 로그 출력"""
        formatted_message = self._format_entry(entry)

        # UTF-8 인코딩 처리를 위한 설정
        try:
            if entry.level == LogLevel.DEBUG:
                print(formatted_message, file=sys.stdout, flush=True)
            elif entry.level == LogLevel.INFO:
                print(formatted_message, file=sys.stdout, flush=True)
            elif entry.level == LogLevel.WARN:
                print(formatted_message, file=sys.stderr, flush=True)
            elif entry.level in (LogLevel.ERROR, LogLevel.FATAL):
                print(formatted_message, file=sys.stderr, flush=True)
            else:
                print(formatted_message, file=sys.stdout, flush=True)
        except UnicodeEncodeError:
            # 인코딩 오류 시 ASCII로 fallback
            safe_message = formatted_message.encode('ascii', 'replace').decode('ascii')
            print(safe_message, file=sys.stderr, flush=True)

    def _format_entry(self, entry: LogEntry) -> str:
        """로그 엔트리 포맷팅"""
        timestamp = datetime.fromtimestamp(entry.timestamp).isoformat()
        namespace = f"[{entry.namespace}]" if entry.namespace else ""
        level = f"[{entry.level.value.upper()}]"
        metadata = f" {json.dumps(entry.metadata, ensure_ascii=False)}" if entry.metadata else ""
        stack = f"\n{entry.stack}" if entry.stack and self.debug_mode else ""

        return f"{timestamp} {level} {namespace} {entry.message}{metadata}{stack}"


class PacaLogger:
    """PACA 로거 클래스"""

    def __init__(
        self,
        namespace: str,
        appenders: List[LogAppender] = None,
        min_level: LogLevel = LogLevel.INFO
    ):
        self.namespace = namespace
        self.appenders = appenders or [ConsoleAppender()]
        self.min_level = min_level
        self._level_order = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARN, LogLevel.ERROR, LogLevel.FATAL]

    @property
    def is_debug_enabled(self) -> bool:
        """디버그 활성화 여부"""
        return self._should_log(LogLevel.DEBUG)

    def debug(self, message: str, meta: KeyValuePair = None) -> None:
        """디버그 로그"""
        self._log(LogLevel.DEBUG, message, meta)

    def info(self, message: str, meta: KeyValuePair = None) -> None:
        """정보 로그"""
        self._log(LogLevel.INFO, message, meta)

    def warn(self, message: str, meta: KeyValuePair = None) -> None:
        """경고 로그"""
        self._log(LogLevel.WARN, message, meta)

    def error(self, message: str, error: Exception = None, meta: KeyValuePair = None) -> None:
        """에러 로그"""
        metadata = dict(meta or {})
        stack = None

        if error:
            metadata['error'] = str(error)
            stack = traceback.format_exc() if isinstance(error, Exception) else None

        self._log(LogLevel.ERROR, message, metadata, stack)

    def fatal(self, message: str, error: Exception = None, meta: KeyValuePair = None) -> None:
        """치명적 로그"""
        metadata = dict(meta or {})
        stack = None

        if error:
            metadata['error'] = str(error)
            stack = traceback.format_exc() if isinstance(error, Exception) else None

        self._log(LogLevel.FATAL, message, metadata, stack)

    def _log(self, level: LogLevel, message: str, metadata: KeyValuePair = None, stack: str = None) -> None:
        """로그 출력"""
        if not self._should_log(level):
            return

        entry = LogEntry(
            timestamp=current_timestamp(),
            level=level,
            namespace=self.namespace,
            message=message,
            metadata=metadata,
            stack=stack
        )

        for appender in self.appenders:
            try:
                appender.write(entry)
            except Exception as e:
                print(f"Failed to write log entry: {e}", file=sys.stderr)

    def _should_log(self, level: LogLevel) -> bool:
        """로그 레벨 확인"""
        current_index = self._level_order.index(self.min_level)
        log_index = self._level_order.index(level)
        return log_index >= current_index


class FileAppender(LogAppender):
    """파일 로그 출력기"""

    def __init__(self, filename: str, debug_mode: bool = False):
        self.filename = filename
        self.debug_mode = debug_mode

    def write(self, entry: LogEntry) -> None:
        """파일에 로그 출력"""
        formatted_message = self._format_entry(entry)

        try:
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write(formatted_message + '\n')
        except IOError as e:
            print(f"Failed to write to log file {self.filename}: {e}", file=sys.stderr)

    def _format_entry(self, entry: LogEntry) -> str:
        """로그 엔트리 포맷팅"""
        timestamp = datetime.fromtimestamp(entry.timestamp).isoformat()
        namespace = f"[{entry.namespace}]" if entry.namespace else ""
        level = f"[{entry.level.value.upper()}]"
        metadata = f" {json.dumps(entry.metadata, ensure_ascii=False)}" if entry.metadata else ""
        stack = f"\n{entry.stack}" if entry.stack and self.debug_mode else ""

        return f"{timestamp} {level} {namespace} {entry.message}{metadata}{stack}"


# 로거 팩토리
_loggers: Dict[str, PacaLogger] = {}


def create_logger(
    namespace: str,
    appenders: List[LogAppender] = None,
    min_level: LogLevel = LogLevel.INFO
) -> PacaLogger:
    """로거 생성"""
    key = f"{namespace}-{len(appenders or [])}-{min_level.value}"

    if key not in _loggers:
        _loggers[key] = PacaLogger(namespace, appenders, min_level)

    return _loggers[key]


def get_logger(namespace: str) -> PacaLogger:
    """기존 로거 조회 또는 새 로거 생성"""
    return create_logger(namespace)


# 기본 로거 인스턴스
default_logger = create_logger('PACA')

# 표준 Python logging과의 브릿지
class PacaLogHandler(logging.Handler):
    """Python 표준 logging을 PACA 로거로 연결"""

    def __init__(self, paca_logger: PacaLogger):
        super().__init__()
        self.paca_logger = paca_logger

    def emit(self, record: logging.LogRecord) -> None:
        """로그 레코드 처리"""
        level_mapping = {
            logging.DEBUG: LogLevel.DEBUG,
            logging.INFO: LogLevel.INFO,
            logging.WARNING: LogLevel.WARN,
            logging.ERROR: LogLevel.ERROR,
            logging.CRITICAL: LogLevel.FATAL
        }

        paca_level = level_mapping.get(record.levelno, LogLevel.INFO)
        message = self.format(record)

        if record.exc_info:
            self.paca_logger.error(message, Exception(record.getMessage()))
        else:
            self.paca_logger._log(paca_level, message)


def setup_python_logging_bridge(namespace: str = 'python') -> None:
    """Python 표준 logging을 PACA 로거로 연결"""
    paca_logger = create_logger(namespace)
    handler = PacaLogHandler(paca_logger)

    # 루트 로거에 핸들러 추가
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)


# ===== Compatibility wrapper for __main__.py =====
# Some modules import `StructuredLogger` directly.
# Provide a thin wrapper over PacaLogger to keep the public API stable.
try:
    from ..types.base import LogLevel  # already imported above; kept for clarity
except Exception:
    pass

class StructuredLogger:
    """
    Thin wrapper around PacaLogger to match expected API.
    - name: logger namespace
    - log_level: "DEBUG" | "INFO" | "WARN" | "ERROR" (string)
    Methods:
      debug/info/warning/warn/error (sync)
      error_async (awaitable), and `async def error(...)` for compatibility
    """

    _LEVEL_MAP = {
        "DEBUG": LogLevel.DEBUG,
        "INFO": LogLevel.INFO,
        "WARN": LogLevel.WARN,
        "WARNING": LogLevel.WARN,
        "ERROR": LogLevel.ERROR,
        "FATAL": LogLevel.FATAL,
    }

    def __init__(self, name: str = "PacaLogger", log_level: str = "INFO"):
        level = self._LEVEL_MAP.get(str(log_level).upper(), LogLevel.INFO)
        # Use existing logger implementation
        self._inner = PacaLogger(namespace=name, min_level=level)

    # level controls
    def set_level(self, log_level: str) -> None:
        level = self._LEVEL_MAP.get(str(log_level).upper(), LogLevel.INFO)
        self._inner.min_level = level

    # passthrough helpers
    def debug(self, message: str, **meta) -> None:
        self._inner.debug(message, meta or None)

    def info(self, message: str, **meta) -> None:
        self._inner.info(message, meta or None)

    def warning(self, message: str, **meta) -> None:
        self._inner.warn(message, meta or None)

    # alias commonly used name
    warn = warning

    def error(self, message: str, **meta) -> None:
        # keep sync path; some code may call without await
        self._inner.error(message, None, meta or None)

    def exception(self, message: str, **meta) -> None:
        """Log an exception while preserving stack information."""
        error = meta.pop("error", None)
        if error is None and "exc_info" in meta:
            error = meta.pop("exc_info")

        if error is None:
            exc_type, exc_value, exc_tb = sys.exc_info()
            if exc_value is not None:
                error = exc_value
            else:
                stack_trace = "".join(traceback.format_stack()[:-1])
                self._inner._log(LogLevel.ERROR, message, meta or None, stack_trace)
                return

        self._inner.error(message, error, meta or None)

    async def error_async(self, message: str, **meta) -> None:
        # allow `await logger.error_async(...)`
        self._inner.error(message, None, meta or None)

    async def exception_async(self, message: str, **meta) -> None:
        """Async-compatible alias for ``exception``."""
        self.exception(message, **meta)

    async_exception = exception_async

    # Some code may `await logger.error(...)`; support that too.
    async def __call_error_async_compat(self, message: str, **meta) -> None:
        self._inner.error(message, None, meta or None)
    # expose the async-compatible name
    async_error = error_async

