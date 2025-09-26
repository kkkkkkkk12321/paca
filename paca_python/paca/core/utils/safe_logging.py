"""
PACA Safe Logging System
Windows 파일 잠금 문제를 해결하는 안전한 로깅 시스템
"""

import os
import sys
import logging
import logging.handlers
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import threading
import queue
import atexit
from .safe_print import safe_print, emoji_to_text


class SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Windows 환경에서 안전한 로그 로테이션 핸들러"""

    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None, delay=False):
        # Windows에서 UTF-8 인코딩 강제
        if encoding is None:
            encoding = 'utf-8'
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self._lock = threading.Lock()

    def emit(self, record):
        """레코드 안전 출력"""
        with self._lock:
            try:
                # 이모지 제거
                if hasattr(record, 'msg') and isinstance(record.msg, str):
                    record.msg = emoji_to_text(record.msg)

                # 파일 핸들 상태 확인
                if self.stream is None:
                    self.stream = self._open()

                super().emit(record)

                # 즉시 플러시
                if hasattr(self.stream, 'flush'):
                    self.stream.flush()

            except (OSError, IOError) as e:
                # 파일 잠금 등의 문제 발생 시 재시도
                if "being used by another process" in str(e):
                    self._retry_emit(record)
                else:
                    self.handleError(record)

    def _retry_emit(self, record, max_retries=3):
        """파일 잠금 시 재시도"""
        for attempt in range(max_retries):
            try:
                time.sleep(0.1 * (attempt + 1))  # 지수적 백오프

                # 스트림 재생성
                if self.stream:
                    self.stream.close()
                    self.stream = None

                self.stream = self._open()
                super().emit(record)
                return
            except Exception:
                if attempt == max_retries - 1:
                    # 최종 실패 시 콘솔로 출력
                    try:
                        safe_print(f"[LOG_ERROR] {record.getMessage()}")
                    except:
                        pass

    def doRollover(self):
        """안전한 로그 로테이션"""
        with self._lock:
            try:
                super().doRollover()
            except (OSError, IOError):
                # 로테이션 실패 시 현재 로그 파일 계속 사용
                pass


class QueuedLoggingHandler(logging.Handler):
    """큐 기반 비동기 로깅 핸들러"""

    def __init__(self, target_handler):
        super().__init__()
        self.target_handler = target_handler
        self.log_queue = queue.Queue()
        self.worker_thread = None
        self.should_stop = threading.Event()
        self._start_worker()

    def _start_worker(self):
        """워커 스레드 시작"""
        self.worker_thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="SafeLoggingWorker"
        )
        self.worker_thread.start()

    def _worker(self):
        """로그 처리 워커"""
        while not self.should_stop.is_set():
            try:
                record = self.log_queue.get(timeout=0.5)
                if record is None:  # 종료 신호
                    break
                self.target_handler.emit(record)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                # 워커에서의 오류는 무시 (무한 루프 방지)
                pass

    def emit(self, record):
        """레코드를 큐에 추가"""
        try:
            self.log_queue.put_nowait(record)
        except queue.Full:
            # 큐가 가득 찬 경우 가장 오래된 레코드 제거 후 추가
            try:
                self.log_queue.get_nowait()
                self.log_queue.put_nowait(record)
            except queue.Empty:
                pass

    def close(self):
        """핸들러 종료"""
        self.should_stop.set()
        if self.worker_thread and self.worker_thread.is_alive():
            # 종료 신호 전송
            self.log_queue.put(None)
            self.worker_thread.join(timeout=1.0)
        super().close()


class SafeLogger:
    """안전한 로거 관리자"""

    def __init__(self, name: str = "PACA", log_dir: Optional[Path] = None):
        self.name = name
        self.log_dir = log_dir or Path("logs")
        self.logger = None
        self.handlers = []
        self._setup_logger()

    def _setup_logger(self):
        """로거 설정"""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)

        # 기존 핸들러 제거
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        self.handlers.clear()

        # 로그 디렉토리 생성
        self.log_dir.mkdir(exist_ok=True)

        # 콘솔 핸들러 (항상 추가)
        self._add_console_handler()

        # 파일 핸들러 (가능한 경우만)
        try:
            self._add_file_handler()
        except Exception as e:
            safe_print(f"[WARNING] File logging not available: {e}")

    def _add_console_handler(self):
        """콘솔 핸들러 추가"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)
        self.handlers.append(console_handler)

    def _add_file_handler(self):
        """파일 핸들러 추가"""
        log_file = self.log_dir / f"{self.name.lower()}.log"

        # 로테이팅 파일 핸들러
        file_handler = SafeRotatingFileHandler(
            str(log_file),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        # 큐 기반 핸들러로 래핑
        queued_handler = QueuedLoggingHandler(file_handler)
        queued_handler.setLevel(logging.DEBUG)

        self.logger.addHandler(queued_handler)
        self.handlers.append(queued_handler)

    def get_logger(self) -> logging.Logger:
        """로거 반환"""
        return self.logger

    def info(self, message: str, **kwargs):
        """정보 로그"""
        self.logger.info(emoji_to_text(str(message)), **kwargs)

    def debug(self, message: str, **kwargs):
        """디버그 로그"""
        self.logger.debug(emoji_to_text(str(message)), **kwargs)

    def warning(self, message: str, **kwargs):
        """경고 로그"""
        self.logger.warning(emoji_to_text(str(message)), **kwargs)

    def error(self, message: str, **kwargs):
        """오류 로그"""
        self.logger.error(emoji_to_text(str(message)), **kwargs)

    def critical(self, message: str, **kwargs):
        """치명적 오류 로그"""
        self.logger.critical(emoji_to_text(str(message)), **kwargs)

    def close(self):
        """로거 종료"""
        for handler in self.handlers:
            try:
                handler.close()
            except:
                pass
        self.handlers.clear()


class LoggerManager:
    """로거 관리자 싱글톤"""

    _instance = None
    _loggers: Dict[str, SafeLogger] = {}
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    # 프로그램 종료 시 정리
                    atexit.register(cls._instance.cleanup)
        return cls._instance

    def get_logger(self, name: str = "PACA", log_dir: Optional[Path] = None) -> SafeLogger:
        """로거 가져오기"""
        if name not in self._loggers:
            with self._lock:
                if name not in self._loggers:
                    self._loggers[name] = SafeLogger(name, log_dir)
        return self._loggers[name]

    def cleanup(self):
        """모든 로거 정리"""
        with self._lock:
            for logger in self._loggers.values():
                logger.close()
            self._loggers.clear()


def get_safe_logger(name: str = "PACA", log_dir: Optional[Path] = None) -> SafeLogger:
    """안전한 로거 가져오기"""
    manager = LoggerManager()
    return manager.get_logger(name, log_dir)


def configure_safe_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    disable_file_logging: bool = False
) -> SafeLogger:
    """안전한 로깅 설정"""

    # 로그 레벨 설정
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # 기본 로그 디렉토리
    if log_dir is None:
        log_dir = Path("logs")

    # 안전한 로거 생성
    logger = get_safe_logger("PACA", log_dir)
    logger.logger.setLevel(numeric_level)

    # 파일 로깅 비활성화 옵션
    if disable_file_logging:
        # 파일 핸들러만 제거
        for handler in logger.handlers[:]:
            if isinstance(handler, (SafeRotatingFileHandler, QueuedLoggingHandler)):
                logger.logger.removeHandler(handler)
                handler.close()
                logger.handlers.remove(handler)

    return logger


# 전역 기본 로거
default_logger = None

def init_default_logger():
    """기본 로거 초기화"""
    global default_logger
    if default_logger is None:
        try:
            default_logger = configure_safe_logging()
        except Exception as e:
            safe_print(f"[WARNING] Logger initialization failed: {e}")


# 모듈 로드 시 기본 로거 초기화
init_default_logger()