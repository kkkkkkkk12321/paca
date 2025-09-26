"""
Webhook Handler
웹훅 수신 및 처리를 위한 핸들러
"""

import asyncio
import json
import hashlib
import hmac
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime
from enum import Enum
import aiohttp
from aiohttp import web

from ...core.types import Result
from ...core.utils.logger import PacaLogger


class WebhookStatus(Enum):
    """웹훅 상태"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class WebhookConfig:
    """웹훅 설정"""
    name: str
    path: str
    methods: List[str] = field(default_factory=lambda: ["POST"])
    secret: Optional[str] = None
    signature_header: str = "X-Signature"
    timestamp_header: str = "X-Timestamp"
    max_body_size: int = 1024 * 1024  # 1MB
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 5.0


@dataclass
class WebhookEvent:
    """웹훅 이벤트"""
    id: str
    config_name: str
    method: str
    path: str
    headers: Dict[str, str]
    body: Union[Dict, str, bytes]
    remote_addr: str
    timestamp: datetime
    status: WebhookStatus = WebhookStatus.PENDING
    error: Optional[str] = None
    processing_time: Optional[float] = None
    retry_count: int = 0


class WebhookHandler:
    """웹훅 핸들러"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.logger = PacaLogger("WebhookHandler")

        # 웹훅 설정들
        self.configs: Dict[str, WebhookConfig] = {}

        # 이벤트 핸들러들
        self.handlers: Dict[str, List[Callable]] = {}

        # 이벤트 저장소
        self.events: Dict[str, WebhookEvent] = {}

        # aiohttp 앱
        self.app = web.Application()
        self.runner = None
        self.site = None

        # 통계
        self.stats = {
            "total_events": 0,
            "successful_events": 0,
            "failed_events": 0,
            "invalid_signature_events": 0,
            "total_processing_time": 0.0
        }

    def register_webhook(self, config: WebhookConfig) -> None:
        """웹훅 등록"""
        self.configs[config.name] = config
        self.handlers[config.name] = []

        # 라우트 등록
        for method in config.methods:
            self.app.router.add_route(
                method,
                config.path,
                self._create_handler(config.name)
            )

        self.logger.info(f"Webhook registered: {config.name} at {config.path}")

    def add_handler(self, config_name: str, handler: Callable) -> None:
        """웹훅 핸들러 추가"""
        if config_name not in self.handlers:
            self.handlers[config_name] = []

        self.handlers[config_name].append(handler)
        self.logger.info(f"Handler added for webhook: {config_name}")

    def _create_handler(self, config_name: str):
        """aiohttp 핸들러 생성"""
        async def handler(request):
            return await self._handle_webhook(request, config_name)
        return handler

    async def _handle_webhook(self, request: web.Request, config_name: str) -> web.Response:
        """웹훅 요청 처리"""
        config = self.configs[config_name]
        event_id = f"{config_name}_{int(time.time() * 1000)}"

        start_time = time.time()

        try:
            # 바디 읽기
            body = await request.read()

            # 크기 제한 확인
            if len(body) > config.max_body_size:
                return web.Response(
                    status=413,
                    text="Request body too large"
                )

            # 이벤트 객체 생성
            event = WebhookEvent(
                id=event_id,
                config_name=config_name,
                method=request.method,
                path=request.path,
                headers=dict(request.headers),
                body=body,
                remote_addr=request.remote,
                timestamp=datetime.now()
            )

            self.events[event_id] = event
            self.stats["total_events"] += 1

            # 서명 검증
            if config.secret:
                if not await self._verify_signature(event, config):
                    event.status = WebhookStatus.FAILED
                    event.error = "Invalid signature"
                    self.stats["invalid_signature_events"] += 1

                    return web.Response(
                        status=401,
                        text="Invalid signature"
                    )

            # 바디 파싱
            await self._parse_body(event)

            # 이벤트 처리
            event.status = WebhookStatus.PROCESSING
            result = await self._process_event(event, config)

            processing_time = time.time() - start_time
            event.processing_time = processing_time
            self.stats["total_processing_time"] += processing_time

            if result.is_success:
                event.status = WebhookStatus.COMPLETED
                self.stats["successful_events"] += 1

                return web.Response(
                    status=200,
                    text="Webhook processed successfully"
                )
            else:
                event.status = WebhookStatus.FAILED
                event.error = result.error
                self.stats["failed_events"] += 1

                # 재시도 스케줄링
                if event.retry_count < config.retry_attempts:
                    asyncio.create_task(self._schedule_retry(event, config))

                return web.Response(
                    status=500,
                    text=f"Webhook processing failed: {result.error}"
                )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)

            # 이벤트 업데이트
            if event_id in self.events:
                event = self.events[event_id]
                event.status = WebhookStatus.FAILED
                event.error = error_msg
                event.processing_time = processing_time

            self.stats["failed_events"] += 1
            self.logger.error(f"Webhook error for {config_name}: {error_msg}")

            return web.Response(
                status=500,
                text=f"Internal server error: {error_msg}"
            )

    async def _verify_signature(self, event: WebhookEvent, config: WebhookConfig) -> bool:
        """서명 검증"""
        try:
            signature_header = config.signature_header
            if signature_header not in event.headers:
                return False

            received_signature = event.headers[signature_header]

            # HMAC 서명 계산
            if isinstance(event.body, bytes):
                body_bytes = event.body
            else:
                body_bytes = str(event.body).encode()

            expected_signature = hmac.new(
                config.secret.encode(),
                body_bytes,
                hashlib.sha256
            ).hexdigest()

            # 서명 비교 (타이밍 공격 방지)
            if received_signature.startswith("sha256="):
                received_signature = received_signature[7:]

            return hmac.compare_digest(expected_signature, received_signature)

        except Exception as e:
            self.logger.error(f"Signature verification error: {str(e)}")
            return False

    async def _parse_body(self, event: WebhookEvent) -> None:
        """바디 파싱"""
        try:
            if isinstance(event.body, bytes):
                body_str = event.body.decode('utf-8')
            else:
                body_str = str(event.body)

            # JSON 파싱 시도
            try:
                event.body = json.loads(body_str)
            except json.JSONDecodeError:
                # JSON이 아니면 문자열로 유지
                event.body = body_str

        except Exception as e:
            self.logger.error(f"Body parsing error: {str(e)}")
            # 원본 바디 유지

    async def _process_event(self, event: WebhookEvent, config: WebhookConfig) -> Result[Any]:
        """이벤트 처리"""
        try:
            handlers = self.handlers.get(config.name, [])

            if not handlers:
                return Result(True, "No handlers registered")

            results = []
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        result = await asyncio.wait_for(
                            handler(event),
                            timeout=config.timeout
                        )
                    else:
                        result = handler(event)

                    results.append(result)

                except asyncio.TimeoutError:
                    error_msg = f"Handler timeout for {config.name}"
                    self.logger.error(error_msg)
                    return Result(False, None, error_msg)

                except Exception as e:
                    error_msg = f"Handler error for {config.name}: {str(e)}"
                    self.logger.error(error_msg)
                    return Result(False, None, error_msg)

            return Result(True, results)

        except Exception as e:
            return Result(False, None, f"Event processing error: {str(e)}")

    async def _schedule_retry(self, event: WebhookEvent, config: WebhookConfig) -> None:
        """재시도 스케줄링"""
        try:
            await asyncio.sleep(config.retry_delay * (2 ** event.retry_count))

            event.retry_count += 1
            event.status = WebhookStatus.RETRYING

            self.logger.info(f"Retrying webhook event: {event.id} (attempt {event.retry_count})")

            result = await self._process_event(event, config)

            if result.is_success:
                event.status = WebhookStatus.COMPLETED
                self.stats["successful_events"] += 1
            else:
                if event.retry_count < config.retry_attempts:
                    # 다시 재시도 스케줄링
                    asyncio.create_task(self._schedule_retry(event, config))
                else:
                    event.status = WebhookStatus.FAILED
                    event.error = f"Max retries exceeded: {result.error}"
                    self.stats["failed_events"] += 1

        except Exception as e:
            event.status = WebhookStatus.FAILED
            event.error = f"Retry error: {str(e)}"
            self.logger.error(f"Retry error for {event.id}: {str(e)}")

    async def start_server(self) -> Result[bool]:
        """웹훅 서버 시작"""
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()

            self.site = web.TCPSite(self.runner, self.host, self.port)
            await self.site.start()

            self.logger.info(f"Webhook server started on {self.host}:{self.port}")
            return Result(True, True)

        except Exception as e:
            self.logger.error(f"Failed to start webhook server: {str(e)}")
            return Result(False, False, str(e))

    async def stop_server(self) -> Result[bool]:
        """웹훅 서버 중지"""
        try:
            if self.site:
                await self.site.stop()
                self.site = None

            if self.runner:
                await self.runner.cleanup()
                self.runner = None

            self.logger.info("Webhook server stopped")
            return Result(True, True)

        except Exception as e:
            self.logger.error(f"Failed to stop webhook server: {str(e)}")
            return Result(False, False, str(e))

    def get_event(self, event_id: str) -> Optional[WebhookEvent]:
        """이벤트 조회"""
        return self.events.get(event_id)

    def get_events(
        self,
        config_name: Optional[str] = None,
        status: Optional[WebhookStatus] = None,
        limit: int = 100
    ) -> List[WebhookEvent]:
        """이벤트 목록 조회"""
        events = list(self.events.values())

        # 필터링
        if config_name:
            events = [e for e in events if e.config_name == config_name]

        if status:
            events = [e for e in events if e.status == status]

        # 최신순 정렬 및 제한
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """통계 정보"""
        avg_processing_time = (
            self.stats["total_processing_time"] / self.stats["successful_events"]
            if self.stats["successful_events"] > 0 else 0
        )

        success_rate = (
            self.stats["successful_events"] / self.stats["total_events"]
            if self.stats["total_events"] > 0 else 0
        )

        return {
            **self.stats,
            "average_processing_time": avg_processing_time,
            "success_rate": success_rate,
            "configured_webhooks": len(self.configs),
            "active_handlers": sum(len(handlers) for handlers in self.handlers.values()),
            "stored_events": len(self.events)
        }

    def clear_events(self, older_than_hours: int = 24) -> int:
        """오래된 이벤트 정리"""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        initial_count = len(self.events)

        self.events = {
            event_id: event
            for event_id, event in self.events.items()
            if event.timestamp > cutoff_time
        }

        cleared_count = initial_count - len(self.events)
        self.logger.info(f"Cleared {cleared_count} old events")
        return cleared_count


# 팩토리 함수들
def create_webhook_config(
    name: str,
    path: str,
    methods: Optional[List[str]] = None,
    secret: Optional[str] = None,
    **kwargs
) -> WebhookConfig:
    """웹훅 설정 생성"""
    return WebhookConfig(
        name=name,
        path=path,
        methods=methods or ["POST"],
        secret=secret,
        **kwargs
    )


# 일반적인 웹훅 핸들러 예시
async def log_webhook_handler(event: WebhookEvent) -> Dict[str, Any]:
    """로깅 웹훅 핸들러"""
    logger = PacaLogger("WebhookHandler")
    logger.info(f"Webhook received: {event.config_name} from {event.remote_addr}")
    return {"status": "logged", "event_id": event.id}


async def echo_webhook_handler(event: WebhookEvent) -> Dict[str, Any]:
    """에코 웹훅 핸들러"""
    return {
        "echo": event.body,
        "headers": event.headers,
        "timestamp": event.timestamp.isoformat()
    }