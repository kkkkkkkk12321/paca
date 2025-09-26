"""
Main Controller
PACA v5 메인 컨트롤러 - 전체 시스템 조정 및 요청 라우팅
"""

import asyncio
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime

from ..core.types.base import ID, Timestamp, Result, create_id, current_timestamp
from ..core.events.emitter import EventEmitter
from ..core.utils.logger import create_logger
from ..core.errors.base import PacaError

class ControllerState(Enum):
    """컨트롤러 상태"""
    IDLE = 'idle'
    INITIALIZING = 'initializing'
    RUNNING = 'running'
    PROCESSING = 'processing'
    PAUSED = 'paused'
    STOPPED = 'stopped'
    ERROR = 'error'

@dataclass
class ControllerConfig:
    """컨트롤러 설정"""
    max_concurrent_requests: int = 10
    request_timeout: float = 30.0
    enable_sentiment_analysis: bool = True
    enable_input_validation: bool = True
    enable_execution_control: bool = True
    log_level: str = 'INFO'
    middleware: List[str] = field(default_factory=list)
    plugins: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RequestContext:
    """요청 컨텍스트"""
    request_id: ID
    user_id: Optional[str]
    session_id: Optional[str]
    timestamp: Timestamp
    input_data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    source: str = 'unknown'

@dataclass
class ResponseContext:
    """응답 컨텍스트"""
    request_id: ID
    response_data: Any
    status_code: int
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class ControllerResult:
    """컨트롤러 결과"""
    success: bool
    data: Any
    error_message: Optional[str] = None
    processing_time: float = 0.0
    request_context: Optional[RequestContext] = None
    response_context: Optional[ResponseContext] = None

class ControllerError(PacaError):
    """컨트롤러 관련 오류"""
    pass

class MainController:
    """메인 컨트롤러 클래스"""

    def __init__(self, config: Optional[ControllerConfig] = None):
        self.config = config or ControllerConfig()
        self.state = ControllerState.IDLE
        self.logger = create_logger(__name__)
        self.event_emitter = EventEmitter()

        # 요청 처리 관련
        self.active_requests: Dict[ID, RequestContext] = {}
        self.request_handlers: Dict[str, Callable] = {}
        self.middleware_stack: List[Callable] = []

        # 통계 및 모니터링
        self.request_count = 0
        self.error_count = 0
        self.average_processing_time = 0.0
        self.start_time = current_timestamp()

        # 컴포넌트 초기화
        self._init_components()
        self._init_middleware()

    def _init_components(self):
        """컴포넌트 초기화"""
        try:
            # Sentiment Analyzer 초기화
            if self.config.enable_sentiment_analysis:
                from .sentiment import SentimentAnalyzer
                self.sentiment_analyzer = SentimentAnalyzer()
                self.logger.info("감정 분석기 초기화 완료")

            # Input Validator 초기화
            if self.config.enable_input_validation:
                from .validation import InputValidator
                self.input_validator = InputValidator()
                self.logger.info("입력 검증기 초기화 완료")

            # Execution Controller 초기화
            if self.config.enable_execution_control:
                from .execution import ExecutionController
                self.execution_controller = ExecutionController()
                self.logger.info("실행 제어기 초기화 완료")

        except Exception as e:
            self.logger.error(f"컴포넌트 초기화 실패: {e}")
            raise ControllerError(f"컴포넌트 초기화 실패: {e}")

    def _init_middleware(self):
        """미들웨어 초기화"""
        # 기본 미들웨어 추가
        self.middleware_stack = [
            self._request_logging_middleware,
            self._rate_limiting_middleware,
            self._error_handling_middleware
        ]

        # 설정에서 추가 미들웨어 로드
        for middleware_name in self.config.middleware:
            try:
                middleware = self._load_middleware(middleware_name)
                self.middleware_stack.append(middleware)
            except Exception as e:
                self.logger.warning(f"미들웨어 로드 실패 {middleware_name}: {e}")

    def _load_middleware(self, middleware_name: str) -> Callable:
        """미들웨어 로드"""
        # 실제 구현에서는 동적 로딩을 통해 미들웨어를 불러옴
        # 여기서는 기본 구현만 제공
        def default_middleware(context: RequestContext, next_handler: Callable):
            return next_handler(context)

        return default_middleware

    async def start(self):
        """컨트롤러 시작"""
        try:
            self.state = ControllerState.INITIALIZING
            self.logger.info("메인 컨트롤러 시작")

            # 컴포넌트 시작
            if hasattr(self, 'execution_controller'):
                await self.execution_controller.start()

            # 요청 핸들러 등록
            self._register_default_handlers()

            self.state = ControllerState.RUNNING
            self.event_emitter.emit('controller_started', {'timestamp': current_timestamp()})
            self.logger.info("메인 컨트롤러 시작 완료")

        except Exception as e:
            self.state = ControllerState.ERROR
            self.logger.error(f"컨트롤러 시작 실패: {e}")
            raise ControllerError(f"컨트롤러 시작 실패: {e}")

    async def stop(self):
        """컨트롤러 정지"""
        try:
            self.state = ControllerState.STOPPED
            self.logger.info("메인 컨트롤러 정지")

            # 활성 요청 완료 대기
            await self._wait_for_active_requests()

            # 컴포넌트 정지
            if hasattr(self, 'execution_controller'):
                await self.execution_controller.stop()

            self.event_emitter.emit('controller_stopped', {'timestamp': current_timestamp()})
            self.logger.info("메인 컨트롤러 정지 완료")

        except Exception as e:
            self.logger.error(f"컨트롤러 정지 실패: {e}")
            raise ControllerError(f"컨트롤러 정지 실패: {e}")

    async def process_request(
        self,
        input_data: Any,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ControllerResult:
        """요청 처리"""
        start_time = time.time()
        request_id = create_id()

        # 요청 컨텍스트 생성
        context = RequestContext(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            timestamp=current_timestamp(),
            input_data=input_data,
            metadata=metadata or {}
        )

        try:
            # 활성 요청 등록
            self.active_requests[request_id] = context
            self.request_count += 1

            # 미들웨어 체인 실행
            result = await self._execute_middleware_chain(context)

            # 처리 시간 계산
            processing_time = time.time() - start_time
            self._update_average_processing_time(processing_time)

            # 응답 컨텍스트 생성
            response_context = ResponseContext(
                request_id=request_id,
                response_data=result,
                status_code=200,
                processing_time=processing_time
            )

            self.logger.info(f"요청 처리 완료: {request_id} ({processing_time:.3f}s)")

            return ControllerResult(
                success=True,
                data=result,
                processing_time=processing_time,
                request_context=context,
                response_context=response_context
            )

        except Exception as e:
            self.error_count += 1
            processing_time = time.time() - start_time

            self.logger.error(f"요청 처리 실패: {request_id} - {e}")

            return ControllerResult(
                success=False,
                data=None,
                error_message=str(e),
                processing_time=processing_time,
                request_context=context
            )

        finally:
            # 활성 요청 제거
            self.active_requests.pop(request_id, None)

    async def _execute_middleware_chain(self, context: RequestContext) -> Any:
        """미들웨어 체인 실행"""
        async def execute_handler(handler_index: int = 0):
            if handler_index >= len(self.middleware_stack):
                # 모든 미들웨어 통과 후 실제 처리
                return await self._process_core_logic(context)

            middleware = self.middleware_stack[handler_index]

            async def next_handler():
                return await execute_handler(handler_index + 1)

            return await middleware(context, next_handler)

        return await execute_handler()

    async def _process_core_logic(self, context: RequestContext) -> Any:
        """핵심 로직 처리"""
        input_data = context.input_data

        # 1. 입력 검증
        if hasattr(self, 'input_validator'):
            validation_result = await self.input_validator.validate(input_data)
            if not validation_result.is_valid:
                raise ControllerError(f"입력 검증 실패: {validation_result.errors}")

        # 2. 감정 분석 (텍스트 입력인 경우)
        sentiment_result = None
        if hasattr(self, 'sentiment_analyzer') and isinstance(input_data, str):
            sentiment_result = await self.sentiment_analyzer.analyze(input_data)
            context.metadata['sentiment'] = sentiment_result.to_dict()

        # 3. 적절한 핸들러로 라우팅
        handler_name = self._determine_handler(input_data, sentiment_result)
        handler = self.request_handlers.get(handler_name, self._default_handler)

        # 4. 핸들러 실행
        result = await handler(context)

        return result

    def _determine_handler(self, input_data: Any, sentiment_result: Any) -> str:
        """적절한 핸들러 결정"""
        # 입력 데이터와 감정 분석 결과를 바탕으로 핸들러 선택
        if isinstance(input_data, dict):
            if 'command' in input_data:
                return 'command_handler'
            elif 'query' in input_data:
                return 'query_handler'
        elif isinstance(input_data, str):
            if sentiment_result and hasattr(sentiment_result, 'emotion_type'):
                if sentiment_result.emotion_type in ['angry', 'frustrated']:
                    return 'emotion_handler'
            return 'text_handler'

        return 'default_handler'

    def _register_default_handlers(self):
        """기본 핸들러 등록"""
        self.request_handlers = {
            'default_handler': self._default_handler,
            'text_handler': self._text_handler,
            'command_handler': self._command_handler,
            'query_handler': self._query_handler,
            'emotion_handler': self._emotion_handler
        }

    async def _default_handler(self, context: RequestContext) -> Any:
        """기본 핸들러"""
        return {
            'response': '요청을 처리했습니다.',
            'timestamp': current_timestamp(),
            'request_id': context.request_id
        }

    async def _text_handler(self, context: RequestContext) -> Any:
        """텍스트 처리 핸들러"""
        text = context.input_data
        return {
            'response': f'텍스트를 처리했습니다: {text[:50]}...' if len(text) > 50 else f'텍스트를 처리했습니다: {text}',
            'length': len(text),
            'timestamp': current_timestamp()
        }

    async def _command_handler(self, context: RequestContext) -> Any:
        """명령어 처리 핸들러"""
        command_data = context.input_data
        command = command_data.get('command', 'unknown')

        return {
            'response': f'명령어 "{command}"를 실행했습니다.',
            'command': command,
            'timestamp': current_timestamp()
        }

    async def _query_handler(self, context: RequestContext) -> Any:
        """쿼리 처리 핸들러"""
        query_data = context.input_data
        query = query_data.get('query', 'unknown')

        return {
            'response': f'쿼리 "{query}"를 처리했습니다.',
            'query': query,
            'timestamp': current_timestamp()
        }

    async def _emotion_handler(self, context: RequestContext) -> Any:
        """감정 기반 처리 핸들러"""
        sentiment = context.metadata.get('sentiment', {})
        emotion_type = sentiment.get('emotion_type', 'neutral')

        responses = {
            'angry': '화가 나셨군요. 도움이 되도록 노력하겠습니다.',
            'frustrated': '답답하셨군요. 더 명확하게 설명해드리겠습니다.',
            'happy': '기쁘시군요! 좋은 하루 되세요.',
            'sad': '슬프시군요. 기분이 나아지도록 도와드리겠습니다.'
        }

        response = responses.get(emotion_type, '감정을 이해했습니다.')

        return {
            'response': response,
            'detected_emotion': emotion_type,
            'timestamp': current_timestamp()
        }

    async def _request_logging_middleware(self, context: RequestContext, next_handler: Callable):
        """요청 로깅 미들웨어"""
        self.logger.info(f"요청 수신: {context.request_id} from {context.user_id}")
        result = await next_handler()
        self.logger.info(f"요청 완료: {context.request_id}")
        return result

    async def _rate_limiting_middleware(self, context: RequestContext, next_handler: Callable):
        """속도 제한 미들웨어"""
        if len(self.active_requests) >= self.config.max_concurrent_requests:
            raise ControllerError("동시 요청 수 제한 초과")

        return await next_handler()

    async def _error_handling_middleware(self, context: RequestContext, next_handler: Callable):
        """오류 처리 미들웨어"""
        try:
            return await next_handler()
        except Exception as e:
            self.logger.error(f"요청 처리 중 오류 발생: {context.request_id} - {e}")
            # 여기서 오류를 다시 발생시켜 상위에서 처리하도록 함
            raise

    async def _wait_for_active_requests(self, timeout: float = 10.0):
        """활성 요청 완료 대기"""
        start_time = time.time()
        while self.active_requests and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)

        if self.active_requests:
            self.logger.warning(f"타임아웃으로 인해 {len(self.active_requests)}개 요청이 강제 종료됨")

    def _update_average_processing_time(self, processing_time: float):
        """평균 처리 시간 업데이트"""
        if self.request_count == 1:
            self.average_processing_time = processing_time
        else:
            # 지수 이동 평균 사용
            alpha = 0.1
            self.average_processing_time = (
                alpha * processing_time +
                (1 - alpha) * self.average_processing_time
            )

    def register_handler(self, name: str, handler: Callable):
        """커스텀 핸들러 등록"""
        self.request_handlers[name] = handler
        self.logger.info(f"핸들러 등록: {name}")

    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        uptime = current_timestamp() - self.start_time

        return {
            'state': self.state.value,
            'uptime_seconds': uptime,
            'total_requests': self.request_count,
            'active_requests': len(self.active_requests),
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'average_processing_time': self.average_processing_time,
            'requests_per_second': self.request_count / max(uptime, 1)
        }

    def get_health_status(self) -> Dict[str, Any]:
        """헬스 체크 상태 반환"""
        stats = self.get_statistics()

        # 헬스 체크 기준
        is_healthy = (
            self.state == ControllerState.RUNNING and
            stats['error_rate'] < 0.05 and  # 5% 미만 오류율
            stats['average_processing_time'] < 5.0 and  # 5초 미만 평균 처리 시간
            len(self.active_requests) < self.config.max_concurrent_requests * 0.8  # 80% 미만 사용률
        )

        return {
            'healthy': is_healthy,
            'status': self.state.value,
            'components': {
                'sentiment_analyzer': hasattr(self, 'sentiment_analyzer'),
                'input_validator': hasattr(self, 'input_validator'),
                'execution_controller': hasattr(self, 'execution_controller')
            },
            'metrics': stats
        }