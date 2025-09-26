"""
API Base Module
API 엔드포인트와 라우팅을 위한 기본 클래스들과 인터페이스
"""

import re
import time
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Awaitable, Union
from dataclasses import dataclass, field

from ..core.types import ID, Timestamp, Result, create_success, create_failure, create_result
from ..core.errors import ValidationError, InfrastructureError
from ..core.events import EventEmitter


class HttpMethod(Enum):
    """HTTP 메서드"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"


class HttpStatus(Enum):
    """HTTP 상태 코드"""
    OK = 200
    CREATED = 201
    NO_CONTENT = 204
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    CONFLICT = 409
    UNPROCESSABLE_ENTITY = 422
    INTERNAL_SERVER_ERROR = 500
    SERVICE_UNAVAILABLE = 503


class ContentType(Enum):
    """콘텐츠 타입"""
    JSON = "application/json"
    XML = "application/xml"
    FORM = "application/x-www-form-urlencoded"
    MULTIPART = "multipart/form-data"
    TEXT = "text/plain"
    HTML = "text/html"


@dataclass
class ApiRequest:
    """API 요청"""
    id: ID
    method: HttpMethod
    url: str
    path: str
    query: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[Any] = None
    timestamp: Timestamp = field(default_factory=lambda: int(time.time() * 1000))
    user_agent: Optional[str] = None
    ip: Optional[str] = None
    user_id: Optional[ID] = None


@dataclass
class ApiResponse:
    """API 응답"""
    request_id: ID
    status: HttpStatus
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[Any] = None
    timestamp: Timestamp = field(default_factory=lambda: int(time.time() * 1000))
    processing_time: float = 0.0
    cached: bool = False


@dataclass
class ApiError:
    """API 에러"""
    name: str
    code: str
    message: str
    details: Optional[Any] = None
    timestamp: Timestamp = field(default_factory=lambda: int(time.time() * 1000))
    request_id: Optional[ID] = None


# Type aliases
RouteHandler = Callable[[ApiRequest], Awaitable[Result[ApiResponse]]]
Middleware = Callable[[ApiRequest, Callable[[], Awaitable[Result[ApiResponse]]]], Awaitable[Result[ApiResponse]]]


@dataclass
class FieldValidation:
    """필드 검증"""
    type: str  # 'string', 'number', 'boolean', 'array', 'object'
    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    enum: Optional[List[Any]] = None
    custom_validator: Optional[Callable[[Any], bool]] = None


@dataclass
class ValidationSchema:
    """검증 스키마"""
    query: Optional[Dict[str, FieldValidation]] = None
    params: Optional[Dict[str, FieldValidation]] = None
    body: Optional[Dict[str, FieldValidation]] = None
    headers: Optional[Dict[str, FieldValidation]] = None


@dataclass
class RateLimitConfig:
    """속도 제한 설정"""
    window_ms: int
    max_requests: int
    skip_successful_requests: bool = False
    skip_failed_requests: bool = False
    key_generator: Optional[Callable[[ApiRequest], str]] = None


@dataclass
class CacheConfig:
    """캐시 설정"""
    ttl: int
    key_generator: Optional[Callable[[ApiRequest], str]] = None
    conditions: Optional[Callable[[ApiRequest], bool]] = None


@dataclass
class RouteDefinition:
    """라우트 정의"""
    method: HttpMethod
    path: str
    handler: RouteHandler
    middlewares: Optional[List[Middleware]] = None
    validation: Optional[ValidationSchema] = None
    authentication: bool = False
    rate_limit: Optional[RateLimitConfig] = None
    cache: Optional[CacheConfig] = None


class BaseRouter(ABC):
    """기본 라우터 추상 클래스"""

    def __init__(self, name: str, events: Optional[EventEmitter] = None):
        self.routes: Dict[str, RouteDefinition] = {}
        self.middlewares: List[Middleware] = []
        self.events = events
        self.logger = logging.getLogger(f"API:{name}")

        # 통계
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0

    def add_route(self, definition: RouteDefinition) -> None:
        """라우트 등록"""
        route_key = self._create_route_key(definition.method, definition.path)
        self.routes[route_key] = definition

        self.logger.debug(
            f"Route registered: {definition.method.value} {definition.path}",
            extra={
                "method": definition.method.value,
                "path": definition.path,
                "authentication": definition.authentication,
                "middlewares": len(definition.middlewares or [])
            }
        )

    def use(self, middleware: Middleware) -> None:
        """미들웨어 등록"""
        self.middlewares.append(middleware)
        self.logger.debug("Middleware registered")

    def get(self, path: str, handler: RouteHandler, **options) -> None:
        """GET 라우트 등록"""
        definition = RouteDefinition(
            method=HttpMethod.GET,
            path=path,
            handler=handler,
            **options
        )
        self.add_route(definition)

    def post(self, path: str, handler: RouteHandler, **options) -> None:
        """POST 라우트 등록"""
        definition = RouteDefinition(
            method=HttpMethod.POST,
            path=path,
            handler=handler,
            **options
        )
        self.add_route(definition)

    def put(self, path: str, handler: RouteHandler, **options) -> None:
        """PUT 라우트 등록"""
        definition = RouteDefinition(
            method=HttpMethod.PUT,
            path=path,
            handler=handler,
            **options
        )
        self.add_route(definition)

    def patch(self, path: str, handler: RouteHandler, **options) -> None:
        """PATCH 라우트 등록"""
        definition = RouteDefinition(
            method=HttpMethod.PATCH,
            path=path,
            handler=handler,
            **options
        )
        self.add_route(definition)

    def delete(self, path: str, handler: RouteHandler, **options) -> None:
        """DELETE 라우트 등록"""
        definition = RouteDefinition(
            method=HttpMethod.DELETE,
            path=path,
            handler=handler,
            **options
        )
        self.add_route(definition)

    async def handle_request(self, request: ApiRequest) -> Result[ApiResponse]:
        """요청 처리"""
        start_time = time.time()
        self.request_count += 1

        try:
            await self._emit_event('request.received', {'request': request})

            # 라우트 찾기
            route = self._find_route(request.method, request.path)
            if not route:
                return self._create_error_response(
                    request.id,
                    HttpStatus.NOT_FOUND,
                    'ROUTE_NOT_FOUND',
                    f'Route not found: {request.method.value} {request.path}',
                    start_time
                )

            # 요청 검증
            validation_result = await self._validate_request(request, route.validation)
            if validation_result.is_failure:
                return self._create_error_response(
                    request.id,
                    HttpStatus.BAD_REQUEST,
                    'VALIDATION_ERROR',
                    validation_result.error.message,
                    start_time
                )

            # 미들웨어 체인 실행
            response = await self._execute_middlewares(request, route)

            response_time = time.time() - start_time
            self.total_response_time += response_time

            await self._emit_event('request.completed', {
                'request_id': request.id,
                'status': response.value.status.value if response.is_success else HttpStatus.INTERNAL_SERVER_ERROR.value,
                'response_time': response_time
            })

            return response

        except Exception as error:
            self.error_count += 1
            self.logger.error(
                f"Request handling failed: {error}",
                extra={
                    "request_id": request.id,
                    "method": request.method.value,
                    "path": request.path
                },
                exc_info=True
            )

            return self._create_error_response(
                request.id,
                HttpStatus.INTERNAL_SERVER_ERROR,
                'INTERNAL_ERROR',
                'Internal server error',
                start_time
            )

    def get_statistics(self) -> Dict[str, Any]:
        """라우터 통계 조회"""
        return {
            'total_requests': self.request_count,
            'error_count': self.error_count,
            'success_rate': (self.request_count - self.error_count) / self.request_count if self.request_count > 0 else 0,
            'average_response_time': self.total_response_time / self.request_count if self.request_count > 0 else 0,
            'routes_count': len(self.routes),
            'middleware_count': len(self.middlewares)
        }

    # Protected methods
    def _find_route(self, method: HttpMethod, path: str) -> Optional[RouteDefinition]:
        """라우트 찾기"""
        route_key = self._create_route_key(method, path)
        route = self.routes.get(route_key)

        if not route:
            # 매개변수가 있는 라우트 검색
            for key, route_def in self.routes.items():
                if key.startswith(f"{method.value}:"):
                    route_path = key[len(method.value) + 1:]
                    if self._match_path(route_path, path):
                        route = route_def
                        # 매개변수 추출 및 요청에 추가
                        # 참고: 실제 구현에서는 request 객체를 수정해야 함
                        break

        return route

    def _match_path(self, route_path: str, request_path: str) -> bool:
        """경로 매칭"""
        route_parts = route_path.split('/')
        path_parts = request_path.split('/')

        if len(route_parts) != len(path_parts):
            return False

        for route_part, path_part in zip(route_parts, path_parts):
            if route_part.startswith(':'):
                continue  # 매개변수는 모든 값과 매치
            if route_part != path_part:
                return False

        return True

    def _extract_params(self, route_path: str, request_path: str) -> Dict[str, str]:
        """매개변수 추출"""
        params = {}
        route_parts = route_path.split('/')
        path_parts = request_path.split('/')

        for route_part, path_part in zip(route_parts, path_parts):
            if route_part.startswith(':'):
                param_name = route_part[1:]
                params[param_name] = path_part

        return params

    async def _validate_request(
        self,
        request: ApiRequest,
        schema: Optional[ValidationSchema]
    ) -> Result[bool]:
        """요청 검증"""
        if not schema:
            return create_result(True, True)

        # Query 매개변수 검증
        if schema.query:
            result = self._validate_fields(request.query, schema.query, 'query')
            if result.is_failure:
                return result

        # URL 매개변수 검증
        if schema.params:
            result = self._validate_fields(request.params, schema.params, 'params')
            if result.is_failure:
                return result

        # Body 검증
        if schema.body:
            result = self._validate_fields(request.body, schema.body, 'body')
            if result.is_failure:
                return result

        # Headers 검증
        if schema.headers:
            result = self._validate_fields(request.headers, schema.headers, 'headers')
            if result.is_failure:
                return result

        return create_result(True, True)

    def _validate_fields(
        self,
        data: Any,
        schema: Dict[str, FieldValidation],
        context: str
    ) -> Result[bool]:
        """필드 검증"""
        for field_name, validation in schema.items():
            value = data.get(field_name) if data else None

            if validation.required and value is None:
                return create_result(
                    False,
                    None,
                    ValidationError(
                        f"Required field '{field_name}' is missing in {context}",
                        {'context': context, 'field_name': field_name, 'value': value},
                        ['Provide a valid value for this required field']
                    )
                )

            if value is not None:
                type_result = self._validate_field_type(value, validation.type, f"{context}.{field_name}")
                if type_result.is_failure:
                    return type_result

                constraint_result = self._validate_field_constraints(value, validation, f"{context}.{field_name}")
                if constraint_result.is_failure:
                    return constraint_result

        return create_result(True, True)

    def _validate_field_type(
        self,
        value: Any,
        expected_type: str,
        field_name: str
    ) -> Result[bool]:
        """필드 타입 검증"""
        if expected_type == 'array':
            if not isinstance(value, list):
                return create_result(
                    False,
                    None,
                    ValidationError(
                        f"Type mismatch for field '{field_name}': expected array, got {type(value).__name__}",
                        {'field_name': field_name, 'value': value, 'expected_type': expected_type, 'actual_type': type(value).__name__},
                        [f"Convert the value to {expected_type} type"]
                    )
                )
        elif expected_type == 'string':
            if not isinstance(value, str):
                return create_result(
                    False,
                    None,
                    ValidationError(
                        f"Type mismatch for field '{field_name}': expected string, got {type(value).__name__}",
                        {'field_name': field_name, 'value': value, 'expected_type': expected_type, 'actual_type': type(value).__name__},
                        [f"Convert the value to {expected_type} type"]
                    )
                )
        elif expected_type == 'number':
            if not isinstance(value, (int, float)):
                return create_result(
                    False,
                    None,
                    ValidationError(
                        f"Type mismatch for field '{field_name}': expected number, got {type(value).__name__}",
                        {'field_name': field_name, 'value': value, 'expected_type': expected_type, 'actual_type': type(value).__name__},
                        [f"Convert the value to {expected_type} type"]
                    )
                )
        elif expected_type == 'boolean':
            if not isinstance(value, bool):
                return create_result(
                    False,
                    None,
                    ValidationError(
                        f"Type mismatch for field '{field_name}': expected boolean, got {type(value).__name__}",
                        {'field_name': field_name, 'value': value, 'expected_type': expected_type, 'actual_type': type(value).__name__},
                        [f"Convert the value to {expected_type} type"]
                    )
                )
        elif expected_type == 'object':
            if not isinstance(value, dict):
                return create_result(
                    False,
                    None,
                    ValidationError(
                        f"Type mismatch for field '{field_name}': expected object, got {type(value).__name__}",
                        {'field_name': field_name, 'value': value, 'expected_type': expected_type, 'actual_type': type(value).__name__},
                        [f"Convert the value to {expected_type} type"]
                    )
                )

        return create_result(True, True)

    def _validate_field_constraints(
        self,
        value: Any,
        validation: FieldValidation,
        field_name: str
    ) -> Result[bool]:
        """필드 제약 조건 검증"""
        if validation.min_length is not None and isinstance(value, str) and len(value) < validation.min_length:
            return create_result(
                False,
                None,
                ValidationError(
                    f"Field '{field_name}' is too short: minimum length is {validation.min_length}",
                    {'field_name': field_name, 'value': value, 'min_length': validation.min_length, 'actual_length': len(value)},
                    [f"Provide a value with at least {validation.min_length} characters"]
                )
            )

        if validation.max_length is not None and isinstance(value, str) and len(value) > validation.max_length:
            return create_result(
                False,
                None,
                ValidationError(
                    f"Field '{field_name}' is too long: maximum length is {validation.max_length}",
                    {'field_name': field_name, 'value': value, 'max_length': validation.max_length, 'actual_length': len(value)},
                    [f"Provide a value with at most {validation.max_length} characters"]
                )
            )

        if validation.min_value is not None and isinstance(value, (int, float)) and value < validation.min_value:
            return create_result(
                False,
                None,
                ValidationError(
                    f"Field '{field_name}' value is too small: minimum value is {validation.min_value}",
                    {'field_name': field_name, 'value': value, 'min_value': validation.min_value},
                    [f"Provide a value greater than or equal to {validation.min_value}"]
                )
            )

        if validation.max_value is not None and isinstance(value, (int, float)) and value > validation.max_value:
            return create_result(
                False,
                None,
                ValidationError(
                    f"Field '{field_name}' value is too large: maximum value is {validation.max_value}",
                    {'field_name': field_name, 'value': value, 'max_value': validation.max_value},
                    [f"Provide a value less than or equal to {validation.max_value}"]
                )
            )

        if validation.pattern and isinstance(value, str) and not re.match(validation.pattern, value):
            return create_result(
                False,
                None,
                ValidationError(
                    f"Field '{field_name}' does not match required pattern: {validation.pattern}",
                    {'field_name': field_name, 'value': value, 'pattern': validation.pattern},
                    [f"Provide a value that matches the pattern {validation.pattern}"]
                )
            )

        if validation.enum and value not in validation.enum:
            return create_result(
                False,
                None,
                ValidationError(
                    f"Field '{field_name}' must be one of: {', '.join(str(v) for v in validation.enum)}",
                    {'field_name': field_name, 'value': value, 'allowed_values': validation.enum},
                    [f"Choose one of the allowed values: {', '.join(str(v) for v in validation.enum)}"]
                )
            )

        if validation.custom_validator and not validation.custom_validator(value):
            return create_result(
                False,
                None,
                ValidationError(
                    f"Field '{field_name}' failed custom validation",
                    {'field_name': field_name, 'value': value},
                    ['Check the custom validation requirements for this field']
                )
            )

        return create_result(True, True)

    async def _execute_middlewares(
        self,
        request: ApiRequest,
        route: RouteDefinition
    ) -> Result[ApiResponse]:
        """미들웨어 체인 실행"""
        all_middlewares = self.middlewares + (route.middlewares or [])

        async def execute_next(index: int) -> Result[ApiResponse]:
            if index >= len(all_middlewares):
                return await route.handler(request)

            middleware = all_middlewares[index]
            return await middleware(request, lambda: execute_next(index + 1))

        return await execute_next(0)

    def _create_route_key(self, method: HttpMethod, path: str) -> str:
        """라우트 키 생성"""
        return f"{method.value}:{path}"

    def _create_error_response(
        self,
        request_id: ID,
        status: HttpStatus,
        code: str,
        message: str,
        start_time: float
    ) -> Result[ApiResponse]:
        """에러 응답 생성"""
        error = ApiError(
            name='ApiError',
            code=code,
            message=message,
            timestamp=int(time.time() * 1000),
            request_id=request_id
        )

        response = ApiResponse(
            request_id=request_id,
            status=status,
            headers={'Content-Type': ContentType.JSON.value},
            body={'error': error},
            timestamp=int(time.time() * 1000),
            processing_time=time.time() - start_time
        )

        return create_result(True, response)

    async def _emit_event(self, event_name: str, data: Any) -> None:
        """이벤트 발행"""
        if self.events:
            await self.events.emit(f"api.{event_name}", data)


# ==========================================
# 내보내기
# ==========================================

__all__ = [
    # Enums
    'HttpMethod', 'HttpStatus', 'ContentType',

    # Data Classes
    'ApiRequest', 'ApiResponse', 'ApiError',
    'FieldValidation', 'ValidationSchema',
    'RateLimitConfig', 'CacheConfig', 'RouteDefinition',

    # Type Aliases
    'RouteHandler', 'Middleware',

    # Classes
    'BaseRouter'
]