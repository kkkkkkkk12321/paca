"""
범용 API 클라이언트
다양한 API와 통신하기 위한 통합 클라이언트
"""

import asyncio
import aiohttp
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging

from ...core.types import Result
from ...core.utils.logger import PacaLogger
from .rate_limiter import RateLimiter
from .auth_manager import AuthManager
from .circuit_breaker import CircuitBreaker


class HTTPMethod(Enum):
    """HTTP 메서드"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class ContentType(Enum):
    """컨텐츠 타입"""
    JSON = "application/json"
    FORM = "application/x-www-form-urlencoded"
    MULTIPART = "multipart/form-data"
    TEXT = "text/plain"
    XML = "application/xml"


@dataclass
class APIEndpoint:
    """API 엔드포인트 정의"""
    name: str
    url: str
    method: HTTPMethod
    auth_required: bool = True
    rate_limit: Optional[int] = None  # requests per minute
    timeout: float = 30.0
    retry_attempts: int = 3
    content_type: ContentType = ContentType.JSON
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class APIRequest:
    """API 요청"""
    endpoint: APIEndpoint
    data: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    files: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = None


@dataclass
class APIResponse:
    """API 응답"""
    status_code: int
    data: Any
    headers: Dict[str, str]
    response_time: float
    endpoint_name: str
    success: bool = True
    error: Optional[str] = None


class UniversalAPIClient:
    """범용 API 클라이언트"""

    def __init__(
        self,
        base_url: str = "",
        default_timeout: float = 30.0,
        max_connections: int = 100,
        enable_rate_limiting: bool = True,
        enable_circuit_breaker: bool = True
    ):
        self.base_url = base_url.rstrip('/')
        self.default_timeout = default_timeout
        self.logger = PacaLogger("UniversalAPIClient")

        # 컴포넌트 초기화
        self.rate_limiter = RateLimiter() if enable_rate_limiting else None
        self.auth_manager = AuthManager()
        self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None

        # 세션 설정
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=20,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        self.session = None

        # 엔드포인트 레지스트리
        self.endpoints: Dict[str, APIEndpoint] = {}

        # 통계
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_response_time": 0.0,
            "cache_hits": 0
        }

        # 캐시
        self.response_cache: Dict[str, Dict] = {}
        self.cache_ttl = 300  # 5분

    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.cleanup()

    async def initialize(self) -> Result[bool]:
        """클라이언트 초기화"""
        try:
            # aiohttp 세션 생성
            timeout = aiohttp.ClientTimeout(total=self.default_timeout)
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=timeout,
                headers={"User-Agent": "PACA-UniversalAPIClient/1.0"}
            )

            self.logger.info("Universal API client initialized successfully")
            return Result(True, True)

        except Exception as e:
            self.logger.error(f"Failed to initialize client: {str(e)}")
            return Result(False, False, str(e))

    def register_endpoint(self, endpoint: APIEndpoint) -> None:
        """엔드포인트 등록"""
        self.endpoints[endpoint.name] = endpoint
        self.logger.debug(f"Registered endpoint: {endpoint.name}")

    def register_endpoints(self, endpoints: List[APIEndpoint]) -> None:
        """다중 엔드포인트 등록"""
        for endpoint in endpoints:
            self.register_endpoint(endpoint)

    async def set_auth(
        self,
        auth_type: str,
        credentials: Dict[str, Any]
    ) -> Result[bool]:
        """인증 설정"""
        return await self.auth_manager.set_auth(auth_type, credentials)

    def _build_url(self, endpoint: APIEndpoint, params: Optional[Dict] = None) -> str:
        """URL 구성"""
        url = endpoint.url
        if not url.startswith('http'):
            url = f"{self.base_url}/{url.lstrip('/')}"

        if params:
            # URL 파라미터 추가
            param_str = "&".join([f"{k}={v}" for k, v in params.items()])
            url += f"?{param_str}" if '?' not in url else f"&{param_str}"

        return url

    def _get_cache_key(self, request: APIRequest) -> str:
        """캐시 키 생성"""
        cache_data = {
            "endpoint": request.endpoint.name,
            "data": request.data,
            "params": request.params
        }
        return str(hash(json.dumps(cache_data, sort_keys=True)))

    def _is_cache_valid(self, cache_key: str) -> bool:
        """캐시 유효성 확인"""
        if cache_key not in self.response_cache:
            return False

        cached_time = self.response_cache[cache_key].get("timestamp", 0)
        return time.time() - cached_time < self.cache_ttl

    async def _prepare_headers(
        self,
        endpoint: APIEndpoint,
        custom_headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """헤더 준비"""
        headers = {}

        # 기본 헤더
        if endpoint.content_type != ContentType.MULTIPART:
            headers["Content-Type"] = endpoint.content_type.value

        # 엔드포인트 기본 헤더
        headers.update(endpoint.headers)

        # 인증 헤더
        if endpoint.auth_required:
            auth_headers = await self.auth_manager.get_auth_headers()
            headers.update(auth_headers)

        # 커스텀 헤더 (최우선)
        if custom_headers:
            headers.update(custom_headers)

        return headers

    async def _prepare_data(
        self,
        endpoint: APIEndpoint,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None
    ) -> Any:
        """요청 데이터 준비"""
        if not data and not files:
            return None

        if files:
            # 멀티파트 데이터
            form_data = aiohttp.FormData()
            if data:
                for key, value in data.items():
                    form_data.add_field(key, str(value))
            for key, file_data in files.items():
                form_data.add_field(key, file_data)
            return form_data

        elif endpoint.content_type == ContentType.JSON:
            return json.dumps(data) if data else None

        elif endpoint.content_type == ContentType.FORM:
            return data

        else:
            return str(data) if data else None

    async def _make_request(self, request: APIRequest) -> Result[APIResponse]:
        """실제 HTTP 요청 실행"""
        if not self.session:
            await self.initialize()

        endpoint = request.endpoint
        start_time = time.time()

        try:
            # Rate limiting 확인
            if self.rate_limiter and endpoint.rate_limit:
                await self.rate_limiter.acquire(endpoint.name, endpoint.rate_limit)

            # Circuit breaker 확인
            if self.circuit_breaker:
                if not await self.circuit_breaker.can_execute(endpoint.name):
                    raise Exception("Circuit breaker is open")

            # URL 및 헤더 준비
            url = self._build_url(endpoint, request.params)
            headers = await self._prepare_headers(endpoint, request.headers)
            data = await self._prepare_data(endpoint, request.data, request.files)

            # 타임아웃 설정
            timeout = request.timeout or endpoint.timeout

            # HTTP 요청 실행
            async with self.session.request(
                method=endpoint.method.value,
                url=url,
                headers=headers,
                data=data,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:

                response_time = time.time() - start_time
                response_text = await response.text()

                # JSON 파싱 시도
                try:
                    response_data = json.loads(response_text) if response_text else {}
                except json.JSONDecodeError:
                    response_data = response_text

                # 응답 객체 생성
                api_response = APIResponse(
                    status_code=response.status,
                    data=response_data,
                    headers=dict(response.headers),
                    response_time=response_time,
                    endpoint_name=endpoint.name,
                    success=200 <= response.status < 300
                )

                if not api_response.success:
                    api_response.error = f"HTTP {response.status}: {response_text}"

                # Circuit breaker 업데이트
                if self.circuit_breaker:
                    if api_response.success:
                        await self.circuit_breaker.record_success(endpoint.name)
                    else:
                        await self.circuit_breaker.record_failure(endpoint.name)

                return Result(api_response.success, api_response)

        except Exception as e:
            response_time = time.time() - start_time
            error_msg = str(e)

            # Circuit breaker 실패 기록
            if self.circuit_breaker:
                await self.circuit_breaker.record_failure(endpoint.name)

            api_response = APIResponse(
                status_code=0,
                data=None,
                headers={},
                response_time=response_time,
                endpoint_name=endpoint.name,
                success=False,
                error=error_msg
            )

            return Result(False, api_response, error_msg)

    async def request(
        self,
        endpoint_name: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        files: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Result[APIResponse]:
        """API 요청 실행"""

        # 엔드포인트 확인
        if endpoint_name not in self.endpoints:
            error_msg = f"Endpoint '{endpoint_name}' not registered"
            self.logger.error(error_msg)
            return Result(False, None, error_msg)

        endpoint = self.endpoints[endpoint_name]

        # 요청 객체 생성
        api_request = APIRequest(
            endpoint=endpoint,
            data=data,
            params=params,
            headers=headers,
            files=files
        )

        # 캐시 확인 (GET 요청만)
        if use_cache and endpoint.method == HTTPMethod.GET:
            cache_key = self._get_cache_key(api_request)
            if self._is_cache_valid(cache_key):
                self.stats["cache_hits"] += 1
                cached_response = self.response_cache[cache_key]["response"]
                self.logger.debug(f"Cache hit for endpoint: {endpoint_name}")
                return Result(True, cached_response)

        # 통계 업데이트
        self.stats["total_requests"] += 1

        # 재시도 로직
        last_error = None
        for attempt in range(endpoint.retry_attempts):
            result = await self._make_request(api_request)

            if result.is_success:
                self.stats["successful_requests"] += 1
                self.stats["total_response_time"] += result.data.response_time

                # 캐시 저장 (GET 요청만)
                if use_cache and endpoint.method == HTTPMethod.GET:
                    cache_key = self._get_cache_key(api_request)
                    self.response_cache[cache_key] = {
                        "response": result.data,
                        "timestamp": time.time()
                    }

                self.logger.info(f"Request successful: {endpoint_name}")
                return result

            else:
                last_error = result.error
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {last_error}")

                # 재시도 대기
                if attempt < endpoint.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # 지수 백오프

        # 모든 재시도 실패
        self.stats["failed_requests"] += 1
        self.logger.error(f"All retry attempts failed for {endpoint_name}: {last_error}")
        return Result(False, None, last_error)

    async def get(
        self,
        endpoint_name: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Result[APIResponse]:
        """GET 요청"""
        return await self.request(endpoint_name, params=params, **kwargs)

    async def post(
        self,
        endpoint_name: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Result[APIResponse]:
        """POST 요청"""
        return await self.request(endpoint_name, data=data, **kwargs)

    async def put(
        self,
        endpoint_name: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Result[APIResponse]:
        """PUT 요청"""
        return await self.request(endpoint_name, data=data, **kwargs)

    async def delete(
        self,
        endpoint_name: str,
        **kwargs
    ) -> Result[APIResponse]:
        """DELETE 요청"""
        return await self.request(endpoint_name, **kwargs)

    async def batch_request(
        self,
        requests: List[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> List[Result[APIResponse]]:
        """배치 요청 처리"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_request(req_data):
            async with semaphore:
                endpoint_name = req_data.get("endpoint")
                return await self.request(
                    endpoint_name,
                    data=req_data.get("data"),
                    params=req_data.get("params"),
                    headers=req_data.get("headers"),
                    files=req_data.get("files")
                )

        tasks = [process_request(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 예외 처리
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    Result(False, None, f"Batch request {i} failed: {str(result)}")
                )
            else:
                processed_results.append(result)

        return processed_results

    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        avg_response_time = (
            self.stats["total_response_time"] / self.stats["successful_requests"]
            if self.stats["successful_requests"] > 0 else 0
        )

        return {
            **self.stats,
            "average_response_time": avg_response_time,
            "success_rate": (
                self.stats["successful_requests"] / self.stats["total_requests"]
                if self.stats["total_requests"] > 0 else 0
            ),
            "cache_hit_rate": (
                self.stats["cache_hits"] / self.stats["total_requests"]
                if self.stats["total_requests"] > 0 else 0
            ),
            "registered_endpoints": len(self.endpoints)
        }

    async def health_check(self) -> Result[Dict[str, Any]]:
        """헬스 체크"""
        try:
            health_data = {
                "status": "healthy",
                "session_status": "active" if self.session and not self.session.closed else "inactive",
                "stats": self.get_stats(),
                "endpoints": list(self.endpoints.keys()),
                "timestamp": datetime.now().isoformat()
            }

            # 간단한 연결 테스트 (옵션)
            if self.base_url:
                try:
                    async with self.session.get(f"{self.base_url}/health", timeout=5) as response:
                        health_data["base_url_status"] = response.status
                except:
                    health_data["base_url_status"] = "unreachable"

            return Result(True, health_data)

        except Exception as e:
            return Result(False, None, f"Health check failed: {str(e)}")

    async def cleanup(self) -> None:
        """리소스 정리"""
        try:
            # 캐시 정리
            self.response_cache.clear()

            # 세션 종료
            if self.session and not self.session.closed:
                await self.session.close()

            # 커넥터 종료
            if self.connector:
                await self.connector.close()

            self.logger.info("Universal API client cleaned up")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")


# 팩토리 함수
def create_universal_client(
    base_url: str = "",
    endpoints: Optional[List[APIEndpoint]] = None,
    **kwargs
) -> UniversalAPIClient:
    """범용 API 클라이언트 생성 헬퍼"""
    client = UniversalAPIClient(base_url=base_url, **kwargs)

    if endpoints:
        client.register_endpoints(endpoints)

    return client