"""
API Registry
API 엔드포인트 및 서비스 등록 관리
"""

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import aiohttp

from ...core.types import Result
from ...core.utils.logger import PacaLogger
from .universal_client import APIEndpoint, HTTPMethod, ContentType


class ServiceStatus(Enum):
    """서비스 상태"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"


@dataclass
class ServiceInfo:
    """서비스 정보"""
    name: str
    base_url: str
    version: str = "1.0.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    health_check_url: Optional[str] = None
    health_check_interval: int = 300  # 5분
    timeout: float = 30.0
    retry_attempts: int = 3
    contact: Dict[str, str] = field(default_factory=dict)
    documentation_url: Optional[str] = None
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EndpointInfo:
    """엔드포인트 정보"""
    endpoint: APIEndpoint
    service_name: str
    category: str = "general"
    deprecated: bool = False
    deprecation_date: Optional[datetime] = None
    replacement_endpoint: Optional[str] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)
    response_schema: Optional[Dict[str, Any]] = None
    error_codes: Dict[str, str] = field(default_factory=dict)


class APIRegistry:
    """API 레지스트리"""

    def __init__(self):
        self.logger = PacaLogger("APIRegistry")

        # 서비스 및 엔드포인트 저장소
        self.services: Dict[str, ServiceInfo] = {}
        self.endpoints: Dict[str, EndpointInfo] = {}

        # 카테고리별 인덱스
        self.categories: Dict[str, List[str]] = {}

        # 태그별 인덱스
        self.tags: Dict[str, List[str]] = {}

        # 헬스체크 관리
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        self.health_history: Dict[str, List[Dict[str, Any]]] = {}

        # 통계
        self.stats = {
            "total_services": 0,
            "healthy_services": 0,
            "total_endpoints": 0,
            "deprecated_endpoints": 0,
            "health_checks_performed": 0,
            "last_registry_update": None
        }

    def register_service(self, service_info: ServiceInfo) -> Result[bool]:
        """서비스 등록"""
        try:
            self.services[service_info.name] = service_info

            # 태그 인덱스 업데이트
            for tag in service_info.tags:
                if tag not in self.tags:
                    self.tags[tag] = []
                if service_info.name not in self.tags[tag]:
                    self.tags[tag].append(service_info.name)

            # 헬스체크 시작
            if service_info.health_check_url:
                self._start_health_check(service_info.name)

            self.stats["total_services"] = len(self.services)
            self.stats["last_registry_update"] = datetime.now()

            self.logger.info(f"Service registered: {service_info.name}")
            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"Failed to register service: {str(e)}")

    def register_endpoint(self, endpoint_info: EndpointInfo) -> Result[bool]:
        """엔드포인트 등록"""
        try:
            endpoint_name = endpoint_info.endpoint.name
            self.endpoints[endpoint_name] = endpoint_info

            # 카테고리 인덱스 업데이트
            category = endpoint_info.category
            if category not in self.categories:
                self.categories[category] = []
            if endpoint_name not in self.categories[category]:
                self.categories[category].append(endpoint_name)

            # 서비스 존재 확인
            if endpoint_info.service_name not in self.services:
                self.logger.warning(f"Service not found for endpoint: {endpoint_name}")

            self.stats["total_endpoints"] = len(self.endpoints)
            if endpoint_info.deprecated:
                self.stats["deprecated_endpoints"] = sum(
                    1 for ep in self.endpoints.values() if ep.deprecated
                )

            self.stats["last_registry_update"] = datetime.now()

            self.logger.info(f"Endpoint registered: {endpoint_name}")
            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"Failed to register endpoint: {str(e)}")

    def register_endpoints_for_service(
        self,
        service_name: str,
        endpoints: List[APIEndpoint],
        category: str = "general"
    ) -> Result[List[str]]:
        """서비스의 여러 엔드포인트 일괄 등록"""
        registered_endpoints = []

        for endpoint in endpoints:
            endpoint_info = EndpointInfo(
                endpoint=endpoint,
                service_name=service_name,
                category=category
            )

            result = self.register_endpoint(endpoint_info)
            if result.is_success:
                registered_endpoints.append(endpoint.name)
            else:
                self.logger.error(f"Failed to register endpoint {endpoint.name}: {result.error}")

        return Result(True, registered_endpoints)

    def get_service(self, service_name: str) -> Optional[ServiceInfo]:
        """서비스 정보 조회"""
        return self.services.get(service_name)

    def get_endpoint(self, endpoint_name: str) -> Optional[EndpointInfo]:
        """엔드포인트 정보 조회"""
        return self.endpoints.get(endpoint_name)

    def list_services(
        self,
        tag: Optional[str] = None,
        status: Optional[ServiceStatus] = None
    ) -> List[ServiceInfo]:
        """서비스 목록 조회"""
        services = list(self.services.values())

        # 태그 필터링
        if tag:
            service_names = self.tags.get(tag, [])
            services = [s for s in services if s.name in service_names]

        # 상태 필터링
        if status:
            services = [s for s in services if s.status == status]

        return services

    def list_endpoints(
        self,
        service_name: Optional[str] = None,
        category: Optional[str] = None,
        include_deprecated: bool = True
    ) -> List[EndpointInfo]:
        """엔드포인트 목록 조회"""
        endpoints = list(self.endpoints.values())

        # 서비스 필터링
        if service_name:
            endpoints = [ep for ep in endpoints if ep.service_name == service_name]

        # 카테고리 필터링
        if category:
            endpoint_names = self.categories.get(category, [])
            endpoints = [ep for ep in endpoints if ep.endpoint.name in endpoint_names]

        # 폐기예정 필터링
        if not include_deprecated:
            endpoints = [ep for ep in endpoints if not ep.deprecated]

        return endpoints

    def search_endpoints(
        self,
        query: str,
        search_fields: List[str] = None
    ) -> List[EndpointInfo]:
        """엔드포인트 검색"""
        if search_fields is None:
            search_fields = ["name", "url", "description"]

        query_lower = query.lower()
        results = []

        for endpoint_info in self.endpoints.values():
            endpoint = endpoint_info.endpoint

            # 검색 필드에서 쿼리 매칭
            for field in search_fields:
                if field == "name" and query_lower in endpoint.name.lower():
                    results.append(endpoint_info)
                    break
                elif field == "url" and query_lower in endpoint.url.lower():
                    results.append(endpoint_info)
                    break
                elif field == "description" and hasattr(endpoint, 'description'):
                    if query_lower in getattr(endpoint, 'description', '').lower():
                        results.append(endpoint_info)
                        break

        return results

    def get_endpoints_by_category(self, category: str) -> List[EndpointInfo]:
        """카테고리별 엔드포인트 조회"""
        endpoint_names = self.categories.get(category, [])
        return [self.endpoints[name] for name in endpoint_names if name in self.endpoints]

    def get_services_by_tag(self, tag: str) -> List[ServiceInfo]:
        """태그별 서비스 조회"""
        service_names = self.tags.get(tag, [])
        return [self.services[name] for name in service_names if name in self.services]

    def deprecate_endpoint(
        self,
        endpoint_name: str,
        replacement_endpoint: Optional[str] = None,
        deprecation_date: Optional[datetime] = None
    ) -> Result[bool]:
        """엔드포인트 폐기 예정 설정"""
        try:
            if endpoint_name not in self.endpoints:
                return Result(False, False, "Endpoint not found")

            endpoint_info = self.endpoints[endpoint_name]
            endpoint_info.deprecated = True
            endpoint_info.deprecation_date = deprecation_date or datetime.now()
            endpoint_info.replacement_endpoint = replacement_endpoint

            self.stats["deprecated_endpoints"] = sum(
                1 for ep in self.endpoints.values() if ep.deprecated
            )

            self.logger.info(f"Endpoint deprecated: {endpoint_name}")
            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"Failed to deprecate endpoint: {str(e)}")

    def _start_health_check(self, service_name: str) -> None:
        """헬스체크 시작"""
        if service_name in self.health_check_tasks:
            return

        async def health_check_loop():
            service = self.services[service_name]
            while service_name in self.services:
                try:
                    result = await self._perform_health_check(service)
                    self._record_health_check(service_name, result)

                    await asyncio.sleep(service.health_check_interval)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Health check error for {service_name}: {str(e)}")
                    await asyncio.sleep(60)  # 1분 후 재시도

        task = asyncio.create_task(health_check_loop())
        self.health_check_tasks[service_name] = task

    async def _perform_health_check(self, service: ServiceInfo) -> Dict[str, Any]:
        """헬스체크 수행"""
        start_time = time.time()

        try:
            timeout = aiohttp.ClientTimeout(total=service.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(service.health_check_url) as response:
                    response_time = time.time() - start_time

                    if 200 <= response.status < 300:
                        service.status = ServiceStatus.HEALTHY
                        status = "healthy"
                    elif 500 <= response.status < 600:
                        service.status = ServiceStatus.UNHEALTHY
                        status = "unhealthy"
                    else:
                        service.status = ServiceStatus.DEGRADED
                        status = "degraded"

                    service.last_health_check = datetime.now()

                    return {
                        "status": status,
                        "response_time": response_time,
                        "status_code": response.status,
                        "timestamp": datetime.now().isoformat()
                    }

        except asyncio.TimeoutError:
            service.status = ServiceStatus.UNHEALTHY
            return {
                "status": "unhealthy",
                "error": "timeout",
                "response_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            service.status = ServiceStatus.UNHEALTHY
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }

    def _record_health_check(self, service_name: str, result: Dict[str, Any]) -> None:
        """헬스체크 결과 기록"""
        if service_name not in self.health_history:
            self.health_history[service_name] = []

        self.health_history[service_name].append(result)

        # 최근 100개 기록만 유지
        if len(self.health_history[service_name]) > 100:
            self.health_history[service_name] = self.health_history[service_name][-100:]

        self.stats["health_checks_performed"] += 1
        self.stats["healthy_services"] = sum(
            1 for s in self.services.values() if s.status == ServiceStatus.HEALTHY
        )

    def get_health_status(self, service_name: str) -> Optional[Dict[str, Any]]:
        """서비스 헬스 상태 조회"""
        service = self.get_service(service_name)
        if not service:
            return None

        history = self.health_history.get(service_name, [])
        recent_checks = history[-10:] if history else []

        # 가용성 계산 (최근 10회 체크 기준)
        if recent_checks:
            healthy_count = sum(1 for check in recent_checks if check["status"] == "healthy")
            availability = healthy_count / len(recent_checks)
        else:
            availability = 0.0

        return {
            "service_name": service_name,
            "current_status": service.status.value,
            "last_check": service.last_health_check.isoformat() if service.last_health_check else None,
            "availability": availability,
            "total_checks": len(history),
            "recent_checks": recent_checks,
            "health_check_url": service.health_check_url
        }

    def get_registry_stats(self) -> Dict[str, Any]:
        """레지스트리 통계"""
        # 카테고리별 엔드포인트 수
        category_counts = {
            category: len(endpoints)
            for category, endpoints in self.categories.items()
        }

        # 서비스별 상태 분포
        status_distribution = {}
        for status in ServiceStatus:
            count = sum(1 for s in self.services.values() if s.status == status)
            status_distribution[status.value] = count

        return {
            **self.stats,
            "category_distribution": category_counts,
            "status_distribution": status_distribution,
            "available_tags": list(self.tags.keys()),
            "available_categories": list(self.categories.keys())
        }

    def export_registry(self) -> Dict[str, Any]:
        """레지스트리 전체 정보 내보내기"""
        return {
            "services": {
                name: asdict(service)
                for name, service in self.services.items()
            },
            "endpoints": {
                name: {
                    "endpoint": asdict(endpoint_info.endpoint),
                    "service_name": endpoint_info.service_name,
                    "category": endpoint_info.category,
                    "deprecated": endpoint_info.deprecated,
                    "deprecation_date": endpoint_info.deprecation_date.isoformat() if endpoint_info.deprecation_date else None,
                    "replacement_endpoint": endpoint_info.replacement_endpoint,
                    "examples": endpoint_info.examples,
                    "response_schema": endpoint_info.response_schema,
                    "error_codes": endpoint_info.error_codes
                }
                for name, endpoint_info in self.endpoints.items()
            },
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "stats": self.get_registry_stats()
            }
        }

    def import_registry(self, registry_data: Dict[str, Any]) -> Result[bool]:
        """레지스트리 정보 가져오기"""
        try:
            # 서비스 가져오기
            for name, service_data in registry_data.get("services", {}).items():
                service_info = ServiceInfo(**service_data)
                self.register_service(service_info)

            # 엔드포인트 가져오기
            for name, endpoint_data in registry_data.get("endpoints", {}).items():
                endpoint = APIEndpoint(**endpoint_data["endpoint"])
                endpoint_info = EndpointInfo(
                    endpoint=endpoint,
                    service_name=endpoint_data["service_name"],
                    category=endpoint_data.get("category", "general"),
                    deprecated=endpoint_data.get("deprecated", False),
                    replacement_endpoint=endpoint_data.get("replacement_endpoint"),
                    examples=endpoint_data.get("examples", []),
                    response_schema=endpoint_data.get("response_schema"),
                    error_codes=endpoint_data.get("error_codes", {})
                )

                if endpoint_data.get("deprecation_date"):
                    endpoint_info.deprecation_date = datetime.fromisoformat(
                        endpoint_data["deprecation_date"]
                    )

                self.register_endpoint(endpoint_info)

            self.logger.info("Registry imported successfully")
            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"Failed to import registry: {str(e)}")

    async def cleanup(self) -> None:
        """리소스 정리"""
        # 헬스체크 태스크 정리
        for task in self.health_check_tasks.values():
            task.cancel()

        await asyncio.gather(*self.health_check_tasks.values(), return_exceptions=True)
        self.health_check_tasks.clear()

        self.logger.info("API Registry cleaned up")


# 팩토리 함수들
def create_service_info(
    name: str,
    base_url: str,
    version: str = "1.0.0",
    **kwargs
) -> ServiceInfo:
    """서비스 정보 생성 헬퍼"""
    return ServiceInfo(
        name=name,
        base_url=base_url,
        version=version,
        **kwargs
    )


def create_endpoint_info(
    endpoint: APIEndpoint,
    service_name: str,
    category: str = "general",
    **kwargs
) -> EndpointInfo:
    """엔드포인트 정보 생성 헬퍼"""
    return EndpointInfo(
        endpoint=endpoint,
        service_name=service_name,
        category=category,
        **kwargs
    )