"""
REST API 클라이언트
RESTful API와의 통신을 위한 전용 클라이언트
"""

import asyncio
import aiohttp
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

from ...core.types import Result
from ...core.utils.logger import PacaLogger
from .universal_client import UniversalAPIClient, APIEndpoint, HTTPMethod, ContentType


class RESTMethodPattern(Enum):
    """REST 메서드 패턴"""
    CRUD = "crud"  # Create, Read, Update, Delete
    COLLECTION = "collection"  # List, Create
    RESOURCE = "resource"  # Get, Update, Delete


@dataclass
class RESTResource:
    """REST 리소스 정의"""
    name: str
    base_path: str
    id_field: str = "id"
    methods: List[HTTPMethod] = field(default_factory=lambda: [
        HTTPMethod.GET, HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.DELETE
    ])
    nested_resources: List[str] = field(default_factory=list)
    custom_endpoints: Dict[str, Dict] = field(default_factory=dict)


class RESTClient(UniversalAPIClient):
    """REST API 전용 클라이언트"""

    def __init__(self, base_url: str, **kwargs):
        super().__init__(base_url, **kwargs)
        self.resources: Dict[str, RESTResource] = {}
        self.logger = PacaLogger("RESTClient")

    def register_resource(self, resource: RESTResource) -> None:
        """REST 리소스 등록"""
        self.resources[resource.name] = resource

        # 기본 CRUD 엔드포인트 자동 생성
        base_path = resource.base_path.rstrip('/')

        # Collection endpoints (리스트, 생성)
        if HTTPMethod.GET in resource.methods:
            list_endpoint = APIEndpoint(
                name=f"{resource.name}_list",
                url=f"{base_path}",
                method=HTTPMethod.GET,
                auth_required=True
            )
            self.register_endpoint(list_endpoint)

        if HTTPMethod.POST in resource.methods:
            create_endpoint = APIEndpoint(
                name=f"{resource.name}_create",
                url=f"{base_path}",
                method=HTTPMethod.POST,
                auth_required=True,
                content_type=ContentType.JSON
            )
            self.register_endpoint(create_endpoint)

        # Resource endpoints (조회, 수정, 삭제)
        if HTTPMethod.GET in resource.methods:
            get_endpoint = APIEndpoint(
                name=f"{resource.name}_get",
                url=f"{base_path}/{{{resource.id_field}}}",
                method=HTTPMethod.GET,
                auth_required=True
            )
            self.register_endpoint(get_endpoint)

        if HTTPMethod.PUT in resource.methods:
            update_endpoint = APIEndpoint(
                name=f"{resource.name}_update",
                url=f"{base_path}/{{{resource.id_field}}}",
                method=HTTPMethod.PUT,
                auth_required=True,
                content_type=ContentType.JSON
            )
            self.register_endpoint(update_endpoint)

        if HTTPMethod.DELETE in resource.methods:
            delete_endpoint = APIEndpoint(
                name=f"{resource.name}_delete",
                url=f"{base_path}/{{{resource.id_field}}}",
                method=HTTPMethod.DELETE,
                auth_required=True
            )
            self.register_endpoint(delete_endpoint)

        # 커스텀 엔드포인트 등록
        for endpoint_name, config in resource.custom_endpoints.items():
            custom_endpoint = APIEndpoint(
                name=f"{resource.name}_{endpoint_name}",
                url=f"{base_path}/{config.get('path', endpoint_name)}",
                method=HTTPMethod(config.get('method', 'GET')),
                auth_required=config.get('auth_required', True),
                content_type=ContentType(config.get('content_type', ContentType.JSON.value))
            )
            self.register_endpoint(custom_endpoint)

        self.logger.info(f"Registered REST resource: {resource.name}")

    def _build_resource_url(self, resource_name: str, resource_id: Optional[str] = None, nested_path: str = "") -> str:
        """리소스 URL 구성"""
        if resource_name not in self.resources:
            raise ValueError(f"Resource '{resource_name}' not registered")

        resource = self.resources[resource_name]
        url = resource.base_path.rstrip('/')

        if resource_id:
            url += f"/{resource_id}"

        if nested_path:
            url += f"/{nested_path.lstrip('/')}"

        return url

    async def list_resources(
        self,
        resource_name: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Result[List[Dict]]:
        """리소스 목록 조회"""
        endpoint_name = f"{resource_name}_list"
        result = await self.get(endpoint_name, params=params, **kwargs)

        if result.is_success and result.data.success:
            # 응답 데이터가 리스트인지 확인
            data = result.data.data
            if isinstance(data, dict) and 'items' in data:
                return Result(True, data['items'])
            elif isinstance(data, list):
                return Result(True, data)
            else:
                return Result(True, [data])

        return Result(False, None, result.error)

    async def get_resource(
        self,
        resource_name: str,
        resource_id: str,
        **kwargs
    ) -> Result[Dict]:
        """단일 리소스 조회"""
        endpoint_name = f"{resource_name}_get"

        # URL에서 {id} 플레이스홀더 교체
        if resource_name in self.resources:
            resource = self.resources[resource_name]
            endpoint = self.endpoints[endpoint_name]
            endpoint.url = endpoint.url.replace(f"{{{resource.id_field}}}", resource_id)

        result = await self.get(endpoint_name, **kwargs)

        if result.is_success and result.data.success:
            return Result(True, result.data.data)

        return Result(False, None, result.error)

    async def create_resource(
        self,
        resource_name: str,
        data: Dict[str, Any],
        **kwargs
    ) -> Result[Dict]:
        """리소스 생성"""
        endpoint_name = f"{resource_name}_create"
        result = await self.post(endpoint_name, data=data, **kwargs)

        if result.is_success and result.data.success:
            return Result(True, result.data.data)

        return Result(False, None, result.error)

    async def update_resource(
        self,
        resource_name: str,
        resource_id: str,
        data: Dict[str, Any],
        **kwargs
    ) -> Result[Dict]:
        """리소스 수정"""
        endpoint_name = f"{resource_name}_update"

        # URL에서 {id} 플레이스홀더 교체
        if resource_name in self.resources:
            resource = self.resources[resource_name]
            endpoint = self.endpoints[endpoint_name]
            endpoint.url = endpoint.url.replace(f"{{{resource.id_field}}}", resource_id)

        result = await self.put(endpoint_name, data=data, **kwargs)

        if result.is_success and result.data.success:
            return Result(True, result.data.data)

        return Result(False, None, result.error)

    async def delete_resource(
        self,
        resource_name: str,
        resource_id: str,
        **kwargs
    ) -> Result[bool]:
        """리소스 삭제"""
        endpoint_name = f"{resource_name}_delete"

        # URL에서 {id} 플레이스홀더 교체
        if resource_name in self.resources:
            resource = self.resources[resource_name]
            endpoint = self.endpoints[endpoint_name]
            endpoint.url = endpoint.url.replace(f"{{{resource.id_field}}}", resource_id)

        result = await self.delete(endpoint_name, **kwargs)

        if result.is_success and result.data.success:
            return Result(True, True)

        return Result(False, False, result.error)

    async def patch_resource(
        self,
        resource_name: str,
        resource_id: str,
        data: Dict[str, Any],
        **kwargs
    ) -> Result[Dict]:
        """리소스 부분 수정 (PATCH)"""
        # PATCH 엔드포인트가 없으면 동적 생성
        endpoint_name = f"{resource_name}_patch"
        if endpoint_name not in self.endpoints:
            if resource_name in self.resources:
                resource = self.resources[resource_name]
                patch_endpoint = APIEndpoint(
                    name=endpoint_name,
                    url=f"{resource.base_path}/{{{resource.id_field}}}",
                    method=HTTPMethod.PATCH,
                    auth_required=True,
                    content_type=ContentType.JSON
                )
                self.register_endpoint(patch_endpoint)

        # URL에서 {id} 플레이스홀더 교체
        if resource_name in self.resources:
            resource = self.resources[resource_name]
            endpoint = self.endpoints[endpoint_name]
            endpoint.url = endpoint.url.replace(f"{{{resource.id_field}}}", resource_id)

        result = await self.request(endpoint_name, data=data, **kwargs)

        if result.is_success and result.data.success:
            return Result(True, result.data.data)

        return Result(False, None, result.error)

    async def search_resources(
        self,
        resource_name: str,
        query: str,
        fields: Optional[List[str]] = None,
        **kwargs
    ) -> Result[List[Dict]]:
        """리소스 검색"""
        params = {"q": query}
        if fields:
            params["fields"] = ",".join(fields)

        return await self.list_resources(resource_name, params=params, **kwargs)

    async def get_nested_resource(
        self,
        parent_resource: str,
        parent_id: str,
        nested_resource: str,
        nested_id: Optional[str] = None,
        **kwargs
    ) -> Result[Union[List[Dict], Dict]]:
        """중첩 리소스 조회"""
        if parent_resource not in self.resources:
            return Result(False, None, f"Parent resource '{parent_resource}' not registered")

        nested_path = nested_resource
        if nested_id:
            nested_path += f"/{nested_id}"

        # 동적 엔드포인트 생성
        endpoint_name = f"{parent_resource}_{nested_resource}"
        if nested_id:
            endpoint_name += "_get"
        else:
            endpoint_name += "_list"

        parent_res = self.resources[parent_resource]
        nested_endpoint = APIEndpoint(
            name=endpoint_name,
            url=f"{parent_res.base_path}/{parent_id}/{nested_resource}" + (f"/{nested_id}" if nested_id else ""),
            method=HTTPMethod.GET,
            auth_required=True
        )
        self.register_endpoint(nested_endpoint)

        result = await self.get(endpoint_name, **kwargs)

        if result.is_success and result.data.success:
            return Result(True, result.data.data)

        return Result(False, None, result.error)

    async def bulk_create(
        self,
        resource_name: str,
        items: List[Dict[str, Any]],
        **kwargs
    ) -> Result[List[Dict]]:
        """벌크 생성"""
        # 벌크 엔드포인트 확인 또는 생성
        bulk_endpoint_name = f"{resource_name}_bulk_create"
        if bulk_endpoint_name not in self.endpoints:
            if resource_name in self.resources:
                resource = self.resources[resource_name]
                bulk_endpoint = APIEndpoint(
                    name=bulk_endpoint_name,
                    url=f"{resource.base_path}/bulk",
                    method=HTTPMethod.POST,
                    auth_required=True,
                    content_type=ContentType.JSON
                )
                self.register_endpoint(bulk_endpoint)

        result = await self.post(bulk_endpoint_name, data={"items": items}, **kwargs)

        if result.is_success and result.data.success:
            return Result(True, result.data.data)

        return Result(False, None, result.error)

    def get_resource_info(self, resource_name: str) -> Optional[Dict[str, Any]]:
        """리소스 정보 조회"""
        if resource_name not in self.resources:
            return None

        resource = self.resources[resource_name]
        return {
            "name": resource.name,
            "base_path": resource.base_path,
            "id_field": resource.id_field,
            "methods": [method.value for method in resource.methods],
            "nested_resources": resource.nested_resources,
            "custom_endpoints": list(resource.custom_endpoints.keys()),
            "registered_endpoints": [
                name for name in self.endpoints.keys()
                if name.startswith(f"{resource_name}_")
            ]
        }

    def list_resources_info(self) -> Dict[str, Dict[str, Any]]:
        """모든 리소스 정보 목록"""
        return {
            name: self.get_resource_info(name)
            for name in self.resources.keys()
        }


# 팩토리 함수
def create_rest_client(
    base_url: str,
    resources: Optional[List[RESTResource]] = None,
    **kwargs
) -> RESTClient:
    """REST 클라이언트 생성 헬퍼"""
    client = RESTClient(base_url, **kwargs)

    if resources:
        for resource in resources:
            client.register_resource(resource)

    return client


# 공통 REST 리소스 팩토리
def create_standard_resource(
    name: str,
    base_path: Optional[str] = None,
    id_field: str = "id"
) -> RESTResource:
    """표준 REST 리소스 생성"""
    return RESTResource(
        name=name,
        base_path=base_path or f"/{name}",
        id_field=id_field
    )