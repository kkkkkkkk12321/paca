"""
GraphQL API 클라이언트
GraphQL API와의 통신을 위한 전용 클라이언트
"""

import asyncio
import aiohttp
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from ...core.types import Result
from ...core.utils.logger import PacaLogger
from .universal_client import UniversalAPIClient, APIEndpoint, HTTPMethod, ContentType


@dataclass
class GraphQLQuery:
    """GraphQL 쿼리"""
    query: str
    variables: Optional[Dict[str, Any]] = None
    operation_name: Optional[str] = None


@dataclass
class GraphQLMutation:
    """GraphQL 뮤테이션"""
    mutation: str
    variables: Optional[Dict[str, Any]] = None
    operation_name: Optional[str] = None


@dataclass
class GraphQLSubscription:
    """GraphQL 구독"""
    subscription: str
    variables: Optional[Dict[str, Any]] = None
    operation_name: Optional[str] = None


@dataclass
class GraphQLResponse:
    """GraphQL 응답"""
    data: Optional[Dict[str, Any]] = None
    errors: Optional[List[Dict[str, Any]]] = None
    extensions: Optional[Dict[str, Any]] = None


class GraphQLClient(UniversalAPIClient):
    """GraphQL API 전용 클라이언트"""

    def __init__(
        self,
        endpoint_url: str,
        subscription_endpoint: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.endpoint_url = endpoint_url
        self.subscription_endpoint = subscription_endpoint or endpoint_url.replace('http', 'ws')
        self.logger = PacaLogger("GraphQLClient")

        # GraphQL 기본 엔드포인트 등록
        self.register_endpoint(APIEndpoint(
            name="graphql",
            url=endpoint_url,
            method=HTTPMethod.POST,
            content_type=ContentType.JSON,
            auth_required=True,
            timeout=30.0
        ))

        # 쿼리 캐시
        self.query_cache: Dict[str, str] = {}
        self.introspection_cache: Optional[Dict] = None

    def _build_request_payload(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        operation_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """GraphQL 요청 페이로드 구성"""
        payload = {"query": query}

        if variables:
            payload["variables"] = variables

        if operation_name:
            payload["operationName"] = operation_name

        return payload

    def _parse_response(self, response_data: Any) -> GraphQLResponse:
        """GraphQL 응답 파싱"""
        if isinstance(response_data, dict):
            return GraphQLResponse(
                data=response_data.get("data"),
                errors=response_data.get("errors"),
                extensions=response_data.get("extensions")
            )
        else:
            return GraphQLResponse(
                errors=[{"message": "Invalid response format", "data": response_data}]
            )

    async def execute_query(
        self,
        query: GraphQLQuery,
        **kwargs
    ) -> Result[GraphQLResponse]:
        """GraphQL 쿼리 실행"""
        payload = self._build_request_payload(
            query.query,
            query.variables,
            query.operation_name
        )

        result = await self.post("graphql", data=payload, **kwargs)

        if result.is_success and result.data.success:
            graphql_response = self._parse_response(result.data.data)

            # GraphQL 에러 체크
            if graphql_response.errors:
                error_messages = [error.get("message", "Unknown error") for error in graphql_response.errors]
                return Result(False, graphql_response, "; ".join(error_messages))

            return Result(True, graphql_response)

        return Result(False, None, result.error)

    async def execute_mutation(
        self,
        mutation: GraphQLMutation,
        **kwargs
    ) -> Result[GraphQLResponse]:
        """GraphQL 뮤테이션 실행"""
        payload = self._build_request_payload(
            mutation.mutation,
            mutation.variables,
            mutation.operation_name
        )

        result = await self.post("graphql", data=payload, **kwargs)

        if result.is_success and result.data.success:
            graphql_response = self._parse_response(result.data.data)

            # GraphQL 에러 체크
            if graphql_response.errors:
                error_messages = [error.get("message", "Unknown error") for error in graphql_response.errors]
                return Result(False, graphql_response, "; ".join(error_messages))

            return Result(True, graphql_response)

        return Result(False, None, result.error)

    async def execute_raw(
        self,
        query_string: str,
        variables: Optional[Dict[str, Any]] = None,
        operation_name: Optional[str] = None,
        **kwargs
    ) -> Result[GraphQLResponse]:
        """원시 GraphQL 쿼리 실행"""
        query = GraphQLQuery(
            query=query_string,
            variables=variables,
            operation_name=operation_name
        )
        return await self.execute_query(query, **kwargs)

    async def introspect(self, use_cache: bool = True) -> Result[Dict[str, Any]]:
        """GraphQL 스키마 인트로스펙션"""
        if use_cache and self.introspection_cache:
            return Result(True, self.introspection_cache)

        introspection_query = """
        query IntrospectionQuery {
          __schema {
            queryType { name }
            mutationType { name }
            subscriptionType { name }
            types {
              ...FullType
            }
            directives {
              name
              description
              locations
              args {
                ...InputValue
              }
            }
          }
        }

        fragment FullType on __Type {
          kind
          name
          description
          fields(includeDeprecated: true) {
            name
            description
            args {
              ...InputValue
            }
            type {
              ...TypeRef
            }
            isDeprecated
            deprecationReason
          }
          inputFields {
            ...InputValue
          }
          interfaces {
            ...TypeRef
          }
          enumValues(includeDeprecated: true) {
            name
            description
            isDeprecated
            deprecationReason
          }
          possibleTypes {
            ...TypeRef
          }
        }

        fragment InputValue on __InputValue {
          name
          description
          type { ...TypeRef }
          defaultValue
        }

        fragment TypeRef on __Type {
          kind
          name
          ofType {
            kind
            name
            ofType {
              kind
              name
              ofType {
                kind
                name
                ofType {
                  kind
                  name
                  ofType {
                    kind
                    name
                    ofType {
                      kind
                      name
                      ofType {
                        kind
                        name
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """

        result = await self.execute_raw(introspection_query)

        if result.is_success and result.data.data:
            self.introspection_cache = result.data.data
            return Result(True, result.data.data)

        return Result(False, None, result.error)

    async def get_schema_types(self) -> Result[List[Dict[str, Any]]]:
        """스키마 타입 목록 조회"""
        introspection_result = await self.introspect()

        if introspection_result.is_success:
            schema = introspection_result.data.get("__schema", {})
            types = schema.get("types", [])
            # 시스템 타입 제외
            user_types = [
                t for t in types
                if not t.get("name", "").startswith("__")
            ]
            return Result(True, user_types)

        return Result(False, None, introspection_result.error)

    async def get_queries(self) -> Result[List[Dict[str, Any]]]:
        """사용 가능한 쿼리 목록 조회"""
        introspection_result = await self.introspect()

        if introspection_result.is_success:
            schema = introspection_result.data.get("__schema", {})
            query_type = schema.get("queryType", {})
            query_type_name = query_type.get("name")

            if query_type_name:
                types = schema.get("types", [])
                for type_def in types:
                    if type_def.get("name") == query_type_name:
                        return Result(True, type_def.get("fields", []))

            return Result(True, [])

        return Result(False, None, introspection_result.error)

    async def get_mutations(self) -> Result[List[Dict[str, Any]]]:
        """사용 가능한 뮤테이션 목록 조회"""
        introspection_result = await self.introspect()

        if introspection_result.is_success:
            schema = introspection_result.data.get("__schema", {})
            mutation_type = schema.get("mutationType", {})
            mutation_type_name = mutation_type.get("name")

            if mutation_type_name:
                types = schema.get("types", [])
                for type_def in types:
                    if type_def.get("name") == mutation_type_name:
                        return Result(True, type_def.get("fields", []))

            return Result(True, [])

        return Result(False, None, introspection_result.error)

    async def batch_execute(
        self,
        operations: List[Union[GraphQLQuery, GraphQLMutation]],
        **kwargs
    ) -> List[Result[GraphQLResponse]]:
        """배치 GraphQL 연산 실행"""
        async def execute_operation(operation):
            if isinstance(operation, GraphQLQuery):
                return await self.execute_query(operation, **kwargs)
            elif isinstance(operation, GraphQLMutation):
                return await self.execute_mutation(operation, **kwargs)
            else:
                return Result(False, None, "Invalid operation type")

        tasks = [execute_operation(op) for op in operations]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 예외 처리
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    Result(False, None, f"Batch operation {i} failed: {str(result)}")
                )
            else:
                processed_results.append(result)

        return processed_results

    def build_query(
        self,
        operation_type: str,
        fields: List[str],
        filters: Optional[Dict[str, Any]] = None,
        pagination: Optional[Dict[str, Any]] = None
    ) -> str:
        """동적 쿼리 빌더"""
        query_parts = [f"{operation_type} {{"]

        # 필터 조건
        args = []
        if filters:
            for key, value in filters.items():
                if isinstance(value, str):
                    args.append(f'{key}: "{value}"')
                else:
                    args.append(f'{key}: {json.dumps(value)}')

        # 페이지네이션
        if pagination:
            for key, value in pagination.items():
                args.append(f'{key}: {value}')

        # 필드 목록
        fields_str = ", ".join(fields)

        if args:
            query_parts.append(f"  {operation_type}({', '.join(args)}) {{ {fields_str} }}")
        else:
            query_parts.append(f"  {operation_type} {{ {fields_str} }}")

        query_parts.append("}")

        return "\n".join(query_parts)

    def build_mutation(
        self,
        mutation_name: str,
        input_data: Dict[str, Any],
        return_fields: List[str]
    ) -> str:
        """동적 뮤테이션 빌더"""
        input_str = json.dumps(input_data).replace('"', '')
        fields_str = ", ".join(return_fields)

        return f"""
        mutation {{
          {mutation_name}(input: {input_str}) {{
            {fields_str}
          }}
        }}
        """

    async def subscribe(
        self,
        subscription: GraphQLSubscription,
        callback: callable,
        **kwargs
    ) -> Result[bool]:
        """GraphQL 구독 (WebSocket 기반)"""
        # WebSocket 구독은 복잡하므로 기본 구현만 제공
        # 실제 구현에는 aiohttp의 WebSocket 기능 사용
        try:
            # WebSocket 연결 설정 (예시)
            self.logger.info(f"Setting up subscription: {subscription.operation_name}")

            # 실제 WebSocket 구현은 추후 확장
            self.logger.warning("WebSocket subscription not fully implemented")

            return Result(True, True)

        except Exception as e:
            return Result(False, False, str(e))

    def create_query(
        self,
        query_string: str,
        variables: Optional[Dict[str, Any]] = None,
        operation_name: Optional[str] = None
    ) -> GraphQLQuery:
        """GraphQL 쿼리 객체 생성"""
        return GraphQLQuery(
            query=query_string,
            variables=variables,
            operation_name=operation_name
        )

    def create_mutation(
        self,
        mutation_string: str,
        variables: Optional[Dict[str, Any]] = None,
        operation_name: Optional[str] = None
    ) -> GraphQLMutation:
        """GraphQL 뮤테이션 객체 생성"""
        return GraphQLMutation(
            mutation=mutation_string,
            variables=variables,
            operation_name=operation_name
        )

    async def validate_query(self, query_string: str) -> Result[bool]:
        """쿼리 유효성 검증 (간단한 문법 체크)"""
        try:
            # 기본적인 GraphQL 문법 체크
            query_string = query_string.strip()

            if not query_string:
                return Result(False, False, "Empty query")

            # 중괄호 균형 체크
            if query_string.count('{') != query_string.count('}'):
                return Result(False, False, "Unbalanced braces")

            # 기본 키워드 체크
            valid_keywords = ['query', 'mutation', 'subscription', 'fragment']
            first_word = query_string.split()[0].lower()

            if first_word not in valid_keywords and not query_string.startswith('{'):
                return Result(False, False, f"Invalid operation type: {first_word}")

            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"Query validation error: {str(e)}")


# 팩토리 함수
def create_graphql_client(
    endpoint_url: str,
    subscription_endpoint: Optional[str] = None,
    **kwargs
) -> GraphQLClient:
    """GraphQL 클라이언트 생성 헬퍼"""
    return GraphQLClient(
        endpoint_url=endpoint_url,
        subscription_endpoint=subscription_endpoint,
        **kwargs
    )