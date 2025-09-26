"""
Authentication Manager
다양한 인증 방식을 통합 관리
"""

import asyncio
import base64
import hashlib
import hmac
import time
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Union
from datetime import datetime, timedelta
from enum import Enum
import os

from ...core.types import Result
from ...core.utils.logger import PacaLogger


class AuthType(Enum):
    """인증 타입"""
    NONE = "none"
    API_KEY = "api_key"
    BASIC = "basic"
    BEARER = "bearer"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    HMAC = "hmac"
    CUSTOM = "custom"


@dataclass
class AuthCredentials:
    """인증 자격 증명"""
    auth_type: AuthType
    credentials: Dict[str, Any]
    expires_at: Optional[datetime] = None
    refresh_token: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OAuthConfig:
    """OAuth 2.0 설정"""
    client_id: str
    client_secret: str
    authorization_url: str
    token_url: str
    redirect_uri: str
    scope: List[str] = field(default_factory=list)
    state: Optional[str] = None


class AuthManager:
    """통합 인증 관리자"""

    def __init__(self):
        self.logger = PacaLogger("AuthManager")

        # 인증 정보 저장
        self.credentials: Dict[str, AuthCredentials] = {}
        self.default_auth: Optional[str] = None

        # OAuth 설정
        self.oauth_configs: Dict[str, OAuthConfig] = {}

        # 토큰 캐시
        self.token_cache: Dict[str, Dict[str, Any]] = {}

    async def set_auth(
        self,
        auth_type: str,
        credentials: Dict[str, Any],
        auth_name: str = "default"
    ) -> Result[bool]:
        """인증 정보 설정"""
        try:
            auth_enum = AuthType(auth_type.lower())

            # 인증 정보 검증
            validation_result = await self._validate_credentials(auth_enum, credentials)
            if not validation_result.is_success:
                return validation_result

            # 만료 시간 계산
            expires_at = None
            if "expires_in" in credentials:
                expires_at = datetime.now() + timedelta(seconds=credentials["expires_in"])

            auth_creds = AuthCredentials(
                auth_type=auth_enum,
                credentials=credentials,
                expires_at=expires_at,
                refresh_token=credentials.get("refresh_token")
            )

            self.credentials[auth_name] = auth_creds

            if self.default_auth is None:
                self.default_auth = auth_name

            self.logger.info(f"Authentication set: {auth_name} ({auth_type})")
            return Result(True, True)

        except ValueError as e:
            return Result(False, False, f"Invalid auth type: {auth_type}")
        except Exception as e:
            return Result(False, False, f"Failed to set auth: {str(e)}")

    async def _validate_credentials(
        self,
        auth_type: AuthType,
        credentials: Dict[str, Any]
    ) -> Result[bool]:
        """인증 정보 유효성 검증"""
        try:
            if auth_type == AuthType.API_KEY:
                if "api_key" not in credentials:
                    return Result(False, False, "API key required")

            elif auth_type == AuthType.BASIC:
                if "username" not in credentials or "password" not in credentials:
                    return Result(False, False, "Username and password required")

            elif auth_type == AuthType.BEARER:
                if "token" not in credentials:
                    return Result(False, False, "Bearer token required")

            elif auth_type == AuthType.OAUTH2:
                required_fields = ["access_token"]
                for field in required_fields:
                    if field not in credentials:
                        return Result(False, False, f"OAuth2 field required: {field}")

            elif auth_type == AuthType.JWT:
                if "token" not in credentials:
                    return Result(False, False, "JWT token required")

            elif auth_type == AuthType.HMAC:
                required_fields = ["access_key", "secret_key"]
                for field in required_fields:
                    if field not in credentials:
                        return Result(False, False, f"HMAC field required: {field}")

            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"Credential validation error: {str(e)}")

    async def get_auth_headers(
        self,
        auth_name: str = "default",
        request_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """인증 헤더 생성"""
        if auth_name == "default":
            auth_name = self.default_auth

        if not auth_name or auth_name not in self.credentials:
            return {}

        auth_creds = self.credentials[auth_name]

        # 토큰 만료 확인 및 갱신
        if auth_creds.expires_at and datetime.now() >= auth_creds.expires_at:
            await self._refresh_token(auth_name)

        return await self._generate_auth_headers(auth_creds, request_data)

    async def _generate_auth_headers(
        self,
        auth_creds: AuthCredentials,
        request_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """인증 타입별 헤더 생성"""
        headers = {}

        if auth_creds.auth_type == AuthType.API_KEY:
            api_key = auth_creds.credentials["api_key"]
            key_location = auth_creds.credentials.get("location", "header")
            key_name = auth_creds.credentials.get("key_name", "X-API-Key")

            if key_location == "header":
                headers[key_name] = api_key
            # URL 파라미터나 쿼리는 여기서 처리하지 않음

        elif auth_creds.auth_type == AuthType.BASIC:
            username = auth_creds.credentials["username"]
            password = auth_creds.credentials["password"]

            credentials_str = f"{username}:{password}"
            encoded = base64.b64encode(credentials_str.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"

        elif auth_creds.auth_type == AuthType.BEARER:
            token = auth_creds.credentials["token"]
            headers["Authorization"] = f"Bearer {token}"

        elif auth_creds.auth_type == AuthType.OAUTH2:
            access_token = auth_creds.credentials["access_token"]
            headers["Authorization"] = f"Bearer {access_token}"

        elif auth_creds.auth_type == AuthType.JWT:
            token = auth_creds.credentials["token"]
            headers["Authorization"] = f"Bearer {token}"

        elif auth_creds.auth_type == AuthType.HMAC:
            hmac_headers = await self._generate_hmac_headers(auth_creds, request_data)
            headers.update(hmac_headers)

        elif auth_creds.auth_type == AuthType.CUSTOM:
            custom_headers = auth_creds.credentials.get("headers", {})
            headers.update(custom_headers)

        return headers

    async def _generate_hmac_headers(
        self,
        auth_creds: AuthCredentials,
        request_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """HMAC 서명 헤더 생성"""
        access_key = auth_creds.credentials["access_key"]
        secret_key = auth_creds.credentials["secret_key"]

        timestamp = str(int(time.time()))
        method = request_data.get("method", "GET") if request_data else "GET"
        path = request_data.get("path", "/") if request_data else "/"

        # 서명 문자열 구성
        string_to_sign = f"{method}\n{path}\n{timestamp}"

        # 요청 본문이 있으면 추가
        if request_data and "body" in request_data:
            body_hash = hashlib.sha256(
                json.dumps(request_data["body"]).encode()
            ).hexdigest()
            string_to_sign += f"\n{body_hash}"

        # HMAC 서명 생성
        signature = hmac.new(
            secret_key.encode(),
            string_to_sign.encode(),
            hashlib.sha256
        ).hexdigest()

        return {
            "Authorization": f"HMAC-SHA256 Credential={access_key}, Signature={signature}",
            "X-Timestamp": timestamp
        }

    async def _refresh_token(self, auth_name: str) -> Result[bool]:
        """토큰 갱신"""
        if auth_name not in self.credentials:
            return Result(False, False, "Auth not found")

        auth_creds = self.credentials[auth_name]

        if auth_creds.auth_type == AuthType.OAUTH2:
            return await self._refresh_oauth_token(auth_name)
        elif auth_creds.auth_type == AuthType.JWT:
            return await self._refresh_jwt_token(auth_name)
        else:
            return Result(False, False, f"Token refresh not supported for {auth_creds.auth_type.value}")

    async def _refresh_oauth_token(self, auth_name: str) -> Result[bool]:
        """OAuth 토큰 갱신"""
        try:
            auth_creds = self.credentials[auth_name]
            refresh_token = auth_creds.refresh_token

            if not refresh_token:
                return Result(False, False, "No refresh token available")

            # OAuth 설정 조회
            oauth_config = self.oauth_configs.get(auth_name)
            if not oauth_config:
                return Result(False, False, "OAuth config not found")

            # 토큰 갱신 요청 (실제 HTTP 요청은 생략)
            # 실제 구현에서는 HTTP 클라이언트를 사용하여 토큰 갱신
            self.logger.info(f"Refreshing OAuth token for: {auth_name}")

            # 갱신된 토큰으로 업데이트 (예시)
            # auth_creds.credentials["access_token"] = new_access_token
            # auth_creds.expires_at = new_expires_at

            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"OAuth token refresh failed: {str(e)}")

    async def _refresh_jwt_token(self, auth_name: str) -> Result[bool]:
        """JWT 토큰 갱신"""
        try:
            # JWT 토큰 갱신 로직 구현
            self.logger.info(f"Refreshing JWT token for: {auth_name}")

            # 실제 구현에서는 JWT 라이브러리를 사용하여 토큰 검증 및 갱신
            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"JWT token refresh failed: {str(e)}")

    def set_oauth_config(self, auth_name: str, config: OAuthConfig) -> None:
        """OAuth 설정"""
        self.oauth_configs[auth_name] = config
        self.logger.info(f"OAuth config set for: {auth_name}")

    def get_authorization_url(self, auth_name: str) -> Optional[str]:
        """OAuth 인증 URL 생성"""
        if auth_name not in self.oauth_configs:
            return None

        config = self.oauth_configs[auth_name]
        params = [
            f"client_id={config.client_id}",
            f"redirect_uri={config.redirect_uri}",
            f"response_type=code"
        ]

        if config.scope:
            params.append(f"scope={'+'.join(config.scope)}")

        if config.state:
            params.append(f"state={config.state}")

        return f"{config.authorization_url}?{'&'.join(params)}"

    async def exchange_code_for_token(
        self,
        auth_name: str,
        authorization_code: str
    ) -> Result[Dict[str, Any]]:
        """OAuth 인증 코드를 토큰으로 교환"""
        try:
            if auth_name not in self.oauth_configs:
                return Result(False, None, "OAuth config not found")

            config = self.oauth_configs[auth_name]

            # 토큰 교환 요청 (실제 HTTP 요청은 생략)
            # 실제 구현에서는 HTTP 클라이언트를 사용
            self.logger.info(f"Exchanging code for token: {auth_name}")

            # 예시 응답
            token_response = {
                "access_token": "example_access_token",
                "token_type": "Bearer",
                "expires_in": 3600,
                "refresh_token": "example_refresh_token"
            }

            # 인증 정보 자동 설정
            await self.set_auth("oauth2", token_response, auth_name)

            return Result(True, token_response)

        except Exception as e:
            return Result(False, None, f"Token exchange failed: {str(e)}")

    def is_authenticated(self, auth_name: str = "default") -> bool:
        """인증 상태 확인"""
        if auth_name == "default":
            auth_name = self.default_auth

        if not auth_name or auth_name not in self.credentials:
            return False

        auth_creds = self.credentials[auth_name]

        # 만료 시간 확인
        if auth_creds.expires_at and datetime.now() >= auth_creds.expires_at:
            return False

        return True

    def get_auth_info(self, auth_name: str = "default") -> Optional[Dict[str, Any]]:
        """인증 정보 조회"""
        if auth_name == "default":
            auth_name = self.default_auth

        if not auth_name or auth_name not in self.credentials:
            return None

        auth_creds = self.credentials[auth_name]

        return {
            "auth_name": auth_name,
            "auth_type": auth_creds.auth_type.value,
            "expires_at": auth_creds.expires_at.isoformat() if auth_creds.expires_at else None,
            "has_refresh_token": auth_creds.refresh_token is not None,
            "is_expired": auth_creds.expires_at and datetime.now() >= auth_creds.expires_at,
            "metadata": auth_creds.metadata
        }

    def list_auth_methods(self) -> Dict[str, Dict[str, Any]]:
        """모든 인증 방법 목록"""
        return {
            name: self.get_auth_info(name)
            for name in self.credentials.keys()
        }

    async def remove_auth(self, auth_name: str) -> Result[bool]:
        """인증 정보 제거"""
        try:
            if auth_name not in self.credentials:
                return Result(False, False, "Auth not found")

            del self.credentials[auth_name]

            if self.default_auth == auth_name:
                self.default_auth = next(iter(self.credentials.keys()), None)

            if auth_name in self.oauth_configs:
                del self.oauth_configs[auth_name]

            if auth_name in self.token_cache:
                del self.token_cache[auth_name]

            self.logger.info(f"Authentication removed: {auth_name}")
            return Result(True, True)

        except Exception as e:
            return Result(False, False, f"Failed to remove auth: {str(e)}")

    def clear_all_auth(self) -> None:
        """모든 인증 정보 제거"""
        self.credentials.clear()
        self.oauth_configs.clear()
        self.token_cache.clear()
        self.default_auth = None
        self.logger.info("All authentication cleared")


# 팩토리 함수들
def create_api_key_auth(
    api_key: str,
    key_name: str = "X-API-Key",
    location: str = "header"
) -> Dict[str, Any]:
    """API 키 인증 생성"""
    return {
        "api_key": api_key,
        "key_name": key_name,
        "location": location
    }


def create_basic_auth(username: str, password: str) -> Dict[str, Any]:
    """Basic 인증 생성"""
    return {
        "username": username,
        "password": password
    }


def create_bearer_auth(token: str) -> Dict[str, Any]:
    """Bearer 토큰 인증 생성"""
    return {
        "token": token
    }


def create_oauth_config(
    client_id: str,
    client_secret: str,
    authorization_url: str,
    token_url: str,
    redirect_uri: str,
    scope: Optional[List[str]] = None
) -> OAuthConfig:
    """OAuth 설정 생성"""
    return OAuthConfig(
        client_id=client_id,
        client_secret=client_secret,
        authorization_url=authorization_url,
        token_url=token_url,
        redirect_uri=redirect_uri,
        scope=scope or []
    )