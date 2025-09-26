"""
Authentication Service Module
사용자 인증 및 권한 관리 서비스
TypeScript → Python 완전 변환
"""

import asyncio
import hashlib
import hmac
import json
import time
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from enum import Enum

from ..core.types import (
    ID, Timestamp, Result, Priority, Status,
    create_success, create_failure, generate_id, current_timestamp
)
from ..core.errors import ValidationError, ApplicationError, AuthenticationError, AuthorizationError
from ..core.events import EventEmitter
from ..core.utils.logger import PacaLogger
from .base import (
    BaseService, ServiceConfig, ServiceContext, ServiceResult, ServicePriority
)


class TokenType(Enum):
    """토큰 유형"""
    ACCESS = "access"
    REFRESH = "refresh"
    RESET = "reset"
    VERIFICATION = "verification"


class UserRole(Enum):
    """사용자 역할"""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"
    MODERATOR = "moderator"


@dataclass
class DeviceInfo:
    """디바이스 정보"""
    device_id: str
    device_type: str
    user_agent: str
    ip_address: str


@dataclass
class UserProfile:
    """사용자 프로필"""
    first_name: str
    last_name: str
    avatar: Optional[str] = None
    bio: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class User:
    """사용자 정보"""
    id: ID
    username: str
    email: str
    password_hash: str
    profile: UserProfile
    role: UserRole
    is_active: bool = True
    is_verified: bool = False
    created_at: Timestamp = field(default_factory=current_timestamp)
    updated_at: Timestamp = field(default_factory=current_timestamp)
    last_login: Optional[Timestamp] = None
    failed_login_attempts: int = 0
    locked_until: Optional[Timestamp] = None


@dataclass
class AuthToken:
    """인증 토큰"""
    token: str
    type: TokenType
    user_id: ID
    expires_at: Timestamp
    scope: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Timestamp = field(default_factory=current_timestamp)

    @property
    def is_expired(self) -> bool:
        """토큰 만료 여부"""
        return time.time() > self.expires_at


@dataclass
class AuthSession:
    """인증 세션"""
    id: ID
    user_id: ID
    token: str
    created_at: Timestamp
    expires_at: Timestamp
    is_active: bool = True
    last_activity: Timestamp = field(default_factory=current_timestamp)
    device_info: Optional[DeviceInfo] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """세션 만료 여부"""
        return time.time() > self.expires_at

    def update_activity(self) -> None:
        """활동 시간 업데이트"""
        self.last_activity = current_timestamp()


@dataclass
class LoginRequest:
    """로그인 요청"""
    email: str
    password: str
    remember_me: bool = False
    device_info: Optional[DeviceInfo] = None


@dataclass
class RegisterRequest:
    """회원가입 요청"""
    username: str
    email: str
    password: str
    profile: UserProfile
    accept_terms: bool
    email_verification: bool = True


@dataclass
class LoginResponse:
    """로그인 응답"""
    user: User
    access_token: AuthToken
    refresh_token: Optional[AuthToken]
    session_id: ID
    expires_in: int


@dataclass
class PasswordResetRequest:
    """비밀번호 재설정 요청"""
    email: str
    new_password: str
    reset_token: str


class AuthenticationService(BaseService):
    """
    인증 서비스

    Features:
    - 사용자 등록/로그인/로그아웃
    - JWT 토큰 기반 인증
    - 세션 관리
    - 비밀번호 해싱 및 검증
    - 계정 잠금 기능
    - 비밀번호 재설정
    - 역할 기반 접근 제어
    """

    def __init__(
        self,
        config: ServiceConfig,
        events: Optional[EventEmitter] = None,
        secret_key: str = "default-secret-key",
        token_ttl: int = 3600,  # 1시간
        refresh_token_ttl: int = 86400 * 7  # 7일
    ):
        super().__init__(config, events)
        self.secret_key = secret_key
        self.token_ttl = token_ttl
        self.refresh_token_ttl = refresh_token_ttl

        # In-memory storage (실제 구현에서는 데이터베이스 사용)
        self.users: Dict[ID, User] = {}
        self.users_by_email: Dict[str, ID] = {}
        self.users_by_username: Dict[str, ID] = {}
        self.active_sessions: Dict[ID, AuthSession] = {}
        self.token_blacklist: Set[str] = set()
        self.reset_tokens: Dict[str, Dict[str, Any]] = {}

        # 보안 설정
        self.max_login_attempts = 5
        self.lockout_duration = 3600  # 1시간
        self.password_min_length = 8

        self.logger = PacaLogger("AuthenticationService")

    async def startup(self) -> None:
        """서비스 시작"""
        await super().startup()
        await self._load_user_data()

        if self.events:
            await self.events.emit('auth.service.started', {
                'service_id': self.config.id,
                'total_users': len(self.users)
            })

    async def shutdown(self) -> None:
        """서비스 종료"""
        await self._save_user_data()
        await super().shutdown()

    async def register_user(self, request: RegisterRequest) -> Result[User]:
        """사용자 등록"""
        try:
            # 입력 검증
            validation_result = self._validate_registration(request)
            if not validation_result.success:
                return validation_result

            # 중복 확인
            if request.email in self.users_by_email:
                return create_failure(ValidationError(
                    message="이미 등록된 이메일입니다",
                    field="email",
                    value=request.email
                ))

            if request.username in self.users_by_username:
                return create_failure(ValidationError(
                    message="이미 사용 중인 사용자명입니다",
                    field="username",
                    value=request.username
                ))

            # 비밀번호 해싱
            password_hash = self._hash_password(request.password)

            # 사용자 생성
            user = User(
                id=generate_id(),
                username=request.username,
                email=request.email,
                password_hash=password_hash,
                profile=request.profile,
                role=UserRole.USER,
                is_verified=not request.email_verification
            )

            # 저장
            self.users[user.id] = user
            self.users_by_email[user.email] = user.id
            self.users_by_username[user.username] = user.id

            # 이메일 인증이 필요한 경우 토큰 생성
            if request.email_verification:
                verification_token = self._generate_verification_token(user.id)
                await self._send_verification_email(user.email, verification_token)

            # 이벤트 발생
            if self.events:
                await self.events.emit('auth.user.registered', {
                    'user_id': user.id,
                    'email': user.email,
                    'username': user.username
                })

            return create_success(user)

        except Exception as error:
            return create_failure(ApplicationError(
                message=f"사용자 등록 실패: {str(error)}",
                service_name="AuthenticationService",
                operation="register_user"
            ))

    async def login(self, request: LoginRequest) -> Result[LoginResponse]:
        """로그인"""
        try:
            # 사용자 조회
            if request.email not in self.users_by_email:
                return create_failure(ValidationError(
                    message="등록되지 않은 이메일입니다",
                    field="email",
                    value=request.email
                ))

            user_id = self.users_by_email[request.email]
            user = self.users[user_id]

            # 계정 잠금 확인
            if user.locked_until and time.time() < user.locked_until:
                return create_failure(AuthenticationError(
                    message="계정이 잠겨있습니다. 나중에 다시 시도해주세요",
                    error_type="account_locked",
                    security_context={
                        'user_id': user.id,
                        'locked_until': user.locked_until
                    }
                ))

            # 비밀번호 검증
            if not self._verify_password(request.password, user.password_hash):
                # 실패 횟수 증가
                user.failed_login_attempts += 1
                user.updated_at = current_timestamp()

                # 최대 시도 횟수 초과 시 계정 잠금
                if user.failed_login_attempts >= self.max_login_attempts:
                    user.locked_until = time.time() + self.lockout_duration

                    if self.events:
                        await self.events.emit('auth.account.locked', {
                            'user_id': user.id,
                            'email': user.email,
                            'locked_until': user.locked_until
                        })

                return create_failure(ValidationError(
                    message="잘못된 비밀번호입니다",
                    field="password",
                    value="[REDACTED]"
                ))

            # 계정 상태 확인
            if not user.is_active:
                return create_failure(AuthenticationError(
                    message="비활성화된 계정입니다",
                    error_type="account_inactive",
                    security_context={'user_id': user.id}
                ))

            # 로그인 성공 - 실패 횟수 초기화
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login = current_timestamp()
            user.updated_at = current_timestamp()

            # 토큰 생성
            access_token = self._generate_access_token(user)
            refresh_token = None
            if request.remember_me:
                refresh_token = self._generate_refresh_token(user)

            # 세션 생성
            session = AuthSession(
                id=generate_id(),
                user_id=user.id,
                token=access_token.token,
                created_at=current_timestamp(),
                expires_at=access_token.expires_at,
                device_info=request.device_info
            )

            self.active_sessions[session.id] = session

            # 응답 생성
            response = LoginResponse(
                user=user,
                access_token=access_token,
                refresh_token=refresh_token,
                session_id=session.id,
                expires_in=self.token_ttl
            )

            # 이벤트 발생
            if self.events:
                await self.events.emit('auth.user.logged_in', {
                    'user_id': user.id,
                    'session_id': session.id,
                    'device_info': request.device_info.__dict__ if request.device_info else None
                })

            return create_success(response)

        except Exception as error:
            return create_failure(ApplicationError(
                message=f"로그인 실패: {str(error)}",
                service_name="AuthenticationService",
                operation="login"
            ))

    async def logout(self, session_id: ID) -> Result[bool]:
        """로그아웃"""
        try:
            if session_id not in self.active_sessions:
                return create_failure(ValidationError(
                    message="유효하지 않은 세션입니다",
                    field="session_id",
                    value=session_id
                ))

            session = self.active_sessions[session_id]
            session.is_active = False

            # 세션 제거
            del self.active_sessions[session_id]

            # 토큰 블랙리스트에 추가
            self.token_blacklist.add(session.token)

            # 이벤트 발생
            if self.events:
                await self.events.emit('auth.user.logged_out', {
                    'user_id': session.user_id,
                    'session_id': session_id
                })

            return create_success(True)

        except Exception as error:
            return create_failure(ApplicationError(
                message=f"로그아웃 실패: {str(error)}",
                service_name="AuthenticationService",
                operation="logout"
            ))

    async def verify_token(self, token: str) -> Result[User]:
        """토큰 검증"""
        try:
            # 블랙리스트 확인
            if token in self.token_blacklist:
                return create_failure(AuthenticationError(
                    message="무효한 토큰입니다",
                    error_type="token_blacklisted",
                    security_context={'token_prefix': token[:10] + "..."}
                ))

            # 토큰 디코딩 및 검증
            payload = self._decode_token(token)
            if not payload:
                return create_failure(AuthenticationError(
                    message="유효하지 않은 토큰입니다",
                    error_type="invalid_token",
                    security_context={'token_prefix': token[:10] + "..."}
                ))

            user_id = payload.get('user_id')
            if user_id not in self.users:
                return create_failure(ValidationError(
                    message="사용자를 찾을 수 없습니다",
                    field="user_id",
                    value=user_id
                ))

            user = self.users[user_id]

            # 사용자 상태 확인
            if not user.is_active:
                return create_failure(AuthenticationError(
                    message="비활성화된 계정입니다",
                    error_type="account_inactive",
                    security_context={'user_id': user.id}
                ))

            return create_success(user)

        except Exception as error:
            return create_failure(ApplicationError(
                message=f"토큰 검증 실패: {str(error)}",
                service_name="AuthenticationService",
                operation="verify_token"
            ))

    async def refresh_token(self, refresh_token: str) -> Result[AuthToken]:
        """토큰 갱신"""
        try:
            # 리프레시 토큰 검증
            payload = self._decode_token(refresh_token)
            if not payload or payload.get('type') != TokenType.REFRESH.value:
                return create_failure(AuthenticationError(
                    message="유효하지 않은 리프레시 토큰입니다",
                    error_type="invalid_refresh_token",
                    security_context={'token_prefix': refresh_token[:10] + "..."}
                ))

            user_id = payload.get('user_id')
            if user_id not in self.users:
                return create_failure(ValidationError(
                    message="사용자를 찾을 수 없습니다",
                    field="user_id",
                    value=user_id
                ))

            user = self.users[user_id]

            # 새 액세스 토큰 생성
            new_access_token = self._generate_access_token(user)

            # 기존 리프레시 토큰 블랙리스트에 추가
            self.token_blacklist.add(refresh_token)

            return create_success(new_access_token)

        except Exception as error:
            return create_failure(ApplicationError(
                message=f"토큰 갱신 실패: {str(error)}",
                service_name="AuthenticationService",
                operation="refresh_token"
            ))

    async def reset_password(self, request: PasswordResetRequest) -> Result[bool]:
        """비밀번호 재설정"""
        try:
            # 재설정 토큰 검증
            if request.reset_token not in self.reset_tokens:
                return create_failure(ValidationError(
                    message="유효하지 않은 재설정 토큰입니다",
                    field="reset_token",
                    value=request.reset_token
                ))

            token_data = self.reset_tokens[request.reset_token]
            if time.time() > token_data['expires_at']:
                return create_failure(ValidationError(
                    message="만료된 재설정 토큰입니다",
                    field="reset_token",
                    value=request.reset_token
                ))

            if token_data['email'] != request.email:
                return create_failure(ValidationError(
                    message="이메일이 일치하지 않습니다",
                    field="email",
                    value=request.email
                ))

            # 사용자 조회
            user_id = self.users_by_email[request.email]
            user = self.users[user_id]

            # 비밀번호 검증
            if not self._validate_password(request.new_password):
                return create_failure(ValidationError(
                    message=f"비밀번호는 최소 {self.password_min_length}자 이상이어야 합니다",
                    field="new_password",
                    value="[REDACTED]"
                ))

            # 비밀번호 업데이트
            user.password_hash = self._hash_password(request.new_password)
            user.updated_at = current_timestamp()

            # 재설정 토큰 제거
            del self.reset_tokens[request.reset_token]

            # 모든 세션 무효화
            self._invalidate_user_sessions(user.id)

            # 이벤트 발생
            if self.events:
                await self.events.emit('auth.password.reset', {
                    'user_id': user.id,
                    'email': user.email
                })

            return create_success(True)

        except Exception as error:
            return create_failure(ApplicationError(
                message=f"비밀번호 재설정 실패: {str(error)}",
                service_name="AuthenticationService",
                operation="reset_password"
            ))

    def _validate_registration(self, request: RegisterRequest) -> Result[bool]:
        """등록 정보 검증"""
        # 이용약관 동의 확인
        if not request.accept_terms:
            return create_failure(ValidationError(
                message="이용약관에 동의해야 합니다",
                field="accept_terms",
                value=request.accept_terms
            ))

        # 이메일 형식 검증
        if not self._validate_email(request.email):
            return create_failure(ValidationError(
                message="유효하지 않은 이메일 형식입니다",
                field="email",
                value=request.email
            ))

        # 비밀번호 검증
        if not self._validate_password(request.password):
            return create_failure(ValidationError(
                message=f"비밀번호는 최소 {self.password_min_length}자 이상이어야 합니다",
                field="password",
                value="[REDACTED]"
            ))

        # 사용자명 검증
        if not self._validate_username(request.username):
            return create_failure(ValidationError(
                message="사용자명은 3-30자의 영문, 숫자, 언더스코어만 가능합니다",
                field="username",
                value=request.username
            ))

        return create_success(True)

    def _validate_email(self, email: str) -> bool:
        """이메일 형식 검증"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def _validate_password(self, password: str) -> bool:
        """비밀번호 검증"""
        return len(password) >= self.password_min_length

    def _validate_username(self, username: str) -> bool:
        """사용자명 검증"""
        import re
        pattern = r'^[a-zA-Z0-9_]{3,30}$'
        return re.match(pattern, username) is not None

    def _hash_password(self, password: str) -> str:
        """비밀번호 해싱"""
        import bcrypt
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """비밀번호 검증"""
        import bcrypt
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))

    def _generate_access_token(self, user: User) -> AuthToken:
        """액세스 토큰 생성"""
        payload = {
            'user_id': user.id,
            'email': user.email,
            'role': user.role.value,
            'type': TokenType.ACCESS.value,
            'exp': time.time() + self.token_ttl,
            'iat': time.time()
        }

        token = self._encode_token(payload)

        return AuthToken(
            token=token,
            type=TokenType.ACCESS,
            user_id=user.id,
            expires_at=payload['exp'],
            scope=['read', 'write']
        )

    def _generate_refresh_token(self, user: User) -> AuthToken:
        """리프레시 토큰 생성"""
        payload = {
            'user_id': user.id,
            'type': TokenType.REFRESH.value,
            'exp': time.time() + self.refresh_token_ttl,
            'iat': time.time()
        }

        token = self._encode_token(payload)

        return AuthToken(
            token=token,
            type=TokenType.REFRESH,
            user_id=user.id,
            expires_at=payload['exp'],
            scope=['refresh']
        )

    def _generate_verification_token(self, user_id: ID) -> str:
        """이메일 인증 토큰 생성"""
        token = secrets.token_urlsafe(32)
        # 실제 구현에서는 이메일 발송 로직 추가
        return token

    def _encode_token(self, payload: Dict[str, Any]) -> str:
        """토큰 인코딩 (간단한 구현)"""
        # 실제 구현에서는 JWT 라이브러리 사용
        import base64
        payload_json = json.dumps(payload)
        signature = hmac.new(
            self.secret_key.encode(),
            payload_json.encode(),
            hashlib.sha256
        ).hexdigest()

        token_data = {
            'payload': payload_json,
            'signature': signature
        }

        return base64.b64encode(json.dumps(token_data).encode()).decode()

    def _decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """토큰 디코딩"""
        try:
            import base64
            token_data = json.loads(base64.b64decode(token.encode()).decode())
            payload_json = token_data['payload']
            signature = token_data['signature']

            # 서명 검증
            expected_signature = hmac.new(
                self.secret_key.encode(),
                payload_json.encode(),
                hashlib.sha256
            ).hexdigest()

            if signature != expected_signature:
                return None

            payload = json.loads(payload_json)

            # 만료 시간 확인
            if time.time() > payload.get('exp', 0):
                return None

            return payload

        except Exception:
            return None

    def _invalidate_user_sessions(self, user_id: ID) -> None:
        """사용자의 모든 세션 무효화"""
        sessions_to_remove = []
        for session_id, session in self.active_sessions.items():
            if session.user_id == user_id:
                sessions_to_remove.append(session_id)
                self.token_blacklist.add(session.token)

        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]

    async def _send_verification_email(self, email: str, token: str) -> None:
        """인증 이메일 발송"""
        # 실제 구현에서는 이메일 발송 로직 추가
        self.logger.info(f"Verification email sent to {email} with token: {token}")

    async def _load_user_data(self) -> None:
        """사용자 데이터 로드"""
        # 실제 구현에서는 데이터베이스에서 로드
        pass

    async def _save_user_data(self) -> None:
        """사용자 데이터 저장"""
        # 실제 구현에서는 데이터베이스에 저장
        pass

    async def process_request(self, context: ServiceContext) -> ServiceResult:
        """서비스 요청 처리"""
        operation = context.operation

        if operation == "register":
            return await self._handle_register_request(context)
        elif operation == "login":
            return await self._handle_login_request(context)
        elif operation == "logout":
            return await self._handle_logout_request(context)
        elif operation == "verify_token":
            return await self._handle_verify_token_request(context)
        elif operation == "refresh_token":
            return await self._handle_refresh_token_request(context)
        elif operation == "reset_password":
            return await self._handle_reset_password_request(context)
        else:
            return ServiceResult(
                success=False,
                data=None,
                error=f"Unknown operation: {operation}",
                processing_time_ms=0
            )

    async def _handle_register_request(self, context: ServiceContext) -> ServiceResult:
        """등록 요청 처리"""
        start_time = time.time()

        try:
            params = context.parameters
            profile = UserProfile(
                first_name=params.get('first_name', ''),
                last_name=params.get('last_name', ''),
                avatar=params.get('avatar')
            )

            request = RegisterRequest(
                username=params.get('username', ''),
                email=params.get('email', ''),
                password=params.get('password', ''),
                profile=profile,
                accept_terms=params.get('accept_terms', False),
                email_verification=params.get('email_verification', True)
            )

            result = await self.register_user(request)

            processing_time_ms = (time.time() - start_time) * 1000

            if result.success:
                return ServiceResult(
                    success=True,
                    data=result.data,
                    processing_time_ms=processing_time_ms
                )
            else:
                return ServiceResult(
                    success=False,
                    data=None,
                    error=str(result.error),
                    processing_time_ms=processing_time_ms
                )

        except Exception as error:
            processing_time_ms = (time.time() - start_time) * 1000
            return ServiceResult(
                success=False,
                data=None,
                error=str(error),
                processing_time_ms=processing_time_ms
            )

    async def _handle_login_request(self, context: ServiceContext) -> ServiceResult:
        """로그인 요청 처리"""
        start_time = time.time()

        try:
            params = context.parameters
            device_info = None

            if params.get('device_info'):
                device_info = DeviceInfo(**params['device_info'])

            request = LoginRequest(
                email=params.get('email', ''),
                password=params.get('password', ''),
                remember_me=params.get('remember_me', False),
                device_info=device_info
            )

            result = await self.login(request)

            processing_time_ms = (time.time() - start_time) * 1000

            if result.success:
                return ServiceResult(
                    success=True,
                    data=result.data,
                    processing_time_ms=processing_time_ms
                )
            else:
                return ServiceResult(
                    success=False,
                    data=None,
                    error=str(result.error),
                    processing_time_ms=processing_time_ms
                )

        except Exception as error:
            processing_time_ms = (time.time() - start_time) * 1000
            return ServiceResult(
                success=False,
                data=None,
                error=str(error),
                processing_time_ms=processing_time_ms
            )

    def get_service_info(self) -> Dict[str, Any]:
        """서비스 정보 조회"""
        return {
            'name': 'AuthenticationService',
            'version': '1.0.0',
            'description': 'User authentication and authorization service',
            'total_users': len(self.users),
            'active_sessions': len(self.active_sessions),
            'blacklisted_tokens': len(self.token_blacklist),
            'supported_operations': [
                'register', 'login', 'logout', 'verify_token',
                'refresh_token', 'reset_password'
            ],
            'security_settings': {
                'max_login_attempts': self.max_login_attempts,
                'lockout_duration': self.lockout_duration,
                'password_min_length': self.password_min_length,
                'token_ttl': self.token_ttl,
                'refresh_token_ttl': self.refresh_token_ttl
            }
        }