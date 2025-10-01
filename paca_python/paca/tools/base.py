"""
도구 기본 클래스 및 인터페이스

ReAct 프레임워크의 기본 도구 아키텍처를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from collections import deque, defaultdict


@dataclass(frozen=True)
class RateLimitRule:
    """단일 작업에 적용되는 속도 제한 규칙"""

    max_calls: int
    per_seconds: float

    def __post_init__(self):
        if self.max_calls <= 0:
            raise ValueError("max_calls must be positive")
        if self.per_seconds <= 0:
            raise ValueError("per_seconds must be positive")

T = TypeVar('T')


class ToolType(Enum):
    """도구 유형 분류"""
    SEARCH = "search"           # 웹 검색, 정보 수집
    FILE = "file"              # 파일 관리, 읽기/쓰기
    ANALYSIS = "analysis"      # 데이터 분석, 검증
    GENERATION = "generation"  # 콘텐츠 생성
    COMMUNICATION = "communication"  # 외부 통신
    UTILITY = "utility"        # 유틸리티 기능


class ToolStatus(Enum):
    """도구 실행 상태"""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class ToolResult:
    """도구 실행 결과"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """결과를 딕셔너리로 변환"""
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'metadata': self.metadata,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat()
        }


class Tool(ABC):
    """도구 기본 클래스"""

    def __init__(self, name: str, tool_type: ToolType, description: str = ""):
        self.id = str(uuid.uuid4())
        self.name = name
        self.tool_type = tool_type
        self.description = description
        self.status = ToolStatus.IDLE
        self.created_at = datetime.now()
        self.last_used = None
        self.usage_count = 0

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """도구 실행 (비동기)"""
        pass

    def execute_sync(self, **kwargs) -> ToolResult:
        """도구 실행 (동기)"""
        import asyncio
        return asyncio.run(self.execute(**kwargs))

    @abstractmethod
    def validate_input(self, **kwargs) -> bool:
        """입력 파라미터 검증"""
        pass

    def get_info(self) -> Dict[str, Any]:
        """도구 정보 반환"""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.tool_type.value,
            'description': self.description,
            'status': self.status.value,
            'usage_count': self.usage_count,
            'created_at': self.created_at.isoformat(),
            'last_used': self.last_used.isoformat() if self.last_used else None
        }

    def _update_usage(self):
        """사용 통계 업데이트"""
        self.usage_count += 1
        self.last_used = datetime.now()


class SafetyPolicy:
    """도구 안전성 정책"""

    def __init__(self):
        self.allowed_operations = set()
        self.blocked_operations = set()
        self.rate_limits: Dict[str, RateLimitRule] = {}
        self._operation_usage: Dict[str, deque] = defaultdict(deque)
        self.sandbox_mode = True

    def is_operation_allowed(self, operation: str, context: Dict[str, Any] = None) -> bool:
        """작업 허용 여부 확인"""
        if operation in self.blocked_operations:
            return False

        if self.allowed_operations and operation not in self.allowed_operations:
            return False

        return True

    def configure_rate_limit(self, operation: str, max_calls: int, per_seconds: float) -> None:
        """지정된 작업에 대한 속도 제한을 설정"""

        self.rate_limits[operation] = RateLimitRule(max_calls=max_calls, per_seconds=per_seconds)

    def clear_rate_limit(self, operation: str) -> None:
        """설정된 속도 제한 제거"""

        self.rate_limits.pop(operation, None)
        self._operation_usage.pop(operation, None)

    def get_rate_limit(self, operation: str, include_global: bool = True) -> Optional[RateLimitRule]:
        """특정 작업 또는 글로벌('*') 속도 제한 규칙 조회"""

        if operation in self.rate_limits:
            return self.rate_limits[operation]
        if include_global and "*" in self.rate_limits:
            return self.rate_limits["*"]
        return None

    def will_exceed_rate_limit(self, operation: str) -> bool:
        """현재 시점에서 호출 시 속도 제한을 초과하는지 확인 (카운트는 증가시키지 않음)"""

        rule = self.get_rate_limit(operation)
        if not rule:
            return False

        now = datetime.utcnow()
        usage_window = now - timedelta(seconds=rule.per_seconds)
        usage = self._operation_usage[operation]

        while usage and usage[0] < usage_window:
            usage.popleft()

        return len(usage) >= rule.max_calls

    def consume_rate_limit(self, operation: str) -> bool:
        """속도 제한 검증 후 호출을 기록"""

        rule = self.get_rate_limit(operation)
        if not rule:
            return True

        if self.will_exceed_rate_limit(operation):
            return False

        self._operation_usage[operation].append(datetime.utcnow())
        return True

    def validate_parameters(self, tool_name: str, params: Dict[str, Any]) -> bool:
        """파라미터 안전성 검증"""
        # 기본 안전성 검사
        if self.sandbox_mode:
            # 샌드박스 모드에서의 제한 사항
            dangerous_params = ['system', 'exec', 'eval', '__import__']
            for param, value in params.items():
                if isinstance(value, str) and any(dp in value for dp in dangerous_params):
                    return False

        return True


class ToolManager(ABC):
    """도구 관리자 기본 클래스"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.safety_policy = SafetyPolicy()

    @abstractmethod
    def register_tool(self, tool: Tool) -> bool:
        """도구 등록"""
        pass

    @abstractmethod
    def get_tool(self, name: str) -> Optional[Tool]:
        """도구 가져오기"""
        pass

    @abstractmethod
    def list_tools(self, tool_type: Optional[ToolType] = None) -> List[Tool]:
        """도구 목록"""
        pass

    @abstractmethod
    async def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """도구 실행"""
        pass


class ToolError(Exception):
    """도구 관련 예외"""
    pass


class ToolValidationError(ToolError):
    """도구 검증 예외"""
    pass


class ToolExecutionError(ToolError):
    """도구 실행 예외"""
    pass


class ToolTimeoutError(ToolError):
    """도구 타임아웃 예외"""
    pass