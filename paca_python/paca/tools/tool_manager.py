"""
PACA 도구 관리자

ReAct 프레임워크와 통합된 도구 관리 시스템
"""

import time
import asyncio
from typing import Dict, List, Optional, Type, Callable, Any
from datetime import datetime, timedelta
import logging
from .base import Tool, ToolResult, ToolManager, ToolType, ToolStatus, SafetyPolicy
from .base import ToolError, ToolValidationError, ToolExecutionError, ToolTimeoutError


class PACAToolManager(ToolManager):
    """PACA 전용 도구 관리자"""

    def __init__(self, safety_policy: Optional[SafetyPolicy] = None):
        super().__init__()
        if safety_policy is None:
            safety_policy = SafetyPolicy()
            # 기본적으로 모든 도구 작업 허용
            safety_policy.allowed_operations = set()  # 빈 세트는 모든 작업 허용
        self.safety_policy = safety_policy
        self.logger = logging.getLogger(__name__)
        self.execution_history: List[Dict[str, Any]] = []
        self.rate_limits: Dict[str, List[datetime]] = {}

    def register_tool(self, tool: Tool) -> bool:
        """도구 등록 (동기)"""
        try:
            if not isinstance(tool, Tool):
                raise ToolValidationError(f"도구 객체가 Tool 클래스를 상속받지 않았습니다: {type(tool)}")

            if tool.name in self.tools:
                self.logger.warning(f"도구 '{tool.name}'이 이미 등록되어 있습니다. 덮어씁니다.")

            self.tools[tool.name] = tool
            self.logger.info(f"도구 '{tool.name}' ({tool.tool_type.value}) 등록 완료")
            return True

        except Exception as e:
            self.logger.error(f"도구 등록 실패: {e}")
            return False

    async def register_tool_async(self, tool: Tool) -> bool:
        """도구 등록 (비동기)"""
        # 비동기 버전은 동기 버전을 호출
        return self.register_tool(tool)

    def unregister_tool(self, tool_name: str) -> bool:
        """도구 등록 해제"""
        if tool_name in self.tools:
            del self.tools[tool_name]
            self.logger.info(f"도구 '{tool_name}' 등록 해제")
            return True
        return False

    def get_tool(self, name: str) -> Optional[Tool]:
        """도구 가져오기"""
        return self.tools.get(name)

    def list_tools(self, tool_type: Optional[ToolType] = None) -> List[Tool]:
        """도구 목록"""
        tools = list(self.tools.values())
        if tool_type:
            tools = [t for t in tools if t.tool_type == tool_type]
        return tools

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """도구 정보 가져오기"""
        tool = self.get_tool(tool_name)
        return tool.get_info() if tool else None

    async def execute_tool(self, tool_name: str, timeout: float = 30.0, **kwargs) -> ToolResult:
        """도구 실행"""
        start_time = time.time()

        try:
            # 도구 존재 확인
            tool = self.get_tool(tool_name)
            if not tool:
                raise ToolValidationError(f"도구 '{tool_name}'을 찾을 수 없습니다.")

            # 안전성 검사
            if not self._validate_execution(tool, kwargs):
                raise ToolValidationError(f"도구 '{tool_name}' 실행이 안전성 정책에 위배됩니다.")

            # 속도 제한 확인
            if not self._check_rate_limit(tool_name):
                raise ToolExecutionError(f"도구 '{tool_name}'의 속도 제한을 초과했습니다.")

            # 도구 상태 업데이트
            tool.status = ToolStatus.RUNNING

            # 타임아웃과 함께 실행
            try:
                result = await asyncio.wait_for(tool.execute(**kwargs), timeout=timeout)
                tool.status = ToolStatus.SUCCESS if result.success else ToolStatus.ERROR
            except asyncio.TimeoutError:
                tool.status = ToolStatus.TIMEOUT
                raise ToolTimeoutError(f"도구 '{tool_name}' 실행이 {timeout}초 후 타임아웃되었습니다.")

            # 실행 시간 계산
            execution_time = time.time() - start_time
            result.execution_time = execution_time

            # 사용 통계 업데이트
            tool._update_usage()

            # 실행 기록 저장
            self._record_execution(tool_name, kwargs, result, execution_time)

            self.logger.info(f"도구 '{tool_name}' 실행 완료 (성공: {result.success}, {execution_time:.2f}초)")
            return result

        except Exception as e:
            tool.status = ToolStatus.ERROR
            execution_time = time.time() - start_time

            error_result = ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )

            self._record_execution(tool_name, kwargs, error_result, execution_time)
            self.logger.error(f"도구 '{tool_name}' 실행 오류: {e}")

            return error_result

    def _validate_execution(self, tool: Tool, params: Dict[str, Any]) -> bool:
        """실행 검증"""
        # 입력 파라미터 검증
        if not tool.validate_input(**params):
            return False

        # 안전성 정책 검증
        if not self.safety_policy.validate_parameters(tool.name, params):
            return False

        # 도구별 작업 허용 여부 확인
        if not self.safety_policy.is_operation_allowed(tool.name):
            return False

        return True

    def _check_rate_limit(self, tool_name: str, max_calls: int = 10,
                         window_minutes: int = 1) -> bool:
        """속도 제한 확인"""

        # SafetyPolicy에 정의된 속도 제한이 우선한다.
        if self.safety_policy.get_rate_limit(tool_name) or self.safety_policy.get_rate_limit("*"):
            return self.safety_policy.consume_rate_limit(tool_name)

        now = datetime.now()
        window_start = now - timedelta(minutes=window_minutes)

        if tool_name not in self.rate_limits:
            self.rate_limits[tool_name] = []

        # 윈도우 시간 내의 호출만 유지
        recent_calls = [
            call_time for call_time in self.rate_limits[tool_name]
            if call_time >= window_start
        ]
        self.rate_limits[tool_name] = recent_calls

        # 제한 확인
        if len(recent_calls) >= max_calls:
            return False

        # 현재 호출 기록
        self.rate_limits[tool_name].append(now)
        return True

    def _record_execution(self, tool_name: str, params: Dict[str, Any],
                         result: ToolResult, execution_time: float):
        """실행 기록 저장"""
        record = {
            'tool_name': tool_name,
            'params': params,
            'success': result.success,
            'error': result.error,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat()
        }

        self.execution_history.append(record)

        # 기록 크기 제한 (최근 1000개만 유지)
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]

    def get_execution_history(self, tool_name: Optional[str] = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """실행 기록 조회"""
        history = self.execution_history

        if tool_name:
            history = [h for h in history if h['tool_name'] == tool_name]

        return history[-limit:] if limit else history

    def get_usage_statistics(self) -> Dict[str, Any]:
        """사용 통계 조회"""
        stats = {
            'total_tools': len(self.tools),
            'tools_by_type': {},
            'total_executions': len(self.execution_history),
            'success_rate': 0.0,
            'average_execution_time': 0.0,
            'most_used_tools': []
        }

        # 도구 유형별 개수
        for tool in self.tools.values():
            tool_type = tool.tool_type.value
            stats['tools_by_type'][tool_type] = stats['tools_by_type'].get(tool_type, 0) + 1

        if self.execution_history:
            # 성공률
            successful = sum(1 for h in self.execution_history if h['success'])
            stats['success_rate'] = successful / len(self.execution_history)

            # 평균 실행 시간
            total_time = sum(h['execution_time'] for h in self.execution_history)
            stats['average_execution_time'] = total_time / len(self.execution_history)

            # 가장 많이 사용된 도구
            tool_usage = {}
            for h in self.execution_history:
                tool_name = h['tool_name']
                tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

            stats['most_used_tools'] = sorted(
                tool_usage.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

        return stats

    def clear_execution_history(self):
        """실행 기록 삭제"""
        self.execution_history.clear()
        self.logger.info("실행 기록이 삭제되었습니다.")

    def reset_tool_status(self):
        """모든 도구 상태 초기화"""
        for tool in self.tools.values():
            tool.status = ToolStatus.IDLE
        self.logger.info("모든 도구 상태가 초기화되었습니다.")

    def validate_all_tools(self) -> Dict[str, bool]:
        """모든 도구 검증"""
        results = {}
        for name, tool in self.tools.items():
            try:
                # 기본 검증 파라미터로 테스트
                results[name] = tool.validate_input()
            except Exception as e:
                self.logger.error(f"도구 '{name}' 검증 실패: {e}")
                results[name] = False
        return results

    def get_tool_status_summary(self) -> Dict[str, int]:
        """도구 상태 요약"""
        status_count = {}
        for tool in self.tools.values():
            status = tool.status.value
            status_count[status] = status_count.get(status, 0) + 1
        return status_count

    def find_tools_by_capability(self, capability: str) -> List[Tool]:
        """기능별 도구 검색"""
        matching_tools = []
        for tool in self.tools.values():
            if (capability.lower() in tool.name.lower() or
                capability.lower() in tool.description.lower()):
                matching_tools.append(tool)
        return matching_tools

    async def health_check(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'total_tools': len(self.tools),
            'tools_status': {},
            'system_healthy': True,
            'issues': []
        }

        for name, tool in self.tools.items():
            try:
                # 간단한 검증으로 도구 상태 확인
                is_healthy = tool.validate_input()
                health_status['tools_status'][name] = {
                    'healthy': is_healthy,
                    'status': tool.status.value,
                    'usage_count': tool.usage_count
                }

                if not is_healthy:
                    health_status['system_healthy'] = False
                    health_status['issues'].append(f"도구 '{name}'에 문제가 있습니다.")

            except Exception as e:
                health_status['tools_status'][name] = {
                    'healthy': False,
                    'status': 'error',
                    'error': str(e)
                }
                health_status['system_healthy'] = False
                health_status['issues'].append(f"도구 '{name}' 상태 확인 실패: {e}")

        return health_status