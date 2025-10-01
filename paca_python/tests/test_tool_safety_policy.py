import pytest

from paca.tools.base import SafetyPolicy, Tool, ToolResult, ToolType
from paca.tools.tool_manager import PACAToolManager


class EchoTool(Tool):
    """단순 성공 응답을 반환하는 테스트 도구"""

    def __init__(self, name: str = "echo"):
        super().__init__(name, ToolType.UTILITY, "Echo test tool")

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, data=kwargs.get("payload", "ok"))

    def validate_input(self, **kwargs) -> bool:
        return True


@pytest.mark.asyncio
async def test_safety_policy_rate_limit_blocks_when_exceeded():
    policy = SafetyPolicy()
    policy.configure_rate_limit("echo", max_calls=2, per_seconds=60)

    assert policy.is_operation_allowed("echo")
    assert policy.consume_rate_limit("echo")
    assert policy.consume_rate_limit("echo")
    assert not policy.consume_rate_limit("echo")
    assert policy.will_exceed_rate_limit("echo")


@pytest.mark.asyncio
async def test_tool_manager_respects_configured_rate_limits():
    policy = SafetyPolicy()
    policy.configure_rate_limit("echo", max_calls=1, per_seconds=60)

    manager = PACAToolManager(safety_policy=policy)
    tool = EchoTool()
    manager.register_tool(tool)

    first = await manager.execute_tool("echo", payload="first")
    assert first.success

    second = await manager.execute_tool("echo", payload="second")
    assert not second.success
    assert "속도 제한" in (second.error or "")


@pytest.mark.asyncio
async def test_global_rate_limit_rule_applies_to_tools_without_specific_entry():
    policy = SafetyPolicy()
    policy.configure_rate_limit("*", max_calls=1, per_seconds=60)

    manager = PACAToolManager(safety_policy=policy)
    tool = EchoTool("generic")
    manager.register_tool(tool)

    assert (await manager.execute_tool("generic")).success
    blocked = await manager.execute_tool("generic")
    assert not blocked.success
    assert "속도 제한" in (blocked.error or "")
