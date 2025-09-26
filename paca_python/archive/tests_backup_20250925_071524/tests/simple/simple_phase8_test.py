"""
Phase 8 간단한 통합 테스트
"""

import asyncio
import sys
import os

# PACA 모듈 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from paca.tools import (
    ReActFramework, PACAToolManager
)
from paca.tools.tools import WebSearchTool, FileManagerTool


async def simple_test():
    """간단한 통합 테스트"""
    print("=== Phase 8 도구 상자 시스템 테스트 ===")

    # 1. 도구 관리자 생성 및 도구 등록
    tool_manager = PACAToolManager()
    web_search = WebSearchTool()
    file_manager = FileManagerTool()

    tool_manager.register_tool(web_search)
    tool_manager.register_tool(file_manager)

    print(f"등록된 도구: {len(tool_manager.tools)}개")

    # 2. 웹 검색 테스트
    print("\n웹 검색 테스트...")
    result = await tool_manager.execute_tool('web_search', query='Python')
    if result.success:
        print(f"웹 검색 성공 - 결과 {len(result.data)}개")
    else:
        print(f"웹 검색 실패: {result.error}")

    # 3. 파일 관리 테스트
    print("\n파일 관리 테스트...")

    # 파일 쓰기
    content = "Test content for PACA Phase 8"
    result = await tool_manager.execute_tool(
        'file_manager',
        operation='write',
        path='test.txt',
        content=content
    )
    if result.success:
        print("파일 쓰기 성공")
    else:
        print(f"파일 쓰기 실패: {result.error}")

    # 파일 읽기
    result = await tool_manager.execute_tool(
        'file_manager',
        operation='read',
        path='test.txt'
    )
    if result.success:
        print("파일 읽기 성공")
        print(f"내용: {result.data}")
    else:
        print(f"파일 읽기 실패: {result.error}")

    # 4. ReAct 프레임워크 테스트
    print("\nReAct 프레임워크 테스트...")
    react = ReActFramework(tool_manager)

    session = await react.create_session(
        goal="간단한 테스트 실행하기",
        max_steps=3
    )
    print(f"세션 생성 완료: {session.id[:8]}...")

    # 생각 단계
    await react.think(session, "테스트를 위해 간단한 작업을 수행하자.")

    # 행동 단계
    await react.act(session, 'web_search', query='test')

    # 관찰 단계
    await react.observe(session, "검색 작업을 완료했다.")

    print(f"ReAct 세션 완료 - 총 {len(session.steps)}단계")

    # 5. 통계 확인
    stats = tool_manager.get_usage_statistics()
    print(f"\n사용 통계:")
    print(f"총 실행: {stats['total_executions']}회")
    print(f"성공률: {stats['success_rate']:.1%}")

    print("\n=== 모든 테스트 완료 ===")
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(simple_test())
        print("테스트 성공!" if success else "테스트 실패!")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()