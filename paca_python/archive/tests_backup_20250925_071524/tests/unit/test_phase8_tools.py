"""
Phase 8 도구 상자 시스템 통합 테스트

ReAct 프레임워크, 웹 검색 도구, 파일 관리 도구의 통합 테스트
"""

import asyncio
import sys
import os
import tempfile
from pathlib import Path

# PACA 모듈 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from paca.tools import (
    ReActFramework, ReActSession, ReActStepType,
    PACAToolManager, SafetyPolicy
)
from paca.tools.tools import WebSearchTool, FileManagerTool


async def test_tool_manager():
    """도구 관리자 테스트"""
    print("=== 도구 관리자 테스트 시작 ===")

    # 도구 관리자 생성
    tool_manager = PACAToolManager()

    # 웹 검색 도구 등록
    web_search = WebSearchTool()
    file_manager = FileManagerTool()

    assert tool_manager.register_tool(web_search), "웹 검색 도구 등록 실패"
    assert tool_manager.register_tool(file_manager), "파일 관리 도구 등록 실패"

    print(f"등록된 도구 수: {len(tool_manager.tools)}")

    # 도구 목록 확인
    tools = tool_manager.list_tools()
    tool_names = [tool.name for tool in tools]
    assert 'web_search' in tool_names, "웹 검색 도구를 찾을 수 없음"
    assert 'file_manager' in tool_names, "파일 관리 도구를 찾을 수 없음"

    print("도구 목록 확인 완료")

    # 도구 정보 조회
    web_info = tool_manager.get_tool_info('web_search')
    assert web_info is not None, "웹 검색 도구 정보 조회 실패"
    assert web_info['type'] == 'search', "웹 검색 도구 타입 불일치"

    print("도구 정보 조회 완료")

    return tool_manager


async def test_web_search_tool(tool_manager):
    """웹 검색 도구 테스트"""
    print("\n=== 웹 검색 도구 테스트 시작 ===")

    # 기본 검색 테스트
    result = await tool_manager.execute_tool('web_search', query='Python 프로그래밍')
    assert result.success, f"웹 검색 실패: {result.error}"
    assert result.data is not None, "검색 결과 데이터가 없음"
    assert isinstance(result.data, list), "검색 결과가 리스트가 아님"

    print(f"✅ 기본 검색 완료 - 결과 수: {len(result.data)}")

    # 검색 결과 구조 확인
    if result.data:
        first_result = result.data[0]
        required_fields = ['title', 'url', 'snippet', 'score']
        for field in required_fields:
            assert field in first_result, f"검색 결과에 '{field}' 필드가 없음"

    print("✅ 검색 결과 구조 확인 완료")

    # 검색 통계 확인
    web_tool = tool_manager.get_tool('web_search')
    stats = web_tool.get_search_statistics()
    assert 'supported_engines' in stats, "검색 통계에 지원 엔진 정보 없음"

    print("✅ 검색 통계 확인 완료")

    return result


async def test_file_manager_tool(tool_manager):
    """파일 관리 도구 테스트"""
    print("\n=== 파일 관리 도구 테스트 시작 ===")

    # 파일 쓰기 테스트
    test_content = "이것은 테스트 파일입니다.\n안녕하세요, PACA!"
    result = await tool_manager.execute_tool(
        'file_manager',
        operation='write',
        path='test_file.txt',
        content=test_content
    )
    assert result.success, f"파일 쓰기 실패: {result.error}"

    print("✅ 파일 쓰기 완료")

    # 파일 읽기 테스트
    result = await tool_manager.execute_tool(
        'file_manager',
        operation='read',
        path='test_file.txt'
    )
    assert result.success, f"파일 읽기 실패: {result.error}"
    assert result.data == test_content, "읽은 내용이 원본과 다름"

    print("✅ 파일 읽기 완료")

    # 파일 목록 테스트
    result = await tool_manager.execute_tool(
        'file_manager',
        operation='list',
        path='.'
    )
    assert result.success, f"파일 목록 조회 실패: {result.error}"
    assert isinstance(result.data, list), "파일 목록이 리스트가 아님"

    # 방금 생성한 파일이 목록에 있는지 확인
    file_names = [item['name'] for item in result.data]
    assert 'test_file.txt' in file_names, "생성한 파일이 목록에 없음"

    print(f"✅ 파일 목록 조회 완료 - 항목 수: {len(result.data)}")

    # 파일 정보 테스트
    result = await tool_manager.execute_tool(
        'file_manager',
        operation='info',
        path='test_file.txt'
    )
    assert result.success, f"파일 정보 조회 실패: {result.error}"
    assert result.data['name'] == 'test_file.txt', "파일 정보의 이름이 다름"
    assert not result.data['is_directory'], "파일이 디렉토리로 인식됨"

    print("✅ 파일 정보 조회 완료")

    # 파일 검색 테스트
    result = await tool_manager.execute_tool(
        'file_manager',
        operation='search',
        path='.',
        search_text='PACA'
    )
    assert result.success, f"파일 검색 실패: {result.error}"
    assert len(result.data) > 0, "검색 결과가 없음"

    print(f"✅ 파일 검색 완료 - 매치 수: {len(result.data)}")

    # 샌드박스 정보 확인
    file_tool = tool_manager.get_tool('file_manager')
    sandbox_info = file_tool.get_sandbox_info()
    assert 'sandbox_path' in sandbox_info, "샌드박스 정보에 경로가 없음"

    print("✅ 샌드박스 정보 확인 완료")

    return result


async def test_react_framework(tool_manager):
    """ReAct 프레임워크 테스트"""
    print("\n=== ReAct 프레임워크 테스트 시작 ===")

    # ReAct 프레임워크 생성
    react = ReActFramework(tool_manager)

    # 세션 생성
    session = await react.create_session(
        goal="Python에 대한 정보를 검색하고 파일에 저장하기",
        max_steps=8
    )
    assert session is not None, "세션 생성 실패"
    assert session.goal != "", "세션 목표가 비어있음"

    print(f"✅ 세션 생성 완료 - ID: {session.id[:8]}...")

    # 단계별 실행 테스트

    # 1. 생각 단계
    step = await react.think(
        session,
        "Python에 대한 최신 정보를 검색해야겠다.",
        confidence=0.8
    )
    assert step.step_type == ReActStepType.THOUGHT, "생각 단계 타입 불일치"
    assert len(session.steps) == 1, "세션 단계 수 불일치"

    print("✅ 생각 단계 완료")

    # 2. 행동 단계 - 웹 검색
    step = await react.act(
        session,
        'web_search',
        query='Python 프로그래밍 언어 특징'
    )
    assert step.step_type == ReActStepType.ACTION, "행동 단계 타입 불일치"
    assert step.result is not None, "행동 결과가 없음"

    print("✅ 웹 검색 행동 완료")

    # 3. 관찰 단계
    if step.result.success:
        observation = f"검색에 성공했습니다. {len(step.result.data)}개의 결과를 찾았습니다."
    else:
        observation = f"검색에 실패했습니다: {step.result.error}"

    step = await react.observe(session, observation, confidence=0.7)
    assert step.step_type == ReActStepType.OBSERVATION, "관찰 단계 타입 불일치"

    print("✅ 관찰 단계 완료")

    # 4. 행동 단계 - 파일 저장
    if len(session.steps) >= 2 and session.steps[1].result.success:
        search_results = session.steps[1].result.data
        summary = f"Python 검색 결과 요약:\n"
        for i, result in enumerate(search_results[:3], 1):
            summary += f"{i}. {result['title']}\n   {result['snippet']}\n\n"
    else:
        summary = "검색 결과를 가져올 수 없어 기본 정보를 저장합니다.\nPython은 간단하고 읽기 쉬운 프로그래밍 언어입니다."

    step = await react.act(
        session,
        'file_manager',
        operation='write',
        path='python_info.txt',
        content=summary
    )
    assert step.step_type == ReActStepType.ACTION, "파일 저장 행동 타입 불일치"

    print("✅ 파일 저장 행동 완료")

    # 5. 성찰 단계
    step = await react.reflect(
        session,
        "Python 정보를 성공적으로 검색하고 파일에 저장했습니다. 목표를 달성했습니다.",
        confidence=0.9
    )
    assert step.step_type == ReActStepType.REFLECTION, "성찰 단계 타입 불일치"

    print("✅ 성찰 단계 완료")

    # 세션 요약
    print(f"\n📊 세션 요약:")
    print(f"   - 총 단계 수: {len(session.steps)}")
    print(f"   - 생각 단계: {len(session.get_steps_by_type(ReActStepType.THOUGHT))}")
    print(f"   - 행동 단계: {len(session.get_steps_by_type(ReActStepType.ACTION))}")
    print(f"   - 관찰 단계: {len(session.get_steps_by_type(ReActStepType.OBSERVATION))}")
    print(f"   - 성찰 단계: {len(session.get_steps_by_type(ReActStepType.REFLECTION))}")

    return session


async def test_system_integration():
    """전체 시스템 통합 테스트"""
    print("\n=== 전체 시스템 통합 테스트 시작 ===")

    # 도구 관리자 테스트
    tool_manager = await test_tool_manager()

    # 개별 도구 테스트
    await test_web_search_tool(tool_manager)
    await test_file_manager_tool(tool_manager)

    # ReAct 프레임워크 테스트
    session = await test_react_framework(tool_manager)

    # 사용 통계 확인
    stats = tool_manager.get_usage_statistics()
    print(f"\n📈 사용 통계:")
    print(f"   - 총 도구 수: {stats['total_tools']}")
    print(f"   - 총 실행 횟수: {stats['total_executions']}")
    print(f"   - 성공률: {stats['success_rate']:.1%}")
    print(f"   - 평균 실행 시간: {stats['average_execution_time']:.3f}초")

    if stats['most_used_tools']:
        print(f"   - 가장 많이 사용된 도구: {stats['most_used_tools'][0][0]} ({stats['most_used_tools'][0][1]}회)")

    # 상태 확인
    health = await tool_manager.health_check()
    print(f"\n🏥 시스템 상태:")
    print(f"   - 전체 상태: {'✅ 정상' if health['system_healthy'] else '❌ 문제 있음'}")
    print(f"   - 도구 상태: {len([s for s in health['tools_status'].values() if s.get('healthy', False)])} / {len(health['tools_status'])} 정상")

    if health['issues']:
        print(f"   - 문제점: {len(health['issues'])}개")
        for issue in health['issues']:
            print(f"     • {issue}")

    print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")

    return {
        'tool_manager': tool_manager,
        'session': session,
        'stats': stats,
        'health': health
    }


async def main():
    """메인 테스트 함수"""
    print("PACA Phase 8 도구 상자 시스템 통합 테스트 시작")
    print("=" * 60)

    try:
        results = await test_system_integration()

        print("\n" + "=" * 60)
        print("✅ 모든 테스트 통과!")
        print("\n📋 테스트 결과 요약:")
        print(f"   • 도구 관리자: 정상 작동")
        print(f"   • 웹 검색 도구: 정상 작동")
        print(f"   • 파일 관리 도구: 정상 작동")
        print(f"   • ReAct 프레임워크: 정상 작동")
        print(f"   • 시스템 통합: 정상 작동")

        print(f"\n🎯 Phase 8 구현 완료!")
        print(f"   • ReAct 프레임워크 ✅")
        print(f"   • 도구 기반 시스템 ✅")
        print(f"   • 웹 검색 도구 (The Scout) ✅")
        print(f"   • 파일 관리 도구 (The Librarian) ✅")
        print(f"   • 통합 테스트 ✅")

        return True

    except Exception as e:
        print(f"\n❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 테스트 실행
    success = asyncio.run(main())
    sys.exit(0 if success else 1)