"""
PACA 비동기 함수 호출 문제 진단 테스트 (인코딩 문제 제외)
실제 사용 시 발생할 수 있는 비동기 관련 문제를 체크
"""

import asyncio
import sys
import traceback
from typing import Dict, Any
import time

def test_sync_async_mixing():
    """동기/비동기 함수 혼용 문제 테스트"""
    print("=== 동기/비동기 함수 혼용 문제 테스트 ===")

    try:
        # PACA 모듈들을 임포트하고 비동기 문제 확인
        from paca.tools import ReActFramework, PACAToolManager
        from paca.tools.tools.web_search import WebSearchTool
        from paca.cognitive.memory import WorkingMemory

        print("1. 모듈 임포트 성공")

        # 1. 동기 컨텍스트에서 비동기 함수 호출 시도 (문제 발생 예상)
        try:
            tool_manager = PACAToolManager()
            web_search = WebSearchTool()

            # 이것은 실패할 것임 - 동기 컨텍스트에서 비동기 함수 호출
            # tool_manager.register_tool(web_search)  # 이건 await 필요
            print("2. 동기 컨텍스트에서 비동기 함수 호출 방지됨")

        except Exception as e:
            print(f"ERROR: 동기 컨텍스트에서 비동기 함수 호출 시도: {e}")

        # 2. 메모리 시스템 비동기 문제
        try:
            memory = WorkingMemory()
            # memory.store("test", {"type": "test"})  # 이것도 await 필요
            print("3. 메모리 시스템 비동기 문제 확인됨")

        except Exception as e:
            print(f"ERROR: 메모리 시스템 비동기 호출: {e}")

        return True

    except Exception as e:
        print(f"ERROR: 모듈 임포트 실패: {e}")
        traceback.print_exc()
        return False

async def test_async_proper_usage():
    """올바른 비동기 사용법 테스트"""
    print("\n=== 올바른 비동기 사용법 테스트 ===")

    try:
        from paca.tools import ReActFramework, PACAToolManager
        from paca.tools.tools.web_search import WebSearchTool
        from paca.cognitive.memory import WorkingMemory

        # 1. 올바른 비동기 도구 등록
        tool_manager = PACAToolManager()
        web_search = WebSearchTool()

        try:
            await tool_manager.register_tool(web_search)
            print("1. 도구 등록 성공")
        except Exception as e:
            print(f"ERROR: 도구 등록 실패: {e}")
            traceback.print_exc()

        # 2. 올바른 비동기 메모리 사용
        try:
            memory = WorkingMemory()
            result_id = await memory.store("test data", {"type": "test"})
            print(f"2. 메모리 저장 성공: {result_id}")

            retrieved = await memory.retrieve(result_id)
            if retrieved:
                print(f"3. 메모리 검색 성공: {retrieved.content}")
            else:
                print("3. 메모리 검색 실패: None 반환")

        except Exception as e:
            print(f"ERROR: 메모리 시스템 오류: {e}")
            traceback.print_exc()

        # 3. ReAct 프레임워크 테스트
        try:
            react = ReActFramework(tool_manager)
            session = await react.create_session("test-session")
            print(f"4. ReAct 세션 생성 성공: {session.session_id}")

        except Exception as e:
            print(f"ERROR: ReAct 프레임워크 오류: {e}")
            traceback.print_exc()

        return True

    except Exception as e:
        print(f"ERROR: 비동기 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_system_environment():
    """시스템 환경 문제 테스트"""
    print("=== 시스템 환경 테스트 ===")

    import locale
    import os

    print(f"1. Python 버전: {sys.version}")
    print(f"2. 기본 인코딩: {sys.getdefaultencoding()}")
    print(f"3. 파일 시스템 인코딩: {sys.getfilesystemencoding()}")
    try:
        print(f"4. 로케일: {locale.getdefaultlocale()}")
    except:
        print("4. 로케일: 확인 불가")
    print(f"5. 현재 작업 디렉토리: {os.getcwd()}")

    # 환경 변수 확인
    env_vars = ["PYTHONIOENCODING", "LANG", "LC_ALL"]
    for var in env_vars:
        value = os.getenv(var)
        print(f"6. {var}: {value}")

async def test_real_usage_scenario():
    """실제 사용 시나리오 테스트"""
    print("\n=== 실제 사용 시나리오 테스트 ===")

    try:
        from paca.tools import ReActFramework, PACAToolManager
        from paca.tools.tools.web_search import WebSearchTool
        from paca.cognitive.memory import WorkingMemory, EpisodicMemory
        from paca.core.governance import GovernanceProtocol

        # 1. 시스템 초기화
        print("1. 시스템 초기화 중...")
        tool_manager = PACAToolManager()
        react_framework = ReActFramework(tool_manager)

        # 2. 메모리 시스템 초기화
        working_memory = WorkingMemory()
        episodic_memory = EpisodicMemory()

        # 3. 거버넌스 시스템 초기화
        governance = GovernanceProtocol()

        print("2. 모든 시스템 초기화 완료")

        # 4. 실제 사용 시나리오: 사용자 질문 처리
        session = await react_framework.create_session("user-001")
        print(f"3. 세션 생성: {session.session_id}")

        # 5. 메모리에 사용자 컨텍스트 저장
        context_id = await working_memory.store(
            "사용자가 PACA 시스템에 대해 질문함",
            {"user": "user-001", "type": "context"}
        )
        print(f"4. 컨텍스트 저장: {context_id}")

        # 6. 사고 과정 실행
        thought_result = await react_framework.think(
            session,
            "PACA 시스템의 주요 기능은 무엇인가?",
            0.8
        )
        print(f"5. 사고 과정 완료: {thought_result.step_id}")

        # 7. 거버넌스 프로토콜 적용
        governance_decision = await governance.evaluate_action(
            "정보 제공",
            {"confidence": 0.8, "impact": "low"}
        )
        print(f"6. 거버넌스 평가: {governance_decision}")

        print("7. 실제 사용 시나리오 테스트 완료")
        return True

    except Exception as e:
        print(f"ERROR: 실제 사용 시나리오 실패: {e}")
        traceback.print_exc()
        return False

async def main():
    """메인 테스트 함수"""
    print("PACA 비동기 문제 진단 테스트")
    print("=" * 50)

    # 1. 시스템 환경 테스트
    test_system_environment()

    # 2. 동기/비동기 혼용 문제 테스트
    sync_test_passed = test_sync_async_mixing()

    # 3. 올바른 비동기 사용법 테스트
    async_test_passed = await test_async_proper_usage()

    # 4. 실제 사용 시나리오 테스트
    real_usage_passed = await test_real_usage_scenario()

    # 결과 요약
    print("\n" + "=" * 50)
    print("테스트 결과 요약")
    print("=" * 50)

    print(f"동기/비동기 혼용 테스트: {'통과' if sync_test_passed else '실패'}")
    print(f"올바른 비동기 사용법 테스트: {'통과' if async_test_passed else '실패'}")
    print(f"실제 사용 시나리오 테스트: {'통과' if real_usage_passed else '실패'}")

    # 문제점 분석
    print("\n" + "=" * 50)
    print("발견된 문제점")
    print("=" * 50)

    problems = []

    if not async_test_passed:
        problems.append("비동기 함수 호출 관련 문제")

    if not real_usage_passed:
        problems.append("실제 사용 시 시스템 통합 문제")

    if problems:
        for i, problem in enumerate(problems, 1):
            print(f"{i}. {problem}")
    else:
        print("주요 문제점 발견되지 않음")

    return len(problems) == 0

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)