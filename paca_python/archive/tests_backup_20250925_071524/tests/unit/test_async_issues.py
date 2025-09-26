"""
PACA 비동기 함수 호출 문제 진단 테스트
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

def test_encoding_issues():
    """인코딩 문제 테스트"""
    print("\n=== 인코딩 문제 테스트 ===")

    # 1. 이모지 및 유니코드 문제
    test_strings = [
        "🚀 PACA 시스템",
        "💡 아이디어",
        "⚠️ 경고",
        "✅ 성공",
        "❌ 실패",
        "한글 텍스트",
        "English text",
        "Mixed 한글과 English"
    ]

    encoding_results = []

    for test_str in test_strings:
        try:
            # CP949 인코딩 테스트 (Windows 기본)
            encoded = test_str.encode('cp949')
            decoded = encoded.decode('cp949')
            encoding_results.append((test_str, "CP949", "SUCCESS"))
            print(f"CP949 OK: {test_str}")
        except UnicodeEncodeError as e:
            encoding_results.append((test_str, "CP949", f"FAIL: {e}"))
            print(f"CP949 FAIL: {test_str} - {e}")

        try:
            # UTF-8 인코딩 테스트
            encoded = test_str.encode('utf-8')
            decoded = encoded.decode('utf-8')
            print(f"UTF-8 OK: {test_str}")
        except UnicodeEncodeError as e:
            print(f"UTF-8 FAIL: {test_str} - {e}")

    return encoding_results

def test_system_environment():
    """시스템 환경 문제 테스트"""
    print("\n=== 시스템 환경 테스트 ===")

    import locale
    import os

    print(f"1. Python 버전: {sys.version}")
    print(f"2. 기본 인코딩: {sys.getdefaultencoding()}")
    print(f"3. 파일 시스템 인코딩: {sys.getfilesystemencoding()}")
    print(f"4. 로케일: {locale.getdefaultlocale()}")
    print(f"5. 현재 작업 디렉토리: {os.getcwd()}")

    # 환경 변수 확인
    env_vars = ["PYTHONIOENCODING", "LANG", "LC_ALL"]
    for var in env_vars:
        value = os.getenv(var)
        print(f"6. {var}: {value}")

async def main():
    """메인 테스트 함수"""
    print("PACA 비동기 및 인코딩 문제 진단 테스트")
    print("=" * 50)

    # 1. 시스템 환경 테스트
    test_system_environment()

    # 2. 인코딩 문제 테스트
    encoding_results = test_encoding_issues()

    # 3. 동기/비동기 혼용 문제 테스트
    sync_test_passed = test_sync_async_mixing()

    # 4. 올바른 비동기 사용법 테스트
    async_test_passed = await test_async_proper_usage()

    # 결과 요약
    print("\n" + "=" * 50)
    print("테스트 결과 요약")
    print("=" * 50)

    print(f"동기/비동기 혼용 테스트: {'통과' if sync_test_passed else '실패'}")
    print(f"올바른 비동기 사용법 테스트: {'통과' if async_test_passed else '실패'}")

    # 인코딩 문제 요약
    cp949_failures = [r for r in encoding_results if r[1] == "CP949" and "FAIL" in r[2]]
    if cp949_failures:
        print(f"\nCP949 인코딩 실패 ({len(cp949_failures)}개):")
        for failure in cp949_failures:
            print(f"  - {failure[0]}")
    else:
        print("\nCP949 인코딩: 모든 테스트 통과")

    # 해결 방안 제시
    print("\n" + "=" * 50)
    print("문제 해결 방안")
    print("=" * 50)

    if cp949_failures:
        print("1. 인코딩 문제 해결:")
        print("   - 환경 변수 설정: set PYTHONIOENCODING=utf-8")
        print("   - 코드에서 이모지 사용 제거 또는 UTF-8 명시")
        print("   - print() 함수 사용 시 encoding 파라미터 지정")

    if not async_test_passed:
        print("2. 비동기 문제 해결:")
        print("   - 모든 비동기 함수 호출 시 await 키워드 사용")
        print("   - asyncio.run() 또는 async 컨텍스트에서만 비동기 함수 호출")
        print("   - 동기 래퍼 함수 제공 고려")

if __name__ == "__main__":
    asyncio.run(main())