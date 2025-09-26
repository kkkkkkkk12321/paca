"""
PACA 실사용 환경 심층 문제점 분석 (인코딩 안전 버전)
"""

import asyncio
import sys
import time
import traceback
import psutil
import gc
from datetime import datetime

def safe_print(text):
    """안전한 출력 함수"""
    try:
        print(text)
    except UnicodeEncodeError:
        # 이모지 제거 후 출력
        clean_text = ''.join(c for c in text if ord(c) < 65536)
        print(clean_text)

def test_memory_leaks():
    """메모리 누수 테스트"""
    safe_print("=== 메모리 누수 테스트 ===")

    try:
        from paca.tools import ReActFramework, PACAToolManager
        from paca.cognitive.memory import WorkingMemory

        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        safe_print(f"초기 메모리: {initial_memory:.2f} MB")

        memory_usage = []

        for i in range(5):  # 반복 횟수 줄임
            tool_manager = PACAToolManager()
            react_framework = ReActFramework(tool_manager)
            working_memory = WorkingMemory()

            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage.append(current_memory)
            safe_print(f"반복 {i+1}: {current_memory:.2f} MB")

            del tool_manager, react_framework, working_memory
            gc.collect()
            time.sleep(0.1)

        memory_increase = memory_usage[-1] - memory_usage[0]
        safe_print(f"메모리 증가: {memory_increase:.2f} MB")

        if memory_increase > 10:  # 임계값 낮춤
            safe_print("WARNING: 메모리 누수 의심")
            return False
        else:
            safe_print("OK: 메모리 사용량 정상")
            return True

    except Exception as e:
        safe_print(f"ERROR: 메모리 테스트 실패: {e}")
        return False

async def test_nested_async_calls():
    """중첩된 비동기 호출 문제 테스트"""
    safe_print("\n=== 중첩 비동기 호출 테스트 ===")

    try:
        from paca.tools import ReActFramework, PACAToolManager
        from paca.cognitive.memory import WorkingMemory

        tool_manager = PACAToolManager()
        react_framework = ReActFramework(tool_manager)
        memory = WorkingMemory()

        # 중첩된 비동기 작업
        async def nested_operation():
            # 세션 생성
            session = await react_framework.create_session("nested-test")

            # 메모리 작업
            content_id = await memory.store("중첩 테스트", {"type": "nested"})
            retrieved = await memory.retrieve(content_id)

            return session, retrieved

        # 여러 중첩 작업 동시 실행
        tasks = [nested_operation() for _ in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if not isinstance(r, Exception))
        safe_print(f"중첩 작업 성공: {success_count}/3")

        if success_count == 3:
            safe_print("OK: 중첩 비동기 호출 정상")
            return True
        else:
            safe_print("WARNING: 중첩 비동기 호출 문제")
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    safe_print(f"  Task {i}: {result}")
            return False

    except Exception as e:
        safe_print(f"ERROR: 중첩 비동기 테스트 실패: {e}")
        traceback.print_exc()
        return False

async def test_session_management():
    """세션 관리 문제 테스트"""
    safe_print("\n=== 세션 관리 테스트 ===")

    try:
        from paca.tools import ReActFramework, PACAToolManager

        tool_manager = PACAToolManager()
        react_framework = ReActFramework(tool_manager)

        # 다수 세션 생성 및 관리
        sessions = []
        for i in range(10):
            session = await react_framework.create_session(f"session-{i}")
            sessions.append(session)

        safe_print(f"생성된 세션 수: {len(sessions)}")

        # 세션 정보 확인
        session_issues = []
        for i, session in enumerate(sessions):
            # 세션 ID 존재 여부 확인
            if not hasattr(session, 'id') and not hasattr(session, 'session_id'):
                session_issues.append(f"세션 {i}: ID 속성 없음")

            # 세션 상태 확인
            if not hasattr(session, 'status'):
                session_issues.append(f"세션 {i}: 상태 속성 없음")

        if session_issues:
            safe_print("WARNING: 세션 관리 문제 발견")
            for issue in session_issues:
                safe_print(f"  {issue}")
            return False
        else:
            safe_print("OK: 세션 관리 정상")
            return True

    except Exception as e:
        safe_print(f"ERROR: 세션 관리 테스트 실패: {e}")
        traceback.print_exc()
        return False

async def test_tool_registration_consistency():
    """도구 등록 일관성 테스트"""
    safe_print("\n=== 도구 등록 일관성 테스트 ===")

    try:
        from paca.tools import PACAToolManager
        from paca.tools.tools.web_search import WebSearchTool

        tool_manager = PACAToolManager()

        # 동일한 도구 여러 번 등록
        web_search1 = WebSearchTool()
        web_search2 = WebSearchTool()

        result1 = tool_manager.register_tool(web_search1)
        result2 = tool_manager.register_tool(web_search2)  # 같은 이름으로 재등록

        safe_print(f"첫 번째 등록: {result1}")
        safe_print(f"두 번째 등록: {result2}")

        # 등록된 도구 수 확인
        tool_count = len(tool_manager.tools)
        safe_print(f"등록된 도구 수: {tool_count}")

        # 도구 정보 확인
        if web_search1.name in tool_manager.tools:
            registered_tool = tool_manager.tools[web_search1.name]
            safe_print(f"등록된 도구 타입: {type(registered_tool)}")

            # 마지막 등록된 도구가 저장되었는지 확인
            if registered_tool is web_search2:
                safe_print("OK: 도구 덮어쓰기 정상 작동")
                return True
            elif registered_tool is web_search1:
                safe_print("WARNING: 도구 덮어쓰기 미작동")
                return False
            else:
                safe_print("WARNING: 예상치 못한 도구 객체")
                return False
        else:
            safe_print("ERROR: 도구 등록 실패")
            return False

    except Exception as e:
        safe_print(f"ERROR: 도구 등록 테스트 실패: {e}")
        traceback.print_exc()
        return False

async def test_memory_search_performance():
    """메모리 검색 성능 테스트"""
    safe_print("\n=== 메모리 검색 성능 테스트 ===")

    try:
        from paca.cognitive.memory import WorkingMemory
        from paca.cognitive.memory.types import SearchQuery

        memory = WorkingMemory()

        # 테스트 데이터 저장
        safe_print("테스트 데이터 저장 중...")
        start_time = time.time()

        for i in range(50):  # 데이터 수 줄임
            await memory.store(f"테스트 데이터 {i}", {"index": i, "type": "test"})

        storage_time = time.time() - start_time
        safe_print(f"저장 시간: {storage_time:.3f}초")

        # 검색 성능 테스트
        search_times = []
        for i in range(10):
            start_time = time.time()

            search_query = SearchQuery(query="테스트", limit=10)
            results = await memory.search(search_query)

            end_time = time.time()
            search_times.append(end_time - start_time)

        avg_search_time = sum(search_times) / len(search_times)
        max_search_time = max(search_times)

        safe_print(f"평균 검색 시간: {avg_search_time:.3f}초")
        safe_print(f"최대 검색 시간: {max_search_time:.3f}초")

        # 성능 기준 체크
        if avg_search_time > 0.1:  # 100ms 이상시 문제
            safe_print("WARNING: 검색 성능 저하")
            return False
        else:
            safe_print("OK: 검색 성능 양호")
            return True

    except Exception as e:
        safe_print(f"ERROR: 검색 성능 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_import_dependencies():
    """의존성 임포트 문제 테스트"""
    safe_print("\n=== 의존성 임포트 테스트 ===")

    critical_modules = [
        "paca.tools",
        "paca.cognitive.memory",
        "paca.governance",  # 이 경로가 문제일 수 있음
        "paca.monitoring",
        "paca.feedback"
    ]

    import_results = {}

    for module in critical_modules:
        try:
            __import__(module)
            import_results[module] = True
            safe_print(f"OK: {module}")
        except ImportError as e:
            import_results[module] = False
            safe_print(f"ERROR: {module} - {e}")
        except Exception as e:
            import_results[module] = False
            safe_print(f"ERROR: {module} - Unexpected error: {e}")

    success_count = sum(import_results.values())
    total_count = len(import_results)

    safe_print(f"임포트 성공: {success_count}/{total_count}")

    if success_count == total_count:
        safe_print("OK: 모든 모듈 임포트 성공")
        return True
    else:
        safe_print("WARNING: 일부 모듈 임포트 실패")
        return False

async def main():
    """메인 테스트 함수"""
    safe_print("PACA 실사용 환경 심층 문제점 분석")
    safe_print("=" * 50)

    test_results = {}

    # 모든 테스트 실행
    test_results['memory_leaks'] = test_memory_leaks()
    test_results['import_dependencies'] = test_import_dependencies()
    test_results['nested_async'] = await test_nested_async_calls()
    test_results['session_management'] = await test_session_management()
    test_results['tool_registration'] = await test_tool_registration_consistency()
    test_results['memory_search'] = await test_memory_search_performance()

    # 결과 요약
    safe_print("\n" + "=" * 50)
    safe_print("심층 분석 결과 요약")
    safe_print("=" * 50)

    passed_tests = sum(test_results.values())
    total_tests = len(test_results)

    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        safe_print(f"{test_name}: {status}")

    safe_print(f"\n전체 테스트: {passed_tests}/{total_tests} 통과")

    # 추가 문제점 식별
    additional_issues = []

    if not test_results['import_dependencies']:
        additional_issues.append("모듈 경로 및 의존성 문제")

    if not test_results['nested_async']:
        additional_issues.append("중첩 비동기 호출 문제")

    if not test_results['session_management']:
        additional_issues.append("세션 관리 API 불일치")

    if not test_results['tool_registration']:
        additional_issues.append("도구 등록 일관성 문제")

    if not test_results['memory_search']:
        additional_issues.append("메모리 검색 성능 문제")

    if additional_issues:
        safe_print(f"\nCRITICAL 추가 문제점:")
        for issue in additional_issues:
            safe_print(f"- {issue}")
    else:
        safe_print("\nOK: 추가 문제점 없음")

    return test_results

if __name__ == "__main__":
    results = asyncio.run(main())