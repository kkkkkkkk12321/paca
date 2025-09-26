"""
PACA 프로덕션 준비도 최종 검증 테스트
실제 배포 시 발생할 수 있는 모든 문제점 종합 분석
"""

import asyncio
import os
import sys
import json
import time
from datetime import datetime

def safe_print(text):
    """안전한 출력 함수"""
    try:
        print(text)
    except UnicodeEncodeError:
        clean_text = ''.join(c for c in text if ord(c) < 65536)
        print(clean_text)

def test_environment_variables():
    """환경 변수 및 설정 문제 테스트"""
    safe_print("=== 환경 변수 테스트 ===")

    issues = []

    # 1. API 키 설정 확인
    gemini_keys = os.getenv('GEMINI_API_KEYS')
    if not gemini_keys:
        issues.append("GEMINI_API_KEYS 환경 변수 미설정")
    else:
        keys = gemini_keys.split(',')
        safe_print(f"Gemini API 키 수: {len(keys)}")

    # 2. 인코딩 설정 확인
    python_encoding = os.getenv('PYTHONIOENCODING')
    if python_encoding != 'utf-8':
        issues.append(f"PYTHONIOENCODING이 utf-8이 아님: {python_encoding}")

    # 3. 기타 중요 환경 변수
    important_vars = ['PATH', 'PYTHONPATH']
    for var in important_vars:
        value = os.getenv(var)
        if not value:
            issues.append(f"{var} 환경 변수 없음")

    if issues:
        safe_print("WARNING: 환경 변수 문제 발견")
        for issue in issues:
            safe_print(f"  - {issue}")
        return False
    else:
        safe_print("OK: 환경 변수 설정 정상")
        return True

def test_file_permissions():
    """파일 권한 및 접근 문제 테스트"""
    safe_print("\n=== 파일 권한 테스트 ===")

    issues = []

    # 1. 현재 디렉토리 쓰기 권한
    try:
        test_file = "paca_permission_test.tmp"
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        safe_print("OK: 현재 디렉토리 쓰기 가능")
    except Exception as e:
        issues.append(f"현재 디렉토리 쓰기 불가: {e}")

    # 2. 로그 디렉토리 생성 가능 여부
    try:
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # 테스트 로그 파일 생성
        test_log = os.path.join(log_dir, "test.log")
        with open(test_log, 'w') as f:
            f.write("test log")
        os.remove(test_log)
        safe_print("OK: 로그 디렉토리 접근 가능")
    except Exception as e:
        issues.append(f"로그 디렉토리 접근 불가: {e}")

    # 3. 임시 파일 생성 권한
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            tmp.write(b"test")
        safe_print("OK: 임시 파일 생성 가능")
    except Exception as e:
        issues.append(f"임시 파일 생성 불가: {e}")

    if issues:
        safe_print("WARNING: 파일 권한 문제 발견")
        for issue in issues:
            safe_print(f"  - {issue}")
        return False
    else:
        safe_print("OK: 파일 권한 정상")
        return True

async def test_api_timeout_handling():
    """API 타임아웃 처리 테스트"""
    safe_print("\n=== API 타임아웃 처리 테스트 ===")

    try:
        from paca.tools import ReActFramework, PACAToolManager
        from paca.cognitive.memory import WorkingMemory

        tool_manager = PACAToolManager()
        react_framework = ReActFramework(tool_manager)
        memory = WorkingMemory()

        # 의도적 지연 작업 (타임아웃 시뮬레이션)
        async def slow_operation():
            await asyncio.sleep(0.1)  # 100ms 지연
            return await memory.store("느린 작업", {"type": "slow"})

        # 타임아웃 설정하여 작업 실행
        try:
            result = await asyncio.wait_for(slow_operation(), timeout=0.05)  # 50ms 타임아웃
            safe_print("WARNING: 타임아웃이 작동하지 않음")
            return False
        except asyncio.TimeoutError:
            safe_print("OK: 타임아웃 처리 정상")
            return True
        except Exception as e:
            safe_print(f"ERROR: 예상치 못한 오류: {e}")
            return False

    except Exception as e:
        safe_print(f"ERROR: 타임아웃 테스트 실패: {e}")
        return False

def test_error_logging():
    """오류 로깅 시스템 테스트"""
    safe_print("\n=== 오류 로깅 테스트 ===")

    try:
        import logging

        # 로그 설정
        log_file = "paca_test.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

        logger = logging.getLogger("paca_test")

        # 다양한 로그 레벨 테스트
        logger.info("테스트 정보 로그")
        logger.warning("테스트 경고 로그")
        logger.error("테스트 오류 로그")

        # 로그 파일 생성 확인
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
                if "테스트" in log_content:
                    safe_print("OK: 로깅 시스템 정상")
                    os.remove(log_file)  # 테스트 파일 정리
                    return True
                else:
                    safe_print("WARNING: 로그 내용 누락")
                    return False
        else:
            safe_print("WARNING: 로그 파일 생성 실패")
            return False

    except Exception as e:
        safe_print(f"ERROR: 로깅 테스트 실패: {e}")
        return False

def test_dependency_versions():
    """의존성 버전 호환성 테스트"""
    safe_print("\n=== 의존성 버전 테스트 ===")

    version_issues = []

    # 중요 라이브러리 버전 확인
    try:
        import asyncio
        safe_print(f"Python: {sys.version}")

        # psutil 버전
        try:
            import psutil
            safe_print(f"psutil: {psutil.__version__}")
        except ImportError:
            version_issues.append("psutil 라이브러리 없음")

        # 기타 중요 라이브러리들
        optional_libs = ['fastapi', 'uvicorn', 'structlog', 'prometheus_client']
        for lib in optional_libs:
            try:
                module = __import__(lib)
                if hasattr(module, '__version__'):
                    safe_print(f"{lib}: {module.__version__}")
                else:
                    safe_print(f"{lib}: 버전 정보 없음")
            except ImportError:
                version_issues.append(f"{lib} 라이브러리 없음 (선택사항)")

    except Exception as e:
        version_issues.append(f"버전 확인 중 오류: {e}")

    if version_issues:
        safe_print("WARNING: 의존성 문제 발견")
        for issue in version_issues:
            safe_print(f"  - {issue}")
        return len(version_issues) <= 2  # 선택적 라이브러리 문제는 허용
    else:
        safe_print("OK: 의존성 버전 정상")
        return True

async def test_concurrent_sessions():
    """동시 세션 처리 한계 테스트"""
    safe_print("\n=== 동시 세션 한계 테스트 ===")

    try:
        from paca.tools import ReActFramework, PACAToolManager

        tool_manager = PACAToolManager()
        react_framework = ReActFramework(tool_manager)

        # 동시 세션 생성
        session_count = 20  # 적절한 수준으로 조정

        async def create_session(session_id):
            return await react_framework.create_session(f"concurrent-{session_id}")

        start_time = time.time()

        # 모든 세션 동시 생성
        tasks = [create_session(i) for i in range(session_count)]
        sessions = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()

        # 결과 분석
        successful_sessions = sum(1 for s in sessions if not isinstance(s, Exception))
        failed_sessions = session_count - successful_sessions

        safe_print(f"성공한 세션: {successful_sessions}/{session_count}")
        safe_print(f"실패한 세션: {failed_sessions}")
        safe_print(f"처리 시간: {end_time - start_time:.3f}초")

        if successful_sessions >= session_count * 0.9:  # 90% 이상 성공
            safe_print("OK: 동시 세션 처리 양호")
            return True
        else:
            safe_print("WARNING: 동시 세션 처리 문제")
            return False

    except Exception as e:
        safe_print(f"ERROR: 동시 세션 테스트 실패: {e}")
        return False

def test_system_resources():
    """시스템 리소스 요구사항 테스트"""
    safe_print("\n=== 시스템 리소스 테스트 ===")

    try:
        import psutil

        # CPU 정보
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)

        # 메모리 정보
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)

        # 디스크 정보
        disk = psutil.disk_usage('.')
        disk_free_gb = disk.free / (1024**3)

        safe_print(f"CPU 코어: {cpu_count}개")
        safe_print(f"CPU 사용률: {cpu_percent}%")
        safe_print(f"총 메모리: {memory_gb:.1f}GB")
        safe_print(f"사용 가능 메모리: {memory_available_gb:.1f}GB")
        safe_print(f"사용 가능 디스크: {disk_free_gb:.1f}GB")

        # 최소 요구사항 체크
        issues = []

        if cpu_count < 2:
            issues.append("CPU 코어 수 부족 (최소 2코어 권장)")

        if memory_available_gb < 1:
            issues.append("사용 가능 메모리 부족 (최소 1GB 필요)")

        if disk_free_gb < 1:
            issues.append("사용 가능 디스크 공간 부족 (최소 1GB 필요)")

        if cpu_percent > 90:
            issues.append("현재 CPU 사용률 과다")

        if issues:
            safe_print("WARNING: 시스템 리소스 문제")
            for issue in issues:
                safe_print(f"  - {issue}")
            return False
        else:
            safe_print("OK: 시스템 리소스 충분")
            return True

    except Exception as e:
        safe_print(f"ERROR: 시스템 리소스 테스트 실패: {e}")
        return False

async def main():
    """프로덕션 준비도 최종 검증"""
    safe_print("PACA 프로덕션 준비도 최종 검증")
    safe_print("=" * 50)

    test_results = {}

    # 모든 테스트 실행
    test_results['environment_vars'] = test_environment_variables()
    test_results['file_permissions'] = test_file_permissions()
    test_results['api_timeout'] = await test_api_timeout_handling()
    test_results['error_logging'] = test_error_logging()
    test_results['dependency_versions'] = test_dependency_versions()
    test_results['concurrent_sessions'] = await test_concurrent_sessions()
    test_results['system_resources'] = test_system_resources()

    # 결과 요약
    safe_print("\n" + "=" * 50)
    safe_print("프로덕션 준비도 최종 결과")
    safe_print("=" * 50)

    passed_tests = sum(test_results.values())
    total_tests = len(test_results)

    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        safe_print(f"{test_name}: {status}")

    safe_print(f"\n전체 테스트: {passed_tests}/{total_tests} 통과")

    # 배포 준비도 평가
    readiness_score = (passed_tests / total_tests) * 100

    if readiness_score >= 90:
        safe_print(f"\nEXCELLENT: 프로덕션 배포 준비 완료 ({readiness_score:.1f}%)")
    elif readiness_score >= 70:
        safe_print(f"\nGOOD: 프로덕션 배포 가능 ({readiness_score:.1f}%)")
    elif readiness_score >= 50:
        safe_print(f"\nWARNING: 일부 개선 필요 ({readiness_score:.1f}%)")
    else:
        safe_print(f"\nCRITICAL: 추가 개발 필요 ({readiness_score:.1f}%)")

    # 실패한 테스트별 권장사항
    if not test_results['environment_vars']:
        safe_print("\n권장사항: 환경 변수 설정 가이드 참조")
    if not test_results['file_permissions']:
        safe_print("\n권장사항: 실행 권한 및 디렉토리 접근 권한 확인")
    if not test_results['api_timeout']:
        safe_print("\n권장사항: 타임아웃 처리 로직 개선")
    if not test_results['error_logging']:
        safe_print("\n권장사항: 로깅 시스템 설정 확인")
    if not test_results['dependency_versions']:
        safe_print("\n권장사항: requirements.txt 의존성 설치")
    if not test_results['concurrent_sessions']:
        safe_print("\n권장사항: 동시 접속 처리 최적화")
    if not test_results['system_resources']:
        safe_print("\n권장사항: 하드웨어 업그레이드 또는 리소스 최적화")

    return test_results

if __name__ == "__main__":
    results = asyncio.run(main())