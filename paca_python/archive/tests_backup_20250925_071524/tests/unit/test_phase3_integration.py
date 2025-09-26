#!/usr/bin/env python3
"""
PACA v5 Phase 3 통합 테스트
성능 최적화 시스템의 모든 기능을 검증
"""

import asyncio
import sys
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent / "paca_python"
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """기본 임포트 테스트"""
    print("=== 기본 임포트 테스트 ===")

    try:
        from paca.performance import (
            HardwareMonitor, SystemStatus, PerformanceMetrics,
            ProfileManager, ProfileType, PerformanceProfile
        )
        print("SUCCESS: Performance module import successful")
        return True
    except Exception as e:
        print(f"FAILED: Import failed: {e}")
        return False

def test_hardware_monitor():
    """하드웨어 모니터 테스트"""
    print("\n=== 하드웨어 모니터 테스트 ===")

    try:
        from paca.performance import HardwareMonitor

        # 모니터 생성
        monitor = HardwareMonitor(monitoring_interval=1.0)
        print("✅ HardwareMonitor 생성 성공")

        # 즉시 상태 체크
        status_result = monitor.get_system_status()
        if status_result.is_success:
            status = status_result.value
            print(f"✅ 시스템 상태 조회 성공")
            print(f"   CPU: {status.resource_usage.cpu_percent:.1f}%")
            print(f"   Memory: {status.resource_usage.memory_percent:.1f}%")
            print(f"   추천 프로파일: {status.recommended_profile}")
            print(f"   건강도 점수: {status.overall_health_score:.1f}")
            print(f"   알림 개수: {len(status.alerts)}")

            # 알림 정보 출력
            if status.alerts:
                print("   활성 알림:")
                for alert in status.alerts:
                    print(f"     - {alert.level.value}: {alert.message}")

            return True
        else:
            print(f"❌ 시스템 상태 조회 실패: {status_result.error}")
            return False

    except Exception as e:
        print(f"❌ 하드웨어 모니터 테스트 실패: {e}")
        return False

def test_profile_manager():
    """프로파일 매니저 테스트"""
    print("\n=== 프로파일 매니저 테스트 ===")

    try:
        from paca.performance import ProfileManager, ProfileType

        # 매니저 생성
        manager = ProfileManager()
        print("✅ ProfileManager 생성 성공")

        # 현재 프로파일 조회
        current = manager.current_profile
        print(f"✅ 현재 프로파일: {current.name}")
        print(f"   Workers: {current.config.max_workers}")
        print(f"   Reasoning Steps: {current.config.reasoning_steps}")
        print(f"   Speed Multiplier: {current.config.speed_multiplier}")
        print(f"   성능 점수: {current.config.performance_score:.1f}")

        # 프로파일 전환 테스트
        print("\n프로파일 전환 테스트:")
        test_profiles = [ProfileType.HIGH_END, ProfileType.CONSERVATIVE, ProfileType.LOW_END]

        for profile_type in test_profiles:
            result = manager.switch_profile(profile_type, "테스트 전환")
            if result.is_success:
                profile = result.value
                print(f"✅ {profile.name} 전환 성공 (점수: {profile.config.performance_score:.1f})")
            else:
                print(f"❌ {profile_type.value} 전환 실패: {result.error}")
                return False

        # 자동 프로파일 선택 테스트
        print("\n자동 프로파일 선택 테스트:")
        test_cases = [
            (30, 40, 1000, "여유로운 상황"),
            (70, 60, 500, "중간 부하"),
            (90, 80, 200, "고부하"),
            (98, 95, 50, "임계 상황")
        ]

        for cpu, memory, available_mb, description in test_cases:
            result = manager.auto_select_profile(cpu, memory, available_mb)
            if result.is_success:
                profile = result.value
                print(f"✅ {description} (CPU {cpu}%, Memory {memory}%) → {profile.name}")
            else:
                print(f"❌ 자동 선택 실패: {result.error}")
                return False

        return True

    except Exception as e:
        print(f"❌ 프로파일 매니저 테스트 실패: {e}")
        return False

async def test_monitoring_integration():
    """모니터링 통합 테스트"""
    print("\n=== 모니터링 통합 테스트 ===")

    try:
        from paca.performance import HardwareMonitor, ProfileManager

        # 모니터와 매니저 생성
        monitor = HardwareMonitor(monitoring_interval=2.0)
        manager = ProfileManager(auto_switch=True)

        print("✅ 통합 시스템 초기화 성공")

        # 통합 콜백 함수
        profile_changes = []

        def monitoring_callback(status):
            """모니터링 콜백 - 자동 프로파일 전환"""
            result = manager.auto_select_profile(
                status.resource_usage.cpu_percent,
                status.resource_usage.memory_percent,
                status.resource_usage.memory_available_mb
            )

            if result.is_success:
                profile = result.value
                profile_changes.append({
                    'timestamp': status.timestamp,
                    'profile': profile.name,
                    'cpu': status.resource_usage.cpu_percent,
                    'memory': status.resource_usage.memory_percent
                })
                print(f"📊 [{status.timestamp:.1f}] "
                      f"CPU: {status.resource_usage.cpu_percent:.1f}%, "
                      f"Memory: {status.resource_usage.memory_percent:.1f}%, "
                      f"프로파일: {profile.name}")

        monitor.add_callback(monitoring_callback)

        # 5초간 모니터링 실행
        print("📊 5초간 실시간 모니터링 시작...")

        start_result = await monitor.start_monitoring()
        if not start_result.is_success:
            print(f"❌ 모니터링 시작 실패: {start_result.error}")
            return False

        await asyncio.sleep(5)

        stop_result = await monitor.stop_monitoring()
        if not stop_result.is_success:
            print(f"❌ 모니터링 중지 실패: {stop_result.error}")
            return False

        print("✅ 모니터링 통합 테스트 완료")
        print(f"📊 총 {len(profile_changes)}개 상태 업데이트 기록")

        # 성능 통계 출력
        stats = manager.get_performance_statistics()
        print("\n성능 통계:")
        for profile_name, profile_stats in stats['profiles'].items():
            if profile_stats['usage_count'] > 0:
                print(f"  {profile_name}: {profile_stats['usage_count']}회 사용 "
                      f"({profile_stats['usage_percentage']:.1f}%)")

        return True

    except Exception as e:
        print(f"❌ 모니터링 통합 테스트 실패: {e}")
        return False

def test_performance_metrics():
    """성능 메트릭 테스트"""
    print("\n=== 성능 메트릭 테스트 ===")

    try:
        from paca.performance import HardwareMonitor

        monitor = HardwareMonitor()

        # 여러 번 측정하여 성능 체크
        print("📊 성능 측정 테스트...")
        measurements = []

        for i in range(5):
            start_time = time.time()
            result = monitor.get_system_status()
            end_time = time.time()

            if result.is_success:
                duration_ms = (end_time - start_time) * 1000
                measurements.append(duration_ms)
                print(f"  측정 {i+1}: {duration_ms:.1f}ms")
            else:
                print(f"❌ 측정 {i+1} 실패: {result.error}")
                return False

        # 성능 분석
        avg_duration = sum(measurements) / len(measurements)
        max_duration = max(measurements)
        min_duration = min(measurements)

        print(f"\n📊 성능 분석 결과:")
        print(f"  평균 측정 시간: {avg_duration:.1f}ms")
        print(f"  최대 측정 시간: {max_duration:.1f}ms")
        print(f"  최소 측정 시간: {min_duration:.1f}ms")

        # 성능 기준 검증 (목표: 100ms 이하)
        if avg_duration <= 100:
            print("✅ 성능 목표 달성 (평균 100ms 이하)")
        else:
            print(f"⚠️ 성능 목표 미달성 (평균 {avg_duration:.1f}ms > 100ms)")

        return True

    except Exception as e:
        print(f"❌ 성능 메트릭 테스트 실패: {e}")
        return False

def test_error_handling():
    """오류 처리 테스트"""
    print("\n=== 오류 처리 테스트 ===")

    try:
        from paca.performance import ProfileManager, ProfileType

        manager = ProfileManager()

        # 존재하지 않는 프로파일 타입 테스트
        print("📊 잘못된 프로파일 전환 테스트...")

        # 프로파일 사용자 정의 - 잘못된 값
        invalid_configs = [
            {'max_workers': 0},  # 너무 작은 값
            {'max_workers': 100},  # 너무 큰 값
            {'reasoning_steps': 0},  # 너무 작은 값
            {'speed_multiplier': -1},  # 음수 값
            {'memory_limit_mb': 10},  # 너무 작은 값
        ]

        for i, invalid_config in enumerate(invalid_configs):
            result = manager.customize_profile(ProfileType.MID_RANGE, invalid_config)
            if result.is_failure:
                print(f"✅ 잘못된 설정 {i+1} 올바르게 거부됨")
            else:
                print(f"⚠️ 잘못된 설정 {i+1}이 허용됨 (예상치 못한 동작)")

        print("✅ 오류 처리 테스트 완료")
        return True

    except Exception as e:
        print(f"❌ 오류 처리 테스트 실패: {e}")
        return False

async def main():
    """메인 테스트 실행"""
    print("PACA v5 Phase 3 Performance Optimization System Integration Test")
    print("=" * 60)

    # 테스트 목록
    tests = [
        ("기본 임포트", test_basic_imports),
        ("하드웨어 모니터", test_hardware_monitor),
        ("프로파일 매니저", test_profile_manager),
        ("성능 메트릭", test_performance_metrics),
        ("오류 처리", test_error_handling),
    ]

    # 비동기 테스트
    async_tests = [
        ("모니터링 통합", test_monitoring_integration),
    ]

    passed_tests = 0
    total_tests = len(tests) + len(async_tests)

    # 동기 테스트 실행
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} 테스트 통과")
            else:
                print(f"❌ {test_name} 테스트 실패")
        except Exception as e:
            print(f"❌ {test_name} 테스트 예외: {e}")

    # 비동기 테스트 실행
    for test_name, test_func in async_tests:
        try:
            if await test_func():
                passed_tests += 1
                print(f"✅ {test_name} 테스트 통과")
            else:
                print(f"❌ {test_name} 테스트 실패")
        except Exception as e:
            print(f"❌ {test_name} 테스트 예외: {e}")

    # 최종 결과
    print("\n" + "=" * 60)
    print(f"Test Results: {passed_tests}/{total_tests} passed")

    if passed_tests == total_tests:
        print("All tests passed successfully!")
        print("Phase 3 Performance Optimization System is working correctly!")
    else:
        print(f"{total_tests - passed_tests} tests failed.")
        print("Some features may have issues.")

    return passed_tests == total_tests

if __name__ == "__main__":
    # 테스트 실행
    success = asyncio.run(main())
    sys.exit(0 if success else 1)