#!/usr/bin/env python3
"""
PACA v5 Phase 3 Simple Integration Test
Performance Optimization System Basic Functionality Test
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent / "paca_python"
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Basic import test"""
    print("=== Basic Import Test ===")

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
    """Hardware monitor test"""
    print("\n=== Hardware Monitor Test ===")

    try:
        from paca.performance import HardwareMonitor

        # Create monitor
        monitor = HardwareMonitor(monitoring_interval=1.0)
        print("SUCCESS: HardwareMonitor created")

        # Get system status immediately
        status_result = monitor.get_system_status()
        if status_result.is_success:
            status = status_result.value
            print("SUCCESS: System status retrieved")
            print(f"   CPU: {status.resource_usage.cpu_percent:.1f}%")
            print(f"   Memory: {status.resource_usage.memory_percent:.1f}%")
            print(f"   Recommended Profile: {status.recommended_profile}")
            print(f"   Health Score: {status.overall_health_score:.1f}")
            print(f"   Alerts: {len(status.alerts)}")

            # Print alert information
            if status.alerts:
                print("   Active Alerts:")
                for alert in status.alerts:
                    print(f"     - {alert.level.value}: {alert.message[:50]}...")

            return True
        else:
            print(f"FAILED: System status query failed: {status_result.error}")
            return False

    except Exception as e:
        print(f"FAILED: Hardware monitor test failed: {e}")
        return False

def test_profile_manager():
    """Profile manager test"""
    print("\n=== Profile Manager Test ===")

    try:
        from paca.performance import ProfileManager, ProfileType

        # Create manager
        manager = ProfileManager()
        print("SUCCESS: ProfileManager created")

        # Get current profile
        current = manager.current_profile
        print(f"SUCCESS: Current profile: {current.name}")
        print(f"   Workers: {current.config.max_workers}")
        print(f"   Reasoning Steps: {current.config.reasoning_steps}")
        print(f"   Speed Multiplier: {current.config.speed_multiplier}")
        print(f"   Performance Score: {current.config.performance_score:.1f}")

        # Profile switching test
        print("\nProfile Switching Test:")
        test_profiles = [ProfileType.HIGH_END, ProfileType.CONSERVATIVE, ProfileType.LOW_END]

        for profile_type in test_profiles:
            result = manager.switch_profile(profile_type, "Test switch")
            if result.is_success:
                profile = result.value
                print(f"SUCCESS: {profile.name} switch (score: {profile.config.performance_score:.1f})")
            else:
                print(f"FAILED: {profile_type.value} switch failed: {result.error}")
                return False

        # Auto profile selection test
        print("\nAuto Profile Selection Test:")
        test_cases = [
            (30, 40, 1000, "Low usage"),
            (70, 60, 500, "Medium load"),
            (90, 80, 200, "High load"),
            (98, 95, 50, "Critical")
        ]

        for cpu, memory, available_mb, description in test_cases:
            result = manager.auto_select_profile(cpu, memory, available_mb)
            if result.is_success:
                profile = result.value
                print(f"SUCCESS: {description} (CPU {cpu}%, Memory {memory}%) -> {profile.name}")
            else:
                print(f"FAILED: Auto selection failed: {result.error}")
                return False

        return True

    except Exception as e:
        print(f"FAILED: Profile manager test failed: {e}")
        return False

async def test_monitoring_integration():
    """Monitoring integration test"""
    print("\n=== Monitoring Integration Test ===")

    try:
        from paca.performance import HardwareMonitor, ProfileManager

        # Create monitor and manager
        monitor = HardwareMonitor(monitoring_interval=2.0)
        manager = ProfileManager(auto_switch=True)

        print("SUCCESS: Integrated system initialized")

        # Integration callback function
        profile_changes = []

        def monitoring_callback(status):
            """Monitoring callback - auto profile switching"""
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
                print(f"MONITOR [{status.timestamp:.1f}] "
                      f"CPU: {status.resource_usage.cpu_percent:.1f}%, "
                      f"Memory: {status.resource_usage.memory_percent:.1f}%, "
                      f"Profile: {profile.name}")

        monitor.add_callback(monitoring_callback)

        # Run monitoring for 5 seconds
        print("MONITOR: Starting 5-second real-time monitoring...")

        start_result = await monitor.start_monitoring()
        if not start_result.is_success:
            print(f"FAILED: Monitoring start failed: {start_result.error}")
            return False

        await asyncio.sleep(5)

        stop_result = await monitor.stop_monitoring()
        if not stop_result.is_success:
            print(f"FAILED: Monitoring stop failed: {stop_result.error}")
            return False

        print("SUCCESS: Monitoring integration test completed")
        print(f"MONITOR: Total {len(profile_changes)} status updates recorded")

        # Print performance statistics
        stats = manager.get_performance_statistics()
        print("\nPerformance Statistics:")
        for profile_name, profile_stats in stats['profiles'].items():
            if profile_stats['usage_count'] > 0:
                print(f"  {profile_name}: {profile_stats['usage_count']} uses "
                      f"({profile_stats['usage_percentage']:.1f}%)")

        return True

    except Exception as e:
        print(f"FAILED: Monitoring integration test failed: {e}")
        return False

def test_performance_metrics():
    """Performance metrics test"""
    print("\n=== Performance Metrics Test ===")

    try:
        from paca.performance import HardwareMonitor

        monitor = HardwareMonitor()

        # Multiple measurements for performance check
        print("PERFORMANCE: Measuring system status query speed...")
        measurements = []

        for i in range(5):
            start_time = time.time()
            result = monitor.get_system_status()
            end_time = time.time()

            if result.is_success:
                duration_ms = (end_time - start_time) * 1000
                measurements.append(duration_ms)
                print(f"  Measurement {i+1}: {duration_ms:.1f}ms")
            else:
                print(f"FAILED: Measurement {i+1} failed: {result.error}")
                return False

        # Performance analysis
        avg_duration = sum(measurements) / len(measurements)
        max_duration = max(measurements)
        min_duration = min(measurements)

        print(f"\nPERFORMANCE Analysis Results:")
        print(f"  Average query time: {avg_duration:.1f}ms")
        print(f"  Maximum query time: {max_duration:.1f}ms")
        print(f"  Minimum query time: {min_duration:.1f}ms")

        # Performance target validation (target: under 100ms)
        if avg_duration <= 100:
            print("SUCCESS: Performance target achieved (average under 100ms)")
        else:
            print(f"WARNING: Performance target not met (average {avg_duration:.1f}ms > 100ms)")

        return True

    except Exception as e:
        print(f"FAILED: Performance metrics test failed: {e}")
        return False

async def main():
    """Main test execution"""
    print("PACA v5 Phase 3 Performance Optimization System Integration Test")
    print("=" * 70)

    # Test list
    tests = [
        ("Basic Import", test_basic_imports),
        ("Hardware Monitor", test_hardware_monitor),
        ("Profile Manager", test_profile_manager),
        ("Performance Metrics", test_performance_metrics),
    ]

    # Async tests
    async_tests = [
        ("Monitoring Integration", test_monitoring_integration),
    ]

    passed_tests = 0
    total_tests = len(tests) + len(async_tests)

    # Run synchronous tests
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
                print(f"PASS: {test_name} test passed")
            else:
                print(f"FAIL: {test_name} test failed")
        except Exception as e:
            print(f"ERROR: {test_name} test exception: {e}")

    # Run asynchronous tests
    for test_name, test_func in async_tests:
        try:
            if await test_func():
                passed_tests += 1
                print(f"PASS: {test_name} test passed")
            else:
                print(f"FAIL: {test_name} test failed")
        except Exception as e:
            print(f"ERROR: {test_name} test exception: {e}")

    # Final results
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("SUCCESS: All tests passed successfully!")
        print("Phase 3 Performance Optimization System is working correctly!")
    else:
        print(f"WARNING: {total_tests - passed_tests} tests failed.")
        print("Some features may have issues.")

    return passed_tests == total_tests

if __name__ == "__main__":
    # Run tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)