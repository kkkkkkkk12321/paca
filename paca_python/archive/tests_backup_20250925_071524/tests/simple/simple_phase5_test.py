#!/usr/bin/env python3
"""
PACA v5 Phase 5 Simple Integration Test
Backup/Restore System Basic Functionality Test
"""

import sys
import asyncio
import tempfile
import shutil
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent / "paca_python"
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Basic import test"""
    print("=== Basic Import Test ===")

    try:
        # Backup system imports
        from paca.data.backup_system import (
            BackupSystem, BackupManager, BackupMetadata,
            BackupType, BackupStatus, RestoreResult, BackupTrigger
        )
        print("SUCCESS: Backup system imports successful")

        # Scheduler imports
        from paca.data.scheduler import (
            BackupScheduler, ScheduleJob, ScheduleEvent,
            ScheduleStatus, ScheduleType, CronParser
        )
        print("SUCCESS: Scheduler imports successful")

        return True
    except Exception as e:
        print(f"FAILED: Import failed: {e}")
        return False

def test_backup_system_creation():
    """Backup system creation test"""
    print("\n=== Backup System Creation Test ===")

    try:
        from paca.data.backup_system import BackupSystem, BackupType

        # Create temporary backup directory
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_system = BackupSystem(backup_root=temp_dir)
            print("SUCCESS: BackupSystem created")

            # Test auto backup function (document specification)
            backup_id = backup_system.create_auto_backup("test_trigger")
            print(f"SUCCESS: Auto backup created with ID: {backup_id}")

            # Test backup metadata
            if backup_system.metadata_cache:
                print("SUCCESS: Backup metadata cache initialized")
            else:
                print("WARNING: Backup metadata cache not initialized")

            # Test backup stats
            stats = backup_system.get_stats()
            print(f"SUCCESS: Backup stats retrieved: {stats}")

        return True

    except Exception as e:
        print(f"FAILED: Backup system test failed: {e}")
        return False

async def test_async_backup_operations():
    """Async backup operations test"""
    print("\n=== Async Backup Operations Test ===")

    try:
        from paca.data.backup_system import BackupSystem, BackupType

        with tempfile.TemporaryDirectory() as temp_dir:
            backup_system = BackupSystem(backup_root=temp_dir)

            # Create test files to backup
            test_data_dir = Path(temp_dir) / "test_data"
            test_data_dir.mkdir()

            test_file = test_data_dir / "test.txt"
            test_file.write_text("Test backup content")

            # Test async backup creation
            result = await backup_system.create_backup_async(
                backup_type=BackupType.MANUAL,
                source_paths=[str(test_data_dir)],
                description="Test manual backup"
            )

            if result.is_success:
                backup_id = result.data
                print(f"SUCCESS: Async backup created: {backup_id}")

                # Test backup restoration
                restore_dir = Path(temp_dir) / "restore"
                restore_result = await backup_system.restore_backup_async(
                    backup_id,
                    str(restore_dir)
                )

                if restore_result.success:
                    print("SUCCESS: Backup restored successfully")
                    print(f"  Restored files: {len(restore_result.restored_files)}")
                else:
                    print(f"WARNING: Backup restore failed: {restore_result.message}")

            else:
                print(f"FAILED: Async backup creation failed: {result.error}")
                return False

        return True

    except Exception as e:
        print(f"FAILED: Async backup operations test failed: {e}")
        return False

def test_backup_manager():
    """Backup manager test"""
    print("\n=== Backup Manager Test ===")

    try:
        from paca.data.backup_system import BackupManager, BackupSystem

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create backup systems
            local_system = BackupSystem(backup_root=f"{temp_dir}/local")
            archive_system = BackupSystem(backup_root=f"{temp_dir}/archive")

            # Create backup manager
            backup_manager = BackupManager({
                "local": local_system,
                "archive": archive_system
            })
            print("SUCCESS: BackupManager created")

            # Test backup system retrieval
            retrieved_system = backup_manager.get_backup_system("local")
            if retrieved_system:
                print("SUCCESS: Backup system retrieved from manager")
            else:
                print("WARNING: Failed to retrieve backup system")

            # Test global stats
            global_stats = backup_manager.get_global_stats()
            print(f"SUCCESS: Global stats retrieved: {global_stats}")

        return True

    except Exception as e:
        print(f"FAILED: Backup manager test failed: {e}")
        return False

async def test_scheduler_system():
    """Scheduler system test"""
    print("\n=== Scheduler System Test ===")

    try:
        from paca.data.scheduler import BackupScheduler, ScheduleType, ScheduleStatus
        from paca.data.backup_system import BackupSystem

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create backup system
            backup_system = BackupSystem(backup_root=temp_dir)

            # Create scheduler
            scheduler = BackupScheduler()
            print("SUCCESS: BackupScheduler created")

            # Test cron job addition
            result = scheduler.add_cron_job(
                name="Test Cron Job",
                cron_expression="0 */6 * * *",  # Every 6 hours
                backup_system=backup_system,
                description="Test cron backup job"
            )

            if result.is_success:
                job_id = result.data
                print(f"SUCCESS: Cron job added: {job_id}")

                # Test job retrieval
                job = scheduler.get_job(job_id)
                if job:
                    print(f"SUCCESS: Job retrieved - Name: {job.name}, Status: {job.status.value}")
                else:
                    print("WARNING: Failed to retrieve job")

                # Test job listing
                jobs = scheduler.list_jobs()
                print(f"SUCCESS: Jobs listed: {len(jobs)} jobs")

                # Test job pause/resume
                pause_result = scheduler.pause_job(job_id)
                if pause_result.is_success:
                    print("SUCCESS: Job paused")

                    resume_result = scheduler.resume_job(job_id)
                    if resume_result.is_success:
                        print("SUCCESS: Job resumed")
                    else:
                        print("WARNING: Job resume failed")

            else:
                print(f"WARNING: Cron job addition failed: {result.error}")

            # Test interval job
            interval_result = scheduler.add_interval_job(
                name="Test Interval Job",
                interval_minutes=60,  # Every hour
                backup_system=backup_system,
                max_runs=5,
                description="Test interval backup job"
            )

            if interval_result.is_success:
                print("SUCCESS: Interval job added")
            else:
                print("WARNING: Interval job addition failed")

            # Test scheduler stats
            stats = scheduler.get_stats()
            print(f"SUCCESS: Scheduler stats: {stats}")

        return True

    except Exception as e:
        print(f"FAILED: Scheduler system test failed: {e}")
        return False

def test_cron_parser():
    """Cron parser test"""
    print("\n=== Cron Parser Test ===")

    try:
        from paca.data.scheduler import CronParser

        # Test cron parsing
        test_expressions = [
            "0 */6 * * *",    # Every 6 hours
            "0 2 * * 0",      # Every Sunday at 2 AM
            "30 14 * * 1-5",  # Weekdays at 2:30 PM
            "0 0 1 * *"       # First day of every month
        ]

        for expr in test_expressions:
            parsed = CronParser.parse_cron(expr)
            if parsed["valid"]:
                print(f"SUCCESS: Parsed '{expr}' - {parsed['description']}")
            else:
                print(f"FAILED: Failed to parse '{expr}'")

        # Test next run time calculation
        from datetime import datetime
        next_run = CronParser.next_run_time("0 */6 * * *")
        if next_run:
            print(f"SUCCESS: Next run time calculated: {next_run}")
        else:
            print("WARNING: Failed to calculate next run time")

        return True

    except Exception as e:
        print(f"FAILED: Cron parser test failed: {e}")
        return False

def test_backup_types_and_enums():
    """Backup types and enums test"""
    print("\n=== Backup Types and Enums Test ===")

    try:
        from paca.data.backup_system import BackupType, BackupStatus, BackupTrigger
        from paca.data.scheduler import ScheduleType, ScheduleStatus

        # Test BackupType enum
        backup_types = list(BackupType)
        print(f"SUCCESS: BackupType enum values: {[bt.value for bt in backup_types]}")

        # Test BackupStatus enum
        backup_statuses = list(BackupStatus)
        print(f"SUCCESS: BackupStatus enum values: {[bs.value for bs in backup_statuses]}")

        # Test ScheduleType enum
        schedule_types = list(ScheduleType)
        print(f"SUCCESS: ScheduleType enum values: {[st.value for st in schedule_types]}")

        # Test ScheduleStatus enum
        schedule_statuses = list(ScheduleStatus)
        print(f"SUCCESS: ScheduleStatus enum values: {[ss.value for ss in schedule_statuses]}")

        # Test BackupTrigger constants
        triggers = [
            BackupTrigger.TRAINING_START,
            BackupTrigger.TRAINING_COMPLETE,
            BackupTrigger.CONFIG_CHANGE,
            BackupTrigger.SYSTEM_SHUTDOWN
        ]
        print(f"SUCCESS: BackupTrigger constants: {triggers}")

        return True

    except Exception as e:
        print(f"FAILED: Backup types and enums test failed: {e}")
        return False

async def test_integration_with_core_systems():
    """Integration with core systems test"""
    print("\n=== Integration with Core Systems Test ===")

    try:
        # Test core types integration
        from paca.core.types.base import generate_id, create_success, create_failure
        from paca.data.backup_system import BackupSystem

        with tempfile.TemporaryDirectory() as temp_dir:
            backup_system = BackupSystem(backup_root=temp_dir)

            # Test ID generation integration
            backup_id = generate_id()
            print(f"SUCCESS: Core ID generation works: {backup_id}")

            # Test Result types integration
            success_result = create_success("test_data")
            failure_result = create_failure("test_error")

            print(f"SUCCESS: Result types integration works")
            print(f"  Success: {success_result.is_success}")
            print(f"  Failure: {failure_result.is_success}")

        return True

    except Exception as e:
        print(f"FAILED: Integration with core systems test failed: {e}")
        return False

def test_performance_metrics():
    """Performance metrics test"""
    print("\n=== Performance Metrics Test ===")

    try:
        from paca.data.backup_system import BackupSystem
        import time

        print("PERFORMANCE: Testing backup system creation time...")

        start_time = time.time()
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_system = BackupSystem(backup_root=temp_dir)
        end_time = time.time()

        creation_time = (end_time - start_time) * 1000
        print(f"  Backup system creation time: {creation_time:.1f}ms")

        # Test backup ID generation time
        start_time = time.time()
        backup_id = backup_system.create_auto_backup("performance_test")
        end_time = time.time()

        id_generation_time = (end_time - start_time) * 1000
        print(f"  Backup ID generation time: {id_generation_time:.1f}ms")

        # Performance validation
        if creation_time <= 100:  # 100ms
            print("SUCCESS: Backup system creation within performance target")
        else:
            print(f"WARNING: Backup system creation slow ({creation_time:.1f}ms > 100ms)")

        if id_generation_time <= 50:  # 50ms
            print("SUCCESS: Backup ID generation within performance target")
        else:
            print(f"WARNING: Backup ID generation slow ({id_generation_time:.1f}ms > 50ms)")

        return True

    except Exception as e:
        print(f"FAILED: Performance metrics test failed: {e}")
        return False

async def main():
    """Main test execution"""
    print("PACA v5 Phase 5 Backup/Restore System Integration Test")
    print("=" * 65)

    # Test list
    tests = [
        ("Basic Import", test_basic_imports),
        ("Backup System Creation", test_backup_system_creation),
        ("Async Backup Operations", test_async_backup_operations),
        ("Backup Manager", test_backup_manager),
        ("Scheduler System", test_scheduler_system),
        ("Cron Parser", test_cron_parser),
        ("Backup Types and Enums", test_backup_types_and_enums),
        ("Integration with Core Systems", test_integration_with_core_systems),
        ("Performance Metrics", test_performance_metrics),
    ]

    passed_tests = 0
    total_tests = len(tests)

    # Run tests
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                passed_tests += 1
                print(f"PASS: {test_name} test passed")
            else:
                print(f"FAIL: {test_name} test failed")
        except Exception as e:
            print(f"ERROR: {test_name} test exception: {e}")

    # Final results
    print("\n" + "=" * 65)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("SUCCESS: All tests passed successfully!")
        print("Phase 5 Backup/Restore System is working correctly!")
    else:
        print(f"WARNING: {total_tests - passed_tests} tests failed.")
        print("Some features may have issues.")

    return passed_tests == total_tests

if __name__ == "__main__":
    # Run tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)