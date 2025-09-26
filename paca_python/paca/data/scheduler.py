#!/usr/bin/env python3
"""
PACA v5 Backup Scheduler System
백업 스케줄링 및 자동화 시스템
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import re

# 조건부 임포트: 패키지 실행시와 직접 실행시 모두 지원
try:
    from ..core.types.base import (
        ID, Timestamp, Result, current_timestamp, generate_id, create_success, create_failure
    )
    from .backup_system import BackupSystem, BackupType, BackupTrigger
except ImportError:
    from paca.core.types.base import (
        ID, Timestamp, Result, current_timestamp, generate_id, create_success, create_failure
    )
    from paca.data.backup_system import BackupSystem, BackupType, BackupTrigger


class ScheduleStatus(Enum):
    """스케줄 상태"""
    ACTIVE = "active"           # 활성
    PAUSED = "paused"          # 일시 정지
    DISABLED = "disabled"      # 비활성화
    ERROR = "error"           # 오류
    COMPLETED = "completed"   # 완료


class ScheduleType(Enum):
    """스케줄 유형"""
    CRON = "cron"             # 크론 스케줄
    INTERVAL = "interval"     # 간격 스케줄
    ONE_TIME = "one_time"     # 일회성 스케줄
    EVENT_BASED = "event_based"  # 이벤트 기반


@dataclass
class ScheduleJob:
    """스케줄 작업"""
    job_id: str
    name: str
    schedule_type: ScheduleType
    schedule_expression: str
    backup_system: BackupSystem
    backup_type: BackupType
    source_paths: List[str]
    status: ScheduleStatus
    created_at: datetime
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    description: Optional[str] = None
    tags: List[str] = None
    max_runs: Optional[int] = None  # 최대 실행 횟수
    timeout_minutes: int = 60       # 타임아웃 (분)

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class ScheduleEvent:
    """스케줄 이벤트"""
    event_id: str
    job_id: str
    event_type: str
    timestamp: datetime
    success: bool
    backup_id: Optional[str] = None
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CronParser:
    """크론 표현식 파서"""

    @staticmethod
    def parse_cron(cron_expression: str) -> Dict[str, Any]:
        """
        크론 표현식 파싱

        Args:
            cron_expression: 크론 표현식 (예: "0 */6 * * *")

        Returns:
            Dict[str, Any]: 파싱된 크론 정보
        """
        try:
            # 기본 크론 필드: 분 시 일 월 요일
            fields = cron_expression.strip().split()

            if len(fields) != 5:
                raise ValueError("Invalid cron expression: must have 5 fields")

            return {
                "minute": fields[0],
                "hour": fields[1],
                "day": fields[2],
                "month": fields[3],
                "day_of_week": fields[4],
                "valid": True,
                "description": CronParser._describe_cron(cron_expression)
            }

        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }

    @staticmethod
    def _describe_cron(cron_expression: str) -> str:
        """크론 표현식 설명"""
        try:
            # 간단한 크론 설명 생성
            fields = cron_expression.strip().split()
            minute, hour, day, month, day_of_week = fields

            descriptions = []

            # 분 처리
            if minute == "*":
                descriptions.append("every minute")
            elif minute == "0":
                descriptions.append("at the top of the hour")
            elif "/" in minute:
                interval = minute.split("/")[1]
                descriptions.append(f"every {interval} minutes")
            else:
                descriptions.append(f"at minute {minute}")

            # 시간 처리
            if hour != "*":
                if "/" in hour:
                    interval = hour.split("/")[1]
                    descriptions.append(f"every {interval} hours")
                else:
                    descriptions.append(f"at {hour}:00")

            # 일 처리
            if day != "*":
                descriptions.append(f"on day {day}")

            # 월 처리
            if month != "*":
                descriptions.append(f"in month {month}")

            # 요일 처리
            if day_of_week != "*":
                days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
                if day_of_week.isdigit():
                    day_name = days[int(day_of_week)]
                    descriptions.append(f"on {day_name}")

            return ", ".join(descriptions)

        except Exception:
            return "Custom schedule"

    @staticmethod
    def next_run_time(cron_expression: str, from_time: datetime = None) -> Optional[datetime]:
        """
        다음 실행 시간 계산

        Args:
            cron_expression: 크론 표현식
            from_time: 기준 시간 (None이면 현재 시간)

        Returns:
            Optional[datetime]: 다음 실행 시간
        """
        try:
            if from_time is None:
                from_time = datetime.now()

            parsed = CronParser.parse_cron(cron_expression)
            if not parsed["valid"]:
                return None

            # 간단한 다음 실행 시간 계산 (실제로는 더 복잡한 로직 필요)
            next_time = from_time + timedelta(minutes=1)

            # 크론 필드에 따른 계산 (간소화된 버전)
            minute = parsed["minute"]
            hour = parsed["hour"]

            if hour == "*/6":  # 6시간마다
                hours_until_next = 6 - (from_time.hour % 6)
                next_time = from_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=hours_until_next)

            elif hour.isdigit() and minute.isdigit():  # 특정 시간
                target_hour = int(hour)
                target_minute = int(minute)

                next_time = from_time.replace(
                    hour=target_hour,
                    minute=target_minute,
                    second=0,
                    microsecond=0
                )

                # 시간이 이미 지났으면 다음 날
                if next_time <= from_time:
                    next_time += timedelta(days=1)

            return next_time

        except Exception:
            return None


class BackupScheduler:
    """백업 스케줄러"""

    def __init__(self):
        """스케줄러 초기화"""
        self.jobs: Dict[str, ScheduleJob] = {}
        self.events: List[ScheduleEvent] = []
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.scheduler_task: Optional[asyncio.Task] = None
        self.running = False
        self.check_interval = 30  # 스케줄 체크 간격 (초)

        # 통계
        self.stats = {
            "total_jobs": 0,
            "active_jobs": 0,
            "completed_runs": 0,
            "failed_runs": 0,
            "total_runtime": 0.0
        }

    async def start(self) -> Result[str]:
        """스케줄러 시작"""
        try:
            if self.running:
                return create_failure("Scheduler is already running")

            self.running = True
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())

            return create_success("Scheduler started successfully")

        except Exception as e:
            return create_failure(f"Failed to start scheduler: {str(e)}")

    async def stop(self) -> Result[str]:
        """스케줄러 중지"""
        try:
            self.running = False

            # 스케줄러 태스크 중지
            if self.scheduler_task:
                self.scheduler_task.cancel()
                try:
                    await self.scheduler_task
                except asyncio.CancelledError:
                    pass

            # 실행 중인 작업들 중지
            for task in self.running_tasks.values():
                task.cancel()

            # 모든 작업 완료 대기
            if self.running_tasks:
                await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)

            self.running_tasks.clear()

            return create_success("Scheduler stopped successfully")

        except Exception as e:
            return create_failure(f"Failed to stop scheduler: {str(e)}")

    def add_cron_job(self,
                    name: str,
                    cron_expression: str,
                    backup_system: BackupSystem,
                    backup_type: BackupType = BackupType.SCHEDULED,
                    source_paths: List[str] = None,
                    description: str = None,
                    tags: List[str] = None) -> Result[str]:
        """
        크론 작업 추가

        Args:
            name: 작업 이름
            cron_expression: 크론 표현식
            backup_system: 백업 시스템
            backup_type: 백업 유형
            source_paths: 소스 경로 목록
            description: 설명
            tags: 태그 목록

        Returns:
            Result[str]: 작업 ID
        """
        try:
            # 크론 표현식 검증
            parsed = CronParser.parse_cron(cron_expression)
            if not parsed["valid"]:
                return create_failure(f"Invalid cron expression: {parsed.get('error', 'Unknown error')}")

            job_id = generate_id()

            # 다음 실행 시간 계산
            next_run = CronParser.next_run_time(cron_expression)

            # 작업 생성
            job = ScheduleJob(
                job_id=job_id,
                name=name,
                schedule_type=ScheduleType.CRON,
                schedule_expression=cron_expression,
                backup_system=backup_system,
                backup_type=backup_type,
                source_paths=source_paths or [],
                status=ScheduleStatus.ACTIVE,
                created_at=datetime.now(),
                next_run=next_run,
                description=description,
                tags=tags or []
            )

            # 작업 등록
            self.jobs[job_id] = job
            self.stats["total_jobs"] += 1
            self.stats["active_jobs"] += 1

            return create_success(job_id)

        except Exception as e:
            return create_failure(f"Failed to add cron job: {str(e)}")

    def add_interval_job(self,
                        name: str,
                        interval_minutes: int,
                        backup_system: BackupSystem,
                        backup_type: BackupType = BackupType.SCHEDULED,
                        source_paths: List[str] = None,
                        max_runs: int = None,
                        description: str = None,
                        tags: List[str] = None) -> Result[str]:
        """
        간격 작업 추가

        Args:
            name: 작업 이름
            interval_minutes: 실행 간격 (분)
            backup_system: 백업 시스템
            backup_type: 백업 유형
            source_paths: 소스 경로 목록
            max_runs: 최대 실행 횟수
            description: 설명
            tags: 태그 목록

        Returns:
            Result[str]: 작업 ID
        """
        try:
            job_id = generate_id()

            # 다음 실행 시간 계산
            next_run = datetime.now() + timedelta(minutes=interval_minutes)

            # 작업 생성
            job = ScheduleJob(
                job_id=job_id,
                name=name,
                schedule_type=ScheduleType.INTERVAL,
                schedule_expression=f"every {interval_minutes} minutes",
                backup_system=backup_system,
                backup_type=backup_type,
                source_paths=source_paths or [],
                status=ScheduleStatus.ACTIVE,
                created_at=datetime.now(),
                next_run=next_run,
                max_runs=max_runs,
                description=description,
                tags=tags or []
            )

            # 작업 등록
            self.jobs[job_id] = job
            self.stats["total_jobs"] += 1
            self.stats["active_jobs"] += 1

            return create_success(job_id)

        except Exception as e:
            return create_failure(f"Failed to add interval job: {str(e)}")

    def pause_job(self, job_id: str) -> Result[str]:
        """작업 일시 정지"""
        if job_id not in self.jobs:
            return create_failure(f"Job {job_id} not found")

        job = self.jobs[job_id]
        if job.status == ScheduleStatus.ACTIVE:
            job.status = ScheduleStatus.PAUSED
            self.stats["active_jobs"] -= 1
            return create_success(f"Job {job_id} paused")
        else:
            return create_failure(f"Job {job_id} is not active")

    def resume_job(self, job_id: str) -> Result[str]:
        """작업 재개"""
        if job_id not in self.jobs:
            return create_failure(f"Job {job_id} not found")

        job = self.jobs[job_id]
        if job.status == ScheduleStatus.PAUSED:
            job.status = ScheduleStatus.ACTIVE
            self.stats["active_jobs"] += 1

            # 다음 실행 시간 재계산
            if job.schedule_type == ScheduleType.CRON:
                job.next_run = CronParser.next_run_time(job.schedule_expression)
            elif job.schedule_type == ScheduleType.INTERVAL:
                # 간격에서 분 추출
                minutes = int(job.schedule_expression.split()[1])
                job.next_run = datetime.now() + timedelta(minutes=minutes)

            return create_success(f"Job {job_id} resumed")
        else:
            return create_failure(f"Job {job_id} is not paused")

    def delete_job(self, job_id: str) -> Result[str]:
        """작업 삭제"""
        if job_id not in self.jobs:
            return create_failure(f"Job {job_id} not found")

        job = self.jobs[job_id]

        # 실행 중인 태스크가 있으면 취소
        if job_id in self.running_tasks:
            self.running_tasks[job_id].cancel()
            del self.running_tasks[job_id]

        # 통계 업데이트
        if job.status == ScheduleStatus.ACTIVE:
            self.stats["active_jobs"] -= 1

        # 작업 삭제
        del self.jobs[job_id]

        return create_success(f"Job {job_id} deleted")

    def get_job(self, job_id: str) -> Optional[ScheduleJob]:
        """작업 조회"""
        return self.jobs.get(job_id)

    def list_jobs(self, status: ScheduleStatus = None) -> List[ScheduleJob]:
        """작업 목록 조회"""
        jobs = list(self.jobs.values())

        if status:
            jobs = [job for job in jobs if job.status == status]

        return sorted(jobs, key=lambda x: x.created_at, reverse=True)

    def get_events(self, job_id: str = None, limit: int = 100) -> List[ScheduleEvent]:
        """이벤트 목록 조회"""
        events = self.events

        if job_id:
            events = [event for event in events if event.job_id == job_id]

        # 최신순 정렬
        events.sort(key=lambda x: x.timestamp, reverse=True)

        return events[:limit]

    async def _scheduler_loop(self):
        """스케줄러 메인 루프"""
        while self.running:
            try:
                await self._check_and_run_jobs()
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Scheduler loop error: {e}")
                await asyncio.sleep(self.check_interval)

    async def _check_and_run_jobs(self):
        """작업 체크 및 실행"""
        now = datetime.now()

        for job in self.jobs.values():
            # 활성 상태이고 실행 시간이 된 작업 확인
            if (job.status == ScheduleStatus.ACTIVE and
                job.next_run and
                job.next_run <= now and
                job.job_id not in self.running_tasks):

                # 최대 실행 횟수 체크
                if job.max_runs and job.run_count >= job.max_runs:
                    job.status = ScheduleStatus.COMPLETED
                    self.stats["active_jobs"] -= 1
                    continue

                # 비동기 작업 실행
                task = asyncio.create_task(self._run_job(job))
                self.running_tasks[job.job_id] = task

    async def _run_job(self, job: ScheduleJob):
        """작업 실행"""
        start_time = datetime.now()
        event_id = generate_id()

        try:
            # 백업 실행
            result = await job.backup_system.create_backup_async(
                backup_type=job.backup_type,
                source_paths=job.source_paths,
                description=f"Scheduled backup: {job.name}",
                tags=job.tags + ["scheduled"]
            )

            # 실행 결과 처리
            duration = (datetime.now() - start_time).total_seconds()
            success = result.is_success()

            # 작업 통계 업데이트
            job.run_count += 1
            job.last_run = start_time

            if success:
                job.success_count += 1
                self.stats["completed_runs"] += 1
                backup_id = result.data if result.is_success() else None
            else:
                job.failure_count += 1
                self.stats["failed_runs"] += 1
                backup_id = None

            self.stats["total_runtime"] += duration

            # 다음 실행 시간 계산
            if job.schedule_type == ScheduleType.CRON:
                job.next_run = CronParser.next_run_time(job.schedule_expression, start_time)
            elif job.schedule_type == ScheduleType.INTERVAL:
                # 간격에서 분 추출
                minutes = int(job.schedule_expression.split()[1])
                job.next_run = start_time + timedelta(minutes=minutes)

            # 이벤트 기록
            event = ScheduleEvent(
                event_id=event_id,
                job_id=job.job_id,
                event_type="backup_execution",
                timestamp=start_time,
                success=success,
                backup_id=backup_id,
                duration_seconds=duration,
                error_message=None if success else result.error,
                metadata={"run_count": job.run_count}
            )

            self.events.append(event)

            # 이벤트 목록 크기 제한 (최대 1000개)
            if len(self.events) > 1000:
                self.events = self.events[-1000:]

        except Exception as e:
            # 오류 처리
            duration = (datetime.now() - start_time).total_seconds()

            job.run_count += 1
            job.failure_count += 1
            job.last_run = start_time
            job.status = ScheduleStatus.ERROR

            self.stats["failed_runs"] += 1
            self.stats["total_runtime"] += duration

            # 오류 이벤트 기록
            event = ScheduleEvent(
                event_id=event_id,
                job_id=job.job_id,
                event_type="backup_execution",
                timestamp=start_time,
                success=False,
                duration_seconds=duration,
                error_message=str(e),
                metadata={"run_count": job.run_count}
            )

            self.events.append(event)

        finally:
            # 실행 중인 태스크에서 제거
            if job.job_id in self.running_tasks:
                del self.running_tasks[job.job_id]

    def get_stats(self) -> Dict[str, Any]:
        """스케줄러 통계"""
        return {
            **self.stats,
            "running_jobs": len(self.running_tasks),
            "uptime_seconds": (datetime.now() - self._start_time).total_seconds() if hasattr(self, '_start_time') else 0
        }


# 편의 함수들
def create_daily_backup_job(name: str, hour: int, minute: int, backup_system: BackupSystem) -> str:
    """일일 백업 작업 생성"""
    cron_expression = f"{minute} {hour} * * *"
    scheduler = BackupScheduler()
    result = scheduler.add_cron_job(name, cron_expression, backup_system)
    return result.data if result.is_success() else None


def create_hourly_backup_job(name: str, backup_system: BackupSystem) -> str:
    """시간별 백업 작업 생성"""
    cron_expression = "0 * * * *"
    scheduler = BackupScheduler()
    result = scheduler.add_cron_job(name, cron_expression, backup_system)
    return result.data if result.is_success() else None


# 테스트 함수
async def test_scheduler():
    """스케줄러 테스트"""
    print("=== Backup Scheduler Test ===")

    # 가상 백업 시스템 생성
    from .backup_system import BackupSystem
    backup_system = BackupSystem("test_backups")

    # 스케줄러 생성
    scheduler = BackupScheduler()

    # 스케줄러 시작
    await scheduler.start()

    # 간격 작업 추가 (1분마다, 최대 3회)
    result = scheduler.add_interval_job(
        name="Test Interval Job",
        interval_minutes=1,
        backup_system=backup_system,
        max_runs=3,
        description="Test interval backup job"
    )

    if result.is_success():
        job_id = result.data
        print(f"Interval job created: {job_id}")

        # 잠시 대기
        await asyncio.sleep(5)

        # 작업 상태 확인
        job = scheduler.get_job(job_id)
        print(f"Job status: {job.status.value}")
        print(f"Run count: {job.run_count}")

    # 스케줄러 중지
    await scheduler.stop()

    return True


if __name__ == "__main__":
    # 직접 실행 테스트
    asyncio.run(test_scheduler())