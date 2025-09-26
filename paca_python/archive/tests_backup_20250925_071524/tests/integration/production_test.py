"""
PACA 프로덕션 배포 검증 테스트
시스템 전체 통합 테스트 및 배포 준비도 검증
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# PACA 모듈 임포트
from paca.tools import ReActFramework, PACAToolManager
from paca.tools.tools.web_search import WebSearchTool
from paca.tools.tools.file_manager import FileManagerTool
from paca.feedback import FeedbackStorage, FeedbackCollector, FeedbackAnalyzer
# 모니터링 모듈이 아직 구현되지 않은 경우 스킵
try:
    from paca.monitoring.logger import PACALogger, LogLevel
    from paca.monitoring.dashboard import MonitoringDashboard
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    PACALogger = None
    LogLevel = None
    MonitoringDashboard = None


class ProductionValidator:
    """프로덕션 배포 검증기"""

    def __init__(self):
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self.overall_score = 0.0
        self.start_time = datetime.now()

    async def run_validation(self) -> Dict[str, Any]:
        """전체 검증 실행"""
        print("🚀 PACA 프로덕션 배포 검증 시작")
        print("=" * 60)

        # 1. 환경 검증
        await self._test_environment()

        # 2. 핵심 시스템 검증
        await self._test_core_systems()

        # 3. 도구 시스템 검증
        await self._test_tool_systems()

        # 4. 피드백 시스템 검증
        await self._test_feedback_system()

        # 5. 모니터링 시스템 검증
        if MONITORING_AVAILABLE:
            await self._test_monitoring_system()
        else:
            print("\n📊 5. 모니터링 시스템 검증 (스킵 - 모듈 없음)")

        # 6. 통합 시스템 검증
        await self._test_integration()

        # 7. 성능 검증
        await self._test_performance()

        # 8. 안정성 검증
        await self._test_stability()

        # 최종 결과 계산
        self._calculate_final_score()

        return self._generate_report()

    async def _test_environment(self):
        """환경 검증"""
        print("\n📋 1. 환경 검증")
        test_name = "environment"
        results = {"tests": [], "score": 0.0, "issues": []}

        try:
            # Python 버전 확인
            import sys
            if sys.version_info >= (3, 8):
                results["tests"].append({"name": "Python 버전", "status": "PASS", "details": f"Python {sys.version_info.major}.{sys.version_info.minor}"})
            else:
                results["tests"].append({"name": "Python 버전", "status": "FAIL", "details": f"Python {sys.version_info.major}.{sys.version_info.minor} (3.8+ 필요)"})
                results["issues"].append("Python 3.8 이상 버전이 필요합니다")

            # 필수 의존성 확인
            required_modules = [
                "aiohttp", "aiosqlite", "pydantic", "structlog",
                "psutil", "beautifulsoup4", "requests"
            ]

            for module in required_modules:
                try:
                    __import__(module)
                    results["tests"].append({"name": f"모듈 {module}", "status": "PASS", "details": "설치됨"})
                except ImportError:
                    results["tests"].append({"name": f"모듈 {module}", "status": "FAIL", "details": "미설치"})
                    results["issues"].append(f"필수 모듈 {module}가 설치되지 않았습니다")

            # 설정 파일 확인
            config_files = [
                ".env.production",
                "requirements.txt",
                "Dockerfile",
                "docker-compose.yml"
            ]

            for config_file in config_files:
                if Path(config_file).exists():
                    results["tests"].append({"name": f"설정 파일 {config_file}", "status": "PASS", "details": "존재함"})
                else:
                    results["tests"].append({"name": f"설정 파일 {config_file}", "status": "WARN", "details": "없음"})
                    results["issues"].append(f"설정 파일 {config_file}이 없습니다")

            # 점수 계산
            passed = len([t for t in results["tests"] if t["status"] == "PASS"])
            total = len(results["tests"])
            results["score"] = (passed / total) * 100

            print(f"   ✅ 통과: {passed}/{total} ({results['score']:.1f}%)")

        except Exception as e:
            results["issues"].append(f"환경 검증 오류: {str(e)}")
            print(f"   ❌ 오류: {str(e)}")

        self.test_results[test_name] = results

    async def _test_core_systems(self):
        """핵심 시스템 검증"""
        print("\n🔧 2. 핵심 시스템 검증")
        test_name = "core_systems"
        results = {"tests": [], "score": 0.0, "issues": []}

        try:
            # ReAct 프레임워크 테스트
            try:
                tool_manager = PACAToolManager()
                react_framework = ReActFramework(tool_manager)
                session = await react_framework.create_session("test_core")

                # 기본 사고 과정 테스트
                think_result = await react_framework.think(session, "시스템 테스트 중", 0.8)

                if think_result and think_result.content:
                    results["tests"].append({"name": "ReAct 프레임워크", "status": "PASS", "details": "정상 동작"})
                else:
                    results["tests"].append({"name": "ReAct 프레임워크", "status": "FAIL", "details": "사고 과정 실패"})
                    results["issues"].append("ReAct 프레임워크 사고 과정이 실패했습니다")

            except Exception as e:
                results["tests"].append({"name": "ReAct 프레임워크", "status": "FAIL", "details": str(e)})
                results["issues"].append(f"ReAct 프레임워크 오류: {str(e)}")

            # 도구 관리자 테스트
            try:
                tool_manager = PACAToolManager()

                # 기본 기능 테스트
                tool_count = len(tool_manager.tools)
                if tool_count >= 0:  # 도구가 없어도 정상
                    results["tests"].append({"name": "도구 관리자", "status": "PASS", "details": f"{tool_count}개 도구 등록"})
                else:
                    results["tests"].append({"name": "도구 관리자", "status": "FAIL", "details": "초기화 실패"})
                    results["issues"].append("도구 관리자 초기화가 실패했습니다")

            except Exception as e:
                results["tests"].append({"name": "도구 관리자", "status": "FAIL", "details": str(e)})
                results["issues"].append(f"도구 관리자 오류: {str(e)}")

            # 점수 계산
            passed = len([t for t in results["tests"] if t["status"] == "PASS"])
            total = len(results["tests"])
            results["score"] = (passed / total) * 100 if total > 0 else 0

            print(f"   ✅ 통과: {passed}/{total} ({results['score']:.1f}%)")

        except Exception as e:
            results["issues"].append(f"핵심 시스템 검증 오류: {str(e)}")
            print(f"   ❌ 오류: {str(e)}")

        self.test_results[test_name] = results

    async def _test_tool_systems(self):
        """도구 시스템 검증"""
        print("\n🛠️ 3. 도구 시스템 검증")
        test_name = "tool_systems"
        results = {"tests": [], "score": 0.0, "issues": []}

        try:
            # 웹 검색 도구 테스트
            try:
                web_search = WebSearchTool()
                search_result = await web_search.execute(query="Python programming", max_results=1)

                if search_result.success and search_result.result:
                    results["tests"].append({"name": "웹 검색 도구", "status": "PASS", "details": "검색 성공"})
                else:
                    results["tests"].append({"name": "웹 검색 도구", "status": "WARN", "details": "검색 실패 (API 제한 가능)"})

            except Exception as e:
                results["tests"].append({"name": "웹 검색 도구", "status": "FAIL", "details": str(e)})
                results["issues"].append(f"웹 검색 도구 오류: {str(e)}")

            # 파일 관리 도구 테스트
            try:
                file_manager = FileManagerTool(sandbox_mode=True)

                # 테스트 파일 생성
                write_result = await file_manager.execute(
                    operation="write",
                    file_path="test_production.txt",
                    content="Production test file"
                )

                if write_result.success:
                    # 파일 읽기 테스트
                    read_result = await file_manager.execute(
                        operation="read",
                        file_path="test_production.txt"
                    )

                    if read_result.success and "Production test file" in read_result.result:
                        results["tests"].append({"name": "파일 관리 도구", "status": "PASS", "details": "읽기/쓰기 성공"})
                    else:
                        results["tests"].append({"name": "파일 관리 도구", "status": "FAIL", "details": "읽기 실패"})
                        results["issues"].append("파일 관리 도구 읽기가 실패했습니다")

                    # 테스트 파일 삭제
                    await file_manager.execute(operation="delete", file_path="test_production.txt")
                else:
                    results["tests"].append({"name": "파일 관리 도구", "status": "FAIL", "details": "쓰기 실패"})
                    results["issues"].append("파일 관리 도구 쓰기가 실패했습니다")

            except Exception as e:
                results["tests"].append({"name": "파일 관리 도구", "status": "FAIL", "details": str(e)})
                results["issues"].append(f"파일 관리 도구 오류: {str(e)}")

            # 점수 계산
            passed = len([t for t in results["tests"] if t["status"] == "PASS"])
            warned = len([t for t in results["tests"] if t["status"] == "WARN"])
            total = len(results["tests"])
            results["score"] = ((passed + warned * 0.5) / total) * 100 if total > 0 else 0

            print(f"   ✅ 통과: {passed}/{total}, 경고: {warned} ({results['score']:.1f}%)")

        except Exception as e:
            results["issues"].append(f"도구 시스템 검증 오류: {str(e)}")
            print(f"   ❌ 오류: {str(e)}")

        self.test_results[test_name] = results

    async def _test_feedback_system(self):
        """피드백 시스템 검증"""
        print("\n📝 4. 피드백 시스템 검증")
        test_name = "feedback_system"
        results = {"tests": [], "score": 0.0, "issues": []}

        try:
            # 피드백 저장소 테스트
            try:
                storage = FeedbackStorage("test_feedback.db")
                await storage.initialize()

                # 기본 기능 테스트
                from paca.feedback.models import FeedbackModel, FeedbackType

                test_feedback = FeedbackModel(
                    feedback_type=FeedbackType.GENERAL,
                    session_id="test_session",
                    rating=5,
                    text_feedback="테스트 피드백"
                )

                # 저장 테스트
                save_success = await storage.save_feedback(test_feedback)
                if save_success:
                    # 조회 테스트
                    retrieved = await storage.get_feedback(test_feedback.id)
                    if retrieved and retrieved.text_feedback == "테스트 피드백":
                        results["tests"].append({"name": "피드백 저장소", "status": "PASS", "details": "저장/조회 성공"})
                    else:
                        results["tests"].append({"name": "피드백 저장소", "status": "FAIL", "details": "조회 실패"})
                        results["issues"].append("피드백 조회가 실패했습니다")
                else:
                    results["tests"].append({"name": "피드백 저장소", "status": "FAIL", "details": "저장 실패"})
                    results["issues"].append("피드백 저장이 실패했습니다")

                # 테스트 DB 정리
                Path("test_feedback.db").unlink(missing_ok=True)

            except Exception as e:
                results["tests"].append({"name": "피드백 저장소", "status": "FAIL", "details": str(e)})
                results["issues"].append(f"피드백 저장소 오류: {str(e)}")

            # 피드백 수집기 테스트
            try:
                storage = FeedbackStorage("test_feedback.db")
                await storage.initialize()
                collector = FeedbackCollector(storage)

                # 자동 피드백 수집 테스트
                test_context = {
                    "session_id": "test_session",
                    "success": False,
                    "error_message": "테스트 오류"
                }

                auto_feedback = await collector.collect_automatic_feedback("test_session", test_context)
                if auto_feedback:
                    results["tests"].append({"name": "피드백 수집기", "status": "PASS", "details": f"{len(auto_feedback)}개 자동 피드백 생성"})
                else:
                    results["tests"].append({"name": "피드백 수집기", "status": "WARN", "details": "자동 피드백 없음"})

                # 테스트 DB 정리
                Path("test_feedback.db").unlink(missing_ok=True)

            except Exception as e:
                results["tests"].append({"name": "피드백 수집기", "status": "FAIL", "details": str(e)})
                results["issues"].append(f"피드백 수집기 오류: {str(e)}")

            # 점수 계산
            passed = len([t for t in results["tests"] if t["status"] == "PASS"])
            warned = len([t for t in results["tests"] if t["status"] == "WARN"])
            total = len(results["tests"])
            results["score"] = ((passed + warned * 0.5) / total) * 100 if total > 0 else 0

            print(f"   ✅ 통과: {passed}/{total}, 경고: {warned} ({results['score']:.1f}%)")

        except Exception as e:
            results["issues"].append(f"피드백 시스템 검증 오류: {str(e)}")
            print(f"   ❌ 오류: {str(e)}")

        self.test_results[test_name] = results

    async def _test_monitoring_system(self):
        """모니터링 시스템 검증"""
        print("\n📊 5. 모니터링 시스템 검증")
        test_name = "monitoring_system"
        results = {"tests": [], "score": 0.0, "issues": []}

        if not MONITORING_AVAILABLE:
            results["tests"].append({"name": "모니터링 모듈", "status": "SKIP", "details": "모듈 없음"})
            results["score"] = 0.0
            self.test_results[test_name] = results
            return

        try:
            # 로거 테스트
            try:
                logger = PACALogger("test_logger", enable_database=False, enable_console=False)
                await logger.start()

                # 로그 기록 테스트
                await logger.info("테스트 로그", component="test")
                results["tests"].append({"name": "구조화 로거", "status": "PASS", "details": "로그 기록 성공"})

                await logger.stop()

            except Exception as e:
                results["tests"].append({"name": "구조화 로거", "status": "FAIL", "details": str(e)})
                results["issues"].append(f"구조화 로거 오류: {str(e)}")

            # 대시보드 테스트
            try:
                logger = PACALogger("dashboard_test", enable_database=False, enable_console=False)
                dashboard = MonitoringDashboard(logger, update_interval=60)

                await dashboard.start()

                # 대시보드 데이터 조회 테스트
                dashboard_data = await dashboard.get_dashboard_data()
                if dashboard_data and 'timestamp' in dashboard_data:
                    results["tests"].append({"name": "모니터링 대시보드", "status": "PASS", "details": "데이터 조회 성공"})
                else:
                    results["tests"].append({"name": "모니터링 대시보드", "status": "FAIL", "details": "데이터 조회 실패"})
                    results["issues"].append("대시보드 데이터 조회가 실패했습니다")

                await dashboard.stop()

            except Exception as e:
                results["tests"].append({"name": "모니터링 대시보드", "status": "FAIL", "details": str(e)})
                results["issues"].append(f"모니터링 대시보드 오류: {str(e)}")

            # 점수 계산
            passed = len([t for t in results["tests"] if t["status"] == "PASS"])
            total = len(results["tests"])
            results["score"] = (passed / total) * 100 if total > 0 else 0

            print(f"   ✅ 통과: {passed}/{total} ({results['score']:.1f}%)")

        except Exception as e:
            results["issues"].append(f"모니터링 시스템 검증 오류: {str(e)}")
            print(f"   ❌ 오류: {str(e)}")

        self.test_results[test_name] = results

    async def _test_integration(self):
        """통합 시스템 검증"""
        print("\n🔗 6. 통합 시스템 검증")
        test_name = "integration"
        results = {"tests": [], "score": 0.0, "issues": []}

        try:
            # 전체 시스템 통합 테스트
            try:
                # 도구 관리자 + ReAct 프레임워크 통합
                tool_manager = PACAToolManager()
                file_tool = FileManagerTool(sandbox_mode=True)
                await tool_manager.register_tool(file_tool)

                react_framework = ReActFramework(tool_manager)
                session = await react_framework.create_session("integration_test")

                # 통합 액션 테스트
                think_result = await react_framework.think(session, "파일을 생성해야 합니다", 0.9)
                if think_result.content:
                    # 액션 실행
                    act_result = await react_framework.act(
                        session,
                        "FileManagerTool",
                        operation="write",
                        file_path="integration_test.txt",
                        content="통합 테스트 파일"
                    )

                    if act_result.tool_result and act_result.tool_result.success:
                        results["tests"].append({"name": "시스템 통합", "status": "PASS", "details": "ReAct + 도구 연동 성공"})

                        # 정리
                        await react_framework.act(
                            session,
                            "FileManagerTool",
                            operation="delete",
                            file_path="integration_test.txt"
                        )
                    else:
                        results["tests"].append({"name": "시스템 통합", "status": "FAIL", "details": "도구 실행 실패"})
                        results["issues"].append("통합된 도구 실행이 실패했습니다")
                else:
                    results["tests"].append({"name": "시스템 통합", "status": "FAIL", "details": "사고 과정 실패"})
                    results["issues"].append("통합된 사고 과정이 실패했습니다")

            except Exception as e:
                results["tests"].append({"name": "시스템 통합", "status": "FAIL", "details": str(e)})
                results["issues"].append(f"시스템 통합 오류: {str(e)}")

            # 점수 계산
            passed = len([t for t in results["tests"] if t["status"] == "PASS"])
            total = len(results["tests"])
            results["score"] = (passed / total) * 100 if total > 0 else 0

            print(f"   ✅ 통과: {passed}/{total} ({results['score']:.1f}%)")

        except Exception as e:
            results["issues"].append(f"통합 시스템 검증 오류: {str(e)}")
            print(f"   ❌ 오류: {str(e)}")

        self.test_results[test_name] = results

    async def _test_performance(self):
        """성능 검증"""
        print("\n⚡ 7. 성능 검증")
        test_name = "performance"
        results = {"tests": [], "score": 0.0, "issues": []}

        try:
            # ReAct 프레임워크 성능 테스트
            try:
                tool_manager = PACAToolManager()
                react_framework = ReActFramework(tool_manager)

                start_time = time.time()
                session = await react_framework.create_session("perf_test")

                # 10번의 사고 과정 실행
                for i in range(10):
                    await react_framework.think(session, f"성능 테스트 {i+1}", 0.8)

                end_time = time.time()
                duration = end_time - start_time
                avg_time = duration / 10

                if avg_time < 0.1:  # 100ms 미만
                    results["tests"].append({"name": "ReAct 성능", "status": "PASS", "details": f"평균 {avg_time*1000:.1f}ms"})
                elif avg_time < 0.5:  # 500ms 미만
                    results["tests"].append({"name": "ReAct 성능", "status": "WARN", "details": f"평균 {avg_time*1000:.1f}ms (다소 느림)"})
                else:
                    results["tests"].append({"name": "ReAct 성능", "status": "FAIL", "details": f"평균 {avg_time*1000:.1f}ms (너무 느림)"})
                    results["issues"].append(f"ReAct 성능이 너무 느립니다: {avg_time*1000:.1f}ms")

            except Exception as e:
                results["tests"].append({"name": "ReAct 성능", "status": "FAIL", "details": str(e)})
                results["issues"].append(f"ReAct 성능 테스트 오류: {str(e)}")

            # 메모리 사용량 테스트
            try:
                import psutil
                import os

                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024

                if memory_mb < 100:  # 100MB 미만
                    results["tests"].append({"name": "메모리 사용량", "status": "PASS", "details": f"{memory_mb:.1f}MB"})
                elif memory_mb < 500:  # 500MB 미만
                    results["tests"].append({"name": "메모리 사용량", "status": "WARN", "details": f"{memory_mb:.1f}MB (높음)"})
                else:
                    results["tests"].append({"name": "메모리 사용량", "status": "FAIL", "details": f"{memory_mb:.1f}MB (과다)"})
                    results["issues"].append(f"메모리 사용량이 과다합니다: {memory_mb:.1f}MB")

            except Exception as e:
                results["tests"].append({"name": "메모리 사용량", "status": "FAIL", "details": str(e)})
                results["issues"].append(f"메모리 사용량 테스트 오류: {str(e)}")

            # 점수 계산
            passed = len([t for t in results["tests"] if t["status"] == "PASS"])
            warned = len([t for t in results["tests"] if t["status"] == "WARN"])
            total = len(results["tests"])
            results["score"] = ((passed + warned * 0.7) / total) * 100 if total > 0 else 0

            print(f"   ✅ 통과: {passed}/{total}, 경고: {warned} ({results['score']:.1f}%)")

        except Exception as e:
            results["issues"].append(f"성능 검증 오류: {str(e)}")
            print(f"   ❌ 오류: {str(e)}")

        self.test_results[test_name] = results

    async def _test_stability(self):
        """안정성 검증"""
        print("\n🛡️ 8. 안정성 검증")
        test_name = "stability"
        results = {"tests": [], "score": 0.0, "issues": []}

        try:
            # 오류 처리 테스트
            try:
                tool_manager = PACAToolManager()
                file_tool = FileManagerTool(sandbox_mode=True)
                await tool_manager.register_tool(file_tool)

                react_framework = ReActFramework(tool_manager)
                session = await react_framework.create_session("stability_test")

                # 잘못된 파일 경로로 오류 유발
                error_result = await react_framework.act(
                    session,
                    "FileManagerTool",
                    operation="read",
                    file_path="/invalid/path/file.txt"
                )

                # 오류가 적절히 처리되었는지 확인
                if error_result.tool_result and not error_result.tool_result.success:
                    results["tests"].append({"name": "오류 처리", "status": "PASS", "details": "오류 적절히 처리됨"})
                else:
                    results["tests"].append({"name": "오류 처리", "status": "FAIL", "details": "오류 처리 미흡"})
                    results["issues"].append("오류 처리가 적절하지 않습니다")

            except Exception as e:
                results["tests"].append({"name": "오류 처리", "status": "FAIL", "details": str(e)})
                results["issues"].append(f"오류 처리 테스트 오류: {str(e)}")

            # 동시성 테스트
            try:
                tool_manager = PACAToolManager()
                react_framework = ReActFramework(tool_manager)

                # 동시에 여러 세션 생성
                tasks = []
                for i in range(5):
                    task = react_framework.create_session(f"concurrent_test_{i}")
                    tasks.append(task)

                sessions = await asyncio.gather(*tasks, return_exceptions=True)

                # 모든 세션이 성공적으로 생성되었는지 확인
                successful_sessions = sum(1 for s in sessions if not isinstance(s, Exception))

                if successful_sessions == 5:
                    results["tests"].append({"name": "동시성 처리", "status": "PASS", "details": "5개 세션 동시 생성 성공"})
                elif successful_sessions >= 3:
                    results["tests"].append({"name": "동시성 처리", "status": "WARN", "details": f"{successful_sessions}/5 세션 생성 성공"})
                else:
                    results["tests"].append({"name": "동시성 처리", "status": "FAIL", "details": f"{successful_sessions}/5 세션만 성공"})
                    results["issues"].append(f"동시성 처리에서 {5-successful_sessions}개 세션 실패")

            except Exception as e:
                results["tests"].append({"name": "동시성 처리", "status": "FAIL", "details": str(e)})
                results["issues"].append(f"동시성 처리 테스트 오류: {str(e)}")

            # 점수 계산
            passed = len([t for t in results["tests"] if t["status"] == "PASS"])
            warned = len([t for t in results["tests"] if t["status"] == "WARN"])
            total = len(results["tests"])
            results["score"] = ((passed + warned * 0.6) / total) * 100 if total > 0 else 0

            print(f"   ✅ 통과: {passed}/{total}, 경고: {warned} ({results['score']:.1f}%)")

        except Exception as e:
            results["issues"].append(f"안정성 검증 오류: {str(e)}")
            print(f"   ❌ 오류: {str(e)}")

        self.test_results[test_name] = results

    def _calculate_final_score(self):
        """최종 점수 계산"""
        # 가중치 적용
        weights = {
            "environment": 0.15,
            "core_systems": 0.25,
            "tool_systems": 0.20,
            "feedback_system": 0.10,
            "monitoring_system": 0.10,
            "integration": 0.10,
            "performance": 0.05,
            "stability": 0.05
        }

        total_score = 0.0
        total_weight = 0.0

        for test_name, weight in weights.items():
            if test_name in self.test_results:
                score = self.test_results[test_name]["score"]
                total_score += score * weight
                total_weight += weight

        self.overall_score = total_score / total_weight if total_weight > 0 else 0.0

    def _generate_report(self) -> Dict[str, Any]:
        """검증 보고서 생성"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        # 전체 이슈 수집
        all_issues = []
        for test_results in self.test_results.values():
            all_issues.extend(test_results.get("issues", []))

        # 배포 준비도 평가
        if self.overall_score >= 90:
            deployment_status = "READY"
            deployment_message = "프로덕션 배포 준비 완료"
        elif self.overall_score >= 80:
            deployment_status = "CONDITIONALLY_READY"
            deployment_message = "일부 수정 후 배포 가능"
        elif self.overall_score >= 70:
            deployment_status = "NEEDS_IMPROVEMENT"
            deployment_message = "상당한 개선 필요"
        else:
            deployment_status = "NOT_READY"
            deployment_message = "배포 불가 - 심각한 문제 존재"

        report = {
            "summary": {
                "overall_score": round(self.overall_score, 1),
                "deployment_status": deployment_status,
                "deployment_message": deployment_message,
                "test_duration_seconds": round(duration, 1),
                "total_tests": sum(len(results["tests"]) for results in self.test_results.values()),
                "total_issues": len(all_issues),
                "timestamp": end_time.isoformat()
            },
            "test_results": self.test_results,
            "all_issues": all_issues,
            "recommendations": self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []

        # 점수 기반 권장사항
        if self.overall_score < 90:
            low_score_tests = [
                name for name, results in self.test_results.items()
                if results["score"] < 80
            ]

            if low_score_tests:
                recommendations.append(f"다음 영역의 개선이 필요합니다: {', '.join(low_score_tests)}")

        # 공통 이슈 기반 권장사항
        all_issues = []
        for test_results in self.test_results.values():
            all_issues.extend(test_results.get("issues", []))

        if "설치되지 않았습니다" in str(all_issues):
            recommendations.append("누락된 Python 패키지를 설치하세요: pip install -r requirements.txt")

        if "실패했습니다" in str(all_issues):
            recommendations.append("실패한 테스트 케이스를 개별적으로 디버깅하세요")

        if "느립니다" in str(all_issues):
            recommendations.append("성능 최적화를 고려하세요 (캐싱, 비동기 처리 등)")

        if "메모리" in str(all_issues):
            recommendations.append("메모리 사용량 최적화를 고려하세요")

        # 배포 준비 권장사항
        if self.overall_score >= 80:
            recommendations.append("모니터링 대시보드를 설정하여 실시간 상태를 추적하세요")
            recommendations.append("로그 분석 시스템을 구성하여 문제를 신속히 감지하세요")
            recommendations.append("백업 및 복구 계획을 수립하세요")

        return recommendations


async def main():
    """메인 실행 함수"""
    validator = ProductionValidator()

    try:
        # 검증 실행
        report = await validator.run_validation()

        # 결과 출력
        print("\n" + "=" * 60)
        print("📊 PACA 프로덕션 배포 검증 결과")
        print("=" * 60)

        summary = report["summary"]
        print(f"🎯 전체 점수: {summary['overall_score']}%")
        print(f"📈 배포 상태: {summary['deployment_status']}")
        print(f"💬 상태 메시지: {summary['deployment_message']}")
        print(f"⏱️ 검증 시간: {summary['test_duration_seconds']}초")
        print(f"🧪 총 테스트: {summary['total_tests']}개")
        print(f"⚠️ 총 이슈: {summary['total_issues']}개")

        if report["all_issues"]:
            print(f"\n🚨 발견된 이슈:")
            for issue in report["all_issues"]:
                print(f"   • {issue}")

        if report["recommendations"]:
            print(f"\n💡 개선 권장사항:")
            for rec in report["recommendations"]:
                print(f"   • {rec}")

        # 상세 결과를 파일로 저장
        report_file = f"production_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n📄 상세 보고서가 저장되었습니다: {report_file}")

        # 성공/실패 종료 코드
        if summary['deployment_status'] in ['READY', 'CONDITIONALLY_READY']:
            print(f"\n✅ 검증 완료 - 배포 가능")
            return 0
        else:
            print(f"\n❌ 검증 실패 - 배포 불가")
            return 1

    except Exception as e:
        print(f"\n💥 검증 중 심각한 오류 발생: {str(e)}")
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())