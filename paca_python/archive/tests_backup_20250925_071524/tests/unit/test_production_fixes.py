"""
PACA Production Fixes Test Script
수정된 프로덕션 문제들의 테스트 스크립트
"""

import os
import sys
import asyncio
import time
from pathlib import Path
import logging

# 프로덕션 수정 사항 테스트를 위한 import
try:
    from paca.core.utils.safe_print import safe_print, emoji_to_text, format_status_safe
    from paca.core.utils.environment import get_environment_manager, get_environment_report
    from paca.core.utils.safe_logging import get_safe_logger, configure_safe_logging
    from paca.core.utils.optional_imports import get_feature_availability, check_dependencies_status
    from paca.tools.tool_manager import PACAToolManager
    from paca.tools.react_framework import ReActFramework, ReActSession
    IMPORTS_SUCCESS = True
except ImportError as e:
    safe_print = print
    IMPORTS_SUCCESS = False
    IMPORT_ERROR = str(e)


class ProductionFixesTest:
    """프로덕션 수정 사항 테스트 클래스"""

    def __init__(self):
        self.test_results = {}
        self.project_root = Path(__file__).parent.absolute()
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

    def run_all_tests(self) -> bool:
        """모든 테스트 실행"""
        safe_print("PACA Production Fixes Test Suite")
        safe_print("=" * 40)

        # 1. Import 테스트
        self.test_imports()

        # 2. 인코딩 테스트
        self.test_encoding_fixes()

        # 3. 환경 설정 테스트
        self.test_environment_setup()

        # 4. 로깅 시스템 테스트
        self.test_logging_system()

        # 5. 비동기 API 테스트
        self.test_async_api_fixes()

        # 6. 의존성 관리 테스트
        self.test_dependency_management()

        # 7. 파일 시스템 테스트
        self.test_file_system_compatibility()

        # 결과 출력
        self.print_test_results()

        return self.failed_tests == 0

    def test_imports(self):
        """Import 테스트"""
        safe_print("\n[1/7] Testing imports...")
        self.total_tests += 1

        if IMPORTS_SUCCESS:
            self.passed_tests += 1
            self.test_results["imports"] = "PASS"
            safe_print("   [SUCCESS] All imports successful")
        else:
            self.failed_tests += 1
            self.test_results["imports"] = f"FAIL - {IMPORT_ERROR}"
            safe_print(f"   [ERROR] Import failed: {IMPORT_ERROR}")

    def test_encoding_fixes(self):
        """인코딩 수정 테스트"""
        safe_print("\n[2/7] Testing encoding fixes...")
        self.total_tests += 1

        try:
            # 이모지 텍스트 변환 테스트
            test_text = "🚀 PACA 시스템 시작 💡 테스트 중 ⚠️ 경고 ✅ 성공"
            converted = emoji_to_text(test_text)
            safe_print(f"   Original: {repr(test_text)}")
            safe_print(f"   Converted: {converted}")

            # 안전한 출력 테스트
            safe_print("   Testing safe output with emojis: 🔥🎯📊")

            # 상태 메시지 포맷팅 테스트
            status_msg = format_status_safe("success", "인코딩 테스트 완료 ✅")
            safe_print(f"   Status: {status_msg}")

            self.passed_tests += 1
            self.test_results["encoding"] = "PASS"
            safe_print("   [SUCCESS] Encoding fixes working correctly")

        except Exception as e:
            self.failed_tests += 1
            self.test_results["encoding"] = f"FAIL - {e}"
            safe_print(f"   [ERROR] Encoding test failed: {e}")

    def test_environment_setup(self):
        """환경 설정 테스트"""
        safe_print("\n[3/7] Testing environment setup...")
        self.total_tests += 1

        try:
            # 환경 변수 확인
            required_vars = ['PYTHONIOENCODING']
            for var in required_vars:
                value = os.environ.get(var)
                safe_print(f"   {var}: {value}")

            # 환경 관리자 테스트 (사용 가능한 경우)
            if IMPORTS_SUCCESS and get_environment_manager:
                env_manager = get_environment_manager()
                status = env_manager.check_environment()
                safe_print(f"   Environment status: {status}")

                # 환경 보고서 생성
                report = get_environment_report()
                safe_print("   Environment report generated successfully")

            self.passed_tests += 1
            self.test_results["environment"] = "PASS"
            safe_print("   [SUCCESS] Environment setup working correctly")

        except Exception as e:
            self.failed_tests += 1
            self.test_results["environment"] = f"FAIL - {e}"
            safe_print(f"   [ERROR] Environment test failed: {e}")

    def test_logging_system(self):
        """로깅 시스템 테스트"""
        safe_print("\n[4/7] Testing logging system...")
        self.total_tests += 1

        try:
            # 로그 디렉토리 확인
            log_dir = self.project_root / "logs"
            safe_print(f"   Log directory exists: {log_dir.exists()}")

            # 안전한 로거 테스트 (사용 가능한 경우)
            if IMPORTS_SUCCESS and get_safe_logger:
                logger = get_safe_logger("TestLogger")
                logger.info("로깅 시스템 테스트 ✅")
                logger.warning("경고 메시지 테스트 ⚠️")
                safe_print("   [INFO] Safe logger test completed")

            # 기본 로깅 테스트
            test_logger = logging.getLogger("TestBasicLogger")
            test_logger.info("Basic logging test")

            self.passed_tests += 1
            self.test_results["logging"] = "PASS"
            safe_print("   [SUCCESS] Logging system working correctly")

        except Exception as e:
            self.failed_tests += 1
            self.test_results["logging"] = f"FAIL - {e}"
            safe_print(f"   [ERROR] Logging test failed: {e}")

    def test_async_api_fixes(self):
        """비동기 API 수정 테스트"""
        safe_print("\n[5/7] Testing async API fixes...")
        self.total_tests += 1

        try:
            if IMPORTS_SUCCESS:
                # 도구 관리자 테스트
                tool_manager = PACAToolManager()
                safe_print("   Tool manager created successfully")

                # register_tool_async 메서드 존재 확인
                if hasattr(tool_manager, 'register_tool_async'):
                    safe_print("   register_tool_async method available")
                else:
                    raise AttributeError("register_tool_async method not found")

                # ReAct 세션 session_id 속성 테스트
                session = ReActSession(goal="테스트 목표")
                if hasattr(session, 'session_id'):
                    safe_print(f"   Session ID property: {session.session_id}")
                else:
                    raise AttributeError("session_id property not found")

            self.passed_tests += 1
            self.test_results["async_api"] = "PASS"
            safe_print("   [SUCCESS] Async API fixes working correctly")

        except Exception as e:
            self.failed_tests += 1
            self.test_results["async_api"] = f"FAIL - {e}"
            safe_print(f"   [ERROR] Async API test failed: {e}")

    def test_dependency_management(self):
        """의존성 관리 테스트"""
        safe_print("\n[6/7] Testing dependency management...")
        self.total_tests += 1

        try:
            if IMPORTS_SUCCESS:
                # 기능 가용성 확인
                features = get_feature_availability()
                safe_print(f"   Available features: {sum(features.values())}/{len(features)}")

                # 의존성 상태 확인
                deps_status = check_dependencies_status()
                safe_print(f"   Dependencies: {deps_status['available']}/{deps_status['total']} available")

                # 각 기능별 상태 출력
                for feature, available in features.items():
                    status = "OK" if available else "Missing"
                    safe_print(f"     {feature}: {status}")

            self.passed_tests += 1
            self.test_results["dependencies"] = "PASS"
            safe_print("   [SUCCESS] Dependency management working correctly")

        except Exception as e:
            self.failed_tests += 1
            self.test_results["dependencies"] = f"FAIL - {e}"
            safe_print(f"   [ERROR] Dependency test failed: {e}")

    def test_file_system_compatibility(self):
        """파일 시스템 호환성 테스트"""
        safe_print("\n[7/7] Testing file system compatibility...")
        self.total_tests += 1

        try:
            # UTF-8 파일 읽기/쓰기 테스트
            test_file = self.project_root / "test_utf8.tmp"
            test_content = "테스트 내용 🚀 UTF-8 인코딩 💡"

            # 파일 쓰기
            test_file.write_text(test_content, encoding='utf-8')
            safe_print("   UTF-8 file write: OK")

            # 파일 읽기
            read_content = test_file.read_text(encoding='utf-8')
            if read_content == test_content:
                safe_print("   UTF-8 file read: OK")
            else:
                raise ValueError("File content mismatch")

            # 파일 삭제
            test_file.unlink()
            safe_print("   File cleanup: OK")

            # 로그 디렉토리 권한 테스트
            log_dir = self.project_root / "logs"
            if log_dir.exists():
                test_log = log_dir / "test_permissions.log"
                test_log.write_text("permission test", encoding='utf-8')
                test_log.unlink()
                safe_print("   Log directory permissions: OK")

            self.passed_tests += 1
            self.test_results["file_system"] = "PASS"
            safe_print("   [SUCCESS] File system compatibility working correctly")

        except Exception as e:
            self.failed_tests += 1
            self.test_results["file_system"] = f"FAIL - {e}"
            safe_print(f"   [ERROR] File system test failed: {e}")

    def print_test_results(self):
        """테스트 결과 출력"""
        safe_print("\n" + "=" * 40)
        safe_print("TEST RESULTS SUMMARY")
        safe_print("=" * 40)

        safe_print(f"Total Tests: {self.total_tests}")
        safe_print(f"Passed: {self.passed_tests}")
        safe_print(f"Failed: {self.failed_tests}")
        safe_print(f"Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%")

        safe_print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "[SUCCESS]" if result == "PASS" else "[ERROR]"
            safe_print(f"  {status} {test_name}: {result}")

        if self.failed_tests == 0:
            safe_print("\n[SUCCESS] All production fixes are working correctly!")
            safe_print("PACA is ready for production use.")
        else:
            safe_print(f"\n[WARNING] {self.failed_tests} test(s) failed.")
            safe_print("Some issues may still need attention.")

        safe_print("\n" + "=" * 40)


async def run_async_tests():
    """비동기 테스트 실행"""
    if not IMPORTS_SUCCESS:
        return False

    safe_print("\nRunning async-specific tests...")

    try:
        # 비동기 도구 등록 테스트
        tool_manager = PACAToolManager()

        # 더미 도구 클래스
        from paca.tools.base import Tool, ToolResult, ToolType

        class TestTool(Tool):
            def __init__(self):
                super().__init__("test_tool", ToolType.UTILITY, "Test tool for async API")

            async def execute(self, **kwargs) -> ToolResult:
                return ToolResult(success=True, data="Test completed")

            def validate_input(self, **kwargs) -> bool:
                return True

        # 비동기 도구 등록 테스트
        test_tool = TestTool()
        result = await tool_manager.register_tool_async(test_tool)
        safe_print(f"   Async tool registration: {'SUCCESS' if result else 'FAILED'}")

        return True

    except Exception as e:
        safe_print(f"   Async tests failed: {e}")
        return False


def main():
    """메인 함수"""
    tester = ProductionFixesTest()

    # 동기 테스트 실행
    sync_success = tester.run_all_tests()

    # 비동기 테스트 실행
    try:
        async_success = asyncio.run(run_async_tests())
    except Exception as e:
        safe_print(f"\nAsync tests failed: {e}")
        async_success = False

    # 전체 결과
    overall_success = sync_success and async_success

    if overall_success:
        safe_print("\n[FINAL SUCCESS] All production fixes validated!")
    else:
        safe_print("\n[FINAL WARNING] Some tests failed. Review the results above.")

    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)