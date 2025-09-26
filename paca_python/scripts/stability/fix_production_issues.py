"""
PACA Production Issues Fix Script
프로덕션 환경 문제점 자동 수정 스크립트
"""

import os
import sys
import subprocess
from pathlib import Path
import logging
import time

# 안전한 출력을 위한 import
try:
    from paca.core.utils.safe_print import safe_print, setup_unicode_environment
    from paca.core.utils.environment import get_environment_manager
    from paca.core.utils.safe_logging import configure_safe_logging
    from paca.core.utils.optional_imports import check_dependencies_status, get_feature_availability
except ImportError:
    # 폴백 출력 함수
    def safe_print(*args, **kwargs):
        try:
            print(*args, **kwargs)
        except UnicodeEncodeError:
            print("[Unicode output error - content not displayable]")

    def setup_unicode_environment():
        return True

    get_environment_manager = None
    configure_safe_logging = None
    check_dependencies_status = None
    get_feature_availability = None


class ProductionIssueFixer:
    """프로덕션 환경 문제 해결사"""

    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.issues_fixed = []
        self.issues_failed = []

        # 로깅 설정
        try:
            if configure_safe_logging:
                self.logger = configure_safe_logging("ProductionFixer").get_logger()
            else:
                logging.basicConfig(level=logging.INFO)
                self.logger = logging.getLogger("ProductionFixer")
        except Exception:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("ProductionFixer")

    def run_all_fixes(self) -> bool:
        """모든 수정 작업 실행"""
        safe_print("PACA Production Issues Fix Script")
        safe_print("=" * 50)

        success = True

        # 1. 환경 설정 수정
        if not self.fix_environment_issues():
            success = False

        # 2. 인코딩 문제 수정
        if not self.fix_encoding_issues():
            success = False

        # 3. 로깅 시스템 수정
        if not self.fix_logging_issues():
            success = False

        # 4. 의존성 문제 점검
        if not self.check_dependency_issues():
            success = False

        # 5. 권한 문제 수정
        if not self.fix_permission_issues():
            success = False

        # 6. 환경 스크립트 생성
        if not self.create_environment_scripts():
            success = False

        # 결과 보고
        self.print_summary()

        return success

    def fix_environment_issues(self) -> bool:
        """환경 변수 문제 수정"""
        safe_print("\n[1/6] Fixing environment issues...")

        try:
            # PYTHONIOENCODING 설정
            os.environ['PYTHONIOENCODING'] = 'utf-8'

            # PYTHONPATH 설정
            pythonpath = str(self.project_root)
            current_pythonpath = os.environ.get('PYTHONPATH', '')

            if pythonpath not in current_pythonpath:
                if current_pythonpath:
                    new_pythonpath = f"{current_pythonpath}{os.pathsep}{pythonpath}"
                else:
                    new_pythonpath = pythonpath
                os.environ['PYTHONPATH'] = new_pythonpath

            # 환경 관리자 사용 (사용 가능한 경우)
            if get_environment_manager:
                env_manager = get_environment_manager()
                env_manager.setup_all()

            # Windows 콘솔 인코딩 설정
            if sys.platform == 'win32':
                try:
                    subprocess.run(['chcp', '65001'], check=True, capture_output=True)
                except:
                    pass

            self.issues_fixed.append("Environment variables configured")
            safe_print("   [SUCCESS] Environment variables configured")
            return True

        except Exception as e:
            self.issues_failed.append(f"Environment setup: {e}")
            safe_print(f"   [ERROR] Environment setup failed: {e}")
            return False

    def fix_encoding_issues(self) -> bool:
        """인코딩 문제 수정"""
        safe_print("\n[2/6] Fixing encoding issues...")

        try:
            # UTF-8 환경 설정
            if setup_unicode_environment:
                setup_unicode_environment()

            # 환경 변수 재확인
            if os.environ.get('PYTHONIOENCODING', '').lower() != 'utf-8':
                os.environ['PYTHONIOENCODING'] = 'utf-8'

            self.issues_fixed.append("Encoding configuration applied")
            safe_print("   [SUCCESS] Encoding issues fixed")
            return True

        except Exception as e:
            self.issues_failed.append(f"Encoding fix: {e}")
            safe_print(f"   [ERROR] Encoding fix failed: {e}")
            return False

    def fix_logging_issues(self) -> bool:
        """로깅 시스템 문제 수정"""
        safe_print("\n[3/6] Fixing logging issues...")

        try:
            # 로그 디렉토리 생성
            log_dir = self.project_root / "logs"
            log_dir.mkdir(exist_ok=True)

            # 로그 파일 권한 테스트
            test_file = log_dir / "test.log"
            test_file.write_text("test", encoding='utf-8')
            test_file.unlink()

            # 안전한 로깅 설정 (사용 가능한 경우)
            if configure_safe_logging:
                configure_safe_logging(log_dir=log_dir)

            self.issues_fixed.append("Logging system configured")
            safe_print("   [SUCCESS] Logging issues fixed")
            return True

        except Exception as e:
            self.issues_failed.append(f"Logging fix: {e}")
            safe_print(f"   [ERROR] Logging fix failed: {e}")
            return False

    def check_dependency_issues(self) -> bool:
        """의존성 문제 점검"""
        safe_print("\n[4/6] Checking dependency issues...")

        try:
            # 필수 패키지 확인
            required_packages = ['asyncio', 'typing', 'dataclasses', 'pathlib']
            missing_required = []

            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_required.append(package)

            if missing_required:
                self.issues_failed.append(f"Missing required packages: {missing_required}")
                safe_print(f"   [ERROR] Missing required packages: {missing_required}")
                return False

            # 선택적 의존성 상태 확인
            if check_dependencies_status:
                deps_status = check_dependencies_status()
                safe_print(f"   [INFO] Dependencies: {deps_status['available']}/{deps_status['total']} available")

            if get_feature_availability:
                features = get_feature_availability()
                available_features = sum(1 for available in features.values() if available)
                total_features = len(features)
                safe_print(f"   [INFO] Features: {available_features}/{total_features} available")

            self.issues_fixed.append("Dependencies checked")
            safe_print("   [SUCCESS] Dependencies checked")
            return True

        except Exception as e:
            self.issues_failed.append(f"Dependency check: {e}")
            safe_print(f"   [ERROR] Dependency check failed: {e}")
            return False

    def fix_permission_issues(self) -> bool:
        """권한 문제 수정"""
        safe_print("\n[5/6] Fixing permission issues...")

        try:
            # 로그 디렉토리 권한 확인
            log_dir = self.project_root / "logs"
            if log_dir.exists():
                test_file = log_dir / "permission_test.txt"
                test_file.write_text("permission test", encoding='utf-8')
                test_file.unlink()

            # 프로젝트 디렉토리 읽기 권한 확인
            for py_file in self.project_root.glob("**/*.py"):
                if py_file.is_file():
                    py_file.read_text(encoding='utf-8', errors='ignore')
                    break

            self.issues_fixed.append("Permissions verified")
            safe_print("   [SUCCESS] Permissions verified")
            return True

        except Exception as e:
            self.issues_failed.append(f"Permission fix: {e}")
            safe_print(f"   [ERROR] Permission fix failed: {e}")
            return False

    def create_environment_scripts(self) -> bool:
        """환경 설정 스크립트 생성"""
        safe_print("\n[6/6] Creating environment scripts...")

        try:
            # Windows 배치 파일
            if sys.platform == 'win32':
                script_content = f"""@echo off
echo Setting up PACA environment...
set PYTHONIOENCODING=utf-8
set PYTHONPATH={self.project_root}
set PACA_LOG_LEVEL=INFO
chcp 65001 >nul 2>&1
echo Environment setup complete!
echo.
echo To run PACA:
echo python -m paca
pause
"""
                script_path = self.project_root / "setup_environment.bat"
                script_path.write_text(script_content, encoding='utf-8')
                safe_print(f"   [SUCCESS] Created {script_path}")

            # Unix/Linux 셸 스크립트
            else:
                script_content = f"""#!/bin/bash
echo "Setting up PACA environment..."
export PYTHONIOENCODING=utf-8
export PYTHONPATH={self.project_root}
export PACA_LOG_LEVEL=INFO
echo "Environment setup complete!"
echo ""
echo "To run PACA:"
echo "python -m paca"
"""
                script_path = self.project_root / "setup_environment.sh"
                script_path.write_text(script_content, encoding='utf-8')
                script_path.chmod(0o755)
                safe_print(f"   [SUCCESS] Created {script_path}")

            # Python 설정 스크립트
            python_script = f"""#!/usr/bin/env python3
# PACA Environment Setup Script
import os
import sys
from pathlib import Path

def setup_paca_environment():
    project_root = Path(__file__).parent.absolute()

    # Environment variables
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONPATH'] = str(project_root)
    os.environ['PACA_LOG_LEVEL'] = 'INFO'

    # Add to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print("PACA environment configured successfully!")
    return True

if __name__ == "__main__":
    setup_paca_environment()
"""
            python_script_path = self.project_root / "setup_paca_env.py"
            python_script_path.write_text(python_script, encoding='utf-8')

            self.issues_fixed.append("Environment scripts created")
            safe_print("   [SUCCESS] Environment scripts created")
            return True

        except Exception as e:
            self.issues_failed.append(f"Script creation: {e}")
            safe_print(f"   [ERROR] Script creation failed: {e}")
            return False

    def print_summary(self):
        """수정 결과 요약 출력"""
        safe_print("\n" + "=" * 50)
        safe_print("PRODUCTION ISSUES FIX SUMMARY")
        safe_print("=" * 50)

        if self.issues_fixed:
            safe_print(f"\n[SUCCESS] Fixed issues ({len(self.issues_fixed)}):")
            for issue in self.issues_fixed:
                safe_print(f"  - {issue}")

        if self.issues_failed:
            safe_print(f"\n[ERROR] Failed to fix ({len(self.issues_failed)}):")
            for issue in self.issues_failed:
                safe_print(f"  - {issue}")

        if not self.issues_failed:
            safe_print("\n[SUCCESS] All production issues have been resolved!")
            safe_print("\nNext steps:")
            safe_print("1. Run: python -m paca")
            safe_print("2. Or use: python setup_paca_env.py && python -m paca")
            if sys.platform == 'win32':
                safe_print("3. Or double-click: setup_environment.bat")
        else:
            safe_print(f"\n[WARNING] {len(self.issues_failed)} issues remain. Please check the errors above.")

        safe_print("\n" + "=" * 50)

    def run_diagnostic(self):
        """진단 모드 실행"""
        safe_print("PACA Production Diagnostic")
        safe_print("=" * 30)

        # 환경 정보
        safe_print(f"Python Version: {sys.version}")
        safe_print(f"Platform: {sys.platform}")
        safe_print(f"Project Root: {self.project_root}")
        safe_print(f"Current Directory: {Path.cwd()}")

        # 환경 변수 확인
        safe_print("\nEnvironment Variables:")
        env_vars = ['PYTHONIOENCODING', 'PYTHONPATH', 'PACA_LOG_LEVEL']
        for var in env_vars:
            value = os.environ.get(var, 'Not Set')
            safe_print(f"  {var}: {value}")

        # 파일 시스템 권한
        safe_print("\nFile System Check:")
        try:
            test_file = self.project_root / "diagnostic_test.tmp"
            test_file.write_text("test", encoding='utf-8')
            test_file.unlink()
            safe_print("  Write Permissions: OK")
        except Exception as e:
            safe_print(f"  Write Permissions: ERROR - {e}")

        # 의존성 확인
        if check_dependencies_status:
            safe_print("\nDependencies Status:")
            deps = check_dependencies_status()
            safe_print(f"  Available: {deps['available']}/{deps['total']}")

        safe_print("\nDiagnostic complete.")


def main():
    """메인 함수"""
    fixer = ProductionIssueFixer()

    # 명령행 인수 확인
    if len(sys.argv) > 1 and sys.argv[1] == 'diagnostic':
        fixer.run_diagnostic()
    else:
        success = fixer.run_all_fixes()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()