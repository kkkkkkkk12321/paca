"""
PACA Environment Configuration Module
환경 설정 자동화 및 관리 모듈
"""

import os
import sys
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from .safe_print import safe_print, setup_unicode_environment


class EnvironmentManager:
    """PACA 환경 관리자"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.environment_vars = {}
        self.required_vars = [
            'PYTHONIOENCODING',
            'PYTHONPATH'
        ]
        self.optional_vars = [
            'GEMINI_API_KEYS',
            'PACA_LOG_LEVEL',
            'PACA_DEBUG_MODE'
        ]

    def setup_all(self) -> bool:
        """모든 환경 설정 수행"""
        success = True

        # 1. 유니코드 환경 설정
        if not setup_unicode_environment():
            safe_print("⚠ Unicode environment setup failed")
            success = False

        # 2. 환경 변수 설정
        if not self.setup_environment_variables():
            safe_print("⚠ Environment variables setup failed")
            success = False

        # 3. 로그 디렉토리 설정
        if not self.setup_log_directory():
            safe_print("⚠ Log directory setup failed")
            success = False

        # 4. Python 경로 설정
        if not self.setup_python_path():
            safe_print("⚠ Python path setup failed")
            success = False

        return success

    def setup_environment_variables(self) -> bool:
        """환경 변수 설정"""
        try:
            # PYTHONIOENCODING 설정
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            self.environment_vars['PYTHONIOENCODING'] = 'utf-8'

            # PACA 프로젝트 루트 경로
            project_root = Path(__file__).parent.parent.parent.parent.absolute()
            pythonpath = str(project_root)

            # PYTHONPATH 설정
            current_pythonpath = os.environ.get('PYTHONPATH', '')
            if pythonpath not in current_pythonpath:
                if current_pythonpath:
                    new_pythonpath = f"{current_pythonpath}{os.pathsep}{pythonpath}"
                else:
                    new_pythonpath = pythonpath
                os.environ['PYTHONPATH'] = new_pythonpath
                self.environment_vars['PYTHONPATH'] = new_pythonpath

            # PACA 기본 설정
            if 'PACA_LOG_LEVEL' not in os.environ:
                os.environ['PACA_LOG_LEVEL'] = 'INFO'
                self.environment_vars['PACA_LOG_LEVEL'] = 'INFO'

            if 'PACA_DEBUG_MODE' not in os.environ:
                os.environ['PACA_DEBUG_MODE'] = 'false'
                self.environment_vars['PACA_DEBUG_MODE'] = 'false'

            safe_print("[SUCCESS] Environment variables configured")
            return True

        except Exception as e:
            self.logger.error(f"Environment variables setup failed: {e}")
            return False

    def setup_python_path(self) -> bool:
        """Python 경로 설정"""
        try:
            project_root = Path(__file__).parent.parent.parent.parent.absolute()
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            safe_print(f"[SUCCESS] Python path configured: {project_root}")
            return True

        except Exception as e:
            self.logger.error(f"Python path setup failed: {e}")
            return False

    def setup_log_directory(self) -> bool:
        """로그 디렉토리 설정"""
        try:
            project_root = Path(__file__).parent.parent.parent.parent.absolute()
            log_dir = project_root / "logs"
            log_dir.mkdir(exist_ok=True)

            # 권한 테스트
            test_file = log_dir / "test.log"
            test_file.write_text("test", encoding='utf-8')
            test_file.unlink()

            safe_print(f"[SUCCESS] Log directory configured: {log_dir}")
            return True

        except Exception as e:
            self.logger.error(f"Log directory setup failed: {e}")
            return False

    def check_environment(self) -> Dict[str, bool]:
        """환경 상태 확인"""
        status = {}

        # 필수 환경 변수 확인
        for var in self.required_vars:
            status[f"env_{var}"] = var in os.environ

        # 선택적 환경 변수 확인
        for var in self.optional_vars:
            status[f"optional_{var}"] = var in os.environ

        # Python 경로 확인
        project_root = Path(__file__).parent.parent.parent.parent.absolute()
        status["python_path"] = str(project_root) in sys.path

        # 로그 디렉토리 확인
        log_dir = project_root / "logs"
        status["log_directory"] = log_dir.exists() and log_dir.is_dir()

        return status

    def get_environment_info(self) -> Dict[str, str]:
        """환경 정보 반환"""
        return {
            'platform': platform.platform(),
            'python_version': sys.version,
            'python_executable': sys.executable,
            'encoding': sys.getdefaultencoding(),
            'file_system_encoding': sys.getfilesystemencoding(),
            'current_directory': str(Path.cwd()),
            'project_root': str(Path(__file__).parent.parent.parent.parent.absolute()),
            **{k: os.environ.get(k, 'Not Set') for k in self.required_vars + self.optional_vars}
        }

    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """환경 설정 검증"""
        issues = []

        # 필수 환경 변수 확인
        for var in self.required_vars:
            if var not in os.environ:
                issues.append(f"Missing required environment variable: {var}")

        # Python 경로 확인
        project_root = Path(__file__).parent.parent.parent.parent.absolute()
        if str(project_root) not in sys.path:
            issues.append("Project root not in Python path")

        # 로그 디렉토리 확인
        log_dir = project_root / "logs"
        if not log_dir.exists():
            issues.append("Log directory does not exist")

        # 인코딩 확인
        if os.environ.get('PYTHONIOENCODING', '').lower() != 'utf-8':
            issues.append("PYTHONIOENCODING is not set to utf-8")

        return len(issues) == 0, issues

    def fix_common_issues(self) -> bool:
        """일반적인 문제 자동 수정"""
        try:
            # 환경 변수 재설정
            self.setup_environment_variables()

            # Python 경로 재설정
            self.setup_python_path()

            # 로그 디렉토리 재생성
            self.setup_log_directory()

            # Windows 콘솔 인코딩 설정
            if platform.system() == 'Windows':
                try:
                    import subprocess
                    subprocess.run(['chcp', '65001'], check=True, capture_output=True)
                except:
                    pass

            safe_print("[SUCCESS] Common issues fixed")
            return True

        except Exception as e:
            self.logger.error(f"Failed to fix common issues: {e}")
            return False

    def create_environment_script(self) -> bool:
        """환경 설정 스크립트 생성"""
        try:
            project_root = Path(__file__).parent.parent.parent.parent.absolute()

            # Windows 배치 파일
            if platform.system() == 'Windows':
                script_path = project_root / "set_environment.bat"
                script_content = f"""@echo off
echo Setting up PACA environment...
set PYTHONIOENCODING=utf-8
set PYTHONPATH={project_root}
set PACA_LOG_LEVEL=INFO
set PACA_DEBUG_MODE=false
chcp 65001 >nul 2>&1
echo Environment setup complete!
"""
                script_path.write_text(script_content, encoding='utf-8')

            # Unix/Linux 셸 스크립트
            else:
                script_path = project_root / "set_environment.sh"
                script_content = f"""#!/bin/bash
echo "Setting up PACA environment..."
export PYTHONIOENCODING=utf-8
export PYTHONPATH={project_root}
export PACA_LOG_LEVEL=INFO
export PACA_DEBUG_MODE=false
echo "Environment setup complete!"
"""
                script_path.write_text(script_content, encoding='utf-8')
                script_path.chmod(0o755)

            safe_print(f"[SUCCESS] Environment script created: {script_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create environment script: {e}")
            return False


# 전역 환경 관리자 인스턴스
_env_manager = None

def get_environment_manager() -> EnvironmentManager:
    """환경 관리자 싱글톤 인스턴스 반환"""
    global _env_manager
    if _env_manager is None:
        _env_manager = EnvironmentManager()
    return _env_manager

def auto_setup_environment() -> bool:
    """자동 환경 설정"""
    manager = get_environment_manager()
    return manager.setup_all()

def check_environment_status() -> Dict[str, bool]:
    """환경 상태 확인"""
    manager = get_environment_manager()
    return manager.check_environment()

def get_environment_report() -> str:
    """환경 상태 보고서 생성"""
    manager = get_environment_manager()

    info = manager.get_environment_info()
    status = manager.check_environment()
    is_valid, issues = manager.validate_configuration()

    report = "PACA Environment Report\n"
    report += "=" * 30 + "\n\n"

    report += "System Information:\n"
    report += f"  Platform: {info['platform']}\n"
    report += f"  Python: {info['python_version']}\n"
    report += f"  Encoding: {info['encoding']}\n\n"

    report += "Environment Status:\n"
    for key, value in status.items():
        status_symbol = "[SUCCESS]" if value else "[ERROR]"
        report += f"  {status_symbol} {key}: {value}\n"

    if issues:
        report += "\nIssues Found:\n"
        for issue in issues:
            report += f"  - {issue}\n"

    report += f"\nOverall Status: {'HEALTHY' if is_valid else 'NEEDS ATTENTION'}\n"

    return report


# 모듈 로드 시 자동 환경 설정 (선택적)
# auto_setup_environment()