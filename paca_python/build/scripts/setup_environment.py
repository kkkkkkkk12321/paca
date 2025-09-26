"""
PACA Environment Setup Script
Windows 환경에서 PACA 실행을 위한 환경 설정 자동화
"""

import os
import sys
import subprocess
from pathlib import Path


def setup_environment_variables():
    """환경 변수 자동 설정"""
    print("Setting up environment variables...")

    # PYTHONIOENCODING 설정
    os.environ['PYTHONIOENCODING'] = 'utf-8'

    # Windows에서 영구 환경 변수 설정
    if sys.platform == 'win32':
        try:
            subprocess.run(['setx', 'PYTHONIOENCODING', 'utf-8'], check=True, capture_output=True)
            print("✓ PYTHONIOENCODING set to utf-8 (permanent)")
        except subprocess.CalledProcessError:
            print("⚠ Failed to set permanent PYTHONIOENCODING (temporary only)")

    # PYTHONPATH 설정
    current_dir = Path(__file__).parent.absolute()
    pythonpath = str(current_dir)

    os.environ['PYTHONPATH'] = pythonpath

    if sys.platform == 'win32':
        try:
            subprocess.run(['setx', 'PYTHONPATH', pythonpath], check=True, capture_output=True)
            print(f"✓ PYTHONPATH set to {pythonpath} (permanent)")
        except subprocess.CalledProcessError:
            print(f"⚠ Failed to set permanent PYTHONPATH (temporary only)")

    return True


def check_dependencies():
    """필수 및 선택적 의존성 확인"""
    print("Checking dependencies...")

    required_packages = [
        'asyncio',
        'typing',
        'dataclasses',
        'pathlib',
        'json',
        'logging'
    ]

    optional_packages = {
        'structlog': 'Enhanced logging support',
        'prometheus_client': 'Metrics collection',
        'psutil': 'System monitoring',
        'uvloop': 'High-performance event loop (Unix only)'
    }

    # 필수 패키지 확인
    missing_required = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"✗ {package} (REQUIRED)")

    # 선택적 패키지 확인
    missing_optional = []
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print(f"✓ {package} - {description}")
        except ImportError:
            missing_optional.append((package, description))
            print(f"○ {package} - {description} (OPTIONAL)")

    if missing_required:
        print(f"\n⚠ Missing required packages: {', '.join(missing_required)}")
        print("Please install with: pip install <package_name>")

    if missing_optional:
        print(f"\n○ Optional packages not installed:")
        for package, description in missing_optional:
            print(f"  - {package}: {description}")
        print("Install with: pip install <package_name>")

    return len(missing_required) == 0


def create_log_directory():
    """로그 디렉토리 생성 및 권한 설정"""
    print("Setting up log directory...")

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Windows에서 로그 파일 권한 설정
    if sys.platform == 'win32':
        try:
            # 읽기/쓰기 권한 확인
            test_file = log_dir / "test.log"
            test_file.write_text("test")
            test_file.unlink()
            print("✓ Log directory permissions OK")
        except Exception as e:
            print(f"⚠ Log directory permission issue: {e}")

    return True


def setup_console_encoding():
    """Windows 콘솔 인코딩 설정"""
    if sys.platform == 'win32':
        print("Setting up Windows console encoding...")
        try:
            # 콘솔 코드페이지를 UTF-8로 설정
            subprocess.run(['chcp', '65001'], check=True, capture_output=True)
            print("✓ Console codepage set to UTF-8")
        except subprocess.CalledProcessError:
            print("⚠ Failed to set console codepage")

    return True


def main():
    """메인 설정 함수"""
    print("PACA Environment Setup")
    print("=" * 30)

    success = True

    # 1. 환경 변수 설정
    if not setup_environment_variables():
        success = False

    # 2. 의존성 확인
    if not check_dependencies():
        print("⚠ Some required dependencies are missing")

    # 3. 로그 디렉토리 설정
    if not create_log_directory():
        success = False

    # 4. 콘솔 인코딩 설정
    if not setup_console_encoding():
        success = False

    print("\n" + "=" * 30)
    if success:
        print("✓ Environment setup completed successfully!")
        print("\nYou can now run PACA with:")
        print("python -m paca")
    else:
        print("⚠ Some setup steps failed. Please check the errors above.")

    return success


if __name__ == "__main__":
    main()