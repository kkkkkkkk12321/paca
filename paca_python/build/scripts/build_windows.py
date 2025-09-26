"""
PACA v5 Windows 배포 스크립트
PyInstaller를 사용하여 실행파일 생성
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def check_requirements():
    """필수 패키지 확인 및 설치"""
    required_packages = [
        'pyinstaller',
        'customtkinter',
        'pillow',
        'pydantic',
        'asyncio'
    ]

    print("필수 패키지 확인 중...")

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 설치됨")
        except ImportError:
            print(f"✗ {package} 설치 필요")
            print(f"설치 중: pip install {package}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ {package} 설치 완료")

def clean_build_directories():
    """이전 빌드 디렉토리 정리"""
    directories_to_clean = ['build', 'dist', '__pycache__']

    print("이전 빌드 파일 정리 중...")

    for directory in directories_to_clean:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"✓ {directory} 디렉토리 정리 완료")

def create_pyinstaller_spec():
    """PyInstaller spec 파일 생성"""
    spec_content = '''
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['desktop_app/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('paca', 'paca'),
        ('desktop_app', 'desktop_app'),
    ],
    hiddenimports=[
        'paca',
        'paca.core',
        'paca.mathematics',
        'paca.services',
        'paca.cognitive',
        'paca.reasoning',
        'customtkinter',
        'PIL',
        'pydantic',
        'asyncio'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='paca-v5-windows',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='desktop_app/assets/icon.ico' if os.path.exists('desktop_app/assets/icon.ico') else None,
)
'''

    with open('paca-v5.spec', 'w', encoding='utf-8') as f:
        f.write(spec_content)

    print("✓ PyInstaller spec 파일 생성 완료")

def build_executable():
    """실행파일 빌드"""
    print("실행파일 빌드 시작...")
    print("이 과정은 몇 분이 소요될 수 있습니다...")

    try:
        # PyInstaller 실행
        result = subprocess.run([
            sys.executable, '-m', 'PyInstaller',
            '--clean',
            '--noconfirm',
            'paca-v5.spec'
        ], check=True, capture_output=True, text=True)

        print("✓ 빌드 성공!")
        print(f"실행파일 위치: {os.path.abspath('dist/paca-v5-windows.exe')}")

        # 빌드 결과 확인
        exe_path = Path('dist/paca-v5-windows.exe')
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"실행파일 크기: {size_mb:.1f} MB")

            if size_mb > 300:
                print("⚠️ 파일 크기가 300MB를 초과합니다.")
            else:
                print("✓ 파일 크기가 적정 범위입니다.")

    except subprocess.CalledProcessError as e:
        print(f"✗ 빌드 실패: {e}")
        print("오류 출력:", e.stderr)
        return False

    return True

def create_installer_script():
    """설치 스크립트 생성"""
    installer_content = '''@echo off
echo PACA v5 설치 프로그램
echo =====================

echo 설치 디렉토리 생성 중...
if not exist "%PROGRAMFILES%\\PACA v5" mkdir "%PROGRAMFILES%\\PACA v5"

echo 실행파일 복사 중...
copy "paca-v5-windows.exe" "%PROGRAMFILES%\\PACA v5\\paca-v5-windows.exe"

echo 바탕화면 바로가기 생성 중...
echo Set oWS = WScript.CreateObject("WScript.Shell") > CreateShortcut.vbs
echo sLinkFile = "%USERPROFILE%\\Desktop\\PACA v5.lnk" >> CreateShortcut.vbs
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> CreateShortcut.vbs
echo oLink.TargetPath = "%PROGRAMFILES%\\PACA v5\\paca-v5-windows.exe" >> CreateShortcut.vbs
echo oLink.Save >> CreateShortcut.vbs
cscript CreateShortcut.vbs
del CreateShortcut.vbs

echo 설치 완료!
echo PACA v5가 바탕화면에 설치되었습니다.
pause
'''

    with open('dist/install.bat', 'w', encoding='utf-8') as f:
        f.write(installer_content)

    print("✓ 설치 스크립트 생성 완료: dist/install.bat")

def create_readme():
    """사용자 가이드 README 생성"""
    readme_content = '''# PACA v5 - 개인 AI 어시스턴트

## 🚀 빠른 시작

### 1. 설치 방법

#### 자동 설치 (권장)
1. `install.bat` 파일을 관리자 권한으로 실행
2. 설치가 완료되면 바탕화면에 "PACA v5" 바로가기가 생성됩니다

#### 수동 설치
1. `paca-v5-windows.exe` 파일을 원하는 폴더에 복사
2. 더블클릭하여 실행

### 2. 사용 방법

1. **기본 대화**: 메시지 입력창에 질문이나 요청사항을 입력
2. **계산 기능**: "2 + 3 계산" 또는 "더하기" 등의 키워드 사용
3. **학습 기능**: "학습", "기억", "저장" 등의 키워드로 학습 시스템 활용
4. **도구 사용**:
   - 계산기: 별도 계산기 창 열기
   - 학습 통계: 학습 현황 확인
   - 시스템 상태: PACA 시스템 상태 확인

### 3. 주요 기능

- **채팅 인터페이스**: 직관적인 대화형 UI
- **계산 시스템**: 수학 연산 및 계산 지원
- **학습 시스템**: 사용자 패턴 학습 및 개선
- **상태 모니터링**: 시스템 성능 및 상태 실시간 확인
- **대화 저장**: 중요한 대화 내용 파일로 저장

### 4. 시스템 요구사항

- **운영체제**: Windows 10 이상
- **메모리**: 최소 4GB RAM (권장 8GB)
- **저장공간**: 최소 500MB 여유 공간
- **기타**: 인터넷 연결 (선택사항)

### 5. 문제 해결

#### 실행 오류
- Windows Defender에서 차단되는 경우: 예외 추가 설정
- "파일을 찾을 수 없습니다" 오류: 모든 파일이 같은 폴더에 있는지 확인

#### 성능 문제
- 메모리 부족: 다른 프로그램 종료 후 재실행
- 느린 응답: 시스템 상태 확인 후 재시작

### 6. 지원 및 피드백

문제가 발생하거나 개선사항이 있으시면:
- GitHub Issues 페이지에 문제 보고
- 이메일을 통한 직접 피드백

---

**PACA v5** - 한국어 특화 개인 AI 어시스턴트
버전: 5.0.0 | 빌드: 2024-09-20
'''

    with open('dist/README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print("✓ 사용자 가이드 생성 완료: dist/README.md")

def test_executable():
    """생성된 실행파일 테스트"""
    exe_path = Path('dist/paca-v5-windows.exe')

    if not exe_path.exists():
        print("✗ 실행파일을 찾을 수 없습니다.")
        return False

    print("실행파일 테스트 중...")
    print("(GUI 창이 열리면 정상 작동 중입니다. 수동으로 닫아주세요.)")

    try:
        # 테스트 실행 (타임아웃 설정)
        process = subprocess.Popen([str(exe_path)],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)

        # 3초 후 프로세스 종료
        import time
        time.sleep(3)
        process.terminate()

        print("✓ 실행파일 테스트 완료")
        return True

    except Exception as e:
        print(f"✗ 실행파일 테스트 실패: {e}")
        return False

def main():
    """메인 빌드 프로세스"""
    print("PACA v5 Windows 배포 빌드 시작")
    print("=" * 50)

    # 1. 필수 패키지 확인
    check_requirements()
    print()

    # 2. 이전 빌드 정리
    clean_build_directories()
    print()

    # 3. PyInstaller spec 파일 생성
    create_pyinstaller_spec()
    print()

    # 4. 실행파일 빌드
    if not build_executable():
        print("빌드 실패로 인해 종료합니다.")
        return
    print()

    # 5. 설치 스크립트 생성
    create_installer_script()
    print()

    # 6. 사용자 가이드 생성
    create_readme()
    print()

    # 7. 실행파일 테스트
    test_executable()
    print()

    print("=" * 50)
    print("PACA v5 Windows 빌드 완료!")
    print()
    print("생성된 파일:")
    print("- dist/paca-v5-windows.exe (실행파일)")
    print("- dist/install.bat (설치 스크립트)")
    print("- dist/README.md (사용자 가이드)")
    print()
    print("배포 준비 완료!")

if __name__ == "__main__":
    main()