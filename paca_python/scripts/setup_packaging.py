"""
PACA v5 패키징 시스템
PyInstaller를 사용한 실행 파일 생성 및 설치 프로그램 생성
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
import zipfile
import json


class PacaPackager:
    """PACA 애플리케이션 패키징 도구"""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.dist_dir = self.project_root / "dist"
        self.build_dir = self.project_root / "build"
        self.assets_dir = self.project_root / "desktop_app" / "assets"

        # 패키징 설정
        self.app_name = "PACA-v5"
        self.version = "5.0.0"
        self.author = "PACA Development Team"
        self.description = "Personal Adaptive Cognitive Assistant v5 - Python Edition"

    def create_executable(self) -> bool:
        """실행 파일 생성"""
        print("📦 실행 파일 생성 시작...")

        try:
            # 기존 빌드 파일 정리
            self._clean_build_dirs()

            # PyInstaller 명령어 구성
            pyinstaller_args = self._build_pyinstaller_args()

            print(f"🔧 PyInstaller 실행: {' '.join(pyinstaller_args)}")

            # PyInstaller 실행
            result = subprocess.run(pyinstaller_args, capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ 실행 파일 생성 완료!")

                # 실행 파일 정보 출력
                exe_path = self.dist_dir / f"{self.app_name}.exe"
                if exe_path.exists():
                    size_mb = exe_path.stat().st_size / 1024 / 1024
                    print(f"📄 실행 파일: {exe_path}")
                    print(f"📏 파일 크기: {size_mb:.1f}MB")

                return True
            else:
                print(f"❌ PyInstaller 실행 실패:")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ 실행 파일 생성 중 오류: {str(e)}")
            return False

    def _clean_build_dirs(self):
        """빌드 디렉토리 정리"""
        print("🧹 이전 빌드 파일 정리...")

        dirs_to_clean = [self.dist_dir, self.build_dir]

        for dir_path in dirs_to_clean:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"  정리됨: {dir_path}")

        # 디렉토리 재생성
        self.dist_dir.mkdir(exist_ok=True)
        self.build_dir.mkdir(exist_ok=True)

    def _build_pyinstaller_args(self) -> List[str]:
        """PyInstaller 명령어 인수 구성"""
        main_script = self.project_root / "desktop_app" / "main.py"

        args = [
            sys.executable, "-m", "PyInstaller",
            str(main_script),
            "--onefile",
            "--windowed",
            f"--name={self.app_name}",
            f"--distpath={self.dist_dir}",
            f"--workpath={self.build_dir}",
            "--clean",
            "--noconfirm"
        ]

        # 아이콘 파일이 있다면 추가
        icon_path = self.assets_dir / "icons" / "paca_icon.ico"
        if icon_path.exists():
            args.extend([f"--icon={icon_path}"])

        # 데이터 파일 추가
        data_additions = self._get_data_additions()
        for addition in data_additions:
            args.extend([f"--add-data={addition}"])

        # 숨겨진 import 추가
        hidden_imports = self._get_hidden_imports()
        for import_name in hidden_imports:
            args.extend([f"--hidden-import={import_name}"])

        # 제외할 모듈들
        excludes = self._get_excludes()
        for exclude in excludes:
            args.extend([f"--exclude-module={exclude}"])

        return args

    def _get_data_additions(self) -> List[str]:
        """데이터 파일 추가 목록"""
        additions = []

        # assets 디렉토리가 있다면 추가
        if self.assets_dir.exists():
            additions.append(f"{self.assets_dir};assets")

        # paca 패키지 추가
        paca_dir = self.project_root / "paca"
        if paca_dir.exists():
            additions.append(f"{paca_dir};paca")

        return additions

    def _get_hidden_imports(self) -> List[str]:
        """숨겨진 import 목록"""
        return [
            "customtkinter",
            "tkinter",
            "PIL",
            "PIL.Image",
            "PIL.ImageTk",
            "numpy",
            "sympy",
            "asyncio",
            "threading",
            "multiprocessing",
            "psutil",
            "paca",
            "paca.core",
            "paca.cognitive",
            "paca.reasoning",
            "paca.mathematics",
            "paca.services",
            "paca.config",
            "paca.data"
        ]

    def _get_excludes(self) -> List[str]:
        """제외할 모듈 목록"""
        return [
            "pytest",
            "unittest",
            "test",
            "tests",
            "setuptools",
            "pip",
            "wheel"
        ]

    def create_installer(self) -> bool:
        """설치 프로그램 생성"""
        print("📦 설치 프로그램 생성 시작...")

        try:
            # NSIS 스크립트 생성 (Windows)
            if sys.platform == "win32":
                return self._create_nsis_installer()
            else:
                print("ℹ️ 현재 Windows만 지원됩니다.")
                return False

        except Exception as e:
            print(f"❌ 설치 프로그램 생성 중 오류: {str(e)}")
            return False

    def _create_nsis_installer(self) -> bool:
        """NSIS 설치 프로그램 생성"""
        nsis_script = self._generate_nsis_script()
        script_path = self.project_root / "installer.nsi"

        # NSIS 스크립트 파일 생성
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(nsis_script)

        print(f"✅ NSIS 스크립트 생성 완료: {script_path}")
        print("ℹ️ NSIS가 설치되어 있다면 다음 명령으로 설치 프로그램을 생성할 수 있습니다:")
        print(f"   makensis {script_path}")

        return True

    def _generate_nsis_script(self) -> str:
        """NSIS 스크립트 생성"""
        exe_name = f"{self.app_name}.exe"
        installer_name = f"{self.app_name}-Setup.exe"

        return f'''
!define APPNAME "{self.app_name}"
!define APPEXE "{exe_name}"
!define VERSION "{self.version}"
!define DESCRIPTION "{self.description}"

Name "${{APPNAME}}"
OutFile "{installer_name}"
InstallDir "$PROGRAMFILES\\${{APPNAME}}"

; Request admin privileges
RequestExecutionLevel admin

Page directory
Page instfiles

Section "Install"
    SetOutPath $INSTDIR
    File "dist\\${{APPEXE}}"

    ; Create start menu shortcuts
    CreateDirectory "$SMPROGRAMS\\${{APPNAME}}"
    CreateShortCut "$SMPROGRAMS\\${{APPNAME}}\\${{APPNAME}}.lnk" "$INSTDIR\\${{APPEXE}}"

    ; Create desktop shortcut
    CreateShortCut "$DESKTOP\\${{APPNAME}}.lnk" "$INSTDIR\\${{APPEXE}}"

    ; Create uninstaller
    WriteUninstaller "$INSTDIR\\Uninstall.exe"
    CreateShortCut "$SMPROGRAMS\\${{APPNAME}}\\Uninstall.lnk" "$INSTDIR\\Uninstall.exe"

    ; Add to Add/Remove Programs
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "DisplayName" "${{APPNAME}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "UninstallString" "$\\"$INSTDIR\\Uninstall.exe$\\""
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "DisplayVersion" "${{VERSION}}"
SectionEnd

Section "Uninstall"
    Delete "$INSTDIR\\${{APPEXE}}"
    Delete "$INSTDIR\\Uninstall.exe"
    Delete "$SMPROGRAMS\\${{APPNAME}}\\${{APPNAME}}.lnk"
    Delete "$SMPROGRAMS\\${{APPNAME}}\\Uninstall.lnk"
    Delete "$DESKTOP\\${{APPNAME}}.lnk"
    RMDir "$SMPROGRAMS\\${{APPNAME}}"
    RMDir "$INSTDIR"

    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}"
SectionEnd
'''

    def create_portable_package(self) -> bool:
        """포터블 패키지 생성"""
        print("📦 포터블 패키지 생성 시작...")

        try:
            exe_path = self.dist_dir / f"{self.app_name}.exe"
            if not exe_path.exists():
                print("❌ 실행 파일이 존재하지 않습니다. 먼저 실행 파일을 생성하세요.")
                return False

            # 포터블 패키지 디렉토리 생성
            portable_dir = self.dist_dir / f"{self.app_name}-Portable"
            portable_dir.mkdir(exist_ok=True)

            # 실행 파일 복사
            shutil.copy2(exe_path, portable_dir / f"{self.app_name}.exe")

            # README 파일 생성
            readme_content = self._generate_portable_readme()
            with open(portable_dir / "README.txt", "w", encoding="utf-8") as f:
                f.write(readme_content)

            # 설정 파일 예시 생성
            config_content = self._generate_default_config()
            with open(portable_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(config_content, f, indent=2, ensure_ascii=False)

            # ZIP 파일 생성
            zip_path = self.dist_dir / f"{self.app_name}-Portable-v{self.version}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in portable_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(portable_dir)
                        zipf.write(file_path, arcname)

            size_mb = zip_path.stat().st_size / 1024 / 1024
            print(f"✅ 포터블 패키지 생성 완료!")
            print(f"📄 패키지: {zip_path}")
            print(f"📏 파일 크기: {size_mb:.1f}MB")

            return True

        except Exception as e:
            print(f"❌ 포터블 패키지 생성 중 오류: {str(e)}")
            return False

    def _generate_portable_readme(self) -> str:
        """포터블 패키지 README 생성"""
        return f"""
{self.app_name} Portable v{self.version}
{self.description}

=== 실행 방법 ===
1. {self.app_name}.exe 파일을 더블클릭하여 실행합니다.
2. 첫 실행 시 초기화에 몇 초가 소요될 수 있습니다.

=== 설정 ===
- config.json 파일을 편집하여 설정을 변경할 수 있습니다.
- 설정 변경 후 프로그램을 재시작해야 적용됩니다.

=== 시스템 요구사항 ===
- Windows 10 이상
- 메모리: 최소 2GB, 권장 4GB
- 저장 공간: 약 {300}MB

=== 문의 ===
PACA Development Team
GitHub: https://github.com/paca-team/paca-python

생성일: {Path(__file__).stat().st_mtime}
"""

    def _generate_default_config(self) -> Dict:
        """기본 설정 파일 생성"""
        return {
            "application": {
                "name": self.app_name,
                "version": self.version,
                "debug": False
            },
            "ai": {
                "enable_learning": True,
                "enable_korean_nlp": True,
                "response_timeout": 5.0,
                "quality_threshold": 0.7
            },
            "ui": {
                "theme": "dark",
                "color_scheme": "blue",
                "auto_save": True,
                "notification_sound": True
            },
            "performance": {
                "memory_limit_mb": 500,
                "max_concurrent_tasks": 4,
                "cache_size": 100
            }
        }

    def validate_build(self) -> bool:
        """빌드 검증"""
        print("🔍 빌드 검증 시작...")

        try:
            exe_path = self.dist_dir / f"{self.app_name}.exe"

            if not exe_path.exists():
                print("❌ 실행 파일이 존재하지 않습니다.")
                return False

            # 파일 크기 검증
            size_mb = exe_path.stat().st_size / 1024 / 1024
            if size_mb > 500:  # 500MB 제한
                print(f"⚠️ 실행 파일이 너무 큽니다: {size_mb:.1f}MB")
                print("   최적화를 고려해보세요.")
            else:
                print(f"✅ 파일 크기 적절: {size_mb:.1f}MB")

            # TODO: 실제 실행 테스트 (옵션)
            print("ℹ️ 실행 테스트는 수동으로 확인하세요.")

            return True

        except Exception as e:
            print(f"❌ 빌드 검증 중 오류: {str(e)}")
            return False

    def get_build_info(self) -> Dict:
        """빌드 정보 반환"""
        exe_path = self.dist_dir / f"{self.app_name}.exe"

        info = {
            "app_name": self.app_name,
            "version": self.version,
            "executable_exists": exe_path.exists(),
            "project_root": str(self.project_root),
            "dist_dir": str(self.dist_dir)
        }

        if exe_path.exists():
            info.update({
                "executable_path": str(exe_path),
                "file_size_mb": exe_path.stat().st_size / 1024 / 1024,
                "modified_time": exe_path.stat().st_mtime
            })

        return info


def main():
    """메인 실행 함수"""
    print("🚀 PACA v5 패키징 시스템 시작\n")

    packager = PacaPackager()

    # 1. 실행 파일 생성
    print("=" * 50)
    print("1️⃣ 실행 파일 생성")
    print("=" * 50)

    if packager.create_executable():
        print("✅ 실행 파일 생성 성공!\n")

        # 2. 빌드 검증
        print("=" * 50)
        print("2️⃣ 빌드 검증")
        print("=" * 50)

        if packager.validate_build():
            print("✅ 빌드 검증 성공!\n")

            # 3. 설치 프로그램 생성
            print("=" * 50)
            print("3️⃣ 설치 프로그램 생성")
            print("=" * 50)

            packager.create_installer()
            print()

            # 4. 포터블 패키지 생성
            print("=" * 50)
            print("4️⃣ 포터블 패키지 생성")
            print("=" * 50)

            packager.create_portable_package()
            print()

            # 5. 최종 정보
            print("=" * 50)
            print("5️⃣ 빌드 정보")
            print("=" * 50)

            build_info = packager.get_build_info()
            for key, value in build_info.items():
                print(f"  {key}: {value}")

            print("\n🎉 PACA v5 패키징 완료!")
            print(f"📂 결과물은 {packager.dist_dir} 디렉토리에서 확인하세요.")

        else:
            print("❌ 빌드 검증 실패")
    else:
        print("❌ 실행 파일 생성 실패")


if __name__ == "__main__":
    main()