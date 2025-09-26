"""
PACA v5 간단 설치 및 실행 스크립트
"""

import os
import sys
import subprocess
from pathlib import Path

def install_dependencies():
    """필수 의존성 설치"""
    dependencies = [
        "customtkinter>=5.2.0",
        "pillow>=10.0.0",
        "pydantic>=2.0.0"
    ]

    print("PACA v5 의존성 설치 중...")

    for dep in dependencies:
        try:
            print(f"설치: {dep}")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", dep
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✓ {dep} 설치 완료")
        except subprocess.CalledProcessError:
            print(f"✗ {dep} 설치 실패")

def create_launcher():
    """실행 스크립트 생성"""
    launcher_content = '''@echo off
title PACA v5 - AI Assistant

echo PACA v5 시작 중...
cd /d "%~dp0"

python desktop_app/main.py

if errorlevel 1 (
    echo.
    echo 오류가 발생했습니다.
    echo Python이 설치되어 있는지 확인해주세요.
    pause
)
'''

    with open("PACA-v5.bat", "w", encoding="utf-8") as f:
        f.write(launcher_content)

    print("✓ 실행 스크립트 생성: PACA-v5.bat")

def create_install_guide():
    """설치 가이드 생성"""
    guide_content = '''# PACA v5 설치 및 사용 가이드

## 🚀 빠른 시작

### 1. 사전 요구사항
- Python 3.8 이상 설치 필요
- Windows 10 이상 권장

### 2. 설치 방법

#### 자동 설치 (권장)
```
python simple_install.py
```

#### 수동 설치
```
pip install customtkinter pillow pydantic
```

### 3. 실행 방법

#### Windows
```
PACA-v5.bat
```

#### Python 직접 실행
```
python desktop_app/main.py
```

### 4. 주요 기능

- **채팅 인터페이스**: 직관적인 대화형 UI
- **계산 시스템**: 수학 연산 및 계산 지원
- **학습 시스템**: 사용자 패턴 학습 및 개선
- **상태 모니터링**: 시스템 성능 실시간 확인

### 5. 사용법

1. **기본 대화**: 메시지 입력창에 질문 입력
2. **계산**: "2 + 3 계산" 또는 "더하기" 키워드 사용
3. **학습**: "학습", "기억", "저장" 키워드 사용
4. **도구**: 사이드바에서 계산기, 통계 등 기능 이용

### 6. 문제 해결

#### 실행 오류
- Python 설치 확인: `python --version`
- 의존성 재설치: `python simple_install.py`

#### 성능 문제
- 다른 프로그램 종료 후 재실행
- 시스템 재시작

---

**PACA v5** - 한국어 특화 개인 AI 어시스턴트
'''

    with open("INSTALL.md", "w", encoding="utf-8") as f:
        f.write(guide_content)

    print("✓ 설치 가이드 생성: INSTALL.md")

def test_installation():
    """설치 테스트"""
    print("PACA v5 설치 테스트 중...")

    try:
        # PACA 모듈 import 테스트
        sys.path.append(str(Path.cwd()))
        from paca.mathematics import Calculator
        from paca.services.learning import LearningService

        # 기본 기능 테스트
        calc = Calculator()
        result = calc.add(2, 3)

        if result.is_success and result.value == 5:
            print("✓ 계산 시스템 정상")
        else:
            print("✗ 계산 시스템 오류")
            return False

        # 학습 서비스 테스트
        learning = LearningService()
        print("✓ 학습 서비스 정상")

        print("✓ 모든 테스트 통과!")
        return True

    except Exception as e:
        print(f"✗ 테스트 실패: {e}")
        return False

def main():
    """메인 설치 프로세스"""
    print("PACA v5 설치 프로그램")
    print("=" * 30)

    # 1. 의존성 설치
    install_dependencies()
    print()

    # 2. 실행 스크립트 생성
    create_launcher()
    print()

    # 3. 설치 가이드 생성
    create_install_guide()
    print()

    # 4. 설치 테스트
    if test_installation():
        print("=" * 30)
        print("PACA v5 설치 완료!")
        print()
        print("실행 방법:")
        print("1. PACA-v5.bat 더블클릭")
        print("2. 또는 python desktop_app/main.py")
        print()
        print("문제가 있으면 INSTALL.md를 참고하세요.")
    else:
        print("=" * 30)
        print("설치 중 문제가 발생했습니다.")
        print("INSTALL.md를 참고하여 수동 설치해주세요.")

if __name__ == "__main__":
    main()