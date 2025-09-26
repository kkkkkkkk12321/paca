# PACA v5 설치 및 사용 가이드

## 🚀 빠른 시작

### 1. 사전 요구사항
- Python 3.8 이상 설치 필요
- Windows 10 이상 권장

### 2. 설치 방법

#### 자동 설치 (권장)
```bash
pip install customtkinter pillow pydantic
```

### 3. 실행 방법

#### Windows
```bash
PACA-v5.bat
```

#### Python 직접 실행
```bash
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

### 6. 테스트 명령어

```bash
# 기본 기능 테스트
python test_final.py

# 통합 테스트
python test_integration.py

# 간단 테스트
python test_simple.py
```

### 7. 문제 해결

#### 실행 오류
- Python 설치 확인: `python --version`
- 의존성 설치: `pip install -r requirements.txt`

#### GUI 관련 오류
- CustomTkinter 설치: `pip install customtkinter`
- PIL/Pillow 설치: `pip install pillow`

#### 성능 문제
- 다른 프로그램 종료 후 재실행
- 시스템 재시작

### 8. 프로젝트 구조

```
paca_python/
├── paca/                 # 핵심 PACA 모듈
├── desktop_app/          # GUI 애플리케이션
├── tests/               # 테스트 파일들
├── docs/                # 문서
├── scripts/             # 유틸리티 스크립트
├── requirements.txt     # 의존성 목록
├── PACA-v5.bat         # Windows 실행 스크립트
└── README.md           # 프로젝트 가이드
```

---

**PACA v5** - 한국어 특화 개인 AI 어시스턴트
버전: 5.0.0 | 빌드: 2024-09-20