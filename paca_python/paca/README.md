# PACA v5 Python Edition

## 🎯 프로젝트 개요
**Personal Adaptive Cognitive Assistant v5** - 인간 유사 인지 처리 시스템으로 ACT-R, SOAR 기반 하이브리드 아키텍처를 통해 추론, 기억, 학습을 통합 처리하는 개인화된 적응형 인지 어시스턴트입니다. TypeScript 원본을 Python으로 완전 변환하여 한국어 자연어 처리 기능을 강화하였습니다.

## 📁 폴더/파일 구조
```
paca/
├── __init__.py              # 패키지 진입점 및 공개 API (51줄)
├── __main__.py              # CLI 실행 진입점 (252줄)
├── system.py                # 통합 PACA 시스템 관리자
├── config.py                # 시스템 설정 및 구성
├── api/                     # API 및 LLM 통합 레이어
├── cognitive/               # 인지 시스템 (메모리, 추론, 인지 모델)
├── core/                    # 핵심 기반 모듈 (타입, 이벤트, 에러, 유틸리티)
├── data/                    # 데이터 관리 및 백업 시스템
├── feedback/                # 피드백 및 사용자 상호작용 분석
├── learning/                # 자동 학습 및 패턴 인식 시스템
├── mathematics/             # 수학적 계산 및 통계 분석
├── monitoring/              # 시스템 모니터링 및 성능 분석
├── performance/             # 성능 최적화 및 벤치마크
├── reasoning/               # 논리적 추론 및 체인 추론 시스템
└── services/                # 외부 서비스 통합 및 관리
```

## ⚙️ 기능 요구사항

**입력**:
- 텍스트 메시지 (한국어/영어 자연어)
- CLI 명령어 옵션 (`--interactive`, `--gui`, `--message`, `--config`)
- 설정 파일 (JSON 형식)

**출력**:
- 처리된 응답 메시지 (한국어/영어)
- 시스템 상태 및 성능 메트릭 (처리시간, 신뢰도)
- GUI 인터페이스 (옵션)

**핵심 로직 흐름**:
1. **시스템 초기화** → **메시지 수신** → **인지 처리** (복잡도 감지, 메모리 검색, 추론 실행) → **응답 생성** → **학습 및 피드백**

## 🛠️ 기술적 요구사항

**언어 및 프레임워크**:
- Python 3.9+ (asyncio, pathlib, argparse, typing)
- Windows 호환성 (UTF-8 인코딩, chcp 65001 설정)
- 포터블 저장소 시스템 (상대 경로 기반)

**핵심 라이브러리**:
- `asyncio`: 비동기 처리 및 동시성
- `pathlib`: 크로스 플랫폼 경로 처리
- `argparse`: CLI 인터페이스
- `typing`: 타입 힌트 및 안전성

**실행 환경**:
- 메모리: 최소 512MB (학습 데이터 포함시 1GB+ 권장)
- 저장소: 100MB (데이터 폴더 포함)
- OS: Windows, Linux, macOS (포터블)

## 🚀 라우팅 및 진입점

**패키지 사용**:
```python
from paca import (
    PacaSystem, PacaConfig, Message,
    CognitiveSystem, ReasoningEngine,
    Result, Status, Priority, LogLevel
)

# 시스템 초기화
config = PacaConfig()
paca = PacaSystem(config)
await paca.initialize()

# 메시지 처리
result = await paca.process_message("안녕하세요")
print(result.data["response"])
```

**CLI 실행**:
```bash
# 대화형 모드
python -m paca --interactive

# 단일 메시지 처리
python -m paca --message "복잡한 수학 문제를 해결해주세요"

# GUI 모드 (선택적)
python -m paca --gui

# 디버그 모드
python -m paca --interactive --debug --log-level DEBUG
```

## 📋 코드 품질 가이드

**네이밍 규칙**:
- 클래스: PascalCase (예: `PacaSystem`, `CognitiveSystem`)
- 함수/변수: snake_case (예: `process_message`, `user_input`)
- 상수: UPPER_SNAKE_CASE (예: `DEFAULT_CONFIG`)
- 프라이빗 멤버: `_underscore_prefix`

**필수 규칙**:
- 모든 public 메서드에 타입 힌트 필수
- 비동기 함수는 async/await 패턴 준수
- 예외 처리: try-except 블록으로 안전성 보장
- 문서화: docstring으로 목적과 매개변수 설명
- 한국어 처리: UTF-8 인코딩 및 Windows 호환성 보장

**에러 처리**:
- `PacaError`: 일반적인 시스템 오류
- `CognitiveError`: 인지 처리 관련 오류
- `ReasoningError`: 추론 시스템 오류

## 🏃‍♂️ 실행 방법

**개발 환경 설치**:
```bash
# 포터블 저장소 초기화
python setup_portable_storage.py

# 패키지 모드로 설치
pip install -e .

# 의존성 확인
python -c "import paca; print('PACA v5 설치 완료')"
```

**기본 실행**:
```bash
# 시스템 정상 동작 확인
python -m paca --version

# 대화형 모드 시작
python -m paca --interactive

# 설정 파일과 함께 실행
python -m paca --config custom_config.json --interactive
```

## 🧪 테스트 방법

**기본 테스트**:
```bash
# 포터블 기능 전체 테스트
python test_complete_portable.py

# 단위 테스트 실행
pytest tests/ -v

# 커버리지 테스트
pytest tests/ --cov=paca --cov-report=html
```

**통합 테스트**:
```bash
# Phase 1 핵심 기능 테스트
python simple_phase1_test.py

# 시스템 통합 테스트
python -c "
import asyncio
from paca import PacaSystem, PacaConfig

async def test():
    system = PacaSystem(PacaConfig())
    result = await system.initialize()
    print(f'초기화: {result.is_success}')

asyncio.run(test())
"
```

**성능 테스트**:
```bash
# 응답 시간 테스트 (목표: <1초)
python -m paca --message "테스트 메시지" --log-level DEBUG

# 메모리 사용량 테스트
python tests/performance/test_memory_usage.py
```

## 💡 추가 고려사항

**보안**:
- 사용자 입력 검증 및 정제 (XSS, 인젝션 방지)
- 민감한 정보 로깅 금지 (API 키, 개인정보)
- 포터블 저장소 권한 관리 (읽기/쓰기 권한)

**성능**:
- 비동기 처리로 응답성 향상 (목표: <1초 응답시간)
- 포터블 저장소 최적화 (JSON 기반 메모리, SQLite DB)
- 메모리 효율성 (세션당 <100MB 사용량)
- 학습 데이터 캐싱 및 지연 로딩

**향후 개선**:
- 다국어 지원 확장 (일본어, 중국어)
- 음성 인식/합성 통합
- 웹 API 서버 모드 추가
- 실시간 협업 기능
- 고급 AI 모델 통합 (GPT, Claude API)
- 모바일 앱 연동 인터페이스