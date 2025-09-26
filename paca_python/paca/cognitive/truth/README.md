# 🎯 프로젝트 개요

PACA v5 Phase 2.2 진실 탐구 프로토콜 (Truth Seeking Protocol) 모듈. 불확실한 정보를 감지하고 외부 검증을 통해 진실을 탐구하는 시스템.

# 📁 폴더/파일 구조

```
paca/cognitive/truth/
├── __init__.py              # 모듈 초기화 및 공개 API
├── truth_seeker.py          # TruthSeeker 메인 클래스
├── README.md               # 본 문서
└── __pycache__/            # Python 캐시 (자동 생성)
```

**파일별 설명:**
- `truth_seeker.py`: 핵심 진실 탐구 엔진, 불확실성 감지, 검증 프로세스
- `__init__.py`: TruthSeeker, UncertaintyType, 관련 타입들 export

# ⚙️ 기능 요구사항

## 입력
- **텍스트 콘텐츠**: 분석할 정보나 주장
- **컨텍스트**: 추가 메타데이터 (중요도, 도메인 등)
- **검증 옵션**: 검증 강도, 시간 제한 등

## 출력
- **불확실성 감지 결과**: 감지된 불확실성 정보와 신뢰도
- **진실 탐구 결과**: 검증 결과, 신뢰도 개선, 지식 업데이트
- **검증 통계**: 처리 시간, 수행된 검증 작업 등

## 핵심 로직 흐름
```
텍스트 입력 → 불확실성 패턴 매칭 → 우선순위 계산 → 외부 검증 → 신뢰도 업데이트 → 지식베이스 저장
```

# 🛠️ 기술적 요구사항

## 언어 및 프레임워크
- **Python 3.8+**: 비동기 프로그래밍 지원
- **asyncio**: 비동기 검증 처리
- **typing**: 타입 힌트 및 안전성

## 주요 라이브러리
- `paca.core.types`: Result, OperationStatus 타입
- `paca.core.logging`: 통합 로깅 시스템
- `paca.tools.truth_seeking`: 기존 진실 탐구 인프라

## 실행 환경
- PACA v5 시스템 내 통합 실행
- Gemini API 연동 (외부 검증용)
- SQLite 데이터베이스 (지식베이스 저장)

# 🚀 라우팅 및 진입점

## API 진입점
```python
from paca.cognitive.truth import TruthSeeker

# 메인 클래스 초기화
truth_seeker = TruthSeeker()

# 불확실성 감지
result = await truth_seeker.detect_uncertainty(content, context)

# 진실 탐구 수행
truth_result = await truth_seeker.seek_truth(query, options)
```

## 주요 메서드
- `detect_uncertainty()`: 불확실성 감지 및 분류
- `seek_truth()`: 종합적 진실 탐구 프로세스
- `verify_claim()`: 특정 주장 검증
- `get_seeking_history()`: 탐구 이력 조회
- `get_knowledge_base_stats()`: 지식베이스 통계

# 📋 코드 품질 가이드

## 주석 규칙
- 모든 public 메서드에 docstring 필수
- 복잡한 로직에 인라인 주석
- 타입 힌트 모든 함수/메서드에 적용

## 네이밍 규칙
- 클래스: PascalCase (TruthSeeker)
- 함수/변수: snake_case (detect_uncertainty)
- 상수: UPPER_SNAKE_CASE (DEFAULT_CONFIDENCE)
- 열거형: PascalCase (UncertaintyType)

## 예외 처리
- 모든 외부 API 호출에 try-catch
- Result 패턴으로 에러 반환
- 로깅을 통한 디버깅 정보 제공

# 🏃‍♂️ 실행 방법

## 설치
```bash
# 프로젝트 루트에서
cd C:\Users\kk\claude\paca\paca_python
pip install -r requirements.txt
```

## 실행
```python
# 기본 사용법
from paca.cognitive.truth import TruthSeeker

truth_seeker = TruthSeeker()
result = await truth_seeker.detect_uncertainty("I'm not sure about this claim", {})
```

## 테스트 명령어
```bash
# 단순 테스트
python simple_phase2_test.py

# 포괄적 테스트
python test_phase2_truth_integrity.py
```

# 🧪 테스트 방법

## 단위 테스트
- `test_uncertainty_detection()`: 5가지 불확실성 유형 감지 테스트
- `test_truth_seeking_process()`: 전체 진실 탐구 워크플로우 테스트
- `test_knowledge_base_updates()`: 지식베이스 업데이트 검증

## 통합 테스트
- Phase 2.2 + 2.3 통합 테스트
- LLM API 연동 테스트
- 기존 truth_seeking 도구와의 호환성 테스트

## 성능 테스트
- 불확실성 감지 응답시간 (<100ms 목표)
- 동시 검증 요청 처리 능력
- 메모리 사용량 모니터링

# 💡 추가 고려사항

## 보안
- 외부 API 호출 시 API 키 보안
- 사용자 입력 검증 및 새니타이징
- 지식베이스 접근 권한 관리

## 성능
- 불확실성 패턴 매칭 최적화
- 검증 결과 캐싱 시스템
- 비동기 처리를 통한 동시성 향상

## 향후 개선
- 머신러닝 기반 불확실성 감지
- 다국어 불확실성 패턴 지원
- 실시간 팩트체킹 API 연동
- 검증 품질 피드백 루프