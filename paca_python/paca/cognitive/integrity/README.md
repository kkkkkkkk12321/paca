# 🎯 프로젝트 개요

PACA v5 Phase 2.3 지적 무결성 점수 (IIS) 시스템 모듈. AI의 정직성, 정확성, 투명성 등을 실시간 추적하고 점수화하여 신뢰도를 관리하는 시스템.

# 📁 폴더/파일 구조

```
paca/cognitive/integrity/
├── __init__.py                 # 모듈 초기화 및 공개 API
├── integrity_scoring.py        # IntegrityScoring 메인 클래스
├── README.md                  # 본 문서
└── __pycache__/               # Python 캐시 (자동 생성)
```

**파일별 설명:**
- `integrity_scoring.py`: 핵심 무결성 점수 엔진, 행동 추적, 점수 계산
- `__init__.py`: IntegrityScoring, BehaviorType, IntegrityDimension 등 export

# ⚙️ 기능 요구사항

## 입력
- **행동 타입**: 11가지 행동 유형 (진실탐구, 소스인용, 오류수정 등)
- **컨텍스트**: 행동의 심각도, 상황 정보
- **증거**: 행동을 뒷받침하는 구체적 증거
- **신뢰도**: 행동 기록의 확실성 정도

## 출력
- **무결성 점수**: 0-100점 실시간 점수
- **차원별 점수**: 6개 차원별 세부 점수
- **무결성 보고서**: 종합 분석 및 트렌드
- **신뢰도 점수**: 외부 시스템용 신뢰도 지표

## 핵심 로직 흐름
```
행동 관찰 → 행동 분류 → 차원별 영향 계산 → 가중평균 점수 → 트렌드 분석 → 보고서 생성
```

# 🛠️ 기술적 요구사항

## 언어 및 프레임워크
- **Python 3.8+**: 비동기 프로그래밍 지원
- **asyncio**: 비동기 행동 기록 처리
- **typing**: 타입 힌트 및 열거형 지원

## 주요 라이브러리
- `paca.core.types`: Result, OperationStatus 타입
- `paca.core.logging`: 통합 로깅 시스템
- `enum`: 행동 타입 및 차원 열거형
- `datetime`: 시간 기반 트렌드 분석

## 실행 환경
- PACA v5 시스템 내 통합 실행
- 실시간 행동 모니터링
- 지속적 점수 업데이트

# 🚀 라우팅 및 진입점

## API 진입점
```python
from paca.cognitive.integrity import IntegrityScoring, BehaviorType

# 메인 클래스 초기화
integrity_scoring = IntegrityScoring()

# 행동 기록
result = await integrity_scoring.record_behavior(
    BehaviorType.TRUTH_SEEKING,
    {'severity': 'normal'},
    ['증거 데이터']
)

# 무결성 보고서 생성
report = integrity_scoring.get_integrity_report()
```

## 주요 메서드
- `record_behavior()`: 행동 기록 및 점수 업데이트
- `get_integrity_report()`: 종합 무결성 보고서
- `detect_dishonesty()`: 부정직 패턴 감지
- `get_trust_score()`: 외부용 신뢰도 점수
- `calculate_dimension_score()`: 차원별 점수 계산

# 📋 코드 품질 가이드

## 주석 규칙
- 모든 public 메서드에 docstring 필수
- 점수 계산 로직에 상세 주석
- 행동 분류 기준 문서화

## 네이밍 규칙
- 클래스: PascalCase (IntegrityScoring)
- 함수/변수: snake_case (record_behavior)
- 상수: UPPER_SNAKE_CASE (DEFAULT_BASE_SCORE)
- 열거형: PascalCase (BehaviorType, IntegrityDimension)

## 예외 처리
- 잘못된 행동 타입 검증
- 점수 범위 유효성 검사
- Result 패턴으로 안전한 에러 처리

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
from paca.cognitive.integrity import IntegrityScoring, BehaviorType

integrity = IntegrityScoring()

# 긍정적 행동 기록
await integrity.record_behavior(
    BehaviorType.SOURCE_CITING,
    {'quality': 'high'},
    ['신뢰할 수 있는 출처 인용함']
)

# 보고서 확인
report = integrity.get_integrity_report()
print(f"현재 무결성 점수: {report['overall_metrics']['score']}")
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
- `test_behavior_recording()`: 11가지 행동 타입 기록 테스트
- `test_score_calculation()`: 6차원 점수 계산 정확성 테스트
- `test_dishonesty_detection()`: 부정직 패턴 감지 테스트
- `test_trend_analysis()`: 시간별 트렌드 분석 테스트

## 통합 테스트
- Phase 2.2 + 2.3 통합 워크플로우
- 실시간 행동 모니터링 테스트
- 대량 행동 데이터 처리 성능 테스트

## 성능 테스트
- 행동 기록 처리 속도 (<50ms 목표)
- 점수 계산 정확성 (±0.1 오차 이내)
- 메모리 효율성 테스트

# 💡 추가 고려사항

## 보안
- 행동 데이터 무결성 보장
- 점수 조작 방지 메커니즘
- 민감한 행동 정보 보호

## 성능
- 실시간 점수 계산 최적화
- 히스토리 데이터 압축 저장
- 차원별 병렬 계산 처리

## 향후 개선
- 머신러닝 기반 행동 패턴 학습
- 개인화된 무결성 기준 설정
- 실시간 무결성 모니터링 대시보드
- 다른 AI 시스템과의 무결성 점수 비교
- 무결성 점수 기반 자동 권한 조정