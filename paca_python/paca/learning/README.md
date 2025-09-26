# Learning Module - PACA Python v5 Phase 2

> Phase 2 완료: IIS 점수 계산, 자율 훈련, 전술/휴리스틱 자동 생성 시스템 추가 (2024-09-21)

## 🎯 프로젝트 개요

PACA 시스템의 자율 학습 엔진으로, AI가 스스로 약점을 분석하고 학습하여 성장하는 시스템입니다. Phase 2에서 IIS 점수 계산, 자율 훈련, 전술/휴리스틱 자동 생성 기능이 추가되었습니다. (12개 파일, 4737줄)

## 📁 폴더/파일 구조

```
paca/learning/
├── __init__.py                    # 모듈 진입점 (Phase 1 + Phase 2 통합)
├── README.md                      # 이 문서
│
├── # Phase 2 새로운 핵심 기능
├── iis_calculator.py              # IIS 점수 계산 시스템 (780줄)
├── autonomous_trainer.py          # 자율 훈련 시스템 (820줄)
├── tactic_generator.py            # 전술/휴리스틱 자동 생성 (980줄)
│
├── # Phase 1 기존 기능 (유지됨)
├── auto/
│   ├── __init__.py                # 자동 학습 시스템 진입점
│   ├── engine.py                  # 자동 학습 엔진
│   └── types.py                   # 학습 타입 정의
├── patterns/
│   ├── detector.py                # 패턴 감지기
│   └── analyzer.py                # 패턴 분석기
└── memory/
    └── storage.py                 # 학습 메모리 저장소
```

## ⚙️ 기능 요구사항

### Phase 2 핵심 기능

**입력**:
- 학습 데이터 (상호작용 횟수, 성공률, 추론 세션, 전술 사용 데이터)
- 상호작용 결과 (성공/실패, 복잡도, 추론 품질, 응답 시간)
- 약점 영역 (전술 숙련도, 문제 복잡도, 추론 품질, 학습 효율성, 적응 속도)

**출력**:
- IIS 점수 (0-100점) + 세부 분석 + 추세 + 개선 제안
- 자율 훈련 결과 (훈련 세션, 개선도, 성과 통계)
- 자동 생성된 전술/휴리스틱 (성공 패턴 기반 전술, 실패 패턴 기반 회피 규칙)

**핵심 로직 흐름**:
1. **IIS 계산**: 학습 데이터 분석 → 5개 구성 요소 점수 계산 → 가중 평균 → 최종 IIS 점수
2. **자율 훈련**: 약점 분석 → 훈련 임무 생성 → 연속 자동 훈련 실행 → 성과 평가
3. **전술 생성**: 성공 패턴 분석 → 전술 추출 → 실패 패턴 분석 → 휴리스틱 생성

## 🛠️ 기술적 요구사항

- Python 3.9+
- 외부 라이브러리: konlpy, asyncio
- 메모리 요구사항: < 100MB
- 비동기 처리 지원 (asyncio)


## 🚀 라우팅 및 진입점

**주요 클래스**:
```python
from paca.engine import AutoLearningSystem
from paca.types import LearningCategory
from paca.types import PatternType
```

**주요 함수**:
```python
get_learning_status(self)
get_generated_knowledge(self)
use(self)
```

## 📋 코드 품질 가이드

**코딩 규칙**:
- 함수명: snake_case (예: process_data, create_result)
- 클래스명: PascalCase (예: DataProcessor, ResultHandler)
- 상수명: UPPER_SNAKE_CASE (예: MAX_RETRY_COUNT)
- 비공개 멤버: _underscore_prefix

**필수 규칙**:
- 모든 public 메서드에 타입 힌트 필수
- 예외 처리: try-except 블록으로 안전성 보장
- 문서화: docstring으로 목적과 매개변수 설명
- 비동기 처리: async/await 패턴 준수
- 테스트: 모든 핵심 기능에 단위 테스트 작성

## 🏃‍♂️ 실행 방법

**설치**:
```bash
# 개발 환경 설치
pip install -e .
# 또는 의존성만 설치
pip install -r requirements.txt
```

**실행**:
```bash
# AutoLearningSystem 사용 예시
python -c "
from paca.learning import AutoLearningSystem
instance = AutoLearningSystem()
print(instance)"
```

## 🧪 테스트 방법

**단위 테스트**:
```bash
pytest tests/test_*.py -v
```

**커버리지 테스트**:
```bash
pytest --cov=paca --cov-report=html
# 결과는 htmlcov/index.html에서 확인
```

**성능 테스트**:
```bash
# 비동기 성능 테스트
python -m pytest tests/test_performance.py -v
```

## 💡 추가 고려사항

**보안**:
- 입력 데이터 검증 및 타입 안전성 보장

**성능**:
- 비동기 처리로 동시성 향상
- 메모리 효율적인 스트리밍 처리
- 복잡한 모듈이므로 캐싱 전략 고려
- 모듈 분할 및 지연 로딩 검토

**향후 개선**:
- 타입 체크 강화 (mypy strict 모드)
- 테스트 커버리지 확대 (목표: 80%+)
- 의존성 최적화 및 번들 크기 감소
- 모니터링 및 로깅 시스템 통합

---

> 이 문서는 PACA v5 Python 변환 프로젝트의 자동 문서화 시스템에 의해 생성되었습니다.
> 수정이 필요한 경우 `scripts/auto_documentation_system.py`를 통해 재생성하세요.
