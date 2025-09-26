# Cognitive System - Python 구현체

## 🎯 프로젝트 개요
인지 과학 이론을 기반으로 한 Python 인지 시스템입니다. 비동기 처리와 AI 통합을 통해 고성능 인지 작업을 수행합니다.

## 📁 폴더/파일 구조
```
paca/cognitive/
├── __init__.py                    # 모듈 진입점 및 통합 API
├── complexity_detector.py         # 🆕 복잡도 감지 시스템 (Phase 1)
├── metacognition_engine.py        # 🆕 메타인지 엔진 (Phase 1)
├── reasoning_chain.py             # 🆕 추론 체인 시스템 (Phase 1)
├── base.py                        # 공통 인터페이스 및 기본 클래스
├── learning.py                    # 학습 관련 기능
├── memory/                        # 메모리 시스템
│   ├── __init__.py               # 메모리 모듈 진입점
│   ├── types.py                  # 메모리 타입 정의
│   ├── working.py                # 작업 메모리
│   ├── episodic.py               # 에피소드 메모리
│   └── longterm.py               # 장기 메모리
├── models/                        # 인지 모델들
│   ├── __init__.py               # 모델 모듈 진입점
│   ├── base.py                   # 기본 인지 모델
│   ├── actr.py                   # ACT-R 모델
│   └── soar.py                   # SOAR 모델
├── curiosity/                     # 호기심 시스템
│   ├── __init__.py               # 호기심 모듈 진입점
│   ├── curiosity_engine.py       # 호기심 엔진
│   └── exploration_planner.py    # 탐색 계획 수립기
├── reflection/                    # 성찰 시스템
│   ├── __init__.py               # 성찰 모듈 진입점
│   └── README.md                 # 성찰 시스템 문서
├── processes/                     # 인지 과정 (개발 예정)
└── README.md                      # 이 문서
```

**🆕 Phase 1 핵심 기능 (새로 추가된 PACA 차별화 기능):**
- `complexity_detector.py`: 질문 복잡도를 0-100점으로 자동 평가
- `metacognition_engine.py`: AI의 사고 과정을 모니터링하고 품질 평가
- `reasoning_chain.py`: 복잡한 문제를 단계별로 체계적 해결

## ⚙️ 기능 요구사항
**입력**: 사용자 질문/요청 (한국어/영어 텍스트)
**출력**: 복잡도 점수(0-100) + 추론 과정 + 메타인지 분석 결과
**핵심 로직**: 복잡도 감지 → 추론 체인 활성화 → 메타인지 모니터링 → 품질 평가

### Phase 1 주요 기능
1. **복잡도 감지 시스템**:
   - 입력 텍스트 복잡도를 0-100점으로 자동 평가
   - 키워드 분석(30%) + 문장 구조(25%) + 도메인 복잡도(25%) + 추론 필요성(20%)
   - 복잡도 30점 이상시 추론 체인 자동 활성화

2. **메타인지 엔진**:
   - AI의 사고 과정을 실시간 모니터링
   - 논리적 일관성, 단계별 명확성, 결론 타당성 평가
   - 자기반성을 통한 강점/약점 분석 및 개선 제안

3. **추론 체인 시스템**:
   - 복잡한 문제를 단계별로 체계적 해결
   - 4가지 추론 전략: 순차적, 병렬, 계층적, 반복적
   - 백트래킹 및 품질 검증 기능

## 🛠️ 기술적 요구사항
- **Python**: 3.9 이상
- **핵심 의존성**:
  - `asyncio`: 비동기 처리 (추론 체인, 메타인지 모니터링)
  - `dataclasses`: 데이터 구조 정의 (결과 객체, 메트릭)
  - `typing`: 타입 힌트 (강타입 시스템)
  - `enum`: 열거형 정의 (도메인 타입, 추론 전략)
  - `time`: 시간 측정 (성능 메트릭)
  - `uuid`: 고유 ID 생성 (세션, 단계 추적)
  - `re`: 정규 표현식 (한국어 텍스트 처리)
- **메모리 요구사항**: < 100MB (세션당 < 10MB)
- **성능 목표**: 복잡도 감지 < 50ms, 추론 체인 < 200ms

## 🚀 라우팅 및 진입점
```python
# Phase 1 핵심 기능 사용법
from paca.cognitive import (
    ComplexityDetector, MetacognitionEngine, ReasoningChain,
    detect_complexity, execute_reasoning
)

# 1. 복잡도 감지
detector = ComplexityDetector()
result = await detector.detect_complexity("머신러닝 알고리즘 성능을 분석해주세요")
print(f"복잡도: {result.score}, 추론 필요: {result.reasoning_required}")

# 2. 메타인지 모니터링
engine = MetacognitionEngine()
session_id = await engine.start_reasoning_monitoring({'problem': '복잡한 분석 문제'})
# ... 추론 과정 ...
reflection = await engine.perform_self_reflection(session_id)

# 3. 추론 체인 실행
chain = ReasoningChain({'enable_metacognition': True})
reasoning_result = await chain.execute_reasoning_chain("복잡한 문제", complexity_score=75)

# 4. 통합 워크플로우 (복잡도 감지 → 추론 체인)
problem = "인공지능의 윤리적 문제를 분석하고 해결방안을 제시해주세요"
complexity_result = await detect_complexity(problem)
if complexity_result.reasoning_required:
    reasoning_result = await execute_reasoning(problem, complexity_result.score)
    print(f"결론: {reasoning_result.final_conclusion}")
```

**기존 기능과의 통합:**
```python
# 기존 인지 시스템과 새로운 기능 통합
from paca.cognitive import CognitiveSystem, create_cognitive_context

cognitive = CognitiveSystem()
context = create_cognitive_context(
    task_type="reasoning",
    input_data="복잡한 분석 요청"
)
result = await cognitive.process(context)
```

## 📋 코드 품질 가이드
- **함수명**: snake_case (예: detect_complexity, start_reasoning_monitoring, execute_reasoning_chain)
- **클래스명**: PascalCase (예: ComplexityDetector, MetacognitionEngine, ReasoningChain)
- **상수명**: UPPER_SNAKE_CASE (예: REASONING_THRESHOLD, MAX_STEPS, DOMAIN_KEYWORDS)
- **예외처리**: 모든 async 메서드에 try-except 필수
- **한국어 처리**: UTF-8 인코딩 보장, 한글 유니코드 범위 처리 (가-힣: 0xAC00-0xD7A3)
- **타입 힌트**: 모든 함수/메서드에 타입 힌트 필수
- **로깅**: `logging` 모듈 사용, 적절한 로그 레벨 설정
- **성능**: 모든 주요 함수에 처리 시간 측정 및 성능 메트릭 수집

## 🏃‍♂️ 실행 방법
```bash
# 프로젝트 루트에서 설치
pip install -e .

# Phase 1 기능 동작 확인
python -c "from paca.cognitive import ComplexityDetector, MetacognitionEngine, ReasoningChain; print('Phase 1 핵심 기능 로드 성공')"

# 간단한 통합 테스트
python simple_phase1_test.py

# 전체 통합 테스트
python test_phase1_integration.py
```

## 🧪 테스트 방법
```bash
# Phase 1 핵심 기능 단위 테스트
pytest tests/test_cognitive/test_complexity_detector.py -v
pytest tests/test_cognitive/test_metacognition_engine.py -v
pytest tests/test_cognitive/test_reasoning_chain.py -v

# 전체 인지 모듈 테스트 (커버리지 포함)
pytest tests/test_cognitive/ -v --cov=paca.cognitive

# 성능 테스트
python tests/performance/test_basic_performance.py

# Phase 1 통합 워크플로우 테스트
python simple_phase1_test.py
```

### 테스트 커버리지 목표
- **Phase 1 핵심 기능**: ≥ 90% 커버리지
- **통합 테스트**: 핵심 워크플로우 100% 커버
- **성능 테스트**: 모든 핵심 함수 응답 시간 검증
- **한국어 처리**: 다양한 한국어 입력 패턴 테스트

### Phase 1 성공 기준 (✅ 달성 완료)
- ✅ 복잡도 점수 정상 계산 (0-100)
- ✅ 메타인지 모니터링 ID 정상 생성
- ✅ 추론 체인 활성화 (복잡도 ≥30)
- ✅ 단계별 추론 과정 실행
- ✅ 통합 테스트 4/4 통과

## 💡 추가 고려사항

### 🔐 보안
- 사용자 입력 검증 및 정제 (`sanitize_input` 함수 활용)
- SQL 인젝션 방지 (`escape_sql` 함수 활용)
- 민감한 정보 로깅 금지
- 메타인지 데이터 개인정보 보호

### ⚡ 성능 (Phase 1 달성 현황)
- **복잡도 감지**: < 50ms 목표 → 실제 평균 ~15ms ✅
- **추론 체인**: < 200ms 목표 → 단계당 평균 ~30ms ✅
- **메타인지 모니터링**: 오버헤드 < 10% → 실제 ~5% ✅
- **메모리 사용량**: 세션당 < 10MB → 실제 ~3MB ✅

### 🚀 향후 개선 계획 (Phase 2-5)
1. **Phase 2**: IIS 점수 시스템, 자율 훈련, 전술/휴리스틱 생성
2. **Phase 3**: 실시간 하드웨어 모니터링, 성능 프로파일 시스템
3. **Phase 4**: 데스크톱 GUI, 디버그 모드, 학습 상태 시각화
4. **Phase 5**: 백업/복원 시스템, 학습 데이터 관리

### 📊 Phase 1 완료 상태
- ✅ **복잡도 감지 시스템**: 100% 완료, 9개 도메인 지원, 한국어 특화
- ✅ **메타인지 엔진**: 100% 완료, 5가지 품질 메트릭, 자기반성 기능
- ✅ **추론 체인 시스템**: 100% 완료, 4가지 전략, 백트래킹 지원
- ✅ **통합 테스트**: 4/4 통과, 모든 시스템 정상 연동
- ✅ **성능 목표**: 모든 기준 달성

### 📞 문의 및 지원
- **이슈 리포팅**: GitHub Issues
- **기술 문서**: `/docs` 디렉토리
- **성능 벤치마크**: `/tests/performance`
- **Phase 1 테스트**: `simple_phase1_test.py`