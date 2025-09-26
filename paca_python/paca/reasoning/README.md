# 🎯 프로젝트 개요

PACA v5 추론 시스템 모듈 - AI 어시스턴트의 논리적 추론, 체인 추론, 병렬 추론 기능을 제공하는 핵심 모듈입니다.

## 📁 폴더/파일 구조

```
reasoning/
├── 📄 __init__.py           # 모듈 초기화 및 공개 API
├── 📄 base.py               # 기본 추론 엔진 및 인터페이스
├── 📁 chains/               # 체인 추론 시스템
├── 📁 metacognition/        # 메타인지 추론 (계획 중)
├── 📁 parallel/             # 병렬 추론 시스템 (계획 중)
└── 📄 README.md             # 이 문서
```

**파일별 설명:**
- `base.py`: ReasoningEngine, InferenceRule, 기본 추론 클래스
- `chains/`: ReasoningChainManager, 단계별 추론 체인 구현
- `metacognition/`: 자기 인식적 추론 및 모니터링 (향후 구현)
- `parallel/`: 병렬 추론 프로세스 및 동시성 관리 (향후 구현)

## ⚙️ 기능 요구사항

**입력:**
- 추론 문제 및 컨텍스트
- 추론 규칙 및 지식베이스
- 추론 전략 및 설정

**출력:**
- 추론 결과 및 확신도
- 추론 과정 및 단계별 기록
- 검증 결과 및 메타데이터

**핵심 로직 흐름:**
1. 추론 문제 분석 및 전략 선택
2. 체인 추론 또는 병렬 추론 실행
3. 각 단계별 검증 및 피드백
4. 최종 결과 종합 및 검증

## 🛠️ 기술적 요구사항

**언어 및 프레임워크:**
- Python 3.8+
- asyncio (비동기 추론 처리)
- concurrent.futures (병렬 처리)

**주요 의존성:**
- `core.types`: 기본 타입 및 ID 시스템
- `core.events`: 이벤트 기반 추론 모니터링
- `cognitive`: 인지 시스템과의 통합

**실행 환경:**
- CPU: 멀티코어 권장 (병렬 추론용)
- 메모리: 최소 256MB (추론 체인 저장용)

## 🚀 라우팅 및 진입점

**주요 진입점:**
```python
from paca.reasoning import (
    ReasoningEngine,
    ReasoningChainManager,
    DEFAULT_REASONING_CONFIG
)

# 기본 추론 엔진
engine = ReasoningEngine(config=DEFAULT_REASONING_CONFIG)

# 체인 추론 매니저
chain_manager = ReasoningChainManager()

# 추론 실행
result = await engine.reason(context, rules)
```

**API 경로:**
- `ReasoningEngine.reason()`: 메인 추론 인터페이스
- `ReasoningChainManager.execute_chain()`: 체인 추론 실행
- `DeductiveReasoningEngine.deduce()`: 연역적 추론

## 📋 코드 품질 가이드

**주석 규칙:**
- 모든 추론 메서드에 입출력 명세 필수
- 복잡한 추론 알고리즘은 수학적 표기법 사용
- 추론 단계는 순서와 조건을 명확히 기술

**네이밍 규칙:**
- 추론 타입: ReasoningType enum 사용
- 추론 단계: ReasoningStep, SubGoal 클래스
- 검증 함수: validate_* 접두사

**예외 처리:**
- ReasoningError: 일반적인 추론 오류
- ValidationError: 추론 결과 검증 실패
- TimeoutError: 추론 시간 초과

## 🏃‍♂️ 실행 방법

**설치:**
```bash
# 프로젝트 루트에서
pip install -e .
```

**기본 사용법:**
```python
import asyncio
from paca.reasoning import ReasoningEngine, ReasoningContext

async def main():
    # 추론 엔진 초기화
    engine = ReasoningEngine()

    # 추론 컨텍스트 생성
    context = ReasoningContext(
        problem="수학 문제",
        facts=["1 + 1 = 2", "2 + 2 = 4"],
        goal="3 + 3의 답 구하기"
    )

    # 추론 실행
    result = await engine.reason(context)
    print(f"추론 결과: {result.conclusion}")

# 실행
asyncio.run(main())
```

**체인 추론 사용법:**
```python
from paca.reasoning import ReasoningChainManager

# 체인 매니저 초기화
chain_manager = ReasoningChainManager()

# 체인 실행
chain_result = await chain_manager.execute_chain(
    problem="복합 문제",
    strategy="step_by_step"
)
```

## 🧪 테스트 방법

**단위 테스트:**
- 각 추론 엔진의 개별 기능 테스트
- 추론 규칙 적용 정확성 검증
- 체인 추론 단계별 검증

**통합 테스트:**
- 전체 추론 파이프라인 테스트
- 다중 추론 전략 통합 검증
- 인지 시스템과의 연동 테스트

**성능 테스트:**
- 추론 응답 시간 (<1초 목표)
- 메모리 사용량 최적화 검증
- 병렬 추론 성능 측정

**논리 정확성 테스트:**
```python
def test_deductive_reasoning():
    """연역적 추론 정확성 테스트"""
    engine = DeductiveReasoningEngine()

    facts = ["모든 사람은 죽는다", "소크라테스는 사람이다"]
    conclusion = engine.deduce(facts)

    assert "소크라테스는 죽는다" in conclusion

def test_chain_reasoning():
    """체인 추론 단계별 테스트"""
    chain = ReasoningChain()

    result = chain.add_step("전제 분석") \
                  .add_step("규칙 적용") \
                  .add_step("결론 도출") \
                  .execute()

    assert result.is_valid
    assert len(result.steps) == 3
```

## 💡 추가 고려사항

**보안:**
- 추론 규칙 무결성 검증
- 악의적인 추론 체인 방지
- 추론 결과 신뢰도 평가

**성능:**
- 추론 결과 캐싱으로 재계산 방지
- 병렬 추론으로 처리 속도 향상
- 메모리 효율적인 체인 관리

**향후 개선:**
- 확률적 추론 시스템 도입
- 기계학습 기반 추론 규칙 학습
- 설명 가능한 추론 과정 제공
- 실시간 추론 성능 모니터링

**추론 전략:**
- **체인 추론**: 단계별 순차 추론 (기본)
- **병렬 추론**: 동시 다중 경로 탐색 (성능 최적화)
- **메타인지 추론**: 추론 과정 자체를 추론 (고도화)

**품질 지표:**
- 추론 정확도: >95% (논리적 문제)
- 추론 속도: <1초 (단순 문제), <5초 (복합 문제)
- 설명 가능성: 모든 추론 단계 추적 가능