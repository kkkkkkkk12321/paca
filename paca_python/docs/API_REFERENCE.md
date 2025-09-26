# PACA v5 API Reference

## 🎯 프로젝트 개요

PACA v5 Python API 완전 참조 문서. 모든 모듈, 클래스, 함수의 사용법과 예제를 포함한 개발자 가이드입니다.

## 📁 API 구조

```
paca/
├── core/                    # 핵심 시스템 API
├── cognitive/               # 인지 처리 API
├── reasoning/               # 추론 엔진 API
├── mathematics/             # 수학 계산 API
├── services/                # 서비스 관리 API
├── learning/                # 학습 시스템 API
├── data/                    # 데이터 관리 API
├── config/                  # 설정 관리 API
├── integrations/            # 외부 통합 API
├── controllers/             # 컨트롤러 API
└── system.py               # 통합 시스템 API
```

## ⚙️ 핵심 API

### Core Types API

#### Result[T] Class
**목적**: 안전한 에러 처리를 위한 결과 타입

```python
from paca.core.types import Result, Ok, Err

# 성공 결과
result = Ok("성공 데이터")
if result.is_success:
    print(result.data)  # "성공 데이터"

# 실패 결과
result = Err("에러 메시지")
if result.is_failure:
    print(result.error)  # "에러 메시지"
```

#### EventBus Class
**목적**: 시스템 간 이벤트 통신

```python
from paca.core.events import EventBus, PacaEvent

# 이벤트 버스 생성
bus = EventBus()

# 이벤트 리스너 등록
@bus.on("user_input")
async def handle_input(event: PacaEvent):
    print(f"입력 받음: {event.data}")

# 이벤트 발생
await bus.emit("user_input", {"text": "안녕하세요"})
```

### Cognitive API

#### CognitiveSystem Class
**목적**: 인지 처리 시스템 관리

```python
from paca.cognitive import CognitiveSystem, CognitiveContext, CognitiveTaskType

# 시스템 초기화
system = CognitiveSystem()

# 인지 컨텍스트 생성
context = CognitiveContext(
    id="ctx_001",
    task_type=CognitiveTaskType.REASONING,
    timestamp=1634567890.0,
    input="논리 문제를 해결해주세요"
)

# 인지 처리 실행
result = await system.process(context)
```

#### BaseCognitiveProcessor Class
**목적**: 커스텀 인지 프로세서 구현

```python
from paca.cognitive.base import BaseCognitiveProcessor
from paca.core.types import Result

class CustomProcessor(BaseCognitiveProcessor):
    async def process(self, context: CognitiveContext) -> Result[Any]:
        # 커스텀 처리 로직
        return Ok({"processed": True})

# 프로세서 등록
system.add_processor(CustomProcessor())
```

### Reasoning API

#### ReasoningEngine Class
**목적**: 논리적 추론 처리

```python
from paca.reasoning import ReasoningEngine, ReasoningType

# 추론 엔진 생성
engine = ReasoningEngine()

# 논리 추론 실행
result = await engine.reason(
    premises=["모든 사람은 죽는다", "소크라테스는 사람이다"],
    reasoning_type=ReasoningType.DEDUCTIVE
)

if result.is_success:
    print(f"결론: {result.data.conclusion}")
    print(f"신뢰도: {result.data.confidence}")
```

#### ReasoningChain Class
**목적**: 다단계 추론 체인

```python
from paca.reasoning.chains import ReasoningChain, ReasoningStep

# 추론 체인 생성
chain = ReasoningChain()

# 추론 단계 추가
chain.add_step(ReasoningStep(
    step_type="premise",
    content="전제 1: 모든 새는 날 수 있다"
))

chain.add_step(ReasoningStep(
    step_type="premise",
    content="전제 2: 펭귄은 새다"
))

# 추론 실행
result = await chain.execute()
```

### Mathematics API

#### Calculator Class
**목적**: 고급 수학 계산

```python
from paca.mathematics import Calculator

calc = Calculator()

# 기본 계산
result = await calc.calculate("2 + 3 * 4")
print(result.data)  # 14.0

# 통계 분석
stats = await calc.statistical_analysis([1, 2, 3, 4, 5])
print(f"평균: {stats.data.mean}")
print(f"표준편차: {stats.data.std}")

# 심볼릭 계산
symbolic = await calc.symbolic_solve("x^2 + 2*x + 1 = 0")
print(f"해: {symbolic.data}")
```

#### StatisticalAnalyzer Class
**목적**: 통계 분석 전문 도구

```python
from paca.mathematics import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

# 상관관계 분석
correlation = await analyzer.correlation([1, 2, 3], [2, 4, 6])
print(f"상관계수: {correlation.data}")

# 회귀 분석
regression = await analyzer.linear_regression(x_data, y_data)
print(f"기울기: {regression.data.slope}")
```

### Services API

#### ServiceManager Class
**목적**: 서비스 생명주기 관리

```python
from paca.services import ServiceManager, BaseService

# 서비스 매니저 생성
manager = ServiceManager()

# 커스텀 서비스 정의
class MyService(BaseService):
    async def start(self):
        print("서비스 시작")

    async def stop(self):
        print("서비스 중지")

# 서비스 등록 및 시작
await manager.register("my_service", MyService())
await manager.start("my_service")
```

### Learning API

#### LearningService Class
**목적**: 학습 시스템 관리

```python
from paca.services.learning import LearningService
from paca.learning import LearningStrategy

# 학습 서비스 생성
learning = LearningService()

# 학습 전략 설정
strategy = LearningStrategy(
    algorithm="reinforcement",
    parameters={"learning_rate": 0.01}
)

# 학습 실행
result = await learning.learn(
    input_data=training_data,
    strategy=strategy
)
```

### Memory API

#### MemoryService Class
**목적**: 메모리 시스템 관리

```python
from paca.services.memory import MemoryService

# 메모리 서비스 생성
memory = MemoryService()

# 데이터 저장
await memory.store("user_preference", {"theme": "dark"})

# 데이터 검색
result = await memory.retrieve("user_preference")
if result.is_success:
    print(result.data)  # {"theme": "dark"}

# 관련 메모리 검색
related = await memory.search_related("user")
```

### Configuration API

#### ConfigManager Class
**목적**: 설정 관리

```python
from paca.config import ConfigManager, ConfigFormat

# 설정 매니저 생성
config = ConfigManager()

# YAML 설정 로드
await config.load_from_file("config.yaml", ConfigFormat.YAML)

# 설정 값 조회
log_level = config.get("logging.level", default="INFO")

# 설정 값 설정
config.set("cognitive.max_processing_time", 30.0)

# 설정 저장
await config.save_to_file("config.yaml")
```

## 🛠️ 고급 API 사용법

### 비동기 처리 패턴

```python
import asyncio
from paca import PacaSystem

async def main():
    # 시스템 초기화
    system = PacaSystem()
    await system.initialize()

    # 비동기 처리 예제
    tasks = []
    for i in range(10):
        task = system.process_async(f"작업 {i}")
        tasks.append(task)

    # 모든 작업 완료 대기
    results = await asyncio.gather(*tasks)

    # 시스템 정리
    await system.shutdown()

# 실행
asyncio.run(main())
```

### 에러 처리 패턴

```python
from paca.core.types import Result
from paca.core.errors import CognitiveError, ReasoningError

async def safe_processing(input_data):
    try:
        # 인지 처리
        cognitive_result = await cognitive_system.process(input_data)
        if cognitive_result.is_failure:
            return Err(f"인지 처리 실패: {cognitive_result.error}")

        # 추론 처리
        reasoning_result = await reasoning_engine.reason(cognitive_result.data)
        if reasoning_result.is_failure:
            return Err(f"추론 실패: {reasoning_result.error}")

        return Ok(reasoning_result.data)

    except CognitiveError as e:
        return Err(f"인지 오류: {e}")
    except ReasoningError as e:
        return Err(f"추론 오류: {e}")
    except Exception as e:
        return Err(f"예상치 못한 오류: {e}")
```

### 이벤트 기반 아키텍처

```python
from paca.core.events import EventBus, PacaEvent

# 글로벌 이벤트 버스
event_bus = EventBus()

# 이벤트 핸들러 등록
@event_bus.on("cognitive_process_complete")
async def on_cognitive_complete(event: PacaEvent):
    print(f"인지 처리 완료: {event.data}")

    # 다음 단계 트리거
    await event_bus.emit("start_reasoning", event.data)

@event_bus.on("start_reasoning")
async def on_start_reasoning(event: PacaEvent):
    # 추론 시작
    reasoning_result = await reasoning_engine.reason(event.data)
    await event_bus.emit("reasoning_complete", reasoning_result)
```

## 🚀 통합 시스템 API

### PacaSystem Class
**목적**: 전체 시스템 통합 관리

```python
from paca.system import PacaSystem

# 시스템 생성 및 초기화
system = PacaSystem()
await system.initialize()

# 종합 처리 (인지 + 추론 + 학습)
result = await system.process_comprehensive(
    input_text="복잡한 문제를 해결해주세요",
    options={
        "enable_learning": True,
        "confidence_threshold": 0.8,
        "max_reasoning_steps": 5
    }
)

# 결과 확인
if result.is_success:
    print(f"응답: {result.data.response}")
    print(f"신뢰도: {result.data.confidence}")
    print(f"처리 시간: {result.data.processing_time}")
```

## 📋 API 사용 가이드라인

### 타입 힌트 사용
모든 API는 완전한 타입 힌트를 제공합니다:

```python
from typing import Optional, List, Dict, Any
from paca.core.types import Result

async def my_function(
    input_data: str,
    options: Optional[Dict[str, Any]] = None
) -> Result[List[str]]:
    # 구현
    return Ok(["결과1", "결과2"])
```

### 비동기 처리 필수
모든 주요 API는 비동기입니다:

```python
# 올바른 사용법
result = await system.process(data)

# 잘못된 사용법 (동기 호출)
# result = system.process(data)  # 에러 발생
```

### Result 타입 처리
모든 API는 Result 타입을 반환합니다:

```python
result = await some_api_call()

# 패턴 1: if 문으로 체크
if result.is_success:
    data = result.data
else:
    error = result.error

# 패턴 2: match 문 사용 (Python 3.10+)
match result:
    case Ok(data):
        print(f"성공: {data}")
    case Err(error):
        print(f"실패: {error}")
```

## 🧪 API 테스트 방법

### 단위 테스트 예제

```python
import pytest
from paca.cognitive import CognitiveSystem
from paca.core.types import Ok, Err

@pytest.mark.asyncio
async def test_cognitive_processing():
    system = CognitiveSystem()

    # 정상 케이스
    result = await system.process("간단한 질문")
    assert result.is_success

    # 에러 케이스
    result = await system.process("")
    assert result.is_failure
    assert "빈 입력" in result.error
```

### 통합 테스트 예제

```python
@pytest.mark.asyncio
async def test_full_pipeline():
    system = PacaSystem()
    await system.initialize()

    # 전체 파이프라인 테스트
    result = await system.process_comprehensive("복잡한 문제")

    assert result.is_success
    assert result.data.confidence > 0.5
    assert result.data.processing_time < 1000  # 1초 이내

    await system.shutdown()
```

## 💡 API 최적화 팁

### 성능 최적화
1. **배치 처리**: 여러 요청을 묶어서 처리
2. **캐싱 활용**: 동일한 입력에 대한 결과 재사용
3. **비동기 병렬 처리**: asyncio.gather() 활용

### 메모리 관리
1. **적절한 정리**: await system.shutdown() 호출
2. **큰 데이터 스트리밍**: 청크 단위 처리
3. **가비지 컬렉션**: gc.collect() 필요시 호출

### 에러 처리
1. **구체적 예외**: 특정 에러 타입 catch
2. **로깅**: 모든 에러 상황 기록
3. **복구 전략**: 실패시 대안 방법 구현

## 📖 버전 정보

- **API 버전**: v5.0.0
- **Python 요구사항**: 3.9+
- **마지막 업데이트**: 2024-09-20
- **호환성**: TypeScript v4.x API와 95% 호환