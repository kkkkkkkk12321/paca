# PACA v5 System Architecture

## 🎯 프로젝트 개요

PACA v5의 전체 시스템 아키텍처 문서. ACT-R과 SOAR 이론을 기반으로 한 하이브리드 인지 아키텍처, 모듈화 설계, 이벤트 기반 통신, 그리고 확장 가능한 플러그인 시스템을 설명합니다.

## 📁 아키텍처 개요

### 계층형 아키텍처 (Layered Architecture)

```
┌─────────────────────────────────────────────────────────────┐
│                    🎨 Presentation Layer                    │
├─────────────────────────────────────────────────────────────┤
│  📱 GUI (CustomTkinter)    │  🖥️ CLI Interface              │
│  🌐 Web Interface          │  📡 API Endpoints              │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    🎯 Application Layer                     │
├─────────────────────────────────────────────────────────────┤
│  🔄 Controllers            │  🎛️ Orchestrators             │
│  📋 Use Cases              │  🔌 Plugin Manager             │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    🧠 Domain Layer                          │
├─────────────────────────────────────────────────────────────┤
│  🧩 Cognitive System      │  🤔 Reasoning Engine           │
│  📚 Learning System       │  🧮 Mathematics Module         │
│  🗄️ Memory System         │  🔍 Analysis Engine            │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                  🏗️ Infrastructure Layer                   │
├─────────────────────────────────────────────────────────────┤
│  💾 Data Storage          │  🔧 Configuration               │
│  📊 Logging & Monitoring  │  🌐 External Integrations      │
│  ⚡ Event Bus             │  🔐 Security Services           │
└─────────────────────────────────────────────────────────────┘
```

### 핵심 설계 원칙

#### 1. 모듈화 (Modularity)
- **단일 책임 원칙**: 각 모듈은 하나의 명확한 책임
- **느슨한 결합**: 모듈 간 최소한의 의존성
- **높은 응집도**: 관련 기능의 논리적 그룹화
- **인터페이스 분리**: 명확한 계약을 통한 상호작용

#### 2. 확장성 (Scalability)
- **수평적 확장**: 새로운 모듈 추가 용이
- **수직적 확장**: 기존 모듈 기능 확장 가능
- **플러그인 아키텍처**: 런타임 기능 확장
- **이벤트 기반**: 비동기 처리와 성능 최적화

#### 3. 유지보수성 (Maintainability)
- **명확한 구조**: 직관적인 폴더/파일 구조
- **문서화**: 9개 섹션 표준 문서화
- **테스트 가능성**: 단위/통합/E2E 테스트 지원
- **코드 품질**: 타입 힌트, 린팅, 포매팅

## ⚙️ 인지 아키텍처 모델

### ACT-R 기반 인지 모델

```
┌─────────────────────────────────────────────────────────────┐
│                     ACT-R Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Perceptual│    │   Cognitive │    │    Motor    │     │
│  │   Module    │    │    Module   │    │   Module    │     │
│  │             │    │             │    │             │     │
│  │ • Vision    │    │ • Working   │    │ • Response  │     │
│  │ • Audition  │    │   Memory    │    │   Planning  │     │
│  │ • Text      │    │ • Goal      │    │ • Execution │     │
│  │   Input     │    │   Buffer    │    │             │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                   │                   │          │
│         └─────────────┬─────────────┬─────────────┘         │
│                       │             │                       │
│              ┌─────────────────────────────┐                │
│              │    Declarative Memory       │                │
│              │                             │                │
│              │ • Facts & Knowledge         │                │
│              │ • Activation Spreading      │                │
│              │ • Retrieval Mechanisms      │                │
│              └─────────────────────────────┘                │
│                       │                                     │
│              ┌─────────────────────────────┐                │
│              │    Procedural Memory        │                │
│              │                             │                │
│              │ • Production Rules          │                │
│              │ • Conflict Resolution       │                │
│              │ • Learning Mechanisms       │                │
│              └─────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### SOAR 기반 문제 해결 모델

```
┌─────────────────────────────────────────────────────────────┐
│                      SOAR Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    ┌─────────────┐                         │
│                    │   Problem   │                         │
│                    │    Space    │                         │
│                    │             │                         │
│                    │ • States    │                         │
│                    │ • Operators │                         │
│                    │ • Goals     │                         │
│                    └─────────────┘                         │
│                           │                                 │
│           ┌───────────────┼───────────────┐                │
│           │               │               │                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │   Working   │ │   Long-term │ │  Chunking   │          │
│  │   Memory    │ │   Memory    │ │  Learning   │          │
│  │             │ │             │ │             │          │
│  │ • Current   │ │ • Rules     │ │ • Pattern   │          │
│  │   State     │ │ • Facts     │ │   Recognition│         │
│  │ • Goals     │ │ • Chunks    │ │ • Explanation│         │
│  │ • Context   │ │             │ │   Based     │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
│           │               │               │                │
│           └───────────────┼───────────────┘                │
│                           │                                 │
│                  ┌─────────────┐                           │
│                  │ Decision    │                           │
│                  │ Procedure   │                           │
│                  │             │                           │
│                  │ • Preference│                           │
│                  │ • Selection │                           │
│                  │ • Conflict  │                           │
│                  │   Resolution│                           │
│                  └─────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### 하이브리드 통합 모델

```python
class HybridCognitiveArchitecture:
    """ACT-R과 SOAR를 통합한 하이브리드 인지 아키텍처"""

    def __init__(self):
        # ACT-R 구성요소
        self.working_memory = WorkingMemory()
        self.declarative_memory = DeclarativeMemory()
        self.procedural_memory = ProceduralMemory()

        # SOAR 구성요소
        self.problem_space = ProblemSpace()
        self.decision_procedure = DecisionProcedure()
        self.chunking_learner = ChunkingLearner()

        # 통합 컨트롤러
        self.cognitive_controller = CognitiveController()

    async def process(self, context: CognitiveContext) -> Result[CognitiveResult]:
        """통합 인지 처리"""
        # 1. ACT-R: 지식 활성화 및 검색
        activated_knowledge = await self.declarative_memory.activate(
            context.input_data
        )

        # 2. SOAR: 문제 공간 설정
        problem_state = await self.problem_space.initialize(
            context, activated_knowledge
        )

        # 3. 하이브리드 추론
        reasoning_result = await self.cognitive_controller.reason(
            problem_state, self.procedural_memory
        )

        # 4. 학습 및 적응
        await self.chunking_learner.learn_from_experience(
            context, reasoning_result
        )

        return reasoning_result
```

## 🛠️ 핵심 모듈 아키텍처

### Core 모듈: 기반 시스템

```
📁 paca/core/
├── 📁 types/                   # 기본 타입 시스템
│   ├── 📄 base.py             # Result, Status, ID 타입
│   ├── 📄 cognitive.py        # 인지 관련 타입
│   └── 📄 reasoning.py        # 추론 관련 타입
├── 📁 events/                  # 이벤트 시스템
│   ├── 📄 base.py             # 이벤트 기본 클래스
│   ├── 📄 emitter.py          # EventEmitter 구현
│   └── 📄 bus.py              # EventBus 중앙 허브
├── 📁 utils/                   # 공통 유틸리티
│   ├── 📄 logger.py           # 구조화된 로깅
│   ├── 📄 timing.py           # 성능 측정
│   └── 📄 validation.py       # 입력 검증
└── 📁 errors/                  # 에러 시스템
    ├── 📄 base.py             # 기본 에러 클래스
    ├── 📄 cognitive.py        # 인지 관련 에러
    └── 📄 reasoning.py        # 추론 관련 에러
```

#### Result 타입 시스템
```python
from typing import TypeVar, Generic, Union
from abc import ABC, abstractmethod

T = TypeVar('T')
E = TypeVar('E')

class Result(Generic[T], ABC):
    """안전한 에러 처리를 위한 Result 타입"""

    @property
    @abstractmethod
    def is_success(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_failure(self) -> bool:
        pass

    @abstractmethod
    def and_then(self, func) -> 'Result[T]':
        """모나드 체이닝"""
        pass

class Ok(Result[T]):
    def __init__(self, data: T):
        self.data = data

    @property
    def is_success(self) -> bool:
        return True

    @property
    def is_failure(self) -> bool:
        return False

class Err(Result[T]):
    def __init__(self, error: str):
        self.error = error

    @property
    def is_success(self) -> bool:
        return False

    @property
    def is_failure(self) -> bool:
        return True
```

### Cognitive 모듈: 인지 처리 시스템

```
📁 paca/cognitive/
├── 📄 base.py                  # 기본 인지 클래스
├── 📄 processors.py            # 인지 프로세서들
├── 📄 context.py               # 인지 컨텍스트
├── 📄 memory.py                # 작업 메모리
├── 📁 models/                  # 인지 모델들
│   ├── 📄 actr.py             # ACT-R 모델
│   ├── 📄 soar.py             # SOAR 모델
│   └── 📄 hybrid.py           # 하이브리드 모델
└── 📁 processes/               # 인지 프로세스
    ├── 📄 attention.py        # 주의 집중
    ├── 📄 perception.py       # 지각 처리
    └── 📄 comprehension.py    # 이해 처리
```

#### 인지 시스템 아키텍처
```python
class CognitiveSystem:
    """인지 시스템 메인 클래스"""

    def __init__(self):
        self.processors: List[BaseCognitiveProcessor] = []
        self.working_memory = WorkingMemory()
        self.long_term_memory = LongTermMemory()
        self.event_bus = EventBus()

    async def process(self, context: CognitiveContext) -> Result[CognitiveResult]:
        """메인 인지 처리 파이프라인"""
        try:
            # 1. 전처리
            preprocessed = await self._preprocess(context)

            # 2. 적절한 프로세서 선택
            processor = await self._select_processor(preprocessed)

            # 3. 인지 처리 실행
            result = await processor.process(preprocessed)

            # 4. 후처리 및 메모리 저장
            final_result = await self._postprocess(result, context)

            return Ok(final_result)

        except Exception as e:
            return Err(f"인지 처리 실패: {e}")

    async def _select_processor(
        self, context: CognitiveContext
    ) -> BaseCognitiveProcessor:
        """컨텍스트에 적합한 프로세서 선택"""
        for processor in self.processors:
            if await processor.can_handle(context):
                return processor

        # 기본 프로세서 반환
        return self.processors[0] if self.processors else DefaultProcessor()
```

### Reasoning 모듈: 추론 엔진

```
📁 paca/reasoning/
├── 📄 base.py                  # 기본 추론 클래스
├── 📄 engines.py               # 추론 엔진들
├── 📄 chains.py                # 추론 체인 관리
├── 📄 rules.py                 # 추론 규칙
├── 📁 strategies/              # 추론 전략
│   ├── 📄 deductive.py        # 연역적 추론
│   ├── 📄 inductive.py        # 귀납적 추론
│   ├── 📄 abductive.py        # 가추법 추론
│   └── 📄 analogical.py       # 유추 추론
└── 📁 solvers/                 # 문제 해결기
    ├── 📄 logical.py          # 논리 문제 해결
    ├── 📄 mathematical.py     # 수학 문제 해결
    └── 📄 causal.py           # 인과 관계 추론
```

#### 추론 엔진 아키텍처
```python
class ReasoningEngine:
    """추론 엔진 메인 클래스"""

    def __init__(self):
        self.strategies = {
            ReasoningType.DEDUCTIVE: DeductiveStrategy(),
            ReasoningType.INDUCTIVE: InductiveStrategy(),
            ReasoningType.ABDUCTIVE: AbductiveStrategy(),
            ReasoningType.ANALOGICAL: AnalogicalStrategy(),
        }
        self.working_memory = WorkingMemory()
        self.rule_base = RuleBase()

    async def reason(
        self,
        premises: List[str],
        reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE
    ) -> Result[ReasoningResult]:
        """추론 실행"""
        try:
            # 1. 전제 파싱 및 검증
            parsed_premises = await self._parse_premises(premises)

            # 2. 추론 전략 선택
            strategy = self.strategies[reasoning_type]

            # 3. 추론 실행
            reasoning_chain = await strategy.reason(
                parsed_premises, self.rule_base
            )

            # 4. 결과 생성
            result = ReasoningResult(
                conclusion=reasoning_chain.conclusion,
                confidence=reasoning_chain.confidence,
                steps=reasoning_chain.steps,
                reasoning_type=reasoning_type
            )

            return Ok(result)

        except Exception as e:
            return Err(f"추론 실패: {e}")
```

### Mathematics 모듈: 수학 계산 시스템

```
📁 paca/mathematics/
├── 📄 calculator.py            # 기본 계산기
├── 📄 symbolic.py              # 심볼릭 계산
├── 📄 statistical.py           # 통계 분석
├── 📄 optimization.py          # 최적화 알고리즘
├── 📁 solvers/                 # 수학 해결기
│   ├── 📄 algebraic.py        # 대수 방정식
│   ├── 📄 differential.py     # 미분 방정식
│   ├── 📄 integral.py         # 적분 계산
│   └── 📄 linear.py           # 선형 시스템
└── 📁 analysis/                # 수학적 분석
    ├── 📄 numerical.py        # 수치 해석
    ├── 📄 complex.py          # 복소수 연산
    └── 📄 matrix.py           # 행렬 연산
```

## 🚀 이벤트 기반 아키텍처

### 이벤트 버스 중심 통신

```
                    ┌─────────────────┐
                    │   Event Bus     │
                    │    (Central)    │
                    └─────────┬───────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
    ┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐
    │  Cognitive    │ │   Reasoning   │ │  Learning     │
    │   Module      │ │    Module     │ │   Module      │
    └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
            │                 │                 │
    ┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐
    │  Mathematics  │ │    Memory     │ │   Services    │
    │   Module      │ │   Module      │ │   Module      │
    └───────────────┘ └───────────────┘ └───────────────┘
```

#### 이벤트 시스템 구현
```python
class EventBus:
    """중앙집중식 이벤트 버스"""

    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}
        self._middleware: List[Callable] = []

    def on(self, event_type: str):
        """이벤트 리스너 데코레이터"""
        def decorator(func):
            if event_type not in self._listeners:
                self._listeners[event_type] = []
            self._listeners[event_type].append(func)
            return func
        return decorator

    async def emit(self, event_type: str, data: Any = None):
        """이벤트 발생"""
        event = PacaEvent(
            type=event_type,
            data=data,
            timestamp=time.time(),
            id=generate_id()
        )

        # 미들웨어 적용
        for middleware in self._middleware:
            event = await middleware(event)

        # 리스너 실행
        if event_type in self._listeners:
            tasks = []
            for listener in self._listeners[event_type]:
                tasks.append(listener(event))

            await asyncio.gather(*tasks, return_exceptions=True)

# 이벤트 기반 모듈 통신 예제
@event_bus.on("cognitive_process_complete")
async def on_cognitive_complete(event: PacaEvent):
    """인지 처리 완료시 추론 시작"""
    reasoning_context = ReasoningContext.from_cognitive_result(
        event.data
    )
    await event_bus.emit("reasoning_start", reasoning_context)

@event_bus.on("reasoning_complete")
async def on_reasoning_complete(event: PacaEvent):
    """추론 완료시 학습 데이터 저장"""
    learning_data = LearningData.from_reasoning_result(
        event.data
    )
    await event_bus.emit("learning_data_available", learning_data)
```

## 📋 데이터 플로우 아키텍처

### 전체 시스템 데이터 플로우

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    User     │────▶│    Input    │────▶│ Validation  │
│   Input     │     │ Processing  │     │ & Parsing   │
└─────────────┘     └─────────────┘     └─────┬───────┘
                                              │
┌─────────────┐     ┌─────────────┐     ┌─────▼───────┐
│   Output    │◀────│ Response    │◀────│ Cognitive   │
│ Formatting  │     │ Generation  │     │ Processing  │
└─────────────┘     └─────────────┘     └─────┬───────┘
                                              │
┌─────────────┐     ┌─────────────┐     ┌─────▼───────┐
│   Memory    │◀────│ Learning &  │◀────│ Reasoning   │
│   Storage   │     │ Adaptation  │     │  Engine     │
└─────────────┘     └─────────────┘     └─────┬───────┘
                                              │
                    ┌─────────────┐     ┌─────▼───────┐
                    │ Mathematics │◀────│   Context   │
                    │ & Analysis  │     │ Enhancement │
                    └─────────────┘     └─────────────┘
```

### 마이크로서비스 스타일 데이터 플로우

```python
class DataFlowOrchestrator:
    """데이터 플로우 오케스트레이터"""

    def __init__(self):
        self.services = {
            'input': InputProcessingService(),
            'cognitive': CognitiveProcessingService(),
            'reasoning': ReasoningService(),
            'mathematics': MathematicsService(),
            'learning': LearningService(),
            'memory': MemoryService(),
            'output': OutputFormattingService()
        }

    async def process_comprehensive(
        self, user_input: str
    ) -> Result[ComprehensiveResponse]:
        """종합적 데이터 처리 파이프라인"""
        try:
            # 1. 입력 처리
            input_result = await self.services['input'].process(user_input)
            if input_result.is_failure:
                return input_result

            # 2. 인지 처리
            cognitive_result = await self.services['cognitive'].process(
                input_result.data
            )

            # 3. 추론 처리 (병렬)
            reasoning_task = self.services['reasoning'].process(
                cognitive_result.data
            )
            math_task = self.services['mathematics'].process(
                cognitive_result.data
            )

            reasoning_result, math_result = await asyncio.gather(
                reasoning_task, math_task
            )

            # 4. 결과 통합
            integrated_result = await self._integrate_results(
                cognitive_result.data,
                reasoning_result.data,
                math_result.data
            )

            # 5. 학습 및 메모리 저장 (백그라운드)
            asyncio.create_task(
                self._background_learning(integrated_result)
            )

            # 6. 출력 포매팅
            output_result = await self.services['output'].format(
                integrated_result
            )

            return output_result

        except Exception as e:
            return Err(f"종합 처리 실패: {e}")
```

## 🧪 확장성 아키텍처

### 플러그인 시스템

```python
class PluginArchitecture:
    """확장 가능한 플러그인 아키텍처"""

    def __init__(self):
        self.plugin_registry: Dict[str, Plugin] = {}
        self.hook_points: Dict[str, List[Callable]] = {}

    def register_plugin(self, plugin: Plugin):
        """플러그인 등록"""
        self.plugin_registry[plugin.name] = plugin

        # 훅 포인트 등록
        for hook_name, handler in plugin.hooks.items():
            if hook_name not in self.hook_points:
                self.hook_points[hook_name] = []
            self.hook_points[hook_name].append(handler)

    async def execute_hook(self, hook_name: str, context: Any) -> List[Any]:
        """훅 포인트 실행"""
        if hook_name not in self.hook_points:
            return []

        tasks = []
        for handler in self.hook_points[hook_name]:
            tasks.append(handler(context))

        return await asyncio.gather(*tasks, return_exceptions=True)

# 플러그인 예제
class KoreanNLPPlugin(Plugin):
    """한국어 자연어 처리 플러그인"""

    @property
    def name(self) -> str:
        return "korean_nlp"

    @property
    def hooks(self) -> Dict[str, Callable]:
        return {
            "preprocessing": self.preprocess_korean,
            "tokenization": self.tokenize_korean,
            "pos_tagging": self.pos_tag_korean
        }

    async def preprocess_korean(self, text: str) -> str:
        """한국어 전처리"""
        # 한국어 특화 전처리 로직
        return processed_text

    async def tokenize_korean(self, text: str) -> List[str]:
        """한국어 토큰화"""
        # KoNLPy 등을 활용한 토큰화
        return tokens
```

### 마이크로서비스 지원 아키텍처

```python
class MicroserviceAdapter:
    """마이크로서비스 아키텍처 어댑터"""

    def __init__(self, service_registry: ServiceRegistry):
        self.service_registry = service_registry
        self.circuit_breaker = CircuitBreaker()

    async def call_service(
        self, service_name: str, method: str, **kwargs
    ) -> Result[Any]:
        """외부 서비스 호출"""
        try:
            service_info = self.service_registry.get_service(service_name)

            if not service_info:
                return Err(f"서비스를 찾을 수 없습니다: {service_name}")

            # 서킷 브레이커 적용
            async with self.circuit_breaker.protect(service_name):
                # HTTP API 호출 또는 내부 서비스 호출
                if service_info.is_external:
                    result = await self._call_external_service(
                        service_info, method, kwargs
                    )
                else:
                    result = await self._call_internal_service(
                        service_info, method, kwargs
                    )

                return result

        except CircuitBreakerOpen:
            return Err(f"서비스 일시 중단: {service_name}")
        except Exception as e:
            return Err(f"서비스 호출 실패: {e}")
```

## 💡 성능 및 확장성 고려사항

### 성능 최적화 아키텍처

#### 1. 비동기 처리
```python
class AsyncArchitecture:
    """비동기 처리 아키텍처"""

    async def parallel_processing(
        self, tasks: List[ProcessingTask]
    ) -> List[Result]:
        """병렬 처리"""
        # CPU 집약적 작업은 ProcessPoolExecutor
        cpu_tasks = [task for task in tasks if task.is_cpu_intensive]
        # I/O 집약적 작업은 비동기 처리
        io_tasks = [task for task in tasks if task.is_io_intensive]

        # 병렬 실행
        cpu_results = await self._execute_cpu_tasks(cpu_tasks)
        io_results = await asyncio.gather(*[
            task.execute() for task in io_tasks
        ])

        return cpu_results + io_results

    async def _execute_cpu_tasks(
        self, tasks: List[ProcessingTask]
    ) -> List[Result]:
        """CPU 집약적 작업 병렬 처리"""
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor() as executor:
            futures = [
                loop.run_in_executor(executor, task.execute)
                for task in tasks
            ]
            return await asyncio.gather(*futures)
```

#### 2. 캐싱 시스템
```python
class CachingArchitecture:
    """계층형 캐싱 아키텍처"""

    def __init__(self):
        self.l1_cache = LRUCache(maxsize=1000)  # 메모리 캐시
        self.l2_cache = RedisCache()            # 분산 캐시
        self.l3_cache = DatabaseCache()         # 영구 저장소

    async def get(self, key: str) -> Optional[Any]:
        """캐시 계층별 조회"""
        # L1 캐시
        if key in self.l1_cache:
            return self.l1_cache[key]

        # L2 캐시
        l2_result = await self.l2_cache.get(key)
        if l2_result:
            self.l1_cache[key] = l2_result
            return l2_result

        # L3 캐시
        l3_result = await self.l3_cache.get(key)
        if l3_result:
            await self.l2_cache.set(key, l3_result)
            self.l1_cache[key] = l3_result
            return l3_result

        return None

    async def set(self, key: str, value: Any, ttl: int = 3600):
        """캐시 계층별 저장"""
        # 모든 계층에 저장
        self.l1_cache[key] = value
        await self.l2_cache.set(key, value, ttl)
        await self.l3_cache.set(key, value)
```

### 확장성 패턴

#### 1. 로드 밸런싱
```python
class LoadBalancer:
    """로드 밸런서"""

    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.servers = []
        self.current_index = 0

    def select_server(self) -> Server:
        """서버 선택 전략"""
        if self.strategy == "round_robin":
            return self._round_robin()
        elif self.strategy == "least_connections":
            return self._least_connections()
        elif self.strategy == "weighted":
            return self._weighted_selection()
        else:
            return self.servers[0]

    def _round_robin(self) -> Server:
        server = self.servers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.servers)
        return server
```

#### 2. 수평적 확장
```python
class HorizontalScaling:
    """수평적 확장 관리"""

    async def scale_out(self, service_name: str, target_instances: int):
        """서비스 인스턴스 증가"""
        current_instances = await self.get_current_instances(service_name)

        for i in range(target_instances - current_instances):
            await self.create_instance(service_name)
            await self.register_instance(service_name, instance_id)

    async def scale_in(self, service_name: str, target_instances: int):
        """서비스 인스턴스 감소"""
        current_instances = await self.get_current_instances(service_name)

        for i in range(current_instances - target_instances):
            instance = await self.select_instance_for_removal(service_name)
            await self.graceful_shutdown(instance)
            await self.deregister_instance(service_name, instance.id)
```

---

**PACA v5의 아키텍처는 지속적으로 진화합니다!** 🚀

*아키텍처 관련 문의사항은 개발팀 또는 GitHub Issues를 통해 연락주세요.*