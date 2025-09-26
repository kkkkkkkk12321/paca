# PACA v5 Developer Guide

## 🎯 프로젝트 개요

PACA v5 Python 개발자를 위한 완전한 기여 가이드. 코드 스타일, 아키텍처 패턴, 개발 워크플로우, 테스트 방법론을 포함한 개발 환경 구축 가이드입니다.

## 📁 개발 환경 구조

```
paca_python/
├── 📁 paca/                    # 메인 패키지
│   ├── 📁 core/                # 핵심 시스템
│   ├── 📁 cognitive/           # 인지 처리
│   ├── 📁 reasoning/           # 추론 엔진
│   ├── 📁 mathematics/         # 수학 계산
│   ├── 📁 services/            # 서비스 계층
│   ├── 📁 learning/            # 학습 시스템
│   ├── 📁 data/                # 데이터 관리
│   ├── 📁 config/              # 설정 관리
│   ├── 📁 integrations/        # 외부 통합
│   └── 📁 controllers/         # 컨트롤러
├── 📁 desktop_app/             # GUI 애플리케이션
├── 📁 scripts/                 # 개발 도구
├── 📁 tests/                   # 테스트 스위트
├── 📁 docs/                    # 문서화
├── 📄 pyproject.toml           # 프로젝트 설정
├── 📄 requirements.txt         # 의존성
└── 📄 README.md                # 프로젝트 개요
```

## ⚙️ 개발 환경 설정

### 필수 도구 설치

#### 1. Python 환경
```bash
# Python 3.9+ 설치 확인
python --version  # Python 3.9.x 이상

# 가상환경 생성
python -m venv venv

# 활성화
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

#### 2. 의존성 설치
```bash
# 기본 의존성
pip install -r requirements.txt

# 개발 의존성
pip install -r requirements-dev.txt

# 또는 한번에
pip install -e .[dev]
```

#### 3. 개발 도구 설정
```bash
# 코드 포매터
pip install black isort

# 타입 체커
pip install mypy

# 린터
pip install flake8 pylint

# 테스트
pip install pytest pytest-asyncio pytest-cov

# 문서화
pip install sphinx sphinx-rtd-theme
```

### IDE 설정

#### VS Code 설정 (.vscode/settings.json)
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.mypyEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

#### PyCharm 설정
```
1. Interpreter: 가상환경 Python 선택
2. Code Style: Black formatter 설정
3. Inspections: MyPy 타입 검사 활성화
4. Run Configuration: pytest 기본 테스트 러너
5. File Watchers: 저장시 자동 포매팅
```

## 🛠️ 코딩 표준

### 코드 스타일 가이드

#### 1. 네이밍 규칙
```python
# 클래스: PascalCase
class CognitiveProcessor:
    pass

# 함수/변수: snake_case
def process_input(user_data: str) -> Result[str]:
    processing_time = 0.5
    return Ok("processed")

# 상수: UPPER_SNAKE_CASE
MAX_PROCESSING_TIME = 30.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

# 비공개: 단일 언더스코어
class MyClass:
    def _internal_method(self):
        pass

    def __private_method(self):  # 강한 비공개
        pass
```

#### 2. 타입 힌트 필수
```python
from typing import Optional, List, Dict, Any, Union
from paca.core.types import Result

# 함수 시그니처
async def process_cognitive_task(
    context: CognitiveContext,
    processors: List[BaseCognitiveProcessor],
    options: Optional[Dict[str, Any]] = None
) -> Result[CognitiveResult]:
    """인지 작업 처리"""
    pass

# 클래스 속성
@dataclass
class CognitiveContext:
    id: str
    task_type: CognitiveTaskType
    timestamp: float
    input_data: str
    metadata: Optional[Dict[str, Any]] = None
```

#### 3. Docstring 표준
```python
def calculate_confidence(
    results: List[float],
    weights: Optional[List[float]] = None
) -> float:
    """
    결과값들의 가중 신뢰도를 계산합니다.

    Args:
        results: 계산 결과값 리스트
        weights: 가중치 리스트 (선택적)

    Returns:
        0.0~1.0 범위의 신뢰도 점수

    Raises:
        ValueError: results가 비어있을 때

    Example:
        >>> calculate_confidence([0.8, 0.9, 0.7])
        0.8
        >>> calculate_confidence([0.8, 0.9], [0.6, 0.4])
        0.84
    """
    if not results:
        raise ValueError("결과값 리스트가 비어있습니다")

    if weights is None:
        return sum(results) / len(results)

    return sum(r * w for r, w in zip(results, weights)) / sum(weights)
```

### 아키텍처 패턴

#### 1. Result 타입 패턴
```python
from paca.core.types import Result, Ok, Err

# 성공/실패를 명시적으로 처리
async def safe_operation(data: str) -> Result[ProcessedData]:
    try:
        # 데이터 검증
        if not data.strip():
            return Err("빈 데이터는 처리할 수 없습니다")

        # 실제 처리
        processed = await process_data(data)

        # 결과 검증
        if processed.confidence < 0.5:
            return Err(f"신뢰도가 너무 낮습니다: {processed.confidence}")

        return Ok(processed)

    except ProcessingError as e:
        return Err(f"처리 오류: {e}")
    except Exception as e:
        return Err(f"예상치 못한 오류: {e}")
```

#### 2. 이벤트 기반 패턴
```python
from paca.core.events import EventBus, PacaEvent

class CognitiveSystemWithEvents:
    def __init__(self):
        self.event_bus = EventBus()
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        @self.event_bus.on("cognitive_process_start")
        async def on_process_start(event: PacaEvent):
            logger.info(f"인지 처리 시작: {event.data}")

        @self.event_bus.on("cognitive_process_complete")
        async def on_process_complete(event: PacaEvent):
            # 후속 처리 트리거
            await self.event_bus.emit("reasoning_start", event.data)

    async def process(self, context: CognitiveContext) -> Result[Any]:
        # 시작 이벤트 발생
        await self.event_bus.emit("cognitive_process_start", {
            "context_id": context.id,
            "task_type": context.task_type
        })

        # 실제 처리
        result = await self._process_internal(context)

        # 완료 이벤트 발생
        await self.event_bus.emit("cognitive_process_complete", result)

        return result
```

#### 3. 의존성 주입 패턴
```python
from abc import ABC, abstractmethod
from typing import Protocol

# 인터페이스 정의
class CognitiveProcessor(Protocol):
    async def process(self, context: CognitiveContext) -> Result[Any]:
        ...

class ReasoningEngine(Protocol):
    async def reason(self, premises: List[str]) -> Result[ReasoningResult]:
        ...

# 의존성 주입을 통한 느슨한 결합
class PacaSystem:
    def __init__(
        self,
        cognitive_processor: CognitiveProcessor,
        reasoning_engine: ReasoningEngine,
        memory_service: MemoryService
    ):
        self.cognitive = cognitive_processor
        self.reasoning = reasoning_engine
        self.memory = memory_service

    async def process_comprehensive(
        self,
        input_text: str
    ) -> Result[ComprehensiveResult]:
        # 인지 처리
        cognitive_result = await self.cognitive.process(
            CognitiveContext.from_text(input_text)
        )
        if cognitive_result.is_failure:
            return Err(cognitive_result.error)

        # 추론 처리
        reasoning_result = await self.reasoning.reason(
            cognitive_result.data.extracted_premises
        )

        # 메모리 저장
        await self.memory.store_interaction(input_text, reasoning_result)

        return Ok(ComprehensiveResult(
            cognitive=cognitive_result.data,
            reasoning=reasoning_result.data
        ))
```

## 🚀 개발 워크플로우

### 1. 기능 개발 프로세스

#### Git 브랜치 전략
```bash
# 기능 브랜치 생성
git checkout -b feature/cognitive-enhancement
git checkout -b bugfix/memory-leak-fix
git checkout -b docs/api-documentation

# 개발 → 테스트 → 커밋
git add .
git commit -m "feat: 인지 처리 성능 30% 향상

- 병렬 처리 도입으로 응답 시간 단축
- 메모리 사용량 20% 감소
- 신뢰도 계산 알고리즘 최적화

Closes #123"

# 푸시 및 PR 생성
git push origin feature/cognitive-enhancement
```

#### 커밋 메시지 표준
```bash
# 타입(스코프): 간단한 설명
#
# 상세 설명 (선택적)
#
# 관련 이슈 참조 (선택적)

feat(cognitive): 새로운 추론 엔진 추가
fix(memory): 메모리 누수 문제 해결
docs(api): API 문서 업데이트
test(unit): 단위 테스트 커버리지 향상
refactor(core): 코드 구조 개선
perf(math): 수학 계산 성능 최적화
style(format): 코드 포매팅 적용
chore(deps): 의존성 업데이트
```

### 2. 테스트 주도 개발 (TDD)

#### 단위 테스트 작성
```python
# tests/test_cognitive.py
import pytest
from paca.cognitive import CognitiveSystem, CognitiveContext, CognitiveTaskType
from paca.core.types import Ok, Err

class TestCognitiveSystem:
    @pytest.fixture
    def cognitive_system(self):
        return CognitiveSystem()

    @pytest.fixture
    def sample_context(self):
        return CognitiveContext(
            id="test_001",
            task_type=CognitiveTaskType.REASONING,
            timestamp=1634567890.0,
            input_data="테스트 입력"
        )

    @pytest.mark.asyncio
    async def test_process_valid_input(self, cognitive_system, sample_context):
        """유효한 입력에 대한 정상 처리 테스트"""
        result = await cognitive_system.process(sample_context)

        assert result.is_success
        assert result.data is not None
        assert result.data.confidence > 0.0

    @pytest.mark.asyncio
    async def test_process_empty_input(self, cognitive_system):
        """빈 입력에 대한 에러 처리 테스트"""
        context = CognitiveContext(
            id="test_002",
            task_type=CognitiveTaskType.REASONING,
            timestamp=1634567890.0,
            input_data=""  # 빈 입력
        )

        result = await cognitive_system.process(context)

        assert result.is_failure
        assert "빈 입력" in result.error

    @pytest.mark.asyncio
    async def test_process_performance(self, cognitive_system, sample_context):
        """성능 요구사항 테스트"""
        import time

        start_time = time.time()
        result = await cognitive_system.process(sample_context)
        end_time = time.time()

        processing_time = (end_time - start_time) * 1000  # ms

        assert result.is_success
        assert processing_time < 120  # 120ms 이내
```

#### 통합 테스트 작성
```python
# tests/integration/test_full_pipeline.py
import pytest
from paca.system import PacaSystem

class TestFullPipeline:
    @pytest.fixture
    async def paca_system(self):
        system = PacaSystem()
        await system.initialize()
        yield system
        await system.shutdown()

    @pytest.mark.asyncio
    async def test_comprehensive_processing(self, paca_system):
        """전체 파이프라인 통합 테스트"""
        input_text = "2 + 2는 얼마인가요?"

        result = await paca_system.process_comprehensive(input_text)

        assert result.is_success
        assert "4" in result.data.response
        assert result.data.confidence > 0.8
        assert result.data.processing_time < 1000  # 1초 이내

    @pytest.mark.asyncio
    async def test_learning_and_memory(self, paca_system):
        """학습 및 기억 기능 테스트"""
        # 첫 번째 상호작용
        await paca_system.process_comprehensive("내 이름은 김철수입니다")

        # 두 번째 상호작용에서 기억 확인
        result = await paca_system.process_comprehensive("내 이름이 뭐였죠?")

        assert result.is_success
        assert "김철수" in result.data.response
```

### 3. 코드 품질 관리

#### 정적 분석 도구
```bash
# 코드 포매팅 (Black)
black paca/ tests/

# Import 정렬 (isort)
isort paca/ tests/

# 타입 체크 (MyPy)
mypy paca/ --strict

# 린팅 (Flake8)
flake8 paca/ tests/

# 코드 복잡도 (Pylint)
pylint paca/ --rcfile=.pylintrc
```

#### Pre-commit 훅 설정
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.942
    hooks:
      - id: mypy
        additional_dependencies: [types-PyYAML]
```

#### CI/CD 파이프라인 (.github/workflows/ci.yml)
```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]

    - name: Lint with flake8
      run: flake8 paca/ tests/

    - name: Type check with mypy
      run: mypy paca/ --strict

    - name: Test with pytest
      run: |
        pytest tests/ -v --cov=paca --cov-report=xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v2
```

## 📋 모듈 개발 가이드

### 새로운 모듈 추가

#### 1. 모듈 구조 생성
```bash
# 새 모듈 디렉토리 생성
mkdir paca/new_module

# 필수 파일 생성
touch paca/new_module/__init__.py
touch paca/new_module/base.py
touch paca/new_module/README.md

# 테스트 파일 생성
mkdir tests/test_new_module
touch tests/test_new_module/__init__.py
touch tests/test_new_module/test_base.py
```

#### 2. 모듈 기본 구조
```python
# paca/new_module/__init__.py
"""
새로운 모듈 - 특정 기능 담당

이 모듈은 PACA v5의 특정 기능을 담당합니다.
"""

from .base import NewModuleProcessor, NewModuleContext

__all__ = [
    "NewModuleProcessor",
    "NewModuleContext",
]
```

```python
# paca/new_module/base.py
"""새로운 모듈의 기본 클래스들"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from paca.core.types import Result
from paca.core.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class NewModuleContext:
    """새 모듈 컨텍스트"""
    id: str
    input_data: Any
    options: Optional[dict] = None

class BaseNewModuleProcessor(ABC):
    """새 모듈 프로세서 기본 클래스"""

    @abstractmethod
    async def process(self, context: NewModuleContext) -> Result[Any]:
        """
        새 모듈의 핵심 처리 로직

        Args:
            context: 처리 컨텍스트

        Returns:
            처리 결과
        """
        pass

class NewModuleProcessor(BaseNewModuleProcessor):
    """구체적인 새 모듈 프로세서 구현"""

    async def process(self, context: NewModuleContext) -> Result[Any]:
        try:
            logger.info(f"새 모듈 처리 시작: {context.id}")

            # 실제 처리 로직
            result = await self._process_internal(context)

            logger.info(f"새 모듈 처리 완료: {context.id}")
            return Result.ok(result)

        except Exception as e:
            logger.error(f"새 모듈 처리 실패: {e}")
            return Result.err(f"처리 실패: {e}")

    async def _process_internal(self, context: NewModuleContext) -> Any:
        """내부 처리 로직"""
        # 구현 필요
        pass
```

#### 3. README.md 작성 (9개 섹션 표준)
```markdown
# 🎯 프로젝트 개요
새로운 모듈의 목적과 역할 설명

# 📁 폴더/파일 구조
모듈 내부 구조와 파일 설명

# ⚙️ 기능 요구사항
입력/출력/핵심 로직 흐름

# 🛠️ 기술적 요구사항
언어/라이브러리/실행 환경

# 🚀 라우팅 및 진입점
API 경로/실행 시작점

# 📋 코드 품질 가이드
주석/네이밍/예외처리 규칙

# 🏃‍♂️ 실행 방법
설치/실행/테스트 명령어

# 🧪 테스트 방법
단위/통합/성능 테스트

# 💡 추가 고려사항
보안/성능/향후 개선
```

### API 확장 가이드

#### 1. 기존 모듈 확장
```python
# 기존 클래스 상속
from paca.cognitive.base import BaseCognitiveProcessor

class EnhancedCognitiveProcessor(BaseCognitiveProcessor):
    """향상된 인지 프로세서"""

    def __init__(self, enhancement_level: float = 1.0):
        super().__init__()
        self.enhancement_level = enhancement_level

    async def process(self, context: CognitiveContext) -> Result[Any]:
        # 기본 처리
        base_result = await super().process(context)

        if base_result.is_failure:
            return base_result

        # 향상된 처리
        enhanced_result = await self._enhance_result(
            base_result.data,
            self.enhancement_level
        )

        return Result.ok(enhanced_result)

    async def _enhance_result(self, base_result: Any, level: float) -> Any:
        """결과 향상 로직"""
        # 구현 필요
        pass
```

#### 2. 플러그인 시스템
```python
# paca/core/plugins.py
from abc import ABC, abstractmethod
from typing import Dict, List, Type

class Plugin(ABC):
    """플러그인 기본 인터페이스"""

    @property
    @abstractmethod
    def name(self) -> str:
        """플러그인 이름"""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """플러그인 버전"""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """플러그인 초기화"""
        pass

    @abstractmethod
    async def process(self, context: Any) -> Result[Any]:
        """플러그인 처리 로직"""
        pass

class PluginManager:
    """플러그인 관리자"""

    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}

    def register(self, plugin: Plugin) -> None:
        """플러그인 등록"""
        self._plugins[plugin.name] = plugin

    async def execute_plugin(self, name: str, context: Any) -> Result[Any]:
        """플러그인 실행"""
        if name not in self._plugins:
            return Result.err(f"플러그인을 찾을 수 없습니다: {name}")

        return await self._plugins[name].process(context)
```

## 🧪 테스트 전략

### 테스트 피라미드

#### 1. 단위 테스트 (80%)
```python
# 각 함수/메서드의 개별 테스트
def test_calculate_confidence():
    result = calculate_confidence([0.8, 0.9, 0.7])
    assert abs(result - 0.8) < 0.01

# 모킹을 활용한 격리 테스트
@pytest.mark.asyncio
async def test_cognitive_processor_with_mock():
    mock_memory = AsyncMock()
    mock_memory.retrieve.return_value = Result.ok("mock_data")

    processor = CognitiveProcessor(memory=mock_memory)
    result = await processor.process(sample_context)

    assert result.is_success
    mock_memory.retrieve.assert_called_once()
```

#### 2. 통합 테스트 (15%)
```python
# 여러 모듈 간의 상호작용 테스트
@pytest.mark.asyncio
async def test_cognitive_reasoning_integration():
    cognitive = CognitiveProcessor()
    reasoning = ReasoningEngine()

    # 인지 처리
    cognitive_result = await cognitive.process(context)

    # 추론 처리
    reasoning_result = await reasoning.reason(
        cognitive_result.data.premises
    )

    assert reasoning_result.is_success
    assert reasoning_result.data.confidence > 0.5
```

#### 3. E2E 테스트 (5%)
```python
# 전체 시스템 테스트
@pytest.mark.asyncio
async def test_full_system_workflow():
    system = PacaSystem()
    await system.initialize()

    # 복잡한 시나리오 테스트
    questions = [
        "2 + 2는 얼마인가요?",
        "앞의 답에 3을 곱하면?",
        "결과가 12가 맞나요?"
    ]

    for question in questions:
        result = await system.process_comprehensive(question)
        assert result.is_success

    await system.shutdown()
```

### 성능 테스트

#### 벤치마크 테스트
```python
# tests/performance/benchmark.py
import time
import asyncio
import statistics
from typing import List

async def benchmark_cognitive_processing(iterations: int = 100) -> Dict[str, float]:
    """인지 처리 성능 벤치마크"""
    system = CognitiveSystem()
    response_times = []

    for _ in range(iterations):
        start_time = time.perf_counter()

        result = await system.process(sample_context)

        end_time = time.perf_counter()
        response_times.append((end_time - start_time) * 1000)  # ms

    return {
        "mean": statistics.mean(response_times),
        "median": statistics.median(response_times),
        "p95": statistics.quantiles(response_times, n=20)[18],  # 95th percentile
        "p99": statistics.quantiles(response_times, n=100)[98],  # 99th percentile
    }

# 성능 요구사항 검증
def test_performance_requirements():
    results = asyncio.run(benchmark_cognitive_processing())

    assert results["mean"] < 120  # 평균 120ms 이내
    assert results["p95"] < 200   # 95% 200ms 이내
    assert results["p99"] < 500   # 99% 500ms 이내
```

## 💡 베스트 프랙티스

### 1. 에러 처리
```python
# 구체적인 예외 처리
try:
    result = await risky_operation()
except CognitiveError as e:
    logger.error(f"인지 처리 오류: {e}")
    return Result.err(f"인지 처리 실패: {e}")
except ReasoningError as e:
    logger.error(f"추론 오류: {e}")
    return Result.err(f"추론 실패: {e}")
except Exception as e:
    logger.error(f"예상치 못한 오류: {e}")
    return Result.err(f"시스템 오류: {e}")

# Result 타입 체이닝
async def complex_operation(input_data: str) -> Result[str]:
    return (await validate_input(input_data)
            .and_then(lambda data: process_data(data))
            .and_then(lambda processed: format_output(processed)))
```

### 2. 로깅
```python
from paca.core.utils.logger import get_logger

logger = get_logger(__name__)

async def process_with_logging(context: CognitiveContext) -> Result[Any]:
    logger.info(
        "인지 처리 시작",
        extra={
            "context_id": context.id,
            "task_type": context.task_type.value,
            "input_length": len(context.input_data)
        }
    )

    start_time = time.perf_counter()

    try:
        result = await _process_internal(context)

        processing_time = (time.perf_counter() - start_time) * 1000

        logger.info(
            "인지 처리 완료",
            extra={
                "context_id": context.id,
                "processing_time_ms": processing_time,
                "confidence": result.confidence if hasattr(result, 'confidence') else None
            }
        )

        return Result.ok(result)

    except Exception as e:
        processing_time = (time.perf_counter() - start_time) * 1000

        logger.error(
            "인지 처리 실패",
            extra={
                "context_id": context.id,
                "processing_time_ms": processing_time,
                "error": str(e)
            }
        )

        return Result.err(f"처리 실패: {e}")
```

### 3. 성능 최적화
```python
# 비동기 배치 처리
async def process_batch(contexts: List[CognitiveContext]) -> List[Result[Any]]:
    """여러 컨텍스트를 병렬로 처리"""
    tasks = [process_single(context) for context in contexts]
    return await asyncio.gather(*tasks, return_exceptions=True)

# 캐싱
from functools import lru_cache
import asyncio

class CachedProcessor:
    def __init__(self):
        self._cache = {}

    async def process(self, context: CognitiveContext) -> Result[Any]:
        # 캐시 키 생성
        cache_key = self._generate_cache_key(context)

        if cache_key in self._cache:
            logger.debug(f"캐시에서 결과 반환: {cache_key}")
            return self._cache[cache_key]

        # 실제 처리
        result = await self._process_internal(context)

        # 캐시 저장 (성공한 결과만)
        if result.is_success:
            self._cache[cache_key] = result

        return result
```

## 📖 기여 가이드라인

### Pull Request 작성

#### 1. PR 템플릿
```markdown
## 변경 사항 요약
- [ ] 새로운 기능 추가
- [ ] 버그 수정
- [ ] 성능 개선
- [ ] 문서 업데이트
- [ ] 리팩토링

## 상세 설명
변경 사항에 대한 자세한 설명

## 테스트
- [ ] 단위 테스트 추가/업데이트
- [ ] 통합 테스트 통과
- [ ] 수동 테스트 완료

## 관련 이슈
Closes #이슈번호

## 체크리스트
- [ ] 코드 스타일 준수 (black, isort)
- [ ] 타입 힌트 추가
- [ ] 문서 업데이트
- [ ] 테스트 커버리지 유지
```

#### 2. 코드 리뷰 가이드
```markdown
### 리뷰어 체크리스트
- [ ] 코드 품질: 가독성, 유지보수성
- [ ] 아키텍처: 설계 원칙 준수
- [ ] 성능: 병목지점 없음
- [ ] 보안: 취약점 없음
- [ ] 테스트: 충분한 커버리지
- [ ] 문서: API 문서 최신화
```

### 이슈 관리

#### 이슈 템플릿
```markdown
### 버그 리포트
**문제 설명**
간단하고 명확한 버그 설명

**재현 단계**
1. 이동할 페이지
2. 클릭할 버튼
3. 스크롤 위치
4. 발생하는 오류

**예상 동작**
정상적으로 작동해야 하는 방식

**실제 동작**
실제로 발생하는 문제

**환경**
- OS: [예: Windows 10]
- Python 버전: [예: 3.9.7]
- PACA 버전: [예: v5.0.0]

**추가 정보**
스크린샷, 로그 파일 등
```

---

**PACA v5 개발에 기여해주셔서 감사합니다!** 🚀

*개발 관련 문의사항은 GitHub Issues나 개발자 포럼을 이용해주세요.*