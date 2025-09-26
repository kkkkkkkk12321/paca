# Core Errors Module - PACA Python v5

## 🎯 프로젝트 개요
PACA 시스템의 통합 에러 처리 시스템으로, 계층적 에러 클래스, 심각도 관리, 컨텍스트 정보, 복구 힌트를 제공하는 체계적인 예외 처리 프레임워크입니다. 인지 시스템과 추론 시스템의 특수한 에러 상황을 전문적으로 처리합니다.

## 📁 폴더/파일 구조
```
errors/
├── __init__.py              # 모듈 진입점 및 통합 API (81줄)
├── base.py                  # 기본 에러 클래스 및 공통 인터페이스
├── cognitive.py             # 인지 시스템 특화 에러 클래스들
├── reasoning.py             # 추론 시스템 특화 에러 클래스들
└── validation.py            # 검증 및 데이터 무결성 에러 클래스들
```

**에러 계층 구조**:
- `base.py`: `PacaError` (기본), `ApplicationError`, `InfrastructureError`, `NetworkError`
- `cognitive.py`: `CognitiveError`, `MemoryError`, `AttentionError`, `ACTRError`, `SOARError`
- `reasoning.py`: `ReasoningError`, `DeductiveReasoningError`, `ChainOfThoughtError`, `MetacognitionError`
- `validation.py`: `ValidationError`, `DataIntegrityError`, `SchemaValidationError`

## ⚙️ 기능 요구사항

**입력**:
- 에러 메시지 및 세부 정보
- 에러 심각도 (LOW, MEDIUM, HIGH, CRITICAL)
- 에러 카테고리 (VALIDATION, SYSTEM, NETWORK, AUTHENTICATION)
- 컨텍스트 정보 (컴포넌트, 단계, 메타데이터)

**출력**:
- 구조화된 에러 객체 (ID, 타임스탬프, 심각도 포함)
- 컨텍스트 정보 (모듈, 함수, 라인 번호, 파일 경로)
- 복구 힌트 및 해결 방안 제안
- JSON 직렬화 가능한 에러 정보

**핵심 로직 흐름**:
1. **에러 발생**: 예외 상황 감지 → 에러 클래스 선택 → 컨텍스트 수집
2. **에러 처리**: 심각도 평가 → 로깅 → 복구 힌트 생성 → 전파 결정
3. **에러 복구**: 복구 가능성 판단 → 자동 복구 시도 → 사용자 알림

## 🛠️ 기술적 요구사항

**언어 및 프레임워크**:
- Python 3.9+ (dataclasses, enum, typing, traceback)
- JSON 직렬화 지원 (에러 정보 저장/전송)
- UUID 기반 고유 에러 ID 생성

**에러 분류 체계**:
- **심각도**: `LOW` (정보), `MEDIUM` (경고), `HIGH` (중요), `CRITICAL` (치명적)
- **카테고리**: `VALIDATION` (검증), `SYSTEM` (시스템), `NETWORK` (네트워크), `AUTHENTICATION` (인증)
- **복구 가능성**: 자동 복구, 수동 개입 필요, 복구 불가능

**특화 에러 도메인**:
- **인지 시스템**: 메모리 오버플로우, 주의 집중 실패, 인지 모델 충돌
- **추론 시스템**: 논리적 비일관성, 추론 체인 단절, 메타인지 오류
- **학습 시스템**: 학습 데이터 부족, 모델 수렴 실패, 과적합

## 🚀 라우팅 및 진입점

**기본 에러 처리**:
```python
from paca.core.errors import (
    PacaError, ErrorSeverity, ErrorCategory,
    CognitiveError, ReasoningError, ValidationError
)

# 기본 에러 발생
try:
    # 위험한 작업
    risky_operation()
except Exception as e:
    raise PacaError(
        message="작업 실행 중 오류 발생",
        severity=ErrorSeverity.HIGH,
        category=ErrorCategory.SYSTEM,
        original_exception=e,
        recovery_hints=["시스템 재시작", "로그 확인"]
    )
```

**인지 시스템 에러 처리**:
```python
from paca.core.errors import CognitiveError, MemoryError, ACTRError

# 메모리 시스템 에러
try:
    memory_system.store_data(large_data)
except MemoryOverflowError:
    raise MemoryError(
        message="메모리 용량 초과",
        component="working_memory",
        metadata={"current_size": "1.2GB", "max_size": "1GB"},
        recovery_hints=["메모리 정리", "데이터 압축", "캐시 지우기"]
    )

# ACT-R 모델 에러
try:
    actr_model.execute_production_rule()
except ProductionRuleConflict:
    raise ACTRError(
        message="프로덕션 룰 충돌",
        model_type="ACT-R",
        rule_name="complex_reasoning_rule",
        conflict_rules=["rule_1", "rule_2"]
    )
```

**추론 시스템 에러 처리**:
```python
from paca.core.errors import (
    ReasoningError, DeductiveReasoningError,
    ChainOfThoughtError, LogicalInconsistencyError
)

# 추론 체인 에러
try:
    reasoning_chain.execute_step(step_number=5)
except ChainBreakError:
    raise ChainOfThoughtError(
        message="추론 체인 단절",
        reasoning_type="deductive",
        step_number=5,
        metadata={"previous_steps": 4, "total_steps": 10},
        recovery_hints=["이전 단계 재검토", "논리 구조 재정립"]
    )

# 논리적 비일관성
try:
    validate_logical_consistency(premises, conclusion)
except InconsistencyDetected:
    raise LogicalInconsistencyError(
        message="논리적 비일관성 감지",
        premises=premises,
        conclusion=conclusion,
        inconsistency_type="contradiction"
    )
```

**에러 컨텍스트 활용**:
```python
from paca.core.errors import ErrorContext, create_error_context

# 에러 컨텍스트 생성
def risky_function():
    try:
        # 위험한 작업
        perform_complex_calculation()
    except Exception as e:
        context = create_error_context(
            module="calculation_engine",
            function="risky_function",
            line_number=42,
            file_path="calculation.py"
        )

        raise PacaError(
            message="계산 처리 중 오류",
            context=context,
            original_exception=e
        )
```

**에러 복구 및 핸들링**:
```python
from paca.core.errors import handle_exception, is_recoverable_error

# 통합 에러 핸들러
async def safe_operation():
    try:
        return await dangerous_operation()
    except PacaError as e:
        # 구조화된 에러 처리
        await handle_exception(
            func_name="safe_operation",
            exception=e,
            severity=e.severity,
            context=e.context
        )

        # 복구 가능 여부 확인
        if is_recoverable_error(e):
            return await retry_operation()
        else:
            raise
```

## 📋 코드 품질 가이드

**네이밍 규칙**:
- 에러 클래스: PascalCase + "Error" 접미사 (예: `CognitiveError`, `ReasoningError`)
- 심각도/카테고리: UPPER_SNAKE_CASE (예: `ErrorSeverity.CRITICAL`)
- 함수: snake_case (예: `handle_exception`, `create_error_context`)
- 상수: UPPER_SNAKE_CASE (예: `MAX_ERROR_COUNT`)

**필수 규칙**:
- 모든 커스텀 에러는 `PacaError`에서 상속
- 에러 메시지는 한국어와 영어 병기 지원
- 모든 에러에 복구 힌트 제공 필수
- 민감한 정보는 에러 메시지에 포함 금지
- 에러 컨텍스트는 디버깅에 필요한 최소 정보만 포함

**에러 설계 원칙**:
- **명확성**: 에러 원인과 해결 방법이 명확해야 함
- **일관성**: 동일한 유형의 에러는 동일한 형식 사용
- **확장성**: 새로운 에러 타입 추가가 용이한 구조
- **보안성**: 에러 메시지를 통한 정보 유출 방지

## 🏃‍♂️ 실행 방법

**기본 에러 테스트**:
```bash
# 에러 클래스 로드 테스트
python -c "
from paca.core.errors import *
print(f'기본 에러: {PacaError.__name__}')
print(f'인지 에러: {CognitiveError.__name__}')
print(f'추론 에러: {ReasoningError.__name__}')
print(f'심각도: {list(ErrorSeverity)}')
print(f'카테고리: {list(ErrorCategory)}')
"

# 에러 생성 및 처리 테스트
python -c "
from paca.core.errors import PacaError, ErrorSeverity, ErrorCategory

try:
    raise PacaError(
        message='테스트 에러',
        severity=ErrorSeverity.HIGH,
        category=ErrorCategory.VALIDATION
    )
except PacaError as e:
    print(f'에러 ID: {e.error_id}')
    print(f'메시지: {e.message}')
    print(f'심각도: {e.severity.value}')
    print(f'카테고리: {e.category.value}')
"
```

**인지 시스템 에러 테스트**:
```bash
python -c "
from paca.core.errors import CognitiveError, MemoryError

# 메모리 에러 시뮬레이션
try:
    raise MemoryError(
        message='메모리 용량 초과',
        component='working_memory',
        metadata={'size': '1.5GB', 'limit': '1GB'}
    )
except CognitiveError as e:
    print(f'인지 에러: {e.message}')
    print(f'컴포넌트: {e.metadata.get(\"cognitive_component\")}')
    print(f'복구 힌트: {e.recovery_hints}')
"
```

**추론 시스템 에러 테스트**:
```bash
python -c "
from paca.core.errors import ReasoningError, ChainOfThoughtError

# 추론 체인 에러 시뮬레이션
try:
    raise ChainOfThoughtError(
        message='추론 체인 단절',
        reasoning_type='deductive',
        step_number=3,
        metadata={'total_steps': 5}
    )
except ReasoningError as e:
    print(f'추론 에러: {e.message}')
    print(f'추론 타입: {e.metadata.get(\"reasoning_type\")}')
    print(f'단계: {e.metadata.get(\"step_number\")}')
"
```

## 🧪 테스트 방법

**단위 테스트**:
```bash
# 개별 에러 클래스 테스트
pytest tests/test_core/test_errors/test_base.py -v
pytest tests/test_core/test_errors/test_cognitive.py -v
pytest tests/test_core/test_errors/test_reasoning.py -v

# 전체 errors 모듈 테스트
pytest tests/test_core/test_errors/ -v --cov=paca.core.errors
```

**통합 테스트**:
```bash
# 에러 처리 시나리오 테스트
python tests/integration/test_error_handling.py

# 에러 복구 메커니즘 테스트
python tests/integration/test_error_recovery.py
```

**성능 테스트**:
```bash
# 에러 생성 성능 테스트
python -c "
import time
from paca.core.errors import PacaError, ErrorSeverity, ErrorCategory

start = time.time()
for i in range(1000):
    try:
        raise PacaError('테스트', ErrorSeverity.LOW, ErrorCategory.VALIDATION)
    except PacaError:
        pass
end = time.time()

print(f'에러 생성 성능: {(end-start)*1000:.2f}ms (1000회)')
"

# 에러 직렬화 성능 테스트
python -c "
import json
import time
from paca.core.errors import PacaError, ErrorSeverity, ErrorCategory

error = PacaError('테스트', ErrorSeverity.HIGH, ErrorCategory.SYSTEM)

start = time.time()
for i in range(100):
    serialized = error.to_dict()
    json.dumps(serialized)
end = time.time()

print(f'에러 직렬화 성능: {(end-start)*1000:.2f}ms (100회)')
"
```

**에러 시나리오 테스트**:
```bash
# 복합 에러 상황 시뮬레이션
python tests/scenarios/test_complex_error_scenarios.py

# 에러 전파 및 처리 체인 테스트
python tests/scenarios/test_error_propagation.py
```

## 💡 추가 고려사항

**보안**:
- 에러 메시지를 통한 시스템 정보 노출 방지
- 스택 트레이스에서 민감한 변수값 필터링
- 프로덕션 환경에서 디버그 정보 제한
- 에러 로그 접근 권한 관리

**성능**:
- 에러 객체 생성 최적화 (목표: <1ms 생성 시간)
- 에러 컨텍스트 수집 최소화 (필요한 정보만)
- 에러 로깅 비동기 처리 (블로킹 방지)
- 메모리 효율적인 에러 스택 관리

**향후 개선**:
- 에러 패턴 분석 및 예측 시스템
- 자동 에러 분류 및 우선순위 결정
- 에러 기반 자동 복구 시스템
- 다국어 에러 메시지 지원 (i18n)
- 에러 시각화 및 대시보드
- 머신러닝 기반 에러 원인 분석