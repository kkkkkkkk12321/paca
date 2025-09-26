# Core Types - Python 구현체

## 🎯 프로젝트 개요
PACA Python 시스템의 핵심 타입 정의 모듈입니다. 시스템 전반에서 사용되는 기본 데이터 타입, 인터페이스, 열거형을 정의하여 타입 안전성과 코드 일관성을 보장합니다.

## 📁 폴더/파일 구조

```
types/
├── __init__.py               # 타입 모듈 초기화 및 공통 타입 export
└── base.py                   # 기본 타입 정의 및 공통 인터페이스
```

## ⚙️ 기능 요구사항

### 입력
- **타입 정의 요청**: 새로운 데이터 타입 정의 필요
- **타입 검증 요청**: 데이터 타입 일치 검증
- **타입 변환 요청**: 타입 간 안전한 변환

### 출력
- **타입 정의**: 표준화된 데이터 타입 스키마
- **검증 결과**: 타입 안전성 검증 결과
- **변환 결과**: 타입 변환된 데이터

### 핵심 로직 흐름
1. **타입 정의** → **검증 규칙 설정** → **런타임 검증** → **타입 안전성 보장** → **오류 처리**

## 🛠️ 기술적 요구사항

### 언어 및 프레임워크
- **Python 3.9+**: 고급 타입 힌트 및 제네릭 지원
- **Typing**: 타입 힌트 및 제네릭
- **Pydantic**: 데이터 검증 및 직렬화

### 주요 기능
- **Generic Types**: 재사용 가능한 제네릭 타입
- **Union Types**: 다중 타입 지원
- **Literal Types**: 리터럴 값 타입
- **Protocol Types**: 구조적 타이핑 지원

## 🚀 라우팅 및 진입점

### 기본 타입 사용
```python
from paca.core.types import (
    CognitiveRequest, CognitiveResponse,
    MemoryItem, LearningData,
    ResultStatus, ErrorCode
)

# 인지 요청 타입
request = CognitiveRequest(
    id="req_001",
    content="분석할 텍스트",
    context={"domain": "science"},
    priority=ResultStatus.HIGH
)

# 메모리 아이템 타입
memory_item = MemoryItem[str](
    key="concept_key",
    value="개념 설명",
    metadata={"importance": 0.8, "timestamp": datetime.now()}
)

# 학습 데이터 타입
learning_data = LearningData(
    input_data="학습 입력",
    expected_output="기대 결과",
    actual_output="실제 결과",
    confidence=0.95
)
```

### 커스텀 타입 정의
```python
from paca.core.types import BaseType
from typing import Generic, TypeVar, Optional

T = TypeVar('T')

class CustomData(BaseType, Generic[T]):
    content: T
    processed: bool = False
    metadata: Optional[dict] = None

    def validate(self) -> bool:
        return self.content is not None
```

## 📋 코드 품질 가이드

### 타입 설계 원칙
- **명확성**: 타입명과 용도가 명확히 구분
- **일관성**: 전체 시스템에서 일관된 타입 규칙
- **안전성**: 런타임 타입 오류 방지

### 타입 명명 규칙
- **클래스**: PascalCase (예: CognitiveRequest, MemoryItem)
- **열거형**: PascalCase (예: ResultStatus, ErrorCode)
- **타입 변수**: 단일 대문자 (예: T, K, V)

## 🏃‍♂️ 실행 방법

### 타입 검증
```python
from paca.core.types import validate_type, TypeValidator

# 런타임 타입 검증
validator = TypeValidator()
is_valid = validator.validate(data, CognitiveRequest)

# 자동 타입 변환
converted = validator.convert(raw_data, target_type=MemoryItem[str])
```

### 타입 가드
```python
from paca.core.types import is_cognitive_request, is_memory_item

def process_data(data: Any) -> None:
    if is_cognitive_request(data):
        # data는 여기서 CognitiveRequest 타입으로 취급됨
        handle_cognitive_request(data)
    elif is_memory_item(data):
        # data는 여기서 MemoryItem 타입으로 취급됨
        handle_memory_item(data)
```

## 🧪 테스트 방법

### 단위 테스트
```bash
pytest tests/core/types/test_base.py -v
pytest tests/core/types/test_validation.py -v
pytest tests/core/types/test_conversion.py -v
```

### 타입 체킹
```bash
mypy paca/core/types/ --strict
pyright paca/core/types/
```

## 🔒 추가 고려사항

### 성능
- **타입 캐싱**: 반복적인 타입 검증 결과 캐싱
- **지연 검증**: 필요한 시점에만 타입 검증 수행
- **최적화**: 빈번히 사용되는 타입의 성능 최적화

### 호환성
- **하위 호환성**: 기존 코드와의 호환성 유지
- **확장성**: 새로운 타입의 쉬운 추가
- **상호운용성**: 다른 모듈과의 타입 호환성

### 향후 개선
- **자동 타입 추론**: 코드 분석을 통한 자동 타입 추론
- **타입 문서화**: 타입 정의 자동 문서 생성
- **런타임 최적화**: 프로덕션에서의 타입 검증 최적화
- **IDE 통합**: 개발 도구와의 향상된 통합 지원