# Core Constants Module - PACA Python v5

## 🎯 프로젝트 개요
PACA 시스템의 모든 상수 정의를 중앙 집중화한 모듈로, 설정값, 제한값, 메시지, 경로 등 시스템 전반에서 사용되는 상수들을 체계적으로 관리합니다. 포터블 저장소와 통합되어 동적 경로 계산을 지원합니다.

## 📁 폴더/파일 구조
```
constants/
├── __init__.py              # 모듈 진입점 및 통합 API (84줄)
├── config.py                # 기본 설정 상수들 (타임아웃, 환경, 설정 클래스)
├── limits.py                # 시스템 제한값 상수들 (메모리, CPU, 파일, 네트워크)
├── messages.py              # 메시지 템플릿 상수들 (에러, 성공, 알림 메시지)
└── paths.py                 # 경로 및 URL 상수들 (API, 파일, 캐시, DB 경로)
```

**파일별 주요 클래스**:
- `config.py`: `DatabaseConfig`, `AIModelConfig`, `MonitoringConfig`, `SecurityConfig`
- `limits.py`: `MemoryLimits`, `CpuLimits`, `FileSizeLimits`, `RateLimits`, `ReasoningLimits`
- `messages.py`: `ErrorMessages`, `SuccessMessages`, `StatusMessages`, `LogTemplates`
- `paths.py`: `ApiEndpoints`, `FilePaths`, `CacheKeys`, `DatabaseTables`, `ExternalUrls`

## ⚙️ 기능 요구사항

**입력**:
- 환경 변수 (PYTHON_ENV, API 키, 디버그 모드)
- 설정 매개변수 (템플릿 포맷팅용 키-값 쌍)
- 검증할 값 (메모리, CPU, 파일 크기 등)

**출력**:
- 시스템 상수값 (타임아웃, 제한값, 경로)
- 포맷팅된 메시지 (에러, 성공, 상태 메시지)
- 동적 계산된 포터블 경로 (데이터, 로그, 캐시, DB)
- 환경별 설정값 (개발/프로덕션/테스트)

**핵심 로직 흐름**:
1. **상수 로드**: 모듈 임포트 → 환경 감지 → 상수 초기화 → 설정 검증
2. **경로 계산**: 포터블 저장소 → 동적 경로 → 디렉토리 생성 → 경로 반환
3. **메시지 포맷**: 템플릿 선택 → 매개변수 바인딩 → 포맷팅 → 메시지 반환

## 🛠️ 기술적 요구사항

**언어 및 프레임워크**:
- Python 3.9+ (typing, pathlib, os, dataclasses)
- 환경 변수 기반 설정 관리
- 포터블 저장소 통합 (동적 경로)

**핵심 상수 카테고리**:
- **성능 제한**: 메모리(1GB), CPU(80%), 파일 크기(100MB)
- **네트워크**: API 타임아웃(60초), 재시도(3회), 캐시 TTL(5분)
- **데이터베이스**: 연결 타임아웃(10초), 쿼리 제한(1000행)
- **인지 시스템**: 복잡도 임계값(30), 추론 단계 제한(10)

**환경별 설정**:
- **개발**: 디버그 모드, 상세 로깅, 낮은 제한값
- **프로덕션**: 최적화 모드, 보안 강화, 높은 성능
- **테스트**: 격리 환경, 모의 설정, 빠른 실행

## 🚀 라우팅 및 진입점

**기본 상수 사용**:
```python
from paca.core.constants import (
    DEFAULT_TIMEOUT, MAX_RETRY_COUNT, MEMORY_LIMITS,
    API_ENDPOINTS, FILE_PATHS, ERROR_MESSAGES
)

# 타임아웃 설정
async def api_call():
    async with aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT) as session:
        # API 호출 로직
        pass

# 메모리 제한 확인
if current_memory > MEMORY_LIMITS.MAX_HEAP_SIZE:
    raise MemoryError("메모리 한계 초과")
```

**동적 경로 사용**:
```python
from paca.core.constants import FILE_PATHS, get_data_file_path

# 포터블 경로 가져오기
config_file = FILE_PATHS.get_config_file()
data_dir = FILE_PATHS.get_data_dir()
log_file = get_log_file_path("error.log")

# 데이터베이스 경로
db_path = FILE_PATHS.get_main_db()
```

**메시지 포맷팅**:
```python
from paca.core.constants import ERROR_MESSAGES, format_message

# 에러 메시지 생성
error_msg = format_message(
    ERROR_MESSAGES.VALIDATION_FAILED,
    {"field": "사용자명", "value": "invalid_name"}
)

# 성공 메시지
success_msg = get_success_message("USER_CREATED", {"username": "홍길동"})
```

**설정 클래스 사용**:
```python
from paca.core.constants import DatabaseConfig, AIModelConfig

# 데이터베이스 설정
db_config = DatabaseConfig(
    host="localhost",
    port=5432,
    timeout=DB_TIMEOUT,
    max_connections=20
)

# AI 모델 설정
model_config = AIModelConfig(
    model_name="gpt-4",
    max_tokens=4096,
    temperature=0.7,
    timeout=API_TIMEOUT
)
```

**환경별 설정**:
```python
from paca.core.constants import is_production, is_development, get_env_config

# 환경 확인
if is_production():
    log_level = "ERROR"
    debug_mode = False
elif is_development():
    log_level = "DEBUG"
    debug_mode = True

# 환경별 설정 로드
env_config = get_env_config()
```

## 📋 코드 품질 가이드

**네이밍 규칙**:
- 상수: UPPER_SNAKE_CASE (예: `MAX_RETRY_COUNT`, `DEFAULT_TIMEOUT`)
- 클래스: PascalCase (예: `MemoryLimits`, `ApiEndpoints`)
- 함수: snake_case (예: `format_message`, `get_env_config`)
- 모듈: snake_case.py (예: `config.py`, `limits.py`)

**필수 규칙**:
- 모든 상수에 타입 힌트 및 Final 지정 필수
- 환경별 설정 지원 (개발/프로덕션/테스트)
- 포터블 경로만 사용 (하드코딩된 절대 경로 금지)
- 메시지 템플릿은 다국어 지원 고려
- 제한값은 성능과 안정성 균형 유지

**상수 설계 원칙**:
- **그룹화**: 관련 상수들을 클래스로 그룹화
- **일관성**: 동일한 단위와 명명 규칙 사용
- **확장성**: 새로운 상수 추가가 용이한 구조
- **검증**: 잘못된 상수 사용 방지를 위한 검증 함수

## 🏃‍♂️ 실행 방법

**기본 설정 확인**:
```bash
# 환경 설정 확인
python -c "
from paca.core.constants import is_production, DEFAULT_TIMEOUT, MEMORY_LIMITS
print(f'환경: {\"프로덕션\" if is_production() else \"개발\"}')
print(f'기본 타임아웃: {DEFAULT_TIMEOUT}초')
print(f'최대 메모리: {MEMORY_LIMITS.MAX_HEAP_SIZE // (1024**3)}GB')
"

# 포터블 경로 확인
python -c "
from paca.core.constants import FILE_PATHS
print(f'데이터 디렉토리: {FILE_PATHS.get_data_dir()}')
print(f'로그 디렉토리: {FILE_PATHS.get_logs_dir()}')
print(f'캐시 디렉토리: {FILE_PATHS.get_cache_dir()}')
"
```

**메시지 템플릿 테스트**:
```bash
python -c "
from paca.core.constants import ERROR_MESSAGES, format_template
template = ERROR_MESSAGES.CONNECTION_FAILED
params = {'host': 'localhost', 'port': 8080, 'error': 'Connection refused'}
message = format_template(template, params)
print(f'에러 메시지: {message}')
"
```

**제한값 검증 테스트**:
```bash
python -c "
from paca.core.constants import MEMORY_LIMITS, validate_limit
import psutil

current_memory = psutil.virtual_memory().used
max_memory = MEMORY_LIMITS.MAX_HEAP_SIZE

try:
    validate_limit(current_memory, max_memory, '메모리 사용량 초과')
    print('메모리 사용량 정상')
except ValueError as e:
    print(f'메모리 경고: {e}')
"
```

## 🧪 테스트 방법

**단위 테스트**:
```bash
# 개별 모듈 테스트
pytest tests/test_core/test_constants/test_config.py -v
pytest tests/test_core/test_constants/test_limits.py -v
pytest tests/test_core/test_constants/test_paths.py -v

# 전체 constants 테스트
pytest tests/test_core/test_constants/ -v --cov=paca.core.constants
```

**통합 테스트**:
```bash
# 포터블 경로 통합 테스트
python -c "
from paca.core.constants import FILE_PATHS
import os

# 모든 경로가 존재하는지 확인
paths = [
    FILE_PATHS.get_data_dir(),
    FILE_PATHS.get_logs_dir(),
    FILE_PATHS.get_cache_dir()
]

for path in paths:
    if os.path.exists(path):
        print(f'✅ {path}')
    else:
        print(f'❌ {path}')
"
```

**환경별 테스트**:
```bash
# 개발 환경 테스트
PYTHON_ENV=development python -c "
from paca.core.constants import is_development, get_env_config
print(f'개발 환경: {is_development()}')
print(f'환경 설정: {get_env_config()}')
"

# 프로덕션 환경 테스트
PYTHON_ENV=production python -c "
from paca.core.constants import is_production, get_env_config
print(f'프로덕션 환경: {is_production()}')
print(f'환경 설정: {get_env_config()}')
"
```

**성능 테스트**:
```bash
# 상수 로드 성능 테스트
python -c "
import time
start = time.time()
from paca.core.constants import *
end = time.time()
print(f'상수 로드 시간: {(end-start)*1000:.2f}ms')
"
```

## 💡 추가 고려사항

**보안**:
- 민감한 상수 (API 키, 비밀번호) 환경 변수 분리
- 프로덕션 환경에서 디버그 정보 노출 방지
- 상수 변조 방지를 위한 Final 타입 강제
- 로그 메시지에서 민감 정보 필터링

**성능**:
- 상수 로드 최적화 (목표: <10ms 로드 시간)
- 동적 경로 계산 캐싱 (중복 계산 방지)
- 메시지 템플릿 컴파일 캐싱
- 메모리 효율적인 상수 저장

**향후 개선**:
- 다국어 메시지 지원 (i18n 통합)
- 런타임 상수 변경 기능 (개발 모드에서만)
- 상수 사용량 모니터링 및 분석
- 환경별 상수 오버라이드 시스템
- 상수 검증 및 타입 안전성 강화
- 클라우드 기반 설정 관리 (AWS/Azure)