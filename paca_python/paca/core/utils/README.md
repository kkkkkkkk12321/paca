# Core Utils Module - PACA Python v5

## 🎯 프로젝트 개요
PACA 시스템의 핵심 유틸리티 모듈로, 포터블 저장소, 안전한 출력, 환경 설정, 비동기 처리, 수학 계산 등 시스템 전반에서 사용되는 공통 기능을 제공합니다.

## 📁 폴더/파일 구조
```
utils/
├── __init__.py              # 모듈 진입점 및 통합 API (80+줄)
├── async_utils.py           # 비동기 처리 유틸리티 (재시도, 배치, 캐시)
├── environment.py           # 환경 설정 자동화 관리 (200+줄)
├── logger.py                # 구조화된 로깅 시스템
├── math_utils.py            # 수학 계산 유틸리티 (통계, 분석)
├── optional_imports.py      # 선택적 의존성 관리
├── portable_storage.py      # 포터블 데이터 저장 관리자 (500+줄)
├── safe_logging.py          # Windows 호환 안전한 로깅
└── safe_print.py            # UTF-8/이모지 안전 출력 (150+줄)
```

**파일별 주요 기능**:
- `portable_storage.py`: JSON/SQLite 포터블 데이터 관리, 동적 경로 계산
- `safe_print.py`: Windows CP949 이모지 처리, UTF-8 환경 설정
- `environment.py`: 자동 환경 변수 설정, Python 경로 관리
- `async_utils.py`: 재시도, 배치 처리, 비동기 캐시, 동시성 제어
- `math_utils.py`: 통계 계산, 이상치 감지, 정규화, 보간

## ⚙️ 기능 요구사항

**입력**:
- 저장할 데이터 (JSON 직렬화 가능한 객체)
- 출력할 텍스트 (한국어, 이모지, UTF-8 문자)
- 환경 설정 요구사항 (Python 경로, API 키)
- 비동기 작업 및 수학 연산 데이터

**출력**:
- 포터블 저장된 데이터 (JSON 파일, SQLite DB)
- 안전하게 처리된 출력 (Windows CP949 호환)
- 자동 설정된 환경 변수 및 Python 경로
- 처리된 비동기 결과 및 수학 계산값

**핵심 로직 흐름**:
1. **저장소 관리**: 경로 계산 → 디렉토리 생성 → 데이터 저장/로드 → 메타데이터 관리
2. **안전 출력**: 문자 감지 → 인코딩 검사 → 이모지 변환 → 안전 출력
3. **환경 설정**: 시스템 감지 → 설정 검증 → 자동 구성 → 경로 설정

## 🛠️ 기술적 요구사항

**언어 및 프레임워크**:
- Python 3.9+ (pathlib, asyncio, typing, dataclasses)
- Windows/Linux/macOS 크로스 플랫폼 지원
- SQLite3, JSON 기반 데이터 저장

**핵심 라이브러리**:
- `pathlib`: 크로스 플랫폼 경로 처리
- `asyncio`: 비동기 작업 및 동시성 관리
- `sqlite3`: 포터블 데이터베이스 저장
- `json`: JSON 직렬화/역직렬화
- `logging`: 구조화된 로깅

**환경 요구사항**:
- 메모리: 최소 64MB (포터블 저장소 포함)
- 저장소: 10-100MB (데이터 양에 따라)
- 권한: 읽기/쓰기 권한 (데이터 폴더)

## 🚀 라우팅 및 진입점

**포터블 저장소 사용**:
```python
from paca.core.utils.portable_storage import get_storage_manager

# 저장소 매니저 초기화
storage = get_storage_manager()

# 데이터 저장
data = {"user": "홍길동", "score": 95}
storage.save_json_data(storage.get_config_file_path("user.json"), data)

# 데이터 로드
loaded = storage.load_json_data(storage.get_config_file_path("user.json"))

# 저장소 정보
info = storage.get_storage_info()
print(f"사용 공간: {info['total_size_mb']:.2f} MB")
```

**안전한 출력 사용**:
```python
from paca.core.utils.safe_print import safe_print, setup_unicode_environment

# 환경 설정
setup_unicode_environment()

# 안전한 출력 (Windows CP949에서도 동작)
safe_print("안녕하세요! 😊 PACA v5입니다.")
safe_print("복잡한 수식: ∑∞ + ∫∂x = ∆y")
```

**환경 설정 사용**:
```python
from paca.core.utils.environment import EnvironmentManager

# 환경 매니저 초기화
env_manager = EnvironmentManager()

# 전체 환경 설정
await env_manager.setup_all()

# 개별 설정
env_manager.setup_python_path()
env_manager.setup_encoding()
```

**비동기 유틸리티 사용**:
```python
from paca.core.utils.async_utils import retry_async, batch_process, AsyncLRUCache

# 재시도 기능
@retry_async(max_retries=3, delay=1.0)
async def unstable_api_call():
    # 불안정한 API 호출
    pass

# 배치 처리
async def process_items():
    items = [1, 2, 3, 4, 5]
    results = await batch_process(items, async_processor, batch_size=2)
    return results
```

## 📋 코드 품질 가이드

**네이밍 규칙**:
- 클래스: PascalCase (예: `PortableStorageManager`, `EnvironmentManager`)
- 함수: snake_case (예: `safe_print`, `get_storage_manager`)
- 상수: UPPER_SNAKE_CASE (예: `DEFAULT_PATHS`, `MAX_RETRY_COUNT`)
- 파일: snake_case.py (예: `portable_storage.py`)

**필수 규칙**:
- 모든 public 함수에 타입 힌트 및 docstring 필수
- 크로스 플랫폼 호환성 보장 (pathlib 사용)
- 예외 처리: 안전한 실패 및 복구 메커니즘
- 로깅: 적절한 로그 레벨과 구조화된 메시지
- 테스트: 핵심 기능의 단위 테스트 작성

**특별 규칙**:
- **포터블 경로**: 절대 경로 금지, 상대 경로만 사용
- **인코딩 안전성**: 모든 텍스트 출력은 safe_print 사용
- **환경 독립성**: 하드코딩된 경로나 설정 금지

## 🏃‍♂️ 실행 방법

**기본 설치**:
```bash
# 포터블 저장소 초기화
python setup_portable_storage.py

# 환경 설정 확인
python -c "
from paca.core.utils.environment import EnvironmentManager
import asyncio

async def check():
    env = EnvironmentManager()
    result = await env.setup_all()
    print(f'환경 설정: {result}')

asyncio.run(check())
"
```

**개별 기능 테스트**:
```bash
# 포터블 저장소 테스트
python -c "
from paca.core.utils.portable_storage import get_storage_manager
storage = get_storage_manager()
print(f'저장소 위치: {storage.base_path}')
"

# 안전한 출력 테스트
python -c "
from paca.core.utils.safe_print import safe_print
safe_print('테스트: 🎯🚀✅ 한글과 이모지')
"
```

**비동기 기능 테스트**:
```bash
python -c "
import asyncio
from paca.core.utils.async_utils import delay, AsyncLRUCache

async def test():
    cache = AsyncLRUCache(max_size=100)
    await cache.set('key', 'value')
    result = await cache.get('key')
    print(f'캐시 테스트: {result}')

asyncio.run(test())
"
```

## 🧪 테스트 방법

**단위 테스트**:
```bash
# 개별 모듈 테스트
pytest tests/test_core/test_utils/test_portable_storage.py -v
pytest tests/test_core/test_utils/test_safe_print.py -v
pytest tests/test_core/test_utils/test_environment.py -v

# 전체 utils 테스트
pytest tests/test_core/test_utils/ -v --cov=paca.core.utils
```

**통합 테스트**:
```bash
# 포터블 기능 전체 테스트
python test_complete_portable.py

# 크로스 플랫폼 테스트
python tests/integration/test_cross_platform.py
```

**성능 테스트**:
```bash
# 저장소 성능 테스트
python tests/performance/test_storage_performance.py

# 비동기 처리 성능 테스트
python tests/performance/test_async_performance.py
```

**특별 테스트**:
```bash
# Windows CP949 환경에서 이모지 테스트
python -c "
import os
os.system('chcp 949')  # CP949로 변경
from paca.core.utils.safe_print import safe_print
safe_print('이모지 테스트: 😊🎯🚀✅❌⚠️')
"

# 포터블 이동 테스트
python tests/portability/test_folder_move.py
```

## 💡 추가 고려사항

**보안**:
- 포터블 저장소 권한 관리 (읽기/쓰기 제한)
- 환경 변수 민감 정보 보호 (API 키 암호화)
- 사용자 입력 검증 (경로 트래버설 방지)
- 로그 파일 민감 정보 필터링

**성능**:
- JSON 파일 압축 및 캐싱 (목표: <10ms 로드 시간)
- SQLite 연결 풀링 및 트랜잭션 최적화
- 비동기 I/O 활용 (파일 읽기/쓰기 병렬화)
- 메모리 효율성 (대용량 데이터 스트리밍)

**향후 개선**:
- 클라우드 저장소 동기화 (Google Drive, OneDrive)
- 암호화된 저장소 지원 (AES-256)
- 버전 관리 시스템 (Git-like 스냅샷)
- 실시간 저장소 모니터링 및 알림
- 네트워크 저장소 지원 (SMB, FTP)
- 모바일 동기화 인터페이스