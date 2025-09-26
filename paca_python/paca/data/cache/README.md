# 🚀 PACA 고급 캐시 시스템

## 🎯 프로젝트 개요

PACA 고급 캐시 시스템은 3계층 아키텍처를 통한 지능형 캐싱 솔루션입니다. L1 메모리 캐시, L2 디스크 캐시, L3 분산 캐시(Redis)를 통해 85% 이상의 캐시 히트율과 10ms 미만의 응답 시간을 목표로 합니다.

## 📁 폴더/파일 구조

```
paca/data/cache/
├── __init__.py              # 캐시 모듈 진입점 및 통합 인터페이스
├── cache_manager.py         # 3계층 캐시 중앙 관리자
├── lru_cache.py            # L1 메모리 캐시 (LRU/LFU/FIFO 정책)
├── ttl_cache.py            # L2 디스크 캐시 (TTL 기반 영속성)
├── hybrid_cache.py         # 메모리+디스크 하이브리드 캐시
├── cache_metrics.py        # 성능 모니터링 및 통계
├── cache_warming.py        # 지능형 캐시 예열 시스템
└── README.md               # 이 문서
```

### 파일별 설명

- **cache_manager.py**: 전체 캐시 시스템의 중앙 관리자로 L1→L2→L3 계층 간 자동 승격/강등 처리
- **lru_cache.py**: 메모리 기반 고성능 LRU 캐시, O(1) 연산과 메모리 사용량 관리
- **ttl_cache.py**: 디스크 기반 TTL 캐시, 영속성과 대용량 저장 지원
- **hybrid_cache.py**: 접근 패턴 기반 hot/cold 데이터 자동 분류 및 배치
- **cache_metrics.py**: 실시간 성능 메트릭 수집, 히트율/응답시간 분석
- **cache_warming.py**: 패턴 분석 기반 지능형 캐시 예열 및 예측

## ⚙️ 기능 요구사항

### 입력/출력 인터페이스

```python
# 기본 캐시 연산
async def get(key: str) -> Optional[Any]           # 캐시에서 값 조회
async def set(key: str, value: Any, ttl: int) -> bool  # 값 저장 (TTL 지정)
async def delete(key: str) -> bool                 # 키 삭제
async def invalidate(pattern: str) -> int          # 패턴 매칭 무효화
async def clear() -> bool                          # 전체 캐시 초기화

# 성능 모니터링
async def get_stats() -> CacheStats               # 종합 통계 조회
async def optimize() -> bool                      # 캐시 최적화 수행
```

### 핵심 로직 흐름

1. **캐시 조회 (get)**: L1 메모리 → L2 디스크 → L3 Redis → 원본 데이터
2. **자동 승격**: L2/L3 히트 시 상위 계층으로 자동 승격
3. **지능형 배치**: 접근 빈도 기반 hot/cold 데이터 분류
4. **백그라운드 최적화**: 만료된 항목 정리, 메모리 최적화
5. **예측 기반 예열**: 패턴 분석을 통한 사전 캐시 로딩

## 🛠️ 기술적 요구사항

### 언어 및 라이브러리
- **Python**: 3.9+ (asyncio, typing 지원)
- **필수 라이브러리**: aiofiles, redis (선택적)
- **개발 도구**: pytest, mypy, black

### 실행 환경
- **메모리**: 최소 512MB (캐시용 100MB + 시스템)
- **디스크**: 1GB 여유 공간 (L2 캐시용)
- **네트워크**: Redis 사용 시 네트워크 연결 필요

### 성능 목표
- **캐시 히트율**: >85%
- **L1 응답시간**: <10ms
- **L2 응답시간**: <50ms
- **메모리 한계**: 100MB (L1), 1GB (L2)
- **동시 처리**: 1000 ops/sec

## 🚀 라우팅 및 진입점

### API 경로 및 실행 시작점

```python
# 전역 캐시 매니저 인스턴스 생성
from paca.data.cache import get_cache_manager, CacheConfig

# 기본 설정으로 초기화
cache_manager = await get_cache_manager()

# 커스텀 설정으로 초기화
config = CacheConfig(
    l1_max_size_mb=200,      # L1 메모리 캐시 200MB
    l2_max_size_mb=2000,     # L2 디스크 캐시 2GB
    enable_l3=True,          # Redis L3 캐시 활성화
    redis_host="localhost",
    redis_port=6379
)
cache_manager = await get_cache_manager(config)
```

### 계층별 진입점

```python
# L1 메모리 캐시 직접 사용
from paca.data.cache import LRUCache
lru_cache = LRUCache(max_size_mb=100)

# L2 디스크 캐시 직접 사용
from paca.data.cache import TTLCache
ttl_cache = TTLCache(cache_dir="./cache", max_size_mb=1000)

# 하이브리드 캐시 사용
from paca.data.cache import HybridCache
hybrid_cache = HybridCache(total_size_mb=1000)
```

## 📋 코드 품질 가이드

### 주석 규칙
- **모든 공개 함수**: docstring 필수 작성
- **복잡한 알고리즘**: 인라인 주석으로 설명
- **타입 힌트**: 모든 함수 매개변수와 반환값에 타입 명시

### 네이밍 규칙
- **클래스**: PascalCase (예: `CacheManager`)
- **함수/변수**: snake_case (예: `get_cache_stats`)
- **상수**: UPPER_SNAKE_CASE (예: `DEFAULT_TTL`)
- **Private 멤버**: 언더스코어 접두사 (예: `_internal_method`)

### 예외처리 규칙
- **모든 외부 I/O**: try-catch 블록으로 보호
- **로그 레벨**: ERROR(실패), WARNING(성능저하), INFO(상태변화), DEBUG(상세정보)
- **Graceful Degradation**: 하위 캐시 계층 장애 시 상위 계층으로 fallback

## 🏃‍♂️ 실행 방법

### 설치

```bash
# 기본 의존성 설치
pip install aiofiles

# Redis 지원 (선택적)
pip install redis

# 개발 도구 설치
pip install pytest pytest-asyncio pytest-benchmark mypy
```

### 기본 사용법

```python
import asyncio
from paca.data.cache import get_cache_manager

async def main():
    # 캐시 매니저 초기화
    cache = await get_cache_manager()

    # 데이터 저장
    await cache.set("user:123", {"name": "John", "age": 30}, ttl=3600)

    # 데이터 조회
    user = await cache.get("user:123")
    print(user)  # {'name': 'John', 'age': 30}

    # 성능 통계 확인
    stats = await cache.get_stats()
    print(f"Hit rate: {stats.hit_rate:.2%}")

    # 종료
    await cache.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### 캐시 워밍 사용법

```python
from paca.data.cache import CacheWarming

async def data_loader(key: str):
    """데이터 로더 함수"""
    # 실제 데이터 소스에서 데이터 로드
    return f"data_for_{key}"

async def setup_warming():
    cache = await get_cache_manager()
    warming = CacheWarming(cache)

    # 데이터 소스 등록
    await warming.register_data_source("main_db", data_loader)

    # 워밍 시작
    await warming.start_warming()

    # 즉시 예열 실행
    await warming.warm_cache_now()
```

## 🧪 테스트 방법

### 단위 테스트

```bash
# 모든 테스트 실행
pytest tests/cache/

# 특정 모듈 테스트
pytest tests/cache/test_cache_manager.py

# 커버리지 포함 테스트
pytest tests/cache/ --cov=paca.data.cache --cov-report=html
```

### 통합 테스트

```bash
# 전체 캐시 시스템 통합 테스트
pytest tests/integration/test_cache_integration.py

# Redis 연동 테스트 (Redis 서버 필요)
pytest tests/integration/test_redis_integration.py
```

### 성능 테스트

```bash
# 성능 벤치마크 실행
pytest tests/performance/test_cache_performance.py --benchmark-only

# 메모리 사용량 테스트
pytest tests/performance/test_memory_usage.py

# 동시성 테스트
pytest tests/performance/test_concurrency.py
```

### 테스트 시나리오

1. **기본 CRUD 연산**: get, set, delete, clear 기본 동작
2. **TTL 만료**: 시간 기반 자동 만료 확인
3. **메모리 한계**: 최대 메모리 사용량 제한 동작
4. **자동 승격/강등**: 계층 간 데이터 이동 로직
5. **동시성**: 여러 스레드/코루틴 동시 접근
6. **장애 복구**: Redis 연결 실패 시 fallback 동작
7. **성능 목표**: 히트율 85%, 응답시간 10ms 달성 확인

## 💡 추가 고려사항

### 보안

- **데이터 암호화**: 민감한 데이터는 저장 전 암호화 권장
- **접근 제어**: Redis 사용 시 AUTH 설정 필수
- **키 충돌 방지**: 네임스페이스 기반 키 관리
- **메모리 덤프 보호**: 메모리 캐시 내용의 의도치 않은 노출 방지

### 성능 최적화

- **메모리 풀링**: 객체 재사용을 통한 GC 압박 감소
- **배치 연산**: 여러 키를 한 번에 처리하는 배치 API
- **압축**: 대용량 데이터 자동 압축 (gzip, lz4)
- **샤딩**: 대규모 환경에서 여러 캐시 인스턴스 분산

### 향후 개선사항

1. **분산 캐시 일관성**: 여러 인스턴스 간 데이터 동기화
2. **머신러닝 기반 예측**: 더 정확한 캐시 예열 패턴 학습
3. **자동 튜닝**: 워크로드 기반 자동 설정 최적화
4. **메트릭 시각화**: 실시간 대시보드 및 알림
5. **백업/복구**: 캐시 데이터 백업 및 복구 기능
6. **클러스터링**: 고가용성을 위한 클러스터 모드 지원

### 모니터링 및 알림

- **성능 저하 감지**: 히트율 70% 미만 시 알림
- **메모리 사용량 경고**: 80% 초과 시 경고
- **오류율 모니터링**: 5% 초과 시 긴급 알림
- **예측 정확도**: 캐시 워밍 효과성 추적

### 운영 가이드

- **모니터링 주기**: 5분 간격 메트릭 수집
- **로그 보존**: 7일간 DEBUG, 30일간 INFO 이상
- **백업 주기**: 일일 캐시 인덱스 백업
- **성능 리포트**: 주간 성능 분석 리포트 생성