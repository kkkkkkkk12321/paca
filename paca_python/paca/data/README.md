# Data Management System - PACA Python v5

## 🎯 프로젝트 개요

PACA v5 데이터 관리 시스템으로, 학습 데이터 자동 백업, 복원, 스케줄링을 제공하는 완전한 데이터 보호 시스템입니다. Phase 5 핵심 기능으로서 시스템의 안정성과 데이터 무결성을 보장하며, 자동화된 백업 스케줄링과 효율적인 복원 기능을 제공합니다.

## 📁 폴더/파일 구조

```
data/
├── __init__.py                 # 모듈 진입점
├── base.py                     # 기존 데이터 저장소 시스템
├── backup_system.py            # 백업/복원 핵심 시스템 (NEW)
├── scheduler.py                # 백업 스케줄링 시스템 (NEW)
└── README.md                   # 이 문서
```

### Phase 5 새로운 핵심 파일들

- **`backup_system.py`** (850줄): 백업/복원 핵심 시스템
  - `BackupSystem`: 자동 백업 시스템 (문서 명세 함수 포함)
  - `BackupManager`: 고급 백업 관리 시스템
  - `BackupMetadata`: 백업 메타데이터 관리
  - `RestoreResult`: 복원 결과 처리

- **`scheduler.py`** (580줄): 백업 스케줄링 시스템
  - `BackupScheduler`: 크론 및 간격 기반 스케줄러
  - `CronParser`: 크론 표현식 파싱 및 분석
  - `ScheduleJob`: 스케줄 작업 관리
  - `ScheduleEvent`: 스케줄 이벤트 추적

## ⚙️ 기능 요구사항

**핵심 입력**: 백업 트리거 이벤트, 스케줄 설정, 복원 요청
**핵심 출력**: 백업 파일 (ZIP), 복원된 데이터, 스케줄 실행 결과
**핵심 로직 흐름**: 트리거 감지 → 백업 생성 → 압축 저장 → 메타데이터 기록 → 스케줄 관리

**주요 기능**:
- 자동 백업 시스템 (훈련 시작시, 설정 변경시, 시스템 종료시)
- 스케줄 기반 백업 (크론 표현식, 간격 기반)
- 압축 백업 및 체크섬 검증
- 메타데이터 기반 백업 관리
- 비동기 백업/복원 처리

## 🛠️ 기술적 요구사항

- **Python**: 3.9+ (비동기 처리, 타입 힌트 지원)
- **필수 라이브러리**:
  - asyncio (비동기 처리)
  - zipfile (압축 백업)
  - hashlib (체크섬 검증)
  - pathlib (경로 처리)
- **선택적 라이브러리**:
  - json (메타데이터 저장)
  - tempfile (테스트용 임시 디렉토리)
- **시스템 요구사항**:
  - OS: Windows 10+, macOS 11+, Ubuntu 20.04+
  - 디스크 공간: 백업 데이터 크기의 2-3배
  - 메모리: 최소 1GB (압축 처리용)

## 🚀 라우팅 및 진입점

### Python 직접 실행
```python
# 기본 백업 시스템 사용
from paca.data import BackupSystem

backup_system = BackupSystem("my_backups")
backup_id = backup_system.create_auto_backup("training_start")

# 비동기 백업 생성
import asyncio
result = await backup_system.create_backup_async(
    backup_type=BackupType.MANUAL,
    source_paths=["paca/learning", "paca/cognitive"]
)

# 백업 복원
restore_result = await backup_system.restore_backup_async(backup_id, "restore_path")
```

### 스케줄러 사용
```python
from paca.data import BackupScheduler, BackupSystem

scheduler = BackupScheduler()
backup_system = BackupSystem("scheduled_backups")

# 매일 2시에 백업 (크론 스케줄)
job_id = scheduler.add_cron_job(
    name="Daily Backup",
    cron_expression="0 2 * * *",
    backup_system=backup_system
)

# 6시간마다 백업 (간격 스케줄)
job_id = scheduler.add_interval_job(
    name="Frequent Backup",
    interval_minutes=360,
    backup_system=backup_system
)
```

### 백업 매니저 사용
```python
from paca.data import BackupManager, BackupSystem

manager = BackupManager({
    "local": BackupSystem("backups/local"),
    "archive": BackupSystem("backups/archive")
})

# 응급 백업
await manager.perform_emergency_backup("system_error_detected")
```

## 📋 코드 품질 가이드

**주석 규칙:**
- 모든 데이터 모델에 필드 설명 및 제약조건 명시
- 쿼리 메서드는 성능 특성 및 제한사항 기술
- 인덱스 및 최적화 전략 문서화

**네이밍 규칙:**
- 데이터 모델: 명사형 클래스명 (UserData, ConversationData)
- 저장소: [Type]DataStore (MemoryDataStore, FileDataStore)
- 쿼리 메서드: find_*, query_*, search_* 접두사

**예외 처리:**
- DataError: 일반적인 데이터 오류
- ValidationError: 데이터 검증 실패
- StorageError: 저장소 관련 오류
- QueryError: 쿼리 실행 오류

## 🏃‍♂️ 실행 방법

**설치:**
```bash
# 프로젝트 루트에서
pip install -e .

# 향후 데이터베이스 지원을 위한 선택적 의존성
pip install sqlalchemy  # ORM
pip install redis       # 캐싱
```

**기본 데이터 저장/검색:**
```python
import asyncio
from paca.data import DataManager, MemoryDataStore, QueryFilter

async def main():
    # 데이터 매니저 초기화
    manager = DataManager()

    # 메모리 저장소 등록
    memory_store = MemoryDataStore()
    manager.register_store("memory", memory_store)

    # 사용자 데이터 저장
    user_data = {
        "name": "김철수",
        "email": "kim@example.com",
        "preferences": {
            "language": "ko",
            "theme": "dark"
        }
    }

    record = await manager.store(
        data=user_data,
        data_type="json",
        store_type="memory"
    )

    print(f"저장된 레코드 ID: {record.id}")

    # 데이터 검색
    filters = [QueryFilter("name", "eq", "김철수")]
    results = await manager.query(filters, store_type="memory")

    for result in results:
        print(f"검색 결과: {result.data}")

# 실행
asyncio.run(main())
```

**고급 쿼리 사용법:**
```python
from paca.data import QueryFilter, QueryOptions

# 복합 조건 검색
filters = [
    QueryFilter("age", "gte", 18),      # 나이 18세 이상
    QueryFilter("city", "eq", "서울"),   # 서울 거주
    QueryFilter("status", "in", ["active", "premium"])  # 활성 또는 프리미엄
]

options = QueryOptions(
    limit=10,           # 최대 10개 결과
    offset=0,           # 첫 페이지
    sort_by="created_at",  # 생성일 정렬
    sort_order="desc"   # 내림차순
)

results = await manager.query(
    filters=filters,
    options=options,
    store_type="memory"
)
```

## 🧪 테스트 방법

**단위 테스트:**
- 각 저장소의 CRUD 연산 테스트
- 데이터 타입별 저장/검색 검증
- 쿼리 필터 및 정렬 기능 테스트

**통합 테스트:**
- 다중 저장소 간 데이터 동기화
- 전체 데이터 워크플로우 검증
- 트랜잭션 및 일관성 검증

**성능 테스트:**
- 대용량 데이터 저장/검색 성능 (<50ms 목표)
- 동시 접근 시 데이터 무결성 검증
- 메모리 사용량 최적화 확인

**데이터 테스트 예시:**
```python
async def test_data_storage():
    """데이터 저장 테스트"""
    manager = DataManager()
    store = MemoryDataStore()
    manager.register_store("test", store)

    # 데이터 저장
    data = {"test": "value", "number": 42}
    record = await manager.store(data, store_type="test")

    assert record.id is not None
    assert record.data == data

async def test_data_query():
    """데이터 검색 테스트"""
    manager = DataManager()
    store = MemoryDataStore()
    manager.register_store("test", store)

    # 테스트 데이터 저장
    await manager.store({"name": "Alice", "age": 25}, store_type="test")
    await manager.store({"name": "Bob", "age": 30}, store_type="test")

    # 검색 테스트
    filters = [QueryFilter("age", "gte", 30)]
    results = await manager.query(filters, store_type="test")

    assert len(results) == 1
    assert results[0].data["name"] == "Bob"

async def test_data_update():
    """데이터 수정 테스트"""
    manager = DataManager()
    store = MemoryDataStore()
    manager.register_store("test", store)

    # 데이터 저장
    record = await manager.store({"name": "Charlie", "age": 35}, store_type="test")

    # 데이터 수정
    updated_record = await manager.update(
        record.id,
        {"name": "Charlie Brown", "age": 36},
        store_type="test"
    )

    assert updated_record.data["name"] == "Charlie Brown"
    assert updated_record.data["age"] == 36
```

## 💡 추가 고려사항

**보안:**
- 민감한 데이터 암호화 저장
- 접근 권한 제어 및 감사 로그
- SQL 인젝션 및 NoSQL 인젝션 방지

**성능:**
- 인덱싱 전략 및 쿼리 최적화
- 데이터 파티셔닝 및 샤딩
- 캐싱 계층을 통한 응답 속도 향상

**향후 개선:**
- 분산 데이터베이스 지원
- 실시간 데이터 동기화
- 자동 백업 및 복구 시스템
- 데이터 품질 모니터링

**저장소 유형별 특성:**
- **MemoryDataStore**: 고속 임시 데이터, 세션 정보
- **FileDataStore**: 설정 파일, 로그 데이터 (계획)
- **DatabaseDataStore**: 영구 데이터, 관계형 정보 (계획)
- **CacheDataStore**: 자주 접근하는 데이터 (계획)

**데이터 타입 지원:**
- **기본 타입**: STRING, INTEGER, FLOAT, BOOLEAN, DATETIME
- **복합 타입**: JSON (객체/배열), BINARY (파일/이미지)
- **특수 타입**: 암호화된 데이터, 압축된 데이터 (계획)

**쿼리 연산자:**
- **비교**: eq (같음), ne (다름), gt/gte (크다/크거나같다), lt/lte (작다/작거나같다)
- **포함**: in (포함), not_in (미포함), like (패턴 매치)
- **논리**: and, or, not 연산자 조합
- **정렬**: asc (오름차순), desc (내림차순)

### 테스트 결과 (2024-09-21)
```
PACA v5 Phase 5 Backup/Restore System Integration Test
=================================================================
RESULTS: 9/9 tests passed
SUCCESS: All tests passed successfully!
Phase 5 Backup/Restore System is working correctly!

성능 메트릭:
- 백업 시스템 생성: 1.6ms (목표: <100ms) ✅
- 백업 ID 생성: 0.1ms (목표: <50ms) ✅
- 비동기 백업: 성공적으로 완료 ✅
- 백업 복원: 1개 파일 성공적으로 복원 ✅
```

## 💡 추가 고려사항

### 성능 최적화
- **압축 효율성**: ZIP 압축으로 평균 60-80% 크기 감소
- **메타데이터 캐싱**: 메모리 기반 메타데이터 캐시로 빠른 조회
- **비동기 처리**: GUI 블록 없는 백업/복원 처리
- **자동 정리**: 오래된 백업 자동 삭제로 디스크 공간 관리

### 보안 고려사항
- **체크섬 검증**: MD5 해시로 파일 무결성 보장
- **접근 제어**: 백업 파일 권한 설정
- **메타데이터 보호**: 백업 정보 안전한 저장
- **암호화 준비**: 향후 암호화 백업 지원 준비

### 향후 개선 계획
- **클라우드 백업**: AWS S3, Google Cloud Storage 지원
- **증분 백업**: 변경된 파일만 백업하는 증분 백업
- **백업 암호화**: AES-256 암호화 백업 지원
- **웹 UI**: 백업 관리 웹 인터페이스
- **백업 검증**: 정기적인 백업 무결성 검사