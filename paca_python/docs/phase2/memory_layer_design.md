# Phase 2 Sprint 2 설계 메모 (Memory Layer 강화)

## 1. 목표 개요
- Working/Episodic/LongTerm 메모리 계층의 TTL, 비동기 IO, 스냅샷 전략 확립
- PacaSystem 전체에서 메모리 만료/정리 로직을 안전하게 사용하도록 통합 포인트 마련
- 회귀 테스트 및 벤치마크 시나리오로 메모리 레이어의 안정성을 검증

## 2. 기능 범위
1. **WorkingMemory 개선 마무리**
   - [x] TTL/자동 만료 루프 도입 (`memory_settings.json`, `tests/phase2/test_memory_layers.py`)
   - [x] TTL 없는 환경 fallback 동작 보장 및 테스트 (`tests/phase2/test_memory_layers.py::test_working_memory_handles_ttl_disabled`)
   - [x] cleanup 루프 종료 시그널 처리(PacaSystem 종료 시 task cancel → `WorkingMemory.shutdown`)
2. **EpisodicMemory**
   - [x] 비동기 저장/로드 (`aiofiles`) 및 보존 기간(`retention_days`) 적용
   - [x] 컨텍스트 스냅샷 구조 정의 (`snapshot_interval_seconds`, `max_snapshot_items`)
   - [x] 테스트 스켈레톤 및 시나리오 완료: `tests/phase2/test_episodic_memory.py`
3. **LongTermMemory**
   - [ ] 배치 저장/로드 인터페이스 정리
   - [x] TTL 대신 우선순위 기반 정리 정책 초안 작성 (`LongTermMemorySettings`, `_apply_cleanup_policy_sync`)
   - [ ] 외부 스토리지 연동 시나리오 문서화

## 3. 구현 순서 제안
1. 설정 로더 공통화 (`memory_settings.json` → MemoryConfiguration 확장)
2. EpisodicMemory async I/O + retention 적용 및 포터블 저장소 연동
3. LongTermMemory 정리 정책 설계 및 최소 구현
4. 테스트 작성
   - `tests/phase2/test_memory_layers.py` 확장 (TTL fallback, shutdown 동작)
   - `tests/phase2/test_episodic_memory.py` 신규 작성 (보존/비동기 검증)
   - `tests/phase2/test_longterm_memory.py` 신규 작성 (정책/강도 정리)
5. 벤치마크 초안 업데이트 (`phase2_bench.py`에 메모리 stress 옵션 추가)

## 4. 테스트 전략 업데이트
- WorkingMemory: TTL 만료, 외부 설정 로드, cleanup cancel 확인
- EpisodicMemory: retention 적용, snapshot 주기, async 저장/로드 실패 처리
- LongTermMemory: 우선순위 정리, 검색 결과 검증

## 5. 로그/운영 고려사항
- 만료/정리 이벤트는 DEBUG 로그, 실패/예외는 WARNING 이상 수준으로 기록
- PacaSystem 종료 시 cleanup task 안전하게 종료
- 운영자가 확인할 수 있도록 `logs/memory/` 아래에 일별 정리 리포트 옵션 고려

## 6. 체크리스트
- [x] WorkingMemory TTL/cleanup 구현 및 테스트
- [x] EpisodicMemory retention & async I/O 완료
- [x] LongTermMemory 정리 정책 초안 및 테스트
- [x] Phase 2 테스트 스위트에 Memory 레이어 통합 (`run_phase_regression_tests.py` 반영)
- [x] 문서/가이드 업데이트 (PACA실작동준비가이드, testing_strategy)

## 7. 다음 단계 제안
- EpisodicMemory 배치 내보내기·재로딩 API 정의 (컨텍스트별 스트림 처리)
- LongTermMemory 외부 스토리지(예: DuckDB, cloud KV) 포맷 조사 및 파일 락 전략 설계 *(세부 초안: docs/phase2/longterm_external_storage.md)*
- Memory 레이어 지표 수집 파이프라인 추가 (`logs/memory/` 주기 보고서) → Phase 2 Sprint 3 후보

## 8. 배치 인터페이스 및 외부 스토리지 계획
### 8.1 EpisodicMemory 배치 내보내기/재로딩
- **Export 메서드(`export_batch`)**: 조건(`since_timestamp`, `limit`)을 받아 최근 일화들을 JSON으로 직렬화. 기본 경로는 `data/memory/episodic/episodic_export.json`.
- **Import 메서드(`import_batch`)**: JSON 파일 또는 리스트 입력을 받아 기존 메모리에 병합. 중복 ID는 건너뛰고, 새 항목은 TTL·인덱스 업데이트까지 수행.
- **비동기 I/O**: `aiofiles` 사용 가능 시 비동기 쓰기, 미설치 환경에서는 스레드 실행으로 대체.
- **스냅샷 연계**: 주기 스냅샷과 동일한 포맷을 활용하여 배포 환경에서도 동일한 리커버리 절차 적용.

### 8.2 LongTermMemory 외부 스토리지 연동
- **Export 메서드(`export_items`)**: 선택 조건(`memory_type`, `limit`)을 받아 메모리 아이템을 JSON 파일로 저장. 외부 백업 및 분석용.
- **Import 메서드(`import_items`)**: JSON 파일을 읽어 SQLite에 UPSERT. `max_items` 정책 위반 시 추가 정리 정책 재사용.
- **외부 DB 고려**: `LongTermMemorySettings`에 `persistent_db`·`database_name`이 이미 존재하므로 DuckDB, 클라우드 파일 시스템 등으로 경로만 교체하면 됨. 향후 파일 락·동시성은 Phase 2 Sprint 3에서 보강.

### 8.3 운영 절차
- **백업 스크립트**: `scripts/tools/memory_backup.py`로 episodic/long_term export·import 지원(추가 옵션으로 S3 등 외부 업로드는 후속).
- **CI 스케줄링**: `.github/workflows/memory_backup.yml`로 매일 자동 백업 및 아티팩트 업로드 구성
- **복원 가이드**: 운영 문서에 `import_batch/import_items` 호출 예시를 추가하여 장애 복구 시 절차 간소화.
- **검증**: pytest Phase2 스위트에 export/import 회귀 테스트 추가(`tests/phase2/test_episodic_memory.py`, `tests/phase2/test_longterm_memory.py`).
