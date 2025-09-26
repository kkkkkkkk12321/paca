# LongTermMemory 외부 스토리지 확장 설계 초안

## 1. 목표
- 장기 메모리를 로컬 SQLite에서 분리하여 다중 노드/배포 환경에서도 공유 가능한 저장소로 이전
- 파일 잠금 및 동시성 이슈를 예방하고, 대용량 백업/복원 시간을 단축

## 2. 후보 스토리지
### 2.1 DuckDB (로컬 파일 기반, Mmap 지원)
- **장점**: Python 내장 드라이버, Parquet 변환 용이, 단일 파일 유지
- **유의점**: 동시 쓰기 시 파일 잠금 필요 → DuckDB `PRAGMA busy_timeout` 활용 및 쓰기 큐 관리 필요

### 2.2 클라우드 오브젝트 스토리지 + 캐시층
- **전략**: 애플리케이션에서는 DuckDB/SQLite를 `tmpfs` 또는 로컬 디스크에 유지하고, 주기적으로 Parquet 스냅샷을 S3에 업로드
- **장점**: 단순하고 비용 효율적
- **단점**: 완전 공유형 DB는 아님 (주기 동기화 필요)

### 2.3 관리형 데이터베이스 (RDS / Aurora / AlloyDB 등)
- **장점**: 트랜잭션, 동시성, 백업 자동화
- **단점**: DB 접속 계층 작성 및 비용 증가, ORM/SQLAlchemy 도입 검토 필요

## 3. 권장 1차 로드맵
1. `LongTermMemorySettings` 확장
   - `storage_adapter`: `sqlite` | `duckdb` | `external`
   - `connection_uri`: 외부 DB 또는 파일 경로
2. 스토리지 어댑터 추상화 레이어 작성 (`paca/cognitive/memory/longterm_storage.py`)
   - 기본 구현은 기존 SQLite 사용
   - DuckDB 어댑터는 파일 잠금 감지 및 재시도 로직 포함
3. 백업/복원 경로 정규화
   - 현재 JSON export/import 외에 Parquet 내보내기 지원
   - S3 업로드(현재 CI) → Parquet 파일을 동시에 올리도록 확장
4. 동시성 전략
   - Async queue 기반 write pipeline → 단일 writer task가 DB에 반영
   - 락 충돌 시 backoff 정책 구현
5. 운영 체크
   - 새 스토리지 어댑터 선택 시 `PACA실작동준비가이드.md`에 절차 업데이트
   - 경량 모니터링: 백업 크기, write latency 로그 수집

## 4. TODO 체크리스트
- [x] `LongTermMemorySettings`에 어댑터/URI 필드 추가
- [x] 스토리지 어댑터 추상화 기본 골격 도입 (SQLite 어댑터 연결 완료)
- [ ] Parquet export/import 지원 및 S3 업로드 연동
- [ ] 동시성 큐 및 백오프 로직 설계 문서화 후 구현
- [ ] 운영 가이드 업데이트 (스토리지 전환 절차)

## 5. 참고 자료
- DuckDB Python API: https://duckdb.org/docs/api/python/overview.html
- AWS S3 CLI 사용법: https://docs.aws.amazon.com/cli/latest/reference/s3/
- GitHub Actions Secrets: https://docs.github.com/en/actions/security-guides/encrypted-secrets
