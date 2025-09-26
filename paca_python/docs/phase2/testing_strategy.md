# Phase 2 테스트 전략

## 1. 테스트 목표
- 복잡도 감지, 메타인지, 메모리, 추론 체인 고도화 기능을 통합적으로 검증
- 회귀 방지를 위해 Phase 1 테스트와 병행 실행
- 성능 요구사항(응답 500ms 이내) 확인을 위한 프로파일링 지표 확보

## 2. 테스트 유형
1. **유닛 테스트**
   - ComplexityDetector: 피처 추출/스코어 산출/카테고리 매핑
   - MetacognitionEngine: 품질 스코어 계산 및 경보 로직
   - WorkingMemory: TTL/만료, 설정 로드(`tests/phase2/test_memory_layers.py`)
   - Memory 모듈: CRUD, TTL, 보존/정책 테스트 (`tests/phase2/test_memory_layers.py`, `test_episodic_memory.py`, `test_longterm_memory.py`)
   - ReasoningChain: 다중 규칙 조합, 전략별 백트래킹 경로/회복 및 전략 전환·ReasoningEngine 에스컬레이션 검증
2. **통합 테스트**
   - `tests/phase2/test_phase2_pipeline.py`: 사용자 입력 → CognitiveSystem → Reasoning → Metacognition까지 전체 흐름 검증
   - 메모리와 복잡도 감지의 상호작용 (대화 이력 기반 난이도 조정)
3. **성능/부하 테스트(선택)**
   - 간단한 async 벤치마크 스크립트 (`scripts/benchmarks/phase2_bench.py` 예정)

## 3. 도구 및 환경
- Pytest + pytest-asyncio
- coverage, pytest-xdist(필요 시)로 병렬 실행
- `PYTHONPATH=. ~/.local/bin/pytest` (CLI 제한 회피 위해 테스트를 묶음별로 실행하는 스크립트 추가 예정)

## 4. 실행 계획
| 주차 | 테스트 범위 | 산출물 |
|------|-------------|--------|
| 스프린트 1 | ComplexityDetector, MetacognitionEngine 유닛/통합 테스트 | `tests/phase2/test_complexity_metacognition.py` |
| 스프린트 2 | Memory 모듈 유닛 + 데이터 저장소 통합 | `tests/phase2/test_memory_layers.py`, `tests/phase2/test_episodic_memory.py`, `tests/phase2/test_longterm_memory.py` |
| 스프린트 3 | ReasoningChain E2E, 복구 시나리오 | `tests/phase2/test_reasoning_chain.py` |

## 5. 회귀 테스트 전략
- Phase 1 테스트(`tests/test_system_basic.py`, `tests/test_reasoning_basic.py`, `tests/test_system_phase1.py`)를 CI 단계에서 함께 실행
- LLM 의존이 없는 환경에서도 성공하도록 `GEMINI_API_KEYS` 비워둔 상태로 테스트
- CLI 환경에서 제한 시간을 우회하기 위해 `scripts/testing/run_phase_regression_tests.py`로 모듈별 pytest 호출을 순차 실행

## 6. TODO
- [x] Phase 2용 테스트 디렉터리(`tests/phase2/`) 생성 및 기본 스켈레톤 추가 *(tests/phase2/test_complexity_metacognition.py)*
- [x] 벤치마크 스크립트 초안 작성 *(scripts/benchmarks/phase2_bench.py)*
- [x] CI/스크립트에 Phase 1 + Phase 2 테스트 묶음 실행 명령어 추가 *(scripts/testing/run_phase_regression_tests.py)*
- [x] ReasoningChain 백트래킹 회귀 테스트 추가 *(tests/phase2/test_reasoning_chain.py)*
- [x] 전략 전환 정책·ReasoningEngine 에스컬레이션 회귀 테스트 확장 *(tests/phase2/test_reasoning_chain.py::test_low_confidence_switches_strategy 등)*
