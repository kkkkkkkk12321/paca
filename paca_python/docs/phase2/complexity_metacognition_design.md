# Phase 2 Sprint 1 설계 메모 (Complexity Detector & Metacognition)

## 1. 기능 개요
- **ComplexityDetector**
  - 입력: 한국어 사용자 발화, 시스템 내부 태스크 메타데이터
  - 출력: 복잡도 스코어(0~1), 카테고리(LOW/MEDIUM/HIGH), 근거 피처 목록
  - 사용처: MetacognitionEngine, CognitiveSystem 라우팅, 학습 우선순위 결정
- **MetacognitionEngine**
  - 입력: ReasoningEngine 결과, 복잡도 스코어, 메모리 상태 요약
  - 출력: 품질 스코어, 경고/권장 액션, 로깅 페이로드
  - 사용처: 시스템 품질 모니터링, fallback 제어, 학습 시스템 신호

## 2. 기술 요소 및 의존성
- 한국어 형태소 분석: `paca/integrations/nlp` 하위의 KoNLPy 래퍼 사용, 필요시 `konlpy`/`mecab` 플러그인
- 통계 피처: 문장 길이, 품사 비율, 의문/조건문 패턴, 도메인 키워드 빈도
- ML/규칙 하이브리드: 초기 버전은 규칙+가중치 합산으로 구현, 모델 전환 대비 인터페이스 분리(`ComplexityFeatureExtractor`, `ComplexityRuleEngine`)
- 로깅: `PacaLogger` DEBUG 채널에 피처/스코어/결정 근거 JSON 덤프

## 3. 구현 계획
1. **ComplexityDetector 리팩터링**
   - `extract_features(text: str) -> Dict[str, float]`
   - `score_features(features) -> Tuple[float, str, List[str]]`
   - 모듈 구조: async 메서드 유지, 캐시 사용(최근 50개 문장)
2. **MetacognitionEngine 확장**
   - ReasoningResult 파싱, Confidence/Execution time 기반 품질지표 계산
   - 복잡도 스코어와 조합해 `QualityLevel` 산출 (e.g., GREEN/YELLOW/RED)
   - 피드백 훅: `emit_quality_alert()` -> EventBus
3. **데이터 의존성**
   - 기준값 저장: `data/config/complexity_thresholds.json`
   - 학습 로그: `logs/metacognition/`에 일별 파일 생성

## 4. 테스트 전략(스프린트 1 범위)
- 유닛 테스트: 피처 추출 함수, 스코어 매핑, 품질 등급 결정 로직
- 시나리오 테스트: `tests/phase2/test_complexity_metacognition.py`
  - 낮은/중간/높은 복잡도 입력 케이스
  - Reasoning 실패/성공 시 메타인지 품질 변화 확인
- 모킹: KoNLPy 통합이 준비되지 않은 환경에서는 간단한 토크나이저 대체

## 5. 리스크 및 대응
- 형태소 분석 라이브러리 설치 어려움 → `requirements-optional.txt` 갱신 및 모킹 옵션 제공
- 성능 저하 가능성 → 피처 계산 캐시, 비동기 병렬화 고려
- 품질 지표 정의 미비 → 초기에는 경험적 값 사용, 로그 수집 후 보정

## 6. 산출물 체크리스트
- [x] `ComplexityDetector` 리팩터링 완료 + 단위 테스트 3종 이상 *(tests/phase2/test_complexity_metacognition.py)*
- [x] `MetacognitionEngine` 품질 스코어 계산 로직 구현 *(품질 레벨/알림/로그 경로 반영)*
- [x] `tests/phase2/test_complexity_metacognition.py` 작성 및 통과
- [x] `tests/phase2/test_phase2_pipeline.py`로 PacaSystem 복잡도→추론→메타인지 통합 경로 검증
- [x] `data/config/complexity_thresholds.json` 초기값 정의
- [x] 로그 파이프라인/문서 업데이트 (`docs/phase2/complexity_metacognition_design.md` 최신화)
