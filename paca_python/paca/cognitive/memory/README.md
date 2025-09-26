# Memory System - Python 구현체

## 🎯 프로젝트 개요
PACA Python 시스템의 메모리 관리 모듈입니다. 작업 메모리, 에피소드 메모리, 장기 메모리를 통해 인지 시스템의 정보 저장 및 검색 기능을 제공합니다.

## 📁 폴더/파일 구조

```
memory/
├── __init__.py               # 메모리 모듈 초기화
├── types.py                  # 메모리 타입 정의
├── working.py                # 작업 메모리 구현
├── episodic.py               # 에피소드 메모리 구현
└── longterm.py               # 장기 메모리 구현
```

## ⚙️ 기능 요구사항

### 입력
- **작업 데이터**: 현재 처리 중인 정보
- **에피소드 정보**: 시간과 컨텍스트가 포함된 경험
- **장기 정보**: 영구 저장이 필요한 지식

### 출력
- **메모리 조회 결과**: 저장된 정보의 검색 결과
- **메모리 상태**: 각 메모리 시스템의 현재 상태
- **관련성 점수**: 검색된 정보의 관련성 평가

### 핵심 로직 흐름
1. **정보 분류** → **적절한 메모리 선택** → **저장/검색** → **관련성 평가** → **결과 반환**

## 🛠️ 기술적 요구사항

### 언어 및 프레임워크
- **Python 3.9+**: 비동기 처리 및 타입 힌트
- **AsyncIO**: 비동기 메모리 작업
- **Dataclasses**: 메모리 항목 구조 정의

### 주요 알고리즘
- **Vector Similarity**: 유사도 기반 검색
- **Temporal Decay**: 시간 기반 망각 모델
- **Priority Queue**: 중요도 기반 관리

## 🚀 라우팅 및 진입점

### 사용 예제
```python
from paca.cognitive.memory import WorkingMemory, EpisodicMemory, LongTermMemory

# 작업 메모리 사용
working_mem = WorkingMemory()
await working_mem.store("current_task", task_data)
result = await working_mem.retrieve("current_task")

# 에피소드 메모리 사용
episodic_mem = EpisodicMemory()
await episodic_mem.store_episode(
    content="학습 경험",
    context={"domain": "수학", "difficulty": "중급"},
    timestamp=datetime.now()
)

# 장기 메모리 사용
longterm_mem = LongTermMemory()
await longterm_mem.store_knowledge(
    key="mathematical_concept",
    value=concept_data,
    importance=0.9
)
```

## 📋 코드 품질 가이드

### 메모리 관리 원칙
- **효율성**: 최소한의 메모리 사용으로 최대 효과
- **일관성**: 모든 메모리 시스템에서 일관된 인터페이스
- **지속성**: 중요한 정보의 안정적인 보존

## 🏃‍♂️ 실행 방법

### 기본 메모리 시스템
```python
from paca.cognitive.memory import create_memory_system

memory_system = create_memory_system()
await memory_system.initialize()

# 정보 저장
await memory_system.store("키", "값", memory_type="working")

# 정보 검색
result = await memory_system.search("검색어")
```

## 🧪 테스트 방법

### 단위 테스트
```bash
pytest tests/cognitive/memory/test_working.py -v
pytest tests/cognitive/memory/test_episodic.py -v
pytest tests/cognitive/memory/test_longterm.py -v
```

## 🔒 추가 고려사항

### 성능
- **캐싱**: 자주 접근하는 정보의 캐싱
- **인덱싱**: 효율적인 검색을 위한 인덱스 구조
- **압축**: 메모리 사용량 최적화

### 향후 개선
- **분산 메모리**: 여러 노드에 걸친 메모리 관리
- **학습 기반**: 사용 패턴 학습을 통한 자동 최적화
- **감정 연동**: 감정 상태와 연관된 메모리 관리