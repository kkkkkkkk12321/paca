# Curiosity System - Python 구현체

## 🎯 프로젝트 개요
PACA Python 시스템의 호기심 모듈입니다. 자발적 학습, 탐색적 행동, 새로운 정보에 대한 관심을 통해 시스템의 자율적 학습과 지적 성장을 촉진합니다.

## 📁 폴더/파일 구조

```
curiosity/
├── __init__.py               # 호기심 시스템 초기화
├── bounded_curiosity.py      # 경계를 가진 호기심 시스템
├── curiosity_engine.py       # 호기심 엔진
├── exploration_planner.py    # 탐색 계획 수립
├── gap_detector.py           # 지식 격차 탐지
├── mission_aligner.py        # 미션 정렬 시스템
└── README.md                 # 모듈 문서
```

## ⚙️ 기능 요구사항

### 입력
- **환경 상태**: 현재 시스템 및 외부 환경 정보
- **학습 경험**: 과거 학습 데이터 및 경험
- **탐색 대상**: 새로운 정보나 미지의 영역

### 출력
- **탐색 방향**: 다음에 탐색할 영역 제안
- **호기심 점수**: 각 대상에 대한 관심도 점수
- **학습 우선순위**: 학습할 내용의 우선순위

### 핵심 로직 흐름
1. **환경 관찰** → **새로움 탐지** → **관심도 계산** → **탐색 방향 결정** → **탐색 실행** → **보상 평가**

## 🛠️ 기술적 요구사항

### 언어 및 프레임워크
- **Python 3.9+**: 비동기 처리 및 타입 힌트
- **NumPy**: 수치 계산 및 벡터 연산
- **Scikit-learn**: 기계학습 알고리즘

### 주요 알고리즘
- **Information Gain**: 정보 획득량 계산
- **Intrinsic Motivation**: 내재적 동기 모델링
- **Exploration vs Exploitation**: 탐색-활용 균형

## 🚀 라우팅 및 진입점

### 사용 예제
```python
from paca.cognitive.curiosity import BoundedCuriosity, CuriosityEngine

# 경계 호기심 시스템 초기화
bounded_curiosity = BoundedCuriosity()
await bounded_curiosity.initialize()

# 탐구 대상 평가
curiosity_level = await bounded_curiosity.evaluate_curiosity(
    stimulus="새로운 정보나 개념",
    context="현재 맥락"
)

# 미션 정렬 확인
alignment_result = await bounded_curiosity.check_mission_alignment(
    exploration_target="탐구하고자 하는 주제"
)

print(f"호기심 수준: {curiosity_level}")
print(f"미션 정렬도: {alignment_result.alignment_score}")
```

## 📋 코드 품질 가이드

### 호기심 모델링 원칙
- **자율성**: 외부 지시 없이 자발적 탐색
- **적응성**: 환경 변화에 따른 관심사 조정
- **효율성**: 한정된 자원 내에서 최적 탐색

## 🏃‍♂️ 실행 방법

### 기본 호기심 시스템
```python
from paca.cognitive.curiosity import CuriosityEngine

engine = CuriosityEngine()
await engine.initialize()

# 호기심 기반 탐색 시작
await engine.start_exploration()
```

## 🧪 테스트 방법

### 단위 테스트
```bash
pytest tests/cognitive/test_curiosity.py -v
```

## 🔒 추가 고려사항

### 성능
- **실시간 계산**: 빠른 호기심 점수 계산
- **메모리 효율성**: 효율적인 관심사 저장

### 향후 개선
- **감정 연동**: 감정 상태와 호기심의 상호작용
- **사회적 호기심**: 다른 에이전트와의 상호작용
- **메타 호기심**: 호기심 자체에 대한 탐구