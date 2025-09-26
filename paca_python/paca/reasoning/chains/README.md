# Reasoning Chains Module - PACA Python v5

## 🎯 프로젝트 개요
PACA 추론 시스템의 체인 추론 모듈로, 단계적 논리 전개, 추론 체인 관리, 백트래킹, 체인 검증을 통해 복잡한 문제를 체계적으로 해결합니다.

## 📁 폴더/파일 구조
```
chains/
├── __init__.py              # 추론 체인 전체 구현 (600+줄)
│   ├── ReasoningStepType    # 추론 단계 타입 열거형
│   ├── ReasoningMethod      # 추론 방법 열거형
│   ├── ReasoningStep        # 개별 추론 단계 클래스
│   ├── ReasoningChain       # 추론 체인 관리 클래스
│   └── ReasoningChainManager# 체인 매니저 클래스
└── README.md                # 모듈 문서
```

## ⚙️ 기능 요구사항
**입력**: 문제 명세, 추론 전략, 초기 조건, 목표 상태
**출력**: 추론 체인 결과, 각 단계별 논리, 결론의 타당성
**핵심 로직**: 문제 분해 → 추론 단계 생성 → 순차 실행 → 검증 → 백트래킹(필요시)

## 🛠️ 기술적 요구사항
- Python 3.9+ (asyncio, dataclasses, typing)
- 논리적 일관성 검증 알고리즘
- 백트래킹 및 대안 경로 탐색

## 🚀 라우팅 및 진입점
```python
from paca.reasoning.chains import ReasoningChainManager, ReasoningChain

# 추론 체인 생성
manager = ReasoningChainManager()
chain = await manager.create_chain(
    problem="복잡한 수학 문제",
    strategy="step_by_step"
)

# 체인 실행
result = await chain.execute()
print(f"결론: {result.conclusion}")
print(f"단계 수: {len(result.steps)}")
```

## 📋 코드 품질 가이드
- 클래스: PascalCase (ReasoningChain, ChainManager)
- 추론 단계: 명확한 입력/출력 정의 필수
- 모든 추론 단계에 논리적 근거 포함
- 백트래킹 조건 명시적 정의

## 🏃‍♂️ 실행 방법
```bash
python -c "from paca.reasoning.chains import ReasoningChainManager; print('Chains module loaded')"
```

## 🧪 테스트 방법
```bash
pytest tests/test_reasoning/test_chains/ -v
```

## 💡 추가 고려사항
**성능**: 병렬 체인 실행, 중간 결과 캐싱
**향후 개선**: 확률적 추론, 그래프 기반 체인, 시각화