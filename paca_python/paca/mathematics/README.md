# 🎯 프로젝트 개요

PACA v5 수학 연산 모듈 - AI 어시스턴트의 수학적 계산, 통계 분석, 수학적 추론 기능을 제공하는 핵심 모듈입니다.

## 📁 폴더/파일 구조

```
mathematics/
├── 📄 __init__.py           # 모듈 초기화 및 공개 API
├── 📄 calculator.py         # 기본 계산기 및 통계 분석기
├── 📁 proof/                # 수학적 증명 시스템 (계획 중)
├── 📁 python/               # Python 수학 라이브러리 통합 (계획 중)
├── 📁 quality/              # 수학 품질 평가 시스템 (계획 중)
├── 📁 reasoning/            # 수학적 추론 엔진 (계획 중)
└── 📄 README.md             # 이 문서
```

**파일별 설명:**
- `calculator.py`: Calculator, StatisticalAnalyzer 클래스 구현
- `__init__.py`: MathematicalReasoningEngine, 품질 평가 시스템
- `proof/`: 수학적 증명 검증 및 생성 시스템 (향후 구현)
- `reasoning/`: 고급 수학적 추론 및 문제 해결 (향후 구현)

## ⚙️ 기능 요구사항

**입력:**
- 수학적 표현식 및 계산 요청
- 통계 데이터 및 분석 요구사항
- 수학 문제 및 증명 요청

**출력:**
- 계산 결과 및 정확성 검증
- 통계 분석 결과 및 시각화 데이터
- 수학적 해결 과정 및 품질 평가

**핵심 로직 흐름:**
1. 수학 표현식 파싱 및 검증
2. 적절한 계산 방법 선택
3. 계산 수행 및 결과 검증
4. 품질 평가 및 피드백 제공

## 🛠️ 기술적 요구사항

**언어 및 프레임워크:**
- Python 3.8+
- numpy (수치 계산)
- sympy (기호 수학)
- scipy (과학 계산)

**주요 의존성:**
- `core.types`: 기본 타입 및 결과 처리
- `core.utils`: 로깅 및 유틸리티
- `core.errors`: 수학 관련 예외 처리

**실행 환경:**
- CPU: 수치 계산 최적화 지원 권장
- 메모리: 최소 128MB (대용량 계산용)

## 🚀 라우팅 및 진입점

**주요 진입점:**
```python
from paca.mathematics import (
    Calculator,
    StatisticalAnalyzer,
    MathematicalReasoningEngine,
    MathQualityEvaluator
)

# 기본 계산기
calc = Calculator()
result = calc.add(2, 3)

# 통계 분석기
analyzer = StatisticalAnalyzer()
stats = analyzer.analyze([1, 2, 3, 4, 5])

# 수학적 추론 엔진
engine = MathematicalReasoningEngine()
solution = engine.solve(expression)
```

**API 경로:**
- `Calculator.calculate()`: 기본 수학 연산
- `StatisticalAnalyzer.analyze()`: 통계 분석
- `MathematicalReasoningEngine.solve()`: 수학 문제 해결

## 📋 코드 품질 가이드

**주석 규칙:**
- 모든 수학 함수에 수학적 표기법 포함
- 복잡한 알고리즘은 수식으로 설명
- 통계 메서드는 통계적 의미 명시

**네이밍 규칙:**
- 수학 함수: 표준 수학 용어 사용 (mean, std, integral)
- 상수: MATH_CONSTANTS로 대문자
- 도메인: MathematicalDomain enum 사용

**예외 처리:**
- CalculationError: 계산 오류 및 수치적 불안정
- DivisionByZeroError: 0으로 나누기 오류
- InvalidExpressionError: 잘못된 수학 표현식

## 🏃‍♂️ 실행 방법

**설치:**
```bash
# 프로젝트 루트에서
pip install -e .
pip install numpy scipy sympy  # 수학 라이브러리
```

**기본 계산 사용법:**
```python
from paca.mathematics import Calculator

# 계산기 초기화
calc = Calculator()

# 기본 연산
print(calc.add(2, 3).value)        # 5.0
print(calc.multiply(4, 5).value)   # 20.0
print(calc.divide(10, 3).value)    # 3.333...

# 고급 연산
print(calc.power(2, 8).value)      # 256.0
print(calc.sqrt(16).value)         # 4.0
```

**통계 분석 사용법:**
```python
from paca.mathematics import StatisticalAnalyzer

# 분석기 초기화
analyzer = StatisticalAnalyzer()

# 데이터 분석
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result = analyzer.analyze(data)

print(f"평균: {result.mean}")
print(f"표준편차: {result.std}")
print(f"중앙값: {result.median}")
```

## 🧪 테스트 방법

**단위 테스트:**
- 각 수학 연산의 정확성 검증
- 경계값 및 예외 상황 테스트
- 수치적 안정성 검증

**통합 테스트:**
- 복합 수학 문제 해결 과정 검증
- 통계 분석 파이프라인 테스트
- 품질 평가 시스템 검증

**성능 테스트:**
- 대용량 데이터 계산 성능 (<100ms 목표)
- 메모리 사용량 최적화 검증
- 병렬 계산 성능 측정

**정확성 테스트:**
```python
def test_basic_operations():
    """기본 연산 정확성 테스트"""
    calc = Calculator()

    assert calc.add(2, 3).value == 5.0
    assert calc.subtract(5, 3).value == 2.0
    assert calc.multiply(4, 5).value == 20.0
    assert calc.divide(10, 2).value == 5.0

def test_statistical_analysis():
    """통계 분석 정확성 테스트"""
    analyzer = StatisticalAnalyzer()
    data = [1, 2, 3, 4, 5]

    result = analyzer.analyze(data)

    assert result.mean == 3.0
    assert result.median == 3.0
    assert abs(result.std - 1.58) < 0.01  # 근사값 검증

def test_mathematical_reasoning():
    """수학적 추론 테스트"""
    engine = MathematicalReasoningEngine()

    expression = MathematicalExpression(
        expression="x^2 + 2x + 1 = 0",
        variables=["x"],
        domain=MathematicalDomain.ALGEBRA,
        metadata={}
    )

    solution = engine.solve(expression)
    assert solution.confidence > 0.8
    assert len(solution.solution_steps) > 0
```

## 💡 추가 고려사항

**보안:**
- 수학 표현식 파싱 시 코드 인젝션 방지
- 무한 루프 및 과도한 계산 방지
- 계산 결과 검증 및 신뢰성 확보

**성능:**
- 복잡한 계산을 위한 캐싱 시스템
- 병렬 처리를 통한 대용량 데이터 처리
- 수치적 최적화 알고리즘 적용

**향후 개선:**
- 기호 수학 (SymPy) 통합 강화
- 기계학습 기반 수학 문제 해결
- 실시간 수학 시각화 기능
- 다국어 수학 표기법 지원

**수학적 도메인 지원:**
- **대수학**: 방정식 해결, 다항식 연산
- **미적분학**: 미분, 적분, 극한 계산
- **기하학**: 도형 계산, 변환 행렬
- **통계학**: 기술통계, 확률분포, 가설검정
- **이산수학**: 조합론, 그래프 이론
- **수론**: 소수, 합동, 디오판틴 방정식

**품질 지표:**
- 계산 정확도: >99.9% (기본 연산)
- 수치 안정성: IEEE 754 표준 준수
- 성능: <10ms (기본 연산), <100ms (복합 연산)
- 교육적 가치: 단계별 해결 과정 제공