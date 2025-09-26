# Learning Patterns Module - PACA Python v5

## 🎯 프로젝트 개요
PACA 학습 시스템의 패턴 인식 및 분석 모듈로, 사용자 행동 패턴, 성공/실패 패턴, 학습 패턴을 감지하고 분석하여 자동 학습을 지원합니다.

## 📁 폴더/파일 구조
```
patterns/
├── __init__.py              # 모듈 진입점
├── detector.py              # 패턴 감지 엔진
└── analyzer.py              # 패턴 분석 및 분류
```

## ⚙️ 기능 요구사항
**입력**: 사용자 상호작용 데이터, 성공/실패 기록, 학습 세션 정보
**출력**: 감지된 패턴, 패턴 분석 결과, 개선 제안
**핵심 로직**: 데이터 수집 → 패턴 감지 → 분류 → 분석 → 학습 피드백

## 🛠️ 기술적 요구사항
- Python 3.9+ (numpy, scipy, sklearn)
- 통계적 패턴 분석 알고리즘
- 시계열 데이터 처리

## 🚀 라우팅 및 진입점
```python
from paca.learning.patterns import PatternDetector, PatternAnalyzer

# 패턴 감지
detector = PatternDetector()
patterns = await detector.detect_patterns(interaction_data)

# 패턴 분석
analyzer = PatternAnalyzer()
analysis = await analyzer.analyze_patterns(patterns)
```

## 📋 코드 품질 가이드
- 클래스: PascalCase (PatternDetector, PatternAnalyzer)
- 패턴 타입: UPPER_SNAKE_CASE (SUCCESS_PATTERN, FAILURE_PATTERN)
- 모든 분석 함수에 신뢰도 점수 포함 필수

## 🏃‍♂️ 실행 방법
```bash
python -c "from paca.learning.patterns import PatternDetector; print('Patterns module loaded')"
```

## 🧪 테스트 방법
```bash
pytest tests/test_learning/test_patterns/ -v
```

## 💡 추가 고려사항
**성능**: 실시간 패턴 감지, 증분 학습
**향후 개선**: 딥러닝 기반 패턴 인식, 예측 모델링