# 🎯 프로젝트 개요

PACA v5 Controllers 모듈 - AI 어시스턴트의 메인 컨트롤러, 감정분석, 실행 제어, 입력 검증을 담당하는 핵심 제어 시스템입니다.

## 📁 폴더/파일 구조

```
controllers/
├── 📄 __init__.py           # 모듈 초기화 및 공개 API
├── 📄 main.py               # 메인 컨트롤러 및 요청 라우팅
├── 📄 sentiment.py          # 감정 분석 시스템
├── 📄 execution.py          # 실행 제어 및 작업 관리
├── 📄 validation.py         # 입력 검증 및 데이터 살균
└── 📄 README.md             # 이 문서
```

**파일별 설명:**
- `main.py`: MainController, 요청 라우팅, 미들웨어 시스템
- `sentiment.py`: SentimentAnalyzer, 한국어 감정 분석, 감정 추세 분석
- `execution.py`: ExecutionController, 작업 실행, 리소스 관리, 동시성 제어
- `validation.py`: InputValidator, 데이터 검증, 보안 검사, 스키마 검증

## ⚙️ 기능 요구사항

**입력:**
- 사용자 요청 및 입력 데이터
- 실행할 작업 및 함수
- 검증할 데이터 및 스키마

**출력:**
- 처리된 응답 및 결과
- 감정 분석 결과 및 추세
- 작업 실행 결과 및 상태
- 검증 결과 및 살균된 데이터

**핵심 로직 흐름:**
1. 요청 수신 및 미들웨어 처리
2. 입력 검증 및 보안 검사
3. 감정 분석 (텍스트 입력 시)
4. 적절한 핸들러로 라우팅
5. 작업 실행 및 결과 반환

## 🛠️ 기술적 요구사항

**언어 및 프레임워크:**
- Python 3.8+
- asyncio (비동기 처리)
- psutil (시스템 리소스 모니터링)
- concurrent.futures (스레드 풀 실행)

**주요 의존성:**
- `core.types`: 기본 타입 및 ID 시스템
- `core.events`: 이벤트 기반 통신
- `core.utils`: 로깅 및 유틸리티
- `core.errors`: 예외 처리 시스템

**실행 환경:**
- 메모리: 최소 512MB (작업 큐 및 실행 관리용)
- CPU: 멀티코어 권장 (동시 작업 처리)

## 🚀 라우팅 및 진입점

**주요 진입점:**
```python
from paca.controllers import (
    MainController,
    SentimentAnalyzer,
    ExecutionController,
    InputValidator
)

# 메인 컨트롤러 초기화
controller = MainController()
await controller.start()

# 요청 처리
result = await controller.process_request(
    input_data="안녕하세요",
    user_id="user123"
)

# 감정 분석
sentiment_analyzer = SentimentAnalyzer()
sentiment = await sentiment_analyzer.analyze("기쁘다")

# 작업 실행
execution_controller = ExecutionController()
task_id = await execution_controller.submit_task(
    "task_name", my_function, args=(arg1, arg2)
)

# 입력 검증
validator = InputValidator()
validation_result = await validator.validate(user_input)
```

**API 경로:**
- `MainController.process_request()`: 메인 요청 처리 인터페이스
- `SentimentAnalyzer.analyze()`: 텍스트 감정 분석
- `ExecutionController.submit_task()`: 작업 제출 및 실행
- `InputValidator.validate()`: 입력 데이터 검증

## 📋 코드 품질 가이드

**주석 규칙:**
- 모든 컨트롤러 메서드에 비즈니스 로직 및 플로우 설명 필수
- 감정 분석 알고리즘은 한국어 특성 및 키워드 설명
- 실행 제어는 동시성 및 리소스 관리 주의사항 기술

**네이밍 규칙:**
- 컨트롤러: [Name]Controller (MainController, ExecutionController)
- 분석기: [Name]Analyzer (SentimentAnalyzer)
- 검증기: [Name]Validator (InputValidator, SchemaValidator)
- 상태: [Module]State enum 사용

**예외 처리:**
- ControllerError: 컨트롤러 관련 오류
- ValidationError: 입력 검증 실패
- ExecutionError: 작업 실행 오류
- SentimentAnalysisError: 감정 분석 오류

## 🏃‍♂️ 실행 방법

**설치:**
```bash
# 프로젝트 루트에서
pip install -e .

# 시스템 리소스 모니터링용
pip install psutil
```

**메인 컨트롤러 사용법:**
```python
import asyncio
from paca.controllers import MainController, ControllerConfig

async def main():
    # 설정 및 초기화
    config = ControllerConfig(
        max_concurrent_requests=10,
        enable_sentiment_analysis=True,
        enable_input_validation=True
    )

    controller = MainController(config)
    await controller.start()

    # 요청 처리
    result = await controller.process_request(
        input_data="오늘 기분이 좋아요!",
        user_id="user123",
        session_id="session456"
    )

    print(f"응답: {result.data}")
    print(f"처리 시간: {result.processing_time:.3f}초")

    await controller.stop()

# 실행
asyncio.run(main())
```

**감정 분석 사용법:**
```python
from paca.controllers import SentimentAnalyzer

# 감정 분석기 초기화
analyzer = SentimentAnalyzer()

# 텍스트 감정 분석
result = await analyzer.analyze("정말 화가 나네요!")

print(f"감정: {result.emotion_type.value}")
print(f"강도: {result.emotion_intensity:.2f}")
print(f"키워드: {result.detected_keywords}")

# 감정 추세 분석 (사용자별)
trend = await analyzer.analyze_emotion_trend("user123", days=7)
if trend:
    print(f"주요 감정: {trend.dominant_emotions}")
    print(f"추세: {trend.trend_direction}")
```

**테스트 실행:**
```bash
# Controllers 모듈 테스트
python -m pytest tests/controllers/ -v

# 커버리지 포함
python -m pytest tests/controllers/ --cov=paca.controllers
```

## 🧪 테스트 방법

**단위 테스트:**
- 각 컨트롤러의 개별 기능 테스트
- 감정 분석 정확성 검증 (한국어 키워드)
- 입력 검증 규칙 및 보안 검사
- 작업 실행 및 동시성 제어

**통합 테스트:**
- 전체 요청 처리 파이프라인 테스트
- 컨트롤러 간 협력 및 데이터 흐름 검증
- 미들웨어 체인 및 오류 처리

**성능 테스트:**
- 동시 요청 처리 능력 (목표: 10개 동시 요청)
- 감정 분석 응답 시간 (<100ms)
- 작업 실행 처리량 및 리소스 사용량

**테스트 시나리오:**
```python
async def test_main_controller():
    """메인 컨트롤러 테스트"""
    controller = MainController()
    await controller.start()

    # 기본 요청 처리
    result = await controller.process_request("안녕하세요")
    assert result.success
    assert result.data is not None

    await controller.stop()

async def test_sentiment_analysis():
    """감정 분석 테스트"""
    analyzer = SentimentAnalyzer()

    # 긍정 감정 테스트
    result = await analyzer.analyze("정말 기쁩니다!")
    assert result.emotion_type == EmotionType.HAPPY
    assert result.emotion_intensity > 0.5

    # 부정 감정 테스트
    result = await analyzer.analyze("너무 슬퍼요")
    assert result.emotion_type == EmotionType.SAD

async def test_execution_controller():
    """실행 제어기 테스트"""
    controller = ExecutionController()
    await controller.start()

    # 작업 제출 및 실행
    def test_task(x, y):
        return x + y

    task_id = await controller.submit_task(
        "addition", test_task, args=(2, 3)
    )

    # 결과 대기 및 확인
    result = await controller.execute_function(test_task, 2, 3)
    assert result == 5

    await controller.stop()

async def test_input_validation():
    """입력 검증 테스트"""
    validator = InputValidator()

    # 안전한 입력
    safe_result = await validator.validate("안전한 텍스트입니다")
    assert safe_result.is_valid

    # 위험한 입력
    dangerous_result = await validator.validate("<script>alert('xss')</script>")
    assert not dangerous_result.is_valid
    assert len(dangerous_result.errors) > 0
```

## 💡 추가 고려사항

**보안:**
- 입력 데이터 XSS, SQL 인젝션 방지
- 작업 실행 시 리소스 제한 및 샌드박싱
- 감정 분석 데이터 개인정보 보호
- 요청 인증 및 권한 검증

**성능:**
- 비동기 처리를 통한 높은 동시성
- 감정 분석 결과 캐싱으로 응답 속도 향상
- 작업 큐 및 우선순위 기반 스케줄링
- 리소스 모니터링 및 자동 스케일링

**향후 개선:**
- 기계학습 기반 고도화된 감정 분석
- 분산 작업 실행 시스템 구축
- 실시간 감정 추세 알림 시스템
- 다국어 감정 분석 지원 확장

**모니터링:**
- 요청 처리 성능 메트릭 수집
- 감정 분석 정확도 추적
- 작업 실행 성공률 및 실패 원인 분석
- 시스템 리소스 사용량 모니터링

**Controllers 모듈 특화 기능:**
- **한국어 감정 분석**: 한국어 특화 키워드 및 문맥 분석
- **실시간 작업 제어**: 동시성 제어 및 리소스 관리
- **계층화된 검증**: 타입, 스키마, 보안 단계별 검증
- **미들웨어 아키텍처**: 확장 가능한 요청 처리 파이프라인

**품질 지표:**
- **응답 성능**: <200ms (단순 요청), <500ms (복합 요청)
- **감정 분석 정확도**: >85% (한국어 텍스트 기준)
- **동시 처리**: 10개 이상 동시 요청 처리
- **안정성**: 99.9% 요청 처리 성공률