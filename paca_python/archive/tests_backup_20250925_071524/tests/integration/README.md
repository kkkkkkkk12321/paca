# PACA 통합 테스트 모음

## 🎯 프로젝트 개요
PACA 시스템의 모듈 간 상호작용과 전체 시스템 동작을 검증하는 통합 테스트 모듈입니다.

## 📁 폴더/파일 구조
```
integration/
├── api_integration/     # API 통합 테스트
│   ├── test_llm_api.py      # LLM API 통합 테스트
│   ├── test_memory_api.py   # 메모리 시스템 통합
│   └── test_feedback_api.py # 피드백 시스템 통합
├── cognitive_integration/  # 인지 시스템 통합 테스트
│   ├── test_memory_cognitive.py # 메모리-인지 통합
│   ├── test_learning_cognitive.py # 학습-인지 통합
│   └── test_reasoning_cognitive.py # 추론-인지 통합
├── system_integration/    # 전체 시스템 통합 테스트
│   ├── test_full_workflow.py # 전체 워크플로우 테스트
│   ├── test_error_handling.py # 오류 처리 통합 테스트
│   └── test_performance_integration.py # 성능 통합 테스트
└── e2e/                   # End-to-End 테스트
    ├── test_user_scenarios.py # 사용자 시나리오 테스트
    ├── test_desktop_app.py    # 데스크탑 앱 E2E
    └── test_api_endpoints.py  # API 엔드포인트 E2E
```

## ⚙️ 기능 요구사항
- **입력**: 시스템 구성, 테스트 시나리오, 모의 데이터
- **출력**: 통합 테스트 결과, 성능 메트릭
- **핵심 로직**: 모듈 간 인터페이스 검증, 데이터 플로우 검증

## 🛠️ 기술적 요구사항
- **언어**: Python
- **프레임워크**: pytest, pytest-asyncio
- **모킹**: unittest.mock, responses
- **데이터**: 테스트 픽스처, 모의 데이터 생성

## 🚀 라우팅 및 진입점
- 통합 테스트 실행: `pytest tests/integration/`
- 특정 모듈: `pytest tests/integration/api_integration/`
- E2E 테스트: `pytest tests/integration/e2e/`

## 📋 코드 품질 가이드
- 실제 환경과 유사한 테스트 환경 구성
- 외부 의존성은 모킹 처리
- 테스트 격리를 위한 fixture 사용
- 명확한 테스트 시나리오 문서화

## 🏃‍♂️ 실행 방법
```bash
# 전체 통합 테스트
pytest tests/integration/ -v

# API 통합 테스트만
pytest tests/integration/api_integration/ -v

# E2E 테스트 (느림)
pytest tests/integration/e2e/ -v --slow

# 병렬 실행
pytest tests/integration/ -n auto
```

## 🧪 테스트 방법
- **통합 테스트**: 모듈 간 인터페이스 검증
- **E2E 테스트**: 전체 사용자 워크플로우 검증
- **성능 테스트**: 통합 환경에서의 성능 측정

## 💡 추가 고려사항
- **보안**: 테스트 환경에서의 보안 검증
- **성능**: 통합 테스트 실행 시간 최적화
- **향후 개선**: 자동 통합 테스트 파이프라인, 시각적 회귀 테스트