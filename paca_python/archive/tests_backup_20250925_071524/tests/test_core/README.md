# PACA 핵심 모듈 테스트 모음

## 🎯 프로젝트 개요
PACA 시스템의 핵심 기능들(core, types, events, utils, errors, constants)을 검증하는 단위 테스트 모듈입니다.

## 📁 폴더/파일 구조
```
test_core/
├── test_types/          # 데이터 타입 테스트
│   ├── test_base_types.py      # 기본 타입 테스트
│   ├── test_memory_types.py    # 메모리 타입 테스트
│   ├── test_cognitive_types.py # 인지 타입 테스트
│   └── test_api_types.py       # API 타입 테스트
├── test_events/         # 이벤트 시스템 테스트
│   ├── test_event_bus.py       # 이벤트 버스 테스트
│   ├── test_event_handlers.py  # 이벤트 핸들러 테스트
│   ├── test_event_types.py     # 이벤트 타입 테스트
│   └── test_pub_sub.py         # Pub/Sub 패턴 테스트
├── test_utils/          # 유틸리티 함수 테스트
│   ├── test_string_utils.py    # 문자열 유틸리티
│   ├── test_file_utils.py      # 파일 유틸리티
│   ├── test_crypto_utils.py    # 암호화 유틸리티
│   └── test_validation_utils.py# 검증 유틸리티
├── test_errors/         # 에러 처리 테스트
│   ├── test_custom_exceptions.py # 커스텀 예외 테스트
│   ├── test_error_handlers.py    # 에러 핸들러 테스트
│   └── test_recovery_mechanisms.py # 복구 메커니즘 테스트
└── test_constants/      # 상수 및 설정 테스트
    ├── test_config_constants.py  # 설정 상수 테스트
    ├── test_system_constants.py  # 시스템 상수 테스트
    └── test_api_constants.py     # API 상수 테스트
```

## ⚙️ 기능 요구사항
- **입력**: 핵심 모듈 함수, 클래스, 상수
- **출력**: 테스트 결과, 커버리지 리포트
- **핵심 로직**: 단위 테스트, 모킹, 어서션 검증

## 🛠️ 기술적 요구사항
- **언어**: Python
- **프레임워크**: pytest, unittest
- **모킹**: unittest.mock
- **커버리지**: pytest-cov

## 🚀 라우팅 및 진입점
- 전체 핵심 테스트: `pytest tests/test_core/`
- 특정 모듈 테스트: `pytest tests/test_core/test_types/`
- 커버리지 측정: `pytest tests/test_core/ --cov=paca.core`

## 📋 코드 품질 가이드
- 테스트 커버리지 90% 이상 유지
- 각 함수별 최소 3개 테스트 케이스
- 경계값 테스트 포함 필수
- Mock 사용으로 의존성 격리

## 🏃‍♂️ 실행 방법
```bash
# 전체 핵심 모듈 테스트
pytest tests/test_core/ -v

# 특정 모듈만 테스트
pytest tests/test_core/test_types/ -v

# 커버리지 포함 테스트
pytest tests/test_core/ --cov=paca.core --cov-report=html

# 빠른 테스트 (단위 테스트만)
pytest tests/test_core/ -m "not integration"
```

## 🧪 테스트 방법
- **단위 테스트**: 개별 함수/클래스 기능 검증
- **통합 테스트**: 모듈 간 인터페이스 검증
- **성능 테스트**: 핵심 함수 성능 측정

## 💡 추가 고려사항
- **보안**: 입력 검증 및 보안 취약점 테스트
- **성능**: 핵심 함수 성능 회귀 방지
- **향후 개선**: 속성 기반 테스트(Property-based testing), 변이 테스트