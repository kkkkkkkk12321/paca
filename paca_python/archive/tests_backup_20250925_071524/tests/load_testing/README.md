# PACA 부하 테스트 시스템

## 🎯 프로젝트 개요
PACA 시스템의 성능 한계와 확장성을 검증하는 부하 테스트로, 대규모 사용자 및 데이터 처리를 시뮬레이션합니다.

## 📁 폴더/파일 구조
```
load_testing/
├── test_concurrent_users.py  # 동시 사용자 부하 테스트
└── test_memory_stress.py     # 메모리 스트레스 테스트
```

## ⚙️ 기능 요구사항
- **입력**: 부하 시나리오, 사용자 수, 실행 시간
- **출력**: 성능 메트릭, 처리량, 응답 시간 분포
- **핵심 로직**: 점진적 부하 증가, 임계점 탐지, 성능 모니터링

## 🛠️ 기술적 요구사항
- **언어**: Python 3.9+
- **프레임워크**: locust, pytest-benchmark
- **라이브러리**: asyncio, multiprocessing, psutil
- **모니터링**: CPU, 메모리, 네트워크 사용량

## 🚀 라우팅 및 진입점
- 동시 사용자 테스트: `python tests/load_testing/test_concurrent_users.py`
- 메모리 스트레스: `python tests/load_testing/test_memory_stress.py`
- Locust 부하 테스트: `locust -f tests/load_testing/locustfile.py`

## 📋 코드 품질 가이드
- 현실적인 부하 시나리오
- 점진적 부하 증가 패턴
- 성능 메트릭 수집 최적화
- 메모리 누수 탐지

## 🏃‍♂️ 실행 방법
```bash
# 동시 사용자 부하 테스트
python -m tests.load_testing.test_concurrent_users --users 100

# 메모리 스트레스 테스트
python -m tests.load_testing.test_memory_stress --duration 300

# 부하 테스트 리포트 생성
locust -f tests/load_testing/ --html reports/load_test.html
```

## 🧪 테스트 방법
- **부하 테스트**: 점진적 사용자 수 증가
- **스트레스 테스트**: 시스템 한계점 탐지
- **안정성 테스트**: 장기간 부하 상황 검증

## 💡 추가 고려사항
- **확장성**: 수평적/수직적 확장성 평가
- **성능**: 병목 지점 식별 및 최적화
- **향후 개선**: 클라우드 기반 부하 테스트, 자동 성능 회귀 검증