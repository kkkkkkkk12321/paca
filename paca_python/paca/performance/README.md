# Performance Module - PACA Python v5

## 🎯 프로젝트 개요

PACA 시스템의 성능 최적화 모듈로, 실시간 하드웨어 모니터링과 동적 성능 프로파일 관리를 담당합니다. 시스템 리소스 상태를 지속적으로 모니터링하고 최적의 성능 프로파일을 자동으로 선택하여 효율적인 시스템 운영을 보장합니다.

## 📁 폴더/파일 구조

```
paca/performance/
├── __init__.py                    # 모듈 진입점 및 공개 API
├── hardware_monitor.py            # 실시간 하드웨어 모니터링 시스템
├── profile_manager.py             # 성능 프로파일 관리 시스템
└── README.md                      # 이 문서
```

### 핵심 파일 설명

- **`hardware_monitor.py`** (760줄): 실시간 시스템 리소스 모니터링, 알림 생성, 성능 메트릭 수집
- **`profile_manager.py`** (580줄): 4가지 성능 프로파일 관리, 자동 프로파일 전환, 사용자 정의 설정
- **`__init__.py`** (36줄): 모듈 API 통합 및 버전 관리

## ⚙️ 기능 요구사항

### 입력
- **시스템 상태**: CPU/메모리/디스크 사용률, 네트워크 트래픽
- **성능 요구사항**: 작업 유형, 예상 지속 시간, 리소스 제약사항
- **사용자 설정**: 프로파일 사용자 정의, 모니터링 간격, 알림 임계값

### 출력
- **시스템 상태 보고서**: 리소스 사용량, 건강도 점수, 성능 알림
- **프로파일 추천**: 최적 성능 프로파일, 전환 이유, 기대 효과
- **성능 메트릭**: 응답 시간, 처리량, 효율성 지표

### 핵심 로직 흐름
1. **모니터링 시작** → 시스템 리소스 지속 관찰 → 임계값 기반 알림 생성
2. **프로파일 분석** → 현재 상태 평가 → 최적 프로파일 추천 → 자동 전환
3. **성능 추적** → 메트릭 수집 → 트렌드 분석 → 개선 제안

## 🛠️ 기술적 요구사항

### 시스템 요구사항
- **Python**: 3.9+ (asyncio, typing 지원)
- **운영체제**: Windows, Linux, macOS
- **메모리**: 최소 128MB (conservative 프로파일), 권장 512MB+
- **CPU**: 단일 코어 이상 (멀티코어 권장)

### 의존성 라이브러리
- **psutil**: 시스템 리소스 모니터링 (CPU, 메모리, 디스크, 네트워크)
- **asyncio**: 비동기 모니터링 및 이벤트 처리
- **concurrent.futures**: 스레드 풀 기반 병렬 처리
- **dataclasses**: 구조화된 데이터 관리
- **enum**: 타입 안전한 상수 정의

### 성능 특성
- **모니터링 오버헤드**: < 5% CPU 사용량
- **메모리 사용량**: < 50MB (기본 설정)
- **응답 시간**: 시스템 상태 조회 < 150ms
- **처리량**: 초당 1-10회 모니터링 가능

## 🚀 라우팅 및 진입점

### 기본 사용법
```python
from paca.performance import HardwareMonitor, ProfileManager

# 하드웨어 모니터링 시작
monitor = HardwareMonitor(monitoring_interval=2.0)
status_result = await monitor.get_system_status()

if status_result.is_success:
    status = status_result.value
    print(f"CPU: {status.resource_usage.cpu_percent:.1f}%")
    print(f"Memory: {status.resource_usage.memory_percent:.1f}%")
    print(f"추천 프로파일: {status.recommended_profile}")
```

### 프로파일 관리
```python
# 프로파일 매니저 생성 및 전환
manager = ProfileManager()

# 자동 프로파일 선택
result = manager.auto_select_profile(
    cpu_percent=75.0,
    memory_percent=60.0,
    available_memory_mb=800
)

if result.is_success:
    profile = result.value
    print(f"선택된 프로파일: {profile.name}")
```

### 지속적 모니터링
```python
# 콜백 기반 실시간 모니터링
def status_callback(status):
    print(f"CPU: {status.resource_usage.cpu_percent:.1f}%")
    if status.alerts:
        for alert in status.alerts:
            print(f"알림: {alert.message}")

monitor.add_callback(status_callback)
await monitor.start_monitoring()
# ... 모니터링 실행 중 ...
await monitor.stop_monitoring()
```

## 📋 코드 품질 가이드

### 명명 규칙
- **클래스**: PascalCase (예: `HardwareMonitor`, `ProfileManager`)
- **함수/메서드**: snake_case (예: `get_system_status`, `auto_select_profile`)
- **상수**: UPPER_SNAKE_CASE (예: `DEFAULT_PROFILES`, `ALERT_THRESHOLDS`)
- **열거형**: PascalCase.UPPER_CASE (예: `ProfileType.HIGH_END`)

### 예외 처리
- 모든 비동기 메서드에 try-except 블록 필수 적용
- `Result[T]` 패턴 사용으로 오류 전파 명시적 처리
- 시스템 리소스 접근 실패시 기본값 반환 및 로깅
- 사용자 입력 검증 및 적절한 오류 메시지 제공

### 주석 및 문서화
- 모든 공개 클래스와 메서드에 docstring 작성
- 복잡한 알고리즘에 대한 상세 주석 포함
- 타입 힌트 완전 적용 (Python 3.9+ 지원)
- 사용 예제 및 주의사항 명시

## 🏃‍♂️ 실행 방법

### 설치 및 초기 설정
```bash
# 프로젝트 루트에서 설치
cd paca_python
pip install -e .

# 의존성 설치 확인
python -c "import psutil; print('psutil version:', psutil.__version__)"
```

### 기본 테스트 실행
```bash
# 통합 테스트 실행
python simple_phase3_test.py

# 개별 모듈 테스트
python -c "from paca.performance import HardwareMonitor; print('Hardware Monitor 로드 성공')"
python -c "from paca.performance import ProfileManager; print('Profile Manager 로드 성공')"
```

### 실제 환경 실행
```bash
# 하드웨어 모니터 독립 실행
python -m paca.performance.hardware_monitor

# 프로파일 매니저 독립 실행
python -m paca.performance.profile_manager
```

## 🧪 테스트 방법

### 단위 테스트
```bash
# 하드웨어 모니터 테스트
python -c "
from paca.performance import HardwareMonitor
monitor = HardwareMonitor()
result = monitor.get_system_status()
assert result.is_success, 'System status check failed'
print('Hardware monitor test passed')
"

# 프로파일 매니저 테스트
python -c "
from paca.performance import ProfileManager, ProfileType
manager = ProfileManager()
result = manager.switch_profile(ProfileType.HIGH_END)
assert result.is_success, 'Profile switch failed'
print('Profile manager test passed')
"
```

### 통합 테스트
```bash
# 전체 통합 테스트 실행
python simple_phase3_test.py

# 성능 측정 테스트
python -c "
import time
from paca.performance import HardwareMonitor

monitor = HardwareMonitor()
start = time.time()
result = monitor.get_system_status()
duration = (time.time() - start) * 1000

print(f'System status query: {duration:.1f}ms')
assert duration < 200, f'Performance too slow: {duration}ms'
print('Performance test passed')
"
```

### 스트레스 테스트
```bash
# 장시간 모니터링 테스트 (30초)
python -c "
import asyncio
from paca.performance import HardwareMonitor

async def stress_test():
    monitor = HardwareMonitor(monitoring_interval=0.5)
    await monitor.start_monitoring()
    await asyncio.sleep(30)
    await monitor.stop_monitoring()
    print('30-second monitoring stress test completed')

asyncio.run(stress_test())
"
```

## 💡 추가 고려사항

### 보안
- **리소스 접근 권한**: 시스템 정보 조회 권한 필요 (일반적으로 문제없음)
- **데이터 개인정보**: 시스템 성능 데이터만 수집, 개인정보 미포함
- **네트워크 보안**: 로컬 모니터링만 수행, 외부 통신 없음
- **권한 상승**: 특별한 관리자 권한 불필요

### 성능 최적화
- **캐싱 전략**: 5분간 시스템 상태 캐싱으로 중복 조회 방지
- **비동기 처리**: asyncio 기반 논블로킹 모니터링 구현
- **메모리 관리**: 순환 버퍼로 히스토리 크기 제한 (기본 100개)
- **CPU 효율성**: psutil 최적화된 시스템 호출 사용

### 확장성 및 호환성
- **다중 플랫폼**: Windows, Linux, macOS 완전 지원
- **Python 버전**: 3.9+ 호환 (f-string, dataclass, async 지원)
- **메모리 확장**: 프로파일별 메모리 제한 설정 가능
- **모니터링 확장**: 사용자 정의 메트릭 추가 가능

### 향후 개선 계획
- **GPU 모니터링**: NVIDIA, AMD GPU 사용률 추가
- **네트워크 상세 분석**: 대역폭, 지연시간, 패킷 손실률
- **디스크 I/O 최적화**: 읽기/쓰기 성능 분석 및 최적화
- **머신러닝 통합**: 사용 패턴 학습을 통한 예측적 프로파일 선택
- **클러스터 모니터링**: 다중 노드 성능 통합 관리

### 문제 해결
- **psutil 설치 문제**: `pip install psutil` 또는 시스템 패키지 매니저 사용
- **권한 오류**: Linux/Mac에서 특정 시스템 정보 접근시 sudo 필요할 수 있음
- **성능 저하**: 모니터링 간격 조정 (`monitoring_interval` 증가)
- **메모리 부족**: conservative 프로파일 사용 또는 히스토리 크기 감소

---

**버전**: v5.0.0
**작성일**: 2024-09-21
**문서 갱신**: Phase 3 성능 최적화 시스템 완료
**다음 단계**: Phase 4 UI/UX 시스템 개발 예정