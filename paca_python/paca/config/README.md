# 🎯 프로젝트 개요

PACA v5 설정 관리 모듈 - AI 어시스턴트의 설정 파일 관리, 환경 변수 처리, 동적 설정 업데이트를 담당하는 핵심 모듈입니다.

## 📁 폴더/파일 구조

```
config/
├── 📄 __init__.py           # 모듈 초기화 및 공개 API
├── 📄 base.py               # 기본 설정 관리 클래스 및 인터페이스
├── 📁 providers/            # 다양한 설정 제공자 (계획 중)
├── 📁 schemas/              # 설정 스키마 정의 (계획 중)
├── 📁 validators/           # 설정 검증 로직 (계획 중)
├── 📁 templates/            # 기본 설정 템플릿 (계획 중)
└── 📄 README.md             # 이 문서
```

**파일별 설명:**
- `base.py`: ConfigManager, BaseConfigProvider, 설정 관리 인터페이스
- `providers/`: FileConfigProvider, DatabaseConfigProvider 등 (향후)
- `schemas/`: Pydantic 스키마를 통한 설정 검증 (향후)
- `validators/`: 커스텀 검증 로직 및 규칙 (향후)
- `templates/`: 기본 설정 파일 템플릿 모음 (향후)

## ⚙️ 기능 요구사항

**입력:**
- JSON/YAML 설정 파일
- 환경 변수 및 명령행 인수
- 동적 설정 업데이트 요청

**출력:**
- 검증된 설정 값
- 설정 변경 알림 및 이벤트
- 설정 백업 및 복원 데이터

**핵심 로직 흐름:**
1. 다중 소스에서 설정 로드 및 병합
2. 스키마 기반 설정 검증
3. 설정 변경 감지 및 알림
4. 동적 설정 업데이트 및 적용

## 🛠️ 기술적 요구사항

**언어 및 프레임워크:**
- Python 3.8+
- json (JSON 파일 처리)
- os (환경 변수 접근)
- pathlib (파일 경로 관리)

**주요 의존성:**
- `core.types`: 기본 타입 및 결과 처리
- `core.events`: 설정 변경 이벤트 시스템
- `core.utils`: 로깅 및 유틸리티

**계획된 의존성:**
- `pydantic`: 설정 검증 및 타입 안전성
- `pyyaml`: YAML 파일 지원
- `watchdog`: 파일 변경 감지

**실행 환경:**
- 파일 시스템: 설정 파일 읽기/쓰기 권한
- 메모리: 최소 32MB (설정 캐시용)

## 🚀 라우팅 및 진입점

**주요 진입점:**
```python
from paca.config import (
    ConfigManager,
    JsonConfigProvider,
    YamlConfigProvider,
    EnvironmentConfigProvider
)

# 설정 매니저 초기화
config_manager = ConfigManager()

# 다양한 설정 제공자 등록
config_manager.add_provider(JsonConfigProvider("config.json"))
config_manager.add_provider(YamlConfigProvider("config.yaml"))
config_manager.add_provider(EnvironmentConfigProvider())

# 설정 로드
await config_manager.load()

# 설정 값 접근
api_key = config_manager.get("api.key")
debug_mode = config_manager.get("system.debug", default=False)
```

**API 경로:**
- `ConfigManager.get()`: 설정 값 조회
- `ConfigManager.set()`: 설정 값 변경
- `ConfigManager.reload()`: 설정 재로드
- `ConfigManager.validate()`: 설정 검증

## 📋 코드 품질 가이드

**주석 규칙:**
- 모든 설정 항목에 설명 및 기본값 명시
- 설정 제공자는 지원 형식 및 제한사항 기술
- 검증 규칙은 유효 범위 및 조건 설명

**네이밍 규칙:**
- 설정 키: 점표기법 (dot notation) - "database.host", "api.timeout"
- 제공자 클래스: [Source]ConfigProvider
- 검증 메서드: validate_* 접두사

**예외 처리:**
- ConfigError: 일반적인 설정 오류
- ValidationError: 설정 검증 실패
- FileNotFoundError: 설정 파일 미발견
- InvalidFormatError: 잘못된 설정 파일 형식

## 🏃‍♂️ 실행 방법

**설치:**
```bash
# 프로젝트 루트에서
pip install -e .

# YAML 지원을 위한 선택적 의존성
pip install pyyaml

# 고급 검증을 위한 선택적 의존성
pip install pydantic
```

**기본 설정 관리:**
```python
import asyncio
from paca.config import ConfigManager, JsonConfigProvider

async def main():
    # 설정 매니저 초기화
    config = ConfigManager()

    # JSON 설정 파일 추가
    config.add_provider(JsonConfigProvider("paca_config.json"))

    # 설정 로드
    await config.load()

    # 설정 값 조회
    app_name = config.get("app.name", default="PACA")
    port = config.get("server.port", default=8000)
    debug = config.get("system.debug", default=False)

    print(f"애플리케이션: {app_name}")
    print(f"포트: {port}")
    print(f"디버그 모드: {debug}")

    # 설정 동적 변경
    await config.set("system.debug", True)

# 실행
asyncio.run(main())
```

**환경 변수 우선순위 설정:**
```python
from paca.config import (
    ConfigManager,
    JsonConfigProvider,
    EnvironmentConfigProvider
)

# 설정 제공자 우선순위 순서로 추가
config = ConfigManager()

# 1. 기본 설정 파일 (가장 낮은 우선순위)
config.add_provider(JsonConfigProvider("default_config.json"))

# 2. 환경별 설정 파일
config.add_provider(JsonConfigProvider("production_config.json"))

# 3. 환경 변수 (가장 높은 우선순위)
config.add_provider(EnvironmentConfigProvider(prefix="PACA_"))

await config.load()

# 환경 변수 PACA_API_KEY가 있으면 파일의 api.key 값을 오버라이드
api_key = config.get("api.key")
```

**설정 파일 예시 (paca_config.json):**
```json
{
  "app": {
    "name": "PACA v5",
    "version": "5.0.0",
    "description": "한국어 특화 개인 AI 어시스턴트"
  },
  "server": {
    "host": "localhost",
    "port": 8000,
    "workers": 4
  },
  "database": {
    "url": "sqlite:///paca.db",
    "pool_size": 10,
    "echo": false
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/paca.log"
  },
  "features": {
    "cognitive_enabled": true,
    "learning_enabled": true,
    "analytics_enabled": false
  }
}
```

## 🧪 테스트 방법

**단위 테스트:**
- 각 설정 제공자의 로드/저장 기능 테스트
- 설정 검증 로직 정확성 검증
- 우선순위 기반 설정 병합 테스트

**통합 테스트:**
- 다중 설정 소스 통합 테스트
- 설정 변경 이벤트 전파 검증
- 파일 변경 감지 및 자동 리로드 테스트

**성능 테스트:**
- 설정 로드 시간 측정 (<10ms 목표)
- 메모리 사용량 최적화 검증
- 동시 설정 접근 성능 테스트

**설정 테스트 예시:**
```python
async def test_config_loading():
    """설정 로드 테스트"""
    config = ConfigManager()

    # 테스트 설정 파일 생성
    test_config = {
        "app": {"name": "Test App"},
        "debug": True
    }

    # JSON 제공자 테스트
    provider = JsonConfigProvider("test_config.json")
    config.add_provider(provider)

    await config.load()

    assert config.get("app.name") == "Test App"
    assert config.get("debug") is True

async def test_config_priority():
    """설정 우선순위 테스트"""
    config = ConfigManager()

    # 낮은 우선순위 설정
    config.add_provider(JsonConfigProvider({
        "value": "from_json",
        "only_json": "json_only"
    }))

    # 높은 우선순위 환경 변수
    os.environ["TEST_VALUE"] = "from_env"
    config.add_provider(EnvironmentConfigProvider(prefix="TEST_"))

    await config.load()

    # 환경 변수가 JSON 설정을 오버라이드
    assert config.get("value") == "from_env"
    assert config.get("only_json") == "json_only"

async def test_config_validation():
    """설정 검증 테스트"""
    config = ConfigManager()

    # 스키마 정의 (향후 기능)
    schema = ConfigSchema({
        "port": {"type": "integer", "min": 1, "max": 65535},
        "host": {"type": "string", "required": True}
    })

    config.set_schema(schema)

    # 유효한 설정
    config.add_provider(JsonConfigProvider({
        "port": 8000,
        "host": "localhost"
    }))

    validation_result = await config.validate()
    assert validation_result.is_valid

    # 무효한 설정
    config.add_provider(JsonConfigProvider({
        "port": 99999,  # 범위 초과
        "host": ""      # 빈 문자열
    }))

    validation_result = await config.validate()
    assert not validation_result.is_valid
    assert len(validation_result.errors) > 0
```

## 💡 추가 고려사항

**보안:**
- 민감한 설정 값 암호화 저장
- 설정 파일 접근 권한 제어
- API 키 및 비밀번호 환경 변수 분리

**성능:**
- 설정 캐싱을 통한 빠른 접근
- 변경된 설정만 선택적 리로드
- 메모리 효율적인 설정 저장

**향후 개선:**
- 설정 변경 기록 및 롤백 기능
- 분산 설정 관리 (Consul, etcd 등)
- 실시간 설정 동기화
- 웹 기반 설정 관리 UI

**설정 카테고리:**
- **애플리케이션**: 기본 앱 정보 및 메타데이터
- **서버**: 네트워크 및 서버 관련 설정
- **데이터베이스**: 연결 정보 및 풀 설정
- **로깅**: 로그 레벨, 형식, 출력 대상
- **기능**: 모듈별 활성화/비활성화
- **성능**: 캐시, 타임아웃, 제한값
- **보안**: 인증, 암호화, 접근 제어

**설정 제공자 우선순위:**
1. **명령행 인수** (최고 우선순위)
2. **환경 변수**
3. **사용자 설정 파일**
4. **시스템 설정 파일**
5. **기본 설정** (최저 우선순위)

**품질 지표:**
- **로드 성능**: <10ms (설정 파일 로드)
- **메모리 효율**: <10MB (전체 설정 저장)
- **검증 정확도**: 100% (잘못된 설정 탐지)
- **업데이트 지연**: <1ms (설정 값 변경 반영)

**실제 사용 시나리오:**
- **개발 환경**: debug=true, 상세 로깅, 로컬 DB
- **프로덕션 환경**: debug=false, 최적화된 설정, 원격 DB
- **테스트 환경**: 모킹 활성화, 임시 저장소, 빠른 실행
- **도커 배포**: 환경 변수 기반 설정, 컨테이너 최적화