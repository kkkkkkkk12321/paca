# LLM 통합 의존성 및 모킹 전략

PACA 시스템은 기본적으로 Google Gemini SDK(`google-genai`)를 사용하도록 설계되어 있습니다. 로컬 개발이나 테스트 환경에서 외부 네트워크 호출을 피하기 위해 아래 전략을 활용할 수 있습니다.

## 1. 의존성 및 환경 변수 정리

- 필수 패키지: `google-genai`
- 기본 환경 변수
  - `GEMINI_API_KEYS`: 콤마로 구분된 API 키 목록. 예) `export GEMINI_API_KEYS="key1,key2"`
  - 모듈은 `config.default.llm.api_keys` 값보다 환경 변수를 우선 병합합니다.
- 테스트나 시연 용도로는 `python -m pip install --user google-genai`로 로컬 사용자 영역에 설치 가능합니다.

## 2. API 키 비활성화/대체 방법

- ConfigManager를 통해 LLM 설정을 재정의하여 네트워크 호출 없이 시스템을 구동할 수 있습니다.
  ```python
  await system.config_manager.initialize()
  system.config_manager.set_value("default", "llm.api_keys", [])
  await system.initialize()
  ```
- 빈 리스트가 전달되면 LLM 초기화가 경고와 함께 생략되고, PACA는 기본 규칙 기반 응답으로 동작합니다.
- 테스트 케이스(`tests/test_system_phase1.py`)에서도 동일한 방법으로 LLM 호출을 비활성화합니다.

## 3. 모킹 전략

### 3.1 GeminiClientManager 모킹

- `GeminiClientManager` 클래스는 `update_api_keys`, `add_api_keys` 등의 인터페이스를 제공하므로, 테스트에서는 간단한 스텁 객체로 치환할 수 있습니다.
  ```python
  class DummyGeminiClient:
      async def initialize(self):
          return Result.success(True)

      async def generate_text(self, request):
          return Result.success(
              GeminiResponse(
                  id=request.id,
                  text="mocked response",
                  model=request.model,
              )
          )
  ```
- `PacaSystem` 초기화 전에 `system.llm_client = DummyGeminiClient()` 형태로 주입하거나, `patch("paca.system.GeminiClientManager", DummyClass)`와 같이 monkeypatch를 사용합니다.

### 3.2 ResponseProcessor 모킹

- 컨텍스트 생성이 복잡한 통합 테스트라면 `create_response_processor()` 반환값을 대체하여 최소한의 문자열 처리만 수행하도록 할 수 있습니다.

## 4. 키 로테이션 검증

- `tests/test_system_phase1.py::test_llm_api_key_management_cycle`와 같이 시스템 레벨에서 API 키 추가/제거/갱신 흐름을 검증하는 테스트를 제공하고 있습니다.
- 독립적으로 `tests/test_gemini_key_manager.py`에서 라운드로빈 로테이션 로직을 확인합니다.

## 5. 권장 워크플로우

1. **로컬 개발**: LLM 키 비활성화 → 규칙 기반 응답과 ReasoningEngine 로직 검증
2. **통합 테스트**: Dummy 클라이언트로 PACA 파이프라인 유지 + LLM 호출 모킹
3. **사전 배포**: 실제 API 키를 `.env` 또는 CI/CD 시크릿으로 주입 후, `_initialize_llm_system`이 성공하는지 확인

> 참고: `config.default.llm.*` 값은 `paca/config/base.py`에 정의되어 있으며, 사용자의 실제 키는 환경 변수 또는 사용자 전용 설정 파일에서 관리하는 것을 권장합니다.
