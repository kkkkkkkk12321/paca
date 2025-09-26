# API LLM Module - PACA Python v5

## 🎯 프로젝트 개요
PACA 시스템의 대형 언어 모델(LLM) 통합 API 모듈로, Google Gemini, OpenAI GPT, Anthropic Claude 등 다양한 LLM과의 통합, API 키 관리, 응답 처리를 담당합니다.

## 📁 폴더/파일 구조
```
llm/
├── __init__.py              # 모듈 진입점 및 통합 API
├── base.py                  # 기본 LLM 인터페이스
├── gemini.py                # Google Gemini API 연동
├── openai_client.py         # OpenAI GPT API 연동
├── claude.py                # Anthropic Claude API 연동
└── manager.py               # LLM 관리자 및 로드 밸런싱
```

## ⚙️ 기능 요구사항
**입력**: 텍스트 프롬프트, 모델 설정, API 키, 요청 매개변수
**출력**: LLM 응답 텍스트, 토큰 사용량, 응답 시간, 신뢰도
**핵심 로직**: API 키 검증 → 모델 선택 → 요청 전송 → 응답 처리 → 에러 핸들링

## 🛠️ 기술적 요구사항
- Python 3.9+ (aiohttp, asyncio, json)
- 비동기 HTTP 클라이언트 (aiohttp)
- API 키 보안 관리 (환경 변수)
- 토큰 사용량 추적 및 제한

## 🚀 라우팅 및 진입점
```python
from paca.api.llm import LLMManager, GeminiClient

# LLM 관리자 초기화
manager = LLMManager()
await manager.initialize()

# 단일 모델 사용
gemini = GeminiClient(api_key="your_api_key")
response = await gemini.generate_text(
    prompt="안녕하세요. PACA v5에 대해 설명해주세요.",
    max_tokens=1000,
    temperature=0.7
)

# 관리자를 통한 사용 (자동 모델 선택)
response = await manager.generate_text(
    prompt="복잡한 수학 문제를 해결해주세요.",
    preferred_model="gpt-4"
)
```

## 📋 코드 품질 가이드
- 클래스: PascalCase (LLMManager, GeminiClient)
- API 메서드: snake_case (generate_text, get_models)
- 모든 API 호출에 타임아웃 설정 필수
- API 키는 환경 변수로만 관리
- 토큰 사용량 로깅 필수

## 🏃‍♂️ 실행 방법
```bash
# 환경 변수 설정
export GEMINI_API_KEYS="your_gemini_api_key"
export OPENAI_API_KEY="your_openai_api_key"

# 모듈 테스트
python -c "
from paca.api.llm import LLMManager
import asyncio

async def test():
    manager = LLMManager()
    models = await manager.get_available_models()
    print(f'사용 가능한 모델: {models}')

asyncio.run(test())
"
```

## 🧪 테스트 방법
```bash
# 단위 테스트
pytest tests/test_api/test_llm/ -v

# 통합 테스트 (API 키 필요)
pytest tests/integration/test_llm_integration.py -v

# 성능 테스트
python tests/performance/test_llm_performance.py
```

## 💡 추가 고려사항
**보안**: API 키 암호화, 요청/응답 로깅 시 민감 정보 제거
**성능**: 요청 배치 처리, 응답 캐싱, 연결 풀링
**향후 개선**: 새로운 LLM 모델 지원, 스트리밍 응답, 비용 최적화