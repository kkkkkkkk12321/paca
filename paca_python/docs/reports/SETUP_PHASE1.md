# PACA v5 Phase 1 설정 가이드

Phase 1 LLM 통합이 완료되었습니다! 🎉 **[2025-01-22 검증 완료]**

## 📋 필수 요구사항

### 1. Python 의존성 설치
```bash
# 핵심 의존성 설치
pip install google-genai>=0.2.0
pip install aiohttp>=3.8.0
pip install python-dotenv>=0.19.0

# 또는 전체 의존성 설치
pip install -r requirements.txt
```

### 2. Gemini API 키 설정

#### 방법 1: 환경변수 (권장)
```bash
# Windows
set GEMINI_API_KEYS=your_key_1,your_key_2,your_key_3

# Linux/Mac
export GEMINI_API_KEYS="your_key_1,your_key_2,your_key_3"
```

#### 방법 2: .env 파일
```bash
# .env 파일 생성
echo "GEMINI_API_KEYS=your_key_1,your_key_2,your_key_3" > .env
echo "PACA_LOG_LEVEL=INFO" >> .env
echo "PACA_SANDBOX_PATH=./sandbox" >> .env
```

### 3. API 키 발급
1. [Google AI Studio](https://aistudio.google.com/app/apikey) 방문
2. 새 API 키 생성
3. 키를 안전한 곳에 저장

## 🚀 실행 방법

### 대화형 모드
```bash
python -m paca --interactive
```

### 단일 메시지
```bash
python -m paca --message "안녕하세요!"
```

### GUI 모드 (선택사항)
```bash
python -m paca --gui
```

## 🧪 테스트 실행

### 통합 테스트
```bash
python test_llm_integration.py
```

### 기본 동작 확인
```bash
python -c "
import asyncio
from paca.system import PacaSystem
async def test():
    paca = PacaSystem()
    result = await paca.initialize()
    print(f'초기화: {result.is_success}')
    if result.is_success:
        response = await paca.process_message('안녕하세요')
        print(f'응답: {response.data.get(\"response\", \"오류\")[:50]}...')
        await paca.cleanup()
asyncio.run(test())
"
```

## 📊 현재 구현 상태

### ✅ 완료된 기능
- [x] Gemini API 클라이언트 (API 키 로테이션 포함)
- [x] LLM 인터페이스 및 추상화
- [x] 응답 품질 검증 및 안전성 필터링
- [x] 컨텍스트 관리 및 대화 히스토리
- [x] 토큰 사용량 모니터링
- [x] PACA 시스템 통합
- [x] 기본 대화 기능

### ⚠️ 제한사항
- LLM 없이도 기본 응답 생성 가능 (fallback)
- 현재는 텍스트 생성만 지원 (이미지 생성은 Phase 2에서)
- 고급 인지 기능들은 아직 LLM과 완전히 연동되지 않음

## 📈 사용 가능한 모델

### Gemini 모델
- `gemini-2.5-pro`: 복잡한 추론 작업 (기본값: 프로덕션용)
- `gemini-2.5-flash`: 빠른 응답 (기본값: 개발/테스트용)
- `gemini-2.5-flash-image-preview`: 이미지 생성 (Phase 2)

### 모델 변경
```python
from paca.system import PacaSystem, PacaConfig
from paca.api.llm.base import ModelType

config = PacaConfig(
    default_llm_model=ModelType.GEMINI_PRO,  # 또는 GEMINI_FLASH
    llm_temperature=0.7,
    llm_max_tokens=2048
)

paca = PacaSystem(config)
```

## 🔧 설정 옵션

### 핵심 설정
```python
PacaConfig(
    # LLM 설정
    gemini_api_keys=["key1", "key2"],
    default_llm_model=ModelType.GEMINI_FLASH,
    llm_temperature=0.7,
    llm_max_tokens=2048,
    enable_llm_caching=True,
    llm_timeout=30.0,

    # 시스템 설정
    max_response_time=5.0,
    enable_learning=True,
    log_level=LogLevel.INFO
)
```

## 🐛 문제 해결

### 1. API 키 오류
```
FAILED: No API keys configured
```
**해결**: `GEMINI_API_KEYS` 환경변수 설정 확인

### 2. 라이브러리 누락
```
Warning: google-genai not installed
```
**해결**: `pip install google-genai` 실행

### 3. 네트워크 오류
```
API request failed: Connection timeout
```
**해결**: 네트워크 연결 및 방화벽 설정 확인

### 4. 응답 품질 문제
```
Response validation failed
```
**해결**:
- 다른 모델 시도 (GEMINI_PRO → GEMINI_FLASH)
- temperature 값 조정 (0.3-0.9)
- max_tokens 증가

## ✅ 구현 완료 상태 (2025-01-22)

### 성공적으로 완료된 기능
- ✅ Gemini API 클라이언트 (API 키 로테이션, 오류 복구)
- ✅ LLM 인터페이스 및 추상화 계층
- ✅ 응답 품질 검증 및 안전성 필터링
- ✅ 컨텍스트 관리 및 대화 히스토리
- ✅ 토큰 사용량 모니터링 및 최적화
- ✅ PACA 시스템과의 완전한 통합
- ✅ 호환성 문제 해결 (Result 타입, Logger)

### 테스트 결과
```
Gemini Client: SUCCESS ✅
Response Processor: SUCCESS ✅
Quality Score: 90.0/100
Token Monitoring: Working
Cache System: Ready
```

## 🔮 다음 단계 (Phase 2) - 준비됨

- [ ] 자기 성찰 루프 (Self-Reflection Loop)
- [ ] 진실 탐구 프로토콜 (Truth Seeking Protocol)
- [ ] 지적 무결성 점수 (IIS) 시스템
- [ ] 이미지 생성 기능
- [ ] 고급 인지 프로세스와 LLM 연동

## 📚 참고 자료

- [Gemini API 문서](https://ai.google.dev/docs)
- [PACA v5 초기 아이디어](../paca_초기아이디어통합계획.md)
- [Python 구현 가이드](./README.md)

---

**축하합니다!** PACA v5의 핵심 LLM 통합이 완료되었습니다. 이제 AI 어시스턴트로서의 기본 기능을 사용할 수 있습니다.