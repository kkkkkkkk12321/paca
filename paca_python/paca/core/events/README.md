# Core Events Module - PACA Python v5

## 🎯 프로젝트 개요
PACA 시스템의 이벤트 기반 통신 시스템으로, 발행/구독 패턴, 이벤트 큐, 비동기 핸들러를 통해 모듈 간 느슨한 결합과 반응형 아키텍처를 구현합니다.

## 📁 폴더/파일 구조
```
events/
├── __init__.py              # 모듈 진입점 및 통합 API
├── base.py                  # 기본 이벤트 클래스 및 인터페이스
├── emitter.py               # 이벤트 발행자 (EventEmitter)
├── handlers.py              # 이벤트 핸들러 및 리스너
└── queue.py                 # 이벤트 큐 및 비동기 처리
```

## ⚙️ 기능 요구사항
**입력**: 이벤트 데이터, 우선순위, 카테고리, 필터 조건
**출력**: 이벤트 발행 결과, 핸들러 실행 결과, 큐 상태 정보
**핵심 로직**: 이벤트 생성 → 필터링 → 큐 처리 → 핸들러 실행 → 결과 반환

## 🛠️ 기술적 요구사항
- Python 3.9+ (asyncio, typing, dataclasses, enum)
- 비동기 이벤트 처리 (asyncio 기반)
- 스레드 안전 이벤트 큐

## 🚀 라우팅 및 진입점
```python
from paca.core.events import EventEmitter, PacaEvent, EventPriority

# 이벤트 발행
emitter = EventEmitter()
event = PacaEvent("user_action", {"action": "login"}, EventPriority.HIGH)
await emitter.emit(event)

# 이벤트 구독
@emitter.on("user_action")
async def handle_user_action(event):
    print(f"User action: {event.data}")
```

## 📋 코드 품질 가이드
- 클래스: PascalCase (EventEmitter, BaseEvent)
- 함수: snake_case (emit_event, handle_event)
- 이벤트 타입: snake_case (user_login, system_error)
- 비동기 핸들러 필수, 타입 힌트 필수

## 🏃‍♂️ 실행 방법
```bash
python -c "from paca.core.events import EventEmitter; print('Events module loaded')"
```

## 🧪 테스트 방법
```bash
pytest tests/test_core/test_events/ -v
```

## 💡 추가 고려사항
**성능**: 이벤트 큐 최적화, 핸들러 병렬 실행
**향후 개선**: 이벤트 영속성, 분산 이벤트 처리, 이벤트 재생