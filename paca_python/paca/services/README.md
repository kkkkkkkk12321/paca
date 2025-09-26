# 🎯 프로젝트 개요

PACA v5 비즈니스 서비스 모듈 - AI 어시스턴트의 핵심 비즈니스 로직을 담당하는 서비스 계층입니다.

## 📁 폴더/파일 구조

```
services/
├── 📄 __init__.py           # 모듈 초기화 및 공개 API
├── 📄 base.py               # 기본 서비스 클래스 및 매니저
├── 📄 auth.py               # 인증 및 사용자 관리 서비스
├── 📄 knowledge.py          # 지식 관리 및 검색 서비스
├── 📄 analytics.py          # 분석 및 메트릭 수집 서비스
├── 📄 notification.py       # 알림 및 메시지 서비스
├── 📄 learning.py           # 학습 세션 및 진도 관리 서비스
├── 📄 memory.py             # 메모리 및 대화 관리 서비스
└── 📄 README.md             # 이 문서
```

**파일별 설명:**
- `base.py`: BaseService, ServiceManager, 공통 서비스 인터페이스
- `auth.py`: AuthenticationService, 사용자 인증 및 세션 관리
- `knowledge.py`: KnowledgeService, 지식베이스 관리 및 검색
- `analytics.py`: AnalyticsService, 사용자 행동 및 시스템 메트릭 분석
- `notification.py`: NotificationService, 다채널 알림 시스템
- `learning.py`: LearningService, 적응형 학습 세션 관리
- `memory.py`: MemoryService, 대화 기록 및 컨텍스트 관리

## ⚙️ 기능 요구사항

**입력:**
- 사용자 요청 및 비즈니스 로직 호출
- 서비스 간 통신 및 데이터 교환
- 외부 API 호출 및 데이터 동기화

**출력:**
- 비즈니스 로직 처리 결과
- 서비스 상태 및 헬스 메트릭
- 사용자 응답 및 알림 데이터

**핵심 로직 흐름:**
1. 서비스 요청 수신 및 검증
2. 비즈니스 로직 실행 및 데이터 처리
3. 다른 서비스와의 통신 및 조정
4. 결과 반환 및 상태 업데이트

## 🛠️ 기술적 요구사항

**언어 및 프레임워크:**
- Python 3.8+
- asyncio (비동기 서비스 처리)
- datetime (시간 관리)
- uuid (고유 ID 생성)

**주요 의존성:**
- `core.types`: 기본 타입 및 ID 시스템
- `core.events`: 이벤트 기반 서비스 통신
- `core.utils`: 로깅 및 설정 관리
- `core.errors`: 서비스 관련 예외 처리

**실행 환경:**
- 메모리: 최소 256MB (서비스 상태 관리용)
- 네트워크: 외부 API 통신 가능

## 🚀 라우팅 및 진입점

**주요 진입점:**
```python
from paca.services import (
    ServiceManager,
    AuthenticationService,
    KnowledgeService,
    LearningService,
    MemoryService
)

# 서비스 매니저 초기화
manager = ServiceManager()

# 개별 서비스 등록
auth_service = AuthenticationService()
knowledge_service = KnowledgeService()

manager.register_service("auth", auth_service)
manager.register_service("knowledge", knowledge_service)

# 서비스 시작
await manager.start_all_services()
```

**API 경로:**
- `ServiceManager.get_service()`: 서비스 인스턴스 검색
- `AuthenticationService.authenticate()`: 사용자 인증
- `KnowledgeService.search()`: 지식 검색
- `LearningService.create_session()`: 학습 세션 생성

## 📋 코드 품질 가이드

**주석 규칙:**
- 모든 서비스 메서드에 비즈니스 로직 설명 필수
- API 인터페이스는 파라미터와 반환값 명시
- 비동기 메서드는 동시성 고려사항 기술

**네이밍 규칙:**
- 서비스 클래스: [Name]Service (AuthenticationService)
- 서비스 메서드: 동사_명사 패턴 (create_session)
- DTO 클래스: Request/Response 접미사

**예외 처리:**
- ServiceError: 일반적인 서비스 오류
- AuthenticationError: 인증 관련 오류
- ValidationError: 입력 데이터 검증 실패

## 🏃‍♂️ 실행 방법

**설치:**
```bash
# 프로젝트 루트에서
pip install -e .
```

**서비스 매니저 사용법:**
```python
import asyncio
from paca.services import ServiceManager, AuthenticationService

async def main():
    # 서비스 매니저 초기화
    manager = ServiceManager()

    # 인증 서비스 등록
    auth_service = AuthenticationService()
    manager.register_service("auth", auth_service)

    # 모든 서비스 시작
    await manager.start_all_services()

    # 서비스 사용
    auth = manager.get_service("auth")
    user = await auth.authenticate("username", "password")

    print(f"인증된 사용자: {user.username}")

# 실행
asyncio.run(main())
```

**개별 서비스 사용법:**
```python
# 학습 서비스 사용
from paca.services import LearningService

learning = LearningService()

# 학습 세션 생성
session = await learning.create_session(
    user_id="user123",
    session_type="vocabulary",
    goal="영어 단어 50개 학습"
)

# 답변 제출
result = await learning.submit_answer(
    session_id=session.id,
    question_id="q1",
    answer="답변 내용"
)
```

## 🧪 테스트 방법

**단위 테스트:**
- 각 서비스의 개별 기능 테스트
- 비즈니스 로직 정확성 검증
- 예외 상황 처리 테스트

**통합 테스트:**
- 서비스 간 통신 및 협력 테스트
- 전체 워크플로우 검증
- 데이터 일관성 검증

**성능 테스트:**
- 서비스 응답 시간 (<200ms 목표)
- 동시 요청 처리 능력 측정
- 메모리 사용량 최적화 검증

**서비스별 테스트:**
```python
async def test_authentication_service():
    """인증 서비스 테스트"""
    auth = AuthenticationService()

    # 사용자 등록
    user = await auth.register("testuser", "password123")
    assert user.username == "testuser"

    # 로그인
    login_result = await auth.authenticate("testuser", "password123")
    assert login_result.is_success

async def test_learning_service():
    """학습 서비스 테스트"""
    learning = LearningService()

    # 세션 생성
    session = await learning.create_session(
        user_id="user1",
        session_type="quiz",
        goal="수학 문제 풀기"
    )

    assert session.status == SessionStatus.ACTIVE
    assert session.user_id == "user1"

async def test_knowledge_service():
    """지식 서비스 테스트"""
    knowledge = KnowledgeService()

    # 지식 저장
    item = await knowledge.store_knowledge(
        content="파이썬은 프로그래밍 언어입니다",
        category="programming"
    )

    # 지식 검색
    results = await knowledge.search("파이썬")
    assert len(results) > 0
    assert "파이썬" in results[0].content
```

## 💡 추가 고려사항

**보안:**
- API 키 및 인증 토큰 안전한 관리
- 사용자 데이터 암호화 및 접근 제어
- 서비스 간 통신 보안 (HTTPS, JWT)

**성능:**
- 서비스 결과 캐싱으로 응답 속도 향상
- 비동기 처리를 통한 동시성 최적화
- 데이터베이스 연결 풀링 관리

**향후 개선:**
- 마이크로서비스 아키텍처로 전환
- 서비스 디스커버리 및 로드 밸런싱
- 실시간 모니터링 및 알람 시스템
- GraphQL API 지원

**서비스 품질 지표:**
- **가용성**: 99.9% 업타임 목표
- **응답성**: <200ms 평균 응답 시간
- **신뢰성**: <0.1% 오류율 유지
- **확장성**: 동시 사용자 1000명 지원

**서비스별 주요 기능:**
- **AuthenticationService**: JWT 토큰, 다중 인증, 세션 관리
- **KnowledgeService**: 전문 검색, 관계형 지식베이스, 자동 분류
- **AnalyticsService**: 실시간 메트릭, 사용자 행동 분석, 대시보드
- **NotificationService**: 다채널 알림, 템플릿 시스템, 스케줄링
- **LearningService**: 적응형 학습, 진도 추적, 개인화 추천
- **MemoryService**: 대화 컨텍스트, 장기 기억, 우선순위 관리