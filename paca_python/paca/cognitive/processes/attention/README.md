# PACA 주의 메커니즘 시스템

## 🎯 프로젝트 개요
인간의 주의 집중 과정을 모델링한 AI 주의 메커니즘으로, 자원 할당과 집중도 관리를 담당합니다.

## 📁 폴더/파일 구조
```
attention/
├── __init__.py              # 모듈 진입점
├── attention_manager.py     # 중앙 주의 관리자
├── focus_controller.py      # 집중도 제어 시스템
├── resource_allocator.py    # 주의 자원 할당자
└── selective_attention.py   # 선택적 주의 시스템
```

## ⚙️ 기능 요구사항
- **입력**: 다중 작업 요청, 우선순위, 자원 제한
- **출력**: 주의 자원 할당, 집중도 스코어, 작업 스케줄링
- **핵심 로직**: 우선순위 기반 자원 분배, 동적 집중도 조절

## 🛠️ 기술적 요구사항
- **언어**: Python 3.9+
- **라이브러리**: asyncio, dataclasses, enum
- **의존성**: paca.cognitive.base
- **동시성**: 비동기 작업 처리

## 🚀 라우팅 및 진입점
- 주의 할당: `AttentionManager.allocate_attention(task, priority)`
- 집중도 조절: `FocusController.adjust_focus(target, intensity)`
- 자원 관리: `ResourceAllocator.manage_resources()`

## 📋 코드 품질 가이드
- 비동기 패턴 일관성 유지
- 자원 사용량 모니터링 필수
- 타임아웃 처리 구현
- 상태 변화 로깅

## 🏃‍♂️ 실행 방법
```bash
# 주의 시스템 테스트
python -m paca.cognitive.processes.attention.attention_manager

# 성능 벤치마크
python -m paca.cognitive.processes.attention --benchmark

# 리소스 모니터링
python -m paca.cognitive.processes.attention --monitor
```

## 🧪 테스트 방법
- **단위 테스트**: 각 컴포넌트별 기능 테스트
- **통합 테스트**: 다중 작업 동시 처리 테스트
- **성능 테스트**: 자원 할당 효율성 측정

## 💡 추가 고려사항
- **성능**: 자원 할당 알고리즘 최적화
- **확장성**: 대규모 동시 작업 지원
- **향후 개선**: 학습 기반 우선순위 조정