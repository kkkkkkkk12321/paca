# PACA 일화적 기억 시스템

## 🎯 프로젝트 개요
특정 시간과 장소에서 발생한 개인적 경험과 사건을 저장하고 관리하는 일화적 기억 모듈입니다.

## 📁 폴더/파일 구조
```
episodic/
├── conversations/   # 대화 기록
├── experiences/     # 사용자 경험 데이터
├── events/          # 시스템 이벤트 로그
├── contexts/        # 상황별 컨텍스트
└── timelines/       # 시간순 기억 구조
```

## ⚙️ 기능 요구사항
- **입력**: 대화 기록, 사용자 상호작용, 이벤트 데이터
- **출력**: 시간순 기억 검색, 컨텍스트 재구성
- **핵심 로직**: 시간 인덱싱, 컨텍스트 연결, 기억 재생

## 🛠️ 기술적 요구사항
- **언어**: Python
- **저장소**: 시계열 데이터베이스
- **인덱싱**: 시간, 참여자, 주제별
- **검색**: 시간 범위, 키워드 기반

## 🚀 라우팅 및 진입점
- 기억 저장: `episodic.store_episode(event, context, timestamp)`
- 기억 검색: `episodic.recall(timeframe, context)`
- 타임라인: `episodic.get_timeline(start, end)`

## 📋 코드 품질 가이드
- 타임스탬프 정확성 보장
- 개인정보 마스킹 처리
- 메모리 압축 정책
- 중요도 기반 우선순위

## 🏃‍♂️ 실행 방법
```bash
# 일화 검색
python -m paca.memory.episodic --recall "yesterday"

# 타임라인 생성
python -m paca.memory.episodic --timeline "2024-01-01:2024-01-31"

# 메모리 정리
python -m paca.memory.episodic --cleanup --older-than 30d
```

## 🧪 테스트 방법
- **단위 테스트**: 일화 저장/검색 테스트
- **통합 테스트**: 타임라인 연속성 검증
- **성능 테스트**: 대용량 일화 검색 성능

## 💡 추가 고려사항
- **보안**: 민감한 개인 정보 보호
- **성능**: 시계열 인덱싱 최적화
- **향후 개선**: 자동 중요도 평가, 감정 태깅