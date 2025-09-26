# PACA 장기 기억 시스템

## 🎯 프로젝트 개요
중요하고 지속적으로 필요한 정보를 영구적으로 저장하고 관리하는 장기 기억 모듈입니다.

## 📁 폴더/파일 구조
```
long_term/
├── core_knowledge/  # 핵심 지식 정보
├── learned_patterns/# 학습된 패턴
├── user_preferences/# 사용자 선호도
├── system_history/  # 시스템 학습 이력
└── permanent/       # 영구 보존 데이터
```

## ⚙️ 기능 요구사항
- **입력**: 검증된 중요 정보, 학습 결과, 패턴
- **출력**: 안정적 지식 검색, 학습 기반 예측
- **핵심 로직**: 지식 통합, 패턴 강화, 영구 저장

## 🛠️ 기술적 요구사항
- **언어**: Python
- **저장소**: 영구 데이터베이스, 백업 시스템
- **압축**: 장기 저장을 위한 데이터 압축
- **인덱싱**: 효율적 검색을 위한 다차원 인덱싱

## 🚀 라우팅 및 진입점
- 지식 통합: `long_term.consolidate(knowledge)`
- 영구 저장: `long_term.store_permanent(data)`
- 패턴 검색: `long_term.retrieve_patterns(query)`

## 📋 코드 품질 가이드
- 데이터 중요도 평가 필수
- 정기적 백업 수행
- 지식 품질 검증 과정
- 데이터 압축 및 최적화

## 🏃‍♂️ 실행 방법
```bash
# 장기 기억 통합
python -m paca.memory.long_term --consolidate

# 패턴 분석
python -m paca.memory.long_term --analyze-patterns

# 백업 실행
python -m paca.memory.long_term --backup
```

## 🧪 테스트 방법
- **단위 테스트**: 영구 저장/검색 테스트
- **통합 테스트**: 지식 통합 과정 검증
- **성능 테스트**: 대규모 데이터 검색 성능

## 💡 추가 고려사항
- **보안**: 중요 데이터 암호화
- **성능**: 계층화된 저장 구조
- **향후 개선**: 자동 중요도 평가, 지능형 압축