# PACA 한국어 특화 테스트 모음

## 🎯 프로젝트 개요
PACA 시스템의 한국어 처리 능력과 한국 문화권 특화 기능을 검증하는 테스트 모듈입니다.

## 📁 폴더/파일 구조
```
korean/
├── language/            # 한국어 언어 처리 테스트
│   ├── test_tokenization.py    # 한국어 토크나이제이션
│   ├── test_morphology.py      # 형태소 분석 테스트
│   ├── test_syntax.py          # 문법 분석 테스트
│   └── test_semantics.py       # 의미 분석 테스트
├── cultural/            # 문화적 맥락 테스트
│   ├── test_honorifics.py      # 존댓말 처리 테스트
│   ├── test_cultural_refs.py   # 문화적 참조 테스트
│   ├── test_idioms.py          # 관용구 처리 테스트
│   └── test_context.py         # 문화적 맥락 이해
├── localization/        # 현지화 테스트
│   ├── test_ui_korean.py       # 한국어 UI 테스트
│   ├── test_date_format.py     # 날짜 형식 테스트
│   ├── test_number_format.py   # 숫자 형식 테스트
│   └── test_currency.py        # 통화 표시 테스트
└── integration/         # 한국어 통합 테스트
    ├── test_korean_chat.py     # 한국어 대화 테스트
    ├── test_korean_reasoning.py# 한국어 추론 테스트
    └── test_korean_learning.py # 한국어 학습 테스트
```

## ⚙️ 기능 요구사항
- **입력**: 한국어 텍스트, 문화적 맥락, 언어 패턴
- **출력**: 정확한 한국어 처리 결과, 문화적 적절성 검증
- **핵심 로직**: 한국어 NLP, 문화적 맥락 이해, 존댓말 처리

## 🛠️ 기술적 요구사항
- **언어**: Python, 한국어 코퍼스
- **라이브러리**: KoNLPy, transformers (한국어 모델)
- **데이터**: 한국어 테스트 데이터셋, 문화적 참조 데이터
- **인코딩**: UTF-8 필수

## 🚀 라우팅 및 진입점
- 한국어 테스트: `pytest tests/korean/`
- 언어 처리 테스트: `pytest tests/korean/language/`
- 문화 테스트: `pytest tests/korean/cultural/`

## 📋 코드 품질 가이드
- 한국어 인코딩 일관성 유지
- 다양한 방언과 표현 방식 고려
- 문화적 민감성 검증
- 존댓말 수준별 테스트 케이스

## 🏃‍♂️ 실행 방법
```bash
# 한국어 전체 테스트
pytest tests/korean/ -v

# 언어 처리만 테스트
pytest tests/korean/language/ -v

# 문화적 맥락 테스트
pytest tests/korean/cultural/ -v

# 현지화 테스트
pytest tests/korean/localization/ -v
```

## 🧪 테스트 방법
- **단위 테스트**: 개별 한국어 처리 기능 테스트
- **통합 테스트**: 한국어 전체 처리 파이프라인 테스트
- **성능 테스트**: 한국어 처리 속도 및 정확도 측정

## 💡 추가 고려사항
- **보안**: 한국어 특수문자 처리 시 보안 이슈
- **성능**: 한국어 형태소 분석 성능 최적화
- **향후 개선**: 북한말 지원, 고어/한자 처리 강화