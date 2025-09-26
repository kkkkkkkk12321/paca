# PACA 설정 관리 시스템

## 🎯 프로젝트 개요
PACA 시스템의 모든 설정 파일을 관리하고 런타임 구성을 제공하는 중앙 설정 관리 모듈입니다.

## 📁 폴더/파일 구조
```
config/
├── base/            # 기본 설정 파일
├── environments/    # 환경별 설정 (dev, prod, test)
├── models/          # AI 모델 설정
├── api/             # API 관련 설정
└── user/            # 사용자 개인 설정
```

## ⚙️ 기능 요구사항
- **입력**: 설정 파일 (YAML, JSON), 환경 변수
- **출력**: 구조화된 설정 객체, 런타임 구성
- **핵심 로직**: 설정 병합, 검증, 환경별 오버라이드

## 🛠️ 기술적 요구사항
- **언어**: Python
- **포맷**: YAML (주), JSON (보조)
- **라이브러리**: PyYAML, pydantic
- **검증**: 스키마 기반 검증

## 🚀 라우팅 및 진입점
- 설정 로드: `config.load_config(env)`
- 설정 업데이트: `config.update_config(key, value)`
- 설정 검증: `config.validate()`

## 📋 코드 품질 가이드
- 설정 키는 점 표기법 사용: `api.llm.max_tokens`
- 기본값 항상 제공
- 민감한 정보는 환경 변수 사용
- 설정 변경 시 검증 수행

## 🏃‍♂️ 실행 방법
```bash
# 설정 확인
python -m paca.config --show

# 설정 검증
python -m paca.config --validate

# 환경 설정
python -m paca.config --env production
```

## 🧪 테스트 방법
- **단위 테스트**: 설정 로드/병합/검증 테스트
- **통합 테스트**: 환경별 설정 적용 테스트
- **성능 테스트**: 설정 로딩 시간 측정

## 💡 추가 고려사항
- **보안**: API 키, 패스워드 암호화
- **성능**: 설정 캐싱으로 로딩 시간 단축
- **향후 개선**: 동적 설정 업데이트, 중앙 집중식 설정 관리