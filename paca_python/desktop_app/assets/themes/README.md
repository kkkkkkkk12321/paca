# PACA 테마 시스템

## 🎯 프로젝트 개요
PACA 데스크탑 애플리케이션의 시각적 테마와 스타일링을 관리하는 테마 시스템 모듈입니다.

## 📁 폴더/파일 구조
```
themes/
├── light/           # 라이트 테마
│   ├── colors.json      # 색상 팔레트
│   ├── styles.css       # 스타일 시트
│   └── components.json  # 컴포넌트 스타일
├── dark/            # 다크 테마
│   ├── colors.json      # 다크 색상 팔레트
│   ├── styles.css       # 다크 스타일
│   └── components.json  # 다크 컴포넌트
└── custom/          # 사용자 커스텀 테마
    └── user_theme.json  # 사용자 정의 테마
```

## ⚙️ 기능 요구사항
- **입력**: 테마 설정, 색상 값, 스타일 정의
- **출력**: 일관된 UI 테마 적용
- **핵심 로직**: 테마 전환, 색상 계산, 스타일 적용

## 🛠️ 기술적 요구사항
- **언어**: Python, CSS, JSON
- **색상**: HEX, RGB, HSL 지원
- **라이브러리**: tkinter.ttk, customtkinter
- **접근성**: WCAG 2.1 AA 준수

## 🚀 라우팅 및 진입점
- 테마 적용: `themes.apply_theme(theme_name)`
- 색상 조회: `themes.get_color(color_name)`
- 커스텀 테마: `themes.create_custom_theme(config)`

## 📋 코드 품질 가이드
- 색상 대비비 4.5:1 이상 유지
- 일관된 색상 네이밍 규칙
- 반응형 스타일 지원
- 테마 변경 시 부드러운 전환

## 🏃‍♂️ 실행 방법
```bash
# 테마 미리보기
python -m desktop_app.assets.themes --preview light

# 테마 검증
python -m desktop_app.assets.themes --validate

# 커스텀 테마 생성
python -m desktop_app.assets.themes --create-custom
```

## 🧪 테스트 방법
- **단위 테스트**: 테마 로딩 및 적용 테스트
- **통합 테스트**: UI 컴포넌트 테마 적용 테스트
- **성능 테스트**: 테마 전환 성능 측정

## 💡 추가 고려사항
- **보안**: 테마 파일 검증
- **성능**: 테마 캐싱 및 지연 로딩
- **향후 개선**: 자동 테마 전환, 시스템 테마 연동