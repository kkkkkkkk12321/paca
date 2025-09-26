# PACA 사용자 정의 테마

## 🎯 프로젝트 개요
사용자가 개인 취향에 맞게 커스터마이징할 수 있는 테마 템플릿과 설정 파일을 제공합니다.

## 📁 폴더/파일 구조
```
custom/
├── custom-template.css  # 사용자 정의 테마 템플릿
└── theme-config.json   # 테마 설정 및 메타데이터
```

## ⚙️ 기능 요구사항
- **입력**: 사용자 색상 선호도, 레이아웃 설정, 접근성 요구사항
- **출력**: 개인화된 테마 CSS, 테마 설정 파일
- **핵심 로직**: 동적 CSS 생성, 실시간 미리보기, 설정 검증

## 🛠️ 기술적 요구사항
- **언어**: CSS3, JSON
- **프레임워크**: CSS 변수, PostCSS
- **호환성**: 모든 브라우저, 고대비 모드
- **검증**: 색상 대비 검증, 접근성 준수

## 🚀 라우팅 및 진입점
- 테마 생성: `CustomThemeBuilder.create_theme(user_preferences)`
- 테마 적용: `ThemeManager.apply_custom_theme(theme_config)`
- 실시간 미리보기: `ThemePreview.preview_changes(css_vars)`

## 📋 코드 품질 가이드
- WCAG 2.1 AA 색상 대비 준수
- CSS 변수 기반 구조화
- 브라우저 호환성 확보
- 성능 최적화된 CSS

## 🏃‍♂️ 실행 방법
```bash
# 테마 빌더 실행
python -m desktop_app.assets.run_theme_builder

# 테마 유효성 검사
python -m desktop_app.assets.validate_custom_theme

# 접근성 검증
python -m desktop_app.assets.check_theme_accessibility
```

## 🧪 테스트 방법
- **색상 대비 테스트**: WCAG 기준 색상 대비 검증
- **브라우저 테스트**: 다양한 브라우저에서 렌더링 확인
- **사용자 테스트**: 실제 사용자 피드백 수집

## 💡 추가 고려사항
- **접근성**: 다양한 시각 능력 사용자 고려
- **성능**: CSS 번들 크기 최적화
- **향후 개선**: AI 기반 색상 추천, 테마 마켓플레이스