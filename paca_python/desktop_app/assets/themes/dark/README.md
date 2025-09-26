# PACA 다크 테마

## 🎯 프로젝트 개요
눈의 피로를 줄이고 야간 작업 환경을 위한 PACA의 다크 모드 테마 구현입니다.

## 📁 폴더/파일 구조
```
dark/
├── dark.css       # 메인 다크 테마 스타일시트
└── dark-vars.css  # 다크 테마 CSS 변수 정의
```

## ⚙️ 기능 요구사항
- **입력**: 시스템 다크 모드 설정, 사용자 테마 선호도
- **출력**: 다크 모드 UI 스타일링, 색상 변수 적용
- **핵심 로직**: 동적 색상 전환, 시스템 테마 동기화, 대비 최적화

## 🛠️ 기술적 요구사항
- **언어**: CSS3, CSS 변수
- **색상**: 고대비 다크 팔레트
- **접근성**: WCAG 2.1 AA 준수
- **성능**: 최적화된 렌더링

## 🚀 라우팅 및 진입점
- 테마 활성화: `ThemeManager.activate_dark_theme()`
- 시스템 동기화: `ThemeSync.sync_with_system_preference()`
- 색상 변수 로드: `ThemeVariables.load_dark_variables()`

## 📋 코드 품질 가이드
- 최소 4.5:1 색상 대비 비율
- 일관된 색상 팔레트 사용
- 부드러운 테마 전환 애니메이션
- 성능 최적화된 CSS 셀렉터

## 🏃‍♂️ 실행 방법
```bash
# 다크 테마 적용 테스트
python -m desktop_app.assets.test_dark_theme

# 색상 대비 검증
python -m desktop_app.assets.verify_dark_contrast

# 다크 모드 토글 테스트
python -m desktop_app.assets.test_theme_toggle
```

## 🧪 테스트 방법
- **색상 대비 테스트**: 모든 텍스트-배경 조합 대비 검증
- **시각적 테스트**: 다양한 컴포넌트 다크 모드 렌더링
- **접근성 테스트**: 스크린 리더 호환성 확인

## 💡 추가 고려사항
- **사용자 경험**: 눈의 피로 최소화
- **접근성**: 저시력 사용자 고려 설계
- **향후 개선**: 자동 밝기 조절, 블루라이트 필터