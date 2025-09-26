# PACA 버튼 아이콘 에셋

## 🎯 프로젝트 개요
PACA 데스크톱 애플리케이션의 버튼 UI 아이콘 모음으로, 다양한 제어 버튼의 시각적 표현을 제공합니다.

## 📁 폴더/파일 구조
```
buttons/
├── start_16_dark.svg      # 시작 버튼 (16px, 다크 테마)
├── start_16_light.svg     # 시작 버튼 (16px, 라이트 테마)
├── start_24_dark.svg      # 시작 버튼 (24px, 다크 테마)
├── start_24_light.svg     # 시작 버튼 (24px, 라이트 테마)
├── start_32_dark.svg      # 시작 버튼 (32px, 다크 테마)
├── start_32_light.svg     # 시작 버튼 (32px, 라이트 테마)
├── stop_16_dark.svg       # 정지 버튼 (16px, 다크 테마)
├── stop_16_light.svg      # 정지 버튼 (16px, 라이트 테마)
├── stop_24_dark.svg       # 정지 버튼 (24px, 다크 테마)
├── stop_24_light.svg      # 정지 버튼 (24px, 라이트 테마)
├── stop_32_dark.svg       # 정지 버튼 (32px, 다크 테마)
├── stop_32_light.svg      # 정지 버튼 (32px, 라이트 테마)
├── pause_16_dark.svg      # 일시정지 버튼 (16px, 다크 테마)
├── pause_16_light.svg     # 일시정지 버튼 (16px, 라이트 테마)
├── pause_24_dark.svg      # 일시정지 버튼 (24px, 다크 테마)
├── pause_24_light.svg     # 일시정지 버튼 (24px, 라이트 테마)
├── pause_32_dark.svg      # 일시정지 버튼 (32px, 다크 테마)
├── pause_32_light.svg     # 일시정지 버튼 (32px, 라이트 테마)
├── refresh_16_dark.svg    # 새로고침 버튼 (16px, 다크 테마)
├── refresh_16_light.svg   # 새로고침 버튼 (16px, 라이트 테마)
├── refresh_24_dark.svg    # 새로고침 버튼 (24px, 다크 테마)
├── refresh_24_light.svg   # 새로고침 버튼 (24px, 라이트 테마)
├── refresh_32_dark.svg    # 새로고침 버튼 (32px, 다크 테마)
├── refresh_32_light.svg   # 새로고침 버튼 (32px, 라이트 테마)
├── settings_16_dark.svg   # 설정 버튼 (16px, 다크 테마)
├── settings_16_light.svg  # 설정 버튼 (16px, 라이트 테마)
├── settings_24_dark.svg   # 설정 버튼 (24px, 다크 테마)
├── settings_24_light.svg  # 설정 버튼 (24px, 라이트 테마)
├── settings_32_dark.svg   # 설정 버튼 (32px, 다크 테마)
└── settings_32_light.svg  # 설정 버튼 (32px, 라이트 테마)
```

## ⚙️ 기능 요구사항
- **입력**: 버튼 타입, 크기, 테마 설정
- **출력**: 적절한 버튼 아이콘 SVG
- **핵심 로직**: 버튼 상태별 아이콘 로딩, 테마 동기화

## 🛠️ 기술적 요구사항
- **포맷**: SVG (벡터 그래픽)
- **크기**: 16x16, 24x24, 32x32
- **테마**: Dark, Light 버전
- **버튼 타입**: start, stop, pause, refresh, settings

## 🚀 라우팅 및 진입점
- 버튼 아이콘 로드: `ButtonIconLoader.load_icon(type, size, theme)`
- 아이콘 상태 변경: `ButtonManager.update_icon_state(button_id, state)`
- 테마 일괄 변경: `ButtonManager.apply_theme_to_all(theme)`

## 📋 코드 품질 가이드
- 버튼별 일관된 시각적 스타일
- 접근성을 위한 충분한 대비
- 직관적인 버튼 의미 전달
- 반응형 크기 조정

## 🏃‍♂️ 실행 방법
```bash
# 버튼 아이콘 검증
python -m desktop_app.assets.validate_button_icons

# 버튼 아이콘 프리뷰
python -m desktop_app.assets.preview_buttons

# 테마별 버튼 테스트
python -m desktop_app.assets.test_button_themes
```

## 🧪 테스트 방법
- **시각적 테스트**: 모든 버튼 타입별 렌더링 확인
- **상호작용 테스트**: 버튼 상태 변화 테스트
- **테마 테스트**: 다크/라이트 테마 전환 검증

## 💡 추가 고려사항
- **사용성**: 직관적인 버튼 인식성
- **접근성**: 색맹 사용자 고려 디자인
- **향후 개선**: 호버 효과, 애니메이션 버튼 상태