# PACA 상태 아이콘 에셋

## 🎯 프로젝트 개요
PACA 시스템의 다양한 상태를 시각적으로 표현하는 상태 표시 아이콘 모음입니다.

## 📁 폴더/파일 구조
```
status/
├── active_12_dark.svg     # 활성 상태 (12px, 다크 테마)
├── active_12_light.svg    # 활성 상태 (12px, 라이트 테마)
├── active_16_dark.svg     # 활성 상태 (16px, 다크 테마)
├── active_16_light.svg    # 활성 상태 (16px, 라이트 테마)
├── active_20_dark.svg     # 활성 상태 (20px, 다크 테마)
├── active_20_light.svg    # 활성 상태 (20px, 라이트 테마)
├── active_24_dark.svg     # 활성 상태 (24px, 다크 테마)
├── active_24_light.svg    # 활성 상태 (24px, 라이트 테마)
├── inactive_12_dark.svg   # 비활성 상태 (12px, 다크 테마)
├── inactive_12_light.svg  # 비활성 상태 (12px, 라이트 테마)
├── inactive_16_dark.svg   # 비활성 상태 (16px, 다크 테마)
├── inactive_16_light.svg  # 비활성 상태 (16px, 라이트 테마)
├── inactive_20_dark.svg   # 비활성 상태 (20px, 다크 테마)
├── inactive_20_light.svg  # 비활성 상태 (20px, 라이트 테마)
├── inactive_24_dark.svg   # 비활성 상태 (24px, 다크 테마)
├── inactive_24_light.svg  # 비활성 상태 (24px, 라이트 테마)
├── loading_12_dark.svg    # 로딩 상태 (12px, 다크 테마)
├── loading_12_light.svg   # 로딩 상태 (12px, 라이트 테마)
├── loading_16_dark.svg    # 로딩 상태 (16px, 다크 테마)
├── loading_16_light.svg   # 로딩 상태 (16px, 라이트 테마)
├── loading_20_dark.svg    # 로딩 상태 (20px, 다크 테마)
├── loading_20_light.svg   # 로딩 상태 (20px, 라이트 테마)
├── loading_24_dark.svg    # 로딩 상태 (24px, 다크 테마)
├── loading_24_light.svg   # 로딩 상태 (24px, 라이트 테마)
├── warning_12_dark.svg    # 경고 상태 (12px, 다크 테마)
├── warning_12_light.svg   # 경고 상태 (12px, 라이트 테마)
├── warning_16_dark.svg    # 경고 상태 (16px, 다크 테마)
├── warning_16_light.svg   # 경고 상태 (16px, 라이트 테마)
├── warning_20_dark.svg    # 경고 상태 (20px, 다크 테마)
├── warning_20_light.svg   # 경고 상태 (20px, 라이트 테마)
├── warning_24_dark.svg    # 경고 상태 (24px, 다크 테마)
├── warning_24_light.svg   # 경고 상태 (24px, 라이트 테마)
├── error_12_dark.svg      # 오류 상태 (12px, 다크 테마)
├── error_12_light.svg     # 오류 상태 (12px, 라이트 테마)
├── error_16_dark.svg      # 오류 상태 (16px, 다크 테마)
├── error_16_light.svg     # 오류 상태 (16px, 라이트 테마)
├── error_20_dark.svg      # 오류 상태 (20px, 다크 테마)
├── error_20_light.svg     # 오류 상태 (20px, 라이트 테마)
├── error_24_dark.svg      # 오류 상태 (24px, 다크 테마)
└── error_24_light.svg     # 오류 상태 (24px, 라이트 테마)
```

## ⚙️ 기능 요구사항
- **입력**: 상태 타입, 크기, 테마 설정
- **출력**: 해당 상태의 시각적 표시 아이콘
- **핵심 로직**: 실시간 상태 업데이트, 상태 변화 애니메이션

## 🛠️ 기술적 요구사항
- **포맷**: SVG (벡터 그래픽)
- **크기**: 12x12, 16x16, 20x20, 24x24
- **테마**: Dark, Light 버전
- **상태 타입**: active, inactive, loading, warning, error

## 🚀 라우팅 및 진입점
- 상태 아이콘 로드: `StatusIconLoader.load_status_icon(status, size, theme)`
- 상태 업데이트: `StatusManager.update_status(component_id, new_status)`
- 실시간 모니터링: `StatusMonitor.watch_status_changes()`

## 📋 코드 품질 가이드
- 상태별 색상 코딩 일관성
- 직관적인 상태 표현
- 접근성을 위한 적절한 대비
- 상태 변화 시 부드러운 전환

## 🏃‍♂️ 실행 방법
```bash
# 상태 아이콘 검증
python -m desktop_app.assets.validate_status_icons

# 상태 아이콘 프리뷰
python -m desktop_app.assets.preview_status_icons

# 상태 변화 시뮬레이션
python -m desktop_app.assets.simulate_status_changes
```

## 🧪 테스트 방법
- **시각적 테스트**: 모든 상태별 아이콘 렌더링 확인
- **상태 전환 테스트**: 상태 변화 시 아이콘 업데이트 검증
- **성능 테스트**: 빠른 상태 변화 시 렌더링 성능

## 💡 추가 고려사항
- **사용성**: 색맹 사용자를 위한 형태 구분
- **성능**: 상태 아이콘 캐싱 메커니즘
- **향후 개선**: 맞춤형 상태 아이콘, 애니메이션 효과