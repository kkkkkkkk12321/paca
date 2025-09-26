# PACA 앱 아이콘 에셋

## 🎯 프로젝트 개요
PACA 데스크톱 애플리케이션의 메인 앱 아이콘 에셋 모음으로, 다양한 크기와 테마를 지원합니다.

## 📁 폴더/파일 구조
```
app/
├── paca_16_dark.svg     # 16x16 다크 테마 아이콘
├── paca_16_light.svg    # 16x16 라이트 테마 아이콘
├── paca_32_dark.svg     # 32x32 다크 테마 아이콘
├── paca_32_light.svg    # 32x32 라이트 테마 아이콘
├── paca_64_dark.svg     # 64x64 다크 테마 아이콘
├── paca_64_light.svg    # 64x64 라이트 테마 아이콘
├── paca_128_dark.svg    # 128x128 다크 테마 아이콘
├── paca_128_light.svg   # 128x128 라이트 테마 아이콘
├── paca_256_dark.svg    # 256x256 다크 테마 아이콘
└── paca_256_light.svg   # 256x256 라이트 테마 아이콘
```

## ⚙️ 기능 요구사항
- **입력**: 테마 설정, DPI 설정
- **출력**: 적절한 크기와 테마의 SVG 아이콘
- **핵심 로직**: 동적 아이콘 로딩, 테마별 아이콘 전환

## 🛠️ 기술적 요구사항
- **포맷**: SVG (벡터 그래픽)
- **크기**: 16x16, 32x32, 64x64, 128x128, 256x256
- **테마**: Dark, Light 버전
- **호환성**: Windows, macOS, Linux

## 🚀 라우팅 및 진입점
- 아이콘 로드: `IconLoader.load_app_icon(size, theme)`
- 테마 전환: `IconManager.switch_theme(theme_name)`
- DPI 조정: `IconManager.get_appropriate_size(dpi)`

## 📋 코드 품질 가이드
- SVG 파일 최적화 필수
- 일관된 네이밍 규칙 준수
- 접근성 고려한 색상 대비
- 파일 크기 최적화

## 🏃‍♂️ 실행 방법
```bash
# 아이콘 유효성 검증
python -m desktop_app.assets.validate_icons

# 아이콘 최적화
python -m desktop_app.assets.optimize_svgs

# 테마별 아이콘 테스트
python -m desktop_app.assets.test_theme_icons
```

## 🧪 테스트 방법
- **시각적 테스트**: 모든 크기별 렌더링 검증
- **테마 테스트**: 다크/라이트 테마 전환 테스트
- **성능 테스트**: 아이콘 로딩 속도 측정

## 💡 추가 고려사항
- **접근성**: 고대비 테마 지원 고려
- **성능**: 아이콘 캐싱 메커니즘
- **향후 개선**: 동적 색상 변경, 애니메이션 아이콘