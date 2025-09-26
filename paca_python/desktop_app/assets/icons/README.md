# PACA 아이콘 에셋 모음

## 🎯 프로젝트 개요
PACA 데스크탑 애플리케이션의 사용자 인터페이스에서 사용하는 모든 아이콘을 관리하는 모듈입니다.

## 📁 폴더/파일 구조
```
icons/
├── app/             # 애플리케이션 메인 아이콘
│   ├── paca-16.ico # 16x16 아이콘
│   ├── paca-32.ico # 32x32 아이콘
│   ├── paca-64.ico # 64x64 아이콘
│   └── paca.svg    # 벡터 아이콘
├── buttons/         # UI 버튼 아이콘
│   ├── play.svg    # 실행 버튼
│   ├── pause.svg   # 일시정지
│   ├── stop.svg    # 중지 버튼
│   └── settings.svg# 설정 버튼
└── status/          # 상태 표시 아이콘
    ├── online.svg  # 온라인 상태
    ├── offline.svg # 오프라인 상태
    ├── error.svg   # 에러 상태
    └── loading.gif # 로딩 애니메이션
```

## ⚙️ 기능 요구사항
- **입력**: 벡터 그래픽, 래스터 이미지
- **출력**: 다해상도 아이콘, 테마별 변형
- **핵심 로직**: 아이콘 스케일링, 색상 변경, 캐싱

## 🛠️ 기술적 요구사항
- **포맷**: SVG (주), PNG, ICO
- **크기**: 16x16, 24x24, 32x32, 48x48, 64x64
- **색상**: 단색, 다색, 테마별 변형
- **최적화**: SVG 최적화, PNG 압축

## 🚀 라우팅 및 진입점
- 아이콘 로드: `icons.get_icon(name, size, theme)`
- 색상 변경: `icons.recolor_icon(icon, color)`
- 캐시 관리: `icons.cache_manager`

## 📋 코드 품질 가이드
- 아이콘 이름: 영어 소문자, 하이픈 구분
- SVG 최적화로 파일 크기 최소화
- 접근성: 의미론적 아이콘 이름
- 일관된 스타일 가이드 준수

## 🏃‍♂️ 실행 방법
```bash
# 아이콘 최적화
python -m desktop_app.assets.icons --optimize

# 아이콘 미리보기
python -m desktop_app.assets.icons --preview

# 누락 아이콘 확인
python -m desktop_app.assets.icons --check-missing
```

## 🧪 테스트 방법
- **단위 테스트**: 아이콘 로딩 및 스케일링 테스트
- **통합 테스트**: 테마별 아이콘 변경 테스트
- **성능 테스트**: 아이콘 렌더링 성능

## 💡 추가 고려사항
- **보안**: 악성 SVG 파일 방지
- **성능**: 아이콘 프리로딩 및 캐싱
- **향후 개선**: 동적 아이콘 생성, 애니메이션 아이콘