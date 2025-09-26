# PACA UI 컴포넌트 라이브러리

## 🎯 프로젝트 개요
PACA 데스크탑 애플리케이션에서 재사용 가능한 UI 컴포넌트들을 제공하는 컴포넌트 라이브러리입니다.

## 📁 폴더/파일 구조
```
components/
├── buttons.py       # 버튼 컴포넌트 모음
│   ├── PrimaryButton    # 주요 액션 버튼
│   ├── SecondaryButton  # 보조 버튼
│   ├── IconButton       # 아이콘 버튼
│   └── ToggleButton     # 토글 버튼
├── dialogs.py       # 대화상자 컴포넌트
│   ├── MessageDialog    # 메시지 대화상자
│   ├── ConfirmDialog    # 확인 대화상자
│   ├── InputDialog      # 입력 대화상자
│   └── FileDialog       # 파일 선택 대화상자
├── panels.py        # 패널 컴포넌트
│   ├── CollapsiblePanel # 접을 수 있는 패널
│   ├── TabPanel         # 탭 패널
│   ├── SplitPanel       # 분할 패널
│   └── StatusPanel      # 상태 표시 패널
└── widgets.py       # 커스텀 위젯
    ├── ChatWidget       # 채팅 위젯
    ├── ProgressWidget   # 진행 표시 위젯
    ├── LogWidget        # 로그 표시 위젯
    └── SettingsWidget   # 설정 위젯
```

## ⚙️ 기능 요구사항
- **입력**: 컴포넌트 속성, 이벤트 핸들러
- **출력**: 렌더링된 UI 컴포넌트, 사용자 상호작용
- **핵심 로직**: 컴포넌트 생명주기, 상태 관리, 이벤트 전파

## 🛠️ 기술적 요구사항
- **언어**: Python
- **GUI 프레임워크**: tkinter, customtkinter
- **디자인 패턴**: 컴포짓, 옵저버
- **스타일**: CSS-like 스타일링 지원

## 🚀 라우팅 및 진입점
- 컴포넌트 생성: `Component(parent, **kwargs)`
- 이벤트 바인딩: `component.bind(event, callback)`
- 스타일 적용: `component.apply_style(style)`

## 📋 코드 품질 가이드
- 컴포넌트 인터페이스 일관성 유지
- 접근성 속성 필수 구현
- 테마 지원 내장
- 반응형 크기 조정 지원

## 🏃‍♂️ 실행 방법
```bash
# 컴포넌트 갤러리 실행
python -m desktop_app.ui.components --gallery

# 컴포넌트 테스트
python -m desktop_app.ui.components --test ButtonComponent

# 스타일 미리보기
python -m desktop_app.ui.components --style-preview
```

## 🧪 테스트 방법
- **단위 테스트**: 각 컴포넌트별 기능 테스트
- **통합 테스트**: 컴포넌트 간 상호작용 테스트
- **성능 테스트**: 렌더링 성능 및 메모리 사용량

## 💡 추가 고려사항
- **보안**: XSS 방지를 위한 입력 검증
- **성능**: 가상화를 통한 대량 데이터 처리
- **향후 개선**: 애니메이션 지원, 드래그 앤 드롭