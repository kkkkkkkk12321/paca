# PACA 사운드 에셋 모음

## 🎯 프로젝트 개요
PACA 데스크탑 애플리케이션의 오디오 피드백과 알림을 위한 사운드 에셋을 관리하는 모듈입니다.

## 📁 폴더/파일 구조
```
sounds/
├── alerts/          # 알림 사운드
│   ├── notification.wav  # 일반 알림
│   ├── warning.wav       # 경고음
│   ├── error.wav         # 오류 알림
│   └── success.wav       # 성공 알림
├── feedback/        # 상호작용 피드백
│   ├── click.wav         # 클릭 사운드
│   ├── hover.wav         # 호버 사운드
│   ├── select.wav        # 선택 사운드
│   └── typing.wav        # 타이핑 피드백
└── ambient/         # 배경 사운드
    ├── thinking.wav      # AI 처리 중
    ├── startup.wav       # 시작 사운드
    └── shutdown.wav      # 종료 사운드
```

## ⚙️ 기능 요구사항
- **입력**: 오디오 파일 (WAV, MP3)
- **출력**: 적절한 볼륨의 사운드 재생
- **핵심 로직**: 사운드 재생, 볼륨 조절, 중복 방지

## 🛠️ 기술적 요구사항
- **포맷**: WAV (고품질), MP3 (압축)
- **샘플링**: 44.1kHz, 16bit
- **길이**: 0.1-3초 (UI 사운드), 최대 10초 (배경음)
- **라이브러리**: pygame, pydub

## 🚀 라우팅 및 진입점
- 사운드 재생: `sounds.play(sound_name, volume)`
- 볼륨 조절: `sounds.set_volume(level)`
- 음소거: `sounds.mute(enabled)`

## 📋 코드 품질 가이드
- 사운드 길이: UI 피드백은 0.5초 이내
- 볼륨 정규화: -12dB 기준
- 접근성: 시각 장애인용 오디오 큐
- 사용자 설정 존중: 시스템 볼륨 연동

## 🏃‍♂️ 실행 방법
```bash
# 사운드 테스트
python -m desktop_app.assets.sounds --test-all

# 볼륨 정규화
python -m desktop_app.assets.sounds --normalize

# 사운드 품질 검사
python -m desktop_app.assets.sounds --quality-check
```

## 🧪 테스트 방법
- **단위 테스트**: 사운드 재생/정지 테스트
- **통합 테스트**: UI 이벤트와 사운드 연동 테스트
- **성능 테스트**: 사운드 로딩 및 재생 지연시간

## 💡 추가 고려사항
- **보안**: 악성 오디오 파일 방지
- **성능**: 사운드 프리로딩 및 메모리 관리
- **향후 개선**: 사용자 커스텀 사운드, 3D 오디오