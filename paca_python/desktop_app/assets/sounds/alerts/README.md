# PACA 알림 사운드 에셋

## 🎯 프로젝트 개요
PACA 시스템의 다양한 알림 상황에서 사용되는 오디오 피드백 사운드 모음입니다.

## 📁 폴더/파일 구조
```
alerts/
├── alert.wav         # 일반 알림 사운드
├── success.wav       # 성공 완료 사운드
├── warning.wav       # 경고 알림 사운드
├── error.wav         # 오류 발생 사운드
└── notification.wav  # 일반 알림 사운드
```

## ⚙️ 기능 요구사항
- **입력**: 알림 타입, 볼륨 설정, 사용자 설정
- **출력**: 해당 상황에 맞는 오디오 피드백
- **핵심 로직**: 상황별 사운드 재생, 볼륨 조절, 음소거 지원

## 🛠️ 기술적 요구사항
- **포맷**: WAV (무손실 오디오)
- **샘플링**: 44.1kHz, 16-bit
- **길이**: 0.5초 ~ 2초 (짧은 피드백)
- **크기**: 최적화된 파일 크기

## 🚀 라우팅 및 진입점
- 알림 사운드 재생: `AlertSoundPlayer.play_alert(alert_type, volume)`
- 사운드 설정: `SoundManager.configure_alerts(enabled, volume)`
- 사용자 설정 로드: `SoundPreferences.load_user_settings()`

## 📋 코드 품질 가이드
- 적절한 볼륨 레벨 유지
- 일관된 오디오 품질
- 사용자 접근성 고려
- 시스템 사운드와 조화

## 🏃‍♂️ 실행 방법
```bash
# 알림 사운드 테스트
python -m desktop_app.assets.test_alert_sounds

# 사운드 파일 검증
python -m desktop_app.assets.validate_sound_files

# 볼륨 레벨 분석
python -m desktop_app.assets.analyze_sound_levels
```

## 🧪 테스트 방법
- **오디오 테스트**: 모든 알림 사운드 재생 확인
- **볼륨 테스트**: 적절한 볼륨 레벨 검증
- **호환성 테스트**: 다양한 오디오 장치에서 재생 확인

## 💡 추가 고려사항
- **접근성**: 청각 장애인을 위한 시각적 알림 연동
- **사용자 경험**: 방해받지 않는 적절한 사운드 레벨
- **향후 개선**: 사용자 정의 알림 사운드, 3D 오디오 효과