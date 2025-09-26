# PACA 배경 사운드 에셋

## 🎯 프로젝트 개요
PACA 사용자의 집중력과 생산성 향상을 위한 배경 음악 및 앰비언트 사운드 모음입니다.

## 📁 폴더/파일 구조
```
ambient/
├── calm.wav       # 차분한 배경 사운드 (휴식, 명상용)
├── energetic.wav  # 활기찬 배경 사운드 (활동적 작업용)
└── focus.wav      # 집중 배경 사운드 (딥워크용)
```

## ⚙️ 기능 요구사항
- **입력**: 사운드 타입, 재생 모드(루프/원샷), 볼륨 설정
- **출력**: 연속적인 배경 음향 효과
- **핵심 로직**: 무한 루프 재생, 페이드 인/아웃, 크로스페이딩

## 🛠️ 기술적 요구사항
- **포맷**: WAV (무손실 오디오)
- **샘플링**: 44.1kHz, 16-bit
- **길이**: 5분 ~ 15분 (루프 가능한 길이)
- **특성**: 매끄러운 루프, 갑작스러운 변화 없음

## 🚀 라우팅 및 진입점
- 배경음 재생: `AmbientPlayer.play_ambient(sound_type, loop=True)`
- 볼륨 조절: `AmbientPlayer.set_volume(level)`
- 페이드 효과: `AmbientPlayer.fade_transition(from_sound, to_sound)`

## 📋 코드 품질 가이드
- 매끄러운 루프 포인트 설정
- 적절한 배경음 레벨 유지
- 사용자 집중 방해 최소화
- 메모리 효율적인 스트리밍

## 🏃‍♂️ 실행 방법
```bash
# 배경 사운드 테스트
python -m desktop_app.assets.test_ambient_sounds

# 루프 포인트 검증
python -m desktop_app.assets.validate_loop_points

# 오디오 스트리밍 테스트
python -m desktop_app.assets.test_audio_streaming
```

## 🧪 테스트 방법
- **루프 테스트**: 매끄러운 반복 재생 확인
- **볼륨 테스트**: 배경음 레벨 적절성 검증
- **페이드 테스트**: 부드러운 전환 효과 확인

## 💡 추가 고려사항
- **사용자 경험**: 장시간 청취해도 피로하지 않은 사운드
- **성능**: 메모리 사용량 최적화
- **향후 개선**: AI 기반 개인화 배경음, 바이노럴 비트