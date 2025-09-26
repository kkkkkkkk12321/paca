# PACA 시스템 모니터링

## 🎯 프로젝트 개요
PACA 시스템의 실시간 성능 모니터링, 메트릭 수집, 알림 시스템을 관리하는 모듈입니다.

## 📁 폴더/파일 구조
```
monitoring/
└── prometheus.yml  # Prometheus 모니터링 설정
```

## ⚙️ 기능 요구사항
- **입력**: 시스템 메트릭, 성능 지표, 헬스 체크 결과
- **출력**: 모니터링 대시보드, 알림, 리포트
- **핵심 로직**: 실시간 메트릭 수집, 임계값 모니터링, 자동 알림

## 🛠️ 기술적 요구사항
- **언어**: Python 3.9+
- **모니터링**: Prometheus, Grafana
- **라이브러리**: prometheus_client, psutil
- **저장소**: Time-series 데이터베이스

## 🚀 라우팅 및 진입점
- Prometheus 시작: `prometheus --config.file=monitoring/prometheus.yml`
- 메트릭 엔드포인트: `http://localhost:8000/metrics`
- 대시보드 접근: `http://localhost:3000` (Grafana)

## 📋 코드 품질 가이드
- 실시간 데이터 처리 최적화
- 메트릭 수집 간격 관리
- 알림 중복 방지 로직
- 대시보드 성능 최적화

## 🏃‍♂️ 실행 방법
```bash
# Prometheus 서버 시작
prometheus --config.file=monitoring/prometheus.yml

# PACA 메트릭 익스포터 시작
python -m paca.monitoring.metrics_exporter

# 헬스 체크 실행
python -m paca.monitoring.health_checker --check-all
```

## 🧪 테스트 방법
- **성능 테스트**: 메트릭 수집 성능 측정
- **알림 테스트**: 임계값 초과 시 알림 발송
- **대시보드 테스트**: 실시간 데이터 표시

## 💡 추가 고려사항
- **확장성**: 대규모 메트릭 처리 능력
- **신뢰성**: 모니터링 시스템 자체 모니터링
- **향후 개선**: AI 기반 이상 탐지, 예측적 알림