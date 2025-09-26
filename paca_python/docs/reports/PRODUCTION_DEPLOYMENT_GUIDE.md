# PACA 프로덕션 배포 가이드

## 개요

PACA (Personal Autonomous Cognitive Agent) 시스템의 프로덕션 환경 배포를 위한 완전한 가이드입니다.

## 시스템 요구사항

### 최소 요구사항
- **OS**: Linux/Windows/macOS
- **Python**: 3.8 이상
- **메모리**: 최소 2GB RAM
- **디스크**: 최소 5GB 여유 공간
- **네트워크**: 인터넷 연결 (웹 검색 API용)

### 권장 요구사항
- **OS**: Ubuntu 20.04 LTS 또는 CentOS 8
- **Python**: 3.11
- **메모리**: 8GB RAM
- **디스크**: 20GB SSD
- **CPU**: 4코어 이상

## 설치 및 설정

### 1. 저장소 클론 및 환경 설정

```bash
# 프로젝트 디렉토리로 이동
cd /opt/paca

# Python 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 또는
venv\Scripts\activate.bat  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env.production` 파일을 복사하고 실제 값으로 수정:

```bash
cp .env.production .env
```

중요한 설정값들:
```env
PACA_ENV=production
PACA_SECRET_KEY=your-secret-key-here
PACA_ENCRYPTION_KEY=your-encryption-key-here

# API 키 설정 (선택사항)
GOOGLE_API_KEY=your-google-api-key
BING_API_KEY=your-bing-api-key

# 데이터베이스 경로
PACA_DB_URL=sqlite:///var/lib/paca/feedback.db

# 로그 설정
PACA_LOG_FILE=/var/log/paca/paca.log
```

### 3. 디렉토리 생성

```bash
sudo mkdir -p /var/lib/paca/sandbox
sudo mkdir -p /var/log/paca
sudo chown -R $USER:$USER /var/lib/paca /var/log/paca
```

## Docker 배포

### 1. Docker 이미지 빌드

```bash
docker build -t paca:latest .
```

### 2. Docker Compose로 실행

```bash
# 프로덕션 환경 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f paca-api

# 상태 확인
docker-compose ps
```

### 3. 서비스 확인

```bash
# API 서버 상태 확인
curl http://localhost:8000/health

# Prometheus 메트릭 확인
curl http://localhost:9090/metrics

# Grafana 대시보드
# http://localhost:3000 (admin/admin)
```

## 수동 배포

### 1. 프로덕션 서버 시작

```bash
python production_server.py
```

### 2. 시스템 서비스 등록 (Linux)

`/etc/systemd/system/paca.service` 파일 생성:

```ini
[Unit]
Description=PACA Production Server
After=network.target

[Service]
Type=simple
User=paca
WorkingDirectory=/opt/paca
Environment=PATH=/opt/paca/venv/bin
ExecStart=/opt/paca/venv/bin/python production_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

서비스 활성화:
```bash
sudo systemctl daemon-reload
sudo systemctl enable paca
sudo systemctl start paca
sudo systemctl status paca
```

## API 사용법

### 1. 기본 엔드포인트

- **헬스 체크**: `GET /health`
- **서비스 정보**: `GET /`
- **등록된 도구**: `GET /api/v1/tools`
- **메트릭**: `GET /metrics`

### 2. PACA 액션 실행

```bash
# 사고 과정 실행
curl -X POST http://localhost:8000/api/v1/execute \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user123",
    "action": "think",
    "content": "안녕하세요, 도움이 필요합니다",
    "confidence": 0.8
  }'

# 도구 실행
curl -X POST http://localhost:8000/api/v1/execute \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user123",
    "action": "act",
    "tool_name": "FileManagerTool",
    "tool_params": {
      "operation": "write",
      "file_path": "test.txt",
      "content": "Hello World"
    }
  }'
```

### 3. 세션 관리

```bash
# 세션 정보 조회
curl http://localhost:8000/api/v1/sessions/user123
```

## 모니터링 및 로깅

### 1. 로그 파일 위치
- **애플리케이션 로그**: `/var/log/paca/paca.log`
- **시스템 로그**: `journalctl -u paca`
- **Docker 로그**: `docker-compose logs paca-api`

### 2. 메트릭 확인
- **Prometheus**: http://localhost:9091
- **Grafana**: http://localhost:3000
- **직접 메트릭**: http://localhost:8000/metrics

### 3. 주요 메트릭
- `paca_requests_total`: 총 요청 수
- `paca_request_duration_seconds`: 요청 처리 시간
- `paca_tool_executions_total`: 도구 실행 횟수

## 백업 및 복구

### 1. 데이터 백업

```bash
# 피드백 데이터베이스 백업
cp /var/lib/paca/feedback.db /backup/feedback_$(date +%Y%m%d).db

# 로그 백업
tar -czf /backup/logs_$(date +%Y%m%d).tar.gz /var/log/paca/

# 설정 파일 백업
cp .env /backup/env_$(date +%Y%m%d).bak
```

### 2. 자동 백업 설정

crontab에 추가:
```bash
# 매일 오전 2시에 백업
0 2 * * * /opt/paca/scripts/backup.sh
```

## 보안 설정

### 1. 방화벽 설정

```bash
# 필요한 포트만 개방
sudo ufw allow 8000/tcp   # API 서버
sudo ufw allow 9090/tcp   # 메트릭 (내부망만)
sudo ufw allow 3000/tcp   # Grafana (내부망만)
```

### 2. SSL/TLS 설정

nginx 리버스 프록시 설정 예시:

```nginx
server {
    listen 443 ssl;
    server_name paca.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3. 접근 제어

```env
# .env 파일에서 허용된 출처 설정
PACA_ALLOW_ORIGINS=["https://yourdomain.com", "https://app.yourdomain.com"]
```

## 성능 튜닝

### 1. Python 설정

```env
# 가비지 컬렉션 최적화
PACA_GC_THRESHOLD_0=700
PACA_GC_THRESHOLD_1=10
PACA_GC_THRESHOLD_2=10

# 메모리 제한
PACA_MAX_MEMORY_MB=1024
```

### 2. 데이터베이스 최적화

```bash
# SQLite 성능 최적화
echo "PRAGMA journal_mode=WAL;" | sqlite3 /var/lib/paca/feedback.db
echo "PRAGMA synchronous=NORMAL;" | sqlite3 /var/lib/paca/feedback.db
```

### 3. 로그 레벨 조정

```env
# 프로덕션에서는 INFO 레벨 권장
PACA_LOG_LEVEL=INFO
```

## 문제 해결

### 1. 일반적인 문제들

**문제**: 서버가 시작되지 않음
```bash
# 포트 사용 확인
netstat -tulpn | grep 8000

# 로그 확인
tail -f /var/log/paca/paca.log
```

**문제**: 메모리 사용량이 높음
```bash
# 메모리 사용량 확인
ps aux | grep python

# 메모리 제한 설정
export PACA_MAX_MEMORY_MB=512
```

**문제**: 도구 실행 실패
```bash
# 샌드박스 디렉토리 권한 확인
ls -la /var/lib/paca/sandbox/

# 도구 등록 상태 확인
curl http://localhost:8000/api/v1/tools
```

### 2. 로그 분석

```bash
# 오류 로그 확인
grep ERROR /var/log/paca/paca.log

# 성능 관련 로그
grep "duration" /var/log/paca/paca.log

# 특정 세션 로그
grep "session_id:user123" /var/log/paca/paca.log
```

### 3. 디버깅 모드

```env
# 개발/디버깅 시에만 사용
PACA_DEBUG=true
PACA_LOG_LEVEL=DEBUG
```

## 업데이트 및 유지보수

### 1. 업데이트 절차

```bash
# 1. 백업 수행
./scripts/backup.sh

# 2. 서비스 중지
sudo systemctl stop paca

# 3. 코드 업데이트
git pull origin main

# 4. 의존성 업데이트
pip install -r requirements.txt

# 5. 데이터베이스 마이그레이션 (필요시)
python scripts/migrate.py

# 6. 서비스 재시작
sudo systemctl start paca

# 7. 상태 확인
curl http://localhost:8000/health
```

### 2. 정기 유지보수

```bash
# 매주 실행할 작업들

# 1. 로그 로테이션
logrotate /etc/logrotate.d/paca

# 2. 오래된 피드백 데이터 정리
python scripts/cleanup_old_data.py --days 90

# 3. 데이터베이스 진공청소
echo "VACUUM;" | sqlite3 /var/lib/paca/feedback.db

# 4. 디스크 사용량 확인
df -h /var/lib/paca
```

## 알려진 제한사항

### 1. 현재 제한사항
- 단일 인스턴스만 지원 (분산 처리 미지원)
- SQLite 사용으로 인한 동시성 제한
- 웹 검색 API 제한에 따른 성능 영향

### 2. 향후 개선 계획
- PostgreSQL/MySQL 지원 추가
- Redis 캐싱 시스템 도입
- 마이크로서비스 아키텍처 전환
- Kubernetes 배포 지원

## 지원 및 문의

### 1. 문서 및 리소스
- 개발 가이드: `docs/development.md`
- API 문서: http://localhost:8000/docs
- 모니터링 대시보드: http://localhost:3000

### 2. 문제 보고
- 이슈는 프로젝트 저장소에 보고
- 로그 파일과 환경 정보 포함 필요
- 재현 가능한 예제 제공 권장

---

**최종 업데이트**: 2025-09-22
**문서 버전**: 1.0.0
**PACA 버전**: Phase 10 완료