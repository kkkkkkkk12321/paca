# PACA v5 Deployment Guide

## 🎯 프로젝트 개요

PACA v5 Python 버전의 완전한 배포 가이드. 개발 환경부터 프로덕션 배포까지, Windows 실행파일 생성, Docker 컨테이너, 클라우드 배포, CI/CD 파이프라인 구축을 포함한 전체 배포 전략을 다룹니다.

## 📁 배포 환경 구조

```
📁 deployment/
├── 📁 local/                  # 로컬 개발 환경
│   ├── 📄 docker-compose.yml  # 로컬 컨테이너 구성
│   └── 📄 .env.local          # 로컬 환경 변수
├── 📁 staging/                # 스테이징 환경
│   ├── 📄 docker-compose.yml  # 스테이징 구성
│   ├── 📄 .env.staging        # 스테이징 환경 변수
│   └── 📄 nginx.conf          # 리버스 프록시 설정
├── 📁 production/             # 프로덕션 환경
│   ├── 📄 docker-compose.yml  # 프로덕션 구성
│   ├── 📄 .env.production     # 프로덕션 환경 변수
│   ├── 📄 nginx.conf          # 로드 밸런서 설정
│   └── 📄 monitoring.yml      # 모니터링 구성
├── 📁 scripts/                # 배포 스크립트
│   ├── 📄 deploy.sh           # 배포 자동화 스크립트
│   ├── 📄 backup.sh           # 백업 스크립트
│   └── 📄 rollback.sh         # 롤백 스크립트
└── 📁 k8s/                    # Kubernetes 매니페스트
    ├── 📄 deployment.yaml     # 배포 매니페스트
    ├── 📄 service.yaml        # 서비스 매니페스트
    └── 📄 ingress.yaml        # 인그레스 매니페스트
```

## ⚙️ 로컬 개발 환경

### 기본 설정

#### 1. 개발 환경 구축
```bash
# 저장소 클론
git clone https://github.com/your-org/paca-v5-python.git
cd paca-v5-python

# Python 가상환경 생성
python -m venv venv

# 가상환경 활성화
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 의존성 설치
pip install -e .[dev]

# 환경 변수 설정
cp .env.example .env.local
```

#### 2. 환경 변수 설정 (.env.local)
```bash
# PACA v5 로컬 개발 환경 설정
PACA_ENV=development
PACA_DEBUG=true
PACA_LOG_LEVEL=DEBUG

# 데이터베이스 설정
DATABASE_URL=sqlite:///./data/paca_dev.db
REDIS_URL=redis://localhost:6379/0

# API 설정
API_HOST=127.0.0.1
API_PORT=8000
API_WORKERS=1

# GUI 설정
GUI_THEME=dark
GUI_LANGUAGE=ko

# 성능 설정
MAX_PROCESSING_TIME=30.0
CONFIDENCE_THRESHOLD=0.7
ENABLE_CACHING=true
CACHE_TTL=3600

# 보안 설정
SECRET_KEY=dev-secret-key-change-in-production
JWT_EXPIRATION=86400

# 외부 서비스
OPENAI_API_KEY=your-openai-api-key
GOOGLE_API_KEY=your-google-api-key
```

#### 3. 로컬 실행
```bash
# CLI 모드
python -m paca "안녕하세요"

# GUI 모드
python -m paca --gui

# API 서버 모드
python -m paca --api --host 0.0.0.0 --port 8000

# 개발 모드 (hot reload)
python -m paca --dev --watch
```

### Docker를 활용한 로컬 개발

#### docker-compose.yml (로컬)
```yaml
version: '3.8'

services:
  paca-app:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - .:/app
      - /app/venv
    ports:
      - "8000:8000"
      - "5000:5000"  # GUI 포트
    environment:
      - PACA_ENV=development
      - DATABASE_URL=postgresql://paca:password@db:5432/paca_dev
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    command: python -m paca --dev --api

  db:
    image: postgres:14
    environment:
      POSTGRES_DB: paca_dev
      POSTGRES_USER: paca
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    volumes:
      - ./deployment/local/nginx.conf:/etc/nginx/nginx.conf
    ports:
      - "80:80"
    depends_on:
      - paca-app

volumes:
  postgres_data:
  redis_data:
```

#### Dockerfile.dev
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

# 애플리케이션 코드 복사
COPY . .

# 개발 모드로 패키지 설치
RUN pip install -e .

# 포트 노출
EXPOSE 8000 5000

# 개발 서버 실행
CMD ["python", "-m", "paca", "--dev", "--api", "--host", "0.0.0.0"]
```

## 🛠️ Windows 실행파일 배포

### PyInstaller를 활용한 실행파일 생성

#### 1. 빌드 스크립트 (build_windows.py)
```python
#!/usr/bin/env python3
"""
PACA v5 Windows 실행파일 빌드 스크립트
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def build_executable():
    """Windows 실행파일 빌드"""
    print("🚀 PACA v5 Windows 실행파일 빌드 시작...")

    # 빌드 환경 검사
    if sys.platform != "win32":
        print("❌ Windows 환경에서만 실행 가능합니다.")
        return False

    # 이전 빌드 정리
    for dir_name in ["build", "dist"]:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"🧹 {dir_name} 폴더 정리 완료")

    # PyInstaller 설정
    pyinstaller_args = [
        "pyinstaller",
        "--onefile",                    # 단일 파일
        "--windowed",                   # GUI 애플리케이션
        "--name=PACA-v5",              # 실행파일 이름
        "--icon=assets/paca_icon.ico", # 아이콘
        "--add-data=paca;paca",        # 패키지 데이터
        "--add-data=assets;assets",     # 에셋 파일
        "--hidden-import=tkinter",      # 숨겨진 임포트
        "--hidden-import=customtkinter",
        "--hidden-import=numpy",
        "--hidden-import=sympy",
        "--exclude-module=pytest",     # 불필요한 모듈 제외
        "--exclude-module=sphinx",
        "desktop_app/main.py"          # 진입점
    ]

    # 빌드 실행
    try:
        result = subprocess.run(pyinstaller_args, check=True)
        print("✅ PyInstaller 빌드 성공")
    except subprocess.CalledProcessError as e:
        print(f"❌ PyInstaller 빌드 실패: {e}")
        return False

    # 추가 파일 복사
    dist_dir = Path("dist")

    # 설정 파일
    shutil.copy("config.yaml", dist_dir / "config.yaml")

    # 라이센스 파일
    shutil.copy("LICENSE", dist_dir / "LICENSE")

    # README 파일
    shutil.copy("README.md", dist_dir / "README.md")

    print("✅ PACA v5 Windows 실행파일 빌드 완료!")
    print(f"📁 실행파일 위치: {dist_dir / 'PACA-v5.exe'}")

    return True

def create_installer():
    """NSIS 설치 프로그램 생성"""
    nsis_script = """
; PACA v5 설치 스크립트
!define APPNAME "PACA v5"
!define APPVERSION "5.0.0"
!define APPEXE "PACA-v5.exe"

Name "${APPNAME}"
OutFile "PACA-v5-Setup.exe"
InstallDir "$PROGRAMFILES\\${APPNAME}"

Section "MainSection" SEC01
    SetOutPath "$INSTDIR"
    File "dist\\PACA-v5.exe"
    File "dist\\config.yaml"
    File "dist\\LICENSE"
    File "dist\\README.md"

    ; 바탕화면 바로가기 생성
    CreateShortCut "$DESKTOP\\${APPNAME}.lnk" "$INSTDIR\\${APPEXE}"

    ; 시작 메뉴 바로가기 생성
    CreateDirectory "$SMPROGRAMS\\${APPNAME}"
    CreateShortCut "$SMPROGRAMS\\${APPNAME}\\${APPNAME}.lnk" "$INSTDIR\\${APPEXE}"
SectionEnd

Section "Uninstall"
    Delete "$INSTDIR\\*.*"
    RMDir "$INSTDIR"
    Delete "$DESKTOP\\${APPNAME}.lnk"
    RMDir /r "$SMPROGRAMS\\${APPNAME}"
SectionEnd
"""

    with open("installer.nsi", "w", encoding="utf-8") as f:
        f.write(nsis_script)

    print("✅ NSIS 설치 스크립트 생성 완료")

if __name__ == "__main__":
    if build_executable():
        create_installer()
```

#### 2. 빌드 실행
```bash
# 빌드 의존성 설치
pip install pyinstaller

# Windows 실행파일 빌드
python build_windows.py

# 결과 확인
ls dist/
# PACA-v5.exe
# config.yaml
# LICENSE
# README.md
```

#### 3. 배포 패키지 생성
```bash
# 압축 파일 생성
tar -czf PACA-v5-Windows-Portable.tar.gz -C dist .

# 또는 PowerShell에서
Compress-Archive -Path dist\* -DestinationPath PACA-v5-Windows-Portable.zip
```

## 🚀 Docker 컨테이너 배포

### 프로덕션 Dockerfile

```dockerfile
# Multi-stage 빌드
FROM python:3.11-slim as builder

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 빌드
COPY . .
RUN pip install --no-cache-dir .

# 프로덕션 이미지
FROM python:3.11-slim

WORKDIR /app

# 필수 런타임 의존성만 설치
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 빌드된 패키지 복사
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 애플리케이션 파일 복사
COPY paca ./paca
COPY config.yaml .
COPY assets ./assets

# 사용자 생성 (보안)
RUN useradd --create-home --shell /bin/bash paca
USER paca

# 포트 노출
EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 엔트리포인트
CMD ["python", "-m", "paca", "--api", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose (프로덕션)

```yaml
version: '3.8'

services:
  paca-app:
    build:
      context: .
      dockerfile: Dockerfile
    image: paca-v5:latest
    container_name: paca-app
    restart: unless-stopped
    environment:
      - PACA_ENV=production
      - DATABASE_URL=postgresql://paca:${DB_PASSWORD}@db:5432/paca_prod
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=${SECRET_KEY}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    networks:
      - paca-network
    depends_on:
      - db
      - redis
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "1.0"
        reservations:
          memory: 1G
          cpus: "0.5"

  db:
    image: postgres:14-alpine
    container_name: paca-db
    restart: unless-stopped
    environment:
      POSTGRES_DB: paca_prod
      POSTGRES_USER: paca
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    networks:
      - paca-network

  redis:
    image: redis:7-alpine
    container_name: paca-redis
    restart: unless-stopped
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    networks:
      - paca-network

  nginx:
    image: nginx:alpine
    container_name: paca-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./deployment/production/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
      - ./logs/nginx:/var/log/nginx
    networks:
      - paca-network
    depends_on:
      - paca-app

  monitoring:
    image: prom/prometheus
    container_name: paca-monitoring
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./deployment/production/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - paca-network

volumes:
  postgres_data:
  redis_data:
  prometheus_data:

networks:
  paca-network:
    driver: bridge
```

### Nginx 설정 (nginx.conf)

```nginx
events {
    worker_connections 1024;
}

http {
    upstream paca_backend {
        server paca-app:8000;
        # 로드 밸런싱을 위한 추가 서버
        # server paca-app-2:8000;
        # server paca-app-3:8000;
    }

    # 로그 포맷
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    # Gzip 압축
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript;

    server {
        listen 80;
        server_name localhost;

        # HTTP에서 HTTPS로 리다이렉트
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name localhost;

        # SSL 설정
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        # 보안 헤더
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

        # 로그
        access_log /var/log/nginx/access.log main;
        error_log /var/log/nginx/error.log;

        # API 프록시
        location /api/ {
            proxy_pass http://paca_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # 타임아웃 설정
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }

        # 정적 파일
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }

        # 헬스체크
        location /health {
            proxy_pass http://paca_backend/health;
        }

        # 웹소켓 지원
        location /ws/ {
            proxy_pass http://paca_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

## 📋 클라우드 배포

### AWS 배포

#### 1. ECS (Elastic Container Service) 배포
```yaml
# task-definition.json
{
  "family": "paca-v5-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "paca-app",
      "image": "your-account.dkr.ecr.region.amazonaws.com/paca-v5:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "PACA_ENV",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:paca/database-url"
        },
        {
          "name": "SECRET_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:paca/secret-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/paca-v5",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### 2. Terraform 인프라 구성
```hcl
# main.tf
provider "aws" {
  region = var.aws_region
}

# VPC
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"

  name = "paca-v5-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway = true
  enable_vpn_gateway = false
}

# ECS Cluster
resource "aws_ecs_cluster" "paca_cluster" {
  name = "paca-v5-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Application Load Balancer
resource "aws_lb" "paca_alb" {
  name               = "paca-v5-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets           = module.vpc.public_subnets

  enable_deletion_protection = false
}

# RDS Database
resource "aws_db_instance" "paca_db" {
  identifier = "paca-v5-db"

  engine         = "postgres"
  engine_version = "14.9"
  instance_class = "db.t3.micro"

  allocated_storage     = 20
  max_allocated_storage = 100
  storage_encrypted     = true

  db_name  = "paca_prod"
  username = "paca"
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.db_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.paca_db.name

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = true
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "paca_redis" {
  name       = "paca-v5-redis-subnet-group"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_cluster" "paca_redis" {
  cluster_id           = "paca-v5-redis"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.paca_redis.name
  security_group_ids   = [aws_security_group.redis_sg.id]
}
```

### Google Cloud Platform (GCP) 배포

#### Cloud Run 배포
```yaml
# cloudbuild.yaml
steps:
  # Docker 이미지 빌드
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/paca-v5:$COMMIT_SHA', '.']

  # 이미지 푸시
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/paca-v5:$COMMIT_SHA']

  # Cloud Run 배포
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'paca-v5'
      - '--image'
      - 'gcr.io/$PROJECT_ID/paca-v5:$COMMIT_SHA'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'
      - '--memory'
      - '2Gi'
      - '--cpu'
      - '1'
      - '--max-instances'
      - '10'
      - '--set-env-vars'
      - 'PACA_ENV=production'

images:
  - 'gcr.io/$PROJECT_ID/paca-v5:$COMMIT_SHA'
```

### Kubernetes 배포

#### Deployment 매니페스트
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: paca-v5-deployment
  labels:
    app: paca-v5
spec:
  replicas: 3
  selector:
    matchLabels:
      app: paca-v5
  template:
    metadata:
      labels:
        app: paca-v5
    spec:
      containers:
      - name: paca-app
        image: paca-v5:latest
        ports:
        - containerPort: 8000
        env:
        - name: PACA_ENV
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: paca-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: paca-secrets
              key: redis-url
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: paca-v5-service
spec:
  selector:
    app: paca-v5
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: paca-v5-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - paca-v5.example.com
    secretName: paca-v5-tls
  rules:
  - host: paca-v5.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: paca-v5-service
            port:
              number: 80
```

## 🧪 CI/CD 파이프라인

### GitHub Actions

```yaml
# .github/workflows/ci-cd.yaml
name: PACA v5 CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]

    - name: Lint with flake8
      run: flake8 paca/ tests/

    - name: Type check with mypy
      run: mypy paca/ --strict

    - name: Test with pytest
      run: |
        pytest tests/ -v --cov=paca --cov-report=xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v2

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v3

    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production

    steps:
    - uses: actions/checkout@v3

    - name: Deploy to production
      run: |
        # 배포 스크립트 실행
        ./deployment/scripts/deploy.sh ${{ github.sha }}
```

### GitLab CI/CD

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"

test:
  stage: test
  image: python:3.11
  before_script:
    - pip install -e .[dev]
  script:
    - flake8 paca/ tests/
    - mypy paca/ --strict
    - pytest tests/ -v --cov=paca --cov-report=xml
  coverage: '/(?i)total.*? (100(?:\.0+)?\%|[1-9]?\d(?:\.\d+)?\%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main

deploy:
  stage: deploy
  image: google/cloud-sdk:alpine
  before_script:
    - echo $GCP_SERVICE_KEY | base64 -d > gcp-key.json
    - gcloud auth activate-service-account --key-file gcp-key.json
    - gcloud config set project $GCP_PROJECT_ID
  script:
    - gcloud run deploy paca-v5
        --image $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
        --region us-central1
        --platform managed
        --allow-unauthenticated
  only:
    - main
  environment:
    name: production
    url: https://paca-v5-prod.example.com
```

## 💡 배포 모니터링

### Prometheus 설정

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'paca-v5'
    static_configs:
      - targets: ['paca-app:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['db:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:80']
```

### Grafana 대시보드

```json
{
  "dashboard": {
    "title": "PACA v5 Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "process_resident_memory_bytes",
            "legendFormat": "Memory Usage"
          }
        ]
      }
    ]
  }
}
```

### 로그 집중화 (ELK Stack)

```yaml
# docker-compose.elk.yml
version: '3.8'

services:
  elasticsearch:
    image: elastic/elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: elastic/logstash:8.8.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch

  kibana:
    image: elastic/kibana:8.8.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:
```

---

**PACA v5와 함께 안정적인 프로덕션 환경을 구축하세요!** 🚀

*배포 관련 문의사항은 DevOps 팀 또는 GitHub Issues를 통해 연락주세요.*