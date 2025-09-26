# PACA v5 Troubleshooting Guide

## 🎯 프로젝트 개요

PACA v5 사용 중 발생할 수 있는 모든 문제들에 대한 체계적인 해결 가이드. 설치 문제부터 성능 이슈, 에러 메시지까지 단계별 해결 방법을 제공합니다.

## 📁 문제 분류 구조

```
🔧 문제 해결 가이드
├── 🚀 설치 및 시작 문제
├── 🖥️ GUI 관련 문제
├── ⚡ 성능 및 응답 문제
├── 🧠 인지 처리 문제
├── 🔢 수학 계산 문제
├── 💾 메모리 및 저장 문제
├── 🌐 네트워크 및 연결 문제
├── 🔒 보안 및 권한 문제
├── 📊 로그 및 디버깅
└── 🚨 응급 복구 방법
```

## ⚙️ 일반적인 문제 해결 절차

### 1단계: 기본 진단
```bash
# 시스템 정보 확인
python --version
pip --version
paca --version

# 패키지 상태 확인
pip list | grep paca
pip check

# 시스템 리소스 확인
# Windows
wmic computersystem get TotalPhysicalMemory
wmic cpu get Name

# Linux/Mac
free -h
lscpu
```

### 2단계: 로그 확인
```bash
# PACA 로그 위치
# Windows: %APPDATA%\PACA\logs\
# Linux/Mac: ~/.paca/logs/

# 최근 로그 확인
tail -f ~/.paca/logs/paca.log

# 에러 로그만 확인
grep -i error ~/.paca/logs/paca.log

# 특정 시간대 로그
grep "2024-09-20 14:" ~/.paca/logs/paca.log
```

### 3단계: 환경 변수 검증
```bash
# 환경 변수 확인
echo $PACA_ENV
echo $PYTHONPATH

# 설정 파일 검증
python -c "from paca.config import ConfigManager; print(ConfigManager().validate())"
```

## 🚀 설치 및 시작 문제

### Python 버전 호환성 문제

#### 문제: "Python 3.9+ required" 오류
```
Error: PACA v5 requires Python 3.9 or higher. Current: Python 3.8.10
```

**해결 방법:**
```bash
# 1. Python 버전 확인
python --version

# 2. Python 3.9+ 설치
# Windows - Python.org에서 다운로드
# Ubuntu
sudo apt update
sudo apt install python3.11

# macOS - Homebrew
brew install python@3.11

# 3. 가상환경 재생성
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
pip install -e .
```

### 의존성 설치 실패

#### 문제: "Failed to build wheel for numpy" 오류
```
ERROR: Failed building wheel for numpy
Building wheel for numpy (setup.py) ... error
```

**해결 방법:**
```bash
# 1. 시스템 의존성 설치
# Ubuntu/Debian
sudo apt-get install python3-dev build-essential

# CentOS/RHEL
sudo yum install python3-devel gcc gcc-c++

# macOS
xcode-select --install

# 2. 업그레이드된 pip 사용
pip install --upgrade pip setuptools wheel

# 3. 바이너리 패키지 설치
pip install --only-binary=all numpy sympy

# 4. conda 환경 사용 (대안)
conda create -n paca python=3.11
conda activate paca
conda install numpy sympy
pip install -e .
```

### 실행 파일 시작 실패

#### 문제: Windows에서 "The application was unable to start correctly (0xc000007b)"
**해결 방법:**
```batch
REM 1. Visual C++ 재배포 패키지 설치
REM Microsoft Visual C++ 2015-2022 Redistributable 다운로드 및 설치

REM 2. .NET Framework 확인
REM .NET Framework 4.8 이상 설치

REM 3. 관리자 권한으로 실행
REM 우클릭 → "관리자 권한으로 실행"

REM 4. 호환성 모드 설정
REM 우클릭 → 속성 → 호환성 → Windows 10 모드
```

#### 문제: "ModuleNotFoundError: No module named 'paca'"
**해결 방법:**
```bash
# 1. 설치 상태 확인
pip show paca

# 2. 개발 모드로 재설치
pip uninstall paca
pip install -e .

# 3. PYTHONPATH 설정
export PYTHONPATH="${PYTHONPATH}:/path/to/paca_python"

# 4. 시스템 설치 (권장하지 않음)
pip install .
```

## 🖥️ GUI 관련 문제

### CustomTkinter 문제

#### 문제: "tkinter.TclError: couldn't connect to display"
**해결 방법:**
```bash
# Linux에서 X11 forwarding 활성화
export DISPLAY=:0.0

# SSH 접속시
ssh -X username@hostname

# WSL에서
# Windows에 VcXsrv 설치 후
export DISPLAY=:0

# 대안: CLI 모드 사용
paca --no-gui "질문 내용"
```

#### 문제: GUI가 흐리게 보임 (Windows)
**해결 방법:**
```python
# 애플리케이션 설정에서 DPI 인식 활성화
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(1)

# 또는 매니페스트 파일 추가
# app.manifest에 dpiAware 설정
```

### 테마 및 렌더링 문제

#### 문제: 다크 모드에서 텍스트가 안 보임
**해결 방법:**
```python
# 설정 파일 수정 (config.yaml)
gui:
  theme: "light"  # 또는 "dark", "auto"
  force_theme_override: true

# 또는 환경 변수
export PACA_GUI_THEME=light
```

#### 문제: 한글 폰트가 깨짐
**해결 방법:**
```python
# 1. 시스템 한글 폰트 설치 확인
# Windows: 맑은 고딕
# macOS: Apple SD Gothic Neo
# Linux: Noto Sans CJK KR

# 2. 폰트 강제 설정
# config.yaml
gui:
  font_family: "맑은 고딕"  # Windows
  font_family: "Apple SD Gothic Neo"  # macOS
  font_family: "Noto Sans CJK KR"  # Linux
  font_size: 12
```

## ⚡ 성능 및 응답 문제

### 응답 속도 느림

#### 문제: 응답 시간이 10초 이상 걸림
**진단:**
```python
# 성능 프로파일링 실행
python -m paca --profile "성능 테스트 질문"

# 메모리 사용량 확인
python -m paca --memory-monitor

# CPU 사용량 모니터링
top -p $(pgrep -f paca)
```

**해결 방법:**
```bash
# 1. 캐시 활성화
export PACA_ENABLE_CACHE=true
export PACA_CACHE_SIZE=1000

# 2. 병렬 처리 설정
export PACA_PARALLEL_WORKERS=4

# 3. 메모리 정리
python -c "
import gc
from paca.core.utils import clear_memory_cache
clear_memory_cache()
gc.collect()
"

# 4. 빠른 응답 모드
paca --fast-mode "질문 내용"
```

### 메모리 사용량 과다

#### 문제: 메모리 사용량이 4GB 이상
**해결 방법:**
```python
# 1. 메모리 프로파일링
python -m memory_profiler paca_script.py

# 2. 메모리 정리 설정
# config.yaml
performance:
  max_memory_usage: "2GB"
  auto_memory_cleanup: true
  cleanup_interval: 300  # 5분마다

# 3. 가비지 컬렉션 강제 실행
import gc
gc.set_threshold(700, 10, 10)  # 더 자주 GC 실행
gc.collect()

# 4. 대화 기록 제한
conversation:
  max_history_length: 100
  auto_cleanup_old_conversations: true
```

### CPU 사용률 100%

#### 문제: CPU 사용률이 지속적으로 높음
**해결 방법:**
```bash
# 1. 프로세스 우선순위 조정
nice -n 10 python -m paca --gui

# 2. CPU 제한 설정
# config.yaml
performance:
  max_cpu_cores: 2
  cpu_throttling: true
  max_cpu_usage: 70

# 3. 백그라운드 작업 중지
pkill -f "background_task"

# 4. 동시 처리 수 제한
threading:
  max_workers: 2
  thread_pool_size: 4
```

## 🧠 인지 처리 문제

### 추론 결과 부정확

#### 문제: 논리적 추론 결과가 명백히 틀림
**진단:**
```python
# 추론 과정 상세 로그 활성화
import logging
logging.getLogger('paca.reasoning').setLevel(logging.DEBUG)

# 추론 체인 시각화
from paca.reasoning import ReasoningChain
chain = ReasoningChain()
chain.enable_debugging()
result = chain.process("추론 문제")
print(chain.get_debug_info())
```

**해결 방법:**
```python
# 1. 신뢰도 임계값 조정
# config.yaml
reasoning:
  confidence_threshold: 0.8  # 기본 0.7에서 상향
  require_high_confidence: true
  fallback_to_simple_logic: true

# 2. 추론 모델 재초기화
from paca.reasoning import ReasoningEngine
engine = ReasoningEngine()
engine.reset_model()
engine.reload_rules()

# 3. 전제 검증 강화
reasoning:
  validate_premises: true
  require_evidence: true
  check_logical_consistency: true
```

### 한국어 처리 문제

#### 문제: 한국어 문장 이해가 부정확
**해결 방법:**
```python
# 1. 한국어 모델 활성화
# config.yaml
language:
  primary: "ko"
  korean_nlp_enabled: true
  tokenizer: "korean_specific"

# 2. KoNLPy 설치 및 설정
pip install konlpy

# 환경 변수 설정
export KONLPY_DATA_PATH=/path/to/konlpy/data

# 3. 형태소 분석기 변경
nlp:
  korean_tokenizer: "mecab"  # 또는 "okt", "komoran"
  use_spacing_correction: true
  handle_informal_language: true
```

### 컨텍스트 기억 실패

#### 문제: 이전 대화 내용을 기억하지 못함
**해결 방법:**
```python
# 1. 메모리 시스템 상태 확인
from paca.services.memory import MemoryService
memory = MemoryService()
print(memory.get_status())
print(f"저장된 대화 수: {memory.count_conversations()}")

# 2. 메모리 저장 설정 확인
# config.yaml
memory:
  enable_conversation_memory: true
  max_conversation_length: 50
  memory_persistence: true
  auto_save_interval: 60

# 3. 메모리 데이터베이스 복구
memory.repair_database()
memory.rebuild_index()
```

## 🔢 수학 계산 문제

### SymPy 계산 오류

#### 문제: "sympy.SympifyError: SympifyError: ..."
**해결 방법:**
```python
# 1. 입력 형식 검증
from paca.mathematics import Calculator
calc = Calculator()

# 잘못된 입력 예제와 수정
bad_input = "2++3"
good_input = "2+3"

# 2. 수식 전처리 활성화
# config.yaml
mathematics:
  enable_input_preprocessing: true
  fix_common_typos: true
  validate_syntax: true

# 3. SymPy 버전 확인 및 업데이트
pip install --upgrade sympy
python -c "import sympy; print(sympy.__version__)"
```

### 복잡한 계산 시간 초과

#### 문제: "Calculation timeout after 30 seconds"
**해결 방법:**
```python
# 1. 계산 타임아웃 조정
# config.yaml
mathematics:
  calculation_timeout: 60  # 60초로 증가
  enable_approximate_solutions: true
  max_computation_complexity: "medium"

# 2. 수치 계산 우선 사용
calc.set_mode("numerical")  # symbolic 대신

# 3. 단계별 계산
complex_expr = "integrate(x^2 * sin(x), x, 0, pi)"
# 단순화
simplified = calc.simplify_expression(complex_expr)
result = calc.calculate(simplified)
```

### 통계 분석 오류

#### 문제: "ValueError: Input array is empty"
**해결 방법:**
```python
# 1. 입력 데이터 검증
def safe_statistical_analysis(data):
    if not data:
        return {"error": "빈 데이터셋입니다"}

    if len(data) < 2:
        return {"error": "최소 2개 이상의 데이터가 필요합니다"}

    # 수치 데이터 변환 시도
    try:
        numeric_data = [float(x) for x in data]
        return calc.statistical_analysis(numeric_data)
    except ValueError as e:
        return {"error": f"수치 변환 실패: {e}"}

# 2. 결측값 처리
# config.yaml
mathematics:
  handle_missing_values: true
  missing_value_strategy: "mean"  # "median", "drop"
  outlier_detection: true
```

## 💾 메모리 및 저장 문제

### 데이터베이스 연결 실패

#### 문제: "OperationalError: no such table: conversations"
**해결 방법:**
```python
# 1. 데이터베이스 초기화
from paca.data import DatabaseManager
db = DatabaseManager()
db.initialize_schema()
db.create_tables()

# 2. 마이그레이션 실행
db.run_migrations()

# 3. 데이터베이스 복구
db.repair_database()

# 4. 백업에서 복원
db.restore_from_backup("backup_20240920.db")
```

### 저장 공간 부족

#### 문제: "OSError: [Errno 28] No space left on device"
**해결 방법:**
```bash
# 1. 디스크 사용량 확인
df -h
du -sh ~/.paca/

# 2. 로그 파일 정리
find ~/.paca/logs/ -name "*.log" -mtime +7 -delete

# 3. 캐시 정리
rm -rf ~/.paca/cache/*

# 4. 오래된 대화 기록 정리
python -c "
from paca.services.memory import MemoryService
memory = MemoryService()
memory.cleanup_old_conversations(days=30)
"

# 5. 설정에서 자동 정리 활성화
# config.yaml
storage:
  auto_cleanup: true
  max_storage_size: "1GB"
  cleanup_interval: "daily"
```

### 메모리 누수

#### 문제: 장시간 사용시 메모리 사용량 지속 증가
**해결 방법:**
```python
# 1. 메모리 누수 탐지
import tracemalloc
tracemalloc.start()

# PACA 사용 후
current, peak = tracemalloc.get_traced_memory()
print(f"현재 메모리: {current / 1024 / 1024:.1f} MB")
print(f"최대 메모리: {peak / 1024 / 1024:.1f} MB")

# 2. 약한 참조 사용 설정
# config.yaml
memory:
  use_weak_references: true
  auto_gc_interval: 300  # 5분마다 가비지 컬렉션
  memory_monitoring: true

# 3. 주기적 재시작
# 시스템 크론탭에 추가
# 0 3 * * * /usr/bin/systemctl restart paca.service
```

## 🌐 네트워크 및 연결 문제

### API 연결 실패

#### 문제: "ConnectionError: Failed to connect to API server"
**해결 방법:**
```bash
# 1. 네트워크 연결 확인
ping google.com
curl -I https://api.openai.com/

# 2. 방화벽 설정 확인
# Windows
netsh advfirewall firewall show rule name="PACA v5"

# Linux
sudo ufw status
sudo iptables -L

# 3. 프록시 설정
export http_proxy=http://proxy.company.com:8080
export https_proxy=http://proxy.company.com:8080

# 4. SSL 인증서 문제 해결
pip install --upgrade certifi
export SSL_CERT_FILE=$(python -m certifi)
```

### DNS 해결 실패

#### 문제: "gaierror: [Errno -2] Name or service not known"
**해결 방법:**
```bash
# 1. DNS 서버 확인
nslookup google.com
dig google.com

# 2. DNS 서버 변경
# /etc/resolv.conf (Linux)
nameserver 8.8.8.8
nameserver 8.8.4.4

# 3. 호스트 파일 확인
# Windows: C:\Windows\System32\drivers\etc\hosts
# Linux/Mac: /etc/hosts

# 4. 네트워크 인터페이스 재시작
# Linux
sudo systemctl restart NetworkManager

# Windows
ipconfig /release
ipconfig /renew
ipconfig /flushdns
```

## 🔒 보안 및 권한 문제

### 권한 거부 오류

#### 문제: "PermissionError: [Errno 13] Permission denied"
**해결 방법:**
```bash
# 1. 파일 권한 확인
ls -la ~/.paca/
ls -la /opt/paca/

# 2. 권한 수정
chmod 755 ~/.paca/
chmod 644 ~/.paca/config.yaml

# 3. 소유권 확인
sudo chown -R $USER:$USER ~/.paca/

# 4. SELinux 확인 (Linux)
getenforce
# Enforcing이면
sudo setsebool -P httpd_can_network_connect 1

# 5. 관리자 권한으로 실행 (Windows)
# 우클릭 → "관리자 권한으로 실행"
```

### 보안 인증서 문제

#### 문제: "SSLCertVerificationError: certificate verify failed"
**해결 방법:**
```python
# 1. 인증서 업데이트
pip install --upgrade certifi

# 2. 환경 변수 설정
import ssl
import certifi
ssl_context = ssl.create_default_context(cafile=certifi.where())

# 3. 기업 프록시 인증서 추가
# config.yaml
security:
  ssl_verify: true
  custom_ca_bundle: "/path/to/corporate-ca.pem"
  trust_corporate_proxy: true

# 4. 임시 우회 (비권장)
security:
  ssl_verify: false  # 개발 환경에서만 사용
```

## 📊 로그 및 디버깅

### 로그 레벨 조정

```python
# 1. 설정 파일에서 조정
# config.yaml
logging:
  level: "DEBUG"  # ERROR, WARNING, INFO, DEBUG
  format: "detailed"
  enable_file_logging: true
  log_file_path: "~/.paca/logs/debug.log"

# 2. 환경 변수로 조정
export PACA_LOG_LEVEL=DEBUG

# 3. 프로그래밍 방식으로 조정
import logging
logging.getLogger('paca').setLevel(logging.DEBUG)
```

### 상세 디버그 정보 활성화

```python
# 모든 모듈의 디버그 정보 활성화
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('paca_debug.log'),
        logging.StreamHandler()
    ]
)

# 특정 모듈만 디버그
logging.getLogger('paca.cognitive').setLevel(logging.DEBUG)
logging.getLogger('paca.reasoning').setLevel(logging.DEBUG)
```

### 성능 프로파일링

```python
# 1. cProfile 사용
python -m cProfile -o profile_output.prof -m paca "테스트 질문"

# 결과 분석
python -c "
import pstats
p = pstats.Stats('profile_output.prof')
p.sort_stats('cumulative').print_stats(20)
"

# 2. line_profiler 사용
pip install line_profiler
kernprof -l -v paca_script.py

# 3. memory_profiler 사용
pip install memory_profiler
python -m memory_profiler paca_script.py
```

## 🚨 응급 복구 방법

### 시스템 초기화

```bash
# 1. 완전 초기화 (주의: 모든 데이터 삭제)
rm -rf ~/.paca/
pip uninstall paca
pip install -e .

# 2. 설정만 초기화
mv ~/.paca/config.yaml ~/.paca/config.yaml.backup
paca --init-config

# 3. 데이터베이스만 초기화
mv ~/.paca/data/ ~/.paca/data_backup/
paca --init-database
```

### 백업에서 복원

```bash
# 1. 설정 백업 복원
cp ~/.paca/backups/config_20240920.yaml ~/.paca/config.yaml

# 2. 데이터베이스 백업 복원
cp ~/.paca/backups/database_20240920.db ~/.paca/data/paca.db

# 3. 전체 시스템 백업 복원
tar -xzf paca_backup_20240920.tar.gz -C ~/
```

### 안전 모드 실행

```bash
# 최소한의 기능으로 실행
paca --safe-mode --no-gui --no-cache "테스트"

# 플러그인 없이 실행
paca --no-plugins "테스트"

# 기본 설정으로 실행
paca --default-config "테스트"
```

## 💡 예방 조치

### 정기 점검 스크립트

```bash
#!/bin/bash
# paca_health_check.sh

echo "PACA v5 건강 상태 점검 시작..."

# 1. 디스크 공간 확인
df -h | grep -E "(/$|/home)" | awk '{print $5}' | grep -q "9[0-9]%" && echo "⚠️  디스크 공간 부족"

# 2. 메모리 사용량 확인
free | grep Mem | awk '{printf "메모리 사용률: %.1f%%\n", $3/$2 * 100.0}'

# 3. PACA 프로세스 확인
pgrep -f paca > /dev/null || echo "❌ PACA 프로세스가 실행되지 않음"

# 4. 로그 오류 확인
tail -100 ~/.paca/logs/paca.log | grep -i error | wc -l |
    awk '{if($1>0) print "⚠️  최근 에러 " $1 "건 발견"}'

# 5. 설정 파일 검증
python -c "
from paca.config import ConfigManager
try:
    config = ConfigManager()
    config.validate()
    print('✅ 설정 파일 정상')
except Exception as e:
    print(f'❌ 설정 파일 오류: {e}')
"

echo "건강 상태 점검 완료"
```

### 자동 백업 설정

```bash
# 크론탭에 추가 (crontab -e)
# 매일 새벽 3시에 백업
0 3 * * * /home/user/scripts/paca_backup.sh

# paca_backup.sh
#!/bin/bash
BACKUP_DIR=~/.paca/backups
DATE=$(date +%Y%m%d)

mkdir -p $BACKUP_DIR

# 설정 백업
cp ~/.paca/config.yaml $BACKUP_DIR/config_$DATE.yaml

# 데이터베이스 백업
cp ~/.paca/data/paca.db $BACKUP_DIR/database_$DATE.db

# 대화 기록 백업
tar -czf $BACKUP_DIR/conversations_$DATE.tar.gz ~/.paca/conversations/

# 오래된 백업 삭제 (30일 이상)
find $BACKUP_DIR -name "*" -mtime +30 -delete

echo "백업 완료: $DATE"
```

### 모니터링 설정

```python
# monitoring.py
import psutil
import time
import logging
from pathlib import Path

def monitor_paca():
    """PACA 프로세스 모니터링"""
    logging.basicConfig(
        filename=Path.home() / '.paca/logs/monitor.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

    while True:
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=1)

            # 메모리 사용률
            memory = psutil.virtual_memory()

            # PACA 프로세스 확인
            paca_processes = [p for p in psutil.process_iter(['name'])
                            if 'paca' in p.info['name'].lower()]

            if cpu_percent > 80:
                logging.warning(f"높은 CPU 사용률: {cpu_percent}%")

            if memory.percent > 90:
                logging.warning(f"높은 메모리 사용률: {memory.percent}%")

            if not paca_processes:
                logging.error("PACA 프로세스가 실행되지 않음")

            time.sleep(60)  # 1분마다 확인

        except Exception as e:
            logging.error(f"모니터링 오류: {e}")
            time.sleep(60)

if __name__ == "__main__":
    monitor_paca()
```

---

**문제 해결에 도움이 되었나요?** 🔧

*추가 지원이 필요하시면 GitHub Issues나 커뮤니티 포럼을 이용해주세요.*