# PACA 포터블 앱 가이드 📱

PACA가 완전한 포터블 애플리케이션으로 변환되었습니다! 이제 프로그램 폴더를 어디로든 이동해도 모든 데이터가 함께 보존됩니다.

## 🎯 포터블 기능 개요

### ✅ 구현된 기능

- **완전 포터블 저장소**: 모든 데이터가 `data/` 폴더에 저장
- **메모리 시스템 포터블화**: Working Memory와 Episodic Memory
- **자동 디렉토리 생성**: 필요한 폴더들이 자동으로 생성
- **크로스 플랫폼 호환**: Windows/Linux/macOS 모두 지원
- **데이터 지속성**: 프로그램 재시작 후에도 모든 데이터 유지

## 📁 디렉토리 구조

```
paca_python/
├── data/                    # 모든 데이터 저장소 (포터블)
│   ├── memory/             # 메모리 시스템 데이터
│   │   ├── working/        # 작업 메모리
│   │   ├── episodic/       # 일화 메모리
│   │   ├── semantic/       # 의미 메모리
│   │   └── long_term/      # 장기 메모리
│   ├── database/           # SQLite 데이터베이스
│   ├── logs/              # 로그 파일
│   ├── config/            # 설정 파일
│   ├── cache/             # 캐시 데이터
│   ├── uploads/           # 업로드 파일
│   └── temp/              # 임시 파일
├── paca/                   # PACA 소스 코드
└── setup_portable_storage.py  # 포터블 설정 스크립트
```

## 🚀 사용 방법

### 1. 포터블 저장소 초기 설정

```bash
# 포터블 저장소 설정 및 테스트
python setup_portable_storage.py

# 메모리 시스템 테스트
python test_portable_memory.py
```

### 2. PACA 실행

```bash
# 일반 실행
python -m paca

# 또는 환경 설정 후 실행
python setup_environment.py
python -m paca
```

### 3. 포터블 이동

프로그램 폴더 전체를 다른 위치로 복사하면 모든 데이터가 함께 이동됩니다:

```bash
# 예시: USB 드라이브로 복사
cp -r paca_python/ /media/usb/paca_portable/

# 새 위치에서 바로 실행 가능
cd /media/usb/paca_portable/
python -m paca
```

## 💾 메모리 시스템

### Working Memory (작업 메모리)
- **저장 위치**: `data/memory/working/working_memory.json`
- **기능**: 단기 작업 데이터, 키-값 저장
- **용량**: 기본 7개 항목 (Miller's Law)

### Episodic Memory (일화 메모리)
- **저장 위치**: `data/memory/episodic/episodic_memory.json`
- **기능**: 경험과 맥락 정보 저장
- **검색**: 시간 기반, 맥락 기반 검색 지원

### 자동 백업 및 복원
- 프로그램 시작 시 자동으로 이전 데이터 로드
- 데이터 변경 시 즉시 저장 (실시간 백업)
- JSON 형태로 가독성 있게 저장

## 🔧 고급 기능

### 데이터 내보내기/가져오기

```python
from paca.core.utils.portable_storage import get_storage_manager

storage_manager = get_storage_manager()

# 모든 데이터 내보내기
storage_manager.export_all_data("/path/to/backup/")

# 데이터 가져오기
storage_manager.import_all_data("/path/from/backup/")
```

### 저장소 정보 확인

```python
# 저장소 상태 확인
info = storage_manager.get_storage_info()
print(f"총 파일 수: {info['total_files']}")
print(f"사용 공간: {info['total_size_mb']:.2f} MB")
```

### 오래된 파일 정리

```python
# 30일 이상 된 파일 정리
cleanup_count = storage_manager.cleanup_old_files("working", days_old=30)
print(f"{cleanup_count}개 파일 정리 완료")
```

## 🔐 보안 및 안정성

### 데이터 안전성
- **원자적 저장**: 파일 쓰기 실패 시 이전 데이터 보존
- **오류 복구**: 손상된 파일 감지 시 자동 초기화
- **인코딩 안전**: UTF-8 기본, Windows CP949 호환

### 성능 최적화
- **지연 로딩**: 필요할 때만 데이터 로드
- **비동기 저장**: 메인 스레드 블로킹 방지
- **압축 저장**: JSON 압축으로 공간 절약

## 🌍 플랫폼 호환성

### Windows
- CP949/UTF-8 인코딩 자동 처리
- Windows 경로 형식 지원
- 파일 잠금 문제 해결

### Linux/macOS
- POSIX 경로 형식 지원
- 권한 관리 최적화
- 심볼릭 링크 처리

## 📊 포터블 기능 검증

다음 명령어로 포터블 기능을 검증할 수 있습니다:

```bash
# 1. 초기 설정 테스트
python setup_portable_storage.py

# 2. 메모리 시스템 테스트
python test_portable_memory.py

# 3. 실제 사용 테스트
python -m paca
```

### 검증 항목
- ✅ 디렉토리 자동 생성
- ✅ 메모리 데이터 저장/로드
- ✅ 데이터 지속성
- ✅ 키-값 저장 시스템
- ✅ 데이터베이스 연결
- ✅ 로그 파일 저장
- ✅ 크로스 플랫폼 호환성

## 🎉 이점

### 사용자 이점
1. **완전한 이동성**: USB, 클라우드, 네트워크 드라이브에서 실행
2. **제로 설정**: 추가 설치나 설정 없이 바로 사용
3. **데이터 안전**: 모든 데이터가 프로그램과 함께 보관
4. **백업 용이**: 폴더 복사만으로 완전 백업

### 개발자 이점
1. **배포 단순화**: 단일 폴더 패키징
2. **디버깅 용이**: 모든 데이터가 로컬에 저장
3. **테스트 격리**: 각 인스턴스가 독립된 데이터 공간
4. **확장성**: 새로운 저장소 타입 쉽게 추가 가능

## 🔮 향후 계획

- **압축 백업**: 자동 데이터 압축 및 백업
- **암호화**: 민감한 데이터 암호화 저장
- **동기화**: 여러 디바이스 간 데이터 동기화
- **클라우드 연동**: 클라우드 스토리지 직접 연동

---

**🎯 결론**: PACA는 이제 완전한 포터블 애플리케이션입니다. 어디서든 실행하고, 모든 데이터를 안전하게 보관하며, 필요할 때 언제든 이동할 수 있습니다!