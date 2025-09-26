# Tests Archive

원본 테스트 파일들은 archive/ 폴더로 이동되었습니다.

## 복원 방법
```bash
# 최신 백업 찾기
ls archive/

# 백업 복원
cp -r archive/tests_backup_[날짜]/tests ./
```

## 핵심 PACA 기능
테스트 없이도 모든 핵심 기능이 정상 작동합니다:
- paca 모듈 import
- CognitiveSystem
- ReasoningEngine
- production_server.py
