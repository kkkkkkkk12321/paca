# Learning Memory Module - PACA Python v5

## 🎯 프로젝트 개요
PACA 학습 시스템의 메모리 저장소로, SQLite 기반 영구 학습 데이터 관리, 학습 포인트 저장, 생성된 전술/휴리스틱 보관을 담당합니다.

## 📁 폴더/파일 구조
```
memory/
├── __init__.py              # 모듈 진입점
└── storage.py               # SQLite 기반 학습 메모리 저장소
```

## ⚙️ 기능 요구사항
**입력**: 학습 포인트, 생성된 전술, 휴리스틱 데이터
**출력**: 저장 결과, 학습 이력, 성능 통계
**핵심 로직**: 데이터 입력 → SQLite 저장 → 인덱싱 → 검색 최적화

## 🛠️ 기술적 요구사항
- Python 3.9+ (sqlite3, json, pathlib)
- 포터블 SQLite 데이터베이스 (data/database/learning_memory.db)
- JSON 직렬화 지원

## 🚀 라우팅 및 진입점
```python
from paca.learning.memory import LearningMemory

# 학습 메모리 초기화
memory = LearningMemory()
await memory.initialize()

# 학습 포인트 저장
learning_point = LearningPoint(
    category=LearningCategory.REASONING,
    content="복잡한 추론 패턴",
    confidence=0.85,
    success_rate=0.92
)
await memory.save_learning_point(learning_point)

# 학습 이력 조회
history = await memory.get_learning_history(limit=10)
```

## 📋 코드 품질 가이드
- 클래스: PascalCase (LearningMemory)
- 테이블명: snake_case (learning_points, generated_tactics)
- 모든 DB 작업에 예외 처리 필수
- 트랜잭션 단위 작업 보장

## 🏃‍♂️ 실행 방법
```bash
python -c "from paca.learning.memory import LearningMemory; print('Learning memory loaded')"
```

## 🧪 테스트 방법
```bash
pytest tests/test_learning/test_memory/ -v
```

## 💡 추가 고려사항
**성능**: 인덱스 최적화, 배치 저장
**향후 개선**: 분산 저장소, 백업/복원, 학습 데이터 압축