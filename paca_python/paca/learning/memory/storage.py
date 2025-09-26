"""
Learning Memory Storage
학습 메모리 저장소 구현
"""

import sqlite3
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from ...core.types import Result, create_success, create_failure
from ...core.utils import generate_id
from ...core.utils.portable_storage import get_storage_manager
from ..auto.types import LearningPoint, GeneratedTactic, GeneratedHeuristic, LearningCategory

logger = logging.getLogger(__name__)


class LearningMemory:
    """
    학습 메모리 저장소
    SQLite 기반 학습 데이터 영구 저장
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # 포터블 저장소 사용
            storage_manager = get_storage_manager()
            db_path = str(storage_manager.get_database_path("learning_memory.db"))

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._initialize_database()

    def _initialize_database(self) -> None:
        """데이터베이스 초기화"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")

                # 학습 포인트 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS learning_points (
                        id TEXT PRIMARY KEY,
                        user_message TEXT NOT NULL,
                        paca_response TEXT NOT NULL,
                        context TEXT NOT NULL,
                        category TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        extracted_knowledge TEXT NOT NULL,
                        conversation_id TEXT,
                        source_pattern TEXT,
                        effectiveness_score REAL DEFAULT 0.0,
                        usage_count INTEGER DEFAULT 0,
                        last_used REAL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        metadata TEXT  -- JSON
                    )
                """)

                # 생성된 전술 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS generated_tactics (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT NOT NULL,
                        context TEXT NOT NULL,
                        category TEXT DEFAULT 'auto_generated',
                        success_count INTEGER DEFAULT 0,
                        total_applications INTEGER DEFAULT 0,
                        effectiveness REAL DEFAULT 0.0,
                        last_used REAL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        tags TEXT,  -- JSON array
                        source_conversations TEXT,  -- JSON array
                        metadata TEXT  -- JSON
                    )
                """)

                # 생성된 휴리스틱 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS generated_heuristics (
                        id TEXT PRIMARY KEY,
                        pattern TEXT NOT NULL,
                        avoidance_rule TEXT NOT NULL,
                        context TEXT NOT NULL,
                        category TEXT DEFAULT 'auto_generated',
                        triggered_count INTEGER DEFAULT 0,
                        avoided_count INTEGER DEFAULT 0,
                        effectiveness REAL DEFAULT 0.0,
                        last_triggered REAL,
                        severity TEXT DEFAULT 'medium',
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        source_conversations TEXT,  -- JSON array
                        metadata TEXT  -- JSON
                    )
                """)

                # 인덱스 생성
                conn.execute("CREATE INDEX IF NOT EXISTS idx_learning_points_category ON learning_points(category)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_learning_points_created_at ON learning_points(created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tactics_effectiveness ON generated_tactics(effectiveness)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_heuristics_effectiveness ON generated_heuristics(effectiveness)")

                conn.commit()
                logger.info("Learning memory database initialized")

        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise

    async def store_learning_point(self, learning_point: LearningPoint) -> Result[str]:
        """학습 포인트 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO learning_points (
                        id, user_message, paca_response, context, category,
                        confidence, extracted_knowledge, conversation_id, source_pattern,
                        effectiveness_score, usage_count, last_used,
                        created_at, updated_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    learning_point.id,
                    learning_point.user_message,
                    learning_point.paca_response,
                    learning_point.context,
                    learning_point.category.value,
                    learning_point.confidence,
                    learning_point.extracted_knowledge,
                    learning_point.conversation_id,
                    learning_point.source_pattern,
                    learning_point.effectiveness_score,
                    learning_point.usage_count,
                    learning_point.last_used,
                    learning_point.created_at,
                    learning_point.updated_at,
                    json.dumps(learning_point.metadata)
                ))
                conn.commit()

            return create_success(learning_point.id)

        except Exception as e:
            return create_failure(f"Failed to store learning point: {str(e)}")

    async def get_learning_points(
        self,
        category: Optional[LearningCategory] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Result[List[LearningPoint]]:
        """학습 포인트 조회"""
        try:
            query = """
                SELECT * FROM learning_points
                WHERE 1=1
            """
            params = []

            if category:
                query += " AND category = ?"
                params.append(category.value)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

            learning_points = []
            for row in rows:
                # 간단한 객체 재구성 (실제로는 더 정교한 역직렬화 필요)
                lp = LearningPoint(
                    id=row['id'],
                    user_message=row['user_message'],
                    paca_response=row['paca_response'],
                    context=row['context'],
                    category=LearningCategory(row['category']),
                    confidence=row['confidence'],
                    extracted_knowledge=row['extracted_knowledge'],
                    conversation_id=row['conversation_id'],
                    source_pattern=row['source_pattern'],
                    effectiveness_score=row['effectiveness_score'],
                    usage_count=row['usage_count'],
                    last_used=row['last_used'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
                learning_points.append(lp)

            return create_success(learning_points)

        except Exception as e:
            return create_failure(f"Failed to get learning points: {str(e)}")

    async def store_tactic(self, tactic: GeneratedTactic) -> Result[str]:
        """전술 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO generated_tactics (
                        id, name, description, context, category,
                        success_count, total_applications, effectiveness, last_used,
                        created_at, updated_at, tags, source_conversations, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tactic.id,
                    tactic.name,
                    tactic.description,
                    tactic.context,
                    tactic.category,
                    tactic.success_count,
                    tactic.total_applications,
                    tactic.effectiveness,
                    tactic.last_used,
                    tactic.created_at,
                    tactic.updated_at,
                    json.dumps(tactic.tags),
                    json.dumps(tactic.source_conversations),
                    json.dumps(tactic.metadata)
                ))
                conn.commit()

            return create_success(tactic.id)

        except Exception as e:
            return create_failure(f"Failed to store tactic: {str(e)}")

    async def store_heuristic(self, heuristic: GeneratedHeuristic) -> Result[str]:
        """휴리스틱 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO generated_heuristics (
                        id, pattern, avoidance_rule, context, category,
                        triggered_count, avoided_count, effectiveness, last_triggered,
                        severity, created_at, updated_at, source_conversations, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    heuristic.id,
                    heuristic.pattern,
                    heuristic.avoidance_rule,
                    heuristic.context,
                    heuristic.category,
                    heuristic.triggered_count,
                    heuristic.avoided_count,
                    heuristic.effectiveness,
                    heuristic.last_triggered,
                    heuristic.severity,
                    heuristic.created_at,
                    heuristic.updated_at,
                    json.dumps(heuristic.source_conversations),
                    json.dumps(heuristic.metadata)
                ))
                conn.commit()

            return create_success(heuristic.id)

        except Exception as e:
            return create_failure(f"Failed to store heuristic: {str(e)}")

    async def get_learning_statistics(self) -> Result[Dict[str, Any]]:
        """학습 통계 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}

                # 학습 포인트 통계
                cursor = conn.execute("SELECT COUNT(*) FROM learning_points")
                stats['total_learning_points'] = cursor.fetchone()[0]

                cursor = conn.execute("SELECT category, COUNT(*) FROM learning_points GROUP BY category")
                stats['learning_points_by_category'] = dict(cursor.fetchall())

                # 전술 통계
                cursor = conn.execute("SELECT COUNT(*) FROM generated_tactics")
                stats['total_tactics'] = cursor.fetchone()[0]

                cursor = conn.execute("SELECT AVG(effectiveness) FROM generated_tactics")
                avg_tactic_effectiveness = cursor.fetchone()[0]
                stats['average_tactic_effectiveness'] = avg_tactic_effectiveness or 0.0

                # 휴리스틱 통계
                cursor = conn.execute("SELECT COUNT(*) FROM generated_heuristics")
                stats['total_heuristics'] = cursor.fetchone()[0]

                cursor = conn.execute("SELECT AVG(effectiveness) FROM generated_heuristics")
                avg_heuristic_effectiveness = cursor.fetchone()[0]
                stats['average_heuristic_effectiveness'] = avg_heuristic_effectiveness or 0.0

                # 최근 활동
                one_week_ago = time.time() - (7 * 24 * 60 * 60)
                cursor = conn.execute("SELECT COUNT(*) FROM learning_points WHERE created_at > ?", (one_week_ago,))
                stats['recent_learning_points'] = cursor.fetchone()[0]

            return create_success(stats)

        except Exception as e:
            return create_failure(f"Failed to get learning statistics: {str(e)}")

    async def cleanup_old_data(self, days_to_keep: int = 90) -> Result[int]:
        """오래된 데이터 정리"""
        try:
            cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
            deleted_count = 0

            with sqlite3.connect(self.db_path) as conn:
                # 오래된 학습 포인트 삭제 (효과성이 낮은 것들)
                cursor = conn.execute("""
                    DELETE FROM learning_points
                    WHERE created_at < ? AND effectiveness_score < 0.3
                """, (cutoff_time,))
                deleted_count += cursor.rowcount

                # 비효과적인 전술 삭제
                cursor = conn.execute("""
                    DELETE FROM generated_tactics
                    WHERE created_at < ? AND effectiveness < 0.3 AND total_applications > 5
                """, (cutoff_time,))
                deleted_count += cursor.rowcount

                # 비효과적인 휴리스틱 삭제
                cursor = conn.execute("""
                    DELETE FROM generated_heuristics
                    WHERE created_at < ? AND effectiveness < 0.3 AND triggered_count > 3
                """, (cutoff_time,))
                deleted_count += cursor.rowcount

                conn.commit()

            return create_success(deleted_count)

        except Exception as e:
            return create_failure(f"Failed to cleanup old data: {str(e)}")

    async def search_learning_points(
        self,
        search_query: str,
        category: Optional[LearningCategory] = None,
        min_confidence: float = 0.0
    ) -> Result[List[LearningPoint]]:
        """학습 포인트 검색"""
        try:
            query = """
                SELECT * FROM learning_points
                WHERE (
                    user_message LIKE ? OR
                    paca_response LIKE ? OR
                    extracted_knowledge LIKE ?
                ) AND confidence >= ?
            """
            params = [f"%{search_query}%", f"%{search_query}%", f"%{search_query}%", min_confidence]

            if category:
                query += " AND category = ?"
                params.append(category.value)

            query += " ORDER BY confidence DESC, created_at DESC"

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

            # 결과 변환 (간단한 버전)
            learning_points = []
            for row in rows:
                lp = LearningPoint(
                    id=row['id'],
                    user_message=row['user_message'],
                    paca_response=row['paca_response'],
                    context=row['context'],
                    category=LearningCategory(row['category']),
                    confidence=row['confidence'],
                    extracted_knowledge=row['extracted_knowledge']
                    # 필요한 다른 필드들도 추가...
                )
                learning_points.append(lp)

            return create_success(learning_points)

        except Exception as e:
            return create_failure(f"Failed to search learning points: {str(e)}")

    def get_db_info(self) -> Dict[str, Any]:
        """데이터베이스 정보 조회"""
        try:
            info = {
                "db_path": str(self.db_path),
                "db_size": self.db_path.stat().st_size if self.db_path.exists() else 0,
                "db_exists": self.db_path.exists()
            }

            if self.db_path.exists():
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    info["tables"] = [row[0] for row in cursor.fetchall()]

            return info

        except Exception as e:
            logger.error(f"Failed to get database info: {str(e)}")
            return {"error": str(e)}