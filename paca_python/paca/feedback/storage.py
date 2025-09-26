"""
피드백 저장소
SQLite 기반 피드백 데이터 저장 및 관리
"""

import sqlite3
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
import aiosqlite
import logging

from .models import (
    FeedbackModel, FeedbackType, FeedbackStatus, UserSession,
    FeedbackContext, FeedbackStats, SentimentScore
)
from ..core.utils.portable_storage import get_storage_manager

logger = logging.getLogger(__name__)


class FeedbackStorage:
    """피드백 저장소 클래스"""

    def __init__(self, db_path: str = None):
        """
        초기화

        Args:
            db_path: 데이터베이스 파일 경로 (기본값: 포터블 경로)
        """
        if db_path is None:
            # 포터블 저장소 사용
            storage_manager = get_storage_manager()
            self.db_path = str(storage_manager.get_database_path("feedback.db"))
        else:
            self.db_path = db_path
        self._ensure_directory()

    def _ensure_directory(self):
        """데이터베이스 디렉토리 확인 및 생성"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """데이터베이스 초기화"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await self._create_tables(db)
                await self._create_indices(db)
                await db.commit()
                logger.info(f"Feedback database initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize feedback database: {e}")
            raise

    async def _create_tables(self, db: aiosqlite.Connection):
        """테이블 생성"""

        # 피드백 테이블
        await db.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',

                -- 피드백 내용
                rating INTEGER,
                text_feedback TEXT,
                sentiment_score INTEGER,

                -- 컨텍스트 정보
                session_id TEXT NOT NULL,
                step_id TEXT,
                tool_name TEXT,
                action_type TEXT,
                execution_time REAL,
                success BOOLEAN,
                error_message TEXT,
                user_query TEXT,
                system_response TEXT,

                -- 메타데이터
                user_id TEXT,
                ip_address TEXT,
                user_agent TEXT,

                -- 처리 정보
                reviewed_by TEXT,
                reviewed_at TEXT,
                resolution_notes TEXT,

                -- 추가 데이터
                metadata TEXT,  -- JSON
                tags TEXT       -- JSON array
            )
        """)

        # 사용자 세션 테이블
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                total_interactions INTEGER DEFAULT 0,
                successful_interactions INTEGER DEFAULT 0,
                failed_interactions INTEGER DEFAULT 0,
                tools_used TEXT,  -- JSON array
                average_response_time REAL DEFAULT 0.0,
                user_satisfaction REAL
            )
        """)

        # 피드백 분석 결과 테이블
        await db.execute("""
            CREATE TABLE IF NOT EXISTS feedback_analysis (
                analysis_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                analysis_data TEXT NOT NULL  -- JSON
            )
        """)

    async def _create_indices(self, db: aiosqlite.Connection):
        """인덱스 생성"""
        indices = [
            "CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type)",
            "CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback(status)",
            "CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_feedback_tool ON feedback(tool_name)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON user_sessions(start_time)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON user_sessions(user_id)",
        ]

        for index_sql in indices:
            await db.execute(index_sql)

    async def save_feedback(self, feedback: FeedbackModel) -> bool:
        """피드백 저장"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                context = feedback.context

                await db.execute("""
                    INSERT INTO feedback (
                        id, timestamp, feedback_type, status,
                        rating, text_feedback, sentiment_score,
                        session_id, step_id, tool_name, action_type,
                        execution_time, success, error_message,
                        user_query, system_response,
                        user_id, ip_address, user_agent,
                        reviewed_by, reviewed_at, resolution_notes,
                        metadata, tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback.id,
                    feedback.timestamp.isoformat(),
                    feedback.feedback_type.value,
                    feedback.status.value,
                    feedback.rating,
                    feedback.text_feedback,
                    feedback.sentiment_score.value if feedback.sentiment_score else None,
                    feedback.session_id,
                    context.step_id if context else None,
                    context.tool_name if context else None,
                    context.action_type if context else None,
                    context.execution_time if context else None,
                    context.success if context else None,
                    context.error_message if context else None,
                    context.user_query if context else None,
                    context.system_response if context else None,
                    feedback.user_id,
                    feedback.ip_address,
                    feedback.user_agent,
                    feedback.reviewed_by,
                    feedback.reviewed_at.isoformat() if feedback.reviewed_at else None,
                    feedback.resolution_notes,
                    json.dumps(feedback.metadata),
                    json.dumps(feedback.tags)
                ))

                await db.commit()
                logger.info(f"Feedback saved: {feedback.id}")
                return True

        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
            return False

    async def get_feedback(self, feedback_id: str) -> Optional[FeedbackModel]:
        """피드백 조회"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row

                async with db.execute(
                    "SELECT * FROM feedback WHERE id = ?", (feedback_id,)
                ) as cursor:
                    row = await cursor.fetchone()

                    if row:
                        return self._row_to_feedback(row)
                    return None

        except Exception as e:
            logger.error(f"Failed to get feedback {feedback_id}: {e}")
            return None

    async def list_feedback(
        self,
        feedback_type: Optional[FeedbackType] = None,
        status: Optional[FeedbackStatus] = None,
        session_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[FeedbackModel]:
        """피드백 목록 조회"""
        try:
            query = "SELECT * FROM feedback WHERE 1=1"
            params = []

            if feedback_type:
                query += " AND feedback_type = ?"
                params.append(feedback_type.value)

            if status:
                query += " AND status = ?"
                params.append(status.value)

            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())

            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row

                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    return [self._row_to_feedback(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to list feedback: {e}")
            return []

    async def update_feedback_status(
        self,
        feedback_id: str,
        status: FeedbackStatus,
        reviewed_by: Optional[str] = None,
        resolution_notes: Optional[str] = None
    ) -> bool:
        """피드백 상태 업데이트"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    UPDATE feedback
                    SET status = ?, reviewed_by = ?, reviewed_at = ?, resolution_notes = ?
                    WHERE id = ?
                """, (
                    status.value,
                    reviewed_by,
                    datetime.now().isoformat(),
                    resolution_notes,
                    feedback_id
                ))

                await db.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to update feedback status: {e}")
            return False

    async def get_feedback_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> FeedbackStats:
        """피드백 통계 조회"""
        try:
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()

            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row

                # 기본 통계
                stats = FeedbackStats()

                # 총 피드백 수
                async with db.execute(
                    "SELECT COUNT(*) as count FROM feedback WHERE timestamp BETWEEN ? AND ?",
                    (start_date.isoformat(), end_date.isoformat())
                ) as cursor:
                    row = await cursor.fetchone()
                    stats.total_feedback = row['count']

                # 타입별 통계
                async with db.execute("""
                    SELECT feedback_type, COUNT(*) as count
                    FROM feedback
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY feedback_type
                """, (start_date.isoformat(), end_date.isoformat())) as cursor:
                    async for row in cursor:
                        stats.feedback_by_type[row['feedback_type']] = row['count']

                # 상태별 통계
                async with db.execute("""
                    SELECT status, COUNT(*) as count
                    FROM feedback
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY status
                """, (start_date.isoformat(), end_date.isoformat())) as cursor:
                    async for row in cursor:
                        stats.feedback_by_status[row['status']] = row['count']

                # 평균 평점
                async with db.execute("""
                    SELECT AVG(rating) as avg_rating
                    FROM feedback
                    WHERE rating IS NOT NULL AND timestamp BETWEEN ? AND ?
                """, (start_date.isoformat(), end_date.isoformat())) as cursor:
                    row = await cursor.fetchone()
                    stats.average_rating = row['avg_rating']

                # 감정 분포
                async with db.execute("""
                    SELECT sentiment_score, COUNT(*) as count
                    FROM feedback
                    WHERE sentiment_score IS NOT NULL AND timestamp BETWEEN ? AND ?
                    GROUP BY sentiment_score
                """, (start_date.isoformat(), end_date.isoformat())) as cursor:
                    async for row in cursor:
                        sentiment = SentimentScore(row['sentiment_score'])
                        stats.sentiment_distribution[sentiment.name] = row['count']

                return stats

        except Exception as e:
            logger.error(f"Failed to get feedback stats: {e}")
            return FeedbackStats()

    async def save_user_session(self, session: UserSession) -> bool:
        """사용자 세션 저장"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO user_sessions (
                        session_id, user_id, start_time, end_time,
                        total_interactions, successful_interactions, failed_interactions,
                        tools_used, average_response_time, user_satisfaction
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    session.user_id,
                    session.start_time.isoformat(),
                    session.end_time.isoformat() if session.end_time else None,
                    session.total_interactions,
                    session.successful_interactions,
                    session.failed_interactions,
                    json.dumps(session.tools_used),
                    session.average_response_time,
                    session.user_satisfaction
                ))

                await db.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to save user session: {e}")
            return False

    async def get_user_session(self, session_id: str) -> Optional[UserSession]:
        """사용자 세션 조회"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row

                async with db.execute(
                    "SELECT * FROM user_sessions WHERE session_id = ?", (session_id,)
                ) as cursor:
                    row = await cursor.fetchone()

                    if row:
                        return UserSession(
                            session_id=row['session_id'],
                            user_id=row['user_id'],
                            start_time=datetime.fromisoformat(row['start_time']),
                            end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None,
                            total_interactions=row['total_interactions'],
                            successful_interactions=row['successful_interactions'],
                            failed_interactions=row['failed_interactions'],
                            tools_used=json.loads(row['tools_used']) if row['tools_used'] else [],
                            average_response_time=row['average_response_time'],
                            user_satisfaction=row['user_satisfaction']
                        )
                    return None

        except Exception as e:
            logger.error(f"Failed to get user session {session_id}: {e}")
            return None

    def _row_to_feedback(self, row: aiosqlite.Row) -> FeedbackModel:
        """데이터베이스 행을 FeedbackModel로 변환"""
        context = None
        if any([row['step_id'], row['tool_name'], row['action_type']]):
            context = FeedbackContext(
                session_id=row['session_id'],
                step_id=row['step_id'],
                tool_name=row['tool_name'],
                action_type=row['action_type'],
                execution_time=row['execution_time'],
                success=row['success'],
                error_message=row['error_message'],
                user_query=row['user_query'],
                system_response=row['system_response']
            )

        return FeedbackModel(
            id=row['id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            feedback_type=FeedbackType(row['feedback_type']),
            status=FeedbackStatus(row['status']),
            rating=row['rating'],
            text_feedback=row['text_feedback'],
            sentiment_score=SentimentScore(row['sentiment_score']) if row['sentiment_score'] is not None else None,
            context=context,
            user_id=row['user_id'],
            session_id=row['session_id'],
            ip_address=row['ip_address'],
            user_agent=row['user_agent'],
            reviewed_by=row['reviewed_by'],
            reviewed_at=datetime.fromisoformat(row['reviewed_at']) if row['reviewed_at'] else None,
            resolution_notes=row['resolution_notes'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            tags=json.loads(row['tags']) if row['tags'] else []
        )

    async def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """오래된 데이터 정리"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            async with aiosqlite.connect(self.db_path) as db:
                # 오래된 피드백 삭제
                async with db.execute(
                    "DELETE FROM feedback WHERE timestamp < ?",
                    (cutoff_date.isoformat(),)
                ) as cursor:
                    deleted_count = cursor.rowcount

                # 오래된 세션 삭제
                await db.execute(
                    "DELETE FROM user_sessions WHERE start_time < ?",
                    (cutoff_date.isoformat(),)
                )

                await db.commit()
                logger.info(f"Cleaned up {deleted_count} old feedback records")
                return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0