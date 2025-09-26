"""
Learning Service Module
학습 세션 관리 및 진행 상황 추적 서비스
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field

from ..core.types.base import ID, Result, BaseEntity, Priority
from ..core.errors.base import ValidationError
from ..core.utils.async_utils import create_logger
from ..core.events.base import EventEmitter, Event
from .base import BaseService, ServiceConfig, ServiceContext, ServiceResult, ServicePriority


class LearningSessionState(Enum):
    """학습 세션 상태"""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class SessionType(Enum):
    """세션 유형"""
    PRACTICE = "practice"
    TEST = "test"
    REVIEW = "review"
    ADAPTIVE = "adaptive"


class SessionStatus(Enum):
    """세션 상태"""
    CREATED = "created"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class LearningAction(Enum):
    """학습 행동"""
    READ = "read"
    PRACTICE = "practice"
    TEST = "test"
    REVIEW = "review"


class LearningResult(Enum):
    """학습 결과"""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    SKIPPED = "skipped"
    PARTIAL = "partial"


@dataclass
class LearningQuestion:
    """학습 질문"""
    id: ID
    content: str
    type: str  # 'multiple-choice', 'text', 'code'
    options: Optional[List[str]] = None
    expected_answer: Optional[Any] = None
    difficulty: int = 1


@dataclass
class LearningAnswer:
    """학습 답변"""
    question_id: ID
    answer: Any
    timestamp: int
    correct: Optional[bool] = None
    explanation: Optional[str] = None


@dataclass
class LearningGoal:
    """학습 목표"""
    id: ID
    title: str
    description: str
    target_score: Optional[int] = None
    deadline: Optional[int] = None


@dataclass
class SessionProgress:
    """세션 진행 상황"""
    current_step: int = 0
    total_steps: int = 0
    completion_percentage: float = 0.0
    time_spent: int = 0
    correct_answers: int = 0
    incorrect_answers: int = 0
    skipped_questions: int = 0


@dataclass
class SessionResults:
    """세션 결과"""
    score: int = 0
    accuracy: float = 0.0
    speed: float = 0.0
    improvement: float = 0.0
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)


@dataclass
class SessionConfig:
    """세션 설정"""
    max_questions: int = 10
    time_limit: Optional[int] = None
    difficulty_level: str = "medium"
    adaptive_difficulty: bool = False
    show_hints: bool = True
    immediate_feedback: bool = True


@dataclass
class LearningSession:
    """학습 세션"""
    user_id: ID
    title: str
    type: SessionType
    status: SessionStatus
    started_at: int
    progress: SessionProgress
    results: SessionResults
    config: SessionConfig
    description: Optional[str] = None

    def __post_init__(self):
        if not hasattr(self, 'id') or not self.id:
            self.id = str(uuid.uuid4())
        if not hasattr(self, 'created_at') or not self.created_at:
            self.created_at = int(time.time() * 1000)
        if not hasattr(self, 'updated_at') or not self.updated_at:
            self.updated_at = self.created_at


@dataclass
class KnowledgeItem:
    """지식 항목"""
    title: str
    content: str
    category: str
    difficulty: int = 1
    tags: List[str] = field(default_factory=list)


@dataclass
class LearningRecord:
    """학습 기록"""
    user_id: ID
    session_id: ID
    knowledge_item_id: ID
    action: LearningAction
    result: LearningResult
    start_time: int
    end_time: int
    duration: int
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CreateSessionRequest:
    """학습 세션 생성 요청"""
    user_id: ID
    title: str
    type: SessionType
    config: SessionConfig
    description: Optional[str] = None
    knowledge_items: Optional[List[ID]] = None


@dataclass
class UpdateSessionRequest:
    """학습 세션 업데이트 요청"""
    title: Optional[str] = None
    description: Optional[str] = None
    config: Optional[SessionConfig] = None
    status: Optional[SessionStatus] = None


@dataclass
class SubmitAnswerRequest:
    """학습 답변 요청"""
    session_id: ID
    knowledge_item_id: ID
    user_answer: Any
    time_spent: int
    start_time: int
    end_time: int


@dataclass
class LearningProgress:
    """학습 진행 상황"""
    session_id: ID
    user_id: ID
    current_step: int
    total_steps: int
    completion_percentage: float
    time_spent: int
    accuracy: float
    streak: int
    recommended_next_items: List[ID] = field(default_factory=list)
    weak_areas: List[str] = field(default_factory=list)
    strong_areas: List[str] = field(default_factory=list)


@dataclass
class LearningStatistics:
    """학습 통계"""
    user_id: ID
    total_sessions: int
    completed_sessions: int
    total_time_spent: int
    average_accuracy: float
    improvement: float
    streak_days: int
    last_session_at: Optional[int] = None
    category_stats: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class LearningRecommendation:
    """학습 추천"""
    knowledge_items: List[KnowledgeItem]
    difficulty: str  # 'easy', 'medium', 'hard'
    estimated_time: int
    reason: str
    priority: int


class LearningService(BaseService):
    """학습 서비스"""

    def __init__(self, cognitive_system=None, events: Optional[EventEmitter] = None):
        config = ServiceConfig(
            id="learning-service",
            name="Learning Service",
            description="Learning session management and progress tracking service",
            version="1.0.0",
            priority=ServicePriority.HIGH,
            auto_start=True,
            max_retries=3,
            retry_delay=1000,
            timeout=60000,
            dependencies=["cognitive-service"],
            enable_metrics=True,
            enable_events=True
        )

        super().__init__(config, events)
        self.cognitive_system = cognitive_system
        self.active_sessions: Dict[ID, LearningSession] = {}
        self.user_progress: Dict[ID, LearningProgress] = {}
        self._stats_update_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """서비스 초기화"""
        # 진행 중인 세션 복구 로직
        await self._restore_active_sessions()

        # 주기적인 통계 업데이트 (5분마다)
        self._stats_update_task = asyncio.create_task(self._periodic_stats_update())

        self.logger.info("Learning service initialized")

    async def cleanup(self) -> None:
        """서비스 정리"""
        # 통계 업데이트 태스크 취소
        if self._stats_update_task:
            self._stats_update_task.cancel()
            try:
                await self._stats_update_task
            except asyncio.CancelledError:
                pass

        # 활성 세션 저장
        await self._save_active_sessions()

        self.active_sessions.clear()
        self.user_progress.clear()
        self.logger.info("Learning service cleaned up")

    async def _periodic_stats_update(self):
        """주기적 통계 업데이트"""
        while True:
            try:
                await asyncio.sleep(300)  # 5분마다
                await self._update_learning_statistics()
            except asyncio.CancelledError:
                break
            except Exception as error:
                self.logger.error(f"Statistics update failed: {error}")

    async def perform_execute(self, context: ServiceContext, *args) -> ServiceResult:
        """액션 실행"""
        if not args:
            return self.create_error_result(
                context, "INVALID_ACTION", "No action specified", False
            )

        action = args[0]
        action_args = args[1:] if len(args) > 1 else []

        handlers = {
            "createSession": self._handle_create_session,
            "getSession": self._handle_get_session,
            "updateSession": self._handle_update_session,
            "deleteSession": self._handle_delete_session,
            "startSession": self._handle_start_session,
            "pauseSession": self._handle_pause_session,
            "resumeSession": self._handle_resume_session,
            "completeSession": self._handle_complete_session,
            "submitAnswer": self._handle_submit_answer,
            "getProgress": self._handle_get_progress,
            "getStatistics": self._handle_get_statistics,
            "getRecommendations": self._handle_get_recommendations,
            "getUserSessions": self._handle_get_user_sessions,
        }

        handler = handlers.get(action)
        if not handler:
            return self.create_error_result(
                context, "INVALID_ACTION", f"Unknown action: {action}", False
            )

        try:
            return await handler(context, *action_args)
        except Exception as error:
            self.logger.error(f"Action {action} failed: {error}")
            return self.create_error_result(
                context, "EXECUTION_ERROR", f"Failed to execute {action}", True
            )

    async def _handle_create_session(
        self, context: ServiceContext, request: CreateSessionRequest
    ) -> ServiceResult:
        """학습 세션 생성"""
        try:
            # 입력 검증
            validation_result = self._validate_create_session_request(request)
            if not validation_result.success:
                return self.create_error_result(
                    context, "VALIDATION_ERROR", validation_result.error.message, False
                )

            session_id = str(uuid.uuid4())
            now = int(time.time() * 1000)

            session = LearningSession(
                id=session_id,
                user_id=request.user_id,
                title=request.title,
                description=request.description,
                type=request.type,
                status=SessionStatus.CREATED,
                started_at=now,
                progress=SessionProgress(
                    current_step=0,
                    total_steps=len(request.knowledge_items) if request.knowledge_items else 0,
                    completion_percentage=0.0,
                    time_spent=0,
                    correct_answers=0,
                    incorrect_answers=0,
                    skipped_questions=0
                ),
                results=SessionResults(),
                config=request.config,
                created_at=now,
                updated_at=now
            )

            self.active_sessions[session_id] = session

            await self.emit_event("learning.session.created", {
                "session_id": session_id,
                "user_id": request.user_id,
                "type": request.type.value
            })

            return self.create_success_result(
                context, session, ["validation", "session_creation", "notification"]
            )

        except Exception as error:
            self.logger.error(f"Session creation failed: {error}")
            return self.create_error_result(
                context, "SESSION_CREATION_ERROR", "Failed to create learning session", True
            )

    async def _handle_get_session(
        self, context: ServiceContext, session_id: ID
    ) -> ServiceResult:
        """학습 세션 조회"""
        try:
            session = self.active_sessions.get(session_id)
            return self.create_success_result(context, session, ["session_lookup"])

        except Exception as error:
            self.logger.error(f"Session retrieval failed: {error}")
            return self.create_error_result(
                context, "SESSION_RETRIEVAL_ERROR", "Failed to retrieve learning session", True
            )

    async def _handle_start_session(
        self, context: ServiceContext, session_id: ID
    ) -> ServiceResult:
        """학습 세션 시작"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return self.create_error_result(
                    context, "SESSION_NOT_FOUND", "Learning session not found", False
                )

            if session.status not in [SessionStatus.CREATED, SessionStatus.PAUSED]:
                return self.create_error_result(
                    context,
                    "INVALID_SESSION_STATE",
                    f"Cannot start session in {session.status.value} state",
                    False
                )

            # 세션 상태 업데이트
            session.status = SessionStatus.IN_PROGRESS
            if session.status == SessionStatus.CREATED:
                session.started_at = int(time.time() * 1000)
            session.updated_at = int(time.time() * 1000)

            # 다음 아이템 가져오기
            next_item = await self._get_next_knowledge_item(session)

            # 사용자 진행 상황 업데이트
            await self._update_user_progress(session)

            await self.emit_event("learning.session.started", {
                "session_id": session_id,
                "user_id": session.user_id
            })

            return self.create_success_result(
                context,
                {"session": session, "next_item": next_item},
                ["session_validation", "session_start", "progress_update"]
            )

        except Exception as error:
            self.logger.error(f"Session start failed: {error}")
            return self.create_error_result(
                context, "SESSION_START_ERROR", "Failed to start learning session", True
            )

    async def _handle_submit_answer(
        self, context: ServiceContext, request: SubmitAnswerRequest
    ) -> ServiceResult:
        """답변 제출 처리"""
        try:
            session = self.active_sessions.get(request.session_id)
            if not session:
                return self.create_error_result(
                    context, "SESSION_NOT_FOUND", "Learning session not found", False
                )

            if session.status != SessionStatus.IN_PROGRESS:
                return self.create_error_result(
                    context, "SESSION_NOT_ACTIVE", "Session is not active", False
                )

            # 인지 시스템을 통한 답변 평가 (있다면)
            is_correct = False
            score = 0
            feedback = "Answer processed"

            if self.cognitive_system:
                try:
                    evaluation_context = {
                        "session_id": request.session_id,
                        "knowledge_item_id": request.knowledge_item_id,
                        "user_answer": request.user_answer,
                        "time_spent": request.time_spent
                    }

                    # 인지 시스템 분석 (간단한 mock 구현)
                    cognitive_result = await self._evaluate_answer_with_cognitive_system(
                        evaluation_context
                    )

                    is_correct = cognitive_result.get("correct", False)
                    score = cognitive_result.get("score", 0)
                    feedback = cognitive_result.get("feedback", "Answer processed")

                except Exception as cognitive_error:
                    self.logger.warning(f"Cognitive evaluation failed: {cognitive_error}")
                    # 기본값 사용

            # 세션 진행 상황 업데이트
            session.progress.current_step += 1
            session.progress.time_spent += request.time_spent

            if is_correct:
                session.progress.correct_answers += 1
            else:
                session.progress.incorrect_answers += 1

            session.progress.completion_percentage = (
                session.progress.current_step / max(1, session.progress.total_steps)
            ) * 100

            session.updated_at = int(time.time() * 1000)

            # 학습 기록 저장
            learning_record = LearningRecord(
                id=str(uuid.uuid4()),
                user_id=session.user_id,
                session_id=request.session_id,
                knowledge_item_id=request.knowledge_item_id,
                action=LearningAction.TEST,
                result=LearningResult.CORRECT if is_correct else LearningResult.INCORRECT,
                start_time=request.start_time,
                end_time=request.end_time,
                duration=request.time_spent,
                context={
                    "device": "web",
                    "session_settings": session.config.__dict__,
                    "environment_factors": {},
                    "cognitive_state": "focused"
                },
                created_at=int(time.time() * 1000),
                updated_at=int(time.time() * 1000)
            )

            # 다음 아이템 가져오기
            next_item = None
            if session.progress.current_step < session.progress.total_steps:
                next_item = await self._get_next_knowledge_item(session)

            # 세션 완료 확인
            if session.progress.current_step >= session.progress.total_steps:
                await self._handle_complete_session(context, request.session_id)

            await self.emit_event("learning.answer.submitted", {
                "session_id": request.session_id,
                "user_id": session.user_id,
                "correct": is_correct,
                "score": score,
                "progress": session.progress.completion_percentage
            })

            return self.create_success_result(
                context,
                {
                    "correct": is_correct,
                    "score": score,
                    "feedback": feedback,
                    "next_item": next_item,
                    "progress": session.progress
                },
                ["answer_evaluation", "progress_update", "record_creation"]
            )

        except Exception as error:
            self.logger.error(f"Answer submission failed: {error}")
            return self.create_error_result(
                context, "ANSWER_SUBMISSION_ERROR", "Failed to process answer submission", True
            )

    async def _handle_get_progress(
        self, context: ServiceContext, user_id: ID
    ) -> ServiceResult:
        """학습 진행 상황 조회"""
        try:
            progress = self.user_progress.get(user_id)
            return self.create_success_result(context, progress, ["progress_lookup"])

        except Exception as error:
            self.logger.error(f"Progress retrieval failed: {error}")
            return self.create_error_result(
                context, "PROGRESS_RETRIEVAL_ERROR", "Failed to retrieve learning progress", True
            )

    async def _handle_get_statistics(
        self, context: ServiceContext, user_id: ID
    ) -> ServiceResult:
        """학습 통계 조회"""
        try:
            # 사용자의 모든 세션 분석
            user_sessions = [
                session for session in self.active_sessions.values()
                if session.user_id == user_id
            ]

            completed_sessions = [
                session for session in user_sessions
                if session.status == SessionStatus.COMPLETED
            ]

            total_time_spent = sum(session.progress.time_spent for session in user_sessions)
            total_correct = sum(session.progress.correct_answers for session in user_sessions)
            total_answers = sum(
                session.progress.correct_answers + session.progress.incorrect_answers
                for session in user_sessions
            )

            average_accuracy = total_correct / max(1, total_answers)

            # 최근 세션 찾기
            last_session = None
            if user_sessions:
                last_session = max(user_sessions, key=lambda s: s.updated_at)

            statistics = LearningStatistics(
                user_id=user_id,
                total_sessions=len(user_sessions),
                completed_sessions=len(completed_sessions),
                total_time_spent=total_time_spent,
                average_accuracy=average_accuracy,
                improvement=0.0,  # 실제 구현에서는 시간별 개선도 계산
                streak_days=0,  # 실제 구현에서는 연속 학습일 계산
                last_session_at=last_session.updated_at if last_session else None,
                category_stats=[]  # 실제 구현에서는 카테고리별 통계 계산
            )

            return self.create_success_result(
                context, statistics, ["session_analysis", "statistics_calculation"]
            )

        except Exception as error:
            self.logger.error(f"Statistics retrieval failed: {error}")
            return self.create_error_result(
                context, "STATISTICS_RETRIEVAL_ERROR", "Failed to retrieve learning statistics", True
            )

    async def _handle_get_recommendations(
        self, context: ServiceContext, user_id: ID
    ) -> ServiceResult:
        """학습 추천 조회"""
        try:
            # 사용자 진행 상황 분석
            progress = self.user_progress.get(user_id)
            statistics_result = await self._handle_get_statistics(context, user_id)

            recommendations = []

            if statistics_result.success and statistics_result.data:
                # 인지 시스템을 통한 추천 생성 (있다면)
                if self.cognitive_system:
                    try:
                        recommendation_context = {
                            "progress": progress,
                            "statistics": statistics_result.data,
                            "action": "generate_recommendations"
                        }

                        cognitive_result = await self._generate_recommendations_with_cognitive_system(
                            recommendation_context
                        )

                        recommendations = cognitive_result.get("recommendations", [])

                    except Exception as cognitive_error:
                        self.logger.warning(f"Cognitive recommendations failed: {cognitive_error}")

                # 기본 추천
                if not recommendations:
                    recommendations = [
                        LearningRecommendation(
                            knowledge_items=[],
                            difficulty="medium",
                            estimated_time=30,
                            reason="Continue with current learning path",
                            priority=1
                        )
                    ]

            return self.create_success_result(
                context,
                recommendations,
                ["progress_analysis", "cognitive_reasoning", "recommendation_generation"]
            )

        except Exception as error:
            self.logger.error(f"Recommendations retrieval failed: {error}")
            return self.create_error_result(
                context, "RECOMMENDATIONS_ERROR", "Failed to generate learning recommendations", True
            )

    # Helper methods
    def _validate_create_session_request(self, request: CreateSessionRequest) -> Result:
        """세션 생성 요청 검증"""
        if not request.user_id or not request.title or not request.type:
            return Result(
                success=False,
                data=None,
                error=ValidationError(
                    "User ID, title, and type are required",
                    context={"request": request}
                )
            )

        if not request.config:
            return Result(
                success=False,
                data=None,
                error=ValidationError(
                    "Session configuration is required",
                    context={"config": request.config}
                )
            )

        return Result(success=True, data=True)

    async def _get_next_knowledge_item(self, session: LearningSession) -> Optional[KnowledgeItem]:
        """다음 지식 항목 가져오기"""
        # 실제 구현에서는 세션 설정과 진행 상황을 고려하여 다음 아이템 선택
        # 여기서는 간단히 None 반환
        return None

    async def _update_user_progress(self, session: LearningSession) -> None:
        """사용자 진행 상황 업데이트"""
        total_answers = session.progress.correct_answers + session.progress.incorrect_answers
        accuracy = session.progress.correct_answers / max(1, total_answers)

        progress = LearningProgress(
            session_id=session.id,
            user_id=session.user_id,
            current_step=session.progress.current_step,
            total_steps=session.progress.total_steps,
            completion_percentage=session.progress.completion_percentage,
            time_spent=session.progress.time_spent,
            accuracy=accuracy,
            streak=0,  # 실제 구현에서는 연속 정답 계산
            recommended_next_items=[],
            weak_areas=[],
            strong_areas=[]
        )

        self.user_progress[session.user_id] = progress

    async def _restore_active_sessions(self) -> None:
        """활성 세션 복구"""
        # 실제 구현에서는 데이터베이스에서 진행 중인 세션 복구
        self.logger.info("Active sessions restored")

    async def _save_active_sessions(self) -> None:
        """활성 세션 저장"""
        # 실제 구현에서는 데이터베이스에 활성 세션 저장
        self.logger.info("Active sessions saved")

    async def _update_learning_statistics(self) -> None:
        """학습 통계 업데이트"""
        # 주기적인 통계 업데이트 로직
        self.logger.debug("Learning statistics updated")

    async def _evaluate_answer_with_cognitive_system(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """인지 시스템을 통한 답변 평가"""
        # 인지 시스템이 있다면 실제 평가 수행
        # 여기서는 간단한 mock 구현
        return {
            "correct": True,  # 임시로 항상 정답으로 처리
            "score": 80,
            "feedback": "Good answer!"
        }

    async def _generate_recommendations_with_cognitive_system(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """인지 시스템을 통한 추천 생성"""
        # 인지 시스템이 있다면 실제 추천 생성
        # 여기서는 간단한 mock 구현
        return {
            "recommendations": [
                LearningRecommendation(
                    knowledge_items=[],
                    difficulty="medium",
                    estimated_time=30,
                    reason="Based on your performance, continue with medium difficulty",
                    priority=1
                )
            ]
        }

    # 미구현 핸들러들 (향후 구현 예정)
    async def _handle_update_session(self, context: ServiceContext, session_id: ID, request: UpdateSessionRequest) -> ServiceResult:
        return self.create_error_result(context, "NOT_IMPLEMENTED", "Method not implemented", False)

    async def _handle_delete_session(self, context: ServiceContext, session_id: ID) -> ServiceResult:
        return self.create_error_result(context, "NOT_IMPLEMENTED", "Method not implemented", False)

    async def _handle_pause_session(self, context: ServiceContext, session_id: ID) -> ServiceResult:
        return self.create_error_result(context, "NOT_IMPLEMENTED", "Method not implemented", False)

    async def _handle_resume_session(self, context: ServiceContext, session_id: ID) -> ServiceResult:
        return self.create_error_result(context, "NOT_IMPLEMENTED", "Method not implemented", False)

    async def _handle_complete_session(self, context: ServiceContext, session_id: ID) -> ServiceResult:
        return self.create_error_result(context, "NOT_IMPLEMENTED", "Method not implemented", False)

    async def _handle_get_user_sessions(self, context: ServiceContext, user_id: ID, options: Any) -> ServiceResult:
        return self.create_error_result(context, "NOT_IMPLEMENTED", "Method not implemented", False)