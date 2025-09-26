"""
피드백 수집기
자동 및 수동 피드백 수집을 위한 컴포넌트
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from .models import (
    FeedbackModel, FeedbackType, FeedbackContext,
    UserSession, SentimentScore, CreateFeedbackRequest
)
from .storage import FeedbackStorage

logger = logging.getLogger(__name__)


@dataclass
class FeedbackTrigger:
    """피드백 트리거 조건"""
    name: str
    condition: Callable[..., bool]
    feedback_type: FeedbackType
    priority: int = 1  # 1=highest, 5=lowest
    enabled: bool = True


class FeedbackCollector:
    """피드백 수집기 클래스"""

    def __init__(self, storage: FeedbackStorage):
        """
        초기화

        Args:
            storage: 피드백 저장소
        """
        self.storage = storage
        self.triggers: List[FeedbackTrigger] = []
        self.sessions: Dict[str, UserSession] = {}
        self.auto_feedback_enabled = True
        self._setup_default_triggers()

    def _setup_default_triggers(self):
        """기본 피드백 트리거 설정"""

        # 도구 실행 실패 시 자동 피드백
        self.add_trigger(
            name="tool_execution_failure",
            condition=lambda context: (
                context.get('success') is False and
                context.get('error_message') is not None
            ),
            feedback_type=FeedbackType.TOOL_EXECUTION,
            priority=1
        )

        # 응답 시간이 너무 긴 경우
        self.add_trigger(
            name="slow_response",
            condition=lambda context: (
                context.get('execution_time', 0) > 30.0  # 30초 초과
            ),
            feedback_type=FeedbackType.PERFORMANCE,
            priority=2
        )

        # 연속 실패 감지
        self.add_trigger(
            name="consecutive_failures",
            condition=lambda context: (
                self._get_session_failure_streak(context.get('session_id', '')) >= 3
            ),
            feedback_type=FeedbackType.USER_EXPERIENCE,
            priority=1
        )

    def add_trigger(
        self,
        name: str,
        condition: Callable[..., bool],
        feedback_type: FeedbackType,
        priority: int = 3,
        enabled: bool = True
    ):
        """피드백 트리거 추가"""
        trigger = FeedbackTrigger(
            name=name,
            condition=condition,
            feedback_type=feedback_type,
            priority=priority,
            enabled=enabled
        )
        self.triggers.append(trigger)
        logger.info(f"Added feedback trigger: {name}")

    async def collect_automatic_feedback(
        self,
        session_id: str,
        context: Dict[str, Any]
    ) -> List[FeedbackModel]:
        """자동 피드백 수집"""
        if not self.auto_feedback_enabled:
            return []

        collected_feedback = []

        try:
            # 세션 업데이트
            await self._update_session(session_id, context)

            # 트리거 확인
            for trigger in self.triggers:
                if not trigger.enabled:
                    continue

                try:
                    if trigger.condition(context):
                        feedback = await self._create_automatic_feedback(
                            session_id, context, trigger
                        )
                        if feedback:
                            collected_feedback.append(feedback)
                            logger.info(f"Auto feedback triggered: {trigger.name}")

                except Exception as e:
                    logger.error(f"Error in trigger {trigger.name}: {e}")

            return collected_feedback

        except Exception as e:
            logger.error(f"Failed to collect automatic feedback: {e}")
            return []

    async def collect_manual_feedback(
        self,
        request: CreateFeedbackRequest,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Optional[FeedbackModel]:
        """수동 피드백 수집"""
        try:
            # 컨텍스트 생성
            context = None
            if request.context:
                context = FeedbackContext(
                    session_id=request.session_id,
                    **request.context
                )

            # 감정 분석 (간단한 규칙 기반)
            sentiment = self._analyze_sentiment(request.text_feedback)

            # 피드백 모델 생성
            feedback = FeedbackModel(
                feedback_type=request.feedback_type,
                rating=request.rating,
                text_feedback=request.text_feedback,
                sentiment_score=sentiment,
                context=context,
                session_id=request.session_id,
                tags=request.tags,
                ip_address=user_context.get('ip_address') if user_context else None,
                user_agent=user_context.get('user_agent') if user_context else None,
                user_id=user_context.get('user_id') if user_context else None
            )

            # 저장
            success = await self.storage.save_feedback(feedback)
            if success:
                logger.info(f"Manual feedback collected: {feedback.id}")
                return feedback
            else:
                logger.error("Failed to save manual feedback")
                return None

        except Exception as e:
            logger.error(f"Failed to collect manual feedback: {e}")
            return None

    async def _create_automatic_feedback(
        self,
        session_id: str,
        context: Dict[str, Any],
        trigger: FeedbackTrigger
    ) -> Optional[FeedbackModel]:
        """자동 피드백 생성"""
        try:
            # 컨텍스트 정보 추출
            feedback_context = FeedbackContext(
                session_id=session_id,
                step_id=context.get('step_id'),
                tool_name=context.get('tool_name'),
                action_type=context.get('action_type'),
                execution_time=context.get('execution_time'),
                success=context.get('success'),
                error_message=context.get('error_message'),
                user_query=context.get('user_query'),
                system_response=context.get('system_response')
            )

            # 자동 생성된 피드백 텍스트
            text_feedback = self._generate_auto_feedback_text(trigger, context)

            # 자동 평점 (실패 시 낮은 점수)
            auto_rating = self._calculate_auto_rating(trigger, context)

            feedback = FeedbackModel(
                feedback_type=trigger.feedback_type,
                rating=auto_rating,
                text_feedback=text_feedback,
                context=feedback_context,
                session_id=session_id,
                tags=[f"auto:{trigger.name}", "system_generated"],
                metadata={
                    "trigger_name": trigger.name,
                    "trigger_priority": trigger.priority,
                    "auto_generated": True
                }
            )

            # 저장
            success = await self.storage.save_feedback(feedback)
            if success:
                return feedback
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to create automatic feedback: {e}")
            return None

    def _generate_auto_feedback_text(
        self,
        trigger: FeedbackTrigger,
        context: Dict[str, Any]
    ) -> str:
        """자동 피드백 텍스트 생성"""
        if trigger.name == "tool_execution_failure":
            tool_name = context.get('tool_name', 'Unknown')
            error_msg = context.get('error_message', 'Unknown error')
            return f"Tool '{tool_name}' execution failed: {error_msg}"

        elif trigger.name == "slow_response":
            exec_time = context.get('execution_time', 0)
            tool_name = context.get('tool_name', 'System')
            return f"'{tool_name}' took {exec_time:.2f} seconds to respond (exceeds 30s threshold)"

        elif trigger.name == "consecutive_failures":
            streak = self._get_session_failure_streak(context.get('session_id', ''))
            return f"User experienced {streak} consecutive failures in this session"

        else:
            return f"Automatic feedback triggered by: {trigger.name}"

    def _calculate_auto_rating(
        self,
        trigger: FeedbackTrigger,
        context: Dict[str, Any]
    ) -> int:
        """자동 평점 계산"""
        if trigger.feedback_type == FeedbackType.TOOL_EXECUTION:
            return 1 if not context.get('success', True) else 4

        elif trigger.feedback_type == FeedbackType.PERFORMANCE:
            exec_time = context.get('execution_time', 0)
            if exec_time > 60:
                return 1
            elif exec_time > 30:
                return 2
            else:
                return 3

        elif trigger.feedback_type == FeedbackType.USER_EXPERIENCE:
            streak = self._get_session_failure_streak(context.get('session_id', ''))
            return max(1, 5 - streak)

        else:
            return 3  # 중립

    def _analyze_sentiment(self, text: Optional[str]) -> Optional[SentimentScore]:
        """간단한 감정 분석"""
        if not text:
            return None

        text_lower = text.lower()

        # 부정적 키워드
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'failed', 'error', 'broken',
            'slow', 'confusing', 'frustrating', 'annoying', 'useless',
            '나쁜', '끔찍한', '실패', '오류', '느린', '짜증', '쓸모없는'
        ]

        # 긍정적 키워드
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'perfect', 'fast', 'helpful',
            'useful', 'clear', 'easy', 'smooth', 'wonderful',
            '좋은', '훌륭한', '완벽한', '빠른', '유용한', '명확한', '쉬운'
        ]

        negative_count = sum(1 for word in negative_words if word in text_lower)
        positive_count = sum(1 for word in positive_words if word in text_lower)

        if negative_count > positive_count + 1:
            return SentimentScore.NEGATIVE
        elif positive_count > negative_count + 1:
            return SentimentScore.POSITIVE
        else:
            return SentimentScore.NEUTRAL

    async def _update_session(self, session_id: str, context: Dict[str, Any]):
        """세션 정보 업데이트"""
        try:
            # 기존 세션 조회 또는 새 세션 생성
            if session_id not in self.sessions:
                existing_session = await self.storage.get_user_session(session_id)
                if existing_session:
                    self.sessions[session_id] = existing_session
                else:
                    self.sessions[session_id] = UserSession(
                        session_id=session_id,
                        user_id=context.get('user_id')
                    )

            session = self.sessions[session_id]

            # 인터랙션 카운트 업데이트
            session.total_interactions += 1

            if context.get('success', True):
                session.successful_interactions += 1
            else:
                session.failed_interactions += 1

            # 사용된 도구 추가
            tool_name = context.get('tool_name')
            if tool_name and tool_name not in session.tools_used:
                session.tools_used.append(tool_name)

            # 응답 시간 업데이트
            exec_time = context.get('execution_time')
            if exec_time is not None:
                if session.average_response_time == 0:
                    session.average_response_time = exec_time
                else:
                    # 이동 평균
                    session.average_response_time = (
                        session.average_response_time * 0.7 + exec_time * 0.3
                    )

            # 세션 저장
            await self.storage.save_user_session(session)

        except Exception as e:
            logger.error(f"Failed to update session {session_id}: {e}")

    def _get_session_failure_streak(self, session_id: str) -> int:
        """세션의 연속 실패 횟수 조회"""
        if session_id not in self.sessions:
            return 0

        session = self.sessions[session_id]
        # 최근 실패 패턴 분석 (간단한 구현)
        if session.total_interactions < 3:
            return 0

        failure_rate = session.failed_interactions / session.total_interactions
        if failure_rate > 0.7:  # 70% 이상 실패
            return session.failed_interactions

        return 0

    async def end_session(self, session_id: str, user_satisfaction: Optional[float] = None):
        """세션 종료"""
        try:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                session.end_time = datetime.now()
                if user_satisfaction is not None:
                    session.user_satisfaction = user_satisfaction

                await self.storage.save_user_session(session)
                logger.info(f"Session ended: {session_id}")

        except Exception as e:
            logger.error(f"Failed to end session {session_id}: {e}")

    def enable_auto_feedback(self, enabled: bool = True):
        """자동 피드백 활성화/비활성화"""
        self.auto_feedback_enabled = enabled
        logger.info(f"Auto feedback {'enabled' if enabled else 'disabled'}")

    def enable_trigger(self, trigger_name: str, enabled: bool = True):
        """특정 트리거 활성화/비활성화"""
        for trigger in self.triggers:
            if trigger.name == trigger_name:
                trigger.enabled = enabled
                logger.info(f"Trigger '{trigger_name}' {'enabled' if enabled else 'disabled'}")
                return

        logger.warning(f"Trigger '{trigger_name}' not found")

    def get_trigger_status(self) -> Dict[str, Dict[str, Any]]:
        """트리거 상태 조회"""
        return {
            trigger.name: {
                "enabled": trigger.enabled,
                "feedback_type": trigger.feedback_type.value,
                "priority": trigger.priority
            }
            for trigger in self.triggers
        }