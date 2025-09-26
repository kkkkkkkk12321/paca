"""
ReAct (Reason + Act) 프레임워크 구현

생각(Reason) + 행동(Act) 프레임워크로 AI의 의사결정 과정을 체계화합니다.
"""

from typing import Any, Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import asyncio
from .base import Tool, ToolResult, ToolManager


class ReActStepType(Enum):
    """ReAct 단계 유형"""
    THOUGHT = "thought"      # 생각 단계
    ACTION = "action"        # 행동 단계
    OBSERVATION = "observation"  # 관찰 단계
    REFLECTION = "reflection"    # 성찰 단계


class ReActSessionStatus(Enum):
    """ReAct 세션 상태"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ReActStep:
    """ReAct 단일 단계"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_type: ReActStepType = ReActStepType.THOUGHT
    content: str = ""
    tool_name: Optional[str] = None
    tool_params: Dict[str, Any] = field(default_factory=dict)
    result: Optional[ToolResult] = None
    confidence: float = 0.0
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """단계를 딕셔너리로 변환"""
        return {
            'id': self.id,
            'step_type': self.step_type.value,
            'content': self.content,
            'tool_name': self.tool_name,
            'tool_params': self.tool_params,
            'result': self.result.to_dict() if self.result else None,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ReActSession:
    """ReAct 실행 세션"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal: str = ""
    status: ReActSessionStatus = ReActSessionStatus.CREATED
    steps: List[ReActStep] = field(default_factory=list)
    max_steps: int = 20
    timeout_seconds: float = 300.0
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    final_result: Any = None

    @property
    def session_id(self) -> str:
        """호환성을 위한 session_id 속성"""
        return self.id

    def add_step(self, step: ReActStep):
        """단계 추가"""
        self.steps.append(step)

    def get_last_step(self) -> Optional[ReActStep]:
        """마지막 단계 가져오기"""
        return self.steps[-1] if self.steps else None

    def get_steps_by_type(self, step_type: ReActStepType) -> List[ReActStep]:
        """특정 유형의 단계들 가져오기"""
        return [step for step in self.steps if step.step_type == step_type]

    def to_dict(self) -> Dict[str, Any]:
        """세션을 딕셔너리로 변환"""
        return {
            'id': self.id,
            'goal': self.goal,
            'status': self.status.value,
            'steps': [step.to_dict() for step in self.steps],
            'max_steps': self.max_steps,
            'timeout_seconds': self.timeout_seconds,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error': self.error,
            'final_result': self.final_result
        }


class ReActFramework:
    """ReAct 프레임워크 메인 클래스"""

    def __init__(self, tool_manager: ToolManager):
        self.tool_manager = tool_manager
        self.active_sessions: Dict[str, ReActSession] = {}
        self.completed_sessions: List[ReActSession] = []

    async def create_session(self, goal: str, max_steps: int = 20,
                           timeout_seconds: float = 300.0) -> ReActSession:
        """새 ReAct 세션 생성"""
        session = ReActSession(
            goal=goal,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds
        )
        self.active_sessions[session.id] = session
        return session

    async def think(self, session: ReActSession, thought: str,
                   confidence: float = 0.5, reasoning: str = "") -> ReActStep:
        """생각 단계 실행"""
        step = ReActStep(
            step_type=ReActStepType.THOUGHT,
            content=thought,
            confidence=confidence,
            reasoning=reasoning
        )
        session.add_step(step)
        return step

    async def act(self, session: ReActSession, tool_name: str,
                 **tool_params) -> ReActStep:
        """행동 단계 실행"""
        step = ReActStep(
            step_type=ReActStepType.ACTION,
            tool_name=tool_name,
            tool_params=tool_params,
            content=f"도구 '{tool_name}' 실행"
        )

        try:
            # 도구 실행
            result = await self.tool_manager.execute_tool(tool_name, **tool_params)
            step.result = result
            step.confidence = 1.0 if result.success else 0.0
        except Exception as e:
            step.result = ToolResult(success=False, error=str(e))
            step.confidence = 0.0

        session.add_step(step)
        return step

    async def observe(self, session: ReActSession, observation: str,
                     confidence: float = 0.7) -> ReActStep:
        """관찰 단계 실행"""
        step = ReActStep(
            step_type=ReActStepType.OBSERVATION,
            content=observation,
            confidence=confidence
        )
        session.add_step(step)
        return step

    async def reflect(self, session: ReActSession, reflection: str,
                     confidence: float = 0.6) -> ReActStep:
        """성찰 단계 실행"""
        step = ReActStep(
            step_type=ReActStepType.REFLECTION,
            content=reflection,
            confidence=confidence
        )
        session.add_step(step)
        return step

    async def run_session(self, session_id: str) -> ReActSession:
        """세션 실행 (자동 ReAct 루프)"""
        if session_id not in self.active_sessions:
            raise ValueError(f"세션 {session_id}를 찾을 수 없습니다.")

        session = self.active_sessions[session_id]
        session.status = ReActSessionStatus.RUNNING

        start_time = datetime.now()

        try:
            # 자동 ReAct 루프 실행
            for step_count in range(session.max_steps):
                # 타임아웃 체크
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > session.timeout_seconds:
                    session.status = ReActSessionStatus.TIMEOUT
                    session.error = "세션이 타임아웃되었습니다."
                    break

                # 목표 달성 여부 확인 (단순한 휴리스틱)
                if self._is_goal_achieved(session):
                    session.status = ReActSessionStatus.COMPLETED
                    session.final_result = self._extract_final_result(session)
                    break

                # 다음 단계 결정 및 실행
                await self._execute_next_step(session)

                # 너무 많은 오류가 발생한 경우 중단
                if self._should_stop_session(session):
                    break

            # 세션 완료 처리
            if session.status == ReActSessionStatus.RUNNING:
                session.status = ReActSessionStatus.COMPLETED
                session.final_result = self._extract_final_result(session)

        except Exception as e:
            session.status = ReActSessionStatus.FAILED
            session.error = str(e)

        finally:
            session.completed_at = datetime.now()
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            self.completed_sessions.append(session)

        return session

    async def _execute_next_step(self, session: ReActSession):
        """다음 단계 실행 결정"""
        last_step = session.get_last_step()

        if not last_step or last_step.step_type == ReActStepType.REFLECTION:
            # 처음이거나 성찰 후면 생각 단계
            await self.think(
                session,
                f"목표 '{session.goal}'를 위해 다음에 무엇을 해야 할까요?",
                confidence=0.7
            )
        elif last_step.step_type == ReActStepType.THOUGHT:
            # 생각 후 행동 단계 - 간단한 예시
            if "검색" in last_step.content or "찾" in last_step.content:
                await self.act(session, "web_search", query=session.goal)
            else:
                await self.act(session, "file_read", path="example.txt")
        elif last_step.step_type == ReActStepType.ACTION:
            # 행동 후 관찰 단계
            result = last_step.result
            if result and result.success:
                await self.observe(
                    session,
                    f"도구 실행 성공: {result.data}",
                    confidence=0.8
                )
            else:
                await self.observe(
                    session,
                    f"도구 실행 실패: {result.error if result else '알 수 없는 오류'}",
                    confidence=0.3
                )
        elif last_step.step_type == ReActStepType.OBSERVATION:
            # 관찰 후 성찰 단계
            await self.reflect(
                session,
                f"현재까지의 진행 상황을 종합해보면...",
                confidence=0.6
            )

    def _is_goal_achieved(self, session: ReActSession) -> bool:
        """목표 달성 여부 확인 (단순한 휴리스틱)"""
        # 성공적인 액션이 3개 이상 있으면 달성으로 간주
        successful_actions = [
            step for step in session.steps
            if step.step_type == ReActStepType.ACTION
            and step.result and step.result.success
        ]
        return len(successful_actions) >= 1

    def _should_stop_session(self, session: ReActSession) -> bool:
        """세션 중단 여부 결정"""
        # 연속된 실패가 5개 이상이면 중단
        recent_steps = session.steps[-5:] if len(session.steps) >= 5 else session.steps
        failed_actions = [
            step for step in recent_steps
            if step.step_type == ReActStepType.ACTION
            and step.result and not step.result.success
        ]
        return len(failed_actions) >= 3

    def _extract_final_result(self, session: ReActSession) -> Any:
        """최종 결과 추출"""
        successful_actions = [
            step for step in session.steps
            if step.step_type == ReActStepType.ACTION
            and step.result and step.result.success
        ]
        if successful_actions:
            return successful_actions[-1].result.data
        return None

    def get_session(self, session_id: str) -> Optional[ReActSession]:
        """세션 가져오기"""
        return self.active_sessions.get(session_id)

    def list_sessions(self, status: Optional[ReActSessionStatus] = None) -> List[ReActSession]:
        """세션 목록"""
        all_sessions = list(self.active_sessions.values()) + self.completed_sessions
        if status:
            return [s for s in all_sessions if s.status == status]
        return all_sessions

    async def cancel_session(self, session_id: str) -> bool:
        """세션 취소"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.status = ReActSessionStatus.CANCELLED
            session.completed_at = datetime.now()
            del self.active_sessions[session_id]
            self.completed_sessions.append(session)
            return True
        return False