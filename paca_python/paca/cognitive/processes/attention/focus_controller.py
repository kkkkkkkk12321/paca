"""
Focus Controller

집중도 제어 시스템으로, 특정 대상에 대한 주의 집중을 관리하고
집중도를 동적으로 조절합니다.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Callable, Union
from uuid import uuid4, UUID

from ...base import BaseCognitiveProcessor


class FocusLevel(Enum):
    """집중 수준 정의"""
    MINIMAL = 1     # 최소 집중 (20% 미만)
    LOW = 2         # 낮은 집중 (20-40%)
    MEDIUM = 3      # 중간 집중 (40-70%)
    HIGH = 4        # 높은 집중 (70-90%)
    MAXIMUM = 5     # 최대 집중 (90% 이상)


class FocusStrategy(Enum):
    """집중 전략"""
    SUSTAINED = auto()      # 지속적 집중
    SELECTIVE = auto()      # 선택적 집중
    DIVIDED = auto()        # 분산 집중
    ALTERNATING = auto()    # 교대 집중


@dataclass
class FocusTarget:
    """집중 대상 정의"""
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    type: str = "general"           # task, stimulus, concept 등
    importance: float = 0.5         # 중요도 (0.0-1.0)
    urgency: float = 0.5           # 긴급도 (0.0-1.0)
    complexity: float = 0.5        # 복잡도 (0.0-1.0)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class FocusResult:
    """집중 결과"""
    target_id: UUID
    achieved_level: FocusLevel
    duration_ms: float
    efficiency_score: float = 0.0   # 집중 효율성 (0.0-1.0)
    distraction_count: int = 0      # 방해 요소 개수
    quality_score: float = 0.0      # 집중 품질 점수
    success: bool = True


class FocusController(BaseCognitiveProcessor):
    """
    집중도 제어 시스템

    특정 대상에 대한 집중을 관리하고, 집중도를 동적으로 조절합니다.
    다양한 집중 전략을 지원하며, 방해 요소를 관리합니다.
    """

    def __init__(self, max_focus_targets: int = 3):
        super().__init__()
        self.max_focus_targets = max_focus_targets

        # 집중 상태 관리
        self._current_targets: Dict[UUID, FocusTarget] = {}
        self._focus_levels: Dict[UUID, float] = {}  # 0.0-1.0
        self._focus_strategy = FocusStrategy.SELECTIVE

        # 집중 메트릭
        self._total_focus_time = 0.0
        self._successful_focus_sessions = 0
        self._total_focus_sessions = 0
        self._distraction_events: List[Dict[str, Any]] = []

        # 적응적 제어
        self._baseline_focus_capacity = 1.0
        self._current_focus_capacity = 1.0
        self._fatigue_level = 0.0
        self._recovery_rate = 0.1

    async def initialize(self) -> bool:
        """집중 제어 시스템 초기화"""
        try:
            self.logger.info("Initializing Focus Controller...")

            # 백그라운드 모니터링 시작
            asyncio.create_task(self._monitor_focus_decay())
            asyncio.create_task(self._manage_fatigue())

            self.logger.info("Focus Controller initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Focus Controller: {e}")
            return False

    async def start_focus(self, target: FocusTarget,
                         desired_level: FocusLevel = FocusLevel.HIGH,
                         strategy: Optional[FocusStrategy] = None) -> bool:
        """
        특정 대상에 대한 집중 시작

        Args:
            target: 집중할 대상
            desired_level: 원하는 집중 수준
            strategy: 집중 전략

        Returns:
            집중 시작 성공 여부
        """
        try:
            # 집중 용량 확인
            if len(self._current_targets) >= self.max_focus_targets:
                if not await self._make_room_for_focus(target):
                    return False

            # 집중 전략 설정
            if strategy:
                self._focus_strategy = strategy

            # 초기 집중도 계산
            initial_focus = self._calculate_initial_focus(target, desired_level)

            # 집중 시작
            self._current_targets[target.id] = target
            self._focus_levels[target.id] = initial_focus

            # 집중 램프업 시작
            asyncio.create_task(self._ramp_up_focus(target.id, desired_level))

            self.logger.info(f"Started focus on target {target.name} "
                           f"with level {desired_level.name}")
            return True

        except Exception as e:
            self.logger.error(f"Error starting focus on {target.name}: {e}")
            return False

    async def adjust_focus(self, target_id: UUID,
                          new_level: FocusLevel) -> bool:
        """
        기존 집중 대상의 집중도 조정

        Args:
            target_id: 조정할 대상 ID
            new_level: 새로운 집중 수준

        Returns:
            조정 성공 여부
        """
        try:
            if target_id not in self._current_targets:
                return False

            target_focus = self._level_to_value(new_level)
            current_focus = self._focus_levels[target_id]

            # 점진적 조정
            await self._gradual_focus_adjustment(target_id, current_focus, target_focus)

            self.logger.info(f"Adjusted focus for target {target_id} to {new_level.name}")
            return True

        except Exception as e:
            self.logger.error(f"Error adjusting focus for {target_id}: {e}")
            return False

    async def stop_focus(self, target_id: UUID) -> Optional[FocusResult]:
        """
        특정 대상에 대한 집중 종료

        Args:
            target_id: 종료할 집중 대상 ID

        Returns:
            집중 결과
        """
        try:
            if target_id not in self._current_targets:
                return None

            target = self._current_targets[target_id]
            focus_duration = (time.time() - target.created_at) * 1000

            # 집중 결과 계산
            result = self._calculate_focus_result(target_id, focus_duration)

            # 집중 정리
            del self._current_targets[target_id]
            del self._focus_levels[target_id]

            # 통계 업데이트
            self._update_focus_statistics(result)

            self.logger.info(f"Stopped focus on target {target.name}")
            return result

        except Exception as e:
            self.logger.error(f"Error stopping focus for {target_id}: {e}")
            return None

    async def get_current_focus_state(self) -> Dict[str, Any]:
        """현재 집중 상태 조회"""
        total_focus = sum(self._focus_levels.values())

        return {
            "active_targets": len(self._current_targets),
            "total_focus_intensity": min(total_focus, 1.0),
            "focus_strategy": self._focus_strategy.name,
            "focus_capacity": self._current_focus_capacity,
            "fatigue_level": self._fatigue_level,
            "targets": [
                {
                    "id": str(target.id),
                    "name": target.name,
                    "focus_level": self._value_to_level(self._focus_levels[target.id]).name,
                    "focus_intensity": self._focus_levels[target.id]
                }
                for target in self._current_targets.values()
            ]
        }

    async def handle_distraction(self, distraction_info: Dict[str, Any]) -> bool:
        """
        방해 요소 처리

        Args:
            distraction_info: 방해 요소 정보

        Returns:
            방해 요소 처리 성공 여부
        """
        try:
            distraction_strength = distraction_info.get("strength", 0.5)
            distraction_type = distraction_info.get("type", "unknown")

            # 방해 요소 기록
            self._distraction_events.append({
                "timestamp": time.time(),
                "type": distraction_type,
                "strength": distraction_strength,
                "info": distraction_info
            })

            # 집중도 감소 적용
            focus_reduction = distraction_strength * 0.1  # 최대 10% 감소

            for target_id in self._focus_levels:
                self._focus_levels[target_id] = max(
                    0.0, self._focus_levels[target_id] - focus_reduction
                )

            # 피로도 증가
            self._fatigue_level = min(1.0, self._fatigue_level + distraction_strength * 0.05)

            self.logger.debug(f"Handled distraction: {distraction_type} "
                            f"(strength: {distraction_strength})")
            return True

        except Exception as e:
            self.logger.error(f"Error handling distraction: {e}")
            return False

    def _calculate_initial_focus(self, target: FocusTarget,
                               desired_level: FocusLevel) -> float:
        """초기 집중도 계산"""
        base_focus = self._level_to_value(desired_level)

        # 대상의 특성을 고려한 조정
        importance_factor = 0.8 + (target.importance * 0.4)  # 0.8-1.2
        urgency_factor = 0.9 + (target.urgency * 0.2)        # 0.9-1.1
        complexity_penalty = 1.0 - (target.complexity * 0.2)  # 0.8-1.0

        # 현재 용량과 피로도 고려
        capacity_factor = self._current_focus_capacity
        fatigue_penalty = 1.0 - self._fatigue_level

        adjusted_focus = (base_focus * importance_factor * urgency_factor *
                         complexity_penalty * capacity_factor * fatigue_penalty)

        return min(1.0, max(0.1, adjusted_focus))

    async def _ramp_up_focus(self, target_id: UUID, desired_level: FocusLevel) -> None:
        """집중도 점진적 증가"""
        target_focus = self._level_to_value(desired_level)
        current_focus = self._focus_levels.get(target_id, 0.0)

        ramp_duration = 2.0  # 2초에 걸쳐 증가
        steps = 20
        step_duration = ramp_duration / steps
        step_increment = (target_focus - current_focus) / steps

        for _ in range(steps):
            if target_id not in self._focus_levels:
                break  # 집중이 중단됨

            self._focus_levels[target_id] = min(
                target_focus,
                self._focus_levels[target_id] + step_increment
            )

            await asyncio.sleep(step_duration)

    async def _gradual_focus_adjustment(self, target_id: UUID,
                                      current: float, target: float) -> None:
        """점진적 집중도 조정"""
        adjustment_duration = 1.0  # 1초에 걸쳐 조정
        steps = 10
        step_duration = adjustment_duration / steps
        step_change = (target - current) / steps

        for _ in range(steps):
            if target_id not in self._focus_levels:
                break

            self._focus_levels[target_id] += step_change
            await asyncio.sleep(step_duration)

    async def _make_room_for_focus(self, new_target: FocusTarget) -> bool:
        """새로운 집중을 위한 공간 확보"""
        if not self._current_targets:
            return True

        # 가장 낮은 중요도의 집중 대상 제거
        lowest_importance = min(
            target.importance for target in self._current_targets.values()
        )

        # 새 대상이 더 중요한 경우에만 공간 확보
        if new_target.importance > lowest_importance:
            for target_id, target in list(self._current_targets.items()):
                if target.importance == lowest_importance:
                    await self.stop_focus(target_id)
                    return True

        return False

    def _calculate_focus_result(self, target_id: UUID, duration_ms: float) -> FocusResult:
        """집중 결과 계산"""
        target = self._current_targets[target_id]
        final_focus = self._focus_levels[target_id]
        achieved_level = self._value_to_level(final_focus)

        # 집중 효율성 계산
        efficiency_score = final_focus * (1.0 - self._fatigue_level)

        # 방해 요소 개수 계산 (최근 집중 기간 중)
        recent_distractions = [
            d for d in self._distraction_events
            if d["timestamp"] > target.created_at
        ]

        # 품질 점수 계산
        quality_score = min(1.0, efficiency_score * (1.0 - len(recent_distractions) * 0.05))

        return FocusResult(
            target_id=target_id,
            achieved_level=achieved_level,
            duration_ms=duration_ms,
            efficiency_score=efficiency_score,
            distraction_count=len(recent_distractions),
            quality_score=quality_score,
            success=quality_score > 0.5
        )

    def _update_focus_statistics(self, result: FocusResult) -> None:
        """집중 통계 업데이트"""
        self._total_focus_sessions += 1
        self._total_focus_time += result.duration_ms

        if result.success:
            self._successful_focus_sessions += 1

        # 피로도 증가
        session_fatigue = result.duration_ms / 60000.0 * 0.1  # 분당 10% 증가
        self._fatigue_level = min(1.0, self._fatigue_level + session_fatigue)

    async def _monitor_focus_decay(self) -> None:
        """집중도 자연 감소 모니터링"""
        while True:
            try:
                decay_rate = 0.01  # 초당 1% 감소

                for target_id in list(self._focus_levels.keys()):
                    current_focus = self._focus_levels[target_id]
                    new_focus = max(0.0, current_focus - decay_rate)
                    self._focus_levels[target_id] = new_focus

                    # 집중도가 너무 낮아지면 자동 종료
                    if new_focus < 0.1:
                        await self.stop_focus(target_id)

                await asyncio.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Error in focus decay monitoring: {e}")
                await asyncio.sleep(5.0)

    async def _manage_fatigue(self) -> None:
        """피로도 관리"""
        while True:
            try:
                # 활성 집중이 없으면 피로도 회복
                if not self._current_targets:
                    self._fatigue_level = max(0.0, self._fatigue_level - self._recovery_rate)

                # 용량 조정
                self._current_focus_capacity = self._baseline_focus_capacity * (1.0 - self._fatigue_level * 0.5)

                await asyncio.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Error in fatigue management: {e}")
                await asyncio.sleep(5.0)

    def _level_to_value(self, level: FocusLevel) -> float:
        """집중 수준을 수치값으로 변환"""
        level_map = {
            FocusLevel.MINIMAL: 0.2,
            FocusLevel.LOW: 0.4,
            FocusLevel.MEDIUM: 0.7,
            FocusLevel.HIGH: 0.9,
            FocusLevel.MAXIMUM: 1.0
        }
        return level_map.get(level, 0.5)

    def _value_to_level(self, value: float) -> FocusLevel:
        """수치값을 집중 수준으로 변환"""
        if value >= 0.9:
            return FocusLevel.MAXIMUM
        elif value >= 0.7:
            return FocusLevel.HIGH
        elif value >= 0.4:
            return FocusLevel.MEDIUM
        elif value >= 0.2:
            return FocusLevel.LOW
        else:
            return FocusLevel.MINIMAL


async def create_focus_controller(max_focus_targets: int = 3) -> FocusController:
    """FocusController 인스턴스 생성 및 초기화"""
    controller = FocusController(max_focus_targets)
    await controller.initialize()
    return controller