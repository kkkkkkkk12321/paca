"""
Curiosity Engine - Main Curiosity System Controller

This module implements the main curiosity engine that coordinates
gap detection, exploration planning, and mission alignment to generate
controlled curiosity-driven autonomous exploration.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any, Union
from enum import Enum
import asyncio
from datetime import datetime, timedelta
import logging
import json

from .gap_detector import GapDetector, LogicalGap, GapAnalysis, GapType, GapSeverity
from .exploration_planner import ExplorationPlanner, ExplorationPlan, ExplorationStrategy, ExplorationScope
from .mission_aligner import MissionAligner, AlignmentCheck, MissionAlignment, AlignmentScore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CuriosityLevel(Enum):
    """Levels of curiosity intensity"""
    DORMANT = "dormant"           # No active curiosity
    LOW = "low"                   # Minimal exploration drive
    MODERATE = "moderate"         # Balanced exploration
    HIGH = "high"                 # Strong exploration drive
    INTENSE = "intense"           # Maximum curiosity (use carefully)

class ExplorationFocus(Enum):
    """Focus areas for curiosity-driven exploration"""
    LOGICAL_GAPS = "logical_gaps"           # Focus on logical inconsistencies
    KNOWLEDGE_GAPS = "knowledge_gaps"       # Focus on missing information
    CAUSAL_UNDERSTANDING = "causal_understanding"  # Focus on cause-effect relationships
    PATTERN_DISCOVERY = "pattern_discovery"  # Focus on finding patterns
    CONTRADICTION_RESOLUTION = "contradiction_resolution"  # Focus on resolving conflicts
    CREATIVE_CONNECTIONS = "creative_connections"  # Focus on novel associations
    SYSTEMATIC_EXPLORATION = "systematic_exploration"  # Focus on comprehensive coverage

@dataclass
class CuriosityConfig:
    """Configuration for the curiosity engine"""
    curiosity_level: CuriosityLevel = CuriosityLevel.MODERATE
    exploration_focus: ExplorationFocus = ExplorationFocus.LOGICAL_GAPS
    max_concurrent_explorations: int = 3
    max_exploration_depth: int = 5
    time_budget_per_exploration: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    require_mission_alignment: bool = True
    minimum_alignment_score: float = 0.5
    auto_execute_aligned_explorations: bool = False
    curiosity_decay_rate: float = 0.1
    gap_sensitivity_threshold: float = 0.3

@dataclass
class CuriosityTrigger:
    """Represents a trigger for curiosity"""
    trigger_id: str
    trigger_type: str
    description: str
    intensity: float
    source_context: Dict[str, Any]
    detected_at: datetime = field(default_factory=datetime.now)

@dataclass
class ExplorationSession:
    """Represents an active exploration session"""
    session_id: str
    triggers: List[CuriosityTrigger]
    gaps: List[LogicalGap]
    plan: Optional[ExplorationPlan]
    alignment_check: Optional[AlignmentCheck]
    status: str = "initialized"  # initialized, planning, executing, completed, aborted
    results: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class CuriosityEngine:
    """
    Main curiosity engine that coordinates gap detection, exploration planning,
    and mission alignment to generate controlled autonomous exploration.
    """

    def __init__(self, config: Optional[CuriosityConfig] = None):
        self.config = config or CuriosityConfig()

        # Initialize sub-systems
        self.gap_detector = GapDetector()
        self.exploration_planner = ExplorationPlanner()
        self.mission_aligner = MissionAligner()

        # State management
        self.active_sessions: Dict[str, ExplorationSession] = {}
        self.completed_sessions: Dict[str, ExplorationSession] = {}
        self.curiosity_triggers: List[CuriosityTrigger] = []
        self.current_curiosity_level = self.config.curiosity_level

        # Counters
        self.session_counter = 0
        self.trigger_counter = 0

        # Performance tracking
        self.exploration_history: List[Dict[str, Any]] = []
        self.success_metrics: Dict[str, float] = {
            "successful_explorations": 0.0,
            "alignment_success_rate": 0.0,
            "insight_generation_rate": 0.0,
            "mission_contribution_score": 0.0
        }

    async def process_input_for_curiosity(
        self,
        input_text: str,
        context: Dict[str, Any] = None
    ) -> List[CuriosityTrigger]:
        """
        Process input text to identify curiosity triggers

        Args:
            input_text: Text to analyze for curiosity triggers
            context: Additional context information

        Returns:
            List of identified curiosity triggers
        """
        if context is None:
            context = {}

        triggers = []

        # Detect logical gaps that could trigger curiosity
        gaps = await self.gap_detector.detect_gaps_in_text(input_text, context)

        for gap in gaps:
            if gap.exploration_potential >= self.config.gap_sensitivity_threshold:
                trigger = CuriosityTrigger(
                    trigger_id=self._next_trigger_id(),
                    trigger_type="logical_gap",
                    description=f"Logical gap detected: {gap.description}",
                    intensity=gap.exploration_potential,
                    source_context={
                        "gap_id": gap.gap_id,
                        "gap_type": gap.gap_type.value,
                        "gap_severity": gap.severity.value,
                        "original_context": context
                    }
                )
                triggers.append(trigger)

        # Detect other curiosity patterns
        triggers.extend(await self._detect_curiosity_patterns(input_text, context))

        # Store triggers
        self.curiosity_triggers.extend(triggers)

        # Limit total triggers to prevent overflow
        if len(self.curiosity_triggers) > 100:
            self.curiosity_triggers = self.curiosity_triggers[-100:]

        return triggers

    async def _detect_curiosity_patterns(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> List[CuriosityTrigger]:
        """Detect other patterns that might trigger curiosity"""

        triggers = []
        text_lower = text.lower()

        # Wonder/curiosity expressions
        wonder_patterns = [
            r'i wonder', r'what if', r'how does', r'why does', r'what causes',
            r'curious about', r'interesting', r'puzzling', r'mysterious'
        ]

        import re
        for pattern in wonder_patterns:
            if re.search(pattern, text_lower):
                trigger = CuriosityTrigger(
                    trigger_id=self._next_trigger_id(),
                    trigger_type="wonder_expression",
                    description=f"Wonder/curiosity expression detected: {pattern}",
                    intensity=0.6,
                    source_context=context
                )
                triggers.append(trigger)

        # Incomplete knowledge indicators
        incomplete_patterns = [
            r'not sure', r'unclear', r'unknown', r'uncertain', r'might be',
            r'possibly', r'perhaps', r'maybe', r'seems like'
        ]

        for pattern in incomplete_patterns:
            if re.search(pattern, text_lower):
                trigger = CuriosityTrigger(
                    trigger_id=self._next_trigger_id(),
                    trigger_type="knowledge_gap",
                    description=f"Knowledge uncertainty detected: {pattern}",
                    intensity=0.4,
                    source_context=context
                )
                triggers.append(trigger)

        # Contradiction indicators
        contradiction_patterns = [
            r'however', r'but', r'although', r'despite', r'contradicts',
            r'conflicts with', r'different from'
        ]

        for pattern in contradiction_patterns:
            if re.search(pattern, text_lower):
                trigger = CuriosityTrigger(
                    trigger_id=self._next_trigger_id(),
                    trigger_type="contradiction",
                    description=f"Potential contradiction detected: {pattern}",
                    intensity=0.7,
                    source_context=context
                )
                triggers.append(trigger)

        return triggers

    async def generate_exploration_from_triggers(
        self,
        triggers: Optional[List[CuriosityTrigger]] = None,
        max_explorations: Optional[int] = None
    ) -> List[ExplorationSession]:
        """
        Generate exploration sessions from curiosity triggers

        Args:
            triggers: Specific triggers to use (if None, uses recent triggers)
            max_explorations: Maximum number of explorations to generate

        Returns:
            List of generated exploration sessions
        """
        if triggers is None:
            # Use recent high-intensity triggers
            triggers = [
                t for t in self.curiosity_triggers[-20:]  # Last 20 triggers
                if t.intensity >= self.config.gap_sensitivity_threshold
            ]

        if max_explorations is None:
            max_explorations = self.config.max_concurrent_explorations

        # Sort triggers by intensity
        triggers = sorted(triggers, key=lambda x: x.intensity, reverse=True)
        triggers = triggers[:max_explorations]

        sessions = []

        for trigger in triggers:
            try:
                session = await self._create_exploration_session([trigger])
                sessions.append(session)
            except Exception as e:
                logger.error(f"Failed to create exploration session for trigger {trigger.trigger_id}: {e}")

        return sessions

    async def _create_exploration_session(
        self,
        triggers: List[CuriosityTrigger]
    ) -> ExplorationSession:
        """Create an exploration session from triggers"""

        session_id = self._next_session_id()

        # Gather related gaps
        related_gaps = []
        for trigger in triggers:
            if trigger.trigger_type == "logical_gap":
                gap_id = trigger.source_context.get("gap_id")
                if gap_id:
                    # Find the corresponding gap
                    for gap in self.gap_detector.detected_gaps:
                        if gap.gap_id == gap_id:
                            related_gaps.append(gap)

        # Create exploration objective based on triggers
        objective = self._synthesize_exploration_objective(triggers, related_gaps)

        # Create exploration plan
        plan = await self.exploration_planner.create_exploration_plan(
            objective=objective,
            context={
                "triggers": [t.trigger_id for t in triggers],
                "gaps": [g.gap_id for g in related_gaps],
                "curiosity_level": self.current_curiosity_level.value
            },
            strategy=self._select_strategy_for_focus(self.config.exploration_focus),
            scope=self._select_scope_for_triggers(triggers)
        )

        # Check mission alignment if required
        alignment_check = None
        if self.config.require_mission_alignment:
            alignment_check = await self.mission_aligner.check_exploration_alignment(
                objective, {"triggers": triggers, "gaps": related_gaps}
            )

        # Create session
        session = ExplorationSession(
            session_id=session_id,
            triggers=triggers,
            gaps=related_gaps,
            plan=plan,
            alignment_check=alignment_check
        )

        # Store session
        self.active_sessions[session_id] = session

        return session

    def _synthesize_exploration_objective(
        self,
        triggers: List[CuriosityTrigger],
        gaps: List[LogicalGap]
    ) -> str:
        """Synthesize an exploration objective from triggers and gaps"""

        if not triggers:
            return "General exploratory investigation"

        # Analyze trigger types
        trigger_types = [t.trigger_type for t in triggers]

        if "logical_gap" in trigger_types:
            gap_descriptions = [g.description for g in gaps]
            if gap_descriptions:
                return f"Investigate logical gaps: {'; '.join(gap_descriptions[:2])}"

        if "contradiction" in trigger_types:
            return "Explore and resolve apparent contradictions"

        if "knowledge_gap" in trigger_types:
            return "Fill identified knowledge gaps and uncertainties"

        if "wonder_expression" in trigger_types:
            return "Explore curiosity-driven questions and wonderings"

        # Fallback
        return f"Investigate triggers: {', '.join(set(trigger_types))}"

    def _select_strategy_for_focus(self, focus: ExplorationFocus) -> ExplorationStrategy:
        """Select exploration strategy based on focus"""

        strategy_map = {
            ExplorationFocus.LOGICAL_GAPS: ExplorationStrategy.SYSTEMATIC,
            ExplorationFocus.KNOWLEDGE_GAPS: ExplorationStrategy.BREADTH_FIRST,
            ExplorationFocus.CAUSAL_UNDERSTANDING: ExplorationStrategy.DEPTH_FIRST,
            ExplorationFocus.PATTERN_DISCOVERY: ExplorationStrategy.CURIOSITY_DRIVEN,
            ExplorationFocus.CONTRADICTION_RESOLUTION: ExplorationStrategy.PRIORITY_BASED,
            ExplorationFocus.CREATIVE_CONNECTIONS: ExplorationStrategy.CURIOSITY_DRIVEN,
            ExplorationFocus.SYSTEMATIC_EXPLORATION: ExplorationStrategy.SYSTEMATIC
        }

        return strategy_map.get(focus, ExplorationStrategy.CURIOSITY_DRIVEN)

    def _select_scope_for_triggers(self, triggers: List[CuriosityTrigger]) -> ExplorationScope:
        """Select exploration scope based on triggers"""

        # Analyze trigger intensity and count
        avg_intensity = sum(t.intensity for t in triggers) / len(triggers) if triggers else 0.5
        trigger_count = len(triggers)

        if trigger_count == 1 and avg_intensity < 0.7:
            return ExplorationScope.FOCUSED
        elif trigger_count <= 2:
            return ExplorationScope.RELATED
        elif avg_intensity >= 0.8:
            return ExplorationScope.DOMAIN
        else:
            return ExplorationScope.RELATED

    async def evaluate_session_for_execution(
        self,
        session_id: str
    ) -> Tuple[bool, List[str]]:
        """
        Evaluate whether an exploration session should be executed

        Args:
            session_id: ID of the session to evaluate

        Returns:
            Tuple of (should_execute, reasons)
        """
        if session_id not in self.active_sessions:
            return False, ["Session not found"]

        session = self.active_sessions[session_id]
        should_execute = True
        reasons = []

        # Check mission alignment
        if session.alignment_check:
            if session.alignment_check.alignment_score < self.config.minimum_alignment_score:
                should_execute = False
                reasons.append(f"Low mission alignment score: {session.alignment_check.alignment_score:.2f}")

            if session.alignment_check.approval_required and not self.config.auto_execute_aligned_explorations:
                should_execute = False
                reasons.append("User approval required for execution")

        # Check resource constraints
        if len(self.active_sessions) >= self.config.max_concurrent_explorations:
            should_execute = False
            reasons.append("Maximum concurrent explorations reached")

        # Check curiosity level
        if self.current_curiosity_level == CuriosityLevel.DORMANT:
            should_execute = False
            reasons.append("Curiosity is currently dormant")

        if should_execute:
            reasons.append("All checks passed - ready for execution")

        return should_execute, reasons

    async def execute_exploration_session(
        self,
        session_id: str,
        force_execution: bool = False
    ) -> Dict[str, Any]:
        """
        Execute an exploration session

        Args:
            session_id: ID of the session to execute
            force_execution: Whether to force execution despite checks

        Returns:
            Execution results
        """
        if session_id not in self.active_sessions:
            return {"error": "Session not found", "session_id": session_id}

        session = self.active_sessions[session_id]

        # Check if execution is allowed
        should_execute, reasons = await self.evaluate_session_for_execution(session_id)

        if not should_execute and not force_execution:
            return {
                "error": "Execution not allowed",
                "reasons": reasons,
                "session_id": session_id
            }

        # Update session status
        session.status = "executing"
        session.started_at = datetime.now()

        try:
            # Execute the exploration plan
            if session.plan:
                execution_results = await self.exploration_planner.execute_plan(session.plan.plan_id)
                session.results = execution_results
            else:
                session.results = {"error": "No plan available for execution"}

            # Update session status
            session.status = "completed"
            session.completed_at = datetime.now()

            # Move to completed sessions
            self.completed_sessions[session_id] = session
            del self.active_sessions[session_id]

            # Update performance metrics
            await self._update_performance_metrics(session)

            return {
                "success": True,
                "session_id": session_id,
                "results": session.results,
                "execution_time": session.completed_at - session.started_at
            }

        except Exception as e:
            session.status = "aborted"
            session.results = {"error": str(e)}

            logger.error(f"Exploration session {session_id} failed: {e}")

            return {
                "error": "Execution failed",
                "exception": str(e),
                "session_id": session_id
            }

    async def _update_performance_metrics(self, session: ExplorationSession):
        """Update performance metrics based on session results"""

        # Simple performance tracking
        if session.status == "completed" and "error" not in session.results:
            self.success_metrics["successful_explorations"] += 1

        if session.alignment_check and session.alignment_check.alignment_score >= self.config.minimum_alignment_score:
            self.success_metrics["alignment_success_rate"] += 1

        # Store exploration history
        self.exploration_history.append({
            "session_id": session.session_id,
            "triggers_count": len(session.triggers),
            "gaps_count": len(session.gaps),
            "alignment_score": session.alignment_check.alignment_score if session.alignment_check else 0.0,
            "execution_time": (session.completed_at - session.started_at).total_seconds() if session.completed_at and session.started_at else 0,
            "status": session.status,
            "timestamp": session.completed_at or datetime.now()
        })

    async def adjust_curiosity_level(
        self,
        new_level: CuriosityLevel,
        reason: str = ""
    ):
        """
        Adjust the current curiosity level

        Args:
            new_level: New curiosity level
            reason: Reason for the adjustment
        """
        old_level = self.current_curiosity_level
        self.current_curiosity_level = new_level

        logger.info(f"Curiosity level changed from {old_level.value} to {new_level.value}. Reason: {reason}")

        # Adjust behavior based on new level
        if new_level == CuriosityLevel.DORMANT:
            # Pause all active explorations
            for session_id in list(self.active_sessions.keys()):
                session = self.active_sessions[session_id]
                if session.status == "executing":
                    session.status = "paused"

        elif new_level == CuriosityLevel.INTENSE:
            # Increase sensitivity and generation
            self.config.gap_sensitivity_threshold *= 0.5
            self.config.max_concurrent_explorations = min(10, self.config.max_concurrent_explorations + 2)

    def get_curiosity_status(self) -> Dict[str, Any]:
        """Get current curiosity engine status"""

        return {
            "curiosity_level": self.current_curiosity_level.value,
            "exploration_focus": self.config.exploration_focus.value,
            "active_sessions": len(self.active_sessions),
            "completed_sessions": len(self.completed_sessions),
            "total_triggers": len(self.curiosity_triggers),
            "recent_triggers": len([t for t in self.curiosity_triggers if (datetime.now() - t.detected_at).total_seconds() < 3600]),
            "performance_metrics": self.success_metrics.copy(),
            "config": {
                "max_concurrent_explorations": self.config.max_concurrent_explorations,
                "minimum_alignment_score": self.config.minimum_alignment_score,
                "gap_sensitivity_threshold": self.config.gap_sensitivity_threshold
            }
        }

    def get_current_status(self) -> Dict[str, Any]:
        """Get current status (alias for get_curiosity_status for compatibility)"""
        return self.get_curiosity_status()

    def _next_session_id(self) -> str:
        """Generate next session ID"""
        self.session_counter += 1
        return f"SESSION_{self.session_counter:04d}"

    def _next_trigger_id(self) -> str:
        """Generate next trigger ID"""
        self.trigger_counter += 1
        return f"TRIGGER_{self.trigger_counter:04d}"

    async def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old completed sessions and triggers"""

        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        # Clean up old completed sessions
        sessions_to_remove = [
            session_id for session_id, session in self.completed_sessions.items()
            if session.completed_at and session.completed_at < cutoff_time
        ]

        for session_id in sessions_to_remove:
            del self.completed_sessions[session_id]

        # Clean up old triggers
        self.curiosity_triggers = [
            trigger for trigger in self.curiosity_triggers
            if trigger.detected_at > cutoff_time
        ]

        logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions and {len(self.curiosity_triggers)} triggers")

    def get_session_details(self, session_id: str) -> Optional[ExplorationSession]:
        """Get details of a specific session"""

        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        elif session_id in self.completed_sessions:
            return self.completed_sessions[session_id]
        else:
            return None