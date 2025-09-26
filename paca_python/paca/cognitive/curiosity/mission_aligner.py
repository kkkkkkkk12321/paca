"""
Mission Aligner - Curiosity-Mission Alignment System

This module ensures that curiosity-driven exploration remains aligned
with user missions, values, and long-term goals while preventing
runaway autonomous behavior.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any, Union
from enum import Enum
import asyncio
from datetime import datetime
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlignmentScore(Enum):
    """Alignment scores for mission compatibility"""
    PERFECT_ALIGNMENT = "perfect_alignment"    # 0.9-1.0
    HIGH_ALIGNMENT = "high_alignment"          # 0.7-0.89
    MODERATE_ALIGNMENT = "moderate_alignment"  # 0.5-0.69
    LOW_ALIGNMENT = "low_alignment"           # 0.3-0.49
    MISALIGNMENT = "misalignment"             # 0.1-0.29
    STRONG_MISALIGNMENT = "strong_misalignment" # 0.0-0.09

class ValueConflict(Enum):
    """Types of value conflicts that may arise"""
    PRIVACY_VIOLATION = "privacy_violation"
    ETHICAL_CONCERN = "ethical_concern"
    RESOURCE_WASTE = "resource_waste"
    GOAL_CONTRADICTION = "goal_contradiction"
    TIME_MISALLOCATION = "time_misallocation"
    SCOPE_OVERREACH = "scope_overreach"
    AUTONOMY_BREACH = "autonomy_breach"

@dataclass
class UserMission:
    """Represents a user's mission or goal"""
    mission_id: str
    title: str
    description: str
    core_values: List[str]
    success_criteria: List[str]
    priority: int = 1
    time_horizon: str = "medium_term"  # short_term, medium_term, long_term
    stakeholders: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class AlignmentCheck:
    """Results of alignment checking process"""
    check_id: str
    exploration_objective: str
    mission_references: List[str]
    alignment_score: float
    alignment_category: AlignmentScore
    supporting_factors: List[str] = field(default_factory=list)
    conflicting_factors: List[str] = field(default_factory=list)
    value_conflicts: List[ValueConflict] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    approval_required: bool = False
    checked_at: datetime = field(default_factory=datetime.now)

@dataclass
class MissionAlignment:
    """Comprehensive mission alignment analysis"""
    analysis_id: str
    total_missions_checked: int
    average_alignment_score: float
    highest_alignment: float
    lowest_alignment: float
    critical_conflicts: List[ValueConflict] = field(default_factory=list)
    alignment_distribution: Dict[AlignmentScore, int] = field(default_factory=dict)
    overall_recommendation: str = ""
    requires_user_approval: bool = False
    analysis_timestamp: datetime = field(default_factory=datetime.now)

class MissionAligner:
    """
    Ensures curiosity-driven exploration remains aligned with user missions
    and prevents runaway autonomous behavior.
    """

    def __init__(self):
        self.user_missions: Dict[str, UserMission] = {}
        self.alignment_checks: Dict[str, AlignmentCheck] = {}
        self.alignment_history: List[AlignmentCheck] = []
        self.check_counter = 0
        self.analysis_counter = 0

        # Default core values (can be overridden by user)
        self.default_core_values = [
            "user_autonomy",
            "transparency",
            "beneficial_outcomes",
            "resource_efficiency",
            "ethical_behavior",
            "privacy_protection"
        ]

    async def add_user_mission(
        self,
        title: str,
        description: str,
        core_values: Optional[List[str]] = None,
        success_criteria: Optional[List[str]] = None,
        priority: int = 1,
        time_horizon: str = "medium_term",
        constraints: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a user mission to the alignment system

        Args:
            title: Mission title
            description: Detailed mission description
            core_values: List of core values for this mission
            success_criteria: Success criteria for the mission
            priority: Priority level (1=highest, 5=lowest)
            time_horizon: Time frame (short_term, medium_term, long_term)
            constraints: Any constraints or limitations

        Returns:
            Mission ID
        """
        if core_values is None:
            core_values = self.default_core_values.copy()
        if success_criteria is None:
            success_criteria = []
        if constraints is None:
            constraints = {}

        mission_id = f"MISSION_{len(self.user_missions) + 1:04d}"

        mission = UserMission(
            mission_id=mission_id,
            title=title,
            description=description,
            core_values=core_values,
            success_criteria=success_criteria,
            priority=priority,
            time_horizon=time_horizon,
            constraints=constraints
        )

        self.user_missions[mission_id] = mission
        logger.info(f"Added user mission: {title} (ID: {mission_id})")

        return mission_id

    async def check_exploration_alignment(
        self,
        exploration_objective: str,
        exploration_context: Dict[str, Any] = None,
        specific_missions: Optional[List[str]] = None
    ) -> AlignmentCheck:
        """
        Check if an exploration objective aligns with user missions

        Args:
            exploration_objective: The objective to check
            exploration_context: Additional context for the exploration
            specific_missions: Specific missions to check against (if None, checks all)

        Returns:
            Alignment check results
        """
        if exploration_context is None:
            exploration_context = {}

        check_id = self._next_check_id()

        # Determine which missions to check against
        missions_to_check = []
        if specific_missions:
            missions_to_check = [
                self.user_missions[mid] for mid in specific_missions
                if mid in self.user_missions
            ]
        else:
            missions_to_check = list(self.user_missions.values())

        if not missions_to_check:
            # No missions defined - use default alignment check
            return await self._default_alignment_check(check_id, exploration_objective)

        # Perform alignment analysis
        alignment_scores = []
        supporting_factors = []
        conflicting_factors = []
        value_conflicts = []

        for mission in missions_to_check:
            score, factors = await self._calculate_mission_alignment(
                exploration_objective, mission, exploration_context
            )
            alignment_scores.append(score)

            if score >= 0.7:
                supporting_factors.extend(factors['supporting'])
            elif score <= 0.3:
                conflicting_factors.extend(factors['conflicting'])
                # Check for value conflicts
                conflicts = await self._detect_value_conflicts(
                    exploration_objective, mission, exploration_context
                )
                value_conflicts.extend(conflicts)

        # Calculate overall alignment
        overall_score = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0
        alignment_category = self._categorize_alignment_score(overall_score)

        # Generate recommendations
        recommendations = await self._generate_alignment_recommendations(
            exploration_objective, overall_score, supporting_factors,
            conflicting_factors, value_conflicts
        )

        # Determine if approval is required
        approval_required = (
            overall_score < 0.5 or
            len(value_conflicts) > 0 or
            any(conflict in [ValueConflict.AUTONOMY_BREACH, ValueConflict.ETHICAL_CONCERN]
                for conflict in value_conflicts)
        )

        check = AlignmentCheck(
            check_id=check_id,
            exploration_objective=exploration_objective,
            mission_references=[m.mission_id for m in missions_to_check],
            alignment_score=overall_score,
            alignment_category=alignment_category,
            supporting_factors=list(set(supporting_factors)),
            conflicting_factors=list(set(conflicting_factors)),
            value_conflicts=list(set(value_conflicts)),
            recommendations=recommendations,
            approval_required=approval_required
        )

        # Store the check
        self.alignment_checks[check_id] = check
        self.alignment_history.append(check)

        return check

    async def _calculate_mission_alignment(
        self,
        exploration_objective: str,
        mission: UserMission,
        context: Dict[str, Any]
    ) -> Tuple[float, Dict[str, List[str]]]:
        """
        Calculate alignment score between exploration and mission

        Returns:
            Tuple of (alignment_score, {'supporting': [...], 'conflicting': [...]})
        """
        objective_lower = exploration_objective.lower()
        mission_desc_lower = mission.description.lower()

        supporting_factors = []
        conflicting_factors = []
        score = 0.5  # Start with neutral score

        # Check for direct keyword matches
        mission_keywords = self._extract_keywords(mission_desc_lower)
        objective_keywords = self._extract_keywords(objective_lower)

        common_keywords = set(mission_keywords) & set(objective_keywords)
        if common_keywords:
            score += min(0.3, len(common_keywords) * 0.1)
            supporting_factors.append(f"Shared keywords: {', '.join(common_keywords)}")

        # Check alignment with core values
        for value in mission.core_values:
            value_alignment = await self._check_value_alignment(objective_lower, value)
            if value_alignment > 0:
                score += value_alignment * 0.1
                supporting_factors.append(f"Aligns with core value: {value}")
            elif value_alignment < 0:
                score += value_alignment * 0.1
                conflicting_factors.append(f"Conflicts with core value: {value}")

        # Check against success criteria
        for criterion in mission.success_criteria:
            criterion_lower = criterion.lower()
            if any(word in objective_lower for word in criterion_lower.split()):
                score += 0.1
                supporting_factors.append(f"Supports success criterion: {criterion}")

        # Apply priority weighting
        priority_weight = (6 - mission.priority) / 5  # Higher priority = higher weight
        score = score * priority_weight

        # Check for time horizon compatibility
        if 'quick' in objective_lower or 'immediate' in objective_lower:
            if mission.time_horizon == 'long_term':
                score -= 0.1
                conflicting_factors.append("Time horizon mismatch (immediate vs long-term)")
        elif 'long' in objective_lower or 'comprehensive' in objective_lower:
            if mission.time_horizon == 'short_term':
                score -= 0.1
                conflicting_factors.append("Time horizon mismatch (long vs short-term)")

        # Ensure score stays within bounds
        score = max(0.0, min(1.0, score))

        return score, {
            'supporting': supporting_factors,
            'conflicting': conflicting_factors
        }

    async def _check_value_alignment(self, objective: str, value: str) -> float:
        """Check alignment with a specific core value"""

        value_patterns = {
            'user_autonomy': {
                'positive': ['user choice', 'user control', 'user decision'],
                'negative': ['automatic', 'without asking', 'autonomous action']
            },
            'transparency': {
                'positive': ['explain', 'show', 'transparent', 'clear'],
                'negative': ['hidden', 'secret', 'opaque', 'unclear']
            },
            'beneficial_outcomes': {
                'positive': ['helpful', 'beneficial', 'useful', 'valuable'],
                'negative': ['harmful', 'wasteful', 'useless', 'detrimental']
            },
            'resource_efficiency': {
                'positive': ['efficient', 'optimized', 'streamlined'],
                'negative': ['wasteful', 'inefficient', 'excessive']
            },
            'ethical_behavior': {
                'positive': ['ethical', 'moral', 'responsible', 'fair'],
                'negative': ['unethical', 'unfair', 'irresponsible', 'biased']
            },
            'privacy_protection': {
                'positive': ['private', 'secure', 'protected', 'confidential'],
                'negative': ['expose', 'share', 'public', 'leak']
            }
        }

        if value not in value_patterns:
            return 0.0

        patterns = value_patterns[value]

        # Check for positive alignment
        positive_matches = sum(1 for pattern in patterns['positive'] if pattern in objective)
        negative_matches = sum(1 for pattern in patterns['negative'] if pattern in objective)

        if positive_matches > negative_matches:
            return min(0.2, positive_matches * 0.1)
        elif negative_matches > positive_matches:
            return max(-0.2, -negative_matches * 0.1)
        else:
            return 0.0

    async def _detect_value_conflicts(
        self,
        exploration_objective: str,
        mission: UserMission,
        context: Dict[str, Any]
    ) -> List[ValueConflict]:
        """Detect potential value conflicts"""

        conflicts = []
        objective_lower = exploration_objective.lower()

        # Privacy violation check
        if any(word in objective_lower for word in ['personal', 'private', 'sensitive']) and \
           any(word in objective_lower for word in ['share', 'expose', 'public']):
            conflicts.append(ValueConflict.PRIVACY_VIOLATION)

        # Ethical concern check
        if any(word in objective_lower for word in ['manipulate', 'deceive', 'exploit']):
            conflicts.append(ValueConflict.ETHICAL_CONCERN)

        # Resource waste check
        if any(word in objective_lower for word in ['excessive', 'unlimited', 'all']) and \
           any(word in objective_lower for word in ['resources', 'time', 'compute']):
            conflicts.append(ValueConflict.RESOURCE_WASTE)

        # Goal contradiction check
        if 'not' in objective_lower or 'opposite' in objective_lower:
            # Simple heuristic - could be enhanced
            conflicts.append(ValueConflict.GOAL_CONTRADICTION)

        # Autonomy breach check
        if any(word in objective_lower for word in ['without asking', 'automatically', 'independent']):
            if 'user_autonomy' in mission.core_values:
                conflicts.append(ValueConflict.AUTONOMY_BREACH)

        # Scope overreach check
        if any(word in objective_lower for word in ['everything', 'all', 'unlimited', 'infinite']):
            conflicts.append(ValueConflict.SCOPE_OVERREACH)

        return conflicts

    async def _default_alignment_check(
        self,
        check_id: str,
        exploration_objective: str
    ) -> AlignmentCheck:
        """Perform default alignment check when no missions are defined"""

        # Check against default values only
        score = 0.7  # Default moderate alignment
        supporting_factors = ["No specific missions defined - using default values"]
        conflicting_factors = []
        value_conflicts = []

        # Check for obvious red flags
        objective_lower = exploration_objective.lower()

        if any(word in objective_lower for word in ['harmful', 'destructive', 'malicious']):
            score = 0.1
            conflicting_factors.append("Potentially harmful objective")
            value_conflicts.append(ValueConflict.ETHICAL_CONCERN)

        if any(word in objective_lower for word in ['unlimited', 'infinite', 'all resources']):
            score -= 0.2
            conflicting_factors.append("Potentially resource-intensive")
            value_conflicts.append(ValueConflict.RESOURCE_WASTE)

        alignment_category = self._categorize_alignment_score(score)

        return AlignmentCheck(
            check_id=check_id,
            exploration_objective=exploration_objective,
            mission_references=[],
            alignment_score=score,
            alignment_category=alignment_category,
            supporting_factors=supporting_factors,
            conflicting_factors=conflicting_factors,
            value_conflicts=value_conflicts,
            recommendations=["Consider defining specific missions for better alignment checking"],
            approval_required=score < 0.5 or len(value_conflicts) > 0
        )

    def _categorize_alignment_score(self, score: float) -> AlignmentScore:
        """Categorize numerical alignment score"""
        if score >= 0.9:
            return AlignmentScore.PERFECT_ALIGNMENT
        elif score >= 0.7:
            return AlignmentScore.HIGH_ALIGNMENT
        elif score >= 0.5:
            return AlignmentScore.MODERATE_ALIGNMENT
        elif score >= 0.3:
            return AlignmentScore.LOW_ALIGNMENT
        elif score >= 0.1:
            return AlignmentScore.MISALIGNMENT
        else:
            return AlignmentScore.STRONG_MISALIGNMENT

    async def _generate_alignment_recommendations(
        self,
        objective: str,
        score: float,
        supporting_factors: List[str],
        conflicting_factors: List[str],
        value_conflicts: List[ValueConflict]
    ) -> List[str]:
        """Generate recommendations based on alignment analysis"""

        recommendations = []

        if score >= 0.7:
            recommendations.append("Exploration is well-aligned with user missions")
            if supporting_factors:
                recommendations.append("Proceed with confidence based on alignment factors")
        elif score >= 0.5:
            recommendations.append("Exploration has moderate alignment - consider refinement")
            if conflicting_factors:
                recommendations.append("Address conflicting factors before proceeding")
        else:
            recommendations.append("Low alignment detected - significant concerns exist")
            recommendations.append("Consider alternative approaches or seek user guidance")

        # Specific recommendations for value conflicts
        for conflict in value_conflicts:
            if conflict == ValueConflict.PRIVACY_VIOLATION:
                recommendations.append("Ensure privacy protection measures are in place")
            elif conflict == ValueConflict.ETHICAL_CONCERN:
                recommendations.append("Review ethical implications before proceeding")
            elif conflict == ValueConflict.RESOURCE_WASTE:
                recommendations.append("Optimize resource usage and set reasonable limits")
            elif conflict == ValueConflict.AUTONOMY_BREACH:
                recommendations.append("Seek explicit user permission before proceeding")
            elif conflict == ValueConflict.SCOPE_OVERREACH:
                recommendations.append("Narrow the scope to more manageable boundaries")

        return recommendations

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Simple keyword extraction - could be enhanced with NLP
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = text.lower().split()
        keywords = [word.strip('.,!?;') for word in words if word not in stop_words and len(word) > 2]
        return keywords

    async def analyze_overall_mission_alignment(
        self,
        exploration_objectives: List[str]
    ) -> MissionAlignment:
        """
        Analyze overall alignment across multiple exploration objectives

        Args:
            exploration_objectives: List of exploration objectives to analyze

        Returns:
            Comprehensive mission alignment analysis
        """
        analysis_id = f"ANALYSIS_{self._next_analysis_id()}"

        all_checks = []
        for objective in exploration_objectives:
            check = await self.check_exploration_alignment(objective)
            all_checks.append(check)

        if not all_checks:
            return MissionAlignment(
                analysis_id=analysis_id,
                total_missions_checked=0,
                average_alignment_score=0.0,
                highest_alignment=0.0,
                lowest_alignment=0.0,
                overall_recommendation="No objectives to analyze"
            )

        # Calculate statistics
        scores = [check.alignment_score for check in all_checks]
        average_score = sum(scores) / len(scores)
        highest_score = max(scores)
        lowest_score = min(scores)

        # Count alignment categories
        alignment_distribution = {}
        for check in all_checks:
            category = check.alignment_category
            alignment_distribution[category] = alignment_distribution.get(category, 0) + 1

        # Collect critical conflicts
        critical_conflicts = []
        requires_approval = False
        for check in all_checks:
            if check.approval_required:
                requires_approval = True
            for conflict in check.value_conflicts:
                if conflict in [ValueConflict.ETHICAL_CONCERN, ValueConflict.AUTONOMY_BREACH]:
                    critical_conflicts.append(conflict)

        # Generate overall recommendation
        if average_score >= 0.7:
            overall_recommendation = "Overall alignment is good - proceed with planned explorations"
        elif average_score >= 0.5:
            overall_recommendation = "Mixed alignment - review individual objectives carefully"
        else:
            overall_recommendation = "Poor overall alignment - significant revision recommended"

        return MissionAlignment(
            analysis_id=analysis_id,
            total_missions_checked=len(all_checks),
            average_alignment_score=average_score,
            highest_alignment=highest_score,
            lowest_alignment=lowest_score,
            critical_conflicts=list(set(critical_conflicts)),
            alignment_distribution=alignment_distribution,
            overall_recommendation=overall_recommendation,
            requires_user_approval=requires_approval
        )

    def _next_check_id(self) -> str:
        """Generate next check ID"""
        self.check_counter += 1
        return f"CHECK_{self.check_counter:04d}"

    def _next_analysis_id(self) -> str:
        """Generate next analysis ID"""
        self.analysis_counter += 1
        return f"{self.analysis_counter:04d}"

    def get_user_missions(self) -> Dict[str, UserMission]:
        """Get all user missions"""
        return self.user_missions.copy()

    def get_alignment_history(self) -> List[AlignmentCheck]:
        """Get alignment check history"""
        return self.alignment_history.copy()

    def update_mission(
        self,
        mission_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update an existing mission

        Args:
            mission_id: ID of mission to update
            updates: Dictionary of fields to update

        Returns:
            True if successful, False if mission not found
        """
        if mission_id not in self.user_missions:
            return False

        mission = self.user_missions[mission_id]
        for field, value in updates.items():
            if hasattr(mission, field):
                setattr(mission, field, value)

        return True

    def remove_mission(self, mission_id: str) -> bool:
        """
        Remove a mission

        Args:
            mission_id: ID of mission to remove

        Returns:
            True if successful, False if mission not found
        """
        if mission_id in self.user_missions:
            del self.user_missions[mission_id]
            return True
        return False