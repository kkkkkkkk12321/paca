"""
Exploration Planner - Strategic Curiosity-Driven Exploration Planning

This module creates strategic plans for exploring detected gaps and
curiosity-driven questions while maintaining alignment with user goals.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any, Union
from enum import Enum
import asyncio
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExplorationStrategy(Enum):
    """Strategies for exploration"""
    BREADTH_FIRST = "breadth_first"      # Explore many areas lightly
    DEPTH_FIRST = "depth_first"          # Deep dive into specific areas
    PRIORITY_BASED = "priority_based"     # Follow priority rankings
    CURIOSITY_DRIVEN = "curiosity_driven" # Follow natural curiosity
    SYSTEMATIC = "systematic"             # Methodical, comprehensive approach
    OPPORTUNISTIC = "opportunistic"       # Take advantage of opportunities

class ExplorationScope(Enum):
    """Scope of exploration activities"""
    FOCUSED = "focused"           # Single specific topic/gap
    RELATED = "related"           # Related topics/connected gaps
    DOMAIN = "domain"             # Entire domain or field
    INTERDISCIPLINARY = "interdisciplinary"  # Across multiple domains
    COMPREHENSIVE = "comprehensive"  # Broad, system-wide exploration

class ExplorationResource(Enum):
    """Types of resources needed for exploration"""
    TIME = "time"
    COMPUTATIONAL = "computational"
    EXTERNAL_DATA = "external_data"
    USER_INPUT = "user_input"
    DOMAIN_EXPERTISE = "domain_expertise"
    CREATIVE_THINKING = "creative_thinking"

@dataclass
class ExplorationStep:
    """Individual step in an exploration plan"""
    step_id: str
    description: str
    method: str
    estimated_duration: timedelta
    required_resources: List[ExplorationResource]
    dependencies: List[str] = field(default_factory=list)
    expected_outcomes: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    risk_level: float = 0.0
    priority: int = 1

@dataclass
class ExplorationPlan:
    """Complete plan for exploring gaps or curiosity-driven questions"""
    plan_id: str
    title: str
    objective: str
    strategy: ExplorationStrategy
    scope: ExplorationScope
    steps: List[ExplorationStep] = field(default_factory=list)
    total_estimated_time: timedelta = field(default_factory=lambda: timedelta(0))
    required_resources: Set[ExplorationResource] = field(default_factory=set)
    success_metrics: Dict[str, str] = field(default_factory=dict)
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "planned"

class ExplorationPlanner:
    """
    Creates strategic plans for curiosity-driven exploration
    while maintaining alignment with user goals and constraints.
    """

    def __init__(self):
        self.exploration_plans: Dict[str, ExplorationPlan] = {}
        self.active_explorations: Set[str] = set()
        self.completed_explorations: Set[str] = set()
        self.plan_counter = 0
        self.step_counter = 0

    async def create_exploration_plan(
        self,
        objective: str,
        context: Dict[str, Any] = None,
        strategy: Optional[ExplorationStrategy] = None,
        scope: Optional[ExplorationScope] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> ExplorationPlan:
        """
        Create a comprehensive exploration plan

        Args:
            objective: The exploration objective
            context: Additional context for planning
            strategy: Preferred exploration strategy
            scope: Desired scope of exploration
            constraints: Resource and other constraints

        Returns:
            Complete exploration plan
        """
        if context is None:
            context = {}
        if constraints is None:
            constraints = {}

        # Auto-select strategy and scope if not provided
        if strategy is None:
            strategy = self._select_optimal_strategy(objective, context, constraints)
        if scope is None:
            scope = self._select_optimal_scope(objective, context, constraints)

        plan = ExplorationPlan(
            plan_id=self._next_plan_id(),
            title=f"Exploration: {objective[:50]}...",
            objective=objective,
            strategy=strategy,
            scope=scope
        )

        # Generate exploration steps
        plan.steps = await self._generate_exploration_steps(
            objective, strategy, scope, context, constraints
        )

        # Calculate total time and resources
        plan.total_estimated_time = sum(
            (step.estimated_duration for step in plan.steps),
            timedelta(0)
        )
        for step in plan.steps:
            plan.required_resources.update(step.required_resources)

        # Set success metrics
        plan.success_metrics = self._define_success_metrics(objective, strategy, scope)

        # Perform risk assessment
        plan.risk_assessment = await self._assess_exploration_risks(plan)

        # Store the plan
        self.exploration_plans[plan.plan_id] = plan

        return plan

    def _select_optimal_strategy(
        self,
        objective: str,
        context: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> ExplorationStrategy:
        """Select the optimal exploration strategy"""

        # Analyze objective characteristics
        objective_lower = objective.lower()

        # Time-constrained -> Priority-based
        if constraints.get('time_limit') or 'quickly' in objective_lower or 'urgent' in objective_lower:
            return ExplorationStrategy.PRIORITY_BASED

        # Deep understanding needed -> Depth-first
        if 'understand' in objective_lower or 'deep' in objective_lower or 'thorough' in objective_lower:
            return ExplorationStrategy.DEPTH_FIRST

        # Broad survey needed -> Breadth-first
        if 'overview' in objective_lower or 'survey' in objective_lower or 'explore' in objective_lower:
            return ExplorationStrategy.BREADTH_FIRST

        # Systematic analysis -> Systematic
        if 'systematic' in objective_lower or 'comprehensive' in objective_lower:
            return ExplorationStrategy.SYSTEMATIC

        # Default to curiosity-driven for open-ended exploration
        return ExplorationStrategy.CURIOSITY_DRIVEN

    def _select_optimal_scope(
        self,
        objective: str,
        context: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> ExplorationScope:
        """Select the optimal exploration scope"""

        objective_lower = objective.lower()

        # Specific focus indicators
        if any(word in objective_lower for word in ['specific', 'particular', 'focused']):
            return ExplorationScope.FOCUSED

        # Related topics indicators
        if any(word in objective_lower for word in ['related', 'connected', 'similar']):
            return ExplorationScope.RELATED

        # Domain-wide indicators
        if any(word in objective_lower for word in ['domain', 'field', 'area']):
            return ExplorationScope.DOMAIN

        # Cross-domain indicators
        if any(word in objective_lower for word in ['interdisciplinary', 'cross', 'multi']):
            return ExplorationScope.INTERDISCIPLINARY

        # Comprehensive indicators
        if any(word in objective_lower for word in ['comprehensive', 'complete', 'entire']):
            return ExplorationScope.COMPREHENSIVE

        # Default to related scope for balanced exploration
        return ExplorationScope.RELATED

    async def _generate_exploration_steps(
        self,
        objective: str,
        strategy: ExplorationStrategy,
        scope: ExplorationScope,
        context: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> List[ExplorationStep]:
        """Generate specific exploration steps based on strategy and scope"""

        steps = []

        # Common initial steps
        steps.append(ExplorationStep(
            step_id=self._next_step_id(),
            description="Define exploration boundaries and success criteria",
            method="analytical_definition",
            estimated_duration=timedelta(minutes=10),
            required_resources=[ExplorationResource.TIME, ExplorationResource.CREATIVE_THINKING],
            expected_outcomes=["Clear boundaries", "Success criteria"],
            success_criteria=["Boundaries are well-defined", "Criteria are measurable"],
            priority=1
        ))

        # Strategy-specific steps
        if strategy == ExplorationStrategy.BREADTH_FIRST:
            steps.extend(await self._generate_breadth_first_steps(objective, scope, context))
        elif strategy == ExplorationStrategy.DEPTH_FIRST:
            steps.extend(await self._generate_depth_first_steps(objective, scope, context))
        elif strategy == ExplorationStrategy.PRIORITY_BASED:
            steps.extend(await self._generate_priority_based_steps(objective, scope, context))
        elif strategy == ExplorationStrategy.SYSTEMATIC:
            steps.extend(await self._generate_systematic_steps(objective, scope, context))
        elif strategy == ExplorationStrategy.CURIOSITY_DRIVEN:
            steps.extend(await self._generate_curiosity_driven_steps(objective, scope, context))
        elif strategy == ExplorationStrategy.OPPORTUNISTIC:
            steps.extend(await self._generate_opportunistic_steps(objective, scope, context))

        # Common final steps
        steps.append(ExplorationStep(
            step_id=self._next_step_id(),
            description="Synthesize findings and generate insights",
            method="synthesis_analysis",
            estimated_duration=timedelta(minutes=20),
            required_resources=[ExplorationResource.TIME, ExplorationResource.CREATIVE_THINKING],
            dependencies=[step.step_id for step in steps],
            expected_outcomes=["Synthesized insights", "Key discoveries"],
            success_criteria=["Insights are novel", "Findings are actionable"],
            priority=len(steps) + 1
        ))

        return steps

    async def _generate_breadth_first_steps(
        self,
        objective: str,
        scope: ExplorationScope,
        context: Dict[str, Any]
    ) -> List[ExplorationStep]:
        """Generate steps for breadth-first exploration"""

        steps = []

        steps.append(ExplorationStep(
            step_id=self._next_step_id(),
            description="Identify key areas for broad exploration",
            method="area_mapping",
            estimated_duration=timedelta(minutes=15),
            required_resources=[ExplorationResource.TIME, ExplorationResource.CREATIVE_THINKING],
            expected_outcomes=["Area map", "Key topics identified"],
            success_criteria=["Comprehensive area coverage", "Topics are relevant"],
            priority=2
        ))

        steps.append(ExplorationStep(
            step_id=self._next_step_id(),
            description="Conduct surface-level exploration of each area",
            method="surface_survey",
            estimated_duration=timedelta(minutes=30),
            required_resources=[ExplorationResource.TIME, ExplorationResource.EXTERNAL_DATA],
            expected_outcomes=["Basic understanding of each area", "Interesting findings"],
            success_criteria=["Each area is covered", "Key patterns identified"],
            priority=3
        ))

        return steps

    async def _generate_depth_first_steps(
        self,
        objective: str,
        scope: ExplorationScope,
        context: Dict[str, Any]
    ) -> List[ExplorationStep]:
        """Generate steps for depth-first exploration"""

        steps = []

        steps.append(ExplorationStep(
            step_id=self._next_step_id(),
            description="Select primary area for deep exploration",
            method="priority_selection",
            estimated_duration=timedelta(minutes=10),
            required_resources=[ExplorationResource.TIME, ExplorationResource.CREATIVE_THINKING],
            expected_outcomes=["Primary area selected", "Rationale documented"],
            success_criteria=["Selection is well-justified", "Area has exploration potential"],
            priority=2
        ))

        steps.append(ExplorationStep(
            step_id=self._next_step_id(),
            description="Conduct deep investigation of selected area",
            method="deep_analysis",
            estimated_duration=timedelta(hours=1),
            required_resources=[
                ExplorationResource.TIME,
                ExplorationResource.COMPUTATIONAL,
                ExplorationResource.DOMAIN_EXPERTISE
            ],
            expected_outcomes=["Deep understanding", "Detailed insights"],
            success_criteria=["Understanding is comprehensive", "Insights are novel"],
            priority=3
        ))

        return steps

    async def _generate_priority_based_steps(
        self,
        objective: str,
        scope: ExplorationScope,
        context: Dict[str, Any]
    ) -> List[ExplorationStep]:
        """Generate steps for priority-based exploration"""

        steps = []

        steps.append(ExplorationStep(
            step_id=self._next_step_id(),
            description="Rank exploration targets by priority",
            method="priority_ranking",
            estimated_duration=timedelta(minutes=15),
            required_resources=[ExplorationResource.TIME, ExplorationResource.CREATIVE_THINKING],
            expected_outcomes=["Priority ranking", "Justification for priorities"],
            success_criteria=["Ranking is logical", "High-value targets identified"],
            priority=2
        ))

        steps.append(ExplorationStep(
            step_id=self._next_step_id(),
            description="Explore highest priority targets first",
            method="prioritized_exploration",
            estimated_duration=timedelta(minutes=45),
            required_resources=[ExplorationResource.TIME, ExplorationResource.EXTERNAL_DATA],
            expected_outcomes=["High-value insights", "Quick wins"],
            success_criteria=["Priorities are addressed", "Value is demonstrated"],
            priority=3
        ))

        return steps

    async def _generate_systematic_steps(
        self,
        objective: str,
        scope: ExplorationScope,
        context: Dict[str, Any]
    ) -> List[ExplorationStep]:
        """Generate steps for systematic exploration"""

        steps = []

        steps.append(ExplorationStep(
            step_id=self._next_step_id(),
            description="Create systematic exploration framework",
            method="framework_design",
            estimated_duration=timedelta(minutes=20),
            required_resources=[ExplorationResource.TIME, ExplorationResource.CREATIVE_THINKING],
            expected_outcomes=["Exploration framework", "Systematic approach"],
            success_criteria=["Framework is comprehensive", "Approach is logical"],
            priority=2
        ))

        steps.append(ExplorationStep(
            step_id=self._next_step_id(),
            description="Execute systematic exploration following framework",
            method="systematic_execution",
            estimated_duration=timedelta(hours=1, minutes=30),
            required_resources=[
                ExplorationResource.TIME,
                ExplorationResource.COMPUTATIONAL,
                ExplorationResource.EXTERNAL_DATA
            ],
            expected_outcomes=["Systematic coverage", "Comprehensive insights"],
            success_criteria=["Framework is followed", "Coverage is complete"],
            priority=3
        ))

        return steps

    async def _generate_curiosity_driven_steps(
        self,
        objective: str,
        scope: ExplorationScope,
        context: Dict[str, Any]
    ) -> List[ExplorationStep]:
        """Generate steps for curiosity-driven exploration"""

        steps = []

        steps.append(ExplorationStep(
            step_id=self._next_step_id(),
            description="Identify curiosity triggers and interesting questions",
            method="curiosity_identification",
            estimated_duration=timedelta(minutes=15),
            required_resources=[ExplorationResource.TIME, ExplorationResource.CREATIVE_THINKING],
            expected_outcomes=["Curiosity triggers", "Interesting questions"],
            success_criteria=["Questions are genuinely interesting", "Triggers are identified"],
            priority=2
        ))

        steps.append(ExplorationStep(
            step_id=self._next_step_id(),
            description="Follow curiosity trails naturally and organically",
            method="organic_exploration",
            estimated_duration=timedelta(minutes=45),
            required_resources=[
                ExplorationResource.TIME,
                ExplorationResource.CREATIVE_THINKING,
                ExplorationResource.EXTERNAL_DATA
            ],
            expected_outcomes=["Natural discoveries", "Unexpected insights"],
            success_criteria=["Exploration feels natural", "Insights are surprising"],
            priority=3
        ))

        return steps

    async def _generate_opportunistic_steps(
        self,
        objective: str,
        scope: ExplorationScope,
        context: Dict[str, Any]
    ) -> List[ExplorationStep]:
        """Generate steps for opportunistic exploration"""

        steps = []

        steps.append(ExplorationStep(
            step_id=self._next_step_id(),
            description="Scan for immediate opportunities and low-hanging fruit",
            method="opportunity_scanning",
            estimated_duration=timedelta(minutes=10),
            required_resources=[ExplorationResource.TIME],
            expected_outcomes=["Opportunity list", "Quick wins identified"],
            success_criteria=["Opportunities are viable", "Wins are achievable"],
            priority=2
        ))

        steps.append(ExplorationStep(
            step_id=self._next_step_id(),
            description="Exploit identified opportunities efficiently",
            method="opportunity_exploitation",
            estimated_duration=timedelta(minutes=30),
            required_resources=[ExplorationResource.TIME, ExplorationResource.EXTERNAL_DATA],
            expected_outcomes=["Quick insights", "Efficient progress"],
            success_criteria=["Opportunities are seized", "Progress is efficient"],
            priority=3
        ))

        return steps

    def _define_success_metrics(
        self,
        objective: str,
        strategy: ExplorationStrategy,
        scope: ExplorationScope
    ) -> Dict[str, str]:
        """Define success metrics for the exploration plan"""

        metrics = {
            "completion_rate": "Percentage of planned steps completed",
            "insight_quality": "Quality and novelty of generated insights",
            "objective_fulfillment": "Degree to which objective was met"
        }

        # Strategy-specific metrics
        if strategy == ExplorationStrategy.BREADTH_FIRST:
            metrics["coverage_breadth"] = "Number of areas explored"
        elif strategy == ExplorationStrategy.DEPTH_FIRST:
            metrics["understanding_depth"] = "Depth of understanding achieved"
        elif strategy == ExplorationStrategy.PRIORITY_BASED:
            metrics["priority_efficiency"] = "Efficiency in addressing priorities"
        elif strategy == ExplorationStrategy.SYSTEMATIC:
            metrics["systematic_completeness"] = "Completeness of systematic coverage"
        elif strategy == ExplorationStrategy.CURIOSITY_DRIVEN:
            metrics["curiosity_satisfaction"] = "Satisfaction of curiosity drives"

        # Scope-specific metrics
        if scope == ExplorationScope.COMPREHENSIVE:
            metrics["comprehensive_coverage"] = "Completeness of comprehensive exploration"

        return metrics

    async def _assess_exploration_risks(self, plan: ExplorationPlan) -> Dict[str, float]:
        """Assess risks associated with the exploration plan"""

        risks = {}

        # Time overrun risk
        if plan.total_estimated_time > timedelta(hours=2):
            risks["time_overrun"] = 0.7
        elif plan.total_estimated_time > timedelta(hours=1):
            risks["time_overrun"] = 0.4
        else:
            risks["time_overrun"] = 0.2

        # Resource unavailability risk
        if ExplorationResource.EXTERNAL_DATA in plan.required_resources:
            risks["data_unavailability"] = 0.3
        if ExplorationResource.DOMAIN_EXPERTISE in plan.required_resources:
            risks["expertise_gap"] = 0.4

        # Complexity risk
        complexity_score = len(plan.steps) * 0.1
        risks["excessive_complexity"] = min(1.0, complexity_score)

        # Scope creep risk
        if plan.scope in [ExplorationScope.INTERDISCIPLINARY, ExplorationScope.COMPREHENSIVE]:
            risks["scope_creep"] = 0.6
        else:
            risks["scope_creep"] = 0.3

        return risks

    async def execute_plan(self, plan_id: str) -> Dict[str, Any]:
        """
        Execute an exploration plan (simulation)

        Args:
            plan_id: ID of the plan to execute

        Returns:
            Execution results
        """
        if plan_id not in self.exploration_plans:
            raise ValueError(f"Plan {plan_id} not found")

        plan = self.exploration_plans[plan_id]
        plan.status = "executing"
        self.active_explorations.add(plan_id)

        execution_results = {
            "plan_id": plan_id,
            "status": "completed",
            "completed_steps": len(plan.steps),
            "total_steps": len(plan.steps),
            "insights_generated": [],
            "execution_time": plan.total_estimated_time,
            "success_metrics": {}
        }

        # Simulate execution by generating sample insights
        for step in plan.steps:
            execution_results["insights_generated"].extend(step.expected_outcomes)

        # Calculate success metrics
        for metric, description in plan.success_metrics.items():
            execution_results["success_metrics"][metric] = 0.8  # Sample success rate

        plan.status = "completed"
        self.active_explorations.remove(plan_id)
        self.completed_explorations.add(plan_id)

        return execution_results

    def _next_plan_id(self) -> str:
        """Generate next plan ID"""
        self.plan_counter += 1
        return f"PLAN_{self.plan_counter:04d}"

    def _next_step_id(self) -> str:
        """Generate next step ID"""
        self.step_counter += 1
        return f"STEP_{self.step_counter:04d}"

    def get_plan_status(self, plan_id: str) -> Optional[str]:
        """Get the status of a specific plan"""
        if plan_id in self.exploration_plans:
            return self.exploration_plans[plan_id].status
        return None

    def list_active_explorations(self) -> List[str]:
        """List all active exploration plan IDs"""
        return list(self.active_explorations)

    def list_completed_explorations(self) -> List[str]:
        """List all completed exploration plan IDs"""
        return list(self.completed_explorations)