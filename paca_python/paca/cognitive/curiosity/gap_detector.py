"""
Gap Detector - Logical Gap and Inconsistency Detection System

This module detects logical gaps, causal inconsistencies, and knowledge
lacunae that could drive productive curiosity and exploration.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
from enum import Enum
import asyncio
from datetime import datetime
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GapType(Enum):
    """Types of logical gaps that can be detected"""
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    CAUSAL_BREAK = "causal_break"
    MISSING_PREMISE = "missing_premise"
    INCOMPLETE_REASONING = "incomplete_reasoning"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"
    ASSUMPTION_GAP = "assumption_gap"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    CONTEXT_MISMATCH = "context_mismatch"

class GapSeverity(Enum):
    """Severity levels for detected gaps"""
    CRITICAL = "critical"          # Fundamental logical problems
    HIGH = "high"                  # Significant inconsistencies
    MEDIUM = "medium"              # Notable gaps worth exploring
    LOW = "low"                    # Minor inconsistencies
    INFORMATIONAL = "informational" # Potential areas for enhancement

@dataclass
class CausalChain:
    """Represents a chain of causal relationships"""
    cause: str
    effect: str
    mechanism: Optional[str] = None
    confidence: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    temporal_order: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LogicalGap:
    """Represents a detected logical gap or inconsistency"""
    gap_id: str
    gap_type: GapType
    severity: GapSeverity
    description: str
    location: str
    context: Dict[str, Any] = field(default_factory=dict)
    related_statements: List[str] = field(default_factory=list)
    potential_resolutions: List[str] = field(default_factory=list)
    exploration_potential: float = 0.0
    detected_at: datetime = field(default_factory=datetime.now)

@dataclass
class GapAnalysis:
    """Comprehensive analysis of detected gaps"""
    total_gaps: int = 0
    gaps_by_type: Dict[GapType, int] = field(default_factory=dict)
    gaps_by_severity: Dict[GapSeverity, int] = field(default_factory=dict)
    critical_gaps: List[LogicalGap] = field(default_factory=list)
    exploration_priorities: List[LogicalGap] = field(default_factory=list)
    overall_coherence_score: float = 0.0
    analysis_timestamp: datetime = field(default_factory=datetime.now)

class GapDetector:
    """
    Detects logical gaps, causal inconsistencies, and knowledge lacunae
    in reasoning chains and knowledge structures.
    """

    def __init__(self):
        self.detected_gaps: List[LogicalGap] = []
        self.causal_chains: List[CausalChain] = []
        self.logical_patterns = self._initialize_logical_patterns()
        self.gap_counter = 0

    def _initialize_logical_patterns(self) -> Dict[str, Any]:
        """Initialize patterns for detecting logical gaps"""
        return {
            'contradiction_indicators': [
                r'however\s*,\s*(?:also|simultaneously)',
                r'but\s+(?:also|at the same time)',
                r'(?:although|while)\s+.*(?:also|simultaneously)',
                r'both\s+.*\s+and\s+(?:not|never)',
                r'always\s+.*\s+never'
            ],
            'causal_indicators': [
                r'because\s+of',
                r'due\s+to',
                r'caused\s+by',
                r'results?\s+(?:in|from)',
                r'leads?\s+to',
                r'therefore',
                r'consequently'
            ],
            'assumption_indicators': [
                r'assuming\s+that',
                r'given\s+that',
                r'if\s+we\s+assume',
                r'presumably',
                r'supposedly',
                r'allegedly'
            ],
            'uncertainty_indicators': [
                r'(?:might|may|could)\s+be',
                r'(?:possibly|perhaps|maybe)',
                r'it\s+(?:seems|appears)',
                r'(?:unclear|uncertain|unknown)'
            ]
        }

    async def detect_gaps_in_text(self, text: str, context: Dict[str, Any] = None) -> List[LogicalGap]:
        """
        Detect logical gaps in a given text

        Args:
            text: The text to analyze
            context: Additional context for analysis

        Returns:
            List of detected logical gaps
        """
        if context is None:
            context = {}

        gaps = []

        # Detect various types of gaps
        gaps.extend(await self._detect_contradictions(text, context))
        gaps.extend(await self._detect_causal_breaks(text, context))
        gaps.extend(await self._detect_missing_premises(text, context))
        gaps.extend(await self._detect_assumption_gaps(text, context))

        # Store detected gaps
        self.detected_gaps.extend(gaps)

        return gaps

    async def _detect_contradictions(self, text: str, context: Dict[str, Any]) -> List[LogicalGap]:
        """Detect logical contradictions in text"""
        gaps = []
        sentences = text.split('.')

        for i, sentence in enumerate(sentences):
            for pattern in self.logical_patterns['contradiction_indicators']:
                if re.search(pattern, sentence.lower()):
                    gap = LogicalGap(
                        gap_id=f"CONTRA_{self._next_gap_id()}",
                        gap_type=GapType.LOGICAL_INCONSISTENCY,
                        severity=GapSeverity.HIGH,
                        description=f"Potential logical contradiction detected: {sentence.strip()}",
                        location=f"Sentence {i+1}",
                        context=context,
                        related_statements=[sentence.strip()],
                        potential_resolutions=[
                            "Clarify the apparent contradiction",
                            "Provide additional context",
                            "Resolve conflicting claims"
                        ],
                        exploration_potential=0.8
                    )
                    gaps.append(gap)

        return gaps

    async def _detect_causal_breaks(self, text: str, context: Dict[str, Any]) -> List[LogicalGap]:
        """Detect breaks in causal reasoning"""
        gaps = []

        # Extract causal claims
        causal_chains = await self._extract_causal_chains(text)

        # Analyze causal consistency
        for chain in causal_chains:
            if chain.confidence < 0.5:
                gap = LogicalGap(
                    gap_id=f"CAUSAL_{self._next_gap_id()}",
                    gap_type=GapType.CAUSAL_BREAK,
                    severity=GapSeverity.MEDIUM,
                    description=f"Weak causal link: {chain.cause} -> {chain.effect}",
                    location="Causal reasoning chain",
                    context=context,
                    related_statements=[f"{chain.cause} causes {chain.effect}"],
                    potential_resolutions=[
                        "Provide stronger evidence for causal link",
                        "Clarify the mechanism",
                        "Consider alternative explanations"
                    ],
                    exploration_potential=0.7
                )
                gaps.append(gap)

        return gaps

    async def _detect_missing_premises(self, text: str, context: Dict[str, Any]) -> List[LogicalGap]:
        """Detect missing premises in logical arguments"""
        gaps = []

        # Look for conclusion indicators without sufficient premises
        conclusion_patterns = [
            r'therefore\s+',
            r'thus\s+',
            r'consequently\s+',
            r'it\s+follows\s+that',
            r'we\s+can\s+conclude'
        ]

        for pattern in conclusion_patterns:
            matches = list(re.finditer(pattern, text.lower()))
            for match in matches:
                # Check if there are sufficient premises before the conclusion
                preceding_text = text[:match.start()]
                premise_count = len(re.findall(r'(?:because|since|given|if)', preceding_text.lower()))

                if premise_count < 1:
                    gap = LogicalGap(
                        gap_id=f"PREMISE_{self._next_gap_id()}",
                        gap_type=GapType.MISSING_PREMISE,
                        severity=GapSeverity.MEDIUM,
                        description="Conclusion drawn without sufficient explicit premises",
                        location=f"Character position {match.start()}",
                        context=context,
                        related_statements=[text[match.start():match.start()+100]],
                        potential_resolutions=[
                            "Provide supporting premises",
                            "Make implicit assumptions explicit",
                            "Strengthen the logical foundation"
                        ],
                        exploration_potential=0.6
                    )
                    gaps.append(gap)

        return gaps

    async def _detect_assumption_gaps(self, text: str, context: Dict[str, Any]) -> List[LogicalGap]:
        """Detect unstated assumptions that might need exploration"""
        gaps = []

        for pattern in self.logical_patterns['assumption_indicators']:
            matches = list(re.finditer(pattern, text.lower()))
            for match in matches:
                gap = LogicalGap(
                    gap_id=f"ASSUME_{self._next_gap_id()}",
                    gap_type=GapType.ASSUMPTION_GAP,
                    severity=GapSeverity.LOW,
                    description="Unstated assumption detected that could be explored",
                    location=f"Character position {match.start()}",
                    context=context,
                    related_statements=[text[match.start():match.start()+100]],
                    potential_resolutions=[
                        "Validate the assumption",
                        "Make the assumption explicit",
                        "Consider alternative assumptions"
                    ],
                    exploration_potential=0.5
                )
                gaps.append(gap)

        return gaps

    async def _extract_causal_chains(self, text: str) -> List[CausalChain]:
        """Extract causal relationships from text"""
        chains = []

        for pattern in self.logical_patterns['causal_indicators']:
            matches = list(re.finditer(pattern, text.lower()))
            for match in matches:
                # Extract cause and effect from context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                segment = text[start:end]

                # Simple extraction (could be enhanced with NLP)
                words = segment.split()
                cause_words = words[:len(words)//2]
                effect_words = words[len(words)//2:]

                chain = CausalChain(
                    cause=" ".join(cause_words[-5:]),  # Last 5 words before indicator
                    effect=" ".join(effect_words[:5]),  # First 5 words after indicator
                    mechanism=None,
                    confidence=0.6,  # Default moderate confidence
                    supporting_evidence=[]
                )
                chains.append(chain)

        return chains

    async def analyze_gap_patterns(self) -> GapAnalysis:
        """
        Analyze patterns in detected gaps to provide comprehensive insights

        Returns:
            Comprehensive gap analysis
        """
        analysis = GapAnalysis()

        if not self.detected_gaps:
            return analysis

        # Count gaps by type and severity
        analysis.total_gaps = len(self.detected_gaps)

        for gap in self.detected_gaps:
            # Count by type
            if gap.gap_type not in analysis.gaps_by_type:
                analysis.gaps_by_type[gap.gap_type] = 0
            analysis.gaps_by_type[gap.gap_type] += 1

            # Count by severity
            if gap.severity not in analysis.gaps_by_severity:
                analysis.gaps_by_severity[gap.severity] = 0
            analysis.gaps_by_severity[gap.severity] += 1

            # Identify critical gaps
            if gap.severity in [GapSeverity.CRITICAL, GapSeverity.HIGH]:
                analysis.critical_gaps.append(gap)

        # Sort by exploration potential for prioritization
        analysis.exploration_priorities = sorted(
            self.detected_gaps,
            key=lambda x: x.exploration_potential,
            reverse=True
        )[:10]  # Top 10 priorities

        # Calculate overall coherence score
        if analysis.total_gaps > 0:
            critical_weight = analysis.gaps_by_severity.get(GapSeverity.CRITICAL, 0) * 1.0
            high_weight = analysis.gaps_by_severity.get(GapSeverity.HIGH, 0) * 0.7
            medium_weight = analysis.gaps_by_severity.get(GapSeverity.MEDIUM, 0) * 0.4
            low_weight = analysis.gaps_by_severity.get(GapSeverity.LOW, 0) * 0.2

            total_weight = critical_weight + high_weight + medium_weight + low_weight
            analysis.overall_coherence_score = max(0.0, 1.0 - (total_weight / analysis.total_gaps))
        else:
            analysis.overall_coherence_score = 1.0

        return analysis

    def _next_gap_id(self) -> str:
        """Generate next gap ID"""
        self.gap_counter += 1
        return f"{self.gap_counter:04d}"

    async def get_exploration_suggestions(self, gap: LogicalGap) -> List[str]:
        """
        Generate specific exploration suggestions for a detected gap

        Args:
            gap: The logical gap to explore

        Returns:
            List of exploration suggestions
        """
        suggestions = []

        if gap.gap_type == GapType.LOGICAL_INCONSISTENCY:
            suggestions.extend([
                "Examine the logical structure more carefully",
                "Look for unstated assumptions",
                "Consider alternative interpretations",
                "Seek additional evidence to resolve the contradiction"
            ])

        elif gap.gap_type == GapType.CAUSAL_BREAK:
            suggestions.extend([
                "Investigate the mechanism connecting cause and effect",
                "Look for intervening variables",
                "Consider temporal relationships",
                "Examine alternative causal explanations"
            ])

        elif gap.gap_type == GapType.MISSING_PREMISE:
            suggestions.extend([
                "Identify what assumptions are being made",
                "Gather evidence for unstated premises",
                "Make implicit reasoning explicit",
                "Strengthen the logical foundation"
            ])

        elif gap.gap_type == GapType.ASSUMPTION_GAP:
            suggestions.extend([
                "Question the validity of the assumption",
                "Look for evidence supporting or contradicting the assumption",
                "Consider what happens if the assumption is false",
                "Explore alternative assumptions"
            ])

        # Add general suggestions
        suggestions.extend(gap.potential_resolutions)

        return list(set(suggestions))  # Remove duplicates

    def clear_gaps(self):
        """Clear all detected gaps (for fresh analysis)"""
        self.detected_gaps.clear()
        self.causal_chains.clear()
        self.gap_counter = 0