"""
Mathematics Module - Entry Point
수학 시스템 모듈 진입점
"""

from .calculator import Calculator, StatisticalAnalyzer

# 공통 타입들
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

class MathematicalDomain(Enum):
    """수학적 도메인"""
    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    GEOMETRY = "geometry"
    STATISTICS = "statistics"
    DISCRETE = "discrete"
    NUMBER_THEORY = "number_theory"

class ProofMethod(Enum):
    """증명 방법"""
    DIRECT = "direct"
    INDIRECT = "indirect"
    CONTRADICTION = "contradiction"
    INDUCTION = "induction"
    CONSTRUCTION = "construction"

class ProblemType(Enum):
    """문제 유형"""
    COMPUTATION = "computation"
    PROOF = "proof"
    MODELING = "modeling"
    OPTIMIZATION = "optimization"
    ANALYSIS = "analysis"

@dataclass
class MathematicalExpression:
    """수학적 표현식"""
    expression: str
    variables: List[str]
    domain: MathematicalDomain
    metadata: Dict[str, Any]

@dataclass
class MathematicalSolution:
    """수학적 해결책"""
    problem_id: str
    solution_steps: List[str]
    final_answer: Any
    method: str
    domain: MathematicalDomain
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class QualityDimensions:
    """품질 차원"""
    correctness: float
    rigor: float
    efficiency: float
    clarity: float
    pedagogical_value: float
    creativity: float

@dataclass
class MathQualityEvaluation:
    """수학 품질 평가"""
    solution_id: str
    overall_score: float
    dimensions: QualityDimensions
    feedback: List[str]
    suggestions: List[str]
    metadata: Dict[str, Any]

class MathQualityEvaluator:
    """수학 품질 평가자"""

    def __init__(self):
        self.criteria = {
            "correctness": 0.4,
            "rigor": 0.2,
            "efficiency": 0.15,
            "clarity": 0.15,
            "pedagogical_value": 0.1
        }

    def evaluate(self, solution: MathematicalSolution) -> MathQualityEvaluation:
        """해결책 평가"""
        # 간단한 평가 로직 (실제로는 더 복잡)
        dimensions = QualityDimensions(
            correctness=0.85,
            rigor=0.75,
            efficiency=0.80,
            clarity=0.70,
            pedagogical_value=0.65,
            creativity=0.60
        )

        overall_score = (
            dimensions.correctness * self.criteria["correctness"] +
            dimensions.rigor * self.criteria["rigor"] +
            dimensions.efficiency * self.criteria["efficiency"] +
            dimensions.clarity * self.criteria["clarity"] +
            dimensions.pedagogical_value * self.criteria["pedagogical_value"]
        )

        return MathQualityEvaluation(
            solution_id=solution.problem_id,
            overall_score=overall_score,
            dimensions=dimensions,
            feedback=["Solution shows good mathematical reasoning"],
            suggestions=["Consider adding more detailed explanations"],
            metadata={}
        )

class MathematicalReasoningEngine:
    """수학적 추론 엔진"""

    def __init__(self):
        self.evaluator = MathQualityEvaluator()

    def solve(self, expression: MathematicalExpression) -> MathematicalSolution:
        """수학 문제 해결"""
        # 간단한 해결 로직 (실제로는 더 복잡)
        solution = MathematicalSolution(
            problem_id=str(hash(expression.expression)),
            solution_steps=[
                "문제 분석",
                "방법 선택",
                "계산 수행",
                "결과 검증"
            ],
            final_answer="해결됨",
            method="직접 계산",
            domain=expression.domain,
            confidence=0.85,
            metadata={}
        )

        return solution

    def evaluate_solution(self, solution: MathematicalSolution) -> MathQualityEvaluation:
        """해결책 평가"""
        return self.evaluator.evaluate(solution)

__all__ = [
    # Calculator
    'Calculator',
    'StatisticalAnalyzer',
    # Mathematics types
    'MathematicalDomain',
    'ProofMethod',
    'ProblemType',
    'MathematicalExpression',
    'MathematicalSolution',
    'QualityDimensions',
    'MathQualityEvaluation',
    'MathQualityEvaluator',
    'MathematicalReasoningEngine'
]