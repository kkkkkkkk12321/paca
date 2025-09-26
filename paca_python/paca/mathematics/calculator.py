"""
Calculator Module
고급 수학 계산 엔진
"""

import numpy as np
import sympy as sp
from typing import Union, List, Dict, Any, Optional
import statistics

from ..core.types.base import Result
from ..core.errors.base import ValidationError, PacaError


class Calculator:
    """고급 수학 계산 엔진"""

    def __init__(self, precision: int = 15):
        self.precision = precision
        self.symbolic_engine = sp
        self.numeric_engine = np

    def add(self, a: float, b: float) -> float:
        """덧셈"""
        return a + b

    def subtract(self, a: float, b: float) -> float:
        """뺄셈"""
        return a - b

    def multiply(self, a: float, b: float) -> float:
        """곱셈"""
        return a * b

    def divide(self, a: float, b: float) -> float:
        """나눗셈"""
        if b == 0:
            raise ValueError("Division by zero")
        return a / b

    def power(self, base: float, exponent: float) -> float:
        """거듭제곱"""
        return base ** exponent

    def sqrt(self, x: float) -> float:
        """제곱근"""
        if x < 0:
            raise ValueError("Cannot calculate square root of negative number")
        return x ** 0.5

    async def calculate(
        self,
        expression: str,
        variables: Optional[Dict[str, float]] = None
    ) -> Result[Union[float, str]]:
        """수학 표현식 계산 (기호/수치 계산 자동 선택)"""
        try:
            if not expression:
                return Result.failure(ValidationError("Expression cannot be empty"))

            # 1. 표현식 파싱
            parsed_expr = self.symbolic_engine.sympify(expression)

            # 2. 변수가 있는 경우 대입
            if variables:
                for var, value in variables.items():
                    parsed_expr = parsed_expr.subs(var, value)

            # 3. 수치 계산 시도
            try:
                numeric_result = float(parsed_expr.evalf(self.precision))
                return Result.success(numeric_result)
            except (TypeError, ValueError):
                # 기호 계산 결과 반환
                symbolic_result = str(parsed_expr)
                return Result.success(symbolic_result)

        except Exception as e:
            return Result.failure(PacaError(f"Calculation error: {str(e)}"))

    async def solve_equation(
        self,
        equation: str,
        variable: str
    ) -> Result[List[float]]:
        """방정식 해결"""
        try:
            if not equation or not variable:
                return Result.failure(ValidationError("Equation and variable are required"))

            # 등호가 있는 경우 처리
            if '=' in equation:
                left, right = equation.split('=', 1)
                eq = self.symbolic_engine.Eq(
                    self.symbolic_engine.sympify(left.strip()),
                    self.symbolic_engine.sympify(right.strip())
                )
            else:
                # 등호가 없으면 =0으로 가정
                eq = self.symbolic_engine.Eq(self.symbolic_engine.sympify(equation), 0)

            solutions = self.symbolic_engine.solve(eq, variable)

            # 수치 해로 변환 (실수만)
            numeric_solutions = []
            for sol in solutions:
                try:
                    if sol.is_real:
                        numeric_val = float(sol.evalf())
                        numeric_solutions.append(numeric_val)
                except (AttributeError, TypeError, ValueError):
                    # 복소수나 기호 해는 무시
                    continue

            return Result.success(numeric_solutions)

        except Exception as e:
            return Result.failure(PacaError(f"Equation solving error: {str(e)}"))

    async def differentiate(
        self,
        expression: str,
        variable: str,
        order: int = 1
    ) -> Result[str]:
        """미분 계산"""
        try:
            if not expression or not variable:
                return Result.failure(ValidationError("Expression and variable are required"))

            if order < 1:
                return Result.failure(ValidationError("Order must be positive"))

            expr = self.symbolic_engine.sympify(expression)
            var = self.symbolic_engine.Symbol(variable)

            # n차 도함수 계산
            derivative = self.symbolic_engine.diff(expr, var, order)
            result = str(derivative)

            return Result.success(result)

        except Exception as e:
            return Result.failure(PacaError(f"Differentiation error: {str(e)}"))

    async def integrate(
        self,
        expression: str,
        variable: str,
        limits: Optional[tuple] = None
    ) -> Result[str]:
        """적분 계산"""
        try:
            if not expression or not variable:
                return Result.failure(ValidationError("Expression and variable are required"))

            expr = self.symbolic_engine.sympify(expression)
            var = self.symbolic_engine.Symbol(variable)

            if limits:
                # 정적분
                if len(limits) != 2:
                    return Result.failure(ValidationError("Limits must be a tuple of (lower, upper)"))

                lower, upper = limits
                integral = self.symbolic_engine.integrate(expr, (var, lower, upper))
            else:
                # 부정적분
                integral = self.symbolic_engine.integrate(expr, var)

            result = str(integral)
            return Result.success(result)

        except Exception as e:
            return Result.failure(PacaError(f"Integration error: {str(e)}"))

    async def factor(self, expression: str) -> Result[str]:
        """인수분해"""
        try:
            if not expression:
                return Result.failure(ValidationError("Expression cannot be empty"))

            expr = self.symbolic_engine.sympify(expression)
            factored = self.symbolic_engine.factor(expr)
            result = str(factored)

            return Result.success(result)

        except Exception as e:
            return Result.failure(PacaError(f"Factoring error: {str(e)}"))

    async def expand(self, expression: str) -> Result[str]:
        """전개"""
        try:
            if not expression:
                return Result.failure(ValidationError("Expression cannot be empty"))

            expr = self.symbolic_engine.sympify(expression)
            expanded = self.symbolic_engine.expand(expr)
            result = str(expanded)

            return Result.success(result)

        except Exception as e:
            return Result.failure(PacaError(f"Expansion error: {str(e)}"))

    async def simplify(self, expression: str) -> Result[str]:
        """단순화"""
        try:
            if not expression:
                return Result.failure(ValidationError("Expression cannot be empty"))

            expr = self.symbolic_engine.sympify(expression)
            simplified = self.symbolic_engine.simplify(expr)
            result = str(simplified)

            return Result.success(result)

        except Exception as e:
            return Result.failure(PacaError(f"Simplification error: {str(e)}"))


class StatisticalAnalyzer:
    """통계 분석 엔진"""

    def __init__(self):
        self.np = np

    async def analyze_dataset(self, data: List[float]) -> Result[Dict[str, float]]:
        """데이터셋 통계 분석"""
        try:
            if not data:
                return Result.failure(ValidationError("Data cannot be empty"))

            if not all(isinstance(x, (int, float)) for x in data):
                return Result.failure(ValidationError("All data points must be numeric"))

            data_array = self.np.array(data)

            analysis = {
                "count": len(data),
                "mean": float(self.np.mean(data_array)),
                "median": float(self.np.median(data_array)),
                "mode": self._calculate_mode(data),
                "std": float(self.np.std(data_array)),
                "var": float(self.np.var(data_array)),
                "min": float(self.np.min(data_array)),
                "max": float(self.np.max(data_array)),
                "range": float(self.np.max(data_array) - self.np.min(data_array)),
                "q1": float(self.np.percentile(data_array, 25)),
                "q2": float(self.np.percentile(data_array, 50)),  # median
                "q3": float(self.np.percentile(data_array, 75)),
                "iqr": float(self.np.percentile(data_array, 75) - self.np.percentile(data_array, 25)),
                "skewness": self._calculate_skewness(data_array),
                "kurtosis": self._calculate_kurtosis(data_array)
            }

            return Result.success(analysis)

        except Exception as e:
            return Result.failure(PacaError(f"Statistical analysis error: {str(e)}"))

    async def correlation(
        self,
        x_data: List[float],
        y_data: List[float]
    ) -> Result[Dict[str, float]]:
        """상관관계 분석"""
        try:
            if not x_data or not y_data:
                return Result.failure(ValidationError("Both datasets are required"))

            if len(x_data) != len(y_data):
                return Result.failure(ValidationError("Datasets must have the same length"))

            x_array = self.np.array(x_data)
            y_array = self.np.array(y_data)

            # 피어슨 상관계수
            correlation_matrix = self.np.corrcoef(x_array, y_array)
            pearson_r = float(correlation_matrix[0, 1])

            # 공분산
            covariance = float(self.np.cov(x_array, y_array)[0, 1])

            result = {
                "pearson_correlation": pearson_r,
                "covariance": covariance,
                "correlation_strength": self._interpret_correlation(abs(pearson_r))
            }

            return Result.success(result)

        except Exception as e:
            return Result.failure(PacaError(f"Correlation analysis error: {str(e)}"))

    async def linear_regression(
        self,
        x_data: List[float],
        y_data: List[float]
    ) -> Result[Dict[str, float]]:
        """선형 회귀 분석"""
        try:
            if not x_data or not y_data:
                return Result.failure(ValidationError("Both datasets are required"))

            if len(x_data) != len(y_data):
                return Result.failure(ValidationError("Datasets must have the same length"))

            x_array = self.np.array(x_data)
            y_array = self.np.array(y_data)

            # 선형 회귀 계수 계산
            coefficients = self.np.polyfit(x_array, y_array, 1)
            slope, intercept = coefficients

            # 예측값 계산
            y_pred = slope * x_array + intercept

            # R-squared 계산
            ss_res = self.np.sum((y_array - y_pred) ** 2)
            ss_tot = self.np.sum((y_array - self.np.mean(y_array)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            result = {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_squared),
                "equation": f"y = {slope:.4f}x + {intercept:.4f}"
            }

            return Result.success(result)

        except Exception as e:
            return Result.failure(PacaError(f"Linear regression error: {str(e)}"))

    def _calculate_mode(self, data: List[float]) -> float:
        """최빈값 계산"""
        try:
            return float(statistics.mode(data))
        except statistics.StatisticsError:
            # 최빈값이 없는 경우 첫 번째 값 반환
            return float(data[0]) if data else 0.0

    def _calculate_skewness(self, data_array: np.ndarray) -> float:
        """왜도 계산"""
        n = len(data_array)
        if n < 3:
            return 0.0

        mean = self.np.mean(data_array)
        std = self.np.std(data_array)

        if std == 0:
            return 0.0

        skewness = self.np.sum(((data_array - mean) / std) ** 3) / n
        return float(skewness)

    def _calculate_kurtosis(self, data_array: np.ndarray) -> float:
        """첨도 계산"""
        n = len(data_array)
        if n < 4:
            return 0.0

        mean = self.np.mean(data_array)
        std = self.np.std(data_array)

        if std == 0:
            return 0.0

        kurtosis = self.np.sum(((data_array - mean) / std) ** 4) / n - 3
        return float(kurtosis)

    def _interpret_correlation(self, abs_correlation: float) -> str:
        """상관관계 강도 해석"""
        if abs_correlation >= 0.9:
            return "very_strong"
        elif abs_correlation >= 0.7:
            return "strong"
        elif abs_correlation >= 0.5:
            return "moderate"
        elif abs_correlation >= 0.3:
            return "weak"
        else:
            return "very_weak"