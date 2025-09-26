"""
Math Utilities Module
수학 계산 및 통계 관련 유틸리티 함수들
"""

import math
import statistics
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class ConfidenceInterval:
    """신뢰도 구간"""
    lower: float
    upper: float
    margin: float


def calculate_mean(values: List[float]) -> float:
    """평균값 계산"""
    if not values:
        return 0.0
    return sum(values) / len(values)


def calculate_median(values: List[float]) -> float:
    """중앙값 계산"""
    if not values:
        return 0.0
    return statistics.median(values)


def calculate_mode(values: List[float]) -> float:
    """최빈값 계산"""
    if not values:
        return 0.0
    try:
        return statistics.mode(values)
    except statistics.StatisticsError:
        # 최빈값이 없는 경우 첫 번째 값 반환
        return values[0]


def calculate_standard_deviation(values: List[float], sample: bool = True) -> float:
    """표준편차 계산"""
    if len(values) < 2:
        return 0.0

    if sample:
        return statistics.stdev(values)
    else:
        return statistics.pstdev(values)


def calculate_variance(values: List[float], sample: bool = True) -> float:
    """분산 계산"""
    if len(values) < 2:
        return 0.0

    if sample:
        return statistics.variance(values)
    else:
        return statistics.pvariance(values)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """최솟값과 최댓값 사이로 값을 제한"""
    return max(min_val, min(max_val, value))


def normalize(value: float, min_val: float, max_val: float) -> float:
    """값을 0-1 범위로 정규화"""
    if max_val == min_val:
        return 0.0
    return clamp((value - min_val) / (max_val - min_val), 0.0, 1.0)


def lerp(start: float, end: float, factor: float) -> float:
    """선형 보간"""
    return start + (end - start) * clamp(factor, 0.0, 1.0)


def calculate_percentile(values: List[float], percentile: float) -> float:
    """퍼센타일 계산"""
    if not values:
        return 0.0

    # Python의 statistics 모듈 사용
    return statistics.quantiles(values, n=100)[int(percentile) - 1]


def calculate_quartiles(values: List[float]) -> Tuple[float, float, float]:
    """사분위수 계산 (Q1, Q2, Q3)"""
    if not values:
        return 0.0, 0.0, 0.0

    quartiles = statistics.quantiles(values, n=4)
    return quartiles[0], quartiles[1], quartiles[2]


def calculate_correlation(x: List[float], y: List[float]) -> float:
    """상관관계 계산 (피어슨 상관계수)"""
    if len(x) != len(y) or len(x) == 0:
        return 0.0

    if len(x) < 2:
        return 0.0

    mean_x = calculate_mean(x)
    mean_y = calculate_mean(y)

    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
    sum_x_squared = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
    sum_y_squared = sum((y[i] - mean_y) ** 2 for i in range(len(y)))

    denominator = math.sqrt(sum_x_squared * sum_y_squared)
    return numerator / denominator if denominator != 0 else 0.0


def log_scale(value: float, base: float = 10.0) -> float:
    """로그 스케일 변환"""
    return math.log(max(value, 1.0)) / math.log(base)


def exp_scale(value: float, base: float = 10.0) -> float:
    """지수 스케일 변환"""
    return base ** value


def degrees_to_radians(degrees: float) -> float:
    """각도를 라디안으로 변환"""
    return math.radians(degrees)


def radians_to_degrees(radians: float) -> float:
    """라디안을 각도로 변환"""
    return math.degrees(radians)


def round_to(value: float, decimals: int) -> float:
    """소수점 반올림"""
    return round(value, decimals)


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """두 점 사이의 거리 계산"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_euclidean_distance(point1: List[float], point2: List[float]) -> float:
    """n차원 유클리드 거리 계산"""
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimensions")

    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))


def calculate_manhattan_distance(point1: List[float], point2: List[float]) -> float:
    """맨하탄 거리 계산"""
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimensions")

    return sum(abs(p1 - p2) for p1, p2 in zip(point1, point2))


def calculate_weighted_mean(values: List[float], weights: List[float]) -> float:
    """가중 평균 계산"""
    if len(values) != len(weights) or len(values) == 0:
        return 0.0

    numerator = sum(v * w for v, w in zip(values, weights))
    denominator = sum(weights)

    return numerator / denominator if denominator != 0 else 0.0


def calculate_moving_average(values: List[float], window_size: int) -> List[float]:
    """이동 평균 계산"""
    if window_size <= 0 or window_size > len(values):
        return values.copy()

    result = []
    for i in range(len(values) - window_size + 1):
        window = values[i:i + window_size]
        result.append(calculate_mean(window))

    return result


def calculate_exponential_moving_average(
    values: List[float],
    alpha: float = 0.3
) -> List[float]:
    """지수 이동 평균 계산"""
    if not values:
        return []

    result = [values[0]]
    for i in range(1, len(values)):
        ema = alpha * values[i] + (1 - alpha) * result[-1]
        result.append(ema)

    return result


def calculate_confidence_interval(
    values: List[float],
    confidence_level: float = 0.95
) -> ConfidenceInterval:
    """신뢰도 구간 계산"""
    if len(values) < 2:
        return ConfidenceInterval(0.0, 0.0, 0.0)

    mean = calculate_mean(values)
    std_dev = calculate_standard_deviation(values, sample=True)
    n = len(values)

    # Z-score 테이블 (간단한 버전)
    z_scores = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576
    }

    z_score = z_scores.get(confidence_level, 1.96)
    margin = z_score * (std_dev / math.sqrt(n))

    return ConfidenceInterval(
        lower=mean - margin,
        upper=mean + margin,
        margin=margin
    )


def calculate_z_score(value: float, mean: float, std_dev: float) -> float:
    """Z-점수 계산"""
    if std_dev == 0:
        return 0.0
    return (value - mean) / std_dev


def calculate_coefficient_of_variation(values: List[float]) -> float:
    """변동계수 계산"""
    if not values:
        return 0.0

    mean = calculate_mean(values)
    if mean == 0:
        return 0.0

    std_dev = calculate_standard_deviation(values)
    return (std_dev / mean) * 100


def calculate_skewness(values: List[float]) -> float:
    """왜도 계산"""
    if len(values) < 3:
        return 0.0

    mean = calculate_mean(values)
    std_dev = calculate_standard_deviation(values, sample=False)

    if std_dev == 0:
        return 0.0

    n = len(values)
    skewness = sum(((x - mean) / std_dev) ** 3 for x in values) / n
    return skewness


def calculate_kurtosis(values: List[float]) -> float:
    """첨도 계산"""
    if len(values) < 4:
        return 0.0

    mean = calculate_mean(values)
    std_dev = calculate_standard_deviation(values, sample=False)

    if std_dev == 0:
        return 0.0

    n = len(values)
    kurtosis = sum(((x - mean) / std_dev) ** 4 for x in values) / n
    return kurtosis - 3  # 정규분포의 첨도(3)를 기준으로 조정


def is_outlier(value: float, values: List[float], method: str = "iqr") -> bool:
    """이상치 판별"""
    if not values or len(values) < 4:
        return False

    if method == "iqr":
        q1, q2, q3 = calculate_quartiles(values)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return value < lower_bound or value > upper_bound

    elif method == "zscore":
        mean = calculate_mean(values)
        std_dev = calculate_standard_deviation(values)
        z_score = calculate_z_score(value, mean, std_dev)
        return abs(z_score) > 3

    return False


def remove_outliers(values: List[float], method: str = "iqr") -> List[float]:
    """이상치 제거"""
    return [v for v in values if not is_outlier(v, values, method)]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """안전한 나눗셈 (0으로 나누기 방지)"""
    return numerator / denominator if denominator != 0 else default


def factorial(n: int) -> int:
    """팩토리얼 계산"""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    return math.factorial(n)


def combinations(n: int, k: int) -> int:
    """조합 계산 (nCk)"""
    if k > n or k < 0:
        return 0
    return math.comb(n, k)


def permutations(n: int, k: int) -> int:
    """순열 계산 (nPk)"""
    if k > n or k < 0:
        return 0
    return math.perm(n, k)


# 별칭 (aliases) for compatibility
calculate_std_dev = calculate_standard_deviation
interpolate = lerp  # Linear interpolation alias


# Error class
class MathUtilsError(Exception):
    """Math utilities 관련 에러"""
    pass