"""
Cache Metrics and Monitoring System for PACA

This module provides comprehensive performance monitoring and analytics
for the cache system with real-time metrics collection and reporting.

Features:
- Real-time performance metrics
- Hit/miss ratio tracking
- Response time analytics
- Memory usage monitoring
- Error rate tracking
- Performance trend analysis
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache statistics data structure"""
    # Basic metrics
    total_requests: int = 0
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0

    # Performance metrics
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0

    # Memory metrics
    memory_usage_mb: float = 0.0
    memory_efficiency: float = 0.0

    # Operational metrics
    uptime_seconds: float = 0.0
    error_rate: float = 0.0
    operations_per_second: float = 0.0

    # Level-specific stats (for multi-level caches)
    l1_stats: Dict[str, Any] = field(default_factory=dict)
    l2_stats: Dict[str, Any] = field(default_factory=dict)
    l3_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceWindow:
    """Performance metrics for a time window"""
    window_start: float
    window_end: float
    requests: int = 0
    hits: int = 0
    response_times: List[float] = field(default_factory=list)
    error_count: int = 0


class CacheMetrics:
    """
    Comprehensive cache metrics collection and analysis

    Features:
    - Real-time metrics collection
    - Performance window analysis
    - Trend detection and alerting
    - Percentile calculations
    - Thread-safe operations
    """

    def __init__(self, window_size_seconds: int = 300, max_windows: int = 288):
        # Configuration
        self.window_size = window_size_seconds  # 5 minutes default
        self.max_windows = max_windows          # 24 hours of 5-min windows

        # Core metrics
        self._total_requests = 0
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._deletes = 0
        self._errors = 0

        # Response time tracking
        self._response_times = deque(maxlen=10000)  # Last 10k requests
        self._hit_times = defaultdict(list)         # Per-level hit times

        # Performance windows for trend analysis
        self._performance_windows = deque(maxlen=max_windows)
        self._current_window: Optional[PerformanceWindow] = None

        # Timing
        self._start_time = time.time()
        self._lock = threading.RLock()

        # Initialize first window
        self._init_current_window()

    def record_hit(self, level: str, response_time: float):
        """Record a cache hit"""
        with self._lock:
            self._total_requests += 1
            self._hits += 1
            self._response_times.append(response_time)
            self._hit_times[level].append(response_time)

            # Update current window
            self._update_current_window(hit=True, response_time=response_time)

    def record_miss(self, response_time: float):
        """Record a cache miss"""
        with self._lock:
            self._total_requests += 1
            self._misses += 1
            self._response_times.append(response_time)

            # Update current window
            self._update_current_window(hit=False, response_time=response_time)

    def record_set(self, response_time: float):
        """Record a cache set operation"""
        with self._lock:
            self._sets += 1
            self._response_times.append(response_time)

    def record_delete(self):
        """Record a cache delete operation"""
        with self._lock:
            self._deletes += 1

    def record_error(self):
        """Record a cache error"""
        with self._lock:
            self._errors += 1

            # Update current window
            self._update_current_window(error=True)

    def get_stats(self) -> CacheStats:
        """Get comprehensive cache statistics"""
        with self._lock:
            return CacheStats(
                total_requests=self._total_requests,
                hits=self._hits,
                misses=self._misses,
                sets=self._sets,
                deletes=self._deletes,
                errors=self._errors,
                hit_rate=self._calculate_hit_rate(),
                miss_rate=self._calculate_miss_rate(),
                avg_response_time_ms=self._calculate_avg_response_time(),
                p95_response_time_ms=self._calculate_percentile(95),
                p99_response_time_ms=self._calculate_percentile(99),
                uptime_seconds=time.time() - self._start_time,
                error_rate=self._calculate_error_rate(),
                operations_per_second=self._calculate_ops_per_second()
            )

    def get_performance_trend(self, windows: int = 12) -> List[Dict[str, Any]]:
        """Get performance trend over recent windows"""
        with self._lock:
            recent_windows = list(self._performance_windows)[-windows:]

            trend_data = []
            for window in recent_windows:
                hit_rate = window.hits / max(window.requests, 1)
                avg_response = sum(window.response_times) / max(len(window.response_times), 1)

                trend_data.append({
                    'timestamp': window.window_start,
                    'requests': window.requests,
                    'hit_rate': hit_rate,
                    'avg_response_time_ms': avg_response * 1000,
                    'error_count': window.error_count
                })

            return trend_data

    def get_level_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics by cache level"""
        with self._lock:
            level_stats = {}

            for level, times in self._hit_times.items():
                if times:
                    level_stats[level] = {
                        'avg_response_time_ms': (sum(times) / len(times)) * 1000,
                        'hit_count': len(times),
                        'p95_response_time_ms': self._calculate_percentile_from_list(times, 95) * 1000,
                        'min_response_time_ms': min(times) * 1000,
                        'max_response_time_ms': max(times) * 1000
                    }

            return level_stats

    def get_recent_performance(self, seconds: int = 300) -> Dict[str, Any]:
        """Get performance metrics for recent time period"""
        with self._lock:
            cutoff_time = time.time() - seconds
            recent_times = [t for t in self._response_times if t > cutoff_time]

            if not recent_times:
                return {}

            return {
                'window_seconds': seconds,
                'request_count': len(recent_times),
                'avg_response_time_ms': (sum(recent_times) / len(recent_times)) * 1000,
                'p95_response_time_ms': self._calculate_percentile_from_list(recent_times, 95) * 1000,
                'requests_per_second': len(recent_times) / seconds
            }

    def detect_performance_issues(self) -> List[Dict[str, Any]]:
        """Detect performance issues and anomalies"""
        issues = []

        with self._lock:
            stats = self.get_stats()

            # High error rate
            if stats.error_rate > 0.05:  # 5%
                issues.append({
                    'type': 'high_error_rate',
                    'severity': 'critical' if stats.error_rate > 0.1 else 'warning',
                    'message': f"Error rate is {stats.error_rate:.2%}",
                    'value': stats.error_rate
                })

            # Low hit rate
            if stats.hit_rate < 0.7 and stats.total_requests > 100:
                issues.append({
                    'type': 'low_hit_rate',
                    'severity': 'critical' if stats.hit_rate < 0.5 else 'warning',
                    'message': f"Hit rate is {stats.hit_rate:.2%}",
                    'value': stats.hit_rate
                })

            # High response time
            if stats.p95_response_time_ms > 100:  # 100ms
                issues.append({
                    'type': 'high_response_time',
                    'severity': 'critical' if stats.p95_response_time_ms > 500 else 'warning',
                    'message': f"P95 response time is {stats.p95_response_time_ms:.1f}ms",
                    'value': stats.p95_response_time_ms
                })

            # Check for performance degradation trend
            trend_issue = self._detect_trend_degradation()
            if trend_issue:
                issues.append(trend_issue)

        return issues

    def reset(self):
        """Reset all metrics"""
        with self._lock:
            self._total_requests = 0
            self._hits = 0
            self._misses = 0
            self._sets = 0
            self._deletes = 0
            self._errors = 0
            self._response_times.clear()
            self._hit_times.clear()
            self._performance_windows.clear()
            self._start_time = time.time()
            self._init_current_window()

    def _calculate_hit_rate(self) -> float:
        """Calculate hit rate"""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def _calculate_miss_rate(self) -> float:
        """Calculate miss rate"""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._misses / total

    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time in milliseconds"""
        if not self._response_times:
            return 0.0
        return (sum(self._response_times) / len(self._response_times)) * 1000

    def _calculate_percentile(self, percentile: int) -> float:
        """Calculate response time percentile in milliseconds"""
        return self._calculate_percentile_from_list(list(self._response_times), percentile) * 1000

    def _calculate_percentile_from_list(self, times: List[float], percentile: int) -> float:
        """Calculate percentile from list of times"""
        if not times:
            return 0.0

        sorted_times = sorted(times)
        index = int((percentile / 100) * len(sorted_times))
        index = min(index, len(sorted_times) - 1)
        return sorted_times[index]

    def _calculate_error_rate(self) -> float:
        """Calculate error rate"""
        total_ops = self._total_requests + self._sets + self._deletes
        if total_ops == 0:
            return 0.0
        return self._errors / total_ops

    def _calculate_ops_per_second(self) -> float:
        """Calculate operations per second"""
        uptime = time.time() - self._start_time
        if uptime == 0:
            return 0.0
        total_ops = self._total_requests + self._sets + self._deletes
        return total_ops / uptime

    def _init_current_window(self):
        """Initialize a new performance window"""
        current_time = time.time()
        self._current_window = PerformanceWindow(
            window_start=current_time,
            window_end=current_time + self.window_size
        )

    def _update_current_window(self, hit: bool = False, response_time: float = 0.0, error: bool = False):
        """Update current performance window"""
        current_time = time.time()

        # Check if we need a new window
        if current_time >= self._current_window.window_end:
            # Archive current window
            self._performance_windows.append(self._current_window)
            # Start new window
            self._init_current_window()

        # Update current window
        if hit or not hit:  # Any request
            self._current_window.requests += 1
            if hit:
                self._current_window.hits += 1
            if response_time > 0:
                self._current_window.response_times.append(response_time)

        if error:
            self._current_window.error_count += 1

    def _detect_trend_degradation(self) -> Optional[Dict[str, Any]]:
        """Detect performance degradation trends"""
        if len(self._performance_windows) < 6:  # Need at least 6 windows
            return None

        recent_windows = list(self._performance_windows)[-6:]

        # Calculate trend in hit rate
        hit_rates = []
        for window in recent_windows:
            if window.requests > 0:
                hit_rates.append(window.hits / window.requests)

        if len(hit_rates) >= 4:
            # Simple trend detection: compare first and last half
            first_half = sum(hit_rates[:len(hit_rates)//2]) / (len(hit_rates)//2)
            second_half = hit_rates[len(hit_rates)//2:]
            second_half_avg = sum(second_half) / len(second_half)

            # Check for significant degradation
            if first_half - second_half_avg > 0.15:  # 15% degradation
                return {
                    'type': 'performance_degradation',
                    'severity': 'warning',
                    'message': f"Hit rate degraded from {first_half:.2%} to {second_half_avg:.2%}",
                    'value': first_half - second_half_avg
                }

        return None


class AdvancedMetrics:
    """Advanced metrics and analytics for cache performance"""

    def __init__(self, metrics: CacheMetrics):
        self.metrics = metrics

    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        stats = self.metrics.get_stats()
        trend = self.metrics.get_performance_trend()
        level_perf = self.metrics.get_level_performance()
        issues = self.metrics.detect_performance_issues()

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'status': 'healthy' if not issues else 'issues_detected',
                'total_requests': stats.total_requests,
                'hit_rate': stats.hit_rate,
                'avg_response_time_ms': stats.avg_response_time_ms,
                'uptime_hours': stats.uptime_seconds / 3600
            },
            'performance': {
                'hit_rate': stats.hit_rate,
                'miss_rate': stats.miss_rate,
                'error_rate': stats.error_rate,
                'response_times': {
                    'average_ms': stats.avg_response_time_ms,
                    'p95_ms': stats.p95_response_time_ms,
                    'p99_ms': stats.p99_response_time_ms
                },
                'throughput': {
                    'operations_per_second': stats.operations_per_second,
                    'total_operations': stats.total_requests + stats.sets + stats.deletes
                }
            },
            'levels': level_perf,
            'trends': trend,
            'issues': issues,
            'recommendations': self._generate_recommendations(stats, issues)
        }

        return report

    def _generate_recommendations(self, stats: CacheStats, issues: List[Dict]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []

        # Hit rate recommendations
        if stats.hit_rate < 0.8:
            recommendations.append("Consider increasing cache size or adjusting TTL values")

        # Response time recommendations
        if stats.avg_response_time_ms > 50:
            recommendations.append("Consider optimizing cache lookup algorithms or adding memory cache layer")

        # Error rate recommendations
        if stats.error_rate > 0.01:
            recommendations.append("Investigate and resolve cache errors to improve reliability")

        # Issue-specific recommendations
        for issue in issues:
            if issue['type'] == 'high_error_rate':
                recommendations.append("Review error logs and implement better error handling")
            elif issue['type'] == 'low_hit_rate':
                recommendations.append("Analyze cache key patterns and consider cache warming strategies")
            elif issue['type'] == 'high_response_time':
                recommendations.append("Consider cache optimization or infrastructure scaling")

        return recommendations


# Global metrics registry
_metrics_registry: Dict[str, CacheMetrics] = {}


def get_metrics(name: str = "default") -> CacheMetrics:
    """Get or create cache metrics instance"""
    if name not in _metrics_registry:
        _metrics_registry[name] = CacheMetrics()
    return _metrics_registry[name]


def reset_metrics(name: str = "default"):
    """Reset metrics for named instance"""
    if name in _metrics_registry:
        _metrics_registry[name].reset()


def get_all_metrics() -> Dict[str, CacheMetrics]:
    """Get all registered metrics instances"""
    return _metrics_registry.copy()