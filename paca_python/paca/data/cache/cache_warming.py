"""
Cache Warming System for PACA

This module provides intelligent cache preloading and warming strategies
to optimize cache performance and reduce cold start penalties.

Features:
- Intelligent key prediction based on access patterns
- Scheduled cache warming
- Priority-based warming strategies
- Background warming processes
- Cache warming analytics
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Callable, Awaitable
from collections import defaultdict, Counter
import json
from pathlib import Path
import pickle
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class WarmingStrategy:
    """Cache warming strategy configuration"""
    name: str
    priority: int = 1                    # Higher priority = warmed first
    max_keys: int = 1000                 # Maximum keys to warm
    key_pattern: Optional[str] = None    # Key pattern to match
    warm_interval: int = 3600            # Warming interval in seconds
    enabled: bool = True


@dataclass
class WarmingStats:
    """Cache warming statistics"""
    total_warmed: int = 0
    successful_warms: int = 0
    failed_warms: int = 0
    last_warming_time: float = 0.0
    warming_duration_ms: float = 0.0
    keys_predicted: int = 0
    prediction_accuracy: float = 0.0


class AccessPatternAnalyzer:
    """Analyzes access patterns to predict future cache needs"""

    def __init__(self, max_patterns: int = 10000):
        self.max_patterns = max_patterns
        self.access_log: List[Dict[str, Any]] = []
        self.pattern_frequencies = Counter()
        self.key_sequences = defaultdict(list)
        self.time_patterns = defaultdict(list)
        self._lock = asyncio.Lock()

    async def record_access(self, key: str, timestamp: Optional[float] = None):
        """Record a cache access for pattern analysis"""
        timestamp = timestamp or time.time()

        async with self._lock:
            # Record access
            access_record = {
                'key': key,
                'timestamp': timestamp,
                'hour': int((timestamp % 86400) // 3600),  # Hour of day
                'day_of_week': int((timestamp // 86400) % 7)  # Day of week
            }

            self.access_log.append(access_record)

            # Limit log size
            if len(self.access_log) > self.max_patterns:
                self.access_log = self.access_log[-self.max_patterns//2:]

            # Update patterns
            await self._update_patterns(access_record)

    async def predict_keys(self, count: int = 100) -> List[str]:
        """Predict keys likely to be accessed soon"""
        async with self._lock:
            predictions = []

            # Time-based predictions
            current_hour = int((time.time() % 86400) // 3600)
            time_based = await self._predict_by_time(current_hour, count // 2)
            predictions.extend(time_based)

            # Frequency-based predictions
            frequency_based = await self._predict_by_frequency(count // 2)
            predictions.extend(frequency_based)

            # Sequence-based predictions
            sequence_based = await self._predict_by_sequence(count // 4)
            predictions.extend(sequence_based)

            # Remove duplicates while preserving order
            seen = set()
            unique_predictions = []
            for key in predictions:
                if key not in seen:
                    seen.add(key)
                    unique_predictions.append(key)

            return unique_predictions[:count]

    async def get_pattern_insights(self) -> Dict[str, Any]:
        """Get insights about access patterns"""
        async with self._lock:
            if not self.access_log:
                return {}

            # Analyze time patterns
            hourly_distribution = defaultdict(int)
            daily_distribution = defaultdict(int)

            for record in self.access_log:
                hourly_distribution[record['hour']] += 1
                daily_distribution[record['day_of_week']] += 1

            # Find peak hours and days
            peak_hour = max(hourly_distribution.items(), key=lambda x: x[1])[0]
            peak_day = max(daily_distribution.items(), key=lambda x: x[1])[0]

            # Calculate key diversity
            unique_keys = len(set(record['key'] for record in self.access_log))
            total_accesses = len(self.access_log)

            return {
                'total_accesses': total_accesses,
                'unique_keys': unique_keys,
                'key_diversity': unique_keys / max(total_accesses, 1),
                'peak_hour': peak_hour,
                'peak_day': peak_day,
                'hourly_distribution': dict(hourly_distribution),
                'daily_distribution': dict(daily_distribution),
                'top_keys': [item[0] for item in self.pattern_frequencies.most_common(10)]
            }

    async def _update_patterns(self, access_record: Dict[str, Any]):
        """Update pattern tracking"""
        key = access_record['key']
        hour = access_record['hour']
        timestamp = access_record['timestamp']

        # Update frequency patterns
        self.pattern_frequencies[key] += 1

        # Update time patterns
        self.time_patterns[hour].append(key)
        if len(self.time_patterns[hour]) > 100:
            self.time_patterns[hour] = self.time_patterns[hour][-50:]

        # Update sequence patterns
        if len(self.access_log) >= 2:
            prev_key = self.access_log[-2]['key']
            self.key_sequences[prev_key].append(key)
            if len(self.key_sequences[prev_key]) > 20:
                self.key_sequences[prev_key] = self.key_sequences[prev_key][-10:]

    async def _predict_by_time(self, current_hour: int, count: int) -> List[str]:
        """Predict keys based on time patterns"""
        if current_hour not in self.time_patterns:
            return []

        # Get keys accessed during this hour historically
        hour_keys = self.time_patterns[current_hour]
        key_counts = Counter(hour_keys)

        return [key for key, _ in key_counts.most_common(count)]

    async def _predict_by_frequency(self, count: int) -> List[str]:
        """Predict keys based on access frequency"""
        return [key for key, _ in self.pattern_frequencies.most_common(count)]

    async def _predict_by_sequence(self, count: int) -> List[str]:
        """Predict keys based on access sequences"""
        if not self.access_log:
            return []

        # Get the last accessed key
        last_key = self.access_log[-1]['key']

        if last_key in self.key_sequences:
            sequence_keys = self.key_sequences[last_key]
            key_counts = Counter(sequence_keys)
            return [key for key, _ in key_counts.most_common(count)]

        return []


class CacheWarming:
    """
    Intelligent cache warming system

    Features:
    - Multiple warming strategies
    - Access pattern analysis
    - Scheduled warming
    - Performance monitoring
    - Adaptive warming based on usage patterns
    """

    def __init__(self, cache_manager, config_dir: str = "warming_config"):
        self.cache_manager = cache_manager
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.pattern_analyzer = AccessPatternAnalyzer()
        self.strategies: List[WarmingStrategy] = []
        self.stats = WarmingStats()

        # Data sources for warming
        self.data_sources: Dict[str, Callable[[str], Awaitable[Any]]] = {}

        # State
        self._warming_active = False
        self._warming_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Load configuration
        asyncio.create_task(self._load_configuration())

    async def add_strategy(self, strategy: WarmingStrategy):
        """Add a warming strategy"""
        async with self._lock:
            self.strategies.append(strategy)
            await self._save_configuration()

    async def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a warming strategy"""
        async with self._lock:
            initial_count = len(self.strategies)
            self.strategies = [s for s in self.strategies if s.name != strategy_name]

            if len(self.strategies) < initial_count:
                await self._save_configuration()
                return True
            return False

    async def register_data_source(self, name: str, source_func: Callable[[str], Awaitable[Any]]):
        """Register a data source for cache warming"""
        self.data_sources[name] = source_func

    async def start_warming(self):
        """Start background cache warming"""
        async with self._lock:
            if self._warming_active:
                return

            self._warming_active = True
            self._warming_task = asyncio.create_task(self._warming_loop())
            logger.info("Cache warming started")

    async def stop_warming(self):
        """Stop background cache warming"""
        async with self._lock:
            self._warming_active = False

            if self._warming_task:
                self._warming_task.cancel()
                try:
                    await self._warming_task
                except asyncio.CancelledError:
                    pass

            logger.info("Cache warming stopped")

    async def warm_cache_now(self, strategy_name: Optional[str] = None) -> WarmingStats:
        """Perform immediate cache warming"""
        start_time = time.time()

        try:
            strategies_to_run = self.strategies
            if strategy_name:
                strategies_to_run = [s for s in self.strategies if s.name == strategy_name]

            if not strategies_to_run:
                logger.warning(f"No warming strategies found")
                return self.stats

            # Sort by priority
            strategies_to_run.sort(key=lambda x: x.priority, reverse=True)

            total_warmed = 0
            successful_warms = 0
            failed_warms = 0

            for strategy in strategies_to_run:
                if not strategy.enabled:
                    continue

                try:
                    warmed, succeeded, failed = await self._execute_strategy(strategy)
                    total_warmed += warmed
                    successful_warms += succeeded
                    failed_warms += failed

                except Exception as e:
                    logger.error(f"Strategy {strategy.name} failed: {e}")
                    failed_warms += 1

            # Update stats
            self.stats.total_warmed += total_warmed
            self.stats.successful_warms += successful_warms
            self.stats.failed_warms += failed_warms
            self.stats.last_warming_time = time.time()
            self.stats.warming_duration_ms = (time.time() - start_time) * 1000

            logger.info(f"Cache warming completed: {successful_warms} successful, {failed_warms} failed")
            return self.stats

        except Exception as e:
            logger.error(f"Cache warming error: {e}")
            self.stats.failed_warms += 1
            return self.stats

    async def predict_and_warm(self, count: int = 100) -> int:
        """Use pattern analysis to predict and warm likely keys"""
        try:
            predicted_keys = await self.pattern_analyzer.predict_keys(count)
            self.stats.keys_predicted = len(predicted_keys)

            warmed_count = 0
            successful_predictions = 0

            for key in predicted_keys:
                try:
                    # Check if key is already cached
                    cached_value = await self.cache_manager.get(key)
                    if cached_value is not None:
                        successful_predictions += 1
                        continue

                    # Try to warm the key
                    if await self._warm_key(key):
                        warmed_count += 1

                except Exception as e:
                    logger.error(f"Failed to warm predicted key {key}: {e}")

            # Update prediction accuracy
            if self.stats.keys_predicted > 0:
                self.stats.prediction_accuracy = successful_predictions / self.stats.keys_predicted

            logger.info(f"Predictive warming: {warmed_count} keys warmed, {successful_predictions}/{len(predicted_keys)} predictions accurate")
            return warmed_count

        except Exception as e:
            logger.error(f"Predictive warming error: {e}")
            return 0

    async def record_access(self, key: str):
        """Record cache access for pattern analysis"""
        await self.pattern_analyzer.record_access(key)

    async def get_warming_stats(self) -> Dict[str, Any]:
        """Get comprehensive warming statistics"""
        pattern_insights = await self.pattern_analyzer.get_pattern_insights()

        return {
            'warming_stats': {
                'total_warmed': self.stats.total_warmed,
                'successful_warms': self.stats.successful_warms,
                'failed_warms': self.stats.failed_warms,
                'success_rate': (
                    self.stats.successful_warms / max(self.stats.total_warmed, 1)
                ),
                'last_warming_time': self.stats.last_warming_time,
                'warming_duration_ms': self.stats.warming_duration_ms,
                'prediction_accuracy': self.stats.prediction_accuracy
            },
            'strategies': [
                {
                    'name': s.name,
                    'priority': s.priority,
                    'max_keys': s.max_keys,
                    'enabled': s.enabled,
                    'warm_interval': s.warm_interval
                }
                for s in self.strategies
            ],
            'pattern_insights': pattern_insights,
            'data_sources': list(self.data_sources.keys())
        }

    async def _execute_strategy(self, strategy: WarmingStrategy) -> Tuple[int, int, int]:
        """Execute a specific warming strategy"""
        logger.debug(f"Executing warming strategy: {strategy.name}")

        warmed_count = 0
        successful_count = 0
        failed_count = 0

        try:
            # Get keys to warm based on strategy
            keys_to_warm = await self._get_keys_for_strategy(strategy)

            for key in keys_to_warm[:strategy.max_keys]:
                try:
                    if await self._warm_key(key):
                        successful_count += 1
                    else:
                        failed_count += 1
                    warmed_count += 1

                except Exception as e:
                    logger.error(f"Failed to warm key {key}: {e}")
                    failed_count += 1

        except Exception as e:
            logger.error(f"Strategy execution error: {e}")
            failed_count += 1

        return warmed_count, successful_count, failed_count

    async def _get_keys_for_strategy(self, strategy: WarmingStrategy) -> List[str]:
        """Get keys to warm for a specific strategy"""
        if strategy.name == "frequency_based":
            return await self.pattern_analyzer.predict_keys(strategy.max_keys)
        elif strategy.name == "time_based":
            current_hour = int((time.time() % 86400) // 3600)
            return await self.pattern_analyzer._predict_by_time(current_hour, strategy.max_keys)
        elif strategy.key_pattern:
            # Pattern-based strategy (would need integration with cache key listing)
            return []
        else:
            # Default: use pattern prediction
            return await self.pattern_analyzer.predict_keys(strategy.max_keys)

    async def _warm_key(self, key: str) -> bool:
        """Warm a specific cache key"""
        try:
            # Check if already cached
            cached_value = await self.cache_manager.get(key)
            if cached_value is not None:
                return True

            # Try each data source to get the value
            for source_name, source_func in self.data_sources.items():
                try:
                    value = await source_func(key)
                    if value is not None:
                        # Cache the value
                        success = await self.cache_manager.set(key, value)
                        if success:
                            logger.debug(f"Warmed key {key} from source {source_name}")
                            return True
                except Exception as e:
                    logger.debug(f"Data source {source_name} failed for key {key}: {e}")

            return False

        except Exception as e:
            logger.error(f"Key warming error for {key}: {e}")
            return False

    async def _warming_loop(self):
        """Background warming loop"""
        while self._warming_active:
            try:
                # Check each strategy's schedule
                current_time = time.time()

                for strategy in self.strategies:
                    if not strategy.enabled:
                        continue

                    # Check if it's time to run this strategy
                    time_since_last = current_time - self.stats.last_warming_time
                    if time_since_last >= strategy.warm_interval:
                        await self._execute_strategy(strategy)

                # Sleep between checks
                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Warming loop error: {e}")
                await asyncio.sleep(60)

    async def _load_configuration(self):
        """Load warming configuration from disk"""
        try:
            config_file = self.config_dir / "warming_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = json.load(f)

                self.strategies = [
                    WarmingStrategy(**strategy_data)
                    for strategy_data in config_data.get('strategies', [])
                ]

                logger.info(f"Loaded {len(self.strategies)} warming strategies")
            else:
                # Create default strategies
                await self._create_default_strategies()

        except Exception as e:
            logger.error(f"Failed to load warming configuration: {e}")
            await self._create_default_strategies()

    async def _save_configuration(self):
        """Save warming configuration to disk"""
        try:
            config_file = self.config_dir / "warming_config.json"
            config_data = {
                'strategies': [
                    {
                        'name': s.name,
                        'priority': s.priority,
                        'max_keys': s.max_keys,
                        'key_pattern': s.key_pattern,
                        'warm_interval': s.warm_interval,
                        'enabled': s.enabled
                    }
                    for s in self.strategies
                ]
            }

            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save warming configuration: {e}")

    async def _create_default_strategies(self):
        """Create default warming strategies"""
        default_strategies = [
            WarmingStrategy(
                name="frequency_based",
                priority=3,
                max_keys=100,
                warm_interval=1800,  # 30 minutes
                enabled=True
            ),
            WarmingStrategy(
                name="time_based",
                priority=2,
                max_keys=50,
                warm_interval=3600,  # 1 hour
                enabled=True
            ),
            WarmingStrategy(
                name="predictive",
                priority=1,
                max_keys=25,
                warm_interval=7200,  # 2 hours
                enabled=True
            )
        ]

        self.strategies = default_strategies
        await self._save_configuration()
        logger.info("Created default warming strategies")