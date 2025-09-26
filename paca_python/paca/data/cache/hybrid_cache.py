"""
Hybrid Cache Implementation for PACA

This module provides a hybrid cache that combines memory and disk storage
with intelligent promotion/demotion policies.

Features:
- Memory + Disk hybrid storage
- Intelligent cache promotion/demotion
- Adaptive sizing based on access patterns
- Hot/Cold data classification
- Background optimization
- Memory pressure handling
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
import sys

from ..base import BaseDataComponent
from .lru_cache import LRUCache
from .ttl_cache import TTLCache
from .cache_metrics import CacheMetrics

logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    """Hybrid cache configuration"""
    memory_ratio: float = 0.3           # Memory:Disk ratio (30:70)
    promotion_threshold: int = 3        # Access count for promotion to memory
    demotion_threshold: int = 300       # Seconds without access for demotion
    hot_data_ttl: int = 1800           # TTL for hot data (30 minutes)
    cold_data_ttl: int = 7200          # TTL for cold data (2 hours)
    optimize_interval: int = 300        # Background optimization interval


class HybridCache(BaseDataComponent):
    """
    Hybrid cache combining memory and disk storage

    Architecture:
    - Hot data: Memory (LRU) for frequently accessed items
    - Cold data: Disk (TTL) for less frequently accessed items
    - Intelligent promotion/demotion based on access patterns

    Features:
    - Automatic hot/cold data classification
    - Memory pressure-aware promotion/demotion
    - Background optimization and rebalancing
    - Adaptive memory/disk ratio based on workload
    """

    def __init__(
        self,
        total_size_mb: int = 1000,
        cache_dir: str = "hybrid_cache",
        config: Optional[HybridConfig] = None
    ):
        super().__init__()
        self.total_size_mb = total_size_mb
        self.cache_dir = cache_dir
        self.config = config or HybridConfig()

        # Calculate memory and disk sizes
        self.memory_size_mb = int(total_size_mb * self.config.memory_ratio)
        self.disk_size_mb = total_size_mb - self.memory_size_mb

        # Cache layers
        self.memory_cache: Optional[LRUCache] = None
        self.disk_cache: Optional[TTLCache] = None

        # Access tracking
        self._access_tracker: Dict[str, Dict[str, Any]] = {}
        self._promotion_candidates: Set[str] = set()
        self._demotion_candidates: Set[str] = set()

        # Statistics
        self.metrics = CacheMetrics()
        self._lock = asyncio.Lock()
        self._initialized = False

        # Background tasks
        self._optimization_task: Optional[asyncio.Task] = None
        self._optimization_running = False

    async def initialize(self) -> bool:
        """Initialize hybrid cache components"""
        try:
            async with self._lock:
                if self._initialized:
                    return True

                # Initialize memory cache
                self.memory_cache = LRUCache(
                    max_size_mb=self.memory_size_mb,
                    max_items=10000
                )
                await self.memory_cache.initialize()

                # Initialize disk cache
                from pathlib import Path
                self.disk_cache = TTLCache(
                    cache_dir=Path(self.cache_dir),
                    max_size_mb=self.disk_size_mb,
                    default_ttl=self.config.cold_data_ttl
                )
                await self.disk_cache.initialize()

                # Initialize tracking
                self._access_tracker = {}
                self._promotion_candidates = set()
                self._demotion_candidates = set()
                self.metrics = CacheMetrics()

                # Start background optimization
                await self._start_optimization_task()

                self._initialized = True
                logger.info(f"HybridCache initialized - Memory: {self.memory_size_mb}MB, Disk: {self.disk_size_mb}MB")
                return True

        except Exception as e:
            logger.error(f"HybridCache initialization failed: {e}")
            return False

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from hybrid cache with intelligent promotion

        Search order: Memory → Disk
        Tracks access patterns for promotion decisions
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Update access tracking
            await self._track_access(key)

            # Try memory first
            value = await self.memory_cache.get(key)
            if value is not None:
                self.metrics.record_hit('memory', time.time() - start_time)
                return value

            # Try disk
            value = await self.disk_cache.get(key)
            if value is not None:
                # Consider promotion based on access pattern
                await self._evaluate_promotion(key)
                self.metrics.record_hit('disk', time.time() - start_time)
                return value

            # Cache miss
            self.metrics.record_miss(time.time() - start_time)
            return None

        except Exception as e:
            logger.error(f"HybridCache get error for key {key}: {e}")
            self.metrics.record_error()
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in hybrid cache with intelligent placement

        Hot data → Memory
        Cold data → Disk
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Determine initial placement
            is_hot = await self._is_hot_data(key, value)

            if is_hot:
                # Store in memory with shorter TTL
                hot_ttl = ttl or self.config.hot_data_ttl
                success = await self.memory_cache.set(key, value, hot_ttl)

                # Also store in disk as backup
                if success:
                    await self.disk_cache.set(key, value, hot_ttl)

            else:
                # Store in disk
                cold_ttl = ttl or self.config.cold_data_ttl
                success = await self.disk_cache.set(key, value, cold_ttl)

            # Update access tracking
            await self._track_access(key, is_write=True)

            if success:
                self.metrics.record_set(time.time() - start_time)

            return success

        except Exception as e:
            logger.error(f"HybridCache set error for key {key}: {e}")
            self.metrics.record_error()
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from both cache layers"""
        if not self._initialized:
            await self.initialize()

        try:
            # Remove from both layers
            memory_success = await self.memory_cache.delete(key)
            disk_success = await self.disk_cache.delete(key)

            # Remove from tracking
            if key in self._access_tracker:
                del self._access_tracker[key]

            self._promotion_candidates.discard(key)
            self._demotion_candidates.discard(key)

            return memory_success or disk_success

        except Exception as e:
            logger.error(f"HybridCache delete error for key {key}: {e}")
            return False

    async def invalidate(self, pattern: str) -> int:
        """Invalidate keys matching pattern from both layers"""
        if not self._initialized:
            await self.initialize()

        try:
            # Invalidate from both layers
            memory_count = await self.memory_cache.invalidate(pattern)
            disk_count = await self.disk_cache.invalidate(pattern)

            # Remove from tracking
            keys_to_remove = [
                key for key in self._access_tracker.keys()
                if self._matches_pattern(key, pattern)
            ]

            for key in keys_to_remove:
                del self._access_tracker[key]
                self._promotion_candidates.discard(key)
                self._demotion_candidates.discard(key)

            return memory_count + disk_count

        except Exception as e:
            logger.error(f"HybridCache invalidate error for pattern {pattern}: {e}")
            return 0

    async def clear(self) -> bool:
        """Clear both cache layers"""
        if not self._initialized:
            await self.initialize()

        try:
            memory_success = await self.memory_cache.clear()
            disk_success = await self.disk_cache.clear()

            # Clear tracking
            self._access_tracker.clear()
            self._promotion_candidates.clear()
            self._demotion_candidates.clear()
            self.metrics.reset()

            return memory_success and disk_success

        except Exception as e:
            logger.error(f"HybridCache clear error: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive hybrid cache statistics"""
        if not self._initialized:
            await self.initialize()

        try:
            base_stats = self.metrics.get_stats().__dict__.copy()

            # Get component stats
            memory_stats = await self.memory_cache.get_stats()
            disk_stats = await self.disk_cache.get_stats()

            # Calculate hybrid-specific metrics
            total_items = memory_stats.get('cache_size_items', 0) + disk_stats.get('cache_size_items', 0)
            memory_hit_ratio = memory_stats.get('hit_rate', 0) * memory_stats.get('cache_size_items', 0)
            disk_hit_ratio = disk_stats.get('hit_rate', 0) * disk_stats.get('cache_size_items', 0)
            overall_hit_ratio = (memory_hit_ratio + disk_hit_ratio) / max(total_items, 1)

            base_stats.update({
                'total_items': total_items,
                'memory_items': memory_stats.get('cache_size_items', 0),
                'disk_items': disk_stats.get('cache_size_items', 0),
                'memory_size_mb': memory_stats.get('cache_size_mb', 0),
                'disk_size_mb': disk_stats.get('cache_size_mb', 0),
                'memory_hit_rate': memory_stats.get('hit_rate', 0),
                'disk_hit_rate': disk_stats.get('hit_rate', 0),
                'overall_hit_rate': overall_hit_ratio,
                'promotion_candidates': len(self._promotion_candidates),
                'demotion_candidates': len(self._demotion_candidates),
                'tracked_keys': len(self._access_tracker),
                'memory_efficiency': await self._calculate_memory_efficiency(),
                'hot_cold_ratio': await self._calculate_hot_cold_ratio()
            })

            return base_stats

        except Exception as e:
            logger.error(f"HybridCache stats error: {e}")
            return {}

    async def optimize(self) -> bool:
        """Optimize cache by promoting/demoting entries"""
        if not self._initialized:
            await self.initialize()

        try:
            # Process promotions
            await self._process_promotions()

            # Process demotions
            await self._process_demotions()

            # Optimize individual caches
            await self.memory_cache.optimize()
            await self.disk_cache.optimize()

            # Cleanup old tracking data
            await self._cleanup_access_tracker()

            return True

        except Exception as e:
            logger.error(f"HybridCache optimization error: {e}")
            return False

    async def _track_access(self, key: str, is_write: bool = False):
        """Track access pattern for key"""
        current_time = time.time()

        if key not in self._access_tracker:
            self._access_tracker[key] = {
                'access_count': 0,
                'first_access': current_time,
                'last_access': current_time,
                'total_accesses': 0,
                'write_count': 0
            }

        tracker = self._access_tracker[key]
        tracker['access_count'] += 1
        tracker['total_accesses'] += 1
        tracker['last_access'] = current_time

        if is_write:
            tracker['write_count'] += 1

        # Update access frequency (exponential smoothing)
        time_diff = current_time - tracker['first_access']
        if time_diff > 0:
            tracker['access_frequency'] = tracker['total_accesses'] / time_diff

    async def _evaluate_promotion(self, key: str):
        """Evaluate if key should be promoted to memory"""
        if key not in self._access_tracker:
            return

        tracker = self._access_tracker[key]

        # Promotion criteria
        should_promote = (
            tracker['access_count'] >= self.config.promotion_threshold and
            key not in self._promotion_candidates and
            not await self._is_in_memory(key)
        )

        if should_promote:
            self._promotion_candidates.add(key)

    async def _evaluate_demotion(self, key: str):
        """Evaluate if key should be demoted to disk"""
        if key not in self._access_tracker:
            return

        tracker = self._access_tracker[key]
        current_time = time.time()

        # Demotion criteria
        time_since_access = current_time - tracker['last_access']
        should_demote = (
            time_since_access >= self.config.demotion_threshold and
            await self._is_in_memory(key) and
            key not in self._demotion_candidates
        )

        if should_demote:
            self._demotion_candidates.add(key)

    async def _process_promotions(self):
        """Process promotion candidates"""
        if not self._promotion_candidates:
            return

        promoted_count = 0
        candidates = list(self._promotion_candidates)
        self._promotion_candidates.clear()

        for key in candidates:
            try:
                # Get from disk
                value = await self.disk_cache.get(key)
                if value is not None:
                    # Promote to memory
                    if await self.memory_cache.set(key, value, self.config.hot_data_ttl):
                        promoted_count += 1

            except Exception as e:
                logger.error(f"Promotion error for key {key}: {e}")

        if promoted_count > 0:
            logger.debug(f"Promoted {promoted_count} entries to memory")

    async def _process_demotions(self):
        """Process demotion candidates"""
        if not self._demotion_candidates:
            return

        demoted_count = 0
        candidates = list(self._demotion_candidates)
        self._demotion_candidates.clear()

        for key in candidates:
            try:
                # Check if still in memory
                value = await self.memory_cache.get(key)
                if value is not None:
                    # Ensure it's in disk cache
                    await self.disk_cache.set(key, value, self.config.cold_data_ttl)

                    # Remove from memory
                    if await self.memory_cache.delete(key):
                        demoted_count += 1

            except Exception as e:
                logger.error(f"Demotion error for key {key}: {e}")

        if demoted_count > 0:
            logger.debug(f"Demoted {demoted_count} entries to disk")

    async def _is_hot_data(self, key: str, value: Any) -> bool:
        """Determine if data should be considered 'hot'"""
        # Check access pattern
        if key in self._access_tracker:
            tracker = self._access_tracker[key]
            return tracker['access_count'] >= self.config.promotion_threshold

        # For new data, use heuristics
        value_size = sys.getsizeof(value)

        # Small, frequently accessed data types are likely hot
        if value_size < 1024:  # < 1KB
            return True

        # Large data starts cold
        if value_size > 100 * 1024:  # > 100KB
            return False

        # Medium data is neutral (goes to disk first)
        return False

    async def _is_in_memory(self, key: str) -> bool:
        """Check if key is in memory cache"""
        try:
            value = await self.memory_cache.get(key)
            return value is not None
        except Exception:
            return False

    async def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency score"""
        try:
            memory_stats = await self.memory_cache.get_stats()
            memory_items = memory_stats.get('cache_size_items', 0)
            memory_hits = memory_stats.get('hits', 0)

            if memory_items == 0:
                return 1.0

            # Efficiency = hits per item
            efficiency = memory_hits / memory_items
            return min(efficiency / 10, 1.0)  # Normalize to 0-1

        except Exception:
            return 0.5

    async def _calculate_hot_cold_ratio(self) -> float:
        """Calculate ratio of hot to cold data"""
        try:
            memory_stats = await self.memory_cache.get_stats()
            disk_stats = await self.disk_cache.get_stats()

            memory_items = memory_stats.get('cache_size_items', 0)
            disk_items = disk_stats.get('cache_size_items', 0)

            total_items = memory_items + disk_items
            if total_items == 0:
                return 0.0

            return memory_items / total_items

        except Exception:
            return 0.0

    async def _cleanup_access_tracker(self):
        """Cleanup old access tracking data"""
        current_time = time.time()
        cleanup_threshold = 24 * 3600  # 24 hours

        keys_to_remove = []
        for key, tracker in self._access_tracker.items():
            if current_time - tracker['last_access'] > cleanup_threshold:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._access_tracker[key]
            self._promotion_candidates.discard(key)
            self._demotion_candidates.discard(key)

        if keys_to_remove:
            logger.debug(f"Cleaned up {len(keys_to_remove)} old access tracking entries")

    async def _start_optimization_task(self):
        """Start background optimization task"""
        if self._optimization_task and not self._optimization_task.done():
            return

        self._optimization_running = True
        self._optimization_task = asyncio.create_task(self._optimization_loop())

    async def _optimization_loop(self):
        """Background optimization loop"""
        while self._optimization_running:
            try:
                await asyncio.sleep(self.config.optimize_interval)
                if self._optimization_running:
                    await self.optimize()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")

    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern"""
        if '*' not in pattern:
            return key == pattern

        if pattern.startswith('*') and pattern.endswith('*'):
            return pattern[1:-1] in key
        elif pattern.startswith('*'):
            return key.endswith(pattern[1:])
        elif pattern.endswith('*'):
            return key.startswith(pattern[:-1])
        else:
            return key == pattern

    async def shutdown(self):
        """Shutdown hybrid cache"""
        try:
            # Stop optimization task
            self._optimization_running = False
            if self._optimization_task:
                self._optimization_task.cancel()
                try:
                    await self._optimization_task
                except asyncio.CancelledError:
                    pass

            # Shutdown component caches
            if self.memory_cache:
                await self.memory_cache.shutdown()

            if self.disk_cache:
                await self.disk_cache.shutdown()

            # Clear tracking data
            self._access_tracker.clear()
            self._promotion_candidates.clear()
            self._demotion_candidates.clear()

            logger.info("HybridCache shutdown completed")

        except Exception as e:
            logger.error(f"HybridCache shutdown error: {e}")