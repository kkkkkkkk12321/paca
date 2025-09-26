"""
LRU Cache Implementation for PACA

This module provides a high-performance Least Recently Used (LRU) cache
with advanced features like memory management, statistics tracking,
and thread-safe operations.

Features:
- O(1) get, set, delete operations
- Memory usage tracking and limits
- Access pattern analytics
- Automatic eviction with LRU policy
- Thread-safe async operations
"""

import asyncio
import logging
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
import weakref
import gc

from ..base import BaseDataComponent
from .cache_metrics import CacheMetrics

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    size_bytes: int


class LRUCache(BaseDataComponent):
    """
    High-performance LRU Cache with memory management

    Features:
    - O(1) operations using OrderedDict
    - Memory usage tracking
    - Automatic eviction based on LRU policy
    - Access pattern analytics
    - Thread-safe async operations
    - Memory limit enforcement
    """

    def __init__(self, max_size_mb: int = 100, max_items: int = 10000):
        super().__init__()
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_items = max_items

        # Core storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

        # Statistics
        self.metrics = CacheMetrics()
        self._current_size_bytes = 0
        self._initialized = False

        # Performance tracking
        self._access_times: List[float] = []
        self._eviction_count = 0

    async def initialize(self) -> bool:
        """Initialize the LRU cache"""
        try:
            async with self._lock:
                if self._initialized:
                    return True

                self._cache = OrderedDict()
                self._current_size_bytes = 0
                self.metrics = CacheMetrics()
                self._access_times = []
                self._eviction_count = 0

                self._initialized = True
                logger.info(f"LRUCache initialized - Max: {self.max_size_bytes / (1024*1024):.1f}MB")
                return True

        except Exception as e:
            logger.error(f"LRUCache initialization failed: {e}")
            return False

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache and mark as recently used

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            async with self._lock:
                if key not in self._cache:
                    self.metrics.record_miss(time.time() - start_time)
                    return None

                # Move to end (most recently used)
                entry = self._cache.pop(key)
                entry.accessed_at = time.time()
                entry.access_count += 1
                self._cache[key] = entry

                # Track access time
                self._access_times.append(time.time() - start_time)
                if len(self._access_times) > 1000:
                    self._access_times = self._access_times[-500:]

                self.metrics.record_hit('l1', time.time() - start_time)
                return entry.value

        except Exception as e:
            logger.error(f"LRUCache get error for key {key}: {e}")
            self.metrics.record_error()
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache with automatic eviction if needed

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live (not used in LRU, but kept for interface compatibility)

        Returns:
            True if successfully cached
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Calculate size
            value_size = self._calculate_size(value)

            async with self._lock:
                # Check if update
                if key in self._cache:
                    old_entry = self._cache.pop(key)
                    self._current_size_bytes -= old_entry.size_bytes

                # Create new entry
                entry = CacheEntry(
                    value=value,
                    created_at=time.time(),
                    accessed_at=time.time(),
                    access_count=1,
                    size_bytes=value_size
                )

                # Check if we need to evict
                await self._ensure_capacity(value_size)

                # Add new entry
                self._cache[key] = entry
                self._current_size_bytes += value_size

                self.metrics.record_set(time.time() - start_time)
                return True

        except Exception as e:
            logger.error(f"LRUCache set error for key {key}: {e}")
            self.metrics.record_error()
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self._lock:
                if key in self._cache:
                    entry = self._cache.pop(key)
                    self._current_size_bytes -= entry.size_bytes
                    return True
                return False

        except Exception as e:
            logger.error(f"LRUCache delete error for key {key}: {e}")
            return False

    async def invalidate(self, pattern: str) -> int:
        """Invalidate keys matching pattern"""
        if not self._initialized:
            await self.initialize()

        try:
            deleted_count = 0
            async with self._lock:
                keys_to_delete = []

                for key in self._cache.keys():
                    if self._matches_pattern(key, pattern):
                        keys_to_delete.append(key)

                for key in keys_to_delete:
                    entry = self._cache.pop(key)
                    self._current_size_bytes -= entry.size_bytes
                    deleted_count += 1

            return deleted_count

        except Exception as e:
            logger.error(f"LRUCache invalidate error for pattern {pattern}: {e}")
            return 0

    async def clear(self) -> bool:
        """Clear all cache entries"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self._lock:
                self._cache.clear()
                self._current_size_bytes = 0
                self.metrics.reset()
                self._access_times = []
                self._eviction_count = 0

            # Force garbage collection
            gc.collect()
            return True

        except Exception as e:
            logger.error(f"LRUCache clear error: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self._lock:
                stats = self.metrics.get_stats().__dict__.copy()

                # Add LRU-specific stats
                stats.update({
                    'cache_size_items': len(self._cache),
                    'cache_size_bytes': self._current_size_bytes,
                    'cache_size_mb': self._current_size_bytes / (1024 * 1024),
                    'max_size_mb': self.max_size_bytes / (1024 * 1024),
                    'memory_usage_percent': (self._current_size_bytes / self.max_size_bytes) * 100,
                    'eviction_count': self._eviction_count,
                    'avg_access_time_ms': (
                        sum(self._access_times) / len(self._access_times) * 1000
                        if self._access_times else 0
                    ),
                    'hottest_keys': await self._get_hottest_keys(5),
                    'memory_efficiency': await self._calculate_memory_efficiency()
                })

                return stats

        except Exception as e:
            logger.error(f"LRUCache stats error: {e}")
            return {}

    async def optimize(self) -> bool:
        """Optimize cache performance"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self._lock:
                # Remove entries with very low access count if memory is high
                if self._current_size_bytes > self.max_size_bytes * 0.8:
                    await self._evict_cold_entries()

                # Compact internal structures
                self._access_times = self._access_times[-100:] if self._access_times else []

            # Force garbage collection
            gc.collect()
            return True

        except Exception as e:
            logger.error(f"LRUCache optimization error: {e}")
            return False

    async def _ensure_capacity(self, needed_bytes: int):
        """Ensure sufficient capacity by evicting LRU entries"""
        while (
            len(self._cache) >= self.max_items or
            self._current_size_bytes + needed_bytes > self.max_size_bytes
        ):
            if not self._cache:
                break

            # Evict least recently used (first item)
            lru_key, lru_entry = self._cache.popitem(last=False)
            self._current_size_bytes -= lru_entry.size_bytes
            self._eviction_count += 1

            logger.debug(f"Evicted LRU key: {lru_key[:50]}...")

    async def _evict_cold_entries(self):
        """Evict entries with low access count"""
        if not self._cache:
            return

        # Calculate median access count
        access_counts = [entry.access_count for entry in self._cache.values()]
        median_access = sorted(access_counts)[len(access_counts) // 2]

        # Evict entries with access count below median
        keys_to_evict = []
        for key, entry in self._cache.items():
            if entry.access_count < median_access and len(keys_to_evict) < len(self._cache) // 4:
                keys_to_evict.append(key)

        for key in keys_to_evict:
            entry = self._cache.pop(key)
            self._current_size_bytes -= entry.size_bytes
            self._eviction_count += 1

    async def _get_hottest_keys(self, count: int) -> List[Dict[str, Any]]:
        """Get most frequently accessed keys"""
        if not self._cache:
            return []

        # Sort by access count
        sorted_items = sorted(
            self._cache.items(),
            key=lambda x: x[1].access_count,
            reverse=True
        )

        return [
            {
                'key': key[:50] + '...' if len(key) > 50 else key,
                'access_count': entry.access_count,
                'size_bytes': entry.size_bytes,
                'age_seconds': time.time() - entry.created_at
            }
            for key, entry in sorted_items[:count]
        ]

    async def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency score"""
        if not self._cache:
            return 1.0

        total_accesses = sum(entry.access_count for entry in self._cache.values())
        total_size = self._current_size_bytes

        if total_size == 0:
            return 1.0

        # Efficiency = accesses per MB
        efficiency = (total_accesses / (total_size / (1024 * 1024)))
        return min(efficiency / 1000, 1.0)  # Normalize to 0-1

    def _calculate_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        try:
            return sys.getsizeof(obj)
        except (TypeError, OverflowError):
            # Fallback for complex objects
            return len(str(obj)) * 2  # Rough estimate

    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern (simple wildcard support)"""
        if '*' not in pattern:
            return key == pattern

        # Simple wildcard matching
        if pattern.startswith('*') and pattern.endswith('*'):
            return pattern[1:-1] in key
        elif pattern.startswith('*'):
            return key.endswith(pattern[1:])
        elif pattern.endswith('*'):
            return key.startswith(pattern[:-1])
        else:
            return key == pattern

    async def shutdown(self):
        """Shutdown cache and cleanup resources"""
        try:
            async with self._lock:
                self._cache.clear()
                self._current_size_bytes = 0
                self._access_times = []

            gc.collect()
            logger.info("LRUCache shutdown completed")

        except Exception as e:
            logger.error(f"LRUCache shutdown error: {e}")


# Specialized LRU cache variants

class LFUCache(LRUCache):
    """Least Frequently Used cache variant"""

    async def _ensure_capacity(self, needed_bytes: int):
        """Ensure capacity by evicting least frequently used entries"""
        while (
            len(self._cache) >= self.max_items or
            self._current_size_bytes + needed_bytes > self.max_size_bytes
        ):
            if not self._cache:
                break

            # Find least frequently used entry
            lfu_key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
            lfu_entry = self._cache.pop(lfu_key)
            self._current_size_bytes -= lfu_entry.size_bytes
            self._eviction_count += 1

            logger.debug(f"Evicted LFU key: {lfu_key[:50]}...")


class FIFOCache(LRUCache):
    """First In First Out cache variant"""

    async def get(self, key: str) -> Optional[Any]:
        """Get value without updating access order (FIFO behavior)"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            async with self._lock:
                if key not in self._cache:
                    self.metrics.record_miss(time.time() - start_time)
                    return None

                entry = self._cache[key]
                entry.accessed_at = time.time()
                entry.access_count += 1

                # Don't move to end (maintain insertion order)

                self.metrics.record_hit('l1', time.time() - start_time)
                return entry.value

        except Exception as e:
            logger.error(f"FIFOCache get error for key {key}: {e}")
            self.metrics.record_error()
            return None