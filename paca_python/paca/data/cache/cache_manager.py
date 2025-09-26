"""
Cache Manager - Central cache orchestrator for PACA system

This module provides the main CacheManager class that coordinates
between multiple cache levels (L1: Memory, L2: Disk, L3: Redis).

Features:
- Multi-level cache hierarchy
- Intelligent cache promotion/demotion
- Automatic failover and recovery
- Performance monitoring
- Memory usage management
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import pickle
import hashlib
from pathlib import Path

from ..base import BaseDataComponent
from .lru_cache import LRUCache
from .ttl_cache import TTLCache
from .cache_metrics import CacheMetrics, CacheStats

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    TTL = "ttl"           # Time To Live
    FIFO = "fifo"         # First In First Out
    HYBRID = "hybrid"     # TTL + LRU 조합


@dataclass
class CacheConfig:
    """Cache configuration settings"""
    l1_max_size_mb: int = 100          # L1 메모리 캐시 최대 크기 (MB)
    l2_max_size_mb: int = 1000         # L2 디스크 캐시 최대 크기 (MB)
    default_ttl: int = 3600            # 기본 TTL (초)
    l1_policy: CachePolicy = CachePolicy.LRU
    l2_policy: CachePolicy = CachePolicy.TTL
    enable_l3: bool = False            # Redis 캐시 활성화
    redis_host: str = "localhost"
    redis_port: int = 6379
    cache_dir: str = "cache_storage"
    enable_compression: bool = True
    auto_warming: bool = True


class CacheManager(BaseDataComponent):
    """
    3-tier cache manager providing unified cache interface

    Architecture:
    L1 (Memory) → L2 (Disk) → L3 (Redis/Network) → Source

    Features:
    - Automatic cache promotion (L2→L1, L3→L2)
    - Intelligent eviction policies
    - Performance monitoring
    - Memory management
    - Async operations
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        super().__init__()
        self.config = config or CacheConfig()
        self.metrics = CacheMetrics()

        # Cache levels
        self.l1_cache: Optional[LRUCache] = None
        self.l2_cache: Optional[TTLCache] = None
        self.l3_cache: Optional[Any] = None  # Redis client

        # Internal state
        self._initialized = False
        self._cache_dir = Path(self.config.cache_dir)
        self._lock = asyncio.Lock()

    async def initialize(self) -> bool:
        """Initialize all cache levels"""
        try:
            async with self._lock:
                if self._initialized:
                    return True

                # Create cache directory
                self._cache_dir.mkdir(parents=True, exist_ok=True)

                # Initialize L1 (Memory) cache
                self.l1_cache = LRUCache(
                    max_size_mb=self.config.l1_max_size_mb,
                    policy=self.config.l1_policy
                )
                await self.l1_cache.initialize()

                # Initialize L2 (Disk) cache
                self.l2_cache = TTLCache(
                    cache_dir=self._cache_dir / "l2",
                    max_size_mb=self.config.l2_max_size_mb,
                    default_ttl=self.config.default_ttl
                )
                await self.l2_cache.initialize()

                # Initialize L3 (Redis) cache if enabled
                if self.config.enable_l3:
                    await self._init_l3_cache()

                self._initialized = True
                logger.info("CacheManager initialized successfully")
                return True

        except Exception as e:
            logger.error(f"Failed to initialize CacheManager: {e}")
            return False

    async def _init_l3_cache(self):
        """Initialize Redis L3 cache"""
        try:
            import redis.asyncio as redis
            self.l3_cache = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                decode_responses=False
            )
            # Test connection
            await self.l3_cache.ping()
            logger.info("L3 Redis cache initialized")
        except ImportError:
            logger.warning("Redis not available, L3 cache disabled")
            self.l3_cache = None
        except Exception as e:
            logger.warning(f"L3 cache initialization failed: {e}")
            self.l3_cache = None

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache with automatic promotion

        Search order: L1 → L2 → L3 → None
        Promotes cache hits to higher levels
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Try L1 first
            value = await self._get_l1(key)
            if value is not None:
                self.metrics.record_hit('l1', time.time() - start_time)
                return value

            # Try L2
            value = await self._get_l2(key)
            if value is not None:
                # Promote to L1
                await self._set_l1(key, value)
                self.metrics.record_hit('l2', time.time() - start_time)
                return value

            # Try L3
            if self.l3_cache:
                value = await self._get_l3(key)
                if value is not None:
                    # Promote to L2 and L1
                    await self._set_l2(key, value)
                    await self._set_l1(key, value)
                    self.metrics.record_hit('l3', time.time() - start_time)
                    return value

            # Cache miss
            self.metrics.record_miss(time.time() - start_time)
            return None

        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self.metrics.record_error()
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in all cache levels

        Stores in all available cache levels with appropriate TTL
        """
        if not self._initialized:
            await self.initialize()

        ttl = ttl or self.config.default_ttl
        start_time = time.time()

        try:
            # Set in all levels
            success_l1 = await self._set_l1(key, value, ttl)
            success_l2 = await self._set_l2(key, value, ttl)
            success_l3 = await self._set_l3(key, value, ttl) if self.l3_cache else True

            success = success_l1 or success_l2 or success_l3

            if success:
                self.metrics.record_set(time.time() - start_time)
            else:
                self.metrics.record_error()

            return success

        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            self.metrics.record_error()
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from all cache levels"""
        if not self._initialized:
            await self.initialize()

        try:
            # Delete from all levels
            success_l1 = await self._delete_l1(key)
            success_l2 = await self._delete_l2(key)
            success_l3 = await self._delete_l3(key) if self.l3_cache else True

            return success_l1 or success_l2 or success_l3

        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False

    async def invalidate(self, pattern: str) -> int:
        """Invalidate keys matching pattern"""
        if not self._initialized:
            await self.initialize()

        total_deleted = 0

        try:
            # Invalidate from all levels
            if self.l1_cache:
                total_deleted += await self.l1_cache.invalidate(pattern)

            if self.l2_cache:
                total_deleted += await self.l2_cache.invalidate(pattern)

            if self.l3_cache:
                total_deleted += await self._invalidate_l3(pattern)

            return total_deleted

        except Exception as e:
            logger.error(f"Cache invalidation error for pattern {pattern}: {e}")
            return 0

    async def clear(self) -> bool:
        """Clear all cache levels"""
        if not self._initialized:
            await self.initialize()

        try:
            success = True

            if self.l1_cache:
                success &= await self.l1_cache.clear()

            if self.l2_cache:
                success &= await self.l2_cache.clear()

            if self.l3_cache:
                success &= await self._clear_l3()

            if success:
                self.metrics.reset()

            return success

        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False

    async def get_stats(self) -> CacheStats:
        """Get comprehensive cache statistics"""
        if not self._initialized:
            await self.initialize()

        try:
            base_stats = self.metrics.get_stats()

            # Add level-specific stats
            l1_stats = await self.l1_cache.get_stats() if self.l1_cache else {}
            l2_stats = await self.l2_cache.get_stats() if self.l2_cache else {}
            l3_stats = await self._get_l3_stats() if self.l3_cache else {}

            return CacheStats(
                **base_stats.__dict__,
                l1_stats=l1_stats,
                l2_stats=l2_stats,
                l3_stats=l3_stats
            )

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return self.metrics.get_stats()

    async def optimize(self) -> bool:
        """Optimize cache performance"""
        if not self._initialized:
            await self.initialize()

        try:
            # Optimize each level
            success = True

            if self.l1_cache:
                success &= await self.l1_cache.optimize()

            if self.l2_cache:
                success &= await self.l2_cache.optimize()

            # Run garbage collection
            await self._cleanup_expired()

            return success

        except Exception as e:
            logger.error(f"Cache optimization error: {e}")
            return False

    # L1 Cache Operations
    async def _get_l1(self, key: str) -> Optional[Any]:
        """Get from L1 cache"""
        if not self.l1_cache:
            return None
        return await self.l1_cache.get(key)

    async def _set_l1(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set in L1 cache"""
        if not self.l1_cache:
            return False
        return await self.l1_cache.set(key, value, ttl)

    async def _delete_l1(self, key: str) -> bool:
        """Delete from L1 cache"""
        if not self.l1_cache:
            return False
        return await self.l1_cache.delete(key)

    # L2 Cache Operations
    async def _get_l2(self, key: str) -> Optional[Any]:
        """Get from L2 cache"""
        if not self.l2_cache:
            return None
        return await self.l2_cache.get(key)

    async def _set_l2(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set in L2 cache"""
        if not self.l2_cache:
            return False
        return await self.l2_cache.set(key, value, ttl)

    async def _delete_l2(self, key: str) -> bool:
        """Delete from L2 cache"""
        if not self.l2_cache:
            return False
        return await self.l2_cache.delete(key)

    # L3 Cache Operations
    async def _get_l3(self, key: str) -> Optional[Any]:
        """Get from L3 cache"""
        if not self.l3_cache:
            return None
        try:
            data = await self.l3_cache.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"L3 get error: {e}")
            return None

    async def _set_l3(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set in L3 cache"""
        if not self.l3_cache:
            return False
        try:
            data = pickle.dumps(value)
            if ttl:
                await self.l3_cache.setex(key, ttl, data)
            else:
                await self.l3_cache.set(key, data)
            return True
        except Exception as e:
            logger.error(f"L3 set error: {e}")
            return False

    async def _delete_l3(self, key: str) -> bool:
        """Delete from L3 cache"""
        if not self.l3_cache:
            return False
        try:
            result = await self.l3_cache.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"L3 delete error: {e}")
            return False

    async def _invalidate_l3(self, pattern: str) -> int:
        """Invalidate L3 cache pattern"""
        if not self.l3_cache:
            return 0
        try:
            keys = await self.l3_cache.keys(pattern)
            if keys:
                return await self.l3_cache.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"L3 invalidate error: {e}")
            return 0

    async def _clear_l3(self) -> bool:
        """Clear L3 cache"""
        if not self.l3_cache:
            return True
        try:
            await self.l3_cache.flushdb()
            return True
        except Exception as e:
            logger.error(f"L3 clear error: {e}")
            return False

    async def _get_l3_stats(self) -> Dict[str, Any]:
        """Get L3 cache statistics"""
        if not self.l3_cache:
            return {}
        try:
            info = await self.l3_cache.info()
            return {
                'used_memory': info.get('used_memory', 0),
                'connected_clients': info.get('connected_clients', 0),
                'total_commands_processed': info.get('total_commands_processed', 0)
            }
        except Exception as e:
            logger.error(f"L3 stats error: {e}")
            return {}

    async def _cleanup_expired(self):
        """Clean up expired cache entries"""
        try:
            if self.l2_cache:
                await self.l2_cache.cleanup_expired()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    async def shutdown(self):
        """Shutdown cache manager"""
        try:
            if self.l1_cache:
                await self.l1_cache.shutdown()

            if self.l2_cache:
                await self.l2_cache.shutdown()

            if self.l3_cache:
                await self.l3_cache.close()

            logger.info("CacheManager shutdown completed")

        except Exception as e:
            logger.error(f"Shutdown error: {e}")


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


async def get_cache_manager(config: Optional[CacheConfig] = None) -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager

    if _cache_manager is None:
        _cache_manager = CacheManager(config)
        await _cache_manager.initialize()

    return _cache_manager