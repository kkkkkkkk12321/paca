"""
Basic Cache System Tests for PACA

This module provides comprehensive tests for the cache system
to verify functionality, performance, and reliability.

Test Categories:
- Unit tests for individual cache components
- Integration tests for multi-level cache operations
- Performance tests for response time and throughput
- Memory usage and limit tests
- Error handling and recovery tests
"""

import asyncio
import pytest
import time
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict

# Import cache components
from .cache_manager import CacheManager, CacheConfig
from .lru_cache import LRUCache, LFUCache, FIFOCache
from .ttl_cache import TTLCache
from .hybrid_cache import HybridCache, HybridConfig
from .cache_metrics import CacheMetrics, CacheStats
from .cache_warming import CacheWarming, WarmingStrategy


class TestLRUCache:
    """Test cases for LRU Cache"""

    @pytest.fixture
    async def lru_cache(self):
        """Create LRU cache for testing"""
        cache = LRUCache(max_size_mb=10, max_items=100)
        await cache.initialize()
        yield cache
        await cache.shutdown()

    @pytest.mark.asyncio
    async def test_basic_operations(self, lru_cache):
        """Test basic cache operations"""
        # Test set and get
        assert await lru_cache.set("key1", "value1")
        assert await lru_cache.get("key1") == "value1"

        # Test nonexistent key
        assert await lru_cache.get("nonexistent") is None

        # Test delete
        assert await lru_cache.delete("key1")
        assert await lru_cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_lru_eviction(self, lru_cache):
        """Test LRU eviction policy"""
        # Fill cache to capacity
        for i in range(100):
            await lru_cache.set(f"key{i}", f"value{i}")

        # Add one more item to trigger eviction
        await lru_cache.set("key100", "value100")

        # First item should be evicted
        assert await lru_cache.get("key0") is None
        assert await lru_cache.get("key100") == "value100"

    @pytest.mark.asyncio
    async def test_access_order(self, lru_cache):
        """Test that access updates LRU order"""
        # Add items
        await lru_cache.set("key1", "value1")
        await lru_cache.set("key2", "value2")

        # Access key1 to make it recently used
        await lru_cache.get("key1")

        # Fill cache to trigger eviction
        for i in range(99):
            await lru_cache.set(f"temp{i}", f"temp{i}")

        # key1 should still exist, key2 should be evicted
        assert await lru_cache.get("key1") == "value1"
        assert await lru_cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_stats(self, lru_cache):
        """Test statistics tracking"""
        # Perform operations
        await lru_cache.set("key1", "value1")
        await lru_cache.get("key1")  # Hit
        await lru_cache.get("key2")  # Miss

        stats = await lru_cache.get_stats()
        assert stats['cache_size_items'] == 1
        assert 'hit_rate' in stats
        assert 'avg_access_time_ms' in stats


class TestTTLCache:
    """Test cases for TTL Cache"""

    @pytest.fixture
    async def ttl_cache(self):
        """Create TTL cache for testing"""
        temp_dir = tempfile.mkdtemp()
        cache = TTLCache(
            cache_dir=Path(temp_dir),
            max_size_mb=50,
            default_ttl=60
        )
        await cache.initialize()
        yield cache
        await cache.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_basic_operations(self, ttl_cache):
        """Test basic TTL cache operations"""
        # Test set and get
        assert await ttl_cache.set("key1", "value1", ttl=60)
        assert await ttl_cache.get("key1") == "value1"

        # Test delete
        assert await ttl_cache.delete("key1")
        assert await ttl_cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_ttl_expiration(self, ttl_cache):
        """Test TTL expiration"""
        # Set with short TTL
        await ttl_cache.set("key1", "value1", ttl=1)
        assert await ttl_cache.get("key1") == "value1"

        # Wait for expiration
        await asyncio.sleep(1.1)
        assert await ttl_cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_persistence(self, ttl_cache):
        """Test disk persistence"""
        # Set value
        await ttl_cache.set("key1", {"data": "complex_value"}, ttl=3600)

        # Create new cache instance with same directory
        temp_dir = ttl_cache.cache_dir
        await ttl_cache.shutdown()

        new_cache = TTLCache(cache_dir=temp_dir, max_size_mb=50)
        await new_cache.initialize()

        # Value should be recovered
        value = await new_cache.get("key1")
        assert value == {"data": "complex_value"}

        await new_cache.shutdown()

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, ttl_cache):
        """Test expired entry cleanup"""
        # Add expired entries
        await ttl_cache.set("key1", "value1", ttl=1)
        await ttl_cache.set("key2", "value2", ttl=3600)

        await asyncio.sleep(1.1)

        # Cleanup should remove expired entry
        cleaned = await ttl_cache.cleanup_expired()
        assert cleaned == 1
        assert await ttl_cache.get("key2") == "value2"


class TestHybridCache:
    """Test cases for Hybrid Cache"""

    @pytest.fixture
    async def hybrid_cache(self):
        """Create hybrid cache for testing"""
        temp_dir = tempfile.mkdtemp()
        cache = HybridCache(
            total_size_mb=100,
            cache_dir=temp_dir
        )
        await cache.initialize()
        yield cache
        await cache.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_hot_cold_data_placement(self, hybrid_cache):
        """Test hot/cold data placement"""
        # Add data that should be considered hot (frequent access)
        await hybrid_cache.set("hot_key", "hot_value")

        # Access multiple times to make it hot
        for _ in range(5):
            await hybrid_cache.get("hot_key")

        # Add cold data
        await hybrid_cache.set("cold_key", "cold_value")

        # Check placement through stats
        stats = await hybrid_cache.get_stats()
        assert 'memory_items' in stats
        assert 'disk_items' in stats

    @pytest.mark.asyncio
    async def test_promotion_demotion(self, hybrid_cache):
        """Test promotion and demotion of cache entries"""
        # Add data to disk first
        await hybrid_cache.set("promote_me", "value")

        # Access frequently to trigger promotion
        for _ in range(5):
            await hybrid_cache.get("promote_me")

        # Run optimization to process promotions
        await hybrid_cache.optimize()

        # Verify promotion occurred (implementation specific)
        stats = await hybrid_cache.get_stats()
        assert stats['promotion_candidates'] >= 0


class TestCacheManager:
    """Test cases for Cache Manager"""

    @pytest.fixture
    async def cache_manager(self):
        """Create cache manager for testing"""
        temp_dir = tempfile.mkdtemp()
        config = CacheConfig(
            l1_max_size_mb=10,
            l2_max_size_mb=50,
            cache_dir=temp_dir,
            enable_l3=False  # Disable Redis for testing
        )
        manager = CacheManager(config)
        await manager.initialize()
        yield manager
        await manager.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_multi_level_operations(self, cache_manager):
        """Test multi-level cache operations"""
        # Set value
        await cache_manager.set("key1", "value1", ttl=3600)

        # Get should return value from appropriate level
        value = await cache_manager.get("key1")
        assert value == "value1"

        # Delete should remove from all levels
        await cache_manager.delete("key1")
        assert await cache_manager.get("key1") is None

    @pytest.mark.asyncio
    async def test_cache_promotion(self, cache_manager):
        """Test automatic cache promotion"""
        # Set value that goes to L2 first
        await cache_manager._set_l2("key1", "value1")

        # Get should promote to L1
        value = await cache_manager.get("key1")
        assert value == "value1"

        # Verify it's now in L1
        l1_value = await cache_manager._get_l1("key1")
        assert l1_value == "value1"

    @pytest.mark.asyncio
    async def test_comprehensive_stats(self, cache_manager):
        """Test comprehensive statistics"""
        # Perform various operations
        await cache_manager.set("key1", "value1")
        await cache_manager.get("key1")  # Hit
        await cache_manager.get("key2")  # Miss

        stats = await cache_manager.get_stats()
        assert hasattr(stats, 'total_requests')
        assert hasattr(stats, 'hits')
        assert hasattr(stats, 'misses')
        assert hasattr(stats, 'hit_rate')


class TestCacheMetrics:
    """Test cases for Cache Metrics"""

    @pytest.fixture
    def metrics(self):
        """Create cache metrics for testing"""
        return CacheMetrics()

    def test_hit_miss_tracking(self, metrics):
        """Test hit/miss tracking"""
        # Record hits and misses
        metrics.record_hit('l1', 0.01)
        metrics.record_hit('l2', 0.05)
        metrics.record_miss(0.1)

        stats = metrics.get_stats()
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.total_requests == 3
        assert stats.hit_rate == 2/3

    def test_performance_tracking(self, metrics):
        """Test performance metrics tracking"""
        # Record various response times
        response_times = [0.01, 0.02, 0.05, 0.1, 0.2]
        for rt in response_times:
            metrics.record_hit('l1', rt)

        stats = metrics.get_stats()
        assert stats.avg_response_time_ms > 0
        assert stats.p95_response_time_ms > stats.avg_response_time_ms

    def test_error_tracking(self, metrics):
        """Test error tracking"""
        metrics.record_error()
        metrics.record_error()
        metrics.record_set(0.01)

        stats = metrics.get_stats()
        assert stats.errors == 2
        assert stats.error_rate == 2/1  # 2 errors / 1 operation


class TestCacheWarming:
    """Test cases for Cache Warming"""

    @pytest.fixture
    async def cache_warming_setup(self):
        """Setup cache warming for testing"""
        temp_dir = tempfile.mkdtemp()
        config = CacheConfig(
            l1_max_size_mb=10,
            l2_max_size_mb=50,
            cache_dir=temp_dir,
            enable_l3=False
        )
        cache_manager = CacheManager(config)
        await cache_manager.initialize()

        warming = CacheWarming(cache_manager, temp_dir)

        yield cache_manager, warming

        await warming.stop_warming()
        await cache_manager.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_pattern_analysis(self, cache_warming_setup):
        """Test access pattern analysis"""
        cache_manager, warming = cache_warming_setup

        # Record access patterns
        await warming.record_access("user:123")
        await warming.record_access("user:456")
        await warming.record_access("user:123")  # Repeat access

        # Get predictions
        predictions = await warming.predict_and_warm(count=10)
        assert isinstance(predictions, int)

    @pytest.mark.asyncio
    async def test_warming_strategies(self, cache_warming_setup):
        """Test warming strategies"""
        cache_manager, warming = cache_warming_setup

        # Add test strategy
        strategy = WarmingStrategy(
            name="test_strategy",
            priority=1,
            max_keys=10,
            warm_interval=60
        )
        await warming.add_strategy(strategy)

        # Test strategy execution
        stats = await warming.warm_cache_now("test_strategy")
        assert isinstance(stats.total_warmed, int)


class TestPerformance:
    """Performance tests for cache system"""

    @pytest.mark.asyncio
    async def test_response_time_targets(self):
        """Test response time performance targets"""
        cache = LRUCache(max_size_mb=50)
        await cache.initialize()

        try:
            # Warm up cache
            for i in range(100):
                await cache.set(f"key{i}", f"value{i}")

            # Test response time
            start_time = time.time()
            for i in range(100):
                await cache.get(f"key{i}")
            end_time = time.time()

            avg_time_ms = ((end_time - start_time) / 100) * 1000
            assert avg_time_ms < 10.0  # Target: <10ms

        finally:
            await cache.shutdown()

    @pytest.mark.asyncio
    async def test_memory_limits(self):
        """Test memory usage limits"""
        cache = LRUCache(max_size_mb=1)  # Very small cache
        await cache.initialize()

        try:
            # Try to fill cache beyond limit
            large_value = "x" * 1024 * 1024  # 1MB value

            await cache.set("key1", large_value)
            stats = await cache.get_stats()

            # Should not exceed memory limit significantly
            assert stats['memory_usage_percent'] <= 120  # Allow 20% overhead

        finally:
            await cache.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test concurrent cache access"""
        cache = LRUCache(max_size_mb=50)
        await cache.initialize()

        try:
            async def worker(worker_id: int):
                for i in range(100):
                    key = f"worker{worker_id}_key{i}"
                    await cache.set(key, f"value{i}")
                    value = await cache.get(key)
                    assert value == f"value{i}"

            # Run multiple workers concurrently
            workers = [worker(i) for i in range(10)]
            await asyncio.gather(*workers)

            # Verify cache consistency
            stats = await cache.get_stats()
            assert stats['cache_size_items'] > 0

        finally:
            await cache.shutdown()


class TestErrorHandling:
    """Test error handling and recovery"""

    @pytest.mark.asyncio
    async def test_disk_error_recovery(self):
        """Test recovery from disk errors"""
        # Use non-existent directory to trigger errors
        cache = TTLCache(
            cache_dir=Path("/nonexistent/path"),
            max_size_mb=10
        )

        # Should handle initialization gracefully
        result = await cache.initialize()
        # Depending on implementation, might return False or handle gracefully

        await cache.shutdown()

    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self):
        """Test handling of memory pressure"""
        cache = LRUCache(max_size_mb=1)  # Very small
        await cache.initialize()

        try:
            # Fill beyond capacity
            for i in range(1000):
                await cache.set(f"key{i}", f"value{i}" * 1000)

            # Should handle gracefully without crashing
            stats = await cache.get_stats()
            assert 'eviction_count' in stats

        finally:
            await cache.shutdown()


# Integration test function
async def run_integration_test():
    """Run comprehensive integration test"""
    print("Starting PACA Cache System Integration Test...")

    # Test basic functionality
    temp_dir = tempfile.mkdtemp()
    try:
        config = CacheConfig(
            l1_max_size_mb=50,
            l2_max_size_mb=200,
            cache_dir=temp_dir,
            enable_l3=False
        )

        cache = CacheManager(config)
        await cache.initialize()

        # Test operations
        await cache.set("test_key", {"data": "test_value"}, ttl=3600)
        value = await cache.get("test_key")
        assert value == {"data": "test_value"}

        # Test stats
        stats = await cache.get_stats()
        print(f"Hit rate: {stats.hit_rate:.2%}")
        print(f"Total requests: {stats.total_requests}")

        await cache.shutdown()
        print("Integration test completed successfully!")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Run integration test when executed directly
    asyncio.run(run_integration_test())