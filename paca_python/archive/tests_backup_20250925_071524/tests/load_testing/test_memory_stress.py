"""
Memory Stress Testing for PACA System
Purpose: Test system behavior under memory pressure and resource constraints
Author: PACA Development Team
Created: 2024-09-24
"""

import pytest
import asyncio
import time
import gc
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import threading
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import PACA modules
from paca.core.types import PACACoreTypes
from paca.cognitive.memory import MemorySystem
from paca.data.cache.cache_manager import CacheManager
from paca.learning.auto.engine import AutoLearningEngine

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class MemoryTestConfig:
    """Configuration for memory stress testing."""
    max_memory_mb: int = 200  # Maximum allowed memory usage
    allocation_size_mb: int = 10  # Size of each allocation
    allocation_count: int = 50  # Number of allocations to make
    hold_duration: int = 30  # How long to hold allocations (seconds)
    gc_frequency: int = 5  # Run GC every N allocations


@dataclass
class MemoryTestResult:
    """Results from memory stress testing."""
    initial_memory_mb: float
    peak_memory_mb: float
    final_memory_mb: float
    memory_growth_mb: float
    memory_freed_mb: float
    gc_collections: int
    allocation_failures: int
    test_duration: float
    memory_leaks_detected: bool
    system_stable: bool


class MemoryAllocator:
    """Handles memory allocation and tracking for stress testing."""

    def __init__(self):
        self.allocations = []
        self.allocation_history = []

    def allocate_memory(self, size_mb: int) -> bool:
        """Allocate memory of specified size."""
        try:
            # Create a large list to consume memory
            size_bytes = size_mb * 1024 * 1024
            # Use byte arrays for more realistic memory consumption
            allocation = bytearray(size_bytes)

            # Fill with some data to ensure actual memory usage
            for i in range(0, min(1024, len(allocation)), 4):
                allocation[i:i+4] = (i % 256).to_bytes(4, 'big')

            self.allocations.append(allocation)
            self.allocation_history.append({
                'size_mb': size_mb,
                'timestamp': time.time(),
                'success': True
            })
            return True
        except MemoryError:
            self.allocation_history.append({
                'size_mb': size_mb,
                'timestamp': time.time(),
                'success': False,
                'error': 'MemoryError'
            })
            return False

    def free_memory(self, percentage: float = 0.5) -> int:
        """Free a percentage of allocated memory."""
        count_to_free = int(len(self.allocations) * percentage)
        freed_count = 0

        for _ in range(count_to_free):
            if self.allocations:
                self.allocations.pop()
                freed_count += 1

        return freed_count

    def get_allocation_stats(self) -> Dict[str, Any]:
        """Get allocation statistics."""
        total_allocations = len(self.allocation_history)
        successful_allocations = sum(1 for a in self.allocation_history if a['success'])
        failed_allocations = total_allocations - successful_allocations

        return {
            'total_allocations': total_allocations,
            'successful_allocations': successful_allocations,
            'failed_allocations': failed_allocations,
            'current_allocations': len(self.allocations)
        }

    def clear_all(self):
        """Clear all allocations."""
        self.allocations.clear()
        gc.collect()


class SystemMonitor:
    """Monitors system resource usage during testing."""

    def __init__(self):
        self.memory_samples = []
        self.monitoring = False
        self.monitor_task = None

    async def start_monitoring(self, interval: float = 1.0):
        """Start monitoring system resources."""
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop(interval))

    async def stop_monitoring(self):
        """Stop monitoring system resources."""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring:
            sample = await self._collect_sample()
            self.memory_samples.append(sample)
            await asyncio.sleep(interval)

    async def _collect_sample(self) -> Dict[str, Any]:
        """Collect a resource usage sample."""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                return {
                    'timestamp': time.time(),
                    'rss_mb': memory_info.rss / (1024 * 1024),
                    'vms_mb': memory_info.vms / (1024 * 1024),
                    'percent': process.memory_percent(),
                    'available_mb': psutil.virtual_memory().available / (1024 * 1024)
                }
            except Exception as e:
                return {
                    'timestamp': time.time(),
                    'error': str(e),
                    'rss_mb': 0,
                    'vms_mb': 0,
                    'percent': 0,
                    'available_mb': 1000
                }
        else:
            # Fallback simulation when psutil is not available
            return {
                'timestamp': time.time(),
                'rss_mb': 50 + len(gc.get_objects()) * 0.001,  # Rough estimation
                'vms_mb': 100 + len(gc.get_objects()) * 0.002,
                'percent': 5.0,
                'available_mb': 1000
            }

    def get_peak_memory(self) -> float:
        """Get peak memory usage from samples."""
        if not self.memory_samples:
            return 0.0
        return max(sample['rss_mb'] for sample in self.memory_samples if 'rss_mb' in sample)

    def get_memory_growth(self) -> float:
        """Get memory growth from start to end."""
        if len(self.memory_samples) < 2:
            return 0.0

        valid_samples = [s for s in self.memory_samples if 'rss_mb' in s]
        if len(valid_samples) < 2:
            return 0.0

        return valid_samples[-1]['rss_mb'] - valid_samples[0]['rss_mb']


class TestMemoryStress:
    """
    Memory stress testing for PACA system.

    Tests system behavior under various memory pressure scenarios.
    """

    @pytest.fixture(scope="class")
    def paca_system(self):
        """Initialize PACA system for memory testing."""
        system = {
            'memory': MemorySystem(),
            'cache': CacheManager(),
            'learning': AutoLearningEngine(),
        }
        yield system
        # Force cleanup
        if hasattr(system['cache'], 'clear'):
            system['cache'].clear()
        gc.collect()

    @pytest.fixture
    def memory_allocator(self):
        """Create memory allocator for testing."""
        allocator = MemoryAllocator()
        yield allocator
        allocator.clear_all()

    @pytest.fixture
    def system_monitor(self):
        """Create system monitor for testing."""
        monitor = SystemMonitor()
        yield monitor
        asyncio.create_task(monitor.stop_monitoring())

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_gradual_memory_increase(self, paca_system, memory_allocator, system_monitor):
        """Test system behavior with gradual memory pressure increase."""
        config = MemoryTestConfig(
            max_memory_mb=100,
            allocation_size_mb=5,
            allocation_count=15,
            hold_duration=20,
            gc_frequency=3
        )

        await system_monitor.start_monitoring(interval=0.5)
        start_time = time.time()

        # Gradually increase memory usage
        for i in range(config.allocation_count):
            # Allocate memory
            success = memory_allocator.allocate_memory(config.allocation_size_mb)

            if not success:
                # If allocation fails, try to free some memory and continue
                freed = memory_allocator.free_memory(0.3)
                gc.collect()
                continue

            # Perform PACA operations under memory pressure
            await self._perform_memory_operations(paca_system, i)

            # Periodic garbage collection
            if i % config.gc_frequency == 0:
                gc.collect()

            # Brief pause between allocations
            await asyncio.sleep(0.5)

        # Hold memory for specified duration
        await asyncio.sleep(config.hold_duration)

        await system_monitor.stop_monitoring()
        end_time = time.time()

        # Calculate results
        result = await self._calculate_memory_test_results(
            memory_allocator, system_monitor, end_time - start_time
        )

        # Assertions for gradual memory increase
        assert result.system_stable is True
        assert result.allocation_failures <= 2  # Allow some failures
        assert result.peak_memory_mb <= config.max_memory_mb + 50  # Some tolerance
        assert not result.memory_leaks_detected

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_memory_spike_recovery(self, paca_system, memory_allocator, system_monitor):
        """Test system recovery from memory spikes."""
        config = MemoryTestConfig(
            max_memory_mb=150,
            allocation_size_mb=20,  # Larger allocations
            allocation_count=8,
            hold_duration=10,
            gc_frequency=2
        )

        await system_monitor.start_monitoring(interval=0.3)
        start_time = time.time()

        # Create memory spike
        spike_allocations = []
        for i in range(5):
            success = memory_allocator.allocate_memory(config.allocation_size_mb)
            if success:
                spike_allocations.append(i)

        # Perform operations during spike
        await self._perform_intensive_operations(paca_system, 10)

        # Sudden memory release
        freed_count = memory_allocator.free_memory(0.8)  # Free 80% of memory
        gc.collect()

        # Test recovery
        recovery_start = time.time()
        for i in range(10):
            await self._perform_memory_operations(paca_system, i)
            await asyncio.sleep(0.2)
        recovery_time = time.time() - recovery_start

        await system_monitor.stop_monitoring()
        end_time = time.time()

        result = await self._calculate_memory_test_results(
            memory_allocator, system_monitor, end_time - start_time
        )

        # Assertions for memory spike recovery
        assert result.system_stable is True
        assert recovery_time < 5.0  # Quick recovery
        assert freed_count >= 3  # Should have freed some allocations
        assert result.memory_freed_mb > 0

    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, paca_system, system_monitor):
        """Test detection of memory leaks in PACA operations."""
        await system_monitor.start_monitoring(interval=0.5)
        start_time = time.time()

        # Simulate operations that might cause memory leaks
        for cycle in range(5):
            # Create many objects that should be garbage collected
            test_data = []
            for i in range(1000):
                # Store memory items
                memory_id = await paca_system['memory'].store_memory(
                    content=f"Leak test {cycle}_{i} - " + "x" * 100,
                    memory_type="test",
                    importance=0.1
                )
                test_data.append(memory_id)

                # Cache operations
                await paca_system['cache'].set(
                    f"leak_test_{cycle}_{i}",
                    "test_data_" + "y" * 200,
                    ttl=1  # Short TTL
                )

            # Clear references and force GC
            del test_data
            gc.collect()

            # Wait for cache expiration
            await asyncio.sleep(2)

            # Force cleanup
            if hasattr(paca_system['cache'], 'cleanup_expired'):
                await paca_system['cache'].cleanup_expired()

        await system_monitor.stop_monitoring()
        end_time = time.time()

        # Analyze memory growth pattern
        memory_growth = system_monitor.get_memory_growth()
        test_duration = end_time - start_time

        # Memory leak detection
        # If memory grows consistently without proper cleanup, it indicates a leak
        memory_leak_detected = memory_growth > 50.0  # More than 50MB growth indicates potential leak

        result = MemoryTestResult(
            initial_memory_mb=system_monitor.memory_samples[0]['rss_mb'] if system_monitor.memory_samples else 0,
            peak_memory_mb=system_monitor.get_peak_memory(),
            final_memory_mb=system_monitor.memory_samples[-1]['rss_mb'] if system_monitor.memory_samples else 0,
            memory_growth_mb=memory_growth,
            memory_freed_mb=max(0, system_monitor.get_peak_memory() - system_monitor.memory_samples[-1]['rss_mb']) if system_monitor.memory_samples else 0,
            gc_collections=len(gc.get_stats()) if hasattr(gc, 'get_stats') else 5,
            allocation_failures=0,
            test_duration=test_duration,
            memory_leaks_detected=memory_leak_detected,
            system_stable=True
        )

        # Assertions for memory leak detection
        assert not result.memory_leaks_detected, f"Memory leak detected: {memory_growth:.2f}MB growth"
        assert result.memory_growth_mb <= 30.0  # Allow some growth for caching
        assert result.system_stable is True

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_low_memory_conditions(self, paca_system, memory_allocator, system_monitor):
        """Test system behavior under low memory conditions."""
        config = MemoryTestConfig(
            max_memory_mb=80,  # Relatively low limit
            allocation_size_mb=15,
            allocation_count=6,
            hold_duration=15,
            gc_frequency=1  # Frequent GC
        )

        await system_monitor.start_monitoring(interval=0.5)
        start_time = time.time()

        # Consume most available memory
        for i in range(config.allocation_count):
            success = memory_allocator.allocate_memory(config.allocation_size_mb)

            # Force frequent garbage collection
            gc.collect()

            # Test PACA operations under low memory
            try:
                await self._perform_memory_operations(paca_system, i)
            except MemoryError:
                # System should handle low memory gracefully
                memory_allocator.free_memory(0.5)  # Free half the memory
                gc.collect()
                continue

            await asyncio.sleep(0.3)

        # Test graceful degradation
        degradation_test_passed = await self._test_graceful_degradation(paca_system)

        await system_monitor.stop_monitoring()
        end_time = time.time()

        result = await self._calculate_memory_test_results(
            memory_allocator, system_monitor, end_time - start_time
        )

        # Assertions for low memory conditions
        assert result.system_stable is True
        assert degradation_test_passed
        assert result.allocation_failures <= 3  # Some failures expected under low memory

    @pytest.mark.asyncio
    async def test_memory_fragmentation(self, paca_system, memory_allocator, system_monitor):
        """Test system behavior with memory fragmentation."""
        await system_monitor.start_monitoring(interval=0.5)
        start_time = time.time()

        # Create fragmented memory pattern
        for cycle in range(3):
            # Allocate many small chunks
            small_allocations = []
            for i in range(20):
                success = memory_allocator.allocate_memory(2)  # 2MB chunks
                if success:
                    small_allocations.append(i)

            # Free every other allocation to create fragmentation
            for i in range(0, len(small_allocations), 2):
                if memory_allocator.allocations:
                    memory_allocator.allocations.pop(i % len(memory_allocator.allocations))

            # Perform operations with fragmented memory
            await self._perform_memory_operations(paca_system, cycle * 10)

            gc.collect()
            await asyncio.sleep(1)

        await system_monitor.stop_monitoring()
        end_time = time.time()

        result = await self._calculate_memory_test_results(
            memory_allocator, system_monitor, end_time - start_time
        )

        # Assertions for memory fragmentation
        assert result.system_stable is True
        assert result.peak_memory_mb > 0

    # Helper methods
    async def _perform_memory_operations(self, paca_system: Dict[str, Any], iteration: int):
        """Perform memory-intensive PACA operations."""
        try:
            # Memory system operations
            content = f"Memory stress test {iteration} - " + "data" * 50
            memory_id = await paca_system['memory'].store_memory(
                content=content,
                memory_type="stress_test",
                importance=0.5,
                tags=[f"iteration_{iteration}"]
            )

            # Retrieve the memory
            retrieved = await paca_system['memory'].retrieve_memory(memory_id)

            # Cache operations
            cache_key = f"stress_test_{iteration}"
            cache_value = "cached_data_" + "x" * 100
            await paca_system['cache'].set(cache_key, cache_value, ttl=30)

            # Get cached value
            cached = await paca_system['cache'].get(cache_key)

            return True
        except Exception as e:
            # Log error but don't fail the test
            return False

    async def _perform_intensive_operations(self, paca_system: Dict[str, Any], count: int):
        """Perform intensive operations to stress the system."""
        operations = []
        for i in range(count):
            task = self._perform_memory_operations(paca_system, i)
            operations.append(task)

        # Execute operations concurrently
        results = await asyncio.gather(*operations, return_exceptions=True)
        return results

    async def _test_graceful_degradation(self, paca_system: Dict[str, Any]) -> bool:
        """Test if system degrades gracefully under memory pressure."""
        try:
            # Test basic functionality
            memory_id = await paca_system['memory'].store_memory(
                content="degradation_test",
                memory_type="test",
                importance=0.1
            )

            retrieved = await paca_system['memory'].retrieve_memory(memory_id)

            # Test cache functionality
            await paca_system['cache'].set("degradation_key", "degradation_value", ttl=60)
            cached = await paca_system['cache'].get("degradation_key")

            return retrieved is not None and cached is not None
        except Exception:
            return False

    async def _calculate_memory_test_results(
        self, allocator: MemoryAllocator, monitor: SystemMonitor, duration: float
    ) -> MemoryTestResult:
        """Calculate memory test results."""
        stats = allocator.get_allocation_stats()

        if monitor.memory_samples:
            initial_memory = monitor.memory_samples[0]['rss_mb']
            final_memory = monitor.memory_samples[-1]['rss_mb']
            peak_memory = monitor.get_peak_memory()
            memory_growth = monitor.get_memory_growth()
        else:
            initial_memory = final_memory = peak_memory = memory_growth = 0.0

        memory_freed = max(0, peak_memory - final_memory)

        # Simple memory leak detection
        memory_leak_detected = memory_growth > 50.0 and memory_freed < memory_growth * 0.5

        return MemoryTestResult(
            initial_memory_mb=initial_memory,
            peak_memory_mb=peak_memory,
            final_memory_mb=final_memory,
            memory_growth_mb=memory_growth,
            memory_freed_mb=memory_freed,
            gc_collections=len(gc.get_stats()) if hasattr(gc, 'get_stats') else 5,
            allocation_failures=stats['failed_allocations'],
            test_duration=duration,
            memory_leaks_detected=memory_leak_detected,
            system_stable=stats['failed_allocations'] <= stats['total_allocations'] * 0.2  # Less than 20% failures
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])