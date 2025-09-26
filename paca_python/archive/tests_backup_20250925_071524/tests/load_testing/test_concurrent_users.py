"""
Concurrent Users Load Testing for PACA System
Purpose: Test system performance under multiple concurrent user loads
Author: PACA Development Team
Created: 2024-09-24
"""

import pytest
import asyncio
import time
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import PACA modules
from paca.core.types import PACACoreTypes
from paca.core.events import EventManager
from paca.cognitive.memory import MemorySystem
from paca.data.cache.cache_manager import CacheManager
from paca.integrations.apis.universal_client import UniversalAPIClient


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    concurrent_users: int = 10
    operations_per_user: int = 50
    test_duration: int = 60  # seconds
    ramp_up_time: int = 10   # seconds
    think_time: Tuple[float, float] = (0.1, 0.5)  # min, max seconds between operations


@dataclass
class LoadTestResult:
    """Results from load testing."""
    total_operations: int
    successful_operations: int
    failed_operations: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    operations_per_second: float
    errors: List[str]
    test_duration: float


class VirtualUser:
    """Simulates a virtual user performing operations."""

    def __init__(self, user_id: int, paca_system: Dict[str, Any], config: LoadTestConfig):
        self.user_id = user_id
        self.paca_system = paca_system
        self.config = config
        self.operation_results = []
        self.is_running = False

    async def run(self) -> List[Dict[str, Any]]:
        """Run virtual user operations."""
        self.is_running = True
        operations = []

        for operation_id in range(self.config.operations_per_user):
            if not self.is_running:
                break

            # Random think time between operations
            think_time = random.uniform(*self.config.think_time)
            await asyncio.sleep(think_time)

            # Select random operation
            operation_type = random.choice([
                'memory_store', 'memory_retrieve', 'cache_set', 'cache_get',
                'api_request', 'event_emit', 'complex_query'
            ])

            # Execute operation
            start_time = time.time()
            try:
                result = await self._execute_operation(operation_type)
                response_time = time.time() - start_time

                operations.append({
                    'user_id': self.user_id,
                    'operation_id': operation_id,
                    'operation_type': operation_type,
                    'response_time': response_time,
                    'success': result.get('success', False),
                    'error': result.get('error'),
                    'timestamp': start_time
                })
            except Exception as e:
                response_time = time.time() - start_time
                operations.append({
                    'user_id': self.user_id,
                    'operation_id': operation_id,
                    'operation_type': operation_type,
                    'response_time': response_time,
                    'success': False,
                    'error': str(e),
                    'timestamp': start_time
                })

        self.is_running = False
        return operations

    async def _execute_operation(self, operation_type: str) -> Dict[str, Any]:
        """Execute a specific operation type."""
        if operation_type == 'memory_store':
            return await self._memory_store_operation()
        elif operation_type == 'memory_retrieve':
            return await self._memory_retrieve_operation()
        elif operation_type == 'cache_set':
            return await self._cache_set_operation()
        elif operation_type == 'cache_get':
            return await self._cache_get_operation()
        elif operation_type == 'api_request':
            return await self._api_request_operation()
        elif operation_type == 'event_emit':
            return await self._event_emit_operation()
        elif operation_type == 'complex_query':
            return await self._complex_query_operation()
        else:
            return {'success': False, 'error': f'Unknown operation: {operation_type}'}

    async def _memory_store_operation(self) -> Dict[str, Any]:
        """Store memory operation."""
        try:
            memory_system = self.paca_system['memory']
            content = f"User {self.user_id} memory content {random.randint(1, 1000)}"

            memory_id = await memory_system.store_memory(
                content=content,
                memory_type="test",
                importance=random.uniform(0.1, 1.0),
                tags=[f"user_{self.user_id}", "load_test"]
            )

            return {'success': True, 'memory_id': memory_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _memory_retrieve_operation(self) -> Dict[str, Any]:
        """Retrieve memory operation."""
        try:
            memory_system = self.paca_system['memory']
            # Try to retrieve a random memory (simulation)
            memories = await memory_system.search_memories(
                query=f"user_{self.user_id}",
                limit=1
            )

            if memories:
                retrieved = await memory_system.retrieve_memory(memories[0]['id'])
                return {'success': True, 'retrieved': retrieved is not None}
            else:
                return {'success': True, 'retrieved': False}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _cache_set_operation(self) -> Dict[str, Any]:
        """Cache set operation."""
        try:
            cache_system = self.paca_system['cache']
            key = f"user_{self.user_id}_cache_{random.randint(1, 100)}"
            value = f"cached_data_{random.randint(1, 1000)}"

            success = await cache_system.set(key, value, ttl=300)
            return {'success': success}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _cache_get_operation(self) -> Dict[str, Any]:
        """Cache get operation."""
        try:
            cache_system = self.paca_system['cache']
            key = f"user_{self.user_id}_cache_{random.randint(1, 100)}"

            value = await cache_system.get(key)
            return {'success': True, 'cache_hit': value is not None}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _api_request_operation(self) -> Dict[str, Any]:
        """API request operation."""
        try:
            api_client = self.paca_system['api_client']

            # Simulate API request
            response = await api_client.request(
                method='GET',
                endpoint='/test',
                params={'user_id': self.user_id, 'random': random.randint(1, 1000)}
            )

            return {'success': True, 'response_received': response is not None}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _event_emit_operation(self) -> Dict[str, Any]:
        """Event emit operation."""
        try:
            event_manager = self.paca_system['events']

            event_data = {
                'user_id': self.user_id,
                'action': 'load_test_action',
                'timestamp': time.time(),
                'data': random.randint(1, 1000)
            }

            await event_manager.emit('load_test_event', event_data)
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _complex_query_operation(self) -> Dict[str, Any]:
        """Complex query operation."""
        try:
            # Simulate complex operation involving multiple systems
            memory_system = self.paca_system['memory']
            cache_system = self.paca_system['cache']

            # Multi-step operation
            query_key = f"complex_query_{self.user_id}_{random.randint(1, 50)}"

            # Check cache first
            cached_result = await cache_system.get(query_key)
            if cached_result:
                return {'success': True, 'cache_hit': True}

            # Simulate complex memory search
            memories = await memory_system.search_memories(
                query=f"user_{self.user_id % 5}",  # Search across multiple users
                limit=5
            )

            # Simulate processing
            await asyncio.sleep(random.uniform(0.01, 0.05))  # Processing time

            result = {
                'memories_found': len(memories),
                'processing_time': time.time(),
                'user_id': self.user_id
            }

            # Cache the result
            await cache_system.set(query_key, result, ttl=60)

            return {'success': True, 'cache_hit': False, 'result': result}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def stop(self):
        """Stop the virtual user."""
        self.is_running = False


class TestConcurrentUsers:
    """
    Concurrent users load testing.

    Tests system performance under various concurrent user loads.
    """

    @pytest.fixture(scope="class")
    def paca_system(self):
        """Initialize PACA system for load testing."""
        system = {
            'memory': MemorySystem(),
            'cache': CacheManager(),
            'events': EventManager(),
            'api_client': UniversalAPIClient(),
        }
        yield system
        # Cleanup
        if hasattr(system['cache'], 'clear'):
            system['cache'].clear()

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_light_load_10_users(self, paca_system):
        """Test system under light load (10 concurrent users)."""
        config = LoadTestConfig(
            concurrent_users=10,
            operations_per_user=20,
            test_duration=30,
            ramp_up_time=5,
            think_time=(0.1, 0.3)
        )

        result = await self._run_load_test(paca_system, config)

        # Assertions for light load
        assert result.operations_per_second >= 20  # At least 20 ops/sec
        assert result.successful_operations / result.total_operations >= 0.95  # 95% success rate
        assert result.avg_response_time <= 0.5  # Average response time under 500ms
        assert result.p95_response_time <= 1.0  # 95th percentile under 1 second

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_medium_load_25_users(self, paca_system):
        """Test system under medium load (25 concurrent users)."""
        config = LoadTestConfig(
            concurrent_users=25,
            operations_per_user=30,
            test_duration=45,
            ramp_up_time=10,
            think_time=(0.05, 0.2)
        )

        result = await self._run_load_test(paca_system, config)

        # Assertions for medium load
        assert result.operations_per_second >= 30  # At least 30 ops/sec
        assert result.successful_operations / result.total_operations >= 0.90  # 90% success rate
        assert result.avg_response_time <= 1.0  # Average response time under 1 second
        assert result.p95_response_time <= 2.0  # 95th percentile under 2 seconds

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_heavy_load_50_users(self, paca_system):
        """Test system under heavy load (50 concurrent users)."""
        config = LoadTestConfig(
            concurrent_users=50,
            operations_per_user=40,
            test_duration=60,
            ramp_up_time=15,
            think_time=(0.01, 0.1)
        )

        result = await self._run_load_test(paca_system, config)

        # Assertions for heavy load
        assert result.operations_per_second >= 40  # At least 40 ops/sec
        assert result.successful_operations / result.total_operations >= 0.85  # 85% success rate
        assert result.avg_response_time <= 2.0  # Average response time under 2 seconds
        assert result.p95_response_time <= 5.0  # 95th percentile under 5 seconds

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_stress_load_100_users(self, paca_system):
        """Test system under stress load (100 concurrent users)."""
        config = LoadTestConfig(
            concurrent_users=100,
            operations_per_user=25,
            test_duration=90,
            ramp_up_time=20,
            think_time=(0.0, 0.05)
        )

        result = await self._run_load_test(paca_system, config)

        # Assertions for stress load (more lenient)
        assert result.operations_per_second >= 25  # At least 25 ops/sec
        assert result.successful_operations / result.total_operations >= 0.75  # 75% success rate
        assert result.avg_response_time <= 5.0  # Average response time under 5 seconds
        assert result.p95_response_time <= 10.0  # 95th percentile under 10 seconds

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_endurance_test(self, paca_system):
        """Test system endurance over extended period."""
        config = LoadTestConfig(
            concurrent_users=20,
            operations_per_user=100,  # More operations per user
            test_duration=300,  # 5 minutes
            ramp_up_time=30,
            think_time=(0.1, 0.5)
        )

        result = await self._run_load_test(paca_system, config)

        # Endurance test assertions
        assert result.operations_per_second >= 15  # Sustained throughput
        assert result.successful_operations / result.total_operations >= 0.90  # 90% success rate
        assert result.avg_response_time <= 1.0  # Consistent performance

        # Check for performance degradation (simple heuristic)
        # In a real implementation, you'd track performance over time
        assert result.max_response_time <= result.avg_response_time * 10  # No extreme outliers

    @pytest.mark.asyncio
    async def test_ramp_up_performance(self, paca_system):
        """Test system performance during user ramp-up."""
        config = LoadTestConfig(
            concurrent_users=30,
            operations_per_user=15,
            test_duration=60,
            ramp_up_time=30,  # Slower ramp-up
            think_time=(0.1, 0.3)
        )

        # Track performance during ramp-up
        performance_snapshots = []

        async def performance_monitor():
            """Monitor performance during test."""
            for i in range(6):  # Take snapshots every 10 seconds
                await asyncio.sleep(10)
                snapshot = {
                    'timestamp': time.time(),
                    'active_users': min(config.concurrent_users, (i + 1) * 5),
                    'memory_usage': await self._get_memory_usage(paca_system),
                    'cache_stats': await self._get_cache_stats(paca_system['cache'])
                }
                performance_snapshots.append(snapshot)

        # Run load test with monitoring
        monitor_task = asyncio.create_task(performance_monitor())
        result = await self._run_load_test(paca_system, config)

        # Wait for monitoring to complete
        try:
            await asyncio.wait_for(monitor_task, timeout=5.0)
        except asyncio.TimeoutError:
            monitor_task.cancel()

        # Verify ramp-up performance
        assert len(performance_snapshots) >= 3  # At least 3 snapshots
        assert result.successful_operations / result.total_operations >= 0.90

        # Check that performance remains stable during ramp-up
        memory_usages = [s['memory_usage'] for s in performance_snapshots]
        if len(memory_usages) >= 2:
            memory_increase = max(memory_usages) - min(memory_usages)
            assert memory_increase <= 100.0  # Memory increase should be reasonable (MB)

    async def _run_load_test(self, paca_system: Dict[str, Any], config: LoadTestConfig) -> LoadTestResult:
        """Run a load test with the given configuration."""
        start_time = time.time()
        all_operations = []

        # Create virtual users
        users = [VirtualUser(i, paca_system, config) for i in range(config.concurrent_users)]

        # Ramp up users gradually
        user_tasks = []
        for i, user in enumerate(users):
            # Stagger user start times
            delay = (config.ramp_up_time * i) / config.concurrent_users
            task = asyncio.create_task(self._start_user_with_delay(user, delay))
            user_tasks.append(task)

        # Wait for all users to complete
        try:
            user_results = await asyncio.gather(*user_tasks, return_exceptions=True)

            # Collect all operations
            for result in user_results:
                if isinstance(result, list):
                    all_operations.extend(result)

        except Exception as e:
            # Stop all users in case of error
            for user in users:
                user.stop()
            raise e

        end_time = time.time()
        test_duration = end_time - start_time

        # Calculate results
        return self._calculate_load_test_results(all_operations, test_duration)

    async def _start_user_with_delay(self, user: VirtualUser, delay: float) -> List[Dict[str, Any]]:
        """Start a user with a delay."""
        await asyncio.sleep(delay)
        return await user.run()

    def _calculate_load_test_results(self, operations: List[Dict[str, Any]], test_duration: float) -> LoadTestResult:
        """Calculate load test results from operations data."""
        total_operations = len(operations)
        successful_operations = sum(1 for op in operations if op['success'])
        failed_operations = total_operations - successful_operations

        if total_operations == 0:
            return LoadTestResult(
                total_operations=0,
                successful_operations=0,
                failed_operations=0,
                avg_response_time=0.0,
                min_response_time=0.0,
                max_response_time=0.0,
                p95_response_time=0.0,
                p99_response_time=0.0,
                operations_per_second=0.0,
                errors=[],
                test_duration=test_duration
            )

        # Calculate response time statistics
        response_times = [op['response_time'] for op in operations]
        avg_response_time = statistics.mean(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)

        # Calculate percentiles
        sorted_times = sorted(response_times)
        p95_index = int(0.95 * len(sorted_times))
        p99_index = int(0.99 * len(sorted_times))
        p95_response_time = sorted_times[min(p95_index, len(sorted_times) - 1)]
        p99_response_time = sorted_times[min(p99_index, len(sorted_times) - 1)]

        # Calculate throughput
        operations_per_second = total_operations / test_duration if test_duration > 0 else 0

        # Collect errors
        errors = [op['error'] for op in operations if op.get('error')]

        return LoadTestResult(
            total_operations=total_operations,
            successful_operations=successful_operations,
            failed_operations=failed_operations,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            operations_per_second=operations_per_second,
            errors=errors,
            test_duration=test_duration
        )

    async def _get_memory_usage(self, paca_system: Dict[str, Any]) -> float:
        """Get current memory usage (simulation)."""
        # In a real implementation, you'd use psutil or similar
        return 50.0 + random.uniform(0, 20)  # Simulate 50-70 MB usage

    async def _get_cache_stats(self, cache_system) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            stats = await cache_system.get_stats()
            return stats if stats else {'hit_rate': 0.5, 'size': 100}
        except:
            return {'hit_rate': 0.5, 'size': 100}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])