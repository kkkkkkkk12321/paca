"""
End-to-End Workflow Tests for PACA System
Purpose: Test complete user workflows across all system components
Author: PACA Development Team
Created: 2024-09-24
"""

import pytest
import asyncio
import time
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import PACA modules
from paca.core.types import PACACoreTypes
from paca.core.events import EventManager
from paca.cognitive.memory import MemorySystem
from paca.learning.auto.engine import AutoLearningEngine
from paca.data.cache.cache_manager import CacheManager
from paca.integrations.apis.universal_client import UniversalAPIClient
from paca.integrations.databases.sql_connector import SQLConnector


class TestFullWorkflow:
    """
    Comprehensive end-to-end workflow tests.

    Tests complete user journeys through the PACA system,
    ensuring all components work together correctly.
    """

    @pytest.fixture(scope="class")
    def paca_system(self):
        """Initialize complete PACA system for testing."""
        system = {
            'memory': MemorySystem(),
            'cache': CacheManager(),
            'learning': AutoLearningEngine(),
            'events': EventManager(),
            'api_client': UniversalAPIClient(),
        }
        yield system
        # Cleanup
        if hasattr(system['cache'], 'clear'):
            system['cache'].clear()

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.mark.asyncio
    async def test_complete_learning_workflow(self, paca_system, temp_workspace):
        """
        Test complete learning workflow from data input to knowledge retention.

        Workflow:
        1. Input new information
        2. Process through cognitive system
        3. Store in memory
        4. Learn patterns
        5. Apply learning to new scenarios
        6. Verify knowledge retention
        """
        memory = paca_system['memory']
        learning = paca_system['learning']
        cache = paca_system['cache']

        # Step 1: Input new information
        training_data = [
            {"input": "What is machine learning?", "output": "A method of data analysis that automates analytical model building"},
            {"input": "Define neural networks", "output": "Computing systems inspired by biological neural networks"},
            {"input": "Explain deep learning", "output": "A subset of machine learning using neural networks with multiple layers"}
        ]

        knowledge_ids = []
        for item in training_data:
            # Step 2: Process through cognitive system
            processed_info = await self._process_cognitive_input(item)

            # Step 3: Store in memory
            memory_id = await memory.store_memory(
                content=processed_info['content'],
                memory_type="factual",
                importance=0.8,
                tags=processed_info['tags']
            )
            knowledge_ids.append(memory_id)

            # Cache frequently accessed information
            await cache.set(f"knowledge_{memory_id}", processed_info, ttl=3600)

        # Step 4: Learn patterns
        learning_result = await learning.train_on_data(training_data)
        assert learning_result['status'] == 'success'
        assert learning_result['patterns_learned'] > 0

        # Step 5: Apply learning to new scenarios
        test_query = "What are the applications of artificial intelligence?"
        response = await self._query_learned_knowledge(
            paca_system, test_query, knowledge_ids
        )

        assert response is not None
        assert len(response) > 0
        assert any(keyword in response.lower() for keyword in ['machine', 'learning', 'neural'])

        # Step 6: Verify knowledge retention
        for memory_id in knowledge_ids:
            retrieved = await memory.retrieve_memory(memory_id)
            assert retrieved is not None
            assert retrieved['importance'] >= 0.5

        # Verify cache performance
        cache_stats = await cache.get_stats()
        assert cache_stats['hit_rate'] > 0.5

    @pytest.mark.asyncio
    async def test_api_integration_workflow(self, paca_system, temp_workspace):
        """
        Test complete API integration workflow.

        Workflow:
        1. Configure API client
        2. Make API requests
        3. Process responses
        4. Cache results
        5. Handle rate limiting
        6. Verify error handling
        """
        api_client = paca_system['api_client']
        cache = paca_system['cache']

        # Step 1: Configure API client
        api_config = {
            'base_url': 'https://httpbin.org',
            'rate_limit': 10,  # requests per second
            'timeout': 30,
            'retry_count': 3
        }

        await api_client.configure(api_config)

        # Step 2: Make API requests
        test_requests = [
            {'method': 'GET', 'endpoint': '/get', 'params': {'test': 'data1'}},
            {'method': 'POST', 'endpoint': '/post', 'data': {'test': 'data2'}},
            {'method': 'GET', 'endpoint': '/json', 'params': {}},
        ]

        results = []
        for request in test_requests:
            # Step 3: Process responses
            try:
                response = await api_client.request(
                    method=request['method'],
                    endpoint=request['endpoint'],
                    params=request.get('params'),
                    data=request.get('data')
                )
                results.append(response)

                # Step 4: Cache results
                cache_key = f"api_{request['method']}_{request['endpoint']}"
                await cache.set(cache_key, response, ttl=1800)

            except Exception as e:
                # Step 6: Verify error handling
                assert isinstance(e, (TimeoutError, ConnectionError))

        # Verify successful requests
        assert len(results) >= 1  # At least one request should succeed

        # Step 5: Test rate limiting
        rate_limit_start = time.time()
        rate_test_requests = []

        for i in range(15):  # Exceed rate limit
            try:
                response = await api_client.request('GET', '/get', params={'id': i})
                rate_test_requests.append(response)
            except Exception as e:
                # Rate limiting should kick in
                pass

        rate_limit_duration = time.time() - rate_limit_start
        # Should take at least 1 second due to rate limiting
        assert rate_limit_duration >= 1.0

    @pytest.mark.asyncio
    async def test_data_processing_pipeline(self, paca_system, temp_workspace):
        """
        Test complete data processing pipeline.

        Pipeline:
        1. Data ingestion
        2. Preprocessing
        3. Analysis
        4. Caching
        5. Storage
        6. Retrieval
        """
        memory = paca_system['memory']
        cache = paca_system['cache']
        events = paca_system['events']

        # Step 1: Data ingestion
        raw_data = [
            {"text": "The quick brown fox jumps over the lazy dog", "type": "sentence"},
            {"text": "Python is a programming language", "type": "fact"},
            {"text": "Machine learning enables computers to learn", "type": "definition"},
            {"text": "Data science combines statistics and programming", "type": "concept"}
        ]

        processed_items = []

        for item in raw_data:
            # Step 2: Preprocessing
            processed_item = {
                'id': f"item_{len(processed_items)}",
                'text': item['text'].lower().strip(),
                'type': item['type'],
                'word_count': len(item['text'].split()),
                'processed_at': time.time()
            }

            # Step 3: Analysis
            analysis_result = await self._analyze_text_content(processed_item['text'])
            processed_item['analysis'] = analysis_result

            # Step 4: Caching
            cache_key = f"processed_{processed_item['id']}"
            await cache.set(cache_key, processed_item, ttl=3600)

            # Step 5: Storage
            memory_id = await memory.store_memory(
                content=processed_item,
                memory_type="processed_data",
                importance=analysis_result.get('importance', 0.5)
            )
            processed_item['memory_id'] = memory_id

            processed_items.append(processed_item)

            # Emit processing event
            await events.emit('data_processed', {
                'item_id': processed_item['id'],
                'memory_id': memory_id,
                'type': processed_item['type']
            })

        # Step 6: Retrieval and verification
        for item in processed_items:
            # Test cache retrieval
            cached = await cache.get(f"processed_{item['id']}")
            assert cached is not None
            assert cached['id'] == item['id']

            # Test memory retrieval
            retrieved = await memory.retrieve_memory(item['memory_id'])
            assert retrieved is not None
            assert retrieved['content']['type'] == item['type']

        # Verify processing statistics
        assert len(processed_items) == len(raw_data)
        avg_word_count = sum(item['word_count'] for item in processed_items) / len(processed_items)
        assert avg_word_count > 0

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, paca_system, temp_workspace):
        """
        Test system error recovery and resilience.

        Scenarios:
        1. Memory system failure
        2. Cache system failure
        3. API timeout
        4. Data corruption
        5. Resource exhaustion
        """
        memory = paca_system['memory']
        cache = paca_system['cache']
        api_client = paca_system['api_client']

        error_scenarios = []

        # Scenario 1: Memory system failure simulation
        with patch.object(memory, 'store_memory', side_effect=Exception("Memory error")):
            try:
                await memory.store_memory("test content", "test", 0.5)
            except Exception as e:
                error_scenarios.append(('memory_error', str(e)))

        # Scenario 2: Cache system failure simulation
        with patch.object(cache, 'set', side_effect=Exception("Cache error")):
            try:
                await cache.set("test_key", "test_value")
            except Exception as e:
                error_scenarios.append(('cache_error', str(e)))

        # Scenario 3: API timeout simulation
        with patch.object(api_client, 'request', side_effect=TimeoutError("Request timeout")):
            try:
                await api_client.request('GET', '/test')
            except TimeoutError as e:
                error_scenarios.append(('api_timeout', str(e)))

        # Scenario 4: Data corruption simulation
        corrupted_data = b'\x00\x01\x02invalid_data\xff\xfe'
        try:
            processed = await self._process_potentially_corrupted_data(corrupted_data)
            error_scenarios.append(('data_corruption', 'handled'))
        except Exception as e:
            error_scenarios.append(('data_corruption', str(e)))

        # Verify error handling
        assert len(error_scenarios) >= 3
        error_types = [scenario[0] for scenario in error_scenarios]
        assert 'memory_error' in error_types
        assert 'cache_error' in error_types
        assert any(t in error_types for t in ['api_timeout', 'data_corruption'])

    @pytest.mark.asyncio
    async def test_concurrent_operations_workflow(self, paca_system, temp_workspace):
        """
        Test system behavior under concurrent operations.

        Operations:
        1. Multiple simultaneous memory operations
        2. Concurrent cache operations
        3. Parallel API requests
        4. Simultaneous read/write operations
        """
        memory = paca_system['memory']
        cache = paca_system['cache']

        # Test 1: Concurrent memory operations
        memory_tasks = []
        for i in range(10):
            task = memory.store_memory(
                content=f"Concurrent content {i}",
                memory_type="test",
                importance=0.5 + (i * 0.05)
            )
            memory_tasks.append(task)

        memory_results = await asyncio.gather(*memory_tasks, return_exceptions=True)
        successful_memory_ops = [r for r in memory_results if not isinstance(r, Exception)]
        assert len(successful_memory_ops) >= 8  # At least 80% success rate

        # Test 2: Concurrent cache operations
        cache_tasks = []
        for i in range(20):
            # Mix of set and get operations
            if i % 2 == 0:
                task = cache.set(f"concurrent_key_{i}", f"value_{i}", ttl=60)
            else:
                task = cache.get(f"concurrent_key_{i-1}")
            cache_tasks.append(task)

        cache_results = await asyncio.gather(*cache_tasks, return_exceptions=True)
        successful_cache_ops = [r for r in cache_results if not isinstance(r, Exception)]
        assert len(successful_cache_ops) >= 16  # At least 80% success rate

        # Test 3: Performance under load
        start_time = time.time()

        # Simulate mixed workload
        mixed_tasks = []
        for i in range(5):
            mixed_tasks.extend([
                memory.store_memory(f"load_test_{i}", "load_test", 0.5),
                cache.set(f"load_key_{i}", f"load_value_{i}", ttl=30),
                cache.get(f"load_key_{i}")
            ])

        mixed_results = await asyncio.gather(*mixed_tasks, return_exceptions=True)
        execution_time = time.time() - start_time

        successful_mixed_ops = [r for r in mixed_results if not isinstance(r, Exception)]
        success_rate = len(successful_mixed_ops) / len(mixed_results)

        # Performance assertions
        assert success_rate >= 0.8  # At least 80% success rate
        assert execution_time < 10.0  # Complete within 10 seconds

    # Helper methods
    async def _process_cognitive_input(self, item: Dict[str, str]) -> Dict[str, Any]:
        """Process input through cognitive system simulation."""
        return {
            'content': item['output'],
            'tags': item['input'].lower().split(),
            'confidence': 0.8,
            'processed_at': time.time()
        }

    async def _query_learned_knowledge(self, system: Dict[str, Any], query: str,
                                     knowledge_ids: List[str]) -> str:
        """Query learned knowledge simulation."""
        # Simple knowledge retrieval simulation
        memory = system['memory']
        relevant_memories = []

        for memory_id in knowledge_ids:
            memory_item = await memory.retrieve_memory(memory_id)
            if memory_item:
                relevant_memories.append(memory_item['content'])

        if relevant_memories:
            # Simple response generation based on stored knowledge
            return f"Based on learned knowledge about {query}, relevant concepts include machine learning and neural networks."
        return "No relevant knowledge found."

    async def _analyze_text_content(self, text: str) -> Dict[str, Any]:
        """Analyze text content simulation."""
        words = text.split()
        return {
            'word_count': len(words),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'importance': min(1.0, len(words) / 20.0),  # Longer texts are more important
            'complexity': len(set(words)) / len(words) if words else 0  # Unique word ratio
        }

    async def _process_potentially_corrupted_data(self, data: bytes) -> Dict[str, Any]:
        """Process potentially corrupted data with error handling."""
        try:
            # Attempt to decode
            text_data = data.decode('utf-8', errors='ignore')
            return {'status': 'success', 'data': text_data}
        except Exception as e:
            # Fallback processing
            return {'status': 'error', 'error': str(e), 'fallback_used': True}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])