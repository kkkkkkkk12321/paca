"""
Advanced Cache System for PACA

This module provides a sophisticated 3-tier caching system:
- L1: Memory cache (LRU/LFU policies)
- L2: Disk cache (TTL-based persistence)
- L3: Distributed cache (Redis integration)

Components:
- CacheManager: Central cache orchestrator
- LRUCache: Memory-based LRU cache implementation
- TTLCache: Time-based expiration cache
- HybridCache: Memory + Disk hybrid cache
- CacheMetrics: Performance monitoring and statistics
- CacheWarming: Intelligent cache preloading

Performance Goals:
- Cache hit rate: >85%
- Memory limit: 100MB (configurable)
- Response time: <10ms for hits
"""

from .cache_manager import CacheManager, CachePolicy
from .lru_cache import LRUCache
from .ttl_cache import TTLCache
from .hybrid_cache import HybridCache
from .cache_metrics import CacheMetrics, CacheStats
from .cache_warming import CacheWarming

__all__ = [
    'CacheManager',
    'CachePolicy',
    'LRUCache',
    'TTLCache',
    'HybridCache',
    'CacheMetrics',
    'CacheStats',
    'CacheWarming'
]

__version__ = "1.0.0"