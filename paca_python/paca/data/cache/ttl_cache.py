"""
TTL Cache Implementation for PACA

This module provides a Time-To-Live (TTL) based cache with disk persistence.
Designed for L2 cache layer with automatic expiration and cleanup.

Features:
- TTL-based automatic expiration
- Disk persistence for durability
- Efficient file-based storage
- Background cleanup of expired entries
- Memory usage optimization
- Compression support
"""

import asyncio
import json
import logging
import pickle
import time
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import aiofiles
import hashlib

from ..base import BaseDataComponent
from .cache_metrics import CacheMetrics

logger = logging.getLogger(__name__)


@dataclass
class TTLEntry:
    """TTL cache entry with expiration metadata"""
    value: Any
    created_at: float
    expires_at: float
    access_count: int
    size_bytes: int
    file_path: Optional[str] = None


class TTLCache(BaseDataComponent):
    """
    TTL-based cache with disk persistence

    Features:
    - Automatic expiration based on TTL
    - Disk persistence for durability
    - Efficient file-based storage
    - Background cleanup processes
    - Compression for large values
    - Memory usage optimization
    """

    def __init__(
        self,
        cache_dir: Path,
        max_size_mb: int = 1000,
        default_ttl: int = 3600,
        enable_compression: bool = True,
        cleanup_interval: int = 300  # 5 minutes
    ):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.enable_compression = enable_compression
        self.cleanup_interval = cleanup_interval

        # Core storage
        self._index: Dict[str, TTLEntry] = {}
        self._lock = asyncio.Lock()

        # Statistics
        self.metrics = CacheMetrics()
        self._current_size_bytes = 0
        self._initialized = False

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_running = False

    async def initialize(self) -> bool:
        """Initialize TTL cache with disk recovery"""
        try:
            async with self._lock:
                if self._initialized:
                    return True

                # Create cache directory
                self.cache_dir.mkdir(parents=True, exist_ok=True)

                # Initialize components
                self._index = {}
                self._current_size_bytes = 0
                self.metrics = CacheMetrics()

                # Recover from disk
                await self._recover_from_disk()

                # Start background cleanup
                await self._start_cleanup_task()

                self._initialized = True
                logger.info(f"TTLCache initialized - Dir: {self.cache_dir}")
                return True

        except Exception as e:
            logger.error(f"TTLCache initialization failed: {e}")
            return False

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if not expired

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            async with self._lock:
                if key not in self._index:
                    self.metrics.record_miss(time.time() - start_time)
                    return None

                entry = self._index[key]

                # Check expiration
                if time.time() > entry.expires_at:
                    await self._remove_entry(key, entry)
                    self.metrics.record_miss(time.time() - start_time)
                    return None

                # Update access info
                entry.access_count += 1

                # Load from disk if needed
                value = await self._load_value(entry)
                if value is None:
                    await self._remove_entry(key, entry)
                    self.metrics.record_miss(time.time() - start_time)
                    return None

                self.metrics.record_hit('l2', time.time() - start_time)
                return value

        except Exception as e:
            logger.error(f"TTLCache get error for key {key}: {e}")
            self.metrics.record_error()
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache with TTL

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds

        Returns:
            True if successfully cached
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        ttl = ttl or self.default_ttl

        try:
            # Calculate size
            value_size = await self._calculate_size(value)

            async with self._lock:
                # Remove existing entry if present
                if key in self._index:
                    await self._remove_entry(key, self._index[key])

                # Check capacity
                await self._ensure_capacity(value_size)

                # Create file path
                file_path = await self._get_file_path(key)

                # Create entry
                entry = TTLEntry(
                    value=value,
                    created_at=time.time(),
                    expires_at=time.time() + ttl,
                    access_count=1,
                    size_bytes=value_size,
                    file_path=str(file_path)
                )

                # Save to disk
                if await self._save_value(entry):
                    self._index[key] = entry
                    self._current_size_bytes += value_size
                    self.metrics.record_set(time.time() - start_time)
                    return True
                else:
                    return False

        except Exception as e:
            logger.error(f"TTLCache set error for key {key}: {e}")
            self.metrics.record_error()
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self._lock:
                if key in self._index:
                    entry = self._index[key]
                    await self._remove_entry(key, entry)
                    return True
                return False

        except Exception as e:
            logger.error(f"TTLCache delete error for key {key}: {e}")
            return False

    async def invalidate(self, pattern: str) -> int:
        """Invalidate keys matching pattern"""
        if not self._initialized:
            await self.initialize()

        try:
            deleted_count = 0
            async with self._lock:
                keys_to_delete = []

                for key in self._index.keys():
                    if self._matches_pattern(key, pattern):
                        keys_to_delete.append(key)

                for key in keys_to_delete:
                    entry = self._index[key]
                    await self._remove_entry(key, entry)
                    deleted_count += 1

            return deleted_count

        except Exception as e:
            logger.error(f"TTLCache invalidate error for pattern {pattern}: {e}")
            return 0

    async def clear(self) -> bool:
        """Clear all cache entries"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self._lock:
                # Remove all files
                for entry in self._index.values():
                    if entry.file_path:
                        try:
                            Path(entry.file_path).unlink(missing_ok=True)
                        except Exception as e:
                            logger.warning(f"Failed to delete file {entry.file_path}: {e}")

                # Clear index
                self._index.clear()
                self._current_size_bytes = 0
                self.metrics.reset()

            return True

        except Exception as e:
            logger.error(f"TTLCache clear error: {e}")
            return False

    async def cleanup_expired(self) -> int:
        """Cleanup expired entries"""
        if not self._initialized:
            await self.initialize()

        try:
            expired_count = 0
            current_time = time.time()

            async with self._lock:
                expired_keys = []

                for key, entry in self._index.items():
                    if current_time > entry.expires_at:
                        expired_keys.append(key)

                for key in expired_keys:
                    entry = self._index[key]
                    await self._remove_entry(key, entry)
                    expired_count += 1

            if expired_count > 0:
                logger.debug(f"Cleaned up {expired_count} expired entries")

            return expired_count

        except Exception as e:
            logger.error(f"TTLCache cleanup error: {e}")
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics"""
        if not self._initialized:
            await self.initialize()

        try:
            async with self._lock:
                stats = self.metrics.get_stats().__dict__.copy()

                # Count expired entries
                current_time = time.time()
                expired_count = sum(
                    1 for entry in self._index.values()
                    if current_time > entry.expires_at
                )

                # Add TTL-specific stats
                stats.update({
                    'cache_size_items': len(self._index),
                    'cache_size_bytes': self._current_size_bytes,
                    'cache_size_mb': self._current_size_bytes / (1024 * 1024),
                    'max_size_mb': self.max_size_bytes / (1024 * 1024),
                    'disk_usage_percent': (self._current_size_bytes / self.max_size_bytes) * 100,
                    'expired_entries': expired_count,
                    'cache_directory': str(self.cache_dir),
                    'compression_enabled': self.enable_compression,
                    'default_ttl': self.default_ttl,
                    'avg_ttl_remaining': await self._calculate_avg_ttl_remaining(),
                    'file_count': len([p for p in self.cache_dir.iterdir() if p.is_file()])
                })

                return stats

        except Exception as e:
            logger.error(f"TTLCache stats error: {e}")
            return {}

    async def optimize(self) -> bool:
        """Optimize cache performance"""
        if not self._initialized:
            await self.initialize()

        try:
            # Cleanup expired entries
            await self.cleanup_expired()

            # Compact cache directory if needed
            await self._compact_cache_files()

            return True

        except Exception as e:
            logger.error(f"TTLCache optimization error: {e}")
            return False

    async def _recover_from_disk(self):
        """Recover cache index from disk files"""
        try:
            if not self.cache_dir.exists():
                return

            index_file = self.cache_dir / "cache_index.json"
            if index_file.exists():
                async with aiofiles.open(index_file, 'r') as f:
                    content = await f.read()
                    index_data = json.loads(content)

                    # Restore index
                    current_time = time.time()
                    for key, entry_data in index_data.items():
                        if entry_data['expires_at'] > current_time:
                            # Check if file exists
                            file_path = Path(entry_data['file_path'])
                            if file_path.exists():
                                entry = TTLEntry(**entry_data)
                                self._index[key] = entry
                                self._current_size_bytes += entry.size_bytes

                logger.info(f"Recovered {len(self._index)} cache entries from disk")

        except Exception as e:
            logger.warning(f"Cache recovery failed: {e}")

    async def _save_index_to_disk(self):
        """Save cache index to disk"""
        try:
            index_file = self.cache_dir / "cache_index.json"
            index_data = {
                key: {
                    'created_at': entry.created_at,
                    'expires_at': entry.expires_at,
                    'access_count': entry.access_count,
                    'size_bytes': entry.size_bytes,
                    'file_path': entry.file_path
                }
                for key, entry in self._index.items()
            }

            async with aiofiles.open(index_file, 'w') as f:
                await f.write(json.dumps(index_data, indent=2))

        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")

    async def _load_value(self, entry: TTLEntry) -> Optional[Any]:
        """Load value from disk"""
        if not entry.file_path:
            return entry.value

        try:
            file_path = Path(entry.file_path)
            if not file_path.exists():
                return None

            async with aiofiles.open(file_path, 'rb') as f:
                data = await f.read()

            # Decompress if needed
            if self.enable_compression and file_path.suffix == '.gz':
                data = gzip.decompress(data)

            # Deserialize
            return pickle.loads(data)

        except Exception as e:
            logger.error(f"Failed to load value from {entry.file_path}: {e}")
            return None

    async def _save_value(self, entry: TTLEntry) -> bool:
        """Save value to disk"""
        try:
            file_path = Path(entry.file_path)

            # Serialize
            data = pickle.dumps(entry.value)

            # Compress if enabled and beneficial
            if self.enable_compression and len(data) > 1024:  # Only compress if >1KB
                compressed_data = gzip.compress(data, compresslevel=6)
                if len(compressed_data) < len(data) * 0.9:  # Only if 10%+ savings
                    data = compressed_data
                    file_path = file_path.with_suffix(file_path.suffix + '.gz')
                    entry.file_path = str(file_path)

            # Save to disk
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(data)

            return True

        except Exception as e:
            logger.error(f"Failed to save value to {entry.file_path}: {e}")
            return False

    async def _remove_entry(self, key: str, entry: TTLEntry):
        """Remove entry from cache and disk"""
        try:
            # Remove from index
            if key in self._index:
                del self._index[key]
                self._current_size_bytes -= entry.size_bytes

            # Remove file
            if entry.file_path:
                file_path = Path(entry.file_path)
                if file_path.exists():
                    file_path.unlink()

        except Exception as e:
            logger.error(f"Failed to remove entry {key}: {e}")

    async def _ensure_capacity(self, needed_bytes: int):
        """Ensure sufficient disk capacity"""
        while self._current_size_bytes + needed_bytes > self.max_size_bytes:
            if not self._index:
                break

            # Find oldest entry to evict
            oldest_key = min(
                self._index.keys(),
                key=lambda k: self._index[k].created_at
            )
            oldest_entry = self._index[oldest_key]
            await self._remove_entry(oldest_key, oldest_entry)

    async def _get_file_path(self, key: str) -> Path:
        """Generate unique file path for key"""
        # Create hash to avoid filesystem issues
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"cache_{key_hash[:16]}.pkl"

    async def _calculate_size(self, obj: Any) -> int:
        """Calculate object size"""
        try:
            # Serialize to get accurate size
            data = pickle.dumps(obj)
            return len(data)
        except Exception:
            # Fallback estimation
            return len(str(obj)) * 2

    async def _calculate_avg_ttl_remaining(self) -> float:
        """Calculate average TTL remaining for all entries"""
        if not self._index:
            return 0.0

        current_time = time.time()
        total_remaining = sum(
            max(0, entry.expires_at - current_time)
            for entry in self._index.values()
        )

        return total_remaining / len(self._index)

    async def _compact_cache_files(self):
        """Compact cache directory by removing orphaned files"""
        try:
            if not self.cache_dir.exists():
                return

            # Get all cache files
            cache_files = {
                p for p in self.cache_dir.iterdir()
                if p.is_file() and p.name.startswith('cache_') and p.suffix in ['.pkl', '.gz']
            }

            # Get referenced files
            referenced_files = {
                Path(entry.file_path) for entry in self._index.values()
                if entry.file_path
            }

            # Remove orphaned files
            orphaned_files = cache_files - referenced_files
            for file_path in orphaned_files:
                try:
                    file_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove orphaned file {file_path}: {e}")

            if orphaned_files:
                logger.info(f"Removed {len(orphaned_files)} orphaned cache files")

        except Exception as e:
            logger.error(f"Cache compaction failed: {e}")

    async def _start_cleanup_task(self):
        """Start background cleanup task"""
        if self._cleanup_task and not self._cleanup_task.done():
            return

        self._cleanup_running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self._cleanup_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                if self._cleanup_running:
                    await self.cleanup_expired()
                    await self._save_index_to_disk()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

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
        """Shutdown cache and cleanup resources"""
        try:
            # Stop cleanup task
            self._cleanup_running = False
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            # Save index
            await self._save_index_to_disk()

            # Clear memory
            async with self._lock:
                self._index.clear()
                self._current_size_bytes = 0

            logger.info("TTLCache shutdown completed")

        except Exception as e:
            logger.error(f"TTLCache shutdown error: {e}")