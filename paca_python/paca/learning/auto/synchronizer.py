"""Synchronization helpers for exporting auto-learning snapshots."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import threading
from typing import Dict, List, Optional, Protocol


@dataclass
class LearningDataSnapshot:
    """Immutable snapshot of the auto-learning engine state."""

    saved_at: float
    learning_points: List[Dict[str, object]]
    generated_tactics: List[Dict[str, object]]
    generated_heuristics: List[Dict[str, object]]
    metrics: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class LearningDataSynchronizer(Protocol):
    """Interface that synchronizes learning snapshots with external systems."""

    async def sync(self, snapshot: LearningDataSnapshot) -> None:  # pragma: no cover - protocol
        ...


class FileLearningDataSynchronizer:
    """Persist snapshots to JSON for ingestion by monitoring dashboards."""

    def __init__(self, export_path: Path) -> None:
        self.export_path = export_path
        self._lock: Optional[asyncio.Lock] = None
        self._lock_guard = threading.Lock()

    async def sync(self, snapshot: LearningDataSnapshot) -> None:
        lock = self._ensure_lock()

        async with lock:
            await asyncio.to_thread(self._write_snapshot, snapshot)

    def _write_snapshot(self, snapshot: LearningDataSnapshot) -> None:
        payload = snapshot.to_dict()
        payload.setdefault("metadata", {})
        payload["metadata"]["export_path"] = str(self.export_path)

        self.export_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        self.export_path.write_text(serialized, encoding="utf-8")

    def _ensure_lock(self) -> asyncio.Lock:
        lock = self._lock
        if lock is not None:
            return lock

        with self._lock_guard:
            lock = self._lock
            if lock is None:
                lock = asyncio.Lock()
                self._lock = lock
        return lock


class CompositeLearningDataSynchronizer:
    """Fan-out synchronizer used to coordinate multiple sinks."""

    def __init__(self, *synchronizers: LearningDataSynchronizer) -> None:
        self._synchronizers = synchronizers

    async def sync(self, snapshot: LearningDataSnapshot) -> None:
        await asyncio.gather(*(sync.sync(snapshot) for sync in self._synchronizers))


def build_default_synchronizer(storage_root: Path) -> LearningDataSynchronizer:
    """Create the default synchronizer that mirrors snapshots to monitoring storage."""

    export_path = storage_root / "monitoring" / "learning_snapshot.json"
    return FileLearningDataSynchronizer(export_path)


__all__ = [
    "LearningDataSnapshot",
    "LearningDataSynchronizer",
    "FileLearningDataSynchronizer",
    "CompositeLearningDataSynchronizer",
    "build_default_synchronizer",
]
