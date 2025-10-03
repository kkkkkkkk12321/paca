"""Utilities that serialize operations pipeline results for monitoring dashboards."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import threading

from typing import Any, Dict, Optional

if False:  # pragma: no cover - type checking only
    from .ops_pipeline import PipelineResult


class OpsMonitoringBridge:
    """Persist operations pipeline outcomes to a monitoring-friendly payload."""

    def __init__(
        self,
        *,
        export_path: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.export_path = export_path or Path("logs/ops/ops_pipeline_state.json")
        self.logger = logger or logging.getLogger("paca.operations.pipeline")
        self._write_lock: Optional[asyncio.Lock] = None
        self._write_lock_loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock_guard = threading.Lock()

    async def publish(self, result: "PipelineResult") -> None:
        payload = self.build_payload(result)

        lock = self._ensure_write_lock()

        async with lock:
            await asyncio.to_thread(self._write_payload, payload)

        # Structured logging for dashboards / alerting systems.
        if self.logger:
            self.logger.info("ops_pipeline_completed", extra={"ops_pipeline": payload})

    def build_payload(self, result: "PipelineResult") -> Dict[str, Any]:
        components = {component.name: component.to_dict() for component in result.components}
        payload: Dict[str, Any] = {
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat(),
            "duration_s": result.duration_s,
            "success": result.success,
            "components": components,
        }
        return payload

    def _write_payload(self, payload: Dict[str, Any]) -> None:
        path = self.export_path
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        path.write_text(serialized, encoding="utf-8")

    def _ensure_write_lock(self) -> asyncio.Lock:
        current_loop = asyncio.get_running_loop()

        lock = self._write_lock
        lock_loop = self._write_lock_loop

        if _needs_new_lock(lock, lock_loop, current_loop):
            with self._lock_guard:
                lock = self._write_lock
                lock_loop = self._write_lock_loop
                if _needs_new_lock(lock, lock_loop, current_loop):
                    lock = asyncio.Lock()
                    self._write_lock = lock
                    self._write_lock_loop = current_loop

        assert lock is not None  # narrow Optional for type-checkers
        return lock


def _needs_new_lock(
    lock_obj: Optional[asyncio.Lock],
    lock_loop: Optional[asyncio.AbstractEventLoop],
    current_loop: asyncio.AbstractEventLoop,
) -> bool:
    return (
        lock_obj is None
        or lock_loop is None
        or lock_loop.is_closed()
        or lock_loop is not current_loop
    )


__all__ = ["OpsMonitoringBridge"]
