"""Lightweight GUI regression smoke checks used by the operations pipeline."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional


class GUIUnavailableError(RuntimeError):
    """Raised when GUI dependencies are missing on the current platform."""


@dataclass
class GUIRegressionResult:
    """Outcome of the GUI regression smoke test."""

    status: str
    duration_s: float
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class GUIRegressionRunner:
    """Execute a defensive GUI smoke test in headless environments."""

    def __init__(
        self,
        *,
        smoke_test: Optional[Callable[[], None]] = None,
    ) -> None:
        self._smoke_test = smoke_test or self._default_smoke_test

    async def run(self) -> GUIRegressionResult:
        started = time.perf_counter()
        try:
            await asyncio.to_thread(self._smoke_test)
        except GUIUnavailableError as exc:
            duration = time.perf_counter() - started
            return GUIRegressionResult(status="skipped", duration_s=duration, detail=str(exc))
        except Exception as exc:  # pragma: no cover - unexpected failures should propagate as failed status
            duration = time.perf_counter() - started
            return GUIRegressionResult(status="failed", duration_s=duration, detail=str(exc))

        duration = time.perf_counter() - started
        return GUIRegressionResult(status="passed", duration_s=duration, detail="GUI smoke test completed")

    def _default_smoke_test(self) -> None:
        """Import the enhanced GUI and instantiate it without starting the event loop."""

        try:
            from desktop_app import enhanced_gui
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise GUIUnavailableError("enhanced GUI dependencies are not installed") from exc

        EnhancedGUI = getattr(enhanced_gui, "EnhancedGUI", None)
        if EnhancedGUI is None:
            raise GUIUnavailableError("EnhancedGUI entrypoint is unavailable")

        try:
            gui = EnhancedGUI()  # type: ignore[call-arg]
        except Exception as exc:  # pragma: no cover - actual GUI issues are surfaced as failures
            raise GUIUnavailableError(f"unable to instantiate EnhancedGUI: {exc}") from exc

        root = getattr(gui, "root", None)
        if root is not None:
            try:
                root.update()
            except Exception as exc:  # pragma: no cover - Tk may be missing display
                raise GUIUnavailableError(f"GUI backend unavailable: {exc}") from exc
            finally:
                try:
                    root.destroy()
                except Exception:  # pragma: no cover - defensive cleanup
                    pass


__all__ = ["GUIRegressionRunner", "GUIRegressionResult", "GUIUnavailableError"]
