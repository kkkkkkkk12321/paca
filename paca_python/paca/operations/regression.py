"""Regression test orchestrators used by the operations pipeline."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Optional, Sequence


@dataclass
class RegressionRunResult:
    """Summary of a single regression module invocation."""

    module: str
    status: str
    duration_s: float
    return_code: int
    log_path: Optional[Path] = None
    stdout: str | None = None
    stderr: str | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        if self.log_path is not None:
            payload["log_path"] = str(self.log_path)
        return payload


class PhaseRegressionRunner:
    """Execute the curated Phase regression pytest modules sequentially."""

    DEFAULT_MODULES: Sequence[str] = (
        "tests/test_system_basic.py",
        "tests/test_reasoning_basic.py",
        "tests/test_system_phase1.py",
        "tests/phase2/test_complexity_metacognition.py",
        "tests/phase2/test_phase2_pipeline.py",
    )

    def __init__(
        self,
        *,
        modules: Optional[Sequence[str]] = None,
        runner: Optional[Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]] = None,
        repo_root: Optional[Path] = None,
        fail_fast: bool = True,
        extra_args: Optional[Sequence[str]] = None,
        log_directory: Optional[Path] = None,
    ) -> None:
        self.modules: Sequence[str] = modules or self.DEFAULT_MODULES
        self._runner = runner or self._default_runner
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]
        self.fail_fast = fail_fast
        self.extra_args = tuple(extra_args or ("--maxfail=1", "-q"))
        self.log_directory = log_directory

    def _default_runner(
        self, command: Sequence[str], cwd: Path
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )

    def _build_command(self, module: str) -> List[str]:
        command = [sys.executable, "-m", "pytest", module]
        command.extend(self.extra_args)
        return command

    async def run(self) -> List[RegressionRunResult]:
        """Execute regression modules sequentially and capture their summaries."""

        results: List[RegressionRunResult] = []
        for module in self.modules:
            command = self._build_command(module)
            started = time.perf_counter()
            completed = await asyncio.to_thread(self._runner, command, self.repo_root)
            duration = time.perf_counter() - started

            status = "passed" if completed.returncode == 0 else "failed"
            log_path: Optional[Path] = None
            if self.log_directory is not None:
                log_directory = self.log_directory
                log_directory.mkdir(parents=True, exist_ok=True)
                log_path = log_directory / f"{Path(module).stem}.log"
                log_path.write_text(
                    (completed.stdout or "") + "\n" + (completed.stderr or ""),
                    encoding="utf-8",
                )

            results.append(
                RegressionRunResult(
                    module=module,
                    status=status,
                    duration_s=duration,
                    return_code=completed.returncode,
                    log_path=log_path,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )
            )

            if status == "failed" and self.fail_fast:
                break

        return results


__all__ = ["PhaseRegressionRunner", "RegressionRunResult"]
