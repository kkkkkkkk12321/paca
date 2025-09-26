"""Helper script to execute Phase 1 regression tests alongside emerging Phase 2 suites.

The CLI sandbox enforces a ~10s wall clock limit per command, so we execute
individual pytest invocations sequentially rather than a single aggregated run.

Usage examples
--------------
python scripts/testing/run_phase_regression_tests.py                # run phase1+phase2
python scripts/testing/run_phase_regression_tests.py --phase phase1 # only legacy suite
python scripts/testing/run_phase_regression_tests.py --extra "-vv"   # pass through flags
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


PHASE1_MODULES: Sequence[str] = (
    "tests/test_system_basic.py",
    "tests/test_reasoning_basic.py",
    "tests/test_system_phase1.py",
)

PHASE2_MODULES: Sequence[str] = (
    "tests/phase2/test_complexity_metacognition.py",
    "tests/phase2/test_phase2_pipeline.py",
    "tests/phase2/test_memory_layers.py",
    "tests/phase2/test_episodic_memory.py",
    "tests/phase2/test_longterm_memory.py",
)


@dataclass
class TestGroup:
    name: str
    modules: Sequence[str]


GROUPS = {
    "phase1": TestGroup("Phase 1 regression", PHASE1_MODULES),
    "phase2": TestGroup("Phase 2 cognitive", PHASE2_MODULES),
}


DEFAULT_PYTEST_ARGS: Sequence[str] = ("--no-cov", "-q", "--maxfail=1")


def build_command(module: str, extra_args: Sequence[str]) -> List[str]:
    """Return a python -m pytest invocation for the given module."""

    pytest_module = [sys.executable, "-m", "pytest", module]
    if extra_args:
        pytest_module.extend(extra_args)
    else:
        pytest_module.extend(DEFAULT_PYTEST_ARGS)
    return pytest_module


def run_command(cmd: Sequence[str], *, cwd: Path) -> int:
    """Execute a command, wiring PYTHONPATH so packages resolve from repo root."""

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(cwd))

    print(f"\n[pytest] running: {' '.join(cmd)}")
    process = subprocess.run(cmd, cwd=cwd, env=env)
    return process.returncode


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase regression test bundles")
    parser.add_argument(
        "--phase",
        choices=("phase1", "phase2", "all"),
        default="all",
        help="Select which suite to execute (default: all)",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed verbatim to pytest (overrides default fast settings)",
    )
    return parser.parse_args()


def iter_selected_groups(selection: str) -> Iterable[TestGroup]:
    if selection == "all":
        yield GROUPS["phase1"]
        yield GROUPS["phase2"]
    else:
        yield GROUPS[selection]


def main() -> int:
    args = parse_arguments()
    repo_root = Path(__file__).resolve().parents[2]

    extra_args: Sequence[str] = args.extra or []

    failures: list[str] = []

    for group in iter_selected_groups(args.phase):
        print(f"\n=== Executing {group.name} suite ===")
        for module in group.modules:
            cmd = build_command(module, extra_args)
            rc = run_command(cmd, cwd=repo_root)
            if rc != 0:
                failures.append(module)

    if failures:
        print("\nTest failures detected in modules:")
        for module in failures:
            print(f" - {module}")
        return 1

    print("\nAll selected test modules completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
