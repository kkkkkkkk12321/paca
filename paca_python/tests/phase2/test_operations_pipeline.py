import asyncio
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "email_validator" not in sys.modules:  # pragma: no cover - dependency shim for tests
    class _EmailNotValidError(Exception):
        ...

    def _validate_email(email, *_args, **_kwargs):
        return types.SimpleNamespace(email=email)

    sys.modules["email_validator"] = types.SimpleNamespace(
        EmailNotValidError=_EmailNotValidError,
        validate_email=_validate_email,
    )

from paca.operations import (
    GUIRegressionResult,
    GUIRegressionRunner,
    OperationsPipeline,
    OpsMonitoringBridge,
    PhaseRegressionRunner,
    PipelineComponentResult,
    PipelineResult,
    RegressionRunResult,
)


class _StubRegressionRunner(PhaseRegressionRunner):
    def __init__(self, results):
        self._results = results

    async def run(self):  # type: ignore[override]
        return self._results


class _StubGUIRunner(GUIRegressionRunner):
    def __init__(self, result: GUIRegressionResult):
        self._result = result

    async def run(self) -> GUIRegressionResult:  # type: ignore[override]
        return self._result


@pytest.mark.asyncio
async def test_operations_pipeline_exports_monitoring_payload(tmp_path: Path):
    regression_results = [
        RegressionRunResult(
            module="tests/test_a.py",
            status="passed",
            duration_s=0.1,
            return_code=0,
        ),
        RegressionRunResult(
            module="tests/test_b.py",
            status="passed",
            duration_s=0.2,
            return_code=0,
        ),
    ]

    monitoring_path = tmp_path / "ops.json"
    pipeline = OperationsPipeline(
        regression_runner=_StubRegressionRunner(regression_results),
        gui_runner=_StubGUIRunner(
            GUIRegressionResult(status="skipped", duration_s=0.0, detail="tk unavailable")
        ),
        monitoring_bridge=OpsMonitoringBridge(export_path=monitoring_path),
    )

    result = await pipeline.run()

    assert result.success
    assert monitoring_path.exists()

    payload = json.loads(monitoring_path.read_text(encoding="utf-8"))
    assert payload["components"]["phase_regression"]["status"] == "passed"
    assert payload["components"]["gui_regression"]["status"] == "skipped"
    assert payload["components"]["phase_regression"]["metadata"]["modules"]


@pytest.mark.asyncio
async def test_phase_regression_runner_accepts_custom_runner(tmp_path: Path):
    executed = []

    def fake_runner(command, cwd):
        executed.append((tuple(command), cwd))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    runner = PhaseRegressionRunner(
        modules=("tests/custom_module.py",),
        runner=fake_runner,
        repo_root=tmp_path,
    )

    results = await runner.run()

    assert executed
    assert results[0].module == "tests/custom_module.py"
    assert results[0].status == "passed"


def test_ops_monitoring_bridge_initializes_without_event_loop(tmp_path: Path):
    bridge = OpsMonitoringBridge(export_path=tmp_path / "ops.json")
    result = PipelineResult(
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        components=[
            PipelineComponentResult(
                name="phase_regression",
                status="passed",
                duration_s=0.1,
            )
        ],
    )

    asyncio.run(bridge.publish(result))

    assert (tmp_path / "ops.json").exists(), "monitoring payload should persist in synchronous context"


def test_ops_monitoring_bridge_survives_multiple_asyncio_run_calls(tmp_path: Path):
    bridge = OpsMonitoringBridge(export_path=tmp_path / "ops.json")
    result = PipelineResult(
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        components=[
            PipelineComponentResult(
                name="phase_regression",
                status="passed",
                duration_s=0.1,
            )
        ],
    )

    locks = []
    loops = []
    observed_closed_states = []

    for _ in range(3):
        asyncio.run(bridge.publish(result))
        assert bridge._write_lock is not None
        assert bridge._write_lock_loop is not None
        locks.append(bridge._write_lock)
        loops.append(bridge._write_lock_loop)
        observed_closed_states.append(bridge._write_lock_loop.is_closed())

    payload = json.loads((tmp_path / "ops.json").read_text(encoding="utf-8"))
    assert payload["components"]["phase_regression"]["status"] == "passed"
    assert all(
        locks[i] is not locks[i + 1] for i in range(len(locks) - 1)
    ), "new asyncio loops should trigger new lock objects"
    assert all(
        loops[i] is not loops[i + 1] for i in range(len(loops) - 1)
    ), "each asyncio.run call should attach a new loop reference"
    assert all(observed_closed_states), "each event loop should be closed after asyncio.run"


def test_operations_pipeline_survives_multiple_asyncio_run_calls(tmp_path: Path):
    regression_results = [
        RegressionRunResult(
            module="tests/test_a.py",
            status="passed",
            duration_s=0.1,
            return_code=0,
        )
    ]

    monitoring_path = tmp_path / "ops.json"
    pipeline = OperationsPipeline(
        regression_runner=_StubRegressionRunner(regression_results),
        gui_runner=_StubGUIRunner(
            GUIRegressionResult(status="passed", duration_s=0.05, detail="")
        ),
        monitoring_bridge=OpsMonitoringBridge(export_path=monitoring_path),
    )

    for _ in range(3):
        result = asyncio.run(pipeline.run())
        assert result.success

    payload = json.loads(monitoring_path.read_text(encoding="utf-8"))
    assert payload["components"]["gui_regression"]["status"] == "passed"

