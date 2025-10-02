import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paca.operations import (
    GUIRegressionResult,
    GUIRegressionRunner,
    OperationsPipeline,
    OpsMonitoringBridge,
    PhaseRegressionRunner,
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
