"""Operations automation pipeline tying regression checks to monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .gui_regression import GUIRegressionRunner
from .monitoring_bridge import OpsMonitoringBridge
from .regression import PhaseRegressionRunner, RegressionRunResult


@dataclass
class PipelineComponentResult:
    """Normalized component result for monitoring serialization."""

    name: str
    status: str
    duration_s: float
    detail: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "status": self.status,
            "duration_s": self.duration_s,
        }
        if self.detail:
            payload["detail"] = self.detail
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass
class PipelineResult:
    """Overall summary of an operations pipeline execution."""

    started_at: datetime
    finished_at: datetime
    components: List[PipelineComponentResult]

    @property
    def duration_s(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def success(self) -> bool:
        for component in self.components:
            if component.status == "failed":
                return False
        # If every component was skipped we consider the run unsuccessful
        return any(component.status == "passed" for component in self.components)


class OperationsPipeline:
    """Coordinates regression suites, GUI checks, and monitoring export."""

    def __init__(
        self,
        *,
        regression_runner: Optional[PhaseRegressionRunner] = None,
        gui_runner: Optional[GUIRegressionRunner] = None,
        monitoring_bridge: Optional[OpsMonitoringBridge] = None,
    ) -> None:
        self.regression_runner = regression_runner or PhaseRegressionRunner()
        self.gui_runner = gui_runner or GUIRegressionRunner()
        self.monitoring_bridge = monitoring_bridge or OpsMonitoringBridge()

    async def run(self) -> PipelineResult:
        started_at = datetime.utcnow()
        components: List[PipelineComponentResult] = []

        regression_results = await self.regression_runner.run()
        regression_status = self._compute_regression_status(regression_results)
        components.append(regression_status)

        gui_result = await self.gui_runner.run()
        components.append(
            PipelineComponentResult(
                name="gui_regression",
                status=gui_result.status,
                duration_s=gui_result.duration_s,
                detail=gui_result.detail,
            )
        )

        finished_at = datetime.utcnow()
        result = PipelineResult(
            started_at=started_at,
            finished_at=finished_at,
            components=components,
        )

        if self.monitoring_bridge is not None:
            await self.monitoring_bridge.publish(result)

        return result

    def _compute_regression_status(
        self, regression_results: List[RegressionRunResult]
    ) -> PipelineComponentResult:
        if not regression_results:
            return PipelineComponentResult(
                name="phase_regression",
                status="skipped",
                duration_s=0.0,
                detail="no regression modules executed",
            )

        status = "passed"
        total_duration = 0.0
        metadata: Dict[str, object] = {"modules": []}
        module_entries: List[Dict[str, object]] = []

        for entry in regression_results:
            total_duration += entry.duration_s
            module_payload: Dict[str, object] = entry.to_dict()
            module_entries.append(module_payload)
            if entry.status == "failed":
                status = "failed"

        metadata["modules"] = module_entries

        return PipelineComponentResult(
            name="phase_regression",
            status=status,
            duration_s=total_duration,
            metadata=metadata,
        )


__all__ = ["OperationsPipeline", "PipelineComponentResult", "PipelineResult"]
