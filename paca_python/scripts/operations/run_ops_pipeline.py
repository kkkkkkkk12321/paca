"""CLI entrypoint that runs the operations automation pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Optional

from paca.operations import (
    GUIRegressionRunner,
    OperationsPipeline,
    OpsMonitoringBridge,
    PhaseRegressionRunner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PACA operations pipeline")
    parser.add_argument(
        "--export",
        type=Path,
        help="Optional path where the monitoring payload should be written",
    )
    parser.add_argument(
        "--no-fail-fast",
        action="store_true",
        help="Run all regression modules even if one fails",
    )
    parser.add_argument(
        "--extra-pytest",
        nargs=argparse.REMAINDER,
        help="Additional pytest arguments passed to regression modules",
    )
    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> int:
    regression_runner = PhaseRegressionRunner(
        fail_fast=not args.no_fail_fast,
        extra_args=tuple(args.extra_pytest or []),
    )

    monitoring_bridge: Optional[OpsMonitoringBridge]
    if args.export:
        monitoring_bridge = OpsMonitoringBridge(export_path=args.export)
    else:
        monitoring_bridge = OpsMonitoringBridge()

    pipeline = OperationsPipeline(
        regression_runner=regression_runner,
        gui_runner=GUIRegressionRunner(),
        monitoring_bridge=monitoring_bridge,
    )

    result = await pipeline.run()
    payload = monitoring_bridge.build_payload(result)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.success else 1


def main() -> int:
    args = parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
