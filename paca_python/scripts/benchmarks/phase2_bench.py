"""Phase 2 cognitive pipeline micro-benchmark skeleton.

This script focuses on the ComplexityDetector + MetacognitionEngine
path introduced during Phase 2 sprint 1.  It is intentionally light-weight and
uses synthetic inputs so that it can run inside constrained CI environments.

Usage
-----
python scripts/benchmarks/phase2_bench.py              # default rounds
python scripts/benchmarks/phase2_bench.py --rounds 50  # customise workload
python scripts/benchmarks/phase2_bench.py --json benchmark.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import sys

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

logging.getLogger("paca").setLevel(logging.CRITICAL)
logging.getLogger("paca.cognitive.metacognition_engine").setLevel(logging.CRITICAL)
logging.getLogger("paca.cognitive.complexity_detector").setLevel(logging.CRITICAL)

from paca.cognitive import ComplexityDetector, MetacognitionEngine


SAMPLE_COMPLEX_INPUTS: Sequence[str] = (
    "만약 사용자가 분산 시스템 설계에서 데이터 일관성과 가용성을 동시에 확보하려면 어떤 전략을 선택해야 할까요?",
    "신규 회원 온보딩 플로우에서 드롭오프율을 15% 이하로 낮추려면 어떤 실험 설계를 적용하는 것이 좋을지 단계별로 정리해 주세요.",
    "한국어 자연어 처리 파이프라인을 구축할 때 형태소 분석기와 워드 임베딩 계층을 어떻게 조합해야 추론 성능이 개선되는지 근거를 들어 설명해 주세요.",
)


@dataclass
class BenchmarkResult:
    name: str
    rounds: int
    elapsed_ms: float

    @property
    def per_round_ms(self) -> float:
        return self.elapsed_ms / self.rounds if self.rounds else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.name,
            "rounds": self.rounds,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "per_round_ms": round(self.per_round_ms, 3),
        }


async def run_complexity_benchmark(detector: ComplexityDetector, rounds: int) -> BenchmarkResult:
    start = time.perf_counter()
    for idx in range(rounds):
        text = SAMPLE_COMPLEX_INPUTS[idx % len(SAMPLE_COMPLEX_INPUTS)]
        await detector.detect_complexity(text)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult("complexity_detect", rounds, elapsed)


async def run_metacognition_benchmark(engine: MetacognitionEngine, rounds: int) -> BenchmarkResult:
    start = time.perf_counter()
    for idx in range(rounds):
        context = {
            "user_id": f"bench_user_{idx}",
            "message": SAMPLE_COMPLEX_INPUTS[idx % len(SAMPLE_COMPLEX_INPUTS)],
            "complexity_score": 70 + (idx % 20),
        }
        session_id = await engine.start_reasoning_monitoring(context)
        await engine.add_reasoning_step(
            session_id=session_id,
            step_description="1. 상황 분석",
            input_data={"premise": "mock premise"},
            output_data={"conclusion": "mock conclusion", "confidence": 0.55},
            processing_time_ms=1200,
        )
        await engine.add_reasoning_step(
            session_id=session_id,
            step_description="2. 전략 비교",
            input_data={"premise": "mock conclusion"},
            output_data={"conclusion": "refined answer", "confidence": 0.45},
            processing_time_ms=1500,
        )
        await engine.end_monitoring_session(session_id)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult("metacognition_quality", rounds, elapsed)


async def run_benchmarks(rounds: int) -> List[BenchmarkResult]:
    detector = ComplexityDetector()
    engine = MetacognitionEngine()

    complexity_result, metacog_result = await asyncio.gather(
        run_complexity_benchmark(detector, rounds),
        run_metacognition_benchmark(engine, rounds),
    )

    return [complexity_result, metacog_result]


def summarise(results: Iterable[BenchmarkResult]) -> Dict[str, Any]:
    per_round = [result.per_round_ms for result in results]
    return {
        "results": [result.to_dict() for result in results],
        "per_round_ms": {
            "mean": round(statistics.mean(per_round), 3),
            "stdev": round(statistics.pstdev(per_round), 3) if len(per_round) > 1 else 0.0,
            "max": round(max(per_round), 3),
            "min": round(min(per_round), 3),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 cognitive benchmark skeleton")
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of synthetic requests per benchmark (default: 10)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path to dump JSON results",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = asyncio.run(run_benchmarks(args.rounds))
    payload = summarise(results)

    print("\nPhase 2 benchmark summary")
    for result in payload["results"]:
        print(
            f"- {result['benchmark']}: {result['rounds']} rounds, "
            f"total {result['elapsed_ms']} ms, per round {result['per_round_ms']} ms"
        )

    stats = payload["per_round_ms"]
    print(
        f"\nPer-round statistics (ms) :: mean {stats['mean']} | "
        f"stdev {stats['stdev']} | max {stats['max']} | min {stats['min']}"
    )

    if args.json:
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"\nJSON results written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
