"""Utilities for merging responses from multiple LLM providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass
class LLMEnsembleCandidate:
    """Single model contribution to an ensemble response."""

    model: str
    content: str
    confidence: float
    metadata: Optional[Dict[str, object]] = None


@dataclass
class EnsembleStrategyResult:
    """Aggregated response emitted by the ensemble merger."""

    content: str
    aggregated_confidence: float
    supporting_models: List[str] = field(default_factory=list)
    combined_metadata: List[Dict[str, object]] = field(default_factory=list)


def merge_candidates(
    candidates: Iterable[LLMEnsembleCandidate],
    *,
    minimum_confidence: float = 0.0,
) -> List[EnsembleStrategyResult]:
    """Merge responses that share the same canonical content."""

    buckets: Dict[str, Dict[str, object]] = {}

    for candidate in candidates:
        if candidate.confidence < minimum_confidence:
            continue

        key = candidate.content.strip()
        bucket = buckets.setdefault(
            key,
            {
                "content": candidate.content,
                "confidence": 0.0,
                "supporting_models": [],
                "metadata": [],
            },
        )

        bucket["confidence"] += max(candidate.confidence, 0.0)
        bucket["supporting_models"].append(candidate.model)
        if candidate.metadata:
            bucket["metadata"].append(candidate.metadata)

    ranked = sorted(
        buckets.values(),
        key=lambda payload: (payload["confidence"], len(payload["supporting_models"])),
        reverse=True,
    )

    results: List[EnsembleStrategyResult] = []
    for payload in ranked:
        count = max(len(payload["supporting_models"]), 1)
        averaged_confidence = payload["confidence"] / count
        results.append(
            EnsembleStrategyResult(
                content=payload["content"],
                aggregated_confidence=averaged_confidence,
                supporting_models=list(payload["supporting_models"]),
                combined_metadata=list(payload["metadata"]),
            )
        )

    return results


__all__ = [
    "LLMEnsembleCandidate",
    "EnsembleStrategyResult",
    "merge_candidates",
]
