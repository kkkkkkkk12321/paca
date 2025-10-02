from paca.api.llm.ensemble import (
    EnsembleStrategyResult,
    LLMEnsembleCandidate,
    merge_candidates,
)


def test_merge_candidates_groups_identical_content():
    candidates = [
        LLMEnsembleCandidate(model="gemini-1", content="답변", confidence=0.9, metadata={"latency": 1.2}),
        LLMEnsembleCandidate(model="claude", content="답변", confidence=0.8, metadata={"latency": 1.4}),
        LLMEnsembleCandidate(model="gpt", content="다른", confidence=0.7),
    ]

    results = merge_candidates(candidates)

    assert len(results) == 2
    top = results[0]
    assert top.content == "답변"
    assert set(top.supporting_models) == {"gemini-1", "claude"}
    assert abs(top.aggregated_confidence - ((0.9 + 0.8) / 2)) < 1e-6
    assert top.combined_metadata[0]["latency"] == 1.2


def test_merge_candidates_respects_confidence_threshold():
    candidates = [
        LLMEnsembleCandidate(model="gemini", content="keep", confidence=0.6),
        LLMEnsembleCandidate(model="gpt", content="drop", confidence=0.2),
    ]

    results = merge_candidates(candidates, minimum_confidence=0.5)
    assert len(results) == 1
    assert results[0].content == "keep"
