import pytest

from paca.reasoning import ReasoningEngine, ReasoningType, InferenceRule


@pytest.mark.asyncio
async def test_deductive_reasoning_modus_ponens():
    engine = ReasoningEngine()
    await engine.initialize()

    result = await engine.reason(
        ReasoningType.DEDUCTIVE,
        premises=[
            "If it rains then ground is wet",
            "It rains",
        ],
    )

    assert result.is_success
    data = result.data
    assert data.conclusion == "ground is wet"
    assert data.metadata.get("fallback_used") is False
    assert any(
        step.inference_rule == InferenceRule.MODUS_PONENS.value
        for step in data.reasoning_steps
    )


@pytest.mark.asyncio
async def test_deductive_reasoning_fallback_used():
    engine = ReasoningEngine()
    await engine.initialize()

    premise = "Knowledge is power"
    result = await engine.reason(
        ReasoningType.DEDUCTIVE,
        premises=[premise],
    )

    assert result.is_success
    data = result.data
    assert data.conclusion == premise
    assert data.metadata.get("fallback_used") is True
    assert any(
        step.inference_rule == InferenceRule.DIRECT_INFERENCE.value
        for step in data.reasoning_steps
    )
    assert pytest.approx(data.confidence, rel=0.01) == 0.7


@pytest.mark.asyncio
async def test_deductive_reasoning_modus_tollens():
    engine = ReasoningEngine()
    await engine.initialize()

    result = await engine.reason(
        ReasoningType.DEDUCTIVE,
        premises=[
            "If system fails then alert triggers",
            "Not alert triggers",
        ],
    )

    assert result.is_success
    data = result.data
    assert any(
        step.inference_rule == InferenceRule.MODUS_TOLLENS.value
        for step in data.reasoning_steps
    )
    assert "not" in data.conclusion.lower()


@pytest.mark.asyncio
async def test_deductive_reasoning_hypothetical_syllogism():
    engine = ReasoningEngine()
    await engine.initialize()

    result = await engine.reason(
        ReasoningType.DEDUCTIVE,
        premises=[
            "If AI learns then AI adapts",
            "If AI adapts then users benefit",
        ],
    )

    assert result.is_success
    data = result.data
    assert any(
        step.inference_rule == InferenceRule.HYPOTHETICAL_SYLLOGISM.value
        for step in data.reasoning_steps
    )
    syllogism_step = next(
        step for step in data.reasoning_steps
        if step.inference_rule == InferenceRule.HYPOTHETICAL_SYLLOGISM.value
    )
    assert "if ai learns" in syllogism_step.conclusion.lower()
    assert "users benefit" in syllogism_step.conclusion.lower()
