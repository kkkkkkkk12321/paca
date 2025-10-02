"""LLM API Module exposing client and ensemble helpers."""

from .base import (
    LLMInterface,
    LLMResponse,
    LLMRequest,
    LLMError,
    LLMConfig,
    ModelType,
    GenerationConfig,
)

from .gemini_client import (
    GeminiClientManager,
    GeminiResponse,
    GeminiConfig,
    create_gemini_client,
)

from .ensemble import (
    LLMEnsembleCandidate,
    EnsembleStrategyResult,
    merge_candidates,
)

__all__ = [
    # Base interfaces
    "LLMInterface",
    "LLMResponse",
    "LLMRequest",
    "LLMError",
    "LLMConfig",
    "ModelType",
    "GenerationConfig",

    # Gemini client
    "GeminiClientManager",
    "GeminiResponse",
    "GeminiConfig",
    "create_gemini_client",

    # Ensemble utilities
    "LLMEnsembleCandidate",
    "EnsembleStrategyResult",
    "merge_candidates",
]
