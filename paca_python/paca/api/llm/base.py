"""
LLM Base Interfaces
LLM 클라이언트를 위한 기본 인터페이스와 타입 정의
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from ...core.types import Result, create_id


class ModelType(Enum):
    """LLM 모델 타입"""
    GEMINI_PRO = "gemini-2.5-pro"
    GEMINI_FLASH = "gemini-2.5-flash"
    GEMINI_IMAGE = "gemini-2.5-flash-image-preview"
    GEMINI_IMAGE_GENERATION = "gemini-2.0-flash-preview-image-generation"


@dataclass
class GenerationConfig:
    """텍스트 생성 설정"""
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    top_k: int = 40
    stop_sequences: List[str] = field(default_factory=list)
    response_mime_type: str = "text/plain"

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop_sequences": self.stop_sequences,
            "response_mime_type": self.response_mime_type
        }


@dataclass
class LLMRequest:
    """LLM 요청 데이터"""
    id: str = field(default_factory=create_id)
    prompt: str = ""
    system_prompt: Optional[str] = None
    model: ModelType = ModelType.GEMINI_FLASH
    config: Optional[GenerationConfig] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LLMResponse:
    """LLM 응답 데이터"""
    id: str
    text: str
    model: ModelType
    usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0

    @property
    def is_success(self) -> bool:
        """성공 여부"""
        return bool(self.text and len(self.text.strip()) > 0)


class LLMError(Exception):
    """LLM 관련 오류"""
    def __init__(self, message: str, error_code: str = "UNKNOWN", details: Optional[Dict] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now()


@dataclass
class LLMConfig:
    """LLM 클라이언트 설정"""
    api_keys: List[str] = field(default_factory=list)
    default_model: ModelType = ModelType.GEMINI_FLASH
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    rate_limit_per_minute: int = 60
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds


class LLMInterface(ABC):
    """LLM 클라이언트 기본 인터페이스"""

    @abstractmethod
    async def generate_text(self, request: LLMRequest) -> Result[LLMResponse]:
        """텍스트 생성"""
        pass

    @abstractmethod
    async def generate_with_context(
        self,
        prompt: str,
        context: List[Dict[str, str]],
        model: Optional[ModelType] = None,
        config: Optional[GenerationConfig] = None
    ) -> Result[LLMResponse]:
        """컨텍스트를 포함한 텍스트 생성"""
        pass

    @abstractmethod
    async def batch_generate(self, requests: List[LLMRequest]) -> List[Result[LLMResponse]]:
        """배치 텍스트 생성"""
        pass

    @abstractmethod
    async def get_available_models(self) -> List[ModelType]:
        """사용 가능한 모델 목록"""
        pass

    @abstractmethod
    async def health_check(self) -> Result[Dict[str, Any]]:
        """서비스 상태 확인"""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """리소스 정리"""
        pass


# 유틸리티 함수들
def create_llm_request(
    prompt: str,
    model: ModelType = ModelType.GEMINI_FLASH,
    system_prompt: Optional[str] = None,
    config: Optional[GenerationConfig] = None,
    context: Optional[Dict[str, Any]] = None
) -> LLMRequest:
    """LLM 요청 생성 헬퍼"""
    return LLMRequest(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        config=config or GenerationConfig(),
        context=context or {}
    )


def create_generation_config(
    temperature: float = 0.7,
    max_tokens: int = 2048,
    **kwargs
) -> GenerationConfig:
    """생성 설정 생성 헬퍼"""
    return GenerationConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
