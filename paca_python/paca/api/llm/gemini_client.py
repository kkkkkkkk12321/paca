"""
Gemini API Client
Google Gemini API와의 통신을 담당하는 클라이언트
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import os
import random

try:
    import google.genai as genai
except ImportError:
    print("Warning: google-genai not installed. Run: pip install google-genai")
    genai = None

from ...core.types import Result
from ...core.utils.logger import PacaLogger as StructuredLogger
from .base import (
    LLMInterface, LLMRequest, LLMResponse, LLMError, LLMConfig,
    ModelType, GenerationConfig
)


@dataclass
class GeminiConfig(LLMConfig):
    """Gemini 전용 설정"""
    api_base_url: str = "https://generativelanguage.googleapis.com"
    safety_settings: Dict[str, str] = field(default_factory=lambda: {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
    })
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    rotation_strategy: str = "round_robin"
    rotation_min_interval: float = 1.0
    model_preferences: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class GeminiResponse(LLMResponse):
    """Gemini 전용 응답"""
    safety_ratings: List[Dict] = field(default_factory=list)
    finish_reason: str = ""
    candidate_count: int = 1


class APIKeyManager:
    """API 키 로테이션 관리"""

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        rotation_strategy: str = "round_robin",
        min_interval_seconds: float = 1.0
    ):
        self.rotation_strategy = rotation_strategy
        self.min_interval_seconds = max(min_interval_seconds, 0.0)
        self.api_keys: List[str] = []
        self.failed_keys: set[str] = set()
        self.last_used: Dict[str, float] = {}
        self.current_index: int = 0
        self.set_keys(api_keys or [])

    def set_keys(self, api_keys: List[str]) -> None:
        """API 키 전체 설정"""
        unique_keys: List[str] = []
        for key in api_keys:
            cleaned = (key or "").strip()
            if cleaned and cleaned not in unique_keys:
                unique_keys.append(cleaned)

        self.api_keys = unique_keys
        self.failed_keys.intersection_update(self.api_keys)
        self.last_used = {key: self.last_used.get(key, 0.0) for key in self.api_keys}
        if self.api_keys:
            self.current_index %= len(self.api_keys)
        else:
            self.current_index = 0

    def add_keys(self, api_keys: List[str]) -> None:
        """API 키 추가"""
        updated = False
        for key in api_keys:
            cleaned = (key or "").strip()
            if cleaned and cleaned not in self.api_keys:
                self.api_keys.append(cleaned)
                self.last_used.setdefault(cleaned, 0.0)
                updated = True

        if updated and self.api_keys:
            self.current_index %= len(self.api_keys)

    def remove_key(self, api_key: str) -> None:
        """특정 API 키 제거"""
        cleaned = (api_key or "").strip()
        if cleaned in self.api_keys:
            self.api_keys.remove(cleaned)
            self.failed_keys.discard(cleaned)
            self.last_used.pop(cleaned, None)
            if self.api_keys:
                self.current_index %= len(self.api_keys)
            else:
                self.current_index = 0

    def get_keys(self) -> List[str]:
        """현재 등록된 키 목록 반환"""
        return list(self.api_keys)

    def get_next_key(self) -> Optional[str]:
        """다음 사용할 API 키 반환"""
        if not self.api_keys:
            return None

        total_keys = len(self.api_keys)
        current_time = time.time()

        if self.rotation_strategy == "random":
            order = random.sample(range(total_keys), total_keys)
        else:
            start_index = self.current_index
            order = [(start_index + offset) % total_keys for offset in range(total_keys)]

        for idx in order:
            key = self.api_keys[idx]
            if key in self.failed_keys:
                continue

            last_used = self.last_used.get(key, 0.0)
            if current_time - last_used < self.min_interval_seconds:
                continue

            self.last_used[key] = current_time
            if self.rotation_strategy != "random":
                self.current_index = (idx + 1) % total_keys
            return key

        available_keys = [key for key in self.api_keys if key not in self.failed_keys]
        if not available_keys:
            self.failed_keys.clear()
            return self.get_next_key()

        oldest_key = min(available_keys, key=lambda k: self.last_used.get(k, 0.0))
        self.last_used[oldest_key] = current_time
        if self.rotation_strategy != "random":
            self.current_index = (self.api_keys.index(oldest_key) + 1) % total_keys
        return oldest_key

    def mark_key_failed(self, key: str) -> None:
        """키를 실패로 표시"""
        cleaned = (key or "").strip()
        if cleaned:
            self.failed_keys.add(cleaned)

    def mark_key_success(self, key: str) -> None:
        """키를 성공으로 표시"""
        cleaned = (key or "").strip()
        if cleaned:
            self.failed_keys.discard(cleaned)


class GeminiClientManager(LLMInterface):
    """Gemini API 클라이언트 관리자"""

    def __init__(self, config: Optional[GeminiConfig] = None):
        self.config = config or GeminiConfig()
        self.logger = StructuredLogger("GeminiClient")
        self.key_manager = APIKeyManager(
            self.config.api_keys,
            rotation_strategy=self.config.rotation_strategy,
            min_interval_seconds=self.config.rotation_min_interval
        )
        self.client = None
        self.is_initialized = False

        self.model_preferences = self.config.model_preferences or {}

        # 캐시 관리
        self.cache = {} if self.config.enable_caching else None
        self.cache_timestamps = {}

        # 통계
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "cache_hits": 0
        }

    def get_api_keys(self) -> List[str]:
        """현재 등록된 API 키 목록"""
        return self.key_manager.get_keys()

    def add_api_keys(self, api_keys: List[str]) -> List[str]:
        """API 키 추가"""
        self.key_manager.add_keys(api_keys)
        self.config.api_keys = self.key_manager.get_keys()
        return self.config.api_keys

    def update_api_keys(self, api_keys: List[str]) -> List[str]:
        """API 키 전체 갱신"""
        self.key_manager.set_keys(api_keys)
        self.config.api_keys = self.key_manager.get_keys()
        return self.config.api_keys

    def remove_api_key(self, api_key: str) -> List[str]:
        """API 키 제거"""
        self.key_manager.remove_key(api_key)
        self.config.api_keys = self.key_manager.get_keys()
        return self.config.api_keys

    async def initialize(self) -> Result[bool]:
        """클라이언트 초기화"""
        try:
            if genai is None:
                raise LLMError("google-genai library not installed", "LIBRARY_MISSING")

            env_keys_raw = os.getenv("GEMINI_API_KEYS", "")
            env_keys = [
                key.strip() for key in env_keys_raw.split(",") if key and key.strip()
            ]

            if env_keys:
                self.add_api_keys(env_keys)

            if not self.config.api_keys:
                raise LLMError("No API keys configured", "NO_API_KEYS")

            # 첫 번째 키로 초기화 테스트
            test_key = self.key_manager.get_next_key()
            if test_key:
                # 클라이언트 초기화
                self.client = genai.Client(api_key=test_key)

                # 헬스 체크
                health_result = await self.health_check()
                if health_result.is_success:
                    self.is_initialized = True
                    self.logger.info("Gemini client initialized successfully")
                    return Result(True, True)
                else:
                    raise LLMError(f"Health check failed: {health_result.error}", "HEALTH_CHECK_FAILED")
            else:
                raise LLMError("No valid API keys available", "NO_VALID_KEYS")

        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {str(e)}")
            return Result(False, False, str(e))

    def _get_cache_key(self, request: LLMRequest) -> str:
        """캐시 키 생성"""
        cache_data = {
            "prompt": request.prompt,
            "system_prompt": request.system_prompt,
            "context": request.context,
            "model": request.model.value,
            "config": request.config.to_dict() if request.config else {}
        }
        return str(hash(json.dumps(cache_data, sort_keys=True, default=str)))

    def _is_cache_valid(self, cache_key: str) -> bool:
        """캐시 유효성 확인"""
        if not self.config.enable_caching or cache_key not in self.cache:
            return False

        timestamp = self.cache_timestamps.get(cache_key, 0)
        return time.time() - timestamp < self.config.cache_ttl

    def _prepare_request_payload(self, request: LLMRequest, generation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Gemini SDK 호출에 사용할 파라미터 구성"""
        contents = self._build_contents_from_request(request)

        payload: Dict[str, Any] = {
            "model": request.model.value,
            "contents": contents,
            "config": generation_config,
        }

        if request.system_prompt:
            payload["system_instruction"] = request.system_prompt

        if self.config.safety_settings:
            payload["safety_settings"] = self.config.safety_settings

        return payload

    def _build_contents_from_request(self, request: LLMRequest) -> List[Dict[str, Any]]:
        """요청 컨텍스트와 현재 프롬프트를 Gemini 메시지 배열로 변환"""
        contents: List[Dict[str, Any]] = []
        context = request.context or {}

        # 1) 명시적으로 전달된 prior_messages 우선 사용
        prior_messages = context.get("prior_messages") or []
        for message in prior_messages:
            text = self._clean_text(message.get("content"))
            if not text:
                continue
            role = self._normalize_role(message.get("role"))
            contents.append(self._make_message(role, text))

        # 2) recent_history를 user/model 턴으로 변환
        for exchange in context.get("recent_history", []) or []:
            user_turn = self._clean_text(exchange.get("user_input"))
            assistant_turn = self._clean_text(exchange.get("assistant_response"))

            if user_turn:
                contents.append(self._make_message("user", user_turn))
            if assistant_turn:
                contents.append(self._make_message("model", assistant_turn))

        # 3) 요약/세션/선호도 등 추가 컨텍스트
        summary_text = self._clean_text(context.get("context_summary"))
        if summary_text:
            contents.append(self._make_message("user", f"[대화 요약]\n{summary_text}"))

        session_context = context.get("session_context") or {}
        session_lines = [f"{key}: {value}" for key, value in session_context.items() if value is not None]
        if session_lines:
            contents.append(self._make_message("user", "[세션 정보]\n" + "\n".join(session_lines)))

        user_preferences = context.get("user_preferences") or {}
        preference_lines = [f"{key}: {value}" for key, value in user_preferences.items() if value is not None]
        if preference_lines:
            contents.append(self._make_message("user", "[사용자 선호]\n" + "\n".join(preference_lines)))

        # 4) 현재 사용자 입력을 항상 마지막에 배치
        prompt_text = self._clean_text(request.prompt)
        contents.append(self._make_message("user", prompt_text or ""))

        return contents

    @staticmethod
    def _make_message(role: str, text: str) -> Dict[str, Any]:
        return {
            "role": role,
            "parts": [{"text": text}]
        }

    @staticmethod
    def _clean_text(value: Optional[str]) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _normalize_role(role: Optional[str]) -> str:
        if not role:
            return "user"
        role_lower = str(role).lower()
        if role_lower in {"model", "assistant"}:
            return "model"
        return "user"

    async def _make_request_with_retry(self, request: LLMRequest) -> Result[GeminiResponse]:
        """재시도 로직을 포함한 요청 처리"""
        last_error = None

        for attempt in range(self.config.max_retries):
            api_key = self.key_manager.get_next_key()
            if not api_key:
                return Result(False, None, "No API keys available")

            try:
                # 새 클라이언트 생성 (키별로)
                client = genai.Client(api_key=api_key)

                # 생성 설정
                generation_config = request.config.to_dict() if request.config else {}

                start_time = time.time()

                request_payload = self._prepare_request_payload(request, generation_config)

                # 요청 실행
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    **request_payload
                )

                processing_time = time.time() - start_time

                # 응답 처리
                if response.text:
                    self.key_manager.mark_key_success(api_key)

                    gemini_response = GeminiResponse(
                        id=request.id,
                        text=response.text,
                        model=request.model,
                        processing_time=processing_time,
                        usage=getattr(response, 'usage_metadata', {}).__dict__ if hasattr(response, 'usage_metadata') else {},
                        finish_reason=getattr(response, 'finish_reason', 'STOP'),
                        candidate_count=len(getattr(response, 'candidates', []))
                    )

                    return Result(True, gemini_response)
                else:
                    last_error = LLMError("Empty response from API", "EMPTY_RESPONSE")

            except Exception as e:
                last_error = LLMError(f"API request failed: {str(e)}", "API_ERROR")
                self.key_manager.mark_key_failed(api_key)

                # 재시도 대기
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))

        return Result(False, None, str(last_error))

    async def generate_text(self, request: LLMRequest) -> Result[LLMResponse]:
        """텍스트 생성"""
        if not self.is_initialized:
            return Result(False, None, "Client not initialized")

        try:
            # 캐시 확인
            cache_key = self._get_cache_key(request)
            if self._is_cache_valid(cache_key):
                self.stats["cache_hits"] += 1
                cached_response = self.cache[cache_key]
                self.logger.debug(f"Cache hit for request {request.id}")
                return Result(True, cached_response)

            # 통계 업데이트
            self.stats["total_requests"] += 1

            # 요청 실행
            result = await self._make_request_with_retry(request)

            if result.is_success:
                self.stats["successful_requests"] += 1
                if result.data.usage:
                    self.stats["total_tokens"] += result.data.usage.get("total_tokens", 0)

                # 캐시 저장
                if self.config.enable_caching:
                    self.cache[cache_key] = result.data
                    self.cache_timestamps[cache_key] = time.time()

                self.logger.info(f"Generated text successfully for request {request.id}")
            else:
                self.stats["failed_requests"] += 1
                self.logger.error(f"Failed to generate text: {result.error}")

            return result

        except Exception as e:
            self.stats["failed_requests"] += 1
            self.logger.error(f"Unexpected error in generate_text: {str(e)}")
            return Result(False, None, f"Unexpected error: {str(e)}")

    async def generate_with_context(
        self,
        prompt: str,
        context: List[Dict[str, str]],
        model: Optional[ModelType] = None,
        config: Optional[GenerationConfig] = None
    ) -> Result[LLMResponse]:
        """컨텍스트를 포함한 텍스트 생성"""

        request_context = {
            "prior_messages": context
        } if context else {}

        request = LLMRequest(
            prompt=prompt,
            model=model or self.config.default_model,
            config=config or self.config.generation_config,
            context=request_context
        )

        return await self.generate_text(request)

    async def batch_generate(self, requests: List[LLMRequest]) -> List[Result[LLMResponse]]:
        """배치 텍스트 생성"""
        results = []

        # 동시 실행을 위한 세마포어 (최대 5개 동시 요청)
        semaphore = asyncio.Semaphore(5)

        async def process_request(req):
            async with semaphore:
                return await self.generate_text(req)

        # 모든 요청을 동시에 처리
        tasks = [process_request(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 예외 처리
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    Result(False, None, f"Batch request {i} failed: {str(result)}")
                )
            else:
                processed_results.append(result)

        return processed_results

    async def get_available_models(self) -> List[ModelType]:
        """사용 가능한 모델 목록"""
        models = [
            ModelType.GEMINI_PRO,
            ModelType.GEMINI_FLASH,
            ModelType.GEMINI_IMAGE,
            ModelType.GEMINI_IMAGE_GENERATION
        ]
        # 중복 제거를 위해 dict 사용
        return list(dict.fromkeys(models))

    async def health_check(self) -> Result[Dict[str, Any]]:
        """서비스 상태 확인"""
        try:
            if not self.is_initialized and genai:
                api_key = self.key_manager.get_next_key()
                if api_key:
                    self.client = genai.Client(api_key=api_key)

            # 간단한 텍스트 생성으로 상태 확인
            test_request = LLMRequest(
                prompt="Hello",
                model=ModelType.GEMINI_FLASH,
                config=GenerationConfig(max_tokens=10)
            )

            result = await self._make_request_with_retry(test_request)

            health_data = {
                "status": "healthy" if result.is_success else "unhealthy",
                "available_keys": len([k for k in self.config.api_keys if k not in self.key_manager.failed_keys]),
                "failed_keys": len(self.key_manager.failed_keys),
                "stats": self.stats,
                "cache_size": len(self.cache) if self.cache else 0,
                "timestamp": datetime.now().isoformat()
            }

            return Result(True, health_data)

        except Exception as e:
            return Result(False, None, f"Health check failed: {str(e)}")

    async def cleanup(self) -> None:
        """리소스 정리"""
        try:
            # 캐시 정리
            if self.cache:
                self.cache.clear()
                self.cache_timestamps.clear()

            # 통계 리셋
            self.stats = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_tokens": 0,
                "cache_hits": 0
            }

            self.is_initialized = False
            self.logger.info("Gemini client cleaned up")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")


# 팩토리 함수
def create_gemini_client(
    api_keys: Optional[List[str]] = None,
    default_model: ModelType = ModelType.GEMINI_FLASH,
    **kwargs
) -> GeminiClientManager:
    """Gemini 클라이언트 생성 헬퍼"""
    config = GeminiConfig(
        api_keys=api_keys or [],
        default_model=default_model,
        **kwargs
    )
    return GeminiClientManager(config)
