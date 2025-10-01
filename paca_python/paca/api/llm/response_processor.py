"""
Response Processing System
LLM 응답의 품질 검증, 안전성 필터링, 컨텍스트 관리를 담당
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

from ...core.types import Result
from ...core.utils.logger import PacaLogger as StructuredLogger
from .base import LLMResponse, LLMError


class SafetyLevel(Enum):
    """안전성 레벨"""
    SAFE = "safe"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    BLOCKED = "blocked"


class QualityLevel(Enum):
    """품질 레벨"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    REJECTED = "rejected"


@dataclass
class ValidationResult:
    """검증 결과"""
    is_valid: bool
    quality_score: float
    safety_level: SafetyLevel
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingMetrics:
    """처리 메트릭"""
    validation_time: float = 0.0
    safety_check_time: float = 0.0
    quality_score: float = 0.0
    token_count: int = 0
    character_count: int = 0


class ContentValidator:
    """컨텐츠 검증기"""

    def __init__(self):
        self.logger = StructuredLogger("ContentValidator")

        # 금지된 패턴들
        self.unsafe_patterns = [
            r'(?i)\b(hack|crack|exploit|vulnerability)\b.*(?:instruction|guide|tutorial)',
            r'(?i)\b(illegal|harmful|dangerous)\b.*(?:activity|action)',
            r'(?i)\b(generate|create|make)\b.*(?:virus|malware|weapon)',
        ]

        # 품질 검증 패턴들
        self.quality_patterns = {
            'coherence': r'[.!?]\s+[A-Z]',  # 문장 구조
            'repetition': r'\b(\w+)\s+\1\b',  # 단어 반복
            'incomplete': r'(?:\.{3,}|\[.*\]|\(.*\))$',  # 불완전한 끝
        }

    async def validate_content(self, text: str) -> ValidationResult:
        """컨텐츠 검증"""
        issues = []
        suggestions = []
        safety_level = SafetyLevel.SAFE
        quality_score = 100.0

        try:
            # 안전성 검증
            safety_result = await self._check_safety(text)
            safety_level = safety_result['level']
            if safety_result['issues']:
                issues.extend(safety_result['issues'])

            # 품질 검증
            quality_result = await self._check_quality(text)
            quality_score = quality_result['score']
            if quality_result['issues']:
                issues.extend(quality_result['issues'])
                suggestions.extend(quality_result['suggestions'])

            # 전체 유효성 판단
            is_valid = (
                safety_level in [SafetyLevel.SAFE, SafetyLevel.CAUTION] and
                quality_score >= 60.0
            )

            return ValidationResult(
                is_valid=is_valid,
                quality_score=quality_score,
                safety_level=safety_level,
                issues=issues,
                suggestions=suggestions,
                metadata={
                    'text_length': len(text),
                    'word_count': len(text.split()),
                    'sentence_count': len(re.findall(r'[.!?]+', text))
                }
            )

        except Exception as e:
            self.logger.error(f"Content validation error: {str(e)}")
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                safety_level=SafetyLevel.UNSAFE,
                issues=[f"Validation failed: {str(e)}"]
            )

    async def _check_safety(self, text: str) -> Dict[str, Any]:
        """안전성 검증"""
        issues = []
        level = SafetyLevel.SAFE

        # 위험한 패턴 검사
        for pattern in self.unsafe_patterns:
            if re.search(pattern, text):
                issues.append(f"Detected potentially unsafe content: {pattern}")
                level = SafetyLevel.UNSAFE

        # 길이 검증
        if len(text) > 50000:  # 너무 긴 응답
            issues.append("Response too long")
            level = SafetyLevel.CAUTION

        # 빈 내용 검증
        if not text.strip():
            issues.append("Empty response")
            level = SafetyLevel.BLOCKED

        return {'level': level, 'issues': issues}

    async def _check_quality(self, text: str) -> Dict[str, Any]:
        """품질 검증"""
        issues = []
        suggestions = []
        score = 100.0

        # 텍스트 기본 검증
        if len(text.strip()) < 10:
            issues.append("Response too short")
            score -= 30

        # 반복성 검증
        repetition_matches = re.findall(self.quality_patterns['repetition'], text)
        if len(repetition_matches) > 3:
            issues.append("Excessive word repetition detected")
            suggestions.append("Reduce repetitive language")
            score -= 20

        # 완성도 검증
        if re.search(self.quality_patterns['incomplete'], text):
            issues.append("Response appears incomplete")
            suggestions.append("Ensure complete responses")
            score -= 15

        # 일관성 검증
        sentence_count = len(re.findall(r'[.!?]+', text))
        word_count = len(text.split())
        if sentence_count > 0 and word_count / sentence_count < 5:
            issues.append("Sentences too short on average")
            score -= 10

        return {
            'score': max(0, score),
            'issues': issues,
            'suggestions': suggestions
        }


class ContextManager:
    """컨텍스트 관리자"""

    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.conversation_history: List[Dict[str, Any]] = []
        self.user_preferences: Dict[str, Any] = {}
        self.session_context: Dict[str, Any] = {}
        self.logger = StructuredLogger("ContextManager")

    async def add_exchange(
        self,
        user_input: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """대화 교환 추가"""
        exchange = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'assistant_response': assistant_response,
            'metadata': metadata or {}
        }

        self.conversation_history.append(exchange)

        # 히스토리 크기 제한
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

        self.logger.debug(f"Added conversation exchange, total: {len(self.conversation_history)}")

    async def get_context_for_request(self, current_input: str) -> Dict[str, Any]:
        """요청을 위한 컨텍스트 생성"""
        context = {
            'session_context': self.session_context.copy(),
            'user_preferences': self.user_preferences.copy(),
            'recent_history': self.conversation_history[-5:] if self.conversation_history else [],
            'current_input': current_input,
            'context_summary': await self._generate_context_summary()
        }

        return context

    async def _generate_context_summary(self) -> str:
        """컨텍스트 요약 생성"""
        if not self.conversation_history:
            return "No previous conversation history."

        summary_lines: List[str] = []
        for exchange in self.conversation_history[-3:]:
            user_input = (exchange.get('user_input') or '').strip()
            assistant_response = (exchange.get('assistant_response') or '').strip()

            if user_input:
                summary_lines.append(f"사용자: {user_input}")
            if assistant_response:
                summary_lines.append(f"PACA: {assistant_response}")

        if not summary_lines:
            return "No recent conversation details available."

        joined = " | ".join(summary_lines)
        # 지나치게 긴 요약은 잘라낸다 (LLM 입력 안정성을 위함)
        return joined[:1000]

    def update_user_preferences(self, preferences: Dict[str, Any]) -> None:
        """사용자 선호도 업데이트"""
        self.user_preferences.update(preferences)

    def set_session_context(self, key: str, value: Any) -> None:
        """세션 컨텍스트 설정"""
        self.session_context[key] = value

    def clear_history(self) -> None:
        """히스토리 초기화"""
        self.conversation_history.clear()


class TokenUsageMonitor:
    """토큰 사용량 모니터"""

    def __init__(self):
        self.daily_usage = {}
        self.session_usage = 0
        self.total_usage = 0
        self.logger = StructuredLogger("TokenMonitor")

    async def track_usage(self, tokens_used: int, model: str) -> Dict[str, Any]:
        """토큰 사용량 추적"""
        today = datetime.now().date().isoformat()

        # 일일 사용량 업데이트
        if today not in self.daily_usage:
            self.daily_usage[today] = {}
        if model not in self.daily_usage[today]:
            self.daily_usage[today][model] = 0

        self.daily_usage[today][model] += tokens_used
        self.session_usage += tokens_used
        self.total_usage += tokens_used

        usage_stats = {
            'tokens_used': tokens_used,
            'session_total': self.session_usage,
            'daily_total': sum(self.daily_usage[today].values()),
            'model_daily': self.daily_usage[today][model],
            'estimated_cost': await self._estimate_cost(tokens_used, model)
        }

        self.logger.info(f"Token usage tracked: {tokens_used} tokens for {model}")
        return usage_stats

    async def _estimate_cost(self, tokens: int, model: str) -> float:
        """비용 추정"""
        # Gemini API 대략적인 가격 (실제 가격은 변동될 수 있음)
        cost_per_1k_tokens = {
            'gemini-2.5-pro': 0.0025,
            'gemini-2.5-flash': 0.00025,
            'gemini-2.5-flash-image-preview': 0.00025
        }

        rate = cost_per_1k_tokens.get(model, 0.001)
        return (tokens / 1000) * rate

    def get_usage_summary(self) -> Dict[str, Any]:
        """사용량 요약"""
        today = datetime.now().date().isoformat()
        return {
            'session_usage': self.session_usage,
            'daily_usage': self.daily_usage.get(today, {}),
            'total_usage': self.total_usage
        }


class ResponseProcessor:
    """통합 응답 처리기"""

    def __init__(self):
        self.validator = ContentValidator()
        self.context_manager = ContextManager()
        self.token_monitor = TokenUsageMonitor()
        self.logger = StructuredLogger("ResponseProcessor")

    async def process_response(
        self,
        response: LLMResponse,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Result[Tuple[LLMResponse, ProcessingMetrics]]:
        """응답 처리"""
        start_time = asyncio.get_event_loop().time()

        try:
            # 컨텍스트 검증
            validation_start = asyncio.get_event_loop().time()
            validation_result = await self.validator.validate_content(response.text)
            validation_time = asyncio.get_event_loop().time() - validation_start

            # 응답이 유효하지 않으면 거부
            if not validation_result.is_valid:
                self.logger.warning(f"Response validation failed: {validation_result.issues}")
                return Result(
                    False,
                    None,
                    f"Response validation failed: {'; '.join(validation_result.issues)}"
                )

            # 토큰 사용량 추적
            token_count = response.usage.get('total_tokens', len(response.text.split()))
            usage_stats = await self.token_monitor.track_usage(token_count, response.model.value)

            # 컨텍스트에 추가
            await self.context_manager.add_exchange(
                user_input,
                response.text,
                {
                    'model': response.model.value,
                    'processing_time': response.processing_time,
                    'validation_score': validation_result.quality_score,
                    'usage_stats': usage_stats
                }
            )

            # 메트릭 생성
            processing_time = asyncio.get_event_loop().time() - start_time
            metrics = ProcessingMetrics(
                validation_time=validation_time,
                safety_check_time=validation_time,  # 같은 과정에서 처리
                quality_score=validation_result.quality_score,
                token_count=token_count,
                character_count=len(response.text)
            )

            # 응답에 처리 정보 추가
            response.metadata.update({
                'validation_result': validation_result,
                'usage_stats': usage_stats,
                'processing_metrics': metrics
            })

            self.logger.info(f"Response processed successfully with score: {validation_result.quality_score}")
            return Result(True, (response, metrics))

        except Exception as e:
            self.logger.error(f"Response processing failed: {str(e)}")
            return Result(False, None, f"Processing failed: {str(e)}")

    async def get_context_for_next_request(self, user_input: str) -> Dict[str, Any]:
        """다음 요청을 위한 컨텍스트 생성"""
        return await self.context_manager.get_context_for_request(user_input)

    async def cleanup(self) -> None:
        """리소스 정리"""
        self.context_manager.clear_history()
        self.logger.info("Response processor cleaned up")


# 팩토리 함수
def create_response_processor() -> ResponseProcessor:
    """응답 처리기 생성 헬퍼"""
    return ResponseProcessor()