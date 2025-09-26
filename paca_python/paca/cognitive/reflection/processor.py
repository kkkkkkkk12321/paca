"""
Self-Reflection Processor
자기 성찰 루프의 메인 처리기
"""

import asyncio
import time
from typing import Optional, Dict, Any

from ...core.types import Result, create_success, create_failure
from ...core.utils.logger import PacaLogger
from ...api.llm.base import LLMRequest, GenerationConfig, ModelType
from .base import (
    ReflectionResult,
    CritiqueResult,
    ReflectionConfig,
    ReflectionLevel,
    create_reflection_config
)
from .critique import CritiqueAnalyzer
from .improvement import IterativeImprover


class SelfReflectionProcessor:
    """
    자기 성찰 루프 처리기
    1차 응답 → 비평적 검토 → 개선된 2차 응답 생성
    """

    def __init__(self, llm_client=None, config: Optional[ReflectionConfig] = None):
        self.llm_client = llm_client
        self.config = config or create_reflection_config()
        self.logger = PacaLogger("SelfReflection")

        # 서브 컴포넌트 초기화
        self.critique_analyzer = CritiqueAnalyzer(llm_client, self.config)
        self.iterative_improver = IterativeImprover(llm_client, self.config)

        # 통계 및 모니터링
        self.stats = {
            "total_reflections": 0,
            "successful_improvements": 0,
            "average_quality_gain": 0.0,
            "average_processing_time": 0.0
        }

    async def process_with_reflection(self, user_input: str, initial_response: str) -> Result[ReflectionResult]:
        """
        자기 성찰을 통한 응답 처리

        Args:
            user_input: 사용자 입력
            initial_response: 초기 생성된 응답

        Returns:
            Result[ReflectionResult]: 성찰 처리 결과
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting reflection process for input: {user_input[:50]}...")

            # 1단계: 초기 응답에 대한 자기 비평 수행
            critique_result = await self.critique_analyzer.analyze_response(
                initial_response, user_input
            )

            if not critique_result.is_success:
                self.logger.error(f"Critique analysis failed: {critique_result.error}")
                return create_failure(f"Critique analysis failed: {critique_result.error}")

            critique = critique_result.data

            # 2단계: 개선 필요성 판단
            final_response = initial_response
            improvement_applied = False
            iterations_performed = 0
            quality_improvement = 0.0

            if critique.needs_improvement and critique.overall_quality_score < self.config.quality_threshold:
                self.logger.info(f"Improvement needed. Current quality: {critique.overall_quality_score:.1f}")

                # 3단계: 반복적 개선 수행
                improvement_result = await self.iterative_improver.improve_iteratively(
                    initial_response, user_input, critique
                )

                if improvement_result.is_success:
                    improved_data = improvement_result.data
                    final_response = improved_data["improved_response"]
                    improvement_applied = True
                    iterations_performed = improved_data["iterations_performed"]
                    quality_improvement = improved_data["quality_improvement"]

                    self.logger.info(f"Improvement completed. Quality gain: {quality_improvement:.1f}")
                else:
                    self.logger.warning(f"Improvement failed: {improvement_result.error}")

            # 4단계: 결과 생성
            processing_time = time.time() - start_time

            reflection_result = ReflectionResult(
                user_input=user_input,
                initial_response=initial_response,
                final_response=final_response,
                critique=critique,
                improvement_applied=improvement_applied,
                iterations_performed=iterations_performed,
                total_processing_time=processing_time,
                quality_improvement=quality_improvement,
                config_used=self.config.to_dict()
            )

            # 통계 업데이트
            await self._update_stats(reflection_result)

            self.logger.info(f"Reflection completed in {processing_time:.2f}s")
            return create_success(reflection_result)

        except asyncio.TimeoutError:
            self.logger.error("Reflection process timed out")
            return create_failure("Reflection process timed out")
        except Exception as e:
            self.logger.error(f"Unexpected error in reflection: {str(e)}")
            return create_failure(f"Reflection failed: {str(e)}")

    async def generate_initial_response(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Result[str]:
        """
        초기 응답 생성 (LLM 클라이언트 사용)

        Args:
            user_input: 사용자 입력
            context: 추가 컨텍스트 정보

        Returns:
            Result[str]: 생성된 초기 응답
        """
        if not self.llm_client:
            return create_failure("LLM client not available")

        try:
            # 컨텍스트 포함한 프롬프트 구성
            prompt = self._build_initial_prompt(user_input, context)

            request = LLMRequest(
                prompt=prompt,
                model=ModelType.GEMINI_FLASH,
                config=GenerationConfig(
                    temperature=0.7,
                    max_tokens=2048
                )
            )

            result = await self.llm_client.generate_text(request)

            if result.is_success:
                response_text = result.data.text.strip()
                self.logger.debug(f"Generated initial response: {response_text[:100]}...")
                return create_success(response_text)
            else:
                return create_failure(f"Failed to generate initial response: {result.error}")

        except Exception as e:
            self.logger.error(f"Error generating initial response: {str(e)}")
            return create_failure(f"Failed to generate response: {str(e)}")

    async def quick_reflection(self, user_input: str, initial_response: str) -> Result[ReflectionResult]:
        """
        빠른 성찰 (기본 수준)

        Args:
            user_input: 사용자 입력
            initial_response: 초기 응답

        Returns:
            Result[ReflectionResult]: 성찰 결과
        """
        # 임시로 낮은 수준의 성찰 설정
        quick_config = ReflectionConfig(
            reflection_level=ReflectionLevel.BASIC,
            quality_threshold=70.0,
            max_iterations=1,
            max_processing_time=10.0
        )

        original_config = self.config
        self.config = quick_config

        try:
            result = await self.process_with_reflection(user_input, initial_response)
            return result
        finally:
            self.config = original_config

    def _build_initial_prompt(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """초기 응답 생성용 프롬프트 구성"""
        prompt = f"사용자 질문: {user_input}\n\n"

        if context:
            prompt += "추가 컨텍스트:\n"
            for key, value in context.items():
                prompt += f"- {key}: {value}\n"
            prompt += "\n"

        prompt += """도움이 되고 정확한 응답을 제공해주세요.
응답은 다음 기준을 만족해야 합니다:
1. 논리적으로 일관성이 있어야 합니다
2. 사실적으로 정확해야 합니다
3. 완전하고 포괄적이어야 합니다
4. 질문과 관련성이 있어야 합니다
5. 명확하고 이해하기 쉬워야 합니다

응답:"""
        return prompt

    async def _update_stats(self, result: ReflectionResult):
        """통계 정보 업데이트"""
        self.stats["total_reflections"] += 1

        if result.improvement_applied:
            self.stats["successful_improvements"] += 1

        # 이동 평균 계산
        n = self.stats["total_reflections"]
        self.stats["average_quality_gain"] = (
            (self.stats["average_quality_gain"] * (n - 1) + result.quality_improvement) / n
        )
        self.stats["average_processing_time"] = (
            (self.stats["average_processing_time"] * (n - 1) + result.total_processing_time) / n
        )

    async def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        success_rate = 0.0
        if self.stats["total_reflections"] > 0:
            success_rate = self.stats["successful_improvements"] / self.stats["total_reflections"] * 100

        return {
            **self.stats,
            "improvement_success_rate": success_rate,
            "config": self.config.to_dict()
        }

    async def cleanup(self):
        """리소스 정리"""
        if self.critique_analyzer:
            await self.critique_analyzer.cleanup()
        if self.iterative_improver:
            await self.iterative_improver.cleanup()

        self.logger.info("SelfReflectionProcessor cleaned up")