"""
Iterative Improver
반복적 개선 처리기
"""

import asyncio
import time
from typing import Dict, Any, Optional

from ...core.types import Result, create_success, create_failure
from ...core.utils.logger import PacaLogger
from ...api.llm.base import LLMRequest, GenerationConfig, ModelType
from .base import CritiqueResult, ReflectionConfig


class IterativeImprover:
    """반복적 개선 처리기"""

    def __init__(self, llm_client=None, config: Optional[ReflectionConfig] = None):
        self.llm_client = llm_client
        self.config = config
        self.logger = PacaLogger("IterativeImprover")

    async def improve_iteratively(self, initial_response: str, user_input: str, critique: CritiqueResult) -> Result[Dict[str, Any]]:
        """
        품질 임계값 달성까지 반복 개선

        Args:
            initial_response: 초기 응답
            user_input: 사용자 입력
            critique: 비평 결과

        Returns:
            Result[Dict]: 개선 결과
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting iterative improvement. Initial quality: {critique.overall_quality_score:.1f}")

            current_response = initial_response
            current_quality = critique.overall_quality_score
            iterations_performed = 0

            # 개선 히스토리 추적
            improvement_history = []

            for iteration in range(self.config.max_iterations):
                if current_quality >= self.config.quality_threshold:
                    self.logger.info(f"Quality threshold reached: {current_quality:.1f}")
                    break

                # 시간 제한 확인
                elapsed_time = time.time() - start_time
                if elapsed_time > self.config.improvement_timeout:
                    self.logger.warning("Improvement process timed out")
                    break

                self.logger.info(f"Iteration {iteration + 1}: Attempting improvement...")

                # 단일 개선 수행
                improvement_result = await self._perform_single_improvement(
                    current_response, user_input, critique
                )

                if not improvement_result.is_success:
                    self.logger.error(f"Improvement iteration {iteration + 1} failed: {improvement_result.error}")
                    break

                improved_data = improvement_result.data
                improved_response = improved_data["improved_response"]
                new_quality = improved_data["quality_score"]

                # 개선 정도 계산
                quality_gain = new_quality - current_quality

                improvement_history.append({
                    "iteration": iteration + 1,
                    "previous_quality": current_quality,
                    "new_quality": new_quality,
                    "quality_gain": quality_gain,
                    "response_length": len(improved_response)
                })

                # 최소 개선 임계값 확인
                if quality_gain < self.config.min_improvement_threshold:
                    self.logger.info(f"Minimal improvement detected ({quality_gain:.1f}). Stopping iterations.")
                    break

                # 결과 업데이트
                current_response = improved_response
                current_quality = new_quality
                iterations_performed = iteration + 1

                self.logger.info(f"Iteration {iteration + 1} completed. Quality: {current_quality:.1f} (+{quality_gain:.1f})")

            # 전체 개선 정도 계산
            total_quality_improvement = current_quality - critique.overall_quality_score

            processing_time = time.time() - start_time

            result_data = {
                "improved_response": current_response,
                "initial_quality": critique.overall_quality_score,
                "final_quality": current_quality,
                "quality_improvement": total_quality_improvement,
                "iterations_performed": iterations_performed,
                "processing_time": processing_time,
                "improvement_history": improvement_history
            }

            self.logger.info(f"Iterative improvement completed. Total gain: {total_quality_improvement:.1f}")
            return create_success(result_data)

        except asyncio.TimeoutError:
            self.logger.error("Iterative improvement timed out")
            return create_failure("Improvement process timed out")
        except Exception as e:
            self.logger.error(f"Error in iterative improvement: {str(e)}")
            return create_failure(f"Improvement failed: {str(e)}")

    async def _perform_single_improvement(self, response: str, user_input: str, critique: CritiqueResult) -> Result[Dict[str, Any]]:
        """단일 개선 수행"""
        try:
            # 개선 프롬프트 생성
            improvement_prompt = self._build_improvement_prompt(response, user_input, critique)

            request = LLMRequest(
                prompt=improvement_prompt,
                model=ModelType.GEMINI_FLASH,
                config=GenerationConfig(
                    temperature=self.config.improvement_model_temperature,
                    max_tokens=2048
                )
            )

            # LLM 호출
            result = await self.llm_client.generate_text(request)

            if not result.is_success:
                return create_failure(f"LLM generation failed: {result.error}")

            improved_response = result.data.text.strip()

            # 개선된 응답의 품질 추정
            estimated_quality = await self._estimate_quality(improved_response, user_input)

            return create_success({
                "improved_response": improved_response,
                "quality_score": estimated_quality,
                "processing_time": result.data.processing_time
            })

        except Exception as e:
            return create_failure(f"Single improvement failed: {str(e)}")

    def _build_improvement_prompt(self, response: str, user_input: str, critique: CritiqueResult) -> str:
        """개선 프롬프트 구성"""
        prompt = f"""다음 응답을 비평 결과를 바탕으로 개선해주세요.

원본 질문: {user_input}

현재 응답:
{response}

비평 분석 결과:
- 전체 품질 점수: {critique.overall_quality_score:.1f}/100
- 논리적 일관성: {critique.logical_consistency:.1f}/100
- 사실적 정확성: {critique.factual_accuracy:.1f}/100
- 완전성: {critique.completeness:.1f}/100
- 관련성: {critique.relevance:.1f}/100
- 명확성: {critique.clarity:.1f}/100

"""

        # 주요 약점들 추가
        if critique.weaknesses:
            prompt += "주요 개선 필요 사항:\n"
            for weakness in critique.weaknesses[:5]:  # 상위 5개
                prompt += f"- {weakness.description} (심각도: {weakness.severity:.1f})\n"

        # 개선 제안들 추가
        if critique.improvements:
            prompt += "\n구체적 개선 제안:\n"
            for improvement in critique.improvements[:3]:  # 상위 3개
                prompt += f"- {improvement.suggestion}\n"

        prompt += f"""

개선 지침:
1. 비평에서 지적된 문제점들을 구체적으로 해결하세요
2. 전체 품질 점수를 {self.config.quality_threshold}점 이상으로 높이세요
3. 원본의 좋은 부분은 유지하면서 문제점만 개선하세요
4. 응답의 구조와 흐름을 개선하세요
5. 더 정확하고 완전한 정보를 제공하세요

개선된 응답:"""

        return prompt

    async def _estimate_quality(self, response: str, user_input: str) -> float:
        """개선된 응답의 품질 추정"""
        try:
            # 간단한 휴리스틱 기반 품질 추정
            quality_score = 50.0  # 기본 점수

            # 길이 기반 평가 (너무 짧거나 길면 감점)
            response_length = len(response)
            if 50 <= response_length <= 2000:
                quality_score += 10.0
            elif response_length < 20:
                quality_score -= 20.0

            # 구조화 평가 (문단, 목록 등)
            if '\n\n' in response or '•' in response or '-' in response:
                quality_score += 5.0

            # 구체성 평가 (예시, 숫자 등)
            if any(char.isdigit() for char in response):
                quality_score += 5.0

            # 정중함 평가
            polite_words = ['감사', '죄송', '안녕하세요', '도움', '참고']
            if any(word in response for word in polite_words):
                quality_score += 5.0

            # 불완전한 문장 감점
            if response.endswith('...') or response.endswith('입니다.') is False:
                quality_score -= 5.0

            return max(0.0, min(100.0, quality_score))

        except Exception as e:
            self.logger.warning(f"Quality estimation failed: {str(e)}")
            return 75.0  # 기본값

    async def quick_improve(self, response: str, user_input: str, target_areas: list = None) -> Result[str]:
        """
        빠른 개선 (특정 영역에 대해서만)

        Args:
            response: 개선할 응답
            user_input: 사용자 입력
            target_areas: 개선할 영역 리스트 (예: ['clarity', 'completeness'])

        Returns:
            Result[str]: 개선된 응답
        """
        try:
            target_areas = target_areas or ['clarity']

            prompt = f"""다음 응답을 {', '.join(target_areas)} 측면에서 빠르게 개선해주세요.

질문: {user_input}
응답: {response}

개선 영역: {', '.join(target_areas)}

개선된 응답:"""

            request = LLMRequest(
                prompt=prompt,
                model=ModelType.GEMINI_FLASH,
                config=GenerationConfig(
                    temperature=0.5,
                    max_tokens=1024
                )
            )

            result = await self.llm_client.generate_text(request)

            if result.is_success:
                return create_success(result.data.text.strip())
            else:
                return create_failure(f"Quick improvement failed: {result.error}")

        except Exception as e:
            return create_failure(f"Quick improvement error: {str(e)}")

    async def cleanup(self):
        """리소스 정리"""
        self.logger.info("IterativeImprover cleaned up")