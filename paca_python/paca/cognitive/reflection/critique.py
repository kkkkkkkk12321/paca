"""
Critique Analyzer
응답에 대한 비평적 분석 수행
"""

import asyncio
import time
import json
import re
from typing import List, Dict, Any, Optional

from ...core.types import Result, create_success, create_failure
from ...core.utils.logger import PacaLogger
from ...api.llm.base import LLMRequest, GenerationConfig, ModelType
from .base import (
    CritiqueResult,
    Weakness,
    Improvement,
    WeaknessType,
    ImprovementType,
    ReflectionConfig,
    calculate_overall_quality
)


class CritiqueAnalyzer:
    """응답에 대한 비평적 분석 수행"""

    def __init__(self, llm_client=None, config: Optional[ReflectionConfig] = None):
        self.llm_client = llm_client
        self.config = config
        self.logger = PacaLogger("CritiqueAnalyzer")

        # 분석 템플릿
        self.critique_prompts = {
            "logical_consistency": self._get_logical_consistency_prompt(),
            "factual_accuracy": self._get_factual_accuracy_prompt(),
            "completeness": self._get_completeness_prompt(),
            "relevance": self._get_relevance_prompt(),
            "clarity": self._get_clarity_prompt()
        }

    async def analyze_response(self, response: str, user_input: str) -> Result[CritiqueResult]:
        """
        응답에 대한 종합적 비평 분석

        Args:
            response: 분석할 응답
            user_input: 원본 사용자 입력

        Returns:
            Result[CritiqueResult]: 비평 분석 결과
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting critique analysis for response: {response[:50]}...")

            # 각 차원별 분석 수행
            analysis_results = await self._perform_multidimensional_analysis(response, user_input)

            if not analysis_results["success"]:
                return create_failure(f"Analysis failed: {analysis_results['error']}")

            scores = analysis_results["scores"]
            weaknesses = analysis_results["weaknesses"]
            improvements = analysis_results["improvements"]

            # 전체 품질 점수 계산
            overall_quality = calculate_overall_quality(
                logical_consistency=scores["logical_consistency"],
                factual_accuracy=scores["factual_accuracy"],
                completeness=scores["completeness"],
                relevance=scores["relevance"],
                clarity=scores["clarity"]
            )

            # 개선 필요성 판단
            needs_improvement = (
                overall_quality < self.config.quality_threshold or
                len([w for w in weaknesses if w.severity >= self.config.severity_threshold]) > 0
            )

            # 비평 추론 생성
            critique_reasoning = await self._generate_critique_reasoning(
                response, user_input, scores, weaknesses
            )

            processing_time = time.time() - start_time

            # 결과 객체 생성
            critique_result = CritiqueResult(
                overall_quality_score=overall_quality,
                needs_improvement=needs_improvement,
                logical_consistency=scores["logical_consistency"],
                factual_accuracy=scores["factual_accuracy"],
                completeness=scores["completeness"],
                relevance=scores["relevance"],
                clarity=scores["clarity"],
                weaknesses=weaknesses,
                improvements=improvements,
                reflection_level=self.config.reflection_level,
                processing_time=processing_time,
                critique_reasoning=critique_reasoning
            )

            self.logger.info(f"Critique completed. Quality score: {overall_quality:.1f}")
            return create_success(critique_result)

        except asyncio.TimeoutError:
            self.logger.error("Critique analysis timed out")
            return create_failure("Analysis timed out")
        except Exception as e:
            self.logger.error(f"Error in critique analysis: {str(e)}")
            return create_failure(f"Analysis failed: {str(e)}")

    async def _perform_multidimensional_analysis(self, response: str, user_input: str) -> Dict[str, Any]:
        """각 차원별 분석을 병렬로 수행"""
        try:
            # 병렬 분석 태스크 생성
            tasks = []
            for dimension, prompt_template in self.critique_prompts.items():
                task = self._analyze_dimension(dimension, response, user_input, prompt_template)
                tasks.append(task)

            # 병렬 실행
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 결과 처리
            scores = {}
            all_weaknesses = []
            all_improvements = []

            for i, result in enumerate(results):
                dimension = list(self.critique_prompts.keys())[i]

                if isinstance(result, Exception):
                    self.logger.error(f"Error analyzing {dimension}: {str(result)}")
                    scores[dimension] = 50.0  # 기본값
                    continue

                if result["success"]:
                    scores[dimension] = result["score"]
                    all_weaknesses.extend(result["weaknesses"])
                    all_improvements.extend(result["improvements"])
                else:
                    self.logger.warning(f"Failed to analyze {dimension}: {result['error']}")
                    scores[dimension] = 50.0  # 기본값

            return {
                "success": True,
                "scores": scores,
                "weaknesses": all_weaknesses,
                "improvements": all_improvements
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _analyze_dimension(self, dimension: str, response: str, user_input: str, prompt_template: str) -> Dict[str, Any]:
        """특정 차원에 대한 분석 수행"""
        try:
            # 프롬프트 구성
            analysis_prompt = prompt_template.format(
                user_input=user_input,
                response=response
            )

            request = LLMRequest(
                prompt=analysis_prompt,
                model=ModelType.GEMINI_FLASH,
                config=GenerationConfig(
                    temperature=self.config.critique_model_temperature,
                    max_tokens=1024
                )
            )

            # LLM 호출
            result = await self.llm_client.generate_text(request)

            if not result.is_success:
                return {"success": False, "error": result.error}

            # 응답 파싱
            analysis_text = result.data.text.strip()
            parsed_result = await self._parse_analysis_result(analysis_text, dimension)

            return {
                "success": True,
                "score": parsed_result["score"],
                "weaknesses": parsed_result["weaknesses"],
                "improvements": parsed_result["improvements"]
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _parse_analysis_result(self, analysis_text: str, dimension: str) -> Dict[str, Any]:
        """LLM 분석 결과 파싱"""
        try:
            # JSON 형태의 응답을 찾아서 파싱 시도
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)

                score = float(parsed.get("score", 50.0))
                score = max(0.0, min(100.0, score))  # 0-100 범위로 제한

                weaknesses = []
                improvements = []

                # 약점 파싱
                for w in parsed.get("weaknesses", []):
                    weakness = Weakness(
                        type=WeaknessType(w.get("type", "unclear_expression")),
                        description=w.get("description", ""),
                        location=w.get("location", "전체"),
                        severity=float(w.get("severity", 0.5)),
                        confidence=float(w.get("confidence", 0.7))
                    )
                    weaknesses.append(weakness)

                # 개선사항 파싱
                for i in parsed.get("improvements", []):
                    improvement = Improvement(
                        weakness_id=None,
                        type=ImprovementType(i.get("type", "clarification")),
                        description=i.get("description", ""),
                        suggestion=i.get("suggestion", ""),
                        priority=float(i.get("priority", 0.5)),
                        estimated_impact=float(i.get("estimated_impact", 0.5))
                    )
                    improvements.append(improvement)

                return {
                    "score": score,
                    "weaknesses": weaknesses,
                    "improvements": improvements
                }

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.warning(f"Failed to parse JSON analysis result: {str(e)}")

        # JSON 파싱 실패 시 텍스트 기반 파싱
        return await self._parse_text_analysis(analysis_text, dimension)

    async def _parse_text_analysis(self, analysis_text: str, dimension: str) -> Dict[str, Any]:
        """텍스트 기반 분석 결과 파싱"""
        # 점수 추출 (기본값: 75.0)
        score_match = re.search(r'점수[:：]?\s*(\d+(?:\.\d+)?)', analysis_text)
        score = 75.0
        if score_match:
            score = float(score_match.group(1))

        # 기본 반환값
        return {
            "score": max(0.0, min(100.0, score)),
            "weaknesses": [],
            "improvements": []
        }

    async def _generate_critique_reasoning(self, response: str, user_input: str, scores: Dict[str, float], weaknesses: List[Weakness]) -> str:
        """비평 추론 생성"""
        reasoning = f"응답 품질 분석 결과:\n"
        reasoning += f"- 논리적 일관성: {scores['logical_consistency']:.1f}/100\n"
        reasoning += f"- 사실적 정확성: {scores['factual_accuracy']:.1f}/100\n"
        reasoning += f"- 완전성: {scores['completeness']:.1f}/100\n"
        reasoning += f"- 관련성: {scores['relevance']:.1f}/100\n"
        reasoning += f"- 명확성: {scores['clarity']:.1f}/100\n\n"

        if weaknesses:
            reasoning += "주요 개선점:\n"
            for weakness in weaknesses[:3]:  # 상위 3개만
                reasoning += f"- {weakness.description}\n"

        return reasoning

    def _get_logical_consistency_prompt(self) -> str:
        """논리적 일관성 분석 프롬프트"""
        return """다음 응답의 논리적 일관성을 분석해주세요.

사용자 질문: {user_input}
응답: {response}

분석 기준:
1. 논리적 모순이 없는가?
2. 추론 과정이 일관성이 있는가?
3. 결론이 전제와 일치하는가?

JSON 형태로 응답해주세요:
{{
    "score": 0-100 점수,
    "weaknesses": [
        {{
            "type": "logical_inconsistency",
            "description": "발견된 문제점",
            "location": "문제 위치",
            "severity": 0.0-1.0,
            "confidence": 0.0-1.0
        }}
    ],
    "improvements": [
        {{
            "type": "correction",
            "description": "개선 설명",
            "suggestion": "구체적 개선안",
            "priority": 0.0-1.0,
            "estimated_impact": 0.0-1.0
        }}
    ]
}}"""

    def _get_factual_accuracy_prompt(self) -> str:
        """사실적 정확성 분석 프롬프트"""
        return """다음 응답의 사실적 정확성을 분석해주세요.

사용자 질문: {user_input}
응답: {response}

분석 기준:
1. 제시된 사실이 정확한가?
2. 출처나 근거가 신뢰할 만한가?
3. 최신 정보를 반영하고 있는가?

JSON 형태로 응답해주세요:
{{
    "score": 0-100 점수,
    "weaknesses": [
        {{
            "type": "factual_error",
            "description": "발견된 문제점",
            "location": "문제 위치",
            "severity": 0.0-1.0,
            "confidence": 0.0-1.0
        }}
    ],
    "improvements": [
        {{
            "type": "evidence_addition",
            "description": "개선 설명",
            "suggestion": "구체적 개선안",
            "priority": 0.0-1.0,
            "estimated_impact": 0.0-1.0
        }}
    ]
}}"""

    def _get_completeness_prompt(self) -> str:
        """완전성 분석 프롬프트"""
        return """다음 응답의 완전성을 분석해주세요.

사용자 질문: {user_input}
응답: {response}

분석 기준:
1. 질문의 모든 부분을 다뤘는가?
2. 필요한 정보가 빠진 것은 없는가?
3. 맥락을 충분히 고려했는가?

JSON 형태로 응답해주세요:
{{
    "score": 0-100 점수,
    "weaknesses": [
        {{
            "type": "incomplete_response",
            "description": "발견된 문제점",
            "location": "문제 위치",
            "severity": 0.0-1.0,
            "confidence": 0.0-1.0
        }}
    ],
    "improvements": [
        {{
            "type": "elaboration",
            "description": "개선 설명",
            "suggestion": "구체적 개선안",
            "priority": 0.0-1.0,
            "estimated_impact": 0.0-1.0
        }}
    ]
}}"""

    def _get_relevance_prompt(self) -> str:
        """관련성 분석 프롬프트"""
        return """다음 응답의 관련성을 분석해주세요.

사용자 질문: {user_input}
응답: {response}

분석 기준:
1. 응답이 질문과 직접적으로 관련되는가?
2. 불필요한 정보는 없는가?
3. 핵심 요점을 다루고 있는가?

JSON 형태로 응답해주세요:
{{
    "score": 0-100 점수,
    "weaknesses": [
        {{
            "type": "irrelevant_content",
            "description": "발견된 문제점",
            "location": "문제 위치",
            "severity": 0.0-1.0,
            "confidence": 0.0-1.0
        }}
    ],
    "improvements": [
        {{
            "type": "restructuring",
            "description": "개선 설명",
            "suggestion": "구체적 개선안",
            "priority": 0.0-1.0,
            "estimated_impact": 0.0-1.0
        }}
    ]
}}"""

    def _get_clarity_prompt(self) -> str:
        """명확성 분석 프롬프트"""
        return """다음 응답의 명확성을 분석해주세요.

사용자 질문: {user_input}
응답: {response}

분석 기준:
1. 표현이 명확하고 이해하기 쉬운가?
2. 모호한 표현은 없는가?
3. 구조가 잘 정리되어 있는가?

JSON 형태로 응답해주세요:
{{
    "score": 0-100 점수,
    "weaknesses": [
        {{
            "type": "unclear_expression",
            "description": "발견된 문제점",
            "location": "문제 위치",
            "severity": 0.0-1.0,
            "confidence": 0.0-1.0
        }}
    ],
    "improvements": [
        {{
            "type": "clarification",
            "description": "개선 설명",
            "suggestion": "구체적 개선안",
            "priority": 0.0-1.0,
            "estimated_impact": 0.0-1.0
        }}
    ]
}}"""

    async def cleanup(self):
        """리소스 정리"""
        self.logger.info("CritiqueAnalyzer cleaned up")