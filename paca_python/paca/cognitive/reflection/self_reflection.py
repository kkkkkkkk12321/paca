"""
Self Reflection Module
자기 성찰 시스템

이 모듈은 PACA의 자기 성찰 루프를 구현합니다:
- 다양한 성찰 타입 (논리적, 윤리적, 실용적)
- 성찰 세션 관리
- 자기 개선 피드백 루프
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
import statistics
import json

# 조건부 임포트
try:
    from ...core.types.base import (
        ID, Timestamp, Result, current_timestamp, generate_id, create_success, create_failure
    )
except ImportError:
    from paca.core.types.base import (
        ID, Timestamp, Result, current_timestamp, generate_id, create_success, create_failure
    )


class ReflectionType(Enum):
    """성찰 타입"""
    LOGICAL = "logical"           # 논리적 성찰
    ETHICAL = "ethical"           # 윤리적 성찰
    PRACTICAL = "practical"       # 실용적 성찰
    EMOTIONAL = "emotional"       # 감정적 성찰
    EPISTEMIC = "epistemic"       # 인식론적 성찰
    METACOGNITIVE = "metacognitive"  # 메타인지적 성찰


class ReflectionDepth(Enum):
    """성찰 깊이"""
    SURFACE = "surface"       # 표면적 검토
    MODERATE = "moderate"     # 중간 깊이
    DEEP = "deep"            # 깊은 성찰
    PROFOUND = "profound"     # 심오한 성찰


class ReflectionOutcome(Enum):
    """성찰 결과"""
    CONFIRMED = "confirmed"           # 기존 답변 확인
    REFINED = "refined"              # 답변 개선
    REVISED = "revised"              # 답변 수정
    REJECTED = "rejected"            # 답변 거부
    UNCERTAIN = "uncertain"          # 불확실성 증가


@dataclass(frozen=True)
class ReflectionPrompt:
    """성찰 프롬프트"""
    prompt_text: str
    focus_areas: List[str]
    expected_insights: List[str]
    reflection_type: ReflectionType


@dataclass(frozen=True)
class ReflectionResult:
    """성찰 결과"""
    reflection_id: ID
    original_content: str
    reflection_type: ReflectionType
    depth: ReflectionDepth
    insights: List[str]
    concerns: List[str]
    improvements: List[str]
    confidence_change: float  # -1.0 to 1.0
    outcome: ReflectionOutcome
    revised_content: Optional[str] = None
    timestamp: Timestamp = field(default_factory=current_timestamp)


@dataclass
class ReflectionCycle:
    """성찰 사이클"""
    cycle_id: ID
    original_question: str
    initial_answer: str
    reflection_results: List[ReflectionResult] = field(default_factory=list)
    final_answer: Optional[str] = None
    cycle_complete: bool = False
    total_iterations: int = 0
    start_timestamp: Timestamp = field(default_factory=current_timestamp)
    end_timestamp: Optional[Timestamp] = None


@dataclass
class ReflectionSession:
    """성찰 세션"""
    session_id: ID
    cycles: List[ReflectionCycle] = field(default_factory=list)
    session_metadata: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    start_timestamp: Timestamp = field(default_factory=current_timestamp)


class SelfReflection:
    """
    자기 성찰 시스템
    PACA의 응답에 대해 다각도로 성찰하고 개선하는 시스템
    """

    def __init__(self):
        self.reflection_sessions: Dict[ID, ReflectionSession] = {}
        self.reflection_templates = self._initialize_reflection_templates()
        self.active_cycles: Dict[ID, ReflectionCycle] = {}

        # 성찰 설정
        self.max_iterations_per_cycle = 3
        self.confidence_threshold = 0.8
        self.improvement_threshold = 0.1

    def _initialize_reflection_templates(self) -> Dict[ReflectionType, List[ReflectionPrompt]]:
        """성찰 템플릿 초기화"""
        return {
            ReflectionType.LOGICAL: [
                ReflectionPrompt(
                    prompt_text="이 답변의 논리적 구조와 추론 과정을 검토하세요",
                    focus_areas=["논리적 일관성", "전제와 결론의 연결", "추론 과정의 타당성"],
                    expected_insights=["논리적 허점", "추론 개선점", "논증 강화 방안"],
                    reflection_type=ReflectionType.LOGICAL
                ),
                ReflectionPrompt(
                    prompt_text="반대 논리나 대안적 해석이 있는지 검토하세요",
                    focus_areas=["반대 의견", "대안적 관점", "논리적 반박"],
                    expected_insights=["반박 가능성", "논리적 약점", "보완 필요사항"],
                    reflection_type=ReflectionType.LOGICAL
                )
            ],

            ReflectionType.ETHICAL: [
                ReflectionPrompt(
                    prompt_text="이 답변이 윤리적으로 적절한지 검토하세요",
                    focus_areas=["도덕적 함의", "가치 중립성", "잠재적 해로움"],
                    expected_insights=["윤리적 우려", "가치 편향", "도덕적 개선점"],
                    reflection_type=ReflectionType.ETHICAL
                ),
                ReflectionPrompt(
                    prompt_text="다양한 윤리적 관점에서 이 답변을 평가하세요",
                    focus_areas=["결과주의", "의무론", "덕윤리학"],
                    expected_insights=["윤리적 갈등", "도덕적 딜레마", "윤리적 균형"],
                    reflection_type=ReflectionType.ETHICAL
                )
            ],

            ReflectionType.PRACTICAL: [
                ReflectionPrompt(
                    prompt_text="이 답변이 실제로 유용하고 실행 가능한지 검토하세요",
                    focus_areas=["실용성", "실행 가능성", "현실적 제약"],
                    expected_insights=["실용적 한계", "실행 장벽", "개선 방안"],
                    reflection_type=ReflectionType.PRACTICAL
                ),
                ReflectionPrompt(
                    prompt_text="사용자의 실제 상황과 요구에 얼마나 부합하는지 검토하세요",
                    focus_areas=["사용자 맥락", "실제 적용성", "구체적 도움"],
                    expected_insights=["맥락 적합성", "실용적 가치", "사용자 중심성"],
                    reflection_type=ReflectionType.PRACTICAL
                )
            ],

            ReflectionType.EPISTEMIC: [
                ReflectionPrompt(
                    prompt_text="이 답변의 지식적 기반과 불확실성을 검토하세요",
                    focus_areas=["지식의 확실성", "증거의 강도", "인식론적 한계"],
                    expected_insights=["지식의 한계", "불확실성 영역", "추가 확인 필요사항"],
                    reflection_type=ReflectionType.EPISTEMIC
                ),
                ReflectionPrompt(
                    prompt_text="이 답변이 적절한 겸손성을 보이는지 검토하세요",
                    focus_areas=["인식론적 겸손", "확신의 적절성", "불확실성 표현"],
                    expected_insights=["과도한 확신", "겸손성 부족", "불확실성 인정"],
                    reflection_type=ReflectionType.EPISTEMIC
                )
            ]
        }

    async def start_reflection_session(self, session_metadata: Optional[Dict[str, Any]] = None) -> Result[ID]:
        """성찰 세션 시작"""
        try:
            session_id = generate_id("reflection_session_")

            session = ReflectionSession(
                session_id=session_id,
                session_metadata=session_metadata or {},
                active=True
            )

            self.reflection_sessions[session_id] = session
            return create_success(session_id)

        except Exception as e:
            return create_failure(e)

    async def begin_reflection_cycle(self, session_id: ID, question: str,
                                   initial_answer: str) -> Result[ID]:
        """성찰 사이클 시작"""
        try:
            if session_id not in self.reflection_sessions:
                return create_failure("유효하지 않은 세션 ID입니다")

            cycle_id = generate_id("reflection_cycle_")

            cycle = ReflectionCycle(
                cycle_id=cycle_id,
                original_question=question,
                initial_answer=initial_answer
            )

            self.reflection_sessions[session_id].cycles.append(cycle)
            self.active_cycles[cycle_id] = cycle

            return create_success(cycle_id)

        except Exception as e:
            return create_failure(e)

    async def perform_reflection(self, cycle_id: ID, reflection_types: List[ReflectionType],
                               depth: ReflectionDepth = ReflectionDepth.MODERATE) -> Result[List[ReflectionResult]]:
        """성찰 수행"""
        try:
            if cycle_id not in self.active_cycles:
                return create_failure("활성화된 사이클이 아닙니다")

            cycle = self.active_cycles[cycle_id]
            current_content = cycle.final_answer or cycle.initial_answer

            results = []

            # 각 성찰 타입별로 성찰 수행
            for reflection_type in reflection_types:
                reflection_result = await self._perform_single_reflection(
                    current_content, reflection_type, depth, cycle
                )
                if reflection_result.is_success:
                    results.append(reflection_result.value)

            # 결과를 사이클에 추가
            cycle.reflection_results.extend(results)
            cycle.total_iterations += 1

            return create_success(results)

        except Exception as e:
            return create_failure(e)

    async def _perform_single_reflection(self, content: str, reflection_type: ReflectionType,
                                       depth: ReflectionDepth, cycle: ReflectionCycle) -> Result[ReflectionResult]:
        """단일 성찰 수행"""
        try:
            # 성찰 프롬프트 선택
            templates = self.reflection_templates.get(reflection_type, [])
            if not templates:
                return create_failure(f"지원하지 않는 성찰 타입: {reflection_type}")

            # 깊이에 따른 프롬프트 선택
            template = self._select_template_by_depth(templates, depth)

            # 성찰 실행
            insights, concerns, improvements = await self._execute_reflection(
                content, template, cycle.original_question
            )

            # 신뢰도 변화 계산
            confidence_change = await self._calculate_confidence_change(
                content, insights, concerns, improvements
            )

            # 결과 판정
            outcome = await self._determine_outcome(
                confidence_change, len(improvements), len(concerns)
            )

            # 개선된 내용 생성
            revised_content = None
            if outcome in [ReflectionOutcome.REFINED, ReflectionOutcome.REVISED]:
                revised_content = await self._generate_improved_content(
                    content, improvements, cycle.original_question
                )

            result = ReflectionResult(
                reflection_id=generate_id("reflection_"),
                original_content=content,
                reflection_type=reflection_type,
                depth=depth,
                insights=insights,
                concerns=concerns,
                improvements=improvements,
                confidence_change=confidence_change,
                outcome=outcome,
                revised_content=revised_content
            )

            return create_success(result)

        except Exception as e:
            return create_failure(e)

    def _select_template_by_depth(self, templates: List[ReflectionPrompt],
                                 depth: ReflectionDepth) -> ReflectionPrompt:
        """깊이에 따른 템플릿 선택"""
        if depth == ReflectionDepth.SURFACE:
            return templates[0] if templates else templates[0]
        elif depth == ReflectionDepth.DEEP:
            return templates[-1] if len(templates) > 1 else templates[0]
        else:
            # 중간 깊이
            mid_index = len(templates) // 2
            return templates[mid_index] if templates else templates[0]

    async def _execute_reflection(self, content: str, template: ReflectionPrompt,
                                original_question: str) -> Tuple[List[str], List[str], List[str]]:
        """성찰 실행"""
        insights = []
        concerns = []
        improvements = []

        # 논리적 성찰
        if template.reflection_type == ReflectionType.LOGICAL:
            logical_insights = await self._analyze_logical_structure(content)
            insights.extend(logical_insights)

            logical_concerns = await self._identify_logical_issues(content)
            concerns.extend(logical_concerns)

            logical_improvements = await self._suggest_logical_improvements(content, logical_concerns)
            improvements.extend(logical_improvements)

        # 윤리적 성찰
        elif template.reflection_type == ReflectionType.ETHICAL:
            ethical_insights = await self._analyze_ethical_implications(content)
            insights.extend(ethical_insights)

            ethical_concerns = await self._identify_ethical_issues(content)
            concerns.extend(ethical_concerns)

            ethical_improvements = await self._suggest_ethical_improvements(content, ethical_concerns)
            improvements.extend(ethical_improvements)

        # 실용적 성찰
        elif template.reflection_type == ReflectionType.PRACTICAL:
            practical_insights = await self._analyze_practical_value(content, original_question)
            insights.extend(practical_insights)

            practical_concerns = await self._identify_practical_limitations(content)
            concerns.extend(practical_concerns)

            practical_improvements = await self._suggest_practical_improvements(content, practical_concerns)
            improvements.extend(practical_improvements)

        # 인식론적 성찰
        elif template.reflection_type == ReflectionType.EPISTEMIC:
            epistemic_insights = await self._analyze_knowledge_basis(content)
            insights.extend(epistemic_insights)

            epistemic_concerns = await self._identify_epistemic_issues(content)
            concerns.extend(epistemic_concerns)

            epistemic_improvements = await self._suggest_epistemic_improvements(content, epistemic_concerns)
            improvements.extend(epistemic_improvements)

        return insights, concerns, improvements

    async def _analyze_logical_structure(self, content: str) -> List[str]:
        """논리 구조 분석"""
        insights = []

        # 전제-결론 구조 분석
        if "because" in content.lower() or "since" in content.lower():
            insights.append("명시적 전제-결론 구조가 있음")

        if "therefore" in content.lower() or "thus" in content.lower():
            insights.append("논리적 결론 도출 시도가 있음")

        # 조건문 분석
        if "if" in content.lower() and "then" in content.lower():
            insights.append("조건적 논리 구조 사용")

        return insights

    async def _identify_logical_issues(self, content: str) -> List[str]:
        """논리적 문제 식별"""
        concerns = []

        # 모순 검사
        contradiction_pairs = [
            ("always", "never"), ("all", "none"), ("certain", "uncertain"),
            ("possible", "impossible"), ("true", "false")
        ]

        content_lower = content.lower()
        for pair in contradiction_pairs:
            if pair[0] in content_lower and pair[1] in content_lower:
                concerns.append(f"잠재적 모순: '{pair[0]}'과 '{pair[1]}' 동시 사용")

        # 과도한 일반화
        if "all" in content_lower or "every" in content_lower or "always" in content_lower:
            concerns.append("과도한 일반화 가능성")

        return concerns

    async def _suggest_logical_improvements(self, content: str, concerns: List[str]) -> List[str]:
        """논리적 개선 제안"""
        improvements = []

        for concern in concerns:
            if "모순" in concern:
                improvements.append("모순되는 표현을 제거하거나 조건을 명확히 하세요")
            elif "일반화" in concern:
                improvements.append("더 구체적이고 한정적인 표현을 사용하세요")

        # 논리 구조 강화
        if "because" not in content.lower() and "since" not in content.lower():
            improvements.append("근거나 이유를 더 명확히 제시하세요")

        return improvements

    async def _analyze_ethical_implications(self, content: str) -> List[str]:
        """윤리적 함의 분석"""
        insights = []

        # 가치 중립성 분석
        value_loaded_words = ["should", "must", "right", "wrong", "good", "bad"]
        if any(word in content.lower() for word in value_loaded_words):
            insights.append("가치 판단이 포함된 내용")

        # 이해관계자 고려
        stakeholder_words = ["people", "users", "society", "individuals", "groups"]
        if any(word in content.lower() for word in stakeholder_words):
            insights.append("이해관계자에 대한 고려가 있음")

        return insights

    async def _identify_ethical_issues(self, content: str) -> List[str]:
        """윤리적 문제 식별"""
        concerns = []

        # 편향성 검사
        bias_indicators = ["obviously", "clearly", "everyone knows", "common sense"]
        if any(indicator in content.lower() for indicator in bias_indicators):
            concerns.append("잠재적 편향이나 주관적 판단")

        # 해로움 가능성
        harm_indicators = ["risk", "danger", "harmful", "damage"]
        if any(indicator in content.lower() for indicator in harm_indicators):
            concerns.append("잠재적 해로움에 대한 충분한 경고 필요")

        return concerns

    async def _suggest_ethical_improvements(self, content: str, concerns: List[str]) -> List[str]:
        """윤리적 개선 제안"""
        improvements = []

        for concern in concerns:
            if "편향" in concern:
                improvements.append("더 객관적이고 다양한 관점을 제시하세요")
            elif "해로움" in concern:
                improvements.append("잠재적 위험에 대한 경고를 강화하세요")

        return improvements

    async def _analyze_practical_value(self, content: str, original_question: str) -> List[str]:
        """실용적 가치 분석"""
        insights = []

        # 구체성 분석
        if any(word in content.lower() for word in ["step", "how", "process", "method"]):
            insights.append("구체적인 방법이나 과정 제시")

        # 실행 가능성
        if any(word in content.lower() for word in ["can", "try", "implement", "apply"]):
            insights.append("실행 가능한 방안 제시")

        return insights

    async def _identify_practical_limitations(self, content: str) -> List[str]:
        """실용적 한계 식별"""
        concerns = []

        # 추상성 검사
        abstract_indicators = ["generally", "typically", "usually", "in theory"]
        if any(indicator in content.lower() for indicator in abstract_indicators):
            concerns.append("지나치게 추상적이거나 이론적")

        # 실행 장벽
        if not any(word in content.lower() for word in ["step", "how", "method", "way"]):
            concerns.append("구체적 실행 방법 부족")

        return concerns

    async def _suggest_practical_improvements(self, content: str, concerns: List[str]) -> List[str]:
        """실용적 개선 제안"""
        improvements = []

        for concern in concerns:
            if "추상적" in concern:
                improvements.append("더 구체적인 예시나 방법을 제시하세요")
            elif "실행 방법" in concern:
                improvements.append("단계별 실행 방법을 추가하세요")

        return improvements

    async def _analyze_knowledge_basis(self, content: str) -> List[str]:
        """지식 기반 분석"""
        insights = []

        # 불확실성 표현
        uncertainty_indicators = ["might", "possibly", "unclear", "uncertain", "may be"]
        if any(indicator in content.lower() for indicator in uncertainty_indicators):
            insights.append("적절한 불확실성 인정")

        # 출처 언급
        source_indicators = ["according to", "research shows", "studies indicate"]
        if any(indicator in content.lower() for indicator in source_indicators):
            insights.append("외부 출처나 연구 참조")

        return insights

    async def _identify_epistemic_issues(self, content: str) -> List[str]:
        """인식론적 문제 식별"""
        concerns = []

        # 과도한 확신
        overconfidence_indicators = ["definitely", "certainly", "absolutely", "guaranteed"]
        if any(indicator in content.lower() for indicator in overconfidence_indicators):
            concerns.append("과도한 확신 표현")

        # 근거 부족
        if not any(word in content.lower() for word in ["because", "evidence", "research", "study"]):
            concerns.append("주장에 대한 근거 부족")

        return concerns

    async def _suggest_epistemic_improvements(self, content: str, concerns: List[str]) -> List[str]:
        """인식론적 개선 제안"""
        improvements = []

        for concern in concerns:
            if "확신" in concern:
                improvements.append("더 겸손한 표현과 불확실성 인정이 필요합니다")
            elif "근거" in concern:
                improvements.append("주장을 뒷받침하는 근거나 출처를 제시하세요")

        return improvements

    async def _calculate_confidence_change(self, content: str, insights: List[str],
                                         concerns: List[str], improvements: List[str]) -> float:
        """신뢰도 변화 계산"""
        # 기본 신뢰도는 중립 (0.0)
        confidence_change = 0.0

        # 긍정적 요소 (통찰)
        confidence_change += len(insights) * 0.1

        # 부정적 요소 (우려사항)
        confidence_change -= len(concerns) * 0.15

        # 개선 가능성
        if improvements:
            confidence_change += 0.05  # 개선 가능성이 있다는 것은 긍정적

        # 범위 제한
        return max(-1.0, min(1.0, confidence_change))

    async def _determine_outcome(self, confidence_change: float,
                               improvement_count: int, concern_count: int) -> ReflectionOutcome:
        """결과 판정"""
        if concern_count == 0 and improvement_count == 0:
            return ReflectionOutcome.CONFIRMED
        elif concern_count > 0 and improvement_count > 3:
            return ReflectionOutcome.REVISED
        elif improvement_count > 0:
            return ReflectionOutcome.REFINED
        elif concern_count > 2:
            return ReflectionOutcome.REJECTED
        else:
            return ReflectionOutcome.UNCERTAIN

    async def _generate_improved_content(self, original_content: str,
                                       improvements: List[str], question: str) -> str:
        """개선된 내용 생성"""
        # 실제 구현에서는 더 정교한 텍스트 생성 알고리즘 사용
        improved_parts = []

        improved_parts.append(f"[원본 답변을 다음 관점에서 개선]\n")

        for i, improvement in enumerate(improvements, 1):
            improved_parts.append(f"{i}. {improvement}")

        improved_parts.append(f"\n[개선된 답변]\n{original_content}")
        improved_parts.append(f"\n[추가 고려사항: {', '.join(improvements)}]")

        return "\n".join(improved_parts)

    async def complete_reflection_cycle(self, cycle_id: ID) -> Result[str]:
        """성찰 사이클 완료"""
        try:
            if cycle_id not in self.active_cycles:
                return create_failure("활성화된 사이클이 아닙니다")

            cycle = self.active_cycles[cycle_id]

            # 최종 답변 결정
            final_answer = await self._determine_final_answer(cycle)
            cycle.final_answer = final_answer
            cycle.cycle_complete = True
            cycle.end_timestamp = current_timestamp()

            # 활성 사이클에서 제거
            del self.active_cycles[cycle_id]

            return create_success(final_answer)

        except Exception as e:
            return create_failure(e)

    async def _determine_final_answer(self, cycle: ReflectionCycle) -> str:
        """최종 답변 결정"""
        if not cycle.reflection_results:
            return cycle.initial_answer

        # 가장 최근의 개선된 답변 찾기
        revised_answers = [r.revised_content for r in cycle.reflection_results
                          if r.revised_content and r.outcome in [ReflectionOutcome.REFINED, ReflectionOutcome.REVISED]]

        if revised_answers:
            return revised_answers[-1]  # 가장 최근 개선된 답변

        # 개선된 답변이 없으면 원본 답변
        return cycle.initial_answer

    async def get_reflection_summary(self, session_id: ID) -> Result[Dict[str, Any]]:
        """성찰 요약 조회"""
        try:
            if session_id not in self.reflection_sessions:
                return create_failure("유효하지 않은 세션 ID입니다")

            session = self.reflection_sessions[session_id]

            total_cycles = len(session.cycles)
            completed_cycles = sum(1 for cycle in session.cycles if cycle.cycle_complete)

            # 성찰 타입별 통계
            reflection_type_counts = {}
            for cycle in session.cycles:
                for result in cycle.reflection_results:
                    rtype = result.reflection_type.value
                    reflection_type_counts[rtype] = reflection_type_counts.get(rtype, 0) + 1

            # 결과별 통계
            outcome_counts = {}
            for cycle in session.cycles:
                for result in cycle.reflection_results:
                    outcome = result.outcome.value
                    outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

            # 평균 신뢰도 변화
            all_confidence_changes = []
            for cycle in session.cycles:
                for result in cycle.reflection_results:
                    all_confidence_changes.append(result.confidence_change)

            avg_confidence_change = statistics.mean(all_confidence_changes) if all_confidence_changes else 0.0

            summary = {
                'session_id': session_id,
                'total_cycles': total_cycles,
                'completed_cycles': completed_cycles,
                'completion_rate': completed_cycles / total_cycles if total_cycles > 0 else 0,
                'reflection_type_distribution': reflection_type_counts,
                'outcome_distribution': outcome_counts,
                'average_confidence_change': avg_confidence_change,
                'total_reflections': sum(len(cycle.reflection_results) for cycle in session.cycles),
                'session_active': session.active,
                'session_duration': current_timestamp() - session.start_timestamp
            }

            return create_success(summary)

        except Exception as e:
            return create_failure(e)

    async def close_reflection_session(self, session_id: ID) -> Result[Dict[str, Any]]:
        """성찰 세션 종료"""
        try:
            if session_id not in self.reflection_sessions:
                return create_failure("유효하지 않은 세션 ID입니다")

            session = self.reflection_sessions[session_id]
            session.active = False

            # 미완료 사이클들 강제 완료
            for cycle in session.cycles:
                if not cycle.cycle_complete and cycle.cycle_id in self.active_cycles:
                    await self.complete_reflection_cycle(cycle.cycle_id)

            # 세션 요약 생성
            summary_result = await self.get_reflection_summary(session_id)

            return summary_result

        except Exception as e:
            return create_failure(e)

    def get_global_reflection_statistics(self) -> Dict[str, Any]:
        """전체 성찰 통계"""
        total_sessions = len(self.reflection_sessions)
        active_sessions = sum(1 for session in self.reflection_sessions.values() if session.active)

        total_cycles = sum(len(session.cycles) for session in self.reflection_sessions.values())
        total_reflections = sum(
            sum(len(cycle.reflection_results) for cycle in session.cycles)
            for session in self.reflection_sessions.values()
        )

        # 가장 일반적인 성찰 타입
        all_reflection_types = []
        for session in self.reflection_sessions.values():
            for cycle in session.cycles:
                for result in cycle.reflection_results:
                    all_reflection_types.append(result.reflection_type.value)

        most_common_type = None
        if all_reflection_types:
            type_counts = {}
            for rtype in all_reflection_types:
                type_counts[rtype] = type_counts.get(rtype, 0) + 1
            most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]

        return {
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'total_cycles': total_cycles,
            'total_reflections': total_reflections,
            'most_common_reflection_type': most_common_type,
            'average_reflections_per_cycle': total_reflections / total_cycles if total_cycles > 0 else 0,
            'current_active_cycles': len(self.active_cycles)
        }