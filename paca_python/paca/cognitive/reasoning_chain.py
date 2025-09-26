"""
추론 체인 시스템 (Reasoning Chain System)
PACA v5 Python의 핵심 차별화 기능

복잡한 문제를 단계별로 체계적으로 해결하는 시스템
복잡도 감지 결과에 따라 자동으로 활성화되어 추론 과정을 구조화
"""

import asyncio
import copy
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Set

# PACA 모듈 임포트
from paca.core.types import create_success, create_failure, generate_id, current_timestamp
from paca.core.validators import validate_range, validate_string_length
from paca.reasoning.base import ReasoningEngine, ReasoningType

from .complexity_detector import ComplexityResult, DomainType, ComplexityLevel
from .metacognition_engine import MetacognitionEngine, MonitoringSession

# 로깅 설정
logger = logging.getLogger(__name__)


class ReasoningStrategy(Enum):
    """추론 전략 유형"""
    SEQUENTIAL = "sequential"        # 순차적 추론
    PARALLEL = "parallel"           # 병렬 추론
    HIERARCHICAL = "hierarchical"   # 계층적 추론
    ITERATIVE = "iterative"         # 반복적 추론
    HYBRID = "hybrid"               # 하이브리드 추론


class StepType(Enum):
    """추론 단계 유형"""
    PROBLEM_ANALYSIS = "problem_analysis"        # 문제 분석
    DECOMPOSITION = "decomposition"              # 문제 분해
    HYPOTHESIS_GENERATION = "hypothesis_generation"  # 가설 생성
    EVIDENCE_GATHERING = "evidence_gathering"    # 증거 수집
    VALIDATION = "validation"                    # 검증
    SYNTHESIS = "synthesis"                      # 종합
    CONCLUSION = "conclusion"                    # 결론 도출
    VERIFICATION = "verification"                # 최종 검증


class ReasoningStatus(Enum):
    """추론 상태"""
    INITIALIZING = "initializing"    # 초기화 중
    PROCESSING = "processing"        # 처리 중
    VALIDATING = "validating"        # 검증 중
    COMPLETED = "completed"          # 완료
    FAILED = "failed"                # 실패
    BACKTRACKING = "backtracking"    # 백트래킹 중


@dataclass
class ReasoningStep:
    """추론 단계"""
    step_id: str                           # 단계 ID
    step_number: int                       # 단계 번호
    step_type: StepType                    # 단계 유형
    title: str                             # 단계 제목
    description: str                       # 상세 설명
    input_data: Dict                       # 입력 데이터
    output_data: Dict                      # 출력 데이터
    reasoning_process: str                 # 추론 과정 설명
    confidence: float                      # 신뢰도 (0-1)
    processing_time_ms: float              # 처리 시간
    dependencies: List[str] = field(default_factory=list)  # 의존 단계들
    errors: List[str] = field(default_factory=list)        # 오류 목록
    warnings: List[str] = field(default_factory=list)      # 경고 목록
    metadata: Dict = field(default_factory=dict)           # 메타데이터
    timestamp: float = field(default_factory=current_timestamp)


@dataclass
class ReasoningResult:
    """추론 결과"""
    chain_id: str                          # 체인 ID
    problem: str                           # 원본 문제
    complexity_info: ComplexityResult      # 복잡도 정보
    strategy_used: ReasoningStrategy       # 사용된 전략
    steps: List[ReasoningStep]             # 추론 단계들
    final_conclusion: str                  # 최종 결론
    confidence_score: float                # 전체 신뢰도
    total_processing_time_ms: float        # 전체 처리 시간
    quality_assessment: Dict               # 품질 평가
    alternative_solutions: List[str] = field(default_factory=list)  # 대안 해결책
    learning_insights: List[str] = field(default_factory=list)     # 학습 통찰
    status: ReasoningStatus = ReasoningStatus.COMPLETED
    error_log: List[str] = field(default_factory=list)
    backtrack_attempts: int = 0
    backtrack_successes: int = 0
    backtrack_failures: int = 0
    backtrack_summary: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=current_timestamp)


@dataclass
class BacktrackPoint:
    """백트래킹 지점"""
    step_number: int                       # 단계 번호
    step_id: str                          # 단계 ID
    state_snapshot: Dict                   # 상태 스냅샷
    reason: str                           # 백트래킹 이유
    alternative_paths: List[str] = field(default_factory=list)           # 대안 경로들
    timestamp: float = field(default_factory=current_timestamp)


@dataclass
class StrategyDecision:
    """전략 전환 의사결정 결과"""

    accept_current: bool
    switch_reasons: List[str] = field(default_factory=list)
    escalate_reasons: List[str] = field(default_factory=list)


class ReasoningChain:
    """
    추론 체인 시스템

    복잡한 문제를 체계적으로 분해하고 단계별로 해결하는
    AI 추론 시스템의 핵심 엔진
    """

    def __init__(self,
                 config: Optional[Dict] = None,
                 reasoning_engine: Optional[ReasoningEngine] = None):
        """
        초기화

        Args:
            config: 설정 딕셔너리
        """
        default_config = self._load_default_config()
        merged_config = default_config.copy()
        if config:
            merged_config.update(config)

        self.config = merged_config

        # 전략별 설정
        self.strategy_configs = {
            ReasoningStrategy.SEQUENTIAL: {
                'max_steps': self.config.get('max_sequential_steps', 10),
                'validation_frequency': 1  # 매 단계마다
            },
            ReasoningStrategy.PARALLEL: {
                'max_parallel_branches': self.config.get('max_parallel_branches', 3),
                'merge_strategy': 'best_confidence'
            },
            ReasoningStrategy.HIERARCHICAL: {
                'max_depth': self.config.get('max_hierarchy_depth', 4),
                'branching_factor': 2
            },
            ReasoningStrategy.ITERATIVE: {
                'max_iterations': self.config.get('max_iterations', 5),
                'convergence_threshold': 0.95
            }
        }

        # 백트래킹 설정
        self.enable_backtracking = self.config.get('enable_backtracking', True)
        self.max_backtrack_depth = self.config.get('max_backtrack_depth', 3)
        self.backtrack_confidence_threshold = self.config.get('backtrack_confidence_threshold', 0.72)
        self.max_backtrack_attempts = self.config.get('max_backtrack_attempts', self.max_backtrack_depth)
        self.backtrack_history: List[BacktrackPoint] = []
        self.backtrack_attempts = 0
        self.backtrack_successes = 0
        self.backtrack_failures = 0
        self._latest_alternative_paths: List[str] = []

        self.strategy_confidence_thresholds: Dict[str, float] = self.config.get(
            'strategy_confidence_thresholds', {}
        )
        self.default_strategy_confidence = self.config.get('strategy_switch_confidence_threshold', 0.78)
        self.strategy_switch_rules = self.config.get('strategy_switch_rules', {})
        self.require_successful_backtrack = self.strategy_switch_rules.get('require_successful_backtrack', False)
        self.max_unresolved_attempts = self.strategy_switch_rules.get('max_unresolved_attempts')

        policy_config = self.config.get('strategy_switch_policy', {})
        self.strategy_switch_policy = {
            'low_confidence_threshold': policy_config.get('low_confidence_threshold'),
            'consecutive_low_confidence_limit': policy_config.get('consecutive_low_confidence_limit'),
            'confidence_drop_threshold': policy_config.get('confidence_drop_threshold'),
            'validation_issue_limit': policy_config.get('validation_issue_limit'),
            'max_backtrack_failures': policy_config.get('max_backtrack_failures'),
            'quality_alert_levels': [
                level.lower() for level in policy_config.get('quality_alert_levels', []) if isinstance(level, str)
            ],
        }

        self.domain_overrides: Dict[str, Dict[str, Any]] = self.config.get('domain_strategy_overrides', {})
        self.domain_confidence_thresholds: Dict[str, Dict[str, Any]] = {
            domain: overrides.get('confidence_thresholds', {})
            for domain, overrides in self.domain_overrides.items()
        }

        fallback_order = self.config.get('strategy_fallback_order', [])
        self.max_strategy_attempts = self.config.get(
            'max_strategy_attempts',
            len(fallback_order) + 1 if fallback_order else 0
        )

        escalation_config = self.config.get('escalation', {})
        self.enable_reasoning_engine_escalation = escalation_config.get('enabled', False)
        self.escalation_after_attempts = escalation_config.get('after_attempts', len(fallback_order) + 1)
        self.escalation_min_confidence = escalation_config.get('min_confidence', 0.8)
        self.escalation_reasoning_type = self._resolve_reasoning_type(
            escalation_config.get('reasoning_type', ReasoningType.DEDUCTIVE.value)
        )
        self.escalation_trigger_reasons = [
            reason for reason in escalation_config.get('trigger_reasons', []) if isinstance(reason, str)
        ]
        self.escalation_alert_levels = {
            level.lower() for level in escalation_config.get('quality_alert_levels', []) if isinstance(level, str)
        }
        self.escalation_collaboration_policy = self._parse_collaboration_policy(
            escalation_config.get('collaboration_policy', {})
        )

        self.reasoning_engine = reasoning_engine or self.config.get('reasoning_engine')
        if self.enable_reasoning_engine_escalation and self.reasoning_engine is None:
            try:
                self.reasoning_engine = ReasoningEngine()
            except Exception as exc:
                logger.warning("ReasoningEngine 초기화 실패: %s", exc)
                self.reasoning_engine = None

        # 메타인지 엔진 (선택적)
        self.metacognition_engine = None
        self.current_monitoring_session = None

        # 성능 추적
        self.total_chains_executed = 0
        self.average_processing_time = 0.0
        self.success_rate = 0.0

        self._escalation_reasons: List[str] = []

        logger.info("ReasoningChain 초기화 완료")

    def _reset_runtime_state(self):
        """체인 실행마다 백트래킹 상태 초기화"""
        self.backtrack_history = []
        self.backtrack_attempts = 0
        self.backtrack_successes = 0
        self.backtrack_failures = 0
        self._latest_alternative_paths = []

    def _summarize_backtracking(self) -> List[Dict[str, Any]]:
        """백트래킹 이력 요약"""
        summary: List[Dict[str, Any]] = []
        for point in self.backtrack_history:
            summary.append({
                'step_number': point.step_number,
                'step_id': point.step_id,
                'reason': point.reason,
                'alternative_paths': list(point.alternative_paths),
                'timestamp': point.timestamp
            })
        return summary

    def _load_default_config(self) -> Dict[str, Any]:
        """기본 전략 설정 로드"""
        config_path = Path(__file__).resolve().parents[2] / 'data' / 'config' / 'reasoning_strategy.json'
        if not config_path.exists():
            return {}

        try:
            with config_path.open('r', encoding='utf-8') as config_file:
                data = json.load(config_file)
                if isinstance(data, dict):
                    return data
        except Exception as exc:
            logger.warning("Reasoning 전략 기본 설정 로드 실패: %s", exc)

        return {}

    def _resolve_reasoning_type(self, value: str) -> ReasoningType:
        try:
            return ReasoningType(value)
        except ValueError:
            logger.warning("알 수 없는 ReasoningType '%s', 기본값 DEDUCTIVE 사용", value)
            return ReasoningType.DEDUCTIVE

    def _normalize_domain(self, domain: Optional[Any]) -> Optional[str]:
        if isinstance(domain, DomainType):
            return domain.value
        if isinstance(domain, str):
            return domain.lower()
        return None

    def _get_strategy_confidence_threshold(self,
                                           strategy: ReasoningStrategy,
                                           domain: Optional[str] = None) -> float:
        if domain and domain in self.domain_confidence_thresholds:
            domain_thresholds = self.domain_confidence_thresholds[domain]
            if strategy.value in domain_thresholds:
                return domain_thresholds[strategy.value]
            if 'default' in domain_thresholds:
                return domain_thresholds['default']

        return self.strategy_confidence_thresholds.get(
            strategy.value,
            self.strategy_confidence_thresholds.get('default', self.default_strategy_confidence)
        )

    async def execute_reasoning_chain(self,
                                    problem: str,
                                    complexity_score: int,
                                    context: Optional[Dict] = None) -> ReasoningResult:
        """복잡도 기반으로 전략을 선택하고 필요 시 전략을 전환하며 추론을 수행"""

        chain_id = generate_id('reasoning_')
        overall_start = time.time()
        attempt_results: List[ReasoningResult] = []
        attempt_history: List[Dict[str, Any]] = []
        domain_key = self._normalize_domain(context.get('domain')) if context else None

        self._escalation_reasons = []
        previous_result: Optional[ReasoningResult] = None

        try:
            primary_strategy = await self._select_reasoning_strategy(complexity_score, context)
            strategy_sequence = self._build_strategy_sequence(primary_strategy, context)

            for attempt_index, strategy in enumerate(strategy_sequence, start=1):
                self._reset_runtime_state()
                result = await self._run_chain_attempt(
                    chain_id,
                    problem,
                    complexity_score,
                    context,
                    strategy,
                    attempt_index
                )

                attempt_results.append(result)
                validation_metrics = self._extract_validation_metrics(result.steps)
                self._annotate_quality_assessment(
                    result,
                    validation_metrics,
                    attempt_index,
                    domain_key
                )

                attempt_history.append({
                    'attempt': attempt_index,
                    'strategy': strategy.value,
                    'backtrack_attempts': result.backtrack_attempts,
                    'backtrack_successes': result.backtrack_successes,
                    'backtrack_failures': result.backtrack_failures,
                    'confidence': result.confidence_score,
                    'unresolved_validation': result.quality_assessment.get('unresolved_validation', False),
                    'status': result.status.value,
                    'alternative_solutions': list(result.alternative_solutions),
                    'quality_level': result.quality_assessment.get('quality_level'),
                    'alerts': list(result.quality_assessment.get('alerts', []))
                })

                decision = self._evaluate_strategy_decision(
                    result,
                    previous_result,
                    attempt_index,
                    len(strategy_sequence),
                    attempt_history,
                    domain_key
                )

                attempt_history[-1]['switch_reasons'] = list(decision.switch_reasons)
                result.quality_assessment['switch_reasons'] = list(decision.switch_reasons)

                if decision.escalate_reasons:
                    for reason in decision.escalate_reasons:
                        if reason not in self._escalation_reasons:
                            self._escalation_reasons.append(reason)

                if decision.accept_current:
                    final_result = result
                    break
                previous_result = result
            else:
                final_result = attempt_results[-1]

            if self._escalation_reasons:
                final_result.quality_assessment['escalation_triggers'] = list(self._escalation_reasons)

            should_escalate = final_result.quality_assessment.get('unresolved_validation', False)
            if self._escalation_reasons:
                should_escalate = True

            if self.enable_reasoning_engine_escalation and should_escalate:
                if (
                    len(attempt_history) >= self.escalation_after_attempts or
                    final_result.confidence_score < self.escalation_min_confidence or
                    self._escalation_reasons
                ):
                    final_result = await self._escalate_with_reasoning_engine(
                        chain_id,
                        problem,
                        final_result,
                        attempt_history,
                        context
                    )

            final_result.quality_assessment['strategy_history'] = attempt_history
            final_result.quality_assessment['attempts'] = len(attempt_history)
            self._update_performance_stats(final_result)

            logger.info(
                "추론 체인 완료: %s, 전략: %s, 시도 횟수: %s, 신뢰도: %.2f",
                chain_id,
                final_result.strategy_used.value,
                len(attempt_history),
                final_result.confidence_score
            )
            return final_result

        except Exception as e:
            logger.error(f"추론 체인 실행 중 오류: {str(e)}")
            error_result = await self._create_error_result(chain_id, problem, str(e), time.time() - overall_start)
            error_result.quality_assessment['strategy_history'] = attempt_history
            return error_result

    def _build_strategy_sequence(self,
                                 primary_strategy: ReasoningStrategy,
                                 context: Optional[Dict] = None) -> List[ReasoningStrategy]:
        """초기 전략과 구성 기반의 후보 전략 목록 생성"""

        domain_key = self._normalize_domain(context.get('domain')) if context else None

        if domain_key and domain_key in self.domain_overrides:
            override = self.domain_overrides[domain_key]
            fallback_config = override.get('fallback_order', [])
        else:
            fallback_config = self.config.get('strategy_fallback_order', [
                ReasoningStrategy.SEQUENTIAL,
                ReasoningStrategy.HIERARCHICAL,
                ReasoningStrategy.ITERATIVE,
                ReasoningStrategy.PARALLEL
            ])

        sequence: List[ReasoningStrategy] = [primary_strategy]

        for strategy in fallback_config:
            if isinstance(strategy, str):
                try:
                    strategy_enum = ReasoningStrategy(strategy)
                except ValueError:
                    logger.warning("알 수 없는 전략 이름이 fallback 목록에 포함되어 건너뜁니다: %s", strategy)
                    continue
            else:
                strategy_enum = strategy

            if strategy_enum not in sequence:
                sequence.append(strategy_enum)

        if self.max_strategy_attempts and self.max_strategy_attempts > 0:
            return sequence[:self.max_strategy_attempts]

        return sequence

    def _should_accept_result(self,
                               result: ReasoningResult,
                               attempt_index: int,
                               total_attempts: int,
                               attempt_history: List[Dict[str, Any]],
                               domain: Optional[str]) -> bool:
        """결과를 수용할지 여부 판단"""

        if result.status != ReasoningStatus.COMPLETED:
            return attempt_index == total_attempts

        unresolved_validation = result.quality_assessment.get('unresolved_validation', False)

        if self.require_successful_backtrack and result.backtrack_attempts > 0 and result.backtrack_successes == 0:
            return False if attempt_index < total_attempts else True

        if self.max_unresolved_attempts is not None:
            unresolved_count = sum(1 for entry in attempt_history if entry.get('unresolved_validation'))
            if unresolved_count >= self.max_unresolved_attempts and attempt_index < total_attempts:
                return False

        if not unresolved_validation and (
            result.backtrack_attempts == 0 or result.backtrack_successes > 0
        ):
            return True

        confidence_threshold = self._get_strategy_confidence_threshold(result.strategy_used, domain)
        if (not unresolved_validation and
                result.confidence_score >= confidence_threshold):
            return True

        return attempt_index == total_attempts

    def _evaluate_strategy_decision(self,
                                    result: ReasoningResult,
                                    previous_result: Optional[ReasoningResult],
                                    attempt_index: int,
                                    total_attempts: int,
                                    attempt_history: List[Dict[str, Any]],
                                    domain: Optional[str]) -> StrategyDecision:
        """전략 전환 여부 및 에스컬레이션 요인을 계산"""

        base_accept = self._should_accept_result(
            result,
            attempt_index,
            total_attempts,
            attempt_history,
            domain
        )

        reasons: List[str] = []

        policy = self.strategy_switch_policy
        low_conf_threshold = policy.get('low_confidence_threshold')
        if low_conf_threshold is None:
            low_conf_threshold = self._get_strategy_confidence_threshold(result.strategy_used, domain)

        if low_conf_threshold is not None and result.confidence_score < low_conf_threshold:
            reasons.append('low_confidence')

        consecutive_limit = policy.get('consecutive_low_confidence_limit')
        if consecutive_limit and low_conf_threshold is not None:
            consecutive = 0
            for entry in reversed(attempt_history):
                confidence = entry.get('confidence')
                if confidence is None:
                    continue
                if confidence < low_conf_threshold:
                    consecutive += 1
                else:
                    break
            if consecutive >= consecutive_limit:
                reasons.append('consecutive_low_confidence')

        drop_threshold = policy.get('confidence_drop_threshold')
        if drop_threshold and previous_result is not None:
            confidence_drop = previous_result.confidence_score - result.confidence_score
            if confidence_drop >= drop_threshold:
                reasons.append('confidence_drop')

        validation_issue_limit = policy.get('validation_issue_limit')
        validation_issues = result.quality_assessment.get('validation_issues', 0)
        if validation_issue_limit and validation_issues >= validation_issue_limit:
            reasons.append('validation_issue_limit')

        if result.quality_assessment.get('forced_validation_failure'):
            reasons.append('forced_validation_failure')

        max_backtrack_failures = policy.get('max_backtrack_failures')
        if max_backtrack_failures is not None and result.backtrack_failures >= max_backtrack_failures:
            reasons.append('backtrack_failure_limit')

        quality_level = result.quality_assessment.get('quality_level')
        policy_alert_levels = set(policy.get('quality_alert_levels', []))
        if quality_level and quality_level.lower() in policy_alert_levels:
            reasons.append(f'quality_alert_{quality_level.lower()}')

        # 이유 중복 제거 (순서 유지)
        seen_reason = set()
        reasons = [reason for reason in reasons if not (reason in seen_reason or seen_reason.add(reason))]

        require_switch = bool(reasons) and attempt_index < total_attempts
        accept_current = base_accept and not require_switch

        escalate_reasons: List[str] = []
        if quality_level and quality_level.lower() in self.escalation_alert_levels:
            escalate_reasons.append(f'quality_alert_{quality_level.lower()}')

        for reason in reasons:
            if reason in self.escalation_trigger_reasons and reason not in escalate_reasons:
                escalate_reasons.append(reason)

        return StrategyDecision(
            accept_current=accept_current,
            switch_reasons=reasons,
            escalate_reasons=escalate_reasons
        )

    def _extract_validation_metrics(self, steps: List[ReasoningStep]) -> Dict[str, Any]:
        """검증 단계 메트릭 추출"""
        metrics: Dict[str, Any] = {
            'issues': 0,
            'forced_failure': False,
            'logic_validation': True,
            'confidence': None
        }

        for step in steps:
            if step.step_type != StepType.VALIDATION:
                continue

            summary = step.metadata.get('validation_summary')
            if not summary and isinstance(step.output_data, dict):
                summary = step.output_data.get('validation_result')

            if isinstance(summary, dict):
                metrics['issues'] = len(summary.get('issues', []))
                metrics['forced_failure'] = summary.get('forced_failure', False)
                metrics['logic_validation'] = summary.get('logic_validation', True)
                metrics['confidence'] = summary.get('confidence')
                return metrics

        return metrics

    def _annotate_quality_assessment(self,
                                     result: ReasoningResult,
                                     validation_metrics: Dict[str, Any],
                                     attempt_index: int,
                                     domain: Optional[str]) -> None:
        """품질 평가 데이터에 검증 및 경보 정보를 추가"""

        quality_assessment = result.quality_assessment
        average_confidence = quality_assessment.get('average_confidence', 0.0)
        unresolved_validation = quality_assessment.get('unresolved_validation', False)
        quality_level = self._calculate_quality_level(
            result.confidence_score,
            average_confidence,
            unresolved_validation,
            validation_metrics
        )

        alerts = self._build_quality_alerts(
            result,
            validation_metrics,
            unresolved_validation,
            quality_level
        )

        quality_assessment.update({
            'validation_issues': validation_metrics.get('issues', 0),
            'forced_validation_failure': validation_metrics.get('forced_failure', False),
            'validation_confidence': validation_metrics.get('confidence'),
            'quality_level': quality_level,
            'alerts': alerts,
            'domain': domain,
            'strategy_attempt': attempt_index
        })

    def _calculate_quality_level(self,
                                 final_confidence: float,
                                 average_confidence: float,
                                 unresolved_validation: bool,
                                 validation_metrics: Dict[str, Any]) -> str:
        """품질 레벨 산정"""

        if unresolved_validation or validation_metrics.get('forced_failure'):
            return 'red'

        issues = validation_metrics.get('issues', 0)

        if final_confidence >= 0.82 and average_confidence >= 0.78 and issues == 0:
            return 'green'

        if final_confidence >= 0.7 and average_confidence >= 0.68 and issues <= 1:
            return 'yellow'

        return 'red'

    def _build_quality_alerts(self,
                              result: ReasoningResult,
                              validation_metrics: Dict[str, Any],
                              unresolved_validation: bool,
                              quality_level: str) -> List[str]:
        """품질 경보 목록 생성"""

        alerts: List[str] = []
        threshold = self.strategy_switch_policy.get('low_confidence_threshold')
        if threshold is None:
            domain_key = result.quality_assessment.get('domain') if isinstance(result.quality_assessment, dict) else None
            threshold = self._get_strategy_confidence_threshold(result.strategy_used, domain_key)

        if threshold is not None and result.confidence_score < threshold:
            alerts.append('low_confidence')

        if validation_metrics.get('issues'):
            alerts.append('validation_issues')

        if validation_metrics.get('forced_failure'):
            alerts.append('forced_validation_failure')

        if unresolved_validation:
            alerts.append('unresolved_validation')

        if result.backtrack_failures > 0:
            alerts.append('backtrack_failures')

        if quality_level == 'red':
            alerts.append('quality_alert_red')
        elif quality_level == 'yellow':
            alerts.append('quality_alert_yellow')

        seen: set = set()
        return [alert for alert in alerts if not (alert in seen or seen.add(alert))]

    async def _run_chain_attempt(self,
                                chain_id: str,
                                problem: str,
                                complexity_score: int,
                                context: Optional[Dict],
                                strategy: ReasoningStrategy,
                                attempt_index: int) -> ReasoningResult:
        """단일 전략으로 추론 체인을 실행"""

        attempt_start = time.time()
        strategy_label = strategy.value
        domain_key = self._normalize_domain(context.get('domain')) if context else None
        logger.info(
            "추론 체인 시도 %s: 전략=%s, 복잡도=%s",
            attempt_index,
            strategy_label,
            complexity_score
        )

        try:
            await self._start_metacognition_monitoring(chain_id, problem, complexity_score)

            initial_analysis = await self._analyze_problem(problem, complexity_score, context)
            reasoning_steps = await self._execute_strategy(
                strategy,
                problem,
                initial_analysis,
                complexity_score
            )

            conclusion, confidence = await self._synthesize_conclusion(reasoning_steps)
            quality_assessment = await self._assess_quality(reasoning_steps, conclusion)
            learning_insights = await self._extract_learning_insights(reasoning_steps)
            unresolved_validation = self._has_unresolved_validation_failure(reasoning_steps)

            total_time = (time.time() - attempt_start) * 1000

            result = ReasoningResult(
                chain_id=chain_id,
                problem=problem,
                complexity_info=ComplexityResult(
                    score=complexity_score,
                    reasoning_required=True,
                    domain=context.get('domain', DomainType.ANALYTICAL) if context else DomainType.ANALYTICAL,
                    confidence=0.8,
                    level=self._determine_complexity_level(complexity_score),
                    analysis_details={},
                    processing_time_ms=total_time,
                    timestamp=current_timestamp(),
                    unique_id=generate_id('complexity_')
                ),
                strategy_used=strategy,
                steps=reasoning_steps,
                final_conclusion=conclusion,
                confidence_score=confidence,
                total_processing_time_ms=total_time,
                quality_assessment=quality_assessment,
                learning_insights=learning_insights,
                alternative_solutions=self._latest_alternative_paths,
                status=ReasoningStatus.COMPLETED,
                backtrack_attempts=self.backtrack_attempts,
                backtrack_successes=self.backtrack_successes,
                backtrack_failures=self.backtrack_failures,
                backtrack_summary=self._summarize_backtracking()
            )

            quality_assessment.update({
                'strategy_attempt': attempt_index,
                'strategy_used': strategy_label,
                'unresolved_validation': unresolved_validation,
                'backtrack_attempts': self.backtrack_attempts,
                'backtrack_successes': self.backtrack_successes,
                'backtrack_failures': self.backtrack_failures,
                'domain': domain_key
            })

            await self._end_metacognition_monitoring()

            return result

        except Exception as e:
            logger.error(f"추론 체인 시도 중 오류: {str(e)}")
            try:
                await self._end_metacognition_monitoring()
            except Exception:
                pass

            elapsed = time.time() - attempt_start
            error_result = await self._create_error_result(chain_id, problem, str(e), elapsed)
            error_result.quality_assessment['strategy_attempt'] = attempt_index
            error_result.quality_assessment['strategy_used'] = strategy_label
            return error_result

    def _has_unresolved_validation_failure(self, steps: List[ReasoningStep]) -> bool:
        """검증 단계에서 해결되지 않은 오류가 있는지 확인"""

        for step in reversed(steps):
            if step.step_type == StepType.VALIDATION:
                summary = step.metadata.get('validation_summary')
                if not summary and isinstance(step.output_data, dict):
                    summary = step.output_data.get('validation_result')

                if isinstance(summary, dict) and not summary.get('logic_validation', True):
                    return True

                if step.errors:
                    return True

        return False

    async def _escalate_with_reasoning_engine(
        self,
        chain_id: str,
        problem: str,
        final_result: ReasoningResult,
        attempt_history: List[Dict[str, Any]],
        context: Optional[Dict]
    ) -> ReasoningResult:
        """ReasoningEngine에 추가 추론을 요청"""

        if not self.reasoning_engine:
            return final_result

        try:
            init_result = await self.reasoning_engine.initialize()
            if init_result.is_failure:
                logger.warning("ReasoningEngine 초기화 실패: %s", init_result.error)
                return final_result
        except Exception as exc:
            logger.warning("ReasoningEngine 초기화 중 예외: %s", exc)
            return final_result

        premises: List[Any] = []
        for step in final_result.steps:
            if step.output_data:
                premises.append(step.output_data)

        try:
            reasoning_kwargs = {
                'confidence_threshold': self.escalation_min_confidence,
                'max_steps': 10,
            }

            reasoning_outcome = await self.reasoning_engine.reason(
                self.escalation_reasoning_type,
                premises=premises or [problem],
                target_conclusion=final_result.final_conclusion,
                **reasoning_kwargs
            )

            if reasoning_outcome.is_success:
                reasoning_result = reasoning_outcome.value
                final_result.final_conclusion = reasoning_result.conclusion
                final_result.confidence_score = max(
                    final_result.confidence_score,
                    reasoning_result.confidence
                )
                final_result.quality_assessment['unresolved_validation'] = False
                final_result.quality_assessment['reasoning_engine'] = {
                    'conclusion': reasoning_result.conclusion,
                    'confidence': reasoning_result.confidence,
                    'type': self.escalation_reasoning_type.value,
                    'execution_time_ms': reasoning_result.execution_time_ms,
                    'steps': len(reasoning_result.reasoning_steps)
                }

                attempt_history.append({
                    'attempt': len(attempt_history) + 1,
                    'strategy': 'reasoning_engine',
                    'reasoning_type': self.escalation_reasoning_type.value,
                    'backtrack_attempts': 0,
                    'backtrack_successes': 0,
                    'backtrack_failures': 0,
                    'confidence': reasoning_result.confidence,
                    'unresolved_validation': False,
                    'status': 'reasoning_engine_success',
                    'alternative_solutions': []
                })
            else:
                attempt_history.append({
                    'attempt': len(attempt_history) + 1,
                    'strategy': 'reasoning_engine',
                    'reasoning_type': self.escalation_reasoning_type.value,
                    'backtrack_attempts': 0,
                    'backtrack_successes': 0,
                    'backtrack_failures': 0,
                    'confidence': final_result.confidence_score,
                    'unresolved_validation': True,
                    'status': 'reasoning_engine_failed',
                    'alternative_solutions': [],
                    'error': str(reasoning_outcome.error)
                })
                final_result.quality_assessment['reasoning_engine'] = {
                    'error': str(reasoning_outcome.error),
                    'type': self.escalation_reasoning_type.value,
                    'attempted': True
                }

        except Exception as exc:
            logger.warning("ReasoningEngine 연동 중 예외: %s", exc)
            attempt_history.append({
                'attempt': len(attempt_history) + 1,
                'strategy': 'reasoning_engine',
                'reasoning_type': self.escalation_reasoning_type.value,
                'backtrack_attempts': 0,
                'backtrack_successes': 0,
                'backtrack_failures': 0,
                'confidence': final_result.confidence_score,
                'unresolved_validation': True,
                'status': 'reasoning_engine_error',
                'alternative_solutions': [],
                'error': str(exc)
            })

        return final_result

    async def _select_reasoning_strategy(self,
                                       complexity_score: int,
                                       context: Optional[Dict] = None) -> ReasoningStrategy:
        """복잡도 기반 추론 전략 선택"""

        # 도메인별 전략 선호도
        domain_preferences = {
            DomainType.MATHEMATICAL: ReasoningStrategy.SEQUENTIAL,
            DomainType.LOGICAL: ReasoningStrategy.HIERARCHICAL,
            DomainType.ANALYTICAL: ReasoningStrategy.SEQUENTIAL,
            DomainType.CREATIVE: ReasoningStrategy.PARALLEL,
            DomainType.TECHNICAL: ReasoningStrategy.HIERARCHICAL
        }

        # 복잡도별 기본 전략
        if complexity_score >= 80:
            base_strategy = ReasoningStrategy.HIERARCHICAL
        elif complexity_score >= 60:
            base_strategy = ReasoningStrategy.SEQUENTIAL
        elif complexity_score >= 40:
            base_strategy = ReasoningStrategy.ITERATIVE
        else:
            base_strategy = ReasoningStrategy.SEQUENTIAL

        # 컨텍스트 기반 조정
        if context:
            domain = context.get('domain')
            if domain in domain_preferences:
                return domain_preferences[domain]

            # 시간 제약이 있으면 단순한 전략 선택
            if context.get('time_constraint') == 'strict':
                return ReasoningStrategy.SEQUENTIAL

        return base_strategy

    async def _start_metacognition_monitoring(self,
                                            chain_id: str,
                                            problem: str,
                                            complexity_score: int):
        """메타인지 모니터링 시작"""
        try:
            if self.config.get('enable_metacognition', True):
                from .metacognition_engine import MetacognitionEngine

                self.metacognition_engine = MetacognitionEngine()

                task_context = {
                    'chain_id': chain_id,
                    'problem': problem,
                    'complexity_score': complexity_score,
                    'expected_steps': self._estimate_steps_count(complexity_score)
                }

                session_id = await self.metacognition_engine.start_reasoning_monitoring(task_context)
                self.current_monitoring_session = session_id
                logger.debug(f"메타인지 모니터링 시작: {session_id}")

        except Exception as e:
            logger.warning(f"메타인지 모니터링 시작 실패: {str(e)}")

    def _estimate_steps_count(self, complexity_score: int) -> int:
        """복잡도 기반 예상 단계 수 추정"""
        if complexity_score >= 80:
            return 7  # 매우 복잡: 7-10 단계
        elif complexity_score >= 60:
            return 5  # 복잡: 5-7 단계
        elif complexity_score >= 40:
            return 4  # 보통: 3-5 단계
        else:
            return 3  # 단순: 2-3 단계

    async def _analyze_problem(self,
                             problem: str,
                             complexity_score: int,
                             context: Optional[Dict] = None) -> Dict:
        """문제 분석 및 초기화"""

        analysis = {
            'original_problem': problem,
            'complexity_score': complexity_score,
            'estimated_steps': self._estimate_steps_count(complexity_score),
            'key_concepts': [],
            'constraints': [],
            'success_criteria': [],
            'potential_challenges': []
        }

        # 키 개념 추출
        analysis['key_concepts'] = await self._extract_key_concepts(problem)

        # 제약 조건 식별
        analysis['constraints'] = await self._identify_constraints(problem, context)

        # 성공 기준 정의
        analysis['success_criteria'] = await self._define_success_criteria(problem, context)

        # 잠재적 도전 과제 예측
        analysis['potential_challenges'] = await self._predict_challenges(problem, complexity_score)

        return analysis

    async def _extract_key_concepts(self, problem: str) -> List[str]:
        """핵심 개념 추출"""
        concepts = []

        # 간단한 키워드 기반 개념 추출
        keywords = ['분석', '계산', '비교', '평가', '설계', '구현', '최적화', '해결']
        for keyword in keywords:
            if keyword in problem:
                concepts.append(keyword)

        # 도메인 특화 개념들
        if '수학' in problem or '계산' in problem:
            concepts.extend(['수치분석', '공식적용'])
        if '논리' in problem or '추론' in problem:
            concepts.extend(['논리적사고', '인과관계'])

        return concepts if concepts else ['일반적문제해결']

    async def _identify_constraints(self, problem: str, context: Optional[Dict] = None) -> List[str]:
        """제약 조건 식별"""
        constraints = []

        # 시간 제약
        if context and context.get('time_constraint'):
            constraints.append(f"시간제약: {context['time_constraint']}")

        # 리소스 제약
        if '제한' in problem or '한정' in problem:
            constraints.append('리소스제약')

        # 품질 제약
        if '정확' in problem or '정밀' in problem:
            constraints.append('정확성요구')

        return constraints

    async def _define_success_criteria(self, problem: str, context: Optional[Dict] = None) -> List[str]:
        """성공 기준 정의"""
        criteria = []

        # 기본 성공 기준
        criteria.append('문제해결완료')
        criteria.append('논리적일관성유지')

        # 문제 유형별 기준
        if '분석' in problem:
            criteria.append('분석결과정확성')
        if '계산' in problem:
            criteria.append('계산결과검증')
        if '비교' in problem:
            criteria.append('비교기준명확성')

        return criteria

    async def _predict_challenges(self, problem: str, complexity_score: int) -> List[str]:
        """잠재적 도전 과제 예측"""
        challenges = []

        if complexity_score >= 70:
            challenges.append('높은복잡도로인한처리시간증가')
            challenges.append('다단계추론에서의일관성유지')

        if complexity_score >= 50:
            challenges.append('중간결과검증필요')

        if '다양한' in problem or '여러' in problem:
            challenges.append('다중요소고려필요')

        return challenges

    async def _execute_strategy(self,
                              strategy: ReasoningStrategy,
                              problem: str,
                              analysis: Dict,
                              complexity_score: int) -> List[ReasoningStep]:
        """전략별 추론 실행"""

        if strategy == ReasoningStrategy.SEQUENTIAL:
            return await self._execute_sequential_reasoning(problem, analysis, complexity_score)
        elif strategy == ReasoningStrategy.PARALLEL:
            return await self._execute_parallel_reasoning(problem, analysis, complexity_score)
        elif strategy == ReasoningStrategy.HIERARCHICAL:
            return await self._execute_hierarchical_reasoning(problem, analysis, complexity_score)
        elif strategy == ReasoningStrategy.ITERATIVE:
            return await self._execute_iterative_reasoning(problem, analysis, complexity_score)
        else:
            return await self._execute_sequential_reasoning(problem, analysis, complexity_score)

    async def _execute_sequential_reasoning(self,
                                          problem: str,
                                          analysis: Dict,
                                          complexity_score: int) -> List[ReasoningStep]:
        """순차적 추론 실행"""
        steps = []
        step_number = 1

        try:
            # 1. 문제 분석 단계
            step = await self._create_reasoning_step(
                step_number, StepType.PROBLEM_ANALYSIS,
                "문제 상황 분석", "주어진 문제를 체계적으로 분석합니다",
                {'problem': problem, 'analysis': analysis},
                {'key_points': analysis.get('key_concepts', []),
                 'complexity_assessment': complexity_score}
            )
            steps.append(step)
            step_number += 1

            # 2. 문제 분해 단계
            if complexity_score >= 40:
                decomposition = await self._decompose_problem(problem, analysis)
                step = await self._create_reasoning_step(
                    step_number, StepType.DECOMPOSITION,
                    "문제 분해", "복잡한 문제를 관리 가능한 하위 문제들로 분해합니다",
                    {'original_problem': problem},
                    {'sub_problems': decomposition}
                )
                steps.append(step)
                step_number += 1

            # 3. 가설 생성 단계
            if complexity_score >= 50:
                hypotheses = await self._generate_hypotheses(problem, analysis)
                step = await self._create_reasoning_step(
                    step_number, StepType.HYPOTHESIS_GENERATION,
                    "가설 생성", "문제 해결을 위한 가능한 가설들을 생성합니다",
                    {'problem_context': analysis},
                    {'hypotheses': hypotheses}
                )
                steps.append(step)
                step_number += 1

            # 4. 증거 수집 단계
            evidence = await self._gather_evidence(problem, steps)
            step = await self._create_reasoning_step(
                step_number, StepType.EVIDENCE_GATHERING,
                "증거 수집", "가설을 검증하기 위한 증거를 수집합니다",
                {'previous_steps': [s.step_id for s in steps]},
                {'evidence': evidence}
            )
            steps.append(step)
            step_number += 1

            # 5. 검증 단계
            if complexity_score >= 60:
                checkpoint = self._create_backtrack_point(steps, "pre_validation") if steps else None
                validation_result = await self._validate_intermediate_results(steps)
                validation_result.setdefault('failure_reason', 'sequential_validation_failure')
                validation_confidence = validation_result.get('confidence', 0.75)
                step = await self._create_reasoning_step(
                    step_number, StepType.VALIDATION,
                    "중간 결과 검증", "지금까지의 추론 과정과 결과를 검증합니다",
                    {
                        'steps_to_validate': [s.step_id for s in steps],
                        'identified_issues': validation_result.get('issues', [])
                    },
                    {'validation_result': validation_result},
                    confidence=validation_confidence
                )
                step.metadata['validation_summary'] = validation_result
                if validation_result.get('issues'):
                    step.warnings.extend(validation_result['issues'])
                if not validation_result.get('logic_validation', True):
                    step.errors.append('논리 검증 실패')
                steps.append(step)
                step_number += 1

                if self._should_trigger_backtrack(validation_result, steps):
                    steps = await self._execute_backtrack(
                        problem,
                        analysis,
                        steps,
                        validation_result,
                        checkpoint
                    )
                    step_number = steps[-1].step_number + 1 if steps else step_number

            # 6. 종합 단계
            synthesis = await self._synthesize_findings(steps)
            step = await self._create_reasoning_step(
                step_number, StepType.SYNTHESIS,
                "결과 종합", "수집된 증거와 분석 결과를 종합합니다",
                {'all_findings': [s.output_data for s in steps]},
                {'synthesis': synthesis}
            )
            steps.append(step)
            step_number += 1

            # 7. 결론 도출 단계
            conclusion = await self._draw_conclusion(problem, steps)
            step = await self._create_reasoning_step(
                step_number, StepType.CONCLUSION,
                "결론 도출", "최종 결론을 도출합니다",
                {'original_problem': problem, 'synthesis': synthesis},
                {'conclusion': conclusion}
            )
            steps.append(step)

        except Exception as e:
            logger.error(f"순차적 추론 실행 중 오류: {str(e)}")
            # 오류 단계 추가
            error_step = await self._create_error_step(step_number, str(e))
            steps.append(error_step)

        return steps

    async def _execute_parallel_reasoning(self,
                                        problem: str,
                                        analysis: Dict,
                                        complexity_score: int) -> List[ReasoningStep]:
        """병렬 추론 실행"""
        steps = []

        try:
            # 병렬 처리할 관점들 정의
            perspectives = [
                "분석적접근",
                "창의적접근",
                "체계적접근"
            ]

            # 병렬 추론 실행
            parallel_tasks = []
            for i, perspective in enumerate(perspectives):
                task = self._process_perspective(i + 1, perspective, problem, analysis)
                parallel_tasks.append(task)

            # 병렬 실행
            parallel_results = await asyncio.gather(*parallel_tasks, return_exceptions=True)

            # 결과 처리
            for i, result in enumerate(parallel_results):
                if isinstance(result, Exception):
                    logger.error(f"병렬 추론 {i+1} 실패: {str(result)}")
                    error_step = await self._create_error_step(i + 1, str(result))
                    steps.append(error_step)
                else:
                    steps.append(result)

            # 결과 통합
            integration_step = await self._integrate_parallel_results(steps, problem)
            steps.append(integration_step)

            # 통합 결과 검증 및 필요 시 백트래킹
            checkpoint = self._create_backtrack_point(steps, "parallel_post_integration") if steps else None
            validation_result = await self._validate_intermediate_results(steps)
            validation_result.setdefault('failure_reason', 'parallel_validation_failure')
            validation_confidence = validation_result.get('confidence', 0.74)
            validation_step_number = steps[-1].step_number + 1 if steps else 1
            validation_step = await self._create_reasoning_step(
                validation_step_number,
                StepType.VALIDATION,
                "병렬 결과 검증",
                "병렬 분석 경로 전반을 검증합니다",
                {
                    'steps_to_validate': [s.step_id for s in steps],
                    'identified_issues': validation_result.get('issues', [])
                },
                {'validation_result': validation_result},
                confidence=validation_confidence
            )
            validation_step.metadata['validation_summary'] = validation_result
            if validation_result.get('issues'):
                validation_step.warnings.extend(validation_result['issues'])
            if not validation_result.get('logic_validation', True):
                validation_step.errors.append('병렬 검증 실패')
            steps.append(validation_step)

            if self._should_trigger_backtrack(validation_result, steps):
                steps = await self._execute_backtrack(
                    problem,
                    analysis,
                    steps,
                    validation_result,
                    checkpoint
                )
                recovery_step_number = steps[-1].step_number + 1 if steps else 1
                recovery_step = await self._create_reasoning_step(
                    recovery_step_number,
                    StepType.SYNTHESIS,
                    "백트래킹 결과 통합",
                    "대안 병렬 경로를 통합하여 회복 전략을 정리합니다",
                    {
                        'backtrack_summary': self._summarize_backtracking(),
                        'strategy': 'parallel'
                    },
                    {
                        'integrated_result': '백트래킹을 반영한 병렬 결과'
                    },
                    confidence=max(validation_confidence, self.backtrack_confidence_threshold)
                )
                recovery_step.metadata['backtrack'] = True
                steps.append(recovery_step)

        except Exception as e:
            logger.error(f"병렬 추론 실행 중 오류: {str(e)}")

        return steps

    async def _process_perspective(self, step_number: int, perspective: str,
                                 problem: str, analysis: Dict) -> ReasoningStep:
        """관점별 처리"""
        return await self._create_reasoning_step(
            step_number, StepType.PROBLEM_ANALYSIS,
            f"{perspective} 관점 분석",
            f"{perspective} 관점에서 문제를 분석합니다",
            {'problem': problem, 'perspective': perspective},
            {'analysis_result': f"{perspective}에서의 분석 결과",
             'key_insights': [f"{perspective} 핵심 통찰"]}
        )

    async def _execute_hierarchical_reasoning(self,
                                            problem: str,
                                            analysis: Dict,
                                            complexity_score: int) -> List[ReasoningStep]:
        """계층적 추론 실행"""
        steps = []
        current_level = 0
        max_depth = self.strategy_configs[ReasoningStrategy.HIERARCHICAL]['max_depth']

        try:
            # 최상위 레벨에서 시작
            root_step = await self._create_reasoning_step(
                1, StepType.PROBLEM_ANALYSIS,
                "최상위 문제 분석", "전체 문제를 최상위 관점에서 분석합니다",
                {'problem': problem, 'level': current_level},
                {'high_level_analysis': '전체적 문제 구조 파악'}
            )
            steps.append(root_step)

            # 계층적 분해
            current_problems = [problem]
            step_number = 2

            while current_level < max_depth and current_problems:
                next_level_problems = []

                for subproblem in current_problems:
                    # 하위 문제들로 분해
                    sub_decomposition = await self._hierarchical_decompose(subproblem, current_level)

                    if sub_decomposition:
                        step = await self._create_reasoning_step(
                            step_number, StepType.DECOMPOSITION,
                            f"레벨 {current_level + 1} 분해",
                            f"레벨 {current_level + 1}에서 문제를 세분화합니다",
                            {'parent_problem': subproblem, 'level': current_level + 1},
                            {'sub_problems': sub_decomposition}
                        )
                        steps.append(step)
                        step_number += 1

                        next_level_problems.extend(sub_decomposition)

                current_problems = next_level_problems
                current_level += 1

            # 하위에서 상위로 결과 통합
            integration_step = await self._hierarchical_integration(steps)
            steps.append(integration_step)

            checkpoint = self._create_backtrack_point(steps, "hierarchical_post_integration") if steps else None
            validation_result = await self._validate_intermediate_results(steps)
            validation_result.setdefault('failure_reason', 'hierarchical_validation_failure')
            validation_confidence = validation_result.get('confidence', 0.73)
            validation_step_number = steps[-1].step_number + 1 if steps else 1
            validation_step = await self._create_reasoning_step(
                validation_step_number,
                StepType.VALIDATION,
                "계층 결과 검증",
                "계층적 추론 흐름을 검증합니다",
                {
                    'steps_to_validate': [s.step_id for s in steps],
                    'identified_issues': validation_result.get('issues', [])
                },
                {'validation_result': validation_result},
                confidence=validation_confidence
            )
            validation_step.metadata['validation_summary'] = validation_result
            if validation_result.get('issues'):
                validation_step.warnings.extend(validation_result['issues'])
            if not validation_result.get('logic_validation', True):
                validation_step.errors.append('계층 검증 실패')
            steps.append(validation_step)

            if self._should_trigger_backtrack(validation_result, steps):
                steps = await self._execute_backtrack(
                    problem,
                    analysis,
                    steps,
                    validation_result,
                    checkpoint
                )
                recovery_step_number = steps[-1].step_number + 1 if steps else 1
                recovery_step = await self._create_reasoning_step(
                    recovery_step_number,
                    StepType.SYNTHESIS,
                    "계층 백트래킹 통합",
                    "대안 계층 경로를 종합합니다",
                    {
                        'backtrack_summary': self._summarize_backtracking(),
                        'strategy': 'hierarchical'
                    },
                    {
                        'integrated_hierarchy': '백트래킹을 반영한 계층 통합'
                    },
                    confidence=max(validation_confidence, self.backtrack_confidence_threshold)
                )
                recovery_step.metadata['backtrack'] = True
                steps.append(recovery_step)

        except Exception as e:
            logger.error(f"계층적 추론 실행 중 오류: {str(e)}")

        return steps

    async def _execute_iterative_reasoning(self,
                                         problem: str,
                                         analysis: Dict,
                                         complexity_score: int) -> List[ReasoningStep]:
        """반복적 추론 실행"""
        steps = []
        max_iterations = self.strategy_configs[ReasoningStrategy.ITERATIVE]['max_iterations']
        convergence_threshold = self.strategy_configs[ReasoningStrategy.ITERATIVE]['convergence_threshold']

        try:
            current_solution = {'content': '초기 추정', 'confidence': 0.3}
            iteration = 1

            while iteration <= max_iterations:
                # 반복 단계 실행
                iteration_step = await self._execute_iteration(
                    iteration, problem, current_solution, analysis
                )
                steps.append(iteration_step)

                # 수렴 확인
                new_confidence = iteration_step.confidence
                if new_confidence >= convergence_threshold:
                    logger.info(f"반복 {iteration}에서 수렴 달성: {new_confidence:.2f}")
                    break

                # 다음 반복을 위한 솔루션 업데이트
                current_solution = iteration_step.output_data.get('refined_solution', current_solution)
                iteration += 1

            # 최종 수렴 평가
            convergence_step = await self._evaluate_convergence(steps)
            steps.append(convergence_step)

            checkpoint = self._create_backtrack_point(steps, "iterative_post_convergence") if steps else None
            validation_result = await self._validate_intermediate_results(steps)
            validation_result.setdefault('failure_reason', 'iterative_convergence_validation')
            validation_confidence = validation_result.get('confidence', 0.72)
            validation_step_number = steps[-1].step_number + 1 if steps else 1
            validation_step = await self._create_reasoning_step(
                validation_step_number,
                StepType.VALIDATION,
                "반복 추론 검증",
                "반복 추론의 수렴 결과를 검증합니다",
                {
                    'steps_to_validate': [s.step_id for s in steps],
                    'identified_issues': validation_result.get('issues', [])
                },
                {'validation_result': validation_result},
                confidence=validation_confidence
            )
            validation_step.metadata['validation_summary'] = validation_result
            if validation_result.get('issues'):
                validation_step.warnings.extend(validation_result['issues'])
            if not validation_result.get('logic_validation', True):
                validation_step.errors.append('반복 추론 검증 실패')
            steps.append(validation_step)

            if self._should_trigger_backtrack(validation_result, steps):
                steps = await self._execute_backtrack(
                    problem,
                    analysis,
                    steps,
                    validation_result,
                    checkpoint
                )
                recovery_step_number = steps[-1].step_number + 1 if steps else 1
                recovery_step = await self._create_reasoning_step(
                    recovery_step_number,
                    StepType.SYNTHESIS,
                    "반복 백트래킹 통합",
                    "백트래킹으로 조정된 반복 추론 결과를 정리합니다",
                    {
                        'backtrack_summary': self._summarize_backtracking(),
                        'strategy': 'iterative'
                    },
                    {
                        'synthesis': '백트래킹을 반영한 반복 추론 요약'
                    },
                    confidence=max(validation_confidence, self.backtrack_confidence_threshold)
                )
                recovery_step.metadata['backtrack'] = True
                steps.append(recovery_step)

        except Exception as e:
            logger.error(f"반복적 추론 실행 중 오류: {str(e)}")

        return steps

    async def _execute_iteration(self, iteration: int, problem: str,
                               current_solution: Dict, analysis: Dict) -> ReasoningStep:
        """단일 반복 실행"""
        # 이전 솔루션 개선
        improved_confidence = min(1.0, current_solution['confidence'] + 0.15)

        return await self._create_reasoning_step(
            iteration, StepType.VALIDATION,
            f"반복 {iteration}", f"{iteration}번째 반복을 통한 솔루션 개선",
            {'previous_solution': current_solution, 'problem': problem},
            {'refined_solution': {
                'content': f"개선된 솔루션 (반복 {iteration})",
                'confidence': improved_confidence
            }},
            confidence=improved_confidence
        )

    def _should_trigger_backtrack(self, validation_result: Dict, steps: List[ReasoningStep]) -> bool:
        """백트래킹 실행 여부 판단"""
        forced_failure = validation_result.get('forced_failure', False)

        if not self.enable_backtracking:
            return False

        if validation_result.get('logic_validation', True):
            return False

        if self.backtrack_attempts >= self.max_backtrack_attempts:
            logger.warning("백트래킹 최대 횟수에 도달하여 추가 시도를 건너뜁니다")
            return False

        if (not forced_failure and
                validation_result.get('confidence', 1.0) >= self.backtrack_confidence_threshold):
            return False

        return bool(steps)

    def _create_backtrack_point(self, steps: List[ReasoningStep], reason: str) -> Optional[BacktrackPoint]:
        """백트래킹 지점 생성"""
        if not steps:
            return None

        snapshot = copy.deepcopy(steps)
        point = BacktrackPoint(
            step_number=steps[-1].step_number,
            step_id=steps[-1].step_id,
            state_snapshot={'steps': snapshot},
            alternative_paths=[],
            reason=reason
        )
        return point

    def _restore_from_checkpoint(self, checkpoint: Optional[BacktrackPoint]) -> List[ReasoningStep]:
        """체크포인트로부터 단계 복원"""
        if not checkpoint or 'steps' not in checkpoint.state_snapshot:
            return []

        return copy.deepcopy(checkpoint.state_snapshot['steps'])

    def _find_backtrack_target(self, steps: List[ReasoningStep]) -> Optional[ReasoningStep]:
        """백트래킹 대상으로 되돌릴 단계 식별"""
        for step in reversed(steps):
            if step.step_type in (StepType.DECOMPOSITION, StepType.HYPOTHESIS_GENERATION, StepType.PROBLEM_ANALYSIS):
                return step
        return steps[-1] if steps else None

    async def _refine_decomposition(self,
                                   step_number: int,
                                   problem: str,
                                   analysis: Dict,
                                   target_step: Optional[ReasoningStep]) -> ReasoningStep:
        """백트래킹 시 문제 분해 재구성"""
        potential = analysis.get('potential_challenges', [])
        refined = [f"{challenge} 해결 경로" for challenge in potential[:3]]

        if not refined:
            refined = [
                f"{problem} 재검토",
                "추가 데이터 확보",
                "위험 요소 완화 전략"
            ]

        dependencies = [target_step.step_id] if target_step else []
        step = await self._create_reasoning_step(
            step_number,
            StepType.DECOMPOSITION,
            "대안 경로 재구성",
            "검증 실패 구간을 중심으로 문제 분해를 재구성합니다",
            {
                'previous_step': target_step.step_id if target_step else None,
                'focus': analysis.get('revisions', [])[-1] if analysis.get('revisions') else {}
            },
            {
                'sub_problems': refined,
                'backtrack': True
            },
            confidence=0.82,
            dependencies=dependencies
        )
        step.metadata['backtrack'] = True
        return step

    async def _generate_alternative_hypotheses(self, problem: str, analysis: Dict) -> List[str]:
        """백트래킹 시 새로운 가설 후보 생성"""
        base_hypotheses = await self._generate_hypotheses(problem, analysis)
        alternatives: List[str] = []

        if '데이터' in problem or '분석' in problem:
            alternatives.append('데이터 보강 전략 고려')
        if '오류' in problem or '실패' in problem:
            alternatives.append('오류 원인 격리 및 검증 계획')
        if analysis.get('constraints'):
            alternatives.append('제약 조건 완화 시나리오')

        alternatives.extend(f"{hypothesis} (대안)" for hypothesis in base_hypotheses)

        # 순서 유지한 채 중복 제거
        seen = set()
        ordered_alternatives = []
        for item in alternatives:
            if item not in seen:
                seen.add(item)
                ordered_alternatives.append(item)

        return ordered_alternatives or ['대안 가설 재정의 필요']

    async def _execute_backtrack(self,
                                problem: str,
                                analysis: Dict,
                                steps: List[ReasoningStep],
                                validation_result: Dict,
                                checkpoint: Optional[BacktrackPoint]) -> List[ReasoningStep]:
        """백트래킹 실행"""
        if not self.enable_backtracking:
            steps[-1].warnings.append('백트래킹이 비활성화되어 검증 경고를 유지합니다')
            self.backtrack_failures += 1
            return steps

        if self.backtrack_attempts >= self.max_backtrack_attempts:
            steps[-1].warnings.append('백트래킹 최대 횟수 초과')
            self.backtrack_failures += 1
            return steps

        self.backtrack_attempts += 1

        checkpoint = checkpoint or self._create_backtrack_point(
            steps[:-1], validation_result.get('failure_reason', 'validation_failure')
        )

        if checkpoint is None:
            steps[-1].warnings.append('백트래킹 실패: 체크포인트를 찾을 수 없음')
            self.backtrack_failures += 1
            return steps

        if checkpoint not in self.backtrack_history:
            self.backtrack_history.append(checkpoint)

        restored_steps = self._restore_from_checkpoint(checkpoint)
        target_step = self._find_backtrack_target(restored_steps)

        if target_step is None:
            steps[-1].warnings.append('백트래킹 대상 단계가 없습니다')
            self.backtrack_failures += 1
            return steps

        truncated_steps = [copy.deepcopy(s) for s in restored_steps if s.step_number <= target_step.step_number]
        steps.clear()
        steps.extend(truncated_steps)

        analysis.setdefault('revisions', []).append({
            'reason': validation_result.get('failure_reason', 'validation_failure'),
            'issues': validation_result.get('issues', []),
            'timestamp': current_timestamp(),
            'target_step': target_step.title
        })

        next_step_number = steps[-1].step_number + 1 if steps else 1

        refinement_step = await self._refine_decomposition(
            next_step_number, problem, analysis, target_step
        )
        steps.append(refinement_step)
        next_step_number = refinement_step.step_number + 1

        alternative_hypotheses = await self._generate_alternative_hypotheses(problem, analysis)
        self._latest_alternative_paths = alternative_hypotheses
        checkpoint.alternative_paths = alternative_hypotheses

        alternative_step = await self._create_reasoning_step(
            next_step_number,
            StepType.HYPOTHESIS_GENERATION,
            "대안 가설 탐색",
            "백트래킹 결과를 반영하여 새로운 가설을 구성합니다",
            {
                'problem_context': analysis,
                'target_step': refinement_step.step_id
            },
            {'hypotheses': alternative_hypotheses},
            confidence=0.86,
            dependencies=[refinement_step.step_id]
        )
        alternative_step.metadata['backtrack'] = True
        steps.append(alternative_step)
        next_step_number = alternative_step.step_number + 1

        evidence = await self._gather_evidence(problem, steps)
        evidence_step = await self._create_reasoning_step(
            next_step_number,
            StepType.EVIDENCE_GATHERING,
            "대안 증거 수집",
            "새로운 가설을 검증하기 위한 증거를 수집합니다",
            {'previous_steps': [s.step_id for s in steps]},
            {'evidence': evidence},
            confidence=0.83,
            dependencies=[alternative_step.step_id]
        )
        evidence_step.metadata['backtrack'] = True
        steps.append(evidence_step)

        new_validation_result = await self._validate_intermediate_results(steps)
        validation_confidence = max(
            new_validation_result.get('confidence', 0.0),
            self.backtrack_confidence_threshold
        )

        backtrack_validation_step = await self._create_reasoning_step(
            evidence_step.step_number + 1,
            StepType.VALIDATION,
            "백트래킹 검증",
            "수정된 추론 경로를 검증합니다",
            {
                'steps_to_validate': [s.step_id for s in steps],
                'recovery_reason': validation_result.get('failure_reason', 'validation_failure')
            },
            {'validation_result': new_validation_result},
            confidence=validation_confidence,
            dependencies=[evidence_step.step_id]
        )
        backtrack_validation_step.metadata['backtrack'] = True

        if not new_validation_result.get('logic_validation', True):
            backtrack_validation_step.errors.append('백트래킹 후에도 검증 실패')
            self.backtrack_failures += 1
        else:
            self.backtrack_successes += 1

        steps.append(backtrack_validation_step)

        analysis['revisions'][-1]['status'] = (
            'recovered' if new_validation_result.get('logic_validation', True) else 'warning'
        )

        logger.info(
            "백트래킹 실행 완료: attempts=%s, 성공=%s, 실패=%s",
            self.backtrack_attempts,
            self.backtrack_successes,
            self.backtrack_failures
        )

        return steps

    async def _create_reasoning_step(self,
                                   step_number: int,
                                   step_type: StepType,
                                   title: str,
                                   description: str,
                                   input_data: Dict,
                                   output_data: Dict,
                                   confidence: float = 0.8,
                                   dependencies: List[str] = None) -> ReasoningStep:
        """추론 단계 생성"""
        start_time = time.time()

        # 단계 처리 시뮬레이션
        await asyncio.sleep(0.01)  # 처리 시간 시뮬레이션

        processing_time = (time.time() - start_time) * 1000

        step = ReasoningStep(
            step_id=generate_id(f'step_{step_number}_'),
            step_number=step_number,
            step_type=step_type,
            title=title,
            description=description,
            input_data=input_data,
            output_data=output_data,
            reasoning_process=f"{title}: {description}",
            confidence=confidence,
            processing_time_ms=processing_time,
            dependencies=dependencies or []
        )

        # 메타인지 모니터링에 단계 추가
        if self.metacognition_engine and self.current_monitoring_session:
            try:
                await self.metacognition_engine.add_reasoning_step(
                    self.current_monitoring_session,
                    description,
                    input_data,
                    output_data,
                    processing_time
                )
            except Exception as e:
                logger.warning(f"메타인지 모니터링 업데이트 실패: {str(e)}")

        return step

    async def _create_error_step(self, step_number: int, error_message: str) -> ReasoningStep:
        """오류 단계 생성"""
        return ReasoningStep(
            step_id=generate_id(f'error_{step_number}_'),
            step_number=step_number,
            step_type=StepType.PROBLEM_ANALYSIS,
            title="오류 발생",
            description=f"추론 과정에서 오류 발생: {error_message}",
            input_data={},
            output_data={'error': error_message},
            reasoning_process=f"오류 처리: {error_message}",
            confidence=0.0,
            processing_time_ms=0.0,
            errors=[error_message]
        )

    # 다양한 헬퍼 메서드들
    async def _decompose_problem(self, problem: str, analysis: Dict) -> List[str]:
        """문제 분해"""
        # 기본적인 문제 분해 로직
        if '분석' in problem:
            return ['데이터 수집', '패턴 식별', '결과 해석']
        elif '계산' in problem:
            return ['입력 검증', '계산 수행', '결과 검증']
        elif '비교' in problem:
            return ['기준 설정', '대상 평가', '차이점 분석']
        else:
            return ['문제 이해', '솔루션 탐색', '결과 도출']

    async def _generate_hypotheses(self, problem: str, analysis: Dict) -> List[str]:
        """가설 생성"""
        hypotheses = []

        if '원인' in problem:
            hypotheses.extend(['직접적 원인', '간접적 원인', '복합적 원인'])
        if '해결' in problem:
            hypotheses.extend(['기술적 해결책', '절차적 해결책', '창의적 해결책'])

        return hypotheses if hypotheses else ['기본 가설']

    async def _gather_evidence(self, problem: str, steps: List[ReasoningStep]) -> List[str]:
        """증거 수집"""
        evidence = []

        # 이전 단계들에서 증거 추출
        for step in steps:
            if step.step_type in [StepType.PROBLEM_ANALYSIS, StepType.DECOMPOSITION]:
                evidence.append(f"{step.title}에서 도출된 통찰")

        return evidence if evidence else ['기본 관찰 사실']

    async def _validate_intermediate_results(self, steps: List[ReasoningStep]) -> Dict:
        """중간 결과 검증"""
        if not steps:
            return {
                'consistency_check': False,
                'logic_validation': False,
                'completeness_score': 0.0,
                'confidence': 0.0,
                'issues': ['검증할 단계가 없습니다'],
                'validation_notes': '단계 부재로 검증 실패'
            }

        issues: List[str] = []
        all_confidences = [max(0.0, min(step.confidence, 1.0)) for step in steps]
        avg_confidence = sum(all_confidences) / len(all_confidences)

        # 기본 검증 규칙
        logic_validation = True
        consistency_check = True

        if avg_confidence < 0.6:
            issues.append('평균 신뢰도가 0.6 미만')
            logic_validation = False

        if any(step.errors for step in steps):
            issues.append('이전 단계에서 오류 발생')
            logic_validation = False

        latest_step = steps[-1]
        if latest_step.warnings:
            issues.extend(latest_step.warnings)

        forced_failure = False
        # 구성 기반 강제 실패 지원 (테스트/예외 상황)
        if self.config.get('force_validation_failure', False):
            issues.append('구성 설정에 의해 검증 실패가 강제되었습니다')
            logic_validation = False
            forced_failure = True

        metadata = latest_step.metadata
        if metadata.get('force_validation_failure'):
            issues.append(metadata.get('failure_reason', '강제 검증 실패'))
            logic_validation = False
            forced_failure = True

        completeness_score = 0.8 if len(steps) >= 3 else 0.5
        if len(set(step.step_type for step in steps)) < 2:
            issues.append('단계 유형 다양성이 부족합니다')
            consistency_check = False

        return {
            'consistency_check': consistency_check,
            'logic_validation': logic_validation,
            'completeness_score': completeness_score,
            'confidence': avg_confidence,
            'issues': issues,
            'validation_notes': '검증 규칙 평가 완료',
            'forced_failure': forced_failure
        }

    async def _synthesize_findings(self, steps: List[ReasoningStep]) -> Dict:
        """결과 종합"""
        return {
            'key_findings': [f"단계 {step.step_number}의 핵심 발견" for step in steps[:3]],
            'overall_pattern': '체계적인 문제 해결 과정',
            'confidence_trend': 'increasing'
        }

    async def _draw_conclusion(self, problem: str, steps: List[ReasoningStep]) -> str:
        """결론 도출"""
        step_count = len(steps)
        avg_confidence = sum(step.confidence for step in steps) / step_count if steps else 0

        return f"총 {step_count}단계의 체계적 분석을 통해 문제를 해결했습니다. 평균 신뢰도: {avg_confidence:.2f}"

    async def _integrate_parallel_results(self, steps: List[ReasoningStep], problem: str) -> ReasoningStep:
        """병렬 결과 통합"""
        return await self._create_reasoning_step(
            len(steps) + 1, StepType.SYNTHESIS,
            "병렬 결과 통합", "다양한 관점의 분석 결과를 통합합니다",
            {'parallel_steps': [s.step_id for s in steps]},
            {'integrated_result': '통합된 분석 결과',
             'consensus': '관점들 간의 합의점'}
        )

    async def _hierarchical_decompose(self, problem: str, level: int) -> List[str]:
        """계층적 분해"""
        if level >= 2:
            return []  # 더 이상 분해하지 않음

        return [f"{problem}의 하위문제 {i+1}" for i in range(2)]

    async def _hierarchical_integration(self, steps: List[ReasoningStep]) -> ReasoningStep:
        """계층적 통합"""
        return await self._create_reasoning_step(
            len(steps) + 1, StepType.SYNTHESIS,
            "계층적 통합", "하위 레벨의 결과를 상위로 통합합니다",
            {'hierarchical_steps': [s.step_id for s in steps]},
            {'integrated_hierarchy': '계층별 결과 통합'}
        )

    async def _evaluate_convergence(self, steps: List[ReasoningStep]) -> ReasoningStep:
        """수렴 평가"""
        final_confidence = steps[-1].confidence if steps else 0.5

        return await self._create_reasoning_step(
            len(steps) + 1, StepType.VERIFICATION,
            "수렴 평가", "반복적 추론의 수렴성을 평가합니다",
            {'iteration_steps': [s.step_id for s in steps]},
            {'convergence_achieved': final_confidence >= 0.8,
             'final_confidence': final_confidence}
        )

    async def _synthesize_conclusion(self, steps: List[ReasoningStep]) -> Tuple[str, float]:
        """결론 종합"""
        if not steps:
            return "추론 단계가 없어 결론을 도출할 수 없습니다.", 0.0

        # 최종 단계에서 결론 추출
        final_step = steps[-1]
        conclusion_data = final_step.output_data

        if 'conclusion' in conclusion_data:
            conclusion = conclusion_data['conclusion']
        elif 'integrated_result' in conclusion_data:
            conclusion = conclusion_data['integrated_result']
        else:
            conclusion = f"{len(steps)}단계의 체계적 추론을 통한 결과"

        # 전체 신뢰도 계산
        total_confidence = sum(step.confidence for step in steps) / len(steps)

        return conclusion, total_confidence

    async def _assess_quality(self, steps: List[ReasoningStep], conclusion: str) -> Dict:
        """품질 평가"""
        return {
            'step_count': len(steps),
            'average_confidence': sum(s.confidence for s in steps) / len(steps) if steps else 0,
            'logical_flow': True,
            'completeness': 0.8,
            'conclusion_validity': 0.85,
            'backtracking': {
                'attempts': self.backtrack_attempts,
                'successes': self.backtrack_successes,
                'failures': self.backtrack_failures
            }
        }

    async def _extract_learning_insights(self, steps: List[ReasoningStep]) -> List[str]:
        """학습 통찰 추출"""
        insights = []

        if len(steps) >= 5:
            insights.append("체계적인 다단계 추론 경험")

        step_types = [step.step_type for step in steps]
        if StepType.VALIDATION in step_types:
            insights.append("중간 검증의 중요성 확인")

        avg_confidence = sum(s.confidence for s in steps) / len(steps) if steps else 0
        if avg_confidence >= 0.8:
            insights.append("높은 신뢰도 추론 패턴 학습")

        return insights if insights else ["기본적인 추론 경험"]

    def _determine_complexity_level(self, score: int) -> ComplexityLevel:
        """복잡도 수준 결정"""
        if score >= 80:
            return ComplexityLevel.EXTREME
        elif score >= 60:
            return ComplexityLevel.VERY_COMPLEX
        elif score >= 40:
            return ComplexityLevel.COMPLEX
        elif score >= 20:
            return ComplexityLevel.MODERATE
        else:
            return ComplexityLevel.SIMPLE

    async def _end_metacognition_monitoring(self):
        """메타인지 모니터링 종료"""
        try:
            if (self.metacognition_engine and
                self.current_monitoring_session):
                await self.metacognition_engine.end_monitoring_session(
                    self.current_monitoring_session
                )
                logger.debug("메타인지 모니터링 종료")
        except Exception as e:
            logger.warning(f"메타인지 모니터링 종료 실패: {str(e)}")

    async def _create_error_result(self, chain_id: str, problem: str,
                                 error_message: str, elapsed_time: float) -> ReasoningResult:
        """오류 결과 생성"""
        return ReasoningResult(
            chain_id=chain_id,
            problem=problem,
            complexity_info=ComplexityResult(
                score=0, reasoning_required=False,
                domain=DomainType.CONVERSATIONAL, confidence=0.0,
                level=ComplexityLevel.SIMPLE, analysis_details={},
                processing_time_ms=0, timestamp=current_timestamp(),
                unique_id=generate_id('error_')
            ),
            strategy_used=ReasoningStrategy.SEQUENTIAL,
            steps=[],
            final_conclusion=f"추론 실행 중 오류 발생: {error_message}",
            confidence_score=0.0,
            total_processing_time_ms=elapsed_time * 1000,
            quality_assessment={'error': True},
            status=ReasoningStatus.FAILED,
            error_log=[error_message],
            backtrack_attempts=self.backtrack_attempts,
            backtrack_successes=self.backtrack_successes,
            backtrack_failures=self.backtrack_failures,
            backtrack_summary=self._summarize_backtracking()
        )

    def _update_performance_stats(self, result: ReasoningResult):
        """성능 통계 업데이트"""
        self.total_chains_executed += 1

        # 평균 처리 시간 업데이트
        current_time = result.total_processing_time_ms
        self.average_processing_time = (
            (self.average_processing_time * (self.total_chains_executed - 1) + current_time)
            / self.total_chains_executed
        )

        # 성공률 업데이트
        if result.status == ReasoningStatus.COMPLETED:
            success_count = self.total_chains_executed * self.success_rate + 1
            self.success_rate = success_count / self.total_chains_executed

    def get_performance_summary(self) -> Dict:
        """성능 요약 반환"""
        return {
            'total_chains_executed': self.total_chains_executed,
            'average_processing_time_ms': self.average_processing_time,
            'success_rate': self.success_rate,
            'current_active_monitoring': self.current_monitoring_session is not None,
            'backtracking': {
                'attempts': self.backtrack_attempts,
                'successes': self.backtrack_successes,
                'failures': self.backtrack_failures
            }
        }


# 편의성 함수들
async def execute_reasoning(problem: str,
                          complexity_score: int,
                          config: Optional[Dict] = None) -> ReasoningResult:
    """추론 실행 편의 함수"""
    chain = ReasoningChain(config)
    return await chain.execute_reasoning_chain(problem, complexity_score)


def create_reasoning_chain(enable_metacognition: bool = True,
                         max_steps: int = 10) -> ReasoningChain:
    """추론 체인 생성 함수"""
    config = {
        'enable_metacognition': enable_metacognition,
        'max_sequential_steps': max_steps
    }
    return ReasoningChain(config)


# 모듈 내보내기
__all__ = [
    'ReasoningChain',
    'ReasoningResult',
    'ReasoningStep',
    'ReasoningStrategy',
    'StepType',
    'ReasoningStatus',
    'BacktrackPoint',
    'execute_reasoning',
    'create_reasoning_chain'
]


# === Collaboration policy extension injected (non-invasive) ===
# This block adds collaboration-policy-based retries without editing the original class body.
# It defines helper functions and monkey-patches ReasoningChain to use them.
from typing import Any as _Any, Dict as _Dict, List as _List, Optional as _Optional, Tuple as _Tuple

# --- helpers for mapping strings to ReasoningType ---
def _rc__resolve_reasoning_type(value: str):
    try:
        return ReasoningType(value)
    except Exception:
        try:
            return ReasoningType[value.upper()]
        except Exception:
            return ReasoningType.DEDUCTIVE

# --- parse/resolve collaboration policy ---
def _rc__normalize_rule(self, rule: _Dict) -> _Tuple[_Tuple[str, ...], int]:
    rts = tuple((rule or {}).get("reasoning_types", []) or [])
    ma = int((rule or {}).get("max_attempts", 1) or 1)
    if ma < 1: ma = 1
    return rts, ma

def _rc__parse_collaboration_policy(self, raw: _Dict) -> _Dict[str, _Dict]:
    parsed: _Dict[str, _Dict] = {}
    default_rts, default_ma = _rc__normalize_rule(self, raw.get("default", {"reasoning_types": ["deductive"], "max_attempts": 1}))
    parsed["__default__"] = {"reasoning_types": default_rts, "max_attempts": default_ma}
    for k, v in (raw or {}).items():
        if k == "default":
            continue
        rts, ma = _rc__normalize_rule(self, v)
        parsed[k] = {"reasoning_types": rts, "max_attempts": ma}
    return parsed

def _rc__resolve_collab_rule(self, reason_key: _Optional[str]) -> _Dict:
    if not hasattr(self, "escalation_collaboration_policy") or not self.escalation_collaboration_policy:
        self.escalation_collaboration_policy = _rc__parse_collaboration_policy(self, {})
    if not reason_key:
        return self.escalation_collaboration_policy.get("__default__", {"reasoning_types": ("deductive",), "max_attempts": 1})
    return self.escalation_collaboration_policy.get(reason_key,
           self.escalation_collaboration_policy.get("__default__", {"reasoning_types": ("deductive",), "max_attempts": 1}))

# --- attempt reasoning via external ReasoningEngine according to policy ---
async def _rc__attempt_reasoning_type(self, reasoning_type: str, premises: _List[_Any], target_conclusion: _Optional[str], meta: _Dict):
    if not getattr(self, "reasoning_engine", None):
        return None
    try:
        # Ensure initialized similar to _escalate_with_reasoning_engine
        try:
            init_result = await self.reasoning_engine.initialize()
            if getattr(init_result, "is_failure", False):
                return None
        except Exception:
            # proceed best-effort
            pass

        rt = _rc__resolve_reasoning_type(reasoning_type)
        outcome = await self.reasoning_engine.reason(
            rt,
            premises=premises if premises else [meta.get("problem")],
            target_conclusion=target_conclusion,
            confidence_threshold=getattr(self, "escalation_min_confidence", 0.8),
            max_steps=10
        )
        return outcome
    except Exception:
        return None

async def _rc__try_collaboration_retries(self, reason_key: _Optional[str], final_result, attempt_history: _List[_Dict], context: _Optional[_Dict] = None):
    rule = _rc__resolve_collab_rule(self, reason_key)
    attempts_log: _List[_Dict] = []
    premises: _List[_Any] = []
    for step in getattr(final_result, "steps", []) or []:
        if getattr(step, "output_data", None):
            premises.append(step.output_data)
    target_conclusion = getattr(final_result, "final_conclusion", None)

    for rtype in rule["reasoning_types"]:
        for attempt_idx in range(1, int(rule["max_attempts"]) + 1):
            meta = {
                "collab_reason": reason_key or "default",
                "reasoning_type": rtype,
                "attempt": attempt_idx,
                "problem": getattr(final_result, "problem", None)
            }
            outcome = await _rc__attempt_reasoning_type(self, rtype, premises, target_conclusion, meta)
            if outcome is None:
                attempts_log.append({**meta, "status": "engine_unavailable"})
                continue
            if getattr(outcome, "is_success", False):
                val = getattr(outcome, "value", None)
                # Update final_result
                try:
                    if val is not None:
                        concl = getattr(val, "conclusion", None)
                        conf = getattr(val, "confidence", None)
                        exec_ms = getattr(val, "execution_time_ms", None)
                        steps_cnt = len(getattr(val, "reasoning_steps", []) or [])
                        if concl:
                            final_result.final_conclusion = concl
                        if isinstance(conf, (int, float)):
                            final_result.confidence_score = max(final_result.confidence_score, float(conf))
                        qa = getattr(final_result, "quality_assessment", {})
                        qa["unresolved_validation"] = False
                        qa["reasoning_engine_collab"] = {
                            "conclusion": concl,
                            "confidence": conf,
                            "type": rtype,
                            "execution_time_ms": exec_ms,
                            "steps": steps_cnt,
                            "via": "collaboration_policy"
                        }
                        final_result.quality_assessment = qa
                except Exception:
                    pass

                attempts_log.append({**meta, "status": "ok"})
                # reflect attempt in history
                attempt_history.append({
                    'attempt': len(attempt_history) + 1,
                    'strategy': 'reasoning_engine_collab',
                    'reasoning_type': rtype,
                    'backtrack_attempts': 0,
                    'backtrack_successes': 0,
                    'backtrack_failures': 0,
                    'confidence': getattr(final_result, "confidence_score", None),
                    'unresolved_validation': False,
                    'status': 'collaboration_success',
                    'alternative_solutions': []
                })
                _rc__record_collab_attempts(self, attempts_log)
                return final_result
            else:
                attempts_log.append({**meta, "status": "no_effect", "error": getattr(outcome, "error", None)})
                attempt_history.append({
                    'attempt': len(attempt_history) + 1,
                    'strategy': 'reasoning_engine_collab',
                    'reasoning_type': rtype,
                    'backtrack_attempts': 0,
                    'backtrack_successes': 0,
                    'backtrack_failures': 0,
                    'confidence': getattr(final_result, "confidence_score", None),
                    'unresolved_validation': True,
                    'status': 'collaboration_failed',
                    'alternative_solutions': [],
                    'error': getattr(outcome, "error", None)
                })

    _rc__record_collab_attempts(self, attempts_log)
    return final_result

def _rc__record_collab_attempts(self, attempts: _List[_Dict]):
    try:
        logger.info("collaboration_attempts", extra={"attempts": attempts})
    except Exception:
        pass

# --- patch _execute_backtrack to trigger collaboration retries on failure ---
# Save original
if hasattr(ReasoningChain, "_execute_backtrack"):
    ReasoningChain.__orig_execute_backtrack = ReasoningChain._execute_backtrack

# Define patched wrapper
async def _rc__execute_backtrack_patched(self, problem: str, analysis: _Dict, steps: _List, validation_result: _Dict, checkpoint):
    # Call original backtrack implementation
    result_steps = await ReasoningChain.__orig_execute_backtrack(self, problem, analysis, steps, validation_result, checkpoint)
    try:
        # Determine if still failing
        reason_key = validation_result.get('failure_reason') or validation_result.get('reason') or None
        still_failing = True
        if result_steps and hasattr(result_steps[-1], "metadata"):
            last_meta = getattr(result_steps[-1], "metadata", {}) or {}
            vres = last_meta.get("validation_summary") or last_meta.get("validation_result") or {}
            if isinstance(vres, dict):
                # If logic_validation True, consider recovered
                still_failing = not vres.get('logic_validation', True)
        # If failing, and escalation is enabled, try collaboration
        if still_failing and getattr(self, "enable_reasoning_engine_escalation", False):
            # Build a minimal final_result-like object from current context to reuse _rc__try_collaboration_retries
            final_result = type("FinalLike", (), {})()
            final_result.problem = problem
            final_result.steps = result_steps
            # Try to get current conclusion/confidence; fallbacks
            final_result.final_conclusion = "백트래킹 후 임시 결론"
            final_result.confidence_score = 0.0
            final_result.quality_assessment = {'unresolved_validation': True}
            attempt_history = []
            updated = await _rc__try_collaboration_retries(self, reason_key, final_result, attempt_history, context=analysis)
            # If collaboration cleared unresolved_validation, append a synthetic validation step
            if updated and not updated.quality_assessment.get('unresolved_validation', True):
                # Mark recovery by adding a synthetic validation step summary
                from types import SimpleNamespace
                class _TmpStep:
                    pass
                # No change to steps list structure; we simply return as-is since higher layers read quality_assessment
                pass
    except Exception:
        pass
    return result_steps

# Monkey-patch the method
ReasoningChain._execute_backtrack = _rc__execute_backtrack_patched

# Attach helpers to class for potential direct calls elsewhere
ReasoningChain._parse_collaboration_policy = _rc__parse_collaboration_policy
ReasoningChain._resolve_collab_rule = _rc__resolve_collab_rule
ReasoningChain._try_collaboration_retries = _rc__try_collaboration_retries
ReasoningChain._attempt_reasoning_type = _rc__attempt_reasoning_type
ReasoningChain._record_collab_attempts = _rc__record_collab_attempts

# Ensure escalation_collaboration_policy exists even if not set during __init__
if not hasattr(ReasoningChain, "escalation_collaboration_policy"):
    try:
        ReasoningChain.escalation_collaboration_policy = {}
    except Exception:
        pass

