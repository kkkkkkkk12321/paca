"""
메타인지 엔진 (Metacognition Engine)
PACA v5 Python의 핵심 차별화 기능

AI의 사고 과정을 모니터링하고 개선하는 시스템
추론 품질을 평가하고 자기반성 메커니즘을 제공
"""

import json
import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 타입 시스템 임포트
from paca.core.types import generate_id, current_timestamp
from paca.core.validators import validate_range, is_valid_json

# 로깅 설정
logger = logging.getLogger(__name__)


class MonitoringPhase(Enum):
    """모니터링 단계"""
    INITIALIZATION = "initialization"    # 초기화
    PROCESSING = "processing"           # 처리 중
    VALIDATION = "validation"           # 검증
    COMPLETION = "completion"           # 완료
    ERROR = "error"                     # 오류


class ReasoningQuality(Enum):
    """추론 품질 등급"""
    EXCELLENT = (90, 100)    # 우수
    GOOD = (75, 89)          # 양호
    MODERATE = (60, 74)      # 보통
    POOR = (40, 59)          # 미흡
    CRITICAL = (0, 39)       # 심각


class QualityLevel(Enum):
    """품질 레벨"""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass
class QualityAssessment:
    """메타인지 품질 평가 결과"""

    score: float
    grade: ReasoningQuality
    level: QualityLevel
    complexity_score: Optional[float] = None
    average_confidence: Optional[float] = None
    alerts: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'score': round(self.score, 2),
            'grade': self.grade.name,
            'level': self.level.value,
            'complexity_score': self.complexity_score,
            'average_confidence': None if self.average_confidence is None else round(self.average_confidence, 3),
            'alerts': self.alerts,
            'recommended_actions': self.recommended_actions,
        }


@dataclass
class QualityMetrics:
    """추론 품질 메트릭"""
    logical_consistency: float = 0.0    # 논리적 일관성 (0-1)
    step_clarity: float = 0.0           # 단계별 명확성 (0-1)
    conclusion_validity: float = 0.0    # 결론 타당성 (0-1)
    completeness: float = 0.0           # 완전성 (0-1)
    efficiency: float = 0.0             # 효율성 (0-1)

    def calculate_overall_score(self) -> float:
        """전체 품질 점수 계산 (0-100)"""
        weights = {
            'logical_consistency': 0.25,
            'step_clarity': 0.20,
            'conclusion_validity': 0.25,
            'completeness': 0.20,
            'efficiency': 0.10
        }

        total = (
            self.logical_consistency * weights['logical_consistency'] +
            self.step_clarity * weights['step_clarity'] +
            self.conclusion_validity * weights['conclusion_validity'] +
            self.completeness * weights['completeness'] +
            self.efficiency * weights['efficiency']
        )

        return min(100.0, max(0.0, total * 100))

    def get_quality_grade(self) -> ReasoningQuality:
        """품질 등급 반환"""
        score = self.calculate_overall_score()
        for grade in ReasoningQuality:
            min_score, max_score = grade.value
            if min_score <= score <= max_score:
                return grade
        return ReasoningQuality.CRITICAL


@dataclass
class ReasoningStep:
    """추론 단계"""
    step_id: str                        # 단계 ID
    step_number: int                    # 단계 번호
    description: str                    # 단계 설명
    input_data: Dict                    # 입력 데이터
    output_data: Dict                   # 출력 데이터
    processing_time_ms: float           # 처리 시간
    quality_score: float                # 단계별 품질 점수
    confidence: float                   # 신뢰도
    errors: List[str] = field(default_factory=list)  # 오류 목록
    warnings: List[str] = field(default_factory=list)  # 경고 목록
    timestamp: float = field(default_factory=current_timestamp)


@dataclass
class MonitoringSession:
    """모니터링 세션"""
    session_id: str                     # 세션 ID
    task_context: Dict                  # 작업 컨텍스트
    start_time: float                   # 시작 시간
    end_time: Optional[float] = None    # 종료 시간
    phase: MonitoringPhase = MonitoringPhase.INITIALIZATION
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    quality_metrics: Optional[QualityMetrics] = None
    quality_assessment: Optional[QualityAssessment] = None
    metadata: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)

    def get_duration_ms(self) -> float:
        """세션 지속 시간 반환 (밀리초)"""
        end = self.end_time if self.end_time else current_timestamp()
        return (end - self.start_time) * 1000


@dataclass
class SelfReflectionResult:
    """자기반성 결과"""
    reflection_id: str                  # 반성 ID
    session_id: str                     # 대상 세션 ID
    strengths: List[str]                # 강점
    weaknesses: List[str]               # 약점
    improvement_suggestions: List[str]  # 개선 제안
    learning_points: List[str]          # 학습 포인트
    overall_assessment: str             # 전체 평가
    confidence_in_assessment: float     # 평가 신뢰도
    timestamp: float = field(default_factory=current_timestamp)


class MetacognitionEngine:
    """
    메타인지 엔진

    AI의 사고 과정을 실시간으로 모니터링하고
    추론 품질을 평가하여 자기개선을 지원하는 시스템
    """
    CONFIG_FILENAME = "complexity_thresholds.json"
    DEFAULT_CONFIG_PATH = (
        Path(__file__).resolve().parent.parent.parent / "data" / "config" / CONFIG_FILENAME
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        settings_bundle = self._load_configuration(config or {})
        metacog_settings = settings_bundle.get('metacognition', settings_bundle)

        self.config = metacog_settings
        self.config_source = metacog_settings.get('__source__', str(self.DEFAULT_CONFIG_PATH))

        self.quality_thresholds = self._prepare_quality_thresholds(
            metacog_settings.get('quality_thresholds')
        )
        self.alert_settings = {
            'high_complexity_score': metacog_settings.get('high_complexity_score', 70),
            'low_confidence': metacog_settings.get('low_confidence', 0.45),
        }
        alerts_override = metacog_settings.get('alerts')
        if isinstance(alerts_override, dict):
            self.alert_settings.update(alerts_override)

        self.log_enabled = metacog_settings.get('logging_enabled', True)
        self.log_directory_setting = metacog_settings.get('log_directory', 'logs/metacognition')
        self.log_directory_path = (
            self._resolve_log_directory(self.log_directory_setting) if self.log_enabled else None
        )

        self.max_history_size = metacog_settings.get('max_history_size', 100)

        self.active_sessions: Dict[str, MonitoringSession] = {}
        self.session_history: List[MonitoringSession] = []

        self.total_sessions = 0
        self.average_quality_score = 0.0

        logger.info(
            "MetacognitionEngine 초기화 완료",
            extra={
                'config_source': self.config_source,
                'log_directory': str(self.log_directory_path) if self.log_directory_path else None,
            },
        )

    def _load_configuration(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        file_config: Dict[str, Any] = {}
        if self.DEFAULT_CONFIG_PATH.exists():
            try:
                with self.DEFAULT_CONFIG_PATH.open('r', encoding='utf-8') as handle:
                    file_config = json.load(handle)
                    file_config.setdefault('metacognition', {})['__source__'] = str(
                        self.DEFAULT_CONFIG_PATH
                    )
            except json.JSONDecodeError as error:
                logger.warning(
                    "MetacognitionEngine 설정 파일 파싱 실패",
                    extra={'error': str(error)},
                )

        return self._deep_merge_dicts(file_config, overrides)

    @staticmethod
    def _deep_merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        result = deepcopy(base)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = MetacognitionEngine._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        return result

    def _prepare_quality_thresholds(
        self, overrides: Optional[Dict[str, Union[int, float]]]
    ) -> Dict[str, float]:
        thresholds = {'green': 80.0, 'yellow': 60.0, 'red': 40.0}
        if overrides:
            for key in ('green', 'yellow', 'red'):
                if key in overrides:
                    thresholds[key] = float(overrides[key])

        if thresholds['green'] < thresholds['yellow']:
            thresholds['green'] = thresholds['yellow']
        if thresholds['yellow'] < thresholds['red']:
            thresholds['yellow'] = thresholds['red']
        return thresholds

    def _resolve_log_directory(self, log_setting: Optional[str]) -> Optional[Path]:
        if not log_setting:
            return None

        path = Path(log_setting)
        if not path.is_absolute():
            base = Path(__file__).resolve().parent.parent.parent
            path = base / log_setting

        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as error:  # pragma: no cover - 파일 시스템 오류 처리
            logger.warning(
                "메타인지 로그 디렉터리 생성 실패",
                extra={'path': str(path), 'error': str(error)},
            )
            return None

        return path

    def _assess_quality(self, session: MonitoringSession) -> Optional[QualityAssessment]:
        if not session.quality_metrics:
            return None

        score = session.quality_metrics.calculate_overall_score()
        grade = session.quality_metrics.get_quality_grade()

        thresholds = self.quality_thresholds
        if score >= thresholds['green']:
            level = QualityLevel.GREEN
        elif score >= thresholds['yellow']:
            level = QualityLevel.YELLOW
        else:
            level = QualityLevel.RED

        complexity_score = session.metadata.get('complexity_score')
        avg_confidence = self._calculate_average_confidence(session)

        alerts: List[str] = []
        recommendations: List[str] = []

        if level != QualityLevel.GREEN and complexity_score is not None:
            if complexity_score >= self.alert_settings.get('high_complexity_score', 70):
                alerts.append('고복잡도 입력에서 품질 저하 감지')
                recommendations.append('추론 체인 fallback 또는 세부 검증 경로 실행')

        if avg_confidence is not None:
            if avg_confidence < self.alert_settings.get('low_confidence', 0.45):
                alerts.append('추론 단계 평균 신뢰도 하락')
                recommendations.append('LLM 응답 모니터링 및 규칙 기반 보강 필요')

        if not session.reasoning_steps:
            alerts.append('추론 단계 데이터 부족')
            recommendations.append('추론 로그 수집 또는 예외 처리 강화')

        assessment = QualityAssessment(
            score=score,
            grade=grade,
            level=level,
            complexity_score=complexity_score,
            average_confidence=avg_confidence,
            alerts=alerts,
            recommended_actions=recommendations,
        )

        session.metadata['quality_assessment'] = assessment.to_dict()
        return assessment

    def _emit_quality_alert(
        self, session: MonitoringSession, assessment: QualityAssessment
    ) -> Optional[Dict[str, Any]]:
        if assessment.level == QualityLevel.GREEN and not assessment.alerts:
            return None

        payload = {
            'session_id': session.session_id,
            'level': assessment.level.value,
            'score': assessment.score,
            'complexity_score': assessment.complexity_score,
            'alerts': assessment.alerts,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }

        log_method = logger.warning if assessment.level != QualityLevel.RED else logger.error
        log_method("Metacognition 품질 경보", extra=payload)
        return payload

    def _write_session_log(self, session: MonitoringSession) -> None:
        if not self.log_directory_path:
            return

        now = datetime.now(timezone.utc)
        log_path = self.log_directory_path / f"{now:%Y%m%d}.log"
        payload = {
            'session_id': session.session_id,
            'ended_at': now.isoformat(),
            'quality': session.quality_assessment.to_dict() if session.quality_assessment else None,
            'quality_metrics': session.quality_metrics.calculate_overall_score() if session.quality_metrics else None,
            'complexity_score': session.metadata.get('complexity_score'),
            'alerts': session.alerts,
        }

        try:
            with log_path.open('a', encoding='utf-8') as handle:
                json.dump(payload, handle, ensure_ascii=False)
                handle.write('\n')
        except Exception as error:  # pragma: no cover - 파일 시스템 오류 처리
            logger.warning(
                "메타인지 로그 기록 실패",
                extra={'path': str(log_path), 'error': str(error)},
            )

    def _calculate_average_confidence(self, session: MonitoringSession) -> Optional[float]:
        confidences = [step.confidence for step in session.reasoning_steps if step.confidence is not None]
        if not confidences:
            return None
        return sum(confidences) / len(confidences)

    async def start_reasoning_monitoring(self, task_context: Dict) -> str:
        """
        추론 과정 모니터링 시작

        Args:
            task_context: 작업 컨텍스트 정보

        Returns:
            str: 모니터링 세션 ID
        """
        try:
            # 세션 ID 생성
            session_id = generate_id('monitoring_')

            # 모니터링 세션 생성
            session = MonitoringSession(
                session_id=session_id,
                task_context=task_context,
                start_time=current_timestamp(),
                phase=MonitoringPhase.INITIALIZATION,
                metadata={
                    'complexity_score': task_context.get('complexity_score', 0),
                    'domain': task_context.get('domain', 'unknown'),
                    'expected_steps': task_context.get('expected_steps', 'unknown')
                }
            )

            # 활성 세션에 추가
            self.active_sessions[session_id] = session

            logger.info(f"추론 모니터링 시작: {session_id}")
            return session_id

        except Exception as e:
            logger.error(f"모니터링 시작 중 오류: {str(e)}")
            raise

    async def update_monitoring_phase(self, session_id: str,
                                    new_phase: MonitoringPhase) -> bool:
        """
        모니터링 단계 업데이트

        Args:
            session_id: 세션 ID
            new_phase: 새로운 단계

        Returns:
            bool: 업데이트 성공 여부
        """
        if session_id not in self.active_sessions:
            logger.warning(f"활성 세션을 찾을 수 없음: {session_id}")
            return False

        session = self.active_sessions[session_id]
        old_phase = session.phase
        session.phase = new_phase

        logger.debug(f"세션 {session_id} 단계 변경: {old_phase.value} -> {new_phase.value}")
        return True

    async def add_reasoning_step(self, session_id: str,
                               step_description: str,
                               input_data: Dict,
                               output_data: Dict,
                               processing_time_ms: float) -> str:
        """
        추론 단계 추가

        Args:
            session_id: 세션 ID
            step_description: 단계 설명
            input_data: 입력 데이터
            output_data: 출력 데이터
            processing_time_ms: 처리 시간

        Returns:
            str: 단계 ID
        """
        if session_id not in self.active_sessions:
            logger.error(f"활성 세션을 찾을 수 없음: {session_id}")
            return ""

        session = self.active_sessions[session_id]

        # 단계 ID 생성
        step_id = generate_id('step_')
        step_number = len(session.reasoning_steps) + 1

        # 단계별 품질 평가
        quality_score = await self._evaluate_step_quality(
            step_description, input_data, output_data, processing_time_ms
        )

        # 단계 생성
        step = ReasoningStep(
            step_id=step_id,
            step_number=step_number,
            description=step_description,
            input_data=input_data,
            output_data=output_data,
            processing_time_ms=processing_time_ms,
            quality_score=quality_score,
            confidence=output_data.get('confidence', 0.5)
        )

        # 세션에 추가
        session.reasoning_steps.append(step)

        logger.debug(f"추론 단계 추가: {session_id}, 단계: {step_number}, 품질: {quality_score:.2f}")
        return step_id

    async def _evaluate_step_quality(self, description: str,
                                   input_data: Dict,
                                   output_data: Dict,
                                   processing_time_ms: float) -> float:
        """단계별 품질 평가"""
        quality_score = 0.0

        # 1. 설명 명확성 (30%)
        description_score = min(1.0, len(description) / 100) * 0.8
        if any(word in description.lower() for word in ['분석', '검증', '결론']):
            description_score += 0.2
        quality_score += description_score * 0.30

        # 2. 데이터 완전성 (25%)
        data_score = 0.0
        if input_data and isinstance(input_data, dict):
            data_score += 0.5
        if output_data and isinstance(output_data, dict):
            data_score += 0.5
        quality_score += data_score * 0.25

        # 3. 처리 효율성 (20%)
        efficiency_score = 1.0
        if processing_time_ms > 1000:  # 1초 초과시 감점
            efficiency_score = max(0.0, 1.0 - (processing_time_ms - 1000) / 5000)
        quality_score += efficiency_score * 0.20

        # 4. 논리적 연결성 (25%)
        logic_score = 0.7  # 기본 점수
        if 'reasoning' in output_data:
            logic_score = 0.9
        quality_score += logic_score * 0.25

        return min(1.0, quality_score)

    async def evaluate_reasoning_quality(self, reasoning_steps: List[Dict]) -> QualityMetrics:
        """
        추론 품질 평가

        Args:
            reasoning_steps: 추론 단계 목록

        Returns:
            QualityMetrics: 품질 메트릭
        """
        if not reasoning_steps:
            return QualityMetrics()

        metrics = QualityMetrics()

        try:
            # 1. 논리적 일관성 평가
            metrics.logical_consistency = await self._evaluate_logical_consistency(reasoning_steps)

            # 2. 단계별 명확성 평가
            metrics.step_clarity = await self._evaluate_step_clarity(reasoning_steps)

            # 3. 결론 타당성 평가
            metrics.conclusion_validity = await self._evaluate_conclusion_validity(reasoning_steps)

            # 4. 완전성 평가
            metrics.completeness = await self._evaluate_completeness(reasoning_steps)

            # 5. 효율성 평가
            metrics.efficiency = await self._evaluate_efficiency(reasoning_steps)

            logger.debug(f"추론 품질 평가 완료: 전체 점수 {metrics.calculate_overall_score():.1f}")

        except Exception as e:
            logger.error(f"품질 평가 중 오류: {str(e)}")

        return metrics

    async def _evaluate_logical_consistency(self, steps: List[Dict]) -> float:
        """논리적 일관성 평가"""
        if len(steps) <= 1:
            return 0.8  # 단일 단계는 기본 점수

        consistency_score = 0.0
        valid_connections = 0

        for i in range(1, len(steps)):
            prev_step = steps[i-1]
            curr_step = steps[i]

            # 이전 단계의 출력이 현재 단계의 입력과 연결되는지 확인
            prev_output = prev_step.get('output_data', {})
            curr_input = curr_step.get('input_data', {})

            # 간단한 키 매칭으로 연결성 확인
            if self._check_data_connection(prev_output, curr_input):
                valid_connections += 1

        consistency_score = valid_connections / (len(steps) - 1) if len(steps) > 1 else 0.8
        return min(1.0, consistency_score)

    async def _evaluate_step_clarity(self, steps: List[Dict]) -> float:
        """단계별 명확성 평가"""
        if not steps:
            return 0.0

        clarity_scores = []

        for step in steps:
            description = step.get('description', '')
            clarity = 0.0

            # 설명 길이
            if len(description) >= 10:
                clarity += 0.3

            # 구체적 동사 포함
            action_words = ['분석', '평가', '계산', '검증', '결론', '추론']
            if any(word in description for word in action_words):
                clarity += 0.4

            # 명확한 목적
            if any(word in description for word in ['위해', '하기', '목적', '결과']):
                clarity += 0.3

            clarity_scores.append(min(1.0, clarity))

        return sum(clarity_scores) / len(clarity_scores)

    async def _evaluate_conclusion_validity(self, steps: List[Dict]) -> float:
        """결론 타당성 평가"""
        if not steps:
            return 0.0

        # 마지막 단계를 결론으로 간주
        final_step = steps[-1]
        output_data = final_step.get('output_data', {})

        validity = 0.0

        # 결론이 명시적으로 제시되었는지
        if 'conclusion' in output_data or 'result' in output_data:
            validity += 0.5

        # 신뢰도 정보가 있는지
        if 'confidence' in output_data:
            confidence = output_data.get('confidence', 0)
            if isinstance(confidence, (int, float)) and 0 <= confidence <= 1:
                validity += 0.3

        # 추론 근거가 있는지
        if 'reasoning' in output_data or 'evidence' in output_data:
            validity += 0.2

        return min(1.0, validity)

    async def _evaluate_completeness(self, steps: List[Dict]) -> float:
        """완전성 평가"""
        if not steps:
            return 0.0

        completeness = 0.0

        # 최소 단계 수 확인
        if len(steps) >= 2:
            completeness += 0.3
        if len(steps) >= 4:
            completeness += 0.2

        # 각 단계가 입력과 출력을 가지는지
        complete_steps = sum(1 for step in steps
                           if step.get('input_data') and step.get('output_data'))
        step_completeness = complete_steps / len(steps)
        completeness += step_completeness * 0.5

        return min(1.0, completeness)

    async def _evaluate_efficiency(self, steps: List[Dict]) -> float:
        """효율성 평가"""
        if not steps:
            return 0.0

        total_time = sum(step.get('processing_time_ms', 0) for step in steps)
        avg_time_per_step = total_time / len(steps)

        # 단계당 평균 시간이 500ms 이하면 효율적
        efficiency = 1.0
        if avg_time_per_step > 500:
            efficiency = max(0.0, 1.0 - (avg_time_per_step - 500) / 2000)

        return efficiency

    def _check_data_connection(self, prev_output: Dict, curr_input: Dict) -> bool:
        """데이터 연결성 확인"""
        if not prev_output or not curr_input:
            return False

        # 공통 키가 있는지 확인
        prev_keys = set(prev_output.keys())
        curr_keys = set(curr_input.keys())

        return len(prev_keys.intersection(curr_keys)) > 0

    async def perform_self_reflection(self, session_id: str) -> SelfReflectionResult:
        """
        자기반성 수행

        Args:
            session_id: 세션 ID

        Returns:
            SelfReflectionResult: 자기반성 결과
        """
        if session_id not in self.active_sessions:
            # 히스토리에서 찾기
            session = None
            for hist_session in self.session_history:
                if hist_session.session_id == session_id:
                    session = hist_session
                    break
            if not session:
                raise ValueError(f"세션을 찾을 수 없음: {session_id}")
        else:
            session = self.active_sessions[session_id]

        try:
            reflection_id = generate_id('reflection_')

            # 강점 분석
            strengths = await self._analyze_strengths(session)

            # 약점 분석
            weaknesses = await self._analyze_weaknesses(session)

            # 개선 제안 생성
            improvements = await self._generate_improvements(session, weaknesses)

            # 학습 포인트 추출
            learning_points = await self._extract_learning_points(session)

            # 전체 평가
            overall_assessment = await self._generate_overall_assessment(session)

            # 평가 신뢰도 계산
            confidence = self._calculate_assessment_confidence(session)

            result = SelfReflectionResult(
                reflection_id=reflection_id,
                session_id=session_id,
                strengths=strengths,
                weaknesses=weaknesses,
                improvement_suggestions=improvements,
                learning_points=learning_points,
                overall_assessment=overall_assessment,
                confidence_in_assessment=confidence
            )

            logger.info(f"자기반성 완료: {session_id}, 신뢰도: {confidence:.2f}")
            return result

        except Exception as e:
            logger.error(f"자기반성 중 오류: {str(e)}")
            raise

    async def _analyze_strengths(self, session: MonitoringSession) -> List[str]:
        """강점 분석"""
        strengths = []

        if session.quality_metrics:
            score = session.quality_metrics.calculate_overall_score()
            if score >= 80:
                strengths.append("전반적으로 높은 품질의 추론 과정")
            if session.quality_metrics.logical_consistency >= 0.8:
                strengths.append("논리적 일관성이 우수함")
            if session.quality_metrics.efficiency >= 0.8:
                strengths.append("효율적인 처리 속도")

        if len(session.reasoning_steps) >= 3:
            strengths.append("체계적인 단계별 접근")

        avg_step_quality = sum(step.quality_score for step in session.reasoning_steps) / len(session.reasoning_steps) if session.reasoning_steps else 0
        if avg_step_quality >= 0.7:
            strengths.append("단계별 품질이 일정하게 유지됨")

        return strengths if strengths else ["기본적인 추론 과정 완료"]

    async def _analyze_weaknesses(self, session: MonitoringSession) -> List[str]:
        """약점 분석"""
        weaknesses = []

        if session.quality_metrics:
            if session.quality_metrics.logical_consistency < 0.6:
                weaknesses.append("논리적 일관성 부족")
            if session.quality_metrics.step_clarity < 0.6:
                weaknesses.append("단계별 설명이 불분명")
            if session.quality_metrics.efficiency < 0.6:
                weaknesses.append("처리 속도 개선 필요")

        if len(session.reasoning_steps) < 2:
            weaknesses.append("추론 단계가 너무 단순함")

        if session.errors:
            weaknesses.append("처리 중 오류 발생")

        return weaknesses if weaknesses else []

    async def _generate_improvements(self, session: MonitoringSession,
                                   weaknesses: List[str]) -> List[str]:
        """개선 제안 생성"""
        improvements = []

        for weakness in weaknesses:
            if "논리적 일관성" in weakness:
                improvements.append("단계 간 연결고리를 더 명확히 설정")
            elif "설명이 불분명" in weakness:
                improvements.append("각 단계의 목적과 방법을 구체적으로 기술")
            elif "처리 속도" in weakness:
                improvements.append("불필요한 계산 과정 최적화")
            elif "단순함" in weakness:
                improvements.append("문제를 더 세분화하여 체계적 접근")

        if not improvements:
            improvements.append("현재 수준을 유지하며 지속적 모니터링")

        return improvements

    async def _extract_learning_points(self, session: MonitoringSession) -> List[str]:
        """학습 포인트 추출"""
        learning_points = []

        if session.reasoning_steps:
            # 가장 높은 품질의 단계 찾기
            best_step = max(session.reasoning_steps, key=lambda s: s.quality_score)
            learning_points.append(f"'{best_step.description}' 방식이 효과적이었음")

        if session.quality_metrics:
            score = session.quality_metrics.calculate_overall_score()
            if score >= 75:
                learning_points.append("이 접근 방식을 향후에도 활용")
            elif score < 60:
                learning_points.append("다른 접근 방식 시도 필요")

        complexity = session.metadata.get('complexity_score', 0)
        if complexity >= 70:
            learning_points.append("고복잡도 문제에 대한 경험 축적")

        return learning_points if learning_points else ["기본적인 추론 경험 축적"]

    async def _generate_overall_assessment(self, session: MonitoringSession) -> str:
        """전체 평가 생성"""
        if not session.quality_metrics:
            return "품질 평가 데이터 부족"

        score = session.quality_metrics.calculate_overall_score()
        grade = session.quality_metrics.get_quality_grade()

        assessments = {
            ReasoningQuality.EXCELLENT: "탁월한 추론 성능을 보여줌",
            ReasoningQuality.GOOD: "전반적으로 양호한 추론 과정",
            ReasoningQuality.MODERATE: "기본적인 요구사항은 충족하나 개선 여지 있음",
            ReasoningQuality.POOR: "상당한 개선이 필요한 상태",
            ReasoningQuality.CRITICAL: "추론 과정에 심각한 문제가 있어 즉시 개선 필요"
        }

        return f"{assessments[grade]} (점수: {score:.1f})"

    def _calculate_assessment_confidence(self, session: MonitoringSession) -> float:
        """평가 신뢰도 계산"""
        confidence = 0.5  # 기본 신뢰도

        # 충분한 데이터가 있으면 신뢰도 증가
        if len(session.reasoning_steps) >= 3:
            confidence += 0.2
        if session.quality_metrics:
            confidence += 0.2
        if session.get_duration_ms() >= 1000:  # 1초 이상
            confidence += 0.1

        return min(1.0, confidence)

    async def end_monitoring_session(self, session_id: str) -> MonitoringSession:
        """
        모니터링 세션 종료

        Args:
            session_id: 세션 ID

        Returns:
            MonitoringSession: 완료된 세션
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"활성 세션을 찾을 수 없음: {session_id}")

        session = self.active_sessions[session_id]
        session.end_time = current_timestamp()
        session.phase = MonitoringPhase.COMPLETION

        # 전체 품질 평가
        if session.reasoning_steps:
            steps_data = [
                {
                    'description': step.description,
                    'input_data': step.input_data,
                    'output_data': step.output_data,
                    'processing_time_ms': step.processing_time_ms
                }
                for step in session.reasoning_steps
            ]
            session.quality_metrics = await self.evaluate_reasoning_quality(steps_data)

        session.quality_assessment = self._assess_quality(session)
        if session.quality_assessment:
            alert_payload = self._emit_quality_alert(session, session.quality_assessment)
            if alert_payload:
                session.alerts.append(alert_payload)

        self._write_session_log(session)

        # 세션 히스토리에 추가
        self.session_history.append(session)
        if len(self.session_history) > self.max_history_size:
            self.session_history.pop(0)

        # 활성 세션에서 제거
        del self.active_sessions[session_id]

        # 성능 통계 업데이트
        self._update_performance_stats(session)

        logger.info(f"모니터링 세션 종료: {session_id}, 지속시간: {session.get_duration_ms():.0f}ms")
        return session

    def _update_performance_stats(self, session: MonitoringSession):
        """성능 통계 업데이트"""
        self.total_sessions += 1

        if session.quality_assessment:
            current_score = session.quality_assessment.score
        elif session.quality_metrics:
            current_score = session.quality_metrics.calculate_overall_score()
        else:
            current_score = 0.0

        if current_score:
            self.average_quality_score = (
                (self.average_quality_score * (self.total_sessions - 1) + current_score)
                / self.total_sessions
            )

    def get_performance_summary(self) -> Dict:
        """성능 요약 반환"""
        return {
            'total_sessions': self.total_sessions,
            'average_quality_score': self.average_quality_score,
            'active_sessions_count': len(self.active_sessions),
            'session_history_count': len(self.session_history)
        }

    def get_session_status(self, session_id: str) -> Optional[Dict]:
        """세션 상태 반환"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            return {
                'session_id': session.session_id,
                'phase': session.phase.value,
                'steps_count': len(session.reasoning_steps),
                'duration_ms': session.get_duration_ms(),
                'is_active': True
            }
        return None


# 편의성 함수들
async def create_metacognition_engine(config: Optional[Dict] = None) -> MetacognitionEngine:
    """메타인지 엔진 생성 함수"""
    return MetacognitionEngine(config)


async def monitor_reasoning_session(task_context: Dict,
                                  config: Optional[Dict] = None) -> Tuple[MetacognitionEngine, str]:
    """추론 세션 모니터링 시작"""
    engine = MetacognitionEngine(config)
    session_id = await engine.start_reasoning_monitoring(task_context)
    return engine, session_id


# 모듈 내보내기
__all__ = [
    'MetacognitionEngine',
    'QualityMetrics',
    'ReasoningStep',
    'MonitoringSession',
    'SelfReflectionResult',
    'MonitoringPhase',
    'ReasoningQuality',
    'create_metacognition_engine',
    'monitor_reasoning_session'
]
