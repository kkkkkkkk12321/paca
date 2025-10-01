"""
PACA 통합 시스템
모든 모듈을 통합하여 완전한 PACA 어시스턴트 시스템 제공
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .core.types import Result, Status, Priority, LogLevel, create_id, current_timestamp
from .core.events import EventBus, EventEmitter
from .core.errors import PacaError, CognitiveError
from .core.utils.logger import PacaLogger
from .cognitive import (
    CognitiveSystem,
    BaseCognitiveProcessor,
    CognitiveContext,
    CognitiveTaskType,
    ComplexityDetector,
    MetacognitionEngine,
    ComplexityResult,
    ComplexityLevel,
    DomainType,
    MonitoringSession,
)
from .reasoning import ReasoningEngine, ReasoningType, ReasoningResult
from .mathematics import Calculator, StatisticalAnalyzer
from .services import ServiceManager, BaseService
from .config import ConfigManager
from .data import DataManager
# LLM 모듈은 선택적으로 임포트
try:
    from .api.llm import GeminiClientManager, GeminiConfig, LLMRequest, ModelType, GenerationConfig, create_gemini_client
    from .api.llm.response_processor import ResponseProcessor, create_response_processor
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"LLM 모듈 임포트 실패: {e}")
    LLM_AVAILABLE = False
    # 더미 클래스들
    class GeminiClientManager: pass
    class GeminiConfig: pass
    class LLMRequest: pass
    class ModelType: pass
    class GenerationConfig: pass
    class ResponseProcessor: pass
    def create_gemini_client(): return None
    def create_response_processor(): return ResponseProcessor()


@dataclass
class PacaConfig:
    """PACA 시스템 설정"""
    max_response_time: float = 5.0
    enable_learning: bool = True
    enable_korean_nlp: bool = True
    log_level: LogLevel = LogLevel.INFO
    memory_limit_mb: int = 500

    # 인지 시스템 설정
    cognitive_quality_threshold: float = 0.7
    cognitive_max_processing_time: float = 3.0

    # 추론 시스템 설정
    reasoning_confidence_threshold: float = 0.7
    reasoning_max_steps: int = 10

    # 학습 설정
    learning_rate: float = 0.1
    learning_feedback_threshold: float = 0.8

    # LLM 설정
    gemini_api_keys: List[str] = field(default_factory=list)
    default_llm_model: Optional[Any] = None  # LLM 사용 가능할 때 ModelType.GEMINI_FLASH로 설정
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    enable_llm_caching: bool = True
    llm_timeout: float = 30.0
    llm_model_preferences: Dict[str, List[str]] = field(default_factory=dict)
    llm_rotation_strategy: str = "round_robin"
    llm_rotation_min_interval: float = 1.0


class Message:
    """메시지 클래스"""
    def __init__(self, content: str, sender: str = "user", timestamp: Optional[datetime] = None):
        self.id = create_id()
        self.content = content
        self.sender = sender
        self.timestamp = timestamp or datetime.now()
        self.metadata: Dict[str, Any] = {}


class PacaSystem:
    """
    PACA v5 통합 시스템
    모든 하위 시스템을 통합하여 완전한 AI 어시스턴트 기능 제공
    """

    def __init__(self, config: Optional[PacaConfig] = None):
        self.config = config or PacaConfig()
        self.logger = PacaLogger("PacaSystem")

        # 시스템 상태
        self.is_initialized = False
        self.status = Status.IDLE
        self.startup_time: Optional[datetime] = None

        # 핵심 컴포넌트
        self.event_bus = EventBus()
        self.config_manager = ConfigManager()
        self.data_storage = DataManager()

        # AI 시스템
        self.cognitive_system: Optional[CognitiveSystem] = None
        self.reasoning_engine: Optional[ReasoningEngine] = None
        self.calculator: Optional[Calculator] = None
        self.statistical_analyzer: Optional[StatisticalAnalyzer] = None

        # LLM 시스템
        self.llm_client: Optional[GeminiClientManager] = None
        self.response_processor: Optional[ResponseProcessor] = None

        # 서비스 관리
        self.service_manager: Optional[ServiceManager] = None

        # 대화 관리
        self.conversation_history: List[Message] = []
        self.user_context: Dict[str, Any] = {}

        # 인지 보조 시스템
        self.complexity_detector: Optional[ComplexityDetector] = None
        self.metacognition_engine: Optional[MetacognitionEngine] = None

        # 성능 메트릭
        self.performance_metrics: Dict[str, Any] = {
            "total_messages": 0,
            "avg_response_time": 0.0,
            "success_rate": 0.0,
            "learning_sessions": 0
        }

    async def initialize(self) -> Result[bool]:
        """시스템 초기화"""
        if self.is_initialized:
            return Result.success(True)

        try:
            self.startup_time = datetime.now()
            self.status = Status.INITIALIZING

            self.logger.info("PACA 시스템 초기화 시작")

            config_result = await self.config_manager.initialize()
            if not config_result.is_success:
                self.status = Status.ERROR
                self.logger.error("설정 관리자 초기화 실패", error=config_result.error)
                return Result.failure(config_result.error)

            self._apply_llm_config()

            storage_result = await self.data_storage.initialize()
            if not storage_result.is_success:
                self.status = Status.ERROR
                self.logger.error("데이터 저장소 초기화 실패", error=storage_result.error)
                return Result.failure(storage_result.error)

            self.cognitive_system = CognitiveSystem(processors=[])
            cognitive_init = await self.cognitive_system.initialize()
            if not cognitive_init.is_success:
                self.status = Status.ERROR
                self.logger.error("인지 시스템 초기화 실패", error=cognitive_init.error)
                return Result.failure(cognitive_init.error)

            self.reasoning_engine = ReasoningEngine()
            reasoning_init = await self.reasoning_engine.initialize()
            if not reasoning_init.is_success:
                self.status = Status.ERROR
                self.logger.error("추론 엔진 초기화 실패", error=reasoning_init.error)
                return Result.failure(reasoning_init.error)

            try:
                self.complexity_detector = ComplexityDetector()
                self.logger.debug("ComplexityDetector ready")
            except Exception as detector_error:
                self.logger.warn(
                    "Complexity detector initialization failed",
                    {"error": str(detector_error)}
                )
                self.complexity_detector = None

            try:
                self.metacognition_engine = MetacognitionEngine()
                self.logger.debug("MetacognitionEngine ready")
            except Exception as metacog_error:
                self.logger.warn(
                    "Metacognition engine initialization failed",
                    {"error": str(metacog_error)}
                )
                self.metacognition_engine = None

            self.calculator = Calculator()
            self.statistical_analyzer = StatisticalAnalyzer()

            await self._initialize_llm_system()

            self.service_manager = ServiceManager()
            service_init = await self.service_manager.initialize()
            if not service_init.is_success:
                self.status = Status.ERROR
                self.logger.error("서비스 관리자 초기화 실패", error=service_init.error)
                return Result.failure(service_init.error)

            await self._setup_event_handlers()

            self.is_initialized = True
            self.status = Status.READY
            self.logger.info("PACA 시스템 초기화 완료")
            return Result.success(True)

        except Exception as error:
            self.status = Status.ERROR
            self.logger.error("PACA 시스템 초기화 실패", error=error)
            return Result.failure(PacaError(f"PACA 시스템 초기화 실패: {str(error)}"))

    async def process_message(self, message: str, user_id: str = "default") -> Result[Dict[str, Any]]:
        """메시지 처리 및 응답 생성"""
        if not self.is_initialized:
            return Result.failure(PacaError("System not initialized"))

        start_time = time.time()

        try:
            processed_input = (message or "").strip()
            if not processed_input:
                return Result.failure(PacaError("Empty message"))

            self.status = Status.PROCESSING

            user_message = Message(processed_input, "user")
            self.conversation_history.append(user_message)

            complexity_result = None
            metacog_session_id = None
            metacog_summary = None
            reasoning_metadata = {"used": False}
            session_closed = False

            if self.complexity_detector or self.metacognition_engine:
                complexity_result, metacog_session_id = await self._prepare_cognitive_analysis(
                    processed_input,
                    user_id
                )

            response_text = None

            if (
                complexity_result
                and complexity_result.reasoning_required
                and self.reasoning_engine is not None
            ):
                reasoning_outcome = await self._handle_reasoning(
                    processed_input,
                    complexity_result,
                    metacog_session_id
                )

                if reasoning_outcome:
                    response_text = reasoning_outcome.get("response")
                    reasoning_metadata = reasoning_outcome.get("metadata", reasoning_metadata)
                    metacog_summary = reasoning_outcome.get("session_summary")
                    session_closed = reasoning_outcome.get("session_closed", False)
                    metacog_session_id = reasoning_outcome.get("session_id")

            if not response_text:
                response_text = await self._generate_basic_response(processed_input, user_id)

            if (
                self.metacognition_engine
                and metacog_session_id
                and not session_closed
            ):
                try:
                    metacog_summary = await self.metacognition_engine.end_monitoring_session(
                        metacog_session_id
                    )
                except Exception as session_error:
                    self.logger.warn("Failed to finalize metacognition session", {"error": str(session_error)})

            bot_message = Message(response_text, "assistant")
            self.conversation_history.append(bot_message)

            processing_time = time.time() - start_time
            await self._update_performance_metrics(processing_time, True)

            analysis_payload = {"reasoning": reasoning_metadata}

            if complexity_result:
                analysis_payload["complexity"] = {
                    "score": complexity_result.score,
                    "level": complexity_result.level.name,
                    "domain": complexity_result.domain.value,
                    "confidence": complexity_result.confidence,
                    "reasoning_required": complexity_result.reasoning_required,
                }

            if metacog_summary:
                analysis_payload["metacognition"] = await self._summarize_metacognition_session(
                    metacog_summary
                )

            result_data = {
                "response": response_text,
                "processing_time": processing_time,
                "confidence": reasoning_metadata.get("confidence", 0.8),
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis_payload
            }

            self.status = Status.READY
            return Result.success(result_data)

        except Exception as error:
            processing_time = time.time() - start_time
            await self._update_performance_metrics(processing_time, False)
            self.status = Status.ERROR
            self.logger.error("Message processing failed", error=error)
            return Result.failure(PacaError(f"Failed to process message: {str(error)}"))

    async def _prepare_cognitive_analysis(
        self,
        message: str,
        user_id: str
    ) -> Tuple[Optional[ComplexityResult], Optional[str]]:
        """복잡도 분석 및 메타인지 세션 준비"""

        complexity_result: Optional[ComplexityResult] = None
        session_id: Optional[str] = None

        if self.complexity_detector:
            try:
                complexity_result = await self.complexity_detector.detect_complexity(message)
            except Exception as error:
                self.logger.warn(
                    "Complexity analysis failed",
                    {"error": str(error)}
                )

        if self.metacognition_engine:
            task_context = {
                "user_id": user_id,
                "message": message,
                "complexity_score": getattr(complexity_result, "score", None),
                "complexity_level": getattr(complexity_result, "level", None).name
                if complexity_result else None,
                "domain": getattr(complexity_result, "domain", None).value
                if complexity_result else None,
                "timestamp": datetime.now().isoformat()
            }
            try:
                session_id = await self.metacognition_engine.start_reasoning_monitoring(task_context)
            except Exception as error:
                self.logger.warn(
                    "Metacognition session start failed",
                    {"error": str(error)}
                )
                session_id = None

        return complexity_result, session_id

    async def _handle_reasoning(
        self,
        message: str,
        complexity_result: ComplexityResult,
        session_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """복잡한 입력을 추론 및 메타인지와 함께 처리"""

        if not self.reasoning_engine:
            return None

        premises, target_conclusion = self._build_reasoning_premises(message, complexity_result)

        reasoning_result = await self.reasoning_engine.reason(
            ReasoningType.DEDUCTIVE,
            premises=premises,
            target_conclusion=target_conclusion
        )

        if not reasoning_result.is_success:
            self.logger.warn(
                "Reasoning engine failed",
                {"error": str(reasoning_result.error) if reasoning_result.error else "unknown"}
            )

            summary = None
            if self.metacognition_engine and session_id:
                try:
                    summary = await self.metacognition_engine.end_monitoring_session(session_id)
                    session_id = None
                except Exception as error:
                    self.logger.warn(
                        "Metacognition session finalization failed",
                        {"error": str(error)}
                    )

            return {
                "response": None,
                "metadata": {
                    "used": False,
                    "error": str(reasoning_result.error) if reasoning_result.error else None
                },
                "session_summary": summary,
                "session_closed": summary is not None,
                "session_id": session_id
            }

        reasoning_data: ReasoningResult = reasoning_result.data
        session_summary: Optional[MonitoringSession] = None
        session_closed = False

        if self.metacognition_engine and session_id:
            await self._record_metacognition_steps(session_id, reasoning_data)
            try:
                session_summary = await self.metacognition_engine.end_monitoring_session(session_id)
                session_id = None
                session_closed = True
            except Exception as error:
                self.logger.warn(
                    "Failed to end metacognition session",
                    {"error": str(error)}
                )

        response_text = self._format_reasoning_response(message, complexity_result, reasoning_data)

        reasoning_metadata = {
            "used": True,
            "conclusion": reasoning_data.conclusion,
            "confidence": reasoning_data.confidence,
            "execution_time_ms": reasoning_data.execution_time_ms,
            "steps": [
                {
                    "rule": step.inference_rule,
                    "premise": step.premise,
                    "confidence": step.confidence
                }
                for step in reasoning_data.reasoning_steps
            ]
        }

        return {
            "response": response_text,
            "metadata": reasoning_metadata,
            "session_summary": session_summary,
            "session_closed": session_closed,
            "session_id": session_id
        }

    def _build_reasoning_premises(
        self,
        message: str,
        complexity_result: ComplexityResult
    ) -> Tuple[List[str], str]:
        """추론 엔진 입력을 위한 전제 구성"""

        domain_phrase = complexity_result.domain.value.replace('_', ' ')
        condition = f"user needs help with {domain_phrase}"
        conclusion = f"provide detailed guidance about {domain_phrase}"

        premises = [
            f"If {condition} then {conclusion}",
            condition,
            f"User message: {message}"
        ]

        return premises, conclusion

    async def _record_metacognition_steps(
        self,
        session_id: str,
        reasoning_data: ReasoningResult
    ) -> None:
        """추론 단계를 메타인지 엔진에 기록"""

        if not self.metacognition_engine:
            return

        steps = reasoning_data.reasoning_steps or []
        step_count = len(steps) or 1
        avg_time = reasoning_data.execution_time_ms / step_count

        for index, step in enumerate(steps, start=1):
            try:
                await self.metacognition_engine.add_reasoning_step(
                    session_id=session_id,
                    step_description=f"{index}. {step.inference_rule.replace('_', ' ')}",
                    input_data={
                        "premise": step.premise,
                        "evidence": getattr(step, 'evidence', None)
                    },
                    output_data={
                        "conclusion": step.conclusion,
                        "confidence": step.confidence
                    },
                    processing_time_ms=avg_time
                )
            except Exception as error:
                self.logger.warn(
                    "Failed to record metacognition step",
                    {"error": str(error)}
                )

    def _format_reasoning_response(
        self,
        message: str,
        complexity_result: ComplexityResult,
        reasoning_data: ReasoningResult
    ) -> str:
        """추론 결과를 기반으로 사용자 응답 생성"""

        domain_messages = {
            DomainType.TECHNICAL: "기술적인 질문으로 판단되어 단계별 안내를 준비했어요.",
            DomainType.MATHEMATICAL: "수학적 사고가 필요한 질문이네요. 핵심 단계를 정리해볼게요.",
            DomainType.LOGICAL: "논리적 추론이 필요한 질문이군요. 전제를 하나씩 확인해보겠습니다.",
            DomainType.ANALYTICAL: "분석 중심 질문이라 구조화된 답변이 필요해 보여요.",
        }

        base_message = domain_messages.get(
            complexity_result.domain,
            "심화 설명이 필요한 질문으로 감지되었습니다."
        )

        detail = reasoning_data.conclusion or "자세한 설명이 필요합니다"
        level = complexity_result.level.name.replace('_', ' ').title()

        return (
            f"{base_message}\n"
            f"- 복잡도 수준: {level} (점수 {complexity_result.score})\n"
            f"- 추천 대응: {detail}\n"
            "필요하다면 추가로 세부 단계를 안내드릴게요."
        )

    async def _summarize_metacognition_session(
        self,
        session: MonitoringSession
    ) -> Dict[str, Any]:
        """메타인지 세션 요약 정보 생성"""

        summary = {
            "duration_ms": session.get_duration_ms(),
            "phase": session.phase.value,
            "session_id": session.session_id
        }

        if session.quality_assessment:
            assessment = session.quality_assessment
            summary["quality_score"] = assessment.score
            summary["quality_grade"] = assessment.grade.name
            summary["quality_level"] = assessment.level.value
            summary["quality_alerts"] = assessment.alerts
            summary["recommended_actions"] = assessment.recommended_actions
            summary["average_confidence"] = assessment.average_confidence
        elif session.quality_metrics:
            quality_score = session.quality_metrics.calculate_overall_score()
            summary["quality_score"] = quality_score
            summary["quality_grade"] = session.quality_metrics.get_quality_grade().name
            summary["quality_level"] = None
            summary["quality_alerts"] = []
            summary["recommended_actions"] = []
            summary["average_confidence"] = None
        else:
            summary["quality_score"] = None
            summary["quality_grade"] = None
            summary["quality_level"] = None
            summary["quality_alerts"] = []
            summary["recommended_actions"] = []
            summary["average_confidence"] = None

        if self.metacognition_engine:
            try:
                reflection = await self.metacognition_engine.perform_self_reflection(session.session_id)
                summary["reflection"] = {
                    "strengths": reflection.strengths[:2],
                    "weaknesses": reflection.weaknesses[:2] if reflection.weaknesses else [],
                    "assessment": reflection.overall_assessment,
                }
            except Exception as error:
                self.logger.warn(
                    "Self reflection generation failed",
                    {"error": str(error)}
                )

        return summary

    async def _generate_basic_response(self, message: str, user_id: str) -> str:
        """기본 응답: LLM 우선 + 429(쿼터초과) 재시도/명확 안내"""
        import asyncio
        import re

        attempts = 3
        backoff = 0.6  # seconds

        for i in range(1, attempts + 1):
            try:
                # 프로젝트의 표준 LLM 파이프라인을 그대로 사용
                return await self._generate_llm_response(
                    cognitive_data={},                 # 최소 컨텍스트
                    original_message=message,
                    context={"user_id": user_id}
                )
            except Exception as e:
                s = str(e)
                is_429 = (
                    "RESOURCE_EXHAUSTED" in s
                    or "429" in s
                    or re.search(r"\brate[- ]?limit\b", s, re.IGNORECASE)
                )

                # 429면 짧게 재시도
                if is_429 and i < attempts:
                    await asyncio.sleep(backoff)
                    backoff *= 1.8
                    continue

                # 재시도 후에도 429면 쿼터 초과를 명확히 안내
                if is_429:
                    return (
                        "지금 LLM 제공자 쿼터(요청 제한)를 초과해 응답을 만들 수 없어요. "
                        "잠시 뒤 다시 시도하거나, 다른 모델/키로 전환해주세요."
                    )

                # 그 외 에러는 규칙 기반 분석으로 대응
                return self._generate_fallback_response(
                    {
                        "failure_reason": s or e.__class__.__name__,
                        "confidence": 0.35,
                    },
                    message,
                )



    async def _apply_learning_feedback(
        self,
        user_message: str,
        bot_response: str,
        cognitive_data: Dict[str, Any]
    ) -> bool:
        """학습 피드백 적용"""
        if not self.config.enable_learning:
            return False

        try:
            # 기본적인 학습 피드백 로직
            # 실제로는 더 복잡한 강화학습 시스템 필요

            confidence = cognitive_data.get("confidence", 0.5)

            if confidence > self.config.learning_feedback_threshold:
                # 성공적인 상호작용 학습
                self.user_context["successful_interactions"] = \
                    self.user_context.get("successful_interactions", 0) + 1

                self.performance_metrics["learning_sessions"] += 1
                return True

            return False

        except Exception as e:
            self.logger.error(f"학습 피드백 적용 오류: {str(e)}", error=e)
            return False

    async def _update_performance_metrics(self, processing_time: float, success: bool):
        """성능 메트릭 업데이트"""
        self.performance_metrics["total_messages"] += 1

        # 평균 응답 시간 계산
        total_messages = self.performance_metrics["total_messages"]
        current_avg = self.performance_metrics["avg_response_time"]
        new_avg = (current_avg * (total_messages - 1) + processing_time) / total_messages
        self.performance_metrics["avg_response_time"] = new_avg

        # 성공률 계산
        if success:
            current_success_count = self.performance_metrics["success_rate"] * (total_messages - 1)
            new_success_rate = (current_success_count + 1) / total_messages
            self.performance_metrics["success_rate"] = new_success_rate
        else:
            current_success_count = self.performance_metrics["success_rate"] * (total_messages - 1)
            new_success_rate = current_success_count / total_messages
            self.performance_metrics["success_rate"] = new_success_rate

    def _apply_llm_config(self) -> None:
        """설정 관리자에서 LLM 관련 설정 반영"""
        config_data = self.config_manager.get_config("default") or {}
        llm_settings = config_data.get("llm") if isinstance(config_data, dict) else {}

        if not isinstance(llm_settings, dict):
            return

        api_keys = llm_settings.get("api_keys", [])
        if isinstance(api_keys, list):
            cleaned_keys = [
                key.strip() for key in api_keys if isinstance(key, str) and key.strip()
            ]
            if cleaned_keys:
                self.config.gemini_api_keys = cleaned_keys

        models = llm_settings.get("models", {})
        if isinstance(models, dict):
            normalized_models: Dict[str, List[str]] = {}
            for key, value in models.items():
                if isinstance(value, list):
                    normalized_models[key] = [
                        str(item).strip() for item in value if str(item).strip()
                    ]
                elif value:
                    normalized_models[key] = [str(value).strip()]

            if normalized_models:
                self.config.llm_model_preferences = normalized_models
                if not self.config.default_llm_model:
                    conversation_models = (
                        normalized_models.get("conversation")
                        or normalized_models.get("conversation_priority")
                        or []
                    )
                    if conversation_models:
                        self.config.default_llm_model = self._resolve_model_type(conversation_models[0])

        rotation = llm_settings.get("rotation", {})
        if isinstance(rotation, dict):
            strategy = rotation.get("strategy")
            if isinstance(strategy, str) and strategy.strip():
                self.config.llm_rotation_strategy = strategy.strip()

            interval_value = rotation.get("min_interval_seconds")
            if interval_value is not None:
                try:
                    interval_float = float(interval_value)
                    if interval_float >= 0:
                        self.config.llm_rotation_min_interval = interval_float
                except (TypeError, ValueError):
                    self.logger.warn(
                        "LLM 키 로테이션 간격 값을 실수로 변환하지 못했습니다",
                        value=interval_value
                    )

        if self.config.default_llm_model:
            self.config.default_llm_model = self._resolve_model_type(self.config.default_llm_model)

    def _resolve_model_type(self, model: Optional[Any]) -> ModelType:
        """문자열 또는 ModelType을 안전하게 변환"""
        if isinstance(model, ModelType):
            return model

        if isinstance(model, str):
            try:
                return ModelType(model)
            except ValueError:
                self.logger.warn(
                    "알 수 없는 LLM 모델이 지정되어 기본 모델을 사용합니다",
                    model=model
                )

        return ModelType.GEMINI_FLASH

    async def _initialize_llm_system(self) -> None:
        """LLM 시스템 초기화"""
        if not LLM_AVAILABLE:
            self.logger.warn("LLM 모듈을 사용할 수 없습니다. 기본 응답 모드로 실행됩니다.")
            self.llm_client = None
            self.response_processor = create_response_processor()
            return

        try:
            # 기본 모델 설정
            default_model = self.config.default_llm_model
            if default_model is None:
                conversation_models = (
                    self.config.llm_model_preferences.get("conversation")
                    or self.config.llm_model_preferences.get("conversation_priority")
                    or []
                )
                if conversation_models:
                    default_model = conversation_models[0]

            default_model = self._resolve_model_type(default_model)

            # Gemini 클라이언트 설정
            gemini_config = GeminiConfig(
                api_keys=self.config.gemini_api_keys or [],
                default_model=default_model,
                timeout=self.config.llm_timeout,
                enable_caching=self.config.enable_llm_caching,
                generation_config=GenerationConfig(
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens
                ),
                rotation_strategy=self.config.llm_rotation_strategy,
                rotation_min_interval=self.config.llm_rotation_min_interval,
                model_preferences=self.config.llm_model_preferences
            )

            # 클라이언트 생성 및 초기화
            self.llm_client = GeminiClientManager(gemini_config)
            init_result = await self.llm_client.initialize()

            if not init_result.is_success:
                self.logger.warn(f"LLM 클라이언트 초기화 실패: {init_result.error}")
                self.logger.info("LLM 없이 시스템을 계속 실행합니다")
                self.llm_client = None
            else:
                self.logger.info("LLM 시스템이 성공적으로 초기화되었습니다")

            # 응답 처리기 초기화
            self.response_processor = create_response_processor()

        except Exception as e:
            self.logger.error(f"LLM 시스템 초기화 오류: {str(e)}", error=e)
            self.llm_client = None
            self.response_processor = create_response_processor()

    async def update_llm_api_keys(self, new_keys: List[str], persist: bool = True) -> Result[bool]:
        """LLM API 키 목록을 갱신"""
        cleaned_keys: List[str] = []
        for key in new_keys or []:
            sanitized = (key or "").strip()
            if sanitized and sanitized not in cleaned_keys:
                cleaned_keys.append(sanitized)

        if not cleaned_keys:
            return Result.failure(PacaError("No valid API keys provided"))

        self.config.gemini_api_keys = cleaned_keys

        if self.llm_client:
            self.llm_client.update_api_keys(cleaned_keys)

        if persist:
            set_result = self.config_manager.set_value("default", "llm.api_keys", cleaned_keys)
            if not set_result.is_success:
                return Result.failure(set_result.error)

        self.logger.info(f"LLM API 키 갱신 완료 (총 {len(cleaned_keys)}개)")
        return Result.success(True)

    async def add_llm_api_key(self, api_key: str, persist: bool = True) -> Result[bool]:
        """LLM API 키 추가"""
        sanitized = (api_key or "").strip()
        if not sanitized:
            return Result.failure(PacaError("Invalid API key provided"))

        if sanitized in self.config.gemini_api_keys:
            return Result.success(True)

        self.config.gemini_api_keys.append(sanitized)

        if self.llm_client:
            self.llm_client.add_api_keys([sanitized])

        if persist:
            set_result = self.config_manager.set_value("default", "llm.api_keys", self.config.gemini_api_keys)
            if not set_result.is_success:
                return Result.failure(set_result.error)

        self.logger.info(f"LLM API 키 추가 완료 (총 {len(self.config.gemini_api_keys)}개)")
        return Result.success(True)

    async def remove_llm_api_key(self, api_key: str, persist: bool = True) -> Result[bool]:
        """LLM API 키 제거"""
        sanitized = (api_key or "").strip()
        if not sanitized or sanitized not in self.config.gemini_api_keys:
            return Result.success(True)

        self.config.gemini_api_keys = [
            key for key in self.config.gemini_api_keys if key != sanitized
        ]

        if self.llm_client:
            self.llm_client.remove_api_key(sanitized)

        if persist:
            set_result = self.config_manager.set_value("default", "llm.api_keys", self.config.gemini_api_keys)
            if not set_result.is_success:
                return Result.failure(set_result.error)

        self.logger.info(f"LLM API 키 제거 완료 (총 {len(self.config.gemini_api_keys)}개)")
        return Result.success(True)

    async def _generate_llm_response(
        self,
        cognitive_data: Dict[str, Any],
        original_message: str,
        context: CognitiveContext
    ) -> str:
        """LLM을 통한 응답 생성"""
        try:
            if not self.llm_client:
                # LLM이 없으면 기본 응답 생성
                return self._generate_fallback_response(cognitive_data, original_message)

            # 컨텍스트 생성
            request_context = await self.response_processor.get_context_for_next_request(original_message)

            # 시스템 프롬프트 구성
            system_prompt = self._build_system_prompt(cognitive_data, request_context)

            # 사용자 프롬프트 구성
            user_prompt = self._build_user_prompt(original_message, cognitive_data)

            # LLM 요청 생성
            llm_request = LLMRequest(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=self.config.default_llm_model,
                config=GenerationConfig(
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens
                ),
                context=request_context
            )

            # LLM 호출
            result = await self.llm_client.generate_text(llm_request)

            if result.is_success:
                # 응답 후처리
                processed_result = await self.response_processor.process_response(
                    result.data,
                    original_message,
                    request_context
                )

                if processed_result.is_success:
                    return processed_result.data[0].text
                else:
                    self.logger.warn(f"응답 처리 실패: {processed_result.error}")
                    return result.data.text

            else:
                err = str(result.error or "")
                # 429 / RATE LIMIT은 예외로 올려서 basic_response의 재시도가 작동하게 한다
                if "RESOURCE_EXHAUSTED" in err or "429" in err or "rate limit" in err.lower():
                    raise RuntimeError(f"LLM_RATE_LIMIT: {err}")
                self.logger.error(f"LLM 응답 생성 실패: {result.error}")
                return self._generate_fallback_response(cognitive_data, original_message)


        except Exception as e:
            s = str(e)
            if "RESOURCE_EXHAUSTED" in s or "429" in s or "rate limit" in s.lower():
                # 429는 위로 던져서 _generate_basic_response의 재시도/안내 로직이 처리하게 함
                raise
            self.logger.error(f"LLM 응답 생성 중 오류: {s}", error=e)
            return self._generate_fallback_response(cognitive_data, original_message)


    def _build_system_prompt(self, cognitive_data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """시스템 프롬프트 구성"""
        system_prompt = """당신은 PACA(Personal Adaptive Cognitive Assistant) v5 AI 어시스턴트입니다.

핵심 특성:
- 지적 정직성을 최우선으로 합니다
- 불확실한 것은 솔직히 모른다고 말합니다
- 사용자의 학습과 성장을 돕습니다
- 단계적이고 논리적인 사고를 합니다

응답 원칙:
1. 정확하고 도움이 되는 정보 제공
2. 불확실할 때는 명확히 표현
3. 사용자의 맥락과 의도 이해
4. 간결하면서도 완전한 답변
5. 한국어로 자연스럽게 대화

"""

        # 인지 데이터에 따른 추가 지침
        confidence = cognitive_data.get("confidence", 0.5)
        if confidence < 0.5:
            system_prompt += "\n현재 사용자의 요청이 다소 불명확하니, 명확화를 위한 질문을 포함하세요."

        # 컨텍스트 정보 추가
        if context.get("context_summary"):
            system_prompt += f"\n\n이전 대화 맥락: {context['context_summary']}"

        return system_prompt

    def _build_user_prompt(self, message: str, cognitive_data: Dict[str, Any]) -> str:
        """사용자 프롬프트 구성"""
        complexity_level = cognitive_data.get("complexity_level", "medium")

        prompt = f"사용자 요청: {message}\n"

        # 복잡도에 따른 추가 정보
        if complexity_level == "high":
            prompt += "\n(복잡한 요청으로 분석됨 - 단계적 접근이 필요할 수 있습니다)"
        elif complexity_level == "low":
            prompt += "\n(간단한 요청으로 분석됨 - 직접적인 답변이 적절합니다)"

        return prompt

    def _generate_fallback_response(self, cognitive_data: Dict[str, Any], original_message: str) -> str:
        """LLM 실패 시 규칙 기반 분석으로 응답 생성"""
        import re
        from collections import Counter

        message = (original_message or "").strip()
        if not message:
            return "내부 분석을 수행할 수 있는 정보가 부족해요. 요청을 조금 더 구체적으로 설명해 주세요."

        normalized = message.lower()

        # 간단한 인사/감사/학습 요청 등 반복 패턴은 친근한 템플릿으로 즉시 응답한다.
        if any(keyword in normalized for keyword in ["안녕", "안뇽", "hello", "hi"]):
            return "안녕하세요! PACA입니다. 오늘은 어떤 도움을 드릴까요?"

        if any(keyword in normalized for keyword in ["고마워", "감사", "thank"]):
            return "천만에요! 도움이 필요하시면 언제든지 말씀해 주세요."

        if any(keyword in normalized for keyword in ["공부", "학습", "도와줘", "도와 주", "help"]):
            return "학습에 대해 무엇이 궁금하신가요? 목표나 현재 수준을 알려주시면 맞춤형으로 도와드릴게요."

        sentences = [s.strip() for s in re.split(r"(?<=[.!?\?\n])\s+", message) if s.strip()]
        summary = sentences[0] if sentences else message[:120]
        if len(summary) > 120:
            summary = summary[:117] + "..."

        tokens = re.findall(r"[가-힣a-zA-Z0-9]{2,}", message)
        stopwords = {
            "그리고", "하지만", "그러나", "그래서", "이어서", "그리고", "그러면", "그러니까",
            "하면", "하면요", "위해", "대한", "어떤", "어떻게", "무엇", "어디", "어느",
            "정도", "조금", "이번", "관련", "사용", "가능", "필요", "있는", "없는",
            "합니다", "하세요", "해주세요", "있나요", "있을까요", "해주세요", "같아요",
        }
        normalized_tokens = [token.lower() for token in tokens if token.lower() not in stopwords]
        keywords = [token for token, _ in Counter(normalized_tokens).most_common(5)]

        question_detected = bool(re.search(r"[?？]$", message) or re.search(r"\b(why|how|what|when|who|where)\b", message, re.IGNORECASE) or re.search(r"(왜|어떻게|무엇|몇|어디|누가)", message))
        task_detected = bool(re.search(r"(만들|구현|설계|작성|코드|빌드|제작)", message))
        issue_detected = bool(re.search(r"(문제|오류|에러|버그|고장|실패)", message))

        intents: List[str] = []
        if question_detected:
            intents.append("정보/질문")
        if task_detected:
            intents.append("실행/구현")
        if issue_detected:
            intents.append("문제 해결")
        if not intents:
            intents.append("일반 대화")

        base_confidence = cognitive_data.get("confidence")
        if base_confidence is None:
            base_confidence = 0.45 + min(len(keywords) * 0.05, 0.2)
            if question_detected:
                base_confidence += 0.05
            if issue_detected:
                base_confidence += 0.05
            base_confidence = max(0.35, min(base_confidence, 0.85))

        analysis_lines: List[str] = []
        analysis_lines.append(f"의도 분류: {', '.join(intents)}")
        if keywords:
            analysis_lines.append(f"핵심 키워드: {', '.join(keywords)}")

        if len(message) > 200:
            analysis_lines.append(f"요청 길이: {len(message)}자 (정보량이 많음)")
        elif len(message) < 40:
            analysis_lines.append("요청 길이: 짧음 (추가 정보 필요 가능)")

        recent_context = []
        if getattr(self, "conversation_history", None):
            user_turns = [
                m.content for m in reversed(self.conversation_history)
                if getattr(m, "sender", "user") == "user"
            ]
            if user_turns:
                # 최근 현재 메시지를 제외한 이전 사용자 메시지들을 참조
                previous_messages = [turn for turn in user_turns[1:4] if turn != message]
                if previous_messages:
                    condensed = " | ".join(prev[:40] + ("..." if len(prev) > 40 else "") for prev in reversed(previous_messages))
                    recent_context.append(f"이전 흐름: {condensed}")

        plan_steps: List[str] = []
        plan_steps.append("요청 의도를 명확히 정리하고 필요한 가정을 세웁니다.")
        if question_detected:
            plan_steps.append("질문에 답하기 위해 필요한 정보/근거 목록을 작성합니다.")
        if task_detected:
            plan_steps.append("실행 가능한 절차나 코드를 단계별로 설계합니다.")
        if issue_detected:
            plan_steps.append("문제 재현 조건과 해결 전략을 점검합니다.")
        plan_steps.append("부족한 정보는 후속 질문으로 확보하고, 대화 로그에 학습 메모를 남깁니다.")

        closing: str
        if base_confidence >= 0.7:
            closing = "현재 규칙 기반 추정 신뢰도는 높지만, 세부 확인을 위해 추가 설명을 부탁드릴 수 있습니다."
        elif base_confidence >= 0.5:
            closing = "규칙 기반 추정 신뢰도는 중간 수준입니다. 중요한 세부사항이 있다면 공유해 주세요."
        else:
            closing = "규칙 기반 추정 신뢰도가 낮아 추가 맥락이 필요합니다. 더 구체적으로 알려주시면 학습에 도움이 됩니다."

        failure_reason = cognitive_data.get("failure_reason")

        observation = {
            "message": message[:200],
            "keywords": keywords,
            "confidence": round(base_confidence, 2),
            "intents": intents,
            "failure_reason": failure_reason,
            "timestamp": datetime.now().isoformat(),
        }
        fallback_memory = self.user_context.setdefault("fallback_observations", [])
        fallback_memory.append(observation)
        if len(fallback_memory) > 25:
            del fallback_memory[0]

        response_lines: List[str] = [
            "LLM 엔진 호출이 실패해 PACA의 규칙 기반 분석으로 응답드립니다.",
            f"요청 요약: {summary}",
        ]

        if recent_context:
            response_lines.extend(recent_context)

        if analysis_lines:
            response_lines.append("상황 해석:")
            response_lines.extend(f"- {line}" for line in analysis_lines)

        if plan_steps:
            response_lines.append("대응 계획:")
            response_lines.extend(f"{index}. {step}" for index, step in enumerate(plan_steps, start=1))

        response_lines.append(closing)

        if failure_reason:
            response_lines.append(f"(참고: LLM 오류로 자동 대체됨 - {failure_reason})")

        response_lines.append("대화를 이어가면 학습 메모를 바탕으로 더 나은 답변을 준비하겠습니다.")

        return "\n".join(response_lines)

    async def _setup_event_handlers(self):
        """이벤트 핸들러 설정"""
        # 시스템 간 이벤트 연결 설정
        pass

    async def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 정보 반환"""
        uptime = None
        if self.startup_time:
            uptime = (datetime.now() - self.startup_time).total_seconds()

        return {
            "status": self.status.value,
            "is_initialized": self.is_initialized,
            "uptime_seconds": uptime,
            "conversation_count": len(self.conversation_history),
            "performance_metrics": self.performance_metrics.copy(),
            "memory_usage": await self._get_memory_usage(),
            "version": "5.0.0"
        }

    async def _get_memory_usage(self) -> Dict[str, Any]:
        """메모리 사용량 정보"""
        # 기본적인 메모리 정보 (실제로는 psutil 등 사용)
        return {
            "conversation_history_size": len(self.conversation_history),
            "user_context_size": len(str(self.user_context)),
            "estimated_mb": len(str(self.conversation_history)) / 1024 / 1024
        }

    async def clear_conversation(self):
        """대화 히스토리 초기화"""
        self.conversation_history.clear()
        self.logger.info("대화 히스토리가 초기화되었습니다")

    async def cleanup(self):
        """시스템 정리"""
        try:
            self.status = Status.SHUTTING_DOWN

            if self.service_manager:
                await self.service_manager.shutdown()

            if self.cognitive_system:
                await self.cognitive_system.cleanup()

            if self.llm_client:
                await self.llm_client.cleanup()

            if self.response_processor:
                await self.response_processor.cleanup()

            if self.data_storage:
                await self.data_storage.cleanup()

            self.logger.info("PACA 시스템 정리 완료")
            self.status = Status.SHUTDOWN

        except Exception as e:
            self.logger.error(f"시스템 정리 중 오류: {str(e)}", error=e)
