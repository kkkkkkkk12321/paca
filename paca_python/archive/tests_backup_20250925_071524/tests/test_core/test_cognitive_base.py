"""
Test cases for cognitive base module
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Any, Dict, List

from paca.cognitive.base import (
    CognitiveState,
    CognitiveTaskType,
    QualityMetrics,
    CognitiveModel,
    MemorySystem,
    CognitiveConfig,
    CognitiveContext,
    ProcessingStep,
    MemoryUpdate,
    LearningData,
    ResourceUsage,
    CognitiveResult,
    CognitiveStatistics,
    BaseCognitiveProcessor,
    CognitiveSystem,
    create_cognitive_context,
    create_processing_step
)
from paca.core.types.base import create_id, current_timestamp
from paca.core.errors.cognitive import CognitiveError, MemoryError, ModelError


class TestCognitiveProcessor(BaseCognitiveProcessor):
    """테스트용 인지 프로세서 구현"""

    async def perform_cognitive_processing(
        self,
        context: CognitiveContext,
        relevant_memory: Any
    ) -> CognitiveResult:
        """테스트용 간단한 인지 처리"""
        start_time = current_timestamp()

        # 간단한 처리 시뮬레이션
        await asyncio.sleep(0.01)  # 10ms 처리 시간

        # 처리 단계 생성
        steps = [
            create_processing_step(
                name="test_step",
                step_type="processing",
                input_data=context.input,
                output_data=f"processed_{context.input}",
                confidence=0.85
            )
        ]

        # 메모리 업데이트
        memory_updates = [
            MemoryUpdate(
                id=create_id(),
                type="store",
                target="test_memory",
                data=context.input,
                timestamp=current_timestamp(),
                confidence=0.8
            )
        ]

        # 학습 데이터
        learning_data = LearningData(
            id=create_id(),
            type="supervised",
            input=context.input,
            expected_output=None,
            actual_output=f"processed_{context.input}",
            feedback=None,
            importance=0.5,
            timestamp=current_timestamp()
        )

        # 리소스 사용량
        resource_usage = ResourceUsage(
            memory_mb=10.0,
            cpu_percent=5.0,
            processing_time_ms=10.0,
            api_calls=1,
            cache_hits=0,
            cache_misses=1
        )

        # 품질 메트릭
        quality_metrics = QualityMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            consistency=0.85,
            completeness=0.90,
            efficiency=0.80,
            reliability=0.85,
            coherence=0.85,
            relevance=0.90,
            confidence=0.85
        )

        return CognitiveResult(
            context_id=context.id,
            output=f"processed_{context.input}",
            confidence=0.85,
            quality_metrics=quality_metrics,
            processing_steps=steps,
            memory_updates=memory_updates,
            learning_data=learning_data,
            processing_time_ms=10.0,
            resource_usage=resource_usage
        )


@pytest.fixture
def cognitive_config():
    """테스트용 인지 설정"""
    return CognitiveConfig(
        id=create_id(),
        name="test_processor",
        description="Test cognitive processor",
        model=CognitiveModel(
            name="test_model",
            version="1.0.0",
            type="test",
            parameters={"param1": "value1"}
        ),
        memory=MemorySystem(
            type="test_memory",
            capacity=1000,
            persistence=True,
            configuration={"cache_size": 100}
        ),
        quality_threshold=0.7,
        max_processing_time=5000.0,
        enable_learning=True,
        enable_memory_persistence=True,
        debug_mode=False
    )


@pytest.fixture
def test_processor(cognitive_config):
    """테스트용 인지 프로세서"""
    return TestCognitiveProcessor(cognitive_config)


@pytest.fixture
def test_context():
    """테스트용 인지 컨텍스트"""
    return create_cognitive_context(
        task_type=CognitiveTaskType.REASONING,
        input_data="test_input",
        metadata={"test": "metadata"},
        quality_requirements=QualityMetrics(
            accuracy=0.8,
            completeness=0.7,
            relevance=0.8
        )
    )


class TestQualityMetrics:
    """QualityMetrics 테스트"""

    def test_quality_metrics_initialization(self):
        """품질 메트릭 초기화 테스트"""
        metrics = QualityMetrics()

        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.consistency == 0.0
        assert metrics.completeness == 0.0
        assert metrics.efficiency == 0.0
        assert metrics.reliability == 0.0
        assert metrics.coherence == 0.0
        assert metrics.relevance == 0.0
        assert metrics.confidence == 0.0

    def test_quality_metrics_with_values(self):
        """값이 있는 품질 메트릭 테스트"""
        metrics = QualityMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75
        )

        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.80
        assert metrics.recall == 0.75


class TestCognitiveContext:
    """CognitiveContext 테스트"""

    def test_create_cognitive_context(self):
        """인지 컨텍스트 생성 테스트"""
        context = create_cognitive_context(
            task_type=CognitiveTaskType.LEARNING,
            input_data="test_input"
        )

        assert context.task_type == CognitiveTaskType.LEARNING
        assert context.input == "test_input"
        assert context.metadata == {}
        assert context.quality_requirements is None
        assert context.constraints is None

    def test_create_cognitive_context_with_metadata(self):
        """메타데이터가 있는 인지 컨텍스트 생성 테스트"""
        metadata = {"key": "value", "number": 42}
        quality_req = QualityMetrics(accuracy=0.9)

        context = create_cognitive_context(
            task_type=CognitiveTaskType.ANALYSIS,
            input_data="test_input",
            metadata=metadata,
            quality_requirements=quality_req
        )

        assert context.task_type == CognitiveTaskType.ANALYSIS
        assert context.input == "test_input"
        assert context.metadata == metadata
        assert context.quality_requirements == quality_req


class TestProcessingStep:
    """ProcessingStep 테스트"""

    def test_create_processing_step(self):
        """처리 단계 생성 테스트"""
        step = create_processing_step(
            name="test_step",
            step_type="processing",
            input_data="input",
            output_data="output",
            confidence=0.9
        )

        assert step.name == "test_step"
        assert step.type == "processing"
        assert step.input == "input"
        assert step.output == "output"
        assert step.confidence == 0.9
        assert step.metadata == {}


class TestBaseCognitiveProcessor:
    """BaseCognitiveProcessor 테스트"""

    @pytest.mark.asyncio
    async def test_processor_initialization(self, cognitive_config):
        """프로세서 초기화 테스트"""
        processor = TestCognitiveProcessor(cognitive_config)

        assert processor.config == cognitive_config
        assert processor.state == CognitiveState.IDLE
        assert processor.logger is not None
        assert processor.events is None

    @pytest.mark.asyncio
    async def test_successful_processing(self, test_processor, test_context):
        """성공적인 처리 테스트"""
        result = await test_processor.process(test_context)

        assert result.context_id == test_context.id
        assert result.output == f"processed_{test_context.input}"
        assert result.confidence == 0.85
        assert len(result.processing_steps) == 1
        assert len(result.memory_updates) == 1
        assert result.learning_data is not None
        assert test_processor.state == CognitiveState.COMPLETED

    @pytest.mark.asyncio
    async def test_processing_with_invalid_input(self, test_processor):
        """잘못된 입력으로 처리 테스트"""
        context = CognitiveContext(
            id=create_id(),
            task_type=CognitiveTaskType.REASONING,
            timestamp=current_timestamp(),
            input=None,  # 잘못된 입력
            metadata={}
        )

        with pytest.raises(ModelError):
            await test_processor.process(context)

    @pytest.mark.asyncio
    async def test_processing_statistics(self, test_processor, test_context):
        """처리 통계 테스트"""
        # 초기 통계
        stats = test_processor.get_statistics()
        assert stats.process_count == 0
        assert stats.success_count == 0
        assert stats.error_count == 0

        # 성공적인 처리
        await test_processor.process(test_context)

        stats = test_processor.get_statistics()
        assert stats.process_count == 1
        assert stats.success_count == 1
        assert stats.error_count == 0
        assert stats.success_rate == 1.0
        assert stats.current_state == CognitiveState.COMPLETED

    @pytest.mark.asyncio
    async def test_quality_validation_failure(self, cognitive_config):
        """품질 검증 실패 테스트"""
        # 높은 품질 요구사항 설정
        cognitive_config.quality_threshold = 0.95
        processor = TestCognitiveProcessor(cognitive_config)

        context = create_cognitive_context(
            task_type=CognitiveTaskType.REASONING,
            input_data="test_input",
            quality_requirements=QualityMetrics(
                coherence=0.99,  # 높은 요구사항
                completeness=0.99,
                relevance=0.99
            )
        )

        with pytest.raises(CognitiveError):
            await processor.process(context)


class TestCognitiveSystem:
    """CognitiveSystem 테스트"""

    @pytest.fixture
    def cognitive_system(self, test_processor):
        """테스트용 인지 시스템"""
        return CognitiveSystem([test_processor])

    @pytest.mark.asyncio
    async def test_system_initialization(self, cognitive_system, test_processor):
        """시스템 초기화 테스트"""
        assert len(cognitive_system.processors) == 1
        assert test_processor.config.id in cognitive_system.processors
        assert cognitive_system.get_processor(test_processor.config.id) == test_processor

    @pytest.mark.asyncio
    async def test_system_processing(self, cognitive_system, test_processor, test_context):
        """시스템 처리 테스트"""
        result = await cognitive_system.process(test_processor.config.id, test_context)

        assert result.context_id == test_context.id
        assert result.output == f"processed_{test_context.input}"

    @pytest.mark.asyncio
    async def test_system_processor_not_found(self, cognitive_system, test_context):
        """존재하지 않는 프로세서 테스트"""
        with pytest.raises(CognitiveError):
            await cognitive_system.process("nonexistent_processor", test_context)

    def test_system_statistics(self, cognitive_system):
        """시스템 통계 테스트"""
        stats = cognitive_system.get_system_statistics()

        assert stats['total_processes'] == 0
        assert stats['total_successes'] == 0
        assert stats['total_errors'] == 0
        assert stats['success_rate'] == 0
        assert stats['processor_count'] == 1
        assert len(stats['processor_statistics']) == 1

    def test_list_processors(self, cognitive_system, test_processor):
        """프로세서 목록 테스트"""
        processors = cognitive_system.list_processors()

        assert len(processors) == 1
        assert test_processor.config.id in processors


if __name__ == "__main__":
    pytest.main([__file__])