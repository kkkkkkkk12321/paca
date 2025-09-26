"""
Cognitive Processes Test Suite

Phase 2 인지 프로세스 시스템의 종합 테스트를 수행합니다.
attention, perception, memory 통합 시스템의 기능을 검증합니다.
"""

import asyncio
import time
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any

# Import cognitive process components
from .attention import (
    create_attention_manager, create_focus_controller,
    AttentionTask, AttentionPriority, FocusTarget, FocusLevel
)
from .perception import (
    create_perception_engine, create_pattern_recognizer,
    create_concept_former, create_sensory_processor,
    SensoryInput, SensoryModality
)
from .cognitive_integrator import (
    create_cognitive_integrator, CognitiveRequest, ProcessingPriority
)


class TestAttentionSystem:
    """주의 시스템 테스트"""

    @pytest.fixture
    async def attention_manager(self):
        """AttentionManager 생성"""
        manager = await create_attention_manager()
        yield manager
        await manager.shutdown()

    @pytest.fixture
    async def focus_controller(self):
        """FocusController 생성"""
        controller = await create_focus_controller()
        yield controller

    @pytest.mark.asyncio
    async def test_attention_allocation(self, attention_manager):
        """주의 자원 할당 테스트"""
        # 주의 작업 생성
        task = AttentionTask(
            name="test_task",
            priority=AttentionPriority.HIGH,
            resource_required=20.0,
            duration_estimate_ms=1000
        )

        # 자원 할당 테스트
        result = await attention_manager.allocate_attention(task)
        assert result is True

        # 상태 확인
        status = await attention_manager.get_attention_status()
        assert status["active_tasks"] >= 1
        assert status["resource_usage_percent"] > 0

        # 작업 완료
        await attention_manager.complete_task(task.id, success=True)

        # 완료 후 상태 확인
        final_status = await attention_manager.get_attention_status()
        assert final_status["active_tasks"] == 0

    @pytest.mark.asyncio
    async def test_focus_control(self, focus_controller):
        """집중 제어 테스트"""
        # 집중 대상 생성
        target = FocusTarget(
            name="test_target",
            importance=0.8,
            urgency=0.7,
            complexity=0.5
        )

        # 집중 시작
        result = await focus_controller.start_focus(target, FocusLevel.HIGH)
        assert result is True

        # 집중 상태 확인
        state = await focus_controller.get_current_focus_state()
        assert state["active_targets"] >= 1
        assert state["total_focus_intensity"] > 0

        # 집중 조정
        adjustment_result = await focus_controller.adjust_focus(target.id, FocusLevel.MAXIMUM)
        assert adjustment_result is True

        # 집중 종료
        focus_result = await focus_controller.stop_focus(target.id)
        assert focus_result is not None
        assert focus_result.target_id == target.id

    @pytest.mark.asyncio
    async def test_attention_overload(self, attention_manager):
        """주의 과부하 처리 테스트"""
        tasks = []

        # 많은 작업 생성하여 과부하 유발
        for i in range(10):
            task = AttentionTask(
                name=f"overload_task_{i}",
                priority=AttentionPriority.NORMAL,
                resource_required=15.0
            )
            tasks.append(task)

        # 모든 작업 할당 시도
        results = []
        for task in tasks:
            result = await attention_manager.allocate_attention(task)
            results.append(result)

        # 일부는 할당되고 일부는 대기열에 들어가야 함
        allocated_count = sum(results)
        assert allocated_count < len(tasks)  # 모든 작업이 즉시 할당되지는 않음

        # 상태 확인
        status = await attention_manager.get_attention_status()
        assert status["state"] in ["DIVIDED", "OVERLOADED"]

        # 모든 작업 완료
        for task in tasks:
            await attention_manager.complete_task(task.id, success=True)


class TestPerceptionSystem:
    """지각 시스템 테스트"""

    @pytest.fixture
    async def perception_engine(self):
        """PerceptionEngine 생성"""
        engine = await create_perception_engine()
        yield engine

    @pytest.fixture
    async def pattern_recognizer(self):
        """PatternRecognizer 생성"""
        recognizer = await create_pattern_recognizer()
        yield recognizer

    @pytest.fixture
    async def concept_former(self):
        """ConceptFormer 생성"""
        former = await create_concept_former()
        yield former

    @pytest.fixture
    async def sensory_processor(self):
        """SensoryProcessor 생성"""
        processor = await create_sensory_processor()
        yield processor

    @pytest.mark.asyncio
    async def test_sensory_processing(self, sensory_processor):
        """감각 데이터 처리 테스트"""
        # 텍스트 데이터 전처리 테스트
        text_data = "Hello World! This is a test sentence with numbers 123."
        processed_text = await sensory_processor.preprocess(text_data, "textual")
        assert isinstance(processed_text, str)
        assert len(processed_text) > 0

        # 특징 추출 테스트
        features = await sensory_processor.extract_features(processed_text, "textual")
        assert isinstance(features, dict)
        assert "length" in features
        assert "word_count" in features

        # 수치 데이터 처리 테스트
        numeric_data = [1, 2, 3, 4, 5]
        processed_numeric = await sensory_processor.preprocess(numeric_data, "numerical")
        assert isinstance(processed_numeric, list)

        numeric_features = await sensory_processor.extract_features(processed_numeric, "numerical")
        assert "count" in numeric_features
        assert "mean" in numeric_features

    @pytest.mark.asyncio
    async def test_pattern_recognition(self, pattern_recognizer):
        """패턴 인식 테스트"""
        # 텍스트 패턴 인식
        text_data = "contact@example.com is an email address"
        patterns = await pattern_recognizer.recognize(text_data, "textual")
        assert isinstance(patterns, list)

        # 이메일 패턴이 인식되었는지 확인
        email_pattern_found = any(
            "email" in pattern.get("name", "").lower()
            for pattern in patterns
        )
        assert email_pattern_found

        # 시퀀스 패턴 인식
        sequence_data = [1, 2, 3, 4, 5]
        sequence_patterns = await pattern_recognizer.recognize(sequence_data, "sequential")
        assert isinstance(sequence_patterns, list)

        # 증가 패턴이 인식되었는지 확인
        increasing_found = any(
            "increasing" in pattern.get("name", "").lower()
            for pattern in sequence_patterns
        )
        assert increasing_found

    @pytest.mark.asyncio
    async def test_concept_formation(self, concept_former):
        """개념 형성 테스트"""
        # 유사한 패턴들로 개념 형성 테스트
        patterns = [
            {
                "name": "greeting_hello",
                "type": "SEMANTIC",
                "confidence": 0.9,
                "features": {"category": "greeting", "formality": "casual"}
            },
            {
                "name": "greeting_hi",
                "type": "SEMANTIC",
                "confidence": 0.8,
                "features": {"category": "greeting", "formality": "casual"}
            },
            {
                "name": "greeting_hey",
                "type": "SEMANTIC",
                "confidence": 0.7,
                "features": {"category": "greeting", "formality": "casual"}
            }
        ]

        concepts = await concept_former.form_concepts(patterns)
        assert isinstance(concepts, list)

        # 인사 관련 개념이 형성되었는지 확인
        if concepts:
            concept = concepts[0]
            assert "greeting" in concept.get("name", "").lower() or \
                   concept.get("features", {}).get("category") == "greeting"

    @pytest.mark.asyncio
    async def test_perception_integration(self, perception_engine):
        """지각 통합 처리 테스트"""
        # 감각 입력 생성
        sensory_input = SensoryInput(
            modality="textual",
            data="Hello world! The quick brown fox jumps over the lazy dog.",
            intensity=1.0,
            confidence=1.0
        )

        # 지각 처리 실행
        result = await perception_engine.process_input(sensory_input)

        assert result.success is True
        assert result.input_id == sensory_input.id
        assert result.processing_time_ms > 0

        # 처리 결과 확인
        assert isinstance(result.recognized_patterns, list)
        assert isinstance(result.formed_concepts, list)
        assert isinstance(result.perceived_objects, list)

        # 지각 상태 확인
        state = await perception_engine.get_perception_state()
        assert "state" in state
        assert "total_processed" in state


class TestCognitiveIntegration:
    """인지 통합 시스템 테스트"""

    @pytest.fixture
    async def cognitive_integrator(self):
        """CognitiveIntegrator 생성"""
        integrator = await create_cognitive_integrator()
        yield integrator

    @pytest.mark.asyncio
    async def test_cognitive_request_processing(self, cognitive_integrator):
        """인지 요청 처리 테스트"""
        # 인지 요청 생성
        request = CognitiveRequest(
            input_data="This is a test message for cognitive processing.",
            modality="textual",
            priority=ProcessingPriority.HIGH,
            require_attention=True,
            require_perception=True,
            require_memory=True
        )

        # 인지 처리 실행
        result = await cognitive_integrator.process_cognitive_request(request)

        assert result.success is True
        assert result.request_id == request.id
        assert result.processing_time_ms > 0
        assert result.confidence_score > 0

        # 각 단계별 결과 확인
        assert isinstance(result.attended_features, dict)
        assert isinstance(result.perceived_patterns, list)
        assert isinstance(result.formed_concepts, list)
        assert isinstance(result.retrieved_memories, list)
        assert isinstance(result.stored_memories, list)

    @pytest.mark.asyncio
    async def test_cognitive_focus_setting(self, cognitive_integrator):
        """인지 집중 설정 테스트"""
        # 집중 대상 설정
        targets = ["textual", "semantic"]
        weights = {"textual": 0.8, "semantic": 0.6}

        result = await cognitive_integrator.set_cognitive_focus(targets, weights)
        assert result is True

        # 집중 상태 확인
        state = await cognitive_integrator.get_cognitive_state()
        assert "attention" in state
        assert "focus" in state
        assert "perception" in state

    @pytest.mark.asyncio
    async def test_multi_modal_processing(self, cognitive_integrator):
        """다중 양상 처리 테스트"""
        requests = [
            CognitiveRequest(
                input_data="Text input for processing",
                modality="textual",
                priority=ProcessingPriority.HIGH
            ),
            CognitiveRequest(
                input_data=[1, 2, 3, 4, 5],
                modality="numerical",
                priority=ProcessingPriority.NORMAL
            ),
            CognitiveRequest(
                input_data={"x": 10, "y": 20},
                modality="spatial",
                priority=ProcessingPriority.NORMAL
            )
        ]

        # 각 요청 처리
        results = []
        for request in requests:
            result = await cognitive_integrator.process_cognitive_request(request)
            results.append(result)

        # 모든 요청이 성공적으로 처리되었는지 확인
        for result in results:
            assert result.success is True
            assert result.confidence_score > 0

        # 전체 인지 상태 확인
        final_state = await cognitive_integrator.get_cognitive_state()
        assert final_state["total_processed"] >= 3


class TestPerformance:
    """성능 테스트"""

    @pytest.mark.asyncio
    async def test_attention_response_time(self):
        """주의 시스템 응답 시간 테스트"""
        manager = await create_attention_manager()

        try:
            # 여러 작업의 평균 응답 시간 측정
            times = []
            for i in range(10):
                start_time = time.time()

                task = AttentionTask(
                    name=f"perf_test_{i}",
                    priority=AttentionPriority.NORMAL,
                    resource_required=10.0
                )

                await manager.allocate_attention(task)
                await manager.complete_task(task.id, success=True)

                end_time = time.time()
                times.append((end_time - start_time) * 1000)

            avg_time = sum(times) / len(times)
            assert avg_time < 100  # 평균 100ms 미만

        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_perception_throughput(self):
        """지각 시스템 처리량 테스트"""
        engine = await create_perception_engine()

        try:
            # 동시 처리 테스트
            inputs = []
            for i in range(5):
                sensory_input = SensoryInput(
                    modality="textual",
                    data=f"Test message {i} for throughput testing",
                    intensity=1.0
                )
                inputs.append(sensory_input)

            start_time = time.time()

            # 병렬 처리
            tasks = [engine.process_input(inp) for inp in inputs]
            results = await asyncio.gather(*tasks)

            end_time = time.time()
            total_time = (end_time - start_time) * 1000

            # 모든 결과가 성공적인지 확인
            for result in results:
                assert result.success is True

            # 처리량 확인 (5개 입력을 2초 내에 처리)
            assert total_time < 2000

        finally:
            pass  # engine cleanup

    @pytest.mark.asyncio
    async def test_memory_integration_speed(self):
        """메모리 통합 속도 테스트"""
        integrator = await create_cognitive_integrator()

        try:
            # 메모리 집약적 요청들
            requests = []
            for i in range(3):
                request = CognitiveRequest(
                    input_data=f"Memory test message {i} with important information",
                    modality="textual",
                    priority=ProcessingPriority.HIGH,
                    require_memory=True
                )
                requests.append(request)

            start_time = time.time()

            # 순차 처리
            for request in requests:
                result = await integrator.process_cognitive_request(request)
                assert result.success is True

            end_time = time.time()
            total_time = (end_time - start_time) * 1000

            # 메모리 통합이 5초 내에 완료되는지 확인
            assert total_time < 5000

        finally:
            pass  # integrator cleanup


# 통합 테스트 함수
async def run_integration_test():
    """종합 통합 테스트 실행"""
    print("🧠 Starting PACA Phase 2 Cognitive Processes Integration Test...")

    try:
        # 1. 인지 통합 시스템 생성
        print("1. Creating cognitive integrator...")
        integrator = await create_cognitive_integrator()

        # 2. 집중 설정 테스트
        print("2. Setting cognitive focus...")
        focus_result = await integrator.set_cognitive_focus(
            ["textual", "semantic"],
            {"textual": 0.9, "semantic": 0.7}
        )
        assert focus_result, "Failed to set cognitive focus"

        # 3. 다양한 인지 처리 테스트
        print("3. Testing cognitive processing...")
        test_requests = [
            CognitiveRequest(
                input_data="Hello world! This is a complex sentence for testing.",
                modality="textual",
                priority=ProcessingPriority.HIGH
            ),
            CognitiveRequest(
                input_data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                modality="numerical",
                priority=ProcessingPriority.NORMAL
            ),
            CognitiveRequest(
                input_data={"message": "spatial data", "coordinates": {"x": 100, "y": 200}},
                modality="spatial",
                priority=ProcessingPriority.NORMAL
            )
        ]

        results = []
        for i, request in enumerate(test_requests):
            print(f"   Processing request {i+1}/{len(test_requests)}...")
            result = await integrator.process_cognitive_request(request)
            results.append(result)

            assert result.success, f"Request {i+1} failed: {result.error_message}"
            print(f"   ✅ Request {i+1} processed successfully (confidence: {result.confidence_score:.2f})")

        # 4. 시스템 상태 확인
        print("4. Checking system state...")
        state = await integrator.get_cognitive_state()

        print(f"   Total processed: {state['total_processed']}")
        print(f"   Success rate: {state['success_rate']:.2%}")
        print(f"   Integration efficiency: {state['integration_efficiency']:.2f}")

        # 5. 성능 확인
        print("5. Performance validation...")
        avg_processing_time = sum(r.processing_time_ms for r in results) / len(results)
        avg_confidence = sum(r.confidence_score for r in results) / len(results)

        print(f"   Average processing time: {avg_processing_time:.2f}ms")
        print(f"   Average confidence: {avg_confidence:.2f}")

        # 성능 기준 확인
        assert avg_processing_time < 1000, f"Processing too slow: {avg_processing_time}ms"
        assert avg_confidence > 0.5, f"Confidence too low: {avg_confidence}"

        print("✅ Phase 2 Integration Test completed successfully!")
        print("\n📊 Test Results Summary:")
        print(f"   - Requests processed: {len(results)}")
        print(f"   - Success rate: 100%")
        print(f"   - Average processing time: {avg_processing_time:.2f}ms")
        print(f"   - Average confidence: {avg_confidence:.2f}")
        print(f"   - Integration efficiency: {state['integration_efficiency']:.2f}")

        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    # 통합 테스트 실행
    asyncio.run(run_integration_test())