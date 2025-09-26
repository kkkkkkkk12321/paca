"""
Core Types Module Tests
"""

import pytest
import time
from paca.core.types import (
    Result, create_success, create_failure, LearningPoint,
    LogLevel, Status, Priority, CognitiveState,
    BaseEntity, PaginationRequest, PaginationResponse
)


class TestResult:
    """Result 타입 테스트"""

    def test_success_result(self):
        """성공 결과 테스트"""
        result = create_success("test data")

        assert result.is_success is True
        assert result.is_failure is False
        assert result.data == "test data"
        assert result.error is None
        assert bool(result) is True

    def test_failure_result(self):
        """실패 결과 테스트"""
        result = create_failure("test error")

        assert result.is_success is False
        assert result.is_failure is True
        assert result.data is None
        assert result.error == "test error"
        assert bool(result) is False

    def test_result_unwrap(self):
        """Result unwrap 테스트"""
        success_result = create_success("test")
        assert success_result.unwrap() == "test"

        failure_result = create_failure("error")
        with pytest.raises(ValueError):
            failure_result.unwrap()

    def test_result_unwrap_or(self):
        """Result unwrap_or 테스트"""
        success_result = create_success("test")
        assert success_result.unwrap_or("default") == "test"

        failure_result = create_failure("error")
        assert failure_result.unwrap_or("default") == "default"

    def test_result_map(self):
        """Result map 테스트"""
        success_result = create_success(5)
        mapped_result = success_result.map(lambda x: x * 2)

        assert mapped_result.is_success is True
        assert mapped_result.data == 10

        failure_result = create_failure("error")
        mapped_failure = failure_result.map(lambda x: x * 2)

        assert mapped_failure.is_success is False


class TestEnums:
    """열거형 테스트"""

    def test_log_level(self):
        """LogLevel 테스트"""
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.ERROR.value == "error"

    def test_status(self):
        """Status 테스트"""
        assert Status.IDLE.value == "idle"
        assert Status.RUNNING.value == "running"
        assert Status.SUCCESS.value == "success"

    def test_priority(self):
        """Priority 테스트"""
        assert Priority.LOW.value == 1
        assert Priority.NORMAL.value == 2
        assert Priority.CRITICAL.value == 4

    def test_cognitive_state(self):
        """CognitiveState 테스트"""
        assert CognitiveState.IDLE.value == "idle"
        assert CognitiveState.PROCESSING.value == "processing"
        assert CognitiveState.LEARNING.value == "learning"


class TestLearningPoint:
    """LearningPoint 테스트"""

    def test_learning_point_creation(self):
        """학습 포인트 생성 테스트"""
        lp = LearningPoint(
            content="test content",
            confidence=0.8,
            importance_score=0.7
        )

        assert lp.content == "test content"
        assert lp.confidence == 0.8
        assert lp.importance_score == 0.7
        assert lp.id is not None
        assert lp.timestamp > 0

    def test_learning_point_validation(self):
        """학습 포인트 유효성 검사 테스트"""
        # 잘못된 confidence 값
        with pytest.raises(ValueError):
            LearningPoint(confidence=1.5)

        # 잘못된 importance_score 값
        with pytest.raises(ValueError):
            LearningPoint(importance_score=-0.1)


class TestBaseEntity:
    """BaseEntity 테스트"""

    def test_base_entity_creation(self):
        """기본 엔티티 생성 테스트"""
        entity = BaseEntity()

        assert entity.id is not None
        assert entity.created_at > 0
        assert entity.updated_at > 0
        assert isinstance(entity.metadata, dict)

    def test_update_timestamp(self):
        """타임스탬프 업데이트 테스트"""
        entity = BaseEntity()
        original_updated_at = entity.updated_at

        time.sleep(0.01)  # 약간의 시간 대기
        entity.update_timestamp()

        assert entity.updated_at > original_updated_at


class TestPagination:
    """페이지네이션 테스트"""

    def test_pagination_request(self):
        """페이지네이션 요청 테스트"""
        request = PaginationRequest(page=1, limit=10)

        assert request.page == 1
        assert request.limit == 10
        assert request.sort_order == "asc"

    def test_pagination_request_validation(self):
        """페이지네이션 요청 유효성 검사"""
        # 잘못된 page 값
        with pytest.raises(ValueError):
            PaginationRequest(page=0, limit=10)

        # 잘못된 limit 값
        with pytest.raises(ValueError):
            PaginationRequest(page=1, limit=0)

        # 잘못된 sort_order 값
        with pytest.raises(ValueError):
            PaginationRequest(page=1, limit=10, sort_order="invalid")

    def test_pagination_response_creation(self):
        """페이지네이션 응답 생성 테스트"""
        items = ["item1", "item2", "item3"]
        response = PaginationResponse.create(
            items=items[:2],  # 2개만 반환
            total_items=10,
            page=1,
            limit=2
        )

        assert len(response.items) == 2
        assert response.total_items == 10
        assert response.total_pages == 5
        assert response.current_page == 1
        assert response.has_next_page is True
        assert response.has_previous_page is False


if __name__ == "__main__":
    pytest.main([__file__])