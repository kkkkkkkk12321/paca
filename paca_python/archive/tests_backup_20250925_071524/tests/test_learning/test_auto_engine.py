"""
Auto Learning Engine Tests
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from paca.learning.auto.engine import AutoLearningSystem
from paca.learning.auto.types import (
    LearningPoint, LearningPattern, PatternType, LearningCategory,
    DatabaseInterface, ConversationMemoryInterface
)


class MockDatabase:
    """테스트용 모의 데이터베이스"""

    def __init__(self):
        self.experiences = []
        self.heuristics = []

    def add_experience(self, name: str, description: str) -> bool:
        self.experiences.append({"name": name, "description": description})
        return True

    def add_heuristic(self, rule: str) -> bool:
        self.heuristics.append({"rule": rule})
        return True

    def get_experiences(self, context=None):
        return self.experiences

    def get_heuristics(self, context=None):
        return self.heuristics


class MockConversationMemory:
    """테스트용 모의 대화 메모리"""

    def __init__(self):
        self.conversations = []
        self.learning_points = []

    def get_recent_conversations(self, limit=10):
        return self.conversations[-limit:]

    def get_conversation_context(self, conversation_id):
        return {"id": conversation_id, "context": "test"}

    def store_learning_point(self, learning_point):
        self.learning_points.append(learning_point)
        return True


class TestAutoLearningSystem:
    """자동 학습 시스템 테스트"""

    @pytest.fixture
    def auto_learning_system(self, tmp_path):
        """테스트용 자동 학습 시스템"""
        db = MockDatabase()
        memory = MockConversationMemory()
        return AutoLearningSystem(
            database=db,
            conversation_memory=memory,
            storage_path=str(tmp_path / "test_learning"),
            enable_korean_nlp=False  # 테스트에서는 KoNLPy 비활성화
        )

    @pytest.mark.asyncio
    async def test_analyze_success_pattern(self, auto_learning_system):
        """성공 패턴 분석 테스트"""
        user_message = "오류가 발생했는데 어떻게 해결하지?"
        paca_response = "해당 문제는 설정을 수정하면 해결됩니다. 완료되었습니다."

        result = await auto_learning_system.analyze_learning_opportunities(
            user_message, paca_response, "test_conversation"
        )

        assert result.is_success is True
        learning_points = result.data

        # 성공 패턴이 감지되었는지 확인
        success_points = [
            lp for lp in learning_points
            if lp.category == LearningCategory.SUCCESS_PATTERN
        ]
        assert len(success_points) > 0

        # 학습 포인트 내용 검증
        success_point = success_points[0]
        assert "해결" in success_point.extracted_knowledge
        assert success_point.confidence > 0.0

    @pytest.mark.asyncio
    async def test_analyze_failure_pattern(self, auto_learning_system):
        """실패 패턴 분석 테스트"""
        user_message = "이 방법을 시도했는데 실패했어요"
        paca_response = "그 방법은 문제가 있습니다. 다른 접근이 필요해요."

        result = await auto_learning_system.analyze_learning_opportunities(
            user_message, paca_response, "test_conversation"
        )

        assert result.is_success is True
        learning_points = result.data

        # 실패 패턴이 감지되었는지 확인
        failure_points = [
            lp for lp in learning_points
            if lp.category == LearningCategory.ERROR_PATTERN
        ]
        assert len(failure_points) > 0

    @pytest.mark.asyncio
    async def test_analyze_preference_pattern(self, auto_learning_system):
        """선호도 패턴 분석 테스트"""
        user_message = "나는 Python을 좋아해. 다른 언어는 별로야."
        paca_response = "Python은 좋은 선택입니다. 관련 도구들을 추천드릴게요."

        result = await auto_learning_system.analyze_learning_opportunities(
            user_message, paca_response, "test_conversation"
        )

        assert result.is_success is True
        learning_points = result.data

        # 선호도 패턴이 감지되었는지 확인
        preference_points = [
            lp for lp in learning_points
            if lp.category == LearningCategory.USER_PREFERENCE
        ]
        assert len(preference_points) > 0

    @pytest.mark.asyncio
    async def test_analyze_knowledge_pattern(self, auto_learning_system):
        """지식 패턴 분석 테스트"""
        user_message = "이 개념을 몰랐는데 배웠어요"
        paca_response = "새로운 개념을 이해하셨군요. 관련 자료를 더 제공하겠습니다."

        result = await auto_learning_system.analyze_learning_opportunities(
            user_message, paca_response, "test_conversation"
        )

        assert result.is_success is True
        learning_points = result.data

        # 지식 패턴이 감지되었는지 확인
        knowledge_points = [
            lp for lp in learning_points
            if lp.category == LearningCategory.KNOWLEDGE_GAP
        ]
        assert len(knowledge_points) > 0

    def test_get_learning_status(self, auto_learning_system):
        """학습 상태 조회 테스트"""
        # 테스트 데이터 추가
        test_point = LearningPoint(
            user_message="test",
            paca_response="test",
            context="test",
            category=LearningCategory.SUCCESS_PATTERN,
            confidence=0.8,
            extracted_knowledge="test knowledge"
        )
        auto_learning_system.learning_points.append(test_point)

        status = auto_learning_system.get_learning_status()

        assert status.learning_points == 1
        assert status.generated_tactics == 0
        assert status.generated_heuristics == 0
        assert status.active_patterns == len(auto_learning_system.learning_patterns)

    def test_get_generated_knowledge(self, auto_learning_system):
        """생성된 지식 조회 테스트"""
        knowledge = auto_learning_system.get_generated_knowledge()

        assert isinstance(knowledge.tactics, list)
        assert isinstance(knowledge.heuristics, list)
        assert isinstance(knowledge.effectiveness_summary, dict)

    def test_pattern_initialization(self, auto_learning_system):
        """패턴 초기화 테스트"""
        patterns = auto_learning_system.learning_patterns

        assert len(patterns) > 0

        # 각 패턴 타입이 있는지 확인
        pattern_types = {pattern.pattern_type for pattern in patterns}
        expected_types = {
            PatternType.SUCCESS,
            PatternType.FAILURE,
            PatternType.PREFERENCE,
            PatternType.KNOWLEDGE,
            PatternType.PERFORMANCE
        }

        assert expected_types.issubset(pattern_types)

        # 각 패턴의 구조 검증
        for pattern in patterns:
            assert isinstance(pattern.keywords, list)
            assert len(pattern.keywords) > 0
            assert isinstance(pattern.context_indicators, list)
            assert 0.0 <= pattern.confidence_threshold <= 1.0
            assert pattern.language == "ko"

    @pytest.mark.asyncio
    async def test_no_pattern_detection(self, auto_learning_system):
        """패턴이 감지되지 않는 경우 테스트"""
        user_message = "일반적인 질문입니다"
        paca_response = "일반적인 답변입니다"

        result = await auto_learning_system.analyze_learning_opportunities(
            user_message, paca_response, "test_conversation"
        )

        assert result.is_success is True
        learning_points = result.data
        assert len(learning_points) == 0  # 패턴이 감지되지 않아야 함

    def test_text_similarity_calculation(self, auto_learning_system):
        """텍스트 유사도 계산 테스트"""
        text1 = "Python 프로그래밍은 재미있다"
        text2 = "Python 프로그래밍은 어렵다"
        text3 = "Java는 객체지향 언어다"

        similarity1 = auto_learning_system._calculate_text_similarity(text1, text2)
        similarity2 = auto_learning_system._calculate_text_similarity(text1, text3)

        # text1과 text2는 더 유사해야 함
        assert similarity1 > similarity2

    def test_context_extraction(self, auto_learning_system):
        """컨텍스트 추출 테스트"""
        user_message = "프로그래밍 문제가 있어요"
        paca_response = "개발 도구를 사용해보세요"

        context = auto_learning_system._extract_context(user_message, paca_response)

        assert context in ["programming", "development", "general"]


if __name__ == "__main__":
    pytest.main([__file__])