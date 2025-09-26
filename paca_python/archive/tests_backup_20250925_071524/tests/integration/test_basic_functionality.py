"""
기본 기능 통합 테스트
PACA v5 Python 변환 프로젝트 - 통합 테스트

이 테스트는 변환된 모듈들이 올바르게 함께 동작하는지 검증합니다.
"""

import pytest
import asyncio
import sys
import os
from typing import Dict, Any

# 테스트 경로 설정
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from paca.core.types import Result, ID, create_success, create_failure, generate_id
from paca.core.errors import ValidationError, AuthenticationError
from paca.core.validators import is_valid_korean_text, is_valid_email
from paca.services.auth import AuthenticationService, LoginRequest, LoginResponse
from paca.services.knowledge import KnowledgeService, KnowledgeItem, KnowledgeSearchQuery, KnowledgeType
from paca.cognitive.base import CognitiveContext, CognitiveTaskType, create_cognitive_context


class TestBasicFunctionality:
    """기본 기능 통합 테스트 클래스"""

    def test_type_system_integration(self):
        """타입 시스템 통합 테스트"""
        # Result 타입 테스트
        success_result = create_success("test data")
        assert success_result.is_success is True
        assert success_result.value == "test data"

        error = ValidationError("test error")
        failure_result = create_failure(error)
        assert failure_result.is_failure is True
        assert failure_result.error == error

        # ID 생성 테스트
        test_id = generate_id()
        assert isinstance(test_id, str)
        assert len(test_id) > 0

        print("Type system integration test passed")

    def test_validation_system(self):
        """검증 시스템 테스트"""
        # 한국어 텍스트 검증
        assert is_valid_korean_text("안녕하세요") is True
        assert is_valid_korean_text("Hello") is False
        assert is_valid_korean_text("안녕 Hello") is True  # 혼합 텍스트

        # 이메일 검증
        assert is_valid_email("test@example.com") is True
        assert is_valid_email("invalid-email") is False

        print("Validation system test passed")

    def test_authentication_service(self):
        """인증 서비스 통합 테스트"""
        # 로그인 요청 생성 테스트
        login_request = LoginRequest(
            email="test@example.com",
            password="testpassword123",
            remember_me=False
        )

        # 요청 객체가 올바르게 생성되는지 확인
        assert login_request is not None
        assert login_request.email == "test@example.com"
        assert login_request.password == "testpassword123"
        assert login_request.remember_me is False

        # AuthenticationService 클래스가 존재하는지 확인
        assert AuthenticationService is not None
        assert hasattr(AuthenticationService, 'login')
        assert hasattr(AuthenticationService, 'logout')
        assert hasattr(AuthenticationService, 'refresh_token')

        print("Authentication service integration test passed")

    def test_knowledge_service(self):
        """지식 관리 서비스 통합 테스트"""
        # 지식 항목 생성 테스트
        knowledge_item = KnowledgeItem(
            id=generate_id(),
            user_id=generate_id(),
            title="테스트 지식",
            content="이것은 테스트용 지식 항목입니다.",
            category="테스트",
            tags=["테스트", "통합테스트"],
            knowledge_type=KnowledgeType.CONCEPT,
            difficulty=3,
            confidence=0.8
        )

        # 지식 항목이 올바르게 생성되는지 확인
        assert knowledge_item is not None
        assert knowledge_item.title == "테스트 지식"
        assert knowledge_item.category == "테스트"
        assert "테스트" in knowledge_item.tags

        # 검색 쿼리 생성 테스트
        search_query = KnowledgeSearchQuery(
            query="테스트",
            categories=["테스트"],
            tags=["테스트"],
            limit=10
        )

        # 검색 쿼리가 올바르게 생성되는지 확인
        assert search_query is not None
        assert search_query.query == "테스트"
        assert search_query.limit == 10

        # KnowledgeService 클래스가 존재하는지 확인
        assert KnowledgeService is not None
        assert hasattr(KnowledgeService, 'create_knowledge_item')
        assert hasattr(KnowledgeService, 'search_knowledge')
        assert hasattr(KnowledgeService, 'get_knowledge_item')

        print("Knowledge service integration test passed")

    def test_cognitive_system(self):
        """인지 시스템 통합 테스트"""
        # 인지 컨텍스트 생성
        context = create_cognitive_context(
            task_type=CognitiveTaskType.REASONING,
            input_data={"question": "테스트 질문입니다."},
            metadata={"source": "integration_test"}
        )

        assert context is not None
        assert context.task_type == CognitiveTaskType.REASONING
        assert context.input["question"] == "테스트 질문입니다."
        assert context.metadata["source"] == "integration_test"

        print("Cognitive system integration test passed")

    def test_error_handling_integration(self):
        """에러 처리 통합 테스트"""
        # ValidationError 테스트
        validation_error = ValidationError("유효성 검사 실패")
        print(f"Validation error message: {str(validation_error)}")
        assert "유효성 검사 실패" in str(validation_error)
        assert isinstance(validation_error, Exception)

        # AuthenticationError 테스트
        auth_error = AuthenticationError("인증 실패")
        print(f"Auth error message: {str(auth_error)}")
        assert "인증 실패" in str(auth_error)
        assert isinstance(auth_error, Exception)

        print("Error handling integration test passed")


class TestSystemIntegration:
    """시스템 통합 테스트 클래스"""

    def test_module_dependencies(self):
        """모듈 의존성 테스트"""
        # 모든 주요 모듈이 올바르게 임포트되는지 확인
        try:
            from paca.core import types, errors, utils, events, constants, validators
            from paca.services import auth, knowledge
            from paca.cognitive import base as cognitive_base

            # 각 모듈의 주요 클래스/함수가 존재하는지 확인
            assert hasattr(types, 'Result')
            assert hasattr(types, 'ID')
            assert hasattr(errors, 'ValidationError')
            assert hasattr(validators, 'is_valid_korean_text')
            assert hasattr(auth, 'AuthenticationService')
            assert hasattr(knowledge, 'KnowledgeService')
            assert hasattr(cognitive_base, 'CognitiveContext')

            print("Module dependencies test passed")

        except ImportError as e:
            pytest.fail(f"Module import failed: {e}")

    def test_korean_language_support(self):
        """한국어 지원 테스트"""
        # 한국어 텍스트 처리
        korean_text = "안녕하세요, PACA v5입니다."

        # 한국어 검증
        assert is_valid_korean_text(korean_text) is True

        # 한국어 에러 메시지
        error = ValidationError("한국어 에러 메시지")
        assert "한국어" in str(error)

        print("Korean language support test passed")


def run_integration_tests():
    """통합 테스트 실행 함수"""
    print("=" * 60)
    print("PACA v5 Python Integration Tests")
    print("=" * 60)

    # 기본 기능 테스트
    basic_tests = TestBasicFunctionality()

    print("\n1. Type System Integration...")
    basic_tests.test_type_system_integration()

    print("\n2. Validation System...")
    basic_tests.test_validation_system()

    print("\n3. Authentication Service...")
    basic_tests.test_authentication_service()

    print("\n4. Knowledge Service...")
    basic_tests.test_knowledge_service()

    print("\n5. Cognitive System...")
    basic_tests.test_cognitive_system()

    print("\n6. Error Handling...")
    basic_tests.test_error_handling_integration()

    # 시스템 통합 테스트
    system_tests = TestSystemIntegration()

    print("\n7. Module Dependencies...")
    system_tests.test_module_dependencies()

    print("\n8. Korean Language Support...")
    system_tests.test_korean_language_support()

    print("\n" + "=" * 60)
    print("All integration tests completed successfully!")
    print("PACA v5 Python conversion is working properly.")
    print("=" * 60)


if __name__ == "__main__":
    run_integration_tests()