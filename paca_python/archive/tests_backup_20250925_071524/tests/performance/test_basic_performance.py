"""
Basic Performance Test
기본 성능 테스트 - PACA v5 Python 시스템 성능 측정
"""

import time
import asyncio
import sys
import os
from typing import List, Dict, Any

# 테스트 경로 설정
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from paca.core.types import create_success, generate_id
from paca.core.validators import is_valid_korean_text, is_valid_email
from paca.services.knowledge import KnowledgeService, KnowledgeItem, KnowledgeType, KnowledgeSearchQuery
from paca.cognitive.base import CognitiveContext, CognitiveTaskType, create_cognitive_context


class PerformanceTest:
    """성능 테스트 클래스"""

    def __init__(self):
        self.results: Dict[str, Any] = {}

    def measure_time(self, func_name: str):
        """시간 측정 데코레이터"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()

                execution_time = (end_time - start_time) * 1000  # milliseconds
                self.results[func_name] = execution_time
                print(f"{func_name}: {execution_time:.2f}ms")

                return result
            return wrapper
        return decorator

    def measure_async_time(self, func_name: str):
        """비동기 시간 측정 데코레이터"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                result = await func(*args, **kwargs)
                end_time = time.time()

                execution_time = (end_time - start_time) * 1000  # milliseconds
                self.results[func_name] = execution_time
                print(f"{func_name}: {execution_time:.2f}ms")

                return result
            return wrapper
        return decorator

    def test_type_system_performance(self):
        """타입 시스템 성능 테스트"""
        start_time = time.time()
        iterations = 1000

        for i in range(iterations):
            # Result 생성
            result = create_success(f"test_data_{i}")

            # ID 생성
            test_id = generate_id()

            # 성공/실패 확인
            assert result.is_success
            assert len(test_id) > 0

        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
        self.results["type_system_operations"] = execution_time
        print(f"type_system_operations: {execution_time:.2f}ms")

        return iterations

    def test_korean_validation_performance(self):
        """한국어 검증 성능 테스트"""
        korean_texts = [
            "안녕하세요",
            "PACA v5 시스템",
            "한국어 텍스트 검증",
            "Hello World",
            "混合 텍스트",
            "123 숫자 포함",
            "특수문자!@# 포함",
            "very long korean text with many characters 매우 긴 한국어 텍스트",
            "단어",
            "가나다라마바사아자차카타파하"
        ]

        iterations = 100

        for _ in range(iterations):
            for text in korean_texts:
                result = is_valid_korean_text(text)
                # 검증 결과가 예상과 맞는지 확인
                has_korean = any('\u3131' <= char <= '\u318e' or '\uac00' <= char <= '\ud7a3' for char in text)
                assert result == has_korean

        return len(korean_texts) * iterations

    def test_email_validation_performance(self):
        """이메일 검증 성능 테스트"""
        test_emails = [
            "test@example.com",
            "user.name@domain.com",
            "invalid-email",
            "user@domain",
            "user@domain.co.kr",
            "@domain.com",
            "user@",
            "user.name+tag@domain.com",
            "user.name@subdomain.domain.com",
            "test123@test123.com"
        ]

        iterations = 100

        start_time = time.time()
        for _ in range(iterations):
            for email in test_emails:
                result = is_valid_email(email)
                # 결과가 불린값인지 확인
                assert isinstance(result, bool)

        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # milliseconds
        self.results["email_validation_performance"] = execution_time
        print(f"email_validation_performance: {execution_time:.2f}ms")

        return len(test_emails) * iterations

    def test_knowledge_item_creation_performance(self):
        """지식 항목 생성 성능 테스트"""
        iterations = 100

        start_time = time.time()
        for i in range(iterations):
            knowledge_item = KnowledgeItem(
                id=generate_id(),
                user_id=generate_id(),
                title=f"테스트 지식 {i}",
                content=f"이것은 {i}번째 테스트용 지식 항목입니다.",
                category="테스트",
                tags=["테스트", f"항목{i}"],
                knowledge_type=KnowledgeType.CONCEPT,
                difficulty=3,
                confidence=0.8
            )

            assert knowledge_item.title == f"테스트 지식 {i}"
            assert knowledge_item.difficulty == 3

        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # milliseconds
        self.results["knowledge_item_creation"] = execution_time
        print(f"knowledge_item_creation: {execution_time:.2f}ms")

        return iterations

    def test_cognitive_context_creation_performance(self):
        """인지 컨텍스트 생성 성능 테스트"""
        iterations = 100

        start_time = time.time()
        for i in range(iterations):
            context = create_cognitive_context(
                task_type=CognitiveTaskType.REASONING,
                input_data={"question": f"테스트 질문 {i}입니다."},
                metadata={"source": "performance_test", "iteration": i}
            )

            assert context.task_type == CognitiveTaskType.REASONING
            assert context.input["question"] == f"테스트 질문 {i}입니다."

        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # milliseconds
        self.results["cognitive_context_creation"] = execution_time
        print(f"cognitive_context_creation: {execution_time:.2f}ms")

        return iterations

    def test_memory_usage(self):
        """메모리 사용량 테스트"""
        import psutil
        import gc

        process = psutil.Process()

        # 초기 메모리 사용량
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 대량의 객체 생성
        large_list = []
        for i in range(1000):
            knowledge_item = KnowledgeItem(
                id=generate_id(),
                user_id=generate_id(),
                title=f"메모리 테스트 지식 {i}",
                content=f"메모리 사용량 테스트를 위한 지식 항목 {i}입니다. " * 10,
                category="메모리테스트",
                tags=["메모리", "테스트", f"항목{i}"],
                knowledge_type=KnowledgeType.CONCEPT,
                difficulty=3,
                confidence=0.8
            )
            large_list.append(knowledge_item)

        # 생성 후 메모리 사용량
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 객체 삭제
        del large_list
        gc.collect()

        # 삭제 후 메모리 사용량
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        memory_increase = peak_memory - initial_memory
        memory_recovered = peak_memory - final_memory

        print(f"Memory Usage Test:")
        print(f"  Initial: {initial_memory:.2f} MB")
        print(f"  Peak: {peak_memory:.2f} MB")
        print(f"  Final: {final_memory:.2f} MB")
        print(f"  Increase: {memory_increase:.2f} MB")
        print(f"  Recovered: {memory_recovered:.2f} MB")
        print(f"  Recovery Rate: {(memory_recovered/memory_increase*100):.1f}%")

        return {
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': peak_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': memory_increase,
            'memory_recovered_mb': memory_recovered,
            'recovery_rate_percent': memory_recovered/memory_increase*100 if memory_increase > 0 else 0
        }

    def run_all_tests(self):
        """모든 성능 테스트 실행"""
        print("=" * 60)
        print("PACA v5 Python Performance Tests")
        print("=" * 60)

        print("\n1. Type System Performance:")
        type_ops = self.test_type_system_performance()
        print(f"   Processed {type_ops} operations")

        print("\n2. Korean Validation Performance:")
        korean_ops = self.test_korean_validation_performance()
        print(f"   Processed {korean_ops} validations")

        print("\n3. Email Validation Performance:")
        email_ops = self.test_email_validation_performance()
        print(f"   Processed {email_ops} validations")

        print("\n4. Knowledge Item Creation Performance:")
        knowledge_ops = self.test_knowledge_item_creation_performance()
        print(f"   Created {knowledge_ops} knowledge items")

        print("\n5. Cognitive Context Creation Performance:")
        cognitive_ops = self.test_cognitive_context_creation_performance()
        print(f"   Created {cognitive_ops} cognitive contexts")

        print("\n6. Memory Usage Test:")
        memory_stats = self.test_memory_usage()

        print("\n" + "=" * 60)
        print("Performance Summary:")
        print("=" * 60)

        for test_name, execution_time in self.results.items():
            print(f"{test_name}: {execution_time:.2f}ms")

        # 성능 기준 검사
        print("\n성능 기준 검사:")
        performance_passed = True

        # 각 작업이 50ms 이하여야 함
        for test_name, execution_time in self.results.items():
            if execution_time > 50:
                print(f"❌ {test_name}: {execution_time:.2f}ms > 50ms (기준 초과)")
                performance_passed = False
            else:
                print(f"✅ {test_name}: {execution_time:.2f}ms ≤ 50ms (기준 통과)")

        # 메모리 회복률이 80% 이상이어야 함
        if memory_stats['recovery_rate_percent'] >= 80:
            print(f"✅ Memory Recovery: {memory_stats['recovery_rate_percent']:.1f}% ≥ 80% (기준 통과)")
        else:
            print(f"❌ Memory Recovery: {memory_stats['recovery_rate_percent']:.1f}% < 80% (기준 초과)")
            performance_passed = False

        print("\n" + "=" * 60)
        if performance_passed:
            print("✅ 모든 성능 테스트 통과!")
            print("PACA v5 Python 시스템이 성능 기준을 만족합니다.")
        else:
            print("❌ 일부 성능 테스트 실패!")
            print("성능 최적화가 필요합니다.")
        print("=" * 60)

        return performance_passed, self.results, memory_stats


def run_performance_tests():
    """성능 테스트 실행 함수"""
    test = PerformanceTest()
    return test.run_all_tests()


if __name__ == "__main__":
    run_performance_tests()