"""
Simple Performance Test
간단한 성능 테스트 - PACA v5 Python 시스템 성능 측정
"""

import time
import sys
import os

# 테스트 경로 설정
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from paca.core.types import create_success, generate_id
from paca.core.validators import is_valid_korean_text, is_valid_email
from paca.services.knowledge import KnowledgeItem, KnowledgeType
from paca.cognitive.base import CognitiveTaskType, create_cognitive_context


def measure_performance(name, func, *args, **kwargs):
    """성능 측정 함수"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    execution_time = (end_time - start_time) * 1000  # milliseconds
    print(f"{name}: {execution_time:.2f}ms")

    return execution_time, result


def test_type_system_performance():
    """타입 시스템 성능 테스트"""
    def run_test():
        for i in range(1000):
            result = create_success(f"test_data_{i}")
            test_id = generate_id()
            assert result.is_success
            assert len(test_id) > 0
        return 1000

    return measure_performance("Type System Operations", run_test)


def test_korean_validation_performance():
    """한국어 검증 성능 테스트"""
    korean_texts = [
        "안녕하세요", "PACA v5 시스템", "한국어 텍스트 검증", "Hello World",
        "混合 텍스트", "123 숫자 포함", "특수문자!@# 포함", "단어"
    ]

    def run_test():
        count = 0
        for _ in range(100):
            for text in korean_texts:
                result = is_valid_korean_text(text)
                count += 1
        return count

    return measure_performance("Korean Validation", run_test)


def test_email_validation_performance():
    """이메일 검증 성능 테스트"""
    test_emails = [
        "test@example.com", "user.name@domain.com", "invalid-email",
        "user@domain", "user@domain.co.kr", "@domain.com"
    ]

    def run_test():
        count = 0
        for _ in range(100):
            for email in test_emails:
                result = is_valid_email(email)
                assert isinstance(result, bool)
                count += 1
        return count

    return measure_performance("Email Validation", run_test)


def test_knowledge_creation_performance():
    """지식 항목 생성 성능 테스트"""
    def run_test():
        for i in range(100):
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
        return 100

    return measure_performance("Knowledge Item Creation", run_test)


def test_cognitive_context_performance():
    """인지 컨텍스트 생성 성능 테스트"""
    def run_test():
        for i in range(100):
            context = create_cognitive_context(
                task_type=CognitiveTaskType.REASONING,
                input_data={"question": f"테스트 질문 {i}입니다."},
                metadata={"source": "performance_test"}
            )
            assert context.task_type == CognitiveTaskType.REASONING
        return 100

    return measure_performance("Cognitive Context Creation", run_test)


def run_performance_tests():
    """모든 성능 테스트 실행"""
    print("=" * 60)
    print("PACA v5 Python Simple Performance Tests")
    print("=" * 60)

    results = {}

    print("\n1. Type System Performance:")
    time_ms, ops = test_type_system_performance()
    results["type_system"] = time_ms
    print(f"   Processed {ops} operations")

    print("\n2. Korean Validation Performance:")
    time_ms, ops = test_korean_validation_performance()
    results["korean_validation"] = time_ms
    print(f"   Processed {ops} validations")

    print("\n3. Email Validation Performance:")
    time_ms, ops = test_email_validation_performance()
    results["email_validation"] = time_ms
    print(f"   Processed {ops} validations")

    print("\n4. Knowledge Item Creation Performance:")
    time_ms, ops = test_knowledge_creation_performance()
    results["knowledge_creation"] = time_ms
    print(f"   Created {ops} knowledge items")

    print("\n5. Cognitive Context Creation Performance:")
    time_ms, ops = test_cognitive_context_performance()
    results["cognitive_context"] = time_ms
    print(f"   Created {ops} cognitive contexts")

    print("\n" + "=" * 60)
    print("Performance Summary:")
    print("=" * 60)

    total_time = sum(results.values())
    performance_passed = True

    for test_name, execution_time in results.items():
        print(f"{test_name}: {execution_time:.2f}ms")

        # 각 작업이 100ms 이하여야 함 (관대한 기준)
        if execution_time > 100:
            print(f"  FAIL: Performance threshold exceeded (> 100ms)")
            performance_passed = False
        else:
            print(f"  PASS: Performance threshold met (<= 100ms)")

    print(f"\nTotal execution time: {total_time:.2f}ms")

    if performance_passed:
        print("\nAll performance tests PASSED!")
        print("PACA v5 Python system meets performance criteria.")
    else:
        print("\nSome performance tests FAILED!")
        print("Performance optimization needed.")

    print("=" * 60)
    return performance_passed, results


if __name__ == "__main__":
    run_performance_tests()