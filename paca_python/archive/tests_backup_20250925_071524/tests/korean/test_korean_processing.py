"""
Korean Language Processing Test
한국어 처리 개선 테스트 - PACA v5 Python 시스템 한국어 지원 검증
"""

import sys
import os

# 테스트 경로 설정
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from paca.core.validators import is_valid_korean_text, validate_korean_name, validate_korean_phone
from paca.core.errors import ValidationError
from paca.services.knowledge import KnowledgeItem, KnowledgeType


def test_korean_text_validation():
    """한국어 텍스트 검증 테스트"""
    print("1. Korean Text Validation Tests:")

    test_cases = [
        # (text, expected_result, description)
        ("안녕하세요", True, "Basic Korean greeting"),
        ("PACA v5", False, "English with numbers"),
        ("안녕 Hello", True, "Mixed Korean-English"),
        ("123456", False, "Numbers only"),
        ("!@#$%", False, "Special characters only"),
        ("한국어 텍스트입니다", True, "Korean sentence"),
        ("가나다라마바사", True, "Korean characters"),
        ("", False, "Empty string"),
        ("ㅏㅓㅗㅜ", True, "Korean vowels"),
        ("ㄱㄴㄷㄹ", True, "Korean consonants"),
        ("한글과 English 혼합", True, "Korean-English mixed"),
        ("숫자123과 한글", True, "Korean with numbers"),
        ("특수문자!와 한글", True, "Korean with special chars"),
    ]

    passed = 0
    total = len(test_cases)

    for text, expected, description in test_cases:
        result = is_valid_korean_text(text)
        status = "PASS" if result == expected else "FAIL"
        print(f"   {status}: '{text}' -> {result} ({description})")
        if result == expected:
            passed += 1

    print(f"   Result: {passed}/{total} tests passed")
    return passed == total


def test_korean_name_validation():
    """한국어 이름 검증 테스트"""
    print("\n2. Korean Name Validation Tests:")

    test_cases = [
        ("김철수", True, "Standard Korean name"),
        ("이영희", True, "Standard Korean name"),
        ("박", False, "Too short (single character)"),
        ("김철수영희박민수", False, "Too long"),
        ("Kim", False, "English name"),
        ("김123", False, "Name with numbers"),
        ("이!@#", False, "Name with special chars"),
        ("홍길동", True, "Historical Korean name"),
        ("최우진", True, "Modern Korean name"),
        ("", False, "Empty string"),
    ]

    passed = 0
    total = len(test_cases)

    for name, expected, description in test_cases:
        result = validate_korean_name(name)
        status = "PASS" if result == expected else "FAIL"
        print(f"   {status}: '{name}' -> {result} ({description})")
        if result == expected:
            passed += 1

    print(f"   Result: {passed}/{total} tests passed")
    return passed == total


def test_korean_phone_validation():
    """한국어 전화번호 검증 테스트"""
    print("\n3. Korean Phone Number Validation Tests:")

    test_cases = [
        ("010-1234-5678", True, "Standard mobile format"),
        ("01012345678", True, "Mobile without hyphens"),
        ("010 1234 5678", True, "Mobile with spaces"),
        ("02-123-4567", True, "Seoul landline"),
        ("031-123-4567", True, "Gyeonggi landline"),
        ("1234-5678", False, "Missing area code"),
        ("010-123-456", False, "Incomplete mobile"),
        ("abc-def-ghij", False, "Non-numeric"),
        ("", False, "Empty string"),
        ("010-1234-56789", False, "Too long"),
        ("010.1234.5678", False, "Wrong separator"),
    ]

    passed = 0
    total = len(test_cases)

    for phone, expected, description in test_cases:
        result = validate_korean_phone(phone)
        status = "PASS" if result == expected else "FAIL"
        print(f"   {status}: '{phone}' -> {result} ({description})")
        if result == expected:
            passed += 1

    print(f"   Result: {passed}/{total} tests passed")
    return passed == total


def test_korean_knowledge_items():
    """한국어 지식 항목 테스트"""
    print("\n4. Korean Knowledge Items Tests:")

    korean_knowledge_items = [
        {
            "title": "한국어 학습 방법",
            "content": "효과적인 한국어 학습 방법에 대해 설명합니다. 먼저 기본 자음과 모음을 익히고, 단어를 늘려가며 문장 구조를 이해하는 것이 중요합니다.",
            "category": "언어학습",
            "tags": ["한국어", "학습", "언어", "교육"]
        },
        {
            "title": "한국 전통 음식",
            "content": "김치, 불고기, 비빔밥 등 한국의 대표적인 전통 음식들을 소개합니다. 각각의 조리법과 영양학적 특성을 설명합니다.",
            "category": "문화",
            "tags": ["음식", "전통", "문화", "요리"]
        },
        {
            "title": "한국사 주요 사건",
            "content": "삼국시대부터 현대까지 한국사의 주요 사건들을 시대순으로 정리했습니다. 각 시대의 특징과 의미를 알아봅시다.",
            "category": "역사",
            "tags": ["역사", "한국사", "사건", "시대"]
        }
    ]

    passed = 0
    total = len(korean_knowledge_items)

    for i, item_data in enumerate(korean_knowledge_items):
        try:
            knowledge_item = KnowledgeItem(
                id=f"korean_test_{i}",
                user_id="test_user",
                title=item_data["title"],
                content=item_data["content"],
                category=item_data["category"],
                tags=item_data["tags"],
                knowledge_type=KnowledgeType.CONCEPT,
                difficulty=3,
                confidence=0.8
            )

            # 한국어 제목 검증
            title_has_korean = is_valid_korean_text(knowledge_item.title)

            # 한국어 내용 검증
            content_has_korean = is_valid_korean_text(knowledge_item.content)

            if title_has_korean and content_has_korean:
                print(f"   PASS: '{knowledge_item.title}' - Korean content created successfully")
                passed += 1
            else:
                print(f"   FAIL: '{knowledge_item.title}' - Korean validation failed")

        except Exception as e:
            print(f"   ERROR: Failed to create knowledge item {i}: {e}")

    print(f"   Result: {passed}/{total} tests passed")
    return passed == total


def test_korean_error_messages():
    """한국어 에러 메시지 테스트"""
    print("\n5. Korean Error Messages Tests:")

    test_cases = [
        ("유효성 검사 실패", "ValidationError with Korean message"),
        ("사용자를 찾을 수 없습니다", "User not found error"),
        ("잘못된 비밀번호입니다", "Invalid password error"),
        ("권한이 없습니다", "Permission denied error"),
        ("서버 오류가 발생했습니다", "Server error message")
    ]

    passed = 0
    total = len(test_cases)

    for message, description in test_cases:
        try:
            error = ValidationError(message)
            error_str = str(error)

            # 에러 메시지에 한국어가 포함되어 있는지 확인
            has_korean = is_valid_korean_text(error_str)

            if has_korean:
                print(f"   PASS: Korean error message created - '{message}'")
                passed += 1
            else:
                print(f"   FAIL: Korean not detected in error - '{error_str}'")

        except Exception as e:
            print(f"   ERROR: Failed to create error message: {e}")

    print(f"   Result: {passed}/{total} tests passed")
    return passed == total


def run_korean_processing_tests():
    """모든 한국어 처리 테스트 실행"""
    print("=" * 60)
    print("PACA v5 Python Korean Language Processing Tests")
    print("=" * 60)

    test_results = []

    # 각 테스트 실행
    test_results.append(test_korean_text_validation())
    test_results.append(test_korean_name_validation())
    test_results.append(test_korean_phone_validation())
    test_results.append(test_korean_knowledge_items())
    test_results.append(test_korean_error_messages())

    # 결과 요약
    passed_tests = sum(test_results)
    total_tests = len(test_results)

    print("\n" + "=" * 60)
    print("Korean Processing Summary:")
    print("=" * 60)

    test_names = [
        "Korean Text Validation",
        "Korean Name Validation",
        "Korean Phone Validation",
        "Korean Knowledge Items",
        "Korean Error Messages"
    ]

    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "PASS" if result else "FAIL"
        print(f"{status}: {name}")

    print(f"\nOverall Result: {passed_tests}/{total_tests} test categories passed")

    if passed_tests == total_tests:
        print("\nAll Korean language processing tests PASSED!")
        print("PACA v5 Python system has excellent Korean language support.")
        print("Korean language processing improvement: VERIFIED")
    else:
        print(f"\n{total_tests - passed_tests} test categories FAILED!")
        print("Korean language processing needs improvement.")

    print("=" * 60)
    return passed_tests == total_tests, test_results


if __name__ == "__main__":
    run_korean_processing_tests()