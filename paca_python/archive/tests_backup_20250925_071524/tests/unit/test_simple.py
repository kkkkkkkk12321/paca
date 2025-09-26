#!/usr/bin/env python3
"""
간단한 모듈 테스트 스크립트
"""

import sys
import os
from pathlib import Path

# PACA 패키지 경로 추가
sys.path.insert(0, str(Path(__file__).parent / "paca"))

def test_basic_imports():
    """기본 모듈 import 테스트"""
    print("기본 모듈 import 테스트 시작...")

    try:
        # Core 모듈
        from paca.core.types.base import ID, Result
        print("  Core types 모듈 OK")

        # Services 모듈
        from paca.services import LearningService, MemoryService
        print("  Services 모듈 OK")

        # Reasoning 모듈
        from paca.reasoning import ReasoningChainManager
        print("  Reasoning 모듈 OK")

        # Mathematics 모듈
        from paca.mathematics import Calculator
        print("  Mathematics 모듈 OK")

        print("모든 기본 모듈 import 성공!")
        return True

    except Exception as e:
        print(f"Import 실패: {e}")
        return False

def test_basic_functionality():
    """기본 기능 테스트"""
    print("\n기본 기능 테스트 시작...")

    try:
        # Calculator 테스트
        from paca.mathematics import Calculator
        calc = Calculator()
        result = calc.add(2, 3)
        print(f"  Calculator: 2 + 3 = {result}")

        # Result 타입 테스트
        from paca.core.types.base import Result
        test_result = Result(success=True, data="test")
        print(f"  Result 타입: success={test_result.success}")

        print("기본 기능 테스트 성공!")
        return True

    except Exception as e:
        print(f"기능 테스트 실패: {e}")
        return False

def main():
    """메인 함수"""
    print("PACA Python 변환 간단 테스트")
    print("=" * 40)

    results = []

    # 1. Import 테스트
    results.append(test_basic_imports())

    # 2. 기능 테스트
    results.append(test_basic_functionality())

    # 결과
    passed = sum(results)
    total = len(results)

    print("\n" + "=" * 40)
    print(f"테스트 결과: {passed}/{total} 통과")

    if passed == total:
        print("모든 테스트 성공!")
        return 0
    else:
        print("일부 테스트 실패")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)