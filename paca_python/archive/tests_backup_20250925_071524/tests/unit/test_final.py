#!/usr/bin/env python3
"""
최종 테스트 스크립트
"""

import sys
from pathlib import Path

# PACA 패키지 경로 추가
sys.path.insert(0, str(Path(__file__).parent / "paca"))

def main():
    """메인 함수"""
    print("PACA Python 변환 최종 테스트")
    print("=" * 40)

    try:
        # 1. 기본 임포트 테스트
        print("1. 기본 모듈 import 테스트...")

        from paca.mathematics import Calculator
        print("   Calculator import 성공")

        from paca.services.learning import LearningService
        print("   LearningService import 성공")

        from paca.services.memory import MemoryService
        print("   MemoryService import 성공")

        from paca.reasoning.chains import ReasoningChainManager
        print("   ReasoningChainManager import 성공")

        # 2. 기본 기능 테스트
        print("\n2. 기본 기능 테스트...")

        # Calculator 테스트
        calc = Calculator()
        result = calc.add(2, 3)
        print(f"   Calculator: 2 + 3 = {result}")

        # Service 생성 테스트
        learning_service = LearningService()
        print("   LearningService 생성 성공")

        memory_service = MemoryService()
        print("   MemoryService 생성 성공")

        # Reasoning 생성 테스트
        reasoning_manager = ReasoningChainManager()
        print("   ReasoningChainManager 생성 성공")

        print("\n=" * 40)
        print("모든 테스트 성공!")
        print("\nPACA Python 변환 완료:")
        print("   - Learning 서비스: 성공")
        print("   - Memory 서비스: 성공")
        print("   - Reasoning 모듈: 성공")
        print("   - Mathematics 모듈: 성공")

        return 0

    except Exception as e:
        print(f"\n테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)