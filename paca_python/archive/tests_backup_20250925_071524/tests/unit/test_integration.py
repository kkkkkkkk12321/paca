#!/usr/bin/env python3
"""
통합 테스트 스크립트
새로 변환된 모듈들의 기본 기능을 테스트합니다.
"""

import sys
import os
import asyncio
import traceback
from pathlib import Path

# PACA 패키지 경로 추가
sys.path.insert(0, str(Path(__file__).parent / "paca"))

def test_imports():
    """모듈 import 테스트"""
    print("모듈 import 테스트 시작...")

    try:
        # Services 모듈 테스트
        print("  Services 모듈...")
        from paca.services import (
            LearningService, MemoryService,
            AuthenticationService, KnowledgeService
        )
        print("    Services 모듈 import 성공")

        # Reasoning 모듈 테스트
        print("  Reasoning 모듈...")
        from paca.reasoning import (
            ReasoningChainManager, ReasoningMethod,
            ReasoningStepType, ReasoningChain
        )
        print("    Reasoning 모듈 import 성공")

        # Mathematics 모듈 테스트
        print("  Mathematics 모듈...")
        from paca.mathematics import (
            MathematicalReasoningEngine, MathQualityEvaluator,
            MathematicalDomain, Calculator
        )
        print("    Mathematics 모듈 import 성공")

        # Core 모듈 테스트
        print("  Core 모듈...")
        from paca.core.types.base import ID, Result
        from paca.core.errors.base import ValidationError
        from paca.core.events.base import EventEmitter
        print("    Core 모듈 import 성공")

        return True

    except Exception as e:
        print(f"    Import 실패: {e}")
        traceback.print_exc()
        return False

async def test_learning_service():
    """Learning 서비스 테스트"""
    print("Learning 서비스 테스트...")

    try:
        from paca.services import LearningService, SessionType, SessionConfig

        # 서비스 생성
        service = LearningService()
        await service.initialize()

        # 세션 생성 요청
        from paca.services.learning import CreateSessionRequest
        request = CreateSessionRequest(
            user_id="test_user",
            title="테스트 학습 세션",
            description="통합 테스트용 학습 세션",
            type=SessionType.PRACTICE,
            config=SessionConfig()
        )

        # 세션 생성
        result = await service._handle_create_session(
            service.create_context("test_req", "test_user", {}),
            request
        )

        if result.success:
            print("    Learning 서비스 기본 기능 동작")
            return True
        else:
            print(f"    Learning 서비스 실패: {result.error}")
            return False

    except Exception as e:
        print(f"    Learning 서비스 오류: {e}")
        traceback.print_exc()
        return False

async def test_memory_service():
    """Memory 서비스 테스트"""
    print("Memory 서비스 테스트...")

    try:
        from paca.services import MemoryService, MemoryType

        # 서비스 생성
        service = MemoryService()

        # 메모리 저장
        result = await service.store(
            memory_type=MemoryType.SHORT_TERM,
            content="테스트 메모리 항목",
            options={
                "tags": ["test", "integration"],
                "metadata": {"test": True}
            }
        )

        if result.success:
            memory_item = result.data
            print(f"    ✅ 메모리 저장 성공: {memory_item.id}")

            # 메모리 조회
            from paca.services.memory import MemoryQuery
            query = MemoryQuery(type=MemoryType.SHORT_TERM)
            retrieve_result = await service.retrieve(query)

            if retrieve_result.success and retrieve_result.data:
                print(f"    ✅ 메모리 조회 성공: {len(retrieve_result.data)}개 항목")
                return True
            else:
                print("    ❌ 메모리 조회 실패")
                return False
        else:
            print(f"    ❌ Memory 서비스 실패: {result.error}")
            return False

    except Exception as e:
        print(f"    ❌ Memory 서비스 오류: {e}")
        traceback.print_exc()
        return False

async def test_reasoning_chain():
    """Reasoning Chain 테스트"""
    print("🔗 Reasoning Chain 테스트...")

    try:
        from paca.reasoning import ReasoningChainManager, ReasoningMethod

        # 체인 관리자 생성
        manager = ReasoningChainManager()

        # 추론 체인 생성
        result = await manager.create_chain(
            name="테스트 추론 체인",
            start_premise="사람은 죽는다",
            method=ReasoningMethod.DEDUCTIVE,
            options={
                "description": "간단한 삼단논법 테스트"
            }
        )

        if result.success:
            chain = result.data
            print(f"    ✅ 추론 체인 생성 성공: {chain.id}")

            # 통계 조회
            stats = manager.get_chain_statistics()
            print(f"    ✅ 통계 조회 성공: {stats['total_chains']}개 체인")
            return True
        else:
            print(f"    ❌ Reasoning Chain 실패: {result.error}")
            return False

    except Exception as e:
        print(f"    ❌ Reasoning Chain 오류: {e}")
        traceback.print_exc()
        return False

def test_mathematics():
    """Mathematics 모듈 테스트"""
    print("📐 Mathematics 모듈 테스트...")

    try:
        from paca.mathematics import (
            MathematicalReasoningEngine, MathematicalExpression,
            MathematicalDomain, Calculator
        )

        # 수학 추론 엔진 생성
        engine = MathematicalReasoningEngine()

        # 수학 표현식 생성
        expression = MathematicalExpression(
            expression="2 + 2",
            variables=[],
            domain=MathematicalDomain.ALGEBRA,
            metadata={"test": True}
        )

        # 문제 해결
        solution = engine.solve(expression)
        print(f"    ✅ 수학 문제 해결 성공: {solution.problem_id}")

        # 해결책 평가
        evaluation = engine.evaluate_solution(solution)
        print(f"    ✅ 해결책 평가 성공: 점수 {evaluation.overall_score:.2f}")

        # 계산기 테스트
        calc = Calculator()
        result = calc.add(2, 3)
        print(f"    ✅ 계산기 테스트 성공: 2 + 3 = {result}")

        return True

    except Exception as e:
        print(f"    ❌ Mathematics 모듈 오류: {e}")
        traceback.print_exc()
        return False

async def main():
    """메인 테스트 함수"""
    print("PACA Python 변환 통합 테스트 시작\n")

    test_results = []

    # 1. Import 테스트
    test_results.append(test_imports())
    print()

    # 2. Learning 서비스 테스트
    test_results.append(await test_learning_service())
    print()

    # 3. Memory 서비스 테스트
    test_results.append(await test_memory_service())
    print()

    # 4. Reasoning Chain 테스트
    test_results.append(await test_reasoning_chain())
    print()

    # 5. Mathematics 모듈 테스트
    test_results.append(test_mathematics())
    print()

    # 결과 요약
    passed = sum(test_results)
    total = len(test_results)

    print("📊 테스트 결과 요약:")
    print(f"  ✅ 성공: {passed}/{total}")
    print(f"  ❌ 실패: {total - passed}/{total}")

    if passed == total:
        print("\n🎉 모든 테스트 통과! PACA Python 변환이 성공적으로 완료되었습니다.")
        return 0
    else:
        print(f"\n⚠️  일부 테스트 실패. {total - passed}개 이슈를 확인하고 수정해주세요.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)