#!/usr/bin/env python3
"""
PACA v5 Phase 1 간단한 테스트
복잡도 감지, 메타인지, 추론 체인 기본 기능 검증
"""

import asyncio
import sys
import os

# 테스트 경로 설정
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

async def test_basic_imports():
    """기본 임포트 테스트"""
    print("=== 기본 임포트 테스트 ===")

    try:
        # 복잡도 감지 시스템
        from paca.cognitive.complexity_detector import ComplexityDetector
        print("ComplexityDetector 임포트 성공")

        # 메타인지 엔진
        from paca.cognitive.metacognition_engine import MetacognitionEngine
        print("MetacognitionEngine 임포트 성공")

        # 추론 체인
        from paca.cognitive.reasoning_chain import ReasoningChain
        print("ReasoningChain 임포트 성공")

        # 통합 임포트
        from paca.cognitive import ComplexityDetector, MetacognitionEngine, ReasoningChain
        print("통합 임포트 성공")

        return True

    except Exception as e:
        print(f"임포트 실패: {str(e)}")
        return False

async def test_complexity_detection():
    """복잡도 감지 기본 테스트"""
    print("\n=== 복잡도 감지 테스트 ===")

    try:
        from paca.cognitive.complexity_detector import ComplexityDetector

        detector = ComplexityDetector()

        # 간단한 테스트
        result = await detector.detect_complexity("안녕하세요")
        print(f"간단한 질문 - 복잡도: {result.score}, 추론 필요: {result.reasoning_required}")

        # 복잡한 테스트
        result = await detector.detect_complexity("머신러닝 알고리즘의 성능을 분석해주세요")
        print(f"복잡한 질문 - 복잡도: {result.score}, 추론 필요: {result.reasoning_required}")

        return True

    except Exception as e:
        print(f"복잡도 감지 테스트 실패: {str(e)}")
        return False

async def test_metacognition():
    """메타인지 기본 테스트"""
    print("\n=== 메타인지 테스트 ===")

    try:
        from paca.cognitive.metacognition_engine import MetacognitionEngine

        engine = MetacognitionEngine()

        # 모니터링 세션 시작
        session_id = await engine.start_reasoning_monitoring({'problem': '테스트 문제'})
        print(f"모니터링 세션 시작: {session_id}")

        # 세션 종료
        session = await engine.end_monitoring_session(session_id)
        print(f"세션 완료, 지속시간: {session.get_duration_ms():.1f}ms")

        return True

    except Exception as e:
        print(f"메타인지 테스트 실패: {str(e)}")
        return False

async def test_reasoning_chain():
    """추론 체인 기본 테스트"""
    print("\n=== 추론 체인 테스트 ===")

    try:
        from paca.cognitive.reasoning_chain import ReasoningChain

        chain = ReasoningChain()

        # 간단한 추론 실행
        result = await chain.execute_reasoning_chain("간단한 문제", 25)
        print(f"추론 완료 - 단계 수: {len(result.steps)}, 신뢰도: {result.confidence_score:.2f}")

        return True

    except Exception as e:
        print(f"추론 체인 테스트 실패: {str(e)}")
        return False

async def main():
    """메인 테스트 실행"""
    print("PACA v5 Phase 1 기본 기능 테스트 시작")
    print("=" * 50)

    tests = [
        test_basic_imports,
        test_complexity_detection,
        test_metacognition,
        test_reasoning_chain
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            success = await test()
            if success:
                passed += 1
                print("테스트 통과\n")
            else:
                print("테스트 실패\n")
        except Exception as e:
            print(f"테스트 오류: {str(e)}\n")

    print("=" * 50)
    print(f"테스트 결과: {passed}/{total} 통과")

    if passed == total:
        print("모든 핵심 기능이 정상적으로 작동합니다!")
        return True
    else:
        print("일부 기능에 문제가 있습니다.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)