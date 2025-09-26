#!/usr/bin/env python3
"""
Simple Phase 2.2 & 2.3 Test
ASCII-only test for Windows console compatibility
"""

import asyncio
import sys
import os

# PACA 모듈 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'paca'))

from paca.cognitive.truth import TruthSeeker, UncertaintyType
from paca.cognitive.integrity import IntegrityScoring, BehaviorType


async def test_truth_seeking():
    """진실 탐구 시스템 간단 테스트"""
    print("Testing Truth Seeking System...")

    try:
        truth_seeker = TruthSeeker()

        # 간단한 테스트 쿼리
        query = "I'm not sure if this information is accurate"

        # 불확실성 감지
        result = await truth_seeker.detect_uncertainty(query, {})

        if result.is_success:
            print(f"  [OK] Detected {len(result.data)} uncertainty markers")
            return True
        else:
            print(f"  [ERROR] {result.error}")
            return False

    except Exception as e:
        print(f"  [ERROR] {str(e)}")
        return False


async def test_integrity_scoring():
    """지적 무결성 점수 시스템 간단 테스트"""
    print("Testing Integrity Scoring System...")

    try:
        integrity_scoring = IntegrityScoring()

        # 행동 기록 테스트
        result = await integrity_scoring.record_behavior(
            BehaviorType.TRUTH_SEEKING,
            {'severity': 'normal'},
            ['Test evidence']
        )

        if result.is_success:
            action = result.data
            print(f"  [OK] Recorded behavior with impact: {action.score_impact:.2f}")

            # 보고서 생성
            report = integrity_scoring.get_integrity_report()
            print(f"  [OK] Current integrity score: {report['overall_metrics']['score']:.1f}")
            return True
        else:
            print(f"  [ERROR] {result.error}")
            return False

    except Exception as e:
        print(f"  [ERROR] {str(e)}")
        return False


async def main():
    """메인 테스트"""
    print("=" * 60)
    print("PACA Phase 2.2 & 2.3 Simple Test")
    print("=" * 60)

    results = []

    # Truth Seeking 테스트
    truth_result = await test_truth_seeking()
    results.append(("Truth Seeking", truth_result))

    # Integrity Scoring 테스트
    integrity_result = await test_integrity_scoring()
    results.append(("Integrity Scoring", integrity_result))

    # 결과 요약
    print("\nTest Results:")
    print("-" * 40)

    passed = 0
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("All tests PASSED!")
        return True
    else:
        print("Some tests FAILED!")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)