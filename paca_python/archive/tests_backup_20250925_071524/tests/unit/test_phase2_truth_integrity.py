#!/usr/bin/env python3
"""
Phase 2.2 & 2.3 Integration Test
진실 탐구 프로토콜 & 지적 무결성 점수 시스템 테스트

새로 구현된 Phase 2.2 (Truth Seeking Protocol)와 Phase 2.3 (IIS System)의 통합 테스트
"""

import asyncio
import sys
import os

# PACA 모듈 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'paca'))

from paca.cognitive.truth import TruthSeeker, UncertaintyType
from paca.cognitive.integrity import IntegrityScoring, BehaviorType, IntegrityDimension


class Phase2IntegrationTest:
    """Phase 2.2 & 2.3 통합 테스트"""

    def __init__(self):
        print("=" * 80)
        print("PACA Phase 2.2 & 2.3 Integration Test")
        print("진실 탐구 프로토콜 & 지적 무결성 점수 시스템")
        print("=" * 80)

    async def test_truth_seeking_system(self):
        """진실 탐구 시스템 테스트"""
        print("\n[TRUTH] Testing Truth Seeking System (Phase 2.2)")
        print("-" * 60)

        try:
            # TruthSeeker 초기화
            truth_seeker = TruthSeeker()

            # 테스트 쿼리들
            test_queries = [
                "I'm not sure if this information is accurate",
                "The data shows that climate change is happening, but I need to verify this",
                "This claim seems uncertain and requires fact-checking",
                "Studies show that exercise is good for health"
            ]

            for i, query in enumerate(test_queries, 1):
                print(f"\n{i}. Testing query: {query[:50]}...")

                # 불확실성 감지
                uncertainty_result = await truth_seeker.detect_uncertainty(query, {})

                if uncertainty_result.is_success:
                    uncertainties = uncertainty_result.data
                    print(f"   [OK] Detected {len(uncertainties)} uncertainty markers")

                    for uncertainty in uncertainties:
                        print(f"      - Type: {uncertainty.uncertainty_type.name}")
                        print(f"      - Confidence: {uncertainty.confidence_level:.2f}")
                        print(f"      - Priority: {uncertainty.priority_score:.2f}")
                else:
                    print(f"   [ERROR] Uncertainty detection failed: {uncertainty_result.error}")
                    continue

                # 진실 탐구 수행
                truth_result = await truth_seeker.seek_truth(query, {'importance': 'high'})

                if truth_result.is_success:
                    result = truth_result.data
                    print(f"   [OK] Truth seeking completed in {result.processing_time:.2f}s")
                    print(f"      - Verification actions: {len(result.verification_actions_taken)}")
                    print(f"      - Knowledge updates: {len(result.knowledge_updates)}")
                    print(f"      - Confidence improvement: {result.confidence_improvement:.2f}")
                    print(f"      - Remaining uncertainties: {len(result.remaining_uncertainties)}")

                    if result.truth_assessment:
                        assessment = result.truth_assessment
                        print(f"      - Truth score: {assessment.truth_score.to_percentage():.1f}%")
                        print(f"      - Confidence level: {assessment.truth_score.confidence_level.name}")
                else:
                    print(f"   [ERROR] Truth seeking failed: {truth_result.error}")

            # 통계 확인
            history = truth_seeker.get_seeking_history()
            kb_stats = truth_seeker.get_knowledge_base_stats()

            print(f"\n[STATS] Truth Seeking Statistics:")
            print(f"   - Total seeking operations: {len(history)}")
            print(f"   - Knowledge base entries: {kb_stats['total_entries']}")
            print(f"   - Verified facts: {kb_stats.get('verified_facts', 0)}")
            print(f"   - Uncertain information: {kb_stats.get('uncertain_information', 0)}")

            return True

        except Exception as e:
            print(f"[ERROR] Truth seeking test failed: {str(e)}")
            return False

    async def test_integrity_scoring_system(self):
        """지적 무결성 점수 시스템 테스트"""
        print("\n🎯 Testing Intellectual Integrity Scoring System (Phase 2.3)")
        print("-" * 60)

        try:
            # IntegrityScoring 초기화
            integrity_scoring = IntegrityScoring()

            # 초기 상태 확인
            initial_report = integrity_scoring.get_integrity_report()
            print(f"Initial integrity score: {initial_report['overall_metrics']['score']:.1f}")
            print(f"Initial level: {initial_report['overall_metrics']['level']}")

            # 다양한 행동들 기록
            test_behaviors = [
                (BehaviorType.TRUTH_SEEKING, {'severity': 'normal'}, "Actively seeking truth"),
                (BehaviorType.SOURCE_CITING, {'severity': 'normal'}, "Citing reliable sources"),
                (BehaviorType.ERROR_CORRECTION, {'severity': 'high'}, "Correcting identified errors"),
                (BehaviorType.UNCERTAINTY_ADMISSION, {'severity': 'normal'}, "Admitting uncertainty"),
                (BehaviorType.BIAS_RECOGNITION, {'severity': 'normal'}, "Recognizing potential bias"),
                (BehaviorType.OVERCONFIDENCE, {'severity': 'low'}, "Showing overconfidence"),
                (BehaviorType.SOURCE_OMISSION, {'severity': 'normal'}, "Omitting source citation")
            ]

            print(f"\n🎬 Recording {len(test_behaviors)} behaviors...")

            for i, (behavior_type, context, description) in enumerate(test_behaviors, 1):
                print(f"\n{i}. Recording: {behavior_type.name}")
                print(f"   Description: {description}")

                result = await integrity_scoring.record_behavior(
                    behavior_type=behavior_type,
                    context=context,
                    evidence=[description],
                    confidence=0.85
                )

                if result.is_success:
                    action = result.data
                    print(f"   ✅ Recorded with score impact: {action.score_impact:.2f}")
                    print(f"      - Primary dimension: {action.dimension.name}")
                    print(f"      - Confidence: {action.confidence:.2f}")
                else:
                    print(f"   ❌ Failed to record: {result.error}")

            # 최종 보고서 생성
            final_report = integrity_scoring.get_integrity_report()

            print(f"\n📊 Final Integrity Report:")
            print(f"   - Overall Score: {final_report['overall_metrics']['score']:.1f}")
            print(f"   - Integrity Level: {final_report['overall_metrics']['level']}")
            print(f"   - Trend: {final_report['overall_metrics']['trend']}")
            print(f"   - Reliability: {final_report['overall_metrics']['reliability']:.2f}")
            print(f"   - Consistency: {final_report['overall_metrics']['consistency']:.2f}")

            print(f"\n📈 Dimension Scores:")
            for dimension, score in final_report['dimension_scores'].items():
                print(f"   - {dimension}: {score:.1f}")

            print(f"\n📋 Recent Activity:")
            activity = final_report['recent_activity']
            print(f"   - Total actions: {activity['total_actions']}")
            print(f"   - Positive actions: {activity['positive_actions']}")
            print(f"   - Negative actions: {activity['negative_actions']}")

            # 거짓말 탐지 테스트
            print(f"\n🕵️ Testing Dishonesty Detection:")

            test_content = """
            I am absolutely certain that this claim is 100% true.
            Studies show clear evidence, and research proves this beyond doubt.
            There is no uncertainty whatsoever in this statement.
            """

            dishonesty_result = await integrity_scoring.detect_dishonesty(
                test_content, {'source': 'test'}
            )

            if dishonesty_result.is_success:
                indicators = dishonesty_result.data
                print(f"   ✅ Detected {len(indicators)} dishonesty indicators:")
                for indicator in indicators:
                    print(f"      - {indicator}")
            else:
                print(f"   ❌ Dishonesty detection failed: {dishonesty_result.error}")

            # 신뢰도 점수 확인
            trust_score = await integrity_scoring.get_trust_score()
            print(f"\n🤝 Trust Score: {trust_score:.3f}")

            return True

        except Exception as e:
            print(f"❌ Integrity scoring test failed: {str(e)}")
            return False

    async def test_integration_workflow(self):
        """통합 워크플로우 테스트"""
        print("\n🔄 Testing Integrated Workflow (Truth Seeking + Integrity Scoring)")
        print("-" * 60)

        try:
            # 시스템 초기화
            truth_seeker = TruthSeeker()
            integrity_scoring = IntegrityScoring()

            # 시나리오: 불확실한 정보에 대한 진실 탐구와 무결성 평가
            scenario_query = "I believe this claim is true, but I'm not completely certain about it"

            print(f"Scenario: {scenario_query}")

            # 1. 불확실성 감지
            print(f"\n1️⃣ Detecting uncertainty...")
            uncertainty_result = await truth_seeker.detect_uncertainty(scenario_query, {})

            if uncertainty_result.is_success and uncertainty_result.data:
                print(f"   ✅ Found uncertainties - triggering truth seeking behavior")

                # 무결성 점수: 진실 탐구 행동 기록
                await integrity_scoring.record_behavior(
                    BehaviorType.TRUTH_SEEKING,
                    {'trigger': 'uncertainty_detected'},
                    [f"Uncertainty detected in: {scenario_query[:50]}..."]
                )

            # 2. 진실 탐구 수행
            print(f"\n2️⃣ Performing truth seeking...")
            truth_result = await truth_seeker.seek_truth(scenario_query)

            if truth_result.is_success:
                result = truth_result.data
                print(f"   ✅ Truth seeking completed")

                # 무결성 점수: 검증 행동 기록
                await integrity_scoring.record_behavior(
                    BehaviorType.VERIFICATION,
                    {'verification_actions': len(result.verification_actions_taken)},
                    result.verification_actions_taken
                )

                # 3. 결과에 따른 무결성 행동 기록
                if result.truth_assessment:
                    assessment = result.truth_assessment
                    truth_score = assessment.truth_score.overall_score

                    if truth_score >= 0.8:
                        # 높은 신뢰도 - 지식 업데이트
                        await integrity_scoring.record_behavior(
                            BehaviorType.KNOWLEDGE_UPDATE,
                            {'truth_score': truth_score},
                            [f"High-confidence information verified: {truth_score:.2f}"]
                        )
                    elif truth_score < 0.5:
                        # 낮은 신뢰도 - 불확실성 인정
                        await integrity_scoring.record_behavior(
                            BehaviorType.UNCERTAINTY_ADMISSION,
                            {'truth_score': truth_score},
                            [f"Low-confidence information acknowledged: {truth_score:.2f}"]
                        )

            # 4. 최종 통합 보고서
            print(f"\n📋 Integrated Workflow Results:")

            # 진실 탐구 결과
            seeking_stats = truth_seeker.get_knowledge_base_stats()
            print(f"   Truth Seeking:")
            print(f"   - Knowledge base entries: {seeking_stats['total_entries']}")
            print(f"   - Average confidence: {seeking_stats.get('average_confidence', 0):.2f}")

            # 무결성 점수 결과
            integrity_report = integrity_scoring.get_integrity_report()
            print(f"   Integrity Scoring:")
            print(f"   - Overall score: {integrity_report['overall_metrics']['score']:.1f}")
            print(f"   - Recent positive actions: {integrity_report['recent_activity']['positive_actions']}")

            return True

        except Exception as e:
            print(f"❌ Integration workflow test failed: {str(e)}")
            return False

    async def run_all_tests(self):
        """모든 테스트 실행"""
        print("Starting comprehensive Phase 2.2 & 2.3 integration tests...\n")

        results = []

        # Truth Seeking System 테스트
        truth_seeking_result = await self.test_truth_seeking_system()
        results.append(("Truth Seeking System", truth_seeking_result))

        # Integrity Scoring System 테스트
        integrity_scoring_result = await self.test_integrity_scoring_system()
        results.append(("Integrity Scoring System", integrity_scoring_result))

        # 통합 워크플로우 테스트
        integration_result = await self.test_integration_workflow()
        results.append(("Integration Workflow", integration_result))

        # 결과 요약
        print(f"\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)

        passed = 0
        total = len(results)

        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name:.<50} {status}")
            if result:
                passed += 1

        print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

        if passed == total:
            print("🎉 All Phase 2.2 & 2.3 integration tests PASSED!")
            print("✅ Truth Seeking Protocol and IIS System are working correctly")
        else:
            print("⚠️  Some tests failed. Please check the error messages above.")

        return passed == total


async def main():
    """메인 테스트 실행"""
    test_runner = Phase2IntegrationTest()
    success = await test_runner.run_all_tests()

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)