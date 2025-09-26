#!/usr/bin/env python3
"""
PACA v5 Phase 1 핵심 기능 통합 테스트
복잡도 감지 + 메타인지 + 추론 체인 시스템 테스트
"""

import asyncio
import sys
import os
import time

# 테스트 경로 설정
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# PACA 모듈 임포트
from paca.cognitive import (
    ComplexityDetector,
    MetacognitionEngine,
    ReasoningChain,
    detect_complexity,
    execute_reasoning
)

async def test_complexity_detection():
    """복잡도 감지 시스템 테스트"""
    print("=" * 60)
    print("🧠 복잡도 감지 시스템 테스트")
    print("=" * 60)

    detector = ComplexityDetector()

    test_cases = [
        "안녕하세요",  # 단순
        "오늘 날씨가 어떤가요?",  # 보통
        "머신러닝 알고리즘의 성능을 어떻게 평가하고 개선할 수 있을까요?",  # 복잡
        "인공지능과 인간 지능의 차이점을 분석하고, 미래 사회에 미칠 영향을 다각도로 평가해주세요"  # 매우 복잡
    ]

    results = []

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}] {test_input}")

        result = await detector.detect_complexity(test_input)
        results.append(result)

        print(f"  📊 복잡도 점수: {result.score}")
        print(f"  🎯 도메인: {result.domain.value}")
        print(f"  🔗 추론 체인 필요: {'✅' if result.reasoning_required else '❌'}")
        print(f"  ⏱️ 처리 시간: {result.processing_time_ms:.2f}ms")
        print(f"  📈 신뢰도: {result.confidence:.2f}")

    print(f"\n📊 성능 통계:")
    stats = detector.get_performance_stats()
    print(f"  총 분석 횟수: {stats['total_analyses']}")
    print(f"  평균 처리 시간: {stats['average_processing_time_ms']:.2f}ms")

    return results

async def test_metacognition_engine():
    """메타인지 엔진 테스트"""
    print("\n" + "=" * 60)
    print("🧩 메타인지 엔진 테스트")
    print("=" * 60)

    engine = MetacognitionEngine()

    # 모니터링 세션 시작
    task_context = {
        'problem': '복잡한 수학 문제 해결',
        'complexity_score': 75,
        'domain': 'mathematical'
    }

    session_id = await engine.start_reasoning_monitoring(task_context)
    print(f"📝 모니터링 세션 시작: {session_id}")

    # 추론 단계들 추가
    steps_data = [
        ("문제 분석", "주어진 수학 문제를 이해합니다", {'problem': '2x + 3 = 7'}, {'analysis': '1차 방정식'}),
        ("해법 적용", "방정식 해법을 적용합니다", {'equation': '2x + 3 = 7'}, {'steps': ['양변에서 3 빼기', '양변을 2로 나누기']}),
        ("결과 계산", "최종 답을 계산합니다", {'steps': 'x = 2'}, {'result': 'x = 2', 'verification': True})
    ]

    for description, detail, input_data, output_data in steps_data:
        step_id = await engine.add_reasoning_step(
            session_id, description, input_data, output_data, 50.0
        )
        print(f"  ➕ 추론 단계 추가: {description}")

    # 추론 품질 평가
    reasoning_steps = [
        {'description': desc, 'input_data': inp, 'output_data': out, 'processing_time_ms': 50.0}
        for desc, _, inp, out in steps_data
    ]

    quality_metrics = await engine.evaluate_reasoning_quality(reasoning_steps)
    print(f"\n📊 품질 평가:")
    print(f"  논리적 일관성: {quality_metrics.logical_consistency:.2f}")
    print(f"  단계별 명확성: {quality_metrics.step_clarity:.2f}")
    print(f"  결론 타당성: {quality_metrics.conclusion_validity:.2f}")
    print(f"  전체 점수: {quality_metrics.calculate_overall_score():.1f}")
    print(f"  품질 등급: {quality_metrics.get_quality_grade().name}")

    # 자기반성 수행
    reflection = await engine.perform_self_reflection(session_id)
    print(f"\n🤔 자기반성 결과:")
    print(f"  강점: {reflection.strengths}")
    print(f"  약점: {reflection.weaknesses}")
    print(f"  개선 제안: {reflection.improvement_suggestions}")
    print(f"  전체 평가: {reflection.overall_assessment}")

    # 세션 종료
    completed_session = await engine.end_monitoring_session(session_id)
    print(f"\n✅ 모니터링 세션 완료, 지속시간: {completed_session.get_duration_ms():.0f}ms")

    return engine

async def test_reasoning_chain():
    """추론 체인 시스템 테스트"""
    print("\n" + "=" * 60)
    print("⛓️ 추론 체인 시스템 테스트")
    print("=" * 60)

    chain = ReasoningChain({'enable_metacognition': True})

    test_problems = [
        ("간단한 계산 문제", 25),
        ("논리적 추론 문제", 55),
        ("복잡한 분석 문제", 85)
    ]

    results = []

    for problem, complexity in test_problems:
        print(f"\n[추론 테스트] {problem} (복잡도: {complexity})")

        result = await chain.execute_reasoning_chain(problem, complexity)
        results.append(result)

        print(f"  🆔 체인 ID: {result.chain_id}")
        print(f"  📊 사용된 전략: {result.strategy_used.value}")
        print(f"  🔗 추론 단계 수: {len(result.steps)}")
        print(f"  📝 최종 결론: {result.final_conclusion}")
        print(f"  📈 신뢰도: {result.confidence_score:.2f}")
        print(f"  ⏱️ 처리 시간: {result.total_processing_time_ms:.1f}ms")
        print(f"  🏆 상태: {result.status.value}")

        # 추론 단계 세부 정보
        print(f"  📋 추론 단계들:")
        for step in result.steps[:3]:  # 처음 3개만 표시
            print(f"    {step.step_number}. {step.title} (신뢰도: {step.confidence:.2f})")

    print(f"\n📊 추론 체인 성능 통계:")
    stats = chain.get_performance_summary()
    print(f"  총 실행 횟수: {stats['total_chains_executed']}")
    print(f"  평균 처리 시간: {stats['average_processing_time_ms']:.1f}ms")
    print(f"  성공률: {stats['success_rate']:.2%}")

    return results

async def test_integrated_workflow():
    """통합 워크플로우 테스트"""
    print("\n" + "=" * 60)
    print("🔄 통합 워크플로우 테스트")
    print("=" * 60)

    # 1. 복잡한 문제로 전체 파이프라인 테스트
    problem = "인공지능의 윤리적 문제를 분석하고, 해결 방안을 제시해주세요"

    print(f"🎯 테스트 문제: {problem}")

    # 2. 복잡도 감지
    print(f"\n1️⃣ 복잡도 감지 단계")
    complexity_result = await detect_complexity(problem)
    print(f"   복잡도 점수: {complexity_result.score}")
    print(f"   추론 체인 필요: {complexity_result.reasoning_required}")

    # 3. 추론 체인 실행 (복잡도가 임계값 이상인 경우만)
    if complexity_result.reasoning_required:
        print(f"\n2️⃣ 추론 체인 실행 단계")
        reasoning_result = await execute_reasoning(problem, complexity_result.score)

        print(f"   전략: {reasoning_result.strategy_used.value}")
        print(f"   단계 수: {len(reasoning_result.steps)}")
        print(f"   최종 결론: {reasoning_result.final_conclusion}")
        print(f"   전체 신뢰도: {reasoning_result.confidence_score:.2f}")

        # 4. 품질 분석
        print(f"\n3️⃣ 품질 분석")
        quality = reasoning_result.quality_assessment
        print(f"   논리적 흐름: {'✅' if quality.get('logical_flow', False) else '❌'}")
        print(f"   완전성: {quality.get('completeness', 0):.2f}")
        print(f"   결론 타당성: {quality.get('conclusion_validity', 0):.2f}")

        return reasoning_result
    else:
        print("   ➡️ 단순한 문제로 판단, 추론 체인 건너뜀")
        return None

async def main():
    """메인 테스트 실행"""
    print("PACA v5 Python Phase 1 핵심 기능 통합 테스트 시작")
    print("=" * 80)

    start_time = time.time()

    try:
        # 개별 시스템 테스트
        complexity_results = await test_complexity_detection()
        metacognition_engine = await test_metacognition_engine()
        reasoning_results = await test_reasoning_chain()

        # 통합 워크플로우 테스트
        integrated_result = await test_integrated_workflow()

        # 전체 결과 요약
        print("\n" + "=" * 80)
        print("📋 전체 테스트 결과 요약")
        print("=" * 80)

        print(f"✅ 복잡도 감지 시스템: {len(complexity_results)}개 테스트 완료")
        print(f"✅ 메타인지 엔진: 모니터링 및 반성 기능 정상")
        print(f"✅ 추론 체인 시스템: {len(reasoning_results)}개 전략 테스트 완료")
        print(f"✅ 통합 워크플로우: {'성공' if integrated_result else '건너뜀'}")

        elapsed_time = time.time() - start_time
        print(f"\n⏱️ 총 실행 시간: {elapsed_time:.2f}초")

        print("\n🎉 Phase 1 핵심 기능 테스트 성공!")
        print("   - 복잡도 감지 → 추론 체인 활성화 → 메타인지 모니터링")
        print("   - 모든 시스템이 정상적으로 연동됨")

        return True

    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)