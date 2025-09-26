#!/usr/bin/env python3
"""
PACA v5 Phase 2 통합 테스트
자율 학습 시스템 기능 테스트

Phase 2 핵심 기능:
- IIS 점수 계산 시스템
- 자율 훈련 시스템
- 전술/휴리스틱 자동 생성
"""

import asyncio
import sys
import time
from typing import Dict, Any

# PACA Phase 2 학습 시스템 임포트
try:
    from paca.learning import (
        # IIS Calculator
        IISCalculator, create_sample_learning_data, create_sample_interaction_result,

        # Autonomous Trainer
        AutonomousTrainer, create_sample_weakness_area, TrainingConfig,

        # Tactic Generator
        TacticGenerator, create_sample_interaction_data, create_sample_tactic
    )
    print("✅ Phase 2 임포트 성공")
except ImportError as e:
    print(f"❌ Phase 2 임포트 실패: {e}")
    sys.exit(1)


async def test_iis_calculator():
    """IIS 점수 계산 시스템 테스트"""
    print("\n=== IIS 점수 계산 시스템 테스트 ===")

    try:
        # 1. IIS 계산기 초기화
        calculator = IISCalculator()
        print("✅ IIS 계산기 초기화 성공")

        # 2. 샘플 학습 데이터 생성
        learning_data = create_sample_learning_data()
        print(f"✅ 학습 데이터 생성: {learning_data.interactions_count}회 상호작용")

        # 3. IIS 점수 계산
        start_time = time.time()
        result = await calculator.calculate_iis_score(learning_data)
        calculation_time = (time.time() - start_time) * 1000

        if result.is_success:
            iis_score = result.value
            print(f"✅ IIS 점수 계산 성공: {iis_score.current_score}점 ({calculation_time:.1f}ms)")
            print(f"   추세: {iis_score.trend.value}")
            print(f"   등급: {iis_score.get_grade()}")
            print(f"   신뢰도: {iis_score.confidence:.2f}")

            # 세부 점수 출력
            breakdown = iis_score.breakdown
            print(f"   세부 점수:")
            print(f"     전술 숙련도: {breakdown.tactic_mastery:.1f}")
            print(f"     문제 해결: {breakdown.problem_solving:.1f}")
            print(f"     추론 품질: {breakdown.reasoning_quality:.1f}")
            print(f"     학습 속도: {breakdown.learning_speed:.1f}")
            print(f"     적응 능력: {breakdown.adaptation_ability:.1f}")

            # 개선 제안
            suggestions = calculator.get_improvement_suggestions(iis_score)
            if suggestions:
                print(f"   개선 제안: {len(suggestions)}개")
                for i, suggestion in enumerate(suggestions[:2], 1):
                    print(f"     {i}. {suggestion}")
        else:
            print(f"❌ IIS 점수 계산 실패: {result.error}")
            return False

        # 4. 상호작용 결과 업데이트 테스트
        interaction_result = create_sample_interaction_result()
        update_result = await calculator.update_iis_from_interaction(interaction_result)

        if update_result.is_success:
            print("✅ 상호작용 결과 업데이트 성공")
        else:
            print(f"❌ 상호작용 결과 업데이트 실패: {update_result.error}")
            return False

        return True

    except Exception as e:
        print(f"❌ IIS 계산 시스템 테스트 오류: {e}")
        return False


async def test_autonomous_trainer():
    """자율 훈련 시스템 테스트"""
    print("\n=== 자율 훈련 시스템 테스트 ===")

    try:
        # 1. 자율 훈련기 초기화
        trainer = AutonomousTrainer()
        print("✅ 자율 훈련기 초기화 성공")

        # 2. 약점 분석
        start_time = time.time()
        weakness_result = await trainer.analyze_weaknesses()
        analysis_time = (time.time() - start_time) * 1000

        if weakness_result.is_success:
            weaknesses = weakness_result.value
            print(f"✅ 약점 분석 완료: {len(weaknesses)}개 발견 ({analysis_time:.1f}ms)")

            for i, weakness in enumerate(weaknesses, 1):
                print(f"   {i}. {weakness.weakness_type.value}: {weakness.current_score:.1f}점 "
                      f"(목표: {weakness.target_score:.1f}점, 심각도: {weakness.severity:.2f})")
        else:
            print(f"❌ 약점 분석 실패: {weakness_result.error}")
            return False

        # 3. 훈련 임무 생성
        if weaknesses:
            mission_result = await trainer.generate_training_missions(weaknesses[:2])  # 상위 2개만

            if mission_result.is_success:
                missions = mission_result.value
                print(f"✅ 훈련 임무 생성: {len(missions)}개")

                for i, mission in enumerate(missions[:3], 1):  # 상위 3개만 출력
                    print(f"   {i}. {mission.description}")
                    print(f"      난이도: {mission.difficulty}/5, 예상 시간: {mission.estimated_duration_minutes}분")
            else:
                print(f"❌ 훈련 임무 생성 실패: {mission_result.error}")
                return False

        # 4. 짧은 훈련 세션 실행
        config = TrainingConfig(
            max_cycles=2,  # 테스트용으로 줄임
            max_session_duration_minutes=5,
            min_improvement_threshold=0.01
        )

        print("🔄 짧은 훈련 세션 시작...")
        start_time = time.time()
        session_result = await trainer.execute_continuous_training(config)
        session_time = time.time() - start_time

        if session_result.is_success:
            session = session_result.value
            print(f"✅ 훈련 세션 완료 ({session_time:.1f}초)")
            print(f"   총 임무: {session.total_missions}, 완료: {session.completed_missions}")
            print(f"   성공률: {session.completed_missions/session.total_missions*100:.1f}%")
            print(f"   전체 개선도: {session.overall_improvement:.3f}")
            print(f"   요약: {session.session_summary}")
        else:
            print(f"❌ 훈련 세션 실패: {session_result.error}")
            return False

        # 5. 훈련 통계 확인
        stats = trainer.get_training_statistics()
        print(f"✅ 훈련 통계:")
        print(f"   총 세션: {stats['total_sessions']}")
        print(f"   총 임무: {stats['total_missions']}")
        print(f"   평균 개선도: {stats['average_improvement']:.3f}")

        return True

    except Exception as e:
        print(f"❌ 자율 훈련 시스템 테스트 오류: {e}")
        return False


async def test_tactic_generator():
    """전술/휴리스틱 생성 시스템 테스트"""
    print("\n=== 전술/휴리스틱 생성 시스템 테스트 ===")

    try:
        # 1. 전술 생성기 초기화
        generator = TacticGenerator()
        print("✅ 전술 생성기 초기화 성공")

        # 2. 샘플 상호작용 데이터 생성
        interaction_data = create_sample_interaction_data()
        print(f"✅ 상호작용 데이터 생성: {len(interaction_data)}개")

        # 3. 성공 패턴에서 전술 추출
        start_time = time.time()
        tactic_result = await generator.extract_successful_patterns(interaction_data)
        extraction_time = (time.time() - start_time) * 1000

        if tactic_result.is_success:
            tactics = tactic_result.value
            print(f"✅ 전술 추출 완료: {len(tactics)}개 ({extraction_time:.1f}ms)")

            for i, tactic in enumerate(tactics, 1):
                print(f"   {i}. {tactic.name} ({tactic.tactic_type.value})")
                print(f"      상태: {tactic.status.value}, 숙련도: {tactic.proficiency:.2f}")
                print(f"      성공률: {tactic.success_rate:.2f}, 효과성: {tactic.effectiveness_score:.2f}")
        else:
            print(f"❌ 전술 추출 실패: {tactic_result.error}")
            return False

        # 4. 실패 사례에서 휴리스틱 생성
        failure_data = [
            {
                'context': 'rushed decision making',
                'actions': ['quick_guess', 'skip_verification'],
                'success': False,
                'timestamp': time.time()
            }
        ]

        heuristic_result = await generator.generate_heuristics(failure_data)

        if heuristic_result.is_success:
            heuristics = heuristic_result.value
            print(f"✅ 휴리스틱 생성 완료: {len(heuristics)}개")

            for i, heuristic in enumerate(heuristics, 1):
                print(f"   {i}. {heuristic.name} ({heuristic.heuristic_type.value})")
                print(f"      조건: {heuristic.condition}")
                print(f"      행동: {heuristic.action}")
                print(f"      신뢰도: {heuristic.confidence:.2f}")
        else:
            print(f"❌ 휴리스틱 생성 실패: {heuristic_result.error}")
            return False

        # 5. 컨텍스트 기반 전술 추천 테스트
        recommendations = await generator.get_recommended_tactics("complex analytical problem", difficulty=0.7)

        if recommendations.is_success:
            recommended_tactics = recommendations.value
            print(f"✅ 전술 추천: {len(recommended_tactics)}개")

            for i, tactic in enumerate(recommended_tactics, 1):
                print(f"   {i}. {tactic.name} (효과성: {tactic.effectiveness_score:.2f})")
        else:
            print(f"❌ 전술 추천 실패: {recommendations.error}")
            return False

        # 6. 전술 숙련도 업데이트 테스트
        if tactics:
            tactic_usage = {
                tactics[0].tactic_id: {
                    'usage_count': 25,
                    'success_rate': 0.88,
                    'proficiency': 0.85
                }
            }

            mastery_result = await generator.update_mastery_levels(tactic_usage)

            if mastery_result.is_success:
                updated_tactics = mastery_result.value
                print(f"✅ 숙련도 업데이트: {len(updated_tactics)}개 전술 변경")
            else:
                print(f"❌ 숙련도 업데이트 실패: {mastery_result.error}")
                return False

        # 7. 학습 통계 확인
        stats = generator.get_learning_statistics()
        print(f"✅ 학습 통계:")
        print(f"   총 전술: {stats['total_tactics']}")
        print(f"   숙련된 전술: {stats['mastered_tactics']}")
        print(f"   숙련률: {stats['mastery_rate']:.1%}")
        print(f"   활성 휴리스틱: {stats['active_heuristics']}")
        print(f"   평균 숙련도: {stats['average_proficiency']:.2f}")

        return True

    except Exception as e:
        print(f"❌ 전술/휴리스틱 생성 시스템 테스트 오류: {e}")
        return False


async def test_integration():
    """통합 테스트 - 시스템간 연동"""
    print("\n=== 시스템 통합 테스트 ===")

    try:
        # 1. 모든 시스템 초기화
        calculator = IISCalculator()
        trainer = AutonomousTrainer(calculator)
        generator = TacticGenerator()

        print("✅ 모든 시스템 초기화 성공")

        # 2. IIS 기반 약점 분석
        learning_data = create_sample_learning_data()
        iis_result = await calculator.calculate_iis_score(learning_data)

        if not iis_result.is_success:
            print("❌ IIS 계산 실패")
            return False

        iis_score = iis_result.value
        print(f"✅ 통합 IIS 점수: {iis_score.current_score}점")

        # 3. 약점 기반 훈련 계획
        weakness_result = await trainer.analyze_weaknesses(learning_data)

        if weakness_result.is_success and weakness_result.value:
            print(f"✅ 약점 기반 훈련 계획 수립: {len(weakness_result.value)}개 약점")
        else:
            print("ℹ️ 발견된 약점 없음 (양호한 상태)")

        # 4. 전술 생성 및 추천
        interaction_data = create_sample_interaction_data()
        tactic_result = await generator.extract_successful_patterns(interaction_data)

        if tactic_result.is_success:
            tactics = tactic_result.value
            print(f"✅ 통합 전술 시스템: {len(tactics)}개 전술 추출")

        # 5. 시스템 성능 요약
        print("\n📊 통합 시스템 성능 요약:")
        print(f"   IIS 점수: {iis_score.current_score}점 ({iis_score.get_grade()}급)")
        print(f"   강점 영역: {iis_score.get_strongest_area()[0]} ({iis_score.get_strongest_area()[1]:.1f}점)")
        print(f"   개선 영역: {iis_score.get_weakest_area()[0]} ({iis_score.get_weakest_area()[1]:.1f}점)")
        print(f"   활용 가능 전술: {len(tactics) if tactic_result.is_success else 0}개")
        print(f"   시스템 신뢰도: {iis_score.confidence:.1%}")

        return True

    except Exception as e:
        print(f"❌ 통합 테스트 오류: {e}")
        return False


async def main():
    """메인 테스트 실행"""
    print("🚀 PACA v5 Phase 2 통합 테스트 시작")
    print("=" * 50)

    test_results = []

    # 개별 시스템 테스트
    test_results.append(await test_iis_calculator())
    test_results.append(await test_autonomous_trainer())
    test_results.append(await test_tactic_generator())

    # 통합 테스트
    test_results.append(await test_integration())

    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약")

    test_names = [
        "IIS 점수 계산 시스템",
        "자율 훈련 시스템",
        "전술/휴리스틱 생성 시스템",
        "시스템 통합"
    ]

    passed_tests = 0
    for i, (name, result) in enumerate(zip(test_names, test_results), 1):
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{i}. {name}: {status}")
        if result:
            passed_tests += 1

    success_rate = passed_tests / len(test_results) * 100
    print(f"\n전체 성공률: {passed_tests}/{len(test_results)} ({success_rate:.1f}%)")

    if success_rate == 100:
        print("🎉 모든 Phase 2 핵심 기능이 정상적으로 작동합니다!")
        print("\n✅ Phase 2 자율 학습 시스템 구현 완료:")
        print("   - IIS 점수 계산으로 AI 학습 수준 정량화")
        print("   - 자율 훈련으로 약점 자동 개선")
        print("   - 전술/휴리스틱 자동 생성으로 지능 축적")
        print("   - 모든 시스템의 유기적 통합 확인")
    else:
        print(f"⚠️ {len(test_results) - passed_tests}개 테스트가 실패했습니다.")
        return 1

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 테스트가 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 예상치 못한 오류: {e}")
        sys.exit(1)