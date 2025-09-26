#!/usr/bin/env python3
"""
PACA v5 Phase 2 직접 테스트
Phase 2 모듈들만 독립적으로 테스트
"""

import asyncio
import sys
import os
import time

# PYTHONPATH 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Phase 2 모듈만 직접 임포트
try:
    # 기본 타입만 먼저 임포트
    from paca.core.types.base import (
        create_success, create_failure, current_timestamp, generate_id
    )
    print("기본 타입 임포트 성공")

    # IIS Calculator 직접 임포트
    sys.path.append(os.path.join(os.path.dirname(__file__), 'paca', 'learning'))

    import iis_calculator
    import autonomous_trainer
    import tactic_generator

    print("Phase 2 모듈 직접 임포트 성공")

except Exception as e:
    print(f"모듈 임포트 실패: {e}")
    sys.exit(1)


async def test_iis_calculator_direct():
    """IIS 계산기 직접 테스트"""
    print("\n=== IIS 계산기 직접 테스트 ===")

    try:
        # IIS 계산기 생성
        calculator = iis_calculator.IISCalculator()

        # 샘플 학습 데이터 생성
        learning_data = iis_calculator.create_sample_learning_data()
        print(f"학습 데이터 생성: {learning_data.interactions_count}회 상호작용")

        # IIS 점수 계산
        start_time = time.time()
        result = await calculator.calculate_iis_score(learning_data)
        calc_time = (time.time() - start_time) * 1000

        if result.is_success:
            iis_score = result.value
            print(f"IIS 점수 계산 성공: {iis_score.current_score}점 ({calc_time:.1f}ms)")
            print(f"등급: {iis_score.get_grade()}")
            print(f"추세: {iis_score.trend.value}")
            print(f"신뢰도: {iis_score.confidence:.2f}")

            # 세부 점수 확인
            breakdown = iis_score.breakdown
            print(f"세부 점수:")
            print(f"  전술 숙련도: {breakdown.tactic_mastery:.1f}")
            print(f"  문제 해결: {breakdown.problem_solving:.1f}")
            print(f"  추론 품질: {breakdown.reasoning_quality:.1f}")
            print(f"  학습 속도: {breakdown.learning_speed:.1f}")
            print(f"  적응 능력: {breakdown.adaptation_ability:.1f}")

            # 개선 제안
            suggestions = calculator.get_improvement_suggestions(iis_score)
            print(f"개선 제안: {len(suggestions)}개")
            for i, suggestion in enumerate(suggestions[:2], 1):
                print(f"  {i}. {suggestion}")

            return True
        else:
            print(f"IIS 점수 계산 실패: {result.error}")
            return False

    except Exception as e:
        print(f"IIS 계산기 테스트 오류: {e}")
        return False


async def test_autonomous_trainer_direct():
    """자율 훈련기 직접 테스트"""
    print("\n=== 자율 훈련기 직접 테스트 ===")

    try:
        # 자율 훈련기 생성
        trainer = autonomous_trainer.AutonomousTrainer()

        # 약점 분석
        weakness_result = await trainer.analyze_weaknesses()

        if weakness_result.is_success:
            weaknesses = weakness_result.value
            print(f"약점 분석 성공: {len(weaknesses)}개 발견")

            for i, weakness in enumerate(weaknesses, 1):
                print(f"  {i}. {weakness.weakness_type.value}: {weakness.current_score:.1f}점 "
                      f"(목표: {weakness.target_score:.1f}점)")

            if weaknesses:
                # 훈련 임무 생성
                mission_result = await trainer.generate_training_missions(weaknesses[:1])

                if mission_result.is_success:
                    missions = mission_result.value
                    print(f"훈련 임무 생성: {len(missions)}개")

                    for i, mission in enumerate(missions[:2], 1):
                        print(f"  {i}. {mission.description} (난이도: {mission.difficulty}/5)")

                    # 짧은 훈련 세션 실행
                    config = autonomous_trainer.TrainingConfig(
                        max_cycles=1,
                        max_session_duration_minutes=1
                    )

                    session_result = await trainer.execute_continuous_training(config)

                    if session_result.is_success:
                        session = session_result.value
                        print(f"훈련 세션 완료:")
                        print(f"  총 임무: {session.total_missions}")
                        print(f"  완료: {session.completed_missions}")
                        print(f"  실패: {session.failed_missions}")
                        print(f"  전체 개선도: {session.overall_improvement:.3f}")

                        return True
                    else:
                        print("훈련 세션 실패")
                        return False
                else:
                    print("훈련 임무 생성 실패")
                    return False
            else:
                print("발견된 약점 없음 (양호한 상태)")
                return True
        else:
            print(f"약점 분석 실패: {weakness_result.error}")
            return False

    except Exception as e:
        print(f"자율 훈련기 테스트 오류: {e}")
        return False


async def test_tactic_generator_direct():
    """전술 생성기 직접 테스트"""
    print("\n=== 전술 생성기 직접 테스트 ===")

    try:
        # 전술 생성기 생성
        generator = tactic_generator.TacticGenerator()

        # 샘플 상호작용 데이터 생성
        interaction_data = tactic_generator.create_sample_interaction_data()
        print(f"상호작용 데이터 생성: {len(interaction_data)}개")

        # 성공 패턴에서 전술 추출
        tactic_result = await generator.extract_successful_patterns(interaction_data)

        if tactic_result.is_success:
            tactics = tactic_result.value
            print(f"전술 추출 성공: {len(tactics)}개")

            for i, tactic in enumerate(tactics, 1):
                print(f"  {i}. {tactic.name} ({tactic.tactic_type.value})")
                print(f"     상태: {tactic.status.value}, 성공률: {tactic.success_rate:.2f}")

            # 실패 사례에서 휴리스틱 생성
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
                print(f"휴리스틱 생성 성공: {len(heuristics)}개")

                for i, heuristic in enumerate(heuristics, 1):
                    print(f"  {i}. {heuristic.name} ({heuristic.heuristic_type.value})")
                    print(f"     조건: {heuristic.condition}")
                    print(f"     행동: {heuristic.action}")

                # 학습 통계
                stats = generator.get_learning_statistics()
                print(f"학습 통계:")
                print(f"  총 전술: {stats['total_tactics']}")
                print(f"  숙련된 전술: {stats['mastered_tactics']}")
                print(f"  숙련률: {stats['mastery_rate']:.1%}")
                print(f"  활성 휴리스틱: {stats['active_heuristics']}")

                return True
            else:
                print("휴리스틱 생성 실패")
                return False
        else:
            print("전술 추출 실패")
            return False

    except Exception as e:
        print(f"전술 생성기 테스트 오류: {e}")
        return False


async def test_integration_direct():
    """통합 테스트"""
    print("\n=== Phase 2 통합 테스트 ===")

    try:
        # 모든 시스템 초기화
        calculator = iis_calculator.IISCalculator()
        trainer = autonomous_trainer.AutonomousTrainer(calculator)
        generator = tactic_generator.TacticGenerator()

        print("모든 시스템 초기화 성공")

        # IIS 기반 통합 워크플로
        learning_data = iis_calculator.create_sample_learning_data()
        iis_result = await calculator.calculate_iis_score(learning_data)

        if iis_result.is_success:
            iis_score = iis_result.value
            print(f"통합 IIS 점수: {iis_score.current_score}점")

            # 상호작용 업데이트
            interaction_result = iis_calculator.create_sample_interaction_result()
            update_result = await calculator.update_iis_from_interaction(interaction_result)

            if update_result.is_success:
                print("상호작용 결과 업데이트 성공")

            # 전술 추천
            interaction_data = tactic_generator.create_sample_interaction_data()
            tactic_result = await generator.extract_successful_patterns(interaction_data)

            if tactic_result.is_success:
                tactics = tactic_result.value
                if tactics:
                    recommendation_result = await generator.get_recommended_tactics(
                        "complex analytical problem", difficulty=0.7
                    )

                    if recommendation_result.is_success:
                        recommended = recommendation_result.value
                        print(f"추천 전술: {len(recommended)}개")

            print("통합 테스트 성공")
            return True
        else:
            print("통합 IIS 계산 실패")
            return False

    except Exception as e:
        print(f"통합 테스트 오류: {e}")
        return False


async def main():
    """메인 테스트 실행"""
    print("PACA v5 Phase 2 직접 테스트 시작")
    print("=" * 50)

    test_results = []
    test_names = [
        "IIS 계산기",
        "자율 훈련기",
        "전술 생성기",
        "시스템 통합"
    ]

    # 개별 테스트 실행
    test_results.append(await test_iis_calculator_direct())
    test_results.append(await test_autonomous_trainer_direct())
    test_results.append(await test_tactic_generator_direct())
    test_results.append(await test_integration_direct())

    # 결과 요약
    print("\n" + "=" * 50)
    print("테스트 결과 요약")

    passed_tests = 0
    for i, (name, result) in enumerate(zip(test_names, test_results), 1):
        status = "통과" if result else "실패"
        print(f"{i}. {name}: {status}")
        if result:
            passed_tests += 1

    success_rate = passed_tests / len(test_results) * 100
    print(f"\n전체 성공률: {passed_tests}/{len(test_results)} ({success_rate:.1f}%)")

    if success_rate == 100:
        print("모든 Phase 2 핵심 기능이 정상적으로 작동합니다!")
        print("\nPhase 2 자율 학습 시스템 구현 완료:")
        print("- IIS 점수 계산 시스템: AI 학습 수준 정량화")
        print("- 자율 훈련 시스템: 약점 자동 개선")
        print("- 전술/휴리스틱 생성 시스템: 지능 축적")
        print("- 시스템 통합: 유기적 연동")
        return 0
    else:
        print(f"{len(test_results) - passed_tests}개 테스트가 실패했습니다.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n테스트가 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n예상치 못한 오류: {e}")
        sys.exit(1)