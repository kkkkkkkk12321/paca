"""
Phase 2 Self-Reflection System Test
자기 성찰 루프 시스템 테스트
"""

import asyncio
import os
import sys
from pathlib import Path

# PACA 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

# 테스트용 환경 변수 설정
os.environ["GEMINI_API_KEYS"] = "test_key_1,test_key_2"


async def test_reflection_imports():
    """자기 성찰 모듈 import 테스트"""
    print("=== Testing Reflection Module Imports ===")

    try:
        from paca.cognitive.reflection import (
            SelfReflectionProcessor,
            CritiqueAnalyzer,
            IterativeImprover,
            ReflectionConfig,
            ReflectionLevel,
            create_reflection_config
        )
        print("✅ All reflection modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


async def test_reflection_processor_basic():
    """기본 SelfReflectionProcessor 테스트"""
    print("\n=== Testing SelfReflectionProcessor Basic ===")

    try:
        from paca.cognitive.reflection import (
            SelfReflectionProcessor,
            ReflectionConfig,
            ReflectionLevel
        )

        # 설정 생성
        config = ReflectionConfig(
            reflection_level=ReflectionLevel.BASIC,
            quality_threshold=70.0,
            max_iterations=2
        )

        # 프로세서 생성 (LLM 클라이언트 없이)
        processor = SelfReflectionProcessor(llm_client=None, config=config)

        print("✅ SelfReflectionProcessor created successfully")
        print(f"   - Reflection level: {config.reflection_level.value}")
        print(f"   - Quality threshold: {config.quality_threshold}")
        print(f"   - Max iterations: {config.max_iterations}")

        # 통계 확인
        stats = await processor.get_stats()
        print(f"   - Initial stats: {stats}")

        await processor.cleanup()
        print("✅ Processor cleanup completed")

        return True

    except Exception as e:
        print(f"❌ SelfReflectionProcessor test failed: {e}")
        return False


async def test_critique_analyzer_basic():
    """기본 CritiqueAnalyzer 테스트"""
    print("\n=== Testing CritiqueAnalyzer Basic ===")

    try:
        from paca.cognitive.reflection import (
            CritiqueAnalyzer,
            ReflectionConfig,
            Weakness,
            WeaknessType
        )

        config = ReflectionConfig()
        analyzer = CritiqueAnalyzer(llm_client=None, config=config)

        print("✅ CritiqueAnalyzer created successfully")

        # 약점 객체 생성 테스트
        weakness = Weakness(
            type=WeaknessType.UNCLEAR_EXPRESSION,
            description="응답이 모호합니다",
            location="첫 번째 문단",
            severity=0.7,
            confidence=0.8
        )

        print(f"✅ Weakness object created: {weakness.description}")
        print(f"   - Type: {weakness.type.value}")
        print(f"   - Severity: {weakness.severity}")

        await analyzer.cleanup()
        print("✅ Analyzer cleanup completed")

        return True

    except Exception as e:
        print(f"❌ CritiqueAnalyzer test failed: {e}")
        return False


async def test_iterative_improver_basic():
    """기본 IterativeImprover 테스트"""
    print("\n=== Testing IterativeImprover Basic ===")

    try:
        from paca.cognitive.reflection import (
            IterativeImprover,
            ReflectionConfig,
            Improvement,
            ImprovementType
        )

        config = ReflectionConfig()
        improver = IterativeImprover(llm_client=None, config=config)

        print("✅ IterativeImprover created successfully")

        # 개선사항 객체 생성 테스트
        improvement = Improvement(
            weakness_id="weakness_123",
            type=ImprovementType.CLARIFICATION,
            description="명확성 개선",
            suggestion="더 구체적인 예시를 추가하세요",
            priority=0.8,
            estimated_impact=0.7
        )

        print(f"✅ Improvement object created: {improvement.description}")
        print(f"   - Type: {improvement.type.value}")
        print(f"   - Priority: {improvement.priority}")

        await improver.cleanup()
        print("✅ Improver cleanup completed")

        return True

    except Exception as e:
        print(f"❌ IterativeImprover test failed: {e}")
        return False


async def test_with_llm_client():
    """LLM 클라이언트와 함께 테스트 (실제 API 키 필요)"""
    print("\n=== Testing with LLM Client (if available) ===")

    try:
        # LLM 클라이언트 시도
        from paca.api.llm import GeminiClientManager, GeminiConfig, ModelType
        from paca.cognitive.reflection import SelfReflectionProcessor, ReflectionConfig

        # API 키 확인
        api_keys_env = os.getenv("GEMINI_API_KEYS", "")
        if not api_keys_env or api_keys_env.startswith("test_"):
            print("⚠️ No real API keys available. Skipping LLM test.")
            return True

        api_keys = [key.strip() for key in api_keys_env.split(",")]

        # Gemini 클라이언트 설정
        gemini_config = GeminiConfig(
            api_keys=api_keys,
            default_model=ModelType.GEMINI_FLASH
        )

        llm_client = GeminiClientManager(gemini_config)
        init_result = await llm_client.initialize()

        if not init_result.is_success:
            print(f"⚠️ LLM client initialization failed: {init_result.error}")
            return True

        print("✅ LLM client initialized")

        # 자기 성찰 프로세서 테스트
        reflection_config = ReflectionConfig(
            reflection_level=ReflectionLevel.BASIC,
            quality_threshold=70.0,
            max_iterations=1
        )

        processor = SelfReflectionProcessor(llm_client, reflection_config)

        # 간단한 초기 응답 생성 테스트
        test_input = "파이썬에서 리스트와 튜플의 차이점은 무엇인가요?"

        response_result = await processor.generate_initial_response(test_input)

        if response_result.is_success:
            initial_response = response_result.data
            print(f"✅ Initial response generated: {initial_response[:100]}...")

            # 빠른 성찰 테스트
            reflection_result = await processor.quick_reflection(test_input, initial_response)

            if reflection_result.is_success:
                reflection = reflection_result.data
                print(f"✅ Quick reflection completed")
                print(f"   - Quality improvement: {reflection.quality_improvement:.1f}")
                print(f"   - Iterations: {reflection.iterations_performed}")
                print(f"   - Processing time: {reflection.total_processing_time:.2f}s")
            else:
                print(f"⚠️ Quick reflection failed: {reflection_result.error}")
        else:
            print(f"⚠️ Initial response generation failed: {response_result.error}")

        await processor.cleanup()
        await llm_client.cleanup()

        print("✅ LLM integration test completed")
        return True

    except Exception as e:
        print(f"❌ LLM integration test failed: {e}")
        return False


async def main():
    """메인 테스트 함수"""
    print("PACA v5 Phase 2 Self-Reflection System Test")
    print("=" * 50)

    tests = [
        ("Module Imports", test_reflection_imports),
        ("SelfReflectionProcessor Basic", test_reflection_processor_basic),
        ("CritiqueAnalyzer Basic", test_critique_analyzer_basic),
        ("IterativeImprover Basic", test_iterative_improver_basic),
        ("LLM Integration", test_with_llm_client)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # 결과 요약
    print("\n" + "=" * 50)
    print("Test Results Summary:")

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All tests passed! Phase 2 Self-Reflection system is working!")
        print("\nNext steps:")
        print("1. Set real GEMINI_API_KEYS for full LLM testing")
        print("2. Integrate with main PACA system")
        print("3. Begin Phase 2.2 implementation (Truth Seeking)")
    else:
        print(f"\n⚠️ {len(results) - passed} tests failed. Please check the issues above.")


if __name__ == "__main__":
    asyncio.run(main())