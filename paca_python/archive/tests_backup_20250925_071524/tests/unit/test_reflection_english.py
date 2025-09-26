"""
Phase 2 Self-Reflection System Test (English)
"""

import asyncio
import os
import sys
from pathlib import Path

# PACA module path
sys.path.insert(0, str(Path(__file__).parent))

# Test environment variables
os.environ["GEMINI_API_KEYS"] = "test_key_1,test_key_2"


async def test_reflection_imports():
    """Test reflection module imports"""
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
        print("SUCCESS: All reflection modules imported successfully")
        return True
    except Exception as e:
        print(f"FAILED: Import failed: {e}")
        return False


async def test_reflection_processor_basic():
    """Test basic SelfReflectionProcessor"""
    print("\n=== Testing SelfReflectionProcessor Basic ===")

    try:
        from paca.cognitive.reflection import (
            SelfReflectionProcessor,
            ReflectionConfig,
            ReflectionLevel
        )

        # Create configuration
        config = ReflectionConfig(
            reflection_level=ReflectionLevel.BASIC,
            quality_threshold=70.0,
            max_iterations=2
        )

        # Create processor (without LLM client)
        processor = SelfReflectionProcessor(llm_client=None, config=config)

        print("SUCCESS: SelfReflectionProcessor created successfully")
        print(f"   - Reflection level: {config.reflection_level.value}")
        print(f"   - Quality threshold: {config.quality_threshold}")
        print(f"   - Max iterations: {config.max_iterations}")

        # Check stats
        stats = await processor.get_stats()
        print(f"   - Initial stats: {stats}")

        await processor.cleanup()
        print("SUCCESS: Processor cleanup completed")

        return True

    except Exception as e:
        print(f"FAILED: SelfReflectionProcessor test failed: {e}")
        return False


async def test_critique_analyzer_basic():
    """Test basic CritiqueAnalyzer"""
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

        print("SUCCESS: CritiqueAnalyzer created successfully")

        # Test weakness object creation
        weakness = Weakness(
            type=WeaknessType.UNCLEAR_EXPRESSION,
            description="Response is ambiguous",
            location="First paragraph",
            severity=0.7,
            confidence=0.8
        )

        print(f"SUCCESS: Weakness object created: {weakness.description}")
        print(f"   - Type: {weakness.type.value}")
        print(f"   - Severity: {weakness.severity}")

        await analyzer.cleanup()
        print("SUCCESS: Analyzer cleanup completed")

        return True

    except Exception as e:
        print(f"FAILED: CritiqueAnalyzer test failed: {e}")
        return False


async def test_iterative_improver_basic():
    """Test basic IterativeImprover"""
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

        print("SUCCESS: IterativeImprover created successfully")

        # Test improvement object creation
        improvement = Improvement(
            type=ImprovementType.CLARIFICATION,
            description="Clarity improvement",
            suggestion="Add more specific examples",
            priority=0.8,
            estimated_impact=0.7
        )

        print(f"SUCCESS: Improvement object created: {improvement.description}")
        print(f"   - Type: {improvement.type.value}")
        print(f"   - Priority: {improvement.priority}")

        await improver.cleanup()
        print("SUCCESS: Improver cleanup completed")

        return True

    except Exception as e:
        print(f"FAILED: IterativeImprover test failed: {e}")
        return False


async def test_config_creation():
    """Test configuration creation and validation"""
    print("\n=== Testing Configuration Creation ===")

    try:
        from paca.cognitive.reflection import (
            ReflectionConfig,
            ReflectionLevel,
            create_reflection_config,
            calculate_overall_quality
        )

        # Test default config
        default_config = create_reflection_config()
        print(f"SUCCESS: Default config created")
        print(f"   - Level: {default_config.reflection_level.value}")
        print(f"   - Threshold: {default_config.quality_threshold}")

        # Test custom config
        custom_config = ReflectionConfig(
            reflection_level=ReflectionLevel.DEEP,
            quality_threshold=90.0,
            max_iterations=5,
            enable_detailed_logging=True
        )
        print(f"SUCCESS: Custom config created")
        print(f"   - Level: {custom_config.reflection_level.value}")
        print(f"   - Threshold: {custom_config.quality_threshold}")

        # Test quality calculation
        overall_quality = calculate_overall_quality(
            logical_consistency=85.0,
            factual_accuracy=90.0,
            completeness=80.0,
            relevance=95.0,
            clarity=88.0
        )
        print(f"SUCCESS: Quality calculation test")
        print(f"   - Overall quality score: {overall_quality:.1f}")

        return True

    except Exception as e:
        print(f"FAILED: Configuration test failed: {e}")
        return False


async def main():
    """Main test function"""
    print("PACA v5 Phase 2 Self-Reflection System Test")
    print("=" * 50)

    tests = [
        ("Module Imports", test_reflection_imports),
        ("SelfReflectionProcessor Basic", test_reflection_processor_basic),
        ("CritiqueAnalyzer Basic", test_critique_analyzer_basic),
        ("IterativeImprover Basic", test_iterative_improver_basic),
        ("Configuration Creation", test_config_creation)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"CRASHED: {test_name} crashed: {e}")
            results.append((test_name, False))

    # Results summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")

    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\nSUCCESS: All tests passed! Phase 2 Self-Reflection system is working!")
        print("\nNext steps:")
        print("1. Set real GEMINI_API_KEYS for full LLM testing")
        print("2. Integrate with main PACA system")
        print("3. Begin Phase 2.2 implementation (Truth Seeking)")
    else:
        print(f"\nWARNING: {len(results) - passed} tests failed. Please check the issues above.")


if __name__ == "__main__":
    asyncio.run(main())