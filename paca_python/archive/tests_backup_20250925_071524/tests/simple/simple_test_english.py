"""
Simple LLM Test in English
"""

import asyncio
import os
import sys
from pathlib import Path

# Add PACA module path
sys.path.insert(0, str(Path(__file__).parent))

# Direct LLM module import test
try:
    from paca.api.llm.base import ModelType, GenerationConfig, LLMRequest
    from paca.api.llm.gemini_client import GeminiClientManager, GeminiConfig
    from paca.api.llm.response_processor import ResponseProcessor
    print("SUCCESS: LLM modules imported successfully")
    LLM_AVAILABLE = True
except Exception as e:
    print(f"FAILED: LLM module import failed: {e}")
    LLM_AVAILABLE = False


async def test_gemini_client():
    """Test Gemini client directly"""
    if not LLM_AVAILABLE:
        print("LLM modules are not available")
        return False

    print("\n=== Gemini Client Test ===")

    # Check API keys
    api_keys_env = os.getenv("GEMINI_API_KEYS", "")
    if not api_keys_env:
        print("WARNING: GEMINI_API_KEYS environment variable not set")
        print("HINT: export GEMINI_API_KEYS=\"your_key1,your_key2\"")

        # Test with dummy keys for initialization test only
        api_keys = ["test_key"]
    else:
        api_keys = [key.strip() for key in api_keys_env.split(",")]
        print(f"SUCCESS: Found {len(api_keys)} API keys")

    # Client configuration
    config = GeminiConfig(
        api_keys=api_keys,
        default_model=ModelType.GEMINI_FLASH,
        enable_caching=True,
        timeout=30.0
    )

    # Create client
    client = GeminiClientManager(config)

    try:
        # Initialization test
        print("\n1. Client initialization...")
        init_result = await client.initialize()

        if init_result.is_success:
            print("   SUCCESS: Initialization completed")

            # Simple text generation test (only if real API keys exist)
            if api_keys_env:  # Only with real API keys
                print("\n2. Text generation test...")

                request = LLMRequest(
                    prompt="Hello! Please say hi briefly.",
                    model=ModelType.GEMINI_FLASH,
                    config=GenerationConfig(
                        temperature=0.7,
                        max_tokens=100
                    )
                )

                result = await client.generate_text(request)

                if result.is_success:
                    print(f"   SUCCESS: Response: {result.data.text[:100]}...")
                    print(f"   TIME: Processing time: {result.data.processing_time:.3f}s")
                else:
                    print(f"   FAILED: {result.error}")
            else:
                print("   WARNING: No API keys, skipping text generation test")

        else:
            print(f"   FAILED: Initialization failed: {init_result.error}")

        # Cleanup
        await client.cleanup()
        print("\n3. Cleanup completed")
        return init_result.is_success

    except Exception as e:
        print(f"   FAILED: Error during test: {str(e)}")
        return False


async def test_response_processor():
    """Test response processor"""
    if not LLM_AVAILABLE:
        return False

    print("\n=== Response Processor Test ===")

    try:
        processor = ResponseProcessor()

        # Create dummy response
        from paca.api.llm.base import LLMResponse
        test_response = LLMResponse(
            id="test_123",
            text="Hello! I am PACA AI assistant.",
            model=ModelType.GEMINI_FLASH,
            usage={"total_tokens": 20},
            processing_time=0.5
        )

        # Response processing test
        result = await processor.process_response(
            test_response,
            "Hello!",
            {}
        )

        if result.is_success:
            print("SUCCESS: Response processing completed")
            processed_response, metrics = result.data
            print(f"   Quality score: {metrics.quality_score:.2f}")
            print(f"   Token count: {metrics.token_count}")
        else:
            print(f"FAILED: Response processing failed: {result.error}")

        await processor.cleanup()
        return result.is_success if result else False

    except Exception as e:
        print(f"FAILED: Response processor test error: {str(e)}")
        return False


async def main():
    """Main test"""
    print("PACA v5 LLM Module Standalone Test")
    print("=" * 40)

    if not LLM_AVAILABLE:
        print("FAILED: LLM modules are not available.")
        print("Please check:")
        print("1. pip install google-genai")
        print("2. Module path and import errors")
        return

    # Run tests
    client_success = await test_gemini_client()
    processor_success = await test_response_processor()

    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"Gemini Client: {'SUCCESS' if client_success else 'FAILED'}")
    print(f"Response Processor: {'SUCCESS' if processor_success else 'FAILED'}")

    if client_success and processor_success:
        print("\nSUCCESS: LLM modules are working properly!")
        print("\nNext steps:")
        print("1. Set GEMINI_API_KEYS environment variable")
        print("2. Run integration test with full PACA system")
    else:
        print("\nWARNING: Some issues occurred.")


if __name__ == "__main__":
    asyncio.run(main())