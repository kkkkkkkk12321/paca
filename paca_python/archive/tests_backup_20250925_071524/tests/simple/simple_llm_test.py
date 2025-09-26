"""
간단한 LLM 테스트
API 임포트 문제를 우회하여 LLM 기능만 테스트
"""

import asyncio
import os
import sys
from pathlib import Path

# PACA 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

# 직접 LLM 모듈 임포트
try:
    from paca.api.llm.base import ModelType, GenerationConfig, LLMRequest
    from paca.api.llm.gemini_client import GeminiClientManager, GeminiConfig
    from paca.api.llm.response_processor import ResponseProcessor
    print("✅ LLM 모듈 임포트 성공")
    LLM_AVAILABLE = True
except Exception as e:
    print(f"❌ LLM 모듈 임포트 실패: {e}")
    LLM_AVAILABLE = False


async def test_gemini_client():
    """Gemini 클라이언트 직접 테스트"""
    if not LLM_AVAILABLE:
        print("LLM 모듈을 사용할 수 없습니다.")
        return False

    print("\n=== Gemini 클라이언트 테스트 ===")

    # API 키 확인
    api_keys_env = os.getenv("GEMINI_API_KEYS", "")
    if not api_keys_env:
        print("⚠️ GEMINI_API_KEYS 환경변수가 설정되지 않음")
        print("💡 export GEMINI_API_KEYS=\"your_key1,your_key2\"로 설정하세요")

        # 테스트용 더미 키로 초기화 테스트만 진행
        api_keys = ["test_key"]
    else:
        api_keys = [key.strip() for key in api_keys_env.split(",")]
        print(f"✅ {len(api_keys)}개 API 키 발견")

    # 클라이언트 설정
    config = GeminiConfig(
        api_keys=api_keys,
        default_model=ModelType.GEMINI_FLASH,
        enable_caching=True,
        timeout=30.0
    )

    # 클라이언트 생성
    client = GeminiClientManager(config)

    try:
        # 초기화 테스트
        print("\n1. 클라이언트 초기화...")
        init_result = await client.initialize()

        if init_result.is_success:
            print("   ✅ 초기화 성공")

            # 간단한 텍스트 생성 테스트 (API 키가 있는 경우)
            if api_keys_env:  # 실제 API 키가 있는 경우에만
                print("\n2. 텍스트 생성 테스트...")

                request = LLMRequest(
                    prompt="안녕하세요! 간단히 인사해주세요.",
                    model=ModelType.GEMINI_FLASH,
                    config=GenerationConfig(
                        temperature=0.7,
                        max_tokens=100
                    )
                )

                result = await client.generate_text(request)

                if result.is_success:
                    print(f"   ✅ 응답: {result.data.text[:100]}...")
                    print(f"   ⏱️ 처리시간: {result.data.processing_time:.3f}초")
                else:
                    print(f"   ❌ 실패: {result.error}")
            else:
                print("   ⚠️ API 키가 없어 텍스트 생성 테스트 건너뜀")

        else:
            print(f"   ❌ 초기화 실패: {init_result.error}")

        # 정리
        await client.cleanup()
        print("\n3. 정리 완료")
        return init_result.is_success

    except Exception as e:
        print(f"   ❌ 테스트 중 오류: {str(e)}")
        return False


async def test_response_processor():
    """응답 처리기 테스트"""
    if not LLM_AVAILABLE:
        return False

    print("\n=== 응답 처리기 테스트 ===")

    try:
        processor = ResponseProcessor()

        # 더미 응답 생성
        from paca.api.llm.base import LLMResponse
        test_response = LLMResponse(
            id="test_123",
            text="안녕하세요! 저는 PACA AI 어시스턴트입니다.",
            model=ModelType.GEMINI_FLASH,
            usage={"total_tokens": 20},
            processing_time=0.5
        )

        # 응답 처리 테스트
        result = await processor.process_response(
            test_response,
            "안녕하세요!",
            {}
        )

        if result.is_success:
            print("✅ 응답 처리 성공")
            processed_response, metrics = result.data
            print(f"   품질 점수: {metrics.quality_score:.2f}")
            print(f"   토큰 수: {metrics.token_count}")
        else:
            print(f"❌ 응답 처리 실패: {result.error}")

        await processor.cleanup()
        return result.is_success if result else False

    except Exception as e:
        print(f"❌ 응답 처리기 테스트 오류: {str(e)}")
        return False


async def main():
    """메인 테스트"""
    print("PACA v5 LLM 모듈 단독 테스트")
    print("="*40)

    if not LLM_AVAILABLE:
        print("❌ LLM 모듈을 사용할 수 없습니다.")
        print("다음을 확인하세요:")
        print("1. pip install google-genai")
        print("2. 모듈 경로 및 import 오류")
        return

    # 테스트 실행
    client_success = await test_gemini_client()
    processor_success = await test_response_processor()

    print("\n" + "="*40)
    print("테스트 결과:")
    print(f"Gemini 클라이언트: {'✅ 성공' if client_success else '❌ 실패'}")
    print(f"응답 처리기: {'✅ 성공' if processor_success else '❌ 실패'}")

    if client_success and processor_success:
        print("\n🎉 LLM 모듈이 정상적으로 작동합니다!")
        print("\n다음 단계:")
        print("1. GEMINI_API_KEYS 환경변수 설정")
        print("2. 전체 PACA 시스템과 통합 테스트")
    else:
        print("\n⚠️ 일부 기능에서 문제가 발생했습니다.")


if __name__ == "__main__":
    asyncio.run(main())