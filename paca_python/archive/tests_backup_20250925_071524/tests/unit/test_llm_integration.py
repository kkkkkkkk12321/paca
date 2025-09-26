"""
PACA v5 LLM 통합 테스트
Phase 1 구현 검증을 위한 테스트 스크립트
"""

import asyncio
import os
import sys
from pathlib import Path

# PACA 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent / "paca"))

from paca.system import PacaSystem, PacaConfig
from paca.api.llm.base import ModelType


async def test_basic_functionality():
    """기본 기능 테스트"""
    print("=== PACA v5 LLM 통합 테스트 ===\n")

    # 1. API 키 설정 확인
    print("1. API 키 설정 확인...")
    api_keys_env = os.getenv("GEMINI_API_KEYS", "")
    if api_keys_env:
        api_keys = [key.strip() for key in api_keys_env.split(",")]
        print(f"   ✅ 환경변수에서 {len(api_keys)}개 API 키 발견")
    else:
        print("   ⚠️ GEMINI_API_KEYS 환경변수가 설정되지 않음")
        print("   💡 export GEMINI_API_KEYS=\"your_key1,your_key2\"로 설정하세요")
        api_keys = []

    # 2. PACA 시스템 설정
    print("\n2. PACA 시스템 설정...")
    config = PacaConfig(
        gemini_api_keys=api_keys,
        default_llm_model=ModelType.GEMINI_FLASH,
        llm_temperature=0.7,
        llm_max_tokens=1024,
        enable_llm_caching=True,
        llm_timeout=30.0
    )
    print("   ✅ 설정 완료")

    # 3. 시스템 초기화
    print("\n3. PACA 시스템 초기화...")
    paca = PacaSystem(config)

    try:
        init_result = await paca.initialize()
        if init_result.is_success:
            print("   ✅ 시스템 초기화 성공")
        else:
            print(f"   ❌ 시스템 초기화 실패: {init_result.error}")
            return False
    except Exception as e:
        print(f"   ❌ 초기화 중 오류: {str(e)}")
        return False

    # 4. 시스템 상태 확인
    print("\n4. 시스템 상태 확인...")
    try:
        status = await paca.get_system_status()
        print(f"   상태: {status['status']}")
        print(f"   초기화됨: {status['is_initialized']}")
        print(f"   버전: {status['version']}")

        # LLM 클라이언트 상태 확인
        if paca.llm_client:
            health_result = await paca.llm_client.health_check()
            if health_result.is_success:
                health_data = health_result.data
                print(f"   LLM 상태: {health_data['status']}")
                print(f"   사용 가능한 키: {health_data['available_keys']}")
            else:
                print(f"   ⚠️ LLM 헬스체크 실패: {health_result.error}")
        else:
            print("   ⚠️ LLM 클라이언트가 초기화되지 않음")
    except Exception as e:
        print(f"   ❌ 상태 확인 오류: {str(e)}")

    # 5. 기본 대화 테스트
    print("\n5. 기본 대화 테스트...")
    test_messages = [
        "안녕하세요!",
        "PACA는 무엇인가요?",
        "2 + 2는 얼마인가요?",
        "오늘 날씨는 어떤가요?"
    ]

    for i, message in enumerate(test_messages, 1):
        print(f"\n   테스트 {i}: '{message}'")
        try:
            result = await paca.process_message(message)
            if result.is_success:
                response = result.data.get("response", "응답 없음")
                processing_time = result.data.get("processing_time", 0)
                confidence = result.data.get("confidence", 0)

                print(f"   ✅ 응답: {response[:100]}...")
                print(f"   ⏱️ 처리시간: {processing_time:.3f}초")
                print(f"   📊 신뢰도: {confidence:.2f}")
            else:
                print(f"   ❌ 실패: {result.error}")
        except Exception as e:
            print(f"   ❌ 오류: {str(e)}")

    # 6. 시스템 정리
    print("\n6. 시스템 정리...")
    try:
        await paca.cleanup()
        print("   ✅ 정리 완료")
    except Exception as e:
        print(f"   ⚠️ 정리 중 오류: {str(e)}")

    return True


async def test_llm_specific_features():
    """LLM 특화 기능 테스트"""
    print("\n=== LLM 특화 기능 테스트 ===")

    # API 키 확인
    api_keys_env = os.getenv("GEMINI_API_KEYS", "")
    if not api_keys_env:
        print("❌ API 키가 없어 LLM 테스트를 건너뜁니다")
        return False

    api_keys = [key.strip() for key in api_keys_env.split(",")]

    config = PacaConfig(
        gemini_api_keys=api_keys,
        default_llm_model=ModelType.GEMINI_FLASH
    )

    paca = PacaSystem(config)
    await paca.initialize()

    if not paca.llm_client:
        print("❌ LLM 클라이언트가 초기화되지 않음")
        return False

    # 다양한 모델 테스트
    models_to_test = [ModelType.GEMINI_FLASH, ModelType.GEMINI_PRO]

    for model in models_to_test:
        print(f"\n테스트 모델: {model.value}")

        # 모델별 설정 변경
        paca.config.default_llm_model = model

        try:
            result = await paca.process_message("한국의 수도는 어디인가요?")
            if result.is_success:
                response = result.data.get("response", "")
                print(f"✅ {model.value} 응답: {response[:100]}...")
            else:
                print(f"❌ {model.value} 실패: {result.error}")
        except Exception as e:
            print(f"❌ {model.value} 오류: {str(e)}")

    await paca.cleanup()
    return True


async def main():
    """메인 테스트 실행"""
    print("PACA v5 Phase 1 구현 테스트를 시작합니다...\n")

    # 기본 기능 테스트
    basic_success = await test_basic_functionality()

    # LLM 특화 기능 테스트 (API 키가 있는 경우)
    llm_success = await test_llm_specific_features()

    print("\n" + "="*50)
    print("테스트 결과 요약:")
    print(f"기본 기능: {'✅ 성공' if basic_success else '❌ 실패'}")
    print(f"LLM 기능: {'✅ 성공' if llm_success else '❌ 실패 (API 키 필요)'}")

    if basic_success:
        print("\n🎉 Phase 1 LLM 통합이 성공적으로 완료되었습니다!")
        print("\n다음 단계:")
        print("1. GEMINI_API_KEYS 환경변수 설정")
        print("2. pip install google-genai 설치")
        print("3. python -m paca --interactive 실행")
    else:
        print("\n⚠️ 일부 기능에서 문제가 발생했습니다.")
        print("로그를 확인하고 설정을 검토하세요.")


if __name__ == "__main__":
    asyncio.run(main())