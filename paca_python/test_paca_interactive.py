#!/usr/bin/env python3
"""
PACA 대화형 테스트 스크립트
StructuredLogger 문제를 우회하여 실제 PACA 시스템 실행
"""

import asyncio
import sys
from paca.system import PacaSystem, PacaConfig
from paca.core.utils.logger import PacaLogger

async def main():
    print("🤖 PACA v5 실제 시스템 테스트")
    print("=" * 50)

    try:
        # 설정 생성
        config = PacaConfig()
        config.log_level = "INFO"
        print("✅ 설정 생성 완료")

        # PACA 시스템 초기화
        print("🚀 PACA 시스템 초기화 중...")
        paca_system = PacaSystem(config)

        result = await paca_system.initialize()
        if not result.is_success:
            print(f"❌ 시스템 초기화 실패: {result.error}")
            return

        print("✅ PACA v5 시스템 준비 완료!")
        print("\n대화를 시작합니다. 'quit'을 입력하면 종료됩니다.\n")

        # 대화형 루프
        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 안녕히 가세요!")
                    break

                if not user_input:
                    continue

                # 메시지 처리
                print("🤔 처리 중...")
                result = await paca_system.process_message(user_input)

                if result.is_success:
                    response = result.data.get("response", "응답을 생성할 수 없습니다.")
                    print(f"PACA: {response}")

                    # 추가 정보 표시
                    processing_time = result.data.get("processing_time", 0)
                    confidence = result.data.get("confidence", 0)
                    if processing_time > 0:
                        print(f"       [처리시간: {processing_time:.3f}s, 신뢰도: {confidence:.2f}]")
                else:
                    print(f"❌ 오류: {result.error}")

                print()

            except KeyboardInterrupt:
                print("\n👋 안녕히 가세요!")
                break
            except EOFError:
                print("\n👋 안녕히 가세요!")
                break

        # 시스템 정리
        await paca_system.cleanup()
        print("🔄 시스템 정리 완료")

    except Exception as e:
        print(f"❌ 실행 중 오류: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())