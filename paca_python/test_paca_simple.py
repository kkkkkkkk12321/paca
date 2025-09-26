#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PACA 단순 테스트 스크립트 (UTF-8 문제 해결)
"""

import asyncio
import sys
import os

# UTF-8 인코딩 강제 설정
if os.name == 'nt':  # Windows
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

from paca.system import PacaSystem, PacaConfig

async def main():
    print("PACA v5 실제 시스템 테스트")
    print("=" * 50)

    try:
        # 설정 생성
        config = PacaConfig()
        print("설정 생성 완료")

        # PACA 시스템 초기화
        print("PACA 시스템 초기화 중...")
        paca_system = PacaSystem(config)

        result = await paca_system.initialize()
        if not result.is_success:
            print(f"시스템 초기화 실패: {result.error}")
            return

        print("PACA v5 시스템 준비 완료!")
        print("\n한국어 테스트를 시작합니다:")

        # 테스트 메시지들
        test_messages = [
            "안녕하세요",
            "파이썬 배우고 싶어요",
            "자바스크립트는 어때요?",
            "리액트 공부하려면?",
            "고마워요"
        ]

        for i, message in enumerate(test_messages, 1):
            print(f"\n[테스트 {i}] 입력: {message}")

            try:
                result = await paca_system.process_message(message)

                if result.is_success:
                    response = result.data.get("response", "응답 생성 실패")
                    print(f"PACA 응답: {response}")

                    # 처리 정보
                    processing_time = result.data.get("processing_time", 0)
                    print(f"처리시간: {processing_time:.3f}초")
                else:
                    print(f"오류: {result.error}")

            except Exception as e:
                print(f"메시지 처리 중 오류: {e}")

        # 시스템 정리
        print("\n시스템 정리 중...")
        await paca_system.cleanup()
        print("테스트 완료!")

    except Exception as e:
        print(f"실행 중 오류: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())