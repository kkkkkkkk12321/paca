"""
PACA Safe Print Utility
Windows CP949 환경에서 이모지 및 UTF-8 문자 안전 출력 모듈
"""

import sys
import os
from typing import Any, Optional
import logging


def safe_print(*args, **kwargs) -> None:
    """
    Windows CP949 환경에서 안전한 출력을 보장하는 함수
    이모지 및 UTF-8 문자 처리 시 오류 방지
    """
    try:
        # 표준 print 시도
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # 인코딩 에러 발생 시 안전한 방식으로 출력
        safe_args = []
        for arg in args:
            try:
                # 문자열 변환 및 이모지 제거
                text = str(arg)
                # 이모지 및 특수 문자 제거/대체
                safe_text = text.encode('ascii', 'ignore').decode('ascii')
                if not safe_text.strip():
                    safe_text = "[Unicode Content]"
                safe_args.append(safe_text)
            except Exception:
                safe_args.append("[Unprintable Content]")

        try:
            print(*safe_args, **kwargs)
        except Exception:
            # 최종 안전장치
            print("[Output Error - Content Not Displayable]")


def setup_unicode_environment() -> bool:
    """
    Windows 환경에서 UTF-8 지원을 위한 환경 설정
    """
    try:
        # 환경 변수 설정
        os.environ['PYTHONIOENCODING'] = 'utf-8'

        # Windows 콘솔 UTF-8 모드 활성화 시도
        if sys.platform == 'win32':
            try:
                import codecs
                sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
                sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
            except Exception:
                pass

        return True
    except Exception as e:
        logging.warning(f"Unicode environment setup failed: {e}")
        return False


def emoji_to_text(text: str) -> str:
    """
    이모지를 텍스트로 변환하는 함수
    """
    emoji_map = {
        '🚀': '[ROCKET]',
        '💡': '[IDEA]',
        '⚠️': '[WARNING]',
        '✅': '[CHECK]',
        '❌': '[ERROR]',
        '🔥': '[FIRE]',
        '🎯': '[TARGET]',
        '📊': '[CHART]',
        '🛠️': '[TOOLS]',
        '🧠': '[BRAIN]',
        '⚡': '[LIGHTNING]',
        '🔍': '[SEARCH]',
        '📝': '[NOTE]',
        '💎': '[GEM]',
        '🔧': '[WRENCH]',
        '📈': '[TRENDING_UP]',
        '🚨': '[ALARM]',
        '🎨': '[ART]',
        '🌟': '[STAR]',
        '🔐': '[LOCK]',
        '⏰': '[CLOCK]',
        '🎪': '[CIRCUS]',
        '🎲': '[DICE]',
        '🎸': '[GUITAR]',
        '🎹': '[PIANO]',
        '🎺': '[TRUMPET]',
        '🥁': '[DRUM]',
        '🎻': '[VIOLIN]'
    }

    for emoji, text_replacement in emoji_map.items():
        text = text.replace(emoji, text_replacement)

    return text


def format_status_safe(status: str, message: str) -> str:
    """
    상태 메시지를 안전한 형태로 포맷팅
    """
    status_map = {
        'success': '[SUCCESS]',
        'error': '[ERROR]',
        'warning': '[WARNING]',
        'info': '[INFO]',
        'debug': '[DEBUG]'
    }

    safe_status = status_map.get(status.lower(), f'[{status.upper()}]')
    safe_message = emoji_to_text(message)

    return f"{safe_status} {safe_message}"


# 모듈 로드 시 자동 설정
setup_unicode_environment()

# 전역 safe_print 함수 export
__all__ = ['safe_print', 'setup_unicode_environment', 'emoji_to_text', 'format_status_safe']