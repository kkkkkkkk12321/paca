import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paca.api.llm.response_processor import ContextManager


@pytest.mark.asyncio
async def test_context_manager_maintains_long_term_summary_and_topics():
    manager = ContextManager(max_history=10, summary_window=4, summary_max_chars=500)

    for index in range(12):
        await manager.add_exchange(f"사용자 메시지 {index}", f"어시스턴트 답변 {index}")

    context = await manager.get_context_for_request("최신 질문")

    assert len(context["recent_history"]) == 4
    assert context["prior_messages"], "최근 기록 외 이전 메시지가 유지되어야 합니다"
    assert context["long_term_summary"], "장기 요약이 생성되어야 합니다"
    assert "장기 요약" in context["context_summary"], "맥락 요약에 장기 요약이 포함되어야 합니다"
    assert "dominant_topics" in context["session_context"], "주요 토픽이 세션 컨텍스트에 포함되어야 합니다"


@pytest.mark.asyncio
async def test_context_manager_updates_user_preferences():
    manager = ContextManager()
    manager.update_user_preferences({"tone": "정중"})
    manager.update_user_preferences({"format": "bullet"})

    assert manager.user_preferences["tone"] == "정중"
    assert manager.user_preferences["format"] == "bullet"

    manager.clear_history()
    assert manager.long_term_summary == ""
    assert not manager.topic_counter
