from __future__ import annotations

import sys, pathlib
# 이 스크립트의 2단계 상위 폴더가 paca_python 이므로, 거기를 경로에 추가
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from typing import Any, Dict, List
import asyncio

from paca.cognitive._collaboration_patch import enable_collab_patch
from paca.cognitive._collab_policy_loader import load_policy

class DummyEngine:
    def run(self, *args):
        print(f"[DummyEngine.run] called with args={args}")

class MinimalRC:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reasoning_engine = DummyEngine()

enable_collab_patch(MinimalRC)

async def main():
    policy_doc = load_policy()
    cfg: Dict[str, Any] = {}
    if policy_doc:
        cfg.update(policy_doc)

    rc = MinimalRC(cfg)
    history: List[Dict[str, Any]] = []

    result = await rc._try_collaboration_retries(   # type: ignore[attr-defined]
        reason="tool_error",
        context={"user": "test"},
        problem="test-problem",
        attempt_history=history,
    )

    print("\n=== RESULT ===")
    print(result)
    print("\n=== HISTORY ===")
    for h in history:
        print(h)

if __name__ == "__main__":
    asyncio.run(main())
