# paca/cognitive/_collab_policy_loader.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "collab_policy.json"
# parents[2]: paca_python/ 를 가리키도록 ( .../paca/cognitive -> parents[0]=cognitive, [1]=paca, [2]=paca_python )

def load_policy(path: Path = DEFAULT_PATH) -> Dict[str, Any]:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        return {}
    except Exception:
        return {}

def apply_to_config(paca_config: Any, policy: Dict[str, Any]) -> None:
    """PacaConfig 같은 설정 객체에 policy를 녹여 넣는다."""
    if not policy:
        return
    # 가장 단순하게 config 객체에 'escalation' 속성을 붙여 둔다.
    try:
        esc = policy.get("escalation")
        if esc:
            # 객체에 속성으로 심어둔다. (ReasoningChain에서 self.config로 복사해 쓰는 전제가 있음)
            if hasattr(paca_config, "__dict__"):
                # dict형 접근도 대비
                try:
                    # dict처럼 쓰는 구현일 수도 있음
                    paca_config["escalation"] = esc  # type: ignore[index]
                except Exception:
                    setattr(paca_config, "escalation", esc)
            else:
                setattr(paca_config, "escalation", esc)
    except Exception:
        pass
