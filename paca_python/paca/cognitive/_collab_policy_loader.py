from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

# paca/cognitive/_collab_policy_loader.py 기준:
# parents[2] == paca_python/  →  paca_python/config/collab_policy.json
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "collab_policy.json"


def load_policy(path: Path = DEFAULT_PATH) -> Dict[str, Any]:
    """
    config/collab_policy.json을 로드해 딕셔너리로 반환.
    파일이 없거나 오류가 나면 빈 dict 반환(안전).
    """
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        return {}
    except Exception:
        return {}


def apply_to_config(paca_config: Any, policy: Dict[str, Any]) -> None:
    """
    JSON의 모든 최상위 키를 paca_config에 반영한다.
    - paca_config가 dict처럼 동작하면 update로 한 번에 병합
    - 아니면 개별 키를 setattr → 실패 시 dict-style 할당까지 시도
    """
    if not policy:
        return

    # 1) dict 인터페이스가 있으면 한 방에 병합
    try:
        paca_config.update(policy)  # type: ignore[attr-defined]
        return
    except Exception:
        pass

    # 2) 개별 키 세팅 (객체/네임스페이스형 대응)
    for k, v in policy.items():
        try:
            setattr(paca_config, k, v)
            continue
        except Exception:
            pass
        try:
            paca_config[k] = v  # type: ignore[index]
        except Exception:
            # 마지막에도 실패하면 조용히 스킵(안전)
            pass
