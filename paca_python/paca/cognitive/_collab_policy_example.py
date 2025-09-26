# paca/cognitive/_collab_policy_example.py

def get_collab_config():
    """
    재시도 정책 샘플.
    - enabled: True면 기능 켬
    - collaboration_policy:
        * default: 모든 상황의 기본 정책
        * tool_error 등 상황별 키를 추가해 다르게 시도 가능
    """
    return {
        "escalation": {
            "enabled": True,
            "collaboration_policy": {
                "default": {  # 기본 정책 (예시)
                    "reasoning_types": ["abductive", "analogical"],
                    "max_attempts": 2
                },
                "tool_error": {  # 도구 오류시 더 많이 시도 (예시)
                    "reasoning_types": ["analytical", "causal"],
                    "max_attempts": 3
                }
            }
        }
    }
