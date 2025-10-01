from paca.cognitive._collab_policy_loader import load_policy


def test_default_collaboration_policy_provides_thresholds(tmp_path):
    policy = load_policy()

    assert policy, "기본 협업 정책 파일을 로드하지 못했습니다."
    assert policy.get("collaboration", {}).get("enabled") is True
    assert "reasoning_confidence_threshold" in policy
    assert policy["reasoning_confidence_threshold"] > 0
