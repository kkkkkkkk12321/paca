# paca/cognitive/_enable_collab_patch.py
try:
    # 프로젝트 구조에 맞게 필요 시 수정
    from paca.cognitive.reasoning_chain import ReasoningChain
    from paca.cognitive._collaboration_patch import enable_collab_patch

    enable_collab_patch(ReasoningChain)
except Exception as e:
    # 패치 실패해도 앱이 죽지 않도록 안전 처리
    print(f"[collab-patch] skip (reason={e})")
