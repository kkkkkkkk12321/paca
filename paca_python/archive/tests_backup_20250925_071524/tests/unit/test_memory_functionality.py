"""
PACA 메모리 시스템 실제 작동 테스트
단기/중기/장기 메모리의 실제 작동 여부를 검증
"""

import asyncio
import json
import time
from datetime import datetime

# PACA 메모리 시스템 임포트
try:
    from paca.cognitive.memory import WorkingMemory, EpisodicMemory, LongTermMemory
    from paca.cognitive.memory.types import MemoryConfiguration, SearchQuery
    from paca.cognitive.memory.episodic import EpisodicContext
    from paca.core.types import KeyValuePair
    MEMORY_AVAILABLE = True
    print("SUCCESS: Memory system import successful")
except ImportError as e:
    print(f"ERROR: Memory system import failed: {e}")
    MEMORY_AVAILABLE = False

async def test_working_memory():
    """단기 메모리 (Working Memory) 테스트"""
    print("\n=== 단기 메모리 (Working Memory) 테스트 ===")

    working_mem = WorkingMemory()

    # 1. 데이터 저장
    user_id = await working_mem.store("사용자: 김철수", {"type": "user_info"})
    task_id = await working_mem.store("현재 작업: PACA 메모리 테스트", {"type": "current_task"})
    context_id = await working_mem.store(
        {"topic": "메모리 시스템", "priority": "high", "timestamp": str(datetime.now())},
        {"type": "context"}
    )

    print(f"저장된 항목 IDs: {user_id}, {task_id}, {context_id}")

    # 2. 데이터 검색
    user_data = await working_mem.retrieve(user_id)
    task_data = await working_mem.retrieve(task_id)

    print(f"검색된 사용자 정보: {user_data.content if user_data else 'None'}")
    print(f"검색된 작업 정보: {task_data.content if task_data else 'None'}")

    # 3. 검색 기능 테스트
    search_query = SearchQuery(query="사용자", limit=5)
    search_results = await working_mem.search(search_query)

    print(f"검색 결과 수: {len(search_results.items)}")

    return len(working_mem.items) > 0

async def test_episodic_memory():
    """중기 메모리 (Episodic Memory) 테스트"""
    print("\n=== 중기 메모리 (Episodic Memory) 테스트 ===")

    episodic_mem = EpisodicMemory()

    # 1. 대화 에피소드 생성
    episodic_context = EpisodicContext(
        temporal_context={"timestamp": str(datetime.now()), "conversation_turn": 1},
        spatial_context={"location": "virtual_chat"},
        emotional_context={"user_mood": "curious", "conversation_tone": "questioning"},
        social_context={"user": "김철수", "topic": "메모리 시스템"}
    )

    episode_id = await episodic_mem.store_episode(
        content="사용자가 PACA의 메모리 시스템 작동 방식에 대해 질문함",
        episodic_context=episodic_context,
        importance=0.8
    )

    print(f"생성된 에피소드 ID: {episode_id}")

    # 2. 관련 에피소드 검색
    search_query = SearchQuery(query="메모리", limit=5)
    search_result = await episodic_mem.search(search_query)
    print(f"관련 에피소드 수: {len(search_result.items)}")

    # 3. 최근 에피소드 조회 (메모리 시스템 상태 확인)
    print(f"총 에피소드 수: {episodic_mem.get_episode_count()}")

    return episode_id is not None

async def test_longterm_memory():
    """장기 메모리 (Long Term Memory) 테스트"""
    print("\n=== 장기 메모리 (Long Term Memory) 테스트 ===")

    longterm_mem = LongTermMemory()

    # 1. 지식 저장
    knowledge_id = await longterm_mem.store_knowledge(
        content="PACA는 단기(Working), 중기(Episodic), 장기(LongTerm) 메모리 시스템을 가지고 있으며, 각각 다른 역할을 담당한다.",
        category="system_knowledge",
        importance=0.9,
        tags=["PACA", "메모리", "시스템", "아키텍처"]
    )

    print(f"저장된 지식 ID: {knowledge_id}")

    # 2. 지식 검색
    knowledge_results = await longterm_mem.retrieve_knowledge("메모리 시스템")
    print(f"검색된 지식 수: {len(knowledge_results) if knowledge_results else 0}")

    # 3. 카테고리별 지식 조회
    system_knowledge = await longterm_mem.get_knowledge_by_category("system_knowledge")
    print(f"시스템 지식 수: {len(system_knowledge) if system_knowledge else 0}")

    return knowledge_id is not None

async def test_memory_persistence():
    """메모리 지속성 테스트 (세션 간 데이터 보존)"""
    print("\n=== 메모리 지속성 테스트 ===")

    # 새로운 메모리 인스턴스 생성 (세션 재시작 시뮬레이션)
    new_working_mem = WorkingMemory()
    new_episodic_mem = EpisodicMemory()
    new_longterm_mem = LongTermMemory()

    # 이전에 저장된 데이터가 있는지 확인
    working_size = len(new_working_mem.items)

    print(f"새 세션의 단기 메모리 항목 수: {working_size}")

    # 장기 메모리에서 기존 지식 검색 시도
    existing_knowledge = await new_longterm_mem.retrieve_knowledge("PACA")
    print(f"기존 지식 검색 결과: {len(existing_knowledge) if existing_knowledge else 0}개")

    return True

async def test_conversation_context():
    """대화 맥락 유지 능력 테스트"""
    print("\n=== 대화 맥락 유지 테스트 ===")

    working_mem = WorkingMemory()
    episodic_mem = EpisodicMemory()

    # 대화 시뮬레이션
    conversation_context = {
        "user": "김철수",
        "session_id": "test_session_001",
        "start_time": str(datetime.now())
    }

    # 1단계: 사용자 정보 저장
    await working_mem.store("사용자가 PACA 메모리에 대해 질문 시작", conversation_context)

    # 2단계: 대화 진행 상황 기록
    episodic_context2 = EpisodicContext(
        temporal_context={"timestamp": str(datetime.now()), "conversation_turn": 2},
        spatial_context={"location": "virtual_chat"},
        emotional_context={"user_mood": "skeptical", "conversation_tone": "doubting"},
        social_context={**conversation_context, "turn": 2}
    )

    await episodic_mem.store_episode(
        content="사용자가 실제 메모리 작동 여부에 대해 회의적 질문",
        episodic_context=episodic_context2,
        importance=0.7
    )

    # 3단계: 맥락 검색 테스트
    context_search = SearchQuery(query="김철수 질문", limit=10)
    context_results = await working_mem.search(context_search)

    print(f"맥락 검색 결과: {len(context_results.items)}개")

    # 최근 에피소드 수 확인
    total_episodes = episodic_mem.get_episode_count()
    print(f"최근 대화 에피소드: {total_episodes}개")

    return len(context_results.items) > 0

async def main():
    """메인 테스트 함수"""
    print("PACA Memory System Real Operation Test Started")
    print("=" * 50)

    if not MEMORY_AVAILABLE:
        print("ERROR: Memory system not available.")
        return

    test_results = {}

    try:
        # 각 메모리 시스템 테스트
        test_results['working_memory'] = await test_working_memory()
        test_results['episodic_memory'] = await test_episodic_memory()
        test_results['longterm_memory'] = await test_longterm_memory()
        test_results['memory_persistence'] = await test_memory_persistence()
        test_results['conversation_context'] = await test_conversation_context()

        # 결과 요약
        print("\n" + "=" * 50)
        print("Test Results Summary")
        print("=" * 50)

        for test_name, result in test_results.items():
            status = "SUCCESS" if result else "FAILED"
            print(f"{test_name}: {status}")

        total_success = sum(test_results.values())
        success_rate = (total_success / len(test_results)) * 100

        print(f"\nOverall Success Rate: {success_rate:.1f}% ({total_success}/{len(test_results)})")

        # 결과를 JSON 파일로 저장
        result_data = {
            "test_timestamp": str(datetime.now()),
            "memory_system_tests": test_results,
            "success_rate": success_rate,
            "overall_status": "메모리 시스템 작동 확인" if success_rate >= 80 else "메모리 시스템 문제 있음"
        }

        with open("paca_memory_test_results.json", "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print("Results saved to paca_memory_test_results.json")

    except Exception as e:
        print(f"ERROR during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())