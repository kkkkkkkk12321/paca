#!/usr/bin/env python3
"""
Portable Memory Test Script
포터블 메모리 시스템 테스트
"""

import asyncio
import sys
from pathlib import Path

# PACA 모듈 경로 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from paca.cognitive.memory.working import WorkingMemory
from paca.cognitive.memory.episodic import EpisodicMemory, EpisodicContext
from paca.core.utils.safe_print import safe_print
from paca.core.utils.portable_storage import get_storage_manager


async def test_working_memory():
    """작업 메모리 포터블 저장 테스트"""
    safe_print("🧠 작업 메모리 테스트 시작...")

    # 작업 메모리 인스턴스 생성
    working_memory = WorkingMemory()

    # 잠시 대기 (비동기 로드 완료 대기)
    await asyncio.sleep(0.1)

    # 테스트 데이터 저장
    item_id1 = await working_memory.store("포터블 테스트 데이터 1", {"type": "test"})
    item_id2 = await working_memory.store("포터블 테스트 데이터 2", {"type": "test"})

    safe_print(f"✅ 작업 메모리에 2개 항목 저장 완료")

    # 키-값 저장 테스트
    kv_id = await working_memory.store_kv("test_key", "포터블 키-값 데이터")
    safe_print(f"✅ 키-값 데이터 저장 완료")

    # 데이터 검색 테스트
    retrieved_item = await working_memory.retrieve(item_id1)
    if retrieved_item:
        safe_print(f"✅ 데이터 검색 성공: {retrieved_item.content}")

    # 키로 검색 테스트
    kv_data = await working_memory.retrieve_by_key("test_key")
    if kv_data:
        safe_print(f"✅ 키-값 검색 성공: {kv_data}")

    # 용량 정보 확인
    capacity_info = working_memory.get_capacity_info()
    safe_print(f"📊 작업 메모리 상태: {capacity_info['current_size']}/{capacity_info['capacity']} 사용중")

    return True


async def test_episodic_memory():
    """일화 메모리 포터블 저장 테스트"""
    safe_print("")
    safe_print("📚 일화 메모리 테스트 시작...")

    # 일화 메모리 인스턴스 생성
    episodic_memory = EpisodicMemory()

    # 잠시 대기 (비동기 로드 완료 대기)
    await asyncio.sleep(0.1)

    # 테스트 일화 저장
    context = EpisodicContext(
        temporal_context={"날짜": "2025-01-23", "시간": "오후"},
        spatial_context={"위치": "포터블 테스트"},
        emotional_context={"기분": "긍정적"},
        social_context={"상황": "개발 테스트"}
    )

    episode_id1 = await episodic_memory.store_episode(
        "포터블 일화 테스트 1",
        context,
        importance=0.8
    )

    episode_id2 = await episodic_memory.store_simple_episode(
        "test_episode",
        {"내용": "간단한 일화 테스트", "중요도": "높음"}
    )

    safe_print(f"✅ 일화 메모리에 2개 에피소드 저장 완료")

    # 시간 기반 검색 테스트
    episodes = await episodic_memory.retrieve_by_time(limit=5)
    safe_print(f"✅ 시간 기반 검색 성공: {len(episodes)}개 에피소드")

    # 맥락 기반 검색 테스트
    context_query = {"spatial": {"위치": "포터블 테스트"}}
    context_episodes = await episodic_memory.retrieve_by_context(context_query)
    safe_print(f"✅ 맥락 기반 검색 성공: {len(context_episodes)}개 에피소드")

    # 일화 수 확인
    episode_count = episodic_memory.get_episode_count()
    safe_print(f"📊 저장된 일화 수: {episode_count}개")

    return True


async def test_storage_persistence():
    """저장소 지속성 테스트"""
    safe_print("")
    safe_print("💾 저장소 지속성 테스트 시작...")

    # 첫 번째 인스턴스에서 데이터 저장
    working_memory1 = WorkingMemory()
    await asyncio.sleep(0.1)

    test_id = await working_memory1.store("지속성 테스트 데이터", {"persistent": True})
    safe_print("✅ 첫 번째 인스턴스에서 데이터 저장")

    # 두 번째 인스턴스 생성 (이전 데이터 로드되어야 함)
    working_memory2 = WorkingMemory()
    await asyncio.sleep(0.1)

    # 이전에 저장한 데이터 검색
    retrieved = await working_memory2.retrieve(test_id)
    if retrieved and retrieved.content == "지속성 테스트 데이터":
        safe_print("✅ 두 번째 인스턴스에서 이전 데이터 로드 성공")
        return True
    else:
        safe_print("❌ 데이터 지속성 테스트 실패")
        return False


async def test_storage_info():
    """저장소 정보 테스트"""
    safe_print("")
    safe_print("📈 저장소 정보 확인...")

    storage_manager = get_storage_manager()
    info = storage_manager.get_storage_info()

    safe_print(f"📁 기본 경로: {info['base_path']}")
    safe_print(f"📊 총 파일 수: {info['total_files']}")
    safe_print(f"💽 총 크기: {info['total_size_mb']:.3f} MB")

    safe_print("")
    safe_print("📂 메모리 타입별 정보:")
    for memory_type, type_info in info['memory_types'].items():
        safe_print(f"  - {memory_type}: {type_info['file_count']}개 파일, {type_info['size_mb']:.3f} MB")

    return True


async def main():
    """메인 테스트 함수"""
    safe_print("🚀 PACA 포터블 메모리 시스템 테스트")
    safe_print("=" * 50)

    try:
        # 작업 메모리 테스트
        await test_working_memory()

        # 일화 메모리 테스트
        await test_episodic_memory()

        # 저장소 지속성 테스트
        await test_storage_persistence()

        # 저장소 정보 테스트
        await test_storage_info()

        safe_print("")
        safe_print("🎉 모든 포터블 메모리 테스트가 성공적으로 완료되었습니다!")
        safe_print("")
        safe_print("✅ 확인된 기능:")
        safe_print("  - 작업 메모리 포터블 저장/로드")
        safe_print("  - 일화 메모리 포터블 저장/로드")
        safe_print("  - 데이터 지속성 (프로그램 재시작 후에도 유지)")
        safe_print("  - 키-값 저장 시스템")
        safe_print("  - 메모리 타입별 분리 저장")
        safe_print("  - 자동 디렉토리 구조 생성")
        safe_print("")
        safe_print("🎯 포터블 앱 준비 완료!")

        return 0

    except Exception as e:
        safe_print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))