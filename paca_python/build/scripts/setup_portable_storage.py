#!/usr/bin/env python3
"""
Portable Storage Setup Script
포터블 저장소 설정 및 초기화 스크립트
"""

import os
import sys
from pathlib import Path

# PACA 모듈 경로 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from paca.core.utils.portable_storage import get_storage_manager
from paca.core.utils.safe_print import safe_print


def main():
    """포터블 저장소 설정 메인 함수"""
    safe_print("🚀 PACA 포터블 저장소 설정을 시작합니다...")
    safe_print("=" * 50)

    try:
        # 저장소 관리자 초기화
        storage_manager = get_storage_manager()

        safe_print("📁 데이터 디렉토리 구조 생성 중...")

        # 디렉토리 구조 정보 출력
        info = storage_manager.get_storage_info()

        safe_print(f"✅ 기본 경로: {info['base_path']}")
        safe_print("")
        safe_print("📂 생성된 디렉토리:")
        for name, path in info['directories'].items():
            safe_print(f"  - {name}: {path}")

        safe_print("")
        safe_print("💾 메모리 타입별 저장소:")
        for memory_type in ["working", "episodic", "semantic", "long_term"]:
            memory_path = storage_manager.get_memory_storage_path(memory_type)
            safe_print(f"  - {memory_type}: {memory_path}")

        # 테스트 데이터 저장
        safe_print("")
        safe_print("🧪 테스트 데이터 저장 중...")

        test_data = {
            "test": "포터블 저장소 테스트",
            "timestamp": "2025-01-23",
            "status": "성공"
        }

        test_file_path = storage_manager.get_config_file_path("test_config.json")
        if storage_manager.save_json_data(test_file_path, test_data):
            safe_print(f"✅ 테스트 데이터 저장 성공: {test_file_path}")
        else:
            safe_print("❌ 테스트 데이터 저장 실패")

        # 테스트 데이터 로드
        loaded_data = storage_manager.load_json_data(test_file_path)
        if loaded_data and loaded_data.get("test") == "포터블 저장소 테스트":
            safe_print("✅ 테스트 데이터 로드 성공")
        else:
            safe_print("❌ 테스트 데이터 로드 실패")

        # 데이터베이스 테스트
        safe_print("")
        safe_print("🗄️ 데이터베이스 연결 테스트 중...")

        try:
            conn = storage_manager.create_sqlite_connection("test.db")
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)")
            cursor.execute("INSERT INTO test (name) VALUES (?)", ("포터블 테스트",))
            conn.commit()

            cursor.execute("SELECT name FROM test WHERE id = ?", (1,))
            result = cursor.fetchone()

            if result and result[0] == "포터블 테스트":
                safe_print("✅ 데이터베이스 연결 및 테스트 성공")
            else:
                safe_print("❌ 데이터베이스 테스트 실패")

            conn.close()

        except Exception as e:
            safe_print(f"❌ 데이터베이스 테스트 오류: {e}")

        # 최종 정보 출력
        safe_print("")
        safe_print("📊 저장소 정보:")
        final_info = storage_manager.get_storage_info()
        safe_print(f"  - 총 파일 수: {final_info['total_files']}")
        safe_print(f"  - 총 크기: {final_info['total_size_mb']:.2f} MB")

        safe_print("")
        safe_print("🎉 포터블 저장소 설정이 완료되었습니다!")
        safe_print("")
        safe_print("📌 주요 특징:")
        safe_print("  - 모든 데이터가 프로그램 폴더 내 'data' 디렉토리에 저장됩니다")
        safe_print("  - 프로그램을 다른 컴퓨터로 복사해도 모든 데이터가 함께 이동됩니다")
        safe_print("  - USB나 클라우드 스토리지에서 바로 실행 가능합니다")
        safe_print("")
        safe_print("🚀 PACA를 시작하려면:")
        safe_print("  python -m paca")

    except Exception as e:
        safe_print(f"❌ 포터블 저장소 설정 중 오류 발생: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())