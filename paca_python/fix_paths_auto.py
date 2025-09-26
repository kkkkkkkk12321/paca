#!/usr/bin/env python3
"""
PACA 경로 수정 자동화 스크립트
29개 파일의 sys.path 패턴을 일괄 수정
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple

def backup_file(file_path: str) -> str:
    """파일 백업 생성"""
    backup_path = f"{file_path}.backup"
    shutil.copy2(file_path, backup_path)
    return backup_path

def fix_path_patterns(file_path: str) -> bool:
    """파일의 경로 패턴을 수정"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # CP949 인코딩으로 재시도
        with open(file_path, 'r', encoding='cp949') as f:
            content = f.read()

    original_content = content
    modified = False

    # 수정 패턴 정의
    patterns = [
        # 패턴 1: 현재 폴더만 참조 (가장 흔한 패턴)
        (
            r'sys\.path\.insert\(0,\s*os\.path\.join\(os\.path\.dirname\(__file__\)\)\)',
            "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))"
        ),

        # 패턴 2: Path(__file__).parent / "paca" 패턴
        (
            r'sys\.path\.insert\(0,\s*str\(Path\(__file__\)\.parent\s*/\s*"paca"\)\)',
            "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))"
        ),

        # 패턴 3: 현재 폴더 '.' 참조
        (
            r'sys\.path\.insert\(0,\s*os\.path\.join\(os\.path\.dirname\(__file__\),\s*\'?\.\'\?\)\)',
            "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))"
        ),

        # 패턴 4: 다양한 공백 처리
        (
            r'sys\.path\.insert\(\s*0\s*,\s*os\.path\.join\(\s*os\.path\.dirname\(\s*__file__\s*\)\s*\)\s*\)',
            "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))"
        )
    ]

    # 패턴별 수정 적용
    for pattern, replacement in patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            modified = True

    # 수정사항이 있으면 파일 저장
    if modified:
        # 백업 생성
        backup_path = backup_file(file_path)
        print(f"  백업 생성: {backup_path}")

        # 수정된 내용 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"  ✅ 수정 완료: {file_path}")
        return True
    else:
        print(f"  ⚠️  수정 패턴 없음: {file_path}")
        return False

def main():
    """메인 실행 함수"""
    print("PACA 경로 수정 자동화 스크립트 시작")
    print("=" * 50)

    # 수정 대상 파일 목록
    target_files = [
        # Tests/Simple (8개)
        "tests/simple/simple_phase1_test.py",
        "tests/simple/simple_phase2_test.py",
        "tests/simple/simple_phase8_test.py",
        "tests/simple/simple_conflict_test.py",
        "tests/simple/simple_integration_test.py",
        "tests/simple/simple_performance_test.py",
        "tests/simple/simple_test_english.py",
        "tests/simple/simple_llm_test.py",

        # Tests/Unit (11개)
        "tests/unit/test_phase1_integration.py",
        "tests/unit/test_phase2_truth_integrity.py",
        "tests/unit/test_phase8_tools.py",
        "tests/unit/test_curiosity_system.py",
        "tests/unit/test_import_conflicts.py",
        "tests/unit/test_reflection_english.py",
        "tests/unit/test_reflection_system.py",
        "tests/unit/test_llm_integration.py",
        "tests/unit/test_final.py",
        "tests/unit/test_simple.py",
        "tests/unit/test_integration.py",

        # Tests/Integration (2개)
        "tests/integration/system_integration_test.py",
        "tests/integration/test_basic_functionality.py",

        # Tests/Performance (2개)
        "tests/performance/test_basic_performance.py",
        "tests/performance/test_simple_performance.py",

        # Tests/Korean (1개)
        "tests/korean/test_korean_processing.py",

        # Tests/Phases (1개) - 특별 패턴이므로 나중에 수동 처리
        # "tests/phases/direct_phase2_test.py",

        # Scripts (4개)
        "scripts/analysis/performance_profiler.py",
        "scripts/analysis/compatibility_example.py",
        "scripts/optimization/memory_optimizer.py",
        "scripts/stability/stability_enhancer.py"
    ]

    success_count = 0
    error_count = 0

    # 현재 스크립트 위치 기준으로 작업 디렉토리 설정
    base_dir = Path(__file__).parent

    print(f"작업 디렉토리: {base_dir}")
    print(f"총 수정 대상: {len(target_files)}개 파일\n")

    # 각 파일 처리
    for i, file_path in enumerate(target_files, 1):
        full_path = base_dir / file_path

        print(f"[{i:2d}/{len(target_files)}] {file_path}")

        if not full_path.exists():
            print(f"  ❌ 파일 없음: {full_path}")
            error_count += 1
            continue

        try:
            if fix_path_patterns(str(full_path)):
                success_count += 1
            else:
                print(f"  ⚪ 수정 불필요")

        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
            error_count += 1

    print("\n" + "=" * 50)
    print("🎯 수정 완료 요약")
    print(f"✅ 성공: {success_count}개")
    print(f"⚠️  수정불필요/오류: {error_count}개")
    print(f"📁 총 파일: {len(target_files)}개")

    if success_count > 0:
        print(f"\n💡 백업 파일들이 생성되었습니다 (*.backup)")
        print("   문제 발생 시 백업으로 복원 가능합니다.")

    print("\n🚀 다음 단계:")
    print("   1. tests/phases/direct_phase2_test.py 수동 확인")
    print("   2. production_server.py 경로 검증")
    print("   3. 수정된 파일들 테스트 실행")

if __name__ == "__main__":
    main()