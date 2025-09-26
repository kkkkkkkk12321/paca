#!/usr/bin/env python3
"""
상대 임포트 문제 해결 테스트
"""

import sys
import os

# PYTHONPATH에 프로젝트 루트 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 이제 절대 임포트로 테스트
try:
    print("=== 절대 임포트 테스트 ===")

    # 1. 기본 타입 절대 임포트
    from paca.core.types.base import (
        create_success, create_failure, current_timestamp, generate_id
    )
    print("✅ 기본 타입 절대 임포트 성공")

    # 2. IIS Calculator 절대 임포트로 수정된 버전 테스트
    print("\n=== IIS Calculator 테스트 ===")

    # 임시로 상대 임포트 문제를 우회하여 클래스만 가져오기
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "iis_calc",
        os.path.join(project_root, "paca", "learning", "iis_calculator.py")
    )

    if spec and spec.loader:
        # 상대 임포트 부분을 임시로 수정하여 로드
        print("⚠️ 상대 임포트 문제로 인해 직접 로드는 불가능")
        print("   해결책: 패키지 설치 또는 절대 임포트로 수정 필요")

    # 3. 해결 방법 시연
    print("\n=== 해결 방법 시연 ===")

    # 방법 1: 조건부 임포트 패턴
    def safe_import():
        try:
            # 상대 임포트 시도
            exec("from ..core.types.base import ID")
            return "relative"
        except ImportError:
            # 절대 임포트로 대체
            from paca.core.types.base import ID
            return "absolute"

    try:
        import_type = safe_import()
        print(f"✅ 조건부 임포트 성공: {import_type}")
    except:
        print("❌ 조건부 임포트도 실패")

    # 방법 2: 시스템 경로 확인
    print(f"\n현재 Python 경로:")
    for i, path in enumerate(sys.path[:5]):
        print(f"  {i+1}. {path}")

    print(f"\n프로젝트 구조:")
    paca_path = os.path.join(project_root, "paca")
    if os.path.exists(paca_path):
        print(f"  ✅ paca/ 폴더 존재: {paca_path}")

        core_path = os.path.join(paca_path, "core")
        if os.path.exists(core_path):
            print(f"  ✅ paca/core/ 폴더 존재")

            types_path = os.path.join(core_path, "types")
            if os.path.exists(types_path):
                print(f"  ✅ paca/core/types/ 폴더 존재")

                base_path = os.path.join(types_path, "base.py")
                if os.path.exists(base_path):
                    print(f"  ✅ paca/core/types/base.py 파일 존재")
                else:
                    print(f"  ❌ base.py 파일 없음")
            else:
                print(f"  ❌ types/ 폴더 없음")
        else:
            print(f"  ❌ core/ 폴더 없음")
    else:
        print(f"  ❌ paca/ 폴더 없음: {paca_path}")

except Exception as e:
    print(f"❌ 테스트 실패: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 결론 ===")
print("상대 임포트 문제 해결 방법:")
print("1. 모든 Phase 2 모듈의 상대 임포트를 절대 임포트로 변경")
print("2. 또는 패키지 설치 후 모듈 형태로 실행")
print("3. 또는 조건부 임포트 패턴 사용")