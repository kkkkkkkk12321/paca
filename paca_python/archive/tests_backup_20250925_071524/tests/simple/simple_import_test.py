#!/usr/bin/env python3
"""
간단한 상대 임포트 문제 테스트
"""

import sys
import os

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("상대 임포트 문제 분석")
print("=" * 40)

# 1. 기본 타입 절대 임포트 테스트
try:
    from paca.core.types.base import create_success, current_timestamp
    print("1. 기본 타입 절대 임포트: 성공")
except ImportError as e:
    print(f"1. 기본 타입 절대 임포트: 실패 - {e}")

# 2. 상대 임포트가 왜 실패하는지 보여주기
print("\n2. 상대 임포트 실패 이유:")
print("   - 파일을 직접 실행할 때: python iis_calculator.py")
print("   - Python은 해당 파일을 패키지가 아닌 스크립트로 인식")
print("   - 따라서 '..' (상위 패키지)를 찾을 수 없음")

# 3. 현재 경로 상황 확인
print(f"\n3. 현재 상황:")
print(f"   - 실행 파일: {__file__}")
print(f"   - 작업 디렉토리: {os.getcwd()}")
print(f"   - Python 경로 첫 번째: {sys.path[0]}")

# 4. 패키지 구조 확인
paca_dir = os.path.join(project_root, "paca")
learning_dir = os.path.join(paca_dir, "learning")
core_dir = os.path.join(paca_dir, "core", "types")

print(f"\n4. 패키지 구조:")
print(f"   - paca/ 존재: {os.path.exists(paca_dir)}")
print(f"   - paca/learning/ 존재: {os.path.exists(learning_dir)}")
print(f"   - paca/core/types/ 존재: {os.path.exists(core_dir)}")

# 5. 해결 방법들
print(f"\n5. 해결 방법:")
print("   방법 1: 절대 임포트 사용")
print("     from paca.core.types.base import ID")
print("   방법 2: 모듈로 실행")
print("     python -m paca.learning.iis_calculator")
print("   방법 3: 패키지 설치")
print("     pip install -e .")

# 6. 실제 문제 파일 확인
iis_file = os.path.join(learning_dir, "iis_calculator.py")
if os.path.exists(iis_file):
    print(f"\n6. 문제 파일 분석:")
    print(f"   파일: {iis_file}")

    with open(iis_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[20:25], 21):
            if 'from ..' in line:
                print(f"   라인 {i}: {line.strip()} <- 문제 지점")

print(f"\n결론: Phase 2 모듈들의 상대 임포트를 절대 임포트로 수정 필요")