#!/usr/bin/env python3
"""
Phase 2 확장성 및 모듈화 리스크 분석
향후 기능 확장시 발생할 수 있는 문제점들을 미리 검토
"""

import os
import sys
import ast
import re
from typing import List, Dict, Set, Tuple

def analyze_import_dependencies():
    """Phase 2 모듈들의 임포트 의존성 분석"""
    print("=== Phase 2 임포트 의존성 분석 ===")

    project_root = os.path.dirname(os.path.abspath(__file__))
    learning_dir = os.path.join(project_root, "paca", "learning")

    phase2_modules = [
        "iis_calculator.py",
        "autonomous_trainer.py",
        "tactic_generator.py"
    ]

    dependencies = {}

    for module in phase2_modules:
        module_path = os.path.join(learning_dir, module)
        if os.path.exists(module_path):
            deps = extract_imports_from_file(module_path)
            dependencies[module] = deps
            print(f"\n📁 {module}:")
            for dep_type, dep_list in deps.items():
                if dep_list:
                    print(f"  {dep_type}: {len(dep_list)}개")
                    for dep in dep_list[:3]:  # 상위 3개만 표시
                        print(f"    - {dep}")

    return dependencies

def extract_imports_from_file(file_path: str) -> Dict[str, List[str]]:
    """파일에서 임포트 구문 추출"""
    imports = {
        "relative": [],      # ..core.types.base
        "absolute": [],      # paca.core.types.base
        "standard": [],      # asyncio, time 등
        "third_party": []    # numpy, pandas 등 (있다면)
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # AST로 파싱하여 임포트 구문 추출
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith(".."):
                    imports["relative"].append(module)
                elif module.startswith("paca"):
                    imports["absolute"].append(module)
                elif module in ["asyncio", "time", "json", "statistics", "math", "random", "re"]:
                    imports["standard"].append(module)
                else:
                    imports["third_party"].append(module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name
                    if module in ["asyncio", "time", "json", "statistics", "math", "random", "re"]:
                        imports["standard"].append(module)
                    else:
                        imports["third_party"].append(module)
    except Exception as e:
        print(f"파일 분석 오류 {file_path}: {e}")

    return imports

def analyze_future_expansion_risks():
    """향후 확장시 예상되는 리스크 분석"""
    print("\n=== 향후 확장 리스크 분석 ===")

    risks = [
        {
            "scenario": "Phase 3 성능 모니터링 모듈 추가",
            "potential_issues": [
                "새 모듈이 Phase 2 모듈들을 임포트할 때 상대 임포트 문제",
                "performance/ 서브패키지 생성시 임포트 경로 변경 필요",
                "하드웨어 모니터링 라이브러리 추가시 의존성 충돌"
            ],
            "risk_level": "Medium"
        },
        {
            "scenario": "Phase 4 GUI 모듈 추가",
            "potential_issues": [
                "desktop_app/ 서브패키지에서 learning 모듈 임포트시 경로 문제",
                "GUI 라이브러리(tkinter, PyQt) 추가시 패키지 구조 변경",
                "실시간 데이터 표시를 위한 순환 임포트 위험"
            ],
            "risk_level": "High"
        },
        {
            "scenario": "모듈 재구성/리팩토링",
            "potential_issues": [
                "learning/ 내부 서브패키지 분할시 모든 상대 임포트 수정 필요",
                "core/ 모듈 구조 변경시 Phase 2 모듈들 일괄 수정",
                "네임스페이스 변경시 외부 사용자 코드 호환성 문제"
            ],
            "risk_level": "High"
        },
        {
            "scenario": "다른 프로젝트에서 부분 사용",
            "potential_issues": [
                "IIS 계산기만 따로 사용하고 싶을 때 core 의존성 문제",
                "전술 생성기만 독립적으로 사용하기 어려움",
                "라이센스 문제로 특정 모듈만 배포시 의존성 해결"
            ],
            "risk_level": "Medium"
        },
        {
            "scenario": "테스트 및 CI/CD 환경",
            "potential_issues": [
                "개별 모듈 단위 테스트시 상대 임포트 문제",
                "CI/CD 파이프라인에서 모듈별 테스트 불가",
                "도커 컨테이너 등 격리 환경에서 실행 문제"
            ],
            "risk_level": "Medium"
        }
    ]

    for i, risk in enumerate(risks, 1):
        print(f"\n🔍 시나리오 {i}: {risk['scenario']}")
        print(f"위험도: {risk['risk_level']}")
        print("예상 문제점:")
        for issue in risk['potential_issues']:
            print(f"  ❌ {issue}")

    return risks

def analyze_current_coupling():
    """현재 Phase 2 모듈들의 결합도 분석"""
    print("\n=== Phase 2 모듈 결합도 분석 ===")

    # Phase 2 모듈간 상호 의존성 확인
    coupling_analysis = {
        "iis_calculator.py": {
            "depends_on": ["core.types.base"],
            "used_by": ["autonomous_trainer", "__init__"],
            "coupling_level": "Low"
        },
        "autonomous_trainer.py": {
            "depends_on": ["core.types.base", "iis_calculator"],
            "used_by": ["__init__"],
            "coupling_level": "Medium"
        },
        "tactic_generator.py": {
            "depends_on": ["core.types.base"],
            "used_by": ["__init__"],
            "coupling_level": "Low"
        }
    }

    for module, info in coupling_analysis.items():
        print(f"\n📦 {module}:")
        print(f"  의존: {', '.join(info['depends_on'])}")
        print(f"  사용처: {', '.join(info['used_by'])}")
        print(f"  결합도: {info['coupling_level']}")

    return coupling_analysis

def suggest_future_proof_solutions():
    """미래 확장을 위한 해결책 제안"""
    print("\n=== 미래 확장을 위한 해결책 ===")

    solutions = [
        {
            "problem": "상대 임포트 의존성",
            "solution": "조건부 임포트 패턴 적용",
            "implementation": """
try:
    from ..core.types.base import ID, Timestamp
except ImportError:
    from paca.core.types.base import ID, Timestamp
            """,
            "benefits": ["모든 실행 방식 지원", "모듈화 확장 용이", "독립 실행 가능"]
        },
        {
            "problem": "모듈간 강결합",
            "solution": "의존성 주입 패턴 도입",
            "implementation": """
class IISCalculator:
    def __init__(self, type_factory=None):
        if type_factory is None:
            from .types import create_success, create_failure
            self.create_success = create_success
            self.create_failure = create_failure
        else:
            self.create_success = type_factory.create_success
            self.create_failure = type_factory.create_failure
            """,
            "benefits": ["테스트 용이성", "모듈 독립성", "확장성 향상"]
        },
        {
            "problem": "패키지 구조 변경 리스크",
            "solution": "공개 API 인터페이스 정의",
            "implementation": """
# paca/learning/api.py
from .iis_calculator import IISCalculator
from .autonomous_trainer import AutonomousTrainer
from .tactic_generator import TacticGenerator

__all__ = ['IISCalculator', 'AutonomousTrainer', 'TacticGenerator']
            """,
            "benefits": ["API 안정성", "하위 호환성", "리팩토링 자유도"]
        }
    ]

    for i, solution in enumerate(solutions, 1):
        print(f"\n💡 해결책 {i}: {solution['problem']}")
        print(f"방법: {solution['solution']}")
        print("구현 예시:")
        print(solution['implementation'])
        print("장점:")
        for benefit in solution['benefits']:
            print(f"  ✅ {benefit}")

def main():
    """메인 분석 실행"""
    print("🔬 PACA Phase 2 확장성 및 모듈화 분석")
    print("=" * 60)

    # 1. 현재 의존성 분석
    dependencies = analyze_import_dependencies()

    # 2. 향후 확장 리스크 분석
    risks = analyze_future_expansion_risks()

    # 3. 현재 결합도 분석
    coupling = analyze_current_coupling()

    # 4. 해결책 제안
    suggest_future_proof_solutions()

    # 5. 종합 결론
    print("\n" + "=" * 60)
    print("🎯 종합 결론 및 권장사항")
    print("=" * 60)

    print("\n❌ 현재 리스크:")
    print("1. 상대 임포트로 인한 모듈 독립 실행 불가")
    print("2. 패키지 구조 변경시 일괄 수정 필요")
    print("3. 새로운 서브패키지 추가시 임포트 경로 문제")
    print("4. CI/CD 및 테스트 환경에서 제약")

    print("\n✅ 권장 해결 방향:")
    print("1. 즉시: 조건부 임포트 패턴 적용 (호환성 확보)")
    print("2. 단기: 공개 API 인터페이스 정의 (API 안정성)")
    print("3. 중기: 의존성 주입 패턴 도입 (모듈 독립성)")
    print("4. 장기: 마이크로서비스 아키텍처 고려")

    print(f"\n🔒 안정성 확신을 위한 필수 조치:")
    print("✅ 조건부 임포트 적용 → 미래 확장시 안전성 보장")
    print("✅ API 인터페이스 정의 → 하위 호환성 유지")
    print("✅ 단위 테스트 강화 → 변경시 안정성 검증")

if __name__ == "__main__":
    main()