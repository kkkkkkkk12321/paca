"""
PACA v5 자동 문서화 생성기
코드 분석 기반 README.md 자동 생성 시스템
"""

import os
import sys
import ast
import inspect
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

@dataclass
class ModuleInfo:
    """모듈 정보"""
    name: str
    path: str
    description: str
    classes: List[str]
    functions: List[str]
    imports: List[str]
    dependencies: List[str]

@dataclass
class FileInfo:
    """파일 정보"""
    name: str
    path: str
    purpose: str
    line_count: int
    classes: List[str]
    functions: List[str]

class CodeAnalyzer:
    """코드 분석 엔진"""

    def __init__(self):
        self.module_cache = {}

    def analyze_python_file(self, file_path: str) -> FileInfo:
        """Python 파일 분석"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            classes = []
            functions = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    if not node.name.startswith('_'):  # 공개 함수만
                        functions.append(node.name)

            # 파일 목적 추출 (docstring에서)
            purpose = ""
            if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Str):
                purpose = tree.body[0].value.s.strip()

            return FileInfo(
                name=os.path.basename(file_path),
                path=file_path,
                purpose=purpose,
                line_count=len(content.splitlines()),
                classes=classes,
                functions=functions
            )

        except Exception as e:
            print(f"파일 분석 실패 {file_path}: {e}")
            return FileInfo(
                name=os.path.basename(file_path),
                path=file_path,
                purpose="분석 실패",
                line_count=0,
                classes=[],
                functions=[]
            )

    def analyze_module(self, module_path: str) -> ModuleInfo:
        """모듈 분석"""
        module_name = os.path.basename(module_path)

        # __init__.py 파일에서 모듈 정보 추출
        init_file = os.path.join(module_path, '__init__.py')
        description = ""
        imports = []

        if os.path.exists(init_file):
            try:
                with open(init_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                tree = ast.parse(content)

                # docstring 추출
                if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Str):
                    description = tree.body[0].value.s.strip()

                # import 문 추출
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)

            except Exception as e:
                print(f"__init__.py 분석 실패 {init_file}: {e}")

        # 모듈 내 Python 파일들 분석
        all_classes = []
        all_functions = []
        dependencies = []

        for file_path in Path(module_path).glob('*.py'):
            if file_path.name != '__init__.py':
                file_info = self.analyze_python_file(str(file_path))
                all_classes.extend(file_info.classes)
                all_functions.extend(file_info.functions)

        # 의존성 분석 (간단한 버전)
        dependencies = [imp for imp in imports if imp.startswith('paca.')]

        return ModuleInfo(
            name=module_name,
            path=module_path,
            description=description,
            classes=list(set(all_classes)),
            functions=list(set(all_functions)),
            imports=imports,
            dependencies=dependencies
        )

class TemplateEngine:
    """템플릿 처리 엔진"""

    def __init__(self):
        self.template = self._load_template()

    def _load_template(self) -> str:
        """9개 섹션 표준 템플릿"""
        return '''# 🎯 프로젝트 개요

{module_description}

## 📁 폴더/파일 구조

```
{folder_structure}
```

{file_descriptions}

## ⚙️ 기능 요구사항

**입력:**
{input_requirements}

**출력:**
{output_requirements}

**핵심 로직 흐름:**
{core_logic_flow}

## 🛠️ 기술적 요구사항

**언어 및 프레임워크:**
{tech_framework}

**주요 의존성:**
{dependencies}

**실행 환경:**
{runtime_environment}

## 🚀 라우팅 및 진입점

**주요 진입점:**
```python
{entry_points}
```

**API 경로:**
{api_routes}

## 📋 코드 품질 가이드

**주석 규칙:**
{comment_rules}

**네이밍 규칙:**
{naming_rules}

**예외 처리:**
{exception_handling}

## 🏃‍♂️ 실행 방법

**설치:**
```bash
{installation_commands}
```

**기본 사용법:**
```python
{usage_example}
```

**테스트 실행:**
```bash
{test_commands}
```

## 🧪 테스트 방법

**단위 테스트:**
{unit_tests}

**통합 테스트:**
{integration_tests}

**성능 테스트:**
{performance_tests}

**테스트 시나리오:**
```python
{test_scenarios}
```

## 💡 추가 고려사항

**보안:**
{security_considerations}

**성능:**
{performance_considerations}

**향후 개선:**
{future_improvements}

**모니터링:**
{monitoring_info}
'''

    def generate_readme(self, module_info: ModuleInfo, template_data: Dict[str, str]) -> str:
        """README.md 생성"""
        return self.template.format(**template_data)

class DependencyMapper:
    """의존성 매핑 도구"""

    def __init__(self):
        self.dependency_graph = {}

    def analyze_dependencies(self, modules: List[ModuleInfo]) -> Dict[str, List[str]]:
        """모듈 간 의존성 분석"""
        dependency_map = {}

        for module in modules:
            dependencies = []
            for dep in module.dependencies:
                # paca 내부 의존성만 추출
                if dep.startswith('paca.'):
                    dep_module = dep.split('.')[1] if len(dep.split('.')) > 1 else dep
                    dependencies.append(dep_module)

            dependency_map[module.name] = dependencies

        return dependency_map

    def generate_dependency_tree(self, dependency_map: Dict[str, List[str]]) -> str:
        """의존성 트리 생성"""
        lines = []
        for module, deps in dependency_map.items():
            lines.append(f"📁 {module}/")
            for dep in deps:
                lines.append(f"  ↳ depends on: {dep}")
            lines.append("")

        return "\n".join(lines)

class DocumentationGenerator:
    """메인 문서화 생성기"""

    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.template_engine = TemplateEngine()
        self.dependency_mapper = DependencyMapper()

    def generate_module_readme(self, module_path: str, output_path: Optional[str] = None) -> str:
        """모듈 README.md 생성"""
        module_info = self.analyzer.analyze_module(module_path)

        # 템플릿 데이터 준비
        template_data = self._prepare_template_data(module_info)

        # README 생성
        readme_content = self.template_engine.generate_readme(module_info, template_data)

        # 파일 저장
        if output_path is None:
            output_path = os.path.join(module_path, 'README.md')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print(f"✓ README.md 생성 완료: {output_path}")
        return readme_content

    def _prepare_template_data(self, module_info: ModuleInfo) -> Dict[str, str]:
        """템플릿 데이터 준비"""
        return {
            'module_description': module_info.description or f"PACA v5 {module_info.name} 모듈",
            'folder_structure': self._generate_folder_structure(module_info),
            'file_descriptions': self._generate_file_descriptions(module_info),
            'input_requirements': "- 모듈별 입력 데이터\n- 설정 및 파라미터",
            'output_requirements': "- 처리된 결과 데이터\n- 상태 및 메트릭",
            'core_logic_flow': "1. 입력 데이터 검증\n2. 핵심 로직 실행\n3. 결과 처리 및 반환",
            'tech_framework': "- Python 3.8+\n- asyncio (비동기 처리)",
            'dependencies': self._generate_dependencies(module_info),
            'runtime_environment': "- 메모리: 최소 128MB\n- Python 환경 필요",
            'entry_points': self._generate_entry_points(module_info),
            'api_routes': self._generate_api_routes(module_info),
            'comment_rules': "- 모든 클래스와 메서드에 docstring 필수\n- 복잡한 로직은 단계별 주석 추가",
            'naming_rules': "- 클래스: PascalCase\n- 메서드: snake_case\n- 상수: UPPER_CASE",
            'exception_handling': "- ModuleError: 일반적인 모듈 오류\n- ValidationError: 입력 검증 실패",
            'installation_commands': "# 프로젝트 루트에서\npip install -e .",
            'usage_example': self._generate_usage_example(module_info),
            'test_commands': f"# {module_info.name} 모듈 테스트\npython -m pytest tests/{module_info.name}/ -v",
            'unit_tests': f"- {module_info.name} 모듈의 개별 기능 테스트\n- 핵심 로직 정확성 검증",
            'integration_tests': "- 다른 모듈과의 통합 테스트\n- 전체 워크플로우 검증",
            'performance_tests': "- 응답 시간 및 처리량 측정\n- 메모리 사용량 최적화 검증",
            'test_scenarios': self._generate_test_scenarios(module_info),
            'security_considerations': "- 입력 데이터 검증 및 sanitization\n- 접근 권한 제어",
            'performance_considerations': "- 캐싱을 통한 성능 최적화\n- 비동기 처리로 동시성 향상",
            'future_improvements': "- 기능 확장 및 최적화\n- 더 나은 사용자 경험 제공",
            'monitoring_info': "- 성능 메트릭 수집\n- 오류 및 예외 추적"
        }

    def _generate_folder_structure(self, module_info: ModuleInfo) -> str:
        """폴더 구조 생성"""
        lines = [f"{module_info.name}/"]
        lines.append("├── 📄 __init__.py           # 모듈 초기화")

        # 모듈 내 파일들 탐색
        module_path = Path(module_info.path)
        for file_path in sorted(module_path.glob('*.py')):
            if file_path.name != '__init__.py':
                lines.append(f"├── 📄 {file_path.name:<20} # {file_path.stem} 구현")

        # 하위 디렉토리들
        for dir_path in sorted(module_path.iterdir()):
            if dir_path.is_dir() and not dir_path.name.startswith('__'):
                lines.append(f"├── 📁 {dir_path.name}/               # {dir_path.name} 하위 시스템")

        lines.append("└── 📄 README.md             # 이 문서")

        return "\n".join(lines)

    def _generate_file_descriptions(self, module_info: ModuleInfo) -> str:
        """파일 설명 생성"""
        lines = ["**파일별 설명:**"]

        # 주요 클래스들을 기반으로 설명 생성
        if module_info.classes:
            main_classes = module_info.classes[:3]  # 상위 3개 클래스
            lines.append(f"- 주요 클래스: {', '.join(main_classes)}")

        if module_info.functions:
            main_functions = module_info.functions[:3]  # 상위 3개 함수
            lines.append(f"- 주요 함수: {', '.join(main_functions)}")

        return "\n".join(lines)

    def _generate_dependencies(self, module_info: ModuleInfo) -> str:
        """의존성 생성"""
        lines = []

        # PACA 내부 의존성
        internal_deps = [dep for dep in module_info.dependencies if dep.startswith('paca.')]
        if internal_deps:
            for dep in internal_deps:
                lines.append(f"- `{dep}`: 내부 모듈 의존성")

        # 외부 의존성 (일반적인 것들)
        external_deps = [dep for dep in module_info.imports if not dep.startswith('paca.')]
        if external_deps:
            for dep in external_deps[:3]:  # 상위 3개만
                lines.append(f"- `{dep}`: 외부 라이브러리")

        return "\n".join(lines) if lines else "- 추가 의존성 없음"

    def _generate_entry_points(self, module_info: ModuleInfo) -> str:
        """진입점 생성"""
        lines = [f"from paca.{module_info.name} import ("]

        # 주요 클래스들 추가
        for cls in module_info.classes[:5]:  # 상위 5개
            lines.append(f"    {cls},")

        lines.append(")")
        lines.append("")
        lines.append(f"# {module_info.name} 사용 예시")

        if module_info.classes:
            main_class = module_info.classes[0]
            lines.append(f"{main_class.lower()} = {main_class}()")
            lines.append(f"result = {main_class.lower()}.process(data)")

        return "\n".join(lines)

    def _generate_api_routes(self, module_info: ModuleInfo) -> str:
        """API 경로 생성"""
        lines = []

        if module_info.functions:
            for func in module_info.functions[:3]:
                lines.append(f"- `{module_info.name}.{func}()`: {func} 기능 실행")

        return "\n".join(lines) if lines else f"- `{module_info.name}.main()`: 메인 인터페이스"

    def _generate_usage_example(self, module_info: ModuleInfo) -> str:
        """사용 예시 생성"""
        lines = [f"from paca.{module_info.name} import {module_info.classes[0] if module_info.classes else 'main'}"]
        lines.append("")

        if module_info.classes:
            main_class = module_info.classes[0]
            lines.append(f"# {main_class} 사용")
            lines.append(f"instance = {main_class}()")
            lines.append("result = instance.process(input_data)")
            lines.append("print(f'결과: {result}')")
        else:
            lines.append(f"# {module_info.name} 모듈 사용")
            lines.append("result = main(input_data)")
            lines.append("print(f'결과: {result}')")

        return "\n".join(lines)

    def _generate_test_scenarios(self, module_info: ModuleInfo) -> str:
        """테스트 시나리오 생성"""
        lines = [f"def test_{module_info.name}_basic():"]
        lines.append(f'    """기본 {module_info.name} 테스트"""')

        if module_info.classes:
            main_class = module_info.classes[0]
            lines.append(f"    instance = {main_class}()")
            lines.append("    result = instance.process(test_data)")
            lines.append("    assert result.is_success")

        return "\n".join(lines)

    def generate_all_missing_readmes(self, paca_path: str):
        """누락된 모든 README.md 생성"""
        paca_modules_path = os.path.join(paca_path, 'paca')

        if not os.path.exists(paca_modules_path):
            print(f"❌ PACA 모듈 경로를 찾을 수 없습니다: {paca_modules_path}")
            return

        generated_count = 0

        # 모든 하위 디렉토리 검사
        for item in os.listdir(paca_modules_path):
            module_path = os.path.join(paca_modules_path, item)

            if os.path.isdir(module_path) and not item.startswith('__'):
                readme_path = os.path.join(module_path, 'README.md')

                # README.md가 없는 경우에만 생성
                if not os.path.exists(readme_path):
                    print(f"📝 {item} 모듈의 README.md 생성 중...")
                    self.generate_module_readme(module_path)
                    generated_count += 1
                else:
                    print(f"✓ {item} 모듈의 README.md가 이미 존재합니다.")

        print(f"\n🎉 총 {generated_count}개의 README.md 파일을 생성했습니다!")

def main():
    """메인 실행 함수"""
    print("PACA v5 자동 문서화 생성기 시작")
    print("=" * 50)

    # PACA 프로젝트 경로
    paca_project_path = str(PROJECT_ROOT)

    # 문서화 생성기 초기화
    doc_generator = DocumentationGenerator()

    # 누락된 README 파일들 자동 생성
    doc_generator.generate_all_missing_readmes(paca_project_path)

    print("\n✅ 자동 문서화 생성 완료!")
    print(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()