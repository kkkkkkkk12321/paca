"""
PACA v5 코드 분석 엔진
Python 코드 구조 분석 및 메타데이터 추출
"""

import os
import ast
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import re

@dataclass
class ClassInfo:
    """클래스 정보"""
    name: str
    bases: List[str]
    methods: List[str]
    properties: List[str]
    docstring: Optional[str]
    decorators: List[str]
    line_number: int
    is_abstract: bool = False

@dataclass
class FunctionInfo:
    """함수 정보"""
    name: str
    args: List[str]
    returns: Optional[str]
    docstring: Optional[str]
    decorators: List[str]
    line_number: int
    is_async: bool = False
    is_property: bool = False

@dataclass
class ImportInfo:
    """임포트 정보"""
    module: str
    names: List[str]
    alias: Optional[str]
    level: int  # relative import level
    line_number: int

@dataclass
class ModuleStructure:
    """모듈 구조 정보"""
    name: str
    path: str
    docstring: Optional[str]
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    constants: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    complexity_score: float = 0.0
    line_count: int = 0

@dataclass
class ProjectStructure:
    """프로젝트 구조 정보"""
    name: str
    root_path: str
    modules: Dict[str, ModuleStructure] = field(default_factory=dict)
    dependency_graph: Dict[str, Set[str]] = field(default_factory=dict)
    test_coverage: Dict[str, float] = field(default_factory=dict)
    documentation_coverage: Dict[str, float] = field(default_factory=dict)

class ASTAnalyzer:
    """AST 기반 코드 분석기"""

    def __init__(self):
        self.current_module = None
        self.analysis_cache = {}

    def analyze_file(self, file_path: str) -> ModuleStructure:
        """Python 파일 AST 분석"""
        if file_path in self.analysis_cache:
            return self.analysis_cache[file_path]

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=file_path)
            module_name = self._get_module_name(file_path)

            structure = ModuleStructure(
                name=module_name,
                path=file_path,
                docstring=self._extract_docstring(tree),
                line_count=len(content.splitlines())
            )

            # AST 노드 순회하여 분석
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node)
                    structure.classes.append(class_info)

                elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    # 클래스 내부 메서드가 아닌 경우만
                    if not self._is_method(node, tree):
                        function_info = self._analyze_function(node)
                        structure.functions.append(function_info)

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_info = self._analyze_import(node)
                    structure.imports.append(import_info)

                elif isinstance(node, ast.Assign):
                    self._analyze_assignment(node, structure)

            # 의존성 및 복잡도 계산
            structure.dependencies = self._extract_dependencies(structure.imports)
            structure.complexity_score = self._calculate_complexity(tree)

            self.analysis_cache[file_path] = structure
            return structure

        except Exception as e:
            print(f"❌ 파일 분석 실패 {file_path}: {e}")
            return ModuleStructure(
                name=self._get_module_name(file_path),
                path=file_path,
                docstring=f"분석 실패: {str(e)}"
            )

    def _get_module_name(self, file_path: str) -> str:
        """파일 경로에서 모듈명 추출"""
        return Path(file_path).stem

    def _extract_docstring(self, node) -> Optional[str]:
        """docstring 추출"""
        if (hasattr(node, 'body') and node.body and
            isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Str)):
            return node.body[0].value.s.strip()
        return None

    def _analyze_class(self, node: ast.ClassDef) -> ClassInfo:
        """클래스 분석"""
        bases = [self._get_name(base) for base in node.bases]
        methods = []
        properties = []

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)

                # @property 데코레이터 확인
                if any(self._get_decorator_name(dec) == 'property' for dec in item.decorator_list):
                    properties.append(item.name)

        decorators = [self._get_decorator_name(dec) for dec in node.decorator_list]
        is_abstract = 'abstractmethod' in decorators or any('ABC' in base for base in bases)

        return ClassInfo(
            name=node.name,
            bases=bases,
            methods=methods,
            properties=properties,
            docstring=self._extract_docstring(node),
            decorators=decorators,
            line_number=node.lineno,
            is_abstract=is_abstract
        )

    def _analyze_function(self, node) -> FunctionInfo:
        """함수 분석"""
        args = [arg.arg for arg in node.args.args]

        # 반환 타입 추출
        returns = None
        if hasattr(node, 'returns') and node.returns:
            returns = self._get_name(node.returns)

        decorators = [self._get_decorator_name(dec) for dec in node.decorator_list]
        is_property = 'property' in decorators

        return FunctionInfo(
            name=node.name,
            args=args,
            returns=returns,
            docstring=self._extract_docstring(node),
            decorators=decorators,
            line_number=node.lineno,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_property=is_property
        )

    def _analyze_import(self, node) -> ImportInfo:
        """import 문 분석"""
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            return ImportInfo(
                module=names[0] if names else "",
                names=names,
                alias=node.names[0].asname if node.names and node.names[0].asname else None,
                level=0,
                line_number=node.lineno
            )

        elif isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names] if node.names else []
            return ImportInfo(
                module=node.module or "",
                names=names,
                alias=None,
                level=node.level,
                line_number=node.lineno
            )

    def _analyze_assignment(self, node: ast.Assign, structure: ModuleStructure):
        """변수 할당 분석"""
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id

                # 상수 판별 (대문자로만 구성)
                if var_name.isupper():
                    structure.constants[var_name] = self._extract_value(node.value)
                else:
                    structure.variables[var_name] = self._extract_value(node.value)

    def _extract_value(self, node) -> Any:
        """AST 노드에서 값 추출"""
        if isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.NameConstant):
            return node.value
        elif isinstance(node, (ast.List, ast.Tuple)):
            return [self._extract_value(elem) for elem in node.elts]
        elif isinstance(node, ast.Dict):
            return {self._extract_value(k): self._extract_value(v)
                   for k, v in zip(node.keys, node.values)}
        else:
            return f"<{type(node).__name__}>"

    def _get_name(self, node) -> str:
        """AST 노드에서 이름 추출"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif hasattr(node, 'id'):
            return node.id
        else:
            return str(node)

    def _get_decorator_name(self, decorator) -> str:
        """데코레이터 이름 추출"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        elif isinstance(decorator, ast.Call):
            return self._get_decorator_name(decorator.func)
        else:
            return str(decorator)

    def _is_method(self, func_node, tree) -> bool:
        """함수가 클래스 메서드인지 확인"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if item == func_node:
                        return True
        return False

    def _extract_dependencies(self, imports: List[ImportInfo]) -> Set[str]:
        """의존성 추출"""
        dependencies = set()

        for imp in imports:
            if imp.module.startswith('paca.'):
                dependencies.add(imp.module)

            # from paca.module import something
            if imp.level > 0:  # relative import
                dependencies.add('paca.' + imp.module if imp.module else 'paca')

        return dependencies

    def _calculate_complexity(self, tree) -> float:
        """순환 복잡도 계산 (McCabe 복잡도 기반)"""
        complexity = 1  # 기본 복잡도

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1

        return complexity

class ProjectAnalyzer:
    """프로젝트 레벨 분석기"""

    def __init__(self):
        self.ast_analyzer = ASTAnalyzer()

    def analyze_project(self, project_root: str) -> ProjectStructure:
        """전체 프로젝트 분석"""
        project_name = os.path.basename(project_root)
        project = ProjectStructure(
            name=project_name,
            root_path=project_root
        )

        # PACA 모듈 디렉토리 찾기
        paca_dir = os.path.join(project_root, 'paca')
        if not os.path.exists(paca_dir):
            print(f"❌ PACA 모듈 디렉토리를 찾을 수 없습니다: {paca_dir}")
            return project

        # 모든 Python 파일 분석
        for root, dirs, files in os.walk(paca_dir):
            # __pycache__ 디렉토리 제외
            dirs[:] = [d for d in dirs if not d.startswith('__pycache__')]

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, paca_dir)
                    module_key = relative_path.replace(os.sep, '.').replace('.py', '')

                    module_structure = self.ast_analyzer.analyze_file(file_path)
                    project.modules[module_key] = module_structure

        # 의존성 그래프 구축
        project.dependency_graph = self._build_dependency_graph(project.modules)

        # 문서화 커버리지 계산
        project.documentation_coverage = self._calculate_doc_coverage(project.modules)

        return project

    def _build_dependency_graph(self, modules: Dict[str, ModuleStructure]) -> Dict[str, Set[str]]:
        """모듈 간 의존성 그래프 구축"""
        graph = defaultdict(set)

        for module_name, module_structure in modules.items():
            for dep in module_structure.dependencies:
                # paca. 접두사 제거하고 모듈명만 추출
                if dep.startswith('paca.'):
                    dep_module = dep[5:]  # 'paca.' 제거
                    if dep_module in modules:
                        graph[module_name].add(dep_module)

        return dict(graph)

    def _calculate_doc_coverage(self, modules: Dict[str, ModuleStructure]) -> Dict[str, float]:
        """문서화 커버리지 계산"""
        coverage = {}

        for module_name, module_structure in modules.items():
            total_items = 1  # 모듈 자체
            documented_items = 1 if module_structure.docstring else 0

            # 클래스 문서화 확인
            total_items += len(module_structure.classes)
            documented_items += sum(1 for cls in module_structure.classes if cls.docstring)

            # 함수 문서화 확인
            total_items += len(module_structure.functions)
            documented_items += sum(1 for func in module_structure.functions if func.docstring)

            # 커버리지 계산
            coverage[module_name] = (documented_items / total_items * 100) if total_items > 0 else 0

        return coverage

    def generate_analysis_report(self, project: ProjectStructure) -> str:
        """분석 리포트 생성"""
        lines = []
        lines.append("# PACA v5 코드 분석 리포트")
        lines.append("=" * 50)
        lines.append("")

        # 프로젝트 개요
        lines.append("## 📊 프로젝트 개요")
        lines.append(f"- **프로젝트명**: {project.name}")
        lines.append(f"- **루트 경로**: {project.root_path}")
        lines.append(f"- **모듈 수**: {len(project.modules)}")
        lines.append("")

        # 모듈별 통계
        lines.append("## 🏗️ 모듈별 통계")
        lines.append("| 모듈 | 클래스 | 함수 | 라인 수 | 복잡도 | 문서화율 |")
        lines.append("|------|--------|------|---------|--------|----------|")

        for module_name, module_structure in sorted(project.modules.items()):
            doc_coverage = project.documentation_coverage.get(module_name, 0)
            lines.append(f"| {module_name} | {len(module_structure.classes)} | "
                        f"{len(module_structure.functions)} | {module_structure.line_count} | "
                        f"{module_structure.complexity_score:.1f} | {doc_coverage:.1f}% |")

        lines.append("")

        # 의존성 그래프
        lines.append("## 🔗 의존성 관계")
        for module, deps in project.dependency_graph.items():
            if deps:
                lines.append(f"- **{module}** → {', '.join(sorted(deps))}")

        lines.append("")

        # 문서화 품질 분석
        lines.append("## 📚 문서화 품질")
        avg_coverage = sum(project.documentation_coverage.values()) / len(project.documentation_coverage)
        lines.append(f"- **평균 문서화율**: {avg_coverage:.1f}%")

        low_doc_modules = [name for name, coverage in project.documentation_coverage.items()
                          if coverage < 50]
        if low_doc_modules:
            lines.append(f"- **문서화 부족 모듈**: {', '.join(low_doc_modules)}")

        lines.append("")

        # 개선 권장사항
        lines.append("## 💡 개선 권장사항")

        high_complexity_modules = [name for name, module in project.modules.items()
                                 if module.complexity_score > 20]
        if high_complexity_modules:
            lines.append(f"- **복잡도 개선 필요**: {', '.join(high_complexity_modules)}")

        if low_doc_modules:
            lines.append(f"- **문서화 개선 필요**: {', '.join(low_doc_modules)}")

        # 순환 의존성 검사
        circular_deps = self._detect_circular_dependencies(project.dependency_graph)
        if circular_deps:
            lines.append(f"- **순환 의존성 발견**: {circular_deps}")

        return "\n".join(lines)

    def _detect_circular_dependencies(self, dependency_graph: Dict[str, Set[str]]) -> List[str]:
        """순환 의존성 탐지"""
        # 간단한 DFS 기반 순환 탐지
        visited = set()
        rec_stack = set()
        circular_paths = []

        def dfs(node, path):
            if node in rec_stack:
                # 순환 발견
                cycle_start = path.index(node)
                circular_paths.append(" → ".join(path[cycle_start:] + [node]))
                return True

            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)

            for neighbor in dependency_graph.get(node, set()):
                if dfs(neighbor, path + [neighbor]):
                    break

            rec_stack.remove(node)
            return False

        for node in dependency_graph:
            if node not in visited:
                dfs(node, [node])

        return circular_paths

def main():
    """메인 실행 함수"""
    print("PACA v5 코드 분석 엔진 시작")
    print("=" * 50)

    # 프로젝트 루트 경로
    project_root = Path(__file__).parent.parent

    # 프로젝트 분석
    analyzer = ProjectAnalyzer()
    project_structure = analyzer.analyze_project(str(project_root))

    # 분석 리포트 생성
    report = analyzer.generate_analysis_report(project_structure)

    # 리포트 저장
    report_path = project_root / "analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✅ 코드 분석 완료!")
    print(f"📊 분석 리포트: {report_path}")
    print(f"📈 분석된 모듈 수: {len(project_structure.modules)}")

    # 간단한 통계 출력
    total_classes = sum(len(m.classes) for m in project_structure.modules.values())
    total_functions = sum(len(m.functions) for m in project_structure.modules.values())
    avg_doc_coverage = sum(project_structure.documentation_coverage.values()) / len(project_structure.documentation_coverage)

    print(f"📋 총 클래스: {total_classes}")
    print(f"🔧 총 함수: {total_functions}")
    print(f"📚 평균 문서화율: {avg_doc_coverage:.1f}%")

if __name__ == "__main__":
    main()