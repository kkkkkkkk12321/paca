"""
PACA v5 의존성 매핑 도구
모듈 간 의존성 관계 분석 및 시각화
"""

import os
import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json

class DependencyType(Enum):
    """의존성 타입"""
    IMPORT = "import"
    INHERITANCE = "inheritance"
    COMPOSITION = "composition"
    FUNCTION_CALL = "function_call"
    TYPE_HINT = "type_hint"

@dataclass
class Dependency:
    """의존성 정보"""
    source: str
    target: str
    dependency_type: DependencyType
    line_number: int
    context: str = ""
    strength: float = 1.0  # 의존성 강도 (0.0 ~ 1.0)

@dataclass
class ModuleDependencies:
    """모듈 의존성 정보"""
    module_name: str
    internal_dependencies: Set[str] = field(default_factory=set)
    external_dependencies: Set[str] = field(default_factory=set)
    dependency_details: List[Dependency] = field(default_factory=list)
    cyclic_dependencies: List[str] = field(default_factory=list)
    fan_in: int = 0  # 이 모듈에 의존하는 다른 모듈 수
    fan_out: int = 0  # 이 모듈이 의존하는 다른 모듈 수

@dataclass
class DependencyGraph:
    """의존성 그래프"""
    modules: Dict[str, ModuleDependencies] = field(default_factory=dict)
    adjacency_list: Dict[str, Set[str]] = field(default_factory=dict)
    reverse_adjacency_list: Dict[str, Set[str]] = field(default_factory=dict)
    cycles: List[List[str]] = field(default_factory=list)
    strongly_connected_components: List[List[str]] = field(default_factory=list)

class DependencyAnalyzer:
    """의존성 분석기"""

    def __init__(self):
        self.paca_modules = set()
        self.analysis_cache = {}

    def analyze_file(self, file_path: str, module_name: str) -> ModuleDependencies:
        """파일의 의존성 분석"""
        if file_path in self.analysis_cache:
            return self.analysis_cache[file_path]

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=file_path)

            dependencies = ModuleDependencies(module_name=module_name)

            # AST 순회하여 의존성 추출
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    self._analyze_import(node, dependencies)
                elif isinstance(node, ast.ClassDef):
                    self._analyze_inheritance(node, dependencies)
                elif isinstance(node, ast.FunctionDef):
                    self._analyze_function_dependencies(node, dependencies)
                elif isinstance(node, ast.Assign):
                    self._analyze_composition(node, dependencies)

            # Fan-in/Fan-out 계산
            dependencies.fan_out = len(dependencies.internal_dependencies)

            self.analysis_cache[file_path] = dependencies
            return dependencies

        except Exception as e:
            print(f"❌ 의존성 분석 실패 {file_path}: {e}")
            return ModuleDependencies(module_name=module_name)

    def _analyze_import(self, node, dependencies: ModuleDependencies):
        """import 문 분석"""
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                self._categorize_dependency(module, dependencies, node.lineno, DependencyType.IMPORT)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module
                # Relative imports 처리
                if node.level > 0:
                    module = f"paca.{module}" if module else "paca"

                self._categorize_dependency(module, dependencies, node.lineno, DependencyType.IMPORT)

    def _analyze_inheritance(self, node: ast.ClassDef, dependencies: ModuleDependencies):
        """상속 관계 분석"""
        for base in node.bases:
            base_name = self._get_node_name(base)
            if base_name and '.' in base_name:
                module = '.'.join(base_name.split('.')[:-1])
                self._categorize_dependency(
                    module, dependencies, node.lineno,
                    DependencyType.INHERITANCE, f"class {node.name} inherits from {base_name}"
                )

    def _analyze_function_dependencies(self, node: ast.FunctionDef, dependencies: ModuleDependencies):
        """함수 내 의존성 분석"""
        # Type hints 분석
        if hasattr(node, 'returns') and node.returns:
            return_type = self._get_node_name(node.returns)
            if return_type and '.' in return_type:
                module = '.'.join(return_type.split('.')[:-1])
                self._categorize_dependency(
                    module, dependencies, node.lineno,
                    DependencyType.TYPE_HINT, f"return type: {return_type}"
                )

        # 함수 호출 분석
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.Call):
                func_name = self._get_node_name(sub_node.func)
                if func_name and '.' in func_name:
                    module = '.'.join(func_name.split('.')[:-1])
                    self._categorize_dependency(
                        module, dependencies, sub_node.lineno,
                        DependencyType.FUNCTION_CALL, f"calls {func_name}"
                    )

    def _analyze_composition(self, node: ast.Assign, dependencies: ModuleDependencies):
        """컴포지션 관계 분석"""
        if isinstance(node.value, ast.Call):
            func_name = self._get_node_name(node.value.func)
            if func_name and '.' in func_name:
                module = '.'.join(func_name.split('.')[:-1])
                self._categorize_dependency(
                    module, dependencies, node.lineno,
                    DependencyType.COMPOSITION, f"creates instance of {func_name}"
                )

    def _get_node_name(self, node) -> Optional[str]:
        """AST 노드에서 이름 추출"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value_name = self._get_node_name(node.value)
            return f"{value_name}.{node.attr}" if value_name else node.attr
        elif hasattr(node, 'id'):
            return node.id
        return None

    def _categorize_dependency(
        self,
        module: str,
        dependencies: ModuleDependencies,
        line_number: int,
        dep_type: DependencyType,
        context: str = ""
    ):
        """의존성 분류"""
        if not module:
            return

        # PACA 내부 모듈인지 확인
        if module.startswith('paca.'):
            clean_module = module[5:]  # 'paca.' 제거
            dependencies.internal_dependencies.add(clean_module)
            strength = self._calculate_dependency_strength(dep_type)
        else:
            dependencies.external_dependencies.add(module)
            strength = 0.5  # 외부 의존성은 낮은 강도

        # 의존성 상세 정보 저장
        dependency = Dependency(
            source=dependencies.module_name,
            target=module,
            dependency_type=dep_type,
            line_number=line_number,
            context=context,
            strength=strength
        )
        dependencies.dependency_details.append(dependency)

    def _calculate_dependency_strength(self, dep_type: DependencyType) -> float:
        """의존성 강도 계산"""
        strength_map = {
            DependencyType.INHERITANCE: 1.0,      # 가장 강한 의존성
            DependencyType.COMPOSITION: 0.8,
            DependencyType.IMPORT: 0.6,
            DependencyType.FUNCTION_CALL: 0.4,
            DependencyType.TYPE_HINT: 0.2         # 가장 약한 의존성
        }
        return strength_map.get(dep_type, 0.5)

class DependencyMapper:
    """의존성 매핑 도구"""

    def __init__(self):
        self.analyzer = DependencyAnalyzer()

    def map_project_dependencies(self, project_root: str) -> DependencyGraph:
        """프로젝트 전체 의존성 매핑"""
        paca_dir = os.path.join(project_root, 'paca')
        if not os.path.exists(paca_dir):
            print(f"❌ PACA 디렉토리를 찾을 수 없습니다: {paca_dir}")
            return DependencyGraph()

        graph = DependencyGraph()

        # 모든 Python 파일 분석
        for root, dirs, files in os.walk(paca_dir):
            # __pycache__ 제외
            dirs[:] = [d for d in dirs if not d.startswith('__pycache__')]

            for file in files:
                if file.endswith('.py') and file != '__init__.py':
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, paca_dir)
                    module_name = relative_path.replace(os.sep, '.').replace('.py', '')

                    # 모듈 의존성 분석
                    module_deps = self.analyzer.analyze_file(file_path, module_name)
                    graph.modules[module_name] = module_deps

        # 인접 리스트 구축
        self._build_adjacency_lists(graph)

        # Fan-in 계산
        self._calculate_fan_in(graph)

        # 순환 의존성 탐지
        graph.cycles = self._detect_cycles(graph)

        # 강하게 연결된 컴포넌트 찾기
        graph.strongly_connected_components = self._find_strongly_connected_components(graph)

        return graph

    def _build_adjacency_lists(self, graph: DependencyGraph):
        """인접 리스트 구축"""
        for module_name, module_deps in graph.modules.items():
            graph.adjacency_list[module_name] = module_deps.internal_dependencies.copy()

            # 역방향 인접 리스트 (fan-in 계산용)
            for dep in module_deps.internal_dependencies:
                if dep not in graph.reverse_adjacency_list:
                    graph.reverse_adjacency_list[dep] = set()
                graph.reverse_adjacency_list[dep].add(module_name)

    def _calculate_fan_in(self, graph: DependencyGraph):
        """Fan-in 계산"""
        for module_name in graph.modules:
            fan_in = len(graph.reverse_adjacency_list.get(module_name, set()))
            graph.modules[module_name].fan_in = fan_in

    def _detect_cycles(self, graph: DependencyGraph) -> List[List[str]]:
        """순환 의존성 탐지 (DFS 기반)"""
        visited = set()
        rec_stack = set()
        cycles = []

        def dfs(node, path):
            if node in rec_stack:
                # 순환 발견
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return True

            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.adjacency_list.get(node, set()):
                if neighbor in graph.modules:  # 실제 존재하는 모듈만
                    if dfs(neighbor, path + [neighbor]):
                        break

            rec_stack.remove(node)
            return False

        for module in graph.modules:
            if module not in visited:
                dfs(module, [module])

        return cycles

    def _find_strongly_connected_components(self, graph: DependencyGraph) -> List[List[str]]:
        """Tarjan 알고리즘으로 강하게 연결된 컴포넌트 찾기"""
        index_map = {}
        lowlink_map = {}
        on_stack = set()
        stack = []
        index = [0]  # mutable integer
        components = []

        def strongconnect(node):
            index_map[node] = index[0]
            lowlink_map[node] = index[0]
            index[0] += 1
            stack.append(node)
            on_stack.add(node)

            for neighbor in graph.adjacency_list.get(node, set()):
                if neighbor in graph.modules:  # 실제 존재하는 모듈만
                    if neighbor not in index_map:
                        strongconnect(neighbor)
                        lowlink_map[node] = min(lowlink_map[node], lowlink_map[neighbor])
                    elif neighbor in on_stack:
                        lowlink_map[node] = min(lowlink_map[node], index_map[neighbor])

            # Root node이면 SCC 생성
            if lowlink_map[node] == index_map[node]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    component.append(w)
                    if w == node:
                        break
                if len(component) > 1:  # 크기가 1보다 큰 컴포넌트만
                    components.append(component)

        for module in graph.modules:
            if module not in index_map:
                strongconnect(module)

        return components

    def generate_dependency_report(self, graph: DependencyGraph) -> str:
        """의존성 분석 리포트 생성"""
        lines = []
        lines.append("# PACA v5 의존성 분석 리포트")
        lines.append("=" * 60)
        lines.append("")

        # 전체 통계
        lines.append("## 📊 전체 통계")
        total_modules = len(graph.modules)
        total_internal_deps = sum(len(m.internal_dependencies) for m in graph.modules.values())
        total_external_deps = sum(len(m.external_dependencies) for m in graph.modules.values())

        lines.append(f"- **총 모듈 수**: {total_modules}")
        lines.append(f"- **내부 의존성**: {total_internal_deps}")
        lines.append(f"- **외부 의존성**: {total_external_deps}")
        lines.append(f"- **순환 의존성**: {len(graph.cycles)}")
        lines.append("")

        # 모듈별 의존성
        lines.append("## 🏗️ 모듈별 의존성")
        lines.append("| 모듈 | Fan-in | Fan-out | 내부 의존성 | 외부 의존성 |")
        lines.append("|------|--------|---------|-------------|-------------|")

        for module_name, module_deps in sorted(graph.modules.items()):
            internal_count = len(module_deps.internal_dependencies)
            external_count = len(module_deps.external_dependencies)
            lines.append(f"| {module_name} | {module_deps.fan_in} | {module_deps.fan_out} | "
                        f"{internal_count} | {external_count} |")

        lines.append("")

        # 순환 의존성 분석
        if graph.cycles:
            lines.append("## 🔄 순환 의존성")
            for i, cycle in enumerate(graph.cycles, 1):
                cycle_str = " → ".join(cycle)
                lines.append(f"{i}. {cycle_str}")
            lines.append("")

        # 강하게 연결된 컴포넌트
        if graph.strongly_connected_components:
            lines.append("## 🔗 강하게 연결된 컴포넌트")
            for i, component in enumerate(graph.strongly_connected_components, 1):
                lines.append(f"{i}. {', '.join(component)}")
            lines.append("")

        # 의존성 강도 분석
        lines.append("## 💪 의존성 강도 분석")
        high_coupling_modules = []
        for module_name, module_deps in graph.modules.items():
            if module_deps.fan_in > 3 or module_deps.fan_out > 5:
                coupling_score = module_deps.fan_in + module_deps.fan_out
                high_coupling_modules.append((module_name, coupling_score))

        if high_coupling_modules:
            high_coupling_modules.sort(key=lambda x: x[1], reverse=True)
            lines.append("**높은 결합도 모듈:**")
            for module, score in high_coupling_modules[:5]:
                lines.append(f"- {module} (결합도: {score})")
        else:
            lines.append("모든 모듈이 적절한 결합도를 유지하고 있습니다.")

        lines.append("")

        # 권장사항
        lines.append("## 💡 개선 권장사항")

        if graph.cycles:
            lines.append("- **순환 의존성 해결**: 의존성 역전 원칙 적용 또는 중간 추상화 계층 도입")

        if high_coupling_modules:
            lines.append("- **결합도 감소**: 높은 결합도 모듈의 책임 분리 고려")

        isolated_modules = [name for name, deps in graph.modules.items()
                          if deps.fan_in == 0 and deps.fan_out == 0]
        if isolated_modules:
            lines.append(f"- **고립된 모듈 검토**: {', '.join(isolated_modules)}")

        lines.append("- **인터페이스 설계**: 모듈 간 명확한 인터페이스 정의")
        lines.append("- **의존성 주입**: 런타임 의존성 구성으로 유연성 향상")

        return "\n".join(lines)

    def generate_dot_graph(self, graph: DependencyGraph, output_path: str):
        """Graphviz DOT 형식으로 의존성 그래프 생성"""
        lines = []
        lines.append("digraph dependencies {")
        lines.append("    rankdir=LR;")
        lines.append("    node [shape=box, style=filled];")
        lines.append("")

        # 노드 정의 (색상은 결합도에 따라)
        for module_name, module_deps in graph.modules.items():
            coupling = module_deps.fan_in + module_deps.fan_out
            if coupling > 8:
                color = "lightcoral"  # 높은 결합도
            elif coupling > 4:
                color = "lightyellow"  # 중간 결합도
            else:
                color = "lightgreen"  # 낮은 결합도

            clean_name = module_name.replace('.', '_')
            lines.append(f'    {clean_name} [label="{module_name}", fillcolor="{color}"];')

        lines.append("")

        # 엣지 정의
        for module_name, module_deps in graph.modules.items():
            clean_source = module_name.replace('.', '_')
            for dep in module_deps.internal_dependencies:
                if dep in graph.modules:  # 실제 존재하는 모듈만
                    clean_target = dep.replace('.', '_')
                    lines.append(f"    {clean_source} -> {clean_target};")

        # 순환 의존성 강조
        if graph.cycles:
            lines.append("")
            lines.append("    /* Cycles */")
            for cycle in graph.cycles:
                for i in range(len(cycle) - 1):
                    source = cycle[i].replace('.', '_')
                    target = cycle[i + 1].replace('.', '_')
                    lines.append(f"    {source} -> {target} [color=red, penwidth=2];")

        lines.append("}")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def export_to_json(self, graph: DependencyGraph, output_path: str):
        """의존성 그래프를 JSON으로 내보내기"""
        data = {
            "modules": {},
            "adjacency_list": {},
            "cycles": graph.cycles,
            "strongly_connected_components": graph.strongly_connected_components
        }

        for module_name, module_deps in graph.modules.items():
            data["modules"][module_name] = {
                "internal_dependencies": list(module_deps.internal_dependencies),
                "external_dependencies": list(module_deps.external_dependencies),
                "fan_in": module_deps.fan_in,
                "fan_out": module_deps.fan_out,
                "dependency_details": [
                    {
                        "source": dep.source,
                        "target": dep.target,
                        "type": dep.dependency_type.value,
                        "line_number": dep.line_number,
                        "context": dep.context,
                        "strength": dep.strength
                    }
                    for dep in module_deps.dependency_details
                ]
            }

        for module, deps in graph.adjacency_list.items():
            data["adjacency_list"][module] = list(deps)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    """메인 실행 함수"""
    print("PACA v5 의존성 매핑 도구")
    print("=" * 50)

    # 프로젝트 루트 경로
    project_root = Path(__file__).parent.parent

    # 의존성 매핑
    mapper = DependencyMapper()
    dependency_graph = mapper.map_project_dependencies(str(project_root))

    # 리포트 생성
    report = mapper.generate_dependency_report(dependency_graph)

    # 리포트 저장
    report_path = project_root / "dependency_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    # JSON 내보내기
    json_path = project_root / "dependency_graph.json"
    mapper.export_to_json(dependency_graph, str(json_path))

    # DOT 그래프 생성
    dot_path = project_root / "dependency_graph.dot"
    mapper.generate_dot_graph(dependency_graph, str(dot_path))

    print(f"✅ 의존성 분석 완료!")
    print(f"📊 리포트: {report_path}")
    print(f"📄 JSON: {json_path}")
    print(f"🌐 DOT 그래프: {dot_path}")

    # 통계 요약
    total_modules = len(dependency_graph.modules)
    cycles_count = len(dependency_graph.cycles)
    scc_count = len(dependency_graph.strongly_connected_components)

    print(f"\n📈 분석 결과:")
    print(f"   총 모듈: {total_modules}")
    print(f"   순환 의존성: {cycles_count}")
    print(f"   강하게 연결된 컴포넌트: {scc_count}")

    if cycles_count > 0:
        print(f"⚠️  순환 의존성이 발견되었습니다. 리포트를 확인하세요.")

if __name__ == "__main__":
    main()