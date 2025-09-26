"""
Automatic README.md Generation System for PACA Python Conversion
자동 README.md 생성 시스템 - 9개 섹션 표준 준수

This system automatically generates comprehensive README.md files with 9 required sections
for every module during the TypeScript to Python conversion process.
"""

import os
import ast
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CodeAnalysis:
    """코드 분석 결과를 담는 데이터 클래스"""
    classes: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    dependencies: List[str]
    apis: List[Dict[str, Any]]
    tests: List[str]
    performance_notes: List[str]
    file_count: int
    line_count: int
    complexity_score: float


class CodeAnalyzer:
    """Python 코드 정적 분석으로 문서화 정보 추출"""

    def __init__(self):
        self.performance_patterns = [
            "async def", "await", "asyncio", "concurrent.futures",
            "multiprocessing", "threading", "cache", "lru_cache"
        ]

    def analyze_module(self, module_path: str) -> CodeAnalysis:
        """모듈 전체 분석"""
        path = Path(module_path)
        if not path.exists():
            return self._empty_analysis()

        analysis = CodeAnalysis(
            classes=[],
            functions=[],
            dependencies=[],
            apis=[],
            tests=[],
            performance_notes=[],
            file_count=0,
            line_count=0,
            complexity_score=0.0
        )

        # Python 파일들 찾기
        python_files = list(path.rglob("*.py"))
        analysis.file_count = len(python_files)

        for py_file in python_files:
            self._analyze_file(py_file, analysis)

        # 복잡도 계산
        analysis.complexity_score = self._calculate_complexity(analysis)

        return analysis

    def _analyze_file(self, file_path: Path, analysis: CodeAnalysis):
        """개별 파일 분석"""
        try:
            content = file_path.read_text(encoding='utf-8')
            analysis.line_count += len(content.splitlines())

            # AST 파싱
            tree = ast.parse(content)

            # 클래스 추출
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    analysis.classes.append({
                        'name': node.name,
                        'docstring': ast.get_docstring(node),
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                        'file': str(file_path)
                    })
                elif isinstance(node, ast.FunctionDef):
                    analysis.functions.append({
                        'name': node.name,
                        'docstring': ast.get_docstring(node),
                        'args': [arg.arg for arg in node.args.args],
                        'file': str(file_path),
                        'is_async': isinstance(node, ast.AsyncFunctionDef)
                    })
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    self._extract_imports(node, analysis)

            # 성능 패턴 찾기
            for pattern in self.performance_patterns:
                if pattern in content:
                    analysis.performance_notes.append(f"{pattern} found in {file_path.name}")

            # API 패턴 찾기 (FastAPI, Flask 등)
            if any(framework in content for framework in ['@app.', '@router.', 'fastapi', 'flask']):
                api_routes = re.findall(r'@\w+\.(get|post|put|delete|patch)\((["\'])([^"\']+)\2\)', content)
                for method, _, route in api_routes:
                    analysis.apis.append({
                        'method': method.upper(),
                        'route': route,
                        'file': str(file_path)
                    })

            # 테스트 파일 확인
            if 'test_' in file_path.name or '_test.py' in file_path.name:
                analysis.tests.append(str(file_path))

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")

    def _extract_imports(self, node: ast.AST, analysis: CodeAnalysis):
        """import 문에서 의존성 추출"""
        if isinstance(node, ast.Import):
            for alias in node.names:
                analysis.dependencies.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                analysis.dependencies.append(node.module)

    def _calculate_complexity(self, analysis: CodeAnalysis) -> float:
        """복잡도 점수 계산 (0.0 - 1.0)"""
        base_score = 0.0

        # 파일 수에 따른 복잡도
        if analysis.file_count > 10:
            base_score += 0.3
        elif analysis.file_count > 5:
            base_score += 0.2
        else:
            base_score += 0.1

        # 클래스/함수 수에 따른 복잡도
        total_components = len(analysis.classes) + len(analysis.functions)
        if total_components > 50:
            base_score += 0.3
        elif total_components > 20:
            base_score += 0.2
        else:
            base_score += 0.1

        # 의존성에 따른 복잡도
        unique_deps = len(set(analysis.dependencies))
        if unique_deps > 15:
            base_score += 0.3
        elif unique_deps > 8:
            base_score += 0.2
        else:
            base_score += 0.1

        # API가 있으면 복잡도 증가
        if analysis.apis:
            base_score += 0.1

        return min(base_score, 1.0)

    def _empty_analysis(self) -> CodeAnalysis:
        """빈 분석 결과 반환"""
        return CodeAnalysis([], [], [], [], [], [], 0, 0, 0.0)


class AutoDocumentationGenerator:
    """
    모든 폴더에 9개 섹션 README.md 자동 생성
    코드 분석 기반으로 정확한 정보 추출
    """

    def __init__(self):
        self.section_templates = self._load_templates()
        self.code_analyzer = CodeAnalyzer()

    def generate_module_readme(self, module_path: str, module_name: str = None) -> str:
        """모듈 분석 후 9개 섹션 완전한 README.md 생성"""
        path = Path(module_path)
        analysis = self.code_analyzer.analyze_module(module_path)

        if not module_name:
            module_name = path.name.replace('_', ' ').title()

        readme_content = {
            "project_overview": self._generate_overview(module_name, analysis),
            "folder_structure": self._generate_structure(path, analysis),
            "functional_requirements": self._generate_requirements(analysis),
            "technical_requirements": self._generate_tech_specs(analysis),
            "routing_entrypoints": self._generate_entrypoints(analysis),
            "code_quality_guide": self._generate_quality_guide(analysis),
            "execution_methods": self._generate_execution_guide(module_name, analysis),
            "testing_methods": self._generate_test_guide(analysis),
            "additional_considerations": self._generate_considerations(analysis)
        }

        formatted_readme = self._format_readme(readme_content, module_name)

        # README.md 파일 생성
        readme_path = path / "README.md"
        readme_path.write_text(formatted_readme, encoding='utf-8')

        return formatted_readme

    def _generate_overview(self, module_name: str, analysis: CodeAnalysis) -> str:
        """프로젝트 개요 섹션 생성"""
        purpose_map = {
            'core': 'PACA 시스템의 핵심 기반 모듈로, 타입 정의, 이벤트 시스템, 에러 처리를 담당',
            'cognitive': '인지 모델(ACT-R, SOAR)과 메모리 시스템을 구현하여 인간과 유사한 사고 과정을 모델링',
            'learning': '자율학습 엔진과 한국어 최적화된 학습 패턴을 통해 지속적인 성능 향상을 제공',
            'reasoning': '논리적 추론과 메타인지 시스템을 통해 복잡한 문제 해결 능력을 구현',
            'mathematics': '수학적 연산과 증명 시스템을 제공하여 정확한 계산과 논리적 추론을 지원',
            'services': 'PACA의 핵심 비즈니스 로직과 서비스 레이어를 구현하여 모듈 간 통신을 관리',
            'integrations': '외부 API(Google AI, KoNLPy)와의 통합을 담당하여 확장 가능한 AI 기능을 제공',
            'data': 'SQLite 기반 데이터 저장소와 스키마 관리를 통해 안정적인 데이터 영속성을 보장',
            'config': '환경 설정과 구성 관리를 담당하여 개발/운영 환경별 유연한 설정을 제공'
        }

        module_key = module_name.lower().replace(' ', '_')
        purpose = purpose_map.get(module_key, f'{module_name} 모듈의 핵심 기능을 담당')

        return f"{purpose}. ({analysis.file_count}개 파일, {analysis.line_count}줄)"

    def _generate_structure(self, path: Path, analysis: CodeAnalysis) -> str:
        """폴더/파일 구조 섹션 생성"""
        structure = f"{path.name}/\n"

        # 실제 파일 구조 생성
        try:
            for item in sorted(path.iterdir()):
                if item.name.startswith('.'):
                    continue

                if item.is_file():
                    if item.suffix == '.py':
                        # Python 파일에 대한 설명 추가
                        description = self._get_file_description(item, analysis)
                        structure += f"├── {item.name:<20} # {description}\n"
                    else:
                        structure += f"├── {item.name}\n"
                elif item.is_dir():
                    structure += f"├── {item.name}/\n"
                    # 하위 디렉토리의 주요 파일들도 표시 (최대 3개)
                    sub_files = [f for f in item.iterdir() if f.is_file() and f.suffix == '.py'][:3]
                    for sub_file in sub_files:
                        structure += f"│   ├── {sub_file.name}\n"
                    if len(list(item.glob('*.py'))) > 3:
                        structure += f"│   └── ... (더 많은 파일들)\n"
        except PermissionError:
            structure += "├── (접근 권한 필요)\n"

        structure += "└── README.md           # 이 문서\n"
        return structure

    def _get_file_description(self, file_path: Path, analysis: CodeAnalysis) -> str:
        """파일별 설명 생성"""
        descriptions = {
            '__init__.py': '모듈 진입점 및 공개 API 정의',
            'types.py': '타입 정의 및 데이터 클래스',
            'events.py': '이벤트 시스템 및 EventBus',
            'errors.py': '커스텀 예외 클래스들',
            'utils.py': '공통 유틸리티 함수들',
            'constants.py': '시스템 상수 및 설정값',
            'base.py': '기본 클래스 및 추상 인터페이스',
            'engine.py': '핵심 엔진 및 처리 로직',
            'models.py': '데이터 모델 및 스키마',
            'config.py': '설정 관리 및 환경 변수'
        }

        filename = file_path.name
        if filename in descriptions:
            return descriptions[filename]

        # 파일 내용 기반 설명 생성
        for cls in analysis.classes:
            if file_path.name in cls['file']:
                return f"{cls['name']} 클래스 구현"

        for func in analysis.functions:
            if file_path.name in func['file'] and not func['name'].startswith('_'):
                return f"{func['name']} 관련 기능"

        return "모듈 구성 요소"

    def _generate_requirements(self, analysis: CodeAnalysis) -> str:
        """기능 요구사항 섹션 생성"""
        reqs = "**입력**: "
        if analysis.classes:
            main_classes = [cls['name'] for cls in analysis.classes[:3]]
            reqs += f"{', '.join(main_classes)} 등의 객체\n"
        else:
            reqs += "모듈 초기화 및 설정 데이터\n"

        reqs += "**출력**: "
        if analysis.apis:
            reqs += f"API 응답 ({len(analysis.apis)}개 엔드포인트)\n"
        elif analysis.functions:
            reqs += f"처리된 결과 객체 및 상태 정보\n"
        else:
            reqs += "모듈 처리 결과 및 상태\n"

        reqs += "**핵심 로직**: "
        if analysis.performance_notes:
            reqs += "비동기 처리 → "
        reqs += "입력 검증 → 데이터 처리 → 결과 반환 → 에러 처리"

        return reqs

    def _generate_tech_specs(self, analysis: CodeAnalysis) -> str:
        """기술적 요구사항 섹션 생성"""
        specs = "- Python 3.9+\n"

        # 주요 의존성 찾기
        important_deps = []
        external_deps = ['asyncio', 'pydantic', 'numpy', 'pandas', 'aiofiles',
                        'customtkinter', 'konlpy', 'transformers', 'google']

        for dep in analysis.dependencies:
            for ext_dep in external_deps:
                if ext_dep in dep.lower():
                    important_deps.append(ext_dep)
                    break

        if important_deps:
            specs += f"- 외부 라이브러리: {', '.join(set(important_deps))}\n"

        # 메모리 요구사항 추정
        if analysis.file_count > 20:
            specs += "- 메모리 요구사항: < 500MB\n"
        elif analysis.file_count > 10:
            specs += "- 메모리 요구사항: < 200MB\n"
        else:
            specs += "- 메모리 요구사항: < 100MB\n"

        # 성능 요구사항
        if analysis.performance_notes:
            specs += "- 비동기 처리 지원 (asyncio)\n"

        if analysis.apis:
            specs += "- API 응답시간: < 200ms\n"

        return specs

    def _generate_entrypoints(self, analysis: CodeAnalysis) -> str:
        """라우팅 및 진입점 섹션 생성"""
        entry = ""

        # API 라우트가 있는 경우
        if analysis.apis:
            entry += "**API 엔드포인트**:\n"
            for api in analysis.apis[:5]:  # 최대 5개만 표시
                entry += f"- {api['method']} {api['route']}\n"
            if len(analysis.apis) > 5:
                entry += f"- ... ({len(analysis.apis) - 5}개 더)\n"
            entry += "\n"

        # 주요 클래스 진입점
        if analysis.classes:
            entry += "**주요 클래스**:\n```python\n"
            for cls in analysis.classes[:3]:
                module_path = Path(cls['file']).stem
                entry += f"from paca.{module_path} import {cls['name']}\n"
            entry += "```\n\n"

        # 주요 함수 진입점
        public_functions = [f for f in analysis.functions if not f['name'].startswith('_')][:3]
        if public_functions:
            entry += "**주요 함수**:\n```python\n"
            for func in public_functions:
                args_str = ', '.join(func['args'])
                entry += f"{func['name']}({args_str})\n"
            entry += "```"

        return entry or "**모듈 import**:\n```python\nimport paca.module_name\n```"

    def _generate_quality_guide(self, analysis: CodeAnalysis) -> str:
        """코드 품질 가이드 섹션 생성"""
        guide = "**코딩 규칙**:\n"
        guide += "- 함수명: snake_case (예: process_data, create_result)\n"
        guide += "- 클래스명: PascalCase (예: DataProcessor, ResultHandler)\n"
        guide += "- 상수명: UPPER_SNAKE_CASE (예: MAX_RETRY_COUNT)\n"
        guide += "- 비공개 멤버: _underscore_prefix\n\n"

        guide += "**필수 규칙**:\n"
        guide += "- 모든 public 메서드에 타입 힌트 필수\n"
        guide += "- 예외 처리: try-except 블록으로 안전성 보장\n"
        guide += "- 문서화: docstring으로 목적과 매개변수 설명\n"

        if analysis.performance_notes:
            guide += "- 비동기 처리: async/await 패턴 준수\n"

        guide += "- 테스트: 모든 핵심 기능에 단위 테스트 작성"

        return guide

    def _generate_execution_guide(self, module_name: str, analysis: CodeAnalysis) -> str:
        """실행 방법 섹션 생성"""
        guide = "**설치**:\n```bash\n"
        guide += "# 개발 환경 설치\n"
        guide += "pip install -e .\n"
        guide += "# 또는 의존성만 설치\n"
        guide += "pip install -r requirements.txt\n```\n\n"

        guide += "**실행**:\n```bash\n"

        if analysis.apis:
            guide += "# API 서버 실행\n"
            guide += f"python -m paca.{module_name.lower().replace(' ', '_')}\n\n"

        # 주요 클래스 사용 예시
        if analysis.classes:
            main_class = analysis.classes[0]['name']
            guide += f"# {main_class} 사용 예시\n"
            guide += "python -c \"\n"
            guide += f"from paca.{module_name.lower().replace(' ', '_')} import {main_class}\n"
            guide += f"instance = {main_class}()\n"
            guide += "print(instance)\"\n"
        else:
            guide += f"# 모듈 테스트\n"
            guide += f"python -c \"import paca.{module_name.lower().replace(' ', '_')}; print('모듈 로드 성공')\"\n"

        guide += "```"

        return guide

    def _generate_test_guide(self, analysis: CodeAnalysis) -> str:
        """테스트 방법 섹션 생성"""
        guide = "**단위 테스트**:\n```bash\n"

        if analysis.tests:
            guide += f"# 전체 테스트 실행\n"
            guide += f"pytest tests/ -v\n\n"
            guide += f"# 특정 모듈 테스트\n"
            test_file = Path(analysis.tests[0]).name
            guide += f"pytest tests/{test_file} -v\n"
        else:
            guide += "pytest tests/test_*.py -v\n"

        guide += "```\n\n"

        guide += "**커버리지 테스트**:\n```bash\n"
        guide += "pytest --cov=paca --cov-report=html\n"
        guide += "# 결과는 htmlcov/index.html에서 확인\n```\n\n"

        guide += "**성능 테스트**:\n```bash\n"
        if analysis.performance_notes:
            guide += "# 비동기 성능 테스트\n"
            guide += "python -m pytest tests/test_performance.py -v\n"
        else:
            guide += "# 응답 시간 테스트\n"
            guide += "python -m timeit \"import paca.module; paca.module.main_function()\"\n"
        guide += "```"

        return guide

    def _generate_considerations(self, analysis: CodeAnalysis) -> str:
        """추가 고려사항 섹션 생성"""
        considerations = "**보안**:\n"

        if analysis.apis:
            considerations += "- API 인증 및 권한 검증 필수\n"
            considerations += "- 입력 데이터 검증 및 SQL 인젝션 방지\n"
        else:
            considerations += "- 입력 데이터 검증 및 타입 안전성 보장\n"

        considerations += "\n**성능**:\n"

        if analysis.performance_notes:
            considerations += "- 비동기 처리로 동시성 향상\n"
            considerations += "- 메모리 효율적인 스트리밍 처리\n"

        if analysis.complexity_score > 0.7:
            considerations += "- 복잡한 모듈이므로 캐싱 전략 고려\n"
            considerations += "- 모듈 분할 및 지연 로딩 검토\n"
        else:
            considerations += "- 응답 시간 최적화 (<100ms 목표)\n"

        considerations += "\n**향후 개선**:\n"
        considerations += "- 타입 체크 강화 (mypy strict 모드)\n"

        if not analysis.tests:
            considerations += "- 테스트 커버리지 확대 (목표: 80%+)\n"

        if len(analysis.dependencies) > 10:
            considerations += "- 의존성 최적화 및 번들 크기 감소\n"

        considerations += "- 모니터링 및 로깅 시스템 통합"

        return considerations

    def _format_readme(self, content: Dict[str, str], module_name: str) -> str:
        """README.md 최종 포맷팅"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        readme = f"""# {module_name} Module - PACA Python v5

> 자동 생성된 문서 (생성시간: {timestamp})

## 🎯 프로젝트 개요

{content['project_overview']}

## 📁 폴더/파일 구조

```
{content['folder_structure']}
```

## ⚙️ 기능 요구사항

{content['functional_requirements']}

## 🛠️ 기술적 요구사항

{content['technical_requirements']}

## 🚀 라우팅 및 진입점

{content['routing_entrypoints']}

## 📋 코드 품질 가이드

{content['code_quality_guide']}

## 🏃‍♂️ 실행 방법

{content['execution_methods']}

## 🧪 테스트 방법

{content['testing_methods']}

## 💡 추가 고려사항

{content['additional_considerations']}

---

> 이 문서는 PACA v5 Python 변환 프로젝트의 자동 문서화 시스템에 의해 생성되었습니다.
> 수정이 필요한 경우 `scripts/auto_documentation_system.py`를 통해 재생성하세요.
"""
        return readme

    def _load_templates(self) -> Dict[str, str]:
        """템플릿 로드 (향후 확장용)"""
        return {}


def generate_all_module_readmes(base_path: str = "C:/Users/kk/claude/paca/paca_python"):
    """모든 모듈에 대해 README.md 생성"""
    generator = AutoDocumentationGenerator()
    base = Path(base_path)

    # paca 하위 모듈들에 대해 README 생성
    paca_modules = base / "paca"
    if paca_modules.exists():
        for module_dir in paca_modules.iterdir():
            if module_dir.is_dir() and not module_dir.name.startswith('__'):
                try:
                    print(f"Generating README for {module_dir.name}...")
                    generator.generate_module_readme(str(module_dir))
                    print(f"[OK] {module_dir.name}/README.md created")
                except Exception as e:
                    print(f"[ERROR] Error generating README for {module_dir.name}: {e}")

    # 루트 README 생성
    try:
        print("Generating main README...")
        generator.generate_module_readme(str(base), "PACA Python v5")
        print("[OK] Main README.md created")
    except Exception as e:
        print(f"[ERROR] Error generating main README: {e}")


if __name__ == "__main__":
    generate_all_module_readmes()