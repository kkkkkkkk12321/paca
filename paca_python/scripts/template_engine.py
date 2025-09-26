"""
PACA v5 템플릿 엔진
9개 섹션 표준 기반 문서 템플릿 처리 시스템
"""

import re
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class TemplateType(Enum):
    """템플릿 타입"""
    README = "readme"
    API_DOC = "api_doc"
    USER_GUIDE = "user_guide"
    DEVELOPER_GUIDE = "developer_guide"
    ARCHITECTURE = "architecture"

@dataclass
class TemplateVariable:
    """템플릿 변수"""
    name: str
    description: str
    default_value: Any
    required: bool = False
    validator: Optional[str] = None

@dataclass
class TemplateSection:
    """템플릿 섹션"""
    title: str
    content: str
    order: int
    required: bool = True
    variables: List[str] = field(default_factory=list)

@dataclass
class Template:
    """템플릿 정의"""
    name: str
    description: str
    template_type: TemplateType
    sections: List[TemplateSection]
    variables: Dict[str, TemplateVariable]
    metadata: Dict[str, Any] = field(default_factory=dict)

class TemplateEngine:
    """템플릿 처리 엔진"""

    def __init__(self):
        self.templates = {}
        self.variable_processors = {}
        self.section_processors = {}
        self._init_default_templates()
        self._init_processors()

    def _init_default_templates(self):
        """기본 템플릿 초기화"""
        # 9개 섹션 표준 README 템플릿
        readme_template = self._create_readme_template()
        self.templates[TemplateType.README] = readme_template

        # API 문서 템플릿
        api_template = self._create_api_template()
        self.templates[TemplateType.API_DOC] = api_template

    def _create_readme_template(self) -> Template:
        """README.md 템플릿 생성"""
        sections = [
            TemplateSection(
                title="🎯 프로젝트 개요",
                content="{module_description}",
                order=1,
                variables=["module_description"]
            ),
            TemplateSection(
                title="📁 폴더/파일 구조",
                content="```\n{folder_structure}\n```\n\n{file_descriptions}",
                order=2,
                variables=["folder_structure", "file_descriptions"]
            ),
            TemplateSection(
                title="⚙️ 기능 요구사항",
                content="""**입력:**
{input_requirements}

**출력:**
{output_requirements}

**핵심 로직 흐름:**
{core_logic_flow}""",
                order=3,
                variables=["input_requirements", "output_requirements", "core_logic_flow"]
            ),
            TemplateSection(
                title="🛠️ 기술적 요구사항",
                content="""**언어 및 프레임워크:**
{tech_framework}

**주요 의존성:**
{dependencies}

**실행 환경:**
{runtime_environment}""",
                order=4,
                variables=["tech_framework", "dependencies", "runtime_environment"]
            ),
            TemplateSection(
                title="🚀 라우팅 및 진입점",
                content="""**주요 진입점:**
```python
{entry_points}
```

**API 경로:**
{api_routes}""",
                order=5,
                variables=["entry_points", "api_routes"]
            ),
            TemplateSection(
                title="📋 코드 품질 가이드",
                content="""**주석 규칙:**
{comment_rules}

**네이밍 규칙:**
{naming_rules}

**예외 처리:**
{exception_handling}""",
                order=6,
                variables=["comment_rules", "naming_rules", "exception_handling"]
            ),
            TemplateSection(
                title="🏃‍♂️ 실행 방법",
                content="""**설치:**
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
```""",
                order=7,
                variables=["installation_commands", "usage_example", "test_commands"]
            ),
            TemplateSection(
                title="🧪 테스트 방법",
                content="""**단위 테스트:**
{unit_tests}

**통합 테스트:**
{integration_tests}

**성능 테스트:**
{performance_tests}

**테스트 시나리오:**
```python
{test_scenarios}
```""",
                order=8,
                variables=["unit_tests", "integration_tests", "performance_tests", "test_scenarios"]
            ),
            TemplateSection(
                title="💡 추가 고려사항",
                content="""**보안:**
{security_considerations}

**성능:**
{performance_considerations}

**향후 개선:**
{future_improvements}

**모니터링:**
{monitoring_info}""",
                order=9,
                variables=["security_considerations", "performance_considerations",
                          "future_improvements", "monitoring_info"]
            )
        ]

        variables = {
            "module_description": TemplateVariable(
                "module_description", "모듈 설명", "PACA v5 모듈", True
            ),
            "folder_structure": TemplateVariable(
                "folder_structure", "폴더 구조", "", True
            ),
            "file_descriptions": TemplateVariable(
                "file_descriptions", "파일 설명", "", True
            ),
            "input_requirements": TemplateVariable(
                "input_requirements", "입력 요구사항", "- 입력 데이터", True
            ),
            "output_requirements": TemplateVariable(
                "output_requirements", "출력 요구사항", "- 출력 결과", True
            ),
            "core_logic_flow": TemplateVariable(
                "core_logic_flow", "핵심 로직 흐름", "1. 입력 처리\n2. 로직 실행\n3. 결과 반환", True
            ),
            "tech_framework": TemplateVariable(
                "tech_framework", "기술 프레임워크", "- Python 3.8+", True
            ),
            "dependencies": TemplateVariable(
                "dependencies", "의존성", "- 내부 의존성 없음", True
            ),
            "runtime_environment": TemplateVariable(
                "runtime_environment", "실행 환경", "- 메모리: 최소 128MB", True
            ),
            "entry_points": TemplateVariable(
                "entry_points", "진입점", "from paca.module import Module", True
            ),
            "api_routes": TemplateVariable(
                "api_routes", "API 경로", "- module.main(): 메인 인터페이스", True
            ),
            "comment_rules": TemplateVariable(
                "comment_rules", "주석 규칙", "- 모든 함수에 docstring 필수", True
            ),
            "naming_rules": TemplateVariable(
                "naming_rules", "네이밍 규칙", "- 클래스: PascalCase\n- 함수: snake_case", True
            ),
            "exception_handling": TemplateVariable(
                "exception_handling", "예외 처리", "- ModuleError: 모듈 관련 오류", True
            ),
            "installation_commands": TemplateVariable(
                "installation_commands", "설치 명령어", "pip install -e .", True
            ),
            "usage_example": TemplateVariable(
                "usage_example", "사용 예시", "# 사용 예시\nprint('Hello, PACA!')", True
            ),
            "test_commands": TemplateVariable(
                "test_commands", "테스트 명령어", "python -m pytest tests/ -v", True
            ),
            "unit_tests": TemplateVariable(
                "unit_tests", "단위 테스트", "- 개별 기능 테스트", True
            ),
            "integration_tests": TemplateVariable(
                "integration_tests", "통합 테스트", "- 전체 워크플로우 테스트", True
            ),
            "performance_tests": TemplateVariable(
                "performance_tests", "성능 테스트", "- 응답 시간 측정", True
            ),
            "test_scenarios": TemplateVariable(
                "test_scenarios", "테스트 시나리오", "def test_basic():\n    assert True", True
            ),
            "security_considerations": TemplateVariable(
                "security_considerations", "보안 고려사항", "- 입력 검증 필수", True
            ),
            "performance_considerations": TemplateVariable(
                "performance_considerations", "성능 고려사항", "- 캐싱 활용", True
            ),
            "future_improvements": TemplateVariable(
                "future_improvements", "향후 개선", "- 기능 확장 계획", True
            ),
            "monitoring_info": TemplateVariable(
                "monitoring_info", "모니터링", "- 성능 메트릭 수집", True
            )
        }

        return Template(
            name="PACA README Template",
            description="PACA v5 9개 섹션 표준 README.md 템플릿",
            template_type=TemplateType.README,
            sections=sections,
            variables=variables,
            metadata={
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "standard": "9-section-paca-v5"
            }
        )

    def _create_api_template(self) -> Template:
        """API 문서 템플릿 생성"""
        sections = [
            TemplateSection(
                title="📚 API Reference",
                content="# {module_name} API Reference\n\n{api_overview}",
                order=1,
                variables=["module_name", "api_overview"]
            ),
            TemplateSection(
                title="🔧 Classes",
                content="{class_documentation}",
                order=2,
                variables=["class_documentation"]
            ),
            TemplateSection(
                title="⚡ Functions",
                content="{function_documentation}",
                order=3,
                variables=["function_documentation"]
            ),
            TemplateSection(
                title="📋 Examples",
                content="{usage_examples}",
                order=4,
                variables=["usage_examples"]
            )
        ]

        variables = {
            "module_name": TemplateVariable(
                "module_name", "모듈명", "Module", True
            ),
            "api_overview": TemplateVariable(
                "api_overview", "API 개요", "모듈 API 설명", True
            ),
            "class_documentation": TemplateVariable(
                "class_documentation", "클래스 문서", "클래스 설명", True
            ),
            "function_documentation": TemplateVariable(
                "function_documentation", "함수 문서", "함수 설명", True
            ),
            "usage_examples": TemplateVariable(
                "usage_examples", "사용 예시", "예시 코드", True
            )
        }

        return Template(
            name="PACA API Template",
            description="PACA v5 API 참조 문서 템플릿",
            template_type=TemplateType.API_DOC,
            sections=sections,
            variables=variables
        )

    def _init_processors(self):
        """변수 및 섹션 프로세서 초기화"""
        # 변수 프로세서
        self.variable_processors = {
            "folder_structure": self._process_folder_structure,
            "file_descriptions": self._process_file_descriptions,
            "class_documentation": self._process_class_documentation,
            "function_documentation": self._process_function_documentation,
            "usage_examples": self._process_usage_examples
        }

        # 섹션 프로세서
        self.section_processors = {
            "format_code_blocks": self._format_code_blocks,
            "add_emojis": self._add_section_emojis,
            "validate_links": self._validate_markdown_links
        }

    def render_template(
        self,
        template_type: TemplateType,
        variables: Dict[str, Any],
        custom_processors: Optional[Dict[str, Any]] = None
    ) -> str:
        """템플릿 렌더링"""

        if template_type not in self.templates:
            raise ValueError(f"Unknown template type: {template_type}")

        template = self.templates[template_type]

        # 변수 검증 및 기본값 적용
        processed_variables = self._process_variables(template, variables)

        # 커스텀 프로세서 적용
        if custom_processors:
            for var_name, processor in custom_processors.items():
                if var_name in processed_variables:
                    processed_variables[var_name] = processor(processed_variables[var_name])

        # 섹션별 렌더링
        rendered_sections = []
        for section in sorted(template.sections, key=lambda s: s.order):
            rendered_content = self._render_section(section, processed_variables)
            rendered_sections.append(f"# {section.title}\n\n{rendered_content}")

        # 최종 문서 생성
        document = "\n\n".join(rendered_sections)

        # 후처리
        document = self._post_process_document(document)

        return document

    def _process_variables(self, template: Template, variables: Dict[str, Any]) -> Dict[str, Any]:
        """변수 처리 및 검증"""
        processed = {}

        for var_name, var_def in template.variables.items():
            if var_name in variables:
                value = variables[var_name]

                # 커스텀 프로세서 적용
                if var_name in self.variable_processors:
                    value = self.variable_processors[var_name](value)

                processed[var_name] = value

            elif var_def.required:
                # 필수 변수가 없으면 기본값 사용
                processed[var_name] = var_def.default_value
                print(f"⚠️ 필수 변수 '{var_name}'이 제공되지 않아 기본값을 사용합니다.")

            else:
                processed[var_name] = var_def.default_value

        return processed

    def _render_section(self, section: TemplateSection, variables: Dict[str, Any]) -> str:
        """섹션 렌더링"""
        content = section.content

        # 변수 치환
        for var_name in section.variables:
            if var_name in variables:
                placeholder = f"{{{var_name}}}"
                content = content.replace(placeholder, str(variables[var_name]))

        return content.strip()

    def _process_folder_structure(self, value: Any) -> str:
        """폴더 구조 처리"""
        if isinstance(value, list):
            return "\n".join(value)
        elif isinstance(value, dict):
            lines = []
            for folder, files in value.items():
                lines.append(f"📁 {folder}/")
                if isinstance(files, list):
                    for file in files:
                        lines.append(f"├── 📄 {file}")
                elif isinstance(files, dict):
                    for file, desc in files.items():
                        lines.append(f"├── 📄 {file:<20} # {desc}")
            return "\n".join(lines)
        else:
            return str(value)

    def _process_file_descriptions(self, value: Any) -> str:
        """파일 설명 처리"""
        if isinstance(value, dict):
            lines = ["**파일별 설명:**"]
            for file, desc in value.items():
                lines.append(f"- `{file}`: {desc}")
            return "\n".join(lines)
        else:
            return str(value)

    def _process_class_documentation(self, classes: List[Dict]) -> str:
        """클래스 문서화 처리"""
        if not classes:
            return "클래스가 정의되지 않았습니다."

        lines = []
        for cls in classes:
            lines.append(f"## {cls.get('name', 'Unknown')}")
            lines.append("")
            lines.append(cls.get('docstring', '설명이 없습니다.'))
            lines.append("")

            if cls.get('methods'):
                lines.append("### Methods")
                for method in cls['methods']:
                    lines.append(f"- `{method}`")
                lines.append("")

        return "\n".join(lines)

    def _process_function_documentation(self, functions: List[Dict]) -> str:
        """함수 문서화 처리"""
        if not functions:
            return "함수가 정의되지 않았습니다."

        lines = []
        for func in functions:
            name = func.get('name', 'unknown')
            args = func.get('args', [])
            lines.append(f"### {name}({', '.join(args)})")
            lines.append("")
            lines.append(func.get('docstring', '설명이 없습니다.'))
            lines.append("")

        return "\n".join(lines)

    def _process_usage_examples(self, examples: Union[str, List[str]]) -> str:
        """사용 예시 처리"""
        if isinstance(examples, list):
            processed_examples = []
            for i, example in enumerate(examples, 1):
                processed_examples.append(f"### Example {i}")
                processed_examples.append("")
                processed_examples.append("```python")
                processed_examples.append(example)
                processed_examples.append("```")
                processed_examples.append("")
            return "\n".join(processed_examples)
        else:
            return f"```python\n{examples}\n```"

    def _format_code_blocks(self, content: str) -> str:
        """코드 블록 포맷팅"""
        # Python 코드 블록에 syntax highlighting 추가
        content = re.sub(
            r'```\n(.*?)\n```',
            r'```python\n\1\n```',
            content,
            flags=re.DOTALL
        )
        return content

    def _add_section_emojis(self, content: str) -> str:
        """섹션에 이모지 추가"""
        emoji_map = {
            "설치": "📦",
            "사용법": "🚀",
            "예시": "💡",
            "테스트": "🧪",
            "API": "⚡"
        }

        for keyword, emoji in emoji_map.items():
            content = content.replace(f"## {keyword}", f"## {emoji} {keyword}")

        return content

    def _validate_markdown_links(self, content: str) -> str:
        """마크다운 링크 검증"""
        # 간단한 링크 형식 검증
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'

        def validate_link(match):
            text, url = match.groups()
            if url.startswith(('http://', 'https://', '#', '/')):
                return match.group(0)  # 유효한 링크
            else:
                return f"`{text}`"  # 코드로 변환

        return re.sub(link_pattern, validate_link, content)

    def _post_process_document(self, document: str) -> str:
        """문서 후처리"""
        # 빈 줄 정리
        document = re.sub(r'\n{3,}', '\n\n', document)

        # 섹션 프로세서 적용
        for processor_name, processor in self.section_processors.items():
            document = processor(document)

        # 마지막 줄바꿈 추가
        if not document.endswith('\n'):
            document += '\n'

        return document

    def create_custom_template(
        self,
        name: str,
        template_type: TemplateType,
        sections: List[TemplateSection],
        variables: Dict[str, TemplateVariable]
    ) -> Template:
        """커스텀 템플릿 생성"""
        template = Template(
            name=name,
            description=f"Custom {template_type.value} template",
            template_type=template_type,
            sections=sections,
            variables=variables,
            metadata={
                "custom": True,
                "created_at": datetime.now().isoformat()
            }
        )

        self.templates[f"custom_{len(self.templates)}"] = template
        return template

    def save_template(self, template: Template, file_path: str):
        """템플릿을 파일에 저장"""
        template_data = {
            "name": template.name,
            "description": template.description,
            "template_type": template.template_type.value,
            "sections": [
                {
                    "title": section.title,
                    "content": section.content,
                    "order": section.order,
                    "required": section.required,
                    "variables": section.variables
                }
                for section in template.sections
            ],
            "variables": {
                name: {
                    "description": var.description,
                    "default_value": var.default_value,
                    "required": var.required,
                    "validator": var.validator
                }
                for name, var in template.variables.items()
            },
            "metadata": template.metadata
        }

        if file_path.endswith('.yaml') or file_path.endswith('.yml'):
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(template_data, f, allow_unicode=True, default_flow_style=False)
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, ensure_ascii=False, indent=2)

    def load_template(self, file_path: str) -> Template:
        """파일에서 템플릿 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                template_data = yaml.safe_load(f)
            else:
                template_data = json.load(f)

        sections = [
            TemplateSection(
                title=section["title"],
                content=section["content"],
                order=section["order"],
                required=section.get("required", True),
                variables=section.get("variables", [])
            )
            for section in template_data["sections"]
        ]

        variables = {
            name: TemplateVariable(
                name=name,
                description=var["description"],
                default_value=var["default_value"],
                required=var.get("required", False),
                validator=var.get("validator")
            )
            for name, var in template_data["variables"].items()
        }

        return Template(
            name=template_data["name"],
            description=template_data["description"],
            template_type=TemplateType(template_data["template_type"]),
            sections=sections,
            variables=variables,
            metadata=template_data.get("metadata", {})
        )

def main():
    """템플릿 엔진 테스트"""
    print("PACA v5 템플릿 엔진 테스트")
    print("=" * 50)

    engine = TemplateEngine()

    # 테스트 변수
    test_variables = {
        "module_description": "PACA v5 테스트 모듈 - 템플릿 엔진 테스트용 모듈입니다.",
        "folder_structure": {
            "test_module": {
                "__init__.py": "모듈 초기화",
                "main.py": "메인 로직",
                "utils.py": "유틸리티 함수"
            }
        },
        "file_descriptions": {
            "__init__.py": "모듈 진입점 및 공개 API",
            "main.py": "핵심 비즈니스 로직",
            "utils.py": "보조 유틸리티 함수들"
        }
    }

    # README 템플릿 렌더링
    readme_content = engine.render_template(
        TemplateType.README,
        test_variables
    )

    # 결과 저장
    output_path = Path(__file__).parent / "test_readme.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"✅ 테스트 README 생성 완료: {output_path}")
    print(f"📝 생성된 문서 크기: {len(readme_content)} 문자")

if __name__ == "__main__":
    main()