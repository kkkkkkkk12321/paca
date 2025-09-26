"""
Config Base Module
설정 관리 기본 클래스들
"""

import json
import os
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..core.types.base import ID, KeyValuePair, Result
from ..core.errors.base import ConfigurationError, ValidationError


class ConfigFormat(Enum):
    """설정 파일 형식"""
    JSON = 'json'
    YAML = 'yaml'
    ENV = 'env'
    TOML = 'toml'


class ConfigScope(Enum):
    """설정 범위"""
    GLOBAL = 'global'
    APPLICATION = 'application'
    MODULE = 'module'
    USER = 'user'
    ENVIRONMENT = 'environment'


@dataclass
class ConfigSchema:
    """설정 스키마"""
    name: str
    type: str  # 'string', 'integer', 'float', 'boolean', 'array', 'object'
    required: bool = False
    default: Any = None
    description: Optional[str] = None
    constraints: KeyValuePair = field(default_factory=dict)


@dataclass
class ConfigItem:
    """설정 항목"""
    key: str
    value: Any
    schema: Optional[ConfigSchema] = None
    source: Optional[str] = None
    scope: ConfigScope = ConfigScope.APPLICATION


class BaseConfigProvider(ABC):
    """기본 설정 제공자 추상 클래스"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def load_config(self, path: str) -> Result[Dict[str, Any]]:
        """설정 로드"""
        pass

    @abstractmethod
    async def save_config(self, path: str, config: Dict[str, Any]) -> Result[bool]:
        """설정 저장"""
        pass

    @abstractmethod
    def supports_format(self, format_type: ConfigFormat) -> bool:
        """형식 지원 여부"""
        pass


class JsonConfigProvider(BaseConfigProvider):
    """JSON 설정 제공자"""

    def __init__(self):
        super().__init__("JsonProvider")

    async def load_config(self, path: str) -> Result[Dict[str, Any]]:
        """JSON 설정 로드"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return Result.success(config)

        except FileNotFoundError:
            return Result.failure(ConfigurationError(
                config_key=path,
                expected_format="JSON file"
            ))
        except json.JSONDecodeError as e:
            return Result.failure(ConfigurationError(
                config_key=path,
                expected_format=f"Valid JSON (error: {str(e)})"
            ))
        except Exception as e:
            return Result.failure(ConfigurationError(
                config_key=path,
                expected_format=f"Readable JSON file (error: {str(e)})"
            ))

    async def save_config(self, path: str, config: Dict[str, Any]) -> Result[bool]:
        """JSON 설정 저장"""
        try:
            # 디렉토리 생성
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return Result.success(True)

        except Exception as e:
            return Result.failure(ConfigurationError(
                config_key=path,
                expected_format=f"Writable JSON file (error: {str(e)})"
            ))

    def supports_format(self, format_type: ConfigFormat) -> bool:
        """JSON 형식 지원"""
        return format_type == ConfigFormat.JSON


class YamlConfigProvider(BaseConfigProvider):
    """YAML 설정 제공자"""

    def __init__(self):
        super().__init__("YamlProvider")

    async def load_config(self, path: str) -> Result[Dict[str, Any]]:
        """YAML 설정 로드"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return Result.success(config or {})

        except FileNotFoundError:
            return Result.failure(ConfigurationError(
                config_key=path,
                expected_format="YAML file"
            ))
        except yaml.YAMLError as e:
            return Result.failure(ConfigurationError(
                config_key=path,
                expected_format=f"Valid YAML (error: {str(e)})"
            ))
        except Exception as e:
            return Result.failure(ConfigurationError(
                config_key=path,
                expected_format=f"Readable YAML file (error: {str(e)})"
            ))

    async def save_config(self, path: str, config: Dict[str, Any]) -> Result[bool]:
        """YAML 설정 저장"""
        try:
            # 디렉토리 생성
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            return Result.success(True)

        except Exception as e:
            return Result.failure(ConfigurationError(
                config_key=path,
                expected_format=f"Writable YAML file (error: {str(e)})"
            ))

    def supports_format(self, format_type: ConfigFormat) -> bool:
        """YAML 형식 지원"""
        return format_type == ConfigFormat.YAML


class EnvironmentConfigProvider(BaseConfigProvider):
    """환경변수 설정 제공자"""

    def __init__(self, prefix: str = "PACA_"):
        super().__init__("EnvironmentProvider")
        self.prefix = prefix

    async def load_config(self, path: str = "") -> Result[Dict[str, Any]]:
        """환경변수 설정 로드"""
        try:
            config = {}
            for key, value in os.environ.items():
                if key.startswith(self.prefix):
                    config_key = key[len(self.prefix):].lower()
                    config[config_key] = self._parse_env_value(value)

            return Result.success(config)

        except Exception as e:
            return Result.failure(ConfigurationError(
                config_key="environment",
                expected_format=f"Valid environment variables (error: {str(e)})"
            ))

    async def save_config(self, path: str, config: Dict[str, Any]) -> Result[bool]:
        """환경변수 설정 저장 (지원되지 않음)"""
        return Result.failure(ConfigurationError(
            config_key="environment",
            expected_format="Environment variables cannot be saved permanently"
        ))

    def supports_format(self, format_type: ConfigFormat) -> bool:
        """환경변수 형식 지원"""
        return format_type == ConfigFormat.ENV

    def _parse_env_value(self, value: str) -> Union[str, int, float, bool]:
        """환경변수 값 파싱"""
        # 불린 값
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # 숫자 값
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # 문자열 값
        return value


class ConfigManager:
    """설정 관리자"""

    def __init__(self):
        self.providers: Dict[ConfigFormat, BaseConfigProvider] = {
            ConfigFormat.JSON: JsonConfigProvider(),
            ConfigFormat.YAML: YamlConfigProvider(),
            ConfigFormat.ENV: EnvironmentConfigProvider()
        }
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.schemas: Dict[str, List[ConfigSchema]] = {}
        self._is_initialized: bool = False

    async def initialize(self) -> Result[bool]:
        """설정 관리자 초기화"""
        if self._is_initialized:
            return Result.success(True)

        try:
            default_config = {
                "system": {
                    "name": "PACA v5",
                    "version": "5.0.0",
                    "debug": False
                },
                "cognitive": {
                    "enable_metacognition": True,
                    "max_reasoning_steps": 10,
                    "quality_threshold": 0.7
                },
                "learning": {
                    "enable_auto_learning": True,
                    "korean_nlp": True,
                    "pattern_detection": True
                },
                "llm": {
                    "provider": "gemini",
                    "models": {
                        "conversation": [
                            "gemini-2.5-pro",
                            "gemini-2.5-flash"
                        ],
                        "image": [
                            "gemini-2.5-flash-image-preview",
                            "gemini-2.0-flash-preview-image-generation"
                        ]
                    },
                    "api_keys": [
                        "AIzaSyBNsviO1QfFcqKfeqVdvUzIk6bQ2McCk00",
                        "AIzaSyDxipPbvDUBQZlLucC8yTqLEc9D-HnqhLw",
                        "AIzaSyBhYUTppmklAYspJgK81w57BbEMgan7YkQ"
                    ],
                    "rotation": {
                        "strategy": "round_robin",
                        "min_interval_seconds": 1.0
                    }
                }
            }

            # 기본 네임스페이스 보관(기존 인터페이스 유지)
            self.configs.setdefault("default", default_config)

            # === [핵심 패치] 기본값을 외부로 노출 + 호환키 + ENV 키 반영 ===
            # 1) 최종 config를 외부에서 접근 가능하게 노출
            self.config = dict(default_config)

            # 2) 학습/메타인지 호환 키 제공 (enabled로도 접근 가능)
            learning = self.config.get("learning", {})
            if "enabled" not in learning:
                learning["enabled"] = bool(learning.get("enable_auto_learning", True))
            self.config["learning"] = learning

            cognitive = self.config.get("cognitive", {})
            metacog_enabled = bool(cognitive.get("enable_metacognition", True))
            # metacognition.enabled 미러링
            self.config["metacognition"] = {
                "enabled": metacog_enabled,
                **self.config.get("metacognition", {})
            }

            # 3) ENV에서 GEMINI_API_KEYS="k1,k2,..." 읽어오면 우선 적용
            import os
            env_keys = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
            if env_keys:
                self.config["llm"]["api_keys"] = env_keys

            # 4) llm 모델/키가 비면 기본값 보강
            if not self.config["llm"].get("models"):
                self.config["llm"]["models"] = default_config["llm"]["models"]
            if not self.config["llm"].get("api_keys"):
                self.config["llm"]["api_keys"] = default_config["llm"]["api_keys"]
            # === [핵심 패치 끝] ===

            self._is_initialized = True
            return Result.success(True)

        except Exception as error:
            return Result.failure(ConfigurationError(
                config_key="initialization",
                expected_format=f"Valid configuration setup (error: {str(error)})"
            ))

    def register_provider(self, format_type: ConfigFormat, provider: BaseConfigProvider) -> None:
        """설정 제공자 등록"""
        self.providers[format_type] = provider

    def register_schema(self, namespace: str, schemas: List[ConfigSchema]) -> None:
        """설정 스키마 등록"""
        self.schemas[namespace] = schemas

    async def load_config(
        self,
        namespace: str,
        source: str,
        format_type: ConfigFormat = ConfigFormat.JSON
    ) -> Result[bool]:
        """설정 로드"""
        if format_type not in self.providers:
            return Result.failure(ConfigurationError(
                config_key=namespace,
                expected_format=f"Supported format (available: {list(self.providers.keys())})"
            ))

        provider = self.providers[format_type]
        config_result = await provider.load_config(source)

        if not config_result.is_success:
            return Result.failure(config_result.error)

        config = config_result.data

        # 스키마 검증
        if namespace in self.schemas:
            validation_result = await self._validate_config(namespace, config)
            if not validation_result.is_success:
                return validation_result

        self.configs[namespace] = config
        return Result.success(True)

    async def save_config(
        self,
        namespace: str,
        destination: str,
        format_type: ConfigFormat = ConfigFormat.JSON
    ) -> Result[bool]:
        """설정 저장"""
        if namespace not in self.configs:
            return Result.failure(ConfigurationError(
                config_key=namespace,
                expected_format="Loaded configuration namespace"
            ))

        if format_type not in self.providers:
            return Result.failure(ConfigurationError(
                config_key=namespace,
                expected_format=f"Supported format (available: {list(self.providers.keys())})"
            ))

        provider = self.providers[format_type]
        config = self.configs[namespace]

        return await provider.save_config(destination, config)

    def get_value(self, namespace: str, key: str, default: Any = None) -> Any:
        """설정 값 조회"""
        if namespace not in self.configs:
            return default

        config = self.configs[namespace]
        keys = key.split('.')

        current = config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current

    def set_value(self, namespace: str, key: str, value: Any) -> Result[bool]:
        """설정 값 설정"""
        if namespace not in self.configs:
            self.configs[namespace] = {}

        config = self.configs[namespace]
        keys = key.split('.')

        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value
        return Result.success(True)

    def get_config(self, namespace: str) -> Optional[Dict[str, Any]]:
        """전체 설정 조회"""
        return self.configs.get(namespace)

    def list_namespaces(self) -> List[str]:
        """네임스페이스 목록"""
        return list(self.configs.keys())

    async def _validate_config(self, namespace: str, config: Dict[str, Any]) -> Result[bool]:
        """설정 검증"""
        schemas = self.schemas.get(namespace, [])

        for schema in schemas:
            value = self._get_nested_value(config, schema.name)

            # 필수 필드 검사
            if schema.required and value is None:
                return Result.failure(ValidationError(
                    message=f"Required field '{schema.name}' is missing",
                    field_name=schema.name
                ))

            # 타입 검사
            if value is not None:
                type_valid = await self._validate_type(value, schema.type)
                if not type_valid:
                    return Result.failure(ValidationError(
                        message=f"Field '{schema.name}' has invalid type. Expected: {schema.type}",
                        field_name=schema.name,
                        field_value=value
                    ))

                # 제약조건 검사
                constraints_valid = await self._validate_constraints(value, schema.constraints)
                if not constraints_valid:
                    return Result.failure(ValidationError(
                        message=f"Field '{schema.name}' violates constraints",
                        field_name=schema.name,
                        field_value=value,
                        constraints=list(schema.constraints.keys()) if schema.constraints else []
                    ))

        return Result.success(True)

    def _get_nested_value(self, config: Dict[str, Any], key: str) -> Any:
        """중첩된 키의 값 조회"""
        keys = key.split('.')
        current = config

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None

        return current

    async def _validate_type(self, value: Any, expected_type: str) -> bool:
        """타입 검증"""
        type_mapping = {
            'string': str,
            'integer': int,
            'float': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }

        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return True  # 알 수 없는 타입은 통과

        return isinstance(value, expected_python_type)

    async def _validate_constraints(self, value: Any, constraints: KeyValuePair) -> bool:
        """제약조건 검증"""
        if not constraints:
            return True

        # 최소/최대 값 검사
        if 'min' in constraints and value < constraints['min']:
            return False

        if 'max' in constraints and value > constraints['max']:
            return False

        # 최소/최대 길이 검사
        if 'min_length' in constraints and len(str(value)) < constraints['min_length']:
            return False

        if 'max_length' in constraints and len(str(value)) > constraints['max_length']:
            return False

        # 패턴 검사
        if 'pattern' in constraints:
            import re
            pattern = constraints['pattern']
            if not re.match(pattern, str(value)):
                return False

        # 허용된 값 검사
        if 'allowed_values' in constraints:
            allowed = constraints['allowed_values']
            if value not in allowed:
                return False

        return True
